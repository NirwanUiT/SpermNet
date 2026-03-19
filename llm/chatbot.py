#!/usr/bin/env python3
"""
llm/chatbot.py

Interactive multi-turn chatbot for sperm motility analysis.

Loads a video's motility summary, per-track kinematics, clinical ground
truth, Markov transition data, and temporal analysis into the system prompt
so the LLM can answer detailed follow-up questions about the sample.

Supports both OpenAI (GPT-4o-mini) and Anthropic (Claude) backends,
selected via ``config.LLM_PROVIDER``.

Usage (CLI):
    python -m llm.chatbot 14          # interactive chat about video 14
    python -m llm.chatbot 14 --all    # load all 20 videos for comparison

Usage (Python):
    from llm.chatbot import SpermAnalysisChatbot
    bot = SpermAnalysisChatbot("14")
    print(bot.chat("Is this sample normal?"))
    print(bot.chat("Which tracks have the highest VCL?"))
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ── Optional LLM backends ────────────────────────────────────────────────────
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
You are an expert andrologist and reproductive biologist with deep knowledge
of WHO 2021 semen analysis guidelines.  You have access to automated CASA
(Computer-Assisted Sperm Analysis) results for the sample(s) described below.

Answer the user's questions accurately and concisely.  When relevant:
• Reference WHO 2021 reference values (≥ 42 % total motility, ≥ 30 %
  progressive motility).
• Cite specific track IDs, velocities, or percentages from the data.
• Explain kinematic parameters (VCL, VSL, VAP, LIN, STR, WOB, ALH, BCF)
  in plain language when asked.
• If a comparison between videos is requested but the data is not loaded,
  say so and suggest loading it.

Use Markdown formatting.  Be concise but thorough.
"""

_PIPELINE_CONTEXT = """\

Pipeline configuration:
• Calibration: {ppm} px/µm  |  {fps} fps
• WHO thresholds: VCL progressive ≥ {vcl_prog} µm/s, STR progressive ≥ {str_prog}, \
VCL immotile ≤ {vcl_imm} µm/s
• Quality filters: conf ≥ 0.4, VCL ≤ 200 µm/s, jitter gate, duration ≥ 0.3 s
"""


def _build_video_context(video_id: str) -> str:
    """Build a context block for one video: summary + per-track table + clinical GT."""
    parts: list[str] = [f"\n--- Video {video_id} ---\n"]

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary_path = config.EVENTS_OUT / f"{video_id}_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        parts.append("Motility summary (JSON):\n```json\n"
                      + json.dumps(summary, indent=2)
                      + "\n```\n")
    else:
        parts.append("(No summary JSON found — run the pipeline first.)\n")

    # ── Per-track kinematics ──────────────────────────────────────────────
    motility_path = config.EVENTS_OUT / f"{video_id}_motility.csv"
    if motility_path.exists():
        df = pd.read_csv(motility_path)
        # Compact table for the context window
        table = df.to_string(index=False, max_rows=200, float_format="%.2f")
        parts.append(f"Per-track kinematics ({len(df)} tracks):\n```\n{table}\n```\n")

        # Top-10 fastest tracks by VCL
        vcl_col = "VCL" if "VCL" in df.columns else "vcl"
        mot_col = ("motility" if "motility" in df.columns
                   else "motility_class" if "motility_class" in df.columns
                   else None)
        if vcl_col in df.columns:
            keep = [c for c in ("track_id", vcl_col, "VSL", "vsl",
                                "VAP", "vap", "LIN", "lin", "STR", "str",
                                mot_col) if c and c in df.columns]
            top10 = df.nlargest(10, vcl_col)[keep]
            parts.append("Top 10 fastest tracks by VCL:\n```\n"
                         + top10.to_string(index=False, float_format="%.2f")
                         + "\n```\n")

        # Motility distribution from per-track data
        if mot_col and mot_col in df.columns:
            dist = df[mot_col].value_counts(normalize=True).mul(100)
            parts.append("Motility distribution (from tracks):\n"
                         + "".join(f"  {cls}: {pct:.1f}%\n"
                                    for cls, pct in dist.items()))

    # ── Clinical ground truth ─────────────────────────────────────────────
    gt_path = config.RAW_DIR / "semen_analysis_data_Train.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        gt.columns = [c.strip() for c in gt.columns]
        row = gt[gt["ID"] == int(video_id)]
        if not row.empty:
            r = row.iloc[0]
            parts.append(
                f"Clinical ground truth:\n"
                f"  Progressive:     {r['Progressive motility (%)']}%\n"
                f"  Non-progressive: {r['Non progressive sperm motility (%)']}%\n"
                f"  Immotile:        {r['Immotile sperm (%)']}%\n"
            )

    return "\n".join(parts)


def _build_system_prompt(video_ids: list[str]) -> str:
    """Assemble the full system prompt with pipeline config + video data."""
    prompt = _BASE_SYSTEM
    prompt += _PIPELINE_CONTEXT.format(
        ppm=config.PIXELS_PER_MICRON,
        fps=config.FPS,
        vcl_prog=config.VCL_PROGRESSIVE_MIN,
        str_prog=config.STR_PROGRESSIVE_MIN,
        vcl_imm=config.VCL_IMMOTILE_MAX,
    )
    for vid in video_ids:
        prompt += _build_video_context(vid)

    # ── Markov transition matrix (global) ─────────────────────────────────
    markov_path = config.MARKOV_OUT / "transition_matrix.csv"
    if markov_path.exists():
        try:
            tm = pd.read_csv(markov_path, index_col=0)
            prompt += "\n--- Markov Transition Matrix (global) ---\n"
            prompt += ("Frame-to-frame transition probabilities between "
                       "motility states:\n```\n")
            prompt += tm.to_string(float_format="%.4f")
            prompt += "\n```\n"
        except Exception:
            pass

    # ── Temporal analysis summary ─────────────────────────────────────────
    temporal_path = config.TEMPORAL_OUT / "temporal_summary.csv"
    if temporal_path.exists():
        try:
            ts = pd.read_csv(temporal_path)
            prompt += "\n--- Temporal Analysis Summary ---\n"
            prompt += ("How motility distributions change over time within "
                       "each video:\n```\n")
            prompt += ts.to_string(index=False, max_rows=60,
                                   float_format="%.2f")
            prompt += "\n```\n"
        except Exception:
            pass

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot class
# ─────────────────────────────────────────────────────────────────────────────

class SpermAnalysisChatbot:
    """Multi-turn conversational chatbot for sperm motility analysis.

    Parameters
    ----------
    video_id : str or list[str]
        One or more VISEM video IDs to load context for.
    model : str
        Model name.  Defaults to the model for the configured provider.
    temperature : float
        Sampling temperature (default: config.LLM_TEMPERATURE → 0.3).
    provider : str
        ``"openai"`` or ``"anthropic"``.  Defaults to ``config.LLM_PROVIDER``.
    """

    def __init__(
        self,
        video_id: str | list[str] = "14",
        model: str | None = None,
        temperature: float | None = None,
        provider: str | None = None,
    ):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.temperature = (temperature if temperature is not None
                            else config.LLM_TEMPERATURE)

        # ── Initialise provider client ────────────────────────────────────
        if self.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError(
                    "anthropic package required.  pip install anthropic"
                )
            self.client = anthropic.Anthropic()
            self.model = model or config.LLM_MODEL_ANTHROPIC
        else:
            if not HAS_OPENAI:
                raise ImportError(
                    "openai package required.  pip install openai"
                )
            self.client = openai.OpenAI()
            self.model = model or config.LLM_MODEL

        if isinstance(video_id, str):
            video_id = [video_id]
        self.video_ids = video_id

        self._system_prompt = _build_system_prompt(self.video_ids)
        # Conversation history (user + assistant turns only)
        self.history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a message and get a response. Maintains conversation history."""
        self.history.append({"role": "user", "content": user_message})

        if self.provider == "anthropic":
            reply = self._chat_anthropic()
        else:
            reply = self._chat_openai()

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _chat_openai(self) -> str:
        """Route chat through the OpenAI API."""
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self.history)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content

    def _chat_anthropic(self) -> str:
        """Route chat through the Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=self.temperature,
            system=self._system_prompt,
            messages=self.history,
        )
        return response.content[0].text

    def reset(self):
        """Clear conversation history (keeps system prompt)."""
        self.history = []

    def load_video(self, video_id: str | list[str]):
        """Switch to a different video (or set of videos). Resets history."""
        if isinstance(video_id, str):
            video_id = [video_id]
        self.video_ids = video_id
        self._system_prompt = _build_system_prompt(self.video_ids)
        self.reset()

    @property
    def turn_count(self) -> int:
        """Number of user turns so far."""
        return sum(1 for m in self.history if m["role"] == "user")

    def __repr__(self) -> str:
        return (f"SpermAnalysisChatbot(videos={self.video_ids}, "
                f"provider={self.provider!r}, model={self.model!r}, "
                f"turns={self.turn_count})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def interactive_cli(bot: SpermAnalysisChatbot):
    """Run an interactive chat session in the terminal."""
    print(f"\n{'='*60}")
    print(f"  Sperm Analysis Chatbot — Videos: {bot.video_ids}")
    print(f"  Model: {bot.model}  |  Type 'quit' to exit")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower().startswith("/load "):
            vids = user_input[6:].strip().split()
            bot.load_video(vids)
            print(f"  [Loaded videos: {vids} — history reset]\n")
            continue
        if user_input.lower() == "/reset":
            bot.reset()
            print("  [Conversation history cleared]\n")
            continue

        try:
            reply = bot.chat(user_input)
            print(f"\nAssistant: {reply}\n")
        except Exception as e:
            print(f"\n  ERROR: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive sperm motility analysis chatbot"
    )
    parser.add_argument("video", nargs="?", default="14",
                        help="Video ID to load (default: 14)")
    parser.add_argument("--all", action="store_true",
                        help="Load all 20 VISEM videos for comparison queries")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default depends on provider)")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "anthropic"],
                        help=f"LLM provider (default: {config.LLM_PROVIDER})")
    args = parser.parse_args()

    if args.all:
        vids = sorted([d.name for d in config.VISEM_ROOT.iterdir() if d.is_dir()])
    else:
        vids = [args.video]

    bot = SpermAnalysisChatbot(vids, model=args.model, provider=args.provider)
    interactive_cli(bot)


if __name__ == "__main__":
    main()
