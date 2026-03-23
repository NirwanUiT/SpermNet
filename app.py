#!/usr/bin/env python3
"""
app.py

Interactive Gradio web UI for the SpermNet sperm motility analysis pipeline.

Tabs
----
1. Analysis      – Upload a tracks CSV (or generate demo data) and compute
                   WHO 2021 motility metrics without needing a GPU or dataset.
2. Visualizations– Motility distribution chart, per-class velocity box-plots,
                   VCL histogram, and trajectory scatter plot.
3. Report        – Template-based or LLM-generated clinical semen report.
4. Chatbot       – Multi-turn LLM chatbot that answers questions about the
                   analysed sample using the Anthropic / OpenAI backend.

Run
---
    python app.py
    # or
    gradio app.py
"""

import json
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – must be set before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from events.detect_events import (
    compute_track_metrics,
    classify_motility,
    filter_tracks,
)

# ── optional LLM imports ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

try:
    from llm.analyze import generate_report_local, generate_report_llm, generate_report_anthropic
    HAS_LLM_ANALYZE = True
except ImportError:
    HAS_LLM_ANALYZE = False

try:
    from llm.chatbot import SpermAnalysisChatbot
    HAS_CHATBOT = True
except ImportError:
    HAS_CHATBOT = False


# ─────────────────────────────────────────────────────────────────────────────
# Color palette (consistent with config.VIS_COLORS)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "progressive":     "#27ae60",
    "non_progressive": "#e67e22",
    "immotile":        "#e74c3c",
}
MOTILITY_ORDER = ["progressive", "non_progressive", "immotile"]
MOTILITY_LABELS = {
    "progressive":     "Progressive",
    "non_progressive": "Non-progressive",
    "immotile":        "Immotile",
}

# WHO 2021 reference values
WHO_TOTAL_MOTILITY = 42.0   # % (progressive + non-progressive)
WHO_PROGRESSIVE    = 30.0   # %


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generator
# ─────────────────────────────────────────────────────────────────────────────

def _random_walk(n: int, step_std: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random-walk trajectory."""
    angles = rng.uniform(0, 2 * np.pi, n)
    steps  = np.abs(rng.normal(step_std, step_std * 0.3, n))
    xs = np.cumsum(steps * np.cos(angles)) + rng.uniform(50, 590)
    ys = np.cumsum(steps * np.sin(angles)) + rng.uniform(50, 430)
    return xs, ys


def generate_demo_tracks(
    n_tracks: int = 120,
    n_frames: int = 300,
    fps: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesize realistic-looking VISEM-style tracking data with a mix of
    progressive, non-progressive, and immotile sperm.

    Returns a DataFrame with columns:
        track_id, frame, cx, cy, x1, y1, x2, y2, conf
    """
    rng = np.random.default_rng(seed)

    # Proportions loosely matching a WHO-normal sample
    DEMO_PROGRESSIVE_RATIO     = 0.38
    DEMO_NON_PROGRESSIVE_RATIO = 0.22

    n_prog   = int(n_tracks * DEMO_PROGRESSIVE_RATIO)
    n_nonpro = int(n_tracks * DEMO_NON_PROGRESSIVE_RATIO)
    n_immot  = n_tracks - n_prog - n_nonpro

    rows = []

    def _make_track(tid, track_length, step_std, drift_std, start_frame=0):
        xs, ys = _random_walk(track_length, step_std, rng)
        if drift_std > 0:
            # Add net drift for progressive sperm
            drift_angle = rng.uniform(0, 2 * np.pi)
            drift_x = np.cumsum(rng.normal(drift_std * np.cos(drift_angle),
                                           drift_std * 0.1, track_length))
            drift_y = np.cumsum(rng.normal(drift_std * np.sin(drift_angle),
                                           drift_std * 0.1, track_length))
            xs = xs + drift_x
            ys = ys + drift_y
        bw = 8  # box half-width
        conf = rng.uniform(0.55, 0.97, track_length)
        for i in range(track_length):
            rows.append({
                "track_id": tid,
                "frame":    start_frame + i,
                "cx":       float(np.clip(xs[i], 5, 635)),
                "cy":       float(np.clip(ys[i], 5, 475)),
                "x1":       float(np.clip(xs[i] - bw, 0, 640)),
                "y1":       float(np.clip(ys[i] - bw, 0, 480)),
                "x2":       float(np.clip(xs[i] + bw, 0, 640)),
                "y2":       float(np.clip(ys[i] + bw, 0, 480)),
                "conf":     float(conf[i]),
            })

    tid = 1
    for _ in range(n_prog):
        # Progressive: large step (≈14–18 px / frame ≈ 25+ µm/s), strong net drift
        length = rng.integers(80, n_frames + 1)
        start  = rng.integers(0, max(1, n_frames - length))
        _make_track(tid, length, step_std=16.0, drift_std=10.0, start_frame=int(start))
        tid += 1

    for _ in range(n_nonpro):
        # Non-progressive: medium step, little drift
        length = rng.integers(40, 200)
        start  = rng.integers(0, max(1, n_frames - length))
        _make_track(tid, length, step_std=8.0, drift_std=0.5, start_frame=int(start))
        tid += 1

    for _ in range(n_immot):
        # Immotile: tiny step, no drift
        length = rng.integers(20, 100)
        start  = rng.integers(0, max(1, n_frames - length))
        _make_track(tid, length, step_std=1.5, drift_std=0.0, start_frame=int(start))
        tid += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(tracks_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compute WHO motility metrics and classify all tracks.

    Returns
    -------
    metrics_df : DataFrame with per-track metrics + motility column.
    summary    : Dict matching the JSON summary written by analyse_video().
    """
    track_ids = tracks_df["track_id"].unique()

    metrics_list = []
    for tid in track_ids:
        track = tracks_df[tracks_df["track_id"] == tid].sort_values("frame")
        m = compute_track_metrics(track)
        if m is not None:
            metrics_list.append(m)

    if not metrics_list:
        return pd.DataFrame(), {}

    # Apply quality filters (use dummy conf=1.0 if no conf column)
    if "conf" not in tracks_df.columns:
        tracks_df = tracks_df.copy()
        tracks_df["conf"] = 1.0

    metrics_list, filter_stats = filter_tracks(metrics_list, tracks_df)

    if not metrics_list:
        return pd.DataFrame(), {}

    for m in metrics_list:
        m["motility"] = classify_motility(m)

    metrics_df = pd.DataFrame(metrics_list)

    total  = len(metrics_df)
    counts = metrics_df["motility"].value_counts()
    prog   = int(counts.get("progressive",     0))
    nonpro = int(counts.get("non_progressive", 0))
    immot  = int(counts.get("immotile",        0))

    summary = {
        "total_tracks":          total,
        "progressive":           prog,
        "non_progressive":       nonpro,
        "immotile":              immot,
        "progressive_pct":       round(100 * prog   / total, 1) if total else 0.0,
        "non_progressive_pct":   round(100 * nonpro / total, 1) if total else 0.0,
        "immotile_pct":          round(100 * immot  / total, 1) if total else 0.0,
        "mean_VCL":              round(float(metrics_df["VCL"].mean()), 2),
        "mean_VSL":              round(float(metrics_df["VSL"].mean()), 2),
        "mean_VAP":              round(float(metrics_df["VAP"].mean()), 2),
        "mean_LIN":              round(float(metrics_df["LIN"].mean()), 3),
        "mean_STR":              round(float(metrics_df["STR"].mean()), 3),
        "mean_WOB":              round(float(metrics_df["WOB"].mean()), 3),
        "mean_ALH":              round(float(metrics_df["ALH"].mean()), 2),
        "mean_BCF":              round(float(metrics_df["BCF"].mean()), 2),
        **filter_stats,
    }
    return metrics_df, summary


def _fmt_who_status(value: float, threshold: float) -> str:
    icon = "✅" if value >= threshold else "⚠️"
    return f"{icon} {value:.1f}% (WHO ref ≥ {threshold:.0f}%)"


def build_summary_html(summary: dict) -> str:
    """Return an HTML card with the analysis summary."""
    if not summary:
        return "<p>No analysis results yet.</p>"

    prog_pct  = summary.get("progressive_pct",   0)
    nonpro_pct = summary.get("non_progressive_pct", 0)
    immot_pct  = summary.get("immotile_pct",      0)
    total      = summary.get("total_tracks",      0)
    total_motile = prog_pct + nonpro_pct

    who_total = _fmt_who_status(total_motile, WHO_TOTAL_MOTILITY)
    who_prog  = _fmt_who_status(prog_pct,    WHO_PROGRESSIVE)

    rows = [
        ("Progressive",     summary.get("progressive", 0),   prog_pct,   COLORS["progressive"]),
        ("Non-progressive", summary.get("non_progressive", 0), nonpro_pct, COLORS["non_progressive"]),
        ("Immotile",        summary.get("immotile", 0),        immot_pct,  COLORS["immotile"]),
    ]

    table_rows = "\n".join(
        f'<tr><td><span style="color:{c};font-weight:bold">●</span> {label}</td>'
        f'<td style="text-align:right">{count}</td>'
        f'<td style="text-align:right">{pct:.1f}%</td></tr>'
        for label, count, pct, c in rows
    )

    kinematic_rows = "\n".join(
        f'<tr><td>{param}</td><td style="text-align:right">{summary.get(key, 0):.2f}</td></tr>'
        for param, key in [
            ("VCL – Curvilinear velocity (µm/s)",    "mean_VCL"),
            ("VSL – Straight-line velocity (µm/s)",  "mean_VSL"),
            ("VAP – Avg path velocity (µm/s)",       "mean_VAP"),
            ("LIN – Linearity (VSL/VCL)",            "mean_LIN"),
            ("STR – Straightness (VSL/VAP)",         "mean_STR"),
            ("WOB – Wobble (VAP/VCL)",               "mean_WOB"),
            ("ALH – Lateral head displacement (µm)", "mean_ALH"),
            ("BCF – Beat-cross frequency (Hz)",      "mean_BCF"),
        ]
    )

    html = f"""
<div style="font-family:sans-serif;max-width:700px">
  <h3 style="margin-top:0">📊 Motility Summary ({total} tracked sperm)</h3>
  <table style="width:100%;border-collapse:collapse;margin-bottom:12px">
    <thead>
      <tr style="background:#f0f0f0">
        <th style="text-align:left;padding:6px">Category</th>
        <th style="text-align:right;padding:6px">Count</th>
        <th style="text-align:right;padding:6px">%</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
    <tfoot>
      <tr style="font-weight:bold;border-top:2px solid #ccc">
        <td style="padding:6px">Total tracked</td>
        <td style="text-align:right;padding:6px">{total}</td>
        <td style="text-align:right;padding:6px">100%</td>
      </tr>
    </tfoot>
  </table>

  <h4 style="margin-bottom:4px">🔬 WHO 2021 Compliance</h4>
  <ul style="margin:4px 0 12px 0">
    <li>Total motility: {who_total}</li>
    <li>Progressive motility: {who_prog}</li>
  </ul>

  <h4 style="margin-bottom:4px">📐 Mean Kinematic Parameters</h4>
  <table style="width:100%;border-collapse:collapse">
    <tbody>{kinematic_rows}</tbody>
  </table>
</div>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig) -> str:
    """Save a matplotlib figure to a temp PNG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def plot_motility_distribution(metrics_df: pd.DataFrame) -> str:
    """Dual pie + bar chart of motility class distribution."""
    if metrics_df is None or metrics_df.empty:
        return None

    counts = metrics_df["motility"].value_counts()
    labels = [MOTILITY_LABELS[k] for k in MOTILITY_ORDER if k in counts]
    values = [counts[k] for k in MOTILITY_ORDER if k in counts]
    colors = [COLORS[k] for k in MOTILITY_ORDER if k in counts]
    total  = len(metrics_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Sperm Motility Classification", fontsize=14, fontweight="bold")

    # ── Pie chart ────────────────────────────────────────────────────────────
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))})",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.set_title(f"Distribution (n = {total})", fontsize=11)

    # ── Bar chart ─────────────────────────────────────────────────────────
    pcts = [100 * v / total for v in values]
    bars = ax2.bar(labels, pcts, color=colors, edgecolor="white", linewidth=1.5)

    # WHO reference lines
    ax2.axhline(WHO_TOTAL_MOTILITY, color="#2980b9", linestyle="--", linewidth=1.2,
                label=f"WHO total motility ({WHO_TOTAL_MOTILITY}%)")
    ax2.axhline(WHO_PROGRESSIVE, color="#8e44ad", linestyle=":",  linewidth=1.2,
                label=f"WHO progressive ({WHO_PROGRESSIVE}%)")
    ax2.legend(fontsize=8, loc="upper right")

    for bar, pct in zip(bars, pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, max(pcts) * 1.25 if pcts else 100)
    ax2.set_title("Motility % vs WHO 2021 Reference", fontsize=11)

    plt.tight_layout()
    return _save_fig(fig)


def plot_velocity_boxplots(metrics_df: pd.DataFrame) -> str:
    """Box plots of VCL, VSL, and VAP split by motility class."""
    if metrics_df is None or metrics_df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    fig.suptitle("Velocity Distributions by Motility Class", fontsize=14, fontweight="bold")

    for ax, col, label in zip(
        axes,
        ["VCL", "VSL", "VAP"],
        ["VCL – Curvilinear (µm/s)", "VSL – Straight-line (µm/s)", "VAP – Avg path (µm/s)"],
    ):
        data  = [metrics_df.loc[metrics_df["motility"] == k, col].dropna().values
                 for k in MOTILITY_ORDER if k in metrics_df["motility"].values]
        lbls  = [MOTILITY_LABELS[k] for k in MOTILITY_ORDER
                 if k in metrics_df["motility"].values]
        clrs  = [COLORS[k] for k in MOTILITY_ORDER if k in metrics_df["motility"].values]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticklabels(lbls, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("µm/s")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return _save_fig(fig)


def plot_vcl_histogram(metrics_df: pd.DataFrame) -> str:
    """Stacked histogram of VCL coloured by motility class."""
    if metrics_df is None or metrics_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4))

    bins = np.linspace(0, metrics_df["VCL"].quantile(0.99), 40)
    for key in MOTILITY_ORDER:
        sub = metrics_df.loc[metrics_df["motility"] == key, "VCL"].dropna()
        if sub.empty:
            continue
        ax.hist(sub, bins=bins, alpha=0.7, color=COLORS[key],
                label=MOTILITY_LABELS[key], edgecolor="white", linewidth=0.5)

    ax.axvline(config.VCL_PROGRESSIVE_MIN, color="#2980b9", linestyle="--",
               linewidth=1.5, label=f"Progressive threshold ({config.VCL_PROGRESSIVE_MIN} µm/s)")
    ax.axvline(config.VCL_IMMOTILE_MAX,    color="#e74c3c", linestyle=":",
               linewidth=1.5, label=f"Immotile threshold ({config.VCL_IMMOTILE_MAX} µm/s)")

    ax.set_xlabel("VCL (µm/s)")
    ax.set_ylabel("Number of tracks")
    ax.set_title("Curvilinear Velocity Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return _save_fig(fig)


def plot_trajectories(tracks_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    """Scatter / trajectory plot of all tracks colour-coded by motility class."""
    if tracks_df is None or tracks_df.empty:
        return None

    motility_map: dict[int, str] = {}
    if metrics_df is not None and not metrics_df.empty and "motility" in metrics_df.columns:
        motility_map = dict(zip(metrics_df["track_id"], metrics_df["motility"]))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    track_ids = tracks_df["track_id"].unique()
    for tid in track_ids:
        t = tracks_df[tracks_df["track_id"] == tid].sort_values("frame")
        xs, ys = t["cx"].values, t["cy"].values
        mot    = motility_map.get(int(tid), "unknown")
        color  = COLORS.get(mot, "#00BFFF")
        ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.6)
        ax.scatter(xs[0], ys[0], c=color, s=8, zorder=5,
                   edgecolors="white", linewidths=0.2)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS[k], label=MOTILITY_LABELS[k])
        for k in MOTILITY_ORDER
        if k in motility_map.values()
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=9, framealpha=0.7)

    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xlabel("x (pixels)", color="white")
    ax.set_ylabel("y (pixels)", color="white")
    ax.set_title("Sperm Trajectories", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    plt.tight_layout()
    return _save_fig(fig)


def plot_kinematic_scatter(metrics_df: pd.DataFrame) -> str:
    """2×2 scatter grid: LIN vs STR, VCL vs VSL, WOB vs BCF, ALH vs BCF."""
    if metrics_df is None or metrics_df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Kinematic Parameter Scatter Plots", fontsize=14, fontweight="bold")

    pairs = [
        ("LIN", "STR",  "Linearity vs Straightness"),
        ("VCL", "VSL",  "VCL vs VSL"),
        ("WOB", "ALH",  "Wobble vs ALH"),
        ("VCL", "BCF",  "VCL vs BCF"),
    ]

    for ax, (xcol, ycol, title) in zip(axes.ravel(), pairs):
        for key in MOTILITY_ORDER:
            sub = metrics_df[metrics_df["motility"] == key]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub[ycol], c=COLORS[key],
                       alpha=0.5, s=18, label=MOTILITY_LABELS[key],
                       edgecolors="none")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title, fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(facecolor=COLORS[k], label=MOTILITY_LABELS[k])
               for k in MOTILITY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return _save_fig(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(summary: dict, sample_name: str, use_llm: bool, provider: str) -> str:
    """Generate a clinical report using template or LLM backend."""
    if not summary:
        return "⚠️ No analysis results. Please run the analysis first."

    if not use_llm or not HAS_LLM_ANALYZE:
        return generate_report_local(summary, sample_name)

    try:
        if provider == "Anthropic (Claude)":
            return generate_report_anthropic(summary, sample_name)
        else:
            return generate_report_llm(summary, sample_name)
    except Exception as exc:
        return (f"⚠️ LLM report generation failed: {exc}\n\n"
                "Falling back to template report:\n\n"
                + generate_report_local(summary, sample_name))


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_inline_context(summary: dict, metrics_df: pd.DataFrame, sample_name: str) -> str:
    """Build a system-prompt context string from in-memory analysis results."""
    if not summary:
        return "(No analysis data loaded — upload and analyse a tracks CSV first.)"

    parts = [f"Sample: {sample_name}\n"]
    parts.append("Motility summary (JSON):\n```json\n"
                 + json.dumps(summary, indent=2) + "\n```\n")

    if metrics_df is not None and not metrics_df.empty:
        table = metrics_df.to_string(index=False, max_rows=100, float_format="%.2f")
        parts.append(f"Per-track kinematics ({len(metrics_df)} tracks):\n```\n{table}\n```\n")

        vcl_col = "VCL"
        mot_col = "motility"
        if vcl_col in metrics_df.columns and mot_col in metrics_df.columns:
            prog = metrics_df[metrics_df[mot_col] == "progressive"]
            if not prog.empty:
                top10 = prog.nlargest(min(10, len(prog)), vcl_col)
                keep  = [c for c in ["track_id", "VCL", "VSL", "VAP", "LIN", "STR"]
                         if c in top10.columns]
                parts.append("Top 10 fastest progressive tracks by VCL:\n```\n"
                             + top10[keep].to_string(index=False, float_format="%.2f")
                             + "\n```\n")

    return "\n".join(parts)


_CHATBOT_SYSTEM = """\
You are an expert andrologist and reproductive biologist with deep knowledge
of WHO 2021 semen analysis guidelines. You have access to CASA (Computer-Assisted
Sperm Analysis) results for the sample described below.

Answer questions accurately and concisely. When relevant:
• Reference WHO 2021 reference values (≥ 42 % total motility, ≥ 30 % progressive).
• Cite specific track IDs, velocities, or percentages from the data.
• Explain kinematic parameters (VCL, VSL, VAP, LIN, STR, WOB, ALH, BCF) in plain
  language when asked.

Pipeline configuration:
• Calibration: {ppm} px/µm  |  {fps} fps
• WHO thresholds: VCL progressive ≥ {vcl_prog} µm/s, STR ≥ {str_prog},
  VCL immotile ≤ {vcl_imm} µm/s

{context}

Use Markdown formatting. Be concise but thorough.
""".format(
    ppm=config.PIXELS_PER_MICRON,
    fps=config.FPS,
    vcl_prog=config.VCL_PROGRESSIVE_MIN,
    str_prog=config.STR_PROGRESSIVE_MIN,
    vcl_imm=config.VCL_IMMOTILE_MAX,
    context="{context}",  # filled at runtime
)


def chat_response(
    message: str,
    history: list,
    summary_state: dict,
    metrics_state,
    sample_name: str,
    provider: str,
) -> tuple[list, list]:
    """Generate a chatbot reply using the in-memory analysis context."""
    if not message.strip():
        return history, history

    context = _build_inline_context(summary_state, metrics_state, sample_name)
    system  = _CHATBOT_SYSTEM.format(context=context)

    # Build message list from history (Gradio 6 messages format)
    messages = []
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": message})

    reply = _llm_call(system, messages, provider)

    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, new_history


def _llm_call(system: str, messages: list[dict], provider: str) -> str:
    """Dispatch an LLM call to OpenAI or Anthropic, or return a fallback."""
    try:
        import os
        if provider == "Anthropic (Claude)":
            import anthropic as _anthropic
            client = _anthropic.Anthropic()
            resp = client.messages.create(
                model=config.LLM_MODEL_ANTHROPIC,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=system,
                messages=messages,
            )
            return resp.content[0].text
        else:
            import openai as _openai
            client = _openai.OpenAI()
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                messages=[{"role": "system", "content": system}] + messages,
            )
            return resp.choices[0].message.content
    except ImportError as exc:
        return (f"⚠️ LLM package not installed: {exc}\n\n"
                "Install `openai` or `anthropic` and set your API key in `.env`.")
    except Exception as exc:
        return (f"⚠️ LLM error: {exc}\n\n"
                "Check that your API key is set in `.env` (see `.env.example`).")


# ─────────────────────────────────────────────────────────────────────────────
# UI callback functions
# ─────────────────────────────────────────────────────────────────────────────

def on_load_demo():
    """Generate demo tracks data and return as a CSV string for preview."""
    df = generate_demo_tracks()
    return df.to_csv(index=False), f"✅ Generated {len(df['track_id'].unique())} demo tracks across {df['frame'].max()+1} frames."


def on_upload_csv(file_obj):
    """Load an uploaded CSV and return preview info."""
    if file_obj is None:
        return "", "No file uploaded."
    try:
        df = pd.read_csv(file_obj)
        required = {"track_id", "frame", "cx", "cy"}
        missing  = required - set(df.columns)
        if missing:
            return "", f"❌ CSV is missing required columns: {missing}"
        n_tracks = df["track_id"].nunique()
        n_frames = df["frame"].nunique()
        return df.to_csv(index=False), (
            f"✅ Loaded {len(df):,} rows — {n_tracks} tracks across {n_frames} frames."
        )
    except Exception as exc:
        return "", f"❌ Failed to read CSV: {exc}"


def on_run_analysis(csv_str: str, sample_name: str):
    """Parse CSV, run analysis, return (summary HTML, metrics CSV, summary dict)."""
    if not csv_str or not csv_str.strip():
        return ("<p>⚠️ No data loaded. Click <b>Load Demo</b> or upload a CSV.</p>",
                "", {}, None)
    try:
        tracks_df = pd.read_csv(StringIO(csv_str))
    except Exception as exc:
        return f"<p>❌ Failed to parse data: {exc}</p>", "", {}, None

    if sample_name.strip() == "":
        sample_name = "sample_01"

    metrics_df, summary = run_analysis(tracks_df)
    if metrics_df is None or metrics_df.empty:
        return ("<p>⚠️ No analysable tracks found. "
                "Ensure tracks have ≥ 10 frames and valid centroids.</p>"),\
               "", {}, None

    summary_html = build_summary_html(summary)
    metrics_csv  = metrics_df.to_csv(index=False)
    return summary_html, metrics_csv, summary, metrics_df


def on_generate_plots(csv_str: str, metrics_csv: str):
    """Generate all visualisation figures from raw tracks + metrics CSVs."""
    if not metrics_csv or not metrics_csv.strip():
        none4 = (None, None, None, None)
        return none4

    metrics_df = pd.read_csv(StringIO(metrics_csv))
    tracks_df  = pd.read_csv(StringIO(csv_str)) if csv_str and csv_str.strip() else pd.DataFrame()

    return (
        plot_motility_distribution(metrics_df),
        plot_velocity_boxplots(metrics_df),
        plot_vcl_histogram(metrics_df),
        plot_trajectories(tracks_df, metrics_df),
        plot_kinematic_scatter(metrics_df),
    )


def on_generate_report(summary_state: dict, sample_name: str,
                       use_llm: bool, provider: str) -> str:
    if not summary_state:
        return "⚠️ Run the analysis first to generate a report."
    return generate_report(summary_state, sample_name or "sample_01", use_llm, provider)


def on_download_report(report_text: str) -> str:
    """Save report markdown to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(report_text)
    tmp.close()
    return tmp.name


def on_download_metrics(metrics_csv: str) -> str:
    """Save metrics CSV to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(metrics_csv)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app layout
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
.tab-header { font-size: 16px; font-weight: bold; }
.who-ok  { color: #27ae60; font-weight: bold; }
.who-warn { color: #e74c3c; font-weight: bold; }
"""

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="SpermNet – Sperm Motility Analysis",
    ) as app:

        # ── Shared state ──────────────────────────────────────────────────
        csv_state     = gr.State("")      # raw tracks CSV string
        summary_state = gr.State({})      # summary dict
        metrics_state = gr.State(None)    # metrics DataFrame

        # ── Header ───────────────────────────────────────────────────────
        gr.Markdown(
            """
# 🔬 SpermNet — Sperm Motility Analysis

Automated **WHO 2021**-compliant CASA (Computer-Assisted Sperm Analysis) using
YOLOv8 detection, BoT-SORT tracking, and LLM-powered clinical reporting.

Upload a **tracks CSV** (columns: `track_id, frame, cx, cy`) from the pipeline,
or click **Load Demo Data** to explore with synthetic data.
"""
        )

        # ─────────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 – Analysis
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📤 Analysis", id="tab_analysis"):
                gr.Markdown("### Step 1 — Load tracking data")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_csv = gr.File(
                            label="Upload tracks CSV",
                            file_types=[".csv"],
                        )
                        load_demo_btn = gr.Button(
                            "🎲 Load Demo Data", variant="secondary"
                        )
                        sample_name_box = gr.Textbox(
                            label="Sample / video name",
                            value="demo_sample",
                            placeholder="e.g. video_14",
                        )
                    with gr.Column(scale=2):
                        load_status = gr.Markdown("*No data loaded yet.*")
                        csv_preview = gr.Dataframe(
                            label="Data preview (first 10 rows)",
                            interactive=False,
                            wrap=False,
                        )

                gr.Markdown("### Step 2 — Run motility analysis")
                run_btn = gr.Button("▶️ Run Analysis", variant="primary", size="lg")

                summary_html = gr.HTML(label="Analysis summary")

                with gr.Accordion("📥 Download per-track metrics CSV", open=False):
                    metrics_csv_state = gr.State("")
                    download_metrics_btn = gr.Button("Prepare download")
                    metrics_download = gr.File(label="Download", interactive=False)

            # ══════════════════════════════════════════════════════════════
            # TAB 2 – Visualizations
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📊 Visualizations", id="tab_vis"):
                gr.Markdown(
                    "Plots are generated automatically after analysis. "
                    "Click **Refresh Plots** if you re-ran the analysis."
                )
                refresh_plots_btn = gr.Button("🔄 Refresh Plots", variant="secondary")

                with gr.Row():
                    plot_dist  = gr.Image(label="Motility Distribution", type="filepath")
                    plot_boxes = gr.Image(label="Velocity Box Plots",    type="filepath")

                with gr.Row():
                    plot_hist  = gr.Image(label="VCL Histogram",         type="filepath")
                    plot_traj  = gr.Image(label="Trajectory Map",        type="filepath")

                plot_scatter = gr.Image(label="Kinematic Scatter Plots", type="filepath")

            # ══════════════════════════════════════════════════════════════
            # TAB 3 – Clinical Report
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📋 Clinical Report", id="tab_report"):
                gr.Markdown(
                    "Generate a structured semen analysis report. "
                    "A local template is always available; LLM requires "
                    "`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` (see `.env.example`)."
                )

                with gr.Row():
                    use_llm_cb   = gr.Checkbox(label="Use LLM (requires API key)", value=False)
                    provider_dd  = gr.Dropdown(
                        choices=["Anthropic (Claude)", "OpenAI (GPT-4o-mini)"],
                        value="Anthropic (Claude)",
                        label="LLM provider",
                        interactive=True,
                    )

                gen_report_btn  = gr.Button("📝 Generate Report", variant="primary")
                report_md       = gr.Markdown(
                    value="*Run the analysis first, then click Generate Report.*"
                )

                with gr.Accordion("📥 Download report", open=False):
                    download_report_btn = gr.Button("Prepare download")
                    report_download     = gr.File(label="Download .md", interactive=False)

            # ══════════════════════════════════════════════════════════════
            # TAB 4 – Chatbot
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("💬 Chatbot", id="tab_chat"):
                gr.Markdown(
                    """
Ask questions about the analysed sample in plain English. The assistant is
pre-loaded with WHO 2021 guidelines and your analysis results.

> **Tip:** Run the analysis in the **Analysis** tab first so the chatbot has
> data to reference. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your
> `.env` file (see `.env.example`) and select the matching provider below.
"""
                )

                with gr.Row():
                    chat_provider_dd = gr.Dropdown(
                        choices=["Anthropic (Claude)", "OpenAI (GPT-4o-mini)"],
                        value="Anthropic (Claude)",
                        label="LLM provider",
                        scale=1,
                    )
                    chat_sample_name = gr.Textbox(
                        label="Sample name (read-only — set in Analysis tab)",
                        interactive=False,
                        scale=2,
                        value="demo_sample",
                    )

                chatbot = gr.Chatbot(
                    label="Sperm Analysis Assistant",
                    height=420,
                    layout="bubble",
                    placeholder="*Run analysis first, then ask questions about your sample.*",
                )
                chat_history_state = gr.State([])

                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. Is this sample normal? Which tracks are most progressive?",
                        lines=2,
                        scale=5,
                    )
                    send_btn  = gr.Button("Send ➤", variant="primary", scale=1)

                clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary", size="sm")

        # ─────────────────────────────────────────────────────────────────
        # Event wiring
        # ─────────────────────────────────────────────────────────────────

        # ── Load demo data ────────────────────────────────────────────────
        def _load_demo():
            csv_str, status = on_load_demo()
            df_preview = pd.read_csv(StringIO(csv_str)).head(10)
            return csv_str, status, df_preview

        load_demo_btn.click(
            fn=_load_demo,
            inputs=[],
            outputs=[csv_state, load_status, csv_preview],
        )

        # ── Upload CSV ────────────────────────────────────────────────────
        def _upload(file_obj):
            csv_str, status = on_upload_csv(file_obj)
            if csv_str:
                df_preview = pd.read_csv(StringIO(csv_str)).head(10)
            else:
                df_preview = pd.DataFrame()
            return csv_str, status, df_preview

        upload_csv.change(
            fn=_upload,
            inputs=[upload_csv],
            outputs=[csv_state, load_status, csv_preview],
        )

        # ── Run analysis ──────────────────────────────────────────────────
        def _run(csv_str, sample_name):
            html, metrics_csv, summary, metrics_df = on_run_analysis(csv_str, sample_name)
            return html, metrics_csv, summary, metrics_df, sample_name

        run_btn.click(
            fn=_run,
            inputs=[csv_state, sample_name_box],
            outputs=[summary_html, metrics_csv_state, summary_state, metrics_state,
                     chat_sample_name],
        )

        # Auto-refresh plots after analysis
        run_btn.click(
            fn=on_generate_plots,
            inputs=[csv_state, metrics_csv_state],
            outputs=[plot_dist, plot_boxes, plot_hist, plot_traj, plot_scatter],
        )

        # ── Refresh plots manually ────────────────────────────────────────
        refresh_plots_btn.click(
            fn=on_generate_plots,
            inputs=[csv_state, metrics_csv_state],
            outputs=[plot_dist, plot_boxes, plot_hist, plot_traj, plot_scatter],
        )

        # ── Download metrics ──────────────────────────────────────────────
        download_metrics_btn.click(
            fn=on_download_metrics,
            inputs=[metrics_csv_state],
            outputs=[metrics_download],
        )

        # ── Generate report ───────────────────────────────────────────────
        gen_report_btn.click(
            fn=on_generate_report,
            inputs=[summary_state, sample_name_box, use_llm_cb, provider_dd],
            outputs=[report_md],
        )

        # ── Download report ───────────────────────────────────────────────
        download_report_btn.click(
            fn=on_download_report,
            inputs=[report_md],
            outputs=[report_download],
        )

        # ── Chatbot send ──────────────────────────────────────────────────
        def _chat(msg, history, summary_s, metrics_s, sample_name, provider):
            new_history, _ = chat_response(
                msg, history, summary_s, metrics_s, sample_name, provider
            )
            return new_history, new_history, ""  # updated chatbot, state, clear input

        send_btn.click(
            fn=_chat,
            inputs=[chat_input, chat_history_state, summary_state, metrics_state,
                    chat_sample_name, chat_provider_dd],
            outputs=[chatbot, chat_history_state, chat_input],
        )
        chat_input.submit(
            fn=_chat,
            inputs=[chat_input, chat_history_state, summary_state, metrics_state,
                    chat_sample_name, chat_provider_dd],
            outputs=[chatbot, chat_history_state, chat_input],
        )

        # ── Clear chat ────────────────────────────────────────────────────
        clear_btn.click(
            fn=lambda: ([], []),
            inputs=[],
            outputs=[chatbot, chat_history_state],
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=_CSS,
    )
