#!/usr/bin/env python3
"""
llm/analyze.py

Generate natural-language semen analysis reports from motility metrics
using an LLM (OpenAI-compatible API or local model).

Usage:
    python -m llm.analyze <video_name>
    python -m llm.analyze --all
    python -m llm.analyze <video_name> --local   # offline template report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv optional; key can be set via env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert andrologist and reproductive biologist. You are given
computer-assisted sperm analysis (CASA) metrics in JSON format. Write a
concise clinical-style report that:
1. States the sample identification.
2. Summarises the motility classification (progressive %, non-progressive %,
   immotile %) and comments on whether the sample meets WHO 2021 reference
   values (≥ 42 % total motility, ≥ 30 % progressive motility).
3. Reports the key kinematic parameters (VCL, VSL, VAP, LIN, STR).
4. Provides a brief clinical interpretation and any caveats about the
   automated analysis.
Keep the report under 300 words. Use Markdown formatting."""

USER_TEMPLATE = """\
Sample: {video}
Date:   {date}

CASA Metrics (JSON):
```json
{metrics_json}
```

Please write the clinical report."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_llm(summary: dict, video_name: str) -> str:
    """Call OpenAI-compatible API to generate a report."""
    if not HAS_OPENAI:
        print("WARNING: openai package not installed. Falling back to local template.")
        return generate_report_local(summary, video_name)

    client = openai.OpenAI()  # uses OPENAI_API_KEY env var

    user_msg = USER_TEMPLATE.format(
        video=video_name,
        date=datetime.now().strftime("%Y-%m-%d"),
        metrics_json=json.dumps(summary, indent=2),
    )

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM API error: {e}")
        print("Falling back to local template report.")
        return generate_report_local(summary, video_name)


# ─────────────────────────────────────────────────────────────────────────────
# Offline template report (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_local(summary: dict, video_name: str) -> str:
    """Generate a structured report using a fixed template (no LLM)."""
    total = summary.get("total_tracks", 0)
    prog_pct = summary.get("progressive_pct", 0)
    nonpro_pct = summary.get("non_progressive_pct", 0)
    immot_pct = summary.get("immotile_pct", 0)
    total_motile_pct = prog_pct + nonpro_pct

    # WHO 2021 reference thresholds
    who_total_motility = 42.0   # %
    who_progressive    = 30.0   # %

    meets_total   = total_motile_pct >= who_total_motility
    meets_prog    = prog_pct >= who_progressive

    status_total = "MEETS" if meets_total else "BELOW"
    status_prog  = "MEETS" if meets_prog else "BELOW"

    report = f"""# Semen Analysis Report

**Sample:** {video_name}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Analysis method:** Automated CASA (YOLOv8 + BoT-SORT)

---

## Motility Classification

| Category         | Count | Percentage |
|------------------|------:|----------:|
| Progressive      | {summary.get('progressive', 0):>5} | {prog_pct:>6.1f}% |
| Non-progressive  | {summary.get('non_progressive', 0):>5} | {nonpro_pct:>6.1f}% |
| Immotile         | {summary.get('immotile', 0):>5} | {immot_pct:>6.1f}% |
| **Total tracked**| **{total}** | |

**Total motility:** {total_motile_pct:.1f}% — **{status_total}** WHO 2021 reference (≥{who_total_motility}%)
**Progressive motility:** {prog_pct:.1f}% — **{status_prog}** WHO 2021 reference (≥{who_progressive}%)

## Kinematic Parameters

| Parameter | Value |
|-----------|------:|
| VCL (µm/s) | {summary.get('mean_VCL', 0):.1f} |
| VSL (µm/s) | {summary.get('mean_VSL', 0):.1f} |
| VAP (µm/s) | {summary.get('mean_VAP', 0):.1f} |
| LIN (VSL/VCL) | {summary.get('mean_LIN', 0):.3f} |
| STR (VSL/VAP) | {summary.get('mean_STR', 0):.3f} |

## Interpretation

{"The sample shows adequate motility with sufficient progressive movement." if meets_total and meets_prog else "The motility parameters are below WHO 2021 reference values, suggesting reduced sperm motility."}

> **Note:** This is an automated analysis. Results should be confirmed by a
> trained andrologist. Pixel-to-micron calibration ({config.PIXELS_PER_MICRON} px/µm)
> should be verified for the specific microscope setup.
"""
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyse_and_report(video_name: str, use_llm: bool = True) -> str:
    """Load motility summary and generate a report."""
    summary_path = config.EVENTS_OUT / f"{video_name}_summary.json"
    if not summary_path.exists():
        print(f"ERROR: Summary not found: {summary_path}")
        print("Run events analysis first:  python -m events.detect_events ...")
        return ""

    with open(summary_path) as f:
        summary = json.load(f)

    if use_llm:
        report = generate_report_llm(summary, video_name)
    else:
        report = generate_report_local(summary, video_name)

    # Save report
    report_path = config.REPORTS_OUT / f"{video_name}_report.md"
    report_path.write_text(report)
    print(f"Report saved → {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="LLM-powered semen analysis report")
    parser.add_argument("video", nargs="?", help="Video name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--local", action="store_true",
                        help="Use offline template instead of LLM API")
    args = parser.parse_args()

    use_llm = not args.local

    if args.all:
        summaries = sorted(config.EVENTS_OUT.glob("*_summary.json"))
        if not summaries:
            print("No summary files found.")
            return
        for sp in summaries:
            vname = sp.stem.replace("_summary", "")
            analyse_and_report(vname, use_llm=use_llm)
    elif args.video:
        report = analyse_and_report(args.video, use_llm=use_llm)
        if report:
            print("\n" + report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
