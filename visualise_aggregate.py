#!/usr/bin/env python3
"""
visualise_aggregate.py

Generate aggregate visualisations across all processed VISEM videos:
  1. Motility distribution heatmap (videos × classes)
  2. VCL/VSL/VAP box plots across videos
  3. GT vs predicted correlation summary
  4. Cohort-level motility pie chart

Usage:
    python visualise_aggregate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

EVAL_DIR = config.OUTPUTS_DIR / "evaluation"
VIS_DIR  = config.VIS_OUT


def load_all_motility() -> pd.DataFrame:
    """Load all per-video motility CSVs into one DataFrame."""
    rows = []
    for csv in sorted(config.EVENTS_OUT.glob("*_motility.csv")):
        df = pd.read_csv(csv)
        vid = csv.stem.replace("_motility", "")
        df["video"] = vid
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_all_summaries() -> pd.DataFrame:
    """Load all per-video summary JSONs."""
    rows = []
    for sp in sorted(config.EVENTS_OUT.glob("*_summary.json")):
        with open(sp) as f:
            s = json.load(f)
        rows.append(s)
    return pd.DataFrame(rows)


def plot_motility_heatmap(summaries: pd.DataFrame):
    """Heatmap: videos (rows) × motility classes (cols)."""
    vids = summaries["video"].astype(str).values
    data = summaries[["progressive_pct", "non_progressive_pct", "immotile_pct"]].values

    fig, ax = plt.subplots(figsize=(8, max(6, len(vids) * 0.4)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Progressive", "Non-progressive", "Immotile"])
    ax.set_yticks(range(len(vids)))
    ax.set_yticklabels(vids)
    ax.set_ylabel("Video ID")
    ax.set_title("Motility Classification — All Videos (%)")

    # Annotate cells
    for i in range(len(vids)):
        for j in range(3):
            val = data[i, j]
            colour = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=colour, fontsize=9)

    plt.colorbar(im, ax=ax, label="%", shrink=0.6)
    fig.tight_layout()
    out = VIS_DIR / "aggregate_motility_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Motility heatmap → {out}")


def plot_velocity_boxplots(all_mot: pd.DataFrame):
    """Box plots of VCL, VSL, VAP grouped by video."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    params = [("VCL", "Curvilinear Velocity"), ("VSL", "Straight-line Velocity"),
              ("VAP", "Average Path Velocity")]

    videos = sorted(all_mot["video"].unique(), key=lambda x: int(x) if x.isdigit() else x)

    for ax, (param, label) in zip(axes, params):
        data = [all_mot[all_mot["video"] == v][param].dropna().values for v in videos]
        bp = ax.boxplot(data, labels=videos, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4ECDC4")
            patch.set_alpha(0.7)
        ax.set_ylabel(f"{param} (µm/s)")
        ax.set_title(f"{label} ({param})")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Video ID")
    fig.suptitle("Velocity Distributions Across Videos", fontsize=14)
    fig.tight_layout()
    out = VIS_DIR / "aggregate_velocity_boxplots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Velocity boxplots → {out}")


def plot_cohort_summary(summaries: pd.DataFrame):
    """Combined summary figure with pie chart and bar chart."""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    # ── (a) Overall cohort pie chart ──
    ax1 = fig.add_subplot(gs[0])
    # Weight by track count
    total_prog = (summaries["progressive_pct"] * summaries["total_tracks"] / 100).sum()
    total_np = (summaries["non_progressive_pct"] * summaries["total_tracks"] / 100).sum()
    total_im = (summaries["immotile_pct"] * summaries["total_tracks"] / 100).sum()
    total = total_prog + total_np + total_im

    vals = [total_prog, total_np, total_im]
    labels_pie = [
        f"Progressive\n({100*total_prog/total:.1f}%)",
        f"Non-progressive\n({100*total_np/total:.1f}%)",
        f"Immotile\n({100*total_im/total:.1f}%)",
    ]
    colors = ["#00CC00", "#FFA500", "#FF3333"]
    ax1.pie(vals, labels=labels_pie, colors=colors, startangle=90,
            textprops={"fontsize": 11}, autopct=lambda p: f"{p*total/100:.0f}")
    ax1.set_title(f"Cohort Motility\n(n={int(total)} tracks, {len(summaries)} videos)",
                  fontsize=12)

    # ── (b) WHO compliance bar chart ──
    ax2 = fig.add_subplot(gs[1])
    vids = summaries["video"].astype(str).values
    total_motility = summaries["progressive_pct"] + summaries["non_progressive_pct"]
    prog_pct = summaries["progressive_pct"].values

    x = np.arange(len(vids))
    width = 0.35

    bars1 = ax2.bar(x - width/2, total_motility, width, label="Total motility (%)",
                    color="#4ECDC4", alpha=0.8)
    bars2 = ax2.bar(x + width/2, prog_pct, width, label="Progressive (%)",
                    color="#00AA00", alpha=0.8)

    # WHO reference lines
    ax2.axhline(42, color="red", linestyle="--", alpha=0.5, label="WHO total motility ≥42%")
    ax2.axhline(30, color="orange", linestyle="--", alpha=0.5, label="WHO progressive ≥30%")

    ax2.set_xlabel("Video ID")
    ax2.set_ylabel("Motility (%)")
    ax2.set_title("WHO 2021 Reference Compliance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(vids, rotation=45)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(0, 110)

    fig.suptitle("VISEM-Tracking Cohort — Sperm Motility Analysis", fontsize=14)
    fig.tight_layout()
    out = VIS_DIR / "aggregate_cohort_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cohort summary → {out}")


def plot_kinematic_correlations(all_mot: pd.DataFrame):
    """Scatter matrix of key kinematic parameters, coloured by motility."""
    params = ["VCL", "VSL", "VAP", "LIN", "STR"]
    colors = {"progressive": "#00AA00", "non_progressive": "#FF8800", "immotile": "#CC0000"}

    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(15, 15))

    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            ax = axes[i, j]
            if i == j:
                # Histogram on diagonal
                for mot, c in colors.items():
                    subset = all_mot[all_mot["motility"] == mot][p1].dropna()
                    if not subset.empty:
                        ax.hist(subset, bins=30, alpha=0.5, color=c, density=True)
                ax.set_ylabel(p1 if j == 0 else "")
            else:
                for mot, c in colors.items():
                    subset = all_mot[all_mot["motility"] == mot]
                    if not subset.empty:
                        ax.scatter(subset[p2], subset[p1], c=c, s=3, alpha=0.3)

            if i == n-1:
                ax.set_xlabel(p2, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(p1, fontsize=9)
            else:
                ax.set_yticklabels([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)
    fig.suptitle("Kinematic Parameter Correlations", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.97])
    out = VIS_DIR / "aggregate_kinematic_scatter.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kinematic scatter → {out}")


def main():
    print("=" * 60)
    print("  AGGREGATE VISUALISATIONS")
    print("=" * 60)

    summaries = load_all_summaries()
    all_mot = load_all_motility()

    if summaries.empty:
        print("No summary files found. Run pipeline first.")
        return

    print(f"  Videos: {len(summaries)}")
    print(f"  Total tracks: {all_mot['track_id'].nunique() if not all_mot.empty else 0}")

    plot_motility_heatmap(summaries)
    plot_velocity_boxplots(all_mot)
    plot_cohort_summary(summaries)
    plot_kinematic_correlations(all_mot)

    print(f"\n  All plots saved to: {VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
