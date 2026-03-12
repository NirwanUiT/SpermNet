#!/usr/bin/env python3
"""
temporal_analysis.py

Temporal Motility Dynamics — Sliding Window Analysis.

Divides each video into time windows and tracks how the proportion of
sperm in each motility class changes over the course of a recording.
Aggregates across all 20 VISEM-Tracking videos.

Outputs:
  outputs/temporal/temporal_dynamics.csv      (per-video per-window)
  outputs/temporal/temporal_aggregate.csv     (mean ± std across videos)
  outputs/temporal/stacked_area_chart.png
  outputs/temporal/line_plot_with_ci.png
  outputs/temporal/per_video_temporal_heatmap.png

Usage:
    python temporal_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from events.detect_events import px_to_um, classify_window

# ── Font setup ────────────────────────────────────────────────────────────────
try:
    fm.findfont("Arial", fallback_to_default=False)
    plt.rcParams["font.family"] = "Arial"
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ── Output directory ──────────────────────────────────────────────────────────
TEMPORAL_DIR = config.TEMPORAL_OUT

# ── States ────────────────────────────────────────────────────────────────────
STATES = ["Progressive", "Non-progressive", "Immotile"]
STATE_COLORS = {"Progressive": "#2ca02c", "Non-progressive": "#ff7f0e", "Immotile": "#d62728"}

# ── Window parameters ────────────────────────────────────────────────────────
WINDOW_SECS = 5.0                           # seconds per window
WINDOW_FRAMES = int(WINDOW_SECS * config.FPS)  # 250 frames at 50 fps
SUB_WINDOW = 25                              # for instantaneous motility


# ═══════════════════════════════════════════════════════════════════════════════
# Motility classification — imported from events.detect_events
# ═══════════════════════════════════════════════════════════════════════════════

# px_to_um and classify_window are imported from events.detect_events


def classify_track_at_frame(track_df: pd.DataFrame, frame: int) -> str:
    """Classify a track's motility at a specific frame using a local sub-window."""
    dt = 1.0 / config.FPS
    frames = track_df["frame"].values
    xs = track_df["cx"].values.astype(float)
    ys = track_df["cy"].values.astype(float)

    # Find closest position to requested frame
    idx = np.searchsorted(frames, frame)
    idx = min(idx, len(frames) - 1)

    half = SUB_WINDOW // 2
    lo = max(0, idx - half)
    hi = min(len(frames), idx + half + 1)

    return classify_window(xs[lo:hi], ys[lo:hi], dt)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-video temporal dynamics
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_video_temporal(track_file: Path) -> pd.DataFrame | None:
    """Compute per-window motility proportions for one video."""
    vid = track_file.stem.replace("_tracks", "")
    df = pd.read_csv(track_file)
    if df.empty:
        return None

    max_frame = int(df["frame"].max())
    n_windows = max_frame // WINDOW_FRAMES
    if n_windows == 0:
        n_windows = 1

    # Pre-sort tracks
    tracks = {}
    for tid in df["track_id"].unique():
        t = df[df["track_id"] == tid].sort_values("frame")
        if len(t) >= config.MIN_TRACK_LENGTH:
            tracks[tid] = t

    records = []
    for w in range(n_windows):
        f_start = w * WINDOW_FRAMES
        f_end = (w + 1) * WINDOW_FRAMES
        f_mid = (f_start + f_end) // 2
        t_mid_s = f_mid / config.FPS

        # Find tracks active in this window
        state_counts = {s: 0 for s in STATES}
        n_active = 0

        for tid, track in tracks.items():
            frames = track["frame"].values
            # Track is active if it overlaps this window
            if frames[-1] < f_start or frames[0] >= f_end:
                continue

            state = classify_track_at_frame(track, f_mid)
            state_counts[state] += 1
            n_active += 1

        if n_active == 0:
            continue

        rec = {
            "video": vid,
            "window": w + 1,
            "t_start_s": f_start / config.FPS,
            "t_mid_s": t_mid_s,
            "t_end_s": f_end / config.FPS,
            "n_active": n_active,
        }
        for s in STATES:
            rec[f"pct_{s}"] = 100.0 * state_counts[s] / n_active
            rec[f"n_{s}"] = state_counts[s]
        records.append(rec)

    if not records:
        return None

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_temporal(all_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std across videos for each window index."""
    pct_cols = [f"pct_{s}" for s in STATES]
    agg = all_df.groupby("window").agg(
        t_mid_s=("t_mid_s", "mean"),
        n_videos=("video", "nunique"),
        **{f"{c}_mean": (c, "mean") for c in pct_cols},
        **{f"{c}_std": (c, "std") for c in pct_cols},
        n_active_mean=("n_active", "mean"),
    ).reset_index()
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stacked_area(agg: pd.DataFrame, save_path: Path):
    """Stacked area chart of mean motility proportions over time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    t = agg["t_mid_s"].values

    bottoms = np.zeros(len(t))
    for s in STATES:
        vals = agg[f"pct_{s}_mean"].values
        ax.fill_between(t, bottoms, bottoms + vals,
                        label=s, color=STATE_COLORS[s], alpha=0.8)
        bottoms += vals

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motility Proportion (%)")
    ax.set_title("Temporal Motility Dynamics — Stacked Area")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stacked area -> {save_path}")


def plot_line_with_ci(agg: pd.DataFrame, save_path: Path):
    """Line plot with shaded ± 1 SD bands."""
    fig, ax = plt.subplots(figsize=(9, 5))
    t = agg["t_mid_s"].values

    for s in STATES:
        mean = agg[f"pct_{s}_mean"].values
        std = agg[f"pct_{s}_std"].values
        ax.plot(t, mean, label=s, color=STATE_COLORS[s], linewidth=2)
        ax.fill_between(t, mean - std, mean + std,
                        color=STATE_COLORS[s], alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motility Proportion (%)")
    ax.set_title("Temporal Motility Dynamics — Mean ± SD")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Line plot -> {save_path}")


def plot_per_video_heatmap(all_df: pd.DataFrame, save_path: Path):
    """Heatmap: rows = videos, columns = windows, values = % progressive."""
    try:
        import seaborn as sns
    except ImportError:
        print("  seaborn not installed; skipping per-video heatmap.")
        return

    pivot = all_df.pivot_table(
        index="video", columns="window",
        values="pct_Progressive", aggfunc="first"
    )

    # Sort by video ID (numeric if possible)
    pivot.index = pivot.index.astype(str)
    try:
        pivot = pivot.loc[sorted(pivot.index, key=int)]
    except ValueError:
        pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2), max(6, pivot.shape[0] * 0.4)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn",
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "% Progressive"},
        ax=ax, vmin=0, vmax=100,
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Video")
    ax.set_title("Per-Video Progressive Motility Over Time")
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-video heatmap -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Trend analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_trends(agg: pd.DataFrame):
    """Linear regression of each motility % over time windows."""
    print(f"\n  Linear Trend Analysis (over {len(agg)} time windows):")
    t = agg["t_mid_s"].values

    for s in STATES:
        mean = agg[f"pct_{s}_mean"].values
        slope, intercept, r, p, se = sp_stats.linregress(t, mean)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {s:>16}: slope = {slope:+.3f} %/s, "
              f"R² = {r**2:.3f}, p = {p:.4f} {sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_temporal_analysis():
    print("=" * 60)
    print("  TEMPORAL MOTILITY DYNAMICS")
    print("=" * 60)

    track_files = sorted(config.TRACK_OUT.glob("*_tracks.csv"))
    if not track_files:
        print("  No track files found.")
        return

    print(f"\n  Found {len(track_files)} video track files")
    print(f"  Window size: {WINDOW_SECS} s ({WINDOW_FRAMES} frames)")

    all_dfs = []
    for tf in track_files:
        vid = tf.stem.replace("_tracks", "")
        print(f"  Processing video {vid}...")
        df = analyse_video_temporal(tf)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("  No temporal data generated.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(TEMPORAL_DIR / "temporal_dynamics.csv", index=False)
    print(f"\n  Per-video data -> {TEMPORAL_DIR / 'temporal_dynamics.csv'}")

    # Aggregate across videos
    agg = aggregate_temporal(all_df)
    agg.to_csv(TEMPORAL_DIR / "temporal_aggregate.csv", index=False)
    print(f"  Aggregate data  -> {TEMPORAL_DIR / 'temporal_aggregate.csv'}")

    # Plots
    print("\n  Generating plots...")
    plot_stacked_area(agg, TEMPORAL_DIR / "stacked_area_chart.png")
    plot_line_with_ci(agg, TEMPORAL_DIR / "line_plot_with_ci.png")
    plot_per_video_heatmap(all_df, TEMPORAL_DIR / "per_video_temporal_heatmap.png")

    # Summary
    print(f"\n{'=' * 60}")
    print("  TEMPORAL ANALYSIS SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  Videos processed: {all_df['video'].nunique()}")
    print(f"  Windows per video: {agg['window'].max()}")
    print(f"  Window duration: {WINDOW_SECS} s")

    print(f"\n  Aggregate Motility Proportions by Window:")
    print(f"  {'Window':>6}  {'t_mid(s)':>8}  {'%Prog':>8}  {'%NonProg':>8}  {'%Immotile':>8}  {'N_active':>8}")
    for _, row in agg.iterrows():
        print(f"  {int(row['window']):>6}  {row['t_mid_s']:>8.1f}  "
              f"{row['pct_Progressive_mean']:>7.1f}%  "
              f"{row['pct_Non-progressive_mean']:>7.1f}%  "
              f"{row['pct_Immotile_mean']:>7.1f}%  "
              f"{row['n_active_mean']:>8.1f}")

    # Best progressive window
    best_idx = agg["pct_Progressive_mean"].idxmax()
    best = agg.loc[best_idx]
    print(f"\n  Best progressive window: #{int(best['window'])} "
          f"(t = {best['t_mid_s']:.1f} s, "
          f"{best['pct_Progressive_mean']:.1f}% progressive)")

    # Trend analysis
    analyse_trends(agg)

    print(f"\n  Outputs in: {TEMPORAL_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_temporal_analysis()
