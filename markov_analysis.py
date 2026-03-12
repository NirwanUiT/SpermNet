#!/usr/bin/env python3
"""
markov_analysis.py

Markov Chain Transition Analysis of sperm motility states.

Computes per-frame instantaneous motility via a sliding window over raw
track positions, then builds transition count/probability matrices and
derives the stationary distribution.

Outputs:
  outputs/markov/transition_matrix.csv
  outputs/markov/transition_matrix.png
  outputs/markov/stationary_distribution.png
  outputs/markov/per_video_transitions.png

Usage:
    python markov_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from events.detect_events import px_to_um, classify_window

# ── Try to use Arial; fall back gracefully ────────────────────────────────────
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
MARKOV_DIR = config.MARKOV_OUT

# ── States ────────────────────────────────────────────────────────────────────
STATES = ["Progressive", "Non-progressive", "Immotile"]
STATE_COLORS = {"Progressive": "#2ca02c", "Non-progressive": "#ff7f0e", "Immotile": "#d62728"}
N_STATES = len(STATES)

# ── Sliding window parameters ────────────────────────────────────────────────
WINDOW_FRAMES = 25  # 0.5 s at 50 fps — enough for instantaneous VCL


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame instantaneous motility
# ═══════════════════════════════════════════════════════════════════════════════

# px_to_um and classify_window are imported from events.detect_events


def compute_frame_states(track_df: pd.DataFrame) -> list:
    """
    Compute per-frame motility state for a single track using a sliding window.

    Returns a list of state labels, one per frame in the track.
    """
    dt = 1.0 / config.FPS
    xs = track_df["cx"].values.astype(float)
    ys = track_df["cy"].values.astype(float)
    n = len(xs)

    if n < WINDOW_FRAMES:
        # Too short: classify entire segment
        state = classify_window(xs, ys, dt)
        return [state] * n

    half = WINDOW_FRAMES // 2
    states = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        state = classify_window(xs[lo:hi], ys[lo:hi], dt)
        states.append(state)
    return states


# ═══════════════════════════════════════════════════════════════════════════════
# Load data + build transitions
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_track_states():
    """
    Load all track CSVs and compute per-frame states.

    Returns dict: video_id -> list of state sequences (one per track).
    """
    video_states = {}
    track_files = sorted(config.TRACK_OUT.glob("*_tracks.csv"))

    for tf in track_files:
        vid = tf.stem.replace("_tracks", "")
        df = pd.read_csv(tf)
        if df.empty:
            continue

        sequences = []
        for tid in df["track_id"].unique():
            track = df[df["track_id"] == tid].sort_values("frame")
            if len(track) < config.MIN_TRACK_LENGTH:
                continue
            seq = compute_frame_states(track)
            sequences.append(seq)

        video_states[vid] = sequences
        n_tracks = len(sequences)
        n_frames = sum(len(s) for s in sequences)
        print(f"  Video {vid:>3}: {n_tracks:>4} tracks, {n_frames:>7} frame-states")

    return video_states


def count_transitions(sequences: list) -> np.ndarray:
    """Count consecutive state transitions into a 3x3 matrix."""
    state_idx = {s: i for i, s in enumerate(STATES)}
    counts = np.zeros((N_STATES, N_STATES), dtype=float)

    for seq in sequences:
        for prev, curr in zip(seq[:-1], seq[1:]):
            i = state_idx.get(prev)
            j = state_idx.get(curr)
            if i is not None and j is not None:
                counts[i, j] += 1

    return counts


def normalise_rows(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a count matrix to get transition probabilities."""
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    return counts / row_sums


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution by finding the left eigenvector
    of T corresponding to eigenvalue 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()  # normalise to sum to 1
    return pi


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_transition_heatmap(T: np.ndarray, save_path: Path,
                            title: str = "Transition Probability Matrix"):
    """Heatmap of the transition probability matrix."""
    try:
        import seaborn as sns
    except ImportError:
        print("  seaborn not installed; skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        T, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=STATES, yticklabels=STATES,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Probability"},
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)
    ax.tick_params(axis="both", which="both", length=0)  # no tick marks
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap -> {save_path}")


def plot_stationary_bar(pi: np.ndarray, save_path: Path):
    """Bar chart of stationary distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(STATES, pi, color=[STATE_COLORS[s] for s in STATES],
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, pi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylabel("Stationary Probability")
    ax.set_title("Stationary Distribution (Markov Chain)")
    ax.set_ylim(0, max(pi) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stationary bar -> {save_path}")


def plot_per_video_transitions(video_matrices: dict, save_path: Path):
    """4x5 grid of small heatmaps, one per video."""
    try:
        import seaborn as sns
    except ImportError:
        print("  seaborn not installed; skipping per-video grid.")
        return

    vids = sorted(video_matrices.keys(),
                  key=lambda x: int(x) if x.isdigit() else x)
    n = len(vids)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
    axes = axes.flatten()

    short_labels = ["P", "NP", "Im"]

    for i, vid in enumerate(vids):
        T = video_matrices[vid]
        sns.heatmap(
            T, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.3, linecolor="white",
            cbar=False, ax=axes[i], vmin=0, vmax=1,
            annot_kws={"fontsize": 8},
        )
        axes[i].set_title(f"Video {vid}", fontsize=10)
        axes[i].tick_params(axis="both", which="both", length=0, labelsize=8)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Video Transition Probability Matrices",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-video grid -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_markov_analysis():
    print("=" * 60)
    print("  MARKOV CHAIN TRANSITION ANALYSIS")
    print("=" * 60)

    # 1. Load all track states
    print("\n  Loading tracks and computing per-frame motility states...")
    video_states = load_all_track_states()

    if not video_states:
        print("  No track data found.")
        return

    # 2. Global transition matrix
    all_sequences = []
    for seqs in video_states.values():
        all_sequences.extend(seqs)

    print(f"\n  Total tracks: {len(all_sequences)}")
    print(f"  Total frame-states: {sum(len(s) for s in all_sequences)}")

    counts = count_transitions(all_sequences)
    T = normalise_rows(counts)
    pi = stationary_distribution(T)

    # 3. Per-video transition matrices
    video_matrices = {}
    for vid, seqs in video_states.items():
        c = count_transitions(seqs)
        video_matrices[vid] = normalise_rows(c)

    # 4. Save transition matrix CSV
    T_df = pd.DataFrame(T, index=STATES, columns=STATES)
    T_df.to_csv(MARKOV_DIR / "transition_matrix.csv")
    print(f"\n  Transition matrix CSV -> {MARKOV_DIR / 'transition_matrix.csv'}")

    # Save counts too
    C_df = pd.DataFrame(counts.astype(int), index=STATES, columns=STATES)
    C_df.to_csv(MARKOV_DIR / "transition_counts.csv")

    # 5. Visualise
    print("\n  Generating plots...")
    plot_transition_heatmap(T, MARKOV_DIR / "transition_matrix.png")
    plot_stationary_bar(pi, MARKOV_DIR / "stationary_distribution.png")
    plot_per_video_transitions(video_matrices,
                               MARKOV_DIR / "per_video_transitions.png")

    # 6. Summary printout
    print(f"\n{'=' * 60}")
    print("  MARKOV ANALYSIS SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  Transition Probability Matrix T:")
    print(f"  {'':>16} {'Progressive':>13} {'Non-prog':>13} {'Immotile':>13}")
    for i, s in enumerate(STATES):
        row = "  ".join(f"{T[i,j]:.4f}" for j in range(N_STATES))
        print(f"  {s:>16}   {row}")

    print(f"\n  Transition Counts:")
    print(f"  {'':>16} {'Progressive':>13} {'Non-prog':>13} {'Immotile':>13}")
    for i, s in enumerate(STATES):
        row = "  ".join(f"{int(counts[i,j]):>10}" for j in range(N_STATES))
        print(f"  {s:>16}   {row}")

    print(f"\n  Most likely transition FROM each state:")
    for i, s in enumerate(STATES):
        j_max = np.argmax(T[i])
        print(f"    {s:>16} -> {STATES[j_max]:<16} (p = {T[i, j_max]:.4f})")

    print(f"\n  Stationary Distribution:")
    for i, s in enumerate(STATES):
        print(f"    {s:>16}: pi = {pi[i]:.4f}  ({100*pi[i]:.1f}%)")

    # Self-transition (persistence) rates
    print(f"\n  Self-transition (persistence) rates:")
    for i, s in enumerate(STATES):
        print(f"    {s:>16}: {T[i,i]:.4f}  ({100*T[i,i]:.1f}%)")

    # Mean sojourn time: 1 / (1 - T[i,i]) frames, then convert to seconds
    print(f"\n  Mean sojourn time (expected duration in each state):")
    for i, s in enumerate(STATES):
        if T[i, i] < 1.0:
            sojourn_frames = 1.0 / (1.0 - T[i, i])
            sojourn_s = sojourn_frames / config.FPS
            print(f"    {s:>16}: {sojourn_frames:.0f} frames = {sojourn_s:.2f} s")
        else:
            print(f"    {s:>16}: absorbing state (no exits)")

    print(f"\n  Outputs in: {MARKOV_DIR}")
    print(f"{'=' * 60}")

    return T, pi, video_matrices


if __name__ == "__main__":
    run_markov_analysis()
