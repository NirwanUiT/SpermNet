#!/usr/bin/env python3
"""
visualise.py

Generate trajectory plots and motility distribution charts for sperm analysis.

Usage:
    python visualise.py <video_name>
    python visualise.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory map
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(video_name: str, bg_frame: int = 0):
    """
    Plot all sperm trajectories overlaid on a background frame.
    Colour-coded by motility class if available.
    """
    tracks_csv = config.TRACK_OUT / f"{video_name}_tracks.csv"
    motility_csv = config.EVENTS_OUT / f"{video_name}_motility.csv"

    if not tracks_csv.exists():
        print(f"No tracks file: {tracks_csv}")
        return

    tracks = pd.read_csv(tracks_csv)
    if tracks.empty:
        print("Tracks file is empty.")
        return

    # Load motility classifications if available
    motility_map = {}
    if motility_csv.exists():
        mot = pd.read_csv(motility_csv)
        motility_map = dict(zip(mot["track_id"], mot["motility"]))

    # Try loading a background frame
    bg_img = None
    frame_dir = config.FRAMES_DIR / video_name
    if frame_dir.exists():
        frames = sorted(frame_dir.glob("*.jpg"))
        if frames and bg_frame < len(frames):
            bg_img = cv2.imread(str(frames[bg_frame]))
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if bg_img is not None:
        ax.imshow(bg_img, alpha=0.4)
        ax.set_xlim(0, bg_img.shape[1])
        ax.set_ylim(bg_img.shape[0], 0)
    else:
        ax.invert_yaxis()

    colour_map = {
        "progressive": "#00FF00",
        "non_progressive": "#FFA500",
        "immotile": "#FF0000",
    }
    default_colour = "#00BFFF"

    track_ids = tracks["track_id"].unique()
    for tid in track_ids:
        t = tracks[tracks["track_id"] == tid].sort_values("frame")
        xs = t["cx"].values
        ys = t["cy"].values

        motility = motility_map.get(tid, "unknown")
        colour = colour_map.get(motility, default_colour)

        ax.plot(xs, ys, color=colour, linewidth=0.8, alpha=0.7)
        # Start marker
        ax.scatter(xs[0], ys[0], c=colour, s=10, zorder=5, edgecolors="white", linewidths=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#00FF00", label=f"Progressive ({sum(1 for v in motility_map.values() if v == 'progressive')})"),
        Patch(facecolor="#FFA500", label=f"Non-progressive ({sum(1 for v in motility_map.values() if v == 'non_progressive')})"),
        Patch(facecolor="#FF0000", label=f"Immotile ({sum(1 for v in motility_map.values() if v == 'immotile')})"),
    ]
    if motility_map:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_title(f"Sperm Trajectories — {video_name}", fontsize=14)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    out_path = config.VIS_OUT / f"{video_name}_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajectory plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Motility distribution chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_motility_chart(video_name: str):
    """
    Create a multi-panel figure:
      (a) Pie chart of motility classes
      (b) VCL histogram
      (c) VSL vs VCL scatter
    """
    motility_csv = config.EVENTS_OUT / f"{video_name}_motility.csv"
    if not motility_csv.exists():
        print(f"No motility file: {motility_csv}")
        return

    df = pd.read_csv(motility_csv)
    if df.empty:
        print("Motility file is empty.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── (a) Pie chart ─────────────────────────────────────────────────────
    counts = df["motility"].value_counts()
    colors = {"progressive": "#00FF00", "non_progressive": "#FFA500", "immotile": "#FF0000"}
    pie_colors = [colors.get(c, "#999999") for c in counts.index]

    axes[0].pie(
        counts.values, labels=counts.index, colors=pie_colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    )
    axes[0].set_title("Motility Distribution")

    # ── (b) VCL histogram ─────────────────────────────────────────────────
    for mot_class in ["progressive", "non_progressive", "immotile"]:
        subset = df[df["motility"] == mot_class]["VCL"]
        if not subset.empty:
            axes[1].hist(
                subset, bins=20, alpha=0.6, label=mot_class,
                color=colors.get(mot_class, "#999"),
            )
    axes[1].set_xlabel("VCL (µm/s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("VCL Distribution by Motility Class")
    axes[1].legend(fontsize=9)

    # ── (c) VSL vs VCL scatter ────────────────────────────────────────────
    for mot_class in ["progressive", "non_progressive", "immotile"]:
        subset = df[df["motility"] == mot_class]
        if not subset.empty:
            axes[2].scatter(
                subset["VCL"], subset["VSL"], alpha=0.5, s=15,
                label=mot_class, color=colors.get(mot_class, "#999"),
            )
    # Reference line (LIN = 1 → VSL = VCL)
    max_val = max(df["VCL"].max(), df["VSL"].max()) * 1.1
    axes[2].plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="LIN=1")
    axes[2].set_xlabel("VCL (µm/s)")
    axes[2].set_ylabel("VSL (µm/s)")
    axes[2].set_title("VSL vs VCL (Linearity)")
    axes[2].legend(fontsize=9)

    fig.suptitle(f"Motility Analysis — {video_name}", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = config.VIS_OUT / f"{video_name}_motility_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Motility chart → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Velocity heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocity_heatmap(video_name: str, grid_size: int = 20):
    """
    Spatial heatmap of mean sperm velocity across the field of view.
    """
    tracks_csv = config.TRACK_OUT / f"{video_name}_tracks.csv"
    motility_csv = config.EVENTS_OUT / f"{video_name}_motility.csv"

    if not tracks_csv.exists() or not motility_csv.exists():
        print("Missing data files for heatmap.")
        return

    tracks = pd.read_csv(tracks_csv)
    mot = pd.read_csv(motility_csv)

    # Merge VCL into track positions
    vcl_map = dict(zip(mot["track_id"], mot["VCL"]))
    tracks["VCL"] = tracks["track_id"].map(vcl_map)
    tracks = tracks.dropna(subset=["VCL"])

    if tracks.empty:
        return

    xmax = tracks["cx"].max()
    ymax = tracks["cy"].max()

    heatmap = np.zeros((grid_size, grid_size))
    counts  = np.zeros((grid_size, grid_size))

    for _, row in tracks.iterrows():
        gx = int(row["cx"] / (xmax + 1) * grid_size)
        gy = int(row["cy"] / (ymax + 1) * grid_size)
        gx = min(gx, grid_size - 1)
        gy = min(gy, grid_size - 1)
        heatmap[gy, gx] += row["VCL"]
        counts[gy, gx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_vcl = np.where(counts > 0, heatmap / counts, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mean_vcl, cmap="hot", interpolation="bilinear", origin="upper")
    plt.colorbar(im, ax=ax, label="Mean VCL (µm/s)")
    ax.set_title(f"Velocity Heatmap — {video_name}")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")

    out_path = config.VIS_OUT / f"{video_name}_velocity_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Velocity heatmap → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def visualise_all():
    track_files = sorted(config.TRACK_OUT.glob("*_tracks.csv"))
    for tf in track_files:
        vname = tf.stem.replace("_tracks", "")
        plot_trajectories(vname)
        plot_motility_chart(vname)
        plot_velocity_heatmap(vname)


def main():
    parser = argparse.ArgumentParser(description="Visualise sperm analysis results")
    parser.add_argument("video", nargs="?", help="Video name")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        visualise_all()
    elif args.video:
        plot_trajectories(args.video)
        plot_motility_chart(args.video)
        plot_velocity_heatmap(args.video)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
