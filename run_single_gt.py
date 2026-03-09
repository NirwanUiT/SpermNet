#!/usr/bin/env python3
"""
run_single_gt.py

Run the sperm motility analysis pipeline on a single VISEM video using
GROUND TRUTH annotations (labels_ftid) for immediate results.

The VISEM labels_ftid format is:
    <feature_id> <class> <cx_norm> <cy_norm> <w_norm> <h_norm>
where feature_id is a consistent track ID across frames.

Stages:
  1. Parse GT annotations → build track CSV  (replaces detect + track)
  2. Compute WHO motility metrics
  3. Generate clinical report
  4. Create visualisations

Usage:
    python run_single_gt.py 11
    python run_single_gt.py 11 --stages gt_tracks events report vis
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Parse GT annotations → track CSV
# ─────────────────────────────────────────────────────────────────────────────

def parse_gt_tracks(video_id: str) -> pd.DataFrame:
    """
    Read all labels_ftid files for a video and build a tracks DataFrame
    identical in format to what the tracker would produce.
    """
    ftid_dir = config.VISEM_ROOT / video_id / "labels_ftid"
    images_dir = config.VISEM_ROOT / video_id / "images"

    if not ftid_dir.exists():
        print(f"ERROR: {ftid_dir} not found")
        return pd.DataFrame()

    # Get image dimensions for denormalisation
    sample_img = next(images_dir.glob("*.jpg"), None)
    if sample_img is None:
        print(f"ERROR: No images found in {images_dir}")
        return pd.DataFrame()
    img = cv2.imread(str(sample_img))
    img_h, img_w = img.shape[:2]
    print(f"  Image size: {img_w}×{img_h}")

    # Parse all ftid annotation files
    ftid_files = sorted(ftid_dir.glob("*_with_ftid.txt"))
    print(f"  Annotation files: {len(ftid_files)}")

    # Map feature string IDs to integer IDs
    fid_to_int = {}
    next_id = 1
    rows = []

    for fpath in ftid_files:
        # Extract frame number from filename: "11_frame_0_with_ftid.txt"
        parts = fpath.stem.replace("_with_ftid", "").split("_frame_")
        if len(parts) < 2:
            continue
        frame_num = int(parts[1])

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 6:
                    continue

                fid = tokens[0]          # feature/track ID (string)
                # cls = int(tokens[1])    # class (always 0 = sperm)
                cx_n = float(tokens[2])  # normalised centre x
                cy_n = float(tokens[3])  # normalised centre y
                w_n  = float(tokens[4])  # normalised width
                h_n  = float(tokens[5])  # normalised height

                # Assign integer track ID
                if fid not in fid_to_int:
                    fid_to_int[fid] = next_id
                    next_id += 1

                # Denormalise to pixel coordinates
                cx = cx_n * img_w
                cy = cy_n * img_h
                bw = w_n * img_w
                bh = h_n * img_h
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                rows.append({
                    "track_id": fid_to_int[fid],
                    "frame":    frame_num,
                    "cx":       cx,
                    "cy":       cy,
                    "x1":       x1,
                    "y1":       y1,
                    "x2":       x2,
                    "y2":       y2,
                    "conf":     1.0,  # ground truth
                })

    df = pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)

    if df.empty:
        print("  WARNING: No annotations parsed.")
        return df

    # Save
    csv_path = config.TRACK_OUT / f"{video_id}_tracks.csv"
    df.to_csv(csv_path, index=False)

    n_tracks = df["track_id"].nunique()
    n_frames = df["frame"].nunique()
    print(f"  → {n_tracks} unique tracks across {n_frames} frames ({len(df)} detections)")
    print(f"  → Saved: {csv_path}")

    # Summary
    track_lengths = df.groupby("track_id").size()
    print(f"  Track lengths: min={track_lengths.min()}, "
          f"median={track_lengths.median():.0f}, "
          f"max={track_lengths.max()}, "
          f"mean={track_lengths.mean():.1f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Motility analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_motility(video_id: str) -> pd.DataFrame:
    """Compute WHO motility metrics from tracks."""
    from events.detect_events import analyse_video
    print(f"\n[EVENTS] Video {video_id}")
    return analyse_video(video_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Report
# ─────────────────────────────────────────────────────────────────────────────

def run_report(video_id: str) -> str:
    """Generate semen analysis report."""
    from llm.analyze import analyse_and_report
    print(f"\n[REPORT] Video {video_id}")
    report = analyse_and_report(video_id, use_llm=False)
    if report:
        print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def run_visualisation(video_id: str):
    """Generate trajectory plots and charts."""
    from visualise import plot_trajectories, plot_motility_chart, plot_velocity_heatmap

    # Symlink VISEM images into frames dir so visualise.py finds them
    src = config.VISEM_ROOT / video_id / "images"
    dst = config.FRAMES_DIR / video_id
    if src.exists() and not dst.exists():
        dst.symlink_to(src)

    print(f"\n[VIS] Video {video_id}")
    plot_trajectories(video_id)
    plot_motility_chart(video_id)
    plot_velocity_heatmap(video_id)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

STAGE_MAP = {
    "gt_tracks": parse_gt_tracks,
    "events":    run_motility,
    "report":    run_report,
    "vis":       run_visualisation,
}

def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline on a single VISEM video using ground-truth annotations"
    )
    parser.add_argument("video_id", help="VISEM video ID (e.g. 11, 12, 13 ...)")
    parser.add_argument("--stages", nargs="+",
                        default=["gt_tracks", "events", "report", "vis"],
                        choices=list(STAGE_MAP.keys()))
    args = parser.parse_args()

    vid = args.video_id

    # Verify video exists
    vid_dir = config.VISEM_ROOT / vid
    if not vid_dir.exists():
        avail = sorted([d.name for d in config.VISEM_ROOT.iterdir() if d.is_dir()])
        print(f"ERROR: Video '{vid}' not found. Available: {avail}")
        sys.exit(1)

    print("=" * 60)
    print(f"  SPERM MOTILITY ANALYSIS — Video {vid} (Ground Truth)")
    print(f"  Stages: {', '.join(args.stages)}")
    print("=" * 60)

    for stage_name in args.stages:
        fn = STAGE_MAP[stage_name]
        fn(vid)

    print("\n" + "=" * 60)
    print(f"  DONE — Video {vid}")
    print(f"  Outputs in: {config.OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
