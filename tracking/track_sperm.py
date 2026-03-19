#!/usr/bin/env python3
"""
tracking/track_sperm.py

Multi-object tracking of sperm cells using Ultralytics trackers.
Supports BoT-SORT (default), ByteTrack, and OC-SORT.

Two modes:
  • video  – run YOLO + tracker directly on an .avi file
  • frames – run YOLO + tracker on pre-extracted JPEG frames

Outputs per-track CSV:  track_id, frame, cx, cy, x1, y1, x2, y2, conf
Saved under outputs/tracks/<video_name>_<tracker>_tracks.csv

Usage:
    python -m tracking.track_sperm video  <path.avi>  [--model weights.pt] [--tracker botsort]
    python -m tracking.track_sperm frames <video_name> [--model weights.pt] [--tracker bytetrack]
    python -m tracking.track_sperm frames --all --tracker ocsort
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_tracker(tracker_type: str | None) -> str:
    """Return the tracker yaml filename, falling back to config default."""
    return tracker_type or config.TRACKER_TYPE


def _tracker_name(tracker_yaml: str) -> str:
    """Extract short name from yaml filename: 'botsort.yaml' → 'botsort'."""
    return Path(tracker_yaml).stem


def _save_tracks_csv(
    df: pd.DataFrame,
    video_name: str,
    tracker_yaml: str,
    frame_count: int,
) -> None:
    """Save tracks CSV with tracker name.  For backward compat, also save
    the plain ``{video_name}_tracks.csv`` when using botsort (the default)."""
    tname = _tracker_name(tracker_yaml)
    # Primary output: includes tracker name
    csv_primary = config.TRACK_OUT / f"{video_name}_{tname}_tracks.csv"
    df.to_csv(csv_primary, index=False)
    n_tracks = df["track_id"].nunique()
    print(f"Tracks saved → {csv_primary}  ({n_tracks} tracks, {len(df)} detections"
          f" over {frame_count} frames)")

    # Backward-compat alias when using the default tracker
    if tname == "botsort":
        csv_compat = config.TRACK_OUT / f"{video_name}_tracks.csv"
        df.to_csv(csv_compat, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core tracking
# ─────────────────────────────────────────────────────────────────────────────

def track_video(
    video_path: str | Path,
    model_path: str | None = None,
    save_video: bool = True,
    tracker_type: str | None = None,
) -> pd.DataFrame:
    """
    Run YOLO detection + tracking on a video file.

    Args:
        tracker_type: Tracker yaml filename (e.g. "bytetrack.yaml").
                      Defaults to config.TRACKER_TYPE when None.

    Returns a DataFrame with columns:
        track_id, frame, cx, cy, x1, y1, x2, y2, conf
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    tracker_yaml = _resolve_tracker(tracker_type)

    weights = model_path or config.YOLO_MODEL
    model = YOLO(weights)

    print(f"Tracking: {video_path.name}  (model={weights}, tracker={tracker_yaml})")

    results = model.track(
        source=str(video_path),
        tracker=tracker_yaml,
        imgsz=config.YOLO_IMGSZ,
        conf=config.YOLO_CONF,
        iou=config.YOLO_IOU,
        persist=config.TRACK_PERSIST,
        stream=True,
        verbose=False,
    )

    rows = []
    writer = None
    frame_idx = 0

    for r in results:
        if r.boxes is not None and r.boxes.id is not None:
            boxes   = r.boxes.xyxy.cpu().numpy()
            ids     = r.boxes.id.cpu().numpy().astype(int)
            confs   = r.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rows.append({
                    "track_id": tid,
                    "frame":    frame_idx,
                    "cx":       cx,
                    "cy":       cy,
                    "x1":       x1,
                    "y1":       y1,
                    "x2":       x2,
                    "y2":       y2,
                    "conf":     conf,
                })

        # Save annotated video
        if save_video:
            annotated = r.plot()
            if writer is None:
                h, w = annotated.shape[:2]
                vis_path = config.VIS_OUT / f"{video_name}_tracked.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(vis_path), fourcc, config.FPS, (w, h))
            writer.write(annotated)

        frame_idx += 1

    if writer:
        writer.release()
        print(f"Annotated video → {vis_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No tracks found.")
        return df

    # Save CSV
    _save_tracks_csv(df, video_name, tracker_yaml, frame_idx)

    return df


def track_from_frames(
    video_name: str,
    model_path: str | None = None,
    tracker_type: str | None = None,
) -> pd.DataFrame:
    """
    Run YOLO detection + tracking on pre-extracted frames.
    Frames are in data/frames/<video_name>/*.jpg

    Args:
        tracker_type: Tracker yaml filename (e.g. "bytetrack.yaml").
                      Defaults to config.TRACKER_TYPE when None.
    """
    frames_dir = config.FRAMES_DIR / video_name
    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found")
        return pd.DataFrame()

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print(f"No .jpg frames in {frames_dir}")
        return pd.DataFrame()

    tracker_yaml = _resolve_tracker(tracker_type)
    weights = model_path or config.YOLO_MODEL
    model = YOLO(weights)

    print(f"Tracking from frames: {video_name}  ({len(frame_files)} frames, "
          f"model={weights}, tracker={tracker_yaml})")

    rows = []
    writer = None

    for frame_idx, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        results = model.track(
            source=img,
            tracker=tracker_yaml,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            persist=config.TRACK_PERSIST,
            verbose=False,
        )
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes   = r.boxes.xyxy.cpu().numpy()
            ids     = r.boxes.id.cpu().numpy().astype(int)
            confs   = r.boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rows.append({
                    "track_id": tid,
                    "frame":    frame_idx,
                    "cx":       cx,
                    "cy":       cy,
                    "x1":       x1,
                    "y1":       y1,
                    "x2":       x2,
                    "y2":       y2,
                    "conf":     conf,
                })

        # Annotated video
        annotated = r.plot()
        if writer is None:
            h, w = annotated.shape[:2]
            vis_path = config.VIS_OUT / f"{video_name}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vis_path), fourcc, config.FPS, (w, h))
        writer.write(annotated)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Frame {frame_idx + 1}/{len(frame_files)} ...")

    if writer:
        writer.release()
        print(f"Annotated video → {vis_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No tracks found.")
        return df

    _save_tracks_csv(df, video_name, tracker_yaml, len(frame_files))
    return df


def track_all_frames(
    model_path: str | None = None,
    tracker_type: str | None = None,
):
    """Track all videos from pre-extracted frames."""
    video_dirs = sorted([d for d in config.FRAMES_DIR.iterdir() if d.is_dir()])
    if not video_dirs:
        print("No video frame directories found.")
        return
    for vdir in video_dirs:
        track_from_frames(vdir.name, model_path=model_path,
                          tracker_type=tracker_type)


def track_all_videos(
    model_path: str | None = None,
    tracker_type: str | None = None,
):
    """Track all .avi videos in data/raw/."""
    videos = sorted(config.RAW_DIR.glob("*.avi"))
    if not videos:
        print("No .avi files in data/raw/")
        return
    for v in videos:
        track_video(v, model_path=model_path, tracker_type=tracker_type)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sperm multi-object tracking")
    parser.add_argument("--tracker", type=str, default="botsort",
                        choices=list(config.TRACKER_CONFIGS.keys()),
                        help="Tracking algorithm (default: botsort)")
    sub = parser.add_subparsers(dest="command")

    # video
    vp = sub.add_parser("video", help="Track directly from .avi file(s)")
    vp.add_argument("path", nargs="?", help="Path to .avi file")
    vp.add_argument("--all", action="store_true", help="Track all .avi files in data/raw/")
    vp.add_argument("--model", type=str, default=None)

    # frames
    fp = sub.add_parser("frames", help="Track from extracted JPEG frames")
    fp.add_argument("name", nargs="?", help="Video name (folder under data/frames/)")
    fp.add_argument("--all", action="store_true", help="Process all videos")
    fp.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    # Resolve tracker yaml from short name
    tracker_yaml = config.TRACKER_CONFIGS[args.tracker]

    if args.command == "video":
        if args.all:
            track_all_videos(model_path=args.model, tracker_type=tracker_yaml)
        elif args.path:
            track_video(args.path, model_path=args.model,
                        tracker_type=tracker_yaml)
        else:
            parser.error("Provide a video path or --all")
    elif args.command == "frames":
        if args.all:
            track_all_frames(model_path=args.model, tracker_type=tracker_yaml)
        elif args.name:
            track_from_frames(args.name, model_path=args.model,
                              tracker_type=tracker_yaml)
        else:
            parser.error("Provide a video name or --all")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
