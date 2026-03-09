#!/usr/bin/env python3
"""
run_single.py

Run the full detection → tracking → motility → report pipeline
on ONE VISEM-Tracking video.

The VISEM dataset already has pre-extracted frames + YOLO labels under:
    data/raw/VISEM_Tracking_Train_v4/Train/<video_id>/
        images/    → JPEG frames
        labels/    → YOLO format (class cx cy w h)
        labels_ftid/ → YOLO + feature/track ID
        <id>.mp4   → original 30fps video

Usage:
    python run_single.py 11                     # video 11, all stages
    python run_single.py 11 --stages detect track events report
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Detection
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(video_id: str, model_path: str | None = None) -> list[dict]:
    """Run YOLOv8 on all frames of one VISEM video."""
    images_dir = config.VISEM_ROOT / video_id / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found")
        return []

    frame_files = sorted(images_dir.glob("*.jpg"))
    print(f"\n[DETECT] Video {video_id}: {len(frame_files)} frames")

    weights = model_path or config.YOLO_MODEL
    model = YOLO(weights)

    all_detections = []
    for idx, fpath in enumerate(frame_files):
        results = model.predict(
            source=str(fpath),
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            verbose=False,
        )
        r = results[0]
        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])
        all_detections.append({"frame": idx, "boxes": boxes})

        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(frame_files)} frames ...")

    # Save
    det_json = config.DETECT_OUT / f"{video_id}_detections.json"
    with open(det_json, "w") as f:
        json.dump(all_detections, f)
    total_boxes = sum(len(d["boxes"]) for d in all_detections)
    print(f"  → {total_boxes} total detections saved to {det_json}")
    return all_detections


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Tracking
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking(video_id: str, model_path: str | None = None) -> pd.DataFrame:
    """Run YOLO + BoT-SORT on the VISEM .mp4 video."""
    # Try video file first
    video_file = config.VISEM_ROOT / video_id / f"{video_id}.mp4"
    if not video_file.exists():
        # Try the 30s clips
        candidates = sorted(config.VISEM_VIDEOS_DIR.glob(f"{video_id}_*.mp4"))
        if candidates:
            video_file = candidates[0]
        else:
            print(f"ERROR: No video file found for {video_id}")
            return pd.DataFrame()

    weights = model_path or config.YOLO_MODEL
    model = YOLO(weights)

    print(f"\n[TRACK] Video {video_id}: {video_file.name}")
    results = model.track(
        source=str(video_file),
        tracker=config.TRACKER_TYPE,
        imgsz=config.YOLO_IMGSZ,
        conf=config.YOLO_CONF,
        iou=config.YOLO_IOU,
        persist=True,
        stream=True,
        verbose=False,
    )

    rows = []
    writer = None
    frame_idx = 0

    for r in results:
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rows.append({
                    "track_id": tid, "frame": frame_idx,
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf,
                })

        # Write annotated video
        annotated = r.plot()
        if writer is None:
            h, w = annotated.shape[:2]
            vis_path = config.VIS_OUT / f"{video_id}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vis_path), fourcc, config.FPS, (w, h))
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  Frame {frame_idx} ...")

    if writer:
        writer.release()
        print(f"  → Annotated video: {vis_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: No tracks found.")
        return df

    csv_path = config.TRACK_OUT / f"{video_id}_tracks.csv"
    df.to_csv(csv_path, index=False)
    n_tracks = df["track_id"].nunique()
    print(f"  → {n_tracks} tracks, {len(df)} detections → {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Motility analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_motility(video_id: str) -> pd.DataFrame:
    """Compute WHO motility metrics from tracks."""
    from events.detect_events import analyse_video
    print(f"\n[EVENTS] Video {video_id}")
    return analyse_video(video_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Report
# ─────────────────────────────────────────────────────────────────────────────

def run_report(video_id: str) -> str:
    """Generate semen analysis report."""
    from llm.analyze import analyse_and_report
    print(f"\n[REPORT] Video {video_id}")
    return analyse_and_report(video_id, use_llm=False)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def run_visualisation(video_id: str):
    """Generate plots."""
    from visualise import plot_trajectories, plot_motility_chart, plot_velocity_heatmap

    # Symlink VISEM images into frames dir so visualise.py can find them
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
    "detect": run_detection,
    "track":  run_tracking,
    "events": run_motility,
    "report": run_report,
    "vis":    run_visualisation,
}

def main():
    parser = argparse.ArgumentParser(description="Run pipeline on a single VISEM video")
    parser.add_argument("video_id", help="VISEM video ID (e.g. 11, 12, 13 ...)")
    parser.add_argument("--stages", nargs="+",
                        default=["detect", "track", "events", "report", "vis"],
                        choices=list(STAGE_MAP.keys()),
                        help="Stages to run")
    parser.add_argument("--model", type=str, default=None, help="Custom YOLO weights")
    args = parser.parse_args()

    vid = args.video_id

    # Verify video exists
    vid_dir = config.VISEM_ROOT / vid
    if not vid_dir.exists():
        avail = sorted([d.name for d in config.VISEM_ROOT.iterdir() if d.is_dir()])
        print(f"ERROR: Video '{vid}' not found. Available: {avail}")
        sys.exit(1)

    print("=" * 60)
    print(f"  SPERM ANALYSIS — Video {vid}")
    print(f"  Stages: {', '.join(args.stages)}")
    print("=" * 60)

    for stage_name in args.stages:
        fn = STAGE_MAP[stage_name]
        if stage_name in ("detect", "track"):
            fn(vid, model_path=args.model)
        else:
            fn(vid)

    print("\n" + "=" * 60)
    print(f"  DONE — Video {vid}")
    print(f"  Outputs: {config.OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
