#!/usr/bin/env python3
"""
detection/detect_sperm.py

Sperm detection using YOLOv8.
  • Train   – fine-tune YOLOv8 on VISEM-Tracking annotations
  • Predict – run inference on extracted frames and save detections

Usage:
    python -m detection.detect_sperm train
    python -m detection.detect_sperm predict [--video VIDEO_NAME]
    python -m detection.detect_sperm predict --all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Allow running as `python -m detection.detect_sperm`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset_yaml(output_path: Path | None = None) -> Path:
    """
    Generate a YOLO-format dataset.yaml pointing to our annotations directory.

    Expected layout under data/annotations/:
        images/
            train/
            val/
        labels/
            train/
            val/
    """
    yaml_path = output_path or config.ANNOTATIONS_DIR / "dataset.yaml"

    content = (
        f"path: {config.ANNOTATIONS_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {config.NUM_CLASSES}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    yaml_path.write_text(content)
    print(f"Dataset YAML written to {yaml_path}")
    return yaml_path


def train(resume: bool = False):
    """Fine-tune YOLOv8 on the sperm dataset."""
    yaml_path = create_dataset_yaml()

    model = YOLO(config.YOLO_MODEL)
    results = model.train(
        data=str(yaml_path),
        epochs=config.YOLO_EPOCHS,
        imgsz=config.YOLO_IMGSZ,
        batch=config.YOLO_BATCH,
        project=str(config.OUTPUTS_DIR / "training"),
        name="sperm_yolov8",
        exist_ok=True,
        resume=resume,
        verbose=True,
    )
    print("Training complete.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_video_frames(
    video_name: str,
    model_path: str | None = None,
    save_visualisations: bool = True,
) -> list[dict]:
    """
    Run YOLO detection on all extracted frames of a single video.

    Returns a list of per-frame detection dicts:
        [{"frame": 0, "boxes": [[x1,y1,x2,y2,conf,cls], ...]}, ...]
    """
    frames_dir = config.FRAMES_DIR / video_name
    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        return []

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return []

    # Load model
    weights = model_path or config.YOLO_MODEL
    model = YOLO(weights)

    all_detections = []
    vis_dir = config.VIS_OUT / "detections" / video_name
    if save_visualisations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running detection on {len(frame_files)} frames from '{video_name}' ...")

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
            cls  = int(box.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])

        all_detections.append({"frame": idx, "boxes": boxes})

        # Save annotated image
        if save_visualisations:
            annotated = r.plot()
            cv2.imwrite(str(vis_dir / fpath.name), annotated)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(frame_files)} frames ...")

    # Save detections JSON
    det_json = config.DETECT_OUT / f"{video_name}_detections.json"
    with open(det_json, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"Detections saved → {det_json}  ({sum(len(d['boxes']) for d in all_detections)} total boxes)")

    return all_detections


def predict_all(model_path: str | None = None):
    """Run detection on every video's extracted frames."""
    if not config.FRAMES_DIR.exists():
        print(f"ERROR: {config.FRAMES_DIR} does not exist. Run extract_frames.py first.")
        return

    video_dirs = sorted([d for d in config.FRAMES_DIR.iterdir() if d.is_dir()])
    if not video_dirs:
        print("No video frame directories found.")
        return

    for vdir in video_dirs:
        predict_video_frames(vdir.name, model_path=model_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO sperm detection")
    sub = parser.add_subparsers(dest="command")

    # train
    tr = sub.add_parser("train", help="Fine-tune YOLOv8 on sperm data")
    tr.add_argument("--resume", action="store_true")

    # predict
    pr = sub.add_parser("predict", help="Run inference on extracted frames")
    pr.add_argument("--video", type=str, help="Video name (folder under data/frames/)")
    pr.add_argument("--all", action="store_true", help="Process all videos")
    pr.add_argument("--model", type=str, default=None, help="Path to custom weights")

    args = parser.parse_args()

    if args.command == "train":
        train(resume=args.resume)
    elif args.command == "predict":
        if args.all:
            predict_all(model_path=args.model)
        elif args.video:
            predict_video_frames(args.video, model_path=args.model)
        else:
            parser.error("Specify --video VIDEO_NAME or --all")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
