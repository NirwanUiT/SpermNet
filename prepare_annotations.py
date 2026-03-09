#!/usr/bin/env python3
"""
prepare_annotations.py

Convert VISEM-Tracking dataset annotations → YOLO format for training.

The VISEM-Tracking dataset provides bounding box annotations in various
formats. This script:
  1. Reads the annotation files from the extracted dataset
  2. Converts them to YOLO format (class cx cy w h — normalised)
  3. Creates train/val splits
  4. Writes to data/annotations/images/{train,val}/ and labels/{train,val}/

Usage:
    python prepare_annotations.py [--val-split 0.2]
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def find_annotation_files(raw_dir: Path) -> list[Path]:
    """
    Search for annotation files in the extracted dataset.
    VISEM-Tracking may store annotations as JSON, CSV, or TXT.
    """
    patterns = ["**/*.json", "**/*.csv", "**/*.txt"]
    ann_files = []
    for pat in patterns:
        ann_files.extend(raw_dir.glob(pat))
    return sorted(ann_files)


def parse_visem_annotations(ann_path: Path) -> list[dict]:
    """
    Parse a VISEM-Tracking annotation file.

    Returns list of dicts:
        [{"frame": int, "x": float, "y": float, "w": float, "h": float}, ...]
    """
    detections = []

    if ann_path.suffix == ".json":
        with open(ann_path) as f:
            data = json.load(f)
        # Handle different JSON structures
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    detections.append({
                        "frame": entry.get("frame", entry.get("frame_id", 0)),
                        "x": entry.get("x", entry.get("bbox_x", 0)),
                        "y": entry.get("y", entry.get("bbox_y", 0)),
                        "w": entry.get("w", entry.get("bbox_w", entry.get("width", 0))),
                        "h": entry.get("h", entry.get("bbox_h", entry.get("height", 0))),
                    })
        elif isinstance(data, dict):
            # Possible {"frames": [{...}, ...]} structure
            frames_data = data.get("frames", data.get("annotations", []))
            for entry in frames_data:
                bboxes = entry.get("bboxes", entry.get("objects", [entry]))
                frame_id = entry.get("frame", entry.get("frame_id", 0))
                for bb in bboxes:
                    detections.append({
                        "frame": frame_id,
                        "x": bb.get("x", bb.get("bbox_x", 0)),
                        "y": bb.get("y", bb.get("bbox_y", 0)),
                        "w": bb.get("w", bb.get("bbox_w", bb.get("width", 0))),
                        "h": bb.get("h", bb.get("bbox_h", bb.get("height", 0))),
                    })

    elif ann_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(ann_path)
        # Try common column naming conventions
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "frame" in cl:
                col_map["frame"] = col
            elif cl in ("x", "bbox_x", "left"):
                col_map["x"] = col
            elif cl in ("y", "bbox_y", "top"):
                col_map["y"] = col
            elif cl in ("w", "width", "bbox_w"):
                col_map["w"] = col
            elif cl in ("h", "height", "bbox_h"):
                col_map["h"] = col

        if all(k in col_map for k in ("x", "y", "w", "h")):
            for _, row in df.iterrows():
                detections.append({
                    "frame": int(row.get(col_map.get("frame", ""), 0)),
                    "x": float(row[col_map["x"]]),
                    "y": float(row[col_map["y"]]),
                    "w": float(row[col_map["w"]]),
                    "h": float(row[col_map["h"]]),
                })

    return detections


def convert_to_yolo(
    detections: list[dict],
    image_w: int,
    image_h: int,
    class_id: int = 0,
) -> dict[int, list[str]]:
    """
    Convert pixel-coordinate bounding boxes to YOLO format.

    Returns {frame_id: ["0 cx cy w h", ...], ...}
    """
    yolo_labels = {}
    for det in detections:
        frame = det["frame"]
        # Convert to YOLO normalised format
        cx = (det["x"] + det["w"] / 2) / image_w
        cy = (det["y"] + det["h"] / 2) / image_h
        w  = det["w"] / image_w
        h  = det["h"] / image_h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w  = max(0, min(1, w))
        h  = max(0, min(1, h))

        label = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        yolo_labels.setdefault(frame, []).append(label)

    return yolo_labels


def prepare_dataset(val_split: float = 0.2, seed: int = 42):
    """
    Full pipeline: find annotations, convert, split, write YOLO layout.
    """
    random.seed(seed)

    # Output structure
    for split in ("train", "val"):
        (config.ANNOTATIONS_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.ANNOTATIONS_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Iterate over video frame directories
    frame_dirs = sorted([d for d in config.FRAMES_DIR.iterdir() if d.is_dir()])
    if not frame_dirs:
        print("No extracted frame directories found. Run extract_frames.py first.")
        return

    total_images = 0
    total_labels = 0

    for fdir in frame_dirs:
        video_name = fdir.name
        print(f"\nProcessing: {video_name}")

        # Find matching annotation file
        ann_candidates = list(config.RAW_DIR.glob(f"**/*{video_name}*"))
        ann_file = None
        for ac in ann_candidates:
            if ac.suffix in (".json", ".csv", ".txt") and ac.is_file():
                ann_file = ac
                break

        # Get image dimensions from first frame
        frames = sorted(fdir.glob("*.jpg"))
        if not frames:
            print(f"  No frames found, skipping.")
            continue

        sample = cv2.imread(str(frames[0]))
        if sample is None:
            print(f"  Cannot read {frames[0].name}, skipping.")
            continue
        img_h, img_w = sample.shape[:2]

        # Parse annotations if available
        yolo_labels = {}
        if ann_file:
            print(f"  Annotation file: {ann_file.name}")
            detections = parse_visem_annotations(ann_file)
            if detections:
                yolo_labels = convert_to_yolo(detections, img_w, img_h)
                print(f"  Parsed {len(detections)} detections across {len(yolo_labels)} frames")
            else:
                print(f"  No detections parsed from {ann_file.name}")
        else:
            print(f"  No annotation file found for {video_name}")
            print(f"  Copying frames without labels (can be used for prediction)")

        # Split frames into train/val
        indices = list(range(len(frames)))
        random.shuffle(indices)
        val_count = max(1, int(len(frames) * val_split))
        val_indices = set(indices[:val_count])

        for idx, fpath in enumerate(frames):
            split = "val" if idx in val_indices else "train"
            frame_num = int(fpath.stem.split("_")[-1])

            # Copy image
            dest_img = config.ANNOTATIONS_DIR / "images" / split / f"{video_name}_{fpath.name}"
            shutil.copy2(fpath, dest_img)
            total_images += 1

            # Write label file (if annotations available for this frame)
            label_name = f"{video_name}_{fpath.stem}.txt"
            dest_lbl = config.ANNOTATIONS_DIR / "labels" / split / label_name

            if frame_num in yolo_labels:
                dest_lbl.write_text("\n".join(yolo_labels[frame_num]) + "\n")
                total_labels += 1
            else:
                # Empty label file (negative sample or unannotated)
                dest_lbl.write_text("")

    # Write dataset.yaml
    yaml_content = (
        f"path: {config.ANNOTATIONS_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {config.NUM_CLASSES}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    yaml_path = config.ANNOTATIONS_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\n{'='*50}")
    print(f"Dataset prepared:")
    print(f"  Images:  {total_images}")
    print(f"  Labels:  {total_labels}")
    print(f"  YAML:    {yaml_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO annotations")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(val_split=args.val_split, seed=args.seed)


if __name__ == "__main__":
    main()
