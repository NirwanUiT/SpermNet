#!/usr/bin/env python3
"""
convert_annotations.py

Organise the VISEM-Tracking dataset into a standard YOLO training layout.

The VISEM-Tracking dataset already has YOLO-format labels:
    <class> <cx_norm> <cy_norm> <w_norm> <h_norm>
with 3 classes:
    0: sperm
    1: cluster
    2: small_pinhead

This script:
  1. Reads images/ and labels/ from each video in VISEM_Tracking_Train_v4/Train/
  2. Creates symlinks (to save disk space) in:
       data/annotations/images/{train,val,test}/
       data/annotations/labels/{train,val,test}/
  3. Splits by VIDEO (not frame) to prevent data leakage:
       Train: 14 videos, Val: 4 videos, Test: 2 videos
  4. Prints class distribution statistics

Usage:
    python convert_annotations.py
    python convert_annotations.py --copy   # copy files instead of symlink
"""

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Video-level split (70/20/10)
# ─────────────────────────────────────────────────────────────────────────────

# 20 videos: 11,12,13,14,15,19,21,22,23,24,29,30,35,36,38,47,52,54,60,82
# Split: 14 train / 4 val / 2 test
# Videos from the same patient must NOT be split across train/val.
# Each VISEM video ID = unique participant, so we can split freely.

TRAIN_VIDEOS = ["11", "12", "13", "15", "19", "21", "22", "24", "30", "35", "36", "38", "54", "82"]
VAL_VIDEOS   = ["14", "23", "47", "60"]
TEST_VIDEOS  = ["29", "52"]

CLASS_NAMES = {0: "sperm", 1: "cluster", 2: "small_pinhead"}


def get_split(video_id: str) -> str:
    """Return train/val/test for a given video ID."""
    if video_id in TRAIN_VIDEOS:
        return "train"
    elif video_id in VAL_VIDEOS:
        return "val"
    elif video_id in TEST_VIDEOS:
        return "test"
    else:
        return "train"  # fallback


def convert(use_copy: bool = False):
    """
    Organise VISEM annotations into YOLO training layout.
    """
    ann_dir = config.ANNOTATIONS_DIR
    visem_train = config.VISEM_ROOT

    if not visem_train.exists():
        print(f"ERROR: {visem_train} not found")
        sys.exit(1)

    # Create output directories
    for split in ("train", "val", "test"):
        (ann_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ann_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([d for d in visem_train.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} video directories in {visem_train}\n")

    total_images = 0
    total_labels = 0
    class_counter = Counter()
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_box_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for vdir in video_dirs:
        vid = vdir.name
        split = get_split(vid)

        images_dir = vdir / "images"
        labels_dir = vdir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"  SKIP {vid}: missing images/ or labels/")
            continue

        image_files = sorted(images_dir.glob("*.jpg"))
        label_files = sorted(labels_dir.glob("*.txt"))

        # Build label lookup
        label_map = {f.stem: f for f in label_files}

        n_linked = 0
        for img in image_files:
            frame_stem = img.stem  # e.g. "11_frame_0"

            # Image destination
            img_dst = ann_dir / "images" / split / img.name
            if not img_dst.exists():
                if use_copy:
                    shutil.copy2(img, img_dst)
                else:
                    img_dst.symlink_to(img.resolve())

            # Label destination
            lbl_src = label_map.get(frame_stem)
            lbl_dst = ann_dir / "labels" / split / f"{frame_stem}.txt"

            if lbl_src and not lbl_dst.exists():
                if use_copy:
                    shutil.copy2(lbl_src, lbl_dst)
                else:
                    lbl_dst.symlink_to(lbl_src.resolve())

                # Count classes
                with open(lbl_src) as f:
                    for line in f:
                        tokens = line.strip().split()
                        if tokens:
                            cls = int(tokens[0])
                            class_counter[cls] += 1
                            split_box_counts[split][cls] += 1

                total_labels += 1
            elif not lbl_src:
                # Empty label file (no annotations for this frame)
                if not lbl_dst.exists():
                    lbl_dst.write_text("")
                total_labels += 1

            total_images += 1
            n_linked += 1

        split_counts[split] += n_linked
        print(f"  {vid:>3}: {n_linked:>5} frames → {split}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images:  {total_images}")
    print(f"  Total labels:  {total_labels}")
    print(f"  Method:        {'copy' if use_copy else 'symlink'}")

    print(f"\n  Split breakdown:")
    for split in ("train", "val", "test"):
        print(f"    {split:>5}: {split_counts[split]:>6} images")

    print(f"\n  Class distribution (total bounding boxes):")
    for cls_id in sorted(class_counter.keys()):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        count = class_counter[cls_id]
        pct = 100 * count / sum(class_counter.values())
        print(f"    {cls_id} ({name:>14}): {count:>8}  ({pct:.1f}%)")

    print(f"\n  Per-split class distribution:")
    for split in ("train", "val", "test"):
        total_split = sum(split_box_counts[split].values())
        if total_split == 0:
            continue
        print(f"    {split}:")
        for cls_id in sorted(split_box_counts[split].keys()):
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            count = split_box_counts[split][cls_id]
            print(f"      {cls_id} ({name:>14}): {count:>8}")

    print(f"\n  Output: {ann_dir}")

    # Print first 10 rows of sample label file
    print(f"\n  Sample label file (first 10 lines of 11_frame_0.txt):")
    sample = ann_dir / "labels" / "train" / "11_frame_0.txt"
    if sample.exists():
        with open(sample) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    print(f"    ... ({total_labels} label files total)")
                    break
                print(f"    {line.rstrip()}")

    print(f"\n{'='*60}")
    return total_images, total_labels


def main():
    parser = argparse.ArgumentParser(description="Convert VISEM annotations to YOLO layout")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks")
    args = parser.parse_args()
    convert(use_copy=args.copy)


if __name__ == "__main__":
    main()
