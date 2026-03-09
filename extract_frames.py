#!/usr/bin/env python3
"""
extract_frames.py

Reads all .avi video files from data/raw/, extracts every frame using OpenCV,
and saves them as JPEGs into data/frames/<video_name>/frame_XXXXX.jpg.

Usage:
    python extract_frames.py
"""

import os
import sys
import cv2
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
FRAMES_DIR = Path(__file__).parent / "data" / "frames"


def extract_frames_from_video(video_path: Path, output_dir: Path) -> int:
    """
    Extract all frames from a single video file.

    Returns the number of frames extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path.name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = output_dir / f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        frame_count += 1

    cap.release()
    return frame_count


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DIR.resolve()}")
        print("Run download_data.py --download first.")
        sys.exit(1)

    # Find all .avi files (case-insensitive)
    video_files = sorted(
        [f for f in RAW_DIR.iterdir() if f.suffix.lower() == ".avi"]
    )

    if not video_files:
        print(f"No .avi files found in {RAW_DIR.resolve()}")
        print("Make sure you have downloaded the dataset first.")
        sys.exit(1)

    print(f"Found {len(video_files)} .avi video(s) in {RAW_DIR.resolve()}\n")

    total_frames = 0
    for video_path in video_files:
        video_name = video_path.stem  # filename without extension
        output_dir = FRAMES_DIR / video_name

        print(f"Processing: {video_path.name} -> {output_dir.relative_to(FRAMES_DIR)}/")
        n_frames = extract_frames_from_video(video_path, output_dir)
        total_frames += n_frames
        print(f"  Extracted {n_frames} frames\n")

    print(f"{'='*50}")
    print(f"Total: {total_frames} frames from {len(video_files)} video(s)")
    print(f"Saved to: {FRAMES_DIR.resolve()}")


if __name__ == "__main__":
    main()
