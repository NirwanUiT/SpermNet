#!/usr/bin/env python3
"""
run_pipeline.py

End-to-end sperm motility analysis pipeline orchestrator.

Stages:
  1. extract   – Extract frames from videos
  2. detect    – Run YOLO sperm detection
  3. track     – Multi-object tracking (BoT-SORT)
  4. events    – Compute WHO motility metrics & classify
  5. report    – Generate clinical semen analysis report
  6. visualise – Plot trajectory maps & motility charts

Usage:
    python run_pipeline.py --video <name>        # single video, all stages
    python run_pipeline.py --all                 # all videos, all stages
    python run_pipeline.py --video <name> --stages track events report
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Import pipeline modules
from detection.detect_sperm import predict_video_frames, predict_all as detect_all
from tracking.track_sperm import (
    track_video, track_from_frames, track_all_videos, track_all_frames,
)
from events.detect_events import analyse_video, analyse_all
from llm.analyze import analyse_and_report
from extract_frames import extract_frames_from_video


VALID_STAGES = ["extract", "detect", "track", "events", "report", "visualise"]

# Trained YOLO weights (from train_detector.py)
TRAINED_WEIGHTS = config.PROJECT_ROOT / "detection" / "weights" / "best.pt"
TRAINING_WEIGHTS = config.OUTPUTS_DIR / "training" / "sperm_yolov8n" / "weights" / "best.pt"


def _find_video_file(video_name: str) -> Path | None:
    """Find the video file for a given video name (supports .mp4, .avi, VISEM dirs)."""
    # Check VISEM dataset first (has both .mp4 and images/)
    visem_video = config.VISEM_ROOT / video_name / f"{video_name}.mp4"
    if visem_video.exists():
        return visem_video

    # Check raw directory for various formats
    for ext in (".mp4", ".avi", ".mkv", ".mov"):
        candidate = config.RAW_DIR / f"{video_name}{ext}"
        if candidate.exists():
            return candidate

    # Glob search
    candidates = list(config.RAW_DIR.glob(f"{video_name}*"))
    video_exts = {".mp4", ".avi", ".mkv", ".mov"}
    for c in candidates:
        if c.suffix.lower() in video_exts:
            return c

    return None


def _find_frames_dir(video_name: str) -> Path | None:
    """Find the frames/images directory for a given video."""
    # VISEM images directory
    visem_imgs = config.VISEM_ROOT / video_name / "images"
    if visem_imgs.exists():
        return visem_imgs

    # Extracted frames
    frames = config.FRAMES_DIR / video_name
    if frames.exists():
        return frames

    return None


def _get_model_path(model_path: str | None) -> str:
    """Resolve model path: user-specified > trained weights > base model."""
    if model_path:
        return model_path
    if TRAINED_WEIGHTS.exists():
        return str(TRAINED_WEIGHTS)
    if TRAINING_WEIGHTS.exists():
        return str(TRAINING_WEIGHTS)
    return config.YOLO_MODEL


def run_pipeline(
    video_name: str | None = None,
    stages: list[str] | None = None,
    model_path: str | None = None,
    use_llm: bool = False,
):
    """
    Run the pipeline for a single video or all videos.
    """
    stages = stages or VALID_STAGES

    print(f"\n{'='*60}")
    print("  SPERM MOTILITY ANALYSIS PIPELINE")
    print(f"{'='*60}")
    mode = f"Video: {video_name}" if video_name else "All videos"
    print(f"  Mode:   {mode}")
    print(f"  Stages: {', '.join(stages)}")
    print(f"{'='*60}\n")

    if video_name:
        _run_single(video_name, stages, model_path, use_llm)
    else:
        _run_all(stages, model_path, use_llm)


def _run_single(video_name: str, stages: list[str], model_path: str | None, use_llm: bool):
    """Pipeline for a single video."""
    resolved_model = _get_model_path(model_path)

    # Stage 1: Extract frames
    if "extract" in stages:
        print("\n▶ STAGE 1: Extract frames")
        frames_dir = _find_frames_dir(video_name)
        if frames_dir:
            n_frames = len(list(frames_dir.glob("*.jpg")))
            print(f"  Frames already available: {n_frames} in {frames_dir}")
            # Symlink into FRAMES_DIR if needed
            dst = config.FRAMES_DIR / video_name
            if not dst.exists() and frames_dir != dst:
                dst.symlink_to(frames_dir)
        else:
            video_file = _find_video_file(video_name)
            if video_file:
                out_dir = config.FRAMES_DIR / video_name
                n = extract_frames_from_video(video_file, out_dir)
                print(f"  Extracted {n} frames → {out_dir}")
            else:
                print(f"  No video or frames found for: {video_name}")
                print("  Skipping subsequent stages.")
                return

    # Stage 2: Detection
    if "detect" in stages:
        print("\n▶ STAGE 2: YOLO sperm detection")
        print(f"  Model: {resolved_model}")
        predict_video_frames(video_name, model_path=resolved_model)

    # Stage 3: Tracking
    if "track" in stages:
        print("\n▶ STAGE 3: Multi-object tracking")
        video_file = _find_video_file(video_name)
        if video_file:
            print(f"  Video: {video_file}")
            track_video(video_file, model_path=resolved_model)
        else:
            print(f"  No video file; tracking from frames...")
            track_from_frames(video_name, model_path=resolved_model)

    # Stage 4: Events / motility metrics
    if "events" in stages:
        print("\n▶ STAGE 4: Motility event analysis")
        analyse_video(video_name)

    # Stage 5: Report generation
    if "report" in stages:
        print("\n▶ STAGE 5: Report generation")
        report = analyse_and_report(video_name, use_llm=use_llm)
        if report:
            print(report[:500] + "..." if len(report) > 500 else report)

    # Stage 6: Visualisation
    if "visualise" in stages:
        print("\n▶ STAGE 6: Visualisation")
        try:
            from visualise import plot_trajectories, plot_motility_chart, plot_velocity_heatmap
            # Ensure frames dir is linked for background image
            frames_dir = _find_frames_dir(video_name)
            dst = config.FRAMES_DIR / video_name
            if frames_dir and not dst.exists() and frames_dir != dst:
                dst.symlink_to(frames_dir)
            plot_trajectories(video_name)
            plot_motility_chart(video_name)
            plot_velocity_heatmap(video_name)
        except ImportError:
            print("  Visualisation module not available. Skipping.")
        except Exception as e:
            print(f"  Visualisation error: {e}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete for: {video_name}")
    print(f"{'='*60}\n")


def _run_all(stages: list[str], model_path: str | None, use_llm: bool):
    """Pipeline for all available videos."""
    video_names = set()

    # 1. VISEM dataset video directories
    if config.VISEM_ROOT.exists():
        for d in config.VISEM_ROOT.iterdir():
            if d.is_dir():
                video_names.add(d.name)

    # 2. Video files in raw directory
    video_exts = {".mp4", ".avi", ".mkv", ".mov"}
    for f in config.RAW_DIR.iterdir():
        if f.suffix.lower() in video_exts:
            video_names.add(f.stem)

    # 3. Extracted frame directories
    if config.FRAMES_DIR.exists():
        for d in config.FRAMES_DIR.iterdir():
            if d.is_dir():
                video_names.add(d.name)

    if not video_names:
        print("No videos or extracted frames found. Download data first.")
        return

    video_list = sorted(video_names, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    print(f"  Found {len(video_list)} videos: {video_list}")

    for v in video_list:
        _run_single(v, stages, model_path, use_llm)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end sperm motility analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_pipeline.py --video sample_001
  python run_pipeline.py --all
  python run_pipeline.py --video sample_001 --stages track events report
  python run_pipeline.py --all --stages detect track --model runs/best.pt
""",
    )
    parser.add_argument("--video", type=str, help="Video name (stem, no extension)")
    parser.add_argument("--all", action="store_true", help="Process all videos")
    parser.add_argument(
        "--stages", nargs="+", choices=VALID_STAGES, default=VALID_STAGES,
        help=f"Pipeline stages to run (default: all). Options: {VALID_STAGES}",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to custom YOLO weights")
    parser.add_argument("--llm", action="store_true", help="Use LLM API for report (needs OPENAI_API_KEY)")
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Specify --video VIDEO_NAME or --all")

    run_pipeline(
        video_name=args.video if args.video else None,
        stages=args.stages,
        model_path=args.model,
        use_llm=args.llm,
    )


if __name__ == "__main__":
    main()
