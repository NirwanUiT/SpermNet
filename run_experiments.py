#!/usr/bin/env python3
"""
run_experiments.py

Orchestrate the full sperm-motility pipeline across multiple
detector + tracker combinations for systematic comparison.

Each experiment runs:
  1. YOLO / RT-DETR detection + multi-object tracking  → tracks CSV
  2. WHO motility event analysis                        → motility CSV + summary JSON

Results are written to per-experiment subdirectories:
  outputs/tracks/{experiment_name}/{video}_tracks.csv
  outputs/events/{experiment_name}/{video}_motility.csv
  outputs/events/{experiment_name}/{video}_summary.json

Usage:
    python run_experiments.py                                # all experiments, all videos
    python run_experiments.py --experiments yolov8n_botsort,yolov8l_botsort
    python run_experiments.py --videos 14,29,47
    python run_experiments.py --skip-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from tracking.track_sperm import track_from_frames
from events.detect_events import analyse_video


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configurations
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "name": "yolov8n_botsort",
        "weights": "outputs/training/sperm_yolov8n/weights/best.pt",
        "tracker": "botsort.yaml",
    },
    {
        "name": "yolov8n_bytetrack",
        "weights": "outputs/training/sperm_yolov8n/weights/best.pt",
        "tracker": "bytetrack.yaml",
    },
    {
        "name": "yolov8n_ocsort",
        "weights": "outputs/training/sperm_yolov8n/weights/best.pt",
        "tracker": "ocsort.yaml",
    },
    {
        "name": "yolov8l_botsort",
        "weights": "outputs/training/sperm_yolov8l/weights/best.pt",
        "tracker": "botsort.yaml",
    },
    {
        "name": "yolov8l_bytetrack",
        "weights": "outputs/training/sperm_yolov8l/weights/best.pt",
        "tracker": "bytetrack.yaml",
    },
    {
        "name": "yolov8l_ocsort",
        "weights": "outputs/training/sperm_yolov8l/weights/best.pt",
        "tracker": "ocsort.yaml",
    },
    {
        "name": "rtdetr-l_botsort",
        "weights": "outputs/training/sperm_rtdetr-l/weights/best.pt",
        "tracker": "botsort.yaml",
    },
    {
        "name": "rtdetr-l_bytetrack",
        "weights": "outputs/training/sperm_rtdetr-l/weights/best.pt",
        "tracker": "bytetrack.yaml",
    },
    {
        "name": "rtdetr-l_ocsort",
        "weights": "outputs/training/sperm_rtdetr-l/weights/best.pt",
        "tracker": "ocsort.yaml",
    },
    # ── YOLO11n ──────────────────────────────────────────────────────────
    {
        "name": "yolo11n_botsort",
        "weights": "outputs/training/sperm_yolo11n/weights/best.pt",
        "tracker": "botsort.yaml",
    },
    {
        "name": "yolo11n_bytetrack",
        "weights": "outputs/training/sperm_yolo11n/weights/best.pt",
        "tracker": "bytetrack.yaml",
    },
    {
        "name": "yolo11n_ocsort",
        "weights": "outputs/training/sperm_yolo11n/weights/best.pt",
        "tracker": "ocsort.yaml",
    },
    # ── YOLOv9t ──────────────────────────────────────────────────────────
    {
        "name": "yolov9t_botsort",
        "weights": "outputs/training/sperm_yolov9t/weights/best.pt",
        "tracker": "botsort.yaml",
    },
    {
        "name": "yolov9t_bytetrack",
        "weights": "outputs/training/sperm_yolov9t/weights/best.pt",
        "tracker": "bytetrack.yaml",
    },
    {
        "name": "yolov9t_ocsort",
        "weights": "outputs/training/sperm_yolov9t/weights/best.pt",
        "tracker": "ocsort.yaml",
    },
]


def _discover_videos() -> list[str]:
    """Return sorted list of video names from data/frames/."""
    return sorted(
        d.name for d in config.FRAMES_DIR.iterdir() if d.is_dir()
    )


def _run_single(
    exp: dict,
    video_name: str,
    skip_existing: bool = False,
) -> dict:
    """Run tracking + analysis for one (experiment, video) pair.

    Returns a status dict: {video, status, tracks, events, error}
    """
    exp_name = exp["name"]
    weights  = config.PROJECT_ROOT / exp["weights"]
    tracker  = exp["tracker"]

    tracks_dir = config.TRACK_OUT / exp_name
    events_dir = config.EVENTS_OUT / exp_name
    tracks_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    tracks_csv  = tracks_dir / f"{video_name}_tracks.csv"
    summary_json = events_dir / f"{video_name}_summary.json"

    # ── Skip check ────────────────────────────────────────────────────────
    if skip_existing and tracks_csv.exists() and summary_json.exists():
        return {"video": video_name, "status": "skipped",
                "tracks": str(tracks_csv), "events": str(summary_json),
                "error": None}

    # ── 1. Tracking ───────────────────────────────────────────────────────
    try:
        df = track_from_frames(
            video_name,
            model_path=str(weights),
            tracker_type=tracker,
        )
        if df.empty:
            return {"video": video_name, "status": "no_tracks",
                    "tracks": None, "events": None, "error": "No tracks produced"}

        # Save to experiment subdirectory
        df.to_csv(tracks_csv, index=False)

    except Exception as exc:
        return {"video": video_name, "status": "error",
                "tracks": None, "events": None, "error": f"Tracking: {exc}"}

    # ── 2. Motility analysis ──────────────────────────────────────────────
    try:
        analyse_video(
            video_name,
            tracks_dir=tracks_dir,
            events_dir=events_dir,
        )
    except Exception as exc:
        return {"video": video_name, "status": "error",
                "tracks": str(tracks_csv), "events": None,
                "error": f"Analysis: {exc}"}

    return {"video": video_name, "status": "ok",
            "tracks": str(tracks_csv), "events": str(summary_json),
            "error": None}


def run_experiments(
    experiment_names: list[str] | None = None,
    video_names: list[str] | None = None,
    skip_existing: bool = False,
) -> dict:
    """Run the full pipeline for each (experiment, video) combination.

    Parameters
    ----------
    experiment_names : Subset of experiment names to run.  ``None`` = all.
    video_names      : Subset of video IDs to process.  ``None`` = all.
    skip_existing    : Skip (experiment, video) pairs that already have output.

    Returns
    -------
    dict mapping experiment name → list of per-video status dicts.
    """
    # Resolve experiments
    if experiment_names is not None:
        exps = [e for e in EXPERIMENTS if e["name"] in experiment_names]
        unknown = set(experiment_names) - {e["name"] for e in exps}
        if unknown:
            print(f"WARNING: Unknown experiments ignored: {unknown}")
    else:
        exps = EXPERIMENTS

    # Resolve videos
    all_videos = _discover_videos()
    if video_names is not None:
        videos = [v for v in video_names if v in all_videos]
        unknown = set(video_names) - set(videos)
        if unknown:
            print(f"WARNING: Unknown videos ignored: {unknown}")
    else:
        videos = all_videos

    if not videos:
        print("No videos to process.")
        return {}

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT RUNNER — {len(exps)} experiments × {len(videos)} videos")
    print(f"{'='*70}\n")

    all_results: dict[str, list[dict]] = {}
    total_t0 = time.time()

    for exp in exps:
        exp_name = exp["name"]
        weights_path = config.PROJECT_ROOT / exp["weights"]

        # Check weights exist
        if not weights_path.exists():
            print(f"⚠  SKIPPING {exp_name}: weights not found at {weights_path}\n")
            all_results[exp_name] = [
                {"video": v, "status": "skipped_no_weights",
                 "tracks": None, "events": None,
                 "error": f"Weights missing: {weights_path}"}
                for v in videos
            ]
            continue

        print(f"━━━ Experiment: {exp_name}  (weights={exp['weights']}, "
              f"tracker={exp['tracker']}) ━━━")

        results = []
        for i, vid in enumerate(videos, 1):
            print(f"\n  [{i}/{len(videos)}] {exp_name} / video {vid}")
            t0 = time.time()
            res = _run_single(exp, vid, skip_existing=skip_existing)
            elapsed = time.time() - t0
            status_symbol = {"ok": "✓", "skipped": "⏭", "error": "✗",
                             "no_tracks": "⊘"}.get(res["status"], "?")
            print(f"  {status_symbol} {res['status']} ({elapsed:.1f}s)")
            if res["error"]:
                print(f"    Error: {res['error']}")
            results.append(res)

        all_results[exp_name] = results
        print()

    # ── Summary table ─────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY  (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  {'Experiment':<25} {'OK':>4} {'Skip':>5} {'Err':>4} {'NoTrk':>6} {'Total':>6}")
    print(f"  {'-'*50}")
    for exp_name, results in all_results.items():
        n_ok   = sum(1 for r in results if r["status"] == "ok")
        n_skip = sum(1 for r in results if r["status"] in ("skipped", "skipped_no_weights"))
        n_err  = sum(1 for r in results if r["status"] == "error")
        n_none = sum(1 for r in results if r["status"] == "no_tracks")
        print(f"  {exp_name:<25} {n_ok:>4} {n_skip:>5} {n_err:>4} {n_none:>6} {len(results):>6}")
    print(f"{'='*70}\n")

    # Save summary JSON
    summary_path = config.OUTPUTS_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results → {summary_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run detector+tracker experiments on VISEM-Tracking")
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated experiment names (default: all). "
             f"Available: {', '.join(e['name'] for e in EXPERIMENTS)}")
    parser.add_argument(
        "--videos", type=str, default=None,
        help="Comma-separated video IDs (default: all)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (experiment, video) pairs that already have output files")
    parser.add_argument(
        "--list", action="store_true",
        help="List available experiments and videos, then exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for e in EXPERIMENTS:
            w = config.PROJECT_ROOT / e["weights"]
            status = "✓" if w.exists() else "✗ (weights missing)"
            print(f"  {e['name']:<25} {status}")
        print(f"\nAvailable videos: {', '.join(_discover_videos())}")
        return

    exp_names = args.experiments.split(",") if args.experiments else None
    vid_names = args.videos.split(",") if args.videos else None

    run_experiments(
        experiment_names=exp_names,
        video_names=vid_names,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
