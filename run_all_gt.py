#!/usr/bin/env python3
"""
run_all_gt.py

Process ALL 20 VISEM-Tracking videos using ground-truth annotations.

Stages per video:
  1. Parse GT labels_ftid → track CSV
  2. Compute WHO motility metrics
  3. Generate clinical report
  4. Create visualisations (trajectories, motility chart, velocity heatmap)

After all videos, runs evaluation against clinical ground truth.

Usage:
    python run_all_gt.py
    python run_all_gt.py --videos 11 12 13
    python run_all_gt.py --skip-vis           # skip slow visualisation stage
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from run_single_gt import parse_gt_tracks, run_motility, run_report, run_visualisation


# All 20 VISEM Training videos
ALL_VIDEOS = [
    "11", "12", "13", "14", "15", "19", "21", "22", "23", "24",
    "29", "30", "35", "36", "38", "47", "52", "54", "60", "82",
]


def process_video(vid: str, skip_vis: bool = False) -> dict:
    """Process a single video and return timing info."""
    t0 = time.time()
    result = {"video": vid, "status": "ok"}

    try:
        # Stage 1: GT tracks
        print(f"\n{'─'*60}")
        print(f"  VIDEO {vid}")
        print(f"{'─'*60}")
        df = parse_gt_tracks(vid)
        if df.empty:
            result["status"] = "no_tracks"
            return result

        # Stage 2: Motility
        run_motility(vid)

        # Stage 3: Report
        run_report(vid)

        # Stage 4: Vis
        if not skip_vis:
            run_visualisation(vid)

    except Exception as e:
        result["status"] = f"error: {e}"
        print(f"  ERROR processing video {vid}: {e}")

    result["time_s"] = round(time.time() - t0, 1)
    return result


def aggregate_results(videos: list[str]):
    """
    Combine per-video summary JSONs into one aggregate CSV + summary.
    """
    rows = []
    for vid in videos:
        summary_path = config.EVENTS_OUT / f"{vid}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            rows.append(s)

    if not rows:
        print("\nNo summary files found to aggregate.")
        return None

    agg = pd.DataFrame(rows)

    # Save aggregate CSV
    agg_path = config.OUTPUTS_DIR / "aggregate_motility.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\nAggregate motility CSV → {agg_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("  AGGREGATE MOTILITY RESULTS — ALL VIDEOS")
    print(f"{'='*80}")
    print(f"{'Video':>6} {'Tracks':>7} {'Prog%':>7} {'NonP%':>7} {'Imm%':>7} "
          f"{'VCL':>7} {'VSL':>7} {'VAP':>7}")
    print(f"{'─'*6:>6} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} "
          f"{'─'*7:>7} {'─'*7:>7} {'─'*7:>7}")
    for _, r in agg.iterrows():
        print(f"{r['video']:>6} {r['total_tracks']:>7} "
              f"{r.get('progressive_pct', 0):>7.1f} "
              f"{r.get('non_progressive_pct', 0):>7.1f} "
              f"{r.get('immotile_pct', 0):>7.1f} "
              f"{r.get('mean_VCL', 0):>7.1f} "
              f"{r.get('mean_VSL', 0):>7.1f} "
              f"{r.get('mean_VAP', 0):>7.1f}")

    print(f"{'─'*6:>6} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} "
          f"{'─'*7:>7} {'─'*7:>7} {'─'*7:>7}")
    print(f"{'MEAN':>6} {agg['total_tracks'].mean():>7.0f} "
          f"{agg['progressive_pct'].mean():>7.1f} "
          f"{agg['non_progressive_pct'].mean():>7.1f} "
          f"{agg['immotile_pct'].mean():>7.1f} "
          f"{agg['mean_VCL'].mean():>7.1f} "
          f"{agg['mean_VSL'].mean():>7.1f} "
          f"{agg['mean_VAP'].mean():>7.1f}")
    print(f"{'='*80}")

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Process all VISEM videos with ground-truth annotations"
    )
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific video IDs to process (default: all 20)")
    parser.add_argument("--skip-vis", action="store_true",
                        help="Skip visualisation stage for faster processing")
    parser.add_argument("--skip-research", action="store_true",
                        help="Skip Markov and temporal research analyses")
    args = parser.parse_args()

    videos = args.videos or ALL_VIDEOS

    # Validate
    avail = sorted([d.name for d in config.VISEM_ROOT.iterdir() if d.is_dir()])
    for v in videos:
        if v not in avail:
            print(f"WARNING: video {v} not in dataset. Available: {avail}")
            videos.remove(v)

    print("=" * 60)
    print("  BATCH PROCESSING — VISEM-Tracking Ground Truth")
    print("=" * 60)
    print(f"  Videos:   {len(videos)}")
    print(f"  Skip vis: {args.skip_vis}")
    print(f"  Skip research: {args.skip_research}")
    print(f"  Output:   {config.OUTPUTS_DIR}")
    print("=" * 60)

    t_start = time.time()
    results = []

    for i, vid in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing video {vid} ...")
        r = process_video(vid, skip_vis=args.skip_vis)
        results.append(r)
        print(f"  → {r['status']} ({r.get('time_s', '?')}s)")

    total_time = time.time() - t_start

    # Aggregate
    agg = aggregate_results(videos)

    # Processing summary
    print(f"\n{'='*60}")
    print("  PROCESSING COMPLETE")
    print(f"{'='*60}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  Processed: {ok}/{len(videos)} videos")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    for r in results:
        if r["status"] != "ok":
            print(f"  FAILED: video {r['video']} — {r['status']}")
    print(f"  Outputs in: {config.OUTPUTS_DIR}")
    print(f"{'='*60}\n")

    # Run evaluation if aggregate exists
    if agg is not None:
        try:
            from evaluate import evaluate_against_gt
            evaluate_against_gt()
        except ImportError:
            print("evaluate.py not found. Skipping GT comparison.")
        except Exception as e:
            print(f"Evaluation error: {e}")

    # Research analyses (Markov + temporal)
    if not args.skip_research:
        print(f"\n{'='*60}")
        print("  RESEARCH ANALYSES")
        print(f"{'='*60}")

        try:
            from markov_analysis import run_markov_analysis
            run_markov_analysis()
        except Exception as e:
            print(f"  Markov analysis error: {e}")

        try:
            from temporal_analysis import run_temporal_analysis
            run_temporal_analysis()
        except Exception as e:
            print(f"  Temporal analysis error: {e}")
    else:
        print("\n  Skipping research analyses (--skip-research)")


if __name__ == "__main__":
    main()
