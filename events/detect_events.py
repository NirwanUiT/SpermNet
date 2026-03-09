#!/usr/bin/env python3
"""
events/detect_events.py

Compute WHO-standard sperm motility metrics from tracking data and
classify each track into: progressive, non-progressive, or immotile.

Metrics computed per track:
  • VCL  – Curvilinear velocity (µm/s)
  • VSL  – Straight-line velocity (µm/s)
  • VAP  – Average path velocity (µm/s)  (5-frame rolling mean)
  • LIN  – Linearity  (VSL / VCL)
  • STR  – Straightness (VSL / VAP)
  • WOB  – Wobble (VAP / VCL)
  • ALH  – Amplitude of lateral head displacement (µm)
  • BCF  – Beat-cross frequency (Hz)

Usage:
    python -m events.detect_events <video_name>
    python -m events.detect_events --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper conversions
# ─────────────────────────────────────────────────────────────────────────────

def px_to_um(px: float) -> float:
    """Convert pixels to microns."""
    return px / config.PIXELS_PER_MICRON


def frame_interval() -> float:
    """Seconds between successive frames."""
    return 1.0 / config.FPS


# ─────────────────────────────────────────────────────────────────────────────
# Per-track metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_track_metrics(track_df: pd.DataFrame) -> dict:
    """
    Compute motility metrics for a single track.

    Parameters
    ----------
    track_df : DataFrame with columns (frame, cx, cy) sorted by frame.

    Returns
    -------
    dict of metric name → value
    """
    dt = frame_interval()

    xs = track_df["cx"].values.astype(float)
    ys = track_df["cy"].values.astype(float)
    n  = len(xs)

    if n < config.MIN_TRACK_LENGTH:
        return None  # too short to analyse

    # ── Point-to-point displacements (pixels) ────────────────────────────
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_dist = np.sqrt(dx**2 + dy**2)        # curvilinear step lengths
    step_dist_um = px_to_um(step_dist)

    # ── VCL: Curvilinear velocity ─────────────────────────────────────────
    vcl = np.mean(step_dist_um) / dt           # µm/s

    # ── VSL: Straight-line velocity ───────────────────────────────────────
    total_displacement_um = px_to_um(
        np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    )
    total_time = (n - 1) * dt
    vsl = total_displacement_um / total_time if total_time > 0 else 0.0

    # ── VAP: Average-path velocity (5-frame smoothed path) ───────────────
    window = min(5, n)
    xs_smooth = uniform_filter1d(xs, size=window)
    ys_smooth = uniform_filter1d(ys, size=window)
    smooth_dx = np.diff(xs_smooth)
    smooth_dy = np.diff(ys_smooth)
    smooth_dist_um = px_to_um(np.sqrt(smooth_dx**2 + smooth_dy**2))
    vap = np.mean(smooth_dist_um) / dt

    # ── Derived ratios ────────────────────────────────────────────────────
    lin = vsl / vcl if vcl > 0 else 0.0        # linearity
    str_ = vsl / vap if vap > 0 else 0.0       # straightness
    wob = vap / vcl if vcl > 0 else 0.0        # wobble

    # ── ALH: Amplitude of lateral head displacement ───────────────────────
    # Deviation of actual path from smoothed path
    lateral = np.sqrt((xs[:-1] - xs_smooth[:-1])**2 + (ys[:-1] - ys_smooth[:-1])**2)
    alh = px_to_um(np.mean(lateral))

    # ── BCF: Beat-cross frequency ─────────────────────────────────────────
    # Count zero-crossings of lateral displacement relative to smoothed path
    lateral_signed = (xs[:-1] - xs_smooth[:-1])  # use x-component
    crossings = np.sum(np.diff(np.sign(lateral_signed)) != 0)
    bcf = crossings / total_time if total_time > 0 else 0.0

    return {
        "track_id":  int(track_df["track_id"].iloc[0]),
        "n_frames":  n,
        "duration_s": total_time,
        "VCL":       round(vcl, 2),
        "VSL":       round(vsl, 2),
        "VAP":       round(vap, 2),
        "LIN":       round(lin, 3),
        "STR":       round(str_, 3),
        "WOB":       round(wob, 3),
        "ALH":       round(alh, 2),
        "BCF":       round(bcf, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Motility classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_motility(row: dict) -> str:
    """
    Classify a track as progressive / non_progressive / immotile
    based on WHO 2021 criteria (configurable in config.py).
    """
    vcl = row["VCL"]
    str_ = row["STR"]

    if vcl <= config.VCL_IMMOTILE_MAX:
        return "immotile"
    elif vcl >= config.VCL_PROGRESSIVE_MIN and str_ >= config.STR_PROGRESSIVE_MIN:
        return "progressive"
    else:
        return "non_progressive"


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis for one video
# ─────────────────────────────────────────────────────────────────────────────

def analyse_video(video_name: str) -> pd.DataFrame:
    """
    Load tracks CSV, compute per-track motility metrics, classify, save.
    """
    csv_path = config.TRACK_OUT / f"{video_name}_tracks.csv"
    if not csv_path.exists():
        print(f"ERROR: Track file not found: {csv_path}")
        print("Run tracking first:  python -m tracking.track_sperm ...")
        return pd.DataFrame()

    tracks_df = pd.read_csv(csv_path)
    if tracks_df.empty:
        print(f"Track file is empty: {csv_path}")
        return pd.DataFrame()

    track_ids = tracks_df["track_id"].unique()
    print(f"Analysing {len(track_ids)} tracks from {video_name} ...")

    metrics_list = []
    for tid in track_ids:
        track = tracks_df[tracks_df["track_id"] == tid].sort_values("frame")
        m = compute_track_metrics(track)
        if m is not None:
            m["motility"] = classify_motility(m)
            metrics_list.append(m)

    if not metrics_list:
        print("No tracks long enough to analyse.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_list)

    # ── Summary statistics ────────────────────────────────────────────────
    total = len(metrics_df)
    counts = metrics_df["motility"].value_counts()
    prog   = counts.get("progressive", 0)
    nonpro = counts.get("non_progressive", 0)
    immot  = counts.get("immotile", 0)

    print(f"\n{'='*50}")
    print(f"Motility Summary – {video_name}")
    print(f"{'='*50}")
    print(f"  Total analysed tracks:  {total}")
    print(f"  Progressive:            {prog:>4}  ({100*prog/total:.1f}%)")
    print(f"  Non-progressive:        {nonpro:>4}  ({100*nonpro/total:.1f}%)")
    print(f"  Immotile:               {immot:>4}  ({100*immot/total:.1f}%)")
    print(f"  Mean VCL:               {metrics_df['VCL'].mean():.1f} µm/s")
    print(f"  Mean VSL:               {metrics_df['VSL'].mean():.1f} µm/s")
    print(f"  Mean VAP:               {metrics_df['VAP'].mean():.1f} µm/s")
    print(f"{'='*50}\n")

    # Save
    out_csv = config.EVENTS_OUT / f"{video_name}_motility.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"Metrics saved → {out_csv}")

    # Save summary JSON
    summary = {
        "video": video_name,
        "total_tracks": total,
        "progressive": int(prog),
        "non_progressive": int(nonpro),
        "immotile": int(immot),
        "progressive_pct": round(100 * prog / total, 1),
        "non_progressive_pct": round(100 * nonpro / total, 1),
        "immotile_pct": round(100 * immot / total, 1),
        "mean_VCL": round(float(metrics_df["VCL"].mean()), 2),
        "mean_VSL": round(float(metrics_df["VSL"].mean()), 2),
        "mean_VAP": round(float(metrics_df["VAP"].mean()), 2),
        "mean_LIN": round(float(metrics_df["LIN"].mean()), 3),
        "mean_STR": round(float(metrics_df["STR"].mean()), 3),
    }
    import json
    summary_path = config.EVENTS_OUT / f"{video_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")

    return metrics_df


def analyse_all():
    """Analyse all tracked videos."""
    track_files = sorted(config.TRACK_OUT.glob("*_tracks.csv"))
    if not track_files:
        print("No track CSVs found. Run tracking first.")
        return

    for tf in track_files:
        video_name = tf.stem.replace("_tracks", "")
        analyse_video(video_name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sperm motility event detection")
    parser.add_argument("video", nargs="?", help="Video name")
    parser.add_argument("--all", action="store_true", help="Analyse all tracked videos")
    args = parser.parse_args()

    if args.all:
        analyse_all()
    elif args.video:
        analyse_video(args.video)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
