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


def classify_window(xs: np.ndarray, ys: np.ndarray, dt: float) -> str:
    """
    Classify a short positional window into a motility state.

    Used by Markov and temporal analyses for per-frame instantaneous
    motility classification via a sliding window over raw positions.

    Parameters
    ----------
    xs, ys : 1-D arrays of centroid positions (pixels).
    dt     : Seconds between consecutive frames (1/FPS).

    Returns
    -------
    One of 'Progressive', 'Non-progressive', 'Immotile'.
    """
    n = len(xs)
    if n < 3:
        return "Immotile"

    # VCL: mean step speed
    dx = np.diff(xs)
    dy = np.diff(ys)
    steps_um = px_to_um(np.sqrt(dx**2 + dy**2))
    vcl = np.mean(steps_um) / dt

    # VSL / VAP for STR
    disp_um = px_to_um(np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2))
    total_t = (n - 1) * dt
    vsl = disp_um / total_t if total_t > 0 else 0.0

    w = min(5, n)
    xs_s = uniform_filter1d(xs.astype(float), size=w)
    ys_s = uniform_filter1d(ys.astype(float), size=w)
    sdx = np.diff(xs_s)
    sdy = np.diff(ys_s)
    vap = np.mean(px_to_um(np.sqrt(sdx**2 + sdy**2))) / dt

    str_ = vsl / vap if vap > 0 else 0.0

    if vcl <= config.VCL_IMMOTILE_MAX:
        return "Immotile"
    elif vcl >= config.VCL_PROGRESSIVE_MIN and str_ >= config.STR_PROGRESSIVE_MIN:
        return "Progressive"
    else:
        return "Non-progressive"


# ─────────────────────────────────────────────────────────────────────────────
# Post-tracking quality filters
# ─────────────────────────────────────────────────────────────────────────────

# Filter thresholds (applied after compute_track_metrics, before classification)
_CONF_MIN        = 0.4    # minimum mean detection confidence per track
_VCL_MAX         = 200.0  # µm/s — anything higher is detection noise
_JITTER_VCL_MIN  = 20.0   # µm/s — VCL floor for the jitter check
_JITTER_LIN_MAX  = 0.02   # linearity ceiling for jitter tracks
_DURATION_MIN    = 0.3    # seconds — second duration gate (supplements MIN_TRACK_LENGTH)


def filter_tracks(
    metrics_list: list[dict],
    tracks_df: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """
    Remove spurious tracks before motility classification.

    Filters (a track is removed if it matches **any**):
      1. Low confidence  – mean detection conf < _CONF_MIN
      2. Unrealistic VCL – VCL > _VCL_MAX µm/s
      3. Jitter          – VCL > _JITTER_VCL_MIN and LIN < _JITTER_LIN_MAX
      4. Short duration  – duration_s < _DURATION_MIN

    Parameters
    ----------
    metrics_list : list of per-track metric dicts (from compute_track_metrics).
    tracks_df    : full tracking DataFrame with columns incl. track_id, conf.

    Returns
    -------
    (filtered_list, filter_stats)
        filtered_list – metrics dicts that survived all filters.
        filter_stats  – dict with counts per filter reason.
    """
    # Pre-compute mean confidence per track_id
    mean_conf = tracks_df.groupby("track_id")["conf"].mean()

    kept: list[dict] = []
    n_low_conf = 0
    n_high_vcl = 0
    n_jitter   = 0
    n_short    = 0

    for m in metrics_list:
        tid = m["track_id"]

        # 1. Low confidence
        if mean_conf.get(tid, 0.0) < _CONF_MIN:
            n_low_conf += 1
            continue

        # 2. Unrealistic velocity
        if m["VCL"] > _VCL_MAX:
            n_high_vcl += 1
            continue

        # 3. Jitter (high VCL, near-zero linearity)
        if m["VCL"] > _JITTER_VCL_MIN and m["LIN"] < _JITTER_LIN_MAX:
            n_jitter += 1
            continue

        # 4. Short duration
        if m["duration_s"] < _DURATION_MIN:
            n_short += 1
            continue

        kept.append(m)

    total_before = len(metrics_list)
    removed = total_before - len(kept)
    print(f"  Filtered: {removed}/{total_before} tracks removed "
          f"({n_low_conf} low-conf, {n_high_vcl} unrealistic-VCL, "
          f"{n_jitter} jitter, {n_short} short-duration)")

    stats = {
        "tracks_before_filter":   total_before,
        "tracks_after_filter":    len(kept),
        "filtered_low_conf":      n_low_conf,
        "filtered_high_vcl":      n_high_vcl,
        "filtered_jitter":        n_jitter,
        "filtered_short_duration": n_short,
    }
    return kept, stats


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis for one video
# ─────────────────────────────────────────────────────────────────────────────

def _load_detection_stats(video_name: str) -> dict | None:
    """
    Load detection JSON and compute average sperm detections per frame.

    Returns dict with keys: avg_detections_per_frame, total_sperm_boxes,
    n_frames_with_detections, n_frames_total.  Returns None if the file
    doesn't exist or is unusable.
    """
    import json as _json

    det_path = config.DETECT_OUT / f"{video_name}_detections.json"
    if not det_path.exists():
        return None

    try:
        with open(det_path) as f:
            data = _json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not read {det_path}: {e}")
        return None

    if not data:
        return None

    # Count class-0 (sperm) detections per frame
    sperm_per_frame = []
    for entry in data:
        n_sperm = sum(1 for b in entry.get("boxes", [])
                      if len(b) >= 6 and int(b[5]) == 0)
        sperm_per_frame.append(n_sperm)

    total_sperm = sum(sperm_per_frame)
    n_frames = len(sperm_per_frame)
    if n_frames == 0 or total_sperm == 0:
        return None

    return {
        "avg_detections_per_frame": total_sperm / n_frames,
        "total_sperm_boxes":       total_sperm,
        "n_frames_with_detections": sum(1 for c in sperm_per_frame if c > 0),
        "n_frames_total":          n_frames,
    }


def analyse_video(video_name: str) -> pd.DataFrame:
    """
    Load tracks CSV, compute per-track motility metrics, classify, save.

    If a detection JSON exists, estimates untracked (assumed immotile)
    sperm from the discrepancy between average detections per frame and
    number of unique tracks.  The motility CSV is unchanged (tracked
    sperm only); the summary JSON and printed output use adjusted counts.
    """
    import json

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
            metrics_list.append(m)

    if not metrics_list:
        print("No tracks long enough to analyse.")
        return pd.DataFrame()

    # ── Post-tracking quality filters ─────────────────────────────────────
    metrics_list, filter_stats = filter_tracks(metrics_list, tracks_df)

    if not metrics_list:
        print("All tracks removed by quality filters.")
        return pd.DataFrame()

    # ── Classify motility (after filtering) ───────────────────────────────
    for m in metrics_list:
        m["motility"] = classify_motility(m)

    metrics_df = pd.DataFrame(metrics_list)

    # ── Raw (tracked-only) counts ─────────────────────────────────────────
    total = len(metrics_df)
    counts = metrics_df["motility"].value_counts()
    prog   = counts.get("progressive", 0)
    nonpro = counts.get("non_progressive", 0)
    immot  = counts.get("immotile", 0)

    # ── Estimate untracked sperm from detection data ──────────────────────
    det_stats = _load_detection_stats(video_name)
    estimated_untracked = 0
    avg_det_per_frame = 0.0
    avg_active_tracks = 0.0
    untracked_fraction = 0.0  # fraction of FOV sperm that are untracked

    if det_stats is not None:
        avg_det_per_frame = det_stats["avg_detections_per_frame"]

        # Compare per-frame: avg detections vs avg active tracks
        frame_counts = tracks_df.groupby("frame")["track_id"].nunique()
        avg_active_tracks = frame_counts.mean()

        estimated_untracked = max(0, round(avg_det_per_frame - avg_active_tracks))
        if avg_det_per_frame > 0:
            untracked_fraction = max(0.0, (avg_det_per_frame - avg_active_tracks) / avg_det_per_frame)
        print(f"  Detection data: avg {avg_det_per_frame:.1f} sperm/frame, "
              f"avg {avg_active_tracks:.1f} active tracks/frame → "
              f"~{estimated_untracked} untracked/frame "
              f"({100*untracked_fraction:.1f}% assumed immotile)")
    else:
        print(f"  No detection JSON for {video_name} — using tracked-only counts")

    # ── Adjusted percentages ──────────────────────────────────────────────
    # Among all sperm in the FOV (detected), the tracked portion has known
    # motility; the untracked portion is assumed immotile.
    tracked_fraction = 1.0 - untracked_fraction
    adj_prog_pct   = round(100 * (prog / total) * tracked_fraction, 1) if total > 0 else 0.0
    adj_nonpro_pct = round(100 * (nonpro / total) * tracked_fraction, 1) if total > 0 else 0.0
    adj_immot_pct  = round(100.0 - adj_prog_pct - adj_nonpro_pct, 1)
    adj_total      = total + estimated_untracked
    adj_immot      = immot + estimated_untracked

    # ── Adjusted counts ───────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"Motility Summary – {video_name}")
    print(f"{'='*50}")
    print(f"  Tracked sperm:          {total}")
    if estimated_untracked > 0:
        print(f"  Estimated untracked:    {estimated_untracked}  (assumed immotile)")
        print(f"  Adjusted total:         {adj_total}")
    print(f"  Progressive:            {prog:>4}  ({adj_prog_pct}%)")
    print(f"  Non-progressive:        {nonpro:>4}  ({adj_nonpro_pct}%)")
    print(f"  Immotile (adjusted):    {adj_immot:>4}  ({adj_immot_pct}%)")
    print(f"  Mean VCL:               {metrics_df['VCL'].mean():.1f} µm/s")
    print(f"  Mean VSL:               {metrics_df['VSL'].mean():.1f} µm/s")
    print(f"  Mean VAP:               {metrics_df['VAP'].mean():.1f} µm/s")
    print(f"{'='*50}\n")

    # ── Save motility CSV (tracked sperm only — unchanged) ────────────────
    out_csv = config.EVENTS_OUT / f"{video_name}_motility.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"Metrics saved → {out_csv}")

    # ── Save summary JSON (both raw and adjusted) ─────────────────────────
    summary = {
        "video": video_name,
        # Quality-filter stats
        **filter_stats,
        # Raw tracked-only counts (post-filter)
        "total_tracks": total,
        "progressive": int(prog),
        "non_progressive": int(nonpro),
        "immotile": int(immot),
        "progressive_pct": round(100 * prog / total, 1),
        "non_progressive_pct": round(100 * nonpro / total, 1),
        "immotile_pct": round(100 * immot / total, 1),
        # Adjusted counts (with untracked-as-immotile correction)
        "estimated_untracked": estimated_untracked,
        "avg_detections_per_frame": round(avg_det_per_frame, 2),
        "avg_active_tracks_per_frame": round(avg_active_tracks, 2),
        "untracked_fraction": round(untracked_fraction, 4),
        "adjusted_total": adj_total,
        "adjusted_immotile": int(adj_immot),
        "adjusted_progressive_pct": adj_prog_pct,
        "adjusted_non_progressive_pct": adj_nonpro_pct,
        "adjusted_immotile_pct": adj_immot_pct,
        # Kinematic means (tracked sperm only)
        "mean_VCL": round(float(metrics_df["VCL"].mean()), 2),
        "mean_VSL": round(float(metrics_df["VSL"].mean()), 2),
        "mean_VAP": round(float(metrics_df["VAP"].mean()), 2),
        "mean_LIN": round(float(metrics_df["LIN"].mean()), 3),
        "mean_STR": round(float(metrics_df["STR"].mean()), 3),
    }
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
