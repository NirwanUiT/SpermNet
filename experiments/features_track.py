#!/usr/bin/env python3
"""Per-track single-cell kinematic features for the pre-registered DFI pilot.

Implements Table 4.1 of paper/prereg_analysis_plan.md on the FROZEN orig20 cohort
(hand-annotated VISEM-Tracking, outputs/tracks/{id}_tracks.csv). Reuses the frozen
pipeline exactly: compute_track_metrics -> filter_tracks -> classify_motility.

Adds PWR = ALH*BCF, TAC = circular SD of turning angles, VAR_V = CV of instantaneous
speed, DUR = duration (diagnostic only, never a predictor).

NO clinical columns are read here (blinding by ordering, plan section 9).

Output: outputs/prereg/features_track.csv (one row per retained track).
Usage:  python -m experiments.features_track
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from events.detect_events import (  # noqa: E402
    compute_track_metrics, filter_tracks, classify_motility,
    px_to_um, frame_interval,
)

OUT_DIR = ROOT / "outputs" / "prereg"
OUT = OUT_DIR / "features_track.csv"

# Frozen orig20 cohort (plain hand-annotated track files only).
PIDS = [11, 12, 13, 14, 15, 19, 21, 22, 23, 24,
        29, 30, 35, 36, 38, 47, 52, 54, 60, 82]


def _extra_metrics(track_df: pd.DataFrame) -> dict:
    """PWR/TAC/VAR_V from raw positions (base metrics come from the frozen fn)."""
    dt = frame_interval()
    xs = track_df["cx"].values.astype(float)
    ys = track_df["cy"].values.astype(float)
    dx = np.diff(xs)
    dy = np.diff(ys)

    # Instantaneous speed (µm/s) and its coefficient of variation.
    step_um = px_to_um(np.sqrt(dx**2 + dy**2))
    v = step_um / dt
    var_v = float(np.std(v) / np.mean(v)) if np.mean(v) > 0 else 0.0

    # Turning angles between successive displacement vectors -> circular SD.
    headings = np.arctan2(dy, dx)
    if len(headings) >= 2:
        turn = np.diff(headings)
        turn = np.arctan2(np.sin(turn), np.cos(turn))  # wrap to (-pi, pi]
        R = np.abs(np.mean(np.exp(1j * turn)))
        tac = float(np.sqrt(-2.0 * np.log(R))) if R > 1e-12 else float(np.pi)
    else:
        tac = 0.0
    return {"TAC": round(tac, 4), "VAR_V": round(var_v, 4)}


def process_video(pid: int) -> pd.DataFrame:
    csv_path = config.TRACK_OUT / f"{pid}_tracks.csv"
    df = pd.read_csv(csv_path)
    metrics, extra = [], {}
    for tid in df["track_id"].unique():
        trk = df[df["track_id"] == tid].sort_values("frame")
        m = compute_track_metrics(trk)
        if m is None:
            continue
        metrics.append(m)
        extra[m["track_id"]] = _extra_metrics(trk)

    kept, stats = filter_tracks(metrics, df)
    rows = []
    for m in kept:
        m["motility"] = classify_motility(m)
        e = extra[m["track_id"]]
        rows.append({
            "video": pid,
            "track_id": m["track_id"],
            "n_frames": m["n_frames"],
            "DUR": m["duration_s"],
            "VCL": m["VCL"], "VSL": m["VSL"], "VAP": m["VAP"],
            "LIN": m["LIN"], "STR": m["STR"], "WOB": m["WOB"],
            "ALH": m["ALH"], "BCF": m["BCF"],
            "PWR": round(m["ALH"] * m["BCF"], 3),
            "TAC": e["TAC"], "VAR_V": e["VAR_V"],
            "motility": m["motility"],
        })
    print(f"  video {pid}: {stats['tracks_after_filter']} kept "
          f"/ {stats['tracks_before_filter']} (removed {stats['tracks_before_filter']-stats['tracks_after_filter']})")
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for pid in PIDS:
        f = process_video(pid)
        if len(f) < 10:
            print(f"  WARNING video {pid}: <10 usable tracks ({len(f)}) -> excluded")
            continue
        frames.append(f)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT, index=False)
    print(f"\n{len(all_df)} tracks across {all_df['video'].nunique()} videos")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
