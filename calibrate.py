#!/usr/bin/env python3
"""
calibrate.py

Fix the PIXELS_PER_MICRON and FPS calibration for the VISEM-Tracking dataset.

VISEM-Tracking microscopy setup (Thambawita et al., Scientific Data 2023):
  - Camera:        UEye UI-2210C
  - Microscope:    Olympus CX31
  - Magnification: 400x
  - Resolution:    640 × 480 pixels
  - Frame rate:    50 fps

At 400x magnification the field of view is approximately 450 µm wide.
  → pixels_per_micron = 640 / 450 ≈ 1.422

This script:
  1. Computes the correct PIXELS_PER_MICRON
  2. Updates config.py with the corrected values
  3. Re-runs WHO motility analysis on Video 11
  4. Prints before/after comparison

Usage:
    python calibrate.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compute correct calibration
# ─────────────────────────────────────────────────────────────────────────────

FRAME_WIDTH_PX   = 640          # pixels
FOV_WIDTH_UM     = 450.0        # µm at 400x magnification
CORRECT_PPM      = FRAME_WIDTH_PX / FOV_WIDTH_UM   # ≈ 1.422
CORRECT_FPS      = 50           # VISEM uses 50 fps, not 30

print("=" * 60)
print("  VISEM-Tracking Calibration")
print("=" * 60)
print(f"  Microscope:        Olympus CX31 @ 400x")
print(f"  Camera:            UEye UI-2210C")
print(f"  Resolution:        {FRAME_WIDTH_PX} × 480 pixels")
print(f"  Field of view:     {FOV_WIDTH_UM} µm (width)")
print(f"  Frame rate:        {CORRECT_FPS} fps")
print(f"  PIXELS_PER_MICRON: {CORRECT_PPM:.3f}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Read old config values and update config.py
# ─────────────────────────────────────────────────────────────────────────────

config_path = Path(__file__).resolve().parent / "config.py"
config_text = config_path.read_text()

# Extract old values
old_ppm_match = re.search(r"PIXELS_PER_MICRON\s*=\s*([\d.]+)", config_text)
old_fps_match = re.search(r"FPS\s*=\s*(\d+)", config_text)
old_ppm = float(old_ppm_match.group(1)) if old_ppm_match else 2.0
old_fps = int(old_fps_match.group(1)) if old_fps_match else 30

print(f"\n  Old PIXELS_PER_MICRON = {old_ppm}")
print(f"  New PIXELS_PER_MICRON = {CORRECT_PPM:.3f}")
print(f"  Old FPS = {old_fps}")
print(f"  New FPS = {CORRECT_FPS}")

# Update config.py
new_config = re.sub(
    r"PIXELS_PER_MICRON\s*=\s*[\d.]+\s*#.*",
    f"PIXELS_PER_MICRON = {CORRECT_PPM:.3f}          # 640px / 450µm FOV at 400x",
    config_text,
)
new_config = re.sub(
    r"FPS\s*=\s*\d+\s*#.*",
    f"FPS               = {CORRECT_FPS}                   # VISEM-Tracking: 50 fps",
    new_config,
)

# Also update CLASS_NAMES to match the 3 VISEM classes
new_config = re.sub(
    r'CLASS_NAMES\s*=\s*\[.*?\]\s*#.*',
    'CLASS_NAMES      = ["sperm", "cluster", "small_pinhead"]  # VISEM 3-class',
    new_config,
)

config_path.write_text(new_config)
print(f"\n  ✅ config.py updated")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Re-run motility analysis on Video 11 with corrected calibration
# ─────────────────────────────────────────────────────────────────────────────

# Reload config with new values
import importlib
import config
importlib.reload(config)

# Verify the values took effect
print(f"\n  Reloaded config:")
print(f"    PIXELS_PER_MICRON = {config.PIXELS_PER_MICRON}")
print(f"    FPS               = {config.FPS}")

# Load existing tracks
tracks_csv = config.TRACK_OUT / "11_tracks.csv"
if not tracks_csv.exists():
    print(f"\n  ERROR: {tracks_csv} not found. Run run_single_gt.py 11 first.")
    sys.exit(1)

tracks_df = pd.read_csv(tracks_csv)

# Compute metrics with OLD calibration
from events.detect_events import compute_track_metrics, classify_motility

def compute_all_metrics(df, ppm, fps):
    """Compute metrics for all tracks with given calibration."""
    # Temporarily override config values
    orig_ppm = config.PIXELS_PER_MICRON
    orig_fps = config.FPS
    config.PIXELS_PER_MICRON = ppm
    config.FPS = fps

    metrics = []
    for tid in df["track_id"].unique():
        track = df[df["track_id"] == tid].sort_values("frame")
        m = compute_track_metrics(track)
        if m is not None:
            m["motility"] = classify_motility(m)
            metrics.append(m)

    # Restore
    config.PIXELS_PER_MICRON = orig_ppm
    config.FPS = orig_fps
    return pd.DataFrame(metrics)


print("\n  Computing metrics with OLD calibration ...")
old_metrics = compute_all_metrics(tracks_df, old_ppm, old_fps)

print("  Computing metrics with NEW calibration ...")
new_metrics = compute_all_metrics(tracks_df, CORRECT_PPM, CORRECT_FPS)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Before/after comparison — 5 sample tracks
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("  BEFORE / AFTER COMPARISON — 5 sample tracks")
print(f"{'='*80}")

# Pick 5 tracks with different motility levels
sample_ids = new_metrics.sort_values("VCL", ascending=False).head(5)["track_id"].values

print(f"\n  {'Track':>6}  {'Old VCL':>10}  {'New VCL':>10}  {'Old VSL':>10}  "
      f"{'New VSL':>10}  {'Old Class':>16}  {'New Class':>16}")
print("  " + "-" * 85)

for tid in sample_ids:
    old_row = old_metrics[old_metrics["track_id"] == tid].iloc[0]
    new_row = new_metrics[new_metrics["track_id"] == tid].iloc[0]
    print(f"  {tid:>6}  {old_row['VCL']:>10.2f}  {new_row['VCL']:>10.2f}  "
          f"{old_row['VSL']:>10.2f}  {new_row['VSL']:>10.2f}  "
          f"{old_row['motility']:>16}  {new_row['motility']:>16}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. New motility classification breakdown
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("  MOTILITY CLASSIFICATION — BEFORE vs AFTER")
print(f"{'='*80}")

for label, df in [("OLD (PPM=2.0, FPS=30)", old_metrics),
                   ("NEW (PPM=1.422, FPS=50)", new_metrics)]:
    total = len(df)
    counts = df["motility"].value_counts()
    prog   = counts.get("progressive", 0)
    nonpro = counts.get("non_progressive", 0)
    immot  = counts.get("immotile", 0)
    print(f"\n  {label}")
    print(f"    Total tracks:     {total}")
    print(f"    Progressive:      {prog:>4}  ({100*prog/total:.1f}%)")
    print(f"    Non-progressive:  {nonpro:>4}  ({100*nonpro/total:.1f}%)")
    print(f"    Immotile:         {immot:>4}  ({100*immot/total:.1f}%)")
    print(f"    Total motility:   {100*(prog+nonpro)/total:.1f}%")
    print(f"    Mean VCL:         {df['VCL'].mean():.2f} µm/s")
    print(f"    Mean VSL:         {df['VSL'].mean():.2f} µm/s")

# WHO 2021 reference
print(f"\n  WHO 2021 references: total motility ≥ 42%, progressive ≥ 30%")

# Compare with ground truth from semen_analysis_data_Train.csv
gt_csv = config.RAW_DIR / "semen_analysis_data_Train.csv"
if gt_csv.exists():
    gt = pd.read_csv(gt_csv)
    v11 = gt[gt["ID"] == 11]
    if not v11.empty:
        row = v11.iloc[0]
        print(f"\n  Ground truth (semen_analysis_data_Train.csv) for Video 11:")
        print(f"    Progressive motility:     {row['Progressive motility (%)']}%")
        print(f"    Non-progressive motility: {row['Non progressive sperm motility (%)']}%")
        print(f"    Immotile:                 {row['Immotile sperm (%)']}%")
        print(f"    Concentration:            {row['Sperm concentration (x10⁶/mL)']} ×10⁶/mL")

# Save the new metrics
new_metrics.to_csv(config.EVENTS_OUT / "11_motility.csv", index=False)

summary = {
    "video": "11",
    "calibration": {"pixels_per_micron": CORRECT_PPM, "fps": CORRECT_FPS},
    "total_tracks": int(len(new_metrics)),
    "progressive": int(new_metrics["motility"].value_counts().get("progressive", 0)),
    "non_progressive": int(new_metrics["motility"].value_counts().get("non_progressive", 0)),
    "immotile": int(new_metrics["motility"].value_counts().get("immotile", 0)),
    "progressive_pct": round(100 * new_metrics["motility"].value_counts().get("progressive", 0) / len(new_metrics), 1),
    "non_progressive_pct": round(100 * new_metrics["motility"].value_counts().get("non_progressive", 0) / len(new_metrics), 1),
    "immotile_pct": round(100 * new_metrics["motility"].value_counts().get("immotile", 0) / len(new_metrics), 1),
    "mean_VCL": round(float(new_metrics["VCL"].mean()), 2),
    "mean_VSL": round(float(new_metrics["VSL"].mean()), 2),
    "mean_VAP": round(float(new_metrics["VAP"].mean()), 2),
    "mean_LIN": round(float(new_metrics["LIN"].mean()), 3),
    "mean_STR": round(float(new_metrics["STR"].mean()), 3),
}
with open(config.EVENTS_OUT / "11_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  ✅ Updated motility CSV → {config.EVENTS_OUT / '11_motility.csv'}")
print(f"  ✅ Updated summary JSON → {config.EVENTS_OUT / '11_summary.json'}")
print("=" * 60)
