#!/usr/bin/env python3
"""Sample-level feature sets A and B for the pre-registered DFI pilot (section 4.2).

Reads outputs/prereg/features_track.csv and produces ONE row per video with:
  Set A (WHO-3): raw tracked-only progressive/non-progressive/immotile percentages.
  Set B (24 kinematic-distribution features): quantiles/IQRs of VCL/LIN/ALH/BCF,
        PWR p90+IQR, TAC med+IQR, VAR_V med, vigorous/intermediate fractions,
        log(#tracks).
Plus diagnostics (DUR median, n_tracks) that are NEVER predictors.

Set C (= A + B) and Set B-orthogonal are derived inside predict_clinical.py.
NO clinical columns are read here.

Output: outputs/prereg/features_sample.csv
Usage:  python -m experiments.features_sample
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "outputs" / "prereg" / "features_track.csv"
OUT = ROOT / "outputs" / "prereg" / "features_sample.csv"

QUANT_METRICS = ["VCL", "LIN", "ALH", "BCF"]

# Column groupings consumed by predict_clinical.py.
SET_A = ["prog_pct", "nonprog_pct", "immot_pct"]


def iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def sample_features(g: pd.DataFrame) -> dict:
    row: dict = {}

    # Set A - raw tracked-only WHO composition.
    n = len(g)
    mc = g["motility"].value_counts()
    row["prog_pct"] = 100.0 * mc.get("progressive", 0) / n
    row["nonprog_pct"] = 100.0 * mc.get("non_progressive", 0) / n
    row["immot_pct"] = 100.0 * mc.get("immotile", 0) / n

    # Set B - kinematic distribution.
    for m in QUANT_METRICS:
        v = g[m].values
        row[f"{m}_p10"] = float(np.percentile(v, 10))
        row[f"{m}_p50"] = float(np.percentile(v, 50))
        row[f"{m}_p90"] = float(np.percentile(v, 90))
        row[f"{m}_iqr"] = iqr(v)
    row["PWR_p90"] = float(np.percentile(g["PWR"].values, 90))
    row["PWR_iqr"] = iqr(g["PWR"].values)
    row["TAC_med"] = float(np.median(g["TAC"].values))
    row["TAC_iqr"] = iqr(g["TAC"].values)
    row["VAR_V_med"] = float(np.median(g["VAR_V"].values))
    row["vigorous_frac"] = float(np.mean(g["VCL"].values > 50.0))
    row["intermediate_frac"] = float(np.mean((g["VCL"].values >= 5.0) & (g["VCL"].values <= 25.0)))
    row["log_ntracks"] = float(np.log(n))

    # Diagnostics (never predictors).
    row["DUR_med"] = float(np.median(g["DUR"].values))
    row["n_tracks"] = n
    return row


def main() -> None:
    df = pd.read_csv(IN)
    rows = []
    for vid, g in df.groupby("video"):
        r = {"video": int(vid)}
        r.update(sample_features(g))
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("video").reset_index(drop=True)
    set_b = [c for c in out.columns
             if c not in ["video", *SET_A, "DUR_med", "n_tracks"]]
    assert len(set_b) == 24, f"Set B must be 24 features, got {len(set_b)}"
    out.to_csv(OUT, index=False)
    print(f"{len(out)} videos | Set A={len(SET_A)} Set B={len(set_b)} features")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
