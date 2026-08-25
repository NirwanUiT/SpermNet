"""Homogeneous continuum null -- mechanism attribution for the per-track
null's g2 (referee round 2; prereg 7f30aff).

The per-track OU null carries quenched cell-to-cell heterogeneity (each track
gets its own mu, var, a). Its block-25 g2 of +0.052 could therefore be
(a) discretisation-intrinsic memory, or (b) mover-stayer aggregation from
pooling tracks with different quenched parameters. This null removes the
heterogeneity: ONE pooled OU parameter set for every track (grand velocity
component means/variances over all steps of all GT tracks, plus globally
pooled lag-1 autocovariance), same track lengths as GT, identical classifier
and scoring.

Pre-registered adjudication (paper/prereg_recalibration.md):
  g2_hom < 25% of +0.052  -> per-track null's g2 is aggregation; rewrite the
                             mechanism sentences (heterogeneity, not
                             discretisation alone, manufactures null g2)
  g2_hom >= 50% of +0.052 -> discretisation-intrinsic memory; new account needed
  in between              -> both mechanisms, report shares

Output: outputs/markov/homogeneous_null.json
Usage:  python -m experiments.homogeneous_null [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from experiments.continuum_null import simulate_ou  # noqa: E402
from experiments.gt_reanchor import GT_DIR, analyse_dir  # noqa: E402

HOM_DIR = ROOT / "outputs" / "tracks_continuum_null_hom"
OUT = config.MARKOV_OUT / "homogeneous_null.json"


def fit_pooled() -> dict:
    """One OU parameter set from ALL steps of ALL GT tracks."""
    vx_all, vy_all = [], []
    num, den = 0.0, 0.0
    for tf in sorted(GT_DIR.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        for _, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            if len(tr) < 3:
                continue
            vx = np.diff(tr["cx"].to_numpy(float))
            vy = np.diff(tr["cy"].to_numpy(float))
            vx_all.append(vx)
            vy_all.append(vy)
    vx = np.concatenate(vx_all)
    vy = np.concatenate(vy_all)
    mu = np.array([vx.mean(), vy.mean()])
    var = np.array([vx.var(), vy.var()])
    # pooled lag-1: centre per component with the GRAND mean (homogeneous model)
    for track_vx, track_vy in zip(vx_all, vy_all):
        cx, cy = track_vx - mu[0], track_vy - mu[1]
        num += float(np.dot(cx[1:], cx[:-1]) + np.dot(cy[1:], cy[:-1]))
        den += float(np.dot(cx, cx) + np.dot(cy, cy))
    a = float(np.clip(num / den if den > 0 else 0.0, -0.995, 0.995))
    return {"mu": mu, "var": var, "a": a}


def materialise(fit: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    HOM_DIR.mkdir(parents=True, exist_ok=True)
    n_tracks = 0
    for tf in sorted(GT_DIR.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        rows = []
        for tid, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            n = len(tr)
            xs = tr["cx"].to_numpy(float)
            ys = tr["cy"].to_numpy(float)
            n_tracks += 1
            if n < 3:
                sx, sy = xs - xs[0], ys - ys[0]
            else:
                sx, sy = simulate_ou(n, fit, rng)
            sx, sy = sx + xs[0], sy + ys[0]
            for k, fr in enumerate(tr["frame"].to_numpy()):
                rows.append({"track_id": tid, "frame": int(fr),
                             "cx": sx[k], "cy": sy[k],
                             "x1": sx[k] - 5, "y1": sy[k] - 5,
                             "x2": sx[k] + 5, "y2": sy[k] + 5, "conf": 1.0})
        out = pd.DataFrame(rows).sort_values(["track_id", "frame"])
        out.to_csv(HOM_DIR / tf.name, index=False)
        print(f"  {tf.stem}: {out['track_id'].nunique()} synthetic tracks", flush=True)
    return {"n_tracks": n_tracks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fit = fit_pooled()
    print(f"pooled OU: mu={fit['mu']}, var={fit['var']}, a={fit['a']:.4f}",
          flush=True)
    print("Simulating homogeneous-continuum tracks ...", flush=True)
    sim = materialise(fit, args.seed)

    res: dict = {
        "model": ("single pooled 2-D AR(1)/OU velocity parameter set for ALL "
                  "tracks (grand mean/variance + pooled lag-1); same track "
                  "lengths as GT; NO quenched heterogeneity"),
        "seed": args.seed,
        "fit": {"mu": fit["mu"].tolist(), "var": fit["var"].tolist(),
                "a": fit["a"]},
        "fit_summary": sim,
    }
    res["null_hom"] = analyse_dir(HOM_DIR, "homogeneous continuum null (pooled OU)")

    # adjudication against per-track null and GT
    het_path = config.MARKOV_OUT / "continuum_null.json"
    gt_path = config.MARKOV_OUT / "gt_reanchor.json"
    if het_path.exists() and gt_path.exists():
        het = json.loads(het_path.read_text())["null"]
        gt = json.loads(gt_path.read_text())["gt"]
        g2_hom = res["null_hom"]["block_g2"]["25"]["g2"]
        g2_het = het["block_g2"]["25"]["g2"]
        g2_gt = gt["block_g2"]["25"]["g2"]
        frac = g2_hom / g2_het if g2_het else None
        res["adjudication"] = {
            "block25_g2": {"gt": g2_gt, "null_per_track": g2_het,
                           "null_homogeneous": g2_hom},
            "hom_fraction_of_per_track": frac,
            "prereg_rule": ("<25% -> aggregation (mover-stayer) attribution; "
                            ">=50% -> discretisation-intrinsic; else both"),
        }
        print(f"\nADJUDICATION: g2 block-25  GT {g2_gt:+.4f}  "
              f"per-track null {g2_het:+.4f}  homogeneous null {g2_hom:+.4f}  "
              f"(hom = {frac:.0%} of per-track)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
