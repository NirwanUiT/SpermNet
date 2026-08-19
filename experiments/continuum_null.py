"""T0.1 -- Continuous-process (Markovian) null: does a memoryless continuum,
passed through the identical windowed classifier, manufacture the paper's
memory statistics?

Null model: per GT track, a 2-D Ornstein-Uhlenbeck (discrete AR(1)) velocity
process -- Markovian by construction -- fit ONLY to that track's velocity
marginal (per-component mean and variance) and pooled lag-1 autocovariance.
Never fit to any memory statistic. Each synthetic track has exactly the same
length as its real counterpart (censoring reproduced by construction), and
per-track fitting preserves quenched cell-to-cell heterogeneity (so the null
is "heterogeneous Markovian continuum", the fair version).

The synthetic tracks are materialised to outputs/tracks_continuum_null/ with
the same CSV schema as outputs/tracks_gt/ and scored with the EXACT pipeline
used for the ground truth (gt_reanchor.analyse_dir): dwell-law AIC
competition, geometric dispersion (CV, tail), frame-level and 0.5s-block
Markov-order tests (g2), EB mover-stayer decomposition, within-cell CV / ICC /
state-controlled serial correlation.

Decision rule (pre-stated):
  - null reproduces g2, within-cell CV and serial drho  -> memory claim collapses
  - null reproduces some                                 -> keep only survivors
  - null reproduces none                                 -> genuine result + control

Output: outputs/markov/continuum_null.json
Usage:  python -m experiments.continuum_null [--seed 0] [--skip-simulate]
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
from experiments.gt_reanchor import GT_DIR, analyse_dir  # noqa: E402

NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "continuum_null.json"


def fit_ou(xs: np.ndarray, ys: np.ndarray) -> dict:
    """Per-track AR(1)/OU velocity fit: component means/variances + pooled lag-1."""
    vx, vy = np.diff(xs), np.diff(ys)
    mu = np.array([vx.mean(), vy.mean()])
    var = np.array([vx.var(), vy.var()])
    cx, cy = vx - mu[0], vy - mu[1]
    num = float(np.dot(cx[1:], cx[:-1]) + np.dot(cy[1:], cy[:-1]))
    den = float(np.dot(cx, cx) + np.dot(cy, cy))
    a = num / den if den > 0 else 0.0
    a = float(np.clip(a, -0.995, 0.995))
    return {"mu": mu, "var": var, "a": a}


def simulate_ou(n: int, fit: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n positions from the fitted AR(1) velocity process."""
    mu, var, a = fit["mu"], fit["var"], fit["a"]
    sd_innov = np.sqrt(np.maximum(var * (1.0 - a * a), 0.0))
    v = np.empty((n - 1, 2))
    v[0] = mu + rng.normal(0.0, np.sqrt(var))
    eps = rng.normal(0.0, 1.0, size=(n - 1, 2)) * sd_innov
    for t in range(1, n - 1):
        v[t] = mu + a * (v[t - 1] - mu) + eps[t]
    pos = np.vstack([[0.0, 0.0], np.cumsum(v, axis=0)])
    return pos[:, 0], pos[:, 1]


def materialise(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    NULL_DIR.mkdir(parents=True, exist_ok=True)
    a_all, n_tracks, n_short = [], 0, 0
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
                n_short += 1
                sx, sy = xs - xs[0], ys - ys[0]
            else:
                fit = fit_ou(xs, ys)
                a_all.append(fit["a"])
                sx, sy = simulate_ou(n, fit, rng)
            sx, sy = sx + xs[0], sy + ys[0]
            for k, fr in enumerate(tr["frame"].to_numpy()):
                rows.append({"track_id": tid, "frame": int(fr),
                             "cx": sx[k], "cy": sy[k],
                             "x1": sx[k] - 5, "y1": sy[k] - 5,
                             "x2": sx[k] + 5, "y2": sy[k] + 5, "conf": 1.0})
        out = pd.DataFrame(rows).sort_values(["track_id", "frame"])
        out.to_csv(NULL_DIR / tf.name, index=False)
        print(f"  {tf.stem}: {out['track_id'].nunique()} synthetic tracks", flush=True)
    a = np.array(a_all)
    return {"n_tracks": n_tracks, "n_too_short": n_short,
            "lag1_a": {"mean": float(a.mean()), "median": float(np.median(a)),
                       "q10": float(np.quantile(a, 0.1)),
                       "q90": float(np.quantile(a, 0.9)),
                       "frac_negative": float((a < 0).mean())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-simulate", action="store_true")
    args = ap.parse_args()

    res: dict = {"model": ("per-track 2-D AR(1)/OU velocity, fit to marginal "
                           "mean/variance + pooled lag-1 autocovariance only; "
                           "same track lengths; quenched per-track parameters"),
                 "seed": args.seed}

    if not args.skip_simulate:
        print("Simulating Markovian-continuum tracks ...", flush=True)
        res["fit_summary"] = materialise(args.seed)

    res["null"] = analyse_dir(NULL_DIR, "continuum-null (per-track OU velocity)")

    # side-by-side with locked GT numbers
    gt_path = config.MARKOV_OUT / "gt_reanchor.json"
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())["gt"]
        res["gt_reference"] = {
            "block25_g2": gt["block_g2"]["25"]["g2"],
            "frame_g2": gt["order_frame"]["g2"],
            "dwell_best": {s: gt["dwell_laws"][s].get("best")
                           for s in gt["dwell_laws"]},
            "geometric": gt["geometric"],
            "icc": gt["memory_decomposition"]["icc_logdwell"]["icc"],
            "serial": gt["memory_decomposition"]["serial_state_controlled"],
        }
        nl = res["null"]
        g2_gt = res["gt_reference"]["block25_g2"]
        g2_nl = nl["block_g2"]["25"]["g2"]
        res["verdict"] = {
            "block25_g2_null_fraction_of_gt": (g2_nl / g2_gt) if g2_gt else None,
            "note": ("fraction ~1 => classifier+continuum manufactures the "
                     "memory (claim collapses); ~0 => genuine result"),
        }
        print(f"\nDECISION: null block-g2 = {g2_nl:+.4f} vs GT {g2_gt:+.4f} "
              f"({res['verdict']['block25_g2_null_fraction_of_gt']:.0%} of GT)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
