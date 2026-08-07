"""Robustness + mechanism test for the log-normal dwell law (Aim B, part 2).

Part 1 (dwell_physics.py) found dwell times are log-normal (not exponential/gamma/
Weibull) in all states and both cohorts. Two things must hold for that to be a real
mechanism rather than an artifact:

  (R) WINDOW INVARIANCE: the sliding-window classifier (default 25 frames) must not
      be CREATING the log-normal shape. We recompute with windows {13,25,51} and
      check the best-fitting law and the rejection of exponential are invariant.

  (M) SUPERSTATISTICS PREDICTION (positive, model-free): if dwells are log-normal
      because each cell's state-escape RATE fluctuates slowly (a continuous hidden
      drive), then successive dwell durations along a track should be SERIALLY
      CORRELATED (the slow variable persists across switches). A memoryless or
      fixed-rate process predicts zero serial correlation. We measure the lag-1
      Spearman correlation of log-dwell along each track vs a within-track shuffle
      null. Positive, above-null correlation = direct evidence of a slow hidden rate.

Output: outputs/markov/dwell_physics_robust.json

Usage: python -m experiments.dwell_physics_robust [--max-tracks 1500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import markov_analysis as ma  # noqa: E402
from markov_analysis import STATES  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"


def iter_tracks(track_dir: Path, max_tracks: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    plain_only = track_dir == ORIG
    for tf in sorted(track_dir.glob("*_tracks.csv")):
        if plain_only:
            nm = tf.stem.replace("_tracks", "")
            if any(nm.endswith(s) for s in ("_botsort", "_bytetrack", "_ocsort", "_reid")):
                continue
        df = pd.read_csv(tf)
        if df.empty:
            continue
        tids = df["track_id"].unique()
        if max_tracks and len(tids) > max_tracks:
            tids = rng.choice(tids, size=max_tracks, replace=False)
            df = df[df["track_id"].isin(tids)]
        for _, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            yield tr.sort_values("frame")


def episodes_of(seq):
    """List of (state, dwell_frames) episodes along one state sequence."""
    eps = []
    run = 1
    for k in range(1, len(seq)):
        if seq[k] == seq[k - 1]:
            run += 1
        else:
            eps.append((seq[k - 1], run))
            run = 1
    eps.append((seq[-1], run))
    return eps


def fit_best(d):
    d = d[d > 0]
    laws = {"exponential": (stats.expon, {"floc": 0}),
            "gamma": (stats.gamma, {"floc": 0}),
            "weibull": (stats.weibull_min, {"floc": 0}),
            "lognormal": (stats.lognorm, {"floc": 0})}
    aic = {}
    for nm, (dist, kw) in laws.items():
        par = dist.fit(d, **kw)
        ll = np.sum(dist.logpdf(d, *par))
        k = len(par) - 1
        aic[nm] = 2 * k - 2 * ll
    best = min(aic, key=aic.get)
    return best, {n: aic[n] - aic[best] for n in aic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = {}

    # ---- (R) window invariance (extra cohort, big n) ----
    print("=== (R) WINDOW INVARIANCE (extra57) ===", flush=True)
    res["window_invariance"] = {}
    for W in (13, 25, 51):
        ma.WINDOW_FRAMES = W
        dwell = {i: [] for i in range(len(STATES))}
        for tr in iter_tracks(EXTRA, args.max_tracks, args.seed):
            seq = [S2I[s] for s in ma.compute_frame_states(tr)]
            for st, run in episodes_of(seq):
                dwell[st].append(run)
        row = {}
        for i, sname in enumerate(STATES):
            d = np.array(dwell[i], float) / config.FPS
            if len(d) < 200:
                continue
            best, daic = fit_best(d)
            row[sname] = {"best": best, "exp_dAIC": daic["exponential"],
                          "logn_vs_gamma": daic["gamma"], "n": len(d),
                          "cv": float(d.std() / d.mean())}
            print(f"  W={W:2d}  {sname:16s} best={best:10s} CV={d.std()/d.mean():.2f}"
                  f"  exp dAIC=+{daic['exponential']:.0f}  gamma dAIC=+{daic['gamma']:.0f}")
        res["window_invariance"][W] = row
    ma.WINDOW_FRAMES = 25  # restore

    # ---- (M) superstatistics: serial correlation of log-dwell along tracks ----
    print("\n=== (M) SERIAL CORRELATION of successive dwell durations ===", flush=True)
    print("    slow hidden rate => positive lag-1 correlation vs within-track shuffle", flush=True)
    res["serial_corr"] = {}
    for cname, cdir in (("orig20", ORIG), ("extra57", EXTRA)):
        logs_a, logs_b = [], []      # lag-1 pairs (log dwell)
        shuf_a, shuf_b = [], []      # within-track shuffled null
        rng = np.random.default_rng(0)
        ntracks = 0
        for tr in iter_tracks(cdir, args.max_tracks, args.seed):
            seq = [S2I[s] for s in ma.compute_frame_states(tr)]
            eps = episodes_of(seq)
            if len(eps) < 4:
                continue
            ntracks += 1
            dur = np.log(np.array([e[1] for e in eps], float))
            logs_a.extend(dur[:-1]); logs_b.extend(dur[1:])
            sh = rng.permutation(dur)
            shuf_a.extend(sh[:-1]); shuf_b.extend(sh[1:])
        r_obs = stats.spearmanr(logs_a, logs_b).correlation
        p_obs = stats.spearmanr(logs_a, logs_b).pvalue
        r_null = stats.spearmanr(shuf_a, shuf_b).correlation
        res["serial_corr"][cname] = {"n_tracks": ntracks, "n_pairs": len(logs_a),
                                     "lag1_spearman": float(r_obs), "p": float(p_obs),
                                     "shuffle_null": float(r_null)}
        print(f"  {cname:8s} n_tracks={ntracks:6d} pairs={len(logs_a):7d}  "
              f"lag-1 rho={r_obs:+.3f} (p={p_obs:.2g})  shuffle null={r_null:+.3f}")

    import json
    json.dump(res, open(OUTDIR / "dwell_physics_robust.json", "w"),
              indent=2, default=float)
    print(f"\nsaved -> {OUTDIR/'dwell_physics_robust.json'}")


if __name__ == "__main__":
    main()
