"""Decompose the memory: quenched cell-to-cell heterogeneity vs within-cell memory.

The log-normal dwell law (dwell_physics.py) and its window/cohort invariance
(dwell_physics_robust.py) are established. But a log-normal pooled dwell distribution
has TWO very different possible origins, with opposite biological meaning:

  (A) QUENCHED SUPERSTATISTICS (population mixture): every cell is individually
      near-memoryless (exponential dwell, CV~1) but cells differ in their escape
      RATE; pooling a log-normal spread of rates yields a log-normal, high-CV pooled
      distribution and *apparent* non-Markovian pooled sequences. The "memory" then
      lives in cell-to-cell heterogeneity, not in any single cell.

  (B) SINGLE-CELL MEMORY: each cell itself has non-exponential (log-normal, CV>1)
      dwells; the memory is intrinsic to the switching dynamics of one cell.

These are DISTINGUISHABLE with our data:
  - decisive test = WITHIN-TRACK dwell CV vs POOLED CV (state-controlled).
        A => within-cell CV ~ 1, pooled CV ~ 2   (variance inflated by mixing)
        B => within-cell CV ~ pooled CV ~ 2       (each cell already dispersed)
  - ICC of log-dwell by track (state-controlled) = fraction of dwell variability
    that is between-cell. High ICC supports (A)'s heterogeneity.
  - state-controlled lag-1 serial correlation of log-dwell along a track = any
    residual within-cell temporal structure.

Output: outputs/markov/memory_decomposition.json

Usage: python -m experiments.memory_decomposition [--max-tracks 4000]
"""
from __future__ import annotations

import argparse
import json
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
        vid = tf.stem.replace("_tracks", "")
        for tid, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            yield f"{vid}:{tid}", tr.sort_values("frame")


def episodes_of(seq):
    eps, run = [], 1
    for k in range(1, len(seq)):
        if seq[k] == seq[k - 1]:
            run += 1
        else:
            eps.append((seq[k - 1], run)); run = 1
    eps.append((seq[-1], run))
    return eps


def analyse(track_dir: Path, max_tracks: int, seed: int):
    # per (track,state): list of dwell (sec); also ordered episode list per track
    per_track_state = {}         # (tkey,state) -> [dwell]
    ordered = {}                 # tkey -> [(state, log_dwell)]
    pooled = {i: [] for i in range(len(STATES))}
    for tkey, tr in iter_tracks(track_dir, max_tracks, seed):
        seq = [S2I[s] for s in ma.compute_frame_states(tr)]
        eps = episodes_of(seq)
        ol = []
        for st, run in eps:
            d = run / config.FPS
            per_track_state.setdefault((tkey, st), []).append(d)
            pooled[st].append(d)
            ol.append((st, np.log(d)))
        ordered[tkey] = ol

    out = {"pooled_cv": {}, "within_track_cv": {}, "icc_logdwell": None,
           "serial_state_controlled": None}

    # pooled CV per state
    for i, s in enumerate(STATES):
        d = np.array(pooled[i], float)
        out["pooled_cv"][s] = float(d.std() / d.mean())

    # within-track per-state CV (tracks with >=5 same-state episodes)
    for i, s in enumerate(STATES):
        cvs = [np.std(v) / np.mean(v) for (tk, st), v in per_track_state.items()
               if st == i and len(v) >= 5 and np.mean(v) > 0]
        cvs = np.array(cvs, float)
        out["within_track_cv"][s] = {
            "n_tracks": int(len(cvs)),
            "median_cv": float(np.median(cvs)) if len(cvs) else None,
            "iqr": [float(np.percentile(cvs, 25)), float(np.percentile(cvs, 75))]
                   if len(cvs) else None,
        }

    # ICC of log-dwell by track, state-controlled (residualise per-state mean)
    rows = []
    state_mean = {i: np.mean(np.log(np.array(pooled[i], float))) for i in range(len(STATES))}
    for tkey, ol in ordered.items():
        for st, ld in ol:
            rows.append((tkey, ld - state_mean[st]))
    if rows:
        df = pd.DataFrame(rows, columns=["tkey", "resid"])
        grp = df.groupby("tkey")["resid"]
        counts = grp.count()
        valid = counts[counts >= 3].index          # need >=3 episodes for a track effect
        df = df[df["tkey"].isin(valid)]
        grand = df["resid"].mean()
        gm = df.groupby("tkey")["resid"]
        n_i = gm.count().values
        mean_i = gm.mean().values
        ss_between = np.sum(n_i * (mean_i - grand) ** 2)
        ss_within = np.sum((df["resid"].values -
                            df.groupby("tkey")["resid"].transform("mean").values) ** 2)
        k = len(n_i)
        N = n_i.sum()
        ms_b = ss_between / (k - 1)
        ms_w = ss_within / (N - k)
        n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
        icc = (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)
        out["icc_logdwell"] = {"icc": float(icc), "n_tracks": int(k),
                               "n_episodes": int(N),
                               "interpretation":
                               "fraction of state-controlled log-dwell variance that is between-cell"}

    # state-controlled lag-1 serial correlation (residual log-dwell) vs shuffle
    a, b, sa, sb = [], [], [], []
    rng = np.random.default_rng(0)
    for tkey, ol in ordered.items():
        if len(ol) < 4:
            continue
        res = np.array([ld - state_mean[st] for st, ld in ol])
        a.extend(res[:-1]); b.extend(res[1:])
        sh = rng.permutation(res)
        sa.extend(sh[:-1]); sb.extend(sh[1:])
    if a:
        r = stats.spearmanr(a, b)
        rn = stats.spearmanr(sa, sb).correlation
        out["serial_state_controlled"] = {
            "lag1_spearman": float(r.correlation), "p": float(r.pvalue),
            "shuffle_null": float(rn), "n_pairs": len(a)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = {}
    for cname, cdir in (("orig20", ORIG), ("extra57", EXTRA)):
        print(f"=== {cname} ===", flush=True)
        r = analyse(cdir, args.max_tracks, args.seed)
        res[cname] = r
        for s in STATES:
            wc = r["within_track_cv"][s]
            print(f"  {s:16s} pooled CV={r['pooled_cv'][s]:.2f}  "
                  f"within-cell median CV={wc['median_cv']}  (n={wc['n_tracks']} tracks)")
        icc = r["icc_logdwell"]
        print(f"  ICC(log-dwell | state) = {icc['icc']:.3f}  "
              f"[between-cell fraction]  (n_tracks={icc['n_tracks']}, n_ep={icc['n_episodes']})")
        sc = r["serial_state_controlled"]
        print(f"  state-controlled lag-1 rho = {sc['lag1_spearman']:+.3f} "
              f"(p={sc['p']:.2g})  shuffle null={sc['shuffle_null']:+.3f}\n")

    json.dump(res, open(OUTDIR / "memory_decomposition.json", "w"), indent=2, default=float)
    print(f"saved -> {OUTDIR/'memory_decomposition.json'}")


if __name__ == "__main__":
    main()
