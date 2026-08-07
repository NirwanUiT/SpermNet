"""Per-cell switching-rate heterogeneity as a new participant-level observable.

The mover-stayer result says the dominant structure in sperm motility is STABLE
cell-to-cell heterogeneity in switching kinetics -- some cells are "stayers" (rarely
change state) and some are "movers". Standard CASA reports only population AVERAGES
and so collapses this axis. Here we build the axis it discards and ask whether it is
(a) a real, reliable participant-level trait and (b) linked to clinical DFI.

Per cell (track) we compute a length-normalised switching rate = number of
motility-state changes / track duration (switches per second), using tracks long
enough (>= MIN_SEQ frames) for a stable estimate. Per participant we summarise the
DISTRIBUTION of that rate across its cells:
    mean_rate   -- the CASA-like average (mover-ness on average)
    cv_rate     -- across-cell coefficient of variation = HETEROGENEITY magnitude
    stayer_frac -- fraction of cells that never switch state
    gini_rate   -- inequality of switching across cells

Reliability: split each participant's cells in half and correlate the heterogeneity
feature between halves across participants (a real trait => high split-half rho).

Clinical: PRE-REGISTERED PRIMARY = partial Spearman(cv_rate, DFI | mean_rate, log
n_cells), run SEPARATELY PER COHORT (never pooled -- the two tracking pipelines carry
a known batch effect that makes pooled dynamics-vs-clinical correlations untrustworthy).
extra57 (single pipeline, larger n) is the primary cohort; orig20 is a check.

Output: outputs/markov/per_cell_kinetics.{json,csv}

Usage: python -m experiments.per_cell_kinetics [--min-seq 50] [--min-cells 30]
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
from markov_analysis import compute_frame_states, STATES  # noqa: E402
from experiments.dfi_features import load_clinical  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"
DFI_COL = "DNA fragmentation index, DFI (%)"


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def partial_spearman(x, y, *ctrls):
    """Spearman correlation of x,y after linearly removing rank-ctrls."""
    x = stats.rankdata(x); y = stats.rankdata(y)
    if ctrls:
        C = np.column_stack([stats.rankdata(c) for c in ctrls])
        C = np.column_stack([np.ones(len(x)), C])
        x = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
        y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    r = stats.spearmanr(x, y)
    return float(r.correlation), float(r.pvalue)


def cohort_files(cohort):
    if cohort == "orig20":
        out = {}
        for tf in sorted(ORIG.glob("*_tracks.csv")):
            name = tf.stem.replace("_tracks", "")
            if any(name.endswith(s) for s in ("_botsort", "_bytetrack", "_ocsort", "_reid")):
                continue
            out[int(name)] = tf
        return out
    return {int(tf.stem.split("_")[0]): tf for tf in sorted(EXTRA.glob("*_tracks.csv"))}


def per_cell_rates(tf, min_seq, max_tracks, rng):
    df = pd.read_csv(tf)
    if df.empty:
        return np.array([])
    tids = df["track_id"].unique()
    if max_tracks and len(tids) > max_tracks:
        tids = rng.choice(tids, size=max_tracks, replace=False)
        df = df[df["track_id"].isin(tids)]
    rates = []
    for _, tr in df.groupby("track_id"):
        if len(tr) < config.MIN_TRACK_LENGTH:
            continue
        tr = tr.sort_values("frame")
        seq = [S2I[s] for s in compute_frame_states(tr)]
        if len(seq) < min_seq:
            continue
        nsw = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
        rates.append(nsw / (len(seq) / config.FPS))
    return np.array(rates, float)


def part_features(rates):
    m = rates.mean()
    return {
        "n_cells": int(len(rates)),
        "mean_rate": float(m),
        "sd_rate": float(rates.std()),
        "cv_rate": float(rates.std() / m) if m > 0 else np.nan,
        "stayer_frac": float(np.mean(rates == 0)),
        "gini_rate": gini(rates),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seq", type=int, default=50)      # >=1 s tracks
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--max-tracks", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clin = load_clinical()
    rng = np.random.default_rng(args.seed)
    res = {"params": vars(args)}
    rows = []

    for cohort in ("orig20", "extra57"):
        print(f"=== {cohort} ===", flush=True)
        files = cohort_files(cohort)
        halfA, halfB = {}, {}
        feats = {}
        for pid, tf in files.items():
            rates = per_cell_rates(tf, args.min_seq, args.max_tracks, rng)
            if len(rates) < args.min_cells:
                continue
            feats[pid] = part_features(rates)
            perm = rng.permutation(len(rates))
            h = len(rates) // 2
            halfA[pid] = part_features(rates[perm[:h]])
            halfB[pid] = part_features(rates[perm[h:]])
            r = {"cohort": cohort, "pid": pid, **feats[pid]}
            if pid in clin.index and np.isfinite(clin.loc[pid, DFI_COL]):
                r["DFI"] = float(clin.loc[pid, DFI_COL])
            rows.append(r)
        pids = list(feats)
        print(f"  {len(pids)} participants with >= {args.min_cells} long cells")

        # split-half reliability of heterogeneity features
        rel = {}
        for key in ("cv_rate", "stayer_frac", "gini_rate"):
            a = [halfA[p][key] for p in pids]
            b = [halfB[p][key] for p in pids]
            rel[key] = float(stats.spearmanr(a, b).correlation)
        print(f"  split-half reliability: "
              + "  ".join(f"{k}={v:+.2f}" for k, v in rel.items()))

        # clinical: partial Spearman(cv_rate, DFI | mean_rate, log n_cells)
        cl = [(feats[p], float(clin.loc[p, DFI_COL])) for p in pids
              if p in clin.index and np.isfinite(clin.loc[p, DFI_COL])]
        clin_res = {}
        if len(cl) >= 12:
            cv = np.array([f["cv_rate"] for f, _ in cl])
            mr = np.array([f["mean_rate"] for f, _ in cl])
            nc = np.log10([f["n_cells"] for f, _ in cl])
            sf = np.array([f["stayer_frac"] for f, _ in cl])
            dfi = np.array([d for _, d in cl])
            r_p, p_p = partial_spearman(cv, dfi, mr, nc)
            r_s, p_s = partial_spearman(sf, dfi, mr, nc)
            r_m, p_m = stats.spearmanr(mr, dfi)
            clin_res = {"n": len(cl),
                        "PRIMARY_cv_rate_partial": {"rho": r_p, "p": p_p},
                        "secondary_stayer_frac_partial": {"rho": r_s, "p": p_s},
                        "ref_mean_rate_vs_DFI": {"rho": float(r_m), "p": float(p_m)}}
            print(f"  DFI n={len(cl)}: PRIMARY cv_rate|mean,n partial rho={r_p:+.3f} "
                  f"(p={p_p:.3f}) | stayer_frac partial rho={r_s:+.3f} (p={p_s:.3f}) "
                  f"| mean_rate vs DFI rho={r_m:+.3f} (p={p_m:.3f})")
        res[cohort] = {"n_participants": len(pids), "reliability": rel,
                       "clinical": clin_res}
        print()

    pd.DataFrame(rows).to_csv(OUTDIR / "per_cell_kinetics.csv", index=False)
    json.dump(res, open(OUTDIR / "per_cell_kinetics.json", "w"), indent=2, default=float)
    print(f"saved -> {OUTDIR/'per_cell_kinetics.json'}")
    print(f"saved -> {OUTDIR/'per_cell_kinetics.csv'}")


if __name__ == "__main__":
    main()
