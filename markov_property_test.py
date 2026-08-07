#!/usr/bin/env python3
"""Is sperm motility actually Markovian? A test of the modelling assumption.

The standard pipeline (and clinical CASA) treats motility either as a static
3-category snapshot or as a first-order Markov chain over {Progressive,
Non-progressive, Immotile}. BOTH assume memorylessness. A first-order Markov
chain has two hard, falsifiable signatures:

  (A) Geometric (discrete-exponential) dwell-time distributions in every state
      -- P(dwell = k) = (1-p) * p^(k-1).  Memory shows up as heavy tails.
  (B) The next state depends ONLY on the current state. A second-order model
      (depends on current + previous) must NOT improve held-out prediction.

We test both on the real VISEM trajectories (high statistical power: ~10^5-10^6
frame-states), with block cross-validation over tracks so nothing is fit and
tested on the same data. If motility is non-Markovian -- in particular if the
immotile state shows long-memory trapping (heavy-tailed dwell) -- that is a real
dynamical finding about sperm behaviour that the WHO snapshot and the first-order
chain both miss, and it changes how motility should be modelled.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import compute_frame_states, STATES  # noqa: E402

TRACKS = config.TRACK_OUT
OUTDIR = ROOT / "outputs" / "markov"
S2I = {s: i for i, s in enumerate(STATES)}
RNG = np.random.default_rng(0)


def load_sequences() -> list[list[int]]:
    """All per-track state sequences (as integer arrays), plain tracks only."""
    seqs = []
    for tf in sorted(TRACKS.glob("*_tracks.csv")):
        name = tf.stem.replace("_tracks", "")
        if any(name.endswith(s) for s in ("_botsort", "_bytetrack", "_ocsort",
                                          "_botsort_reid")):
            continue
        df = pd.read_csv(tf)
        if df.empty:
            continue
        for tid in df["track_id"].unique():
            tr = df[df["track_id"] == tid].sort_values("frame")
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            seqs.append([S2I[s] for s in compute_frame_states(tr)])
    return seqs


# ── (A) dwell-time distributions ──────────────────────────────────────────────
def dwell_times(seqs) -> dict[int, list[int]]:
    dwell = defaultdict(list)
    for seq in seqs:
        if not seq:
            continue
        cur, run = seq[0], 1
        for s in seq[1:]:
            if s == cur:
                run += 1
            else:
                dwell[cur].append(run)
                cur, run = s, 1
        dwell[cur].append(run)   # final run (right-censored, kept for shape)
    return dwell


def test_geometric(dwell_list, state_name):
    """Compare empirical dwell survival to the best-fit geometric (Markov) model.

    Returns dict with the geometric MLE, a dispersion ratio (Var/mean^2; ==
    (1) for exact geometric in the large-mean limit -> >1 means heavy tail),
    and the fraction of probability mass in the tail beyond what geometric
    predicts.
    """
    d = np.asarray(dwell_list, dtype=float)
    n = len(d)
    mean = d.mean()
    # geometric MLE on support {1,2,...}: p_stay = 1 - 1/mean
    p_stay = max(1e-6, 1 - 1.0 / mean)
    # coefficient of variation: geometric CV ~ sqrt(p_stay) -> ~1 for large mean.
    cv = d.std() / mean
    # tail mass: P(dwell > 3*mean) empirical vs geometric
    thr = 3 * mean
    emp_tail = float((d > thr).mean())
    geo_tail = float(p_stay ** np.floor(thr))     # P(dwell > thr) for geometric
    # KS test against the fitted geometric
    k = np.arange(1, int(d.max()) + 1)
    geo_pmf = (1 - p_stay) * p_stay ** (k - 1)
    geo_cdf = np.cumsum(geo_pmf)
    emp_cdf = np.array([(d <= kk).mean() for kk in k])
    ks = float(np.max(np.abs(emp_cdf - geo_cdf)))
    return {
        "state": state_name, "n_episodes": n, "mean_dwell_frames": mean,
        "mean_dwell_s": mean / config.FPS, "cv": cv,
        "emp_tail_gt3mean": emp_tail, "geo_tail_gt3mean": geo_tail,
        "tail_excess_x": emp_tail / geo_tail if geo_tail > 0 else np.inf,
        "ks_vs_geometric": ks,
    }


# ── (B) Markov order via held-out predictive log-likelihood ──────────────────
def fit_counts(seqs, order):
    """Conditional next-state counts given the last `order` states."""
    tab = defaultdict(lambda: np.zeros(len(STATES)))
    for seq in seqs:
        if len(seq) <= order:
            continue
        for i in range(order, len(seq)):
            ctx = tuple(seq[i - order:i]) if order > 0 else ()
            tab[ctx][seq[i]] += 1
    return tab


def predictive_ll(train_seqs, test_seqs, order, alpha=0.5):
    tab = fit_counts(train_seqs, order)
    # global fallback (0-order) for unseen contexts
    g = np.zeros(len(STATES))
    for seq in train_seqs:
        for s in seq:
            g[s] += 1
    g = (g + alpha) / (g.sum() + alpha * len(STATES))
    ll, ntok = 0.0, 0
    for seq in test_seqs:
        if len(seq) <= order:
            continue
        for i in range(order, len(seq)):
            ctx = tuple(seq[i - order:i]) if order > 0 else ()
            c = tab.get(ctx)
            if c is None or c.sum() == 0:
                p = g
            else:
                p = (c + alpha) / (c.sum() + alpha * len(STATES))
            ll += np.log(p[seq[i]] + 1e-12)
            ntok += 1
    return ll, ntok


def cv_order(seqs, orders=(0, 1, 2, 3), folds=5):
    idx = np.arange(len(seqs))
    RNG.shuffle(idx)
    parts = np.array_split(idx, folds)
    out = {}
    for order in orders:
        tot_ll, tot_n = 0.0, 0
        for f in range(folds):
            test_i = set(parts[f].tolist())
            train = [seqs[i] for i in idx if i not in test_i]
            test = [seqs[i] for i in parts[f]]
            ll, n = predictive_ll(train, test, order)
            tot_ll += ll
            tot_n += n
        out[order] = tot_ll / tot_n      # mean per-token log-likelihood
    return out


def main():
    print("loading state sequences ...")
    seqs = load_sequences()
    nstates = sum(len(s) for s in seqs)
    print(f"  {len(seqs)} tracks, {nstates} frame-states "
          f"(@{config.FPS} fps)\n")

    # ---- (A) dwell-time memory test ----------------------------------------
    print("=== (A) Dwell-time distributions vs geometric (Markov) ===")
    print("    geometric/Markov => CV~1, tail_excess~1x. Heavy memory => CV>1, "
          "tail_excess>>1x.\n")
    dwell = dwell_times(seqs)
    rows = []
    for i, name in enumerate(STATES):
        if len(dwell[i]) < 30:
            continue
        r = test_geometric(dwell[i], name)
        rows.append(r)
        print("  %-16s episodes=%6d  mean=%5.2f s  CV=%.2f  "
              "tail>3mean: emp=%.4f geo=%.4f (%.1fx)  KS=%.3f"
              % (name, r["n_episodes"], r["mean_dwell_s"], r["cv"],
                 r["emp_tail_gt3mean"], r["geo_tail_gt3mean"],
                 r["tail_excess_x"], r["ks_vs_geometric"]))
    pd.DataFrame(rows).to_csv(OUTDIR / "dwell_memory_test.csv", index=False)

    # ---- (B) Markov-order test ---------------------------------------------
    print("\n=== (B) Markov order: held-out per-token log-likelihood (5-fold CV) ===")
    print("    higher = better prediction. If 2nd>1st>0th, motility has memory.\n")
    ll = cv_order(seqs)
    base = ll[0]
    for order in sorted(ll):
        gain = ll[order] - ll[order - 1] if order > 0 else 0.0
        tag = ""
        if order == 1:
            tag = "  <- vs i.i.d."
        elif order >= 2:
            tag = "  <- MEMORY beyond first-order" if gain > 1e-3 else "  (no gain)"
        print("  order %d:  mean logL/token = % .4f   (delta vs prev = %+.4f)%s"
              % (order, ll[order], gain, tag))

    # perplexity reduction summary
    print("\n  Interpretation:")
    g1 = ll[1] - ll[0]
    g2 = ll[2] - ll[1]
    print(f"    1st-order over i.i.d. : +{g1:.4f} logL/token "
          f"(perplexity x{np.exp(-g1):.3f})")
    print(f"    2nd-order over 1st    : +{g2:.4f} logL/token "
          f"(perplexity x{np.exp(-g2):.3f})")
    if g2 > 0.01:
        print("    => second-order memory present: motility is NOT first-order Markov.")
    else:
        print("    => no meaningful second-order gain: first-order is adequate.")

    pd.DataFrame([ll]).to_csv(OUTDIR / "markov_order_cv.csv", index=False)
    print(f"\nsaved -> {OUTDIR/'dwell_memory_test.csv'}")
    print(f"saved -> {OUTDIR/'markov_order_cv.csv'}")


if __name__ == "__main__":
    main()
