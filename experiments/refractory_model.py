"""Mechanistic budget of the non-Markovian memory on TRUE hand annotations.

The GT re-anchor (gt_reanchor.py) established, on hand-annotated tracks only:
  - block-decorrelated 2nd-order memory g2 ~ +0.02..0.04 (window-robust);
  - within-cell dwell CV ~ 0.77 << 1: single cells are MORE regular than a
    Poisson switcher (sub-exponential dwells) -- the pooled heavy tail is
    between-cell mixing;
  - state-controlled lag-1 serial correlation of consecutive residual
    log-dwells is NEGATIVE relative to the within-track shuffle null
    (rho = -0.125 vs +0.117): after a long dwell the next dwell runs short --
    refractoriness / homeostatic alternation, exactly the fast ingredient the
    slow-latent generative model (generative_model.py) predicted was missing;
  - quenched between-cell rate heterogeneity ICC ~ 0.10.

This script asks WHICH of those measured ingredients actually produce the
block-level memory. It builds a semi-Markov simulator over the empirical
embedded topology and adds one calibrated ingredient at a time -- each
calibrated to ITS OWN statistic, never to g2 -- so the block-modal g2 of every
variant is a zero-free-parameter prediction:

  V0  first-order embedded chain + exponential dwells, homogeneous (null)
  V1  + quenched per-cell log-rate offset  (calibrated to ICC)
  V2  + gamma dwell shape                  (calibrated to within-cell CV)
  V3  + refractory AR(1) on residual log-dwell (calibrated to serial delta-rho)
  V4  + second-order embedded state chain  (calibrated to embedded transition counts)

For every variant we re-measure, with the IDENTICAL estimators used on the real
data: block-modal g2, pooled dwell CV, within-cell CV, ICC, serial delta-rho.
The rung at which simulated g2 reaches the real value identifies the memory's
mechanistic origin; if no rung reaches it, the honest residual is quantified.

Output: outputs/markov/refractory_model.json
Usage:  python -m experiments.refractory_model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from markov_property_test import cv_order, dwell_times  # noqa: E402
from experiments.replicate_markov_extra import load_sequences_from  # noqa: E402
from experiments.generative_model import block_downsample  # noqa: E402

NS = len(STATES)
GT_DIR = ROOT / "outputs" / "tracks_gt"
OUT = config.MARKOV_OUT / "refractory_model.json"


# ------------------------------------------------------------------- episodes
def episodes_of(seq):
    """[(state, run_frames)] for one frame-level sequence."""
    eps, run = [], 1
    for k in range(1, len(seq)):
        if seq[k] == seq[k - 1]:
            run += 1
        else:
            eps.append((seq[k - 1], run))
            run = 1
    eps.append((seq[-1], run))
    return eps


def frames_of(eps, L):
    """Expand episodes back to a frame-level sequence cut to L frames."""
    out = []
    for st, run in eps:
        out.extend([st] * int(run))
        if len(out) >= L:
            break
    return out[:L]


# ----------------------------------------------------- estimators (shared!)
def block_g2(seqs, block=25):
    blk = block_downsample(seqs, block=block)
    if len(blk) < 10:
        return None
    ll = cv_order(blk, orders=(0, 1, 2), folds=5)
    return ll[2] - ll[1]


def pooled_cv(seqs):
    dw = dwell_times(seqs)
    out = []
    for i in range(NS):
        d = np.asarray(dw.get(i, []), float)
        out.append(float(d.std() / d.mean()) if len(d) >= 30 else None)
    return out


def episode_stats(ep_lists):
    """Within-cell CV, ICC and serial delta-rho with the estimators of
    memory_decomposition.py, applied to per-track episode lists."""
    pooled = defaultdict(list)
    per_ts = defaultdict(list)
    for ti, eps in enumerate(ep_lists):
        for st, run in eps:
            pooled[st].append(run / config.FPS)
            per_ts[(ti, st)].append(run / config.FPS)

    within_cv = {}
    for i, s in enumerate(STATES):
        cvs = [np.std(v) / np.mean(v) for (ti, st), v in per_ts.items()
               if st == i and len(v) >= 5 and np.mean(v) > 0]
        within_cv[s] = float(np.median(cvs)) if cvs else None

    state_mean = {i: float(np.mean(np.log(pooled[i]))) for i in range(NS)
                  if len(pooled[i])}

    # ICC on state-controlled residual log-dwell (tracks >=3 episodes)
    rows = []
    for ti, eps in enumerate(ep_lists):
        for st, run in eps:
            rows.append((ti, np.log(run / config.FPS) - state_mean[st]))
    icc = None
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["t", "r"])
        cnt = df.groupby("t")["r"].count()
        df = df[df["t"].isin(cnt[cnt >= 3].index)]
        if df["t"].nunique() >= 10:
            grand = df["r"].mean()
            gm = df.groupby("t")["r"]
            n_i, mean_i = gm.count().values, gm.mean().values
            ssb = np.sum(n_i * (mean_i - grand) ** 2)
            ssw = np.sum((df["r"].values -
                          df.groupby("t")["r"].transform("mean").values) ** 2)
            k, N = len(n_i), n_i.sum()
            msb, msw = ssb / (k - 1), ssw / (N - k)
            n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
            icc = float((msb - msw) / (msb + (n0 - 1) * msw))

    # serial: lag-1 Spearman of residuals, data vs within-track shuffle
    rng = np.random.default_rng(0)
    a, b, sa, sb = [], [], [], []
    for ti, eps in enumerate(ep_lists):
        if len(eps) < 4:
            continue
        res = np.array([np.log(run / config.FPS) - state_mean[st]
                        for st, run in eps])
        a.extend(res[:-1]); b.extend(res[1:])
        sh = rng.permutation(res)
        sa.extend(sh[:-1]); sb.extend(sh[1:])
    serial = None
    if len(a) > 100:
        r = stats.spearmanr(a, b).correlation
        rn = stats.spearmanr(sa, sb).correlation
        serial = {"rho": float(r), "shuffle_null": float(rn),
                  "delta": float(r - rn)}

    return {"within_cv": within_cv, "icc": icc, "serial": serial}


def embedded_g2(ep_lists):
    """Held-out order test on the embedded (durationless) state sequences."""
    seqs = [[st for st, _ in eps] for eps in ep_lists if len(eps) >= 4]
    if len(seqs) < 10:
        return None
    ll = cv_order(seqs, orders=(0, 1, 2), folds=5)
    return ll[2] - ll[1]


# --------------------------------------------------------------- calibration
def calibrate(ep_lists, seqs):
    """Everything the simulator needs, from GT episodes only."""
    # first-order embedded off-diagonal transitions
    C1 = np.full((NS, NS), 0.5)
    # second-order embedded transitions (context = previous, current)
    C2 = np.full((NS, NS, NS), 0.5)
    np.fill_diagonal(C1, 0.0)
    for eps in ep_lists:
        sts = [st for st, _ in eps]
        for x, y in zip(sts[:-1], sts[1:]):
            C1[x, y] += 1
        for x, y, z in zip(sts[:-2], sts[1:-1], sts[2:]):
            C2[x, y, z] += 1
    for a in range(NS):
        C1[a, a] = 0.0
        for b in range(NS):
            C2[a, b, b] = 0.0
    Q1 = C1 / C1.sum(1, keepdims=True)
    Q2 = C2 / np.maximum(C2.sum(2, keepdims=True), 1e-9)

    # per-state mean dwell (frames), interior episodes only (censor-aware-ish)
    mean_frames = np.zeros(NS)
    for i in range(NS):
        d = [run for eps in ep_lists for j, (st, run) in enumerate(eps)
             if st == i and 0 < j < len(eps) - 1]
        d = d or [run for eps in ep_lists for st, run in eps if st == i]
        mean_frames[i] = np.mean(d)

    # within-cell gamma shape from within-cell CV (CV = 1/sqrt(k))
    es = episode_stats(ep_lists)
    wcv = [es["within_cv"][s] or 1.0 for s in STATES]
    shape = np.array([1.0 / (c * c) for c in wcv])

    # quenched per-cell log-rate SD from ICC: sigma^2 = icc/(1-icc)*var_within
    icc = max(es["icc"] or 0.0, 0.0)
    var_within = float(np.mean(polygamma(1, shape)))  # var of log-gamma
    sigma_cell = float(np.sqrt(icc / (1 - icc) * var_within)) if icc > 0 else 0.0

    # start state occupancy + track lengths
    occ = np.zeros(NS)
    for s in seqs:
        for x in s:
            occ[x] += 1
    occ = occ / occ.sum()
    tracklens = np.array([len(s) for s in seqs])

    return {"Q1": Q1, "Q2": Q2, "mean_frames": mean_frames, "shape": shape,
            "sigma_cell": sigma_cell, "occ": occ, "tracklens": tracklens,
            "target_serial_delta": es["serial"]["delta"],
            "measured": es}


# ----------------------------------------------------------------- simulator
def simulate(cal, rng, het=False, gamma_shape=False, phi=0.0, second_order=False,
             n_cells=None):
    """One simulated cohort of matched-length tracks; returns frame seqs + eps.

    Refractoriness is a Gaussian-copula AR(1) on successive dwells of a cell:
    the marginal dwell law stays EXACTLY the calibrated gamma (so V3 differs
    from V2 only in serial coupling), while consecutive normal scores carry
    correlation phi (negative = long dwell -> next dwell short).
    """
    L_all = cal["tracklens"]
    n = n_cells or len(L_all)
    seqs, ep_lists = [], []
    for i in range(n):
        L = int(L_all[i % len(L_all)])
        eta = rng.normal(0.0, cal["sigma_cell"]) if het else 0.0
        s = int(rng.choice(NS, p=cal["occ"]))
        prev = None
        eps = []
        tot = 0
        z_prev = rng.normal()  # stationary N(0,1) copula score
        while tot < L:
            k = cal["shape"][s] if gamma_shape else 1.0
            base = cal["mean_frames"][s] * np.exp(eta)
            if phi != 0.0:
                z = phi * z_prev + rng.normal(0.0, np.sqrt(max(1 - phi * phi, 1e-9)))
                u = float(stats.norm.cdf(z))
                run = float(stats.gamma.ppf(np.clip(u, 1e-9, 1 - 1e-9),
                                            k, scale=base / k))
                z_prev = z
            else:
                run = rng.gamma(k, base / k)
            run = max(1, int(round(run)))
            eps.append((s, run))
            tot += run
            # next state
            if second_order and prev is not None:
                p = cal["Q2"][prev, s]
                if p.sum() <= 0:
                    p = cal["Q1"][s]
            else:
                p = cal["Q1"][s]
            prev = s
            s = int(rng.choice(NS, p=p / p.sum()))
        seqs.append(frames_of(eps, L))
        # recompute episodes from the CUT frame sequence (censoring realistic)
        ep_lists.append(episodes_of(seqs[-1]))
    return seqs, ep_lists


def calibrate_phi(cal, het, gamma_shape, target_delta, n_cells):
    """1-D search on phi to match the serial delta-rho (never touches g2)."""
    best_phi, best_err = 0.0, np.inf
    for phi in np.linspace(-0.95, 0.0, 20):
        _, eps = simulate(cal, np.random.default_rng(12345), het=het,
                          gamma_shape=gamma_shape, phi=phi, n_cells=n_cells)
        es = episode_stats(eps)
        if es["serial"] is None:
            continue
        err = abs(es["serial"]["delta"] - target_delta)
        if err < best_err:
            best_err, best_phi = err, phi
    return best_phi


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cells", type=int, default=0,
                    help="simulated cells (0 = match real track count x4)")
    args = ap.parse_args()
    t0 = time.time()

    print("loading GT sequences ...", flush=True)
    seqs = load_sequences_from(GT_DIR)
    ep_lists = [episodes_of(s) for s in seqs]
    n_cells = args.n_cells or 4 * len(seqs)

    print("measuring real GT statistics ...", flush=True)
    real = {
        "block_g2": block_g2(seqs),
        "pooled_cv": pooled_cv(seqs),
        "embedded_g2": embedded_g2(ep_lists),
        **episode_stats(ep_lists),
    }
    print(f"  real: block_g2={real['block_g2']:+.4f}  "
          f"embedded_g2={real['embedded_g2']:+.4f}  "
          f"icc={real['icc']:.3f}  serial_delta={real['serial']['delta']:+.3f}")

    cal = calibrate(ep_lists, seqs)
    print(f"  calib: mean_frames={np.round(cal['mean_frames'],1)}  "
          f"shape={np.round(cal['shape'],2)}  sigma_cell={cal['sigma_cell']:.3f}")

    print("calibrating refractory phi to serial delta-rho ...", flush=True)
    phi = calibrate_phi(cal, het=True, gamma_shape=True,
                        target_delta=cal["target_serial_delta"],
                        n_cells=min(n_cells, 2000))
    print(f"  phi = {phi:+.3f}")

    variants = {
        "V0_hom_exp": dict(het=False, gamma_shape=False, phi=0.0, second_order=False),
        "V1_+heterogeneity": dict(het=True, gamma_shape=False, phi=0.0, second_order=False),
        "V2_+gamma_shape": dict(het=True, gamma_shape=True, phi=0.0, second_order=False),
        "V3_+refractory": dict(het=True, gamma_shape=True, phi=phi, second_order=False),
        "V4_+embedded_2nd_order": dict(het=True, gamma_shape=True, phi=phi, second_order=True),
    }

    ladder = {}
    for vi, (name, kw) in enumerate(variants.items()):
        rng = np.random.default_rng(args.seed + 101 * (vi + 1))
        sseqs, seps = simulate(cal, rng, n_cells=n_cells, **kw)
        es = episode_stats(seps)
        row = {
            "block_g2": block_g2(sseqs),
            "pooled_cv": pooled_cv(sseqs),
            "embedded_g2": embedded_g2(seps),
            "within_cv": es["within_cv"],
            "icc": es["icc"],
            "serial_delta": es["serial"]["delta"] if es["serial"] else None,
            "params": {k: (float(v) if isinstance(v, float) else v)
                       for k, v in kw.items()},
        }
        ladder[name] = row
        print(f"  {name:24s} g2={row['block_g2']:+.4f}  "
              f"emb_g2={row['embedded_g2']:+.4f}  icc={row['icc']:.3f}  "
              f"serial_d={row['serial_delta']:+.3f}  "
              f"cv={[None if c is None else round(c,2) for c in row['pooled_cv']]}",
              flush=True)

    g2r = real["block_g2"]
    fractions = {k: (v["block_g2"] / g2r if v["block_g2"] is not None else None)
                 for k, v in ladder.items()}
    result = {
        "real": real,
        "calibration": {
            "mean_frames": cal["mean_frames"].tolist(),
            "gamma_shape": cal["shape"].tolist(),
            "sigma_cell": cal["sigma_cell"],
            "phi_refractory": phi,
            "Q1": cal["Q1"].tolist(),
        },
        "ladder": ladder,
        "fraction_of_real_g2": fractions,
        "n_real_tracks": len(seqs),
        "n_sim_cells": n_cells,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nfractions of real g2: "
          + "  ".join(f"{k}={v:.0%}" for k, v in fractions.items() if v is not None))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
