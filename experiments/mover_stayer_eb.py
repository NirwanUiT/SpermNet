"""Empirical-Bayes FAIR mover-stayer null — bracket the aggregation share from BELOW.

mover_stayer_null.py showed the over-fit heterogeneous-memoryless null (HET) is an
UPPER bound on how much of the pooled non-Markovian signal population heterogeneity
can explain, because fitting each short track its own transition matrix over-fits
noise and injects EXTRA (unreal) heterogeneity. The real g2 still exceeded that
upper bound, proving genuine within-cell memory exists. To PIN the split we need a
FAIR null that reproduces the *estimated true* between-cell heterogeneity, neither
over-fit nor shrunk to homogeneity.

We fit a hierarchical Dirichlet-multinomial: for each context state a, the per-cell
transition row p_{i,a} ~ Dirichlet(k_a * m_a) with m_a = pooled row (sums to 1) and
concentration k_a estimated by maximum marginal (Dirichlet-multinomial) likelihood
over the observed per-track transition counts. Small k_a = strong between-cell
heterogeneity; large k_a = near-homogeneous. The EB null then DRAWS fresh
p_{i,a} ~ Dirichlet(k_a m_a) per track and simulates matched-length sequences: it
reproduces exactly the estimated real heterogeneity without over-fitting.

Bracket: HOM (k->inf, no heterogeneity, g2~=0) <= EB-fair <= HET (over-fit, upper
bound). If real > EB-fair, genuine within-cell memory is confirmed with a fair null,
and (EB-fair - HOM) / (real - HOM) is the estimated heterogeneity (aggregation) share.

Output: outputs/markov/mover_stayer_eb.json

Usage: python -m experiments.mover_stayer_eb [--max-tracks 3000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from markov_property_test import cv_order  # noqa: E402
from experiments.replicate_markov_extra import nonoverlap_sequences_from  # noqa: E402
from experiments.mover_stayer_null import (  # noqa: E402
    NS, sim_from_M, fit_matrix, block_dwell_cv, g2_of,
)

ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"


def track_count_matrices(seqs):
    """Per-track 3x3 transition count matrices (raw counts, no smoothing)."""
    mats = []
    for seq in seqs:
        C = np.zeros((NS, NS), float)
        for a, b in zip(seq[:-1], seq[1:]):
            C[a, b] += 1
        mats.append(C)
    return mats


def dm_negll(logk, counts, m):
    """Neg Dirichlet-multinomial marginal LL for one context row across tracks.

    counts: (T,NS) per-track next-state counts for this context; m: (NS,) sums to 1.
    """
    k = np.exp(logk)
    alpha = k * m
    n = counts.sum(1)
    ll = (gammaln(k) - gammaln(n + k)
          + (gammaln(counts + alpha) - gammaln(alpha)).sum(1))
    return -float(ll.sum())


def fit_concentration(counts, m):
    counts = counts[counts.sum(1) > 0]
    if len(counts) < 5:
        return np.inf
    res = minimize_scalar(dm_negll, bounds=(np.log(0.05), np.log(1e5)),
                          args=(counts, m), method="bounded")
    return float(np.exp(res.x))


def analyse(track_dir, max_tracks, seed):
    real = [s for s in nonoverlap_sequences_from(track_dir, block=25,
            max_tracks=max_tracks, seed=seed) if len(s) >= 3]
    rng = np.random.default_rng(seed)
    mats = track_count_matrices(real)

    # pooled rows m_a (Laplace so all >0)
    P = np.full((NS, NS), 0.5, float)
    for C in mats:
        P += C
    m = P / P.sum(1, keepdims=True)

    # EB concentration per context row
    k = np.zeros(NS)
    for a in range(NS):
        col = np.array([C[a] for C in mats])          # (T,NS) counts for context a
        k[a] = fit_concentration(col, m[a])

    # simulate EB-fair, over-fit HET, HOM
    eb, het, hom = [], [], []
    starts = np.array([s[0] for s in real])
    for s in real:
        L = len(s)
        Meb = np.vstack([rng.dirichlet(k[a] * m[a]) if np.isfinite(k[a]) else m[a]
                         for a in range(NS)])
        eb.append(sim_from_M(Meb, L, s[0], rng))
        het.append(sim_from_M(fit_matrix(s), L, s[0], rng))
        hom.append(sim_from_M(m, L, rng.choice(starts), rng))

    _, g2_r = g2_of(real)
    _, g2_e = g2_of(eb)
    _, g2_h = g2_of(het)
    _, g2_m = g2_of(hom)
    denom = g2_r - g2_m
    return {
        "n_tracks": len(real),
        "n_block_states": int(sum(len(s) for s in real)),
        "eb_concentration_k": {STATES[a]: k[a] for a in range(NS)},
        "g2": {"real": g2_r, "eb_fair": g2_e, "het_overfit": g2_h, "hom": g2_m},
        "dwell_cv": {"real": block_dwell_cv(real), "eb_fair": block_dwell_cv(eb),
                     "het_overfit": block_dwell_cv(het), "hom": block_dwell_cv(hom)},
        "aggregation_share_eb": float((g2_e - g2_m) / denom) if denom > 0 else None,
        "aggregation_share_het_upper": float((g2_h - g2_m) / denom) if denom > 0 else None,
        "genuine_memory_share_eb": float((g2_r - g2_e) / denom) if denom > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = {}
    for cname, cdir in (("orig20", ORIG), ("extra57", EXTRA)):
        print(f"=== {cname} (decorrelated 0.5s blocks) ===", flush=True)
        r = analyse(cdir, args.max_tracks, args.seed)
        res[cname] = r
        g = r["g2"]
        print(f"  {r['n_tracks']} tracks, {r['n_block_states']} block-states")
        print(f"  EB concentration k per context: "
              + "  ".join(f"{s}={r['eb_concentration_k'][s]:.1f}" for s in STATES))
        print(f"  g2:  REAL {g['real']:+.4f} | EB-fair {g['eb_fair']:+.4f} | "
              f"HET-overfit {g['het_overfit']:+.4f} | HOM {g['hom']:+.4f}")
        print(f"  aggregation share: EB-fair={r['aggregation_share_eb']:.0%}  "
              f"(HET upper bound={r['aggregation_share_het_upper']:.0%})")
        print(f"  genuine within-cell memory share (EB): "
              f"{r['genuine_memory_share_eb']:.0%}\n")

    json.dump(res, open(OUTDIR / "mover_stayer_eb.json", "w"), indent=2, default=float)
    print(f"saved -> {OUTDIR/'mover_stayer_eb.json'}")


if __name__ == "__main__":
    main()
