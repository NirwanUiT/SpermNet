"""Mover-stayer / aggregation NULL for Finding #1 (the referee-killer test).

Finding #1 = pooled sperm-motility state sequences are NON-MARKOVIAN: on
decorrelated 0.5 s blocks, a 2nd-order model beats 1st-order by ~+0.057 logL/token.
memory_decomposition.py then showed WITHIN a single cell dwells are near-memoryless
(CV~1) while the POOLED population is dispersed (CV~2) -> the population is
heterogeneous (mover-stayer). That raises the central alternative explanation:

    Could a population of purely MEMORYLESS (1st-order) cells that merely DIFFER in
    their transition matrices reproduce the pooled 2nd-order advantage by AGGREGATION
    alone (a Simpson/mixture effect), with no within-cell memory at all?

We test this by parametric bootstrap under that null:
  1. take the REAL decorrelated-block sequences (one per track);
  2. HET null: fit each track its OWN 1st-order Markov matrix (Laplace a=0.5) and
     simulate a matched-length sequence from it. This preserves the full between-cell
     heterogeneity but every cell is EXACTLY 1st-order (memoryless).
  3. HOM null: simulate every track from the single POOLED 1st-order matrix
     (no heterogeneity) -> sanity floor, should show ~0 second-order gain.
  4. run the identical held-out 5-fold order test (cv_order) on real / HET / HOM and
     compare the 2nd-over-1st gain g2.

Logic: fitting M_i from a short track OVER-fits, injecting EXTRA idiosyncratic
heterogeneity, so HET g2 is an UPPER BOUND on the aggregation artifact. If the REAL
g2 exceeds even this inflated HET null, the memory is genuinely WITHIN-cell and
Finding #1 survives. If real ~ HET, the pooled memory is a heterogeneity artifact
and Finding #1 must be reframed as population structure, not single-cell memory.

Output: outputs/markov/mover_stayer_null.json

Usage: python -m experiments.mover_stayer_null [--max-tracks 3000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from markov_property_test import cv_order  # noqa: E402
from experiments.replicate_markov_extra import nonoverlap_sequences_from  # noqa: E402

NS = len(STATES)
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"


def fit_matrix(seq, alpha=0.5):
    M = np.full((NS, NS), alpha, float)
    for a, b in zip(seq[:-1], seq[1:]):
        M[a, b] += 1
    M /= M.sum(1, keepdims=True)
    return M


def sim_from_M(M, length, start, rng):
    s = [int(start)]
    cum = np.cumsum(M, axis=1)
    for _ in range(length - 1):
        u = rng.random()
        s.append(int(np.searchsorted(cum[s[-1]], u)))
    return s


def block_dwell_cv(seqs):
    """Pooled dwell CV (in block units) over all states, as a dispersion summary."""
    dw = []
    for seq in seqs:
        run = 1
        for k in range(1, len(seq)):
            if seq[k] == seq[k - 1]:
                run += 1
            else:
                dw.append(run); run = 1
        dw.append(run)
    dw = np.array(dw, float)
    return float(dw.std() / dw.mean())


def g2_of(seqs):
    ll = cv_order(seqs)
    return ll, float(ll[2] - ll[1])


def analyse(track_dir, max_tracks, seed):
    real = nonoverlap_sequences_from(track_dir, block=25,
                                     max_tracks=max_tracks, seed=seed)
    real = [s for s in real if len(s) >= 3]
    rng = np.random.default_rng(seed)

    # pooled matrix for HOM null
    Mpool = np.full((NS, NS), 0.5, float)
    starts = []
    for seq in real:
        starts.append(seq[0])
        for a, b in zip(seq[:-1], seq[1:]):
            Mpool[a, b] += 1
    Mpool /= Mpool.sum(1, keepdims=True)
    starts = np.array(starts)

    het, hom = [], []
    for seq in real:
        L = len(seq)
        het.append(sim_from_M(fit_matrix(seq), L, seq[0], rng))
        hom.append(sim_from_M(Mpool, L, rng.choice(starts), rng))

    ll_r, g2_r = g2_of(real)
    ll_h, g2_h = g2_of(het)
    ll_m, g2_m = g2_of(hom)
    return {
        "n_tracks": len(real),
        "n_block_states": int(sum(len(s) for s in real)),
        "real": {"ll_order": ll_r, "g2": g2_r, "dwell_cv": block_dwell_cv(real)},
        "het_null": {"ll_order": ll_h, "g2": g2_h, "dwell_cv": block_dwell_cv(het)},
        "hom_null": {"ll_order": ll_m, "g2": g2_m, "dwell_cv": block_dwell_cv(hom)},
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
        print(f"  {r['n_tracks']} tracks, {r['n_block_states']} block-states")
        for k, lab in (("real", "REAL           "),
                       ("het_null", "HET-1st null   "),
                       ("hom_null", "HOM-1st null   ")):
            b = r[k]
            print(f"  {lab} 2nd-over-1st g2 = {b['g2']:+.4f}/token   "
                  f"block-dwell CV = {b['dwell_cv']:.2f}")
        verdict = ("GENUINE within-cell memory (real >> het null)"
                   if r["real"]["g2"] > r["het_null"]["g2"] + 0.01
                   else "AMBIGUOUS: aggregation can explain it")
        print(f"  => {verdict}\n")

    json.dump(res, open(OUTDIR / "mover_stayer_null.json", "w"),
              indent=2, default=float)
    print(f"saved -> {OUTDIR/'mover_stayer_null.json'}")


if __name__ == "__main__":
    main()
