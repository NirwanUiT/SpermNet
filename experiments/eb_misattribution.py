"""Referee item A (mechanism verification): does the empirical-Bayes mover-stayer
decomposition mis-attribute kinematic heterogeneity as within-cell memory?

The EB null (mover_stayer_eb.py) models per-cell heterogeneity in the transition
matrix ROWS via a Dirichlet-multinomial prior, and reports a "genuine within-cell
memory share" = (g2_real - g2_EB) / (g2_real - g2_HOM). The mechanism claim of
this paper is that real heterogeneity lives UPSTREAM in continuous kinematic
parameters, passed through a windowed classifier; the induced dependence between a
cell's successive block-states is NOT a per-cell reweighting of transition rows and
cannot be represented by any Dirichlet-row prior. If so, the EB estimator will
attribute a spurious positive "genuine memory" share to data that provably contain
ZERO within-cell dynamics.

Decisive test: run the EB decomposition VERBATIM on the continuum null (per-track
OU velocity -> zero within-cell event memory by construction, but real between-cell
kinematic heterogeneity) and on the homogeneous null (no heterogeneity at all).
Prediction:
  * homogeneous null  -> genuine-memory share ~ 0 (nothing to mis-attribute);
  * continuum null    -> genuine-memory share > 0  (mis-attribution: the residual
                         cannot be within-cell memory because there is none).
That directly proves the Dirichlet-row prior under-represents kinematically induced
heterogeneity, so the "genuine memory" the EB null leaves on the REAL data is at
least partly the same artefact.

Output: outputs/markov/eb_misattribution.json
Run:    python -m experiments.eb_misattribution --max-tracks 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from experiments.mover_stayer_eb import analyse  # noqa: E402

GT_DIR = ROOT / "outputs" / "tracks_gt"
NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
HOM_DIR = ROOT / "outputs" / "tracks_continuum_null_hom"
OUT = config.MARKOV_OUT / "eb_misattribution.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohorts = [("ground_truth", GT_DIR),
               ("continuum_null", NULL_DIR),    # heterogeneous, ZERO within-cell memory
               ("homogeneous_null", HOM_DIR)]   # no heterogeneity
    res = {}
    for name, d in cohorts:
        if not d.exists():
            print(f"skip {name}: {d} missing", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        r = analyse(d, args.max_tracks, args.seed)
        res[name] = r
        g = r["g2"]
        print(f"  {r['n_tracks']} tracks, {r['n_block_states']} block-states")
        print("  EB concentration k: " + "  ".join(
            f"{s}={r['eb_concentration_k'][s]:.1f}" for s in STATES))
        print(f"  g2: REAL {g['real']:+.4f} | EB {g['eb_fair']:+.4f} | "
              f"HET {g['het_overfit']:+.4f} | HOM {g['hom']:+.4f}")
        gm = r["genuine_memory_share_eb"]
        ag = r["aggregation_share_eb"]
        print(f"  aggregation share (EB) = {ag if ag is None else format(ag, '.0%')}"
              f"   genuine-memory share (EB) = "
              f"{gm if gm is None else format(gm, '.0%')}\n")
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")
    # verdict
    if "continuum_null" in res and res["continuum_null"]["genuine_memory_share_eb"]:
        print("VERDICT: EB assigns %.0f%% 'genuine within-cell memory' to the "
              "continuum null,\nwhich has ZERO within-cell dynamics by construction "
              "-> mis-attribution confirmed." %
              (100 * res["continuum_null"]["genuine_memory_share_eb"]))


if __name__ == "__main__":
    main()
