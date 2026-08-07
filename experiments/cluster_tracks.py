#!/usr/bin/env python3
"""H3 structural analysis (pre-reg section 7): does the single-cell kinematic
distribution recover the WHO trichotomy?

Pooled per-track features (VCL, LIN, ALH, BCF, PWR, TAC), globally z-scored:
  1. GMM k=1..8 (full cov, 50 restarts) -> BIC curve, selected k*.
  2. Stability: 500 bootstrap GMMs refit at k*, ARI vs full-data labelling.
  3. Geometry vs WHO: NMI(GMM components, WHO classes via frozen thresholds).
  4. Batch guard: chi2(component, video); flag any video >50% of a component.
  5. Locked interpretation rule -> H3 supported / trichotomy-is-fine.

Reads outputs/prereg/features_track.csv. NO clinical data.
Output: outputs/prereg/cluster_tracks.json (+ per-track labels csv for figures).
Usage:  python -m experiments.cluster_tracks [--restarts 50] [--boot 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "outputs" / "prereg" / "features_track.csv"
OUT = ROOT / "outputs" / "prereg" / "cluster_tracks.json"
LABELS = ROOT / "outputs" / "prereg" / "cluster_labels.csv"

FEATS = ["VCL", "LIN", "ALH", "BCF", "PWR", "TAC"]
KMAX = 10


def fit_gmm(X: np.ndarray, k: int, restarts: int, seed: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=k, covariance_type="full", n_init=restarts,
        max_iter=500, reg_covar=1e-5, random_state=seed,
    ).fit(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=50)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(IN)
    X_raw = df[FEATS].values.astype(float)
    mu, sd = X_raw.mean(0), X_raw.std(0)
    sd[sd == 0] = 1.0
    X = (X_raw - mu) / sd
    n = len(X)
    print(f"{n} tracks x {len(FEATS)} features")

    # 1. BIC curve.
    bic = {}
    models = {}
    for k in range(1, KMAX + 1):
        g = fit_gmm(X, k, args.restarts, args.seed)
        bic[k] = float(g.bic(X))
        models[k] = g
        print(f"  k={k}  BIC={bic[k]:.1f}")
    kstar = int(min(bic, key=bic.get))
    print(f"BIC-selected k* = {kstar}")

    gstar = models[kstar]
    labels = gstar.predict(X)

    # 2. Stability via bootstrap.
    aris = []
    rng = np.random.default_rng(args.seed)
    for b in range(args.boot):
        idx = rng.integers(0, n, n)
        gb = fit_gmm(X[idx], kstar, 10, int(rng.integers(1e9)))
        aris.append(adjusted_rand_score(labels, gb.predict(X)))
    aris = np.array(aris)
    med_ari = float(np.median(aris))
    stable = med_ari >= 0.6
    print(f"median ARI (stability) = {med_ari:.3f}  -> {'STABLE' if stable else 'UNSTABLE'}")

    # 3. Geometry vs WHO.
    who = df["motility"].values
    nmi = float(normalized_mutual_info_score(who, labels))
    ari_who = float(adjusted_rand_score(who, labels))
    print(f"NMI(GMM, WHO) = {nmi:.3f}   ARI(GMM, WHO) = {ari_who:.3f}")

    # 4. Batch guard.
    ct = pd.crosstab(labels, df["video"])
    chi2, p_chi2, _, _ = chi2_contingency(ct)
    comp_frac = ct.div(ct.sum(axis=1), axis=0)  # per component, fraction from each video
    dominated = {int(c): {"video": int(comp_frac.columns[comp_frac.loc[c].values.argmax()]),
                          "frac": float(comp_frac.loc[c].max())}
                 for c in comp_frac.index}
    flags = [c for c, d in dominated.items() if d["frac"] > 0.5]
    print(f"batch chi2 p={p_chi2:.2e} | video>50%-of-component flags: {flags}")

    # 5. Locked interpretation rule.
    supported = (stable and kstar != 3) or (kstar == 3 and nmi < 0.5)
    # Honesty guard: if BIC has no interior minimum (k* sits at the cap), the data
    # are better described as a CONTINUUM than as k* discrete modes; H3(a) still
    # holds (the WHO 3-way split is not the natural description) but we must NOT
    # claim a specific number of sperm 'types'.
    no_interior_min = kstar == KMAX
    if supported:
        if no_interior_min:
            verdict = (
                "H3 SUPPORTED (continuum reading): BIC keeps improving to the k cap "
                f"(k*={kstar}), so the kinematic distribution is a CONTINUUM, not {kstar} "
                f"discrete modes; the WHO trichotomy (NMI={nmi:.2f}) is a coarse "
                "discretisation of it, not its natural geometry.")
        else:
            verdict = (
                "H3 SUPPORTED: kinematic geometry does NOT reduce to the WHO trichotomy "
                f"(k*={kstar}"
                + (", stable, k!=3" if kstar != 3 else f", k=3 but NMI={nmi:.2f}<0.5") + ").")
    else:
        verdict = ("H3 NOT supported: the WHO trichotomy is a reasonable compression "
                   f"(k*={kstar}, NMI={nmi:.2f}, median ARI={med_ari:.2f}).")
    print(verdict)

    df_out = df.copy()
    df_out["gmm_comp"] = labels
    df_out.to_csv(LABELS, index=False)

    out = {
        "n_tracks": n, "features": FEATS,
        "bic_curve": bic, "kstar": kstar,
        "stability": {"median_ari": med_ari, "ari_q05": float(np.percentile(aris, 5)),
                      "ari_q95": float(np.percentile(aris, 95)), "stable": stable,
                      "n_boot": args.boot},
        "geometry_vs_who": {"nmi": nmi, "ari": ari_who},
        "batch_guard": {"chi2": float(chi2), "p": float(p_chi2),
                        "dominant_video_per_component": dominated, "flagged": flags},
        "component_means_zspace": {int(c): gstar.means_[c].tolist() for c in range(kstar)},
        "component_weights": gstar.weights_.tolist(),
        "bic_no_interior_min": bool(no_interior_min),
        "h3_supported": bool(supported), "verdict": verdict,
        "zscore": {"mu": mu.tolist(), "sd": sd.tolist()},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
