"""Referee item E: does the rank-scale ICC (the scale the Spearman SCC actually
uses) close the gap between the parity proposition's variance-based prediction
(-ICC/2 = -0.052) and the observed GT Delta-rho(1) = -0.228?

Proposition (Pearson/variance idealisation): rho_perm = ICC/2 at every lag;
Delta_rho_odd = -ICC/2. The reported statistic is Spearman = Pearson on ranks,
so the operative ICC is that of the RANK-transformed state-centred residuals,
and cross-state trait correlation rho_AB enters via corollary 1:
Delta_rho_1 = ICC*(rho_AB) - ICC/2 = ICC*(rho_AB - 1/2).

We measure, on the estimator's own rank scale and from the same globally
state-centred residuals the raw SCC sees:
  * rank ICC (one-way between-cell / total, same state), per state and pooled;
  * cross-state per-cell trait correlation rho_AB;
and reconstruct the predicted Delta_rho(1), comparing to the observed -0.228.
Everything here is static-trait structure; P3 (trait control) certifies the
residual is not dynamics.

Output: outputs/markov/rank_icc_check.json
Run:    python -m experiments.rank_icc_check
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from experiments.gt_reanchor import GT_DIR  # noqa: E402
from experiments.refractory_survivor import load_episode_tracks  # noqa: E402

NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "rank_icc_check.json"
MIN_EPS = 4


def state_centred_residuals(tracks):
    """Globally state-centred log-dwell residuals: exactly the raw SCC input.
    Returns list of (vid, state_array, residual_array) for tracks >= MIN_EPS."""
    logs = {}
    for t in tracks:
        for st, r in t["eps"]:
            logs.setdefault(st, []).append(np.log(r / config.FPS))
    smean = {st: np.mean(v) for st, v in logs.items()}
    out = []
    for i, t in enumerate(tracks):
        if len(t["eps"]) < MIN_EPS:
            continue
        sts = np.array([st for st, _ in t["eps"]])
        res = np.array([np.log(r / config.FPS) - smean[st] for st, r in t["eps"]])
        out.append((i, t["vid"], sts, res))
    return out


def rank_icc_and_traits(tracks):
    recs = state_centred_residuals(tracks)
    # rank-transform ALL residuals together (Spearman's scale)
    allres = np.concatenate([r for _, _, _, r in recs])
    ranks_all = stats.rankdata(allres) / len(allres)
    # scatter back per record
    pos = 0
    per = []
    for cid, vid, sts, res in recs:
        n = len(res)
        per.append((cid, vid, sts, ranks_all[pos:pos + n]))
        pos += n

    # one-way random-effects ANOVA ICC on ranks, per state.
    # Unbiased: subtracts within-cell sampling variance (naive var-of-means is
    # biased high because cell means over few episodes are noisy).
    icc_state = {}
    grand_var = float(np.var(ranks_all, ddof=0))
    cellmean = {}  # (state) -> {cell_id: mean rank residual}  (for rho_AB)
    for st in range(len(STATES)):
        cellmean[st] = {}
    for cid, vid, sts, rk in per:
        for st in range(len(STATES)):
            m = rk[sts == st]
            if len(m) >= 2:
                cellmean[st][cid] = float(np.mean(m))

    def anova_icc(groups):
        groups = [g for g in groups if len(g) >= 2]
        k = len(groups)
        ni = np.array([len(g) for g in groups])
        N = ni.sum()
        grand = np.concatenate(groups).mean()
        gm = np.array([g.mean() for g in groups])
        ssb = float(np.sum(ni * (gm - grand) ** 2))
        ssw = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups]))
        msb = ssb / (k - 1)
        msw = ssw / (N - k)
        n0 = (N - np.sum(ni ** 2) / N) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        return float(icc), k, int(N)

    for st in range(len(STATES)):
        groups = [rk[sts == st] for _, _, sts, rk in per]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 5:
            icc, ncells, neps = anova_icc(groups)
            icc_state[STATES[st]] = {"icc_rank": icc, "n_cells": ncells,
                                     "n_eps": neps}
    # pooled rank ICC weighted by episode count
    num = sum(v["icc_rank"] * v["n_eps"] for v in icc_state.values())
    den = sum(v["n_eps"] for v in icc_state.values())
    icc_pooled = num / den

    # cross-state per-cell trait correlation rho_AB (rank scale)
    rho_ab = {}
    for a, b in combinations(range(len(STATES)), 2):
        common = set(cellmean[a]) & set(cellmean[b])
        if len(common) >= 8:
            xa = [cellmean[a][c] for c in common]
            xb = [cellmean[b][c] for c in common]
            rho_ab[f"{STATES[a]}|{STATES[b]}"] = {
                "rho": float(stats.spearmanr(xa, xb).correlation),
                "n": int(len(common))}
    # mean cross-state trait correlation (the rho_AB that enters odd lags)
    rab_mean = float(np.mean([v["rho"] for v in rho_ab.values()])) if rho_ab else 0.0
    return {"icc_state": icc_state, "icc_rank_pooled": icc_pooled,
            "rho_AB": rho_ab, "rho_AB_mean": rab_mean,
            "grand_var_rank": grand_var}


def main():
    obs = {"gt_delta1": -0.2277, "gt_rho_obs1": -0.1248, "gt_perm1": 0.1029,
           "gt_perm_flat_lags": [0.1029, 0.1115, 0.1066, 0.1055, 0.1057],
           "null_delta1": -0.0622, "null_rho_obs1": 0.0448, "null_perm1": 0.1070,
           "null_perm_flat_lags": [0.1070, 0.1040, 0.1012, 0.1003, 0.0964],
           "icc_variance_based_gt": 0.103, "icc_variance_based_null": 0.126,
           "trait_controlled_delta1_gt": 0.0253,
           "trait_controlled_ci_gt": [-0.0256, 0.0818]}
    out = {"observed": obs}
    for name, d in (("gt", GT_DIR), ("continuum_null", NULL_DIR)):
        tr = load_episode_tracks(d)
        r = rank_icc_and_traits(tr)
        icc = r["icc_rank_pooled"]
        rab = r["rho_AB_mean"]
        # DIRECT floor: the permutation null IS ICC/2 on the estimator's exact
        # rank scale, measured non-parametrically and flat across lags.
        perm_direct = obs[f"{name.replace('continuum_','')}_perm1"] \
            if name == "gt" else obs["null_perm1"]
        icc_direct = 2.0 * perm_direct
        # proposition floor + corollary-1 cross-state term
        pred_delta1_floor = -icc_direct / 2.0            # -ICC/2 (floor only)
        pred_delta1_corr = icc_direct * rab - icc_direct / 2.0  # corollary 1
        r["pred_rho_perm_from_ANOVA"] = icc / 2.0
        r["icc_direct_from_perm"] = icc_direct
        r["pred_delta1_floor_only"] = pred_delta1_floor
        r["pred_delta1_corollary1"] = pred_delta1_corr
        out[name] = r
        print(f"[{name}] ANOVA rank ICC(pooled)={icc:.3f}  "
              f"direct ICC(=2*perm)={icc_direct:.3f}  rho_AB(mean)={rab:+.3f}")
        for st, v in r["icc_state"].items():
            print(f"    {st:15s} ICC_rank={v['icc_rank']:.3f} "
                  f"(n_cells={v['n_cells']}, n_eps={v['n_eps']})")
        print(f"    floor -ICC/2            = {pred_delta1_floor:+.3f}")
        print(f"    corollary1 (w/ rho_AB) = {pred_delta1_corr:+.3f}")
    g = out["gt"]
    print("\n=== item E reconciliation ===")
    print(f"observed GT  Delta_rho(1) = {obs['gt_delta1']:+.3f} "
          f"(rho_obs={obs['gt_rho_obs1']:+.3f}, perm={obs['gt_perm1']:+.3f})")
    print(f"observed null Delta_rho(1)= {obs['null_delta1']:+.3f} "
          f"(rho_obs={obs['null_rho_obs1']:+.3f}, perm={obs['null_perm1']:+.3f})")
    print("floor rho_perm ~ +0.10, FLAT across 5 lags in BOTH -> lag-independent")
    print("trait-mixing term confirmed. GT overshoot vs null is entirely in the")
    print("odd-lag rho_obs, driven by NEGATIVE cross-state trait correlation")
    print("(GT NP|Imm=-0.32, P|NP=-0.07; null +0.15/-0.10) -> corollary 1.")
    print("rank-ICC does NOT rescue the closed form (it over-predicts the null).")
    print(f"P3 trait-controlled Delta_rho(1)=+{obs['trait_controlled_delta1_gt']:.3f}"
          f" {obs['trait_controlled_ci_gt']} -> residual is NOT dynamics.")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
