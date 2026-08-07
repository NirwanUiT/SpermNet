#!/usr/bin/env python3
"""Confirmatory prediction protocol (pre-reg sections 5-6): does single-cell
kinematic structure predict DNA integrity beyond WHO motility categories?

Leakage-safe: clinical columns are merged ONLY here (blinding by ordering).

Models (feature sets, section 4.2):
  A     = WHO-3 (baseline)
  B     = 24 kinematic-distribution features
  C     = A + B (INCREMENTAL test; H1 primary)
  Bperp = B residualised on A within each training fold (H1b primary)
  mean  = training-mean predictor (null)

Ridge, nested leave-one-video-out. Inner alpha via the exact ridge LOO/hat-matrix
closed form. Inference by paired permutation (10k) on the target vector; BCa
bootstrap CIs. Secondary targets (HDS/vitality/morphology) Holm-corrected (H2).
Rank (Spearman) co-primary reported alongside MAE.

Prior context (declared in the plan): a memory/dynamics DFI test already ran null
(experiments/dfi_predict.py). This asks the distinct kinematic-distribution question.

Output: outputs/prereg/predict_clinical.json
Usage:  python -m experiments.predict_clinical [--perm 10000] [--boot 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "outputs" / "prereg" / "features_sample.csv"
CLIN = ROOT / "data" / "raw" / "visem_full_clinical" / "semen_analysis_data.csv"
OUT = ROOT / "outputs" / "prereg" / "predict_clinical.json"

SET_A = ["prog_pct", "nonprog_pct", "immot_pct"]
NON_FEATURE = ["video", "DUR_med", "n_tracks", *SET_A]
# Grid spans up to strong shrinkage so a p=27 model at n=18 can regularise toward
# the training mean rather than explode (see plan deviations log 2026-08-07).
ALPHAS = np.logspace(-3, 6, 19)

TARGETS = {
    "DFI": "DNA fragmentation index",
    "HDS": "High DNA stainability",
    "vitality": "Sperm vitality",
    "morphology": "Normal spermatozoa",
}
SECONDARY = ["HDS", "vitality", "morphology"]


# ── clinical ────────────────────────────────────────────────────────────────
def load_clinical() -> pd.DataFrame:
    df = pd.read_csv(CLIN, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip().str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID")


def pick_col(clin: pd.DataFrame, needle: str) -> str:
    hits = [c for c in clin.columns if needle.lower() in c.lower()]
    if not hits:
        raise KeyError(needle)
    return hits[0]


# ── ridge nested LOO ────────────────────────────────────────────────────────
def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _inner_loo_mae(Xs, yc, alphas):
    """Ridge LOO MAE for every alpha via the hat-matrix shortcut (no intercept:
    Xs standardised, yc centred)."""
    U, s, _ = np.linalg.svd(Xs, full_matrices=False)
    Uty = U.T @ yc
    U2 = U ** 2
    out = np.empty(len(alphas))
    for k, a in enumerate(alphas):
        d = s ** 2 / (s ** 2 + a)
        yhat = U @ (d * Uty)
        h = U2 @ d
        denom = np.clip(1.0 - h, 1e-8, None)
        out[k] = np.mean(np.abs((yc - yhat) / denom))
    return out


def _build_design(name, A, B, tr, te):
    if name == "A":
        return A[tr], A[te]
    if name == "B":
        return B[tr], B[te]
    if name == "C":
        return np.hstack([A, B])[tr], np.hstack([A, B])[te]
    if name == "Bperp":
        # residualise B on [1, A] using training rows only.
        Atr = np.hstack([np.ones((len(tr), 1)), A[tr]])
        coef, *_ = np.linalg.lstsq(Atr, B[tr], rcond=None)
        Ate = np.hstack([np.ones((len(te), 1)), A[te]])
        return B[tr] - Atr @ coef, B[te] - Ate @ coef
    raise ValueError(name)


def nested_loo_predict(name, A, B, y):
    n = len(y)
    preds = np.empty(n)
    for j in range(n):
        tr = np.array([i for i in range(n) if i != j])
        te = np.array([j])
        Xtr, Xte = _build_design(name, A, B, tr, te)
        Xs, Xes = _standardise(Xtr, Xte)
        ymean = y[tr].mean()
        yc = y[tr] - ymean
        maes = _inner_loo_mae(Xs, yc, ALPHAS)
        alpha = ALPHAS[int(np.argmin(maes))]
        w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ yc)
        preds[j] = float((Xes @ w)[0]) + ymean
    return preds


def mean_predict(y):
    n = len(y)
    return np.array([y[[i for i in range(n) if i != j]].mean() for j in range(n)])


# ── metrics ─────────────────────────────────────────────────────────────────
def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))


def r2_loo(pred, y):
    sse = np.sum((y - pred) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - sse / sst) if sst > 0 else float("nan")


def rho(pred, y):
    return float(spearmanr(pred, y).statistic)


# ── inference ───────────────────────────────────────────────────────────────
def perm_test(stat_obs, stat_perm_fn, y, n_perm, seed, greater=True):
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        s = stat_perm_fn(yp)
        if (s >= stat_obs) if greater else (s <= stat_obs):
            count += 1
    return (1 + count) / (n_perm + 1)


def bca_ci(stat_fn, m, n_boot, seed, alpha=0.05):
    """BCa CI for a statistic computed on m items resampled by index."""
    rng = np.random.default_rng(seed)
    theta = stat_fn(np.arange(m))
    boots = np.array([stat_fn(rng.integers(0, m, m)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return float("nan"), float("nan")
    z0 = norm.ppf(np.clip(np.mean(boots < theta), 1e-6, 1 - 1e-6))
    jack = np.array([stat_fn(np.delete(np.arange(m), i)) for i in range(m)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5 + 1e-12)
    a = num / den
    def adj(q):
        z = norm.ppf(q)
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.percentile(boots, 100 * adj(alpha / 2))
    hi = np.percentile(boots, 100 * adj(1 - alpha / 2))
    return float(lo), float(hi)


def evaluate_target(A, B, y, name, n_perm, n_boot, seed):
    preds = {m: nested_loo_predict(m, A, B, y) for m in ["A", "B", "C", "Bperp"]}
    preds["mean"] = mean_predict(y)
    point = {m: {"mae": mae(p, y), "r2": r2_loo(p, y), "rho": rho(p, y)}
             for m, p in preds.items()}

    # BCa CIs on MAE (abs-error vector) and Spearman (pred-obs pairs).
    for m, p in preds.items():
        ae = np.abs(p - y)
        point[m]["mae_ci"] = bca_ci(lambda idx: np.mean(ae[idx]), len(y), n_boot, seed)
        point[m]["rho_ci"] = bca_ci(
            lambda idx: spearmanr(p[idx], y[idx]).statistic, len(y), n_boot, seed + 1)

    res = {"n": int(len(y)), "point": point}

    # H1 incremental: MAE(A) - MAE(C) > 0  (C better).
    d_inc = point["A"]["mae"] - point["C"]["mae"]
    p_inc = perm_test(
        d_inc, lambda yp: mae(nested_loo_predict("A", A, B, yp), yp)
                          - mae(nested_loo_predict("C", A, B, yp), yp),
        y, n_perm, seed, greater=True)
    # rank co-primary: rho(C) - rho(A) > 0.
    d_rank = point["C"]["rho"] - point["A"]["rho"]
    p_rank = perm_test(
        d_rank, lambda yp: rho(nested_loo_predict("C", A, B, yp), yp)
                           - rho(nested_loo_predict("A", A, B, yp), yp),
        y, n_perm, seed + 2, greater=True)
    # H1b orthogonalised: MAE(mean) - MAE(Bperp) > 0 (Bperp beats null).
    d_orth = point["mean"]["mae"] - point["Bperp"]["mae"]
    p_orth = perm_test(
        d_orth, lambda yp: mae(mean_predict(yp), yp)
                           - mae(nested_loo_predict("Bperp", A, B, yp), yp),
        y, n_perm, seed + 3, greater=True)
    # descriptive raw B vs A.
    d_bva = point["A"]["mae"] - point["B"]["mae"]

    res["tests"] = {
        "incremental_C_vs_A": {"delta_mae": d_inc, "p": p_inc},
        "rank_C_vs_A": {"delta_rho": d_rank, "p": p_rank},
        "orthogonal_Bperp_vs_null": {"delta_mae": d_orth, "p": p_orth},
        "descriptive_B_vs_A": {"delta_mae": d_bva},
    }
    return res


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - rank) * p))
        out[k] = adj
        prev = adj
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samp = pd.read_csv(SAMPLE)
    clin = load_clinical()
    set_b_cols = [c for c in samp.columns if c not in NON_FEATURE]
    assert len(set_b_cols) == 24, len(set_b_cols)

    merged = samp.set_index("video").join(
        clin[[pick_col(clin, TARGETS[t]) for t in TARGETS]].rename(
            columns={pick_col(clin, TARGETS[t]): t for t in TARGETS}),
        how="left")

    A_all = merged[SET_A].values.astype(float)
    B_all = merged[set_b_cols].values.astype(float)

    results = {"cohort": "orig20 (clean, hand-annotated)",
               "n_videos": int(len(merged)),
               "n_perm": args.perm, "n_boot": args.boot,
               "set_b_features": set_b_cols,
               "prior_dfi_null": "experiments/dfi_predict.py (memory features, n=77, p=0.567)",
               "targets": {}}

    for t in TARGETS:
        mask = merged[t].notna().values
        y = merged[t].values.astype(float)[mask]
        A, B = A_all[mask], B_all[mask]
        print(f"\n=== {t} (n={mask.sum()}) ===")
        r = evaluate_target(A, B, y, t, args.perm, args.boot, args.seed)
        # survivorship diagnostic: median DUR vs target.
        dur = merged["DUR_med"].values[mask]
        r["diagnostic_DURmed_vs_target_rho"] = float(spearmanr(dur, y).statistic)
        results["targets"][t] = r
        ti = r["tests"]
        print(f"  MAE  A={r['point']['A']['mae']:.3f}  C={r['point']['C']['mae']:.3f} "
              f"Bperp={r['point']['Bperp']['mae']:.3f}  mean={r['point']['mean']['mae']:.3f}")
        print(f"  incremental C_vs_A: dMAE={ti['incremental_C_vs_A']['delta_mae']:.3f} "
              f"p={ti['incremental_C_vs_A']['p']:.4f} | rank p={ti['rank_C_vs_A']['p']:.4f} "
              f"| Bperp_vs_null p={ti['orthogonal_Bperp_vs_null']['p']:.4f}")

    # H2 Holm across secondary targets (incremental test).
    sec_p = {t: results["targets"][t]["tests"]["incremental_C_vs_A"]["p"] for t in SECONDARY}
    results["H2_holm_incremental"] = holm(sec_p)
    results["primary_DFI_incremental_p"] = results["targets"]["DFI"]["tests"]["incremental_C_vs_A"]["p"]

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nprimary DFI incremental p = {results['primary_DFI_incremental_p']:.4f}")
    print(f"H2 Holm (secondary incremental): {results['H2_holm_incremental']}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
