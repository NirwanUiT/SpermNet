"""Pre-registered DFI prediction test (Tier 1 primary analysis).

Question (single primary endpoint, fixed BEFORE seeing results):
  Do sperm-motility *dynamics / memory* features add out-of-sample predictive
  value for the DNA-fragmentation index (DFI) BEYOND the static CASA snapshot
  (kinematics + motility composition) that a standard semen analysis reports?

Design:
  - n ~ 77 participants, target = DFI (%).
  - Two nested feature blocks:
        CASA   = kinematics means/sds (VCL,VSL,VAP,LIN,STR,WOB,ALH,BCF)
                 + motility composition (frac progressive/nonprog/immotile)
        MEMORY = self-transition (dwell) probs, directional hysteresis
                 asymmetries, dwell-time mean & CV per state.
  - Model: median-impute -> standardize -> Ridge (alpha by inner 5-fold CV).
  - Evaluation: RepeatedKFold (5 folds x 20 repeats); pool out-of-fold
    predictions per repeat; score = Spearman rho(pred, true) and R2.
  - PRIMARY ENDPOINT: delta_rho = rho(CASA+MEMORY) - rho(CASA), tested against a
    label-permutation null (is the improvement above chance?).
  - Secondary: MEMORY-only performance; RandomForest robustness; per-feature
    univariate Spearman vs DFI (FDR-corrected) for interpretability only.

Honesty: DFI is a clinically-validated SURROGATE, not a reproductive outcome.
A null result is reported as a null. No endpoint switching.

Usage: python -m experiments.dfi_predict
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "outputs" / "markov" / "dfi_features.csv"
OUT = ROOT / "outputs" / "markov" / "dfi_prediction.json"

CASA_COLS = (
    [f"{k}_{m}" for k in ["VCL", "VSL", "VAP", "LIN", "STR", "WOB", "ALH", "BCF"]
     for m in ["mean", "sd"]]
    + ["frac_progressive", "frac_nonprog", "frac_immotile"]
)
MEM_COLS = (
    ["P_stay_prog", "P_stay_nonprog", "P_stay_immotile",
     "asym_Prog_Non-", "asym_Non-_Immo", "asym_Prog_Immo"]
    + [f"dwell_mean_{s}" for s in ["Prog", "Non-", "Immo"]]
    + [f"dwell_cv_{s}" for s in ["Prog", "Non-", "Immo"]]
)


def ridge_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("est", GridSearchCV(Ridge(),
                             {"alpha": [0.1, 1, 3, 10, 30, 100, 300]},
                             cv=KFold(5, shuffle=True, random_state=0),
                             scoring="neg_mean_squared_error")),
    ])


def rf_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", RandomForestRegressor(n_estimators=400, max_depth=4,
                                      min_samples_leaf=5, random_state=0)),
    ])


def repeated_oof(X, y, make_pipe, n_splits=5, n_repeats=20, seed=0):
    """Out-of-fold predictions pooled per repeat -> list of (rho, r2)."""
    rng = np.random.default_rng(seed)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    preds = np.full((n_repeats, len(y)), np.nan)
    rep = -1
    for i, (tr, te) in enumerate(rkf.split(X)):
        if i % n_splits == 0:
            rep += 1
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        preds[rep, te] = pipe.predict(X[te])
    scores = []
    for r in range(n_repeats):
        p = preds[r]
        rho = stats.spearmanr(p, y).correlation
        ss_res = np.sum((y - p) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        scores.append((rho, r2))
    return np.array(scores), preds


def main() -> None:
    df = pd.read_csv(FEAT)
    y = df["DFI"].values.astype(float)
    casa = [c for c in CASA_COLS if c in df.columns]
    mem = [c for c in MEM_COLS if c in df.columns]
    both = casa + mem
    print(f"n={len(df)} participants | CASA feats={len(casa)} | MEM feats={len(mem)}")
    print(f"DFI: mean={y.mean():.1f}  sd={y.std():.1f}  range=[{y.min():.0f},{y.max():.0f}]")
    print(f"sources: {df['source'].value_counts().to_dict()}")

    res = {"n": int(len(df)), "casa_feats": casa, "mem_feats": mem,
           "dfi_mean": float(y.mean()), "dfi_sd": float(y.std())}

    blocks = {"CASA": casa, "MEMORY": mem, "CASA+MEMORY": both}
    oof_store = {}
    print("\n=== Ridge, RepeatedKFold(5x20), pooled OOF ===")
    for name, cols in blocks.items():
        X = df[cols].values.astype(float)
        scores, preds = repeated_oof(X, y, ridge_pipe)
        oof_store[name] = scores[:, 0]
        rho_m, rho_s = scores[:, 0].mean(), scores[:, 0].std()
        r2_m, r2_s = scores[:, 1].mean(), scores[:, 1].std()
        res[f"ridge_{name}"] = {"rho_mean": float(rho_m), "rho_sd": float(rho_s),
                                "r2_mean": float(r2_m), "r2_sd": float(r2_s)}
        print(f"  {name:14s}  rho={rho_m:+.3f}±{rho_s:.3f}   R2={r2_m:+.3f}±{r2_s:.3f}")

    # ---- PRIMARY ENDPOINT: does MEMORY add to CASA? ----
    d = oof_store["CASA+MEMORY"] - oof_store["CASA"]
    delta = float(d.mean())
    # paired across repeats (descriptive; repeats are correlated so also permute)
    t_p = float(stats.wilcoxon(oof_store["CASA+MEMORY"], oof_store["CASA"]).pvalue) \
        if np.any(d != 0) else 1.0
    print(f"\n*** PRIMARY: delta_rho(CASA+MEM - CASA) = {delta:+.3f} "
          f"(Wilcoxon across repeats p={t_p:.3g}) ***")

    # ---- label-permutation null for the improvement ----
    rng = np.random.default_rng(0)
    nperm = 200
    Xc = df[casa].values.astype(float)
    Xb = df[both].values.astype(float)
    null = np.zeros(nperm)
    for k in range(nperm):
        yp = rng.permutation(y)
        sc_c, _ = repeated_oof(Xc, yp, ridge_pipe, n_repeats=5, seed=k)
        sc_b, _ = repeated_oof(Xb, yp, ridge_pipe, n_repeats=5, seed=k)
        null[k] = sc_b[:, 0].mean() - sc_c[:, 0].mean()
    p_perm = float((np.sum(null >= delta) + 1) / (nperm + 1))
    res["primary"] = {"delta_rho": delta, "wilcoxon_p": t_p,
                      "perm_p": p_perm, "nperm": nperm}
    print(f"*** permutation p (improvement above chance) = {p_perm:.3f} ***")

    # ---- RandomForest robustness ----
    print("\n=== RandomForest robustness (5x10) ===")
    for name, cols in blocks.items():
        X = df[cols].values.astype(float)
        scores, _ = repeated_oof(X, y, rf_pipe, n_repeats=10)
        res[f"rf_{name}"] = {"rho_mean": float(scores[:, 0].mean()),
                             "r2_mean": float(scores[:, 1].mean())}
        print(f"  {name:14s}  rho={scores[:,0].mean():+.3f}  R2={scores[:,1].mean():+.3f}")

    # ---- univariate (interpretation only, FDR) ----
    print("\n=== Univariate Spearman vs DFI (BH-FDR) ===")
    uni = []
    for c in both:
        x = df[c].values.astype(float)
        ok = ~np.isnan(x)
        rho = stats.spearmanr(x[ok], y[ok]).correlation
        p = stats.spearmanr(x[ok], y[ok]).pvalue
        uni.append((c, rho, p))
    uni.sort(key=lambda r: r[2])
    ps = np.array([u[2] for u in uni])
    m = len(ps)
    bh = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
    res["univariate"] = []
    for (c, rho, p), q in zip(uni, bh):
        block = "MEM" if c in mem else "CASA"
        res["univariate"].append({"feat": c, "block": block, "rho": float(rho),
                                  "p": float(p), "q": float(q)})
        if q < 0.10:
            print(f"  [{block}] {c:18s} rho={rho:+.3f}  p={p:.3g}  q={q:.3g}")
    if not any(u["q"] < 0.10 for u in res["univariate"]):
        print("  (no feature survives BH-FDR q<0.10)")

    import json
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(f"\nsaved -> {OUT}")

    # ---- honest verdict ----
    print("\n================ VERDICT ================")
    best = max(["CASA", "MEMORY", "CASA+MEMORY"],
               key=lambda b: res[f"ridge_{b}"]["rho_mean"])
    print(f"best block (Ridge rho): {best} ({res['ridge_'+best]['rho_mean']:+.3f})")
    if p_perm < 0.05 and delta > 0:
        print("=> Memory features ADD significant out-of-sample DFI signal beyond CASA.")
    else:
        print("=> Memory features do NOT add significant signal beyond CASA (honest null).")


if __name__ == "__main__":
    main()
