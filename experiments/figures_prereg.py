#!/usr/bin/env python3
"""Figures for the pre-registered kinematic-phenotyping pilot (pre-reg section 9).

  Fig P1  clinical prediction: MAE by model (BCa CIs) + predicted-vs-observed DFI.
  Fig P2  H3 model selection: BIC curve (k* marked) + bootstrap ARI stability.
  Fig P3  H3 geometry: tracks in (VCL, STR) coloured by GMM component vs WHO planes.
  Fig P4  per-video GMM composition + survivorship diagnostic (DUR vs DFI).

Reads outputs/prereg/{predict_clinical,cluster_tracks}.json, cluster_labels.csv,
features_{track,sample}.csv. Recomputes DFI predictions via predict_clinical.
Output: outputs/prereg/figures/*.png
Usage:  python -m experiments.figures_prereg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.predict_clinical import (  # noqa: E402
    load_clinical, pick_col, nested_loo_predict, mean_predict,
    SET_A, NON_FEATURE, TARGETS,
)

PRE = ROOT / "outputs" / "prereg"
FIG = PRE / "figures"
COL = {"A": "#8c8c8c", "B": "#4c72b0", "C": "#c44e52", "Bperp": "#55a868", "mean": "#cccccc"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _load_design():
    samp = pd.read_csv(PRE / "features_sample.csv")
    clin = load_clinical()
    bcols = [c for c in samp.columns if c not in NON_FEATURE]
    merged = samp.set_index("video").join(
        clin[[pick_col(clin, TARGETS[t]) for t in TARGETS]].rename(
            columns={pick_col(clin, TARGETS[t]): t for t in TARGETS}), how="left")
    return merged, bcols


def fig1(pred_json):
    merged, bcols = _load_design()
    m = merged["DFI"].notna().values
    y = merged["DFI"].values.astype(float)[m]
    A = merged[SET_A].values.astype(float)[m]
    B = merged[bcols].values.astype(float)[m]
    pA = nested_loo_predict("A", A, B, y)
    pC = nested_loo_predict("C", A, B, y)

    pt = pred_json["targets"]["DFI"]["point"]
    tests = pred_json["targets"]["DFI"]["tests"]
    order = ["mean", "A", "B", "Bperp", "C"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = np.arange(len(order))
    maes = [pt[k]["mae"] for k in order]
    los = [pt[k]["mae"] - pt[k]["mae_ci"][0] for k in order]
    his = [pt[k]["mae_ci"][1] - pt[k]["mae"] for k in order]
    ax[0].bar(xs, maes, yerr=[los, his], capsize=4,
              color=[COL[k] for k in order], edgecolor="k", linewidth=0.6)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(order)
    ax[0].set_ylabel("LOO MAE for DFI (%)")
    ax[0].set_title(f"DFI prediction (n={len(y)})\n"
                    f"incremental C vs A: p={tests['incremental_C_vs_A']['p']:.3f} | "
                    f"Bperp vs null: p={tests['orthogonal_Bperp_vs_null']['p']:.3f}")

    ax[1].scatter(y, pA, c=COL["A"], label=f"WHO-3 (A) ρ={pt['A']['rho']:.2f}", s=40)
    ax[1].scatter(y, pC, c=COL["C"], label=f"WHO-3+kin (C) ρ={pt['C']['rho']:.2f}", s=40)
    lim = [min(y.min(), pA.min(), pC.min()), max(y.max(), pA.max(), pC.max())]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlabel("observed DFI (%)"); ax[1].set_ylabel("LOO-predicted DFI (%)")
    ax[1].legend(fontsize=8); ax[1].set_title("Predicted vs observed")
    fig.tight_layout(); fig.savefig(FIG / "figP1_clinical.png", dpi=150); plt.close(fig)
    print("saved figP1_clinical.png")


def fig2(cl):
    bic = {int(k): v for k, v in cl["bic_curve"].items()}
    ks = sorted(bic)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, [bic[k] for k in ks], "o-", color="#4c72b0")
    ax[0].axvline(cl["kstar"], color="#c44e52", ls="--", label=f"k*={cl['kstar']}")
    ax[0].axvline(3, color="gray", ls=":", label="WHO k=3")
    ax[0].set_xlabel("GMM components k"); ax[0].set_ylabel("BIC"); ax[0].legend()
    ax[0].set_title("Model selection")
    st = cl["stability"]
    ax[1].bar([0], [st["median_ari"]],
              yerr=[[st["median_ari"] - st["ari_q05"]], [st["ari_q95"] - st["median_ari"]]],
              capsize=5, color="#55a868", edgecolor="k")
    ax[1].axhline(0.6, color="r", ls="--", label="stability 0.6")
    ax[1].set_xticks([0]); ax[1].set_xticklabels(["bootstrap ARI"])
    ax[1].set_ylim(0, 1); ax[1].legend()
    ax[1].set_title(f"Stability (median ARI={st['median_ari']:.2f}, "
                    f"NMI vs WHO={cl['geometry_vs_who']['nmi']:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "figP2_selection.png", dpi=150); plt.close(fig)
    print("saved figP2_selection.png")


def fig3(cl):
    df = pd.read_csv(PRE / "cluster_labels.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for c in sorted(df["gmm_comp"].unique()):
        s = df[df["gmm_comp"] == c]
        ax.scatter(s["VCL"], s["STR"], s=8, alpha=0.5, label=f"comp {c}")
    ax.axvline(25, color="k", ls="--", lw=0.8)
    ax.axvline(5, color="k", ls=":", lw=0.8)
    ax.axhline(0.5, color="k", ls="-.", lw=0.8)
    ax.set_xlabel("VCL (µm/s)"); ax.set_ylabel("STR")
    ax.set_xlim(0, min(200, df["VCL"].quantile(0.99)))
    ax.set_title(f"GMM geometry vs WHO planes (k*={cl['kstar']}, "
                 f"NMI={cl['geometry_vs_who']['nmi']:.2f})")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout(); fig.savefig(FIG / "figP3_geometry.png", dpi=150); plt.close(fig)
    print("saved figP3_geometry.png")


def fig4(pred_json):
    df = pd.read_csv(PRE / "cluster_labels.csv")
    comp = pd.crosstab(df["video"], df["gmm_comp"], normalize="index")
    merged, _ = _load_design()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    bottom = np.zeros(len(comp))
    for c in comp.columns:
        ax[0].bar(range(len(comp)), comp[c].values, bottom=bottom, label=f"comp {c}")
        bottom += comp[c].values
    ax[0].set_xticks(range(len(comp))); ax[0].set_xticklabels(comp.index, rotation=90, fontsize=7)
    ax[0].set_ylabel("component fraction"); ax[0].set_xlabel("video")
    ax[0].set_title("Per-video GMM composition (batch guard)"); ax[0].legend(fontsize=8)

    mm = merged["DFI"].notna()
    rho = pred_json["targets"]["DFI"]["diagnostic_DURmed_vs_target_rho"]
    ax[1].scatter(merged["DUR_med"][mm], merged["DFI"][mm], s=40, color="#4c72b0")
    ax[1].set_xlabel("median track duration (s)"); ax[1].set_ylabel("DFI (%)")
    ax[1].set_title(f"Survivorship diagnostic\nSpearman(DUR, DFI)={rho:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "figP4_diagnostics.png", dpi=150); plt.close(fig)
    print("saved figP4_diagnostics.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pred_json = json.loads((PRE / "predict_clinical.json").read_text())
    cl = json.loads((PRE / "cluster_tracks.json").read_text())
    fig1(pred_json)
    fig2(cl)
    fig3(cl)
    fig4(pred_json)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
