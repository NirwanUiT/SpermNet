"""Assemble the paper figures for the pipeline-artefact / refractory-survivor study.

Figure 1  a memoryless continuum manufactures the phenomenology -> fig1_dwell_law.png
Figure 2  the survivor: refractory switching vs every control   -> fig2_decomposition.png
Figure 3  heterogeneity is a reliable trait but not clinically  -> fig3_heterogeneity.png
          incremental over CASA composition

Figure 1 recomputes dwell samples from outputs/tracks_gt and
outputs/tracks_continuum_null; Figures 2-3 are built from the summary JSONs
(refractory_survivor, threshold_hysteresis, per_cell_kinetics, stayer_dfi).

Usage: python -m experiments.make_paper_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from markov_analysis import STATES  # noqa: E402
from experiments.dwell_physics import dwell_episodes, EXTRA  # noqa: E402
from experiments.gt_reanchor import GT_DIR  # noqa: E402

NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
MK = ROOT / "outputs" / "markov"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"Progressive": "#1b7837", "Non-progressive": "#e08214", "Immotile": "#762a83"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def load(name):
    return json.load(open(MK / f"{name}.json"))


# ---------------------------------------------------------------- Figure 1
def figure1():
    dwell_gt = dwell_episodes(GT_DIR, max_tracks=0, seed=0)
    dwell_nl = dwell_episodes(NULL_DIR, max_tracks=0, seed=0)
    g2_gt = load("gt_reanchor")["gt"]["block_g2"]
    g2_nl = load("continuum_null")["null"]["block_g2"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # (A) dwell survival: GT (solid) vs memoryless continuum null (dashed)
    for i, s in enumerate(STATES):
        for dwell, ls, alpha in ((dwell_gt, "-", 1.0), (dwell_nl, "--", 0.75)):
            d = np.sort(dwell[i]); d = d[d > 0]
            surv = 1.0 - np.arange(1, len(d) + 1) / len(d)
            ax[0].step(d, surv, where="post", color=COL[s], lw=1.6,
                       ls=ls, alpha=alpha, label=s if ls == "-" else None)
        # exponential (memoryless-switching) reference on GT
        d = np.sort(dwell_gt[i]); d = d[d > 0]
        ax[0].plot(d, np.exp(-d / d.mean()), color=COL[s], ls=":", lw=0.9, alpha=0.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylim(1e-4, 1)
    ax[0].set_xlabel("dwell time (s)"); ax[0].set_ylabel("survival  P(T > t)")
    ax[0].set_title("A  A memoryless continuum reproduces the dwell laws",
                    loc="left", fontsize=11, weight="bold")
    ax[0].plot([], [], color="0.4", ls="-", label="ground truth")
    ax[0].plot([], [], color="0.4", ls="--", label="continuum null (no switching biology)")
    ax[0].plot([], [], color="0.4", ls=":", label="exponential reference")
    ax[0].legend(fontsize=7.5, frameon=False, loc="lower left")

    # (B) second-order memory gain: null >= observed at every window
    windows = ["13", "25", "51"]
    x = np.arange(len(windows)); w = 0.36
    ax[1].bar(x - w / 2, [g2_gt[wd]["g2"] for wd in windows], w,
              color="#2166ac", label="ground truth")
    ax[1].bar(x + w / 2, [g2_nl[wd]["g2"] for wd in windows], w,
              color="#b2182b", alpha=0.85, label="memoryless continuum null")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{wd} frames" for wd in windows])
    ax[1].set_xlabel("classifier window / block size")
    ax[1].set_ylabel(r"2nd-order memory gain $g_2$ (nats / token)")
    ax[1].set_title("B  The null manufactures MORE memory than observed",
                    loc="left", fontsize=11, weight="bold")
    ax[1].set_ylim(0, 0.063)
    ax[1].legend(fontsize=8, frameon=False, loc="upper left")
    ax[1].text(0.02, 0.80, "the window-robustness of the observed gain\n"
               "is reproduced by the null as well", transform=ax[1].transAxes,
               fontsize=7.5, color="0.35")

    fig.tight_layout()
    fig.savefig(FIG / "fig1_dwell_law.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_dwell_law.png")


# ---------------------------------------------------------------- Figure 2
def figure2():
    rs = load("refractory_survivor")
    th = load("threshold_hysteresis")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6),
                           gridspec_kw={"width_ratios": [1, 1.6]})

    # (A) forest plot: the survivor vs its controls
    rows = [
        ("ground truth", rs["gt"]["delta_rho"], rs["gt"]["ci95"], "#2166ac"),
        ("flicker-merged (34 % of episodes)", rs["gt_flicker_merged_25f"]["delta_rho"],
         rs["gt_flicker_merged_25f"]["ci95"], "#2166ac"),
        ("flicker-merged (38 % of episodes)", rs["gt_flicker_merged_50f"]["delta_rho"],
         rs["gt_flicker_merged_50f"]["ci95"], "#2166ac"),
        ("memoryless continuum null", rs["continuum_null"]["delta_rho"],
         rs["continuum_null"]["ci95"], "#b2182b"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (name, pt, ci, col) in zip(ys, rows):
        ax[0].plot(ci, [y, y], color=col, lw=2.2, solid_capstyle="butt")
        ax[0].plot(pt, y, "o", color=col, ms=7)
        ax[0].text(-0.005, y + 0.22, name, ha="right", fontsize=8.5, color="0.2")
    ax[0].axvline(0, color="k", lw=0.7, ls=":")
    ax[0].set_yticks([])
    ax[0].set_xlim(-0.38, 0.06)
    ax[0].set_xlabel(r"refractoriness  $\Delta\rho$  (video-cluster 95 % CI)")
    ax[0].set_title("A  The survivor and its controls", loc="left",
                    fontsize=11, weight="bold")
    d = rs["gt_minus_null"]
    ax[0].text(0.02, 0.03, f"GT − null = {d['point']:+.2f} "
               f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]",
               transform=ax[0].transAxes, fontsize=8, color="0.35")

    # (B) classifier sweep: GT invariant, null artefact swings with design
    cfgs = []
    for k, v in th["threshold_sweep"].items():
        cfgs.append((k, v))
    for k, v in th["window_sweep"].items():
        cfgs.append((k, v))
    for k, v in th["hysteresis"].items():
        cfgs.append((k.replace("margin_", "hyst "), v))
    x = np.arange(len(cfgs))
    for i, (name, v) in enumerate(cfgs):
        lo, hi = v["gt"]["ci95"]
        ax[1].plot([i, i], [lo, hi], color="#2166ac", lw=1.8, alpha=0.9)
        ax[1].plot(i, v["gt"]["delta_rho"], "o", color="#2166ac", ms=5,
                   label="ground truth" if i == 0 else None)
        if v.get("null_ci95"):
            nlo, nhi = v["null_ci95"]
            ax[1].plot([i + 0.22, i + 0.22], [nlo, nhi], color="#b2182b",
                       lw=1.4, alpha=0.7)
        ax[1].plot(i + 0.22, v["null_delta_rho"], "s", mfc="none",
                   mec="#b2182b", ms=5,
                   label="continuum null" if i == 0 else None)
    ax[1].axvline(10.6, color="0.8", lw=0.8)
    ax[1].axvline(12.6, color="0.8", lw=0.8)
    ax[1].text(5.0, 0.06, "threshold placement (11)", ha="center", fontsize=8, color="0.35")
    ax[1].text(11.6, 0.06, "window", ha="center", fontsize=8, color="0.35")
    ax[1].text(14.1, 0.06, "hysteresis", ha="center", fontsize=8, color="0.35")
    ax[1].axhline(0, color="k", lw=0.7, ls=":")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([n for n, _ in cfgs], rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel(r"$\Delta\rho$")
    ax[1].set_title("B  GT invariant across 16 classifier designs; "
                    "the null's artefact is design-dependent", loc="left",
                    fontsize=11, weight="bold")
    ax[1].legend(fontsize=8, frameon=False, loc="lower left")
    ax[1].text(0.99, 0.02, "hysteresis classifiers are latches: they manufacture\n"
               "refractoriness on memoryless input", transform=ax[1].transAxes,
               fontsize=7.5, color="0.35", ha="right")

    fig.tight_layout()
    fig.savefig(FIG / "fig2_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_decomposition.png")


# ---------------------------------------------------------------- Figure 3
def figure3():
    pck = load("per_cell_kinetics")
    st = load("stayer_dfi")
    cohorts = ["orig20", "extra57"]
    labels = {"orig20": "VISEM-Tracking (20)", "extra57": "independent (57)"}

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # (A) split-half reliability of heterogeneity features
    feats = ["cv_rate", "stayer_frac", "gini_rate"]
    x = np.arange(len(feats)); w = 0.38
    for c, off in zip(cohorts, (-w / 2, w / 2)):
        ax[0].bar(x + off, [pck[c]["reliability"][f] for f in feats], w,
                  label=labels[c])
    ax[0].set_xticks(x); ax[0].set_xticklabels(["CV of\nswitch rate", "stayer\nfraction",
                                                "Gini of\nswitch rate"], fontsize=8.5)
    ax[0].set_ylabel("split-half reliability (Spearman)")
    ax[0].set_ylim(0, 1); ax[0].axhline(0.8, color="0.5", ls=":", lw=0.8)
    ax[0].set_title("A  Heterogeneity is a reliable trait", loc="left",
                    fontsize=11, weight="bold")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")

    # (B) stayer->DFI collapses once CASA composition is controlled
    x = np.arange(len(cohorts)); w = 0.38
    raw = [st[c]["state_resolved"]["stayer_frac"]["rho"] for c in cohorts]
    ctl = [st[c]["stayer_given_composition"]["rho"] for c in cohorts]
    ax[1].bar(x - w / 2, raw, w, color="#c2679a", label="| track count only")
    ax[1].bar(x + w / 2, ctl, w, color="#cccccc", label="| + CASA composition")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels([labels[c] for c in cohorts], fontsize=9)
    ax[1].set_ylabel(r"stayer-fraction vs DFI  (partial $\rho$)")
    ax[1].set_title("B  Not incremental over CASA (null)", loc="left",
                    fontsize=11, weight="bold")
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")

    # (C) state-resolved: stable-progressive vs stable-immotile pull DFI oppositely
    comps = ["stable_prog_frac", "stable_immo_frac"]
    names = ["stable\nprogressive", "stable\nimmotile"]
    x = np.arange(len(comps))
    for c, off in zip(cohorts, (-w / 2, w / 2)):
        ax[2].bar(x + off, [st[c]["state_resolved"][k]["rho"] for k in comps], w,
                  label=labels[c])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=9)
    ax[2].set_ylabel(r"vs DFI  (Spearman $\rho$)")
    ax[2].set_title("C  Stayers just re-encode composition", loc="left",
                    fontsize=11, weight="bold")
    ax[2].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(FIG / "fig3_heterogeneity.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_heterogeneity.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print(f"figures -> {FIG}")
