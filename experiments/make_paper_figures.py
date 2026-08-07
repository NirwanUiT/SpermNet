"""Assemble the paper figures for the non-Markovian sperm-motility study.

Figure 1  dwell-time law is log-normal (not exponential)      -> fig1_dwell_law.png
Figure 2  decomposing the memory: single-cell vs mover-stayer -> fig2_decomposition.png
Figure 3  heterogeneity is a reliable trait but not clinically -> fig3_heterogeneity.png
          incremental over CASA composition

Figure 1 recomputes a modest dwell-time sample; Figures 2-3 are built from the
summary JSONs already on disk (mover_stayer_eb/null, memory_decomposition,
per_cell_kinetics, stayer_dfi).

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
    dwell = dwell_episodes(EXTRA, max_tracks=500, seed=0)
    rob = load("dwell_physics_robust")["window_invariance"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # (A) empirical survival on log-log + fitted exponential vs log-normal
    for i, s in enumerate(STATES):
        d = np.sort(dwell[i])
        d = d[d > 0]
        surv = 1.0 - np.arange(1, len(d) + 1) / len(d)
        ax[0].step(d, surv, where="post", color=COL[s], lw=1.8, label=s)
        # fitted exponential (Markov/memoryless) -- dashed
        mu = d.mean()
        ax[0].plot(d, np.exp(-d / mu), color=COL[s], ls=":", lw=1.0, alpha=0.8)
        # fitted log-normal -- thin solid
        sh, loc, sc = stats.lognorm.fit(d, floc=0)
        ax[0].plot(d, 1 - stats.lognorm.cdf(d, sh, loc, sc),
                   color=COL[s], ls="--", lw=1.0, alpha=0.9)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(d.min(), None); ax[0].set_ylim(1e-4, 1)
    ax[0].set_xlabel("dwell time (s)"); ax[0].set_ylabel("survival  P(T > t)")
    ax[0].set_title("A  Dwell-time survival (57-participant cohort)", loc="left",
                    fontsize=11, weight="bold")
    ax[0].plot([], [], color="0.4", ls=":", label="exponential fit")
    ax[0].plot([], [], color="0.4", ls="--", label="log-normal fit")
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")

    # (B) exponential is rejected by a huge margin at every window size
    windows = ["13", "25", "51"]
    x = np.arange(len(windows))
    w = 0.26
    for j, s in enumerate(STATES):
        vals = [rob[wd][s]["exp_dAIC"] for wd in windows]
        ax[1].bar(x + (j - 1) * w, vals, w, color=COL[s], label=s)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{wd} frames" for wd in windows])
    ax[1].set_xlabel("classifier window")
    ax[1].set_ylabel(r"$\Delta$AIC  (exponential $-$ log-normal)")
    ax[1].set_title("B  Exponential rejected at every window", loc="left",
                    fontsize=11, weight="bold")
    ax[1].axhline(10, color="0.5", ls=":", lw=0.8)
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].text(0.02, 0.02, "log-normal is the best law in all states, both cohorts,\n"
               "and every window (higher bar = exponential more strongly rejected)",
               transform=ax[1].transAxes, fontsize=7.5, color="0.35")

    fig.tight_layout()
    fig.savefig(FIG / "fig1_dwell_law.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_dwell_law.png")


# ---------------------------------------------------------------- Figure 2
def figure2():
    eb = load("mover_stayer_eb")
    dec = load("memory_decomposition")
    cohorts = ["orig20", "extra57"]
    labels = {"orig20": "VISEM-Tracking (20)", "extra57": "independent (57)"}

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # (A) held-out 2nd-over-1st gain: real vs the three nulls
    keys = ["real", "eb_fair", "het_overfit", "hom"]
    names = ["observed", "EB-fair\nheterogeneity", "over-fit\nheterogeneity",
             "homogeneous\nmemoryless"]
    cols = ["#2166ac", "#67a9cf", "#b2abd2", "#bbbbbb"]
    x = np.arange(len(keys)); w = 0.38
    for c, off in zip(cohorts, (-w / 2, w / 2)):
        g = eb[c]["g2"]
        ax[0].bar(x + off, [g[k] for k in keys], w, label=labels[c],
                  color=[cols[i] for i in range(len(keys))],
                  edgecolor="k" if c == "extra57" else "none", linewidth=0.6)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8.5)
    ax[0].set_ylabel("2nd-order gain (log-lik / token)")
    ax[0].set_title("A  Only real data shows full memory", loc="left",
                    fontsize=11, weight="bold")
    ax[0].text(0.02, 0.95, "solid edge = 57-cohort", transform=ax[0].transAxes,
               fontsize=7.5, color="0.35", va="top")

    # (B) fair split of the memory into two sources
    for k, c in enumerate(cohorts):
        agg = eb[c]["aggregation_share_eb"]
        gen = eb[c]["genuine_memory_share_eb"]
        ax[1].bar(k, gen * 100, color="#2166ac", label="single-cell memory" if k == 0 else None)
        ax[1].bar(k, agg * 100, bottom=gen * 100, color="#67a9cf",
                  label="mover-stayer heterogeneity" if k == 0 else None)
        ax[1].text(k, gen * 50, f"{gen:.0%}", ha="center", color="w", fontsize=10, weight="bold")
        ax[1].text(k, gen * 100 + agg * 50, f"{agg:.0%}", ha="center", color="w",
                   fontsize=10, weight="bold")
    ax[1].set_xticks(range(len(cohorts)))
    ax[1].set_xticklabels([labels[c] for c in cohorts], fontsize=9)
    ax[1].set_ylabel("share of the non-Markovian signal (%)")
    ax[1].set_ylim(0, 100)
    ax[1].set_title("B  Two comparable sources of memory", loc="left",
                    fontsize=11, weight="bold")
    ax[1].legend(fontsize=8, frameon=False, loc="upper center")

    # (C) single cells near-memoryless (CV~1); population dispersed (CV~2)
    x = np.arange(len(STATES)); w = 0.38
    c = "extra57"
    pooled = [dec[c]["pooled_cv"][s] for s in STATES]
    within = [dec[c]["within_track_cv"][s]["median_cv"] for s in STATES]
    ax[2].bar(x - w / 2, within, w, color="#2166ac", label="within single cell (median)")
    ax[2].bar(x + w / 2, pooled, w, color="#67a9cf", label="pooled population")
    ax[2].axhline(1.0, color="k", ls=":", lw=1, label="memoryless (CV=1)")
    ax[2].set_xticks(x); ax[2].set_xticklabels([s[:4] for s in STATES])
    ax[2].set_ylabel("dwell-time CV")
    ax[2].set_title("C  Cells ~memoryless, population dispersed", loc="left",
                    fontsize=11, weight="bold")
    ax[2].legend(fontsize=8, frameon=False)

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
