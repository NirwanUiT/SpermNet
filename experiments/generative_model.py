"""A minimal mechanistic model of sperm motility-state dynamics.

The empirical analyses (markov_property_test.py, replicate_markov_extra.py,
dwell_censoring.py, mover_stayer_eb.py) MEASURE three robust, cross-cohort facts
about the per-frame motility-state sequence (Progressive / Non-progressive /
Immotile):

  (1) dwell times are heavy-tailed (log-normal beats exponential; pooled CV >> 1);
  (2) the sequence is non-Markovian: a 2nd-order model beats 1st-order out of
      sample by g2 = +0.057-0.059 nats/token on decorrelated 0.5 s blocks;
  (3) an empirical-Bayes decomposition attributes ~55% of that memory to genuine
      *within-cell* structure and ~45% to *between-cell* heterogeneity (movers vs
      stayers), and g2 EXCEEDS the heterogeneity-only upper bound.

Measuring is not explaining. This script asks how much of that phenomenology a
*minimal doubly-stochastic (slow-latent) mechanism* can generate, and -- crucially
-- whether static cell-to-cell heterogeneity ALONE is enough (which would overturn
fact 3), or whether within-cell modulation is required, and if so whether it is
even sufficient.

Mechanism ("vigour" latents on top of calibrated timescales):
  Each cell i is a 3-state continuous-time switch driven by two slow internal
  variables (independent stationary Ornstein-Uhlenbeck processes, mean 0, var 1):
      z^r_i(t) : correlation time tau_r  -- a slow "rate gear"
      z^v_i(t) : correlation time tau_v  -- a slow "vigour" biasing WHICH state
  plus a quenched per-cell rate offset eta_i ~ N(0, sigma_het^2). The dynamics:
      timing   : escape hazard = lambda_base[s] * exp(eta_i) * exp(kappa * z^r_i(t))
      identity : on a switch, target s' ~ Q_base[s,s'] * exp(beta * V[s'] * z^v_i(t))
    - lambda_base[s] : per-state baseline rate CALIBRATED to the empirical mean
                       dwell of state s (not fitted). Sets the timescales.
    - V = [+1, 0, -1] over [Progressive, Non-progressive, Immotile]: high vigour
                       biases toward Progressive, low toward Immotile.
    - Q_base         : empirical embedded (off-diagonal, row-normalised) topology.
  Because consecutive dwells share the slowly drifting z^v, the coarse 3-state
  readout carries state momentum -- the generative source of 2nd-order memory.
  kappa=beta=0 collapses to a memoryless mixture (only static rate heterogeneity).

Censoring is reproduced by cutting each simulated cell to a track length drawn
from the empirical track-length distribution, then scoring with the IDENTICAL
dwell-CV and held-out-order (g2) functions used on the real sequences.

What the analysis reports (all falsifiable, all honest):
  * NULL (kappa=beta=0): pure static rate heterogeneity. If this already reaches
    the real g2, the "memory" is an aggregation artifact (fact 3 wrong).
  * BEST joint fit of (kappa, beta): the closest (dwell-CV, g2) the slow-latent
    mechanism can reach, and what FRACTION of the empirical g2 it explains.
  * CEILING: g2 with identity coupling driven to saturation -- the most memory a
    slow-latent mechanism can possibly generate at matched dwell dispersion.
  * VALIDATION: do the fitted couplings reproduce the same fraction on the
    held-out 57-participant cohort (different detector/tracker)?

Interpretation: a slow-latent mechanism is necessary (NULL fails) and reproduces
heavy-tailed dwells, but if the CEILING falls well below the empirical g2, slow
internal modulation is INSUFFICIENT and the residual memory reflects faster
sequential structure (directional momentum / refractoriness) -- a concrete,
testable prediction rather than a fitted fudge factor.

Output: outputs/markov/generative_model.json

Usage:
    python -m experiments.generative_model [--fit-max-tracks 0] \
        [--val-max-tracks 400] [--n-cells 4000] [--tau-r 3.0] [--tau-v 30.0] \
        [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import compute_frame_states, STATES  # noqa: E402
from markov_property_test import dwell_times, cv_order  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
NS = len(STATES)
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"
BLOCK = 25  # 0.5 s at 50 fps -- decorrelation block for the g2 (order) test


# --------------------------------------------------------------------------- data
def load_real_sequences(track_dir: Path, max_tracks: int = 0, seed: int = 0):
    """Per-track frame-level state sequences (list[list[int]]).

    Reuses the orig20 plain-file filter (skip *_botsort/_bytetrack/_ocsort/_reid)
    so we never mix the same video's alternative-tracker exports into the cohort.
    """
    rng = np.random.default_rng(seed)
    plain_only = track_dir == ORIG
    seqs = []
    for tf in sorted(track_dir.glob("*_tracks.csv")):
        if plain_only:
            nm = tf.stem.replace("_tracks", "")
            if any(nm.endswith(s) for s in ("_botsort", "_bytetrack", "_ocsort", "_reid")):
                continue
        df = pd.read_csv(tf)
        if df.empty:
            continue
        tids = df["track_id"].unique()
        if max_tracks and len(tids) > max_tracks:
            tids = rng.choice(tids, size=max_tracks, replace=False)
            df = df[df["track_id"].isin(tids)]
        for tid, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            tr = tr.sort_values("frame")
            seqs.append([S2I[s] for s in compute_frame_states(tr)])
    return seqs


# --------------------------------------------------------------------- statistics
def block_downsample(seqs, block: int = BLOCK):
    """Collapse each frame-level sequence to one state per non-overlapping block
    (the block's modal state). This is the state-sequence analogue of the
    kinematic 0.5 s re-classification used for the decisive g2 test: it removes
    the trivial frame-to-frame persistence so the order comparison is fair.
    Applied IDENTICALLY to real and simulated sequences.
    """
    out = []
    for seq in seqs:
        a = np.asarray(seq, dtype=int)
        n = len(a) // block
        if n < 2:
            continue
        blk = a[: n * block].reshape(n, block)
        # modal state per block (ties -> lowest index; deterministic)
        modes = np.array([np.bincount(row, minlength=NS).argmax() for row in blk])
        out.append(modes.tolist())
    return out


def summary_stats(seqs, seed: int = 0):
    """The apples-to-apples summary vector computed on ANY set of state sequences.

    Returns per-state pooled dwell mean (frames) + CV, and the block-decorrelated
    held-out order log-likelihoods with g2 = ll[2]-ll[1].
    """
    dw = dwell_times(seqs)
    per_state = {}
    for i, name in enumerate(STATES):
        d = np.asarray(dw.get(i, []), dtype=float)
        if len(d) >= 30:
            per_state[name] = {
                "n_episodes": int(len(d)),
                "mean_frames": float(d.mean()),
                "cv": float(d.std() / d.mean()),
            }
        else:
            per_state[name] = {"n_episodes": int(len(d)), "mean_frames": None, "cv": None}
    blk = block_downsample(seqs)
    ll = cv_order(blk, orders=(0, 1, 2), folds=5) if len(blk) >= 10 else {0: None, 1: None, 2: None}
    g2 = (ll[2] - ll[1]) if (ll.get(2) is not None and ll.get(1) is not None) else None
    return {"per_state": per_state, "ll_order": ll, "g2": g2, "n_blocks": len(blk)}


def lognorm_vs_exp_aic(seqs):
    """Pooled interior-dwell log-normal vs exponential AIC gap on a set of
    sequences (positive => log-normal preferred). Interior runs only (drop the
    first/last run of each track) to avoid boundary censoring, matching the
    dwell-law test's spirit. Quick sanity check that the model makes heavy tails.
    """
    runs = []
    for seq in seqs:
        a = np.asarray(seq)
        if len(a) < 3:
            continue
        chg = np.flatnonzero(np.diff(a) != 0)
        # run lengths between change points; interior = drop first and last run
        bounds = np.concatenate(([-1], chg, [len(a) - 1]))
        rl = np.diff(bounds)
        if len(rl) > 2:
            runs.extend(rl[1:-1].tolist())
    d = np.asarray(runs, dtype=float)
    d = d[d > 0]
    if len(d) < 50:
        return None
    # exponential MLE (on continuous approx of the discrete dwell)
    lam = 1.0 / d.mean()
    ll_exp = np.sum(stats.expon.logpdf(d, scale=1.0 / lam))
    aic_exp = 2 * 1 - 2 * ll_exp
    # lognormal MLE
    s, loc, scale = stats.lognorm.fit(d, floc=0)
    ll_ln = np.sum(stats.lognorm.logpdf(d, s, loc=0, scale=scale))
    aic_ln = 2 * 2 - 2 * ll_ln
    return {"delta_aic_ln_minus_exp": float(aic_exp - aic_ln),
            "n_dwells": int(len(d))}


# --------------------------------------------------------------- calibration
def calibrate(seqs):
    """Derive baseline rates, embedded topology, stationary start, and the
    track-length distribution from the real sequences. These fix the model's
    timescales and geometry; only (sigma_het, kappa) are fitted afterwards.
    """
    dw = dwell_times(seqs)
    mean_frames = np.array([
        np.mean(dw[i]) if len(dw.get(i, [])) else 20.0 for i in range(NS)
    ])
    lambda_base = 1.0 / mean_frames  # per-frame escape hazard baseline

    # embedded transition topology: off-diagonal state-change counts, row-normalised
    Q = np.zeros((NS, NS))
    for seq in seqs:
        a = np.asarray(seq)
        chg = np.flatnonzero(np.diff(a) != 0)
        for k in chg:
            Q[a[k], a[k + 1]] += 1
    for r in range(NS):
        rs = Q[r].sum()
        if rs > 0:
            Q[r] /= rs
        else:
            Q[r] = np.ones(NS) / NS  # unused fallback
        Q[r, r] = 0.0  # embedded chain never self-loops (dwell handles persistence)

    # stationary start distribution = overall frame occupancy
    occ = np.zeros(NS)
    for seq in seqs:
        for s in seq:
            occ[s] += 1
    occ = occ / occ.sum()

    tracklens = np.array([len(s) for s in seqs], dtype=int)
    return {"lambda_base": lambda_base, "mean_frames": mean_frames,
            "Q": Q, "occ": occ, "tracklens": tracklens}


# vigour axis over STATES = [Progressive, Non-progressive, Immotile]:
# high internal vigour -> Progressive, low -> Immotile. Used to let the slow
# latent bias WHICH state a cell prefers, not just how fast it switches.
V_AXIS = np.array([1.0, 0.0, -1.0])


# ---------------------------------------------------------------- simulator
def _ou(L, phi, sig, rng):
    z = np.empty(L)
    z[0] = rng.normal(0.0, 1.0)
    eps = rng.normal(0.0, sig, size=L)
    for t in range(1, L):
        z[t] = phi * z[t - 1] + eps[t]
    return z


def simulate(cal, sigma_het, kappa, beta, tau_r_frames, tau_v_frames,
             n_cells, seed=0):
    """Generate n_cells frame-level state sequences under the two-latent model.

    Each cell draws a quenched rate offset eta ~ N(0, sigma_het^2), a track length
    from the empirical distribution, and two independent slow latents:
      z^r (rate gear, corr. time tau_r, coupling kappa) drives the escape hazard;
      z^v (vigour, corr. time tau_v, coupling beta) biases the target state via the
          ordered axis V. tau_v >> track length makes z^v a quasi-static per-cell
          preference (between-cell identity heterogeneity); finite tau_v adds
          genuine within-cell drift.
    kappa=beta=0 => memoryless mixture (only static rate heterogeneity via eta).
    """
    rng = np.random.default_rng(seed)
    lam = cal["lambda_base"]
    Q = cal["Q"]
    occ_c = np.cumsum(cal["occ"])
    tl = cal["tracklens"]
    phi_r = np.exp(-1.0 / max(tau_r_frames, 1e-6))
    sig_r = np.sqrt(max(1.0 - phi_r * phi_r, 1e-9))
    phi_v = np.exp(-1.0 / max(tau_v_frames, 1e-6))
    sig_v = np.sqrt(max(1.0 - phi_v * phi_v, 1e-9))

    lens = tl[rng.integers(0, len(tl), size=n_cells)]
    etas = rng.normal(0.0, sigma_het, size=n_cells)

    seqs = []
    for c in range(n_cells):
        L = int(lens[c])
        if L < config.MIN_TRACK_LENGTH:
            continue
        base_c = lam * np.exp(etas[c])                 # per-state hazard, this cell
        mod = np.exp(kappa * _ou(L, phi_r, sig_r, rng))  # timing modulation
        tilt = np.exp(beta * np.outer(_ou(L, phi_v, sig_v, rng), V_AXIS))  # L x NS
        u_switch = rng.random(L)
        u_jump = rng.random(L)
        s = int(min(np.searchsorted(occ_c, rng.random()), NS - 1))
        out = np.empty(L, dtype=np.int8)
        for t in range(L):
            out[t] = s
            haz = base_c[s] * mod[t]
            if u_switch[t] < -np.expm1(-haz):          # 1 - exp(-haz)
                p = Q[s] * tilt[t]                      # z^v-tilted target
                tot = p.sum()
                if tot <= 0:
                    s = int(min(np.searchsorted(occ_c, u_jump[t]), NS - 1))
                else:
                    s = int(min(np.searchsorted(np.cumsum(p) / tot, u_jump[t]), NS - 1))
        seqs.append(out.tolist())
    return seqs


# ------------------------------------------------------------------- fitting
def _stat_vector(stats_dict):
    """Extract the numeric target vector [cvP, cvNP, cvI, g2] (None-safe)."""
    ps = stats_dict["per_state"]
    cvs = [ps[STATES[i]]["cv"] for i in range(NS)]
    return cvs + [stats_dict["g2"]]


def cv_distance(sim_stats, real_stats):
    """Relative squared error on the 3 per-state dwell CVs ONLY. Used to find the
    parameter set that best reproduces the heavy-tailed dwell dispersion (fact 1);
    we then read off how much second-order memory (g2) that CV-matched set makes."""
    rv = _stat_vector(real_stats)
    sv = _stat_vector(sim_stats)
    d = 0.0
    for i in range(NS):
        if rv[i] and sv[i]:
            d += ((sv[i] - rv[i]) / rv[i]) ** 2
    return d


def scan(cal, real_stats, sigma_het, kappa_grid, beta_grid,
         tau_r_frames, tau_v_frames, n_cells, seed):
    """Grid over (kappa, beta) at fixed timescales. Returns the grid plus two
    honest summaries:
      cv_matched : the config that best reproduces the dwell CVs (fact 1), and the
                   g2 it happens to produce -- the memory achievable AT MATCHED
                   dwell dispersion.
      max_g2     : the config with the largest g2 anywhere in the grid, and how
                   badly it over/under-disperses the CVs to get there.
    """
    grid = []
    cv_matched = None
    max_g2 = None
    for kp in kappa_grid:
        for bt in beta_grid:
            sim = simulate(cal, sigma_het, kp, bt, tau_r_frames, tau_v_frames,
                           n_cells, seed=seed)
            ss = summary_stats(sim)
            cvd = cv_distance(ss, real_stats)
            row = {"kappa": float(kp), "beta": float(bt), "g2": ss["g2"],
                   "cv_distance": float(cvd),
                   "cv": [ss["per_state"][STATES[i]]["cv"] for i in range(NS)]}
            grid.append(row)
            if cv_matched is None or cvd < cv_matched["cv_distance"]:
                cv_matched = dict(row)
            if max_g2 is None or (ss["g2"] is not None and ss["g2"] > max_g2["g2"]):
                max_g2 = dict(row)
    return grid, cv_matched, max_g2


# ----------------------------------------------------------------------- main
def analyse(fit_dir, fit_name, val_dir, val_name, args):
    t0 = time.time()
    tau_r = args.tau_r * config.FPS
    print(f"[{fit_name}] loading real sequences (fit cohort) ...", flush=True)
    fit_seqs = load_real_sequences(fit_dir, max_tracks=args.fit_max_tracks, seed=args.seed)
    real_stats = summary_stats(fit_seqs)
    cal = calibrate(fit_seqs)
    real_g2 = real_stats["g2"]
    tl = cal["tracklens"]
    print(f"[{fit_name}] {len(fit_seqs)} tracks | mean dwell frames = "
          f"{np.round(cal['mean_frames'],1)} | median track {int(np.median(tl))}f "
          f"| real g2 = {real_g2:+.4f} | real cv = {np.round(_stat_vector(real_stats)[:3],2)}",
          flush=True)

    kappa_grid = np.round(np.arange(0.0, args.kappa_max + 1e-9, args.kappa_step), 3)
    beta_grid = np.round(np.arange(0.0, args.beta_max + 1e-9, args.beta_step), 3)
    frac = lambda g: (g / real_g2) if (g is not None and real_g2) else None

    # memoryless null: both latents off (only whatever sigma_het is set; default 0)
    null_stats = summary_stats(simulate(cal, args.sigma_het, 0.0, 0.0, tau_r,
                                        args.tau_v_between * config.FPS,
                                        args.n_cells, seed=args.seed))
    print(f"[{fit_name}] MEMORYLESS null (kappa=beta=0): g2={null_stats['g2']:+.4f} "
          f"({100*frac(null_stats['g2']):.0f}% of real)", flush=True)

    # two vigour regimes, distinguished only by the vigour correlation time:
    #   between-cell : tau_v >> track length  => quasi-static per-cell preference
    #   within-cell  : tau_v ~ track length   => vigour drifts during a cell's life
    regimes = {}
    for label, tau_v_s in (("between_cell", args.tau_v_between),
                           ("within_cell", args.tau_v_within)):
        print(f"[{fit_name}] scan {label} (tau_v={tau_v_s}s): "
              f"{len(kappa_grid)}x{len(beta_grid)} sims ...", flush=True)
        grid, cvm, mxg = scan(cal, real_stats, args.sigma_het, kappa_grid,
                              beta_grid, tau_r, tau_v_s * config.FPS,
                              args.n_cells, args.seed)
        regimes[label] = {
            "tau_v_seconds": tau_v_s,
            "cv_matched": cvm, "cv_matched_fraction": frac(cvm["g2"]),
            "max_g2": mxg, "max_g2_fraction": frac(mxg["g2"]),
            "grid": grid,
        }
        print(f"[{fit_name}]   CV-matched: kappa={cvm['kappa']} beta={cvm['beta']} "
              f"cv={np.round(cvm['cv'],2)} g2={cvm['g2']:+.4f} "
              f"({100*frac(cvm['g2']):.0f}% of real)", flush=True)
        print(f"[{fit_name}]   max-g2   : kappa={mxg['kappa']} beta={mxg['beta']} "
              f"cv={np.round(mxg['cv'],2)} g2={mxg['g2']:+.4f} "
              f"({100*frac(mxg['g2']):.0f}% of real)", flush=True)

    # pick the regime whose CV-matched config makes the most memory as the model's
    # honest best "at matched dwell dispersion", and sanity-check its dwell law
    best_label = max(regimes, key=lambda L: regimes[L]["cv_matched"]["g2"] or -1)
    best_cvm = regimes[best_label]["cv_matched"]
    best_tau_v = regimes[best_label]["tau_v_seconds"] * config.FPS
    ln_aic = lognorm_vs_exp_aic(
        simulate(cal, args.sigma_het, best_cvm["kappa"], best_cvm["beta"],
                 tau_r, best_tau_v, args.n_cells, seed=args.seed + 1))

    # the "mechanism" config: the strong-coupling point that generates the most
    # memory (necessarily over-dispersing dwells). This is the config whose
    # cross-cohort reproducibility tests whether the SLOW-LATENT MECHANISM itself
    # (not a Pareto endpoint) transfers to an independent cohort.
    mech_label = max(regimes, key=lambda L: regimes[L]["max_g2"]["g2"] or -1)
    mech = regimes[mech_label]["max_g2"]
    mech_tau_v = regimes[mech_label]["tau_v_seconds"] * config.FPS

    # decisive readouts
    null_frac = frac(null_stats["g2"])
    best_frac = frac(best_cvm["g2"])
    heterogeneity_suffices = (regimes["between_cell"]["cv_matched_fraction"] or 0) >= 0.8
    latent_necessary = null_frac is not None and null_frac < 0.2
    within_beats_between = (
        (regimes["within_cell"]["max_g2"]["g2"] or -1)
        > 1.15 * (regimes["between_cell"]["max_g2"]["g2"] or -1))
    slow_latent_sufficient = (regimes[mech_label]["max_g2_fraction"] or 0) >= 0.8

    # cross-cohort validation: replay BOTH the memoryless null AND the strong-
    # coupling mechanism config on the independent cohort (recalibrated).
    val = None
    if val_dir is not None:
        print(f"[{val_name}] loading validation cohort (cap {args.val_max_tracks}/video) ...",
              flush=True)
        val_seqs = load_real_sequences(val_dir, max_tracks=args.val_max_tracks, seed=args.seed)
        val_real = summary_stats(val_seqs)
        val_cal = calibrate(val_seqs)
        v_g2 = val_real["g2"]
        val_mech = summary_stats(simulate(val_cal, args.sigma_het, mech["kappa"],
                                         mech["beta"], tau_r, mech_tau_v,
                                         args.n_cells, seed=args.seed + 2))
        val_null = summary_stats(simulate(val_cal, args.sigma_het, 0.0, 0.0, tau_r,
                                         mech_tau_v, args.n_cells, seed=args.seed + 3))
        val = {
            "cohort": val_name, "n_tracks": len(val_seqs),
            "real_g2": v_g2, "real_cv": _stat_vector(val_real)[:3],
            "mechanism_config": {"regime": mech_label, "kappa": mech["kappa"],
                                 "beta": mech["beta"]},
            "mech_g2": val_mech["g2"], "mech_cv": _stat_vector(val_mech)[:3],
            "mech_fraction_of_real_g2": (val_mech["g2"] / v_g2) if v_g2 else None,
            "null_g2": val_null["g2"],
            "mean_frames": val_cal["mean_frames"].tolist(),
        }
        print(f"[{val_name}] real g2={v_g2:+.4f} | mechanism g2={val_mech['g2']:+.4f} "
              f"({100*val_mech['g2']/v_g2:.0f}% of real, cv={np.round(val['mech_cv'],2)}) "
              f"| null g2={val_null['g2']:+.4f}", flush=True)

    result = {
        "fit_cohort": fit_name,
        "n_fit_tracks": len(fit_seqs),
        "model": "two-latent (rate gear z^r + vigour z^v) doubly-stochastic switch",
        "params": {
            "sigma_het": args.sigma_het,
            "tau_r_seconds": args.tau_r,
            "tau_v_between_seconds": args.tau_v_between,
            "tau_v_within_seconds": args.tau_v_within,
            "V_axis": V_AXIS.tolist(),
            "block_frames": BLOCK,
            "median_track_frames": int(np.median(tl)),
        },
        "calibration": {
            "mean_dwell_frames": cal["mean_frames"].tolist(),
            "lambda_base_per_frame": cal["lambda_base"].tolist(),
            "embedded_Q": cal["Q"].tolist(),
            "occupancy": cal["occ"].tolist(),
            "n_tracklen_samples": int(len(cal["tracklens"])),
        },
        "real_stats": real_stats,
        "memoryless_null": {"g2": null_stats["g2"], "fraction_of_real_g2": null_frac,
                            "cv": _stat_vector(null_stats)[:3]},
        "regimes": regimes,
        "best_regime": best_label,
        "mechanism_config": {"regime": mech_label, "kappa": mech["kappa"],
                             "beta": mech["beta"], "g2": mech["g2"],
                             "fraction_of_real_g2": regimes[mech_label]["max_g2_fraction"],
                             "cv": mech["cv"]},
        "decisive": {
            "real_g2": real_g2,
            "memoryless_null_fraction": null_frac,
            "between_cell_max_g2_fraction": regimes["between_cell"]["max_g2_fraction"],
            "within_cell_max_g2_fraction": regimes["within_cell"]["max_g2_fraction"],
            "best_cv_matched_fraction": best_frac,
            "mechanism_max_g2_fraction": regimes[mech_label]["max_g2_fraction"],
            "heterogeneity_alone_suffices": bool(heterogeneity_suffices),
            "slow_latent_necessary": bool(latent_necessary),
            "within_cell_beats_between_cell": bool(within_beats_between),
            "slow_latent_sufficient": bool(slow_latent_sufficient),
            "pareto_tension_cv_vs_memory": bool(
                (best_frac is not None and best_frac < 0.2)
                and (regimes[mech_label]["max_g2_fraction"] or 0) > 0.4),
        },
        "lognormal_vs_exp_aic_on_sim": ln_aic,
        "validation": val,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-max-tracks", type=int, default=0,
                    help="cap tracks/video on FIT cohort (0 = all; orig20 is small)")
    ap.add_argument("--val-max-tracks", type=int, default=400,
                    help="cap tracks/video on VALIDATION cohort (extra57 is huge)")
    ap.add_argument("--n-cells", type=int, default=4000)
    ap.add_argument("--sigma-het", type=float, default=0.0,
                    help="quenched per-cell log-rate SD (static rate heterogeneity)")
    ap.add_argument("--tau-r", type=float, default=3.0,
                    help="rate-gear OU correlation time in seconds")
    ap.add_argument("--tau-v-between", type=float, default=30.0,
                    help="vigour corr. time (s) for the between-cell/quasi-static regime")
    ap.add_argument("--tau-v-within", type=float, default=1.5,
                    help="vigour corr. time (s) for the within-cell-drift regime")
    ap.add_argument("--kappa-max", type=float, default=3.0)
    ap.add_argument("--kappa-step", type=float, default=0.5)
    ap.add_argument("--beta-max", type=float, default=6.0)
    ap.add_argument("--beta-step", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-val", action="store_true", help="skip cross-cohort validation")
    ap.add_argument("--fit-dir", type=str, default=None,
                    help="alternative fit track directory (e.g. outputs/tracks_gt); "
                         "output goes to generative_model_<fit-name>.json")
    ap.add_argument("--fit-name", type=str, default=None,
                    help="cohort label for --fit-dir (default: dir name)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    val_dir = None if args.no_val else EXTRA
    if args.fit_dir:
        fit_dir = Path(args.fit_dir)
        fit_name = args.fit_name or fit_dir.name
        out = OUTDIR / f"generative_model_{fit_name}.json"
    else:
        fit_dir, fit_name = ORIG, "orig20"
        out = OUTDIR / "generative_model.json"
    res = analyse(fit_dir, fit_name, val_dir, "extra57", args)

    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
