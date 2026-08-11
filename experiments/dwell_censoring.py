"""Censoring-aware robustness of the dwell law and the within-cell/between-cell split.

WHY THIS EXISTS
---------------
The dwell-law result (dwell_physics.py, log-normal >> exponential) and the memory
decomposition (memory_decomposition.py: within-cell CV ~ 1 vs pooled CV ~ 2, i.e.
"single cells are near-memoryless, the memory lives in cell-to-cell heterogeneity")
both treat EVERY dwell episode as fully observed. But a finite track truncates two
episodes per track:
  * the FIRST run started before the track began   (left boundary),
  * the LAST  run may continue after the track ends (right boundary).
Both are only LOWER BOUNDS on the true dwell -> naively counting them as complete
dwells DEFLATES observed dwells and DEFLATES the within-cell dispersion. Worse, the
within-cell CV in memory_decomposition.py is computed only on cells with >= 5 same-
state episodes, which selects fast-switching (short-dwell) cells -> a second deflation.
A referee will (correctly) say the "single cells are near-memoryless" reading could be
a censoring + selection artefact.

WHAT THIS DOES (two censoring-aware analyses, self-contained; numpy/scipy only)
-------------------------------------------------------------------------------
(1) CENSORED PARAMETRIC LAW. Re-fit exp / gamma / weibull / lognormal per state and
    cohort by CENSORED maximum likelihood: interior episodes contribute log f(x),
    boundary episodes contribute log S(x) (survival, x is a lower bound). Redo AIC.
    Question: does log-normal still beat exponential once censoring is respected?

(2) CENSORING-AND-SELECTION-AWARE VARIANCE SPLIT. Fit a Gaussian frailty (one-way
    random-effects Tobit) model to log-dwell, per state:
            log d_ij = mu_state + b_i + e_ij ,  b_i ~ N(0, tau^2),  e_ij ~ N(0, sigma^2)
    where boundary episodes are right-censored (log d_ij >= log c_ij). Fit by EM with
    Gauss-Hermite quadrature over the cell random effect b_i and truncated-normal
    E-steps for the censored episodes. This uses ALL tracks (no >=5 selection) and
    respects truncation, so:
        within-cell log-sd  = sigma      -> within-cell CV = sqrt(exp(sigma^2) - 1)
        between-cell log-sd = tau        -> ICC = tau^2 / (tau^2 + sigma^2)
        pooled CV           = sqrt(exp(sigma^2 + tau^2) - 1)
    Question: after correcting censoring AND selection, are single cells still much
    less dispersed than the pool (supporting quenched heterogeneity), or does the
    within-cell dispersion rise (forcing us to qualify the near-memoryless reading)?

Both outcomes are publishable if reported honestly. Output:
    outputs/markov/dwell_censoring.json

Usage: python -m experiments.dwell_censoring [--max-tracks 3000] [--gh 24]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import markov_analysis as ma  # noqa: E402
from markov_analysis import STATES  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"
LOG2PI = float(np.log(2.0 * np.pi))
# log-scale SD of the natural log of an Exponential r.v. = sqrt(psi'(1)) = pi/sqrt(6).
# This is the memoryless benchmark: a cell whose own dwells are exponential has within-
# cell log-dwell SD exactly EXP_LOGSD, regardless of its rate. within-cell SD above this
# => the cell's OWN dwells are over-dispersed relative to memoryless (single-cell memory);
# at this value => the cell is memoryless and all extra pooled spread is heterogeneity.
EXP_LOGSD = float(np.pi / np.sqrt(6.0))  # ~= 1.2825


# --------------------------------------------------------------------------- data
def iter_tracks(track_dir: Path, max_tracks: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    plain_only = track_dir == ORIG
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
        vid = tf.stem.replace("_tracks", "")
        for tid, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            yield f"{vid}:{tid}", tr.sort_values("frame")


def episodes_with_censor(seq):
    """Return list of (state, run_len). A run is censored (boundary-truncated) if it is
    the first or last run of the track. Interior runs are complete dwells."""
    runs = []
    run = 1
    for k in range(1, len(seq)):
        if seq[k] == seq[k - 1]:
            run += 1
        else:
            runs.append((seq[k - 1], run)); run = 1
    runs.append((seq[-1], run))
    out = []
    for j, (st, r) in enumerate(runs):
        boundary = (j == 0) or (j == len(runs) - 1)   # first OR last run of the track
        out.append((st, r, boundary))
    return out


def collect(track_dir: Path, max_tracks: int, seed: int):
    """Gather per-state complete/censored durations and per-episode arrays for EM."""
    dur = {i: {"complete": [], "censored": []} for i in range(len(STATES))}
    # episode-level arrays for frailty EM
    ep_state, ep_logd, ep_cens, ep_cell = [], [], [], []
    cell_ids = {}
    n_tracks = 0
    n_single_episode = 0
    for tkey, tr in iter_tracks(track_dir, max_tracks, seed):
        seq = [S2I[s] for s in ma.compute_frame_states(tr)]
        eps = episodes_with_censor(seq)
        n_tracks += 1
        if len(eps) == 1:
            n_single_episode += 1
        cid = cell_ids.setdefault(tkey, len(cell_ids))
        for st, run, boundary in eps:
            d = run / config.FPS
            dur[st]["censored" if boundary else "complete"].append(d)
            ep_state.append(st)
            ep_logd.append(np.log(d))
            ep_cens.append(1 if boundary else 0)
            ep_cell.append(cid)
    arrs = dict(
        state=np.asarray(ep_state, int),
        logd=np.asarray(ep_logd, float),
        cens=np.asarray(ep_cens, bool),
        cell=np.asarray(ep_cell, int),
        n_cells=len(cell_ids),
        n_tracks=n_tracks,
        n_single_episode=n_single_episode,
    )
    return dur, arrs


# --------------------------------------------------------- (1) censored parametric law
def _nll_factory(dist, complete, censored):
    c = np.asarray(complete, float)
    z = np.asarray(censored, float)

    def nll(theta):
        p = np.exp(theta)  # positive params
        try:
            if dist == "expon":
                ll = stats.expon.logpdf(c, scale=p[0]).sum()
                if z.size:
                    ll += stats.expon.logsf(z, scale=p[0]).sum()
            elif dist == "gamma":
                ll = stats.gamma.logpdf(c, p[0], scale=p[1]).sum()
                if z.size:
                    ll += stats.gamma.logsf(z, p[0], scale=p[1]).sum()
            elif dist == "weibull":
                ll = stats.weibull_min.logpdf(c, p[0], scale=p[1]).sum()
                if z.size:
                    ll += stats.weibull_min.logsf(z, p[0], scale=p[1]).sum()
            elif dist == "lognorm":
                ll = stats.lognorm.logpdf(c, p[0], scale=p[1]).sum()
                if z.size:
                    ll += stats.lognorm.logsf(z, p[0], scale=p[1]).sum()
            else:
                raise ValueError(dist)
        except Exception:
            return 1e18
        if not np.isfinite(ll):
            return 1e18
        return -ll

    return nll


def fit_censored_laws(complete, censored):
    c = np.asarray(complete, float)
    n = c.size + len(censored)
    laws = {}
    # sensible init from complete-only fits
    m = c.mean() if c.size else 1.0
    s = c.std() if c.size else 1.0
    inits = {
        "expon": ([np.log(max(m, 1e-3))], 1),
        "gamma": ([np.log(max((m / max(s, 1e-3)) ** 2, 0.1)), np.log(max(s ** 2 / max(m, 1e-3), 1e-3))], 2),
        "weibull": ([np.log(1.2), np.log(max(m, 1e-3))], 2),
        "lognorm": ([np.log(max(np.std(np.log(c)) if c.size else 0.6, 0.1)),
                     np.log(max(np.exp(np.mean(np.log(c))) if c.size else m, 1e-3))], 2),
    }
    for dist, (x0, k) in inits.items():
        nll = _nll_factory(dist, c, censored)
        best = None
        for method in ("Nelder-Mead", "Powell"):
            try:
                r = optimize.minimize(nll, np.asarray(x0, float), method=method,
                                      options={"maxiter": 5000})
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
        if best is None:
            laws[dist] = {"aic": np.inf, "ll": -np.inf, "k": k}
            continue
        ll = -best.fun
        aic = 2 * k - 2 * ll
        laws[dist] = {"aic": float(aic), "ll": float(ll), "k": k,
                      "params": [float(v) for v in np.exp(best.x)]}
    best_name = min(laws, key=lambda d: laws[d]["aic"])
    d_exp = laws["expon"]["aic"] - laws[best_name]["aic"]
    return {"laws": laws, "best": best_name, "delta_aic_vs_exponential": float(d_exp),
            "n_complete": int(c.size), "n_censored": int(len(censored)), "n": int(n)}


# ------------------------------------------------- (2) frailty EM (random-effects Tobit)
def frailty_em(logd, cens, cell, gh_nodes=24, max_iter=250, tol=1e-6):
    """One-way random-effects Tobit on log-dwell (single state).
       log d_ij = mu + b_i + e_ij ; boundary episodes right-censored at logd (a lower bound).
       Returns mu, sigma (within-cell log-sd), tau (between-cell log-sd)."""
    y = np.asarray(logd, float)
    is_c = np.asarray(cens, bool)
    # reindex cells to 0..M-1 densely
    uniq, cell_idx = np.unique(cell, return_inverse=True)
    M = uniq.size
    N = y.size
    x_gh, w_gh = hermegauss(gh_nodes)          # nodes/weights for int f(x) e^{-x^2/2} dx
    logw = np.log(w_gh) - 0.5 * LOG2PI          # prior weight for b = tau*x

    # init from marginal moments
    mu = float(np.mean(y))
    var = float(np.var(y)) + 1e-3
    sigma = np.sqrt(0.7 * var)
    tau = np.sqrt(0.3 * var)

    prev = None
    for _ in range(max_iter):
        b = tau * x_gh                          # (K,)
        K = b.size
        m = mu + b[:, None]                     # (K,1) broadcasts over episodes
        ll = np.empty((K, N))
        comp = ~is_c
        # complete: normal logpdf; censored: normal logsf (lower bound = y)
        ll[:, comp] = (-0.5 * LOG2PI - np.log(sigma)
                       - 0.5 * ((y[None, comp] - m) / sigma) ** 2)
        if is_c.any():
            ll[:, is_c] = stats.norm.logsf(y[None, is_c], loc=m, scale=sigma)
        # aggregate episode ll to per-cell ll under each node
        cell_ll = np.zeros((K, M))
        for k in range(K):
            np.add.at(cell_ll[k], cell_idx, ll[k])
        # posterior over nodes per cell
        logpost = logw[:, None] + cell_ll       # (K,M)
        logZ = _logsumexp(logpost, axis=0)      # (M,)
        post = np.exp(logpost - logZ[None, :])   # (K,M) sums to 1 over K

        # map posterior to episodes
        post_ep = post[:, cell_idx]             # (K,N)

        # E[b] per episode, E[b^2] per cell
        Eb_ep = np.einsum("k,kn->n", b, post_ep)          # (N,)
        Eb2_cell = np.einsum("k,km->m", b ** 2, post)     # (M,)

        # E[y] and E[(y-mu-b)^2] per episode (posterior-averaged over nodes)
        Ey_ep = np.empty(N)
        Eres2_ep = np.empty(N)
        # complete episodes
        if comp.any():
            Ey_ep[comp] = y[comp]
            # (y-mu-b_k)^2 averaged over nodes
            diff = (y[None, comp] - (mu + b[:, None]))     # (K, n_comp)
            Eres2_ep[comp] = np.einsum("kn,kn->n", post_ep[:, comp], diff ** 2)
        # censored episodes: truncated normal (lower truncation at c=y)
        if is_c.any():
            cc = y[None, is_c]                              # (1, n_c) lower bound
            mk = mu + b[:, None]                            # (K,1)
            alpha = (cc - mk) / sigma                        # (K, n_c)
            # inverse Mills ratio lambda = phi(a)/S(a); stable via logs
            logphi = -0.5 * LOG2PI - 0.5 * alpha ** 2
            logsf = stats.norm.logsf(alpha)
            lam = np.exp(logphi - logsf)
            Ey_k = mk + sigma * lam                          # E[y | b_k, y>=c]
            Eres2_k = sigma ** 2 * (1.0 + alpha * lam)       # E[(y-mu-b_k)^2 | ...]
            pc = post_ep[:, is_c]
            Ey_ep[is_c] = np.einsum("kn,kn->n", pc, Ey_k)
            Eres2_ep[is_c] = np.einsum("kn,kn->n", pc, Eres2_k)

        # M-step
        mu_new = float(np.mean(Ey_ep - Eb_ep))
        sigma_new = float(np.sqrt(max(np.mean(Eres2_ep), 1e-9)))
        tau_new = float(np.sqrt(max(np.mean(Eb2_cell), 1e-9)))
        # log-dwell SD is physically bounded (dwells span ~1/FPS..~10 s -> log-range ~6,
        # SD <~ 3). Clip to keep the censored-EM from running away on flat likelihoods.
        sigma_new = float(np.clip(sigma_new, 0.05, 4.0))
        tau_new = float(np.clip(tau_new, 1e-3, 4.0))

        cur = np.array([mu_new, sigma_new, tau_new])
        mu, sigma, tau = mu_new, sigma_new, tau_new
        if prev is not None and np.max(np.abs(cur - prev)) < tol:
            break
        prev = cur

    # final posterior cell effects E[b_i] keyed by ORIGINAL cell id (for the serial test)
    b = tau * x_gh
    m = mu + b[:, None]
    ll = np.empty((b.size, N))
    comp = ~is_c
    ll[:, comp] = (-0.5 * LOG2PI - np.log(sigma) - 0.5 * ((y[None, comp] - m) / sigma) ** 2)
    if is_c.any():
        ll[:, is_c] = stats.norm.logsf(y[None, is_c], loc=m, scale=sigma)
    cell_ll = np.zeros((b.size, M))
    for k in range(b.size):
        np.add.at(cell_ll[k], cell_idx, ll[k])
    logpost = logw[:, None] + cell_ll
    post = np.exp(logpost - _logsumexp(logpost, axis=0)[None, :])
    Eb_cell = np.einsum("k,km->m", b, post)                 # (M,)
    eb_by_cell = {int(uniq[j]): float(Eb_cell[j]) for j in range(M)}

    # effective within-cell sample: cells with >=2 complete (interior) episodes
    comp_counts = np.zeros(M, int)
    np.add.at(comp_counts, cell_idx[comp], 1)
    n_cells_within = int((comp_counts >= 2).sum())

    icc = tau ** 2 / (tau ** 2 + sigma ** 2)
    within_cv = float(np.sqrt(np.expm1(min(sigma ** 2, 20.0))))
    pooled_cv = float(np.sqrt(np.expm1(min(sigma ** 2 + tau ** 2, 20.0))))
    # decisive test: is the cell's OWN dwell over-dispersed vs a memoryless (exponential) cell?
    ratio = float(sigma / EXP_LOGSD)
    if ratio >= 1.15:
        verdict = "single-cell memory (over-dispersed vs exponential)"
    elif ratio <= 0.87:
        verdict = "sub-exponential (more regular than memoryless)"
    else:
        verdict = "consistent with memoryless single cells"
    return {"mu": float(mu), "sigma_within": float(sigma), "tau_between": float(tau),
            "icc": float(icc), "within_cell_cv": within_cv, "pooled_cv": pooled_cv,
            "exp_logsd_benchmark": EXP_LOGSD,
            "within_over_exponential": ratio, "within_verdict": verdict,
            "n_cells": int(M), "n_cells_within": n_cells_within, "n_episodes": int(N),
            "frac_censored": float(is_c.mean()),
            "_eb_by_cell": eb_by_cell}


def _logsumexp(a, axis=0):
    amax = np.max(a, axis=axis, keepdims=True)
    return (amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))).squeeze(axis)


# ------------------------------------------------------------- naive (biased) reference
def naive_within_cv(logd, cell, state, min_ep=5):
    """Reproduce the memory_decomposition.py within-cell CV (biased) for direct
       comparison: per-cell CV of dwell (natural scale) on cells with >= min_ep episodes."""
    d = np.exp(logd)
    df = pd.DataFrame({"cell": cell, "d": d})
    cvs = []
    for _, g in df.groupby("cell"):
        if len(g) >= min_ep and g["d"].mean() > 0:
            cvs.append(g["d"].std() / g["d"].mean())
    cvs = np.array(cvs, float)
    pooled_cv = float(d.std() / d.mean())
    return {"naive_within_median_cv": float(np.median(cvs)) if cvs.size else None,
            "naive_within_n_cells": int(cvs.size),
            "naive_pooled_cv": pooled_cv}


def _within_group_lag1(vals, grp, i0, i1, n_perm, seed):
    """Observed lag-1 Spearman of (vals[i0], vals[i1]) vs a within-group permutation null
       (shuffle values within each cell, preserving the cell's value multiset -> preserves
       all between-cell heterogeneity, destroys temporal order). Returns obs, null mean/sd,
       z and two-sided empirical p."""
    obs = float(stats.spearmanr(vals[i0], vals[i1]).correlation)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        key = rng.random(vals.size)
        p = np.lexsort((key, grp))          # groups stay contiguous; random order within
        vp = vals[p]
        null[b] = stats.spearmanr(vp[i0], vp[i1]).correlation
    nm, ns = float(null.mean()), float(null.std() + 1e-12)
    z = (obs - nm) / ns
    p_emp = float((np.sum(np.abs(null - nm) >= abs(obs - nm)) + 1) / (n_perm + 1))
    return {"lag1_spearman": obs, "null_mean": nm, "null_sd": ns,
            "z": float(z), "p_perm": p_emp, "n_pairs": int(i0.size)}


def _pair_index(grp):
    """Adjacent same-group index pairs on a (cell)-contiguous flat array."""
    same = grp[:-1] == grp[1:]
    i0 = np.flatnonzero(same)
    return i0, i0 + 1


def serial_memory(arrs, n_perm=200, seed=0):
    """Censoring-clean within-cell temporal memory. State-residualise log-dwell (state
       means from COMPLETE episodes), then test lag-1 dependence of consecutive dwells
       within a cell against a within-cell permutation null (no demeaning -> no demeaning
       artefact; the null carries the same heterogeneity as the data). Two variants:
         clean  : COMPLETE (interior) episodes only -> censoring cannot bias it;
         naive  : all episodes incl. the truncated first/last (boundary) episodes -> what
                  an uncorrected analysis sees.
       A clean lag-1 that exceeds its permutation null is genuine single-cell memory."""
    state, logd, cens, cell = arrs["state"], arrs["logd"], arrs["cens"], arrs["cell"]
    comp = ~cens
    smean = {}
    for i in range(len(STATES)):
        m = comp & (state == i)
        smean[i] = float(logd[m].mean()) if m.any() else 0.0
    res = logd - np.array([smean[int(s)] for s in state])
    order = np.arange(state.size)

    out = {}
    # clean: complete episodes only
    df = pd.DataFrame({"cell": cell, "order": order, "res": res})[comp]
    df = df.sort_values(["cell", "order"], kind="stable")
    grp = df["cell"].to_numpy()
    vals = df["res"].to_numpy()
    i0, i1 = _pair_index(grp)
    out["clean"] = _within_group_lag1(vals, grp, i0, i1, n_perm, seed) if i0.size >= 20 else None
    # naive: all episodes (incl. truncated boundary episodes)
    dfa = pd.DataFrame({"cell": cell, "order": order, "res": res}).sort_values(
        ["cell", "order"], kind="stable")
    grpa = dfa["cell"].to_numpy()
    valsa = dfa["res"].to_numpy()
    j0, j1 = _pair_index(grpa)
    out["naive_incl_censored"] = (_within_group_lag1(valsa, grpa, j0, j1, n_perm, seed)
                                  if j0.size >= 20 else None)
    return out


# ------------------------------------------------------------------------------- driver
def analyse(track_dir: Path, max_tracks: int, seed: int, gh: int):
    dur, arrs = collect(track_dir, max_tracks, seed)
    res = {"n_tracks": arrs["n_tracks"], "n_cells": arrs["n_cells"],
           "n_single_episode_tracks": arrs["n_single_episode"],
           "frac_single_episode": float(arrs["n_single_episode"] / max(arrs["n_tracks"], 1)),
           "law": {}, "frailty": {}}
    for i, s in enumerate(STATES):
        # (1) censored parametric law
        res["law"][s] = fit_censored_laws(dur[i]["complete"], dur[i]["censored"])
        # (2) frailty EM + naive reference, per state
        mask = arrs["state"] == i
        if mask.sum() >= 20 and np.unique(arrs["cell"][mask]).size >= 5:
            fr = frailty_em(arrs["logd"][mask], arrs["cens"][mask], arrs["cell"][mask],
                            gh_nodes=gh)
            fr.pop("_eb_by_cell", None)                # not needed downstream; keep JSON small
            fr.update(naive_within_cv(arrs["logd"][mask], arrs["cell"][mask], s))
            res["frailty"][s] = fr
        else:
            res["frailty"][s] = None
    # (3) censoring-clean, model-free within-cell serial memory (pooled across states)
    res["serial_memory"] = serial_memory(arrs, seed=seed)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gh", type=int, default=24)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for cname, cdir in (("orig20", ORIG), ("extra57", EXTRA)):
        if not cdir.exists():
            print(f"[skip] {cname}: {cdir} missing", flush=True)
            continue
        print(f"=== {cname} ===", flush=True)
        r = analyse(cdir, args.max_tracks, args.seed, args.gh)
        out[cname] = r
        print(f"  tracks={r['n_tracks']}  single-episode(fully censored)="
              f"{r['frac_single_episode']*100:.1f}%", flush=True)
        for s in STATES:
            law = r["law"][s]
            print(f"  [{s:16s}] censored law: best={law['best']:8s} "
                  f"dAIC(exp)={law['delta_aic_vs_exponential']:10.1f} "
                  f"(nc={law['n_complete']}, ncen={law['n_censored']})", flush=True)
            fr = r["frailty"][s]
            if fr:
                print(f"                   frailty: sigma_w={fr['sigma_within']:.2f} "
                      f"(exp bench {fr['exp_logsd_benchmark']:.2f}, ratio "
                      f"{fr['within_over_exponential']:.2f}) tau={fr['tau_between']:.2f} "
                      f"ICC={fr['icc']:.2f}  n_cells_within={fr['n_cells_within']}",
                      flush=True)
                print(f"                            -> {fr['within_verdict']}  | naive "
                      f"within-med-CV={fr['naive_within_median_cv']}", flush=True)
        sm = r.get("serial_memory")
        if sm and sm.get("clean"):
            c = sm["clean"]
            print(f"  within-cell serial memory (censoring-clean): lag1 rho="
                  f"{c['lag1_spearman']:+.4f} vs null {c['null_mean']:+.4f}+/-{c['null_sd']:.4f} "
                  f"z={c['z']:+.1f} p={c['p_perm']:.2g}  n_pairs={c['n_pairs']}", flush=True)
            nv = sm.get("naive_incl_censored")
            if nv:
                print(f"     naive control incl. censored: lag1 rho="
                      f"{nv['lag1_spearman']:+.4f} vs null {nv['null_mean']:+.4f} "
                      f"z={nv['z']:+.1f}", flush=True)
        print(flush=True)

    json.dump(out, open(OUTDIR / "dwell_censoring.json", "w"), indent=2, default=float)
    print(f"saved -> {OUTDIR/'dwell_censoring.json'}", flush=True)


if __name__ == "__main__":
    main()
