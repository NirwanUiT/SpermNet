"""T0.3 + T0.4 -- Dwell-law resolution audit and immotile censoring audit (GT).

T0.3 (resolution): mean motile dwells (0.85s / 0.54s) sit near the 0.5s
classifier window, so the law competition may have < 1 decade of usable range.
We report per state: minimum dwell, fraction of episodes below 1x/2x/3x the
window; re-fit the 4-law competition restricted to dwells >= 3x window; and
re-run the whole competition at windows BELOW the mean dwell (5 and 9 frames,
plus 13/25 for continuity), reusing the identical classifier code path.

T0.4 (immotile censoring): 907 immotile episodes, mean 494 frames vs median
track 445 frames -- most may simply be whole tracks. We report per state the
fraction of episodes touching one/both track boundaries; compare the immotile
dwell distribution against the track-length distribution (KS); re-fit all GT
dwell laws by CENSORED maximum likelihood (interior episodes = density,
boundary episodes = survival), exactly as dwell_censoring.py did for automated
data; and redo the GT-vs-automated immotile contrast with the GT tracks
truncated to match the automated track-length distribution.

Output: outputs/markov/dwell_resolution_audit.json
Usage:  python -m experiments.dwell_resolution_audit
"""
from __future__ import annotations

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
import markov_analysis as ma  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from experiments.dwell_physics import fit_laws  # noqa: E402
from experiments.dwell_censoring import (  # noqa: E402
    episodes_with_censor,
    fit_censored_laws,
    iter_tracks,
)
from experiments.gt_reanchor import GT_DIR, BASELINE_DIR  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
OUT = config.MARKOV_OUT / "dwell_resolution_audit.json"


def frame_states_at_window(tr: pd.DataFrame, w: int) -> list[str]:
    """Identical sliding-window classification at an arbitrary window size."""
    old = ma.WINDOW_FRAMES
    ma.WINDOW_FRAMES = w
    try:
        return ma.compute_frame_states(tr)
    finally:
        ma.WINDOW_FRAMES = old


def collect_episodes(track_dir: Path, window: int):
    """Per-state episodes (seconds) with boundary flags + track lengths (frames)."""
    eps = {i: {"interior": [], "boundary_one": [], "boundary_both": []}
           for i in range(len(STATES))}
    track_lens = []
    for _, tr in iter_tracks(track_dir, max_tracks=0):
        seq = [S2I[s] for s in frame_states_at_window(tr, window)]
        track_lens.append(len(seq))
        runs = episodes_with_censor(seq)
        for j, (st, run, boundary) in enumerate(runs):
            d = run / config.FPS
            if len(runs) == 1:
                eps[st]["boundary_both"].append(d)
            elif boundary:
                eps[st]["boundary_one"].append(d)
            else:
                eps[st]["interior"].append(d)
    return eps, np.array(track_lens)


def law_summary(d: np.ndarray) -> dict:
    if len(d) < 60:
        return {"n": int(len(d)), "note": "too few episodes"}
    f = fit_laws(d)
    return {"n": int(len(d)), "best": f["best"],
            "dAIC_exp_vs_best": f["exponential"]["dAIC"],
            "dAIC_lognorm": f["lognormal"]["dAIC"],
            "dAIC_gamma": f["gamma"]["dAIC"],
            "cv": float(np.std(d) / np.mean(d))}


def resolution_block(eps: dict, window: int) -> dict:
    """T0.3 per-state resolution report + restricted refit at this window."""
    w_s = window / config.FPS
    out = {}
    for i, s in enumerate(STATES):
        d = np.array(eps[i]["interior"] + eps[i]["boundary_one"]
                     + eps[i]["boundary_both"])
        if len(d) == 0:
            out[s] = {"n": 0}
            continue
        r = {
            "n": int(len(d)),
            "min_dwell_s": float(d.min()),
            "frac_below_1x_window": float((d < 1 * w_s).mean()),
            "frac_below_2x_window": float((d < 2 * w_s).mean()),
            "frac_below_3x_window": float((d < 3 * w_s).mean()),
            "fit_all": law_summary(d),
            "fit_restricted_ge_3x_window": law_summary(d[d >= 3 * w_s]),
        }
        out[s] = r
    return out


def censoring_block(eps: dict, track_lens: np.ndarray) -> dict:
    """T0.4: boundary fractions, immotile-vs-tracklength, censored MLE."""
    out: dict = {"boundary_fractions": {}, "censored_mle": {}}
    for i, s in enumerate(STATES):
        n_i = len(eps[i]["interior"])
        n_b1 = len(eps[i]["boundary_one"])
        n_b2 = len(eps[i]["boundary_both"])
        n = n_i + n_b1 + n_b2
        out["boundary_fractions"][s] = {
            "n_total": n, "n_interior": n_i,
            "frac_touching_boundary": (n_b1 + n_b2) / n if n else None,
            "frac_whole_track": n_b2 / n if n else None,
        }
        complete = np.array(eps[i]["interior"])
        censored = np.array(eps[i]["boundary_one"] + eps[i]["boundary_both"])
        if len(complete) + len(censored) >= 60 and len(complete) >= 20:
            out["censored_mle"][s] = fit_censored_laws(complete, censored)
        else:
            out["censored_mle"][s] = {"note": "too few episodes",
                                      "n_complete": int(len(complete)),
                                      "n_censored": int(len(censored))}

    # immotile dwell distribution vs track-length distribution
    imm = np.array(eps[S2I["Immotile"]]["interior"]
                   + eps[S2I["Immotile"]]["boundary_one"]
                   + eps[S2I["Immotile"]]["boundary_both"]) * config.FPS
    tl = track_lens.astype(float)
    if len(imm) > 30:
        ks = stats.ks_2samp(imm, tl)
        out["immotile_vs_tracklength"] = {
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "immotile_quantiles_frames": {q: float(np.quantile(imm, q))
                                          for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "tracklen_quantiles_frames": {q: float(np.quantile(tl, q))
                                          for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        }
    return out


def truncated_contrast(seed: int = 0) -> dict:
    """GT immotile stats after truncating GT tracks to the automated
    track-length distribution (controls the GT-vs-automated immotile contrast
    for track length)."""
    rng = np.random.default_rng(seed)
    base_lens = []
    for _, tr in iter_tracks(BASELINE_DIR, max_tracks=0):
        base_lens.append(len(tr))
    base_lens = np.array(base_lens)

    eps = {i: [] for i in range(len(STATES))}
    n_trunc = 0
    for _, tr in iter_tracks(GT_DIR, max_tracks=0):
        n = len(tr)
        eligible = base_lens[base_lens <= n]
        if len(eligible) == 0:
            L = n
        else:
            L = int(rng.choice(eligible))
            n_trunc += 1
        start = rng.integers(0, n - L + 1)
        sub = tr.iloc[start:start + L]
        if len(sub) < config.MIN_TRACK_LENGTH:
            continue
        seq = [S2I[s] for s in ma.compute_frame_states(sub)]
        run = 1
        for k in range(1, len(seq)):
            if seq[k] == seq[k - 1]:
                run += 1
            else:
                eps[seq[k - 1]].append(run / config.FPS)
                run = 1
        eps[seq[-1]].append(run / config.FPS)

    out = {"note": ("GT tracks truncated to random segments with lengths drawn "
                    "from the automated (baseline) track-length distribution; "
                    "dwell stats recomputed identically"),
           "baseline_median_len": float(np.median(base_lens)),
           "n_truncated": n_trunc}
    for i, s in enumerate(STATES):
        d = np.array(eps[i])
        if len(d) < 60:
            out[s] = {"n": int(len(d)), "note": "too few"}
            continue
        out[s] = law_summary(d)
        out[s]["mean_dwell_s"] = float(d.mean())
    return out


def main() -> None:
    res: dict = {}

    print("T0.4/T0.3 @ window 25 (headline window) ...", flush=True)
    t0 = time.time()
    eps25, track_lens = collect_episodes(GT_DIR, 25)
    res["window_25"] = {"resolution": resolution_block(eps25, 25),
                        "censoring": censoring_block(eps25, track_lens)}
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    for s in STATES:
        r = res["window_25"]["resolution"][s]
        b = res["window_25"]["censoring"]["boundary_fractions"][s]
        print(f"  {s:16s} n={r['n']:5d} min={r['min_dwell_s']:.2f}s "
              f"<3xW={r['frac_below_3x_window']:.0%} "
              f"boundary={b['frac_touching_boundary']:.0%} "
              f"whole-track={b['frac_whole_track']:.0%}", flush=True)
        cm = res["window_25"]["censoring"]["censored_mle"][s]
        if "best" in cm:
            print(f"      censored-MLE best={cm['best']} "
                  f"dAIC_exp=+{cm['delta_aic_vs_exponential']:.0f}", flush=True)
        rr = r.get("fit_restricted_ge_3x_window", {})
        if "best" in rr:
            print(f"      restricted>=3xW: n={rr['n']} best={rr['best']} "
                  f"dAIC_exp=+{rr['dAIC_exp_vs_best']:.0f}", flush=True)

    for w in (5, 9, 13):
        print(f"T0.3 @ window {w} ...", flush=True)
        t0 = time.time()
        eps, _ = collect_episodes(GT_DIR, w)
        res[f"window_{w}"] = {"resolution": resolution_block(eps, w)}
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
        for s in STATES:
            r = res[f"window_{w}"]["resolution"][s]
            if "best" in r.get("fit_all", {}):
                print(f"  {s:16s} n={r['n']:6d} best={r['fit_all']['best']:9s} "
                      f"exp dAIC=+{r['fit_all']['dAIC_exp_vs_best']:.0f} "
                      f"cv={r['fit_all']['cv']:.2f}", flush=True)

    print("T0.4 length-matched GT-vs-automated immotile contrast ...", flush=True)
    res["length_matched_gt"] = truncated_contrast()
    for s in STATES:
        r = res["length_matched_gt"].get(s, {})
        if "best" in r:
            print(f"  {s:16s} n={r['n']:5d} best={r['best']:9s} cv={r['cv']:.2f} "
                  f"mean={r['mean_dwell_s']:.2f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
