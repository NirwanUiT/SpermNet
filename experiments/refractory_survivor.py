"""T0.6 + T1.1 + T1.4 -- Is the refractory serial anti-correlation real?

The continuum null (T0.1) reproduces the dwell laws, the block g2, the ICC and
the sub-exponential within-cell CV -- but only ~a quarter of the serial
anti-correlation (null drho ~ -0.06 vs GT -0.24). This script subjects that
sole surviving statistic to its remaining failure modes:

  (1) FLICKER (T0.6): a 1-2-block misclassification inside a long dwell
      produces long-short-long triplets => spurious negative lag-1. We measure
      the frequency of short episodes flanked by the same state on both sides
      and recompute drho after merging them (thresholds 25 and 50 frames).
  (2) CLUSTER INFERENCE (T1.1): dwell pairs are nested in cells nested in 20
      videos. All CIs here are video-level cluster bootstraps; we also report
      leave-one-video-out ranges. The pooled p=2.5e-20 is retired.
  (3) PER-CELL ESTIMATOR (T1.4): per-cell lag-1 Spearman, Fisher-z averaged,
      with cluster-bootstrap CI -- no pooling across cells at all.
  (4) CONTINUUM-NULL CALIBRATION: the identical estimator run on the
      T0.1 synthetic tracks; the claim is GT minus null, with a bootstrap CI
      on the difference.

Estimator (identical to memory_decomposition.analyse): state-controlled
residual log-dwell, lag-1 Spearman over successive within-track episode pairs,
against a within-track permutation null (preserves each cell's dwell multiset,
so censoring/heterogeneity cannot fake it).

Output: outputs/markov/refractory_survivor.json
Usage:  python -m experiments.refractory_survivor [--reps 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import markov_analysis as ma  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from experiments.memory_decomposition import episodes_of, iter_tracks  # noqa: E402
from experiments.gt_reanchor import GT_DIR  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "refractory_survivor.json"


# ------------------------------------------------------------------ loading
def load_episode_tracks(track_dir: Path) -> list[dict]:
    """Per track: video id + ordered (state, dwell_frames) episode list."""
    tracks = []
    for tkey, tr in iter_tracks(track_dir, max_tracks=0):
        vid = tkey.split(":")[0]
        seq = [S2I[s] for s in ma.compute_frame_states(tr)]
        tracks.append({"vid": vid, "eps": episodes_of(seq)})
    return tracks


def merge_flickers(eps: list[tuple], thr_frames: int) -> tuple[list[tuple], int]:
    """Merge episodes <= thr flanked by the same state on both sides."""
    eps = list(eps)
    merged = 0
    changed = True
    while changed:
        changed = False
        for j in range(1, len(eps) - 1):
            st_p, r_p = eps[j - 1]
            st, r = eps[j]
            st_n, r_n = eps[j + 1]
            if r <= thr_frames and st_p == st_n and st != st_p:
                eps[j - 1] = (st_p, r_p + r + r_n)
                del eps[j:j + 2]
                merged += 1
                changed = True
                break
    return eps, merged


# ------------------------------------------------------------------ statistic
def residual_series(tracks: list[dict]) -> dict:
    """Per video: list of per-track residual log-dwell arrays (>=4 episodes)."""
    logs = {i: [] for i in range(len(STATES))}
    for t in tracks:
        for st, r in t["eps"]:
            logs[st].append(np.log(r / config.FPS))
    smean = {i: np.mean(v) if v else 0.0 for i, v in logs.items()}
    byvid: dict[str, list[np.ndarray]] = {}
    for t in tracks:
        if len(t["eps"]) < 4:
            continue
        res = np.array([np.log(r / config.FPS) - smean[st] for st, r in t["eps"]])
        byvid.setdefault(t["vid"], []).append(res)
    return byvid


def delta_rho(byvid: dict, vids: list[str], rng: np.random.Generator,
              n_perm: int = 20) -> tuple[float, float, float]:
    """Pooled lag-1 Spearman minus within-track permutation null over vids."""
    a, b = [], []
    tracks = [res for v in vids for res in byvid.get(v, [])]
    for res in tracks:
        a.extend(res[:-1]); b.extend(res[1:])
    if len(a) < 30:
        return np.nan, np.nan, np.nan
    rho = stats.spearmanr(a, b).correlation
    nulls = []
    for _ in range(n_perm):
        sa, sb = [], []
        for res in tracks:
            sh = rng.permutation(res)
            sa.extend(sh[:-1]); sb.extend(sh[1:])
        nulls.append(stats.spearmanr(sa, sb).correlation)
    null = float(np.mean(nulls))
    return float(rho), null, float(rho - null)


def per_cell_delta(byvid: dict, vids: list[str], rng: np.random.Generator,
                   n_perm: int = 50, min_eps: int = 6) -> dict:
    """Per-cell lag-1 Spearman minus per-cell permutation null, Fisher-z mean."""
    deltas, zs = [], []
    for v in vids:
        for res in byvid.get(v, []):
            if len(res) < min_eps:
                continue
            r = stats.spearmanr(res[:-1], res[1:]).correlation
            if not np.isfinite(r):
                continue
            nl = []
            for _ in range(n_perm):
                sh = rng.permutation(res)
                rn = stats.spearmanr(sh[:-1], sh[1:]).correlation
                if np.isfinite(rn):
                    nl.append(rn)
            deltas.append(r - np.mean(nl))
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
    return {"n_cells": len(deltas),
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "fisher_z_mean_rho": float(np.tanh(np.mean(zs))) if zs else None}


def cluster_bootstrap(byvid: dict, reps: int, seed: int) -> dict:
    """Video-level bootstrap CI for delta-rho + leave-one-video-out range."""
    rng = np.random.default_rng(seed)
    vids = sorted(byvid.keys())
    rho, null, delta = delta_rho(byvid, vids, np.random.default_rng(seed + 1))
    boots = []
    for _ in range(reps):
        sample = list(rng.choice(vids, size=len(vids), replace=True))
        _, _, d = delta_rho(byvid, sample, rng, n_perm=5)
        if np.isfinite(d):
            boots.append(d)
    boots = np.array(boots)
    loo = []
    for v in vids:
        _, _, d = delta_rho(byvid, [w for w in vids if w != v],
                            np.random.default_rng(seed + 2), n_perm=10)
        loo.append(d)
    return {"rho": rho, "perm_null": null, "delta_rho": delta,
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
            "boot_reps": int(len(boots)),
            "p_boot_ge_0": float((boots >= 0).mean()),
            "loo_range": [float(min(loo)), float(max(loo))],
            "n_videos": len(vids)}


def flicker_stats(tracks: list[dict], thr: int) -> dict:
    n_flicker = 0
    merged_tracks = []
    for t in tracks:
        eps, m = merge_flickers(t["eps"], thr)
        n_flicker += m
        merged_tracks.append({"vid": t["vid"], "eps": eps})
    n_eps = sum(len(t["eps"]) for t in tracks)
    return {"n_flickers_merged": n_flicker,
            "flicker_frac_of_episodes": n_flicker / n_eps if n_eps else None,
            "tracks": merged_tracks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res: dict = {}

    print("loading GT episode tracks ...", flush=True)
    gt_tracks = load_episode_tracks(GT_DIR)
    print("loading continuum-null episode tracks ...", flush=True)
    null_tracks = load_episode_tracks(NULL_DIR)

    print("GT: cluster-bootstrap delta-rho ...", flush=True)
    gt_byvid = residual_series(gt_tracks)
    res["gt"] = cluster_bootstrap(gt_byvid, args.reps, args.seed)
    res["gt"]["per_cell"] = per_cell_delta(gt_byvid, sorted(gt_byvid),
                                           np.random.default_rng(args.seed + 3))
    print(f"  GT drho = {res['gt']['delta_rho']:+.3f} "
          f"CI95 {res['gt']['ci95']} LOO {res['gt']['loo_range']} "
          f"per-cell mean {res['gt']['per_cell']['mean_delta']}", flush=True)

    print("continuum null: cluster-bootstrap delta-rho ...", flush=True)
    nl_byvid = residual_series(null_tracks)
    res["continuum_null"] = cluster_bootstrap(nl_byvid, args.reps, args.seed + 10)
    print(f"  null drho = {res['continuum_null']['delta_rho']:+.3f} "
          f"CI95 {res['continuum_null']['ci95']}", flush=True)

    # bootstrap the GT-minus-null difference (independent video resampling)
    rng = np.random.default_rng(args.seed + 20)
    gvids, nvids = sorted(gt_byvid), sorted(nl_byvid)
    diffs = []
    for _ in range(min(args.reps, 2000)):
        gs = list(rng.choice(gvids, size=len(gvids), replace=True))
        ns = list(rng.choice(nvids, size=len(nvids), replace=True))
        _, _, dg = delta_rho(gt_byvid, gs, rng, n_perm=5)
        _, _, dn = delta_rho(nl_byvid, ns, rng, n_perm=5)
        if np.isfinite(dg) and np.isfinite(dn):
            diffs.append(dg - dn)
    diffs = np.array(diffs)
    res["gt_minus_null"] = {
        "point": res["gt"]["delta_rho"] - res["continuum_null"]["delta_rho"],
        "ci95": [float(np.percentile(diffs, 2.5)),
                 float(np.percentile(diffs, 97.5))],
        "p_boot_ge_0": float((diffs >= 0).mean())}
    print(f"  GT-minus-null = {res['gt_minus_null']['point']:+.3f} "
          f"CI95 {res['gt_minus_null']['ci95']}", flush=True)

    for thr in (25, 50):
        print(f"flicker merge @ {thr} frames ...", flush=True)
        fk = flicker_stats(gt_tracks, thr)
        byvid = residual_series(fk["tracks"])
        cb = cluster_bootstrap(byvid, min(args.reps, 2000), args.seed + thr)
        res[f"gt_flicker_merged_{thr}f"] = {
            "n_flickers_merged": fk["n_flickers_merged"],
            "flicker_frac_of_episodes": fk["flicker_frac_of_episodes"],
            **{k: cb[k] for k in ("delta_rho", "ci95", "loo_range")}}
        print(f"  merged {fk['n_flickers_merged']} "
              f"({fk['flicker_frac_of_episodes']:.1%} of episodes) -> "
              f"drho = {cb['delta_rho']:+.3f} CI95 {cb['ci95']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
