"""T0.4 (referee B5) -- Wild cluster bootstrap for the serial-dwell statistic.

With only 20 video clusters, the pairs-cluster (resampling) bootstrap can be
anticonservative. This re-tests the headline serial statistics with the
Cameron-Gelbach-Miller wild cluster bootstrap-t (Rademacher weights) on
video-level statistics:

  delta_v = SCC_1(video v) - perm_null(video v)

for (a) GT raw state-controlled residuals (headline delta = -0.228),
(b) GT trait-controlled residuals (the powered null, +0.025), (c) the
continuum null raw (-0.06). Also reports the cell counts entering each
estimator -- in particular n cells in the trait-controlled subset (>= 2
episodes of every visited state), which the manuscript must state.

Output: outputs/markov/wild_cluster.json
Usage:  python -m experiments.wild_cluster
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from experiments.gt_reanchor import GT_DIR  # noqa: E402
from experiments.point_process_stats import (residual_byvid,  # noqa: E402
                                             scc_at_lag, scc_perm_null)
from experiments.refractory_survivor import load_episode_tracks  # noqa: E402

NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "wild_cluster.json"
MIN_EPS = 4
N_WILD = 9999
LAGS = (1, 2)


def video_deltas(byvid: dict, k: int, seed: int) -> dict[str, float]:
    """delta_v computed within each video alone."""
    out = {}
    for v in sorted(byvid):
        rng = np.random.default_rng(seed)
        obs = scc_at_lag(byvid, [v], k)
        if not np.isfinite(obs):
            continue
        null = scc_perm_null(byvid, [v], k, rng, n_perm=50)
        if np.isfinite(null):
            out[v] = obs - null
    return out


def wild_test(deltas: dict[str, float], seed: int) -> dict:
    """Wild cluster bootstrap-t, Rademacher weights, H0: mean delta = 0."""
    d = np.array(list(deltas.values()))
    g = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(g)
    t_obs = mean / se
    rng = np.random.default_rng(seed)
    t_star = np.empty(N_WILD)
    for b in range(N_WILD):
        w = rng.choice([-1.0, 1.0], size=g)
        db = w * d                      # residuals under imposed H0 (mu = 0)
        mb = db.mean()
        sb = db.std(ddof=1) / np.sqrt(g)
        t_star[b] = mb / sb if sb > 0 else 0.0
    p = float((np.sum(np.abs(t_star) >= abs(t_obs)) + 1) / (N_WILD + 1))
    t_crit = float(np.percentile(np.abs(t_star), 97.5))
    return {"n_clusters": g, "mean_video_delta": float(mean),
            "cluster_se": float(se), "t": float(t_obs),
            "wild_p_two_sided": p,
            "wild_ci95": [float(mean - t_crit * se), float(mean + t_crit * se)]}


def main() -> None:
    res: dict = {"method": ("Cameron-Gelbach-Miller wild cluster bootstrap-t, "
                            "Rademacher weights, video-level delta = "
                            "SCC_k(video) - within-track perm null(video), "
                            f"{N_WILD} draws"),
                 "min_episodes": MIN_EPS}

    for name, d in (("gt", GT_DIR), ("continuum_null", NULL_DIR)):
        print(f"loading {name} episodes ...", flush=True)
        tracks = load_episode_tracks(d)
        raw = residual_byvid(tracks, MIN_EPS)
        tc = residual_byvid(tracks, MIN_EPS, per_cell_state=True)
        n_raw = sum(len(v) for v in raw.values())
        n_tc = sum(len(v) for v in tc.values())
        res[name] = {"n_cells_raw": n_raw, "n_cells_trait_controlled": n_tc,
                     "n_videos_raw": len(raw), "n_videos_tc": len(tc)}
        print(f"  {name}: {n_raw} cells raw ({len(raw)} videos), "
              f"{n_tc} cells trait-controlled ({len(tc)} videos)", flush=True)
        for variant, byvid in (("raw", raw), ("trait_controlled", tc)):
            for k in LAGS:
                dv = video_deltas(byvid, k, seed=100 + k)
                w = wild_test(dv, seed=200 + k)
                pooled = scc_at_lag(byvid, sorted(byvid), k) - scc_perm_null(
                    byvid, sorted(byvid), k, np.random.default_rng(7), n_perm=50)
                w["pooled_delta"] = float(pooled)
                res[name][f"{variant}_lag{k}"] = w
                print(f"  {name} {variant} lag{k}: pooled {pooled:+.3f} | "
                      f"video-mean {w['mean_video_delta']:+.3f} "
                      f"[{w['wild_ci95'][0]:+.3f}, {w['wild_ci95'][1]:+.3f}] "
                      f"wild p = {w['wild_p_two_sided']:.4f} "
                      f"({w['n_clusters']} clusters)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
