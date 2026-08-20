#!/usr/bin/env python3
"""
T2.1 — Spike-train statistics for motility switching: serial correlation
coefficients SCC(k) and the Fano factor of switch counts.

Treats each cell's switching record as a point process (neuroscience toolkit):

* SCC(k): correlation of state-controlled residual log-dwells at episode lag k,
  pooled within cells, k = 1..5. A single-timescale adaptation / resource-
  recovery mechanism predicts SCC(1) < 0 and SCC(k>=2) ~ 0; a slowly wandering
  rate predicts positive SCC decaying over many lags (the bacterial pattern).
* Fano factor F(T): variance/mean of switch counts in tiled windows of length
  T, pooled over cells contributing >= 3 windows. Refractory/regular switching
  predicts F < 1 at short-to-mid T.

Everything is computed identically on ground truth and on the memoryless
continuum null, with video-cluster bootstrap CIs.

Output: outputs/markov/point_process_stats.json
Run:    python -m experiments.point_process_stats --reps 2000
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
from experiments.gt_reanchor import GT_DIR  # noqa: E402
from experiments.refractory_survivor import load_episode_tracks  # noqa: E402

NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "point_process_stats.json"
MAX_LAG = 5
FANO_T = (0.5, 1.0, 2.0, 4.0, 8.0)  # seconds


# ------------------------------------------------------------------ SCC(k)
def residual_byvid(tracks: list[dict], min_eps: int,
                   per_cell_state: bool = False) -> dict:
    """Video -> list of per-track state-controlled residual log-dwell arrays.

    per_cell_state=True centres each dwell by its OWN CELL's mean log-dwell in
    that state (cells needing >=2 episodes of the state), which removes ALL
    static per-cell state-specific rate traits by construction; the
    within-track permutation null inherits the induced centring bias, so
    delta = obs - perm remains an unbiased test of *dynamic* serial structure.
    """
    logs: dict[int, list[float]] = {}
    for t in tracks:
        for st, r in t["eps"]:
            logs.setdefault(st, []).append(np.log(r / config.FPS))
    smean = {st: np.mean(v) for st, v in logs.items()}
    byvid: dict[str, list[np.ndarray]] = {}
    for t in tracks:
        if len(t["eps"]) < min_eps:
            continue
        if per_cell_state:
            cell_mean: dict[int, float] = {}
            for st in {s for s, _ in t["eps"]}:
                v = [np.log(r / config.FPS) for s, r in t["eps"] if s == st]
                cell_mean[st] = np.mean(v) if len(v) >= 2 else None
            if any(m is None for m in cell_mean.values()):
                continue
            res = np.array([np.log(r / config.FPS) - cell_mean[st]
                            for st, r in t["eps"]])
        else:
            res = np.array([np.log(r / config.FPS) - smean[st]
                            for st, r in t["eps"]])
        byvid.setdefault(t["vid"], []).append(res)
    return byvid


def scc_at_lag(byvid: dict, vids: list[str], k: int) -> float:
    a, b = [], []
    for v in vids:
        for res in byvid.get(v, []):
            if len(res) > k:
                a.extend(res[:-k]); b.extend(res[k:])
    if len(a) < 30:
        return np.nan
    return float(stats.spearmanr(a, b).correlation)


def scc_perm_null(byvid: dict, vids: list[str], k: int,
                  rng: np.random.Generator, n_perm: int = 20) -> float:
    """Within-track permutation preserves multisets; removes serial order."""
    vals = []
    for _ in range(n_perm):
        a, b = [], []
        for v in vids:
            for res in byvid.get(v, []):
                if len(res) > k:
                    sh = rng.permutation(res)
                    a.extend(sh[:-k]); b.extend(sh[k:])
        vals.append(stats.spearmanr(a, b).correlation)
    return float(np.mean(vals))


def scc_block(byvid: dict, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vids = sorted(byvid.keys())
    out = {}
    for k in range(1, MAX_LAG + 1):
        point = scc_at_lag(byvid, vids, k)
        null = scc_perm_null(byvid, vids, k, np.random.default_rng(seed + k))
        boots = []
        for _ in range(reps):
            sample = list(rng.choice(vids, size=len(vids), replace=True))
            s = scc_at_lag(byvid, sample, k)
            n = scc_perm_null(byvid, sample, k, rng, n_perm=3)
            if np.isfinite(s) and np.isfinite(n):
                boots.append(s - n)
        boots = np.array(boots)
        out[f"lag{k}"] = {
            "scc": point, "perm_null": null, "delta": point - null,
            "delta_ci95": [float(np.percentile(boots, 2.5)),
                           float(np.percentile(boots, 97.5))]}
    return out


# ------------------------------------------------------------------ Fano
def switch_times(eps: list[tuple]) -> np.ndarray:
    """Switch event times (s) within a track: cumulative dwell boundaries."""
    if len(eps) < 2:
        return np.array([])
    return np.cumsum([r for _, r in eps])[:-1] / config.FPS


# ------------------------------------------------------- power / injection
def inject_refractory(tracks: list[dict], phi: float,
                      seed: int) -> list[dict]:
    """Synthetic tracks with the SAME episode/trait structure as `tracks`
    but log-dwell residuals following AR(1) with coefficient phi.

    Preserves each cell's episode count, state sequence, and per-cell
    per-state mean log-dwell; replaces the residuals. phi=0 is the negative
    control; phi=-0.3 injects genuine dynamic refractoriness. Used to measure
    the trait-controlled estimator's power/attenuation.
    """
    rng = np.random.default_rng(seed)
    # global state means + within-cell residual sd per state
    logs: dict[int, list[float]] = {}
    for t in tracks:
        for st, r in t["eps"]:
            logs.setdefault(st, []).append(np.log(r / config.FPS))
    gmean = {st: np.mean(v) for st, v in logs.items()}
    resid: dict[int, list[float]] = {st: [] for st in gmean}
    for t in tracks:
        cm = {}
        for st in {s for s, _ in t["eps"]}:
            v = [np.log(r / config.FPS) for s, r in t["eps"] if s == st]
            cm[st] = np.mean(v) if len(v) >= 2 else gmean[st]
        for st, r in t["eps"]:
            resid[st].append(np.log(r / config.FPS) - cm[st])
    sd = {st: max(np.std(v), 1e-3) for st, v in resid.items()}

    out = []
    for t in tracks:
        cm = {}
        for st in {s for s, _ in t["eps"]}:
            v = [np.log(r / config.FPS) for s, r in t["eps"] if s == st]
            cm[st] = np.mean(v) if len(v) >= 2 else gmean[st]
        n = len(t["eps"])
        z = np.empty(n)
        z[0] = rng.standard_normal()
        for i in range(1, n):
            z[i] = phi * z[i - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
        eps_new = []
        for (st, _), zi in zip(t["eps"], z):
            r = max(1, int(round(np.exp(cm[st] + sd[st] * zi) * config.FPS)))
            eps_new.append((st, r))
        out.append({"vid": t["vid"], "eps": eps_new})
    return out


def fano_at(tracks: list[dict], vids: set[str], T: float) -> float:
    """Median per-track Fano (within-cell dispersion; immune to rate mixing)."""
    fanos = []
    for t in tracks:
        if t["vid"] not in vids:
            continue
        dur = sum(r for _, r in t["eps"]) / config.FPS
        n_win = int(dur // T)
        if n_win < 4:
            continue
        st = switch_times(t["eps"])
        counts = np.array([np.sum((st >= w * T) & (st < (w + 1) * T))
                           for w in range(n_win)], float)
        if counts.mean() > 0:
            fanos.append(counts.var(ddof=1) / counts.mean())
    if len(fanos) < 20:
        return np.nan
    return float(np.median(fanos))


def fano_block(tracks: list[dict], reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vids = sorted({t["vid"] for t in tracks})
    out = {}
    for T in FANO_T:
        point = fano_at(tracks, set(vids), T)
        boots = []
        for _ in range(reps):
            sample = rng.choice(vids, size=len(vids), replace=True)
            f = fano_at(tracks, set(sample), T)
            if np.isfinite(f):
                boots.append(f)
        boots = np.array(boots)
        if not np.isfinite(point) or len(boots) < 50:
            out[f"T{T}"] = {"fano": None, "ci95": None,
                            "note": "insufficient tracks at this window"}
            continue
        out[f"T{T}"] = {"fano": point,
                        "ci95": [float(np.percentile(boots, 2.5)),
                                 float(np.percentile(boots, 97.5))]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--min-eps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res: dict = {"min_episodes": args.min_eps, "fano_windows_s": list(FANO_T)}

    print("loading GT episodes ...", flush=True)
    gt = load_episode_tracks(GT_DIR)
    print("loading continuum-null episodes ...", flush=True)
    nl = load_episode_tracks(NULL_DIR)

    for name, tracks in (("gt", gt), ("continuum_null", nl)):
        print(f"{name}: SCC(k) with cluster bootstrap ...", flush=True)
        byvid = residual_byvid(tracks, args.min_eps)
        res[name] = {"scc": scc_block(byvid, args.reps, args.seed)}
        for k, v in res[name]["scc"].items():
            print(f"  {name} {k}: delta {v['delta']:+.3f} CI {v['delta_ci95']}",
                  flush=True)
        # decisive variant: per-cell-per-state centring removes static traits
        print(f"{name}: SCC(k), trait-controlled ...", flush=True)
        byvid_tc = residual_byvid(tracks, args.min_eps, per_cell_state=True)
        res[name]["scc_trait_controlled"] = scc_block(byvid_tc, args.reps,
                                                      args.seed + 50)
        for k, v in res[name]["scc_trait_controlled"].items():
            print(f"  {name} TC {k}: delta {v['delta']:+.3f} CI {v['delta_ci95']}",
                  flush=True)
        print(f"{name}: Fano factor (per-track) ...", flush=True)
        res[name]["fano"] = fano_block(tracks, min(args.reps, 1000),
                                       args.seed + 7)
        for k, v in res[name]["fano"].items():
            if v["fano"] is None:
                print(f"  {name} {k}: insufficient tracks", flush=True)
            else:
                print(f"  {name} {k}: F {v['fano']:.3f} CI {v['ci95']}", flush=True)

    # power test: can the trait-controlled estimator detect injected phi=-0.3?
    res["power_test"] = {}
    for phi in (-0.3, 0.0):
        inj = inject_refractory(gt, phi, args.seed + 99)
        byvid_inj = residual_byvid(inj, args.min_eps, per_cell_state=True)
        blk = scc_block(byvid_inj, min(args.reps, 500), args.seed + 60)
        res["power_test"][f"phi_{phi}"] = {k: blk[k]["delta"] for k in blk} | {
            "lag1_ci95": blk["lag1"]["delta_ci95"]}
        print(f"power test phi={phi}: TC lag1 delta "
              f"{blk['lag1']['delta']:+.3f} CI {blk['lag1']['delta_ci95']}",
              flush=True)

    # verdict: dynamic refractoriness must survive trait control on GT
    g = res["gt"]["scc_trait_controlled"]
    res["verdict"] = {
        "gt_tc_scc1_negative": bool(g["lag1"]["delta_ci95"][1] < 0),
        "gt_tc_scc": {k: g[k]["delta"] for k in g},
        "power_lag1_at_phi_-0.3": res["power_test"]["phi_-0.3"]["lag1"],
        "note": ("raw SCC parity pattern (odd<0, even>0) is compatible with "
                 "static per-cell state traits; the trait-controlled variant "
                 "is the decisive dynamic test")}
    print("\nVERDICT:", json.dumps(res["verdict"], indent=2), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
