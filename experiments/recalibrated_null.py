"""Referee item C: recalibrate the null on the surface the classifier actually
reads.

The pre-registered continuum null fits a per-track AR(1) velocity process to the
raw velocity marginal (mean, variance, lag-1). That preserves the velocity
marginal but leaves the *windowed-VCL* autocorrelation too short, so the
classifier crosses thresholds ~1.4x too often and the null over-counts episodes
in every state. This over-switching is the one soft spot in the dwell-law
comparison (the g2 conclusion is already secured independently by the
homogeneous null and the phase diagram, which need no calibration).

Here we build a supplementary AR(2) null: per track we fit lag-1 AND lag-2
velocity autocorrelation (Yule-Walker), preserving the velocity marginal
variance exactly while adding the second timescale AR(1) misses, so that the
windowed-VCL ACF out to lag 50 -- and hence the switch rate -- match the ground
truth far better. We then re-run ONLY the dwell rows on this recalibrated null.
This is NOT a revision of the pre-registered primary null; it is a robustness
check that the dwell-law conclusion survives matching the switch rate. We fit
kinematic autocorrelation of the *continuous velocity surface*, never any
event-sequence memory statistic.

Output: outputs/tracks_continuum_null_ar2/, outputs/markov/recalibrated_null.json
Run:    python -m experiments.recalibrated_null --materialise
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import markov_analysis as ma  # noqa: E402
from experiments.gt_reanchor import GT_DIR  # noqa: E402
from experiments.refractory_survivor import load_episode_tracks  # noqa: E402
from experiments.dwell_physics import fit_laws  # noqa: E402

AR1_DIR = ROOT / "outputs" / "tracks_continuum_null"
AR2_DIR = ROOT / "outputs" / "tracks_continuum_null_ar2"
OUT = config.MARKOV_OUT / "recalibrated_null.json"
W = ma.WINDOW_FRAMES - 1
S2I = {s: i for i, s in enumerate(ma.STATES)}


def fit_ar2(xs: np.ndarray, ys: np.ndarray) -> dict:
    """Per-track AR(2) velocity fit via pooled Yule-Walker; preserves marginal
    variance. Falls back to AR(1) when non-stationary."""
    vx, vy = np.diff(xs), np.diff(ys)
    mu = np.array([vx.mean(), vy.mean()])
    var = np.array([vx.var(), vy.var()])
    cx, cy = vx - mu[0], vy - mu[1]
    den = float(np.dot(cx, cx) + np.dot(cy, cy))
    if den <= 0 or len(cx) < 4:
        return {"mu": mu, "var": var, "a1": 0.0, "a2": 0.0}
    r1 = float(np.dot(cx[1:], cx[:-1]) + np.dot(cy[1:], cy[:-1])) / den
    r2 = float(np.dot(cx[2:], cx[:-2]) + np.dot(cy[2:], cy[:-2])) / den
    d = 1.0 - r1 * r1
    if abs(d) < 1e-9:
        a1, a2 = np.clip(r1, -0.995, 0.995), 0.0
    else:
        a1 = r1 * (1.0 - r2) / d
        a2 = (r2 - r1 * r1) / d
    # enforce stationarity of AR(2); else drop to AR(1)
    if not (abs(a2) < 0.999 and a1 + a2 < 0.999 and a2 - a1 < 0.999):
        a1, a2 = float(np.clip(r1, -0.995, 0.995)), 0.0
    return {"mu": mu, "var": var, "a1": float(a1), "a2": float(a2)}


def simulate_ar2(n: int, fit: dict, rng: np.random.Generator):
    mu, var, a1, a2 = fit["mu"], fit["var"], fit["a1"], fit["a2"]
    # innovation variance so the stationary variance equals var
    r1 = a1 / (1.0 - a2) if abs(1.0 - a2) > 1e-9 else 0.0
    scale = max(1.0 - a1 * r1 - a2 * (a1 * r1 + a2), 1e-6)
    sd_innov = np.sqrt(np.maximum(var * scale, 0.0))
    m = n - 1
    v = np.empty((m, 2))
    v[0] = rng.normal(0.0, np.sqrt(var))
    if m > 1:
        v[1] = a1 * v[0] + rng.normal(0.0, np.sqrt(var))
    eps = rng.normal(0.0, 1.0, size=(m, 2)) * sd_innov
    for t in range(2, m):
        v[t] = a1 * v[t - 1] + a2 * v[t - 2] + eps[t]
    v += mu
    pos = np.vstack([[0.0, 0.0], np.cumsum(v, axis=0)])
    return pos[:, 0], pos[:, 1]


def materialise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    AR2_DIR.mkdir(parents=True, exist_ok=True)
    for tf in sorted(GT_DIR.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        rows = []
        for tid, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            n = len(tr)
            xs = tr["cx"].to_numpy(float)
            ys = tr["cy"].to_numpy(float)
            if n < 4:
                sx, sy = xs - xs[0], ys - ys[0]
            else:
                sx, sy = simulate_ar2(n, fit_ar2(xs, ys), rng)
            sx, sy = sx + xs[0], sy + ys[0]
            for k, fr in enumerate(tr["frame"].to_numpy()):
                rows.append({"track_id": tid, "frame": int(fr),
                             "cx": sx[k], "cy": sy[k],
                             "x1": sx[k] - 5, "y1": sy[k] - 5,
                             "x2": sx[k] + 5, "y2": sy[k] + 5, "conf": 1.0})
        pd.DataFrame(rows).sort_values(["track_id", "frame"]).to_csv(
            AR2_DIR / tf.name, index=False)
        print(f"  {tf.stem}: materialised", flush=True)


def wvcl_acf(track_dir: Path, max_lag: int = 50) -> np.ndarray:
    """Mean per-track windowed-VCL autocorrelation, lags 1..max_lag."""
    acc = np.zeros(max_lag)
    cnt = np.zeros(max_lag)
    for tf in sorted(track_dir.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        for _, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            x = tr["cx"].to_numpy(float)
            y = tr["cy"].to_numpy(float)
            sp = np.hypot(np.diff(x), np.diff(y))
            if len(sp) <= W:
                continue
            cs = np.concatenate([[0.0], np.cumsum(sp)])
            w = (cs[W:] - cs[:-W]) / W
            w = w - w.mean()
            v0 = float(np.dot(w, w))
            if v0 <= 0:
                continue
            for L in range(1, max_lag + 1):
                if len(w) > L:
                    acc[L - 1] += np.dot(w[L:], w[:-L]) / v0
                    cnt[L - 1] += 1
    return acc / np.maximum(cnt, 1)


def dwell_rows(track_dir: Path) -> dict:
    tracks = load_episode_tracks(track_dir)
    per = {i: [] for i in range(len(ma.STATES))}
    n_switch = 0
    n_frames = 0
    for t in tracks:
        for st, r in t["eps"]:
            per[st].append(r / config.FPS)
        n_switch += max(0, len(t["eps"]) - 1)
        n_frames += sum(r for _, r in t["eps"])
    out = {"switch_rate_per_s": n_switch / (n_frames / config.FPS)
           if n_frames else float("nan")}
    for st, name in ((S2I["Progressive"], "P"), (S2I["Non-progressive"], "NP"),
                     (S2I["Immotile"], "I")):
        d = np.array(per[st], float)
        d = d[d > 0]
        rec = {"n_episodes": int(len(d))}
        if len(d) >= 100:
            f = fit_laws(d)
            rec["dAIC_exp_per_episode"] = float(f["exponential"]["dAIC"] / len(d))
            rec["best"] = f["best"]
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.materialise:
        print("materialising AR(2) recalibrated null ...", flush=True)
        materialise(args.seed)

    res = {}
    print("windowed-VCL ACF (lags 1..50) ...", flush=True)
    acf_gt = wvcl_acf(GT_DIR)
    acf_ar1 = wvcl_acf(AR1_DIR)
    acf_ar2 = wvcl_acf(AR2_DIR)
    res["wvcl_acf"] = {"lags": list(range(1, 51)),
                       "gt": acf_gt.tolist(), "ar1": acf_ar1.tolist(),
                       "ar2": acf_ar2.tolist()}
    # ACF match error (mean abs deviation from GT over lags 1..50)
    res["acf_mad_vs_gt"] = {"ar1": float(np.mean(np.abs(acf_ar1 - acf_gt))),
                            "ar2": float(np.mean(np.abs(acf_ar2 - acf_gt)))}
    print("dwell rows ...", flush=True)
    res["gt"] = dwell_rows(GT_DIR)
    res["ar1_null"] = dwell_rows(AR1_DIR)
    res["ar2_null"] = dwell_rows(AR2_DIR)
    OUT.write_text(json.dumps(res, indent=2))

    def line(name, d):
        return (f"{name:10s} switch={d['switch_rate_per_s']:.2f}/s  "
                f"P n={d['P']['n_episodes']} dAIC/ep={d['P'].get('dAIC_exp_per_episode', float('nan')):.3f}  "
                f"NP n={d['NP']['n_episodes']} dAIC/ep={d['NP'].get('dAIC_exp_per_episode', float('nan')):.3f}  "
                f"I n={d['I']['n_episodes']} dAIC/ep={d['I'].get('dAIC_exp_per_episode', float('nan')):.3f}")
    print("\nACF MAD vs GT:  AR1 %.4f  AR2 %.4f" % (
        res["acf_mad_vs_gt"]["ar1"], res["acf_mad_vs_gt"]["ar2"]))
    print("ACF@lag10  GT %.3f  AR1 %.3f  AR2 %.3f" % (
        acf_gt[9], acf_ar1[9], acf_ar2[9]))
    print("ACF@lag25  GT %.3f  AR1 %.3f  AR2 %.3f" % (
        acf_gt[24], acf_ar1[24], acf_ar2[24]))
    print(line("GT", res["gt"]))
    print(line("AR1 null", res["ar1_null"]))
    print(line("AR2 null", res["ar2_null"]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
