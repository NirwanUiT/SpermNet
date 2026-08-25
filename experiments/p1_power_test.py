"""T0.1 (referee B4) -- P1 injection power test: can the continuum-null
protocol DETECT genuine switching dynamics when they exist?

Injected process, per GT track: v_t = mu + c_{m(t)} * sqrt(var) .* u_t, where
u_t is a unit-variance AR(1) with the track's fitted lag-1 persistence a, and
m(t) is a two-mode semi-Markov schedule with gamma(k) dwells (mean 2 s,
CV = 1/sqrt(k)) alternating amplitude factors c_lo, c_hi with c_hi/c_lo = R
and E[c^2] = 1. By construction the injected cohort matches GT on EXACTLY the
surfaces the OU null is fit to -- velocity marginal (mean, variance) and lag-1
autocovariance (mode dwells >> 1 frame) -- while containing genuine two-mode
switching biology (prereg 7f30aff: matching constraint pre-specified).

P1 is then executed verbatim on each injected cohort: fit per-track OU to the
injected tracks, simulate the matched memoryless null, score both with the
identical classifier machinery. Detection statistic = block-25 g2 (injected
minus matched null); secondary: per-episode dwell dAIC(exp vs best) and
switch rate. R = 1 is the negative control (zero injected dynamics).

Detection floor is reported in mode-dwell CV x mode-speed separation, the
latter both in um/s and as a fraction of the classifier threshold gap
(25 - 5 = 20 um/s).

Output: outputs/markov/p1_power_test.json
Usage:  python -m experiments.p1_power_test
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from experiments.continuum_null import fit_ou  # noqa: E402
from experiments.gt_reanchor import (GT_DIR, block_g2_at_window,  # noqa: E402
                                     dwell_law_block)

TMP = ROOT / "outputs" / "p1_power_tmp"
OUT = config.MARKOV_OUT / "p1_power_test.json"

MEAN_MODE_DWELL_F = 100          # 2 s at 50 fps
R_GRID = (1.0, 1.25, 1.5, 2.0, 3.0)   # mode amplitude ratio; 1.0 = negative control
K_GRID = (1, 4, 16)              # gamma shape; dwell CV = 1, 0.5, 0.25
SEEDS = (0, 1)
THRESHOLD_GAP_UM_S = config.VCL_PROGRESSIVE_MIN - config.VCL_IMMOTILE_MAX  # 20


def mode_schedule(n: int, k: float, rng: np.random.Generator) -> np.ndarray:
    sched = np.empty(n, dtype=int)
    m = int(rng.integers(0, 2))
    t = 0
    while t < n:
        d = max(1, int(round(rng.gamma(k, MEAN_MODE_DWELL_F / k))))
        sched[t:t + d] = m
        t += d
        m = 1 - m
    return sched


def simulate_injected(n: int, fit: dict, R: float, k: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-mode amplitude-modulated AR(1); returns positions + mode schedule."""
    mu, var, a = fit["mu"], fit["var"], fit["a"]
    sd_innov = np.sqrt(max(1.0 - a * a, 0.0))
    u = np.empty((n - 1, 2))
    u[0] = rng.normal(0.0, 1.0, 2)
    eps = rng.normal(0.0, 1.0, size=(n - 1, 2)) * sd_innov
    for t in range(1, n - 1):
        u[t] = a * u[t - 1] + eps[t]
    sched = mode_schedule(n - 1, k, rng)
    c_lo = np.sqrt(2.0 / (1.0 + R * R))
    c_hi = R * c_lo
    c = np.where(sched == 0, c_lo, c_hi)[:, None]
    v = mu + c * np.sqrt(var) * u
    pos = np.vstack([[0.0, 0.0], np.cumsum(v, axis=0)])
    return pos[:, 0], pos[:, 1], sched


def write_cohort(d: Path, tracks: dict[str, list[dict]]) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name, rows in tracks.items():
        pd.DataFrame(rows).sort_values(["track_id", "frame"]).to_csv(
            d / name, index=False)


def rows_for(tid, frames, sx, sy) -> list[dict]:
    return [{"track_id": tid, "frame": int(fr), "cx": sx[k], "cy": sy[k],
             "x1": sx[k] - 5, "y1": sy[k] - 5,
             "x2": sx[k] + 5, "y2": sy[k] + 5, "conf": 1.0}
            for k, fr in enumerate(frames)]


def load_gt_fits() -> list[tuple[str, int, np.ndarray, np.ndarray, dict]]:
    out = []
    for tf in sorted(GT_DIR.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        for tid, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            xs = tr["cx"].to_numpy(float)
            ys = tr["cy"].to_numpy(float)
            fit = fit_ou(xs, ys) if len(tr) >= 3 else None
            out.append((tf.name, tid, tr["frame"].to_numpy(), (xs, ys), fit))
    return out

def switch_rate(d: Path) -> float:
    from markov_analysis import compute_frame_states
    sw, dur = 0, 0.0
    for tf in sorted(d.glob("*_tracks.csv")):
        for _, tr in pd.read_csv(tf).groupby("track_id"):
            tr = tr.sort_values("frame")
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            states = compute_frame_states(tr)
            sw += sum(1 for i in range(1, len(states))
                      if states[i] != states[i - 1])
            dur += len(states) / config.FPS
    return sw / dur if dur else float("nan")


def score(d: Path) -> dict:
    g2 = block_g2_at_window(d, 25)
    laws = dwell_law_block(d)
    per_ep = {s: (v["dAIC_exp_vs_best"] / v["n"] if v.get("n") and "dAIC_exp_vs_best" in v else None)
              for s, v in laws.items()}
    return {"g2_25": g2["g2"], "n_block_states": g2.get("n_block_states"),
            "dAIC_per_episode": per_ep,
            "n_episodes": {s: v.get("n") for s, v in laws.items()},
            "switch_rate_per_s": switch_rate(d)}


def run_config(gt_fits, R: float, k: float, seed: int) -> dict:
    rng = np.random.default_rng(seed * 10007 + int(R * 100) * 31 + int(k))
    inj_dir, null_dir = TMP / "inj", TMP / "null"
    inj_tracks: dict[str, list[dict]] = {}
    sep_um_s = []
    # 1) injected cohort
    for name, tid, frames, (xs, ys), fit in gt_fits:
        rows = inj_tracks.setdefault(name, [])
        n = len(frames)
        if fit is None:
            sx, sy = xs, ys
            rows.extend(rows_for(tid, frames, sx, sy))
            continue
        sx, sy, sched = simulate_injected(n, fit, R, k, rng)
        sx, sy = sx + xs[0], sy + ys[0]
        rows.extend(rows_for(tid, frames, sx, sy))
        if R > 1.0 and n > 10:
            sp = np.hypot(np.diff(sx), np.diff(sy)) / config.PIXELS_PER_MICRON * config.FPS
            lo, hi = sp[sched == 0], sp[sched == 1]
            if len(lo) > 5 and len(hi) > 5:
                sep_um_s.append(abs(float(np.mean(hi)) - float(np.mean(lo))))
    write_cohort(inj_dir, inj_tracks)

    # 2) P1 verbatim: fit per-track OU to the INJECTED tracks, simulate null
    from experiments.continuum_null import simulate_ou
    null_tracks: dict[str, list[dict]] = {}
    for tf in sorted(inj_dir.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        rows = null_tracks.setdefault(tf.name, [])
        for tid, tr in df.groupby("track_id"):
            tr = tr.sort_values("frame")
            xs = tr["cx"].to_numpy(float)
            ys = tr["cy"].to_numpy(float)
            n = len(tr)
            if n < 3:
                sx, sy = xs, ys
            else:
                f = fit_ou(xs, ys)
                sx, sy = simulate_ou(n, f, rng)
                sx, sy = sx + xs[0], sy + ys[0]
            rows.extend(rows_for(tid, tr["frame"].to_numpy(), sx, sy))
    write_cohort(null_dir, null_tracks)

    inj = score(inj_dir)
    nul = score(null_dir)
    res = {
        "R": R, "k": k, "mode_dwell_cv": 1.0 / np.sqrt(k), "seed": seed,
        "mode_speed_separation_um_s_median": (float(np.median(sep_um_s))
                                              if sep_um_s else 0.0),
        "separation_frac_of_threshold_gap": (float(np.median(sep_um_s)) /
                                             THRESHOLD_GAP_UM_S if sep_um_s else 0.0),
        "injected": inj, "matched_null": nul,
        "detection_delta_g2": (inj["g2_25"] - nul["g2_25"]
                               if inj["g2_25"] is not None and nul["g2_25"] is not None
                               else None),
    }
    return res


def main() -> None:
    print("Loading GT per-track OU fits ...", flush=True)
    gt_fits = load_gt_fits()
    print(f"  {len(gt_fits)} tracks", flush=True)

    results = []
    for R in R_GRID:
        for k in (K_GRID if R > 1.0 else (1,)):   # R=1: mode irrelevant, one control
            for seed in SEEDS:
                r = run_config(gt_fits, R, k, seed)
                results.append(r)
                print(f"R={R:4.2f} k={k:2d} (dwell CV {r['mode_dwell_cv']:.2f}) "
                      f"seed={seed}  sep={r['mode_speed_separation_um_s_median']:6.2f} um/s "
                      f"({r['separation_frac_of_threshold_gap']:.2f} of gap)  "
                      f"g2 inj {r['injected']['g2_25']:+.4f} vs null "
                      f"{r['matched_null']['g2_25']:+.4f}  "
                      f"DELTA {r['detection_delta_g2']:+.4f}  "
                      f"sw {r['injected']['switch_rate_per_s']:.2f}/"
                      f"{r['matched_null']['switch_rate_per_s']:.2f}", flush=True)

    out = {
        "design": ("two-mode amplitude-modulated AR(1) per GT track; matched to "
                   "GT velocity marginal + lag-1 autocov by construction "
                   "(E[c^2]=1, mode dwell >> 1 frame); gamma(k) mode dwells, "
                   "mean 2 s; P1 executed verbatim on each injected cohort"),
        "negative_control": "R=1.0 (zero injected dynamics)",
        "threshold_gap_um_s": THRESHOLD_GAP_UM_S,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
