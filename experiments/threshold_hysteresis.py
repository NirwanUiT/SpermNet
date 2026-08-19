#!/usr/bin/env python3
"""
T0.5 — Threshold sweep + hysteresis classifier: last remaining threat to the
surviving refractory signature (delta-rho).

The refractory statistic could in principle be manufactured by the *placement*
of the classifier thresholds (dwells alternate long/short because the VCL
process meanders across a fixed cut) rather than by biology. Two controls:

1. Threshold sweep: recompute per-frame states on the ground-truth tracks for
   a grid of (VCL_IMMOTILE_MAX, VCL_PROGRESSIVE_MIN, STR_PROGRESSIVE_MIN)
   perturbations and re-estimate delta-rho with video-cluster bootstrap CIs,
   alongside the matched continuum null at identical settings.

2. Hysteresis (Schmitt-trigger) classifier: entering a state requires crossing
   a strict threshold, leaving it a loose one (relative margin m). This is the
   standard suppressor of threshold-crossing flicker: if delta-rho is a
   boundary artefact it must shrink toward the null under hysteresis.

Also: delta-rho at alternative classifier windows (13/51 frames).

Output: outputs/markov/threshold_hysteresis.json
Run:    python -m experiments.threshold_hysteresis --reps 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import markov_analysis as ma  # noqa: E402
from markov_analysis import STATES  # noqa: E402
from events.detect_events import px_to_um  # noqa: E402
from experiments.memory_decomposition import episodes_of, iter_tracks  # noqa: E402
from experiments.gt_reanchor import GT_DIR  # noqa: E402
from experiments.refractory_survivor import (  # noqa: E402
    residual_series, cluster_bootstrap, delta_rho)

NULL_DIR = ROOT / "outputs" / "tracks_continuum_null"
OUT = config.MARKOV_OUT / "threshold_hysteresis.json"

IMM, PROG, NP = (STATES.index("Immotile"), STATES.index("Progressive"),
                 STATES.index("Non-progressive"))
BASE = (config.VCL_IMMOTILE_MAX, config.VCL_PROGRESSIVE_MIN,
        config.STR_PROGRESSIVE_MIN)


# ------------------------------------------------------------- kinematics
def window_vcl_str(xs: np.ndarray, ys: np.ndarray, dt: float) -> tuple[float, float]:
    """Numerically identical to events.detect_events.classify_window."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    dx, dy = np.diff(xs), np.diff(ys)
    vcl = np.mean(px_to_um(np.sqrt(dx**2 + dy**2))) / dt
    disp = px_to_um(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    vsl = disp / ((n - 1) * dt)
    w = min(5, n)
    xs_s = uniform_filter1d(xs.astype(float), size=w)
    ys_s = uniform_filter1d(ys.astype(float), size=w)
    vap = np.mean(px_to_um(np.hypot(np.diff(xs_s), np.diff(ys_s)))) / dt
    return float(vcl), float(vsl / vap) if vap > 0 else 0.0


def frame_kinematics(tr, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (VCL, STR) via the same sliding window as compute_frame_states."""
    dt = 1.0 / config.FPS
    xs = tr["cx"].values.astype(float)
    ys = tr["cy"].values.astype(float)
    n = len(xs)
    if n < window:
        v, s = window_vcl_str(xs, ys, dt)
        return np.full(n, v), np.full(n, s)
    half = window // 2
    vcl = np.empty(n)
    strv = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vcl[i], strv[i] = window_vcl_str(xs[lo:hi], ys[lo:hi], dt)
    return vcl, strv


def load_kinematics(track_dir: Path, window: int) -> list[dict]:
    out = []
    for tkey, tr in iter_tracks(track_dir, max_tracks=0):
        vcl, strv = frame_kinematics(tr, window)
        out.append({"vid": tkey.split(":")[0], "vcl": vcl, "str": strv})
    return out


# ------------------------------------------------------------- classifiers
def states_fixed(vcl, strv, v_imm, v_prog, s_prog) -> np.ndarray:
    st = np.full(len(vcl), NP)
    st[vcl <= v_imm] = IMM
    st[(vcl >= v_prog) & (strv >= s_prog) & (vcl > v_imm)] = PROG
    return st


def states_hysteresis(vcl, strv, margin: float) -> np.ndarray:
    """Schmitt trigger: strict entry, loose exit, per boundary."""
    v_imm, v_prog, s_prog = BASE
    lo_i, hi_i = v_imm * (1 - margin), v_imm * (1 + margin)
    lo_p, hi_p = v_prog * (1 - margin), v_prog * (1 + margin)
    lo_s, hi_s = s_prog * (1 - margin), s_prog * (1 + margin)
    st = int(states_fixed(vcl[:1], strv[:1], v_imm, v_prog, s_prog)[0])
    out = np.empty(len(vcl), dtype=int)
    out[0] = st
    for k in range(1, len(vcl)):
        v, s = vcl[k], strv[k]
        if st == IMM:
            if v > hi_i:  # must exceed strict cut to leave immotile
                st = PROG if (v >= hi_p and s >= hi_s) else NP
        elif st == PROG:
            if v <= lo_i:
                st = IMM
            elif v < lo_p or s < lo_s:  # loose exit from progressive
                st = NP
        else:  # NP
            if v <= lo_i:  # strict entry to immotile
                st = IMM
            elif v >= hi_p and s >= hi_s:  # strict entry to progressive
                st = PROG
        out[k] = st
    return out


# ------------------------------------------------------------- evaluation
def eps_tracks(kins: list[dict], state_fn) -> list[dict]:
    return [{"vid": k["vid"], "eps": episodes_of(list(state_fn(k["vcl"], k["str"])))}
            for k in kins]


def evaluate(kins_gt, kins_nl, state_fn, reps, null_reps, seed) -> dict:
    gt_byvid = residual_series(eps_tracks(kins_gt, state_fn))
    cb = cluster_bootstrap(gt_byvid, reps, seed)
    nl_byvid = residual_series(eps_tracks(kins_nl, state_fn))
    _, _, d_nl = delta_rho(nl_byvid, sorted(nl_byvid),
                           np.random.default_rng(seed + 1))
    nl_cb = cluster_bootstrap(nl_byvid, null_reps, seed + 2) if null_reps else None
    # composition + episode counts (sanity)
    n_eps = sum(len(t["eps"]) for t in eps_tracks(kins_gt, state_fn))
    return {"gt": {k: cb[k] for k in ("delta_rho", "ci95", "loo_range", "rho",
                                      "perm_null", "n_videos")},
            "null_delta_rho": float(d_nl),
            "null_ci95": nl_cb["ci95"] if nl_cb else None,
            "gt_minus_null_point": cb["delta_rho"] - float(d_nl),
            "n_gt_episodes": int(n_eps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=800)
    ap.add_argument("--null-reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res: dict = {"baseline_thresholds": {"v_imm": BASE[0], "v_prog": BASE[1],
                                         "s_prog": BASE[2]},
                 "window_frames": ma.WINDOW_FRAMES}

    print("computing GT kinematics (w25) ...", flush=True)
    kins_gt = load_kinematics(GT_DIR, ma.WINDOW_FRAMES)
    print("computing continuum-null kinematics (w25) ...", flush=True)
    kins_nl = load_kinematics(NULL_DIR, ma.WINDOW_FRAMES)

    # sanity: baseline reproduces the committed classifier exactly
    for tkey, tr in iter_tracks(GT_DIR, max_tracks=5):
        ref = [STATES.index(s) for s in ma.compute_frame_states(tr)]
        vcl, strv = frame_kinematics(tr, ma.WINDOW_FRAMES)
        assert list(states_fixed(vcl, strv, *BASE)) == ref, "classifier mismatch"
    print("sanity check passed: baseline == committed classifier", flush=True)

    # ---- threshold sweep ------------------------------------------------
    v_imm0, v_prog0, s_prog0 = BASE
    grid = {
        "baseline": BASE,
        "v_imm_3": (3.0, v_prog0, s_prog0),
        "v_imm_4": (4.0, v_prog0, s_prog0),
        "v_imm_6": (6.0, v_prog0, s_prog0),
        "v_imm_8": (8.0, v_prog0, s_prog0),
        "v_prog_20": (v_imm0, 20.0, s_prog0),
        "v_prog_30": (v_imm0, 30.0, s_prog0),
        "s_prog_040": (v_imm0, v_prog0, 0.40),
        "s_prog_060": (v_imm0, v_prog0, 0.60),
        "joint_loose": (3.0, 20.0, 0.40),
        "joint_strict": (8.0, 30.0, 0.60),
    }
    res["threshold_sweep"] = {}
    for i, (name, (vi, vp, sp)) in enumerate(grid.items()):
        fn = lambda v, s, vi=vi, vp=vp, sp=sp: states_fixed(v, s, vi, vp, sp)
        r = evaluate(kins_gt, kins_nl, fn, args.reps, args.null_reps,
                     args.seed + 100 * i)
        res["threshold_sweep"][name] = {"thresholds": [vi, vp, sp], **r}
        print(f"  {name:12s} GT drho {r['gt']['delta_rho']:+.3f} "
              f"CI {r['gt']['ci95']}  null {r['null_delta_rho']:+.3f}  "
              f"diff {r['gt_minus_null_point']:+.3f}", flush=True)

    # ---- hysteresis -----------------------------------------------------
    res["hysteresis"] = {}
    for i, m in enumerate((0.10, 0.20, 0.30)):
        fn = lambda v, s, m=m: states_hysteresis(v, s, m)
        r = evaluate(kins_gt, kins_nl, fn, args.reps, args.null_reps,
                     args.seed + 5000 + 100 * i)
        res["hysteresis"][f"margin_{int(m*100)}pct"] = {"margin": m, **r}
        print(f"  hysteresis {m:.0%}: GT drho {r['gt']['delta_rho']:+.3f} "
              f"CI {r['gt']['ci95']}  null {r['null_delta_rho']:+.3f}  "
              f"diff {r['gt_minus_null_point']:+.3f}", flush=True)

    # ---- window sweep for delta-rho ------------------------------------
    res["window_sweep"] = {}
    for i, w in enumerate((13, 51)):
        print(f"recomputing kinematics @ window {w} ...", flush=True)
        kg = load_kinematics(GT_DIR, w)
        kn = load_kinematics(NULL_DIR, w)
        fn = lambda v, s: states_fixed(v, s, *BASE)
        r = evaluate(kg, kn, fn, args.reps, args.null_reps,
                     args.seed + 9000 + 100 * i)
        res["window_sweep"][f"w{w}"] = r
        print(f"  w{w}: GT drho {r['gt']['delta_rho']:+.3f} CI {r['gt']['ci95']} "
              f" null {r['null_delta_rho']:+.3f}", flush=True)

    # ---- verdict --------------------------------------------------------
    all_cfg = ({k: v for k, v in res["threshold_sweep"].items()}
               | {k: v for k, v in res["hysteresis"].items()}
               | {k: v for k, v in res["window_sweep"].items()})
    deltas = {k: v["gt"]["delta_rho"] for k, v in all_cfg.items()}
    survives = all(v["gt"]["ci95"][1] < 0 and v["gt_minus_null_point"] < 0
                   for v in all_cfg.values())
    res["verdict"] = {
        "gt_delta_rho_range": [min(deltas.values()), max(deltas.values())],
        "all_ci_exclude_zero": bool(all(v["gt"]["ci95"][1] < 0
                                        for v in all_cfg.values())),
        "all_more_negative_than_null": bool(all(v["gt_minus_null_point"] < 0
                                                for v in all_cfg.values())),
        "survives_T05": bool(survives)}
    print("\nVERDICT:", json.dumps(res["verdict"], indent=2), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
