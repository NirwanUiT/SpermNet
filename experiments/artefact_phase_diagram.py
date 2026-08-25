"""T1.1 (referee U1) -- Artefact phase diagram: where does a windowed threshold
classifier manufacture event dynamics from a memoryless continuum?

Synthetic cohorts of 2-D AR(1)/OU velocity tracks (Markovian by construction,
no switching, no dynamics) swept over three axes:

  1. W/tau  -- classifier window (fixed W = 25 frames) over velocity
               relaxation time tau (frames); tau grid gives W/tau 0.5 .. 25.
  2. threshold placement -- (immotile, progressive) cuts at percentiles of
               each config's own windowed-VCL marginal, so occupancy is
               controlled and only geometry varies.
  3. trait dispersion -- per-track lognormal amplitude multipliers with
               sigma_disp in {0, 0.3, 0.6}; 0 = homogeneous cohort.
               (The aggregation axis: predicted to control block g2.)

Classifier: state per frame from the sliding 25-frame VCL (I below t_lo, P
above t_hi, NP between) -- the 1-D core of the CASA rule, isolating the
mechanism. Readouts per grid point, computed with the paper's own machinery:
block g2 (cv_order on non-overlapping W-frame block states), per-episode
dwell dAIC(exp vs best) and dwell CV per state (frame-level episodes), and
switch rate.

Output: outputs/markov/artefact_phase_diagram.json
Usage:  python -m experiments.artefact_phase_diagram
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from experiments.dwell_physics import fit_laws  # noqa: E402
from markov_property_test import cv_order  # noqa: E402

OUT = config.MARKOV_OUT / "artefact_phase_diagram.json"

N_TRACKS = 600
N_FRAMES = 1500
W = 25
TAU_GRID = (1.0, 2.0, 3.3, 6.0, 12.0, 25.0, 50.0)     # frames
PCT_GRID = ((10, 60), (25, 75), (40, 90))             # (immotile, progressive)
DISP_GRID = (0.0, 0.3, 0.6)                           # lognormal sigma
SEED = 0


def simulate_wvcl(tau: float, disp: float, rng: np.random.Generator) -> np.ndarray:
    """(N_TRACKS, n_windows) sliding-window mean speed for the cohort."""
    a = np.exp(-1.0 / tau)
    sd_innov = np.sqrt(1.0 - a * a)
    amp = np.exp(rng.normal(0.0, disp, N_TRACKS)) if disp > 0 else np.ones(N_TRACKS)
    kern = np.ones(W - 1) / (W - 1)
    out = np.empty((N_TRACKS, N_FRAMES - 1 - (W - 2)))
    for i in range(N_TRACKS):
        eps = rng.normal(0.0, 1.0, size=(N_FRAMES - 1, 2)) * sd_innov
        v = np.empty((N_FRAMES - 1, 2))
        v[0] = rng.normal(0.0, 1.0, 2)
        for t in range(1, N_FRAMES - 1):
            v[t] = a * v[t - 1] + eps[t]
        speed = amp[i] * np.hypot(v[:, 0], v[:, 1])
        out[i] = np.convolve(speed, kern, mode="valid")
    return out


def states_from(wvcl: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    s = np.ones_like(wvcl, dtype=np.int8)          # NP
    s[wvcl < t_lo] = 2                             # I
    s[wvcl > t_hi] = 0                             # P
    return s


def dwell_stats(states: np.ndarray) -> dict:
    eps: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for row in states:
        # interior episodes only (censoring-free readout)
        change = np.flatnonzero(np.diff(row) != 0)
        if len(change) < 2:
            continue
        starts = change[:-1] + 1
        ends = change[1:] + 1
        for s0, e0 in zip(starts, ends):
            eps[int(row[s0])].append(e0 - s0)
    out = {}
    for st, name in ((0, "P"), (1, "NP"), (2, "I")):
        d = np.array(eps[st], float) / config.FPS
        if len(d) < 100:
            out[name] = {"n": int(len(d))}
            continue
        f = fit_laws(d)
        out[name] = {"n": int(len(d)), "best": f["best"],
                     "dAIC_exp_per_episode": float(f["exponential"]["dAIC"] / len(d)),
                     "cv": float(d.std() / d.mean())}
    return out


def block_g2(states: np.ndarray) -> float:
    seqs = []
    for row in states:
        blk = row[::W]                 # state at non-overlapping window starts
        if len(blk) >= 3:
            seqs.append(blk.tolist())
    ll = cv_order(seqs, orders=(1, 2), folds=5)
    return float(ll[2] - ll[1])


def main() -> None:
    results = []
    for tau in TAU_GRID:
        rng = np.random.default_rng(SEED + int(tau * 10))
        for disp in DISP_GRID:
            wvcl = simulate_wvcl(tau, disp, np.random.default_rng(
                SEED + int(tau * 10) * 100 + int(disp * 10)))
            flat = wvcl.ravel()
            for p_lo, p_hi in PCT_GRID:
                t_lo, t_hi = np.percentile(flat, [p_lo, p_hi])
                st = states_from(wvcl, t_lo, t_hi)
                sw = float(np.mean(np.diff(st, axis=1) != 0)) * config.FPS
                g2 = block_g2(st)
                dw = dwell_stats(st)
                row = {"tau_frames": tau, "W_over_tau": W / tau,
                       "disp_sigma": disp, "pct": [p_lo, p_hi],
                       "switch_rate_per_s": sw, "block_g2": g2,
                       "dwell": dw}
                results.append(row)
                dnp = dw.get("NP", {})
                print(f"tau={tau:5.1f} (W/tau {W/tau:5.2f}) disp={disp:.1f} "
                      f"pct={p_lo}/{p_hi}  g2={g2:+.4f}  sw={sw:5.2f}/s  "
                      f"NP dAIC/ep={dnp.get('dAIC_exp_per_episode', float('nan')):.3f} "
                      f"CV={dnp.get('cv', float('nan')):.2f}", flush=True)

    out = {"design": ("memoryless 2-D AR(1) velocity cohorts, "
                      f"{N_TRACKS} tracks x {N_FRAMES} frames, W={W}; "
                      "1-D two-threshold windowed-VCL classifier; thresholds "
                      "at percentiles of each config's own wVCL marginal"),
           "axes": {"tau_frames": list(TAU_GRID), "pct": [list(p) for p in PCT_GRID],
                    "disp_sigma": list(DISP_GRID)},
           "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
