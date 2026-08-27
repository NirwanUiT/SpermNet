"""Referee item D (part 2): place the ground truth on the phase diagram's INPUT
axes, not its outcome axis.

Two inputs are measured directly from the ground-truth trajectories, independent
of any event statistic:
  * W/tau  -- classifier window W=25 frames over the velocity relaxation time
              tau, estimated from the pooled lag-1 autocorrelation of the
              per-track centroid velocity (a = exp(-1/tau)).
  * sigma_hat -- amplitude dispersion. The pathological statistic SD(log RMS)
              is dominated by the immotile zero-speed tail; instead we use the
              between-track variance ratio on the classifier's OWN windowed-VCL
              surface, eta^2 = Var(per-track mean wVCL) / Var(all wVCL), which
              is bounded in [0,1], scale-free, and robust to the zero tail. The
              same eta^2 is measured from homogeneous-through-dispersed
              synthetic cohorts simulated at the fitted tau; inverting eta^2_GT
              onto that monotone curve gives sigma_hat on the sim's disp axis.

With (W/tau, sigma_hat) fixed by measurement, the phase diagram cell predicts
BOTH block g2 and NP dwell dAIC/episode; the observed ground-truth values are
then a two-for-one prediction, not a fit.

Output: outputs/markov/gt_amplitude_dispersion.json
Run:    python -m experiments.gt_amplitude_dispersion
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from experiments.artefact_phase_diagram import W, simulate_wvcl  # noqa: E402

GT_DIR = ROOT / "outputs" / "tracks_gt"
OUT = config.MARKOV_OUT / "gt_amplitude_dispersion.json"
MIN_FRAMES = 50
SIGMA_GRID = np.round(np.arange(0.0, 1.31, 0.1), 2)


def gt_track_stats():
    """Per track: wVCL window array (classifier surface), lag-1 velocity autocorr."""
    k = W - 1
    groups, ac = [], []
    for f in sorted(GT_DIR.glob("*_tracks.csv")):
        df = pd.read_csv(f)
        for _, g in df.groupby("track_id"):
            g = g.sort_values("frame")
            x = g["cx"].to_numpy(float)
            y = g["cy"].to_numpy(float)
            if len(x) < MIN_FRAMES:
                continue
            vx = np.diff(x)
            vy = np.diff(y)
            speed = np.hypot(vx, vy)
            if len(speed) <= k:
                continue
            csum = np.concatenate([[0.0], np.cumsum(speed)])
            wvcl = (csum[k:] - csum[:-k]) / k          # boxcar, classifier surface
            groups.append(wvcl)
            num = np.sum(vx[1:] * vx[:-1]) + np.sum(vy[1:] * vy[:-1])
            den = np.sum(vx * vx) + np.sum(vy * vy)
            ac.append(num / den if den > 0 else 0.0)
    return groups, np.array(ac)


def eta2_from_groups(groups: list[np.ndarray]) -> float:
    """ANOVA eta^2: length-weighted between-group SS / total SS, in [0,1]."""
    pooled = np.concatenate(groups)
    grand = pooled.mean()
    ssb = float(np.sum([len(g) * (g.mean() - grand) ** 2 for g in groups]))
    sst = float(np.sum((pooled - grand) ** 2))
    return ssb / sst


def sim_eta2(tau: float, disp: float, rng: np.random.Generator) -> float:
    wvcl = simulate_wvcl(tau, disp, rng)           # (N_TRACKS, n_windows)
    return eta2_from_groups([row for row in wvcl])


def main():
    groups, ac = gt_track_stats()
    n_tracks = len(groups)
    a_hat = float(np.median(ac))
    a_hat = min(max(a_hat, 1e-3), 0.999)
    tau_hat = float(-1.0 / np.log(a_hat))
    w_over_tau = W / tau_hat

    eta_gt = eta2_from_groups(groups)
    # sim eta^2(sigma) at fitted tau (3 seeds averaged), monotone increasing
    curve = []
    for s in SIGMA_GRID:
        vals = [sim_eta2(tau_hat, float(s), np.random.default_rng(100 + j))
                for j in range(3)]
        curve.append(float(np.mean(vals)))
    curve = np.array(curve)
    # invert eta_gt onto the curve
    if eta_gt <= curve[0]:
        sigma_hat = 0.0
    elif eta_gt >= curve[-1]:
        sigma_hat = float(SIGMA_GRID[-1])
    else:
        sigma_hat = float(np.interp(eta_gt, curve, SIGMA_GRID))

    out = {
        "n_tracks": n_tracks,
        "lag1_autocorr_median": a_hat,
        "tau_frames": tau_hat,
        "W_over_tau": w_over_tau,
        "eta2_gt": eta_gt,
        "sigma_grid": [float(s) for s in SIGMA_GRID],
        "eta2_sim_curve": [float(c) for c in curve],
        "sigma_hat": sigma_hat,
        "note": ("sigma_hat = disp axis value whose synthetic cohort reproduces "
                 "the GT between-track wVCL variance ratio at the fitted tau; "
                 "W_over_tau on the phase diagram's W/tau axis. Both inputs are "
                 "measured from GT trajectories, not event statistics. NOTE: "
                 "centroid-velocity autocorr is attenuated by tracking jitter, "
                 "so tau_hat is a lower bound and W/tau an upper bound."),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"GT tracks (>= {MIN_FRAMES} frames): {n_tracks}")
    print(f"lag-1 velocity autocorr (median): {a_hat:.4f}  "
          f"-> tau_hat = {tau_hat:.2f} frames, W/tau = {w_over_tau:.2f}")
    print(f"eta^2 GT (between-track wVCL variance ratio) = {eta_gt:.3f}")
    print("sim eta^2(sigma) at tau_hat:")
    for s, c in zip(SIGMA_GRID, curve):
        print(f"    sigma={s:.1f}  eta^2={c:.3f}")
    print(f"sigma_hat (inverted onto sim disp axis) = {sigma_hat:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

