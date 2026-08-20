"""T1.5: power analysis for the pre-registered DFI clinical primary.

The primary (per_cell_kinetics.py) is a partial Spearman of switch-rate CV vs DFI,
controlling for mean switch rate and track count, run per cohort (orig20 n=16,
extra57 n=57). Both were null. A reviewer will rightly ask: what effect size could
these cohorts have detected? This script answers by simulation, using the IDENTICAL
partial_spearman estimator.

Design: latent bivariate normal (x*, y*) with correlation rho_true; both variables
additionally load 0.3 on each of two control covariates (so the raw correlation is
confounded and the partial estimator must remove it, as in the real analysis).
Power = fraction of reps with two-sided p < 0.05. The minimum detectable effect
(MDE) at 80 % power is interpolated on the rho grid; the mean recovered partial
Spearman is reported per grid point so the latent-Pearson -> Spearman mapping is
explicit.

Output: outputs/markov/dfi_power.json

Usage: python -m experiments.dfi_power [--reps 4000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.per_cell_kinetics import partial_spearman  # noqa: E402

OUT = ROOT / "outputs" / "markov" / "dfi_power.json"

COHORTS = {"orig20": 16, "extra57": 57}
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CTRL_LOAD = 0.3
ALPHA = 0.05


def simulate_power(n, rho, reps, rng):
    hits = 0
    rho_hat = []
    for _ in range(reps):
        c1 = rng.standard_normal(n)
        c2 = rng.standard_normal(n)
        z = rng.standard_normal((n, 2))
        xs = z[:, 0]
        ys = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
        x = xs + CTRL_LOAD * (c1 + c2)
        y = ys + CTRL_LOAD * (c1 + c2)
        r, p = partial_spearman(x, y, c1, c2)
        rho_hat.append(r)
        if p < ALPHA:
            hits += 1
    return hits / reps, float(np.mean(rho_hat))


def mde_at(power_curve, target=0.80):
    """Linear interpolation of the rho grid at the target power."""
    for (r0, p0), (r1, p1) in zip(power_curve, power_curve[1:]):
        if p0 < target <= p1:
            return r0 + (target - p0) / (p1 - p0) * (r1 - r0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    res = {"reps": args.reps, "alpha": ALPHA, "ctrl_load": CTRL_LOAD,
           "note": ("power of the pre-registered primary (partial Spearman, "
                    "2 rank controls) by simulation with the identical estimator; "
                    "rho is the latent Pearson of the residual bivariate normal")}
    for name, n in COHORTS.items():
        curve = []
        print(f"{name} (n={n}):", flush=True)
        for rho in RHO_GRID:
            pw, rh = simulate_power(n, rho, args.reps, rng)
            curve.append({"rho_true": rho, "power": pw, "mean_partial_spearman": rh})
            print(f"  rho={rho:.1f}: power {pw:.3f}  (recovered rho_s {rh:+.3f})",
                  flush=True)
        mde = mde_at([(c["rho_true"], c["power"]) for c in curve])
        # power at the observed (null) point estimates, |rho|
        obs = {"orig20": 0.356, "extra57": 0.117}[name]
        pw_obs, _ = simulate_power(n, obs, args.reps, rng)
        res[name] = {"n": n, "curve": curve,
                     "mde_80pct_power": mde,
                     "power_at_observed_point_estimate": {"rho": obs, "power": pw_obs}}
        print(f"  -> MDE at 80% power: rho = {mde:.3f}" if mde else
              "  -> MDE at 80% power: beyond grid", flush=True)
        print(f"  -> power at observed |rho|={obs}: {pw_obs:.3f}", flush=True)

    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
