"""Physics of the motility memory (Aim B): what KIND of non-Markovian process?

Finding #1 showed dwell times are non-geometric (CV>1, heavy tails). "Not
exponential" is not yet a mechanism. Here we identify the DISTRIBUTIONAL FORM of
the state dwell times and the structure of the temporal memory, using the robust
within-video data (millions of switching events per cohort) — which, unlike the
cross-participant clinical correlations, is NOT subject to the cohort batch effect.

We ask, per state and per cohort (orig VISEM-Tracking vs our extra 57):
  (1) Which law best describes dwell times: exponential (Markov/memoryless),
      gamma, Weibull, log-normal, or power-law (scale-free)?  MLE + AIC.
  (2) Is the heavy tail scale-free? Estimate the tail exponent on the survival
      function beyond x_min (mitigates the sliding-window censoring of short dwells).
  (3) Does the FORM replicate across the two independently-tracked cohorts? A
      mechanism-level result should be cohort-invariant even if absolute dwell
      scales differ.

Output: outputs/markov/dwell_physics.{json,png}

Usage: python -m experiments.dwell_physics [--max-tracks 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import compute_frame_states, STATES  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"


def dwell_episodes(track_dir: Path, max_tracks: int, seed: int = 0):
    """Pooled dwell episodes (in seconds) per state from a track directory."""
    rng = np.random.default_rng(seed)
    dwell = {i: [] for i in range(len(STATES))}
    plain_only = track_dir == ORIG
    for tf in sorted(track_dir.glob("*_tracks.csv")):
        if plain_only:
            name = tf.stem.replace("_tracks", "")
            if any(name.endswith(s) for s in ("_botsort", "_bytetrack",
                                              "_ocsort", "_reid")):
                continue
        df = pd.read_csv(tf)
        if df.empty:
            continue
        tids = df["track_id"].unique()
        if max_tracks and len(tids) > max_tracks:
            tids = rng.choice(tids, size=max_tracks, replace=False)
            df = df[df["track_id"].isin(tids)]
        for _, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            tr = tr.sort_values("frame")
            seq = [S2I[s] for s in compute_frame_states(tr)]
            run = 1
            for k in range(1, len(seq)):
                if seq[k] == seq[k - 1]:
                    run += 1
                else:
                    dwell[seq[k - 1]].append(run)
                    run = 1
            dwell[seq[-1]].append(run)
    return {i: np.array(v, float) / config.FPS for i, v in dwell.items()}


def fit_laws(d: np.ndarray):
    """MLE fits + AIC for exponential/gamma/weibull/lognormal on dwell times."""
    d = d[d > 0]
    out = {}
    fits = {
        "exponential": (stats.expon, dict(floc=0)),
        "gamma": (stats.gamma, dict(floc=0)),
        "weibull": (stats.weibull_min, dict(floc=0)),
        "lognormal": (stats.lognorm, dict(floc=0)),
    }
    for name, (dist, kw) in fits.items():
        par = dist.fit(d, **kw)
        ll = np.sum(dist.logpdf(d, *par))
        k = len(par) - ("floc" in kw)  # free params (loc fixed)
        aic = 2 * k - 2 * ll
        out[name] = {"params": [float(p) for p in par], "loglik": float(ll),
                     "aic": float(aic), "k": int(k)}
    best = min(out, key=lambda n: out[n]["aic"])
    for n in out:
        out[n]["dAIC"] = out[n]["aic"] - out[best]["aic"]
    out["best"] = best
    return out


def tail_exponent(d: np.ndarray, xmin_q: float = 0.5):
    """Hill-style tail exponent on the survival function beyond x_min (quantile)."""
    d = np.sort(d[d > 0])
    xmin = np.quantile(d, xmin_q)
    tail = d[d >= xmin]
    if len(tail) < 50:
        return None
    # Hill estimator for P(X>x) ~ x^-alpha
    alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
    return {"xmin_s": float(xmin), "n_tail": int(len(tail)),
            "alpha_hill": float(alpha)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohorts = {"orig20": ORIG, "extra57": EXTRA}
    res = {}
    fig, axes = plt.subplots(len(STATES), 2, figsize=(11, 11))
    for col, (cname, cdir) in enumerate(cohorts.items()):
        print(f"\n########## cohort {cname} ({cdir.name}) ##########", flush=True)
        dw = dwell_episodes(cdir, args.max_tracks, args.seed)
        res[cname] = {}
        for i, sname in enumerate(STATES):
            d = dw[i]
            if len(d) < 200:
                continue
            cv = float(d.std() / d.mean())
            laws = fit_laws(d)
            tail = tail_exponent(d)
            res[cname][sname] = {"n": int(len(d)), "mean_s": float(d.mean()),
                                 "cv": cv, "laws": laws, "tail": tail}
            print(f"  {sname:16s} n={len(d):7d}  mean={d.mean():.3f}s  CV={cv:.2f}"
                  f"  best={laws['best']:11s}"
                  f"  (exp dAIC=+{laws['exponential']['dAIC']:.0f})"
                  + (f"  tail alpha={tail['alpha_hill']:.2f}" if tail else ""))
            # survival plot (log-log) for extra cohort column layout
            ax = axes[i, col]
            ds = np.sort(d)
            surv = 1.0 - np.arange(len(ds)) / len(ds)
            ax.loglog(ds, surv, ".", ms=2, alpha=0.4, label="empirical")
            xx = np.logspace(np.log10(ds[ds > 0][0]), np.log10(ds.max()), 100)
            ax.loglog(xx, stats.expon.sf(xx, *stats.expon.fit(d, floc=0)),
                      "r-", lw=1, label="exponential (Markov)")
            lg = laws["best"]
            distmap = {"gamma": stats.gamma, "weibull": stats.weibull_min,
                       "lognormal": stats.lognorm, "exponential": stats.expon}
            ax.loglog(xx, distmap[lg].sf(xx, *laws[lg]["params"]), "g-", lw=1,
                      label=f"best: {lg}")
            ax.set_title(f"{cname} — {sname}", fontsize=9)
            ax.set_xlabel("dwell (s)"); ax.set_ylabel("P(>t)")
            if i == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / "dwell_physics.png", dpi=140)

    import json
    json.dump(res, open(OUTDIR / "dwell_physics.json", "w"), indent=2, default=float)

    # cross-cohort form consistency
    print("\n=== FORM consistency across independently-tracked cohorts ===")
    for sname in STATES:
        b = [res[c].get(sname, {}).get("laws", {}).get("best")
             for c in cohorts if sname in res[c]]
        print(f"  {sname:16s} best-law by cohort: {b}")
    print(f"\nsaved -> {OUTDIR/'dwell_physics.json'}")
    print(f"saved -> {OUTDIR/'dwell_physics.png'}")


if __name__ == "__main__":
    main()
