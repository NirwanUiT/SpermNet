"""Is the 'stayer fraction' -> DFI link a genuine new axis, or just % immotile?

per_cell_kinetics.py found a CROSS-COHORT-CONSISTENT secondary signal: participants
whose sperm populations contain more "stayer" cells (cells that hold one motility
state) have LOWER DNA fragmentation (orig20 partial rho=-0.55 p=.027; extra57
-0.45 p=.0005). It was a secondary (not the pre-registered primary, which was null),
so it is hypothesis-generating and must survive the confounds that killed the lipid
result before it can be taken seriously.

Decisive checks (run per cohort SEPARATELY -- never pooled across the two tracking
pipelines):
  1. STATE-RESOLVED: split stayers into stable-Progressive / stable-Non-prog /
     stable-Immotile. If the DFI link is carried by stable-IMMOTILE cells it is just
     immotility (known DFI correlate); if by stable-PROGRESSIVE cells it is a genuine
     'stable good-swimmer' axis.
  2. Control for STATIC CASA composition (frac_progressive, frac_immotile): does
     stayer_frac predict DFI BEYOND composition? (the new-axis test)
  3. Control for age / BMI / abstinence (the covariates that vanished the lipid effect).
  4. Consistency of sign/magnitude across the two independent cohorts.

Output: outputs/markov/stayer_dfi.json

Usage: python -m experiments.stayer_dfi [--min-seq 50] [--min-cells 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import compute_frame_states, STATES  # noqa: E402
from experiments.dfi_features import load_clinical  # noqa: E402
from experiments.per_cell_kinetics import cohort_files, partial_spearman  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
OUTDIR = ROOT / "outputs" / "markov"
DFI_COL = "DNA fragmentation index, DFI (%)"
PARTICIP = ROOT / "data" / "raw" / "visem_full_clinical" / "participant_related_data.csv"


def load_particip():
    df = pd.read_csv(PARTICIP, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False),
                              errors="coerce")
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID")


def stayer_features(tf, min_seq, max_tracks, rng):
    df = pd.read_csv(tf)
    if df.empty:
        return None
    tids = df["track_id"].unique()
    if max_tracks and len(tids) > max_tracks:
        tids = rng.choice(tids, size=max_tracks, replace=False)
        df = df[df["track_id"].isin(tids)]
    n = 0
    stay = 0
    stay_state = np.zeros(len(STATES))
    frame_comp = np.zeros(len(STATES))
    for _, tr in df.groupby("track_id"):
        if len(tr) < config.MIN_TRACK_LENGTH:
            continue
        tr = tr.sort_values("frame")
        seq = [S2I[s] for s in compute_frame_states(tr)]
        if len(seq) < min_seq:
            continue
        n += 1
        for s in seq:
            frame_comp[s] += 1
        nsw = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
        if nsw == 0:
            stay += 1
            stay_state[seq[0]] += 1
    if n < 1:
        return None
    comp = frame_comp / frame_comp.sum()
    return {"n_cells": n, "stayer_frac": stay / n,
            "stable_prog_frac": stay_state[0] / n,
            "stable_nonprog_frac": stay_state[1] / n,
            "stable_immo_frac": stay_state[2] / n,
            "frac_progressive": comp[0], "frac_nonprog": comp[1],
            "frac_immotile": comp[2]}


def build(cohort, min_seq, min_cells, max_tracks, seed, clin, part):
    rng = np.random.default_rng(seed)
    rows = []
    for pid, tf in cohort_files(cohort).items():
        f = stayer_features(tf, min_seq, max_tracks, rng)
        if f is None or f["n_cells"] < min_cells:
            continue
        if pid not in clin.index or not np.isfinite(clin.loc[pid, DFI_COL]):
            continue
        row = {"pid": pid, **f, "DFI": float(clin.loc[pid, DFI_COL])}
        if pid in part.index:
            row["age"] = part.loc[pid].get("Age (years)")
            row["bmi"] = part.loc[pid].get("Body mass index (kg/m²)")
            row["abst"] = part.loc[pid].get("Abstinence time(days)")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seq", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--max-tracks", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clin, part = load_clinical(), load_particip()
    res = {}
    for cohort in ("orig20", "extra57"):
        d = build(cohort, args.min_seq, args.min_cells, args.max_tracks, args.seed,
                  clin, part)
        print(f"=== {cohort}  (n={len(d)}) ===", flush=True)
        y = d["DFI"].values
        ncl = np.log10(d["n_cells"].values)

        def ps(x, *ctrls):
            return partial_spearman(x, y, *ctrls)

        # state-resolved stayer fractions vs DFI (control n_cells)
        sr = {}
        for k in ("stayer_frac", "stable_prog_frac", "stable_nonprog_frac",
                  "stable_immo_frac"):
            r, p = ps(d[k].values, ncl)
            sr[k] = {"rho": r, "p": p}
            print(f"  {k:20s} | n_cells   rho={r:+.3f} (p={p:.3f})")

        # new-axis test: stayer_frac | composition (+ n_cells)
        r_comp, p_comp = ps(d["stayer_frac"].values, ncl,
                            d["frac_immotile"].values, d["frac_progressive"].values)
        print(f"  stayer_frac | +composition       rho={r_comp:+.3f} (p={p_comp:.3f})")

        # + age/bmi/abstinence (only rows with all covariates)
        full = None
        m = d.dropna(subset=["age", "bmi", "abst"])
        if len(m) >= 12:
            r_full, p_full = partial_spearman(
                m["stayer_frac"].values, m["DFI"].values,
                np.log10(m["n_cells"].values), m["frac_immotile"].values,
                m["frac_progressive"].values, m["age"].values, m["bmi"].values,
                m["abst"].values)
            full = {"rho": r_full, "p": p_full, "n": int(len(m))}
            print(f"  stayer_frac | +comp+age+bmi+abst rho={r_full:+.3f} "
                  f"(p={p_full:.3f}, n={len(m)})")

        # reference: frac_immotile vs DFI (the known correlate)
        r_im, p_im = stats.spearmanr(d["frac_immotile"].values, y)
        print(f"  [ref] frac_immotile vs DFI        rho={r_im:+.3f} (p={p_im:.3f})")
        res[cohort] = {"n": len(d), "state_resolved": sr,
                       "stayer_given_composition": {"rho": r_comp, "p": p_comp},
                       "stayer_full_controls": full,
                       "ref_frac_immotile_DFI": {"rho": float(r_im), "p": float(p_im)}}
        print()

    json.dump(res, open(OUTDIR / "stayer_dfi.json", "w"), indent=2, default=float)
    print(f"saved -> {OUTDIR/'stayer_dfi.json'}")


if __name__ == "__main__":
    main()
