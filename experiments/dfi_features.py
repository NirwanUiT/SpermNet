"""Per-participant feature extraction for the DFI prediction test (Tier 1).

For each participant with tracks we compute TWO feature blocks:

  CASA snapshot features (what a clinic / standard CASA reports — a static
  description of the population at one instant):
    mean & sd of VCL, VSL, VAP, LIN, STR, WOB, ALH, BCF across tracks;
    motility composition = fraction of tracks {progressive, non-prog, immotile}.

  Memory / dynamics features (our Finding #1 — how individual cells SWITCH
  state over time, beyond the static composition):
    self-transition (dwell) probabilities P(P->P), P(NP->NP), P(I->I);
    directional hysteresis asymmetries  P(i->j) - P(j->i)  for each pair;
    dwell-time mean and coefficient-of-variation per state (CV>1 => memory).

These are merged with the clinical DNA-fragmentation index (DFI) so we can ask:
do the *dynamics* features predict DFI better than the *static* CASA snapshot?

Participants come from two disjoint sources (kept identical in processing):
  outputs/tracks        -> 20 VISEM-Tracking participants (plain {id}_tracks.csv)
  outputs/tracks_extra  -> 57 extra participants (our detector+tracker)

Output: outputs/markov/dfi_features.csv  (one row per participant)

Usage: python -m experiments.dfi_features [--max-tracks 4000] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import compute_frame_states, STATES  # noqa: E402
from events.detect_events import compute_track_metrics  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
NS = len(STATES)
ORIG = config.TRACK_OUT
EXTRA = ROOT / "outputs" / "tracks_extra"
CLIN = ROOT / "data" / "raw" / "visem_full_clinical" / "semen_analysis_data.csv"
OUT = ROOT / "outputs" / "markov" / "dfi_features.csv"

CASA_KEYS = ["VCL", "VSL", "VAP", "LIN", "STR", "WOB", "ALH", "BCF"]


def load_clinical() -> pd.DataFrame:
    """Parse the semicolon/European-decimal clinical CSV -> numeric, ID-indexed."""
    df = pd.read_csv(CLIN, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = (df[c].str.strip().str.replace(",", ".", regex=False))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ID"] = df["ID"].astype(int)
    return df.set_index("ID")


def participant_files() -> dict[int, Path]:
    """pid -> track csv, preferring plain VISEM-Tracking files, then extras."""
    out: dict[int, Path] = {}
    for tf in sorted(EXTRA.glob("*_tracks.csv")):
        out[int(tf.stem.split("_")[0])] = tf
    for tf in sorted(ORIG.glob("*_tracks.csv")):
        name = tf.stem.replace("_tracks", "")
        if any(name.endswith(s) for s in ("_botsort", "_bytetrack", "_ocsort",
                                          "_botsort_reid")):
            continue
        out[int(name)] = tf            # plain file overrides extra if both exist
    return out


def transition_and_dwell(seqs: list[list[int]]):
    """1st-order transition matrix + dwell-time stats from state sequences."""
    counts = np.zeros((NS, NS))
    dwell = {i: [] for i in range(NS)}
    for seq in seqs:
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
        # dwell episodes
        run = 1
        for k in range(1, len(seq)):
            if seq[k] == seq[k - 1]:
                run += 1
            else:
                dwell[seq[k - 1]].append(run)
                run = 1
        dwell[seq[-1]].append(run)
    P = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    return P, dwell


def features_for(pid: int, tf: Path, rng, max_tracks: int) -> dict | None:
    df = pd.read_csv(tf)
    if df.empty:
        return None
    tids = df["track_id"].unique()
    if len(tids) > max_tracks:
        tids = rng.choice(tids, size=max_tracks, replace=False)
    sub = df[df["track_id"].isin(tids)]

    casa_rows = []
    seqs = []
    comp = np.zeros(NS)        # frame-weighted state composition
    for _, tr in sub.groupby("track_id"):
        if len(tr) < config.MIN_TRACK_LENGTH:
            continue
        tr = tr.sort_values("frame")
        m = compute_track_metrics(tr)
        if m is not None:
            casa_rows.append(m)
        st = [S2I[s] for s in compute_frame_states(tr)]
        seqs.append(st)
        for s in st:
            comp[s] += 1
    if not casa_rows or comp.sum() == 0:
        return None

    casa = pd.DataFrame(casa_rows)
    feat: dict[str, float] = {"pid": pid, "n_tracks": int(len(seqs)),
                              "n_frames": int(comp.sum())}
    # CASA snapshot: distribution moments
    for k in CASA_KEYS:
        feat[f"{k}_mean"] = float(casa[k].mean())
        feat[f"{k}_sd"] = float(casa[k].std())
    comp = comp / comp.sum()
    feat["frac_progressive"] = float(comp[S2I["Progressive"]])
    feat["frac_nonprog"] = float(comp[S2I["Non-progressive"]])
    feat["frac_immotile"] = float(comp[S2I["Immotile"]])

    # Memory / dynamics
    P, dwell = transition_and_dwell(seqs)
    feat["P_stay_prog"] = float(P[S2I["Progressive"], S2I["Progressive"]])
    feat["P_stay_nonprog"] = float(P[S2I["Non-progressive"], S2I["Non-progressive"]])
    feat["P_stay_immotile"] = float(P[S2I["Immotile"], S2I["Immotile"]])
    pairs = [("Progressive", "Non-progressive"), ("Non-progressive", "Immotile"),
             ("Progressive", "Immotile")]
    for a, b in pairs:
        ia, ib = S2I[a], S2I[b]
        feat[f"asym_{a[:4]}_{b[:4]}"] = float(P[ia, ib] - P[ib, ia])
    for i, sname in enumerate(STATES):
        d = np.array(dwell[i], dtype=float)
        if len(d) >= 10:
            feat[f"dwell_mean_{sname[:4]}"] = float(d.mean() / config.FPS)
            feat[f"dwell_cv_{sname[:4]}"] = float(d.std() / d.mean()) if d.mean() > 0 else np.nan
        else:
            feat[f"dwell_mean_{sname[:4]}"] = np.nan
            feat[f"dwell_cv_{sname[:4]}"] = np.nan
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=4000,
                    help="cap tracks/participant for tractability (random subsample)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    clin = load_clinical()
    files = participant_files()
    print(f"{len(files)} participants with tracks; {len(clin)} clinical records")

    rows = []
    for pid in sorted(files):
        if pid not in clin.index:
            print(f"  pid {pid}: no clinical record, skip")
            continue
        dfi = clin.loc[pid, "DNA fragmentation index, DFI (%)"]
        f = features_for(pid, files[pid], rng, args.max_tracks)
        if f is None:
            print(f"  pid {pid}: no usable tracks, skip")
            continue
        f["DFI"] = float(dfi)
        # carry the clinically-measured motility for sanity comparison
        f["clin_prog_motility"] = float(clin.loc[pid, "Progressive motility (%)"])
        f["source"] = "orig" if files[pid].parent == ORIG else "extra"
        rows.append(f)
        print(f"  pid {pid:3d}: {f['n_tracks']:6d} tracks  DFI={dfi:.1f}  "
              f"frac_prog={f['frac_progressive']:.2f}")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nsaved {len(out)} participants x {out.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
