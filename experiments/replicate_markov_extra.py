"""Replicate the non-Markovian finding on the 57 INDEPENDENT extra participants.

Finding #1 (motility is non-Markovian / has 2nd-order memory) was established on
the 20 VISEM-Tracking participants. Here we re-run the same decisive tests on the
57 participants that were NOT part of VISEM-Tracking (tracks produced by our own
detector+tracker in outputs/tracks_extra). If the direction holds on a disjoint
cohort, the finding is a robust property of human sperm motility, not an artifact
of the 20-video annotation set.

Same machinery as markov_property_test.py (reused verbatim), just pointed at a
different track directory. Reports per-state dwell CV (geometric/Markov => ~1) and
held-out 5-fold per-token log-likelihood for Markov orders 0-3.

Usage: python -m experiments.replicate_markov_extra
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
from markov_property_test import (  # noqa: E402
    dwell_times, test_geometric, cv_order,
)
from events.detect_events import classify_window  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
EXTRA = ROOT / "outputs" / "tracks_extra"
ORIG = config.TRACK_OUT
OUTDIR = ROOT / "outputs" / "markov"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_sequences_from(track_dir: Path, max_tracks: int = 0,
                        seed: int = 0) -> list[list[int]]:
    """Per-track state sequences from a directory of {pid}_tracks.csv files.

    Tracks in tracks_extra carry a `clip` column; track_ids are already unique
    per participant (offset per clip), and every track lives within one clip, so
    grouping by track_id and sorting by frame yields a clean within-clip sequence.

    If max_tracks > 0, randomly subsample that many tracks per participant (the
    decisive Finding-#1 test used ~1371 tracks total, so a few-thousand-per-
    participant cap retains far more than enough statistical power while keeping
    the pure-python sliding-window classification tractable across 1.2M tracks).
    """
    rng = np.random.default_rng(seed)
    seqs: list[list[int]] = []
    for tf in sorted(track_dir.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        if df.empty:
            continue
        tids = df["track_id"].unique()
        if max_tracks and len(tids) > max_tracks:
            tids = rng.choice(tids, size=max_tracks, replace=False)
            df = df[df["track_id"].isin(tids)]
        for tid, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            tr = tr.sort_values("frame")
            seqs.append([S2I[s] for s in compute_frame_states(tr)])
    return seqs


def run_block(name: str, seqs: list[list[int]]) -> dict:
    nstates = sum(len(s) for s in seqs)
    print(f"\n########## {name}: {len(seqs)} tracks, {nstates} frame-states "
          f"(@{config.FPS} fps) ##########")

    print("\n=== (A) Dwell-time CV vs geometric (Markov => CV~1) ===")
    dwell = dwell_times(seqs)
    drows = []
    for i, sname in enumerate(STATES):
        if len(dwell[i]) < 30:
            continue
        r = test_geometric(dwell[i], sname)
        drows.append(r)
        print("  %-16s episodes=%7d  mean=%5.2f s  CV=%.2f  "
              "tail_excess=%.1fx  KS=%.3f"
              % (sname, r["n_episodes"], r["mean_dwell_s"], r["cv"],
                 r["tail_excess_x"], r["ks_vs_geometric"]))

    print("\n=== (B) Markov order: held-out logL/token (5-fold CV) ===")
    ll = cv_order(seqs)
    for order in sorted(ll):
        gain = ll[order] - ll[order - 1] if order > 0 else 0.0
        print("  order %d:  logL/token = % .4f   (delta vs prev = %+.4f)"
              % (order, ll[order], gain))
    g2 = ll[2] - ll[1]
    verdict = ("NON-MARKOVIAN (2nd-order memory present)" if g2 > 0.01
               else "first-order adequate")
    print(f"  => 2nd-over-1st gain = {g2:+.4f}/token  -> {verdict}")
    return {"cohort": name, "n_tracks": len(seqs), "n_states": nstates,
            "ll_order": ll, "g2_over_1": g2, "dwell": drows}


def nonoverlap_sequences_from(track_dir: Path, block: int = 25,
                              max_tracks: int = 0, seed: int = 0) -> list[list[int]]:
    """One state per non-overlapping `block`-frame window, per track.

    Decorrelates the sliding-window autocorrelation so the Markov-ORDER test is
    a fair apples-to-apples comparison with the decisive Finding-#1 test
    (hmm_vs_markov.py used 0.5 s = 25-frame blocks).
    """
    dt = 1.0 / config.FPS
    rng = np.random.default_rng(seed)
    seqs: list[list[int]] = []
    for tf in sorted(track_dir.glob("*_tracks.csv")):
        df = pd.read_csv(tf)
        if df.empty:
            continue
        tids = df["track_id"].unique()
        if max_tracks and len(tids) > max_tracks:
            tids = rng.choice(tids, size=max_tracks, replace=False)
            df = df[df["track_id"].isin(tids)]
        for tid, tr in df.groupby("track_id"):
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            tr = tr.sort_values("frame")
            xs = tr["cx"].to_numpy(float)
            ys = tr["cy"].to_numpy(float)
            seq = [S2I[classify_window(xs[i:i + block], ys[i:i + block], dt)]
                   for i in range(0, len(xs) - block + 1, block)]
            if len(seq) >= 3:
                seqs.append(seq)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=3000,
                    help="cap tracks/participant (random subsample); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading INDEPENDENT 57-participant cohort (outputs/tracks_extra) "
          f"[max_tracks={args.max_tracks}] ...", flush=True)
    extra = load_sequences_from(EXTRA, max_tracks=args.max_tracks, seed=args.seed)
    res_extra = run_block("EXTRA-57 (independent replication)", extra)

    # decorrelated-block order test (apples-to-apples with Finding #1 decisive test)
    print("\n=== (C) DECORRELATED 0.5s blocks: Markov order (held-out logL/token) ===",
          flush=True)
    blk = nonoverlap_sequences_from(EXTRA, block=25, max_tracks=args.max_tracks,
                                    seed=args.seed)
    ntok = sum(len(s) for s in blk)
    print(f"    {len(blk)} tracks, {ntok} block-states", flush=True)
    llb = cv_order(blk)
    for order in sorted(llb):
        gain = llb[order] - llb[order - 1] if order > 0 else 0.0
        print("  order %d:  logL/token = % .4f   (delta vs prev = %+.4f)"
              % (order, llb[order], gain))
    g2b = llb[2] - llb[1]
    verdictb = ("NON-MARKOVIAN (2nd-order memory present)" if g2b > 0.01
                else "first-order adequate")
    print(f"  => decorrelated 2nd-over-1st gain = {g2b:+.4f}/token  -> {verdictb}")
    res_extra["block_order_ll"] = llb
    res_extra["block_g2_over_1"] = g2b
    res_extra["n_block_states"] = ntok

    import json
    out = OUTDIR / "replication_extra.json"
    json.dump({k: (v if k != "dwell" else v) for k, v in res_extra.items()},
              open(out, "w"), indent=2, default=float)
    pd.DataFrame(res_extra["dwell"]).to_csv(
        OUTDIR / "dwell_memory_test_extra.csv", index=False)
    print(f"\nsaved -> {out}")
    print(f"saved -> {OUTDIR/'dwell_memory_test_extra.csv'}")


if __name__ == "__main__":
    main()
