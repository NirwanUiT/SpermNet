#!/usr/bin/env python3
"""Windowing-confound control for the non-Markovian motility result.

The per-frame states in markov_property_test.py come from a 25-frame OVERLAPPING
sliding window, which manufactures short-range autocorrelation and could fake
both heavy dwell tails and second-order "memory". Here we rebuild the state
sequence from NON-OVERLAPPING windows (each frame-block classified once, blocks
share no frames) and re-run the identical memory tests. Surviving memory is real.

We sweep the block size so the conclusion is not tied to one window length.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from events.detect_events import classify_window  # noqa: E402
from markov_property_test import (  # noqa: E402
    dwell_times, test_geometric, cv_order, STATES,
)

TRACKS = config.TRACK_OUT
S2I = {s: i for i, s in enumerate(STATES)}


def nonoverlap_sequences(block: int) -> list[list[int]]:
    """One state per non-overlapping block of `block` frames, per track."""
    dt = 1.0 / config.FPS
    seqs = []
    for tf in sorted(TRACKS.glob("*_tracks.csv")):
        name = tf.stem.replace("_tracks", "")
        if any(name.endswith(s) for s in ("_botsort", "_bytetrack", "_ocsort",
                                          "_reid")):
            continue
        df = pd.read_csv(tf)
        if df.empty:
            continue
        for tid in df["track_id"].unique():
            tr = df[df["track_id"] == tid].sort_values("frame")
            if len(tr) < config.MIN_TRACK_LENGTH:
                continue
            xs = tr["cx"].to_numpy(float)
            ys = tr["cy"].to_numpy(float)
            seq = []
            for i in range(0, len(xs) - block + 1, block):
                s = classify_window(xs[i:i + block], ys[i:i + block], dt)
                seq.append(S2I[s])
            if len(seq) >= 3:
                seqs.append(seq)
    return seqs


def main():
    print("Windowing-confound control: non-overlapping blocks\n")
    for block in (25, 15, 50):
        seqs = nonoverlap_sequences(block)
        ntok = sum(len(s) for s in seqs)
        print(f"=== block = {block} frames ({block/config.FPS:.2f} s); "
              f"{len(seqs)} tracks, {ntok} block-states ===")

        # (A) dwell tails (in block units)
        dwell = dwell_times(seqs)
        for i, name in enumerate(STATES):
            if len(dwell[i]) < 30:
                continue
            r = test_geometric(dwell[i], name)
            print("   %-16s CV=%.2f  tail_excess=%.1fx  KS=%.3f  (n=%d)"
                  % (name, r["cv"], r["tail_excess_x"], r["ks_vs_geometric"],
                     r["n_episodes"]))

        # (B) Markov order on decorrelated sequence
        ll = cv_order(seqs, orders=(0, 1, 2))
        g1 = ll[1] - ll[0]
        g2 = ll[2] - ll[1]
        verdict = ("MEMORY survives (non-Markovian)" if g2 > 0.01
                   else "collapses to first-order (was a window artifact)")
        print("   order logL/token: 0=%.4f 1=%.4f 2=%.4f | "
              "1st-gain=+%.4f 2nd-gain=+%.4f -> %s\n"
              % (ll[0], ll[1], ll[2], g1, g2, verdict))


if __name__ == "__main__":
    main()
