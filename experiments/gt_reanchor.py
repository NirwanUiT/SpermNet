"""Re-anchor every headline statistic on the TRUE hand annotations.

PROVENANCE FINDING (2026-08-18): the plain orig20 track files
(outputs/tracks/{vid}_tracks.csv) are byte-identical to the BoT-SORT+ReID
variant -- run_single_gt.py's ground-truth export was silently overwritten by a
tracker run on 2026-04-27, BEFORE any of the paper's analyses were executed. So
every "hand-annotated orig20" number in the manuscript was actually computed on
automated tracker output, which the fidelity audit (tracker_fidelity.py) shows
inflates the memory statistic ~2x and the dwell dispersion substantially.

This script is the correction. It (1) materialises a clean, separate GT track
directory (outputs/tracks_gt/, NEVER touching outputs/tracks/) directly from the
VISEM labels_ftid annotation files, and (2) re-runs the paper's headline
analyses on it by importing the EXACT functions used originally -- nothing is
reimplemented, so any change in the numbers is attributable to the data alone:

    - dwell-law competition (exp/gamma/weibull/lognormal MLE + AIC) per state
      [experiments.dwell_physics.fit_laws / dwell_episodes]
    - geometric-dwell dispersion test (CV, tail excess)
      [markov_property_test.test_geometric / dwell_times]
    - frame-level Markov-order test (held-out logL/token, 5-fold)
      [markov_property_test.cv_order]
    - decorrelated 0.5s-block kinematic-reclassify order test = the headline g2
      [experiments.replicate_markov_extra.nonoverlap_sequences_from + cv_order]
    - window-robustness of the block g2 (13/25/51-frame classification windows)
    - EB (Dirichlet-multinomial) mover-stayer decomposition of the block g2
      [experiments.mover_stayer_eb.analyse]
    - within-cell vs pooled dwell CV, ICC, state-controlled serial correlation
      [experiments.memory_decomposition.analyse]

For each statistic we also report the automated-baseline (outputs/tracks/) value
computed identically, so the tracking-induced bias is explicit.

Output: outputs/markov/gt_reanchor.json

Usage: python -m experiments.gt_reanchor [--skip-materialise]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES, compute_frame_states  # noqa: E402
from markov_property_test import cv_order, dwell_times, test_geometric  # noqa: E402
from events.detect_events import classify_window  # noqa: E402
from experiments.dwell_physics import dwell_episodes, fit_laws  # noqa: E402
from experiments.replicate_markov_extra import (  # noqa: E402
    load_sequences_from,
    nonoverlap_sequences_from,
)
from experiments import mover_stayer_eb  # noqa: E402
from experiments import memory_decomposition  # noqa: E402

S2I = {s: i for i, s in enumerate(STATES)}
GT_DIR = ROOT / "outputs" / "tracks_gt"
BASELINE_DIR = config.TRACK_OUT  # botsort_reid output (provenance-verified)
OUT = config.MARKOV_OUT / "gt_reanchor.json"

VIDEOS = ["11", "12", "13", "14", "15", "19", "21", "22", "23", "24",
          "29", "30", "35", "36", "38", "47", "52", "54", "60", "82"]


# ----------------------------------------------------------- GT materialisation
def materialise_gt() -> None:
    """Write outputs/tracks_gt/{vid}_tracks.csv from labels_ftid annotations."""
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for vid in VIDEOS:
        dst = GT_DIR / f"{vid}_tracks.csv"
        ftid_dir = config.VISEM_ROOT / vid / "labels_ftid"
        images_dir = config.VISEM_ROOT / vid / "images"
        sample_img = next(images_dir.glob("*.jpg"), None)
        if not ftid_dir.exists() or sample_img is None:
            print(f"  {vid}: MISSING annotations, skipped")
            continue
        img = cv2.imread(str(sample_img))
        img_h, img_w = img.shape[:2]
        fid_to_int: dict[str, int] = {}
        next_id = 1
        rows = []
        for fpath in sorted(ftid_dir.glob("*_with_ftid.txt")):
            parts = fpath.stem.replace("_with_ftid", "").split("_frame_")
            if len(parts) < 2:
                continue
            frame_num = int(parts[1])
            with open(fpath) as f:
                for line in f:
                    tok = line.split()
                    if len(tok) < 6:
                        continue
                    fid = tok[0]
                    cx = float(tok[2]) * img_w
                    cy = float(tok[3]) * img_h
                    bw = float(tok[4]) * img_w
                    bh = float(tok[5]) * img_h
                    if fid not in fid_to_int:
                        fid_to_int[fid] = next_id
                        next_id += 1
                    rows.append({
                        "track_id": fid_to_int[fid], "frame": frame_num,
                        "cx": cx, "cy": cy,
                        "x1": cx - bw / 2, "y1": cy - bh / 2,
                        "x2": cx + bw / 2, "y2": cy + bh / 2,
                        "conf": 1.0,
                    })
        df = (pd.DataFrame(rows)
              .sort_values(["track_id", "frame"]).reset_index(drop=True))
        df.to_csv(dst, index=False)
        print(f"  {vid}: {df['track_id'].nunique()} GT tracks -> {dst.name}")


# ------------------------------------------------------------------- analyses
def dwell_law_block(track_dir: Path) -> dict:
    """Per-state 4-law AIC competition + geometric dispersion test."""
    dw_s = dwell_episodes(track_dir, max_tracks=0)
    laws = {}
    for i, s in enumerate(STATES):
        d = dw_s[i]
        if len(d) < 100:
            laws[s] = {"n": int(len(d)), "note": "too few episodes"}
            continue
        f = fit_laws(d)
        laws[s] = {
            "n": int(len(d)),
            "best": f["best"],
            "dAIC_exp_vs_best": f["exponential"]["dAIC"],
            "dAIC": {k: f[k]["dAIC"] for k in
                     ("exponential", "gamma", "weibull", "lognormal")},
        }
    return laws


def geometric_block(seqs) -> list[dict]:
    dw = dwell_times(seqs)
    rows = []
    for i, s in enumerate(STATES):
        if len(dw[i]) < 30:
            continue
        rows.append(test_geometric(dw[i], s))
    return rows


def order_block(seqs) -> dict:
    ll = cv_order(seqs, orders=(0, 1, 2, 3), folds=5)
    return {"ll": {str(k): v for k, v in ll.items()},
            "g2": ll[2] - ll[1], "g3": ll[3] - ll[2]}


def block_g2_at_window(track_dir: Path, block: int) -> dict:
    seqs = [s for s in nonoverlap_sequences_from(track_dir, block=block)
            if len(s) >= 3]
    if len(seqs) < 10:
        return {"block": block, "g2": None, "n_tracks": len(seqs)}
    ll = cv_order(seqs, orders=(0, 1, 2), folds=5)
    return {"block": block, "g2": ll[2] - ll[1],
            "n_tracks": len(seqs),
            "n_block_states": int(sum(len(s) for s in seqs))}


def analyse_dir(track_dir: Path, name: str) -> dict:
    print(f"\n=== {name} ({track_dir}) ===", flush=True)
    t0 = time.time()

    seqs = load_sequences_from(track_dir)
    print(f"  {len(seqs)} tracks, {sum(len(s) for s in seqs)} frame-states")

    out: dict = {"name": name, "n_tracks": len(seqs),
                 "n_frame_states": int(sum(len(s) for s in seqs))}

    print("  dwell-law competition ...", flush=True)
    out["dwell_laws"] = dwell_law_block(track_dir)
    for s in STATES:
        r = out["dwell_laws"][s]
        if "best" in r:
            print(f"    {s:16s} n={r['n']:6d}  best={r['best']:9s}  "
                  f"exp dAIC=+{r['dAIC_exp_vs_best']:.0f}")

    print("  geometric dispersion ...", flush=True)
    out["geometric"] = geometric_block(seqs)
    for r in out["geometric"]:
        print(f"    {r['state']:16s} mean={r['mean_dwell_s']:6.2f}s  CV={r['cv']:.2f}  "
              f"tail_excess={r['tail_excess_x']:.1f}x")

    print("  frame-level order test ...", flush=True)
    out["order_frame"] = order_block(seqs)
    print(f"    frame-level g2 = {out['order_frame']['g2']:+.4f}")

    print("  block kinematic-reclassify g2 (windows 13/25/51) ...", flush=True)
    out["block_g2"] = {str(b): block_g2_at_window(track_dir, b)
                       for b in (13, 25, 51)}
    for b, r in out["block_g2"].items():
        print(f"    window {b:>3s}f: g2 = "
              f"{r['g2']:+.4f}  ({r.get('n_block_states', 0)} block-states)")

    print("  EB mover-stayer decomposition ...", flush=True)
    out["eb"] = mover_stayer_eb.analyse(track_dir, max_tracks=0, seed=0)
    g = out["eb"]["g2"]

    def _f(x):
        return f"{x:+.4f}" if x is not None else "n/a"
    share = out["eb"].get("genuine_memory_share_eb")
    print(f"    g2: REAL {_f(g['real'])} | EB {_f(g['eb_fair'])} | "
          f"HOM {_f(g['hom'])}  -> genuine-memory share "
          f"{f'{share:.0%}' if share is not None else 'n/a'}")

    print("  within-cell vs pooled decomposition ...", flush=True)
    out["memory_decomposition"] = memory_decomposition.analyse(
        track_dir, max_tracks=0, seed=0)
    md = out["memory_decomposition"]
    print(f"    ICC = {md['icc_logdwell']['icc']:.3f}  "
          f"serial rho = {md['serial_state_controlled']['lag1_spearman']:+.3f}")

    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-materialise", action="store_true")
    args = ap.parse_args()

    if not args.skip_materialise:
        print("Materialising GT tracks from labels_ftid ...")
        materialise_gt()

    res = {
        "provenance_note": (
            "outputs/tracks/{vid}_tracks.csv verified byte-identical to the "
            "_botsort_reid variant (mtime 2026-04-27): the GT export was "
            "overwritten by a tracker run before any analyses. outputs/tracks_gt/ "
            "is the true hand annotation, rebuilt from labels_ftid."),
        "gt": analyse_dir(GT_DIR, "orig20-GT (hand annotations)"),
        "baseline": analyse_dir(BASELINE_DIR, "orig20-baseline (botsort_reid)"),
    }

    # headline inflation ratios, explicit
    def _ratio(fa, fb):
        return (fa / fb) if (fa is not None and fb not in (None, 0)) else None
    g2_gt = res["gt"]["block_g2"]["25"]["g2"]
    g2_bl = res["baseline"]["block_g2"]["25"]["g2"]
    res["inflation"] = {
        "block25_g2_gt": g2_gt, "block25_g2_baseline": g2_bl,
        "g2_inflation_x": _ratio(g2_bl, g2_gt),
        "note": "baseline/gt ratio of the headline decorrelated-block g2",
    }
    print(f"\nHEADLINE: GT block-g2 = {g2_gt:+.4f} | baseline = {g2_bl:+.4f} "
          f"| inflation = {res['inflation']['g2_inflation_x']:.2f}x")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
