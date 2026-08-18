"""Does tracking-association quality predict DOWNSTREAM DYNAMICAL fidelity?

Standard multi-object-tracking (MOT) metrics -- MOTA, IDF1, ID switches --
score how well predicted identities match a per-frame ground truth. But when
tracking is a *means* to a dynamical inference (dwell-time laws, second-order
memory, mover-stayer structure), the question that actually matters is whether
the recovered trajectories reproduce the correct *dynamics*, not whether every
identity is associated correctly. Those are not the same thing, and nobody
checks the second.

This script runs the check on the 20 hand-annotated VISEM-Tracking videos, for
which we have both ground-truth feature-ID trajectories (labels_ftid) and four
automated detector+tracker pipelines that share a detector but differ in the
association stage:

    default   : the frozen-baseline tracker (outputs/tracks/{v}_tracks.csv)
    botsort   : BoT-SORT
    bytetrack : ByteTrack
    botsort_reid : BoT-SORT + appearance re-identification

For every pipeline we compute, against the SAME ground truth and with the SAME
scoring code used in the main paper:

    ASSOCIATION quality (vs GT):  coverage (recall of GT trajectories),
        ID switches per GT track, fragmentation rate.
    DYNAMICAL fidelity (vs GT dynamics): the block-decorrelated second-order
        memory g2, per-state dwell coefficient of variation, and the
        log-normal-vs-exponential dwell-law delta-AIC.

We then ask two honest questions:
    (i)  ROBUSTNESS -- do the paper's qualitative conclusions (heavy-tailed
         dwells, g2 > 0) survive under every tracker AND under ground truth?
    (ii) DISSOCIATION -- across pipelines, does better association (higher
         coverage / fewer switches) actually mean dynamics closer to GT, or do
         the MOT metrics fail to predict dynamical fidelity?

Nothing here is tuned. Ground truth is read fresh from the annotation files and
is NEVER written back to outputs/tracks. Output: outputs/markov/tracker_fidelity.json

Usage:
    python -m experiments.tracker_fidelity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_analysis import STATES  # noqa: E402
# Reuse the EXACT scoring code used on the real/simulated sequences in the paper.
from experiments.generative_model import (  # noqa: E402
    S2I,
    NS,
    summary_stats,
    lognorm_vs_exp_aic,
    compute_frame_states,
)

VIDEOS = ["11", "12", "13", "14", "15", "19", "21", "22", "23", "24",
          "29", "30", "35", "36", "38", "47", "52", "54", "60", "82"]

# suffix -> human label. "" is the frozen-baseline default tracker.
VARIANTS = {
    "": "default",
    "_botsort": "botsort",
    "_bytetrack": "bytetrack",
    "_botsort_reid": "botsort_reid",
}

IOU_MATCH = 0.5
OUT = config.MARKOV_OUT / "tracker_fidelity.json"


# --------------------------------------------------------------------- loaders
def read_gt(video_id: str) -> pd.DataFrame:
    """Ground-truth tracks from labels_ftid. Non-destructive (never writes)."""
    ftid_dir = config.VISEM_ROOT / video_id / "labels_ftid"
    images_dir = config.VISEM_ROOT / video_id / "images"
    if not ftid_dir.exists():
        return pd.DataFrame()
    sample_img = next(images_dir.glob("*.jpg"), None)
    if sample_img is None:
        return pd.DataFrame()
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
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)


def load_variant(video_id: str, suffix: str) -> pd.DataFrame:
    path = config.TRACK_OUT / f"{video_id}{suffix}_tracks.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def seqs_from_df(df: pd.DataFrame) -> list[list[int]]:
    """Per-track motility-state sequences (identical pipeline to the paper)."""
    if df.empty:
        return []
    seqs = []
    for _, tr in df.groupby("track_id"):
        if len(tr) < config.MIN_TRACK_LENGTH:
            continue
        tr = tr.sort_values("frame")
        seqs.append([S2I[s] for s in compute_frame_states(tr)])
    return seqs


# --------------------------------------------------- association metrics vs GT
def _iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def _frame_boxes(df: pd.DataFrame) -> dict[int, dict[int, tuple]]:
    fm: dict[int, dict[int, tuple]] = {}
    for r in df.itertuples(index=False):
        fm.setdefault(int(r.frame), {})[int(r.track_id)] = (
            float(r.x1), float(r.y1), float(r.x2), float(r.y2))
    return fm


def association_metrics(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Coverage (recall of GT tracks), ID switches per GT track, frag rate."""
    if pred.empty or gt.empty:
        return {"coverage": None, "idsw_per_gt": None, "frag_rate": None,
                "n_gt_tracks": 0}
    pf = _frame_boxes(pred)
    gf = _frame_boxes(gt)
    frames = sorted(set(pf) & set(gf))

    # ID switches: for each GT track, count changes in the matched pred ID.
    gt_last: dict[int, int] = {}
    switches = 0
    # coverage numerator: GT tracks matched on >=50% of their frames.
    gt_hit: dict[int, int] = {}
    gt_tot: dict[int, int] = {}
    for fn in frames:
        gb = gf[fn]; pb = pf[fn]
        used: set[int] = set()
        for gtid, gbox in gb.items():
            gt_tot[gtid] = gt_tot.get(gtid, 0) + 1
            best_iou, best_pid = 0.0, -1
            for pid, pbox in pb.items():
                if pid in used:
                    continue
                v = _iou(gbox, pbox)
                if v > best_iou:
                    best_iou, best_pid = v, pid
            if best_iou >= IOU_MATCH and best_pid >= 0:
                used.add(best_pid)
                gt_hit[gtid] = gt_hit.get(gtid, 0) + 1
                if gtid in gt_last and gt_last[gtid] != best_pid:
                    switches += 1
                gt_last[gtid] = best_pid

    gt_ids = gt["track_id"].unique()
    n_gt = len(gt_ids)
    covered = sum(1 for g in gt_ids
                  if gt_tot.get(g, 0) > 0 and gt_hit.get(g, 0) / gt_tot[g] >= 0.5)
    coverage = covered / n_gt if n_gt else None

    # fragmentation: fraction of PREDICTED tracks shorter than 0.5 s.
    plens = pred.groupby("track_id").size().values
    frag = float((plens < 0.5 * config.FPS).mean()) if len(plens) else None

    return {"coverage": coverage,
            "idsw_per_gt": switches / n_gt if n_gt else None,
            "frag_rate": frag, "n_gt_tracks": int(n_gt)}


# ------------------------------------------------------------------- dynamics
def dynamical_stats(seqs: list[list[int]]) -> dict:
    if len(seqs) < 5:
        return {"g2": None, "dwell_cv": None, "delta_aic_ln_minus_exp": None,
                "n_tracks": len(seqs)}
    ss = summary_stats(seqs)
    cv = [ss["per_state"][s]["cv"] for s in STATES]
    ln = lognorm_vs_exp_aic(seqs)
    return {
        "g2": ss["g2"],
        "n_blocks": ss["n_blocks"],
        "dwell_cv": cv,
        "dwell_mean_frames": [ss["per_state"][s]["mean_frames"] for s in STATES],
        "delta_aic_ln_minus_exp": ln["delta_aic_ln_minus_exp"] if ln else None,
        "n_dwells": ln["n_dwells"] if ln else None,
        "n_tracks": len(seqs),
    }


def _rel_cv_error(cv, cv_gt) -> float | None:
    if cv is None or cv_gt is None:
        return None
    e = [abs(a - b) / b for a, b in zip(cv, cv_gt)
         if a is not None and b not in (None, 0)]
    return float(np.mean(e)) if e else None


# ------------------------------------------------------------------------ main
def main() -> None:
    print("Loading ground truth and tracker variants for 20 videos ...\n")

    # Pool sequences per source across all videos; collect per-video assoc metrics.
    gt_seqs: list[list[int]] = []
    var_seqs: dict[str, list[list[int]]] = {v: [] for v in VARIANTS}
    per_video_assoc: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    per_video_dyn: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    gt_per_video_dyn: list[dict] = []

    for vid in VIDEOS:
        gt = read_gt(vid)
        if gt.empty:
            print(f"  video {vid}: NO GT, skipping")
            continue
        gs = seqs_from_df(gt)
        gt_seqs.extend(gs)
        gt_per_video_dyn.append({"video": vid, **dynamical_stats(gs)})

        for suf, name in VARIANTS.items():
            pv = load_variant(vid, suf)
            if pv.empty:
                per_video_assoc[suf].append({"video": vid})
                continue
            var_seqs[suf].extend(seqs_from_df(pv))
            am = association_metrics(pv, gt)
            per_video_assoc[suf].append({"video": vid, **am})
            per_video_dyn[suf].append(
                {"video": vid, **dynamical_stats(seqs_from_df(pv))})
        print(f"  video {vid}: GT tracks={gt['track_id'].nunique()}  done")

    # Cohort-pooled dynamics (the headline numbers).
    gt_dyn = dynamical_stats(gt_seqs)
    print("\n=== GROUND TRUTH (pooled dynamics) ===")
    print(f"  g2={gt_dyn['g2']}  dwell_cv={gt_dyn['dwell_cv']}  "
          f"dAIC(ln-exp)={gt_dyn['delta_aic_ln_minus_exp']}  "
          f"n_tracks={gt_dyn['n_tracks']}")

    table = []
    for suf, name in VARIANTS.items():
        dyn = dynamical_stats(var_seqs[suf])
        # cohort association = mean over videos
        av = [a for a in per_video_assoc[suf] if a.get("coverage") is not None]
        cov = float(np.mean([a["coverage"] for a in av])) if av else None
        idsw = float(np.mean([a["idsw_per_gt"] for a in av])) if av else None
        frag = float(np.mean([a["frag_rate"] for a in av])) if av else None
        row = {
            "variant": name,
            # association quality vs GT
            "coverage": cov, "idsw_per_gt": idsw, "frag_rate": frag,
            # dynamics
            "g2": dyn["g2"], "dwell_cv": dyn["dwell_cv"],
            "delta_aic_ln_minus_exp": dyn["delta_aic_ln_minus_exp"],
            "n_tracks": dyn["n_tracks"],
            # dynamical fidelity vs GT
            "g2_abs_err": (abs(dyn["g2"] - gt_dyn["g2"])
                           if dyn["g2"] is not None and gt_dyn["g2"] is not None
                           else None),
            "dwell_cv_rel_err": _rel_cv_error(dyn["dwell_cv"], gt_dyn["dwell_cv"]),
            "g2_positive": (dyn["g2"] is not None and dyn["g2"] > 0),
            "lognormal_preferred": (dyn["delta_aic_ln_minus_exp"] is not None
                                    and dyn["delta_aic_ln_minus_exp"] > 0),
        }
        table.append(row)
        print(f"\n=== {name} ===")
        print(f"  ASSOC  coverage={cov}  idsw/gt={idsw}  frag={frag}")
        print(f"  DYN    g2={dyn['g2']}  cv={dyn['dwell_cv']}  "
              f"dAIC={dyn['delta_aic_ln_minus_exp']}  n_tracks={dyn['n_tracks']}")
        print(f"  FIDEL  g2_abs_err={row['g2_abs_err']}  "
              f"cv_rel_err={row['dwell_cv_rel_err']}  "
              f"g2>0={row['g2_positive']}  ln>exp={row['lognormal_preferred']}")

    # Dissociation test: does association quality predict dynamical fidelity?
    # Across the variants, correlate coverage (and -idsw) with g2_abs_err.
    def _corr(xs, ys):
        xy = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(xy) < 3:
            return None
        x = np.array([p[0] for p in xy]); y = np.array([p[1] for p in xy])
        if x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    covs = [r["coverage"] for r in table]
    idsws = [r["idsw_per_gt"] for r in table]
    g2errs = [r["g2_abs_err"] for r in table]
    cverrs = [r["dwell_cv_rel_err"] for r in table]
    dissociation = {
        "corr_coverage_vs_g2err": _corr(covs, g2errs),
        "corr_idsw_vs_g2err": _corr(idsws, g2errs),
        "corr_coverage_vs_cverr": _corr(covs, cverrs),
        "corr_idsw_vs_cverr": _corr(idsws, cverrs),
        "note": ("positive corr(idsw, err) or negative corr(coverage, err) => "
                 "better association predicts better dynamical fidelity; "
                 "near-zero => MOT metrics do NOT capture dynamical fidelity."),
    }

    robust = {
        "g2_positive_all_variants": all(r["g2_positive"] for r in table),
        "g2_positive_gt": gt_dyn["g2"] is not None and gt_dyn["g2"] > 0,
        "lognormal_all_variants": all(r["lognormal_preferred"] for r in table),
        "lognormal_gt": (gt_dyn["delta_aic_ln_minus_exp"] is not None
                         and gt_dyn["delta_aic_ln_minus_exp"] > 0),
    }

    result = {
        "ground_truth": gt_dyn,
        "variants": table,
        "robustness": robust,
        "dissociation": dissociation,
        "gt_per_video_dyn": gt_per_video_dyn,
        "per_video_assoc": per_video_assoc,
        "config": {"videos": VIDEOS, "iou_match": IOU_MATCH,
                   "min_track_length": config.MIN_TRACK_LENGTH, "fps": config.FPS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    OUT.write_text(json.dumps(result, indent=2, default=_default))
    print("\n=== ROBUSTNESS ===")
    for k, v in robust.items():
        print(f"  {k}: {v}")
    print("\n=== DISSOCIATION (does association predict dynamical fidelity?) ===")
    for k, v in dissociation.items():
        if k != "note":
            print(f"  {k}: {v}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
