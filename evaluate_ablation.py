#!/usr/bin/env python3
"""
evaluate_ablation.py

Ablation study: compare GT-ceiling, YOLOv8n, and YOLOv8l pipelines
on the held-out (val+test) videos.

Produces:
  - Per-model evaluation metrics (MAE, RMSE, Pearson r, bias)
  - Side-by-side ablation table
  - Comparison plots

Usage:
    python evaluate_ablation.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HELD_OUT_VIDEOS = ["14", "23", "47", "60", "29", "52"]
VAL_VIDEOS = ["14", "23", "47", "60"]
TEST_VIDEOS = ["29", "52"]

GT_CSV = config.RAW_DIR / "semen_analysis_data_Train.csv"
EVAL_DIR = config.OUTPUTS_DIR / "evaluation_ablation"

ALL_VIDEOS = [
    "11", "12", "13", "14", "15", "19", "21", "22", "23", "24",
    "29", "30", "35", "36", "38", "47", "52", "54", "60", "82",
]

VIDEO_SETS = {
    "val": VAL_VIDEOS,
    "test": TEST_VIDEOS,
    "held_out": HELD_OUT_VIDEOS,
    "all": ALL_VIDEOS,
}

# Clean event output directories (each pipeline's results saved separately)
GT_EVENTS_DIR = config.OUTPUTS_DIR / "events_gt_clean"
YOLOV8L_EVENTS_DIR = config.OUTPUTS_DIR / "events_yolov8l_backup"
YOLOV8N_EVENTS_DIR = config.OUTPUTS_DIR / "events_yolov8n_clean"

# Legacy directories for backward compatibility
LEGACY_DIRS: dict[str, tuple[Path, bool]] = {
    "GT-ceiling": (GT_EVENTS_DIR, False),       # use_adjusted=False
    "YOLOv8n-legacy": (YOLOV8N_EVENTS_DIR, True),
    "YOLOv8l-legacy": (YOLOV8L_EVENTS_DIR, True),
}

# Directory where run_experiments.py writes per-experiment event results
EXPERIMENTS_EVENTS_DIR = config.OUTPUTS_DIR / "events"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth() -> pd.DataFrame:
    """Load clinical semen analysis ground truth."""
    df = pd.read_csv(GT_CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ID": "video",
        "Progressive motility (%)": "gt_progressive",
        "Non progressive sperm motility (%)": "gt_non_progressive",
        "Immotile sperm (%)": "gt_immotile",
    })
    df["video"] = df["video"].astype(str)
    return df


def load_summaries(events_dir: Path, video_ids: list[str],
                   use_adjusted: bool = True) -> pd.DataFrame:
    """Load motility summary JSONs for specific videos from a directory.
    
    Parameters
    ----------
    use_adjusted : bool
        If True, prefer adjusted percentages (for YOLO pipelines where
        detection JSON is from the same model). If False, use raw
        percentages (for GT pipeline, where no model-specific detection
        JSON exists).
    """
    rows = []
    for vid in video_ids:
        sp = events_dir / f"{vid}_summary.json"
        if not sp.exists():
            print(f"  WARNING: {sp} not found")
            continue
        with open(sp) as f:
            s = json.load(f)

        # Choose raw or adjusted percentages
        if use_adjusted and "adjusted_progressive_pct" in s:
            prog = s["adjusted_progressive_pct"]
            nonpro = s["adjusted_non_progressive_pct"]
            immot = s["adjusted_immotile_pct"]
        else:
            prog = s.get("progressive_pct", 0)
            nonpro = s.get("non_progressive_pct", 0)
            immot = s.get("immotile_pct", 0)

        rows.append({
            "video": str(s["video"]),
            "pred_progressive": prog,
            "pred_non_progressive": nonpro,
            "pred_immotile": immot,
            "total_tracks": s.get("total_tracks", 0),
            "adjusted_total": s.get("adjusted_total", s.get("total_tracks", 0)),
            "untracked_fraction": s.get("untracked_fraction", 0),
            "mean_VCL": s.get("mean_VCL", 0),
        })
    return pd.DataFrame(rows)


def compute_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, Pearson r, bias for each motility category."""
    merged = pd.merge(gt_df, pred_df, on="video", how="inner")
    if merged.empty:
        return {}

    results = {"n_videos": len(merged)}
    for cat in ["progressive", "non_progressive", "immotile"]:
        gt_col = f"gt_{cat}"
        pred_col = f"pred_{cat}"
        y_true = merged[gt_col].values.astype(float)
        y_pred = merged[pred_col].values.astype(float)

        m_mae = np.mean(np.abs(y_true - y_pred))
        m_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        bias = np.mean(y_pred - y_true)
        if len(y_true) >= 3:
            r, p = stats.pearsonr(y_true, y_pred)
        else:
            r, p = 0.0, 1.0

        results[cat] = {
            "MAE": round(m_mae, 2),
            "RMSE": round(m_rmse, 2),
            "r": round(r, 3),
            "p": round(p, 4),
            "bias": round(bias, 2),
        }

    results["merged"] = merged
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model discovery & loading
# ─────────────────────────────────────────────────────────────────────────────

def discover_experiments() -> dict[str, Path]:
    """Scan outputs/events/ for experiment subdirectories.

    Each subdirectory name is an experiment name (e.g. "yolov8n_botsort",
    "rtdetr-l_botsort").  Returns a dict mapping experiment name → events
    directory Path.
    """
    experiments: dict[str, Path] = {}
    if not EXPERIMENTS_EVENTS_DIR.is_dir():
        return experiments
    for child in sorted(EXPERIMENTS_EVENTS_DIR.iterdir()):
        if child.is_dir():
            # Check the subdir actually contains at least one summary JSON
            if any(child.glob("*_summary.json")):
                experiments[child.name] = child
    return experiments


def load_experiment_results(
    events_dir: Path,
    video_ids: list[str],
    use_adjusted: bool = True,
) -> pd.DataFrame:
    """Load all _summary.json files from *events_dir* for *video_ids*.

    Returns a DataFrame with columns:
        video, progressive_pct, non_progressive_pct, immotile_pct
    (plus extra bookkeeping columns matching the existing format).
    """
    rows: list[dict] = []
    for vid in video_ids:
        sp = events_dir / f"{vid}_summary.json"
        if not sp.exists():
            continue
        with open(sp) as f:
            s = json.load(f)

        if use_adjusted and "adjusted_progressive_pct" in s:
            prog = s["adjusted_progressive_pct"]
            nonpro = s["adjusted_non_progressive_pct"]
            immot = s["adjusted_immotile_pct"]
        else:
            prog = s.get("progressive_pct", 0)
            nonpro = s.get("non_progressive_pct", 0)
            immot = s.get("immotile_pct", 0)

        rows.append({
            "video": str(s["video"]),
            "pred_progressive": prog,
            "pred_non_progressive": nonpro,
            "pred_immotile": immot,
            "total_tracks": s.get("total_tracks", 0),
            "adjusted_total": s.get("adjusted_total", s.get("total_tracks", 0)),
            "untracked_fraction": s.get("untracked_fraction", 0),
            "mean_VCL": s.get("mean_VCL", 0),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multi(
    experiment_names: list[str] | None = None,
    video_set: str = "held_out",
) -> pd.DataFrame:
    """Evaluate any set of experiments against clinical ground truth.

    Parameters
    ----------
    experiment_names : list[str] | None
        Experiment names to evaluate.  ``None`` = all discovered + legacy.
    video_set : str
        Which video IDs to use: "val", "test", "held_out", or "all".

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per (experiment, category).
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    video_ids = VIDEO_SETS.get(video_set, HELD_OUT_VIDEOS)

    # Load clinical ground truth
    gt = load_ground_truth()
    gt_filtered = gt[gt["video"].isin(video_ids)].copy()
    print(f"Clinical GT available for {len(gt_filtered)} / {len(video_ids)} "
          f"videos (set='{video_set}')")

    # Collect experiment sources: {name: (events_dir, use_adjusted)}
    sources: dict[str, tuple[Path, bool]] = {}

    # Legacy directories (backward compat)
    for name, (edir, adj) in LEGACY_DIRS.items():
        if edir.is_dir():
            sources[name] = (edir, adj)

    # Discovered experiment subdirectories
    discovered = discover_experiments()
    for name, edir in discovered.items():
        sources[name] = (edir, True)  # experiment outputs always use adjusted

    # Filter to requested experiments
    if experiment_names:
        sources = {
            k: v for k, v in sources.items() if k in experiment_names
        }

    if not sources:
        print("No experiments found to evaluate.")
        return pd.DataFrame()

    print(f"Evaluating {len(sources)} experiment(s): "
          f"{', '.join(sources.keys())}")

    # Evaluate each experiment
    all_metrics: dict[str, dict] = {}
    all_preds: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for exp_name, (events_dir, use_adjusted) in sources.items():
        print(f"\n━━━ {exp_name} ━━━")
        pred_df = load_experiment_results(events_dir, video_ids, use_adjusted)
        if pred_df.empty:
            print(f"  No results found in {events_dir}")
            continue

        metrics = compute_metrics(gt_filtered, pred_df)
        if not metrics:
            print(f"  No overlapping videos with GT")
            continue

        all_metrics[exp_name] = metrics
        all_preds[exp_name] = pred_df

        for cat in ["progressive", "non_progressive", "immotile"]:
            if cat in metrics:
                m = metrics[cat]
                summary_rows.append({
                    "experiment": exp_name,
                    "category": cat,
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "r": m["r"],
                    "p": m["p"],
                    "bias": m["bias"],
                    "n_videos": metrics["n_videos"],
                })

    if not summary_rows:
        print("\nNo valid results to compare.")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(summary_rows)

    # ── Print comparison table ─────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  MULTI-MODEL COMPARISON  (videos: {video_set})")
    print(f"{'='*95}")
    print(f"  {'Experiment':<25} {'Category':<18} {'MAE':>7} {'RMSE':>7} "
          f"{'Bias':>7} {'r':>7} {'p':>8} {'n':>4}")
    print(f"  {'─'*25} {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*4}")

    for exp_name in all_metrics:
        metrics = all_metrics[exp_name]
        for cat in ["progressive", "non_progressive", "immotile"]:
            if cat in metrics:
                m = metrics[cat]
                print(f"  {exp_name:<25} {cat:<18} {m['MAE']:>7.1f} "
                      f"{m['RMSE']:>7.1f} {m['bias']:>+7.1f} {m['r']:>7.3f} "
                      f"{m['p']:>8.4f} {metrics['n_videos']:>4}")
        print()
    print(f"{'='*95}")

    # ── Save results ───────────────────────────────────────────────────────
    comparison_df.to_csv(
        EVAL_DIR / "multi_model_comparison.csv", index=False,
    )
    print(f"\n  Comparison CSV → {EVAL_DIR / 'multi_model_comparison.csv'}")

    multi_json = {}
    for exp_name, metrics in all_metrics.items():
        multi_json[exp_name] = {
            "n_videos": metrics.get("n_videos", 0),
            **{cat: metrics[cat] for cat in
               ["progressive", "non_progressive", "immotile"]
               if cat in metrics},
        }
    with open(EVAL_DIR / "multi_model_metrics.json", "w") as f:
        json.dump(multi_json, f, indent=2)
    print(f"  Metrics JSON  → {EVAL_DIR / 'multi_model_metrics.json'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_multi_comparison(all_metrics, video_set)
    _plot_scatter(gt_filtered, all_preds)

    return comparison_df


def _plot_multi_comparison(all_metrics: dict, video_set: str):
    """Grouped bar chart: MAE per experiment for each motility category."""
    categories = ["progressive", "non_progressive", "immotile"]
    cat_labels = ["Progressive", "Non-progressive", "Immotile"]
    exp_names = list(all_metrics.keys())
    n_exp = len(exp_names)
    n_cats = len(categories)

    if n_exp == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_exp), 6))

    x = np.arange(n_cats)
    width = 0.8 / n_exp
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(n_exp, 3))

    max_mae = 0.0
    for i, exp in enumerate(exp_names):
        metrics = all_metrics[exp]
        maes = [metrics[cat]["MAE"] if cat in metrics else 0 for cat in categories]
        max_mae = max(max_mae, max(maes))
        offset = i * width - (n_exp - 1) * width / 2
        bars = ax.bar(
            x + offset, maes, width,
            label=exp, color=cmap(i), alpha=0.85,
        )
        for bar, val in zip(bars, maes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=12)
    ax.set_ylabel("MAE (%)", fontsize=12)
    ax.set_title(
        f"Multi-Model Comparison: MAE (videos: {video_set})", fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_ylim(0, max_mae * 1.35 if max_mae > 0 else 10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = EVAL_DIR / "multi_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy evaluation (kept as fallback)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth()
    gt_held = gt[gt["video"].isin(HELD_OUT_VIDEOS)].copy()
    print(f"Clinical GT available for {len(gt_held)} held-out videos: "
          f"{sorted(gt_held['video'].tolist())}")

    # ── 1. GT-ceiling pipeline (use RAW percentages — no valid detection JSON) ──
    print("\n━━━ GT-ceiling (annotations-based pipeline) ━━━")
    gt_pred = load_summaries(GT_EVENTS_DIR, HELD_OUT_VIDEOS, use_adjusted=False)
    gt_metrics = compute_metrics(gt_held, gt_pred)

    # ── 2. YOLOv8l pipeline (use adjusted — detection JSON from same model) ──
    print("\n━━━ YOLOv8l (trained detector pipeline) ━━━")
    yolov8l_pred = load_summaries(YOLOV8L_EVENTS_DIR, HELD_OUT_VIDEOS, use_adjusted=True)
    yolov8l_metrics = compute_metrics(gt_held, yolov8l_pred)

    # ── 3. YOLOv8n pipeline (use adjusted — detection JSON from same model) ──
    print("\n━━━ YOLOv8n (nano detector pipeline) ━━━")
    yolov8n_pred = load_summaries(YOLOV8N_EVENTS_DIR, HELD_OUT_VIDEOS, use_adjusted=True)
    yolov8n_metrics = compute_metrics(gt_held, yolov8n_pred)

    # ── 4. Print ablation table ────────────────────────────────────────────
    all_models = {}
    if gt_metrics:
        all_models["GT-ceiling"] = gt_metrics
    if yolov8n_metrics:
        all_models["YOLOv8n"] = yolov8n_metrics
    if yolov8l_metrics:
        all_models["YOLOv8l"] = yolov8l_metrics

    all_preds = {}
    if not gt_pred.empty:
        all_preds["GT-ceiling"] = gt_pred
    if not yolov8n_pred.empty:
        all_preds["YOLOv8n"] = yolov8n_pred
    if not yolov8l_pred.empty:
        all_preds["YOLOv8l"] = yolov8l_pred

    print(f"\n{'='*90}")
    print("  ABLATION TABLE: Held-out Videos (Val + Test)")
    print(f"{'='*90}")
    print(f"  {'Model':<15} {'Category':<18} {'MAE':>7} {'RMSE':>7} {'Bias':>7} "
          f"{'r':>7} {'p':>8}")
    print(f"  {'─'*15:<15} {'─'*18:<18} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} "
          f"{'─'*7:>7} {'─'*8:>8}")
    for model_name, metrics in all_models.items():
        for cat in ["progressive", "non_progressive", "immotile"]:
            if cat in metrics:
                m = metrics[cat]
                print(f"  {model_name:<15} {cat:<18} {m['MAE']:>7.1f} {m['RMSE']:>7.1f} "
                      f"{m['bias']:>+7.1f} {m['r']:>7.3f} {m['p']:>8.4f}")
        print()
    print(f"{'='*90}")

    # ── 5. Per-video comparison ────────────────────────────────────────────
    print(f"\n  Per-video comparison (GT-ceil | YOLOv8n | YOLOv8l):")
    print(f"  {'Video':>6} │ {'GT_P':>5} │ {'Ceil':>5} {'Nano':>5} {'Lrg':>5} │ "
          f"{'GT_NP':>6} │ {'Ceil':>5} {'Nano':>5} {'Lrg':>5} │ "
          f"{'GT_Im':>6} │ {'Ceil':>5} {'Nano':>5} {'Lrg':>5}")
    sep = f"  {'─'*6} │ {'─'*5} │ {'─'*5} {'─'*5} {'─'*5} │ {'─'*6} │ {'─'*5} {'─'*5} {'─'*5} │ {'─'*6} │ {'─'*5} {'─'*5} {'─'*5}"
    print(sep)

    for vid in sorted(HELD_OUT_VIDEOS, key=int):
        gt_row = gt_held[gt_held["video"] == vid]
        if gt_row.empty:
            continue
        gp = gt_row["gt_progressive"].iloc[0]
        gnp = gt_row["gt_non_progressive"].iloc[0]
        gim = gt_row["gt_immotile"].iloc[0]

        vals = {}
        for mname, pred_df in all_preds.items():
            row = pred_df[pred_df["video"] == vid]
            if not row.empty:
                vals[mname] = {
                    "p": row["pred_progressive"].iloc[0],
                    "np": row["pred_non_progressive"].iloc[0],
                    "im": row["pred_immotile"].iloc[0],
                }
            else:
                vals[mname] = {"p": float("nan"), "np": float("nan"), "im": float("nan")}

        gc = vals.get("GT-ceiling", {"p":float("nan"),"np":float("nan"),"im":float("nan")})
        yn = vals.get("YOLOv8n", {"p":float("nan"),"np":float("nan"),"im":float("nan")})
        yl = vals.get("YOLOv8l", {"p":float("nan"),"np":float("nan"),"im":float("nan")})

        print(f"  {vid:>6} │ {gp:>5.0f} │ {gc['p']:>5.1f} {yn['p']:>5.1f} {yl['p']:>5.1f} │ "
              f"{gnp:>6.0f} │ {gc['np']:>5.1f} {yn['np']:>5.1f} {yl['np']:>5.1f} │ "
              f"{gim:>6.0f} │ {gc['im']:>5.1f} {yn['im']:>5.1f} {yl['im']:>5.1f}")

    # ── 6. Track count comparison ──────────────────────────────────────────
    print(f"\n  Track counts (held-out videos):")
    print(f"  {'Video':>6} │ {'GT-ceil':>8} {'YOLOv8n':>8} {'YOLOv8l':>8}")
    print(f"  {'─'*6} │ {'─'*8} {'─'*8} {'─'*8}")
    for vid in sorted(HELD_OUT_VIDEOS, key=int):
        gc_t = yn_t = yl_t = 0
        for mname, pred_df in all_preds.items():
            row = pred_df[pred_df["video"] == vid]
            if not row.empty:
                t = int(row["total_tracks"].iloc[0])
                if mname == "GT-ceiling":
                    gc_t = t
                elif mname == "YOLOv8n":
                    yn_t = t
                elif mname == "YOLOv8l":
                    yl_t = t
        print(f"  {vid:>6} │ {gc_t:>8} {yn_t:>8} {yl_t:>8}")

    # ── 7. Save results ───────────────────────────────────────────────────
    ablation_json = {}
    for model_name, metrics in all_models.items():
        ablation_json[model_name] = {
            "n_videos": metrics.get("n_videos", 0),
            **{cat: metrics[cat] for cat in ["progressive", "non_progressive", "immotile"]
               if cat in metrics}
        }
    with open(EVAL_DIR / "ablation_metrics.json", "w") as f:
        json.dump(ablation_json, f, indent=2)
    print(f"\n  Ablation JSON → {EVAL_DIR / 'ablation_metrics.json'}")

    # ── 8. Ablation comparison plot ────────────────────────────────────────
    _plot_ablation(all_models)

    # ── 9. Per-video scatter plots ─────────────────────────────────────────
    _plot_scatter(gt_held, all_preds)

    return all_models


def _plot_ablation(all_models: dict):
    """Bar chart comparing MAE across models for each motility category."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["progressive", "non_progressive", "immotile"]
    cat_labels = ["Progressive", "Non-progressive", "Immotile"]
    model_names = list(all_models.keys())
    n_models = len(model_names)
    n_cats = len(categories)

    x = np.arange(n_cats)
    width = 0.8 / n_models
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    for i, model in enumerate(model_names):
        metrics = all_models[model]
        maes = [metrics[cat]["MAE"] if cat in metrics else 0 for cat in categories]
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2, maes,
                      width, label=model, color=colors[i % len(colors)], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=12)
    ax.set_ylabel("MAE (%)", fontsize=12)
    ax.set_title("Ablation Study: MAE by Model (Held-out Videos)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(m[cat]["MAE"] for m in all_models.values()
                       for cat in categories if cat in m) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = EVAL_DIR / "ablation_mae_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ablation plot → {out}")


def _plot_scatter(gt_held: pd.DataFrame, all_preds: dict):
    """Scatter plots: predicted vs GT for each model and motility category."""
    categories = [("progressive", "Progressive"), ("non_progressive", "Non-progressive"),
                  ("immotile", "Immotile")]
    model_names = list(all_preds.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]
    colors = {"GT-ceiling": "#2196F3", "YOLOv8n": "#FF9800", "YOLOv8l": "#4CAF50"}

    for i, model_name in enumerate(model_names):
        pred_df = all_preds[model_name]
        merged = pd.merge(gt_held, pred_df, on="video", how="inner")
        color = colors.get(model_name, "#888888")

        for j, (cat, label) in enumerate(categories):
            ax = axes[i, j]
            gt_col = f"gt_{cat}"
            pred_col = f"pred_{cat}"
            x = merged[gt_col].values.astype(float)
            y = merged[pred_col].values.astype(float)

            ax.scatter(x, y, c=color, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

            lim = max(max(x.max(), y.max()) * 1.15, 10)
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)

            if len(x) >= 3:
                r, p = stats.pearsonr(x, y)
                slope, intercept, *_ = stats.linregress(x, y)
                xs = np.linspace(0, lim, 50)
                ax.plot(xs, slope * xs + intercept, color=color, alpha=0.5,
                        label=f"r={r:.3f}")
                ax.legend(fontsize=9)

            for _, row in merged.iterrows():
                ax.annotate(str(row["video"]), (row[gt_col], row[pred_col]),
                            fontsize=8, alpha=0.6, ha="center", va="bottom")

            ax.set_xlim(-2, lim)
            ax.set_ylim(-2, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"Clinical GT (%)")
            ax.set_ylabel(f"Predicted (%)")
            ax.set_title(f"{model_name} – {label}")

    fig.suptitle("Ablation: Predicted vs Clinical Ground Truth", fontsize=14, y=1.01)
    fig.tight_layout()
    out = EVAL_DIR / "ablation_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate experiments against clinical GT",
    )
    p.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment names to evaluate (default: all).",
    )
    p.add_argument(
        "--videos",
        type=str,
        default="held_out",
        choices=["val", "test", "held_out", "all"],
        help="Which video set to evaluate on (default: held_out).",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help="Run legacy three-model ablation (run_ablation) instead.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.legacy:
        run_ablation()
    else:
        exp_list = (
            [e.strip() for e in args.experiments.split(",")]
            if args.experiments
            else None
        )
        evaluate_multi(
            experiment_names=exp_list,
            video_set=args.videos,
        )
