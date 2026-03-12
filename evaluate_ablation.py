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

# Clean event output directories (each pipeline's results saved separately)
GT_EVENTS_DIR = config.OUTPUTS_DIR / "events_gt_clean"
YOLOV8L_EVENTS_DIR = config.OUTPUTS_DIR / "events_yolov8l_backup"
YOLOV8N_EVENTS_DIR = config.OUTPUTS_DIR / "events_yolov8n_clean"


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
# Main evaluation
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


if __name__ == "__main__":
    run_ablation()
