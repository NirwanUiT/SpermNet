#!/usr/bin/env python3
"""
evaluate.py

Compare pipeline motility predictions against clinical ground truth
from semen_analysis_data_Train.csv.

Generates:
  - outputs/evaluation/gt_comparison.csv
  - outputs/evaluation/gt_comparison_plot.png
  - outputs/evaluation/bland_altman.png
  - outputs/evaluation/correlation_matrix.png
  - Console summary with MAE, RMSE, correlation per metric

Usage:
    python evaluate.py
    python evaluate.py --threshold-sweep   # test different VCL/STR thresholds
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
# Load ground truth
# ─────────────────────────────────────────────────────────────────────────────

GT_CSV = config.RAW_DIR / "semen_analysis_data_Train.csv"
EVAL_DIR = config.OUTPUTS_DIR / "evaluation"


def load_ground_truth() -> pd.DataFrame:
    """Load clinical semen analysis ground truth."""
    if not GT_CSV.exists():
        print(f"ERROR: {GT_CSV} not found")
        return pd.DataFrame()

    df = pd.read_csv(GT_CSV)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ID": "video",
        "Progressive motility (%)": "gt_progressive",
        "Non progressive sperm motility (%)": "gt_non_progressive",
        "Immotile sperm (%)": "gt_immotile",
        "Sperm concentration (x10⁶/mL)": "gt_concentration",
        "Sperm vitality (%)": "gt_vitality",
    })
    df["video"] = df["video"].astype(str)
    return df


def load_predictions() -> pd.DataFrame:
    """Load pipeline motility summaries for all processed videos."""
    rows = []
    for sp in sorted(config.EVENTS_OUT.glob("*_summary.json")):
        with open(sp) as f:
            s = json.load(f)
        rows.append({
            "video":              str(s["video"]),
            "pred_progressive":   s.get("progressive_pct", 0),
            "pred_non_progressive": s.get("non_progressive_pct", 0),
            "pred_immotile":      s.get("immotile_pct", 0),
            "pred_total_tracks":  s.get("total_tracks", 0),
            "pred_mean_VCL":      s.get("mean_VCL", 0),
            "pred_mean_VSL":      s.get("mean_VSL", 0),
            "pred_mean_VAP":      s.get("mean_VAP", 0),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def pearson_r(y_true, y_pred):
    if len(y_true) < 3:
        return 0, 1
    r, p = stats.pearsonr(y_true, y_pred)
    return r, p


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_against_gt():
    """Compare pipeline predictions against clinical ground truth."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth()
    pred = load_predictions()

    if gt.empty or pred.empty:
        print("Cannot evaluate: missing ground truth or predictions.")
        return

    # Merge on video ID
    merged = pd.merge(gt, pred, on="video", how="inner")
    if merged.empty:
        print("No matching videos between GT and predictions.")
        return

    # Save merged CSV
    merged.to_csv(EVAL_DIR / "gt_comparison.csv", index=False)
    print(f"\nGT comparison CSV → {EVAL_DIR / 'gt_comparison.csv'}")

    # Compute metrics
    metrics = {}
    for cat in ["progressive", "non_progressive", "immotile"]:
        gt_col = f"gt_{cat}"
        pred_col = f"pred_{cat}"
        if gt_col in merged.columns and pred_col in merged.columns:
            m_mae = mae(merged[gt_col], merged[pred_col])
            m_rmse = rmse(merged[gt_col], merged[pred_col])
            r, p = pearson_r(merged[gt_col].values, merged[pred_col].values)
            bias = np.mean(merged[pred_col].values - merged[gt_col].values)
            metrics[cat] = {"MAE": m_mae, "RMSE": m_rmse, "r": r, "p": p, "bias": bias}

    # Print results table
    print(f"\n{'='*70}")
    print("  EVALUATION: Pipeline vs Clinical Ground Truth")
    print(f"{'='*70}")
    print(f"  Videos evaluated: {len(merged)}")
    print(f"\n  {'Category':<20} {'MAE':>7} {'RMSE':>7} {'Bias':>7} {'r':>7} {'p':>8}")
    print(f"  {'─'*20:<20} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*8:>8}")
    for cat, m in metrics.items():
        print(f"  {cat:<20} {m['MAE']:>7.1f} {m['RMSE']:>7.1f} {m['bias']:>+7.1f} "
              f"{m['r']:>7.3f} {m['p']:>8.4f}")
    print(f"{'='*70}")

    # Per-video comparison table
    print(f"\n  {'Video':>6} {'GT_P':>5} {'Pr_P':>5} {'GT_NP':>6} {'Pr_NP':>6} "
          f"{'GT_Im':>6} {'Pr_Im':>6}")
    print(f"  {'─'*6:>6} {'─'*5:>5} {'─'*5:>5} {'─'*6:>6} {'─'*6:>6} {'─'*6:>6} {'─'*6:>6}")
    for _, r in merged.iterrows():
        print(f"  {r['video']:>6} {r['gt_progressive']:>5.0f} {r['pred_progressive']:>5.1f} "
              f"{r['gt_non_progressive']:>6.0f} {r['pred_non_progressive']:>6.1f} "
              f"{r['gt_immotile']:>6.0f} {r['pred_immotile']:>6.1f}")

    # Save metrics JSON
    metrics_json = {cat: {k: round(float(v), 4) for k, v in m.items()}
                    for cat, m in metrics.items()}
    metrics_json["n_videos"] = len(merged)
    with open(EVAL_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n  Metrics JSON → {EVAL_DIR / 'evaluation_metrics.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_comparison(merged)
    _plot_bland_altman(merged)
    _plot_per_video_bars(merged)

    return merged, metrics


def _plot_comparison(merged: pd.DataFrame):
    """Scatter plots: predicted vs ground truth for each motility class."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cats = [("progressive", "Progressive"), ("non_progressive", "Non-progressive"),
            ("immotile", "Immotile")]
    colors = ["#00AA00", "#FF8800", "#CC0000"]

    for ax, (cat, label), color in zip(axes, cats, colors):
        gt_col = f"gt_{cat}"
        pred_col = f"pred_{cat}"
        x = merged[gt_col].values
        y = merged[pred_col].values

        ax.scatter(x, y, c=color, s=50, alpha=0.7, edgecolors="white", linewidths=0.5)

        # Identity line
        lim = max(max(x.max(), y.max()) * 1.1, 10)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="Perfect")

        # Regression line
        if len(x) >= 3:
            slope, intercept, r, p, se = stats.linregress(x, y)
            xs = np.linspace(0, lim, 50)
            ax.plot(xs, slope * xs + intercept, color=color, alpha=0.5,
                    label=f"r={r:.3f}")

        # Annotate with video IDs
        for _, row in merged.iterrows():
            ax.annotate(str(row["video"]), (row[gt_col], row[pred_col]),
                        fontsize=7, alpha=0.6, ha="center", va="bottom")

        ax.set_xlabel(f"Ground Truth {label} (%)")
        ax.set_ylabel(f"Predicted {label} (%)")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_xlim(-2, lim)
        ax.set_ylim(-2, lim)
        ax.set_aspect("equal")

    fig.suptitle("Pipeline vs Clinical Ground Truth", fontsize=14, y=1.02)
    fig.tight_layout()
    out = EVAL_DIR / "gt_comparison_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot → {out}")


def _plot_bland_altman(merged: pd.DataFrame):
    """Bland-Altman plots for each motility class."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cats = [("progressive", "Progressive"), ("non_progressive", "Non-progressive"),
            ("immotile", "Immotile")]
    colors = ["#00AA00", "#FF8800", "#CC0000"]

    for ax, (cat, label), color in zip(axes, cats, colors):
        gt_col = f"gt_{cat}"
        pred_col = f"pred_{cat}"
        x = merged[gt_col].values.astype(float)
        y = merged[pred_col].values.astype(float)

        mean_val = (x + y) / 2
        diff = y - x  # pred - gt

        mean_diff = np.mean(diff)
        std_diff = np.std(diff) * 1.96

        ax.scatter(mean_val, diff, c=color, s=50, alpha=0.7, edgecolors="white")
        ax.axhline(mean_diff, color="k", linestyle="-", alpha=0.5, label=f"Bias={mean_diff:+.1f}")
        ax.axhline(mean_diff + std_diff, color="gray", linestyle="--", alpha=0.4,
                    label=f"+1.96SD={mean_diff+std_diff:+.1f}")
        ax.axhline(mean_diff - std_diff, color="gray", linestyle="--", alpha=0.4,
                    label=f"-1.96SD={mean_diff-std_diff:+.1f}")
        ax.axhline(0, color="k", linestyle=":", alpha=0.2)

        ax.set_xlabel(f"Mean {label} (%)")
        ax.set_ylabel(f"Difference (Pred - GT) (%)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Bland-Altman Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    out = EVAL_DIR / "bland_altman.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bland-Altman plot → {out}")


def _plot_per_video_bars(merged: pd.DataFrame):
    """Grouped bar chart: GT vs predicted for each video."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    cats = [("progressive", "Progressive"), ("non_progressive", "Non-progressive"),
            ("immotile", "Immotile")]
    colors_gt = ["#006600", "#995500", "#880000"]
    colors_pred = ["#00CC00", "#FFAA33", "#FF3333"]

    merged_sorted = merged.sort_values("video")
    x = np.arange(len(merged_sorted))
    width = 0.35

    for ax, (cat, label), c_gt, c_pred in zip(axes, cats, colors_gt, colors_pred):
        gt_vals = merged_sorted[f"gt_{cat}"].values
        pred_vals = merged_sorted[f"pred_{cat}"].values

        ax.bar(x - width/2, gt_vals, width, label="Ground Truth", color=c_gt, alpha=0.8)
        ax.bar(x + width/2, pred_vals, width, label="Predicted", color=c_pred, alpha=0.8)

        ax.set_ylabel(f"{label} (%)")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(merged_sorted["video"].values, rotation=45)
    axes[-1].set_xlabel("Video ID")

    fig.suptitle("Per-Video: Ground Truth vs Predicted Motility", fontsize=14)
    fig.tight_layout()
    out = EVAL_DIR / "per_video_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-video bar chart → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep():
    """
    Test different VCL and STR threshold combinations to find the
    configuration that minimises MAE against clinical ground truth.
    """
    from events.detect_events import compute_track_metrics

    gt = load_ground_truth()
    if gt.empty:
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load all track CSVs
    all_tracks = {}
    for tf in sorted(config.TRACK_OUT.glob("*_tracks.csv")):
        vid = tf.stem.replace("_tracks", "")
        all_tracks[vid] = pd.read_csv(tf)

    if not all_tracks:
        print("No track CSVs found. Run pipeline first.")
        return

    # Pre-compute per-track metrics for all videos
    print("Pre-computing track metrics for all videos...")
    video_metrics = {}
    for vid, tracks_df in all_tracks.items():
        metrics_list = []
        for tid in tracks_df["track_id"].unique():
            track = tracks_df[tracks_df["track_id"] == tid].sort_values("frame")
            m = compute_track_metrics(track)
            if m is not None:
                metrics_list.append(m)
        if metrics_list:
            video_metrics[vid] = pd.DataFrame(metrics_list)
    print(f"  Computed metrics for {len(video_metrics)} videos.")

    # Sweep parameters
    vcl_prog_range = np.arange(10, 60, 5)       # VCL_PROGRESSIVE_MIN
    str_prog_range = np.arange(0.40, 0.95, 0.05) # STR_PROGRESSIVE_MIN
    vcl_imm_range  = np.arange(2, 15, 1)         # VCL_IMMOTILE_MAX

    best_mae = float("inf")
    best_params = {}
    results = []

    for vcl_prog in vcl_prog_range:
        for str_prog in str_prog_range:
            for vcl_imm in vcl_imm_range:
                total_mae = 0
                n = 0
                for vid, mdf in video_metrics.items():
                    gt_row = gt[gt["video"] == vid]
                    if gt_row.empty:
                        continue

                    # Classify with these thresholds
                    def classify(row):
                        v = row["VCL"]
                        s = row["STR"]
                        if v <= vcl_imm:
                            return "immotile"
                        elif v >= vcl_prog and s >= str_prog:
                            return "progressive"
                        else:
                            return "non_progressive"

                    mdf_copy = mdf.copy()
                    mdf_copy["motility"] = mdf_copy.apply(classify, axis=1)
                    total_t = len(mdf_copy)
                    if total_t == 0:
                        continue

                    counts = mdf_copy["motility"].value_counts()
                    pred_p = 100 * counts.get("progressive", 0) / total_t
                    pred_np = 100 * counts.get("non_progressive", 0) / total_t
                    pred_im = 100 * counts.get("immotile", 0) / total_t

                    gt_p = float(gt_row["gt_progressive"].iloc[0])
                    gt_np = float(gt_row["gt_non_progressive"].iloc[0])
                    gt_im = float(gt_row["gt_immotile"].iloc[0])

                    total_mae += abs(pred_p - gt_p) + abs(pred_np - gt_np) + abs(pred_im - gt_im)
                    n += 1

                if n > 0:
                    avg_mae = total_mae / n
                    results.append({
                        "VCL_PROG_MIN": vcl_prog,
                        "STR_PROG_MIN": round(str_prog, 2),
                        "VCL_IMM_MAX": vcl_imm,
                        "avg_total_mae": round(avg_mae, 2),
                    })
                    if avg_mae < best_mae:
                        best_mae = avg_mae
                        best_params = {
                            "VCL_PROGRESSIVE_MIN": float(vcl_prog),
                            "STR_PROGRESSIVE_MIN": round(float(str_prog), 2),
                            "VCL_IMMOTILE_MAX": float(vcl_imm),
                            "avg_total_MAE": round(avg_mae, 2),
                        }

    # Save sweep results
    sweep_df = pd.DataFrame(results).sort_values("avg_total_mae")
    sweep_df.to_csv(EVAL_DIR / "threshold_sweep.csv", index=False)

    print(f"\n{'='*60}")
    print("  THRESHOLD SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  Tested: {len(results)} parameter combinations")
    print(f"  Best average total MAE: {best_mae:.2f}")
    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Top 10 configurations:")
    print(sweep_df.head(10).to_string(index=False))
    print(f"\n  Full sweep → {EVAL_DIR / 'threshold_sweep.csv'}")
    print(f"{'='*60}")

    # Save best params
    with open(EVAL_DIR / "best_thresholds.json", "w") as f:
        json.dump(best_params, f, indent=2)

    return best_params


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against ground truth")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Run threshold parameter sweep")
    args = parser.parse_args()

    if args.threshold_sweep:
        threshold_sweep()
    else:
        evaluate_against_gt()


if __name__ == "__main__":
    main()
