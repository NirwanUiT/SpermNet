#!/usr/bin/env python3
"""
generate_architecture.py

Generates a publication-quality pipeline architecture diagram.
Output: outputs/visualisations/pipeline_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Layout constants ─────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 22
MARGIN = 0.5

# Colors
C_INPUT    = "#4A90D9"   # blue
C_DETECT   = "#E8913A"   # orange
C_TRACK    = "#50B86C"   # green
C_MOTILITY = "#9B59B6"   # purple
C_OUTPUT   = "#E74C3C"   # red
C_RESEARCH = "#1ABC9C"   # teal
C_LLM      = "#F39C12"   # amber
C_BG       = "#FAFAFA"
C_ARROW    = "#555555"
C_SUBTEXT  = "#666666"

STAGE_COLORS = [C_INPUT, C_DETECT, C_TRACK, C_MOTILITY, C_OUTPUT, C_RESEARCH, C_LLM]


def draw_box(ax, x, y, w, h, text, color, fontsize=13, subtext=None,
             alpha=0.15, bold=True, text_color=None, radius=0.02):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=(*plt.cm.colors.to_rgba(color)[:3], alpha),
        edgecolor=color, linewidth=2.0,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    tc = text_color or color
    ax.text(x, y + (0.15 if subtext else 0), text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=tc, zorder=5)
    if subtext:
        ax.text(x, y - 0.25, subtext,
                ha="center", va="center", fontsize=9.5,
                color=C_SUBTEXT, style="italic", zorder=5)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.0, style="->",
               connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        connectionstyle=connectionstyle,
        mutation_scale=18, zorder=3,
    )
    ax.add_patch(arrow)


def draw_stage_label(ax, x, y, number, label, color):
    """Draw a stage number badge + label."""
    circle = plt.Circle((x, y), 0.35, color=color, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(number), ha="center", va="center",
            fontsize=14, fontweight="bold", color="white", zorder=6)
    ax.text(x + 0.55, y, label, ha="left", va="center",
            fontsize=14, fontweight="bold", color=color, zorder=5)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(FIG_W/2, FIG_H - 0.6, "Automated Sperm Motility Analysis Pipeline",
            ha="center", va="center", fontsize=22, fontweight="bold", color="#2C3E50")
    ax.text(FIG_W/2, FIG_H - 1.1,
            "Detection  \u2192  Tracking  \u2192  WHO 2021 Classification  \u2192  LLM Reporting",
            ha="center", va="center", fontsize=12, color=C_SUBTEXT)

    # =====================================================================
    #  ROW 1: INPUT DATA (y ~ 19)
    # =====================================================================
    y1 = 19.0
    draw_stage_label(ax, 1.2, y1 + 0.8, 1, "Input Data", C_INPUT)

    draw_box(ax, 4.5, y1, 3.2, 1.2, "VISEM-Tracking\nDataset", C_INPUT, 12,
             subtext="20 videos \u00b7 50 fps \u00b7 640\u00d7480")
    draw_box(ax, 9.0, y1, 3.0, 1.2, "Video Frames", C_INPUT, 12,
             subtext="1,470 frames/video\n29,370 total")
    draw_box(ax, 13.5, y1, 3.2, 1.2, "GT Annotations", C_INPUT, 12,
             subtext="YOLO format \u00b7 3 classes\nsperm, cluster, small_pinhead")

    draw_arrow(ax, 6.1, y1, 7.5, y1)
    draw_arrow(ax, 10.5, y1, 11.9, y1)

    # =====================================================================
    #  ROW 2: DETECTION (y ~ 16.2)
    # =====================================================================
    y2 = 16.5
    draw_stage_label(ax, 1.2, y2 + 0.8, 2, "Detection", C_DETECT)

    # Main detection box
    draw_box(ax, 5.5, y2, 5.5, 1.5, "Fine-tuned Object Detector", C_DETECT, 14,
             subtext="Conf \u2265 0.25 \u00b7 NMS IoU = 0.45 \u00b7 640px input")

    # Model variants
    models_x = [11.0, 13.0, 15.0]
    models_y = y2 + 0.45
    model_names = ["YOLOv8n", "YOLOv8l", "YOLO11n"]
    model_info  = ["3.0M params", "43.6M params", "2.6M params"]
    for i, (mx, name, info) in enumerate(zip(models_x, model_names, model_info)):
        draw_box(ax, mx, models_y, 1.7, 0.7, name, C_DETECT, 10, alpha=0.10)
        ax.text(mx, models_y - 0.52, info, ha="center", va="center",
                fontsize=8, color=C_SUBTEXT)

    models_x2 = [11.0, 13.0]
    models_y2 = y2 - 0.65
    model_names2 = ["YOLOv9t", "RT-DETR-l"]
    model_info2  = ["2.0M params", "~32M params"]
    for i, (mx, name, info) in enumerate(zip(models_x2, model_names2, model_info2)):
        draw_box(ax, mx, models_y2, 1.7, 0.7, name, C_DETECT, 10, alpha=0.10)
        ax.text(mx, models_y2 - 0.52, info, ha="center", va="center",
                fontsize=8, color=C_SUBTEXT)

    # Training info box
    draw_box(ax, 15.5, y2, 3.5, 1.5, "Training", C_DETECT, 11,
             subtext="50 epochs \u00b7 batch 16 \u00b7 AMP\nA100 GPU \u00b7 1.2\u20136.7 hrs\nEarly stopping", alpha=0.08)

    draw_arrow(ax, 9.0, y1 - 0.6, 5.5, y2 + 0.75)  # frames -> detection
    draw_arrow(ax, 8.25, y2, 10.15, models_y)  # detection -> models

    # =====================================================================
    #  ROW 3: TRACKING (y ~ 13.2)
    # =====================================================================
    y3 = 13.5
    draw_stage_label(ax, 1.2, y3 + 0.8, 3, "Multi-Object Tracking", C_TRACK)

    draw_box(ax, 6.0, y3, 5.0, 1.3, "Multi-Object Tracker", C_TRACK, 14,
             subtext="Track persistence enabled \u00b7 Per-frame ID assignment")

    # Tracker variants
    draw_box(ax, 12.5, y3 + 0.3, 2.5, 0.7, "BoT-SORT", C_TRACK, 11, alpha=0.10)
    draw_box(ax, 15.3, y3 + 0.3, 2.5, 0.7, "ByteTrack", C_TRACK, 11, alpha=0.10)
    ax.text(13.9, y3 - 0.45, "(ByteTrack best for\nprogressive motility)", ha="center",
            va="center", fontsize=8.5, color=C_SUBTEXT, style="italic")

    draw_arrow(ax, 5.5, y2 - 0.75, 6.0, y3 + 0.65)  # detection -> tracking
    draw_arrow(ax, 8.5, y3, 11.25, y3 + 0.3)  # tracking -> variants

    # Output
    draw_box(ax, 6.0, y3 - 1.5, 3.5, 0.8, "Track CSVs", C_TRACK, 10,
             subtext="per-frame (x, y, w, h, conf, track_id)", alpha=0.08)
    draw_arrow(ax, 6.0, y3 - 0.65, 6.0, y3 - 1.1)

    # =====================================================================
    #  ROW 4: MOTILITY ANALYSIS (y ~ 10)
    # =====================================================================
    y4 = 10.0
    draw_stage_label(ax, 1.2, y4 + 0.8, 4, "WHO 2021 Motility Analysis", C_MOTILITY)

    # Main analysis box
    draw_box(ax, 6.0, y4, 6.0, 1.6, "Kinematic Parameter Computation", C_MOTILITY, 13,
             subtext="Pixel\u2192\u00b5m conversion (1.422 px/\u00b5m) \u00b7 Min track: 10 frames")

    # WHO parameters
    params = [
        ("VCL", "Curvilinear\nVelocity"),
        ("VSL", "Straight-Line\nVelocity"),
        ("VAP", "Average Path\nVelocity"),
        ("LIN", "Linearity\nVSL/VCL"),
        ("STR", "Straightness\nVSL/VAP"),
        ("WOB", "Wobble\nVAP/VCL"),
    ]
    px_start = 3.0
    px_step = 2.15
    py = y4 - 1.8
    for i, (name, desc) in enumerate(params):
        px = px_start + i * px_step
        draw_box(ax, px, py, 1.8, 0.9, name, C_MOTILITY, 11, subtext=desc, alpha=0.08)

    draw_arrow(ax, 6.0, y3 - 1.9, 6.0, y4 + 0.8)  # tracks -> motility

    # Classification thresholds
    y_class = y4 - 3.3
    draw_box(ax, 9.0, y_class, 10.0, 1.3,
             "Motility Classification (Optimised Thresholds)", C_MOTILITY, 13)

    # Three classes
    cls_y = y_class - 1.2
    draw_box(ax, 5.0, cls_y, 3.2, 0.9, "Progressive", "#27AE60", 12,
             subtext="VCL \u2265 25 \u00b5m/s  &  STR \u2265 0.50", alpha=0.15)
    draw_box(ax, 9.0, cls_y, 3.2, 0.9, "Non-Progressive", "#E67E22", 12,
             subtext="VCL > 5 \u00b5m/s  &  not progressive", alpha=0.15)
    draw_box(ax, 13.0, cls_y, 3.2, 0.9, "Immotile", "#C0392B", 12,
             subtext="VCL \u2264 5 \u00b5m/s", alpha=0.15)

    draw_arrow(ax, 6.0, y4 - 0.8, 9.0, y_class + 0.65)
    draw_arrow(ax, 7.0, y_class - 0.65, 5.0, cls_y + 0.45)
    draw_arrow(ax, 9.0, y_class - 0.65, 9.0, cls_y + 0.45)
    draw_arrow(ax, 11.0, y_class - 0.65, 13.0, cls_y + 0.45)

    # =====================================================================
    #  ROW 5: OUTPUTS (y ~ 3.5)
    # =====================================================================
    y5 = 3.5
    draw_stage_label(ax, 1.2, y5 + 1.5, 5, "Outputs", C_OUTPUT)

    # LLM Report
    draw_box(ax, 3.5, y5, 3.5, 1.8, "LLM Clinical\nReport", C_LLM, 13,
             subtext="Claude Sonnet 4 /\nGPT-4o-mini\nWHO 2021 interpretation")

    # Visualisations
    draw_box(ax, 8.0, y5, 3.5, 1.8, "Visualisations", C_OUTPUT, 13,
             subtext="Trajectories\nMotility bar charts\nVelocity heatmaps")

    # Evaluation
    draw_box(ax, 12.5, y5, 3.5, 1.8, "Evaluation", C_OUTPUT, 13,
             subtext="MAE, RMSE, Pearson r\nBland-Altman plots\nAblation study")

    # Research
    draw_box(ax, 16.5, y5, 2.8, 1.8, "Research\nAnalyses", C_RESEARCH, 12,
             subtext="Markov chains\nTemporal dynamics")

    # Arrows from classification to outputs
    mid_cls_y = cls_y - 0.45
    draw_arrow(ax, 5.0, mid_cls_y, 3.5, y5 + 0.9)
    draw_arrow(ax, 9.0, mid_cls_y, 8.0, y5 + 0.9)
    draw_arrow(ax, 13.0, mid_cls_y, 12.5, y5 + 0.9)
    draw_arrow(ax, 13.0, mid_cls_y, 16.5, y5 + 0.9,
               connectionstyle="arc3,rad=-0.15")

    # =====================================================================
    #  ROW 6: KEY RESULTS BANNER (y ~ 1.0)
    # =====================================================================
    y6 = 1.0
    banner = FancyBboxPatch(
        (1.5, y6 - 0.5), FIG_W - 3, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#ECF0F1", edgecolor="#BDC3C7", linewidth=1.5,
    )
    ax.add_patch(banner)
    ax.text(FIG_W/2, y6, "Key Results:  Progressive MAE = 8.01%  \u00b7  "
            "Best config: YOLOv9t + ByteTrack (MAE 7.78%, r = 0.799)  \u00b7  "
            "20 videos  \u00b7  A100 GPU  \u00b7  ~15 hrs total training",
            ha="center", va="center", fontsize=11, color="#2C3E50", fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = "outputs/visualisations/pipeline_architecture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Pipeline architecture saved to: {out_path}")


if __name__ == "__main__":
    main()
