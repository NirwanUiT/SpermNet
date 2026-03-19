#!/usr/bin/env python3
"""
train_detector.py

Train YOLOv8 or RT-DETR on the VISEM-Tracking sperm detection dataset.

Prerequisites:
    1. Run convert_annotations.py first to set up the YOLO data layout
    2. The visem.yaml dataset config must exist

Usage — YOLO (CNN-based):
    python train_detector.py                          # YOLOv8n (default)
    python train_detector.py --model-size l           # YOLOv8l with auto-tuned LR & freeze
    python train_detector.py --model-size l --lr0 0.001 --freeze 5  # manual overrides
    python train_detector.py --model path/to/custom.pt              # custom weights
    python train_detector.py --epochs 100 --batch 8 --resume

Usage — RT-DETR (transformer-based):
    python train_detector.py --architecture rtdetr                   # RT-DETR-l (default)
    python train_detector.py --architecture rtdetr --model-size x    # RT-DETR-x
    python train_detector.py --architecture rtdetr --lr0 0.0002      # custom LR
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from ultralytics import YOLO

# ── Auto-tuning defaults for large models ──
_LARGE_MODEL_SIZES = {"l", "x"}
_LARGE_DEFAULT_FREEZE = 10
_LARGE_DEFAULT_LR0 = 0.0005

# ── RT-DETR defaults ──
_RTDETR_DEFAULT_LR0 = 0.0001
_RTDETR_DEFAULT_BATCH = 8
_RTDETR_SIZE_MAP = {"l": "rtdetr-l", "x": "rtdetr-x"}  # other sizes fall back to "rtdetr-l"


def train(
    epochs: int = 200,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 30,
    resume: bool = False,
    model_name: str = "yolov8n.pt",
    lr0: float = 0.001,
    lrf: float = 0.1,
    freeze: int = 0,
    architecture: str = "yolo",
):
    """Train YOLOv8 or RT-DETR on the VISEM-Tracking dataset.

    Args:
        lr0:          Initial learning rate (default 0.001; use lower for
                      large pretrained models / transformers).
        lrf:          Final LR multiplier (final_lr = lr0 * lrf).
        freeze:       Number of backbone layers to freeze (0 = train all;
                      ignored for RT-DETR).
        architecture: "yolo" or "rtdetr".
    """

    yaml_path = config.ANNOTATIONS_DIR / "visem.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found.")
        print("Run convert_annotations.py first.")
        sys.exit(1)

    # Derive run name from model variant
    variant = Path(model_name).stem  # e.g. "yolov8l"
    run_name = f"sperm_{variant}"

    # Weights output directory
    weights_dir = config.PROJECT_ROOT / "detection" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    arch_label = "RT-DETR" if architecture == "rtdetr" else "YOLOv8"
    print("=" * 60)
    print(f"  {arch_label} TRAINING — VISEM-Tracking Sperm Detection")
    print("=" * 60)
    print(f"  Architecture: {architecture}")
    print(f"  Model:        {model_name}")
    print(f"  Dataset:      {yaml_path}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch:        {batch}")
    print(f"  Image size:   {imgsz}")
    print(f"  Patience:     {patience}")
    print(f"  lr0:          {lr0}")
    print(f"  lrf:          {lrf}  (final LR = {lr0 * lrf:.6f})")
    if architecture == "yolo":
        print(f"  Freeze:       {freeze} backbone layers")
    else:
        print(f"  Freeze:       N/A (RT-DETR)")
    print("=" * 60)

    # Load model
    model = YOLO(model_name)

    # Build train kwargs — shared between architectures
    train_kwargs = dict(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(config.OUTPUTS_DIR / "training"),
        name=run_name,
        exist_ok=True,
        resume=resume,
        verbose=True,
        plots=True,           # generates training curves
        save=True,
        save_period=10,       # save checkpoint every 10 epochs
        workers=8,
        device="0",           # use first GPU; change to "cpu" if no GPU
        amp=True,             # mixed precision
        cos_lr=True,          # cosine LR schedule
        lr0=lr0,
        lrf=lrf,              # final LR = lr0 * lrf
    )

    if architecture == "rtdetr":
        # RT-DETR (transformer): no mosaic, AdamW optimizer, no freeze
        train_kwargs.update(
            optimizer="AdamW",   # transformers train best with AdamW
            mosaic=0.0,          # mosaic disabled for RT-DETR
            close_mosaic=0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mixup=0.0,           # no mixup for transformers
        )
    else:
        # YOLO (CNN): full augmentation pipeline, SGD
        train_kwargs.update(
            optimizer="SGD",     # explicit SGD — 'auto' overrides lr0!
            mosaic=1.0,          # mosaic augmentation
            close_mosaic=20,     # disable mosaic last 20 epochs
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,        # rotation
            translate=0.1,
            scale=0.5,
            flipud=0.5,          # sperm swim any direction
            fliplr=0.5,
            mixup=0.1,           # light mixup
        )
        if freeze > 0:
            train_kwargs["freeze"] = freeze

    # Train
    results = model.train(**train_kwargs)

    # Copy best weights
    train_dir = config.OUTPUTS_DIR / "training" / run_name
    best_src = train_dir / "weights" / "best.pt"
    best_dst = weights_dir / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n  ✅ Best weights saved: {best_dst}")
    else:
        print(f"\n  WARNING: best.pt not found at {best_src}")

    last_src = train_dir / "weights" / "last.pt"
    last_dst = weights_dir / "last.pt"
    if last_src.exists():
        shutil.copy2(last_src, last_dst)
        print(f"  ✅ Last weights saved: {last_dst}")

    # Copy training curves
    curves_src = train_dir / "results.png"
    curves_dst = config.OUTPUTS_DIR / "training_curves.png"
    if curves_src.exists():
        shutil.copy2(curves_src, curves_dst)
        print(f"  ✅ Training curves:   {curves_dst}")

    # Also copy confusion matrix if available
    for plot_name in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                      "P_curve.png", "R_curve.png", "F1_curve.png", "PR_curve.png"]:
        src = train_dir / plot_name
        if src.exists():
            shutil.copy2(src, config.OUTPUTS_DIR / plot_name)

    # Print validation results
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Training directory: {train_dir}")
    print(f"  Best weights:      {best_dst}")
    print(f"  Training curves:   {curves_dst}")

    # Run validation on the best model
    print(f"\n  Running validation on best model ...")
    best_model = YOLO(str(best_dst))
    val_results = best_model.val(data=str(yaml_path), imgsz=imgsz, verbose=True)

    print(f"\n  Validation mAP@0.5:     {val_results.box.map50:.4f}")
    print(f"  Validation mAP@0.5:0.95: {val_results.box.map:.4f}")

    # Per-class results
    if hasattr(val_results.box, 'maps') and val_results.box.maps is not None:
        class_names = ["sperm", "cluster", "small_pinhead"]
        for i, name in enumerate(class_names):
            if i < len(val_results.box.maps):
                print(f"  {name:>14} AP@0.5: {val_results.box.maps[i]:.4f}")

    print(f"\n{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 or RT-DETR on VISEM-Tracking")
    parser.add_argument("--architecture", type=str, default="yolo",
                        choices=["yolo", "rtdetr"],
                        help="Model family: 'yolo' (CNN) or 'rtdetr' "
                             "(transformer). Default: yolo.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size (default: 16 for YOLO, 8 for RT-DETR)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Custom model path (overrides --model-size)")
    parser.add_argument("--model-size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="Model variant size (default: n). "
                             "Ignored when --model is provided.")
    parser.add_argument("--lr0", type=float, default=None,
                        help="Initial learning rate (default: 0.001 YOLO, "
                             "0.0001 RT-DETR; auto-set for l/x YOLO models)")
    parser.add_argument("--lrf", type=float, default=0.1,
                        help="Final LR = lr0 * lrf (default: 0.1)")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Backbone layers to freeze (default: 0; "
                             "auto-set to 10 for l/x YOLO models; "
                             "ignored for RT-DETR)")
    args = parser.parse_args()

    architecture = args.architecture

    # ── Resolve model name ──
    if args.model is not None:
        model_name = args.model
        stem = Path(model_name).stem.lower()
        model_size = stem[-1] if stem[-1] in {"n", "s", "m", "l", "x"} else "n"
    elif architecture == "rtdetr":
        # Map size → RT-DETR variant; default to rtdetr-l
        rtdetr_variant = _RTDETR_SIZE_MAP.get(args.model_size, "rtdetr-l")
        model_name = f"{rtdetr_variant}.pt"
        model_size = args.model_size
    else:
        model_size = args.model_size
        model_name = f"yolov8{model_size}.pt"

    # ── Resolve hyperparameters ──
    lr0 = args.lr0
    freeze = args.freeze
    lr0_explicit = args.lr0 is not None
    freeze_explicit = args.freeze is not None
    batch = args.batch

    if architecture == "rtdetr":
        # RT-DETR-specific defaults
        if not lr0_explicit:
            lr0 = _RTDETR_DEFAULT_LR0
        if batch is None:
            batch = _RTDETR_DEFAULT_BATCH
        freeze = 0  # freeze is not used for RT-DETR
        print(f"  Using RT-DETR config: lr0={lr0}, batch={batch}, no mosaic")
    else:
        # YOLO auto-tuning for large models
        if model_size in _LARGE_MODEL_SIZES:
            if not freeze_explicit:
                freeze = _LARGE_DEFAULT_FREEZE
            if not lr0_explicit:
                lr0 = _LARGE_DEFAULT_LR0
            if not freeze_explicit or not lr0_explicit:
                print(f"  Auto-tuning: freeze={freeze}, lr0={lr0} "
                      f"for large model fine-tuning ({model_name})")

    # Fall back to standard defaults if not set by auto-tuning or CLI
    if lr0 is None:
        lr0 = 0.001
    if freeze is None:
        freeze = 0
    if batch is None:
        batch = 16

    train(
        epochs=args.epochs,
        batch=batch,
        imgsz=args.imgsz,
        patience=args.patience,
        resume=args.resume,
        model_name=model_name,
        lr0=lr0,
        lrf=args.lrf,
        freeze=freeze,
        architecture=architecture,
    )


if __name__ == "__main__":
    main()
