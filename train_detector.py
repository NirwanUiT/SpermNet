#!/usr/bin/env python3
"""
train_detector.py

Train YOLOv8n on the VISEM-Tracking sperm detection dataset.

Prerequisites:
    1. Run convert_annotations.py first to set up the YOLO data layout
    2. The visem.yaml dataset config must exist

Usage:
    python train_detector.py
    python train_detector.py --epochs 100 --batch 8 --resume
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from ultralytics import YOLO


def train(
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 10,
    resume: bool = False,
    model_name: str = "yolov8n.pt",
):
    """Train YOLOv8n on the VISEM-Tracking dataset."""

    yaml_path = config.ANNOTATIONS_DIR / "visem.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found.")
        print("Run convert_annotations.py first.")
        sys.exit(1)

    # Weights output directory
    weights_dir = config.PROJECT_ROOT / "detection" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  YOLOv8 TRAINING — VISEM-Tracking Sperm Detection")
    print("=" * 60)
    print(f"  Model:     {model_name}")
    print(f"  Dataset:   {yaml_path}")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Patience:  {patience}")
    print("=" * 60)

    # Load model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(config.OUTPUTS_DIR / "training"),
        name="sperm_yolov8n",
        exist_ok=True,
        resume=resume,
        verbose=True,
        plots=True,           # generates training curves
        save=True,
        save_period=10,       # save checkpoint every 10 epochs
        workers=4,
        device="0",           # use first GPU; change to "cpu" if no GPU
        amp=True,             # mixed precision
    )

    # Copy best weights
    train_dir = config.OUTPUTS_DIR / "training" / "sperm_yolov8n"
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
    parser = argparse.ArgumentParser(description="Train YOLOv8n on VISEM-Tracking")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model (default: yolov8n.pt)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        resume=args.resume,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
