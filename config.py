#!/usr/bin/env python3
"""
config.py

Centralised configuration for the sperm motility analysis pipeline.
All paths, model parameters, and analysis thresholds live here.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
FRAMES_DIR      = DATA_DIR / "frames"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# ── VISEM-Tracking dataset paths ─────────────────────────────────────────────
VISEM_ROOT      = RAW_DIR / "VISEM_Tracking_Train_v4" / "Train"
VISEM_VIDEOS_DIR = RAW_DIR / "visem-extracted-30s-selected-20-videos-excluding-first-30s" / "visem-extracted-30s-selected-20-videos-excluding-first-30s"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
DETECT_OUT      = OUTPUTS_DIR / "detections"
TRACK_OUT       = OUTPUTS_DIR / "tracks"
EVENTS_OUT      = OUTPUTS_DIR / "events"
REPORTS_OUT     = OUTPUTS_DIR / "reports"
VIS_OUT         = OUTPUTS_DIR / "visualisations"
MARKOV_OUT      = OUTPUTS_DIR / "markov"
TEMPORAL_OUT    = OUTPUTS_DIR / "temporal"

# ── Detection settings ────────────────────────────────────────────────────────
YOLO_MODEL       = "yolov8n.pt"          # base model for fine-tuning / inference
YOLO_IMGSZ       = 640                   # input image size
YOLO_CONF        = 0.25                  # confidence threshold
YOLO_IOU         = 0.45                  # NMS IoU threshold
YOLO_EPOCHS      = 50                    # training epochs (fine-tuning)
YOLO_BATCH       = 16                    # training batch size
CLASS_NAMES      = ["sperm", "cluster", "small_pinhead"]  # VISEM 3-class
NUM_CLASSES      = len(CLASS_NAMES)

# ── Tracking settings ─────────────────────────────────────────────────────────
TRACKER_TYPE     = "botsort.yaml"        # ultralytics tracker config (default)
TRACKER_CONFIGS  = {                     # supported trackers
    "botsort":   "botsort.yaml",
    "bytetrack": "bytetrack.yaml",
    "ocsort":    "ocsort.yaml",
}
TRACK_PERSIST    = True                  # persist tracks across frames

# ── Motility analysis (WHO 2021 thresholds) ───────────────────────────────────
# Pixel-to-micron conversion (must be calibrated per microscope setup)
PIXELS_PER_MICRON = 1.422          # 640px / 450µm FOV at 400x
FPS               = 50                   # VISEM-Tracking: 50 fps

# Curvilinear velocity (VCL) thresholds in µm/s
VCL_PROGRESSIVE_MIN   = 25.0            # VCL ≥ 25 µm/s → may be progressive
VCL_IMMOTILE_MAX      = 5.0             # VCL ≤ 5 µm/s → immotile

# Straightness (STR = VSL/VAP) threshold
STR_PROGRESSIVE_MIN   = 0.50            # STR ≥ 0.50 for progressive motility (optimised via threshold sweep)

# Minimum track length (in frames) to classify
MIN_TRACK_LENGTH = 10

# ── LLM settings ──────────────────────────────────────────────────────────────
LLM_PROVIDER     = "anthropic"            # "openai" or "anthropic"
LLM_MODEL        = "gpt-4o-mini"          # OpenAI model
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"  # Anthropic model
LLM_TEMPERATURE  = 0.3
LLM_MAX_TOKENS   = 2048

# ── Visualisation ─────────────────────────────────────────────────────────────
VIS_TRAIL_LENGTH = 30                    # number of past positions to draw
VIS_COLORS       = {
    "progressive":     (0, 255, 0),      # green
    "non_progressive": (0, 165, 255),    # orange
    "immotile":        (0, 0, 255),      # red
}

# ── Ensure output dirs exist ─────────────────────────────────────────────────
for d in [DETECT_OUT, TRACK_OUT, EVENTS_OUT, REPORTS_OUT, VIS_OUT, MARKOV_OUT, TEMPORAL_OUT]:
    d.mkdir(parents=True, exist_ok=True)
