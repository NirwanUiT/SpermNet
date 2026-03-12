# Sperm Motility Analysis Pipeline

Automated sperm motility analysis from microscopy video using deep learning
detection, multi-object tracking, and WHO 2021 kinematic classification.

Built on the [VISEM-Tracking](https://zenodo.org/records/7293726) dataset
(20 videos, 640×480, 50 fps, 400× magnification).

---

## Architecture

```
Raw Video / GT Annotations
        │
        ▼
┌──────────────────┐     ┌──────────────────┐
│  YOLOv8 Detector │ or  │  GT Label Parser  │
│  (detection/)    │     │  (run_single_gt)  │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         ▼                        ▼
   ┌─────────────────────────────────┐
   │   BoT-SORT Tracker (tracking/) │
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │  WHO Motility Metrics (events/) │
   │  VCL, VSL, VAP, LIN, STR, WOB, │
   │  ALH, BCF + quality filters     │
   └──────────────┬──────────────────┘
                  │
        ┌─────────┼──────────┬──────────────┐
        ▼         ▼          ▼              ▼
   ┌─────────┐ ┌───────┐ ┌──────────┐ ┌──────────┐
   │ Classify │ │ Plots │ │  Markov  │ │ Temporal │
   │ P/NP/Im │ │ (vis) │ │  Chain   │ │ Dynamics │
   └────┬────┘ └───────┘ └──────────┘ └──────────┘
        │
        ▼
   ┌─────────────────────────────────┐
   │  LLM Clinical Report (llm/)    │
   │  + Interactive Chatbot          │
   └─────────────────────────────────┘
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/NirwanUiT/WP4.git
cd WP4/sperm_analysis
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python download_data.py          # downloads from Zenodo (~5.9 GB)
python extract_frames.py --all   # extract video frames
```

### 3. Run the pipeline (ground-truth annotations)

```bash
# Process all 20 videos
python run_all_gt.py

# Specific videos only
python run_all_gt.py --videos 14 29

# Skip visualisations for speed
python run_all_gt.py --skip-vis

# Skip Markov/temporal research analyses
python run_all_gt.py --skip-research
```

### 4. Run with YOLO detection (trained model)

```bash
python run_pipeline.py --all
python run_single.py 14
```

### 5. Evaluation

```bash
python evaluate.py                # GT comparison, Bland-Altman, correlation
python evaluate_ablation.py       # 3-way ablation (GT-ceiling vs YOLOv8n vs YOLOv8l)
```

### 6. Research analyses (standalone)

```bash
python markov_analysis.py         # Markov chain transition analysis
python temporal_analysis.py       # temporal motility dynamics
```

### 7. LLM chatbot

```bash
# Set your OpenAI API key
cp .env.example .env
# Edit .env with your key

# Interactive chatbot
python -m llm.chatbot 14          # for video 14
python -m llm.chatbot --all       # load all 20 videos
```

---

## Project Structure

```
sperm_analysis/
├── config.py                 # Centralised paths, thresholds, settings
├── run_all_gt.py             # Batch pipeline (GT annotations)
├── run_pipeline.py           # Batch pipeline (YOLO detection)
├── run_single_gt.py          # Single-video GT pipeline
├── run_single.py             # Single-video YOLO pipeline
│
├── detection/
│   └── detect_sperm.py       # YOLOv8 training + inference
├── tracking/
│   └── track_sperm.py        # BoT-SORT multi-object tracker
├── events/
│   └── detect_events.py      # WHO motility metrics, classification, quality filters
├── llm/
│   ├── analyze.py            # One-shot clinical report generation
│   └── chatbot.py            # Multi-turn interactive chatbot
├── visualise.py              # Per-video plots (trajectories, motility, heatmaps)
├── visualise_aggregate.py    # Cross-video aggregate visualisations
│
├── markov_analysis.py        # Markov chain transition analysis
├── temporal_analysis.py      # Temporal motility dynamics
├── evaluate.py               # GT evaluation (MAE, RMSE, Pearson, Bland-Altman)
├── evaluate_ablation.py      # 3-way detector ablation study
│
├── calibrate.py              # Microscope pixel-to-micron calibration
├── download_data.py          # Zenodo dataset downloader
├── extract_frames.py         # Video → frame extraction
├── convert_annotations.py    # VISEM annotation → YOLO format
├── prepare_annotations.py    # Dataset split preparation
├── train_detector.py         # YOLOv8 fine-tuning script
│
├── demo_pipeline.ipynb       # Full interactive demo notebook
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
│
├── data/                     # Dataset (not tracked)
│   ├── raw/                  # VISEM-Tracking videos + annotations
│   ├── frames/               # Extracted video frames
│   └── annotations/          # YOLO-format annotations
│
├── outputs/                  # Pipeline outputs (not tracked)
│   ├── tracks/               # Per-video track CSVs
│   ├── events/               # Motility metrics + summaries
│   ├── reports/              # LLM clinical reports
│   ├── visualisations/       # Plots + tracked videos
│   ├── evaluation/           # GT comparison metrics + plots
│   ├── evaluation_ablation/  # Ablation study results
│   ├── markov/               # Transition matrices + heatmaps
│   └── temporal/             # Temporal dynamics CSVs + plots
│
├── detection/weights/        # Trained YOLO weights (not tracked)
└── tests/                    # Unit tests
```

## Calibration

| Parameter | Value | Source |
|-----------|-------|--------|
| FOV width | 450 µm | 400× magnification standard |
| Image width | 640 px | VISEM-Tracking videos |
| `PIXELS_PER_MICRON` | 1.422 | 640 / 450 |
| `FPS` | 50 | VISEM-Tracking specification |

## WHO 2021 Motility Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VCL_PROGRESSIVE_MIN` | 25.0 µm/s | Minimum VCL for progressive motility |
| `STR_PROGRESSIVE_MIN` | 0.50 | Minimum straightness (optimised via threshold sweep) |
| `VCL_IMMOTILE_MAX` | 5.0 µm/s | Maximum VCL for immotile classification |
| `MIN_TRACK_LENGTH` | 10 frames | Minimum track length for analysis |

## Quality Filters

Applied post-tracking, before motility classification:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Low confidence | mean conf < 0.4 | Remove uncertain detections |
| Unrealistic VCL | VCL > 200 µm/s | Remove detection noise |
| Jitter | VCL > 20 & LIN < 0.02 | Remove stationary noise tracks |
| Short duration | < 0.3 s | Supplement minimum frame count |

## Evaluation Results

**GT-ceiling pipeline** (threshold-optimised, quality-filtered):

| Metric | Progressive | Non-progressive | Immotile |
|--------|-------------|-----------------|----------|
| MAE (%) | 9.4 | 10.6 | 13.9 |

**Detector ablation** (3-way comparison):

| Detector | Immotile MAE (%) | Notes |
|----------|------------------|-------|
| GT-ceiling | 16.1 | Upper bound |
| YOLOv8n | 17.9 | Lightweight |
| YOLOv8l | 21.7 | Larger model, early-stopped |

## Dataset

[VISEM-Tracking](https://zenodo.org/records/7293726) — 20 training videos of
human spermatozoa recorded at 50 fps, 640×480 resolution, 400× magnification.

3 annotated classes: sperm, cluster, small/pinhead.

## License

This project is for academic/research purposes.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{sperm_motility_pipeline_2026,
  title={Automated Sperm Motility Analysis Pipeline Using Deep Learning and WHO 2021 Guidelines},
  author={N. Barnard},
  year={2026},
  url={https://github.com/NirwanUiT/WP4}
}
```
