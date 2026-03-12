"""Tests for config.py — verify paths and constants are sane."""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def test_project_root_exists():
    assert config.PROJECT_ROOT.is_dir(), f"PROJECT_ROOT does not exist: {config.PROJECT_ROOT}"


def test_output_dirs_defined():
    for attr in ["DETECT_OUT", "TRACK_OUT", "EVENTS_OUT", "REPORTS_OUT",
                 "VIS_OUT", "MARKOV_OUT", "TEMPORAL_OUT"]:
        path = getattr(config, attr, None)
        assert path is not None, f"config.{attr} is not defined"
        assert isinstance(path, Path), f"config.{attr} is not a Path"


def test_calibration_constants():
    assert config.PIXELS_PER_MICRON > 0, "PIXELS_PER_MICRON must be positive"
    assert config.FPS > 0, "FPS must be positive"
    assert config.PIXELS_PER_MICRON == 1.422, "Expected 1.422 (640px / 450µm)"
    assert config.FPS == 50, "Expected 50 fps (VISEM-Tracking)"


def test_who_thresholds():
    assert config.VCL_PROGRESSIVE_MIN > 0
    assert config.VCL_IMMOTILE_MAX > 0
    assert config.VCL_PROGRESSIVE_MIN > config.VCL_IMMOTILE_MAX, \
        "Progressive threshold must exceed immotile threshold"
    assert 0 < config.STR_PROGRESSIVE_MIN <= 1.0, \
        "STR must be in (0, 1]"
    assert config.MIN_TRACK_LENGTH >= 1


def test_class_names():
    assert len(config.CLASS_NAMES) == 3
    assert "sperm" in config.CLASS_NAMES
