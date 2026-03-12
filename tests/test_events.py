"""Tests for events/detect_events.py — motility metrics & classification."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from events.detect_events import (
    px_to_um,
    frame_interval,
    classify_motility,
    classify_window,
)


# ── px_to_um ──────────────────────────────────────────────────────────────────

def test_px_to_um_zero():
    assert px_to_um(0) == 0.0


def test_px_to_um_known():
    result = px_to_um(config.PIXELS_PER_MICRON)
    assert abs(result - 1.0) < 1e-6, "1 PIXELS_PER_MICRON pixels should equal 1 µm"


def test_px_to_um_array():
    arr = np.array([0, config.PIXELS_PER_MICRON, 2 * config.PIXELS_PER_MICRON])
    result = px_to_um(arr)
    np.testing.assert_allclose(result, [0, 1.0, 2.0], atol=1e-6)


# ── frame_interval ────────────────────────────────────────────────────────────

def test_frame_interval():
    dt = frame_interval()
    assert abs(dt - 1.0 / config.FPS) < 1e-12


# ── classify_motility ─────────────────────────────────────────────────────────

def test_classify_immotile():
    row = {"VCL": 2.0, "STR": 0.9}
    assert classify_motility(row) == "immotile"


def test_classify_progressive():
    row = {"VCL": 50.0, "STR": 0.8}
    assert classify_motility(row) == "progressive"


def test_classify_non_progressive_low_str():
    row = {"VCL": 50.0, "STR": 0.2}
    assert classify_motility(row) == "non_progressive"


def test_classify_non_progressive_low_vcl():
    row = {"VCL": 15.0, "STR": 0.9}
    assert classify_motility(row) == "non_progressive"


def test_classify_boundary_immotile():
    row = {"VCL": config.VCL_IMMOTILE_MAX, "STR": 1.0}
    assert classify_motility(row) == "immotile"


# ── classify_window ───────────────────────────────────────────────────────────

def test_classify_window_too_short():
    """Windows with < 3 points should return Immotile."""
    xs = np.array([100.0, 101.0])
    ys = np.array([200.0, 200.0])
    dt = 1.0 / config.FPS
    assert classify_window(xs, ys, dt) == "Immotile"


def test_classify_window_stationary():
    """Stationary sperm should be Immotile."""
    xs = np.full(30, 100.0)
    ys = np.full(30, 200.0)
    dt = 1.0 / config.FPS
    assert classify_window(xs, ys, dt) == "Immotile"


def test_classify_window_fast_straight():
    """Fast straight-line motion should be Progressive."""
    dt = 1.0 / config.FPS
    # Move 5 px per frame in x → ~3.5 µm/frame → ~175 µm/s at 50fps
    n = 30
    xs = np.arange(n, dtype=float) * 5.0
    ys = np.zeros(n)
    result = classify_window(xs, ys, dt)
    assert result == "Progressive", f"Expected Progressive, got {result}"


def test_classify_window_returns_valid_state():
    """Any output must be one of the 3 valid states."""
    dt = 1.0 / config.FPS
    xs = np.random.RandomState(42).randn(25).cumsum() * 2
    ys = np.random.RandomState(43).randn(25).cumsum() * 2
    result = classify_window(xs, ys, dt)
    assert result in ("Progressive", "Non-progressive", "Immotile")
