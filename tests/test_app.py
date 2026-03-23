#!/usr/bin/env python3
"""
tests/test_app.py

Unit tests for app.py UI helper functions.
Tests cover demo-data generation, motility analysis, summary HTML building,
and report generation — all without requiring a GPU or dataset files.
"""

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (
    generate_demo_tracks,
    run_analysis,
    build_summary_html,
    generate_report,
    on_load_demo,
    on_run_analysis,
    _build_inline_context,
)


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generation
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDemoTracks:
    def test_returns_dataframe(self):
        df = generate_demo_tracks(n_tracks=20, n_frames=100)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_demo_tracks(n_tracks=20, n_frames=100)
        for col in ("track_id", "frame", "cx", "cy", "x1", "y1", "x2", "y2", "conf"):
            assert col in df.columns, f"Missing column: {col}"

    def test_track_count(self):
        df = generate_demo_tracks(n_tracks=30, n_frames=100, seed=0)
        assert df["track_id"].nunique() == 30

    def test_coordinates_in_frame(self):
        df = generate_demo_tracks(n_tracks=20, n_frames=100, seed=1)
        assert df["cx"].between(0, 640).all()
        assert df["cy"].between(0, 480).all()

    def test_confidence_in_range(self):
        df = generate_demo_tracks(n_tracks=20, n_frames=100, seed=2)
        assert df["conf"].between(0.0, 1.0).all()

    def test_reproducible_with_seed(self):
        df1 = generate_demo_tracks(seed=42)
        df2 = generate_demo_tracks(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_demo_tracks(seed=1)
        df2 = generate_demo_tracks(seed=2)
        assert not df1["cx"].equals(df2["cx"])


# ─────────────────────────────────────────────────────────────────────────────
# run_analysis
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def demo_tracks_df():
    return generate_demo_tracks(n_tracks=60, n_frames=200, seed=7)


class TestRunAnalysis:
    def test_returns_dataframe_and_summary(self, demo_tracks_df):
        metrics_df, summary = run_analysis(demo_tracks_df)
        assert isinstance(metrics_df, pd.DataFrame)
        assert isinstance(summary, dict)

    def test_metrics_has_motility_column(self, demo_tracks_df):
        metrics_df, _ = run_analysis(demo_tracks_df)
        assert "motility" in metrics_df.columns

    def test_motility_values_valid(self, demo_tracks_df):
        metrics_df, _ = run_analysis(demo_tracks_df)
        valid = {"progressive", "non_progressive", "immotile"}
        assert set(metrics_df["motility"].unique()).issubset(valid)

    def test_summary_keys_present(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        for key in ("total_tracks", "progressive", "non_progressive", "immotile",
                    "progressive_pct", "non_progressive_pct", "immotile_pct",
                    "mean_VCL", "mean_VSL", "mean_VAP"):
            assert key in summary, f"Missing summary key: {key}"

    def test_percentages_sum_to_100(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        total = (summary["progressive_pct"]
                 + summary["non_progressive_pct"]
                 + summary["immotile_pct"])
        # Allow a small rounding tolerance (each pct is rounded to 1 decimal)
        PERCENTAGE_TOLERANCE = 0.5
        assert abs(total - 100.0) < PERCENTAGE_TOLERANCE, f"Percentages sum to {total}, not 100"

    def test_counts_match_total(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        total = (summary["progressive"]
                 + summary["non_progressive"]
                 + summary["immotile"])
        assert total == summary["total_tracks"]

    def test_kinematic_means_positive(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        for key in ("mean_VCL", "mean_VSL", "mean_VAP"):
            assert summary[key] >= 0.0, f"{key} is negative"

    def test_empty_dataframe_returns_empty(self):
        empty_df = pd.DataFrame(columns=["track_id", "frame", "cx", "cy", "conf"])
        metrics_df, summary = run_analysis(empty_df)
        assert metrics_df.empty
        assert summary == {}


# ─────────────────────────────────────────────────────────────────────────────
# build_summary_html
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSummaryHtml:
    def test_returns_html_string(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        html = build_summary_html(summary)
        assert isinstance(html, str)
        assert "<table" in html

    def test_contains_who_compliance_section(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        html = build_summary_html(summary)
        assert "WHO 2021" in html

    def test_contains_kinematic_parameters(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        html = build_summary_html(summary)
        assert "VCL" in html
        assert "VSL" in html

    def test_empty_summary_returns_fallback(self):
        html = build_summary_html({})
        assert "No analysis" in html


# ─────────────────────────────────────────────────────────────────────────────
# generate_report (template mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_template_report_contains_sample_name(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        report = generate_report(summary, "test_vid_42", use_llm=False, provider="")
        assert "test_vid_42" in report

    def test_template_report_is_markdown(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        report = generate_report(summary, "s1", use_llm=False, provider="")
        assert "#" in report  # Markdown headings

    def test_template_report_has_who_thresholds(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        report = generate_report(summary, "s1", use_llm=False, provider="")
        assert "WHO" in report

    def test_no_summary_returns_warning(self):
        report = generate_report({}, "s1", use_llm=False, provider="")
        assert "⚠️" in report


# ─────────────────────────────────────────────────────────────────────────────
# on_load_demo / on_run_analysis (callback wrappers)
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbacks:
    def test_on_load_demo_returns_csv_and_status(self):
        csv_str, status = on_load_demo()
        assert csv_str.strip() != ""
        assert "Generated" in status
        df = pd.read_csv(StringIO(csv_str))
        assert "track_id" in df.columns

    def test_on_run_analysis_with_valid_csv(self):
        csv_str, _ = on_load_demo()
        html, metrics_csv, summary, metrics_df = on_run_analysis(csv_str, "demo")
        assert isinstance(html, str)
        assert "<table" in html
        assert isinstance(summary, dict)
        assert summary.get("total_tracks", 0) > 0

    def test_on_run_analysis_empty_csv_returns_warning(self):
        html, metrics_csv, summary, metrics_df = on_run_analysis("", "demo")
        assert "⚠️" in html or "No data" in html

    def test_on_run_analysis_default_sample_name(self):
        csv_str, _ = on_load_demo()
        html, _, summary, _ = on_run_analysis(csv_str, "")
        # Should use default "sample_01" without error
        assert isinstance(summary, dict)


# ─────────────────────────────────────────────────────────────────────────────
# _build_inline_context
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildInlineContext:
    def test_no_summary_returns_fallback(self):
        ctx = _build_inline_context({}, None, "vid_1")
        assert "No analysis" in ctx

    def test_includes_sample_name(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        ctx = _build_inline_context(summary, demo_tracks_df, "my_sample")
        assert "my_sample" in ctx

    def test_includes_json_summary(self, demo_tracks_df):
        _, summary = run_analysis(demo_tracks_df)
        ctx = _build_inline_context(summary, None, "vid_1")
        assert "total_tracks" in ctx

    def test_includes_per_track_table_when_metrics_provided(self, demo_tracks_df):
        metrics_df, summary = run_analysis(demo_tracks_df)
        ctx = _build_inline_context(summary, metrics_df, "vid_1")
        assert "track_id" in ctx.lower() or "VCL" in ctx
