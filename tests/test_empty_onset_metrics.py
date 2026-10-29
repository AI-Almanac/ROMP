"""Test that an onset table with no forecast onsets yields empty metrics."""

import pandas as pd

from momp.stats.benchmark import compute_onset_metrics_with_windows


def test_year_with_no_forecast_onsets_yields_empty_metrics():
    """An empty onset table gives an empty metrics frame and zeroed summary counts."""
    metrics_df, summary = compute_onset_metrics_with_windows(
        pd.DataFrame(), tolerance_days=3, verification_window=(1, 15)
    )
    assert metrics_df.empty
    assert list(metrics_df.columns[:2]) == ["lat", "lon"]
    assert summary["total_grid_points"] == 0
    assert summary["overall_true_positive"] == 0
