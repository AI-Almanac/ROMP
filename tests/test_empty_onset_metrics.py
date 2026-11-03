"""Test that an onset table with no forecast onsets yields empty metrics."""

import pandas as pd

from hcwf.config.model import Window
from hcwf.metrics.contingency import onset_metrics


def test_year_with_no_forecast_onsets_yields_empty_metrics():
    """An empty onset table gives an empty metrics frame and zeroed summary counts."""
    metrics_df, summary = onset_metrics(pd.DataFrame(), Window(1, 15, 3), back_dated=False,
                                          default_dims=("lat", "lon"))
    assert metrics_df.empty
    assert list(metrics_df.columns[:2]) == ["lat", "lon"]
    assert summary["total_grid_points"] == 0
    assert summary["overall_true_positive"] == 0
