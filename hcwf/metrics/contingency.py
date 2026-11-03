"""Onset contingency metrics (TP/FP/FN/TN, MAE) for model or climatology forecasts across years."""

from hcwf.spatial import axes

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


def onset_metrics(onset_df, window, *, back_dated, default_dims):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows.

    Returns one row per grid point (or evaluation unit, with a unit column instead of lat and lon), in order of first
    appearance in onset_df, and summary totals.
    With forecast_history "observed" (`back_dated`, see bins.reports_back_dated) the rows also have `n_back_dated`: the
    cases whose forecast onset is dated on or before the init, which count as the first lead; 0 unless the window
    starts at lead 1. "none" runs get no such column. `window` is the Window (span and tolerance); `default_dims` are the
    spatial dims of a table with no rows (point_dims).
    """
    report_back_dated = back_dated
    tolerance_days, verification_window = window.tolerance, window.span
    log.debug(f"Computing onset metrics with tolerance = {tolerance_days} days")
    log.debug(f"Verification window starts {verification_window} days after initialization")
    log.debug(f"Forecast window length: {verification_window[1]} days")

    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]
    points = list(axes.spatial_dims(onset_df.columns) if len(onset_df.columns) else default_dims)
    columns = [*points, 'total_forecasts', 'true_positive', 'true_negative', 'false_positive', 'false_negative',
               'num_onset', 'num_no_onset', 'mae_combined', 'mae_tp_only', 'num_tp_errors', 'num_fp_errors',
               'tolerance_days', 'verification_window', 'forecast_days']

    if onset_df.empty:
        # Small domains (e.g. a user focus area) can have no forecast onsets in a year.
        onset_df = pd.DataFrame(columns=points)
    unique_locations = onset_df[points].drop_duplicates()

    log.debug(f"Processing {len(unique_locations)} unique grid points...")

    if len(onset_df):
        day = np.timedelta64(1, "D")
        code = onset_df.groupby(points, sort=False).ngroup().to_numpy()
        n_points = len(unique_locations)
        first_row = np.unique(code, return_index=True)[1]

        obs_onset = pd.to_datetime(onset_df['obs_onset_date']).to_numpy().astype("datetime64[ns]")
        model_onset = pd.to_datetime(onset_df['onset_date']).to_numpy().astype("datetime64[ns]")
        t_init = pd.to_datetime(onset_df['init_time']).to_numpy().astype("datetime64[ns]")
        # every row of a grid point shares its observed onset
        gt_grd = obs_onset[first_row][code]

        valid_window_start = t_init + forecast_bin_start * day
        valid_window_end = t_init + forecast_bin_end * day
        is_obs_onset_in_whole_window = (t_init + day <= gt_grd) & (gt_grd <= valid_window_end)

        has_model_onset = ~np.isnat(model_onset)
        # A forecast onset on or before the init (back-dated by the observed history) is
        # judged as if it fell on the first lead: it can be a hit or a false alarm in a
        # window starting at lead 1 and is ignored, like any early onset, in later windows.
        model_onset_in_window = np.where(has_model_onset & (model_onset <= t_init), t_init + day, model_onset)

        in_valid_window = (has_model_onset & (valid_window_start <= model_onset_in_window)
                           & (model_onset_in_window <= valid_window_end))
        with np.errstate(invalid="ignore"):
            abs_diff_days = np.abs((model_onset - gt_grd) // day)
        true_positive = in_valid_window & (abs_diff_days <= tolerance_days)
        false_positive = in_valid_window & ~true_positive
        past_window = ~has_model_onset | (~in_valid_window & (model_onset_in_window > valid_window_end))
        false_negative = past_window & is_obs_onset_in_whole_window
        true_negative = past_window & ~is_obs_onset_in_whole_window

        def count(mask):
            """Count the cases in each group where the mask holds."""
            return np.bincount(code, weights=mask, minlength=n_points).astype(np.int64)

        def error_sum(mask):
            """Sum the absolute onset error in days over the masked cases in each group."""
            return np.bincount(code, weights=np.where(mask, abs_diff_days, 0), minlength=n_points)

        tp, fp = count(true_positive), count(false_positive)
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = np.where(tp + fp > 0, (error_sum(true_positive) + error_sum(false_positive)) / (tp + fp), np.nan)
            mae_tp_only = np.where(tp > 0, error_sum(true_positive) / tp, np.nan)
        back_dated = has_model_onset & (model_onset <= t_init)
        metrics_df = pd.DataFrame({
            **{point: unique_locations[point].to_numpy() for point in points},
            'total_forecasts': np.bincount(code, minlength=n_points),
            'true_positive': tp,
            'true_negative': count(true_negative),
            'false_positive': fp,
            'false_negative': count(false_negative),
            'num_onset': count(is_obs_onset_in_whole_window),
            'num_no_onset': count(~is_obs_onset_in_whole_window),
            'mae_combined': mae,
            'mae_tp_only': mae_tp_only,
            'num_tp_errors': tp,
            'num_fp_errors': fp,
            'tolerance_days': tolerance_days,
            'verification_window': [verification_window] * n_points,
            'forecast_days': forecast_bin_end,
        })
        if report_back_dated:
            metrics_df['n_back_dated'] = count(back_dated) if forecast_bin_start == 1 else np.zeros(n_points, np.int64)
    else:
        metrics_df = pd.DataFrame(columns=columns[:len(points) + 9] + (['n_back_dated'] if report_back_dated else []))

    summary_stats = {
        'total_grid_points': len(metrics_df),
        'total_forecasts': metrics_df['total_forecasts'].sum(),
        'overall_true_positive': metrics_df['true_positive'].sum(),
        'overall_true_negative': metrics_df['true_negative'].sum(),
        'overall_false_positive': metrics_df['false_positive'].sum(),
        'overall_false_negative': metrics_df['false_negative'].sum(),
        'overall_num_onset': metrics_df['num_onset'].sum(),
        'overall_num_no_onset': metrics_df['num_no_onset'].sum(),
        'overall_mae_combined': metrics_df['mae_combined'].mean(),
        'overall_mae_tp_only': metrics_df['mae_tp_only'].mean(),
        'tolerance_days': tolerance_days,
        'verification_window': verification_window,
        'forecast_days': forecast_bin_end
    }

    return metrics_df, summary_stats
