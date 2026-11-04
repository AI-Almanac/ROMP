"""Aggregate per-year onset verification metrics into spatial FAR, MR and MAE maps."""
import numpy as np
import xarray as xr
import pandas as pd
import logging

from hcwf.spatial import axes

log = logging.getLogger(__name__)


def year_points(metrics_df, onset_da, spatial, coords):
    """The rows of a year's metrics table that stand for a location of the output grid, and where they sit on it.

    A location's first row stands for it; locations outside the grid and those without an observed onset this year
    are skipped. Returns (rows, index) with one index array per spatial dim (lat and lon, or unit).
    """
    shape = tuple(len(coords[d]) for d in spatial)
    points = metrics_df.drop_duplicates(list(spatial))
    flat = axes.flat_positions(points, spatial, coords)
    in_grid = flat >= 0
    points, flat = points[in_grid], flat[in_grid]
    has_onset = ~pd.isna(onset_da.transpose(*spatial).values).ravel()[flat]
    points, flat = points[has_onset], flat[has_onset]
    return points, np.unravel_index(flat, shape)


def create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
    """Create spatial maps of False Alarm Rate, Miss Rate, yearly MAE, and mean MAE across years.

    Parameters
    ----------
    metrics_df_dict : dict
        Maps year to a per-grid-point metrics DataFrame (lat and lon, or unit; false_positive,
        true_negative, false_negative, num_onset, mae_combined).
    onset_da_dict : dict
        Maps year to the observed onset DataArray; its grid defines the output
        grid, and grid points with NaN onset in a year are skipped for that year.

    Returns
    -------
    dict
        'false_alarm_rate', 'miss_rate', 'mean_mae' and 'mae_<year>' lat/lon (or unit) DataArrays.
    """
    first_year = list(onset_da_dict.keys())[0]
    spatial = axes.spatial_dims(onset_da_dict[first_year].dims)
    coords = axes.coords_of(onset_da_dict[first_year], spatial)

    log.debug("Creating spatial FAR, Miss Rate, yearly MAE, and mean MAE maps...")
    log.debug(f"Grid dimensions: {' x '.join(f'{len(coords[d])} {d}' for d in spatial)}")
    log.debug(f"Years: {list(metrics_df_dict.keys())}")

    spatial_metrics = {}

    shape = tuple(len(coords[d]) for d in spatial)
    totals = {name: np.zeros(shape) for name in ('false_positive', 'true_negative', 'false_negative', 'num_onset')}
    has_any_valid_data = np.zeros(shape, dtype=bool)
    mae_sum = np.zeros(shape)
    mae_count = np.zeros(shape, dtype=np.int64)
    yearly_mae_maps = {}

    for year, metrics_df in metrics_df_dict.items():
        yearly_mae_maps[year] = np.full(shape, np.nan)
        if len(metrics_df) == 0:
            continue

        points, at = year_points(metrics_df, onset_da_dict[year], spatial, coords)

        has_any_valid_data[at] = True
        for name, total in totals.items():
            total[at] += points[name].to_numpy()

        mae = points['mae_combined'].to_numpy(dtype=float)
        valid_mae = ~np.isnan(mae)
        valid_at = tuple(ix[valid_mae] for ix in at)
        yearly_mae_maps[year][valid_at] = mae[valid_mae]
        mae_sum[valid_at] += mae[valid_mae]
        mae_count[valid_at] += 1

    false_alarm_denominator = totals['false_positive'] + totals['true_negative']
    with np.errstate(invalid='ignore', divide='ignore'):
        false_alarm_rate_map = np.where(
            has_any_valid_data, np.where(false_alarm_denominator > 0, totals['false_positive'] / false_alarm_denominator, 0), np.nan)
        miss_rate_map = np.where(
            has_any_valid_data, np.where(totals['num_onset'] > 0, totals['false_negative'] / totals['num_onset'], 0), np.nan)
        mean_mae_map = np.where(mae_count > 0, mae_sum / np.maximum(mae_count, 1), np.nan)

    spatial_metrics['false_alarm_rate'] = xr.DataArray(
        false_alarm_rate_map,
        coords=[(d, coords[d]) for d in spatial],
        name='false_alarm_rate',
        attrs={'description': 'False Alarm Rate = sum(FP) / sum(FP + TN) across all valid years'}
    )

    spatial_metrics['miss_rate'] = xr.DataArray(
        miss_rate_map,
        coords=[(d, coords[d]) for d in spatial],
        name='miss_rate',
        attrs={'description': 'Miss Rate = sum(FN) / sum(total_onsets) across all valid years'}
    )

    spatial_metrics['mean_mae'] = xr.DataArray(
        mean_mae_map,
        coords=[(d, coords[d]) for d in spatial],
        name='mean_mae',
        attrs={'description': 'Mean MAE across all valid years (omitting NaN values)'}
    )

    for year, mae_map in yearly_mae_maps.items():
        spatial_metrics[f'mae_{year}'] = xr.DataArray(
            mae_map,
            coords=[(d, coords[d]) for d in spatial],
            name=f'mae_{year}',
            attrs={'description': f'Mean Absolute Error for year {year}'}
        )

    return spatial_metrics


