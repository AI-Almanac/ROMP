"""Aggregate per-year onset verification metrics into spatial FAR, MR and MAE maps."""
import numpy as np
import xarray as xr
import pandas as pd


def create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
    """Create spatial maps of False Alarm Rate, Miss Rate, yearly MAE, and mean MAE across years.

    Parameters
    ----------
    metrics_df_dict : dict
        Maps year to a per-grid-point metrics DataFrame (lat, lon, false_positive,
        true_negative, false_negative, num_onset, mae_combined).
    onset_da_dict : dict
        Maps year to the observed onset DataArray; its grid defines the output
        grid, and grid points with NaN onset in a year are skipped for that year.

    Returns
    -------
    dict
        'false_alarm_rate', 'miss_rate', 'mean_mae' and 'mae_<year>' lat/lon DataArrays.
    """
    first_year = list(onset_da_dict.keys())[0]
    lats = onset_da_dict[first_year].lat.values
    lons = onset_da_dict[first_year].lon.values

    print(f"Creating spatial FAR, Miss Rate, yearly MAE, and mean MAE maps...")
    print(f"Grid dimensions: {len(lats)} lats x {len(lons)} lons")
    print(f"Years: {list(metrics_df_dict.keys())}")

    spatial_metrics = {}

    shape = (len(lats), len(lons))
    grid = pd.MultiIndex.from_product([lats, lons])
    totals = {name: np.zeros(shape) for name in ('false_positive', 'true_negative', 'false_negative', 'num_onset')}
    has_any_valid_data = np.zeros(shape, dtype=bool)
    mae_sum = np.zeros(shape)
    mae_count = np.zeros(shape, dtype=np.int64)
    yearly_mae_maps = {}

    for year, metrics_df in metrics_df_dict.items():
        yearly_mae_maps[year] = np.full(shape, np.nan)
        if len(metrics_df) == 0:
            continue

        # a grid point's first row stands for it; points without an observed onset this year are skipped
        points = metrics_df.drop_duplicates(['lat', 'lon'])
        flat = grid.get_indexer(pd.MultiIndex.from_frame(points[['lat', 'lon']]))
        in_grid = flat >= 0
        points, flat = points[in_grid], flat[in_grid]
        has_onset = ~pd.isna(onset_da_dict[year].values).ravel()[flat]
        points, flat = points[has_onset], flat[has_onset]
        i, j = np.unravel_index(flat, shape)

        has_any_valid_data[i, j] = True
        for name, total in totals.items():
            total[i, j] += points[name].to_numpy()

        mae = points['mae_combined'].to_numpy(dtype=float)
        valid_mae = ~np.isnan(mae)
        yearly_mae_maps[year][i[valid_mae], j[valid_mae]] = mae[valid_mae]
        mae_sum[i[valid_mae], j[valid_mae]] += mae[valid_mae]
        mae_count[i[valid_mae], j[valid_mae]] += 1

    false_alarm_denominator = totals['false_positive'] + totals['true_negative']
    with np.errstate(invalid='ignore', divide='ignore'):
        false_alarm_rate_map = np.where(
            has_any_valid_data, np.where(false_alarm_denominator > 0, totals['false_positive'] / false_alarm_denominator, 0), np.nan)
        miss_rate_map = np.where(
            has_any_valid_data, np.where(totals['num_onset'] > 0, totals['false_negative'] / totals['num_onset'], 0), np.nan)
        mean_mae_map = np.where(mae_count > 0, mae_sum / np.maximum(mae_count, 1), np.nan)

    spatial_metrics['false_alarm_rate'] = xr.DataArray(
        false_alarm_rate_map,
        coords=[('lat', lats), ('lon', lons)],
        name='false_alarm_rate',
        attrs={'description': 'False Alarm Rate = sum(FP) / sum(FP + TN) across all valid years'}
    )

    spatial_metrics['miss_rate'] = xr.DataArray(
        miss_rate_map,
        coords=[('lat', lats), ('lon', lons)],
        name='miss_rate',
        attrs={'description': 'Miss Rate = sum(FN) / sum(total_onsets) across all valid years'}
    )

    spatial_metrics['mean_mae'] = xr.DataArray(
        mean_mae_map,
        coords=[('lat', lats), ('lon', lons)],
        name='mean_mae',
        attrs={'description': 'Mean MAE across all valid years (omitting NaN values)'}
    )

    for year, mae_map in yearly_mae_maps.items():
        spatial_metrics[f'mae_{year}'] = xr.DataArray(
            mae_map,
            coords=[('lat', lats), ('lon', lons)],
            name=f'mae_{year}',
            attrs={'description': f'Mean Absolute Error for year {year}'}
        )

    return spatial_metrics


