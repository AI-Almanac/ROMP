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

    false_alarm_rate_map = np.full((len(lats), len(lons)), np.nan)
    miss_rate_map = np.full((len(lats), len(lons)), np.nan)
    mean_mae_map = np.full((len(lats), len(lons)), np.nan)

    yearly_mae_maps = {}
    for year in metrics_df_dict.keys():
        yearly_mae_maps[year] = np.full((len(lats), len(lons)), np.nan)

    for i, lat_val in enumerate(lats):
        for j, lon_val in enumerate(lons):

            total_FP = 0
            total_TN = 0
            total_FN = 0
            total_num_onset = 0

            mae_values = []
            has_any_valid_data = False

            for year, metrics_df in metrics_df_dict.items():
                obs_onset_val = onset_da_dict[year].isel(lat=i, lon=j).values

                if pd.isna(obs_onset_val):
                    continue

                grid_data = metrics_df[(metrics_df['lat'] == lat_val) & (metrics_df['lon'] == lon_val)]

                if len(grid_data) > 0:
                    has_any_valid_data = True
                    row = grid_data.iloc[0]

                    total_FP += row['false_positive']
                    total_TN += row['true_negative']
                    total_FN += row['false_negative']
                    total_num_onset += row['num_onset']

                    mae_val = row['mae_combined']
                    if not pd.isna(mae_val):
                        yearly_mae_maps[year][i, j] = mae_val
                        mae_values.append(mae_val)

            if has_any_valid_data:
                if (total_FP + total_TN) > 0:
                    false_alarm_rate_map[i, j] = total_FP / (total_FP + total_TN)
                else:
                    false_alarm_rate_map[i, j] = 0

                if total_num_onset > 0:
                    miss_rate_map[i, j] = total_FN / total_num_onset
                else:
                    miss_rate_map[i, j] = 0

                if len(mae_values) > 0:
                    mean_mae_map[i, j] = np.mean(mae_values)

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


