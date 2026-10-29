"""The vectorised metrics, climatology-forecast, pair-building and spatial-map code gives the old loops' results.

The `_old_*` functions are the loop implementations they replaced, kept as references. Frames must match in
values, dtypes and row order; the only differences are the documented ones (see the tests that name them).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.metrics.error import create_spatial_far_mr_mae
from momp.stats.benchmark import compute_onset_metrics_with_windows
from momp.stats.bins import create_climatological_forecast_obs_pairs, create_forecast_observation_pairs_with_bins
from momp.stats.climatology import compute_climatology_as_forecast
from itertools import product
from datetime import datetime, timedelta


# --------------------------------------------------------------------------------------------
# Reference implementations: the loops as they were before they were vectorised.
# --------------------------------------------------------------------------------------------

def _old_has_model_onset_early(model_onset, t_init):
    """Whether a forecast onset exists and lies on or before the init date."""
    return not pd.isna(model_onset) and model_onset <= t_init


def _old_compute_onset_metrics_with_windows(onset_df, *, tolerance_days, verification_window, **kwargs):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows."""
    print(f"Computing onset metrics with tolerance = {tolerance_days} days")
    print(f"Verification window starts {verification_window} days after initialization")
    print(f"Forecast window length: {verification_window[1]} days")

    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]

    results_list = []
    if onset_df.empty:
        # Small domains (e.g. a user focus area) can have no forecast onsets in a year.
        onset_df = pd.DataFrame(columns=['lat', 'lon'])
    unique_locations = onset_df[['lat', 'lon']].drop_duplicates()

    print(f"Processing {len(unique_locations)} unique grid points...")

    for idx, (_, row) in enumerate(unique_locations.iterrows()):
        lat, lon = row['lat'], row['lon']

        grid_data = onset_df[(onset_df['lat'] == lat) & (onset_df['lon'] == lon)].copy()

        grid_data['obs_onset_dt'] = pd.to_datetime(grid_data['obs_onset_date'])
        grid_data['model_onset_dt'] = pd.to_datetime(grid_data['onset_date'])
        grid_data['init_dt'] = pd.to_datetime(grid_data['init_time'])

        TP = 0
        FP = 0
        FN = 0
        TN = 0
        num_onset = 0
        num_no_onset = 0
        mae_tp = []
        mae_fp = []

        gt_grd = grid_data['obs_onset_dt'].iloc[0]

        for _, init_row in grid_data.iterrows():
            t_init = init_row['init_dt']
            model_onset = init_row['model_onset_dt']

            # A forecast onset on or before the init (back-dated by the observed history) is
            # judged as if it fell on the first lead: it can be a hit or a false alarm in a
            # window starting at lead 1 and is ignored, like any early onset, in later windows.
            if _old_has_model_onset_early(model_onset, t_init):
                model_onset_in_window = t_init + pd.Timedelta(days=1)
            else:
                model_onset_in_window = model_onset

            valid_window_start = t_init + pd.Timedelta(days=forecast_bin_start)
            valid_window_end = t_init + pd.Timedelta(days=forecast_bin_end)

            whole_forecast_window_start = t_init + pd.Timedelta(days=1)
            whole_forecast_window_end = t_init + pd.Timedelta(days=forecast_bin_end)

            is_obs_onset_in_whole_window = whole_forecast_window_start <= gt_grd <= whole_forecast_window_end

            if is_obs_onset_in_whole_window:
                num_onset += 1
            else:
                num_no_onset += 1

            has_model_onset = not pd.isna(model_onset)

            if has_model_onset:
                is_model_in_valid_window = valid_window_start <= model_onset_in_window <= valid_window_end

                if is_model_in_valid_window:
                    abs_diff_days = abs((model_onset - gt_grd).days)

                    if abs_diff_days <= tolerance_days:
                        TP += 1
                        mae_tp.append(abs_diff_days)
                    else:
                        FP += 1
                        mae_fp.append(abs_diff_days)

                else:
                    if model_onset_in_window > valid_window_end: #make sure no model onset in early bins
                        if is_obs_onset_in_whole_window:
                            FN += 1
                        else:
                            TN += 1

            else:
                if is_obs_onset_in_whole_window:
                    FN += 1
                else:
                    TN += 1

        total_forecasts = len(grid_data)

        mae_combined = mae_tp + mae_fp
        mae = np.mean(mae_combined) if len(mae_combined) > 0 else np.nan
        mae_tp_only = np.mean(mae_tp) if len(mae_tp) > 0 else np.nan

        result = {
            'lat': lat,
            'lon': lon,
            'total_forecasts': total_forecasts,
            'true_positive': TP,
            'true_negative': TN,
            'false_positive': FP,
            'false_negative': FN,
            'num_onset': num_onset,
            'num_no_onset': num_no_onset,
            'mae_combined': mae,
            'mae_tp_only': mae_tp_only,
            'num_tp_errors': len(mae_tp),
            'num_fp_errors': len(mae_fp),
            'tolerance_days': tolerance_days,
            'verification_window': verification_window,
            'forecast_days': forecast_bin_end
        }
        results_list.append(result)

    metrics_df = pd.DataFrame(results_list)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=[
            'lat', 'lon', 'total_forecasts', 'true_positive', 'true_negative',
            'false_positive', 'false_negative', 'num_onset', 'num_no_onset',
            'mae_combined', 'mae_tp_only',
        ])

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


def _old_compute_climatology_as_forecast(climatological_onset_doy, year, init_dates, observed_onset_da,
                                   *, max_forecast_day, earliest_onset_date, **kwargs):
    """
    Use climatology as a forecast model for the given initialization dates.
    Only processes forecasts initialized before the observed onset date.

    Parameters:
    climatological_onset_doy: xarray DataArray with climatological onset day of year
    year: int, year to evaluate
    init_dates: pandas DatetimeIndex with initialization dates
    observed_onset_da: xarray DataArray with observed onset dates for filtering
    max_forecast_day: int, maximum forecast day to consider
    earliest_onset_date: optional (month, day) tuple; if set, only onsets on or after that date count (falsy = no filter)

    Returns:
    pandas DataFrame with climatology forecast results
    """

    results_list = []

    # Get dimensions
    lats = climatological_onset_doy.lat.values
    lons = climatological_onset_doy.lon.values

    print(f"Processing climatology as forecast for {len(init_dates)} init times x {len(lats)} lats x {len(lons)} lons...")
    print(f"Year: {year}")

    # Track statistics
    total_potential_inits = 0
    valid_inits = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    onsets_forecasted = 0

    # Loop over all initialization dates and grid points
    for t_idx, init_time in enumerate(init_dates):

        init_date = pd.to_datetime(init_time)
        year = init_date.year

        if earliest_onset_date:
            earliest_date = datetime(year, *earliest_onset_date)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):

                total_potential_inits += 1

                # Get observed onset date for this grid point
                try:
                    obs_onset = observed_onset_da.isel(lat=i, lon=j).values
                except:
                    skipped_no_obs += 1
                    continue

                # Skip if no observed onset
                if pd.isna(obs_onset):
                    skipped_no_obs += 1
                    continue

                # Convert observed onset to datetime
                obs_onset_dt = pd.to_datetime(obs_onset)

                # Only process if forecast was initialized before observed onset
                if init_date >= obs_onset_dt:
                    skipped_late_init += 1
                    continue

                valid_inits += 1

                # Get climatological onset day of year for this grid point
                clim_onset_doy = climatological_onset_doy.isel(lat=i, lon=j).values

                # Skip if no climatological onset available
                # Bug fix from original code which directly skip/continue if no clim onset
                if np.isnan(clim_onset_doy):
                    onset_day, onset_date = None, None
                    result = {
                        'init_time': init_time,
                        'lat': lat,
                        'lon': lon,
                        'onset_day': onset_day,  # None if no onset forecasted
                        'onset_date': onset_date.strftime('%Y-%m-%d') if onset_date is not None else None,
                        'climatological_onset_doy': clim_onset_doy,
                        # Verbatim old code: this reads the previous cell's date (the bug the new code fixes).
                        'climatological_onset_date': clim_onset_date.strftime('%Y-%m-%d'),  # noqa: F821
                        'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
                    }
                    results_list.append(result)
                    continue

                # Convert climatological day of year to actual date for this year
                try:
                    clim_onset_date = datetime(year, 1, 1) + timedelta(days=int(clim_onset_doy) - 1)
                    clim_onset_date = pd.to_datetime(clim_onset_date)
                except:
                    continue  # Skip if invalid day of year

                # Check if climatological onset is within forecast window
                forecast_window_start = init_date + pd.Timedelta(days=1)
                forecast_window_end = init_date + pd.Timedelta(days=max_forecast_day)

                onset_day = None
                onset_date = None

                if forecast_window_start <= clim_onset_date <= forecast_window_end:
                    # Climatological onset is within forecast window
                    onset_day = (clim_onset_date - init_date).days

                    # Apply the earliest onset date filter if requested
                    if earliest_onset_date:
                        if clim_onset_date.date() >= earliest_date.date():
                            # Valid onset on or after the earliest onset date
                            onset_date = clim_onset_date
                            onsets_forecasted += 1
                        else:
                            # Reset if before the earliest onset date
                            onset_day = None
                            onset_date = None
                    else:
                        # No date filtering
                        onset_date = clim_onset_date
                        onsets_forecasted += 1

                # Store result
                result = {
                    'init_time': init_time,
                    'lat': lat,
                    'lon': lon,
                    'onset_day': onset_day,  # None if no onset forecasted
                    'onset_date': onset_date.strftime('%Y-%m-%d') if onset_date is not None else None,
                    'climatological_onset_doy': clim_onset_doy,
                    'climatological_onset_date': clim_onset_date.strftime('%Y-%m-%d'),
                    'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
                }
                results_list.append(result)

    # Convert to DataFrame
    climatology_forecast_df = pd.DataFrame(results_list)

    print(f"\nClimatology Forecast Summary:")
    print(f"Total potential initializations: {total_potential_inits}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid initializations processed: {valid_inits}")
    print(f"Onsets forecasted: {onsets_forecasted}")
    print(f"Forecast rate: {onsets_forecasted/valid_inits:.3f}" if valid_inits > 0 else "Forecast rate: 0.000")

    if earliest_onset_date:
        print(f"Note: Only onsets on or after {earliest_onset_date[0]:02d}-{earliest_onset_date[1]:02d} (MM-DD) were counted due to earliest_onset_date")

    return climatology_forecast_df


def _old_create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, *, day_bins, max_forecast_day, **kwargs):
    """
    Create forecast-observation pairs using specified day bins, including a final bin for "after max_forecast_day".

    Parameters:
    -----------
    onset_all_members : DataFrame
        DataFrame with ensemble member onset predictions
    onset_da : xarray.DataArray
        Observed onset dates
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day. Members without onset get assigned to "after day X" bin
    """

    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    results_list = []

    # Get unique combinations of init_time, lat, lon from the filtered forecast data
    forecast_groups = onset_all_members.groupby(['init_time', 'lat', 'lon'])

    # Add the "after max_forecast_day" bin
    extended_bins = ((-float('inf'), min_forecast_day - 1),) + day_bins + ((max_forecast_day + 1, float('inf')),)

    print(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}")
    print(f"Including 'after day {max_forecast_day}' bin for members without onset in forecast window")

    for (init_time, lat, lon), group in forecast_groups:

        # Get observed onset for this location
        try:
            lat_idx = np.where(np.abs(onset_da.lat.values - lat) < 0.01)[0][0]
            lon_idx = np.where(np.abs(onset_da.lon.values - lon) < 0.01)[0][0]
            obs_date = onset_da.isel(lat=lat_idx, lon=lon_idx).values
        except:
            continue

        # Handle different types of obs_onset values
        if hasattr(obs_date, 'item'):
            obs_date = obs_date.item()

        # Skip if no observed onset
        if pd.isna(obs_date):
            continue

        # Convert dates for comparison
        init_date = pd.to_datetime(init_time)
        obs_date_dt = pd.to_datetime(obs_date)

        # Double-check: Only use forecasts initialized before the observed onset
        if init_date >= obs_date_dt:
            continue

        total_members = len(group)

        # For each day bin (including the "after max_forecast_day" bin)
        for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):

            if bin_start == -float('inf'):
                bin_label = f'Before day {min_forecast_day}'

                # Check if observed onset occurs before window start
                forecast_start_date = init_date + pd.Timedelta(days=min_forecast_day)
                observed_onset = int(obs_date_dt.date() < forecast_start_date.date())

                # Count members that didn't predict onset within forecast window
                members_with_onset_in_bin = 0

                for member_idx, member_row in group.iterrows():
                    member_onset_day = member_row['onset_day']

                    # Member predicts < min_forecast_day
                    if pd.notna(member_onset_day) and member_onset_day < min_forecast_day:
                        members_with_onset_in_bin += 1

            # Handle the "after max_forecast_day" bin differently
            elif bin_start > max_forecast_day:
                bin_label = f'After day {max_forecast_day}'

                # Check if observed onset occurs after max_forecast_day
                forecast_end_date = init_date + pd.Timedelta(days=max_forecast_day)
                observed_onset = int(obs_date_dt.date() > forecast_end_date.date())

                # Count members that didn't predict onset within forecast window
                members_with_onset_in_bin = 0

                for member_idx, member_row in group.iterrows():
                    member_onset_day = member_row['onset_day']

                    # Member predicts "after day X" if onset_day is NaN or > max_forecast_day
                    if pd.isna(member_onset_day) or member_onset_day > max_forecast_day:
                        members_with_onset_in_bin += 1

            else:
                # Regular bin within forecast window
                bin_label = f'Days {bin_start}-{bin_end}'

                # Calculate the date range for this bin
                bin_start_date = init_date + pd.Timedelta(days=bin_start)
                bin_end_date = init_date + pd.Timedelta(days=bin_end)

                # Check if observed onset falls within this day bin
                observed_onset = int(bin_start_date.date() <= obs_date_dt.date() <= bin_end_date.date())

                # Calculate ensemble probability for this day bin
                members_with_onset_in_bin = 0

                for member_idx, member_row in group.iterrows():
                    member_onset_day = member_row['onset_day']

                    if pd.notna(member_onset_day) and bin_start <= member_onset_day <= bin_end:
                        members_with_onset_in_bin += 1

            # Calculate probability
            predicted_prob = members_with_onset_in_bin / total_members

            # Store result
            result = {
                'init_time': init_time,
                'lat': lat,
                'lon': lon,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_label': bin_label,
                'predicted_prob': predicted_prob,
                'observed_onset': observed_onset,
                'members_with_onset': members_with_onset_in_bin,
                'total_members': total_members,
                'year': pd.to_datetime(init_time).year,
                'obs_onset_date': obs_date_dt.strftime('%Y-%m-%d'),
                'bin_index': bin_idx
            }
            results_list.append(result)

    # Convert to DataFrame
    forecast_obs_df = pd.DataFrame(results_list)

    print(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    # Show distribution across bins
    print(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean'
    }).round(3)
    print(bin_stats)

    return forecast_obs_df


def _old_create_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, *,
                                             day_bins, max_forecast_day, **kwargs):
    """
    Create forecast-observation pairs using climatological ensemble where each year is a member.
    Uses day-of-year instead of calendar dates for onset comparison.

    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates for all years
    target_year : int
        The year to use as "truth" for observations
    init_dates : list or pandas.DatetimeIndex
        Initialization dates for forecasts
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day

    Returns:
    --------
    DataFrame with forecast-observation pairs
    """

    # adjust max_forecast_day to the end of day_bins
    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    results_list = []

    # Get the observed onset for the target year
    if target_year not in clim_onset.year.values:
        raise ValueError(f"Target year {target_year} not found in climatological dataset")

    obs_onset_da = clim_onset.sel(year=target_year)

    # Use ALL years as ensemble members (including target year) ### redundant
    ensemble_years = list(clim_onset.year.values)
    ensemble_onset_da = clim_onset.sel(year=ensemble_years)

    # Create extended bins including "before initialization" and "after max_forecast_day" bins
    extended_bins = ((-float('inf'), min_forecast_day - 1),) + day_bins + ((max_forecast_day + 1, float('inf')),)

    print(f"Creating climatological forecasts for target year {target_year}")
    print(f"Using {len(ensemble_years)} years as ensemble members: {ensemble_years}")
    print(f"Processing {len(init_dates)} initialization dates")
    print(f"Day bins: {day_bins}")
    print(f"Extended bins include: 'Before day {min_forecast_day}' and 'After day {max_forecast_day}' ")
    print(f"Using day-of-year method for onset comparison")

    # Get the actual lat/lon coordinates from the data
    lats = obs_onset_da.lat.values
    lons = obs_onset_da.lon.values

    # Create unique lat-lon pairs (no repetition)
    unique_pairs = list(product(lons, lats))

    print(f"Processing {len(unique_pairs)} unique lat-lon pairs")

    # Process each initialization date and location
    for init_date in init_dates:
        init_date = pd.to_datetime(init_date)
        init_doy = init_date.dayofyear  # Day of year for initialization

        # Loop over unique lat-lon pairs
        for pair_idx, (lon, lat) in enumerate(unique_pairs):

            # Get observed onset for this location and target year
            try:
                lat_idx = np.where(np.abs(obs_onset_da.lat.values - lat) < 0.01)[0][0]
                lon_idx = np.where(np.abs(obs_onset_da.lon.values - lon) < 0.01)[0][0]
                obs_onset = obs_onset_da.isel(lat=lat_idx, lon=lon_idx).values
            except:
                continue

            # Handle different types of obs_onset values
            if hasattr(obs_onset, 'item'):
                obs_onset = obs_onset.item()

            # Skip if no observed onset
            if pd.isna(obs_onset):
                continue

            obs_onset_dt = pd.to_datetime(obs_onset)
            obs_onset_doy = obs_onset_dt.dayofyear  # Day of year for observed onset

            # Only process forecasts that are initialized BEFORE the observed onset (by day of year)
            if init_doy >= obs_onset_doy:
                continue

            # Get ensemble member onsets for this location using the same indices
            ensemble_onsets = ensemble_onset_da.isel(lat=lat_idx, lon=lon_idx).values

            # Convert ensemble onsets to days from initialization using day-of-year
            ensemble_forecast_days = []
            ensemble_years_with_data = []

            for ens_idx, ens_onset in enumerate(ensemble_onsets):
                ens_year = ensemble_years[ens_idx]

                if pd.notna(ens_onset):
                    ens_onset_dt = pd.to_datetime(ens_onset)
                    ens_onset_doy = ens_onset_dt.dayofyear  # Day of year for ensemble onset

                    # Calculate days from initialization using day-of-year difference
                    days_from_init = ens_onset_doy - init_doy
                    ensemble_forecast_days.append(days_from_init)
                    ensemble_years_with_data.append(ens_year)
                else:
                    # No onset predicted by this member
                    ensemble_forecast_days.append(None)
                    ensemble_years_with_data.append(ens_year)

            total_members = len(ensemble_years)

            # First pass: calculate total members with onset across all bins
            total_members_with_onset = 0
            bin_members_onset = []  # Store for second pass

            for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
                members_with_onset_in_bin = 0

                # Handle the "before initialization" bin
                if bin_start == -float('inf'):
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day <= min_forecast_day - 1:
                            members_with_onset_in_bin += 1

                # Handle the "after max_forecast_day" bin
                elif bin_start > max_forecast_day:
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day > max_forecast_day:
                            members_with_onset_in_bin += 1

                else:
                    # Regular bin within forecast window
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and bin_start <= member_onset_day <= bin_end:
                            members_with_onset_in_bin += 1

                bin_members_onset.append(members_with_onset_in_bin)
                total_members_with_onset += members_with_onset_in_bin

            # Skip if no members showed onset
            if total_members_with_onset == 0:
                continue

            # Second pass: For each day bin, calculate probabilities using total_members_with_onset
            for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):

                members_with_onset_in_bin = bin_members_onset[bin_idx]

                # Track which years contribute to this bin
                contributing_years = []

                # Handle the "before initialization" bin
                if bin_start == -float('inf'):
                    bin_label = f'Before day {min_forecast_day}'

                    # Check if observed onset occurs before initialization (by day of year)
                    observed_onset = int(obs_onset_doy <= init_doy + min_forecast_day - 1 )

                    # Get contributing years
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day <= 0:
                            contributing_years.append(ensemble_years_with_data[i])

                # Handle the "after max_forecast_day" bin
                elif bin_start > max_forecast_day:
                    bin_label = f'After day {max_forecast_day}'

                    # Check if observed onset occurs after max_forecast_day (by day of year)
                    obs_days_from_init = obs_onset_doy - init_doy

                    observed_onset = int(obs_days_from_init > max_forecast_day)

                    # Get contributing years
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day > max_forecast_day:
                            contributing_years.append(ensemble_years_with_data[i])

                else:
                    # Regular bin within forecast window
                    bin_label = f'Days {bin_start}-{bin_end}'

                    # Check if observed onset falls within this day bin (by day of year)
                    obs_days_from_init = obs_onset_doy - init_doy
                    observed_onset = int(bin_start <= obs_days_from_init <= bin_end)

                    # Get contributing years
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and bin_start <= member_onset_day <= bin_end:
                            contributing_years.append(ensemble_years_with_data[i])

                # Calculate probability using only members that showed onset
                predicted_prob = members_with_onset_in_bin / total_members_with_onset

                # Convert contributing years to string for storage
                contributing_years_str = ','.join(map(str, sorted(contributing_years))) if contributing_years else ''

                # Store result
                result = {
                    'init_time': init_date.strftime('%Y-%m-%d'),
                    'lat': lat,
                    'lon': lon,
                    'bin_start': bin_start,
                    'bin_end': bin_end,
                    'bin_label': bin_label,
                    'predicted_prob': predicted_prob,
                    'observed_onset': observed_onset,
                    'members_with_onset': members_with_onset_in_bin,
                    'total_members': total_members,
                    'total_members_with_onset': total_members_with_onset,  # New field
                    'contributing_years': contributing_years_str,
                    'n_contributing_years': len(contributing_years),
                    'year': target_year,
                    'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d'),
                    'obs_onset_doy': obs_onset_doy,
                    'init_doy': init_doy,
                    'obs_days_from_init_doy': obs_days_from_init if 'obs_days_from_init' in locals() else (obs_onset_doy - init_doy),
                    'bin_index': bin_idx,
                    'forecast_type': 'climatological_doy'
                }
                results_list.append(result)

    # Convert to DataFrame
    forecast_obs_df = pd.DataFrame(results_list)

    if len(forecast_obs_df) == 0:
        print("Warning: No forecast-observation pairs generated")
        return forecast_obs_df

    print(f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs")
    print(f"Unique lat-lon pairs processed: {len(unique_pairs)}")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    # Verify uniqueness
    unique_locations_in_output = len(forecast_obs_df[['lat', 'lon']].drop_duplicates())
    print(f"Unique locations in output: {unique_locations_in_output}")

    # Show distribution across bins
    print(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean',
        'n_contributing_years': 'mean',
        'total_members_with_onset': 'mean'
    }).round(3)
    print(bin_stats)

    return forecast_obs_df


def _old_create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
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


# --------------------------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------------------------

LATS = np.array([10.0, 10.25, 10.5, 10.75], dtype="float32")
LONS = np.array([70.0, 70.25, 70.5, 70.75, 71.0], dtype="float32")
INITS = pd.DatetimeIndex(["2020-05-04", "2020-05-07", "2020-05-11", "2020-05-14", "2020-05-18", "2020-05-21",
                          "2020-05-25", "2020-05-28", "2020-06-01", "2020-06-04"])


def _grid(values, dtype=float):
    """Wrap values as a DataArray on the test lat/lon grid."""
    return xr.DataArray(np.asarray(values, dtype=dtype), coords=[("lat", LATS), ("lon", LONS)])


def _observed_onsets(seed=0):
    """Observed onset dates in May-June 2020, with some cells without an onset (NaT) and some very early."""
    rng = np.random.default_rng(seed)
    days = rng.integers(-5, 40, size=(len(LATS), len(LONS)))
    dates = (np.datetime64("2020-05-10") + days.astype("timedelta64[D]")).astype("datetime64[ns]")
    dates[0, 0] = np.datetime64("2020-06-20")
    dates[1, 2] = np.datetime64("NaT")
    dates[3, 4] = np.datetime64("NaT")
    return _grid(dates, dtype="datetime64[ns]")


def _assert_frames_identical(old, new):
    """Assert two frames are exactly equal in values and dtypes and both have a default integer index."""
    pd.testing.assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True), check_dtype=True, check_exact=True)
    assert list(old.index) == list(range(len(old))) == list(new.index)


# --------------------------------------------------------------------------------------------
# compute_climatology_as_forecast
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("earliest_onset_date", [None, (5, 20)])
@pytest.mark.parametrize("max_forecast_day", [10, 30])
def test_climatology_as_forecast_matches_loops(earliest_onset_date, max_forecast_day):
    """The vectorised climatology-as-forecast matches the loop version apart from the documented differences."""
    rng = np.random.default_rng(1)
    doy = rng.integers(125, 175, size=(len(LATS), len(LONS))).astype(float)
    doy[2, 3] = np.nan
    doy[3, 0] = np.nan
    clim = _grid(doy)
    observed = _observed_onsets()
    args = (clim, 2020, INITS, observed)
    kwargs = dict(max_forecast_day=max_forecast_day, earliest_onset_date=earliest_onset_date)
    old = _old_compute_climatology_as_forecast(*args, **kwargs)
    new = compute_climatology_as_forecast(*args, **kwargs)

    # Documented differences: the doy column was an object column of 0-d arrays and is now float64, and
    # rows without a climatological onset now have no climatological_onset_date (the loop repeated the
    # previous cell's date there). Neither column is read by the metrics.
    assert old["climatological_onset_doy"].dtype == object
    assert new["climatological_onset_doy"].dtype == np.float64
    no_clim = new["climatological_onset_doy"].isna()
    assert no_clim.any()
    assert new.loc[no_clim, "climatological_onset_date"].isna().all()
    expected = old.copy()
    expected["climatological_onset_doy"] = expected["climatological_onset_doy"].astype(float)
    expected.loc[no_clim, "climatological_onset_date"] = None
    _assert_frames_identical(expected, new)


def test_climatology_as_forecast_without_valid_cases_is_empty():
    """Climatology-as-forecast gives an empty frame when no case has a valid onset."""
    clim = _grid(np.full((len(LATS), len(LONS)), 150.0))
    observed = _grid(np.full((len(LATS), len(LONS)), np.datetime64("NaT"), dtype="datetime64[ns]"), dtype="datetime64[ns]")
    kwargs = dict(max_forecast_day=30, earliest_onset_date=None)
    assert compute_climatology_as_forecast(clim, 2020, INITS, observed, **kwargs).empty
    assert _old_compute_climatology_as_forecast(clim, 2020, INITS, observed, **kwargs).empty


# --------------------------------------------------------------------------------------------
# compute_onset_metrics_with_windows
# --------------------------------------------------------------------------------------------

def _onset_frame(seed=0):
    """Forecast rows per (init, cell), cells interleaved, with onsets before init, in and past the windows, and none."""
    rng = np.random.default_rng(seed)
    observed = _observed_onsets().values
    rows = []
    for init in INITS:
        for i, lat in enumerate(LATS):
            for j, lon in enumerate(LONS):
                if np.isnat(observed[i, j]) or init >= pd.Timestamp(observed[i, j]):
                    continue
                offset = rng.integers(-3, 40)
                onset = None if rng.random() < 0.25 else (init + pd.Timedelta(days=int(offset))).strftime("%Y-%m-%d")
                rows.append({"init_time": init, "lat": lat, "lon": lon, "onset_day": offset if onset else None,
                             "onset_date": onset, "obs_onset_date": pd.Timestamp(observed[i, j]).strftime("%Y-%m-%d")})
    frame = pd.DataFrame(rows)
    # interleave the cells the way they come out of a run, then break the grouping by init
    return frame.sample(frac=1.0, random_state=seed).sort_values("init_time", kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("verification_window, tolerance_days", [((1, 15), 3), ((16, 30), 5), ((6, 10), 2)])
def test_onset_metrics_match_loops(verification_window, tolerance_days):
    """The vectorised onset metrics and summary match the loop version."""
    onset_df = _onset_frame()
    assert (pd.to_datetime(onset_df["onset_date"]) <= pd.to_datetime(onset_df["init_time"])).any()
    kwargs = dict(tolerance_days=tolerance_days, verification_window=verification_window)
    old_metrics, old_summary = _old_compute_onset_metrics_with_windows(onset_df, **kwargs)
    new_metrics, new_summary = compute_onset_metrics_with_windows(onset_df, **kwargs)
    _assert_frames_identical(old_metrics, new_metrics)
    assert old_summary.keys() == new_summary.keys()
    for key, value in old_summary.items():
        assert new_summary[key] == value or (value != value and new_summary[key] != new_summary[key]), key


def test_onset_metrics_of_no_forecasts_match_loops():
    """Onset metrics of an empty forecast frame match the loop version."""
    kwargs = dict(tolerance_days=3, verification_window=(1, 15))
    old_metrics, _ = _old_compute_onset_metrics_with_windows(pd.DataFrame([]), **kwargs)
    new_metrics, _ = compute_onset_metrics_with_windows(pd.DataFrame([]), **kwargs)
    pd.testing.assert_frame_equal(old_metrics, new_metrics)


# --------------------------------------------------------------------------------------------
# create_forecast_observation_pairs_with_bins
# --------------------------------------------------------------------------------------------

def _member_frame(seed=0, members=6):
    """Build random per-member onset-day rows for every init, cell and member."""
    rng = np.random.default_rng(seed)
    rows = []
    for init in INITS:
        for i, lat in enumerate(LATS):
            for j, lon in enumerate(LONS):
                for member in range(members):
                    day = np.nan if rng.random() < 0.3 else float(rng.integers(-2, 22))
                    rows.append({"init_time": init, "lat": lat, "lon": lon, "member": member, "onset_day": day})
    return pd.DataFrame(rows)


def test_forecast_observation_pairs_match_loops():
    """The vectorised forecast-observation pairs match the loop version."""
    members = _member_frame()
    onset_da = _observed_onsets()
    kwargs = dict(day_bins=((1, 5), (6, 10), (11, 15)), max_forecast_day=15)
    old = _old_create_forecast_observation_pairs_with_bins(members, onset_da, **kwargs)
    new = create_forecast_observation_pairs_with_bins(members, onset_da, **kwargs)
    assert len(new) > 0 and (members["onset_day"] <= 0).any()
    _assert_frames_identical(old, new)


def test_forecast_observation_pairs_skip_cells_off_the_grid():
    """Forecast-observation pairs skip cells whose coordinates are off the grid, as the loop version does."""
    members = _member_frame(members=3)
    members.loc[members["lat"] == LATS[1], "lat"] = np.float32(50.0)
    onset_da = _observed_onsets()
    kwargs = dict(day_bins=((1, 7), (8, 14)), max_forecast_day=14)
    _assert_frames_identical(_old_create_forecast_observation_pairs_with_bins(members, onset_da, **kwargs),
                             create_forecast_observation_pairs_with_bins(members, onset_da, **kwargs))


# --------------------------------------------------------------------------------------------
# create_climatological_forecast_obs_pairs
# --------------------------------------------------------------------------------------------

def _climatological_onsets(years=range(2010, 2021), seed=2):
    """Build random climatological onset dates per year and cell, with some missing."""
    rng = np.random.default_rng(seed)
    shape = (len(years), len(LATS), len(LONS))
    dates = (np.array([f"{y}-01-01" for y in years], dtype="datetime64[D]")[:, None, None]
             + rng.integers(120, 190, size=shape).astype("timedelta64[D]")).astype("datetime64[ns]")
    dates[rng.random(shape) < 0.15] = np.datetime64("NaT")
    dates[:, 3, 4] = np.datetime64("NaT")
    return xr.DataArray(dates, coords=[("year", list(years)), ("lat", LATS), ("lon", LONS)])


@pytest.mark.parametrize("day_bins", [((1, 5), (6, 10), (11, 15)), ((3, 9), (10, 16))])
def test_climatological_pairs_match_loops(day_bins):
    """The vectorised climatological pairs match the loop version apart from the documented difference."""
    clim_onset = _climatological_onsets()
    kwargs = dict(day_bins=day_bins, max_forecast_day=day_bins[-1][1])
    old = _old_create_climatological_forecast_obs_pairs(clim_onset, 2020, INITS, **kwargs)
    new = create_climatological_forecast_obs_pairs(clim_onset, 2020, INITS, **kwargs)
    assert len(new) > 0

    # Documented difference: in the loops the "Before" row of every case after the first carried the previous
    # case's obs_days_from_init_doy; it now holds the case's own value, like the other bins of the case.
    before = new["bin_index"] == 0
    assert (new["obs_days_from_init_doy"] == new["obs_onset_doy"] - new["init_doy"]).all()
    assert (old["obs_days_from_init_doy"][~before] == new["obs_days_from_init_doy"][~before]).all()
    assert (old["obs_days_from_init_doy"][before] != new["obs_days_from_init_doy"][before]).any()
    expected = old.copy()
    expected["obs_days_from_init_doy"] = new["obs_days_from_init_doy"]
    _assert_frames_identical(expected, new)


def test_climatological_pairs_without_cases_are_empty():
    """Climatological pairs are empty when no init has a valid case."""
    clim_onset = _climatological_onsets()
    late = [pd.Timestamp("2020-12-01")]
    kwargs = dict(day_bins=((1, 5),), max_forecast_day=5)
    assert len(create_climatological_forecast_obs_pairs(clim_onset, 2020, late, **kwargs)) == 0
    assert len(_old_create_climatological_forecast_obs_pairs(clim_onset, 2020, late, **kwargs)) == 0


# --------------------------------------------------------------------------------------------
# create_spatial_far_mr_mae
# --------------------------------------------------------------------------------------------

def _metrics_frame(seed, observed):
    """Build random per-cell metrics rows, including a repeated grid point."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, lat in enumerate(LATS):
        for j, lon in enumerate(LONS):
            if rng.random() < 0.2:
                continue
            fp, tn, fn = (int(x) for x in rng.integers(0, 6, size=3))
            rows.append({"lat": lat, "lon": lon, "false_positive": fp, "true_negative": tn, "false_negative": fn,
                         "num_onset": int(rng.integers(0, 5)),
                         "mae_combined": np.nan if rng.random() < 0.3 else float(rng.integers(0, 90)) / 7})
    frame = pd.DataFrame(rows)
    # a repeated grid point: only its first row counts
    return pd.concat([frame, frame.iloc[:2]], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_spatial_maps_match_loops():
    """The vectorised spatial maps match the loop version."""
    onset_da_dict = {2019: _observed_onsets(3), 2020: _observed_onsets(4)}
    metrics_df_dict = {year: _metrics_frame(year, onset_da_dict[year]) for year in onset_da_dict}
    old = _old_create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
    new = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
    assert list(old) == list(new) == ["false_alarm_rate", "miss_rate", "mean_mae", "mae_2019", "mae_2020"]
    for key in old:
        xr.testing.assert_identical(old[key], new[key])
    assert new["mean_mae"].notnull().any() and new["false_alarm_rate"].isnull().any()


def test_spatial_maps_with_an_empty_year_match_loops():
    """The spatial maps match the loop version when one year has no metrics."""
    onset_da_dict = {2019: _observed_onsets(3), 2020: _observed_onsets(4)}
    metrics_df_dict = {2019: _metrics_frame(1, onset_da_dict[2019]),
                       2020: pd.DataFrame(columns=["lat", "lon", "false_positive", "true_negative", "false_negative",
                                                   "num_onset", "mae_combined"])}
    old = _old_create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
    new = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
    for key in old:
        xr.testing.assert_identical(old[key], new[key])
