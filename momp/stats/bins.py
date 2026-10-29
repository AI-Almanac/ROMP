"""Build forecast-observation pairs per day bin (from ensemble members or climatology) for probabilistic verification."""

import pandas as pd
import numpy as np
from momp.io.input import load_thresh_file, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_all_members
from momp.utils.practical import restore_args
from itertools import product


def extract_day_range(bin_label):
    """ extract the start day of each bin for sorting purpose """
    if 'Days ' in bin_label:
        try:
            day_part = bin_label.replace('Days ', '').split('-')[0]
            return int(day_part)
        except:
            return 999
    return 999


def get_target_bins(brier_forecast, brier_climatology):
    """Extract and sort target bins"""
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    common_bins = all_forecast_bins.intersection(all_clim_bins)

    target_bins = []
    for bin_label in common_bins:
        if (bin_label.startswith('Days ') and
            not bin_label.startswith('After') and
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)

    return sorted(target_bins, key=extract_day_range)


# Function to create forecast-observation pairs with specified day bins for probabilistic verification
def create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, *, day_bins, max_forecast_day, **kwargs):
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


## This function creates forecast-observation pairs using climatological ensemble where each year is a member
def create_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, *,
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


# This function creates the observed forecast pairs for multiple years (core monsoon zone grids) and combines them
def multi_year_forecast_obs_pairs(*, years, obs_dir, obs_file_pattern, obs_var,
                                  thresh_file, thresh_var, wet_threshold,
                                  date_filter_year, init_days, start_date, end_date,
                                  model_dir, model_var, unit_cvt, file_pattern,
                                  wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, fallback_date, earliest_onset_date,
                                  members, onset_percentage_threshold, max_forecast_day, day_bins, **kwargs):
    """Main function to perform multi-year reliability analysis."""

    kwargs = restore_args(multi_year_forecast_obs_pairs, kwargs, locals())

    print(f"Processing years: {years}")

    thresh_slice = load_thresh_file(**kwargs)

    # Initialize list to store all forecast-observation pairs
    all_forecast_obs_pairs = []

    # Process each year
    for year in years:
        print(f"\n{'-'*50}")
        print(f"Processing year {year}")

        try:
            # Load model and observation data
            print("Loading S2S model data...")
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            p_model_slice = p_model # !!!!! region subset

            print("Loading observational rainfall data...")
            rainfall_ds = load_imd_rainfall(year, **kwargs)
            rainfall_ds_slice = rainfall_ds #!!!!! region subset

            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds_slice, thresh_slice, year, **kwargs)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")

            print("Computing onset for all ensemble members...")
            onset_all_members, _ = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da, **kwargs)
            print(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")

            print("Creating forecast-observation pairs...")
            forecast_obs_pairs = create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, **kwargs)

            # Add to master list
            all_forecast_obs_pairs.append(forecast_obs_pairs)

            print(f"Year {year} completed: {len(forecast_obs_pairs)} forecast-observation pairs")

        except FileNotFoundError as e:
            # Reforecast archives commonly have year gaps; skip years whose
            # model or observation file is absent instead of aborting the run.
            print(f"Skipping year {year} (missing input file): {e}")
            continue

    # Combine all years
    print(f"\n{'-'*50}")
    print("Combining all years")

    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    # Print final summary statistics
    print(f"\nFinal Summary Statistics:")
    print(f"Years processed: {years}")
    return combined_forecast_obs


def multi_year_climatological_forecast_obs_pairs(clim_onset, *, years, day_bins, max_forecast_day, date_filter_year, init_days, start_date, end_date, **kwargs):
    """
    Create climatological forecast-observation pairs for multiple target years.

    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates
    years : list
        Years to use as truth for observations
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins
    max_forecast_day : int, default=15
        Maximum forecast day

    Returns:
    --------
    DataFrame with combined forecast-observation pairs from all target years
    """

    kwargs = restore_args(multi_year_climatological_forecast_obs_pairs, kwargs, locals())

    clim_onset_slice = clim_onset #!!!!! region subset

    all_forecast_obs_pairs = []

    for target_year in years:
        print(f"\n{'-'*50}")
        print(f"Processing target year {target_year}")

        try:
            # Get initialization dates for this year
            init_dates = get_initialization_dates(target_year, **kwargs)

            # Create forecast-observation pairs for this year
            forecast_obs_pairs = create_climatological_forecast_obs_pairs(
                clim_onset_slice,
                target_year,
                init_dates,
                **kwargs
            )

            if len(forecast_obs_pairs) > 0:
                all_forecast_obs_pairs.append(forecast_obs_pairs)
                print(f"Target year {target_year} completed: {len(forecast_obs_pairs)} pairs")
            else:
                print(f"No pairs generated for target year {target_year}")

        except Exception as e:
            print(f"Error processing target year {target_year}: {e}")
            continue

    # Combine all years
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any target year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    print(f"\n{'='*50}")
    print("CLIMATOLOGICAL FORECAST SUMMARY")
    print(f"{'='*50}")
    print(f"Target years processed: {years}")
    print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
    print(f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}")
    print(f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}")

    return combined_forecast_obs
