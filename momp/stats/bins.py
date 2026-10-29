"""Build forecast-observation pairs per day bin (from ensemble members or climatology) for probabilistic verification."""

import pandas as pd
import numpy as np
from momp.io.input import load_thresholds, load_history, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_all_members, check_alignment_once
from momp.utils.practical import restore_args


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
def _first_close(values, targets, tolerance=0.01):
    """Index in values of the first entry within tolerance of each target, -1 where there is none."""
    found = np.full(len(targets), -1, dtype=np.int64)
    for n, target in enumerate(targets):
        close = np.flatnonzero(np.abs(values - target) < tolerance)
        if len(close):
            found[n] = close[0]
    return found


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

    Returns one row per forecast case (init_time, lat, lon, in sorted order) and bin; cases without an
    observed onset, or initialized on or after it, are left out.
    """

    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    # Unique combinations of init_time, lat, lon from the filtered forecast data
    forecast_groups = onset_all_members.groupby(['init_time', 'lat', 'lon'])

    # Add the "after max_forecast_day" bin
    extended_bins = ((-float('inf'), min_forecast_day - 1),) + day_bins + ((max_forecast_day + 1, float('inf')),)

    print(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}")
    print(f"Including 'after day {max_forecast_day}' bin for members without onset in forecast window")

    member_case = forecast_groups.ngroup().to_numpy()
    case_size = forecast_groups.size()
    cases = case_size.index
    total_members = case_size.to_numpy()
    init_times = cases.get_level_values('init_time')
    lats = cases.get_level_values('lat').to_numpy(dtype=float)
    lons = cases.get_level_values('lon').to_numpy(dtype=float)

    # Observed onset of each case's location, a case without a matching grid point is left out
    unique_lats, lat_of_case = np.unique(lats, return_inverse=True)
    unique_lons, lon_of_case = np.unique(lons, return_inverse=True)
    lat_idx = _first_close(onset_da.lat.values, unique_lats)[lat_of_case]
    lon_idx = _first_close(onset_da.lon.values, unique_lons)[lon_of_case]
    located = (lat_idx >= 0) & (lon_idx >= 0)
    obs_date = np.full(len(cases), np.datetime64("NaT"), dtype="datetime64[ns]")
    obs_date[located] = onset_da.values.astype("datetime64[ns]")[lat_idx[located], lon_idx[located]]
    init_date = pd.DatetimeIndex(init_times).values.astype("datetime64[ns]")
    # Only use forecasts initialized before the observed onset
    kept = np.flatnonzero(located & ~np.isnat(obs_date) & (init_date < obs_date))

    member_day = onset_all_members['onset_day'].to_numpy(dtype=float)
    has_onset = ~np.isnan(member_day)
    obs_day = obs_date[kept].astype("datetime64[D]")
    init_kept = init_date[kept]

    def date_after(days):
        """Return the dates the given number of days after each kept init date."""
        return (init_kept + np.timedelta64(days, "D")).astype("datetime64[D]")

    members_in_bin = []
    observed_onset = []
    labels = []
    for bin_start, bin_end in extended_bins:
        if bin_start == -float('inf'):
            labels.append(f'Before day {min_forecast_day}')
            member_in_bin = has_onset & (member_day < min_forecast_day)
            observed = obs_day < date_after(min_forecast_day)
        elif bin_start > max_forecast_day:
            labels.append(f'After day {max_forecast_day}')
            # Members predict "after day X" if onset_day is NaN or > max_forecast_day
            member_in_bin = ~has_onset | (member_day > max_forecast_day)
            observed = obs_day > date_after(max_forecast_day)
        else:
            labels.append(f'Days {bin_start}-{bin_end}')
            member_in_bin = has_onset & (bin_start <= member_day) & (member_day <= bin_end)
            observed = (date_after(bin_start) <= obs_day) & (obs_day <= date_after(bin_end))
        members_in_bin.append(
            np.bincount(member_case, weights=member_in_bin, minlength=len(cases)).astype(np.int64)[kept])
        observed_onset.append(observed.astype(np.int64))
    members_with_onset = np.stack(members_in_bin, axis=1)
    n_bins = len(extended_bins)

    def per_case(values):
        """Repeat each per-case value once for every bin."""
        return np.repeat(values, n_bins)

    forecast_obs_df = pd.DataFrame({
        'init_time': per_case(init_times[kept]) if len(kept) else [],
        'lat': per_case(lats[kept]),
        'lon': per_case(lons[kept]),
        'bin_start': np.tile(np.array([b[0] for b in extended_bins], dtype=float), len(kept)),
        'bin_end': np.tile(np.array([b[1] for b in extended_bins], dtype=float), len(kept)),
        'bin_label': np.tile(np.array(labels, dtype=object), len(kept)),
        'predicted_prob': (members_with_onset / total_members[kept][:, None]).ravel(),
        'observed_onset': np.stack(observed_onset, axis=1).ravel(),
        'members_with_onset': members_with_onset.ravel(),
        'total_members': per_case(total_members[kept]),
        'year': per_case(pd.DatetimeIndex(init_kept).year.to_numpy().astype(np.int64)),
        'obs_onset_date': per_case(pd.DatetimeIndex(obs_date[kept]).strftime('%Y-%m-%d').to_numpy(dtype=object)),
        'bin_index': np.tile(np.arange(n_bins), len(kept)),
    })

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
    DataFrame with forecast-observation pairs, rows in order of init date, lon, lat and bin
    """

    # adjust max_forecast_day to the end of day_bins
    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    # Get the observed onset for the target year
    if target_year not in clim_onset.year.values:
        raise ValueError(f"Target year {target_year} not found in climatological dataset")

    obs_onset_da = clim_onset.sel(year=target_year)

    # Use ALL years as ensemble members (including target year) ### redundant
    ensemble_years = list(clim_onset.year.values)

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

    print(f"Processing {len(lats) * len(lons)} unique lat-lon pairs")

    lat_idx = _first_close(lats, lats)
    lon_idx = _first_close(lons, lons)
    obs_onset = obs_onset_da.values.astype("datetime64[ns]")[np.ix_(lat_idx, lon_idx)]
    has_obs = ~np.isnat(obs_onset)
    obs_onset_doy = np.where(has_obs, pd.DatetimeIndex(obs_onset.ravel()).dayofyear.to_numpy().reshape(obs_onset.shape),
                             0).astype(np.int64)
    ensemble_onset = clim_onset.transpose("year", "lat", "lon").values.astype("datetime64[ns]")[:, lat_idx][:, :, lon_idx]
    ensemble_doy = np.where(np.isnat(ensemble_onset), np.nan,
                            pd.DatetimeIndex(ensemble_onset.ravel()).dayofyear.to_numpy()
                            .reshape(ensemble_onset.shape).astype(float))

    inits = pd.DatetimeIndex([pd.to_datetime(init_date) for init_date in init_dates])
    init_doy = inits.dayofyear.to_numpy().astype(np.int64)

    # A case is a location with an observed onset and an init before it (by day of year),
    # ordered by init date, then lon, then lat
    cases = np.nonzero(has_obs.T[None] & (init_doy[:, None, None] < obs_onset_doy.T[None]))
    init_case, lon_case, lat_case = cases
    case_obs_doy = obs_onset_doy[lat_case, lon_case]
    obs_days_from_init = case_obs_doy - init_doy[init_case]

    # Days from init of each year's onset at the case's location, NaN where the year has none
    member_days = ensemble_doy[:, lat_case, lon_case].T - init_doy[init_case][:, None]
    has_onset = ~np.isnan(member_days)

    members_in_bin = []
    contributing = []
    observed_onset = []
    labels = []
    for bin_start, bin_end in extended_bins:
        if bin_start == -float('inf'):
            labels.append(f'Before day {min_forecast_day}')
            members_in_bin.append(has_onset & (member_days <= min_forecast_day - 1))
            contributing.append(has_onset & (member_days <= 0))
            observed = case_obs_doy <= init_doy[init_case] + min_forecast_day - 1
        elif bin_start > max_forecast_day:
            labels.append(f'After day {max_forecast_day}')
            members_in_bin.append(has_onset & (member_days > max_forecast_day))
            contributing.append(members_in_bin[-1])
            observed = obs_days_from_init > max_forecast_day
        else:
            labels.append(f'Days {bin_start}-{bin_end}')
            members_in_bin.append(has_onset & (bin_start <= member_days) & (member_days <= bin_end))
            contributing.append(members_in_bin[-1])
            observed = (bin_start <= obs_days_from_init) & (obs_days_from_init <= bin_end)
        observed_onset.append(observed.astype(np.int64))

    members_with_onset = np.stack([in_bin.sum(axis=1) for in_bin in members_in_bin], axis=1)
    total_members_with_onset = members_with_onset.sum(axis=1)
    # Skip if no members showed onset
    kept = np.flatnonzero(total_members_with_onset > 0)
    n_bins = len(extended_bins)

    # The years of each bin's members as comma-separated text, built once per distinct set of years
    year_order = np.argsort(np.array(ensemble_years), kind="stable")
    sorted_years = [ensemble_years[n] for n in year_order]
    place_value = 1 << np.arange(len(ensemble_years), dtype=np.int64)
    contributing_years = np.empty((len(kept), n_bins), dtype=object)
    n_contributing_years = np.empty((len(kept), n_bins), dtype=np.int64)
    for b, in_bin in enumerate(contributing):
        in_bin = in_bin[kept][:, year_order]
        sets, which = np.unique(in_bin.astype(np.int64) @ place_value, return_inverse=True)
        names = np.array([','.join(str(sorted_years[n]) for n in range(len(sorted_years)) if (code >> n) & 1)
                          for code in sets], dtype=object)
        contributing_years[:, b] = names[which]
        n_contributing_years[:, b] = in_bin.sum(axis=1)

    def per_case(values):
        """Repeat each per-case value once for every bin."""
        return np.repeat(values, n_bins)

    def per_bin(values):
        """Tile the per-bin values across all kept cases."""
        return np.tile(values, len(kept))

    # Predicted probability uses only members that showed onset
    members_kept = members_with_onset[kept]
    obs_days_kept = obs_days_from_init[kept]
    obs_days_column = np.repeat(obs_days_kept[:, None], n_bins, axis=1)

    forecast_obs_df = pd.DataFrame({
        'init_time': per_case(inits.strftime('%Y-%m-%d').to_numpy(dtype=object)[init_case[kept]]),
        'lat': per_case(lats[lat_case[kept]]),
        'lon': per_case(lons[lon_case[kept]]),
        'bin_start': per_bin(np.array([b[0] for b in extended_bins], dtype=float)),
        'bin_end': per_bin(np.array([b[1] for b in extended_bins], dtype=float)),
        'bin_label': per_bin(np.array(labels, dtype=object)),
        'predicted_prob': (members_kept / total_members_with_onset[kept][:, None]).ravel(),
        'observed_onset': np.stack(observed_onset, axis=1)[kept].ravel(),
        'members_with_onset': members_kept.ravel(),
        'total_members': len(ensemble_years),
        'total_members_with_onset': per_case(total_members_with_onset[kept]),
        'contributing_years': contributing_years.ravel(),
        'n_contributing_years': n_contributing_years.ravel(),
        'year': target_year,
        'obs_onset_date': per_case(pd.DatetimeIndex(obs_onset[lat_case[kept], lon_case[kept]])
                                   .strftime('%Y-%m-%d').to_numpy(dtype=object)),
        'obs_onset_doy': per_case(case_obs_doy[kept]),
        'init_doy': per_case(init_doy[init_case[kept]]),
        'obs_days_from_init_doy': obs_days_column.ravel(),
        'bin_index': per_bin(np.arange(n_bins)),
        'forecast_type': 'climatological_doy',
    })

    if len(forecast_obs_df) == 0:
        print("Warning: No forecast-observation pairs generated")
        return forecast_obs_df

    print(f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs")
    print(f"Unique lat-lon pairs processed: {len(lats) * len(lons)}")
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

    thresh_slice = load_thresholds(**kwargs)

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
            check_alignment_once(rainfall_ds_slice, p_model_slice, **kwargs)

            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds_slice, thresh_slice, year, **kwargs)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")

            print("Computing onset for all ensemble members...")
            onset_all_members, _ = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da,
                                                           history=load_history(year, rainfall_ds_slice, **kwargs), **kwargs)
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
