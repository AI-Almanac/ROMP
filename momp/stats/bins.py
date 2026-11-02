import pandas as pd
import numpy as np
from momp.io.input import load_thresh_file, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_all_members
#from momp.lib.control import restore_args
from momp.utils.practical import restore_args
#from momp.stats.climatology import compute_climatological_onset_dataset


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
    #day_bins = kwargs["stats_day_bins"]
    #max_forecast_day = kwargs["max_forecast_day"]

    max_forecast_day = max(day_bins)[1]
    min_forecast_day = min(day_bins)[0]

    results_list = []

    # Get unique combinations of init_time, lat, lon from the filtered forecast data
    forecast_groups = onset_all_members.groupby(['init_time', 'lat', 'lon'])

    # Add the "after max_forecast_day" bin
    #extended_bins = day_bins + ((max_forecast_day + 1, float('inf')),)
    extended_bins = ((-float('inf'), min_forecast_day - 1),) + day_bins + ((max_forecast_day + 1, float('inf')),)
#    print("\n\n\n XXXXXXXX extended_bins = ", extended_bins)

    print(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}")
    print(f"Including 'after day {max_forecast_day}' bin for members without onset in forecast window")

    for (init_time, lat, lon), group in forecast_groups:

#        print("init_time, lat, lon = ", init_time, lat, lon)
        # Get observed onset for this location
        try:
#            print("onset_da.lat.values = ", onset_da.lat.values)
#            print("lat = ,", lat)
#            print("np.abs(onset_da.lat.values - lat) = ", np.abs(onset_da.lat.values - lat))
            lat_idx = np.where(np.abs(onset_da.lat.values - lat) < 0.01)[0][0]
            lon_idx = np.where(np.abs(onset_da.lon.values - lon) < 0.01)[0][0]
#            print("lat_idx = ", lat_idx, "  lon_idx = ", lon_idx)
#            print("onset_da = ", onset_da)
            obs_date = onset_da.isel(lat=lat_idx, lon=lon_idx).values
        except:
            continue

        # Handle different types of obs_onset values
        if hasattr(obs_date, 'item'):
            obs_date = obs_date.item()

#        print("AAAAAA")
        # Skip if no observed onset
        if pd.isna(obs_date):
            continue

        # Convert dates for comparison
        init_date = pd.to_datetime(init_time)
        obs_date_dt = pd.to_datetime(obs_date)

        # Double-check: Only use forecasts initialized before the observed onset
        if init_date >= obs_date_dt:
            continue

#        print("BBBBBB")
        
        total_members = len(group)

        # For each day bin (including the "after max_forecast_day" bin)
        for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):

            if bin_start == -float('inf'):
            #if bin_start < min_forecast_day:
                bin_label = f'Before day {min_forecast_day}'

                # Check if observed onset occurs before window start 
                forecast_start_date = init_date + pd.Timedelta(days=min_forecast_day)
                observed_onset = int(obs_date_dt.date() < forecast_start_date.date())

                # Count members that didn't predict onset within forecast window
                members_with_onset_in_bin = 0
                #total_members = len(group)

                for member_idx, member_row in group.iterrows():
                    member_onset_day = member_row['onset_day']

                    # Member predicts < min_forecast_day
                    if pd.notna(member_onset_day) and member_onset_day < min_forecast_day:
                        members_with_onset_in_bin += 1


            # Handle the "after max_forecast_day" bin differently
            #if bin_start > max_forecast_day:
            elif bin_start > max_forecast_day:
                bin_label = f'After day {max_forecast_day}'

                # Check if observed onset occurs after max_forecast_day
                forecast_end_date = init_date + pd.Timedelta(days=max_forecast_day)
                observed_onset = int(obs_date_dt.date() > forecast_end_date.date())

                # Count members that didn't predict onset within forecast window
                members_with_onset_in_bin = 0
                #total_members = len(group)

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
                #total_members = len(group)

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
                #'bin_start': bin_start if bin_start <= max_forecast_day else max_forecast_day + 1,
                #'bin_end': bin_end if bin_end <= max_forecast_day else float('inf'),
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
#    print("result_list = ", results_list)
#    print("forecast_obs_df = ", forecast_obs_df)
    #if 2 > 1:
    #    import os
    #    import pickle
    #    fout = os.path.join(kwargs['dir_out'], "forecast_obs_df.pkl")
    #    with open(fout, "wb") as f:
    #        pickle.dump(forecast_obs_df, f)

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
    DataFrame with forecast-observation pairs, one row per (init date, grid point, bin),
    ordered by init date, then longitude, then latitude, then bin.
    """

    if target_year not in clim_onset.year.values:
        raise ValueError(f"Target year {target_year} not found in climatological dataset")

    extended_bins = _extended_day_bins(day_bins)
    min_forecast_day = extended_bins[0][1] + 1
    max_forecast_day = extended_bins[-1][0] - 1

    # Use ALL years as ensemble members (including target year)
    ensemble_years = list(clim_onset.year.values)

    print(f"Creating climatological forecasts for target year {target_year}")
    print(f"Using {len(ensemble_years)} years as ensemble members: {ensemble_years}")
    print(f"Processing {len(init_dates)} initialization dates")
    print(f"Day bins: {day_bins}")
    print(f"Extended bins include: 'Before day {min_forecast_day}' and 'After day {max_forecast_day}' ")
    print("Using day-of-year method for onset comparison")

    grid = _ClimatologyGrid.from_onset(clim_onset, target_year)
    print(f"Processing {grid.n_cells} unique lat-lon pairs")

    frames = [
        _climatological_pairs_for_init(grid, pd.to_datetime(init_date), extended_bins, target_year)
        for init_date in init_dates
    ]
    frames = [frame for frame in frames if len(frame) > 0]

    if not frames:
        print("Warning: No forecast-observation pairs generated")
        return pd.DataFrame()

    forecast_obs_df = pd.concat(frames, ignore_index=True)

    print(f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs")
    print(f"Unique lat-lon pairs processed: {grid.n_cells}")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    unique_locations_in_output = len(forecast_obs_df[['lat', 'lon']].drop_duplicates())
    print(f"Unique locations in output: {unique_locations_in_output}")

    print("\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean',
        'n_contributing_years': 'mean',
        'total_members_with_onset': 'mean'
    }).round(3)
    print(bin_stats)

    return forecast_obs_df


def _extended_day_bins(day_bins):
    """day_bins framed by an open 'before' bin and an open 'after' bin.

    Every bin, including the open ones, holds the onsets whose day from
    initialization d satisfies start <= d <= end.
    """
    min_forecast_day = min(day_bins)[0]
    max_forecast_day = max(day_bins)[1]
    return (
        ((-np.inf, min_forecast_day - 1),)
        + tuple(day_bins)
        + ((max_forecast_day + 1, np.inf),)
    )


def _bin_label(bin_start, bin_end):
    if bin_start == -np.inf:
        return f'Before day {bin_end + 1}'
    if bin_end == np.inf:
        return f'After day {bin_start - 1}'
    return f'Days {bin_start}-{bin_end}'


def _onset_day_of_year(onset_dates):
    """Day of year of each onset date, NaN where there was no onset."""
    return np.asarray(pd.DatetimeIndex(onset_dates).dayofyear, dtype=float)


class _ClimatologyGrid:
    """Onset day-of-year per grid cell, flattened lon-major (lon outer, lat inner)."""

    def __init__(self, cell_lon, cell_lat, years, member_doy, obs_dates, obs_doy):
        self.cell_lon = cell_lon
        self.cell_lat = cell_lat
        self.years = years
        self.member_doy = member_doy
        self.obs_dates = obs_dates
        self.obs_doy = obs_doy

    @property
    def n_cells(self):
        return self.cell_lon.size

    @classmethod
    def from_onset(cls, clim_onset, target_year):
        grid = clim_onset.transpose("year", "lon", "lat")
        lons, lats = grid.lon.values, grid.lat.values
        years = grid.year.values
        member_dates = grid.values.reshape(len(years), -1)
        obs_dates = grid.sel(year=target_year).values.ravel()
        return cls(
            cell_lon=np.repeat(lons, lats.size),
            cell_lat=np.tile(lats, lons.size),
            years=years,
            member_doy=_onset_day_of_year(member_dates.ravel()).reshape(member_dates.shape),
            obs_dates=obs_dates,
            obs_doy=_onset_day_of_year(obs_dates),
        )


def _year_lists(member_in_bin, years):
    """Comma-separated sorted years for each row of a (row, year) membership mask."""
    order = np.argsort(years)
    year_names = np.array([str(year) for year in years[order]])
    packed = np.ascontiguousarray(np.packbits(member_in_bin[:, order], axis=1))
    row_keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    unique_keys, pattern_of_row = np.unique(row_keys, return_inverse=True)
    patterns = np.unpackbits(
        unique_keys.view(np.uint8).reshape(-1, packed.shape[1]), axis=1, count=len(years)
    ).astype(bool)
    names = np.array([','.join(year_names[pattern]) for pattern in patterns], dtype=object)
    return names[pattern_of_row.ravel()]


def _climatological_pairs_for_init(grid, init_date, extended_bins, target_year):
    """Forecast-observation rows for one initialization date, one per (cell, bin).

    Only cells whose observed onset falls after initialization and where at
    least one member year has an onset in some bin produce rows.
    """
    init_doy = init_date.dayofyear
    starts = np.array([bin_start for bin_start, _ in extended_bins], dtype=float)
    ends = np.array([bin_end for _, bin_end in extended_bins], dtype=float)

    member_days = grid.member_doy - init_doy
    in_bin = (member_days >= starts[:, None, None]) & (member_days <= ends[:, None, None])
    members_in_bin = in_bin.sum(axis=1)
    members_with_onset = members_in_bin.sum(axis=0)

    cells = np.flatnonzero((grid.obs_doy > init_doy) & (members_with_onset > 0))
    n_bins = len(extended_bins)
    n_rows = cells.size * n_bins

    def per_row(cell_values):
        return np.repeat(cell_values[cells], n_bins)

    obs_days = grid.obs_doy[cells] - init_doy
    observed = (obs_days >= starts[:, None]) & (obs_days <= ends[:, None])
    members = members_in_bin[:, cells].T.ravel()
    total_with_onset = per_row(members_with_onset)
    member_in_bin = in_bin[:, :, cells].transpose(2, 0, 1).reshape(n_rows, len(grid.years))

    return pd.DataFrame({
        'init_time': init_date.strftime('%Y-%m-%d'),
        'lat': per_row(grid.cell_lat),
        'lon': per_row(grid.cell_lon),
        'bin_start': np.tile(starts, cells.size),
        'bin_end': np.tile(ends, cells.size),
        'bin_label': np.tile([_bin_label(*b) for b in extended_bins], cells.size),
        'predicted_prob': members / total_with_onset,
        'observed_onset': observed.T.ravel().astype(int),
        'members_with_onset': members,
        'total_members': len(grid.years),
        'total_members_with_onset': total_with_onset,
        'contributing_years': _year_lists(member_in_bin, grid.years),
        'n_contributing_years': members,
        'year': target_year,
        'obs_onset_date': per_row(np.datetime_as_string(grid.obs_dates, unit='D')),
        'obs_onset_doy': per_row(grid.obs_doy).astype(int),
        'init_doy': init_doy,
        'obs_days_from_init_doy': np.repeat(obs_days, n_bins).astype(int),
        'bin_index': np.tile(np.arange(n_bins), cells.size),
        'forecast_type': 'climatological_doy',
    })



# This function creates the observed forecast pairs for multiple years (core monsoon zone grids) and combines them
def multi_year_forecast_obs_pairs(*, years, obs_dir, obs_file_pattern, obs_var,
                                  thresh_file, thresh_var, wet_threshold,
                                  date_filter_year, init_days, start_date, end_date,
                                  model_dir, model_var, unit_cvt, file_pattern,
                                  wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, fallback_date, mok,
                                  members, onset_percentage_threshold, max_forecast_day, day_bins, **kwargs):
    """Main function to perform multi-year reliability analysis."""

    kwargs = restore_args(multi_year_forecast_obs_pairs, kwargs, locals())
#    print("\n\n\nmax_forecast_day = ", max_forecast_day)
#    print("max_forecast_day kwargs = ", kwargs['max_forecast_day'])

    #members = kwargs['members']
    #probabilistic = kwargs['probabilistic']

    #mok = kwargs["mok"]
    #window = kwargs["wet_spell"]
    #wet_init = kwargs["wet_init"]
    #dry_spell = kwargs["dry_spell"]
    #dry_extent = kwargs["dry_extent"]
    #dry_threshold = kwargs["dry_threshold"]
    #max_forecast_day = kwargs['max_forecast_day']

    print(f"Processing years: {years}")

    thresh_slice = load_thresh_file(**kwargs)

    # Initialize list to store all forecast-observation pairs
    all_forecast_obs_pairs = []
                 
    # Process each year
    for year in years:
        print(f"\n{'-'*50}")
        print(f"Processing year {year}")
        #print(f"{'='*50}")
                        
        try:            
            # Load model and observation data
            print("Loading S2S model data...")
            #p_model,_ = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
#            p_model_slice = p_model.sel(lat=inside_lats, lon=inside_lons)
            p_model_slice = p_model # !!!!! region subset
                    
            print("Loading observational rainfall data...")
            rainfall_ds = load_imd_rainfall(year, **kwargs)
#            rainfall_ds_slice = rainfall_ds.sel(lat=inside_lats, lon=inside_lons)
            rainfall_ds_slice = rainfall_ds #!!!!! region subset

            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds_slice, thresh_slice, year, **kwargs)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")
                        
            print("Computing onset for all ensemble members...")
            onset_all_members, _ = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da, **kwargs)
            print(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")
    
#            print("onset_all_members = ", onset_all_members)
#            print("onset_da = ", onset_da)
            print("Creating forecast-observation pairs...")
            forecast_obs_pairs = create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, **kwargs)

#            print("DONE!!!")
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
    #print(f"{'='*50}")

    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    # Print final summary statistics
    print(f"\nFinal Summary Statistics:")
    print(f"Years processed: {years}")
    return combined_forecast_obs



#def multi_year_climatological_forecast_obs_pairs(clim_onset, *, years_clim, day_bins, max_forecast_day, date_filter_year, init_days, start_date, end_date, **kwargs):
def multi_year_climatological_forecast_obs_pairs(clim_onset, *, years, day_bins, max_forecast_day, date_filter_year, init_days, start_date, end_date, **kwargs):
    """
    Create climatological forecast-observation pairs for multiple target years.

    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates
    #years_clim : list
    years : list
        Years to use as truth for observations
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins
    max_forecast_day : int, default=15
        Maximum forecast day
    #mok : bool, default=True
    #    Whether to use MOK date filter

    Returns:
    --------
    DataFrame with combined forecast-observation pairs from all target years
    """

    kwargs = restore_args(multi_year_climatological_forecast_obs_pairs, kwargs, locals())

    clim_onset_slice = clim_onset #!!!!! region subset

    all_forecast_obs_pairs = []

    #for target_year in years_clim:
    for target_year in years:
        print(f"\n{'-'*50}")
        print(f"Processing target year {target_year}")
        #print(f"{'='*50}")

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
    #print(f"Target years processed: {years_clim}")
    print(f"Target years processed: {years}")
    print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
    print(f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}")
    print(f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}")

    return combined_forecast_obs



