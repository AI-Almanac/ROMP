"""Onset detection on observed rainfall and on deterministic or ensemble model forecasts."""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from momp.utils.practical import restore_args
from itertools import product


def find_first_true(arr):
    """
    Find first occurrence of onset condition for each grid point.

    Works if arr contains floats, integers, or NaNs.
    Returns the index of the first True value, or -1 if none.
    """
    # Convert to boolean, treat NaN as False
    arr_bool = np.asarray(arr, dtype=float)  # ensure numeric
    arr_bool = np.nan_to_num(arr_bool, nan=0.0)  # NaN -> 0
    arr_bool = arr_bool.astype(bool)  # convert to boolean

    if arr_bool.any():
        return int(np.argmax(arr_bool))  # first True index
    else:
        return -1


def detect_onset(day, forecast_series, thresh, *, wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, **kwargs):
    """
    detect onset for model forecast
    ---
    forecast_series: grid point xarray time series
    """

    # !!! day start from index 1 in data as step
    start_idx = day - 1

    if dry_extent <= wet_spell:
        end_idx = start_idx + wet_spell

        if end_idx <= len(forecast_series):
            window_series = forecast_series[start_idx:end_idx]

            # Check basic onset condition
            if np.all(window_series >=  wet_init) and np.nansum(window_series) > thresh:
                return True

            # check if followed by dry spell
    else:
        end_idx = start_idx + wet_spell
        end_idx_dry = start_idx + dry_extent

        if end_idx_dry <= len(forecast_series):
            window_series = forecast_series[start_idx:end_idx]
            dry_series = forecast_series[start_idx:end_idx_dry]

            if np.all(window_series >=  wet_init) and np.nansum(window_series) > thresh:

                dry_bool = dry_series < wet_init
                consec_dry = np.convolve(dry_bool, np.ones(dry_spell, dtype=int), 'valid')
                has_dry_spell = np.any(consec_dry == dry_spell)

                if not has_dry_spell:
                    return True


# Function to detect observed onset dates based on rainfall threshold file
def detect_observed_onset(rain_slice, thresh_slice, year, *, wet_init, wet_spell, 
                          dry_spell, dry_threshold, dry_extent, start_date, end_date, fallback_date, earliest_onset_date, 
                          extend_end_day=47, **kwargs):
    """Detect observed onset dates for a given year."""

    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]
    fallback_MMDD = fallback_date

    end_date = datetime(year, *end_MMDD)
    if extend_end_day:
        end_date = end_date + timedelta(days=extend_end_day)

    # Start the search at earliest_onset_date if set, otherwise at start_date's month and day
    if earliest_onset_date:
        start_date = datetime(year, *earliest_onset_date)
        date_label = f"{earliest_onset_date[0]:02d}-{earliest_onset_date[1]:02d}"

    else:
        start_date = datetime(year, *start_MMDD)  # default start_date 
        date_label = f"{start_MMDD[0]:02d}-{start_MMDD[1]:02d}"

    # Find start date index
    time_dates = pd.to_datetime(rain_slice.time.values)
    start_idx_candidates = np.where(time_dates >= start_date)[0]

    if len(start_idx_candidates) == 0 and fallback_date:
        print(f"Warning: {date_label} not found in data for year {year}")
        fallback_date = datetime(year, *fallback_MMDD)
        start_idx = np.where(time_dates >= fallback_date)[0][0]
        print(f"Using fallback date: April 1st")
    else:
        start_idx = start_idx_candidates[0]

    # Subset rain_slice from start date onward
    rain_subset = rain_slice.isel(time=slice(start_idx, None)).sel(time=slice(None,end_date))

    # Create rolling 5-day sums
    rolling_sum = rain_subset.rolling(time=wet_spell, min_periods=wet_spell, center=False).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(wet_spell-1))

    # Create onset condition
    wet_day_condition = rain_subset >= wet_init
    wet_day_spell = (wet_day_condition.rolling(time=wet_spell,
                                                       min_periods=wet_spell, center=False).reduce(np.all))

    first_day_condition = wet_day_spell.shift(time=-(wet_spell-1))
    first_day_condition = first_day_condition.fillna(False).astype(bool) # convert nans to bool

    sum_condition = rolling_sum_aligned > thresh_slice

    # check false onset
    if dry_extent >= dry_spell and dry_extent > wet_spell:
        dry_day_condition = rain_subset < wet_init

        dry_day_spell = (dry_day_condition.rolling(time=dry_spell, min_periods=dry_spell)
                         .sum() == dry_spell)

        has_dry_after = (dry_day_spell.rolling(time=dry_extent+1, min_periods=1)
                         .sum() > 0 ).shift(time=-dry_extent).fillna(False).astype(bool)

        no_dry_after = xr.apply_ufunc(np.logical_not, has_dry_after)

        onset_condition = first_day_condition & sum_condition & no_dry_after

    else:
        onset_condition = first_day_condition & sum_condition

    onset_indices = xr.apply_ufunc(
        find_first_true,
        onset_condition,
        input_core_dims=[['time']],
        output_dtypes=[int],
        vectorize=True
    )

    # Convert indices to actual dates
    valid_mask = onset_indices.values >= 0
    time_coords = rain_subset.time.values
    onset_dates_array = np.full(onset_indices.shape, np.datetime64('NaT'), dtype='datetime64[ns]')

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices[i, j].values)
                if 0 <= idx < len(time_coords):
                    onset_dates_array[i, j] = time_coords[idx]

    # Create final onset date DataArray
    onset_da = xr.DataArray(
        onset_dates_array,
        coords=[('lat', rain_slice.lat.values), ('lon', rain_slice.lon.values)],
        name='onset_date'
    )

    return onset_da


def compute_onset_for_deterministic_model(p_model, thresh_slice, onset_da, *,
                                          wet_init, wet_spell, dry_spell, dry_threshold, dry_extent,
                                          max_forecast_day, earliest_onset_date, end_date, **kwargs):

    """Compute onset dates for deterministic model forecast."""

    kwargs = restore_args(compute_onset_for_deterministic_model, kwargs, locals())

    results_list = []

    init_times = p_model.init_time.values
    lats = p_model.lat.values
    lons = p_model.lon.values

    print(f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons...")

    if dry_extent <= wet_spell:
        max_steps_needed = max_forecast_day + wet_spell - 1
    else:
        max_steps_needed = max_forecast_day + dry_extent

    full_steps = p_model.sizes['step']
    if full_steps < max_steps_needed:
        raise ValueError(f"Not enough forecast time steps: model steps {full_steps} \
                < min steps required {max_steps_needed}, consider decrese dry_extent value")

    total_potential_inits = 0
    valid_inits = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    onsets_found = 0

    for t_idx, init_time in enumerate(init_times):

        init_date = pd.to_datetime(init_time)
        year = init_date.year

        if earliest_onset_date:
            earliest_date = datetime(year, *earliest_onset_date) 
        else:
            earliest_date = None

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):

                total_potential_inits += 1

                try:
                    obs_onset = onset_da.isel(lat=i, lon=j).values
                except:
                    skipped_no_obs += 1
                    continue

                if pd.isna(obs_onset):
                    skipped_no_obs += 1
                    continue

                obs_onset_dt = pd.to_datetime(obs_onset)

                if init_date >= obs_onset_dt:
                    skipped_late_init += 1
                    continue

                valid_inits += 1
                # check if thresh is a 2-D array or scalar
                if not np.isscalar(thresh_slice):
                    thresh = thresh_slice.isel(lat=i, lon=j).values
                else:
                    thresh = thresh_slice

                try:
                    forecast_series = p_model.isel(
                        init_time=t_idx,
                        lat=i,
                        lon=j,
                    ).sel(step=slice(1, max_steps_needed)).values

                    if len(forecast_series) < max_steps_needed:
                        onset_day = None
                    else:
                        onset_day = None

                        for day in range(1, max_forecast_day + 1):

                            isonset = detect_onset(day, forecast_series, thresh, **kwargs)

                            if isonset:
                                # Calculate the actual date this forecast day represents
                                forecast_date = init_date + pd.Timedelta(days=day)

                                # With earliest_onset_date set, only count onsets on or after that date
                                if earliest_onset_date:
                                    if forecast_date.date() >= earliest_date.date():
                                        onset_day = day
                                        break
                                else:
                                    onset_day = day
                                    break

                except Exception as e:
                    print(f"Error at init_time {t_idx}, lat {i}, lon {j}: {e}")
                    onset_day = None

                onset_date = None
                if onset_day is not None:
                    onsets_found += 1
                    onset_date = init_date + pd.Timedelta(days=onset_day)

                result = {
                    'init_time': init_time,
                    'lat': lat,
                    'lon': lon,
                    'onset_day': onset_day,
                    'onset_date': onset_date.strftime('%Y-%m-%d') if onset_date is not None else None,
                    'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')
                }
                results_list.append(result)

    onset_df = pd.DataFrame(results_list)

    print(f"\nProcessing Summary:")
    print(f"Total potential initializations: {total_potential_inits}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid initializations processed: {valid_inits}")
    print(f"Onsets found: {onsets_found}")
    print(f"Onset rate: {onsets_found/valid_inits:.3f}" if valid_inits > 0 else "Onset rate: 0.000")

    return onset_df     


# Function to compute onset dates for all ensemble members and save it as DataFrame
# in binned_skill_score_cmz.py
def _compute_onset_for_all_members_loop(p_model, thresh_slice, onset_da, *, wet_init, wet_spell,
                                        dry_spell, dry_threshold, dry_extent, members, onset_percentage_threshold,
                                        max_forecast_day, earliest_onset_date, end_date, **kwargs):

    """Compute onset dates for each ensemble member, initialization time, and grid point."""

    kwargs = restore_args(_compute_onset_for_all_members_loop, kwargs, locals())

    results_list = []
    results_mean_list = []

    # Get dimensions
    init_times = p_model.init_time.values

    if not members:
        members = p_model.member.values

    # Get the actual lat/lon coordinates from the data
    lats = p_model.lat.values
    lons = p_model.lon.values

    # Create unique lat-lon pairs (no repetition)
    unique_pairs = list(product(lons, lats))

    print(f"Processing {len(init_times)} init times x {len(unique_pairs)} unique locations x {len(members)} members...")

    if dry_extent <= wet_spell:
        max_steps_needed = max_forecast_day + wet_spell - 1
    else:
        max_steps_needed = max_forecast_day + dry_extent

    full_steps = p_model.sizes['step']
    if full_steps < max_steps_needed:
        raise ValueError(f"Not enough forecast time steps: model steps {full_steps} \
                < min steps required {max_steps_needed}, consider decrese dry_extent value")

    # Track statistics
    total_potential_forecasts = 0
    valid_inits = 0
    valid_forecasts = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    ensemble_onsets_found = 0

    # Loop over all combinations
    for t_idx, init_time in enumerate(init_times):
        if t_idx % 5 == 0:
            print(f"Processing init time {t_idx+1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}")

        init_date = pd.to_datetime(init_time)
        year = init_date.year

        if earliest_onset_date:
            earliest_date = datetime(year, *earliest_onset_date)

        # Loop over unique lat-lon pairs only
        for lon_idx, lat_idx in product(range(len(lons)), range(len(lats))):
            lon = lons[lon_idx]
            lat = lats[lat_idx]

            total_potential_forecasts += len(members)

            # Get observed onset date for this location
            try:
                obs_onset = onset_da.isel(lat=lat_idx, lon=lon_idx).values
            except:
                skipped_no_obs += len(members)
                continue

            # Skip if no observed onset
            if pd.isna(obs_onset):
                skipped_no_obs += len(members)
                continue

            # Convert observed onset to datetime
            obs_onset_dt = pd.to_datetime(obs_onset)

            # Only process if forecast was initialized before observed onset
            if init_date >= obs_onset_dt:
                skipped_late_init += len(members)
                continue

            valid_inits +=1

            # Get threshold for this location
            if not np.isscalar(thresh_slice):
                thresh = thresh_slice.isel(lat=lat_idx, lon=lon_idx).values
            else:
                thresh = thresh_slice

            # Collect onset days for all members at this init/location
            member_onset_days = []

            for m_idx, member in enumerate(members):

                valid_forecasts += 1

                try:
                    # Extract forecast time series for this member and location
                    forecast_series = p_model.isel(
                        init_time=t_idx,
                        lat=lat_idx,
                        lon=lon_idx,
                        ).sel(step=slice(1, max_steps_needed)).sel(member=member).values

                    if len(forecast_series) < max_steps_needed:
                        member_onset_days.append(None)
                        continue

                    # Check for onset on each possible day
                    onset_day = None

                    for day in range(1, max_forecast_day + 1):

                        isonset = detect_onset(day, forecast_series, thresh, **kwargs)

                        if isonset:
                            # Calculate the actual date this forecast day represents
                            forecast_date = init_date + pd.Timedelta(days=day)

                            # With earliest_onset_date set, only count onsets on or after that date
                            if earliest_onset_date:
                                if forecast_date.date() >= earliest_date.date():
                                    onset_day = day
                                    break
                            else:
                                onset_day = day
                                break

                    member_onset_days.append(onset_day)

                    # Store result
                    result = {
                        'init_time': init_time,
                        'lat': lat,
                        'lon': lon,
                        'member': member,
                        'onset_day': onset_day,
                        'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')
                    }
                    results_list.append(result)

                except Exception as e:
                    print(f"Error at init_time {t_idx}, location ({lon}, {lat}), member {member}: {e}")
                    raise
                    continue

            # Now check if at least 50% of members have onset
            valid_onsets = [day for day in member_onset_days if day is not None]
            onset_count = len(valid_onsets)
            total_members = len(member_onset_days)
            onset_percentage = onset_count / total_members if total_members > 0 else 0

            # Determine ensemble onset day
            ensemble_onset_day = None
            ensemble_onset_date = None
            if onset_percentage >= onset_percentage_threshold:  # At least 50% of members have onset
                # Use rounding of mean onset day
                mean_onset = np.mean(valid_onsets)
                ensemble_onset_day = int(round(mean_onset))
                ensemble_onsets_found += 1
                ensemble_onset_date = init_date + pd.Timedelta(days=ensemble_onset_day)

            # Store result
            result_mean = {
                'init_time': init_time,
                'lat': lat,
                'lon': lon,
                'onset_day': ensemble_onset_day,  # None if <50% members have onset
                'onset_date': ensemble_onset_date.strftime('%Y-%m-%d') if ensemble_onset_date is not None else None,
                'member_onset_count': onset_count,
                'total_members': total_members,
                'onset_percentage': onset_percentage,
                'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
            }
            results_mean_list.append(result_mean)

    # Convert to DataFrame
    onset_df = pd.DataFrame(results_list)
    onset_mean_df = pd.DataFrame(results_mean_list)

    print(f"\nProcessing Summary:")
    print(f"Total potential forecasts: {total_potential_forecasts}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid forecasts processed: {valid_forecasts}")
    print(f"Generated {len(onset_df)} member-forecast combinations")
    print(f"Found onset in {onset_df['onset_day'].notna().sum()} cases")
    print(f"Onset rate: {onset_df['onset_day'].notna().mean():.3f}")
    print(f"Ensemble onsets found (≥50% members): {ensemble_onsets_found}")
    print(f"Ensemble onset rate: {ensemble_onsets_found/valid_inits:.3f}" if valid_inits > 0 else "Ensemble onset rate: 0.000")

    # Check for uniqueness
    unique_combinations = onset_df.groupby(['init_time', 'lat', 'lon', 'member']).size()
    if (unique_combinations > 1).any():
        print(f"Warning: Found {(unique_combinations > 1).sum()} duplicate combinations!")
    else:
        print("✓ All init_time-lat-lon-member combinations are unique")

    return onset_df, onset_mean_df


def _valid_vectorized_member_inputs(p_model, members):
    """Check that p_model has the expected dims and contains all requested members."""
    expected_dims = {"init_time", "member", "step", "lat", "lon"}
    if not expected_dims.issubset(set(p_model.dims)):
        return False

    if members:
        model_members = set(p_model.member.values.tolist())
        return set(members).issubset(model_members)

    return True


# How many values (init dates x members x grid points x lead days) onset
# detection works on at once. Enough to keep the array work efficient, small
# enough that a year of a large ensemble never needs several full-size copies.
_ONSET_BLOCK_ELEMENTS = 16_000_000


def _first_onset_steps(rain, thresh_slice, valid_step, *, wet_init, wet_spell, max_forecast_day):
    """The first lead day a wet spell starts on, per init date, member and grid point.

    NaN where none starts within max_forecast_day. A spell starts on day k when
    days k .. k + wet_spell - 1 are all at least wet_init and their total
    exceeds the threshold. This gives the same answer as the rolling-window
    formulation it replaces, a block of init dates at a time: wet days are
    counted with a running sum rather than by materializing every window, and
    the spell total is added day by day in the order numpy's rolling sum adds it.

    rain: (init_time, member, lat, lon, step), in memory.
    valid_step: (init_time, step) mask of lead days allowed, or None.
    """
    values = rain.values
    steps = rain.step.values.astype(float)
    n_windows = values.shape[-1] - wet_spell + 1
    onset_days = np.full(values.shape[:-1], np.nan)
    if n_windows <= 0:
        return onset_days
    usable = steps[:n_windows] <= max_forecast_day
    if isinstance(thresh_slice, xr.DataArray):
        grid = rain.isel(init_time=0, member=0, step=0, drop=True)
        thresh = thresh_slice.broadcast_like(grid).transpose("lat", "lon").values[..., None]
    else:
        thresh = np.asarray(thresh_slice)

    per_init = max(1, int(np.prod(values.shape[1:])))
    block = max(1, _ONSET_BLOCK_ELEMENTS // per_init)
    for start in range(0, values.shape[0], block):
        chunk = values[start : start + block]
        counts = np.cumsum(chunk >= wet_init, axis=-1, dtype=np.int32)
        counts = np.concatenate(
            [np.zeros(counts.shape[:-1] + (1,), dtype=np.int32), counts], axis=-1
        )
        all_wet = (counts[..., wet_spell:] - counts[..., :n_windows]) == wet_spell
        total = chunk[..., :n_windows].copy()
        for offset in range(1, wet_spell):
            total += chunk[..., offset : offset + n_windows]
        onset = all_wet & (total > thresh) & usable
        if valid_step is not None:
            onset &= valid_step[start : start + block, None, None, None, :n_windows]
        first = onset.argmax(axis=-1)
        onset_days[start : start + block] = np.where(onset.any(axis=-1), steps[first], np.nan)
    return onset_days


def compute_onset_for_all_members_vectorized(
    p_model,
    thresh_slice,
    onset_da,
    *,
    wet_init,
    wet_spell,
    dry_spell,
    dry_threshold,
    dry_extent,
    members,
    onset_percentage_threshold,
    max_forecast_day,
    earliest_onset_date,
    end_date,
    **kwargs,
):
    """Vectorized onset detection for the common no-post-onset-dry-window case."""

    if dry_extent > wet_spell:
        raise ValueError("vectorized onset detection supports only dry_extent <= wet_spell")

    if not members:
        members = tuple(p_model.member.values.tolist())
    else:
        members = tuple(members)

    max_steps_needed = max_forecast_day + wet_spell - 1
    full_steps = p_model.sizes["step"]
    if full_steps < max_steps_needed:
        raise ValueError(
            f"Not enough forecast time steps: model steps {full_steps} "
            f"< min steps required {max_steps_needed}, consider decrese dry_extent value"
        )

    # Selecting every member, in order, is a no-op that would still copy the cube.
    if list(p_model.member.values) != list(members):
        p_model = p_model.sel(member=list(members))
    p_model = p_model.sel(step=slice(1, max_steps_needed))
    rain = p_model.transpose("init_time", "member", "lat", "lon", "step")
    if isinstance(thresh_slice, xr.DataArray):
        # The inner join the element-wise comparison with the threshold made.
        rain, thresh_slice = xr.align(rain, thresh_slice, join="inner")

    valid_step = None
    if earliest_onset_date:
        init_times = pd.to_datetime(rain.init_time.values)
        step_values = rain.step.values.astype(int)
        valid_step = np.zeros((len(init_times), len(step_values)), dtype=bool)
        for t_idx, init_date in enumerate(init_times):
            earliest_date = pd.Timestamp(datetime(init_date.year, *earliest_onset_date))
            forecast_dates = init_date + pd.to_timedelta(step_values, unit="D")
            valid_step[t_idx, :] = forecast_dates.date >= earliest_date.date()

    obs_onset = onset_da.transpose("lat", "lon")
    obs_values = obs_onset.values.astype("datetime64[ns]")
    obs_valid = ~np.isnat(obs_values)
    init_values = rain.init_time.values.astype("datetime64[ns]")
    valid_case_values = obs_valid[None, :, :] & (init_values[:, None, None] < obs_values[None, :, :])
    valid_case = xr.DataArray(
        valid_case_values,
        dims=("init_time", "lat", "lon"),
        coords={
            "init_time": rain.init_time,
            "lat": rain.lat,
            "lon": rain.lon,
        },
    )

    onset_day_values = _first_onset_steps(
        rain,
        thresh_slice,
        valid_step,
        wet_init=wet_init,
        wet_spell=wet_spell,
        max_forecast_day=max_forecast_day,
    )

    onset_da_members = xr.DataArray(
        onset_day_values,
        dims=("init_time", "member", "lat", "lon"),
        coords={
            "init_time": rain.init_time,
            "member": rain.member,
            "lat": rain.lat,
            "lon": rain.lon,
        },
        name="onset_day",
    ).where(valid_case)

    obs_onset_for_cases = obs_onset.broadcast_like(valid_case).where(valid_case)
    obs_onset_for_members = obs_onset_for_cases.broadcast_like(onset_da_members)

    onset_df = onset_da_members.to_dataframe().reset_index()
    onset_df["obs_onset_date"] = (
        obs_onset_for_members.to_dataframe(name="obs_onset_date")
        .reset_index()["obs_onset_date"]
    )
    onset_df = onset_df[onset_df["obs_onset_date"].notna()].copy()
    onset_df["obs_onset_date"] = pd.to_datetime(onset_df["obs_onset_date"]).dt.strftime("%Y-%m-%d")
    onset_df["onset_day"] = onset_df["onset_day"].where(onset_df["onset_day"].notna(), None)

    onset_count = onset_da_members.notnull().sum("member")
    total_members = len(members)
    onset_percentage = onset_count / total_members if total_members > 0 else 0
    ensemble_onset_day = onset_da_members.mean("member", skipna=True).round()
    ensemble_onset_day = ensemble_onset_day.where(onset_percentage >= onset_percentage_threshold)
    ensemble_onset_day = ensemble_onset_day.where(valid_case)

    onset_mean_df = ensemble_onset_day.to_dataframe(name="onset_day").reset_index()
    onset_mean_df["member_onset_count"] = (
        onset_count.to_dataframe(name="member_onset_count").reset_index()["member_onset_count"]
    )
    onset_mean_df["total_members"] = total_members
    onset_mean_df["onset_percentage"] = (
        onset_percentage.to_dataframe(name="onset_percentage").reset_index()["onset_percentage"]
    )
    onset_mean_df["obs_onset_date"] = (
        obs_onset_for_cases.to_dataframe(name="obs_onset_date").reset_index()["obs_onset_date"]
    )
    onset_mean_df = onset_mean_df[onset_mean_df["obs_onset_date"].notna()].copy()
    onset_mean_df["onset_date"] = None

    has_ensemble_onset = onset_mean_df["onset_day"].notna()
    if has_ensemble_onset.any():
        init_dates = pd.to_datetime(onset_mean_df.loc[has_ensemble_onset, "init_time"])
        onset_days = onset_mean_df.loc[has_ensemble_onset, "onset_day"].astype(int)
        onset_dates = init_dates + pd.to_timedelta(onset_days, unit="D")
        onset_mean_df.loc[has_ensemble_onset, "onset_date"] = onset_dates.dt.strftime("%Y-%m-%d")

    onset_mean_df["obs_onset_date"] = pd.to_datetime(onset_mean_df["obs_onset_date"]).dt.strftime("%Y-%m-%d")
    onset_mean_df["onset_day"] = onset_mean_df["onset_day"].where(onset_mean_df["onset_day"].notna(), None)

    onset_df = onset_df[["init_time", "lat", "lon", "member", "onset_day", "obs_onset_date"]]
    onset_mean_df = onset_mean_df[
        [
            "init_time",
            "lat",
            "lon",
            "onset_day",
            "onset_date",
            "member_onset_count",
            "total_members",
            "onset_percentage",
            "obs_onset_date",
        ]
    ]

    print(f"\nProcessing Summary:")
    print(f"Generated {len(onset_df)} member-forecast combinations")
    print(f"Found onset in {onset_df['onset_day'].notna().sum()} cases")
    print(f"Onset rate: {onset_df['onset_day'].notna().mean():.3f}")

    return onset_df, onset_mean_df


def compute_onset_for_all_members(p_model, thresh_slice, onset_da, *, wet_init, wet_spell,
                                  dry_spell, dry_threshold, dry_extent, members, onset_percentage_threshold,
                                  max_forecast_day, earliest_onset_date, end_date, **kwargs):
    """Compute per-member and ensemble onsets, using the vectorized path when it applies.

    Uses compute_onset_for_all_members_vectorized when dry_extent <= wet_spell and
    p_model has the expected dims and members; otherwise falls back to the loop version.
    Returns (member onset DataFrame, ensemble onset DataFrame).
    """

    kwargs = restore_args(compute_onset_for_all_members, kwargs, locals())

    if dry_extent <= wet_spell and _valid_vectorized_member_inputs(p_model, members):
        return compute_onset_for_all_members_vectorized(p_model, thresh_slice, onset_da, **kwargs)

    return _compute_onset_for_all_members_loop(p_model, thresh_slice, onset_da, **kwargs)
