"""Loading of observed rainfall, thresholds and (deterministic/ensemble) model forecasts from NetCDF files."""
import os
from datetime import datetime
import pandas as pd
import xarray as xr
from momp.utils.standard import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from momp.utils.region import region_select
from momp.utils.practical import restore_args


def onset_steps_needed(*, max_forecast_day=None, wet_spell=None, dry_extent=None, **kwargs):
    """The last lead day ensemble onset detection reads, or None if it can't be told.

    Both paths in momp.stats.detect stop here: the vectorized one at
    max_forecast_day + wet_spell - 1, the per-member loop at that or at
    max_forecast_day + dry_extent when the dry check reaches further.
    """
    if not all(isinstance(value, int) for value in (max_forecast_day, wet_spell, dry_extent)):
        return None
    return max_forecast_day + max(wet_spell - 1, dry_extent)


def load_thresh_file(*, thresh_file, thresh_var, wet_threshold, region, **kwargs):
    """Return the region-selected 2-D threshold from thresh_file, or the scalar wet_threshold if no file is given."""
    if thresh_file:
        thresh_ds = xr.open_dataset(thresh_file)
        thresh_da = thresh_ds[thresh_var]

        thresh_da = region_select(thresh_da, region=region,  **kwargs)

    else:
        thresh_da = wet_threshold

    return thresh_da


def get_initialization_dates(year, *, date_filter_year, init_days, start_date, end_date, **kwargs):
    """
    Get initialization dates for a given year: the days between start_date and end_date
    (month/day only) whose weekday in date_filter_year is in init_days (0 = Monday).
    """
    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]

    start_date = datetime(date_filter_year, *start_MMDD)
    end_date = datetime(date_filter_year, *end_MMDD)
    date_range = pd.date_range(start_date, end_date, freq='D')

    filtered_dates = date_range[date_range.weekday.isin(init_days)]

    # Convert to the requested year
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{year}-%m-%d'))

    return filtered_dates_yr


def load_imd_rainfall(year, *, obs_dir, obs_file_pattern, obs_var, obs_unit_cvt, region,
                      grid_point=False, lat_select=None, lon_select=None, **kwargs):
    """Load IMD daily rainfall NetCDF for a given year."""

    file_patterns = [p.format(year) for p in obs_file_pattern]

    obs_file = None
    for pattern in file_patterns:
        test_path = f"{obs_dir}/{pattern}"
        if os.path.exists(test_path):
            obs_file = test_path
            break

    if obs_file is None:
        available_files = [f for f in os.listdir(obs_dir) if f.endswith('.nc')]
        raise FileNotFoundError(
            f"No observation file found for year {year} in {obs_dir}. "
            f"Tried patterns: {file_patterns}. "
            f"Available files: {available_files}"
        )

    ds = xr.open_dataset(obs_file)

    # Standardize dimension names
    ds = dim_fmt(ds)

    ds = region_select(ds, region=region, **kwargs)

    rainfall = ds[obs_var]

    if obs_unit_cvt:
        rainfall *= obs_unit_cvt

    if grid_point:
        rainfall = rainfall.sel(lat=[lat_select], lon=[lon_select])

    return rainfall


def get_forecast_deterministic_twice_weekly(year, *, model_dir, model_var, date_filter_year, init_days,
                                            start_date, end_date, unit_cvt, file_pattern, region, **kwargs):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    Filters for Mondays and Thursdays in the specified year.
    The forecast file is expected to be named file_pattern.format(year) in the model_dir with
    variable "tp" being daily accumulated rainfall with dimensions (init_time, lat, lon, step).

    Parameters:
    year: int, year to load data for

    Returns:
    p_model: ndarray, precipitation data
    """

    kwargs = restore_args(get_forecast_deterministic_twice_weekly, kwargs, locals())

    fname = file_pattern.format(year)
    file_path = os.path.join(model_dir, fname)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Filter for twice weekly data from daily for the specified year based on 2024 Monday and Thursday dates (to match with IFS CY48R1 reforecasts)

    filtered_dates_yr = get_initialization_dates(year, **kwargs)

    # Load data using xarray
    ds = xr.open_dataset(file_path)

    ds = dim_fmt_model(ds, kwargs.get("model_dims"))

    # Find common dates between desired dates and available dates
    # redundant, len(ds.init_time) or ds.sizes['init_time'] is same as len(matching_times)
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {year}")
    ds = ds.sel(init_time=matching_times)

    ds = region_select(ds, **kwargs)

    # Check if 'step' dimension exists and conditionally slice
    if 'step' in ds.dims:
        # Check if the first value of 'step' is 0, then slice to exclude it
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
    else:
        raise KeyError("'step' dimension not found in dataset")

    p_model = ds[model_var]

    if unit_cvt:
        p_model *= unit_cvt

    ds.close()

    return p_model


def get_forecast_probabilistic_twice_weekly(year, *, model_dir, model_var, date_filter_year, init_days,
                                            start_date, end_date, unit_cvt, file_pattern,
                                            members, region, **kwargs):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    """

    kwargs = restore_args(get_forecast_probabilistic_twice_weekly, kwargs, locals())

    fname = file_pattern.format(year)
    file_path = os.path.join(model_dir, fname)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filtered_dates_yr = get_initialization_dates(year, **kwargs)

    # Load data using xarray
    ds = xr.open_dataset(file_path)

    ds = dim_fmt_model_ensemble(ds, kwargs.get("model_dims"))

    # Find common dates between desired dates and available dates
    # redundant
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]

    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {year}")

    # Select only the matching initialization times
    ds = ds.sel(init_time=matching_times)

    if 'step' in ds.dims:
        # Check if the first value of 'step' is 0, then slice to exclude it
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
        # Onset detection reads no further than this lead day, so loading the
        # rest of a year's ensemble only costs memory (and masking time).
        steps_needed = onset_steps_needed(**kwargs)
        if steps_needed is not None:
            ds = ds.sel(step=slice(None, steps_needed))
    else:
        raise KeyError("'step' dimension not found in dataset")

    if members:
        ds = ds.sel(member = list(members) )

    ds = region_select(ds, **kwargs)

    p_model = ds[model_var]  # in mm

    if unit_cvt:
        p_model *= unit_cvt

    ds.close()

    return p_model
