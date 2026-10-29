"""Loading of observed rainfall, thresholds and (deterministic/ensemble) model forecasts from NetCDF files."""
import hashlib
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from momp.lib.onset_config import Thresholds, forecast_reach, history_needs_previous_year, is_legacy, onset_setup
from momp.onset import definition as definition_mod
from momp.onset.climatology import ThresholdCache, compute_thresholds_from_forecasts, harmonic_seasons
from momp.onset.ingest import IngestInfo, ingest, ingest_forecast_info
from momp.utils.standard import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from momp.utils.region import region_select
from momp.utils.practical import restore_args
import logging

log = logging.getLogger(__name__)


def onset_steps_needed(*, max_forecast_day=None, wet_spell=None, dry_extent=None, **kwargs):
    """The last lead day onset detection reads, or None if it can't be told.

    With a definition, the largest day any requirement or veto of the compiled
    plan reads for the last candidate day max_forecast_day, so loading never
    truncates what the rule reads. For the old flat keys, the formula of the
    old detection: max_forecast_day + max(wet_spell - 1, dry_extent).
    """
    if is_legacy(kwargs):
        if not all(isinstance(value, int) for value in (max_forecast_day, wet_spell, dry_extent)):
            return None
        return max_forecast_day + max(wet_spell - 1, dry_extent)
    if not isinstance(max_forecast_day, int):
        return None
    return max_forecast_day + forecast_reach(onset_setup(kwargs)) - 1


def load_thresh_file(*, thresh_file, thresh_var, wet_threshold, region, **kwargs):
    """Return the region-selected 2-D threshold from thresh_file, or the scalar wet_threshold if no file is given."""
    if thresh_file:
        thresh_ds = xr.open_dataset(thresh_file)
        thresh_da = thresh_ds[thresh_var]

        thresh_da = region_select(thresh_da, region=region, **{**kwargs, "mask": None})

    else:
        thresh_da = wet_threshold

    return thresh_da


class _LazyRain:
    """A DailyRain whose data is read only when something other than its grid and source is asked for.

    The threshold cache keys on the grid and source, so a cache hit never loads the reference years.
    """

    def __init__(self, probe, load):
        """Take dims, coords and source from the probe (one year); `load` returns the full DailyRain."""
        self.dims, self.coords, self.source = probe.dims, probe.coords, probe.source
        self._load, self._full = load, None

    def __getattr__(self, name):
        """Load the full reference rainfall on first use and read the attribute from it."""
        if name.startswith("_"):
            raise AttributeError(name)
        if self._full is None:
            self._full = self._load()
        return getattr(self._full, name)


_reported_ingest = set()


def ingest_model_forecast(cube, setup, kwargs, source=None):
    """Normalise a model's forecast cube (already in mm, dims init_time, [member], lat, lon, step) to ForecastRain.

    Uses the model's own `accumulation` ("daily", "since_init" or "auto") and `daily_boundary` from kwargs
    (the config's per-model values; "daily" when absent), so accumulated fields are differenced and sub-daily steps summed to days
    exactly as momp.onset.ingest.ingest_forecast does. source labels the data (default: the model's name); how the
    days were made is logged once per model.
    """
    source = source or str(kwargs.get("model") or "forecast")
    accumulation = kwargs.get("accumulation") or "daily"
    boundary = kwargs.get("daily_boundary") or None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IngestInfo)
        forecast, info = ingest_forecast_info(cube, day_boundary=setup.forecast_boundary, lead_label=setup.lead_label,
                                              units="mm", source=source, accumulation=accumulation,
                                              daily_boundary=boundary)
    forecast.ingest_info = info
    key = (str(kwargs.get("model")), info.accumulation, info.resolution, info.daily_boundary)
    if key not in _reported_ingest:
        _reported_ingest.add(key)
        text = f"{kwargs.get('model') or source}: accumulation {info.accumulation} [{info.accumulation_evidence}]; steps {info.resolution}"
        if info.daily_boundary:
            text += f"; days cut at {info.daily_boundary}"
        log.info(text + "".join(f"; {note}" for note in info.notes))
    return forecast


def _reference_rain(setup, years, kwargs):
    """The observations of years[0]..years[1] as ingested DailyRain, labelled with their source."""
    source = (f"{kwargs.get('obs')}:{kwargs.get('obs_dir')}:{kwargs.get('obs_file_pattern')}:"
              f"{kwargs.get('obs_var')}:cvt={kwargs.get('obs_unit_cvt')}")
    return ingest(
        xr.concat([load_imd_rainfall(year, **kwargs) for year in range(years[0], years[1] + 1)], dim="time"),
        day_boundary=setup.obs_day_boundary, units="mm", source=source)


def _harmonic_cached(cache_dir, rain, years):
    """The harmonic season analysis of the reference years, read from or written to the cache directory."""
    grid = hashlib.sha256()
    for dim in rain.dims[:-1]:
        grid.update(dim.encode())
        grid.update(np.ascontiguousarray(rain.coords[dim]).tobytes())
    payload = json.dumps(["harmonic", rain.source, [int(years[0]), int(years[1])], grid.hexdigest()])
    path = Path(cache_dir) / f"harmonic_{hashlib.sha256(payload.encode()).hexdigest()[:16]}.nc"
    if path.exists():
        with xr.open_dataset(path) as stored:
            return stored.load()
    result = harmonic_seasons(rain, years)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(path)
    return result


def _model_forecasts(setup, years, kwargs):
    """Yield the model's forecasts of each reference year as ForecastRain, with ROMP's loader and unit conversion."""
    loader = get_forecast_probabilistic_twice_weekly if kwargs.get("probabilistic") \
        else get_forecast_deterministic_twice_weekly
    found = False
    for year in range(years[0], years[1] + 1):
        try:
            cube = loader(year, **kwargs)
        except FileNotFoundError as exc:
            log.warning(f"Model climatology: skipping year {year} ({exc})")
            continue
        found = True
        order = ("init_time",) + (("member",) if "member" in cube.dims else ()) + ("lat", "lon", "step")
        yield ingest_model_forecast(cube.transpose(*order), setup, kwargs, str(kwargs.get("model")))
    if not found:
        raise FileNotFoundError(f"no forecast file of {kwargs.get('model')} for the years {years[0]}-{years[1]}")


def _model_thresholds(setup, pending, years, cache_dir, kwargs):
    """The model's own climatology of the pending thresholds, read from or written to the cache directory.

    Each verification window of the run (verification_window_list, already
    shortened to the usable leads) gets its own threshold, pooled over the leads
    in that window (spec section 13); without the list one threshold pools every lead.
    """
    window = (1, int(kwargs["max_forecast_day"]))
    windows = [(int(a), int(b)) for a, b in kwargs["verification_window_list"]] \
        if kwargs.get("verification_window_list") else None
    key = json.dumps([str(kwargs.get("model")), str(kwargs.get("model_dir")), [int(years[0]), int(years[1])],
                      list(window), windows, definition_mod.definition_hash(setup.definition), [repr(p) for p in pending],
                      setup.lead_label, [int(d) for d in kwargs.get("init_days", ())],
                      None if not kwargs.get("members") else [int(m) for m in kwargs["members"]]])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    paths = {name: Path(cache_dir) / f"model_{name}_{digest}.nc" for name, _ in pending}
    out = {}
    if all(path.exists() for path in paths.values()):
        for name, path in paths.items():
            with xr.open_dataarray(path) as stored:
                out[name] = stored.load()
        return out
    names = {name for name, _ in pending}
    from dataclasses import replace
    reduced = replace(setup.definition, thresholds=tuple((n, sp) for n, sp in setup.definition.thresholds if n in names))
    computed = compute_thresholds_from_forecasts(reduced, _model_forecasts(setup, years, kwargs), years, window, windows)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for name, array in computed.items():
        array.to_netcdf(paths[name])
        out[name] = array
    return out


def load_onset_thresholds(setup, *, dir_out, region, **kwargs):
    """The named thresholds of the definition, as {name: DataArray(lat, lon)} in mm.

    Files given in `thresholds` are used as they are; the others are computed from
    the observations of threshold_years and cached under the run's cache_dir (shared by runs), or
    dir_out/.onset_cache when none is set.
    The returned Thresholds also carries season_climatology for derived seasons.
    """
    thresholds = Thresholds()
    if not setup.uses_thresholds:
        return thresholds
    for name, path, var in setup.threshold_files:
        with xr.open_dataset(path) as ds:
            thresholds[name] = region_select(ds[var], region=region, **{**kwargs, "mask": None}).load()
    pending = [(name, spec) for name, spec in setup.definition.thresholds if name not in thresholds]
    if not pending and not setup.derived_seasons:
        return thresholds
    years = setup.threshold_years
    cache_dir = Path(kwargs["cache_dir"]) if kwargs.get("cache_dir") else Path(dir_out) / ".onset_cache"
    if pending and setup.threshold_source == "model":
        thresholds.update(_model_thresholds(setup, pending, years, cache_dir,
                                            {"dir_out": dir_out, "region": region, **kwargs}))
        pending = []
        if not setup.derived_seasons:
            return thresholds
    probe = _reference_rain(setup, (years[0], years[0]), {"dir_out": dir_out, "region": region, **kwargs})
    full = _LazyRain(probe, lambda: _reference_rain(setup, years, {"region": region, **kwargs}))
    cache = ThresholdCache(cache_dir)
    for name, spec in pending:
        thresholds[name] = cache.get_or_compute(spec, full, years, name)
    if setup.derived_seasons:
        thresholds.season_climatology = _harmonic_cached(cache_dir, full, years)
    return thresholds


_loaded_thresholds = {}


def load_thresholds(**kwargs):
    """The threshold the onset detection takes as thresh_slice: the old scalar or 2-D threshold for the flat keys, else the definition's named thresholds."""
    if is_legacy(kwargs):
        return load_thresh_file(**kwargs)
    setup = onset_setup(kwargs)
    key = (id(setup), str(kwargs.get("model")), str(kwargs.get("max_forecast_day")), str(kwargs.get("verification_window_list")), str(kwargs.get("dir_out")), str(kwargs.get("obs_dir")), str(kwargs.get("obs_var")),
           str(kwargs.get("region")))
    if key not in _loaded_thresholds:
        _loaded_thresholds[key] = load_onset_thresholds(setup, **kwargs)
    return _loaded_thresholds[key]


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

    ds = region_select(ds, region=region, dataset_name=str(kwargs.get("obs") or "obs"), **kwargs)

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

    ds = region_select(ds, dataset_name=str(kwargs.get("model") or "model"), **kwargs)

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

    ds = region_select(ds, dataset_name=str(kwargs.get("model") or "model"), **kwargs)

    p_model = ds[model_var]  # in mm

    if unit_cvt:
        p_model *= unit_cvt

    ds.close()

    return p_model


def load_history(year, current, **kwargs):
    """The observed rain to stitch before each forecast of `year`, or None when forecast_history is "none".

    `current` is the year's observations (already loaded). The previous year's file is
    added when a season or the rule's look-back reaches before 1 January; a missing
    previous file is skipped, which makes the earliest searches INSUFFICIENT_DATA.
    """
    if is_legacy(kwargs):
        return None
    setup = onset_setup(kwargs)
    if setup.forecast_history != "observed":
        return None
    if not history_needs_previous_year(setup):
        return current
    try:
        previous = load_imd_rainfall(year - 1, **kwargs)
    except FileNotFoundError:
        return current
    return xr.concat([previous, current], dim="time")
