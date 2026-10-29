"""Loading of observed rainfall, thresholds and (deterministic/ensemble) model forecasts from NetCDF files or zarr stores.

Loaders read only what the run needs: the year's inits, the leads the onset rule reads, the members asked for and
the region's box are selected on the lazy array before any data is decoded; the cell mask and the unit
conversion are then applied in place on the one array that was read.
"""
import hashlib
import json
import os
import threading
import warnings
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from momp.utils.workers import ordered_prefetch, worker_count
from momp.lib.onset_config import Thresholds, forecast_reach, history_needs_previous_year, is_legacy, onset_setup, setup_key
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


def is_zarr(path):
    """True if path is a zarr store: a name ending in .zarr, or a directory holding .zarray/.zgroup/zarr.json."""
    text = str(path).rstrip("/")
    if text.endswith(".zarr"):
        return True
    folder = Path(text)
    return folder.is_dir() and any((folder / name).exists() for name in ("zarr.json", ".zgroup", ".zarray"))


_stores = {}
_stores_lock = threading.Lock()


def open_store(path):
    """The lazy Dataset of a zarr store (chunks=None: nothing is decoded until a selection is loaded), kept open per path."""
    key = str(path)
    with _stores_lock:
        if key not in _stores:
            _stores[key] = xr.open_zarr(key, chunks=None)
        return _stores[key]


_obs_years = {}
_obs_lock = threading.Lock()


def clear_obs_cache():
    """Forget the observation years held for the current run (called when a run starts and ends)."""
    with _obs_lock:
        _obs_years.clear()


def _in_memory(da):
    """The DataArray with its data in memory (a no-op for one already read)."""
    return da if da.variable._in_memory else da.load()


def _scale_in_place(da, factor):
    """Multiply the loaded array by factor in its own memory (float32 stays float32), if there is a factor."""
    if factor:
        data = da.values
        np.multiply(data, factor, out=data)
    return da


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


def _obs_source_label(kwargs):
    """The text that labels the reference observations as a data source (dataset, folder, pattern, variable, unit factor)."""
    return (f"{kwargs.get('obs')}:{kwargs.get('obs_dir')}:{kwargs.get('obs_file_pattern')}:"
            f"{kwargs.get('obs_var')}:cvt={kwargs.get('obs_unit_cvt')}")


def _run_years(kwargs):
    """The years a run scores (and the year before each, for the observed history): read once, held for the run.

    Climatology and threshold years outside these are read, used and dropped (the observed-onset cache already
    shares what they yield), so the held observations stay the size of the evaluation period.
    """
    years = {int(y) for y in (kwargs.get("years") or ()) if not isinstance(kwargs.get("years"), str)}
    return years | {y - 1 for y in years}


class _NotContiguous(Exception):
    """The reference years do not follow each other day by day, so they cannot be merged year by year."""


def _stream_reference(setup, years, kwargs, source):
    """Ingest the reference years one at a time into one preallocated DailyRain (peak memory: the result plus one year).

    Gives what ingest of the concatenated years gives, for standard-calendar years that follow each other
    day by day on one grid; raises _NotContiguous otherwise. Years the run also evaluates stay in the
    observation cache; the others are read, ingested and dropped.
    """
    from momp.onset.model import DailyRain

    first = date(years[0], 1, 1)
    total = (date(years[1], 12, 31) - first).days + 1
    out, offset = None, 0
    for year, (rain, error) in zip(range(years[0], years[1] + 1), obs_years(range(years[0], years[1] + 1), kwargs)):
        if error is not None:
            raise error
        one = ingest(rain, day_boundary=setup.obs_day_boundary, units="mm", source=source)
        del rain
        if one.calendar != "standard" or one.tenths.shape[-1] != (date(year, 12, 31) - date(year, 1, 1)).days + 1 \
                or (date(year, 1, 1) - first).days != offset:
            raise _NotContiguous
        if out is None:
            shape = one.tenths.shape[:-1] + (total,)
            out = DailyRain(tenths=np.empty(shape, dtype=np.int32), missing=np.empty(shape, dtype=bool),
                            dims=one.dims, coords={**one.coords, "time": np.empty(total, dtype="datetime64[ns]")},
                            year=np.empty(total, dtype=np.int64), month=np.empty(total, dtype=np.int64),
                            day=np.empty(total, dtype=np.int64), calendar=one.calendar,
                            day_boundary=one.day_boundary, source=source)
        elif one.dims != out.dims or any(not np.array_equal(one.coords[d], out.coords[d]) for d in one.dims[:-1]):
            raise _NotContiguous
        n = one.tenths.shape[-1]
        out.tenths[..., offset:offset + n] = one.tenths
        out.missing[..., offset:offset + n] = one.missing
        for name in ("year", "month", "day"):
            getattr(out, name)[offset:offset + n] = getattr(one, name)
        out.coords["time"][offset:offset + n] = one.coords["time"]
        offset += n
    return out


def _reference_year(year, kwargs, keep=None):
    """One reference year's rainfall: held for the run if the run evaluates that year, else read and not kept."""
    keep = _run_years(kwargs) if keep is None else keep
    return load_imd_rainfall(year, **kwargs) if year in keep else _read_obs_uncached(year, kwargs)


def _reference_rain(setup, years, kwargs):
    """The observations of years[0]..years[1] as ingested DailyRain, labelled with their source.

    Years are read and ingested one at a time into one array rather than concatenated as rainfall first.
    """
    source = _obs_source_label(kwargs)
    if years[0] != years[1]:
        try:
            return _stream_reference(setup, years, kwargs, source)
        except _NotContiguous:
            pass
    return ingest(
        xr.concat([_reference_year(year, kwargs) for year in range(years[0], years[1] + 1)], dim="time"),
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
    key = (setup_key(kwargs), str(kwargs.get("model")), str(kwargs.get("max_forecast_day")), str(kwargs.get("verification_window_list")), str(kwargs.get("dir_out")), str(kwargs.get("obs_dir")), str(kwargs.get("obs_var")),
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


def _obs_source(year, obs_dir, obs_file_pattern):
    """Where the year's observations are: (zarr store path, None) or (None, NetCDF file path); FileNotFoundError if absent."""
    if is_zarr(obs_dir):
        return str(obs_dir), None
    file_patterns = [p.format(year) for p in obs_file_pattern]

    for pattern in file_patterns:
        test_path = f"{obs_dir}/{pattern}"
        if os.path.exists(test_path):
            return None, test_path

    available_files = [f for f in os.listdir(obs_dir) if f.endswith('.nc')]
    raise FileNotFoundError(
        f"No observation file found for year {year} in {obs_dir}. "
        f"Tried patterns: {file_patterns}. "
        f"Available files: {available_files}"
    )


_OBS_ARGS = ("obs_dir", "obs_file_pattern", "obs_var", "obs_unit_cvt", "region")
"""The settings that locate and read the observations (the rest of a config is passed on as kwargs)."""
_GRID_ARGS = ("grid_point", "lat_select", "lon_select")
"""Settings applied after a year is read (a single cell is picked out of it)."""
_GROUP_BYTES = 1_500_000_000
"""The most observation data read in one block when years are read together (a block is read, sliced per year, freed)."""


def _read_obs_range(first, last, store, obs_var, obs_unit_cvt, region, kwargs):
    """The region's box of a zarr store's observations from Jan 1 of first to Dec 31 of last, read in one pass.

    Equal, year by year, to _read_obs_year (the same cells masked, the same unit factor); an archive chunked over
    the whole time axis is decoded once for all the years instead of once per year.
    """
    ds = dim_fmt(open_store(store))
    ds = ds[[obs_var]].sel(time=slice(f"{first}-01-01", f"{last}-12-31"))
    da = region_select(ds[[obs_var]], read_by="time", **{**kwargs, "region": region,
                                                         "dataset_name": str(kwargs.get("obs") or "obs")})[obs_var]
    return _scale_in_place(_in_memory(da), obs_unit_cvt)


def _long_time_chunks(store, obs_var) -> bool:
    """Whether the store's time chunks are longer than a year, so each year read decodes data of other years too."""
    try:
        da = dim_fmt(open_store(store))[obs_var]
        chunks = da.encoding.get("chunks")
        return bool(chunks) and chunks[da.dims.index("time")] > 366
    except Exception:  # noqa: BLE001 - an unreadable layout is read year by year, as before
        return False


def _year_slice(block, year, store):
    """One year of a block read by _read_obs_range (a view of it); FileNotFoundError when the year is not in it."""
    sub = block.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if sub.sizes["time"] == 0:
        raise FileNotFoundError(f"No observations for year {year} in {store}")
    return sub


def _estimated_year_bytes(store, obs_var) -> int:
    """An upper bound of one year of the store's variable in memory (its whole grid, 366 days, as stored)."""
    da = dim_fmt(open_store(store))[obs_var]
    cells = int(np.prod([n for dim, n in da.sizes.items() if dim != "time"], dtype=np.int64))
    return max(1, cells * da.dtype.itemsize * 366)


def obs_years(years, kwargs, hold=True):
    """Yield (rain, None), or (None, error), for each of the years' observations in order, as load_imd_rainfall gives them.

    The years the run holds (the evaluated years and the year before each) are taken from, and kept in, the run's
    observation cache. A zarr store chunked over many years is read in blocks of about 1.5 GB, each decoded once
    and sliced per year; any other source is read a year at a time, the next few ahead in threads (zarr only;
    NetCDF files are read one at a time, as HDF5 is not thread-safe). Memory stays at one block, or
    a few years, however many years are asked for. hold=False reads without keeping even the held years.
    """
    years = list(years)
    if not years:
        return
    args = {k: kwargs[k] for k in _OBS_ARGS}
    skip = _OBS_ARGS + _GRID_ARGS
    rest = {k: v for k, v in kwargs.items() if k not in skip}
    keep = _run_years(kwargs) if hold else set()

    def locate(year):
        """(observation-cache key, zarr store or None, file path or None) of a year."""
        store, path = _obs_source(year, args["obs_dir"], args["obs_file_pattern"])
        return _obs_key(year, store, path, args["obs_var"], args["obs_unit_cvt"], args["region"], rest), store, path

    def held(year):
        """The array the run holds for the year, or None."""
        with _obs_lock:
            return _obs_years.get(locate(year)[0])

    def finish(year, rainfall, view=False):
        """Hold the year if the run does (a copy when it is a view of a block, so the block can be freed), then apply grid_point."""
        if year in keep:
            with _obs_lock:
                rainfall = _obs_years.setdefault(locate(year)[0], rainfall.copy(deep=True) if view else rainfall)
        if kwargs.get("grid_point"):
            rainfall = rainfall.sel(lat=[kwargs["lat_select"]], lon=[kwargs["lon_select"]])
        return rainfall

    try:
        store = locate(years[0])[1]
    except FileNotFoundError:
        store = None
    if store is not None and len(years) > 1 and _long_time_chunks(store, args["obs_var"]):
        per_block = max(1, _GROUP_BYTES // _estimated_year_bytes(store, args["obs_var"]))
        for start in range(0, len(years), per_block):
            group = years[start:start + per_block]
            need = [year for year in group if held(year) is None]
            block = _read_obs_range(min(need), max(need), store, args["obs_var"], args["obs_unit_cvt"],
                                    args["region"], rest) if need else None
            for year in group:
                try:
                    rainfall = held(year)
                    if rainfall is None:
                        yield finish(year, _year_slice(block, year, store), view=True), None
                    else:
                        yield finish(year, rainfall), None
                except FileNotFoundError as exc:
                    yield None, exc
            del block
        return

    # NetCDF/HDF5 readers are not thread-safe: only a zarr store is read by several threads at once
    readers = worker_count(kwargs, len(years)) if store is not None else 1

    def single(year):
        """One year read on its own (or the held one), or the error that stopped it."""
        try:
            rainfall = held(year)
            if rainfall is None:
                _, store, path = locate(year)
                rainfall = _read_obs_year(year, store, path, args["obs_var"], args["obs_unit_cvt"], args["region"],
                                          rest)
            return rainfall, None
        except Exception as exc:  # noqa: BLE001 - handed to the caller, which decides what a missing year means
            return None, exc

    for year, (rainfall, error) in zip(years, ordered_prefetch(single, years, readers)):
        yield (finish(year, rainfall), None) if error is None else (None, error)


def _read_obs_year(year, store, path, obs_var, obs_unit_cvt, region, kwargs):
    """Read one year of one variable: select the year and the region's box on the lazy array, read once, mask in place."""
    if store is not None:
        ds = open_store(store)
        ds = dim_fmt(ds)
        ds = ds[[obs_var]].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if ds.sizes["time"] == 0:
            raise FileNotFoundError(f"No observations for year {year} in {store}")
    else:
        ds = dim_fmt(xr.open_dataset(path))
    da = region_select(ds[[obs_var]], read_by="time", **{**kwargs, "region": region,
                                                         "dataset_name": str(kwargs.get("obs") or "obs")})[obs_var]
    da = _in_memory(da)
    if path is not None:
        ds.close()
    return _scale_in_place(da, obs_unit_cvt)


def _obs_key(year, store, path, obs_var, obs_unit_cvt, region, kwargs):
    """What identifies a loaded observation year: the source (with its size and time), the settings that shape it and the cells."""
    source = store or path
    try:
        stat = os.stat(source)
        ident = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        ident = None
    return (year, str(source), ident, obs_var, obs_unit_cvt, str(region), kwargs.get("land_only", True),
            kwargs.get("shp_only", True), str(kwargs.get("nc_mask")), str(kwargs.get("obs") or "obs"),
            json.dumps(kwargs.get("mask"), sort_keys=True, default=str))


def load_imd_rainfall(year, *, obs_dir, obs_file_pattern, obs_var, obs_unit_cvt, region,
                      grid_point=False, lat_select=None, lon_select=None, **kwargs):
    """Load observed daily rainfall for a given year (from the year's NetCDF file, or from a .zarr store of all years).

    The region's box and, for a store, the year are selected before any data is read, and the cell mask and the
    unit conversion are applied in place. The years the run scores (kwargs["years"], and the year before each)
    are read once per run: later calls (other windows, cases, the climatology) get the same array back, so treat
    it as read-only. Other years are read afresh when asked for.
    """
    store, path = _obs_source(year, obs_dir, obs_file_pattern)
    key = _obs_key(year, store, path, obs_var, obs_unit_cvt, region, kwargs)
    with _obs_lock:
        rainfall = _obs_years.get(key)
    if rainfall is None and store is not None and year in _run_years(kwargs) and _long_time_chunks(store, obs_var):
        # an archive chunked over many years: read all the years the run holds in one pass, not one decode per year
        everything = {**kwargs, "obs_dir": obs_dir, "obs_file_pattern": obs_file_pattern, "obs_var": obs_var,
                      "obs_unit_cvt": obs_unit_cvt, "region": region}
        for _ in obs_years(sorted(_run_years(kwargs)), everything):
            pass
        with _obs_lock:
            rainfall = _obs_years.get(key)
    if rainfall is None:
        rainfall = _read_obs_year(year, store, path, obs_var, obs_unit_cvt, region, kwargs)
        if year in _run_years(kwargs):
            with _obs_lock:
                rainfall = _obs_years.setdefault(key, rainfall)

    if grid_point:
        rainfall = rainfall.sel(lat=[lat_select], lon=[lon_select])

    return rainfall


def _read_obs_uncached(year, kwargs):
    """Read one observation year for a single use: the held copy if there is one, else a read that is not kept."""
    args = {k: kwargs[k] for k in ("obs_dir", "obs_file_pattern", "obs_var", "obs_unit_cvt", "region")}
    rest = {k: v for k, v in kwargs.items() if k not in args}
    store, path = _obs_source(year, args["obs_dir"], args["obs_file_pattern"])
    with _obs_lock:
        held = _obs_years.get(_obs_key(year, store, path, args["obs_var"], args["obs_unit_cvt"], args["region"], rest))
    if held is None:
        held = _read_obs_year(year, store, path, args["obs_var"], args["obs_unit_cvt"], args["region"], rest)
    if rest.get("grid_point"):
        held = held.sel(lat=[rest["lat_select"]], lon=[rest["lon_select"]])
    return held


def _open_forecast(year, model_dir, file_pattern, ensemble, model_dims, filtered_dates_yr):
    """Open the year's forecasts lazily and select its inits; returns (Dataset, close function).

    A model_dir that is a zarr store holds all years: the inits of the year are picked by time. Otherwise the
    year's NetCDF file file_pattern.format(year) is opened. Raises FileNotFoundError when the year is absent.
    """
    fix = dim_fmt_model_ensemble if ensemble else dim_fmt_model
    if is_zarr(model_dir):
        ds = fix(open_store(model_dir), model_dims)
        close = lambda: None  # noqa: E731 - the store handle is shared
    else:
        file_path = os.path.join(model_dir, file_pattern.format(year))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ds = fix(xr.open_dataset(file_path), model_dims)
        close = ds.close
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        if is_zarr(model_dir) and not (available_init_times.year == year).any():
            raise FileNotFoundError(f"No forecasts of {year} in {model_dir}")
        raise ValueError(f"No matching initialization times found for year {year}")
    return ds.sel(init_time=matching_times), close


def _load_forecast(year, *, model_dir, model_var, unit_cvt, file_pattern, region, members, ensemble, kwargs):
    """Read one year of forecasts as a loaded DataArray in mm, reading only what the run needs.

    Inits, leads (up to what the onset rule reads), members and the region's box are selected on the lazy array;
    the selection is read once, the cell mask and the unit conversion (x * unit_cvt, in float32 as before) are
    applied in place on it. The file's step 0, if present, is dropped.
    """
    ds, close = _open_forecast(year, model_dir, file_pattern, ensemble, kwargs.get("model_dims"),
                               get_initialization_dates(year, **kwargs))
    try:
        if 'step' not in ds.dims:
            raise KeyError("'step' dimension not found in dataset")
        # Check if the first value of 'step' is 0, then slice to exclude it
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
        # Onset detection reads no further than this lead day, so loading the rest only costs memory.
        steps_needed = onset_steps_needed(**kwargs)
        if steps_needed is not None:
            ds = ds.sel(step=slice(None, steps_needed))
        if ensemble and members:
            ds = ds.sel(member=list(members))
        da = region_select(ds[[model_var]], read_by="init_time",
                           **{**kwargs, "region": region, "dataset_name": str(kwargs.get("model") or "model")})[model_var]
        da = _in_memory(da)
    finally:
        close()
    return _scale_in_place(da, unit_cvt)


def get_forecast_deterministic_twice_weekly(year, *, model_dir, model_var, date_filter_year, init_days,
                                            start_date, end_date, unit_cvt, file_pattern, region, **kwargs):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    Filters for Mondays and Thursdays in the specified year.
    The forecast file is expected to be named file_pattern.format(year) in the model_dir (or model_dir is a
    .zarr store of all years) with variable "tp" being daily accumulated rainfall with dimensions
    (init_time, lat, lon, step).

    Parameters:
    year: int, year to load data for

    Returns:
    p_model: DataArray, precipitation data in mm, read once with the region, inits and leads selected first
    """

    kwargs = restore_args(get_forecast_deterministic_twice_weekly, kwargs, locals())
    return _load_forecast(year, model_dir=model_dir, model_var=model_var, unit_cvt=unit_cvt,
                          file_pattern=file_pattern, region=region, members=None, ensemble=False, kwargs=kwargs)


def get_forecast_probabilistic_twice_weekly(year, *, model_dir, model_var, date_filter_year, init_days,
                                            start_date, end_date, unit_cvt, file_pattern,
                                            members, region, **kwargs):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    Reads like get_forecast_deterministic_twice_weekly, with the members asked for selected before the read.
    """

    kwargs = restore_args(get_forecast_probabilistic_twice_weekly, kwargs, locals())
    return _load_forecast(year, model_dir=model_dir, model_var=model_var, unit_cvt=unit_cvt,
                          file_pattern=file_pattern, region=region, members=members, ensemble=True, kwargs=kwargs)


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
