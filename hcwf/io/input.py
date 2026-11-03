"""Loading of observed rainfall, thresholds and (deterministic/ensemble) model forecasts from NetCDF files or zarr stores.

Loaders read only what the run needs: the year's inits, the leads the onset rule reads, the members asked for and
the region's box are selected on the lazy array before any data is decoded; the cell mask and the unit
conversion are then applied in place on the one array that was read.

The readers take the typed specs of a run (`hcwf.config.model`): `load_observations(year, obs, area, options, select)`,
`load_forecast(year, model, ev, area, options, onset, ensemble=...)` and `load_thresholds(onset, obs, area, options, ev,
output, model=...)`. What a single call chooses and a config does not hold (units to keep, members, issue dates, the
files' years, a grid point, the implicit masks) is a `Selection`. The old keyword names (`load_imd_rainfall(year,
**config)`, `get_forecast_*`, `load_thresholds(**config)`, `units_spec(config, source)`, ...) stay as `# R2-bridge`
wrappers that rebuild the typed inputs with `hcwf.config.flat_bridge` (step S11 deletes them).

A dataset is read from one file per year (`file_pattern` with {} for the year), a zarr store, or an explicit list of
files (`paths`, any number per year, optionally filtered by `years`): see `year_files`. Data on a string "unit"
dimension (admin units) is read without a region box; `unit_series` and `read_on_units` aggregate cells onto units
the way onset_blending's regrid and cell transform do, reading only the storage tiles that feed them.
"""
from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import xarray as xr
from hcwf.cache import NETCDF, NETCDF_GROUPS, Store, file_identity
from hcwf.config.model import Selection
from hcwf.stats import observed_cache
from hcwf.utils.workers import ordered_prefetch, worker_count
from hcwf.lib.onset_config import Thresholds, forecast_reach, history_needs_previous_year, onset_setup
from hcwf.onset import definition as definition_mod
from hcwf.onset.climatology import ThresholdCache, compute_thresholds_from_forecasts, harmonic_seasons
from hcwf.lib.dataset_schema import DatasetError, file_year, is_zarr, resolve_paths
from hcwf.onset.thresholds import ThresholdScale, read_threshold_grid, read_threshold_table, scale_values
from hcwf.onset.ingest import IngestInfo, ingest, ingest_forecast_info, observation_shift_days, unit_scale
from hcwf.spatial import axes, ids as unit_ids
from hcwf.spatial.units import UNITS_KEYS, EvaluationUnits, UnitsSpec
from hcwf.utils.standard import (dim_fmt, dim_fmt_model, dim_fmt_model_ensemble, find_dim, rename_model_dims)
from hcwf.utils.region import implicit_masks, load_blocks, region_box, region_select
import logging

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.config.model import (AreaSpec, EvaluationSpec, ModelSpec, ObsSpec, OnsetOptions,
                                   ParallelSpec)
    from hcwf.lib.onset_config import OnsetSetup

log = logging.getLogger(__name__)

_NO_SELECTION = Selection()


def _bridge():
    """R2-bridge: the adapter that rebuilds typed inputs from the flat names (imported late: it imports the config)."""
    from hcwf.config import flat_bridge

    return flat_bridge


def steps_needed(max_forecast_day, onset: OnsetSetup | None):
    """The last lead day onset detection reads, or None if it can't be told.

    The largest day any requirement or veto of the compiled plan reads for the last candidate day
    max_forecast_day, so loading never truncates what the rule reads.
    """
    if not isinstance(max_forecast_day, int):
        return None
    return max_forecast_day + forecast_reach(onset) - 1


def onset_steps_needed(*, max_forecast_day=None, **kwargs):
    """`steps_needed` of a flat config."""  # R2-bridge
    return steps_needed(max_forecast_day, onset_setup(kwargs) if isinstance(max_forecast_day, int) else None)


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
    """Forget the observation years held for the current run, and the shared reference rainfall (called when a run starts and ends)."""
    with _obs_lock:
        _obs_years.clear()
    release_reference()


@dataclass(frozen=True, eq=False)
class _ObsRead:
    """What reading the observations depends on, besides the year: the typed inputs of one call (see `load_observations`)."""

    obs: ObsSpec
    area: AreaSpec
    options: OnsetOptions
    select: Selection
    onset: OnsetSetup | None
    run_years: tuple
    dir_fig: object
    parallel: ParallelSpec | None = None


@dataclass(frozen=True, eq=False)
class _ForecastRead:
    """What reading a model's forecasts depends on, besides the year: the typed inputs of one call (see `load_forecast`)."""

    model: ModelSpec
    ev: EvaluationSpec
    area: AreaSpec
    options: OnsetOptions
    select: Selection
    onset: OnsetSetup | None
    dir_fig: object
    ensemble: bool


def _mask_text(area) -> str:
    """The `mask:` section as text (a cache key)."""
    return "null" if not area.mask else area.mask.key()


_reference = {"key": None, "rain": None, "first": None, "last": None}
"""The reference (threshold and climatology) years as one ingested DailyRain, kept from the thresholds to the climatology."""


def _reference_key(obs, area, select, boundary):
    """What the shared reference rain depends on: the observations as read and cut, and the day boundary they are ingested with."""
    land, shp = implicit_masks(select)
    return (str(obs.dir), str(None if obs.file_pattern is None else list(obs.file_pattern)), str(obs.var),
            str(obs.unit_cvt), str(area.region), land, shp, str(area.nc_mask), str(obs.name or "obs"),
            _mask_text(area), bool(select.grid_point), str(select.lat), str(select.lon), str(boundary))


def release_reference():
    """Drop the shared reference rainfall (its memory is the size of the reference years' tenths and flags)."""
    _reference.update(key=None, rain=None, first=None, last=None)


def reference_held() -> bool:
    """Whether the thresholds left their reference rainfall for the climatology to use (see reference_rain_year)."""
    return _reference["rain"] is not None


def reference_rain_year(year, obs, area, select, boundary):
    """One year of the shared reference rain as a DailyRain (views of it, nothing copied), or None when it is not held.

    The thresholds read and ingest the reference years once (see _stream_reference); the climatological onsets of
    those years use that same ingested rain instead of reading and converting every year again. The year's rain is
    exactly what ingest gives for the year read on its own.
    """
    held = _reference
    if held["rain"] is None or not held["first"] <= year <= held["last"] \
            or held["key"] != _reference_key(obs, area, select, boundary):
        return None
    from hcwf.onset.model import DailyRain

    rain = held["rain"]
    start = (date(year, 1, 1) - date(held["first"], 1, 1)).days
    days = slice(start, start + (date(year, 12, 31) - date(year, 1, 1)).days + 1)
    return DailyRain(tenths=rain.tenths[..., days], missing=rain.missing[..., days], dims=rain.dims,
                     coords={**rain.coords, "time": rain.coords["time"][days]}, year=rain.year[days],
                     month=rain.month[days], day=rain.day[days], calendar=rain.calendar,
                     day_boundary=rain.day_boundary, source="observations")


def reference_year(year, kwargs, boundary):
    """`reference_rain_year` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return reference_rain_year(year, got.obs, got.area, got.select, boundary)


def _in_memory(da):
    """The DataArray with its data in memory (a no-op for one already read)."""
    return da if da.variable._in_memory else da.load()


def _scale_in_place(da, factor):
    """Multiply the loaded array by factor in its own memory (float32 stays float32), if there is a factor."""
    if factor:
        data = da.values
        np.multiply(data, factor, out=data)
    return da


class _LazyRain:
    """A DailyRain whose data is read only when something other than its grid, source and resolution is asked for.

    The threshold cache keys on those, so a cache hit never loads the reference years.
    """

    def __init__(self, probe, load):
        """Take dims, coords and source from the probe (one year); `load` returns the full DailyRain."""
        self.dims, self.coords, self.source, self.resolution = probe.dims, probe.coords, probe.source, probe.resolution
        self._load, self._full = load, None

    def __getattr__(self, name):
        """Load the full reference rainfall on first use and read the attribute from it."""
        if name.startswith("_"):
            raise AttributeError(name)
        if self._full is None:
            self._full = self._load()
        return getattr(self._full, name)


_reported_ingest = set()


def ingest_options(kwargs, forecast=False) -> dict:
    """`OnsetOptions.ingest` of a flat config (the keyword options of `ingest`, or of `ingest_forecast_info`)."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return got.options.ingest(forecast, got.obs.units, check_units=got.select.check_units)


def ingest_forecast(cube, onset, model, options, select=_NO_SELECTION, source=None):
    """Normalise a model's forecast cube (already in mm, dims init_time, [member], lat, lon, step) to ForecastRain.

    Uses the model's own `accumulation` ("daily", "since_init" or "auto") and `daily_boundary` (the config's per-model
    values; "daily" when absent), so accumulated fields are differenced and sub-daily steps summed to days
    exactly as hcwf.onset.ingest.ingest_forecast does. source labels the data (default: the model's name); how the
    days were made is logged once per model.
    """
    source = source or str(model.name or "forecast")
    accumulation = model.accumulation or "daily"
    boundary = model.daily_boundary or None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IngestInfo)
        forecast, info = ingest_forecast_info(cube, day_boundary=onset.forecast_boundary, lead_label=onset.lead_label,
                                              units="mm", source=source, accumulation=accumulation,
                                              daily_boundary=boundary,
                                              **options.ingest(True, None, check_units=select.check_units))
    forecast.ingest_info = info
    key = (str(model.name), info.accumulation, info.resolution, info.daily_boundary)
    if key not in _reported_ingest:
        _reported_ingest.add(key)
        text = f"{model.name or source}: accumulation {info.accumulation} [{info.accumulation_evidence}]; steps {info.resolution}"
        if info.daily_boundary:
            text += f"; days cut at {info.daily_boundary}"
        log.info(text + "".join(f"; {note}" for note in info.notes))
    return forecast


def ingest_model_forecast(cube, setup, kwargs, source=None):
    """`ingest_forecast` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return ingest_forecast(cube, setup, got.model, got.options, got.select, source)


def _obs_source_label(read, onset):
    """The text that labels the reference observations as a data source (dataset, folder, pattern, variable, unit
    factor, and for units what turns the files into series)."""
    obs = read.obs
    label = (f"{obs.name}:{obs.dir}:{None if obs.file_pattern is None else list(obs.file_pattern)}:"
             f"{obs.var}:cvt={obs.unit_cvt}")
    if obs.units is not None:
        label += (f":units={obs.units!r}:options={sorted(read.options.ingest(False, obs.units).items())}:"
                  f"dims={obs.dims_map()}:labels={onset.obs_day_boundary}/{onset.lead_label}")
    return label


def _run_years(run_years):
    """The years a run scores (and the year before each, for the observed history): read once, held for the run.

    Climatology and threshold years outside these are read, used and dropped (the observed-onset cache already
    shares what they yield), so the held observations stay the size of the evaluation period.
    """
    years = {int(y) for y in run_years}
    return years | {y - 1 for y in years}


class _NotContiguous(Exception):
    """The reference years do not follow each other day by day, so they cannot be merged year by year."""


def _stream_reference(onset, years, read, source):
    """Ingest the reference years one at a time into one preallocated DailyRain (peak memory: the result plus one year).

    Gives what ingest of the concatenated years gives, for standard-calendar years that follow each other
    day by day on one grid; raises _NotContiguous otherwise. Years the run also evaluates stay in the
    observation cache; the others are read, ingested and dropped.
    """
    from hcwf.onset.model import DailyRain

    first = date(years[0], 1, 1)
    total = (date(years[1], 12, 31) - first).days + 1
    out, offset = None, 0
    options = read.options.ingest(False, read.obs.units)
    for year, (rain, error) in zip(range(years[0], years[1] + 1), _obs_years_of(read, range(years[0], years[1] + 1))):
        if error is not None:
            raise error
        one = ingest(rain, day_boundary=onset.obs_day_boundary, units="mm", source=source, **options)
        del rain
        if one.calendar != "standard" or one.tenths.shape[-1] != (date(year, 12, 31) - date(year, 1, 1)).days + 1 \
                or (date(year, 1, 1) - first).days != offset:
            raise _NotContiguous
        if out is None:
            shape = one.tenths.shape[:-1] + (total,)
            out = DailyRain(tenths=np.empty(shape, dtype=one.tenths.dtype), missing=np.empty(shape, dtype=bool),
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


def _reference_year(year, read, keep=None):
    """One reference year's rainfall: held for the run if the run evaluates that year, else read and not kept."""
    keep = _run_years(read.run_years) if keep is None else keep
    rain = _load_observation(year, read) if year in keep else _read_obs_uncached(year, read)
    return rain.drop_vars("present") if "present" in rain.coords else rain  # which units have a series, not rain


def _reference_rain(onset, years, read):
    """The observations of years[0]..years[1] as ingested DailyRain, labelled with their source.

    Years are read and ingested one at a time into one array rather than concatenated as rainfall first.
    """
    source = _obs_source_label(read, onset)
    if years[0] != years[1]:
        try:
            rain = _stream_reference(onset, years, read, source)
        except _NotContiguous:
            pass
        else:
            _reference.update(key=_reference_key(read.obs, read.area, read.select, onset.obs_day_boundary), rain=rain,
                              first=years[0], last=years[1])
            return rain
    return ingest(
        xr.concat([_reference_year(year, read) for year in range(years[0], years[1] + 1)], dim="time"),
        day_boundary=onset.obs_day_boundary, units="mm", source=source, **read.options.ingest(False, read.obs.units))


def _harmonic_cached(cache_dir, rain, years, files=()):
    """The harmonic season analysis of the reference years, read from or written to the cache directory."""
    parts = {"source": rain.source, "years": [int(years[0]), int(years[1])], "files": list(files),
             "grid": {dim: rain.coords[dim] for dim in rain.dims[:-1]}}
    return Store("harmonic_seasons", cache_dir, codec=NETCDF, label="Harmonic seasons").get_or_compute(
        parts, lambda: harmonic_seasons(rain, years))


def _model_forecasts(onset, years, read: _ForecastRead):
    """Yield the model's forecasts of each reference year as ForecastRain, with HCWF's loader and unit conversion."""
    found = False
    for year in range(years[0], years[1] + 1):
        try:
            cube = load_forecast(year, read.model, read.ev, read.area, read.options, onset, ensemble=read.ensemble,
                                 select=read.select, dir_fig=read.dir_fig)
        except FileNotFoundError as exc:
            log.warning(f"Model climatology: skipping year {year} ({exc})")
            continue
        found = True
        order = ("init_time",) + (("member",) if "member" in cube.dims else ()) + ("lat", "lon", "step")
        yield ingest_forecast(cube.transpose(*order), onset, read.model, read.options, read.select,
                              str(read.model.name))
    if not found:
        raise FileNotFoundError(f"no forecast file of {read.model.name} for the years {years[0]}-{years[1]}")


def _model_files(years, read: _ForecastRead) -> list:
    """The identity of what holds the model's forecasts of each year: its files, or the zarr store, or the missing year."""
    found = []
    model, select = read.model, read.select
    for year in range(int(years[0]), int(years[1]) + 1):
        try:
            sources = _forecast_sources(year, model.dir, model.file_pattern, select.model_paths,
                                        select.model_files_years)
        except (FileNotFoundError, TypeError, AttributeError):
            found.append([f"missing {year}"])
            continue
        found.append([file_identity(f if f is not None else model.dir) for f in sources])
    return found


def _model_read_key(read: _ForecastRead) -> dict:
    """The settings that change the model's forecasts as read (grid, units, steps, days), beside the files."""
    model, ev, area, options, select = read.model, read.ev, read.area, read.options, read.select
    return {"region": area.region, "mask": None if not area.mask else area.mask.key(), "land_only": select.land_only,
            "shp_only": select.shp_only, "nc_mask": area.nc_mask, "model_var": model.var,
            "unit_cvt": model.unit_cvt, "accumulation": model.accumulation, "daily_boundary": model.daily_boundary,
            "model_dims": model.dims_map(), "file_pattern": model.file_pattern, "model_paths": select.model_paths,
            "model_files_years": select.model_files_years, "model_dir": model.dir,
            "date_filter_year": ev.date_filter_year, "start_date": ev.start_date, "end_date": ev.end_date,
            "probabilistic": ev.probabilistic, "max_forecast_day": ev.max_forecast_day,
            "rain_resolution": options.rain_resolution, "missing": options.missing, "negatives": options.negatives,
            "check_units": select.check_units}


def _units_key_part(obs_units, model_units):
    """The evaluation units of the observations and the model, as a cache key (the grid cells, none given, as before)."""
    if obs_units is None and model_units is None:
        return {key: None for key in UNITS_KEYS}
    return {"observations": repr(obs_units), "model": repr(model_units)}


def _model_thresholds(onset, pending, years, cache_dir, read: _ForecastRead, obs_units):
    """The model's own climatology of the pending thresholds, read from or written to the cache directory.

    Each verification window of the run (the windows of `ev`, already
    shortened to the usable leads) gets its own threshold, pooled over the leads
    in that window (spec section 13); without windows one threshold pools every lead.
    """
    model, ev = read.model, read.ev
    window = (1, int(ev.max_forecast_day))
    windows = [(int(w.start), int(w.end)) for w in ev.windows] if ev.windows else None
    parts = {"model": str(model.name), "model_files": _model_files(years, read),
             "years": [int(years[0]), int(years[1])], "window": list(window), "windows": windows,
             "definition": definition_mod.definition_hash(onset.definition), "pending": [repr(p) for p in pending],
             "lead_label": onset.lead_label, "init_days": [int(d) for d in ev.init_days],
             "members": None if not ev.members else [int(m) for m in ev.members],
             "read": _model_read_key(read),
             "units": _units_key_part(obs_units, model.units),
             "units_files": units_files(model.units),
             "boundary": onset.forecast_boundary}
    names = {name for name, _ in pending}

    def compute():
        reduced = replace(onset.definition,
                          thresholds=tuple((n, sp) for n, sp in onset.definition.thresholds if n in names))
        return compute_thresholds_from_forecasts(reduced, _model_forecasts(onset, years, read), years, window, windows)

    store = Store("model_thresholds", cache_dir, codec=NETCDF_GROUPS, label="Model thresholds")
    return store.get_or_compute(parts, compute)


def load_onset_thresholds(onset, obs, area, options, ev, output, *, model=None, select=_NO_SELECTION,
                          parallel=None, cache_dir=None):
    """The named thresholds of the definition, as {name: DataArray(lat, lon)} in mm.

    Files given in `thresholds` are used as they are; the others are computed from
    the observations of threshold_years and cached under the run's cache_dir (shared by runs), or
    the output's data folder/.onset_cache when none is set.
    The returned Thresholds also carries season_climatology for derived seasons.
    """
    thresholds = Thresholds()
    if not onset.uses_thresholds:
        return thresholds
    scale = ThresholdScale(**options.threshold_scale) if options.threshold_scale else None
    for name, path, var in onset.threshold_files:
        if var is None:   # a table by unit id
            table_options = {k: v for k, v in (options.threshold_table or {}).items() if v is not None}
            thresholds[name] = read_threshold_table(path, scale=scale, **table_options)
        else:
            thresholds[name] = read_threshold_grid(
                path, var, select=lambda da: region_select(da, area=replace(area, mask=None), select=select),
                scale=scale)
    for name, mm in onset.threshold_constants:
        thresholds[name] = mm
    pending = [(name, spec) for name, spec in onset.definition.thresholds if name not in thresholds]
    if not pending and not onset.derived_seasons:
        return thresholds
    years = onset.threshold_years
    cache_dir = Path(cache_dir) if cache_dir else Path(output.data_dir) / ".onset_cache"
    if pending and onset.threshold_source == "model":
        read = _ForecastRead(model, ev, area, options, select, onset, output.figure_dir, bool(ev.probabilistic))
        thresholds.update(_model_thresholds(onset, pending, years, cache_dir, read, obs.units))
        pending = []
        if not onset.derived_seasons:
            return thresholds
    read = _ObsRead(obs, area, options, select, onset, tuple(ev.years), output.figure_dir, parallel)
    probe = _reference_rain(onset, (years[0], years[0]), read)
    full = _LazyRain(probe, lambda: _reference_rain(onset, years, read))
    files = observed_cache.observation_files(obs, select.obs_files_years, range(years[0], years[1] + 1))
    threshold_options = {key: value for key, value in (("max_missing_fraction", options.threshold_max_missing),
                                                       ("partial_years", options.threshold_partial_years))
                         if value is not None}
    cache = ThresholdCache(cache_dir, files + units_files(obs.units))
    for name, spec in pending:
        thresholds[name] = scale_values(cache.get_or_compute(spec, full, years, name, **threshold_options), scale)
    if onset.derived_seasons:
        thresholds.season_climatology = _harmonic_cached(cache_dir, full, years, files)
    return thresholds


_loaded_thresholds = {}


def _setup_key(onset) -> str:
    """Text identifying the onset setup the thresholds are computed with."""
    return repr((definition_mod.definition_hash(onset.definition), onset.season, onset.threshold_years,
                 onset.threshold_files, onset.threshold_constants, onset.obs_day_boundary, onset.lead_label,
                 onset.threshold_source, onset.forecast_history, onset.dry_check))


def _thresholds_key(onset, obs, options, select, model) -> str:
    """What else the loaded thresholds depend on: how they are computed and which observations on which units."""
    return repr([("threshold_scale", options.threshold_scale), ("threshold_table", options.threshold_table),
                 ("threshold_max_missing", options.threshold_max_missing),
                 ("threshold_partial_years", options.threshold_partial_years), ("obs_paths", obs.paths),
                 ("obs_files_years", select.obs_files_years), ("obs_dims", obs.dims_map()),
                 ("obs_day_boundary", obs.day_boundary), ("model_lead_label", onset.lead_label),
                 ("ingest", options.ingest(False, obs.units)), ("obs_units", obs.units),
                 ("model_units", None if model is None else model.units)])


def load_thresholds(onset=None, obs=None, area=None, options=None, ev=None, output=None, *, model=None,
                    select=_NO_SELECTION, parallel=None, cache_dir=None, **kwargs):
    """The definition's named thresholds (what the onset detection takes as thresh_slice).

    `model` is the forecast model the thresholds come from when threshold_source is "model". Computed ones are cached
    under `cache_dir`, or the output's data folder; the loaded set is kept for the run (see `_loaded_thresholds`).
    R2-bridge: called with only the keys of a flat config (`load_thresholds(**config)`) it rebuilds these from them.
    """
    if onset is None:   # R2-bridge
        got = _bridge().reader_inputs(kwargs)
        return load_thresholds(got.onset(), got.obs, got.area, got.options, got.ev, got.output, model=got.model,
                               select=got.select, parallel=got.parallel, cache_dir=got.cache_dir)
    key = (_setup_key(onset), str(None if model is None else model.name), str(ev.max_forecast_day),
           str(None if not ev.windows else [w.span for w in ev.windows]), str(output.data_dir), str(obs.dir),
           str(obs.var), str(area.region), _thresholds_key(onset, obs, options, select, model))
    if key not in _loaded_thresholds:
        _loaded_thresholds[key] = load_onset_thresholds(onset, obs, area, options, ev, output, model=model,
                                                        select=select, parallel=parallel, cache_dir=cache_dir)
    return _loaded_thresholds[key]


def _warm_masks(lazy, var, dataset_name, area, select, dir_fig):
    """Build the cell masks of a lazily opened dataset's grid (what its first read would build), without reading data.

    `lazy` is the (dim-standardised) Dataset; a zero field on its region box is masked in place with the run's
    settings, which fills the same mask caches (and logs the same mask text) as the real read.
    """
    from hcwf.utils.region import mask_region_loaded, region_box

    box = region_box(lazy[[var]], region=area.region)
    lat, lon = np.asarray(box["lat"].values), np.asarray(box["lon"].values)
    field = xr.DataArray(np.zeros((len(lat), len(lon)), dtype="float32"), dims=("lat", "lon"),
                         coords={"lat": lat, "lon": lon})
    mask_region_loaded(xr.Dataset({var: field}), area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)


def warm_up(onset, obs, area, options, ev, output, *, model=None, select=_NO_SELECTION, parallel=None,
            cache_dir=None):
    """Read or build in this process what every year of a run shares: the thresholds and the cell masks of both grids.

    Done once before the years go to worker processes, which then get these (see hcwf.stats.parallel.snapshot)
    instead of each building them, and whose messages are logged once. The first of the evaluation years picks the
    NetCDF files whose grids are looked at. Thresholds are loaded as the years would load them; the masks are best effort
    (a grid that cannot be looked at is left to the workers, which raise the real error if there is one).
    """
    load_thresholds(onset, obs, area, options, ev, output, model=model, select=select, parallel=parallel,
                    cache_dir=cache_dir)
    years = ev.years
    if not years or obs.units is not None:   # units are read without a cell mask
        return
    year = int(list(years)[0])
    dir_fig = output.figure_dir
    try:
        store, path = _obs_source(year, obs.dir, obs.file_pattern, obs.paths, select.obs_files_years)
        path = path[0] if isinstance(path, tuple) else path
        lazy = dim_fmt(open_store(store) if store is not None else xr.open_dataset(path))
        try:
            _warm_masks(lazy, obs.var, str(obs.name or "obs"), area, select, dir_fig)
        finally:
            if path is not None:
                lazy.close()
    except Exception as exc:  # noqa: BLE001 - only a warm-up
        log.debug(f"obs mask pre-warm skipped: {exc}")
    try:
        fix = dim_fmt_model_ensemble if ev.probabilistic else dim_fmt_model
        if select.model_paths:
            path = year_files(select.model_paths, model.dir, select.model_files_years, year)[0]
            lazy = fix(xr.open_dataset(path), model.dims_map())
        elif is_zarr(model.dir):
            lazy = fix(open_store(model.dir), model.dims_map())
            path = None
        else:
            path = os.path.join(model.dir, model.file_pattern.format(year))
            lazy = fix(xr.open_dataset(path), model.dims_map())
        try:
            _warm_masks(lazy, model.var, str(model.name or "model"), area, select, dir_fig)
        finally:
            if path is not None:
                lazy.close()
    except Exception as exc:  # noqa: BLE001 - only a warm-up (a climatology source has no forecast files)
        log.debug(f"model mask pre-warm skipped: {exc}")


def prewarm(kwargs):
    """`warm_up` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return warm_up(got.onset(), got.obs, got.area, got.options, got.ev, got.output, model=got.model,
                   select=got.select, parallel=got.parallel, cache_dir=got.cache_dir)


def thresholds_of(rc, model, select=_NO_SELECTION):
    """`load_thresholds` of a run (`rc`: RunConfig) for the source `model` (ModelSpec)."""
    return load_thresholds(rc.onset, rc.observations, rc.area, rc.onset_options, rc.evaluation, rc.output, model=model,
                           select=select, parallel=rc.parallel, cache_dir=rc.cache_dir)


def warm_up_of(rc, model, select=_NO_SELECTION):
    """`warm_up` of a run for the source `model`."""
    return warm_up(rc.onset, rc.observations, rc.area, rc.onset_options, rc.evaluation, rc.output, model=model,
                   select=select, parallel=rc.parallel, cache_dir=rc.cache_dir)


def observations_of(year, rc, select=_NO_SELECTION):
    """`load_observations` of a run."""
    return load_observations(year, rc.observations, rc.area, rc.onset_options, select,
                             onset=rc.onset if rc.observations.units is not None else None,
                             run_years=rc.evaluation.years, dir_fig=rc.output.figure_dir)


def forecast_of(year, rc, model, *, ensemble, select=_NO_SELECTION):
    """`load_forecast` of a run for the source `model`."""
    steps = rc.onset if isinstance(rc.evaluation.max_forecast_day, int) else None
    return load_forecast(year, model, rc.evaluation, rc.area, rc.onset_options, steps, ensemble=ensemble,
                         select=select, dir_fig=rc.output.figure_dir)


def history_of(year, current, rc, select=_NO_SELECTION):
    """`observed_history` of a run."""
    return observed_history(year, current, rc.onset, rc.observations, rc.area, rc.onset_options, select,
                            run_years=rc.evaluation.years, dir_fig=rc.output.figure_dir)


def _init_dates(year, date_filter_year, init_days, start_date, end_date):
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


def initialization_dates(year, ev):
    """The initialization dates of `year`: the days between ev.start_date and ev.end_date (month/day only) whose weekday
    in ev.date_filter_year is in ev.init_days (0 = Monday)."""
    return _init_dates(year, ev.date_filter_year, ev.init_days, ev.start_date, ev.end_date)


def get_initialization_dates(year, *, date_filter_year, init_days, start_date, end_date, **kwargs):
    """`initialization_dates` from the four keys of a flat config."""  # R2-bridge
    return _init_dates(year, date_filter_year, init_days, start_date, end_date)


def year_files(paths, base, years_filter, year) -> list[str]:
    """The files of `year` among those a dataset lists in `paths` (names or globs, relative ones under `base`).

    The year of a file is the first 19xx or 20xx in its name (the rule of onset_blending); `years_filter` keeps only
    the files of those years first. Any number of files may fall in a year (live files hold one init each); they
    come in the order resolve_paths gives, by year and name.
    """
    try:
        files = resolve_paths(paths, base, years_filter)
    except DatasetError as exc:
        raise FileNotFoundError(str(exc)) from None
    return [str(f) for f in files if file_year(f) == year]


def _obs_source(year, obs_dir, obs_file_pattern, paths=None, years_filter=None):
    """Where the year's observations are: (zarr store path, None) or (None, NetCDF file path); FileNotFoundError if absent.

    With `paths` (see year_files) the year's listed files are the source: one path, or a tuple of paths when
    several files hold the year.
    """
    if paths:
        found = year_files(paths, obs_dir or None, years_filter, year)
        if not found:
            raise FileNotFoundError(f"No observation file found for year {year} among the listed paths {list(paths)}")
        return None, found[0] if len(found) == 1 else tuple(found)
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


_GROUP_BYTES = 1_500_000_000
"""The most observation data read in one block when years are read together (a block is read, sliced per year, freed)."""


def _locate(year, read: _ObsRead):
    """(zarr store or None, file path or None) holding the year's observations; FileNotFoundError when absent."""
    return _obs_source(year, read.obs.dir, read.obs.file_pattern, read.obs.paths, read.select.obs_files_years)


def _read_obs_range(first, last, store, read: _ObsRead):
    """The region's box of a zarr store's observations from Jan 1 of first to Dec 31 of last, read in one pass.

    Equal, year by year, to _read_obs_year (the same cells masked, the same unit factor); an archive chunked over
    the whole time axis is decoded once for all the years instead of once per year.
    """
    obs = read.obs
    ds = dim_fmt(open_store(store))
    ds = ds[[obs.var]].sel(time=slice(f"{first}-01-01", f"{last}-12-31"))
    da = region_select(ds[[obs.var]], area=read.area, select=read.select, read_by="time",
                       dataset_name=str(obs.name or "obs"), dir_fig=read.dir_fig)[obs.var]
    return _scale_in_place(_in_memory(da), obs.unit_cvt)


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


def read_obs_years(years, obs, area, options, select=_NO_SELECTION, *, onset=None, run_years=(), dir_fig=None,
                   parallel=None, hold=True):
    """Yield (rain, None), or (None, error), for each of the years' observations in order, as load_observations gives them.

    The years the run holds (`run_years` and the year before each) are taken from, and kept in, the run's
    observation cache. A zarr store chunked over many years is read in blocks of about 1.5 GB, each decoded once
    and sliced per year; any other source is read a year at a time, the next few ahead in threads (zarr only;
    NetCDF files are read one at a time, as HDF5 is not thread-safe). Memory stays at one block, or
    a few years, however many years are asked for. hold=False reads without keeping even the held years.
    """
    return _obs_years_of(_ObsRead(obs, area, options, select, onset, tuple(run_years), dir_fig, parallel), years, hold)


def obs_years(years, kwargs, hold=True):
    """`read_obs_years` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return read_obs_years(years, got.obs, got.area, got.options, got.select, onset=got.onset_if_units(),
                          run_years=got.ev.years, dir_fig=got.dir_fig, parallel=got.parallel, hold=hold)


def _obs_years_of(read: _ObsRead, years, hold=True):
    """The generator behind `read_obs_years`."""
    years = list(years)
    if not years:
        return
    obs, select = read.obs, read.select
    keep = _run_years(read.run_years) if hold else set()

    def locate(year):
        """(observation-cache key, zarr store or None, file path or None) of a year."""
        store, path = _locate(year, read)
        return _obs_key(year, store, path, read), store, path

    def held(year):
        """The array the run holds for the year, or None."""
        with _obs_lock:
            return _obs_years.get(locate(year)[0])

    def finish(year, rainfall, view=False):
        """Hold the year if the run does (a copy when it is a view of a block, so the block can be freed), then apply grid_point."""
        if year in keep:
            with _obs_lock:
                rainfall = _obs_years.setdefault(locate(year)[0], rainfall.copy(deep=True) if view else rainfall)
        if select.grid_point:
            rainfall = rainfall.sel(lat=[select.lat], lon=[select.lon])
        return rainfall

    try:
        store = locate(years[0])[1]
    except FileNotFoundError:
        store = None
    if store is not None and len(years) > 1 and _long_time_chunks(store, obs.var):
        per_block = max(1, _GROUP_BYTES // _estimated_year_bytes(store, obs.var))
        for start in range(0, len(years), per_block):
            group = years[start:start + per_block]
            need = [year for year in group if held(year) is None]
            block = _read_obs_range(min(need), max(need), store, read) if need else None
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
    readers = _workers(read.parallel, len(years)) if store is not None else 1

    def single(year):
        """One year read on its own (or the held one), or the error that stopped it."""
        try:
            rainfall = held(year)
            if rainfall is None:
                _, store, path = locate(year)
                rainfall = _read_obs_year(year, store, path, read)
            return rainfall, None
        except Exception as exc:  # noqa: BLE001 - handed to the caller, which decides what a missing year means
            return None, exc

    for year, (rainfall, error) in zip(years, ordered_prefetch(single, years, readers)):
        yield (finish(year, rainfall), None) if error is None else (None, error)


def _workers(parallel, tasks) -> int:
    """The workers for `tasks` independent tasks under the run's parallel setting (the defaults when it is None)."""
    config = {} if parallel is None else {"parallel": parallel.enabled, "parallel_workers": parallel.workers}  # R2-bridge
    return worker_count(config, tasks)


def _read_obs_year(year, store, path, read: _ObsRead):
    """Read one year of one variable: select the year and the region's box on the lazy array, read once, mask in place.

    A tuple of files (a year listed in several files) is read file by file and joined along time.
    """
    obs = read.obs
    if isinstance(path, tuple):
        parts = [_read_obs_year(year, store, one, read) for one in path]
        return xr.concat(parts, dim="time").sortby("time")
    if store is not None:
        ds = open_store(store)
        ds = dim_fmt(ds)
        ds = ds[[obs.var]].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if ds.sizes["time"] == 0:
            raise FileNotFoundError(f"No observations for year {year} in {store}")
    else:
        ds = dim_fmt(rename_model_dims(xr.open_dataset(path), obs.dims_map()))
    if obs.units is not None:
        try:
            return _read_obs_units(ds[obs.var], obs.units, obs.unit_cvt, read)
        finally:
            if path is not None:
                ds.close()
    da = region_select(ds[[obs.var]], area=read.area, select=read.select, read_by="time",
                       dataset_name=str(obs.name or "obs"), dir_fig=read.dir_fig)[obs.var]
    da = _in_memory(da)
    if path is not None:
        ds.close()
    return _scale_in_place(da, obs.unit_cvt)


def _units_key(read: _ObsRead) -> tuple:
    """What reading observations onto units adds to the key of a held year (nothing for grid cells)."""
    obs = read.obs
    if obs.units is None:
        return ()
    return (repr(obs.units), repr(units_files(obs.units)), read.options.missing,
            repr(sorted(read.select.keep_unit_ids or ())), repr(obs.dims_map()), str(obs.day_boundary),
            str(None if read.onset is None else read.onset.lead_label))


def _obs_key(year, store, path, read: _ObsRead):
    """What identifies a loaded observation year: the source (with its size and time), the settings that shape it and the cells."""
    obs, area = read.obs, read.area
    source = store or path
    try:
        stats = [os.stat(p) for p in (source if isinstance(source, tuple) else (source,))]
        ident = tuple((stat.st_size, stat.st_mtime_ns) for stat in stats)
        ident = ident[0] if len(ident) == 1 else ident
    except OSError:
        ident = None
    land, shp = implicit_masks(read.select)
    return (year, str(source), ident, obs.var, obs.unit_cvt, str(area.region), land, shp, str(area.nc_mask),
            str(obs.name or "obs"), _mask_text(area)) + _units_key(read)


def load_observations(year, obs, area, options, select=_NO_SELECTION, *, onset=None, run_years=(), dir_fig=None):
    """Load observed daily rainfall for a given year (from the year's NetCDF file, or from a .zarr store of all years).

    The region's box and, for a store, the year are selected before any data is read, and the cell mask and the
    unit conversion are applied in place. The years the run scores (`run_years`, and the year before each)
    are read once per run: later calls (other windows, cases, the climatology) get the same array back, so treat
    it as read-only. Other years are read afresh when asked for.

    With obs.paths (names or globs; select.obs_files_years keeps the files of those years) the year's files are the
    listed ones with that year in their name, instead of obs.file_pattern.format(year) in obs.dir. A region of None, or
    data on a "unit" dimension, is read without a box or cell mask; observations on units need `onset` (the day
    boundary and lead label their dates are moved by). `select` may pick one grid point and keep some units.
    """
    return _load_observation(year, _ObsRead(obs, area, options, select, onset, tuple(run_years), dir_fig))


def load_imd_rainfall(year, **kwargs):
    """`load_observations` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return load_observations(year, got.obs, got.area, got.options, got.select, onset=got.onset_if_units(),
                             run_years=got.ev.years, dir_fig=got.dir_fig)


def _load_observation(year, read: _ObsRead):
    """The body of `load_observations`."""
    obs, select = read.obs, read.select
    store, path = _locate(year, read)
    key = _obs_key(year, store, path, read)
    with _obs_lock:
        rainfall = _obs_years.get(key)
    if rainfall is None and store is not None and year in _run_years(read.run_years) \
            and _long_time_chunks(store, obs.var):
        # an archive chunked over many years: read all the years the run holds in one pass, not one decode per year
        for _ in _obs_years_of(read, sorted(_run_years(read.run_years))):
            pass
        with _obs_lock:
            rainfall = _obs_years.get(key)
    if rainfall is None:
        rainfall = _read_obs_year(year, store, path, read)
        if year in _run_years(read.run_years):
            with _obs_lock:
                rainfall = _obs_years.setdefault(key, rainfall)

    if select.grid_point:
        rainfall = rainfall.sel(lat=[select.lat], lon=[select.lon])

    return rainfall


def _read_obs_uncached(year, read: _ObsRead):
    """Read one observation year for a single use: the held copy if there is one, else a read that is not kept."""
    store, path = _locate(year, read)
    with _obs_lock:
        held = _obs_years.get(_obs_key(year, store, path, read))
    if held is None:
        held = _read_obs_year(year, store, path, read)
    if read.select.grid_point:
        held = held.sel(lat=[read.select.lat], lon=[read.select.lon])
    return held


def _forecast_sources(year, model_dir, file_pattern, paths=None, years_filter=None) -> list:
    """What holds the year's forecasts: [None] for a zarr store (all years), else the file paths.

    One file, file_pattern.format(year) in model_dir, or the listed `paths` (see year_files) of the year, one init
    or many in each. Raises FileNotFoundError when the year is absent.
    """
    if paths:
        found = year_files(paths, model_dir or None, years_filter, year)
        if not found:
            raise FileNotFoundError(f"No forecast file found for year {year} among the listed paths {list(paths)}")
        return found
    if is_zarr(model_dir):
        return [None]
    file_path = os.path.join(model_dir, file_pattern.format(year))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return [file_path]


def _fix_dims_for(ensemble):
    """The function that standardises a forecast's dimension names: members required (True), absent (False) or optional
    (None, as the units pipeline reads files with or without them)."""
    if ensemble is None:
        def optional_member(ds, model_dims=None):
            ds = dim_fmt_model(ds, model_dims)
            member = find_dim(list(ds.coords), "member", fuzzy=True)
            return ds if member in (None, "member") else ds.rename({member: "member"})
        return optional_member
    return dim_fmt_model_ensemble if ensemble else dim_fmt_model


def _open_forecast(year, model_dir, file_pattern, ensemble, model_dims, filtered_dates_yr, file_path=None, multi=False):
    """Open the year's forecasts lazily and select its inits; returns (Dataset, close function).

    filtered_dates_yr None keeps every init of the file (the units pipeline reads whole files).

    A model_dir that is a zarr store holds all years: the inits of the year are picked by time. Otherwise the
    year's NetCDF file file_pattern.format(year), or `file_path` when the caller names it, is opened. Raises
    FileNotFoundError when the year is absent. With `multi` (one of several files of the year) a file without a
    wanted init gives None instead of an error.
    """
    fix = _fix_dims_for(ensemble)
    if file_path is None and is_zarr(model_dir):
        ds = fix(open_store(model_dir), model_dims)
        close = lambda: None  # noqa: E731 - the store handle is shared
    else:
        file_path = file_path or os.path.join(model_dir, file_pattern.format(year))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ds = fix(xr.open_dataset(file_path), model_dims)
        close = ds.close
    if filtered_dates_yr is None:
        return ds, close
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        if multi:
            close()
            return None, None
        if is_zarr(model_dir) and file_path is None and not (available_init_times.year == year).any():
            raise FileNotFoundError(f"No forecasts of {year} in {model_dir}")
        raise ValueError(f"No matching initialization times found for year {year}")
    return ds.sel(init_time=matching_times), close


def _forecast_on_units(da, spec: UnitsSpec, keep_ids, unit_cvt):
    """The selected issues of a forecast variable on the run's evaluation units, dims (init_time, [member], unit, step).

    The units are those of the observations (their ids, as the same specification builds them for this source), the
    millimetre factor is the file's declared unit or unit_cvt (as `UnitForecast` reads it).
    """
    units, positions = select_units(da, spec, keep_ids)
    out = read_units(da, spec, units, positions, block_dim="init_time", scale=declared_scale(da, unit_cvt))
    return out.transpose("init_time", *[d for d in ("member", "unit", "step") if d in out.dims])


def _load_forecast(year, read: _ForecastRead):
    """Read one year of forecasts as a loaded DataArray in mm, reading only what the run needs.

    Inits, leads (up to what the onset rule reads), members and the region's box are selected on the lazy array;
    the selection is read once, the cell mask and the unit conversion (x * unit_cvt, in float32 as before) are
    applied in place on it. The file's step 0, if present, is dropped. A year listed in several files
    (select.model_paths) is read file by file and joined along init_time.
    """
    model, ev, select = read.model, read.ev, read.select
    sources = _forecast_sources(year, model.dir, model.file_pattern, select.model_paths, select.model_files_years)
    dates = initialization_dates(year, ev)
    spec = model.units
    parts = []
    for file_path in sources:
        ds, close = _open_forecast(year, model.dir, model.file_pattern, read.ensemble, model.dims_map(), dates,
                                   file_path, multi=len(sources) > 1)
        if ds is None:
            continue
        try:
            if 'step' not in ds.dims:
                raise KeyError("'step' dimension not found in dataset")
            # Onset detection reads no further than this lead day, so loading the rest only costs memory.
            ds = select_leads(ds, None, steps_needed(ev.max_forecast_day, read.onset))
            if read.ensemble and ev.members:
                ds = select_members(ds, ev.members)
            if spec is not None:   # evaluation units: the rain is averaged onto them as it is read
                parts.append(_forecast_on_units(ds[model.var], spec, select.keep_unit_ids, model.unit_cvt))
                continue
            da = region_select(ds[[model.var]], area=read.area, select=select, read_by="init_time",
                               dataset_name=str(model.name or "model"), dir_fig=read.dir_fig)[model.var]
            parts.append(_in_memory(da))
        finally:
            close()
    if not parts:
        raise ValueError(f"No matching initialization times found for year {year}")
    da = parts[0] if len(parts) == 1 else xr.concat(parts, dim="init_time").sortby("init_time")
    return da if spec is not None else _scale_in_place(da, model.unit_cvt)   # units are scaled as they are averaged


def load_forecast(year, model, ev, area, options, onset, *, ensemble, select=_NO_SELECTION, dir_fig=None):
    """Load a model's forecasts of one year: the twice-weekly initializations between ev.start_date and ev.end_date.

    Filters for the init days (0 = Monday) of ev.date_filter_year. The forecast file is expected to be named
    model.file_pattern.format(year) in model.dir (or model.dir is a .zarr store of all years, or select.model_paths
    lists the files, any number per year) with the variable model.var, daily accumulated rainfall with dimensions
    (init_time, [member], lat, lon, step). With `ensemble` the members ev.members are selected before the read.
    `onset` says which lead days the onset rule reads (the rest are not loaded); `options` is the run's reading options
    (the ingest takes them: see `ingest_forecast`).

    Returns the DataArray of precipitation in mm, read once with the region, inits and leads selected first.
    """
    return _load_forecast(year, _ForecastRead(model, ev, area, options, select, onset, dir_fig, ensemble))


def _flat_forecast(year, kwargs, ensemble):
    """`load_forecast` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return load_forecast(year, got.model, got.ev, got.area, got.options, got.onset_if_steps(), ensemble=ensemble,
                         select=got.select, dir_fig=got.dir_fig)


def get_forecast_deterministic_twice_weekly(year, **kwargs):
    """`load_forecast` of a flat config, without members."""  # R2-bridge
    return _flat_forecast(year, kwargs, False)


def get_forecast_probabilistic_twice_weekly(year, **kwargs):
    """`load_forecast` of a flat config, with the members asked for."""  # R2-bridge
    return _flat_forecast(year, kwargs, True)


def observed_history(year, current, onset, obs, area, options, select=_NO_SELECTION, *, run_years=(), dir_fig=None):
    """The observed rain to stitch before each forecast of `year`, or None when forecast_history is "none".

    `current` is the year's observations (already loaded). The previous year's file is
    added when a season or the rule's look-back reaches before 1 January; a missing
    previous file is skipped, which makes the earliest searches INSUFFICIENT_DATA.
    """
    if onset.forecast_history != "observed":
        return None
    if not history_needs_previous_year(onset):
        return current
    try:
        previous = load_observations(year - 1, obs, area, options, select, onset=onset, run_years=run_years,
                                     dir_fig=dir_fig)
    except FileNotFoundError:
        return current
    return xr.concat([previous, current], dim="time")


def load_history(year, current, **kwargs):
    """`observed_history` of a flat config."""  # R2-bridge
    got = _bridge().reader_inputs(kwargs)
    return observed_history(year, current, got.onset(), got.obs, got.area, got.options, got.select,
                            run_years=got.ev.years, dir_fig=got.dir_fig)


# ---------------------------------------------------------------------------------------------------------
# Rain on units: source cells -> unit series (onset_blending's regrid and cell transform, as one transform)
# ---------------------------------------------------------------------------------------------------------

SOURCE_POLICIES = ("flag", "keep", "zero", "drop")
"""Missing-rain policies applied to the source cells before they are averaged onto units: "flag" and "keep" leave
NaN as it is (a unit averages its observed sources), "zero" counts NaN as 0 mm and "drop" removes every source cell
with a missing day in a calendar year from that year's means (see `unit_series`)."""


def unit_transform(units, lat, lon, positions=None, *, tolerance=None, wrap_longitude=False, convention=None):
    """The transform of `units` (hcwf.spatial.EvaluationUnits) from the cells of a (lat, lon) grid, restricted to the
    units at `positions` (all when None).

    The weights are matched to the grid's cells by exact coordinates (as onset_blending matches them) or within
    `tolerance`; wrap_longitude first puts the weights' longitudes in the grid's convention. `convention` is the
    grid-id convention of weights keyed by source ids.
    """
    from hcwf.spatial import SparseTransform

    full = units.transform(lat, lon, tolerance=tolerance, convention=convention, wrap_longitude=wrap_longitude)
    if positions is None:
        return full
    positions = np.asarray(positions, dtype=np.int64)
    out = SparseTransform(full.weighted_matrix[positions], full._source_ids, tuple(np.asarray(full.target_ids)[positions]),
                          keys=full.keys, convention=full.convention, grid=full.grid)
    return out


def _cell_positions(transform, lat, lon):
    """Each cell of the (lat, lon) grid (latitude-major) as its position among the transform's sources, or None when the
    sources are the grid's cells themselves."""
    if transform.keys == "coordinate":
        return None
    cell_lat, cell_lon = np.meshgrid(np.asarray(lat, float), np.asarray(lon, float), indexing="ij")
    return transform.positions(cell_lat.reshape(-1), cell_lon.reshape(-1))


def feeding_cells(transform, lat, lon):
    """Boolean (nlat, nlon): the grid cells that are a source of some unit of the transform (what a read must keep)."""
    sources = np.unique(transform.weighted_matrix.indices)
    place = _cell_positions(transform, lat, lon)
    keep = np.zeros(len(lat) * len(lon), dtype=bool)
    if place is None:
        keep[sources] = True
    else:
        keep = np.isin(place, sources)
    return keep.reshape(len(lat), len(lon))


def unit_series(da, transform, *, missing="flag", post_scale=None, support=False):
    """Average a loaded DataArray with lat and lon dims onto the units of `transform`: dims become (the others, "unit").

    Each unit is the weighted mean of its observed (finite) source cells, in float64 mm, NaN where none is observed
    (SparseTransform.observed_mean). `missing` is applied to the sources first ("zero" counts NaN as 0 mm; "drop" needs
    a "time" dim and removes a cell from the means of every calendar year in which it has a missing day, see
    SOURCE_POLICIES). `post_scale` multiplies the means (a unit factor applied after the averaging, as onset_blending
    does for a regrid; apply it to `da` first for a cell transform). The array is read in column blocks, so working
    memory stays one block beyond the result. With `support` the result is (series, (unit, year) booleans of which
    units have a source that year, the years).
    """
    if missing not in SOURCE_POLICIES:
        raise ValueError(f"missing must be one of {SOURCE_POLICIES}, got {missing!r}")
    lat, lon = np.asarray(da["lat"].values), np.asarray(da["lon"].values)
    others = [d for d in da.dims if d not in ("lat", "lon")]
    if missing == "drop" and others != ["time"]:
        raise ValueError("missing 'drop' needs data with the dims (time, lat, lon)")
    view = np.moveaxis(da.values, [da.dims.index("lat"), da.dims.index("lon")], [-2, -1])
    rest_shape = view.shape[:-2]
    cells = view.reshape(-1, len(lat) * len(lon))
    place = _cell_positions(transform, lat, lon)
    n_unit, n_source = transform.shape
    used = None if place is None else np.flatnonzero(place >= 0)
    out = np.empty((cells.shape[0], n_unit))

    def sources(rows):
        """(source, column) float64 values of the given rows of `cells`."""
        block = cells[rows]
        if place is None:
            return block.T.astype(float)
        values = np.full((n_source, block.shape[0]), np.nan)
        values[place[used]] = block[:, used].T.astype(float)
        return values

    years = present = None
    if missing == "drop":
        times = da["time"]
        years = np.asarray(times.dt.year.values)
        distinct = np.unique(years)
        present = np.ones((n_source, len(distinct)), dtype=bool)
        for y, year in enumerate(distinct):
            rows = np.flatnonzero(years == year)
            values = sources(rows)
            incomplete = np.isnan(values).any(axis=1)
            values[incomplete] = np.nan
            present[incomplete, y] = False
            out[rows] = transform.observed_mean(values).T
    else:
        for start in range(0, cells.shape[0], 256):
            rows = slice(start, min(cells.shape[0], start + 256))
            values = sources(rows)
            if missing == "zero":
                values = np.where(np.isnan(values), 0.0, values)
            out[rows] = transform.observed_mean(values).T
    if post_scale is not None and post_scale != 1.0:
        out *= post_scale
    coords = {d: da[d].values for d in others if d in da.coords}
    coords["unit"] = np.asarray(transform.target_ids, dtype=str)
    result = xr.DataArray(out.reshape(rest_shape + (n_unit,)), dims=tuple(others) + ("unit",), coords=coords,
                          attrs=da.attrs, name=da.name)
    if not support:
        return result
    if present is None:
        distinct = np.unique(np.asarray(da["time"].dt.year.values)) if "time" in da.coords else np.array([0])
        present = np.ones((n_source, len(distinct)), dtype=bool)
    return result, transform.supported(present), distinct


def read_on_units(da, units, *, block_dim, region=None, positions=None, tolerance=None, wrap_longitude=False,
                  convention=None, scale=None, post_scale=None, missing="flag", support=False):
    """Read a lazy DataArray (dims named as HCWF's, lat and lon among them) onto `units`, reading only what feeds them.

    The transform of `units` from the array's own cells is compiled (`unit_transform`; the array's lat and lon are
    matched as they are, so give it the cut or uncut array in the order you want matched), only the storage tiles
    holding a source cell of the wanted `positions` are read (block by block along block_dim, `load_blocks`), `scale`
    multiplies the cells in place before they are averaged (as for a cell transform), and `unit_series` averages them
    (post_scale, missing and support as there). Returns the unit series, dims (the others, "unit").

    With a `region` the array is first cut to the region's box (latitude ascending, longitude in the region's
    convention, so use wrap_longitude with weights written in the other one); no cell mask is applied, the weights
    decide which cells count. Without a region the file's own order and extent are read.
    """
    if region is not None:
        da = region_box(da, region=region)
    lat, lon = np.asarray(da["lat"].values), np.asarray(da["lon"].values)
    transform = unit_transform(units, lat, lon, positions, tolerance=tolerance, wrap_longitude=wrap_longitude,
                               convention=convention)
    if scale and da.variable._in_memory:
        da = da.copy(deep=True)  # the caller's array is not scaled in place
    loaded = _scale_in_place(load_blocks(da, block_dim, keep=feeding_cells(transform, lat, lon)), scale)
    return unit_series(loaded, transform, missing=missing, post_scale=post_scale, support=support)


def read_unit_dim(da, *, block_dim, ids=None, scale=None):
    """Read the kept units of a lazy DataArray that already has a "unit" dimension (pre-aggregated admin units).

    `ids` lists the units to keep, in the order wanted (default: all, in file order); only the storage tiles that hold
    one are read. `scale` multiplies in place. Returns the array with its units compacted to the kept ones.
    """
    labels = np.asarray(da["unit"].values).astype(str)
    if ids is None:
        wanted = np.arange(labels.size)
    else:
        index = {label: i for i, label in enumerate(labels)}
        missing = [str(i) for i in ids if str(i) not in index]
        if missing:
            raise KeyError(f"units not in the data: {', '.join(missing[:10])}")
        wanted = np.array([index[str(i)] for i in ids], dtype=np.int64)
    keep = np.zeros(labels.size, dtype=bool)
    keep[wanted] = True
    loaded = load_blocks(da, block_dim, keep=keep, keep_dims=("unit",))
    return _scale_in_place(loaded.isel(unit=wanted), scale)  # isel with positions copies: nothing of the caller's is scaled


# ---------------------------------------------------------------------------------------------------------
# Evaluation units: the config's `units:` section, reading a file onto units, lead and member selection
# ---------------------------------------------------------------------------------------------------------

def _units_of(source, name: str = "observations") -> UnitsSpec | None:
    """The evaluation units of a source: the UnitsSpec itself, or that of an ObsSpec, ModelSpec or ReferenceSpec (None:
    the grid cells are used). R2-bridge: a flat config is resolved for source `name` (see `UnitsSpec.from_flat`)."""
    if isinstance(source, Mapping):
        return UnitsSpec.from_flat(source, name)
    return source if source is None or isinstance(source, UnitsSpec) else source.units


def units_spec(source, name: str = "observations") -> UnitsSpec | None:
    """The evaluation units of a source: its resolved UnitsSpec, or None when the config has no `units:` keys (the grid
    cells are used). `source` is the run's ObsSpec, ModelSpec or ReferenceSpec; `UnitsSpec.from_flat` made it once, when the
    RunConfig was built. R2-bridge: a flat config and a source name ("observations" or a model's name) are resolved
    the same way."""
    return _units_of(source, name)


def units_files(source, name: str = "observations") -> list:
    """The identities (path, size, time) of the files the source's units specification names; none without units."""
    spec = _units_of(source, name)
    return [] if spec is None else [file_identity(p) for p in spec.files()]


def point_dims(source) -> tuple:
    """The spatial dims of a run's onset arrays: ("unit",) on evaluation units, else ("lat", "lon") (see hcwf.spatial.axes).
    `source` is the observations' ObsSpec (or its UnitsSpec; R2-bridge: a flat config)."""
    return axes.UNIT if _units_of(source) is not None else axes.GRID


def unit_table(source, ids) -> EvaluationUnits:
    """The run's evaluation units of the string `ids`, in that order: their names, centroids and, for polygons, geometry.

    `source` is the observations' ObsSpec (or its UnitsSpec; R2-bridge: a flat config). Polygon (shapefile) and weights
    units come from the specification; ids of cells or of a pre-aggregated file are labelled with the centroids table,
    or with the cell they spell ("<lat>_<lon>").
    """
    spec, ids = _units_of(source), np.asarray(ids).astype(str)
    units = spec.build(ids=None if (spec.shapefile or spec.weights) else ids)
    order = pd.Index(units.ids).get_indexer(ids)
    if (order < 0).any():
        raise ValueError(f"units not among the run's units: {', '.join(ids[order < 0][:10])}")
    table = units.table.iloc[order].reset_index(drop=True)
    for n in np.flatnonzero(table["lat"].isna().to_numpy()):
        parsed = unit_ids.parse_grid_id(table["id"][n])
        if parsed is not None:
            table.loc[n, ["lat", "lon"]] = parsed
    geometry = None if units.geometry is None else [units.geometry[i] for i in order]
    return EvaluationUnits(table, units.kind, geometry=geometry)


def declared_scale(da, unit_cvt) -> float | None:
    """Millimetres per unit of the array's rainfall: its declared `units` attribute, else unit_cvt; None for 1."""
    declared = da.attrs.get("units")
    declared = str(declared).strip() if declared is not None and str(declared).strip() else None
    scale = unit_scale(declared) if declared is not None else (unit_cvt or 1.0)
    return None if scale == 1.0 else float(scale)


def select_units(da, spec: UnitsSpec, keep_ids=None):
    """(units, positions of those used, ordered by id) for an array with lat and lon, or a "unit", dimension."""
    if "unit" in da.dims:
        return spec.select(ids=np.asarray(da["unit"].values), keep_ids=keep_ids)
    return spec.select(lat=da["lat"].values, lon=da["lon"].values, keep_ids=keep_ids)


def read_units(da, spec: UnitsSpec, units, positions, *, block_dim, scale=None, missing="flag", support=False):
    """The array's rain on the used units, dims (the others, "unit"): float64 for averaged units, the file's dtype for
    its own cells or units. A regrid scales after averaging, a cell transform before (see UnitsSpec); with `support`
    a cell transform also returns which units have a source each year and the years."""
    if "unit" in da.dims:
        return read_unit_dim(da, block_dim=block_dim, ids=units.ids[positions], scale=scale)
    common = dict(block_dim=block_dim, positions=positions, convention=spec.convention, missing=missing,
                  support=support)
    if spec.how == "regrid":
        return read_on_units(da, units, post_scale=scale, **common)
    out = read_on_units(da, units, scale=scale, **common)
    return out.astype(da.dtype, copy=False) if spec.how == "cells" else out


def _read_obs_units(da, spec: UnitsSpec, unit_cvt, read: _ObsRead):
    """One year of observations on the units, dims (time, unit), float64 millimetres, dates moved onto the forecasts'
    labels. The coordinate `present` says which units have a series this year (a cell transform without a source, or a
    missing day under the "drop" policy, leave a unit out)."""
    da = da.transpose("time", ...)
    stamps = pd.DatetimeIndex(da["time"].values)
    years = np.unique(stamps.year.to_numpy())
    if len(years) != 1:
        raise ValueError(f"observations on units are read a year per file; this one holds the years {years.tolist()}")
    units, positions = select_units(da, spec, read.select.keep_unit_ids)
    scale, policy = declared_scale(da, unit_cvt), read.options.missing or "flag"
    present = np.ones(len(positions), dtype=bool)
    if spec.how == "cell_transform" and "unit" not in da.dims:
        series, supported, _ = read_units(da, spec, units, positions, block_dim="time", scale=scale, missing=policy,
                                          support=True)
        present, policy = np.asarray(supported)[:, 0].copy(), "keep"  # the policy was applied to the source cells
    else:
        series = read_units(da, spec, units, positions, block_dim="time", scale=scale)
    ids = units.ids[positions]
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("points share the unit ids")
    values = np.asarray(series.transpose("time", "unit").values, dtype=float)
    if policy == "drop":
        present &= ~np.isnan(values).any(axis=0)
    if read.onset is None:
        raise ValueError("observations on units need the onset setup (the day boundary and lead label their dates move by)")
    shift = observation_shift_days(read.onset.obs_day_boundary, read.onset.lead_label)
    dates = (stamps.normalize() + pd.Timedelta(days=shift)).to_numpy()
    return xr.DataArray(values, dims=("time", "unit"), coords={"time": dates, "unit": ids.astype(str),
                                                               "present": ("unit", present)}, attrs={"units": "mm"})


def select_leads(ds, first=None, last=None, *, strict=False, rain_horizon=None, where="forecast"):
    """The lead days of a dataset or array with a `step` dimension, `first`..`last` (None: from the first positive
    step, to the last).

    Without `strict` this is HCWF's reading: a step 0 is dropped and the steps up to `last` kept as they come. With
    `strict` (onset_blending's) the days must be whole, distinct and consecutive; they are put in ascending order and
    the step labels become 1..n, the days themselves a `lead_day` coordinate. `rain_horizon` (max_day, policy) checks
    first that the file holds the lead days 1..max_day, exactly ("exact") or with later days permitted ("truncate").
    """
    if not strict and rain_horizon is None:
        if first is None:
            if ds["step"][0].values == 0:
                ds = ds.sel(step=slice(1, None))
        else:
            ds = ds.sel(step=slice(first, None))
        return ds if last is None else ds.sel(step=slice(None, last))
    values = np.asarray(ds["step"].values).reshape(-1)
    numbers = []
    for v in values:
        try:
            numbers.append(int(v))
        except (TypeError, ValueError):
            numbers.append(None)
    if rain_horizon is not None and rain_horizon[0] is not None:
        _check_rain_horizon([v for v, d in zip(values, numbers) if d is not None and (first is None or d >= first)],
                            int(rain_horizon[0]), where, allow_extra=str(rain_horizon[1] or "exact").lower() == "truncate")
    keep = np.array([d is not None and (first is None or d >= first) and (last is None or d <= last) for d in numbers])
    if not keep.any():
        raise ValueError(f"Day filtering removed all columns for: {where}")
    positions = np.flatnonzero(keep)
    days = np.asarray([numbers[p] for p in positions])
    order = np.argsort(days, kind="stable")
    positions, days = positions[order], days[order]
    if len(set(days.tolist())) != len(days) or any(b - a != 1 for a, b in zip(days, days[1:])):
        raise ValueError(f"{where}: lead days must be consecutive and distinct, got {days[:8].tolist()}")
    out = ds.isel(step=positions)
    return out.assign_coords(lead_day=("step", days)).assign_coords(step=np.arange(1, len(days) + 1))


def _whole_day(value) -> int | None:
    """A lead day as a whole number, None for anything else."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if np.isfinite(numeric) and numeric.is_integer() else None


def _check_rain_horizon(day_values, rain_day_max: int, where: str, *, allow_extra: bool) -> None:
    """A strict rain horizon needs the lead days 1..rain_day_max exactly (later days too, with allow_extra)."""
    parsed = [_whole_day(v) for v in day_values]
    invalid = [repr(v) for v, d in zip(day_values, parsed) if d is None]
    valid = [d for d in parsed if d is not None]
    duplicates = sorted({d for d in valid if valid.count(d) > 1})
    actual, required = set(valid), set(range(1, rain_day_max + 1))
    missing, extra = sorted(required - actual), sorted(actual - required)
    problems = []
    if invalid:
        problems.append(f"non-integer day values={invalid[:10]}")
    if duplicates:
        problems.append(f"duplicate days={duplicates[:10]}")
    if missing:
        problems.append(f"missing days={missing[:10]}")
    if [d for d in extra if d < 1]:
        problems.append(f"nonpositive days={[d for d in extra if d < 1][:10]}")
    if [d for d in extra if d > rain_day_max] and not allow_extra:
        problems.append(f"extra days={[d for d in extra if d > rain_day_max][:10]}")
    if problems:
        requirement = (f"days 1..{rain_day_max}, with later days permitted" if allow_extra
                       else f"exactly days 1..{rain_day_max}")
        raise ValueError(f"{where}: strict rainfall horizon requires {requirement}; " + "; ".join(problems))


def select_members(ds, members=None, *, at_most=None, among=None, where="forecast"):
    """The members of a dataset or array with a `member` dimension.

    `members` (HCWF's run setting) names members that must exist. `at_most` and `among` (onset_blending's) keep the
    members present whose number is at most that and among those; none left is an error.
    """
    if members:
        return ds.sel(member=list(members))
    numbers = np.asarray(ds["member"].values).astype(int)
    keep = np.ones(len(numbers), dtype=bool)
    if at_most is not None:
        keep &= numbers <= at_most
    if among is not None:
        keep &= np.isin(numbers, among)
    if not keep.any():
        raise ValueError(f"{where}: none of its members are read")
    return ds if keep.all() else ds.isel(member=np.flatnonzero(keep))


_RUN_VALUES = 1 << 23
"""Values (members x units x lead days) of the issues one run of a forecast file holds at most."""


class UnitForecast:
    """One forecast file read onto evaluation units, a run of issues at a time (memory stays one run).

    Opens the file lazily as `_load_forecast` does (dimension names, lead days, members), keeps every issue, and puts
    the units of `spec` (those among `keep_ids`) in id order. `issues`, `ids`, `days` (the lead days kept, their step
    labels renumbered 1..n) and `members` describe it; `runs()` yields (first, stop, rain) with rain over (init_time,
    member, unit, step) in the file's dtype for cells and float64 for averaged units.
    When the first lead day kept is d (steps are renumbered 1..n) init_time is the issue plus d - 1 days, so every step
    has its true date.
    `issue_dates` (dates or ISO texts) keeps only the issues on those days, and reads nothing of the others: a file
    that has none of them has no issues and `runs()` yields nothing.
    """

    def __init__(self, path, model, spec: UnitsSpec, *, keep_ids=None, issue_dates=None, select=_NO_SELECTION):
        """`model` is the forecast's ModelSpec (variable, dimension names, unit factor); `select` picks the lead days
        (first_lead, last_lead, rain_horizon) and members (max_member, member_ids). R2-bridge: a flat config in place of
        `model` gives both."""
        if isinstance(model, Mapping):   # R2-bridge
            got = _bridge().reader_inputs(model)
            model, select = got.model, got.select
        self.path, name = path, Path(path).name
        variable = model.var
        ds, self._close = _open_forecast(None, None, None, None, model.dims_map(), None, file_path=str(path))
        try:
            if variable not in ds.variables:
                raise ValueError(f"{name} has no {variable!r} variable (it has: {', '.join(map(str, ds.variables))})")
            da = ds[[variable]][variable]
            if "step" not in da.dims or "init_time" not in da.dims:
                raise ValueError(f"{name}: no {variable!r} variable over issue time, lead day, optional member and "
                                 "spatial dimensions only")
            self.scale = declared_scale(da, model.unit_cvt)
            da = select_leads(da, select.first_lead, select.last_lead, strict=True,
                              rain_horizon=select.rain_horizon, where=f"NetCDF file {name}")
            self.days = np.asarray(da["lead_day"].values)
            if "member" not in da.dims:
                da = da.expand_dims(member=[0])
            elif select.max_member is not None or select.member_ids is not None:
                da = select_members(da, at_most=select.max_member, among=select.member_ids, where=name)
            self.issues = pd.DatetimeIndex(da["init_time"].values).normalize().to_numpy()
            if issue_dates is not None:
                wanted = np.array([np.datetime64(d, "D") for d in issue_dates])
                rows = np.flatnonzero(np.isin(self.issues.astype("datetime64[D]"), wanted))
                da, self.issues = da.isel(init_time=rows), self.issues[rows]
            spatial = [d for d in da.dims if d not in ("init_time", "member", "step")]
            self.da = da.transpose("init_time", "member", "step", *spatial)
            self.spec = spec
            self.units, self.positions = select_units(self.da, spec, keep_ids)
            self.ids = self.units.ids[self.positions]
            self.members = np.asarray(self.da["member"].values)
        except BaseException:
            self._close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._close()

    def runs(self):
        """(first, stop, rain) for each run of issues (whole storage chunks along init_time)."""
        da = self.da
        per_issue = max(1, da.sizes["member"] * len(self.ids) * len(self.days))
        rows = max(1, _RUN_VALUES // per_issue)
        chunks = da.encoding.get("chunksizes") or da.encoding.get("chunks")
        if chunks and len(chunks) == da.ndim:
            stored = int(chunks[da.dims.index("init_time")])
            rows = max(1, -(-rows // stored)) * stored
        n_issue = len(self.issues)
        for start in range(0, n_issue, max(1, min(rows, n_issue))):
            stop = min(n_issue, start + rows)
            rain = read_units(da.isel(init_time=slice(start, stop)), self.spec, self.units, self.positions,
                              block_dim="init_time", scale=self.scale)
            labelled = self.issues[start:stop] + np.timedelta64(int(self.days[0]) - 1, "D")
            yield start, stop, rain.transpose("init_time", "member", "unit", "step").assign_coords(init_time=labelled)


def unit_forecast_bytes(path, variable) -> int:
    """Peak memory to process one forecast file on units: one run of issues and its working copies."""
    import math

    import netCDF4

    with netCDF4.Dataset(path) as ds:
        var = ds.variables.get(variable)
        if var is None:
            return 512 * 2**20
        values = math.prod(var.shape)
    return int(min(values, _RUN_VALUES * 4) * 30) + 512 * 2**20
