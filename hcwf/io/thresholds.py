"""The named thresholds of an onset definition: files and constants as given, the rest computed from the observations (or the
model's own forecasts) of the threshold years and cached.

`load_thresholds(onset, obs, area, options, ev, output, model=...)` keeps the loaded set for the run.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from hcwf.cache import NETCDF, NETCDF_GROUPS, Store, file_identity, memo
from hcwf.config.model import NO_SELECTION
from hcwf.config.onset import Thresholds
from hcwf.io.forecasts import ForecastRead, ingest_forecast, load_forecast
from hcwf.io.observations import ObsRead
from hcwf.io.reference import LazyRain, reference_rain
from hcwf.io.stores import forecast_sources
from hcwf.onset import definition as definition_mod
from hcwf.onset import observed_cache
from hcwf.onset.climatology import (
    ThresholdCache,
    compute_thresholds_from_forecasts,
    harmonic_seasons,
)
from hcwf.onset.thresholds import (
    ThresholdScale,
    read_threshold_grid,
    read_threshold_table,
    scale_values,
)
from hcwf.spatial.onto_units import units_files
from hcwf.spatial.select import region_select
from hcwf.spatial.units import UNITS_KEYS

log = logging.getLogger(__name__)

def _harmonic_cached(cache_dir, rain, years, files=()):
    """The harmonic season analysis of the reference years, read from or written to the cache directory."""
    parts = {"source": rain.source, "years": [int(years[0]), int(years[1])], "files": list(files),
             "grid": {dim: rain.coords[dim] for dim in rain.dims[:-1]}}
    return Store("harmonic_seasons", cache_dir, codec=NETCDF, label="Harmonic seasons").get_or_compute(
        parts, lambda: harmonic_seasons(rain, years))


def _model_forecasts(onset, years, read: ForecastRead):
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


def _model_files(years, read: ForecastRead) -> list:
    """The identity of what holds the model's forecasts of each year: its files, or the zarr store, or the missing year."""
    found = []
    model, select = read.model, read.select
    for year in range(int(years[0]), int(years[1]) + 1):
        try:
            sources = forecast_sources(year, model.dir, model.file_pattern, select.model_paths,
                                        select.model_files_years)
        except (FileNotFoundError, TypeError, AttributeError):
            found.append([f"missing {year}"])
            continue
        found.append([file_identity(f if f is not None else model.dir) for f in sources])
    return found


def _model_read_key(read: ForecastRead) -> dict:
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


def _model_thresholds(onset, pending, years, cache_dir, read: ForecastRead, obs_units):
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


def load_onset_thresholds(onset, obs, area, options, ev, output, *, model=None, select=NO_SELECTION,
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
        read = ForecastRead(model, ev, area, options, select, onset, output.figure_dir, bool(ev.probabilistic))
        thresholds.update(_model_thresholds(onset, pending, years, cache_dir, read, obs.units))
        pending = []
        if not onset.derived_seasons:
            return thresholds
    read = ObsRead(obs, area, options, select, onset, tuple(ev.years), output.figure_dir, parallel)
    probe = reference_rain(onset, (years[0], years[0]), read)
    full = LazyRain(probe, lambda: reference_rain(onset, years, read))
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


_loaded_thresholds = memo("io.thresholds.loaded", maxsize=32, share=True)
"""The threshold sets loaded for the run, by `load_thresholds`' key; the workers start with the parent's."""


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
                    select=NO_SELECTION, parallel=None, cache_dir=None):
    """The definition's named thresholds (what the onset detection takes as thresh_slice).

    `model` is the forecast model the thresholds come from when threshold_source is "model". Computed ones are cached
    under `cache_dir`, or the output's data folder; the loaded set is kept for the run (see `_loaded_thresholds`).
    """
    key = (_setup_key(onset), str(None if model is None else model.name), str(ev.max_forecast_day),
           str(None if not ev.windows else [w.span for w in ev.windows]), str(output.data_dir), str(obs.dir),
           str(obs.var), str(area.region), _thresholds_key(onset, obs, options, select, model))
    return _loaded_thresholds.get_or_make(key, lambda: load_onset_thresholds(
        onset, obs, area, options, ev, output, model=model, select=select, parallel=parallel, cache_dir=cache_dir))


def thresholds_of(rc, model, select=NO_SELECTION):
    """`load_thresholds` of a run (`rc`: RunConfig) for the source `model` (ModelSpec)."""
    return load_thresholds(rc.onset, rc.observations, rc.area, rc.onset_options, rc.evaluation, rc.output, model=model,
                           select=select, parallel=rc.parallel, cache_dir=rc.cache_dir)
