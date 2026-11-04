"""The named thresholds of an onset definition: files and constants as given, the rest computed and cached.

The computed thresholds come from the observations (or the model's own forecasts) of the threshold years, through
`hcwf.onset.climatology` and `hcwf.onset.thresholds`, and are cached under a fingerprint of their inputs
(`hcwf.cache.Store`). The reference rainfall they need is read by `hcwf.io.reference`, the forecasts by
`hcwf.io.forecasts`. `load_thresholds` keeps the loaded set for the run; `hcwf.io.warmup` calls it before the years
go to workers.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from hcwf.cache import NETCDF, NETCDF_GROUPS, Store, file_identity, memo
from hcwf.config.model import NO_SELECTION
from hcwf.config.onset import Thresholds
from hcwf.io.forecasts import ForecastRead, ingest_forecast, load_forecast
from hcwf.io.observations import ObsRead
from hcwf.io.reference import LazyRain, reference_probe, reference_rain
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

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.onset.model import ForecastRain

log = logging.getLogger(__name__)


def _harmonic_cached(cache_dir, rain, years, files=()):
    """Return the harmonic season analysis of the reference years, read from or written to the cache directory.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        The cache folder.
    rain : hcwf.onset.model.DailyRain or hcwf.io.reference.LazyRain
        The reference rainfall; only its source and grid are read on a cache hit.
    years : tuple of (int, int)
        The first and last reference year.
    files : sequence of str, default ()
        The identity of the observation files, part of the cache key.

    Returns
    -------
    xarray.Dataset
        The harmonic seasons (see `hcwf.onset.climatology.harmonic_seasons`).
    """
    parts = {"source": rain.source, "years": [int(years[0]), int(years[1])], "files": list(files),
             "grid": {dim: rain.coords[dim] for dim in rain.dims[:-1]}}
    return Store("harmonic_seasons", cache_dir, codec=NETCDF, label="Harmonic seasons").get_or_compute(
        parts, lambda: harmonic_seasons(rain, years))


def _model_forecasts(onset, years, read: ForecastRead) -> Iterator[ForecastRain]:
    """Yield the model's forecasts of each reference year as `hcwf.onset.model.ForecastRain`.

    Each year is loaded with HCWF's loader and unit conversion, so the thresholds are computed from the same
    forecasts the run evaluates. A year without a forecast file is skipped with a warning.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.
    years : tuple of (int, int)
        The first and last reference year.
    read : ForecastRead
        The typed inputs of the forecast reading.

    Yields
    ------
    hcwf.onset.model.ForecastRain
        The year's forecasts, dims ``(init_time, [member], lat, lon, step)``.

    Raises
    ------
    FileNotFoundError
        No forecast file exists for any of the years.
    """
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
    """Return the identity of what holds the model's forecasts of each year, for the cache key.

    Parameters
    ----------
    years : tuple of (int, int)
        The first and last reference year.
    read : ForecastRead
        The typed inputs of the forecast reading.

    Returns
    -------
    list of list of str
        For each year in order, the identities of its files (or of the zarr store), or ``["missing <year>"]``.
    """
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
    """Return the settings that change the model's forecasts as read (grid, units, steps, days), beside the files.

    Parameters
    ----------
    read : ForecastRead
        The typed inputs of the forecast reading.

    Returns
    -------
    dict
        The settings, keyed by name, for the cache fingerprint.
    """
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
    """Return the evaluation units of the observations and the model as a cache key.

    Parameters
    ----------
    obs_units : hcwf.spatial.units.UnitsSpec or None
        The observation units, None for grid cells.
    model_units : hcwf.spatial.units.UnitsSpec or None
        The model units, None for grid cells.

    Returns
    -------
    dict
        ``None`` for each key of `hcwf.spatial.units.UNITS_KEYS` when both are None (grid cells, as before), else
        the ``repr`` of each.
    """
    if obs_units is None and model_units is None:
        return {key: None for key in UNITS_KEYS}
    return {"observations": repr(obs_units), "model": repr(model_units)}


def _model_thresholds(onset, pending, years, cache_dir, read: ForecastRead, obs_units):
    """Return the model's own climatology of the pending thresholds, read from or written to the cache directory.

    Each verification window of the run (the windows of ``ev``, already shortened to the usable leads) gets its own
    threshold, pooled over the leads in that window (spec section 13); without windows one threshold pools every lead.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.
    pending : list of tuple
        The ``(name, spec)`` of the thresholds still to compute.
    years : tuple of (int, int)
        The first and last threshold year.
    cache_dir : pathlib.Path
        The cache folder.
    read : ForecastRead
        The typed inputs of the forecast reading.
    obs_units : hcwf.spatial.units.UnitsSpec or None
        The observation units, None for grid cells (part of the cache key).

    Returns
    -------
    dict of str to xarray.DataArray
        The thresholds by name, in mm.
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
        """Compute the pending thresholds from the model's forecasts."""
        reduced = replace(onset.definition,
                          thresholds=tuple((n, sp) for n, sp in onset.definition.thresholds if n in names))
        return compute_thresholds_from_forecasts(reduced, _model_forecasts(onset, years, read), years, window, windows)

    store = Store("model_thresholds", cache_dir, codec=NETCDF_GROUPS, label="Model thresholds")
    return store.get_or_compute(parts, compute)


def load_onset_thresholds(onset, obs, area, options, ev, output, *, model=None, select=NO_SELECTION,
                          parallel=None, cache_dir=None):
    """Load or compute the named thresholds of the definition.

    Files given in ``thresholds`` are used as they are; the others are computed from the observations of
    ``threshold_years`` (or the model's forecasts when ``threshold_source`` is ``"model"``) and cached under the
    run's ``cache_dir`` (shared by runs), or the output's ``data_dir/.onset_cache`` when none is set. The returned
    thresholds also carry ``season_climatology`` for derived seasons.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    ev : hcwf.config.model.EvaluationSpec
        The evaluation settings.
    output : hcwf.config.model.OutputSpec
        The output settings (``data_dir``, ``figure_dir``).
    model : hcwf.config.model.ModelSpec, optional
        The forecast model, needed when the thresholds come from it.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    parallel : hcwf.config.model.ParallelSpec, optional
        The parallel settings for reading the observations.
    cache_dir : str or pathlib.Path, optional
        The cache folder; None uses ``<output.data_dir>/.onset_cache``.

    Returns
    -------
    hcwf.config.onset.Thresholds
        The thresholds by name, each a ``(lat, lon)`` array (or a table by unit id) in mm, or a constant.
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
    probe = reference_probe(onset, years[0], read)
    full = LazyRain(probe, lambda: reference_rain(onset, years, read, probe=probe))
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
    """Return text identifying the onset setup the thresholds are computed with.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.

    Returns
    -------
    str
        The ``repr`` of the definition hash and of the settings that change the thresholds.
    """
    return repr((definition_mod.definition_hash(onset.definition), onset.season, onset.threshold_years,
                 onset.threshold_files, onset.threshold_constants, onset.obs_day_boundary, onset.lead_label,
                 onset.threshold_source, onset.forecast_history, onset.dry_check))


def _thresholds_key(onset, obs, options, select, model) -> str:
    """Return what else the loaded thresholds depend on: how they are computed and which observations on which units.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    select : hcwf.config.model.Selection
        The run's selection settings.
    model : hcwf.config.model.ModelSpec or None
        The forecast model, None when there is none.

    Returns
    -------
    str
        The ``repr`` of the settings.
    """
    return repr([("threshold_scale", options.threshold_scale), ("threshold_table", options.threshold_table),
                 ("threshold_max_missing", options.threshold_max_missing),
                 ("threshold_partial_years", options.threshold_partial_years), ("obs_paths", obs.paths),
                 ("obs_files_years", select.obs_files_years), ("obs_dims", obs.dims_map()),
                 ("obs_day_boundary", obs.day_boundary), ("model_lead_label", onset.lead_label),
                 ("ingest", options.ingest(False, obs.units)), ("obs_units", obs.units),
                 ("model_units", None if model is None else model.units)])


def load_thresholds(onset, obs, area, options, ev, output, *, model=None,
                    select=NO_SELECTION, parallel=None, cache_dir=None):
    """Return the definition's named thresholds (what the onset detection takes as ``thresh_slice``).

    ``model`` is the forecast model the thresholds come from when ``threshold_source`` is ``"model"``. Computed ones
    are cached under ``cache_dir``, or the output's data folder; the loaded set is kept for the run (see
    `_loaded_thresholds`) and is the same object on the next call with the same inputs.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup.
    obs : hcwf.config.model.ObsSpec
        The observation dataset settings.
    area : hcwf.config.model.AreaSpec
        The region and mask settings.
    options : hcwf.config.model.OnsetOptions
        The onset options.
    ev : hcwf.config.model.EvaluationSpec
        The evaluation settings.
    output : hcwf.config.model.OutputSpec
        The output settings.
    model : hcwf.config.model.ModelSpec, optional
        The forecast model the thresholds may come from.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.
    parallel : hcwf.config.model.ParallelSpec, optional
        The parallel settings for reading the observations.
    cache_dir : str or pathlib.Path, optional
        The cache folder; None uses the output's data folder.

    Returns
    -------
    hcwf.config.onset.Thresholds
        The thresholds (see `load_onset_thresholds`).
    """
    key = (_setup_key(onset), str(None if model is None else model.name), str(ev.max_forecast_day),
           str(None if not ev.windows else [w.span for w in ev.windows]), str(output.data_dir), str(obs.dir),
           str(obs.var), str(area.region), _thresholds_key(onset, obs, options, select, model))
    return _loaded_thresholds.get_or_make(key, lambda: load_onset_thresholds(
        onset, obs, area, options, ev, output, model=model, select=select, parallel=parallel, cache_dir=cache_dir))


def thresholds_of(rc, model, select=NO_SELECTION):
    """Call `load_thresholds` with the settings of a run for the source ``model``.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration.
    model : hcwf.config.model.ModelSpec
        The forecast source.
    select : hcwf.config.model.Selection, default NO_SELECTION
        The run's selection settings.

    Returns
    -------
    hcwf.config.onset.Thresholds
        The thresholds (see `load_onset_thresholds`).
    """
    return load_thresholds(rc.onset, rc.observations, rc.area, rc.onset_options, rc.evaluation, rc.output, model=model,
                           select=select, parallel=rc.parallel, cache_dir=rc.cache_dir)
