"""Detect onsets on observed rainfall and on deterministic or ensemble model forecasts.

A thin adapter between HCWF's pipeline and `hcwf.onset`: it settles units, day boundaries and thresholds, runs
`hcwf.onset.engine`, and shapes the results into the arrays and DataFrames the metrics code expects.

The functions take the typed inputs of `hcwf.config.model`: the run (``rc``, a `hcwf.config.model.RunConfig`), the
case (source and window) and the per-call ``select`` (a `hcwf.config.model.Selection`). The module reads and writes
the observed-onset cache of `hcwf.onset.observed_cache`, and runs the day-boundary check of
`hcwf.onset.alignment` once per model and run.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import memo
from hcwf.config.model import Selection
from hcwf.io.forecasts import ingest_forecast
from hcwf.config.onset import forecast_reach
from hcwf.onset import engine
from hcwf.onset import observed_cache
from hcwf.onset.alignment import check_forecast_alignment
from hcwf.onset.ingest import ingest
from hcwf.onset.model import DailyRain, Status
from hcwf.onset.thresholds import unit_thresholds
from hcwf.spatial import axes
import logging

log = logging.getLogger(__name__)

_GRID_TOLERANCE = 1e-3
_ALIGNMENT_MAX_INITS = 40
_ALIGNMENT_MAX_MEMBERS = 10
_alignment_done = memo("onset.detect.alignment_done")
"""The (model, observations, lead label) alignment checks made this run: {key: True}."""


def engine_options(rc, forecast=False) -> dict:
    """Return the run options of the engine (horizon and missing-rain policy).

    A value the run does not give is HCWF's own default.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run; its ``onset_options`` and ``observations.units`` are read.
    forecast : bool, default False
        Whether the options are for forecast detection rather than observed detection.

    Returns
    -------
    dict
        The keyword arguments for `hcwf.onset.engine.detect_observed` or `hcwf.onset.engine.detect_forecast`.
    """
    return rc.onset_options.engine(rc.observations.units, forecast)


def _on_grid(da, lat, lon):
    """Put a per-cell array on the lat/lon grid of the rainfall.

    Cells the array lacks become NaN; cells are matched to the nearest grid value within ``1e-3`` degrees.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        The per-cell values; returned unchanged when it has no ``lat`` and ``lon`` dims.
    lat : array_like of float64
        The latitudes of the rainfall grid.
    lon : array_like of float64
        The longitudes of the rainfall grid.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        `da` reindexed to ``lat`` and ``lon``.
    """
    if "lat" not in da.dims or "lon" not in da.dims:
        return da
    return da.reindex(lat=lat, lon=lon, method="nearest", tolerance=_GRID_TOLERANCE)


def _at_cells(da, lat, lon):
    """Put a per-cell array on a list of cells instead of the lat/lon grid.

    Each listed cell takes the grid cell within ``1e-3`` degrees, NaN where there is none (as `_on_grid`).

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        The values on a lat/lon grid; returned unchanged when it has no ``lat`` and ``lon`` dims.
    lat : array_like of float64, shape (cell,)
        The latitudes of the listed cells.
    lon : array_like of float64, shape (cell,)
        The longitudes of the listed cells.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The values with one ``cell`` dimension in place of ``lat`` and ``lon``, which are dropped as coordinates.
    """
    if "lat" not in da.dims or "lon" not in da.dims:
        return da
    rows = da.indexes["lat"].get_indexer(np.asarray(lat), method="nearest", tolerance=_GRID_TOLERANCE)
    cols = da.indexes["lon"].get_indexer(np.asarray(lon), method="nearest", tolerance=_GRID_TOLERANCE)
    out = da.isel(lat=xr.DataArray(np.maximum(rows, 0), dims="cell"), lon=xr.DataArray(np.maximum(cols, 0), dims="cell"))
    out = out.drop_vars(["lat", "lon"], errors="ignore")
    missing = (rows < 0) | (cols < 0)
    return out.where(xr.DataArray(~missing, dims="cell")) if missing.any() else out


def _like(da, coords):
    """Put a per-cell or per-unit array on the spatial axes `coords`.

    Cells are matched within tolerance and units by id; whatever is missing is NaN.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        The values to move.
    coords : dict
        The spatial coordinates of the rainfall: ``unit``, or ``cell`` with ``lat`` and ``lon`` over it, or ``lat`` and
        ``lon`` of a grid.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        `da` on those axes.
    """
    if "unit" in coords:
        return da.reindex(unit=np.asarray(coords["unit"]).astype(str))
    if "cell" in coords:
        return _at_cells(da, coords["lat"], coords["lon"]).assign_coords(cell=np.asarray(coords["cell"]))
    return _on_grid(da, coords["lat"], coords["lon"])


def _on_units(value, ids, convention=None):
    """Build a threshold over the units of the rainfall, joined by unit id.

    Parameters
    ----------
    value : float or pandas.DataFrame or numpy.ndarray
        A number of mm, a table of ``(id, onset_thresh)``, or an array over ``unit``; a table must cover the units (see
        `hcwf.onset.thresholds.unit_thresholds`).
    ids : array_like
        The unit ids of the rainfall.
    convention : str, optional
        The grid-id convention of the units, or None when it is established or inferred.

    Returns
    -------
    xarray.DataArray
        The threshold with dim ``unit`` and the ids (as ``str``) as coordinate.
    """
    return xr.DataArray(unit_thresholds(value, np.asarray(ids).astype(str), convention), dims=("unit",),
                        coords={"unit": np.asarray(ids).astype(str)})


def _field(value, coords, convention=None):
    """Put one threshold on the rainfall's own spatial axes (cells of the grid, or units).

    Parameters
    ----------
    value : float or xarray.DataArray or pandas.DataFrame
        The threshold; a number is returned as it is (unless the rainfall is over units).
    coords : dict
        The spatial coordinates of the rainfall (see `_like`).
    convention : str, optional
        The grid-id convention of the units.

    Returns
    -------
    float or xarray.DataArray
        The threshold on the axes of `coords`.
    """
    if "unit" in coords:
        return _on_units(value, coords["unit"], convention)
    if not isinstance(value, xr.DataArray):
        return value
    return _like(value, coords)


def _thresholds(thresh_slice, coords, per_window=True, convention=None):
    """Build the thresholds for the engine.

    Model-climatology thresholds carry a ``window`` dim (one per verification window). Forecast detection keeps it;
    ``per_window=False`` (observed onsets, which have no lead) keeps the first window.

    Parameters
    ----------
    thresh_slice : mapping or None
        The named thresholds from `hcwf.io.input.load_thresholds`; None or empty gives no thresholds.
    coords : dict
        The rainfall's spatial coordinates (``lat`` and ``lon``, or ``unit``).
    per_window : bool, default True
        Keep the ``window`` dim; False takes its first window.
    convention : str, optional
        The grid-id convention of the units.

    Returns
    -------
    dict
        The named thresholds on the rainfall's cells or units, or a fixed amount.
    """
    out = {name: _field(value, coords, convention) for name, value in dict(thresh_slice or {}).items()}
    if not per_window:
        out = {name: (value.isel(window=0, drop=True) if "window" in getattr(value, "dims", ()) else value)
               for name, value in out.items()}
    return out


def _season_climatology(setup, thresh_slice, coords):
    """Return the harmonic season analysis on the rainfall grid, for derived seasons only.

    Parameters
    ----------
    setup : hcwf.config.onset.OnsetSetup
        The onset setup; its ``derived_seasons`` flag decides whether the analysis is needed.
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds`; its ``season_climatology`` attribute is read.
    coords : dict
        The rainfall's spatial coordinates (see `_like`).

    Returns
    -------
    xarray.Dataset or xarray.DataArray or None
        The season climatology on the rainfall grid, or None when the definition has no derived seasons.

    Raises
    ------
    ValueError
        The definition has derived seasons but no season climatology was loaded, or the rainfall is over units.
    """
    if not setup.derived_seasons:
        return None
    climatology = getattr(thresh_slice, "season_climatology", None)
    if climatology is None:
        raise ValueError("this definition has derived seasons but no season climatology was loaded; "
                         "pass the thresholds returned by hcwf.io.input.load_thresholds")
    if "lat" not in coords:
        raise ValueError("derived seasons are analysed on a grid; evaluation units do not support them")
    return _like(climatology, coords)


def unit_convention(rc):
    """Return the grid-id convention of the run's units.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run; ``observations.units`` is read.

    Returns
    -------
    str or None
        The convention, or None when it is established or inferred.
    """
    spec = rc.observations.units
    return None if spec is None else spec.convention


def detect_observed_dataset(rain_slice, thresh_slice, year, rc, select=Selection()):
    """Run the engine on one year of observed rainfall.

    The Dataset has dims ``(year, season, *spatial)`` (``lat`` and ``lon``, or ``unit``) with one year and the selected
    season (both kept as length-1 dims) and the variables ``onset_date``, ``onset_index`` and ``status`` of
    `hcwf.onset.engine.detect_observed`; its ``calendar`` attribute is the calendar of the dates. `detect_observed`
    and the NetCDF export (`hcwf.runs.export`) both read their values from here, so they cannot disagree.

    Parameters
    ----------
    rain_slice : xarray.DataArray or hcwf.onset.model.DailyRain
        Daily rainfall in mm with dims ``(time, lat, lon)`` or ``(time, unit)``, or that rainfall already ingested.
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds`.
    year : int
        The year to detect.
    rc : hcwf.config.model.RunConfig
        The run.
    select : hcwf.config.model.Selection, default Selection()
        The per-call selection; a ``reference_onset`` (a ``ReferenceOnset``, for evaluation units) starts each search no
        earlier than that date.

    Returns
    -------
    tuple of (xarray.Dataset, hcwf.config.onset.OnsetSetup)
        The engine Dataset and the onset setup of the run.
    """
    setup = rc.onset
    rain = rain_slice if isinstance(rain_slice, DailyRain) else ingest(
        rain_slice, day_boundary=setup.obs_day_boundary, units="mm", source="observations",
        **rc.onset_options.ingest(False, rc.observations.units))
    coords = {d: rain.coords[d] for d in rain.dims[:-1]}
    reference = select.reference_onset
    search = None if reference is None else {"reference": reference.search_start([int(year)], coords["unit"])}
    result = engine.detect_observed(
        setup.definition, rain, thresholds=_thresholds(thresh_slice, coords, per_window=False,
                                                       convention=unit_convention(rc)), years=[year],
        season_climatology=_season_climatology(setup, thresh_slice, coords), backend=setup.backend,
        search_start=search, **engine_options(rc))
    if search:
        result = result.isel(start=1)
    result = result.sel(year=[year], season=[setup.season])
    result.attrs["calendar"] = rain.calendar
    return result, setup


def detect_observed(rain_slice, thresh_slice, year, rc, select=Selection()):
    """Detect the observed onset dates for one year.

    `rain_slice` is daily rainfall in mm with dims ``(time, lat, lon)``, or ``(time, unit)`` for evaluation units (the
    reader's ``present`` coordinate says which units have a series), or that rainfall already ingested as a
    `hcwf.onset.model.DailyRain` (the climatology years the thresholds were computed from), or a function that reads
    it, called only when the onsets are not cached (for cells the function needs a ``probe`` attribute whose ``coords``
    hold the lat and lon: the cache's fingerprint reads the grid from it instead of from the rain).

    While a run has the observed-onset cache active (`hcwf.onset.observed_cache`) the dates of a year already detected
    under the same fingerprint are returned without detecting again.

    Parameters
    ----------
    rain_slice : xarray.DataArray or hcwf.onset.model.DailyRain or callable
        The rainfall, as described above; a callable takes no arguments.
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds` (the definition's named thresholds).
    year : int
        The year to detect.
    rc : hcwf.config.model.RunConfig
        The run.
    select : hcwf.config.model.Selection, default Selection()
        The per-call selection (see `detect_observed_dataset`).

    Returns
    -------
    xarray.DataArray of datetime64[ns]
        The onset dates over ``(lat, lon)`` or ``(unit)``, ``NaT`` where there is no onset. For units it carries the
        coordinates ``present`` and ``onset_index`` (the 1-based day of the onset in the year's series, NaN without
        one).
    """
    reader = rain_slice if callable(rain_slice) else None
    cached = observed_cache.lookup(getattr(reader, "probe", None) if reader else rain_slice, thresh_slice, year, rc, select)
    if cached is not None:
        return cached
    if reader is not None:
        rain_slice = reader()
    result, setup = detect_observed_dataset(rain_slice, thresh_slice, year, rc, select)
    spatial = ("lat", "lon") if "lat" in result["onset_date"].dims else tuple(result["onset_date"].dims[2:])
    onset = result["onset_date"].isel(year=0, season=0).transpose(*spatial)
    status = result["status"].isel(year=0, season=0).transpose(*spatial)
    counts = {s.name: int((status == s).sum()) for s in Status}
    log.debug(f"Observed onset {year}, cells by status: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    values = onset.values.astype("datetime64[ns]")
    out = xr.DataArray(values, coords=[(d, result[d].values) for d in spatial], name="onset_date")
    if isinstance(rain_slice, xr.DataArray) and "present" in rain_slice.coords:
        present = np.asarray(rain_slice["present"].values, dtype=bool)
        found = (status.values == int(Status.ONSET)) & present
        index = result["onset_index"].isel(year=0, season=0).transpose(*spatial).values
        out = out.copy(data=np.where(found, values, np.datetime64("NaT"))).assign_coords(
            present=(spatial, present), onset_index=(spatial, np.where(found, index + 1, np.nan)))
    observed_cache.store(rain_slice, thresh_slice, year, rc, select, out)
    return out


def detect_forecast(p_model, thresh_slice, onset_da, rc, case, select=Selection(), *, members=None, history=None,
                    search_start=None, restrict_to_season=True, onset_count=False):
    """Run the engine on a forecast cube and line it up with the observed onsets.

    The result is a dict of numpy arrays shaped ``(init_time, [member], lat, lon)``: the lead day after init of the
    onset (``"day"``: NaN for none; 0 or below for an onset back-dated by the observed history), the date and the
    status; plus the valid-case mask (init before the observed onset) and the observed onset dates ``(init_time, lat,
    lon)``. ``"result"`` is the engine Dataset these arrays come from (season kept as a length-1 dim, with the ensemble
    summary for members) and ``"calendar"`` the calendar of its dates; the NetCDF export (`hcwf.runs.export`) reads
    them.

    The cube may hold only some cells of the grid (the loader's ``cells``: ``(init_time, [member], step, cell)``, with
    ``lat`` and ``lon`` as coordinates over ``cell``): only those are ingested and detected, and the arrays and the
    ``"place"`` function that names a point's columns (``lat`` and ``lon``) are over them. `p_model` may also be a
    function that reads the cube; it is called here, and the cube is dropped once ingested (peak memory: the cube and
    its tenths, not the cube through the detection).

    The cube may be over evaluation units instead of ``lat`` and ``lon`` (``(init_time, member, unit, step)``).
    `search_start` (variants that start the search later, see `hcwf.onset.engine.detect_forecast`),
    `restrict_to_season` (False: one season covering every lead) and `onset_count` (the member onset counts of
    `count_members`) may then be given; with `onset_count` there is no observed onset to line up with (`onset_da` is
    None) and the dict holds ``"counts"`` ``(init_time, start, unit, lead)`` instead of the day, date and status
    arrays.

    Parameters
    ----------
    p_model : xarray.DataArray or callable
        The forecast cube in mm, or a function with no arguments that reads it.
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds`.
    onset_da : xarray.DataArray or None
        The observed onset dates (`detect_observed`); None with `onset_count`.
    rc : hcwf.config.model.RunConfig
        The run.
    case : object
        The case (``case.model`` is the forecast model's settings).
    select : hcwf.config.model.Selection, default Selection()
        The per-call selection.
    members : sequence, optional
        The members to keep; None keeps the cube's.
    history : xarray.DataArray, optional
        Observed rainfall before the init, used when the run's ``forecast_history`` is ``"observed"``.
    search_start : dict, optional
        Variants that start the search later, passed to the engine.
    restrict_to_season : bool, default True
        False gives one season covering every lead.
    onset_count : bool, default False
        Return the member onset counts instead of the onset arrays.

    Returns
    -------
    dict
        ``"setup"``, ``"result"``, ``"calendar"`` and ``"init"`` always. With `onset_count`: ``"counts"`` and
        ``"start"`` (the names of the search starts). Otherwise ``"spatial"`` (the spatial dims), ``"axes"``,
        ``"place"`` (a function from an index tuple to the point's columns), ``"member"`` (or None), ``"day"``,
        ``"date"``, ``"status"``, ``"valid"``, ``"obs"`` (observed dates, ``datetime64[ns]``) and ``"totals"`` (potential
        initializations, cases with an observed onset).
    """
    setup = rc.onset
    max_forecast_day = rc.evaluation.max_forecast_day
    steps_needed = max_forecast_day + forecast_reach(setup) - 1 if isinstance(max_forecast_day, int) else None
    cube = (p_model() if callable(p_model) else p_model).sel(step=slice(1, steps_needed))
    p_model = None
    if "member" in cube.dims and members and list(cube["member"].values) != list(members):
        cube = cube.sel(member=list(members))  # the loader already took them; selecting again would copy the cube
    cells = "cell" in cube.dims
    spatial = ("cell",) if cells else axes.spatial_dims(cube.dims)
    order = ("init_time",) + (("member",) if "member" in cube.dims else ()) + spatial + ("step",)
    forecast = ingest_forecast(cube.transpose(*order), setup, case.model, rc.onset_options, select, "forecast")
    coords = {d: forecast.coords[d] for d in spatial}
    if cells:
        cell_lat, cell_lon = np.asarray(cube["lat"].values), np.asarray(cube["lon"].values)
        coords = {**coords, "lat": cell_lat, "lon": cell_lon}
    cube = None
    stitched = None
    if history is not None and setup.forecast_history == "observed":
        on_grid = _like(history, coords).transpose(*spatial, ...)
        stitched = ingest(on_grid, day_boundary=setup.obs_day_boundary, units="mm", source="observed history")
    count = bool(onset_count)
    result = engine.detect_forecast(
        setup.definition, forecast, thresholds=_thresholds(thresh_slice, coords, convention=unit_convention(rc)),
        lead_window=(1, max_forecast_day), backend=setup.backend, history=stitched,
        season_climatology=_season_climatology(setup, thresh_slice, coords),
        restrict_to_season=restrict_to_season, search_start=search_start,
        onset_count=count, **engine_options(rc, forecast=True))
    season = setup.season
    if not restrict_to_season:   # one season covering every lead, named for itself
        season = result["season"].values[0]
    export_result = result.sel(season=[season])
    result = result.sel(season=season)
    if not count and bool((result["status"] == Status.CENSORED).any()):
        warnings.warn("some forecast searches are CENSORED although the lead window was shortened to fit the rule; "
                      "they are reported as no onset. Is max_forecast_day larger than the loaded leads allow?",
                      UserWarning)
    init = np.asarray(forecast.coords["init_time"])
    if count:
        counts = result["onset_count"].transpose("init_time", "start", *spatial, "lead")
        return {"setup": setup, "result": export_result, "calendar": forecast.calendar, "init": init,
                "counts": counts, "start": [str(s) for s in counts["start"].values]}
    lead_dims = (("init_time", "member") if "member" in forecast.dims else ("init_time",)) + spatial
    dates = result["onset_date"].transpose(*lead_dims).values
    status = result["status"].transpose(*lead_dims).values
    obs = (onset_da.reindex(unit=np.asarray(coords["unit"]).astype(str)) if "unit" in coords else
           _like(onset_da, coords)).transpose(*spatial)
    obs_values = obs.values.astype("datetime64[ns]")
    valid = ~np.isnat(obs_values)[None] & (axes.broadcast_rows(init.astype("datetime64[ns]"), obs_values.ndim + 1)
                                           < obs_values[None])
    found = status == Status.ONSET
    init_day = init.astype("datetime64[D]").reshape((-1,) + (1,) * (dates.ndim - 1))
    days_after_init = (dates.astype("datetime64[D]") - init_day) / np.timedelta64(1, "D")
    return {"setup": setup, "result": export_result, "calendar": forecast.calendar, "spatial": spatial,
            "init": init, "axes": {d: np.asarray(coords[d]) for d in spatial},
            "place": (lambda index: {"lat": cell_lat[index[0]], "lon": cell_lon[index[0]]}) if cells else
            (lambda index: axes.columns_of(spatial, {d: np.asarray(coords[d]) for d in spatial}, index)),
            "member": np.asarray(forecast.coords["member"]) if "member" in forecast.dims else None,
            "day": np.where(found, days_after_init, np.nan), "date": dates, "status": status,
            "valid": valid, "obs": obs_values, "totals": (int(obs.size * len(init)), int((~np.isnat(obs_values)).sum()))}


def count_members(cube, thresh_slice, rc, case, select=Selection(), *, search_start=None):
    """Count the members whose onset falls on each lead day, for an ensemble forecast on evaluation units.

    `cube` has dims ``(init_time, member, unit, step)`` in mm, its steps 1..n, with n the run's ``max_forecast_day``. The
    counts are the number of members whose onset falls on each lead day 0..n, from the search that starts at lead 1
    (``"base"`` first) and from each variant of `search_start` (the fixed cutoff date, the reference onset); the start
    names list them in that order.

    Parameters
    ----------
    cube : xarray.DataArray
        The forecast cube in mm.
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds`.
    rc : hcwf.config.model.RunConfig
        The run.
    case : object
        The case (see `detect_forecast`).
    select : hcwf.config.model.Selection, default Selection()
        The per-call selection.
    search_start : dict, optional
        Variants that start the search later, passed to the engine.

    Returns
    -------
    tuple of (xarray.DataArray, list of str, numpy.ndarray)
        The counts with dims ``(init_time, start, unit, lead)``, the names of the starts, and the init times.
    """
    out = detect_forecast(cube, thresh_slice, None, rc, case, select, search_start=search_start, onset_count=True,
                          restrict_to_season=False)
    return out["counts"], out["start"], out["init"]


def _days(values):
    """Cut dates to whole days, the dates of the onset tables.

    Parameters
    ----------
    values : array_like of datetime64
        The dates; ``NaT`` stays ``NaT``.

    Returns
    -------
    numpy.ndarray of datetime64[ns]
        The dates at midnight.
    """
    return np.asarray(values).astype("datetime64[D]").astype("datetime64[ns]")


def _print_back_dated(day, valid):
    """Log how many scored forecast onsets lie on or before the init (back-dated by the observed history).

    Parameters
    ----------
    day : numpy.ndarray of float64
        The onset lead day, shaped ``(init_time, [member], ...)``, NaN for none.
    valid : numpy.ndarray of bool
        The valid-case mask, shaped ``(init_time, ...)``.
    """
    mask = np.broadcast_to(valid[:, None] if day.ndim == valid.ndim + 1 else valid, day.shape)
    count = int((day[mask] <= 0).sum())
    if count:
        log.debug(f"Back-dated forecast onsets (onset_day <= 0, scored as documented in docs/onset_in_hcwf.md): {count}")


def _print_status_counts(status, valid):
    """Log how many forecast searches ended in each status, and warn about PRIOR ones.

    Parameters
    ----------
    status : numpy.ndarray of int
        The `hcwf.onset.model.Status` codes, shaped ``(init_time, [member], ...)``.
    valid : numpy.ndarray of bool
        The valid-case mask, shaped ``(init_time, ...)``; only valid cases are counted.
    """
    mask = np.broadcast_to(valid[:, None] if status.ndim == valid.ndim + 1 else valid, status.shape)
    counts = np.bincount(status[mask].astype(np.int64), minlength=len(Status))
    log.debug("Forecast onset searches by status: "
          + ", ".join(f"{s.name} {int(counts[s])}" for s in Status if counts[s]))
    if counts[Status.PRIOR]:
        warnings.warn(f"{int(counts[Status.PRIOR])} forecast searches before the observed onset are PRIOR; the "
                      "observed history confirmed an onset that the observed onset date does not show. They are "
                      "reported as no onset.", UserWarning)


def _names(status):
    """Convert an array of status codes to status names.

    Parameters
    ----------
    status : array_like of int
        The `hcwf.onset.model.Status` codes.

    Returns
    -------
    numpy.ndarray of str
        The names, in the shape of `status`.
    """
    labels = np.array([s.name for s in Status])
    return labels[np.asarray(status, dtype=np.int64)]


def detect_deterministic(p_model, thresh_slice, onset_da, rc, case, select=Selection(), history=None):
    """Compute the onset dates of a deterministic model forecast.

    Parameters
    ----------
    p_model : xarray.DataArray or callable
        The forecast in mm with dims ``(init_time, lat, lon, step)``, or ``(init_time, unit, step)`` on evaluation units
        (or the cells of the loader's ``cells``, or a function that reads the cube: see `detect_forecast`).
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds`.
    onset_da : xarray.DataArray
        The observed onset dates (`detect_observed`).
    rc : hcwf.config.model.RunConfig
        The run.
    case : object
        The case (see `detect_forecast`).
    select : hcwf.config.model.Selection, default Selection()
        The per-call selection.
    history : xarray.DataArray, optional
        Observed rainfall before the init (see `detect_forecast`).

    Returns
    -------
    pandas.DataFrame
        One row per ``(init, cell or unit)`` in that order, for those with an observed onset and an init before it, with
        columns ``init_time``, ``lat`` and ``lon`` (or ``unit``), ``onset_day`` (lead day, None if no onset),
        ``onset_date`` (``datetime64[ns]`` days, ``NaT`` if none), ``obs_onset_date`` and ``onset_status``. Empty when
        there is no valid case.
    """
    out = detect_forecast(p_model, thresh_slice, onset_da, rc, case, select, history=history)
    valid = out["valid"]
    spatial = out["spatial"]
    log.debug(f"Processing {len(out['init'])} init times x {valid[0].size} {'/'.join(spatial)} points...")
    i, *where = np.nonzero(valid)
    at = (i, *where)
    day = out["day"][at]
    has = ~np.isnan(day)
    onset_date = np.where(has, _days(out["date"][at]), np.datetime64("NaT", "ns"))
    columns = {
        "init_time": out["init"][i], **out["place"](where),
        "onset_day": [int(d) if h else None for d, h in zip(day, has)],
        "onset_date": onset_date,
        "obs_onset_date": _days(out["obs"][tuple(where)]),
    }
    columns["onset_status"] = _names(out["status"][at]).tolist()
    onset_df = pd.DataFrame(columns) if len(i) else pd.DataFrame([])
    total, with_obs = out["totals"]
    log.debug("\nProcessing Summary:")
    log.debug(f"Total potential initializations: {total}")
    log.debug(f"Skipped (no observed onset): {total - with_obs * len(out['init'])}")
    log.debug(f"Skipped (initialized after observed onset): {with_obs * len(out['init']) - len(i)}")
    log.debug(f"Valid initializations processed: {len(i)}")
    log.debug(f"Onsets found: {int(has.sum())}")
    log.debug(f"Onset rate: {has.sum() / len(i):.3f}" if len(i) else "Onset rate: 0.000")
    _print_status_counts(out["status"], valid)
    _print_back_dated(out["day"], valid)
    return onset_df


def _ensemble_status(status, threshold, has_onset):
    """Give the status of an ensemble onset.

    The status is ``ONSET`` if the ensemble has an onset, else the most informative way the members fell short: the
    code (``INSUFFICIENT_DATA`` or ``CENSORED``) that together with the onsets reaches the member fraction, else
    ``NOT_FOUND``.

    Parameters
    ----------
    status : numpy.ndarray of int, shape (init_time, member, ...)
        The `hcwf.onset.model.Status` codes of the members.
    threshold : float
        The fraction of members needed.
    has_onset : numpy.ndarray of bool, shape (init_time, ...)
        Whether the ensemble has an onset.

    Returns
    -------
    numpy.ndarray of object, shape of `has_onset`
        The status names.
    """
    names = np.full(has_onset.shape, Status.NOT_FOUND.name, dtype=object)
    total = status.shape[1]
    for code in (Status.INSUFFICIENT_DATA, Status.CENSORED):
        reach = ((status == Status.ONSET).sum(axis=1) + (status == code).sum(axis=1)) / total >= threshold
        names[reach] = code.name
    names[has_onset] = Status.ONSET.name
    return names


def ensemble_onset(day, init, threshold):
    """Compute the ensemble onset HCWF scores from the member onset days.

    The ensemble has an onset when the fraction of members with one is at least `threshold`; its day (days after the
    init) is the mean member day rounded half to even.

    Parameters
    ----------
    day : numpy.ndarray of float64, shape (init_time, member, lat, lon) or (init_time, member, unit)
        The member onset day, NaN for none.
    init : array_like of datetime64
        The init dates.
    threshold : float
        The least fraction of members with an onset for the ensemble to have one.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The count of members with an onset, their fraction, the ensemble onset day (NaN for none) and the ensemble onset
        date, each shaped ``(init_time, lat, lon)`` or ``(init_time, unit)``. The date is only meaningful where the day
        is not NaN.
    """
    n_member = day.shape[1]
    count = (~np.isnan(day)).sum(axis=1)
    fraction = count / n_member if n_member > 0 else np.zeros(count.shape)
    with np.errstate(invalid="ignore"):
        mean_day = np.nanmean(day, axis=1) if n_member else np.full(count.shape, np.nan)
    ensemble_day = np.where(fraction >= threshold, np.round(mean_day), np.nan)
    ensemble_date = axes.broadcast_rows(pd.DatetimeIndex(init).values, ensemble_day.ndim) + (
        np.nan_to_num(ensemble_day, nan=0).astype("int64").astype("timedelta64[D]"))
    return count, fraction, ensemble_day, ensemble_date


def detect_ensemble(p_model, thresh_slice, onset_da, rc, case, select=Selection(), history=None):
    """Compute the per-member and the ensemble onsets.

    The ensemble has an onset when the fraction of members with one is at least ``onset_percentage_threshold``; its day
    is the mean member onset day rounded half to even.

    Parameters
    ----------
    p_model : xarray.DataArray or callable
        The forecast in mm with dims ``(init_time, member, lat, lon, step)``, or ``(init_time, member, unit, step)`` on
        evaluation units (or the cells of the loader's ``cells``, or a function that reads the cube: see
        `detect_forecast`).
    thresh_slice : object
        The thresholds from `hcwf.io.input.load_thresholds`.
    onset_da : xarray.DataArray
        The observed onset dates (`detect_observed`).
    rc : hcwf.config.model.RunConfig
        The run.
    case : object
        The case (see `detect_forecast`).
    select : hcwf.config.model.Selection, default Selection()
        The per-call selection.
    history : xarray.DataArray, optional
        Observed rainfall before the init (see `detect_forecast`).

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The member onset table and the ensemble onset table; their date columns are ``datetime64[ns]`` days (``NaT``:
        none).
    """
    members = rc.evaluation.members
    threshold = rc.onset_options.percentage_threshold
    out = detect_forecast(p_model, thresh_slice, onset_da, rc, case, select, members=members, history=history)
    valid, day = out["valid"], out["day"]
    n_init, n_member = day.shape[0], day.shape[1]
    log.debug(f"Processing {n_init} init times x {valid[0].size} unique locations x {n_member} members...")

    member_day = day
    count, fraction, ensemble_day, ensemble_date = ensemble_onset(day, out["init"], threshold)

    grid = valid.shape
    flat_valid = np.broadcast_to(valid[:, None], (grid[0], n_member) + grid[1:])
    member_rows = tuple(np.nonzero(flat_valid))
    member_index = pd.Index(np.flatnonzero(flat_valid.ravel()))
    ens_rows = tuple(np.nonzero(valid))
    ens_index = pd.Index(np.flatnonzero(valid.ravel()))
    mi, mm, *member_where = member_rows
    ei, *ens_where = ens_rows
    member_at, ens_at = (mi, mm, *member_where), (ei, *ens_where)

    member_values = member_day[member_at]
    onset_df = pd.DataFrame({
        "init_time": out["init"][mi], **out["place"](member_where),
        "member": out["member"][mm],
        "onset_day": member_values,
        "obs_onset_date": _days(out["obs"][tuple(member_where)]),
    }, index=member_index)
    chosen = ensemble_day[ens_at]
    has_onset = ~np.isnan(chosen)
    ensemble_df = pd.DataFrame({
        "init_time": out["init"][ei], **out["place"](ens_where),
        "onset_day": chosen,
        "onset_date": np.where(has_onset, _days(ensemble_date[ens_at]), np.datetime64("NaT", "ns")),
        "member_onset_count": count[ens_at], "total_members": n_member,
        "onset_percentage": fraction[ens_at],
        "obs_onset_date": _days(out["obs"][tuple(ens_where)]),
    }, index=ens_index)
    onset_df["onset_status"] = _names(out["status"][member_at])
    ensemble_df["onset_status"] = _ensemble_status(out["status"][(ei, slice(None), *ens_where)], threshold, has_onset)

    log.debug("\nProcessing Summary:")
    log.debug(f"Generated {len(onset_df)} member-forecast combinations")
    log.debug(f"Found onset in {int(np.sum(~np.isnan(member_values)))} cases")
    log.debug(f"Onset rate: {np.mean(~np.isnan(member_values)):.3f}" if len(member_values) else "Onset rate: nan")
    log.debug(f"Ensemble onsets found (>= {threshold:.0%} of members): {int(has_onset.sum())}")
    log.debug(f"Ensemble onset rate: {has_onset.mean():.3f}" if len(has_onset) else "Ensemble onset rate: 0.000")
    _print_status_counts(out["status"], valid)
    _print_back_dated(member_day, valid)
    return onset_df, ensemble_df


def _cells_on_grid(cube, lat, lon):
    """Put a cube of cells on a lat/lon grid.

    Parameters
    ----------
    cube : xarray.DataArray
        The values, with other dims and ``cell``, and ``lat`` and ``lon`` as coordinates over the cells.
    lat : array_like of float64
        The latitudes of the grid.
    lon : array_like of float64
        The longitudes of the grid.

    Returns
    -------
    xarray.DataArray
        The cube with dims ``(*others, lat, lon)``, NaN where the grid cell has no cell of the cube; cells are matched
        within ``1e-3`` degrees.
    """
    rows = pd.Index(lat).get_indexer(np.asarray(cube["lat"].values), method="nearest", tolerance=_GRID_TOLERANCE)
    cols = pd.Index(lon).get_indexer(np.asarray(cube["lon"].values), method="nearest", tolerance=_GRID_TOLERANCE)
    found = (rows >= 0) & (cols >= 0)
    others = [d for d in cube.dims if d != "cell"]
    data = np.full(tuple(cube.sizes[d] for d in others) + (len(lat), len(lon)), np.nan, dtype=cube.dtype)
    data[..., rows[found], cols[found]] = cube.transpose(*others, "cell").values[..., found]
    coords = {d: c for d, c in cube.coords.items() if "cell" not in c.dims}
    return xr.DataArray(data, dims=(*others, "lat", "lon"), coords={**coords, "lat": lat, "lon": lon}, attrs=cube.attrs)


def alignment_report(rain_slice, p_model, rc, case):
    """Compare observations with forecast leads 1-3.

    Subsamples the inits and members, because `hcwf.onset.alignment` is expensive on whole years. Never raises: a
    failure is returned as a note.

    Parameters
    ----------
    rain_slice : xarray.DataArray
        The observed daily rainfall in mm with dims ``(time, lat, lon)``.
    p_model : xarray.DataArray
        The forecast cube in mm.
    rc : hcwf.config.model.RunConfig
        The run.
    case : object
        The case (``case.model`` is the forecast model's settings).

    Returns
    -------
    tuple of (hcwf.onset.alignment.ForecastAlignmentReport or None, str)
        The report (None when the check could not run) and a note that says why it did not (empty when it ran).
    """
    setup = rc.onset
    try:
        cube = p_model.sel(step=slice(1, 3))
        stride = max(1, -(-cube.sizes["init_time"] // _ALIGNMENT_MAX_INITS))
        cube = cube.isel(init_time=slice(None, None, stride))
        if "member" in cube.dims:
            cube = cube.isel(member=slice(0, _ALIGNMENT_MAX_MEMBERS))
        if "cell" in cube.dims:
            cube = _cells_on_grid(cube, rain_slice["lat"].values, rain_slice["lon"].values)
        obs = ingest(rain_slice, day_boundary=setup.obs_day_boundary, units="mm", source="observations")
        forecast = ingest_forecast(cube, setup, case.model, rc.onset_options, Selection(),
                                   str(case.model.name or "forecast"))
        return check_forecast_alignment(obs, forecast, leads=(1, 2, 3)), ""
    except Exception as exc:  # noqa: BLE001 - a failed check is reported, not raised
        return None, f"could not run: {exc}"


def check_alignment(rain_slice, p_model, rc, case):
    """Compare observations with forecast leads 1-3 once per model and run, and warn if they look shifted.

    Skipped (with a note in the log) when ``check_alignment`` is off or the observation day boundary is unknown. Never
    raises: a failed check is reported and the run goes on.

    Parameters
    ----------
    rain_slice : xarray.DataArray
        The observed daily rainfall in mm with dims ``(time, lat, lon)``.
    p_model : xarray.DataArray
        The forecast cube in mm.
    rc : hcwf.config.model.RunConfig
        The run.
    case : object
        The case (``case.model`` is the forecast model's settings).

    Returns
    -------
    None
        A `UserWarning` is issued when the forecasts look shifted against the observations.
    """
    setup = rc.onset
    key = (case.model.name, rc.observations.dir, setup.lead_label)
    if key in _alignment_done or not setup.check_alignment:
        return
    _alignment_done[key] = True
    report, note = alignment_report(rain_slice, p_model, rc, case)
    if report is None:
        log.info(f"Alignment check skipped: {note}")
        return
    log.info(f"Alignment check, {case.model.name} against observations:")
    log.info(report.summary())
    if report.verdict.startswith("likely shifted"):
        warnings.warn(f"{case.model.name} forecasts look shifted against the observations "
                      f"({report.verdict}); check obs_day_boundary and model_lead_label.", UserWarning)

