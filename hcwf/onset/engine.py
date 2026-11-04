"""Turn seasons and forecast leads into onset jobs, run a backend, and package the results as xarray Datasets.

This module is the bridge between the data containers of `hcwf.onset.model` (`~hcwf.onset.model.DailyRain`, `~hcwf.onset.model.ForecastRain`) and the
detection backends in `hcwf.onset.backends`. The public entry points are `detect_observed` (one search per year,
season and cell of observed rain) and `detect_forecast` (one search per init, member, season and cell of a forecast,
optionally stitched onto an observed history). Each builds the jobs (`build_observed_jobs`, `build_forecast_jobs`,
`build_stitched_jobs`), compiles the definition with `hcwf.onset.plan.compile`, runs the chosen backend in chunks of
series, and returns an `xarray.Dataset` of onset dates or leads and `~hcwf.onset.model.Status` codes. Season bounds come from
`hcwf.onset.calendar`.
"""
from __future__ import annotations

import datetime as _dt
import importlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr

from hcwf.onset import calendar as cal
from hcwf.onset import plan as plan_mod
from hcwf.onset.model import (
    DEFAULT_HORIZON, MISSING_POLICIES, DailyRain, Definition, DerivedSeasons, ForecastRain, Jobs, Status, StitchedJobs, resolution_of)

NO_PRESET = -1
"""The ``preset`` value of a job that still needs a search (a `~hcwf.onset.model.Status` code is never negative)."""
MAX_TABLE_CELLS = 20_000_000
"""Series x time cells whose prefix tables are built at one time."""

_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
"""Calendar names handled with ``datetime64`` arithmetic."""
_BACKEND_MODULES = {"reference": "reference", "numba": "numba_cpu", "gpu": "gpu"}
"""Module name under ``hcwf.onset.backends`` of each backend name."""


@dataclass
class JobLayout:
    """How a flat `~hcwf.onset.model.Jobs` array maps back to output dims.

    The job array reshapes to ``shape + (n_series,)``.
    """

    shape: tuple[int, ...]
    """The outer shape of the jobs."""
    preset: np.ndarray
    """A `~hcwf.onset.model.Status` code per job decided without a search (``NO_PRESET`` otherwise), int8 of shape (job,)."""
    season_names: list[str]
    """The labels of the season axis."""
    n_series: int
    """The number of backend series."""


@dataclass
class StitchedLayout(JobLayout):
    """The `JobLayout` of stitched jobs, which are ordered (season, init, [member], cell).

    ``n_series = n_init * n_member * n_cell``.
    """

    n_init: int = 0
    """The number of inits."""
    n_member: int = 1
    """The number of members (1 without members)."""
    n_cell: int = 0
    """The number of cells."""


def _load_backend(name: str):
    """Import the backend module for a backend name.

    Parameters
    ----------
    name : {"reference", "numba", "gpu"}
        The backend; the module is ``hcwf.onset.backends.reference``, ``.numba_cpu`` or ``.gpu``.

    Returns
    -------
    module
        The imported backend module, which provides ``detect`` and ``detect_stitched``.

    Raises
    ------
    ValueError
        ``name`` is not a known backend.
    ImportError
        The backend module cannot be imported (for example the GPU stack is missing).
    """
    if name not in _BACKEND_MODULES:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(_BACKEND_MODULES)}")
    module = f"hcwf.onset.backends.{_BACKEND_MODULES[name]}"
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"backend {name!r} is not available ({module}): {exc}") from exc


def _threshold_windows(plan: plan_mod.Plan, thresholds, threshold_windows) -> np.ndarray | None:
    """Return the window start leads for thresholds that have a ``"window"`` dim.

    ``threshold_windows`` overrides the starts; otherwise they come from the ``window_first`` coordinate that
    `~hcwf.onset.climatology.compute_thresholds_from_forecasts` writes.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled plan; its amount specs say which thresholds are used.
    thresholds : dict of str to (xarray.DataArray or numpy.ndarray or float), optional
        The threshold values by name, in mm; a `xarray.DataArray` may have a ``"window"`` dim.
    threshold_windows : sequence of int, optional
        One start lead per window, overriding the ``window_first`` coordinate.

    Returns
    -------
    numpy.ndarray of int64, shape (window,) or None
        The start lead of each window, or ``None`` when no threshold has a ``"window"`` dim.

    Raises
    ------
    ValueError
        A windowed threshold has no start leads, or their number differs from its ``"window"`` size.
    """
    starts = None if threshold_windows is None else np.asarray(threshold_windows, dtype=np.int64)
    for spec in plan.amount_specs:
        value = (thresholds or {}).get(spec.name) if spec.kind != "fixed" else None
        if isinstance(value, xr.DataArray) and "window" in value.dims:
            found = starts if starts is not None else (
                value["window_first"].values.astype(np.int64) if "window_first" in value.coords else None)
            if found is None or len(found) != value.sizes["window"]:
                raise ValueError("windowed thresholds need threshold_windows (one start lead per window)")
            return found
    return None


def _series_thresholds(plan: plan_mod.Plan, thresholds, spatial_dims: tuple[str, ...],
                       spatial_shape: tuple[int, ...], series_shape: tuple[int, ...],
                       first_window_only: bool = False) -> dict[str, np.ndarray]:
    """Return per-series float arrays for the thresholds the plan needs, broadcast from per-cell values.

    A threshold with a ``"window"`` dim becomes ``(series, W)``; ``first_window_only`` keeps just its first column (used
    for history-only searches, whose candidates all lie before lead 1).

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled plan; its amount specs say which thresholds are needed (fixed amounts are skipped).
    thresholds : dict of str to (xarray.DataArray or numpy.ndarray or float), optional
        The threshold values by name, in mm: a scalar, a per-cell array or a `xarray.DataArray` over the spatial dims.
    spatial_dims : tuple of str
        The spatial dim names of the data, in order.
    spatial_shape : tuple of int
        The sizes of the spatial dims.
    series_shape : tuple of int
        The shape the values are broadcast to (the series dims before flattening).
    first_window_only : bool, default False
        Keep only the first window of a windowed threshold.

    Returns
    -------
    dict of str to numpy.ndarray of float64, shape (series,) or (series, window)
        The thresholds by name, flattened over the series.

    Raises
    ------
    KeyError
        A needed threshold is missing from ``thresholds``.
    """
    out = {}
    for spec in plan.amount_specs:
        if spec.kind == "fixed" or spec.name in out:
            continue
        if not thresholds or spec.name not in thresholds:
            raise KeyError(f"threshold {spec.name!r} is needed by the definition but was not supplied")
        value = thresholds[spec.name]
        if isinstance(value, xr.DataArray) and "window" in value.dims:
            value = value.transpose(*spatial_dims, "window").values.astype(np.float64)
            if first_window_only:
                value = value[..., 0]
            else:
                n_win = value.shape[-1]
                out[spec.name] = np.broadcast_to(value, (*series_shape, n_win)).reshape(-1, n_win)
                continue
        if isinstance(value, xr.DataArray):
            if value.ndim and set(value.dims) == set(spatial_dims):
                value = value.transpose(*spatial_dims)
            value = value.values
        value = np.asarray(value, dtype=np.float64)
        if value.size == 1:
            value = value.reshape(())
        else:
            value = value.reshape(spatial_shape)
        out[spec.name] = np.broadcast_to(value, series_shape).reshape(-1)
    return out


def _run(plan: plan_mod.Plan, tenths: np.ndarray, missing: np.ndarray, thresholds: dict[str, np.ndarray],
         jobs: Jobs, preset: np.ndarray, backend: str, windows: np.ndarray | None = None
         ) -> tuple[np.ndarray, np.ndarray]:
    """Run the backend over the jobs that need a search, chunked over series, and return the day and status of every job.

    Jobs preset to a status are not searched; jobs whose series is invalid (too little data for the thresholds) become
    ``INSUFFICIENT_DATA``. The series are processed in chunks of at most ``MAX_TABLE_CELLS`` series-by-time cells so the
    backend's prefix tables stay bounded.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled plan.
    tenths : numpy.ndarray of int, shape (series, time)
        Daily rain in tenths of mm.
    missing : numpy.ndarray of bool, shape (series, time)
        True where a day is missing.
    thresholds : dict of str to numpy.ndarray of float64
        Per-series thresholds from `_series_thresholds`.
    jobs : hcwf.onset.model.Jobs
        The jobs; ``jobs.series`` indexes the first axis of ``tenths``.
    preset : numpy.ndarray of int8, shape (job,)
        A `~hcwf.onset.model.Status` code for jobs decided without a search, ``NO_PRESET`` (-1) otherwise.
    backend : {"reference", "numba", "gpu"}
        The backend to run.
    windows : numpy.ndarray of int64, optional
        The start leads of ``(series, W)`` thresholds; a job's lead is its day index plus 1.

    Returns
    -------
    tuple of (numpy.ndarray of int64, numpy.ndarray of int8)
        The onset day index of each job (-1 where none) and its `~hcwf.onset.model.Status` code, each shape (job,).
    """
    extra = {} if windows is None else {"amount_windows": windows}
    module = _load_backend(backend)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=tenths.shape[0],
                                                exact=resolution_of(tenths) == "exact")
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.where(preset >= 0, preset, Status.NOT_FOUND).astype(np.int8)
    active = preset < 0
    bad = active & invalid[jobs.series]
    status[bad] = Status.INSUFFICIENT_DATA
    active &= ~bad
    idx = np.nonzero(active)[0]
    if len(idx) == 0:
        return day, status
    idx = idx[np.argsort(jobs.series[idx], kind="stable")]
    sorted_series = jobs.series[idx]
    per_chunk = max(1, MAX_TABLE_CELLS // max(tenths.shape[1], 1))
    edges = np.arange(0, tenths.shape[0] + per_chunk, per_chunk)
    bounds = np.searchsorted(sorted_series, edges)
    for k in range(len(edges) - 1):
        sel = idx[bounds[k]:bounds[k + 1]]
        if len(sel) == 0:
            continue
        s0, s1 = edges[k], min(edges[k + 1], tenths.shape[0])
        sub = Jobs(jobs.series[sel] - s0, jobs.start[sel], jobs.end[sel], jobs.data_end[sel],
                   None if jobs.origin is None else jobs.origin[sel])
        sub_amounts = {key: value[s0:s1] for key, value in amounts.items()}
        out = module.detect(plan, tenths[s0:s1], missing[s0:s1], sub_amounts, sub, **extra)
        day[sel] = out.day
        status[sel] = out.status
    return day, status


def _climatology_doys(definition: Definition, climatology: xr.Dataset, spatial_dims: tuple[str, ...], n_series: int):
    """Return the per-cell season bounds of a season climatology.

    The start and end variables are the ones the definition's `~hcwf.onset.model.DerivedSeasons` bounds refer to.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        A definition whose seasons are `~hcwf.onset.model.DerivedSeasons` with climatological bounds.
    climatology : xarray.Dataset
        The season climatology: ``onset_doy`` and ``cessation_doy`` over ``(season, *spatial)`` and, optionally,
        ``n_seasons`` over the spatial dims.
    spatial_dims : tuple of str
        The spatial dim names of the grid.
    n_series : int
        The number of cells (the product of the spatial sizes).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray or None, list of str)
        The start and end day of year, each float64 of shape (K, cell); the number of seasons of each cell (shape
        (cell,)) or ``None`` when the climatology has no ``n_seasons``; and the season names.

    Raises
    ------
    ValueError
        The seasons have no climatological bounds, or the climatology does not match the grid.
    """
    seasons = definition.seasons
    for ref in (seasons.start, seasons.end):
        if ref.climatological is None:
            raise ValueError("derived seasons need climatological bounds")
    var = {"onset": "onset_doy", "cessation": "cessation_doy"}
    n_k = climatology.sizes["season"]
    try:
        onset = climatology[var[seasons.start.climatological]].transpose("season", *spatial_dims).values.reshape(n_k, n_series)
        cessation = climatology[var[seasons.end.climatological]].transpose("season", *spatial_dims).values.reshape(n_k, n_series)
        n_seasons = None
        if "n_seasons" in climatology:
            n_seasons = climatology["n_seasons"].transpose(*spatial_dims).values.reshape(n_series)
    except ValueError as exc:
        raise ValueError(f"season_climatology does not match the grid ({n_series} cells over {spatial_dims}): {exc}") from exc
    names = [str(s) for s in climatology["season"].values] if "season" in climatology.coords \
        else [f"season_{k + 1}" for k in range(n_k)]
    return onset, cessation, n_seasons, names


def _derived_bounds(definition: Definition, keys: np.ndarray, years: np.ndarray, climatology: xr.Dataset,
                    spatial_dims: tuple[str, ...], n_series: int):
    """Return the first and last day indices of every derived season, plus a "no such season" mask.

    The bounds follow `hcwf.onset.calendar.doy_bounds`; when a season's start lies before the data but January 1 is
    inside it, the start is rescued from the climatological day of year and an open end is set to a very large index
    (the caller clamps it).

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        A definition with `~hcwf.onset.model.DerivedSeasons`.
    keys : numpy.ndarray of int64, shape (time,)
        The label keys (``year*10000 + month*100 + day``) of the time axis.
    years : numpy.ndarray of int64, shape (year,)
        The years of the output.
    climatology : xarray.Dataset
        The season climatology (see `_climatology_doys`).
    spatial_dims : tuple of str
        The spatial dim names of the grid.
    n_series : int
        The number of cells.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, list of str)
        The first and last day index (int64) and the absent mask (bool), each shape (K, year, cell), and the names.
    """
    seasons = definition.seasons
    onset, cessation, n_seasons, names = _climatology_doys(definition, climatology, spatial_dims, n_series)
    n_k = onset.shape[0]
    jan1 = cal.find_days(keys, years, 1, 1)
    firsts, lasts, none = [], [], []
    for k in range(n_k):
        s_doy = onset[k]
        e_doy = cessation[k]
        first, last = cal.doy_bounds(keys, years, s_doy, e_doy, seasons.start.offset_days, seasons.end.offset_days)
        start_off, _, _, valid = cal.doy_offsets(s_doy, e_doy, seasons.start.offset_days, seasons.end.offset_days)
        absent = ~valid
        if n_seasons is not None:
            absent |= n_seasons <= k
        rescue = (first < 0) & ~absent[None, :] & (jan1[:, None] >= 0)
        start_rescued = jan1[:, None] + start_off[None, :]
        first = np.where(rescue, start_rescued, first)
        last = np.where(rescue & (last < 0), np.iinfo(np.int64).max // 4, last)
        firsts.append(first)
        lasts.append(last)
        none.append(np.broadcast_to(absent[None, :], first.shape))
    return np.stack(firsts), np.stack(lasts), np.stack(none), names


def build_observed_jobs(definition: Definition, rain: DailyRain, years=None,
                        season_climatology: xr.Dataset | None = None) -> tuple[Jobs, JobLayout]:
    """Build one job per (year, season, cell), in that C order, with the cells as backend series.

    Fixed seasons use `hcwf.onset.calendar.fixed_bounds`; `~hcwf.onset.model.DerivedSeasons` use `hcwf.onset.calendar.doy_bounds` on
    ``season_climatology`` (``n_seasons``, ``onset_doy``, ``cessation_doy`` with dims ``(season, *spatial)``), which is
    then required. A season start outside the data presets the job to ``INSUFFICIENT_DATA``; a cell without that
    derived season presets ``NOT_FOUND``. A season end past the data is clamped to the last day (the horizon flag then
    decides). ``data_end`` is the length of the time axis.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition, whose seasons say where to search.
    rain : hcwf.onset.model.DailyRain
        The observed daily rain, with dims ``(*spatial, time)``.
    years : array_like of int, optional
        The years to search; default the unique years of ``rain``.
    season_climatology : xarray.Dataset, optional
        Required for `~hcwf.onset.model.DerivedSeasons`: ``n_seasons``, ``onset_doy`` and ``cessation_doy`` over ``(season, *spatial)``.

    Returns
    -------
    tuple of (hcwf.onset.model.Jobs, JobLayout)
        The jobs and the layout that maps them back to ``(year, season)``.

    Raises
    ------
    ValueError
        The seasons are derived and ``season_climatology`` is missing or does not match the grid.
    """
    keys = cal.label_keys(rain.year, rain.month, rain.day)
    n_time = len(keys)
    years = np.unique(rain.year) if years is None else np.asarray(years, dtype=np.int64)
    spatial_dims = rain.dims[:-1]
    n_series = int(np.prod(rain.tenths.shape[:-1], dtype=np.int64))
    if isinstance(definition.seasons, DerivedSeasons):
        if season_climatology is None:
            raise ValueError("derived seasons need season_climatology (n_seasons, onset_doy, cessation_doy)")
        first, last, absent, names = _derived_bounds(definition, keys, years, season_climatology,
                                                     spatial_dims, n_series)
    else:
        names = [s.name for s in definition.seasons]
        firsts, lasts = [], []
        for season in definition.seasons:
            lo, hi = cal.fixed_bounds(keys, years, season.start, season.end)
            firsts.append(np.broadcast_to(lo[:, None], (len(years), n_series)))
            lasts.append(np.broadcast_to(hi[:, None], (len(years), n_series)))
        first, last = np.stack(firsts), np.stack(lasts)
        absent = np.zeros(first.shape, dtype=bool)
    first, last = first.swapaxes(0, 1), last.swapaxes(0, 1)
    absent = absent.swapaxes(0, 1)
    bad_start = (first < 0) | (first >= n_time)
    end = np.minimum(np.where(last < 0, n_time - 1, last), n_time - 1)
    preset = np.full(first.shape, NO_PRESET, dtype=np.int8)
    preset[bad_start] = Status.INSUFFICIENT_DATA
    preset[absent] = Status.NOT_FOUND
    skip = preset >= 0
    start = np.where(skip, 0, first)
    end = np.where(skip, -1, end)
    shape = first.shape[:-1]
    series = np.broadcast_to(np.arange(n_series, dtype=np.int64), first.shape)
    jobs = Jobs(series.ravel().copy(), start.ravel().astype(np.int64), end.ravel().astype(np.int64),
                np.full(first.size, n_time, dtype=np.int64))
    return jobs, JobLayout(shape, preset.ravel(), names, n_series)




def _add_days(calendar: str, year, month, day, offsets: np.ndarray):
    """Return the labels of (year, month, day) plus a number of days in a calendar.

    Standard calendars use ``datetime64`` arithmetic; other calendars (for example ``noleap`` or ``360_day``) use
    ``cftime``.

    Parameters
    ----------
    calendar : str
        The calendar name; ``"standard"``, ``"gregorian"`` and ``"proleptic_gregorian"`` are standard.
    year : array_like of int
        The years of the dates, shape (n_dates,).
    month : array_like of int
        The months, shape (n_dates,).
    day : array_like of int
        The days of the month, shape (n_dates,).
    offsets : numpy.ndarray of int64, shape (n_offsets,)
        The days to add (may be negative).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The year, month and day of every date plus every offset, each int64 of shape (n_dates, n_offsets).
    """
    year, month, day = (np.asarray(a, dtype=np.int64) for a in (year, month, day))
    offsets = np.asarray(offsets, dtype=np.int64)
    if calendar in _STANDARD_CALENDARS:
        months = (year - 1970) * 12 + month - 1
        base = months.astype("datetime64[M]").astype("datetime64[D]") + (day - 1).astype("timedelta64[D]")
        dates = base[:, None] + offsets[None, :].astype("timedelta64[D]")
        out_m = dates.astype("datetime64[M]")
        out_y = out_m.astype(np.int64) // 12 + 1970
        out_month = out_m.astype(np.int64) % 12 + 1
        out_day = (dates - out_m.astype("datetime64[D]")).astype(np.int64) + 1
        return out_y, out_month, out_day
    import cftime
    out = np.empty((3, len(year), len(offsets)), dtype=np.int64)
    steps = [_dt.timedelta(days=int(n)) for n in offsets]
    for i in range(len(year)):
        base = cftime.datetime(int(year[i]), int(month[i]), int(day[i]), calendar=calendar)
        for j, step in enumerate(steps):
            moved = base + step
            out[:, i, j] = moved.year, moved.month, moved.day
    return out[0], out[1], out[2]


def _dates_from_labels(calendar: str, year: np.ndarray, month: np.ndarray, day: np.ndarray,
                       valid: np.ndarray) -> np.ndarray:
    """Return dates for year, month and day label arrays.

    Parameters
    ----------
    calendar : str
        The calendar name.
    year : numpy.ndarray of int64
        The year labels.
    month : numpy.ndarray of int64, same shape as ``year``
        The month labels.
    day : numpy.ndarray of int64, same shape as ``year``
        The day-of-month labels.
    valid : numpy.ndarray of bool
        Where the label is a real date.

    Returns
    -------
    numpy.ndarray
        For standard calendars ``datetime64[ns]`` with ``NaT`` where not valid; otherwise an object array of
        ``cftime.datetime`` with ``None`` where not valid. Same shape as ``year``.
    """
    if calendar in _STANDARD_CALENDARS:
        year, month, day = (np.where(valid, a, fill) for a, fill in ((year, 1970), (month, 1), (day, 1)))
        months = ((year - 1970) * 12 + month - 1).astype("datetime64[M]")
        dates = months.astype("datetime64[D]") + (day - 1).astype("timedelta64[D]")
        return np.where(valid, dates.astype("datetime64[ns]"), np.datetime64("NaT", "ns"))
    import cftime
    out = np.full(year.shape, None, dtype=object)
    flat = np.nonzero(valid.ravel())[0]
    uniq = {}
    target = out.reshape(-1)
    y, m, d = year.reshape(-1), month.reshape(-1), day.reshape(-1)
    for i in flat:
        key = (int(y[i]), int(m[i]), int(d[i]))
        if key not in uniq:
            uniq[key] = cftime.datetime(*key, calendar=calendar)
        target[i] = uniq[key]
    return out


def _status_var(values: np.ndarray, dims: tuple[str, ...]) -> xr.DataArray:
    """Return status codes as an int8 `xarray.DataArray` with CF ``flag_values`` and ``flag_meanings``.

    Parameters
    ----------
    values : numpy.ndarray
        The `~hcwf.onset.model.Status` codes.
    dims : tuple of str
        The dim names of ``values``.

    Returns
    -------
    xarray.DataArray of int8
        The status variable.
    """
    return xr.DataArray(values.astype(np.int8), dims=dims, attrs={
        "long_name": "onset search outcome",
        "flag_values": np.array([s.value for s in Status], dtype=np.int8),
        "flag_meanings": " ".join(s.name for s in Status)})


def _spatial_coords(dims: tuple[str, ...], coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the coordinate arrays known for the given dims.

    Parameters
    ----------
    dims : tuple of str
        The dim names wanted.
    coords : dict of str to numpy.ndarray
        The coordinates available, by dim name.

    Returns
    -------
    dict of str to numpy.ndarray
        The entries of ``coords`` whose name is in ``dims``.
    """
    return {d: coords[d] for d in dims if d in coords}


BASE_START = "base"
"""Name of the search start taken from the season itself, first on the "start" axis when search_start is given."""


def _missing_policy(data: DailyRain | ForecastRain, missing: str | None) -> str:
    """Return the missing-rain policy of a run: the argument if given, else the one ingest recorded.

    Parameters
    ----------
    data : hcwf.onset.model.DailyRain or hcwf.onset.model.ForecastRain
        The rain; ``data.missing_policy`` is the policy ingest recorded.
    missing : str, optional
        The policy to use instead; one of ``MISSING_POLICIES``.

    Returns
    -------
    str
        The policy.

    Raises
    ------
    ValueError
        The policy is not in ``MISSING_POLICIES``.
    """
    policy = data.missing_policy if missing is None else missing
    if policy not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {policy!r}")
    return policy


def _flat_missing(data: DailyRain | ForecastRain, policy: str, n_series: int) -> np.ndarray:
    """Return the missing flags as (series, time), all False under ``"zero"`` (the days already hold 0).

    Parameters
    ----------
    data : hcwf.onset.model.DailyRain or hcwf.onset.model.ForecastRain
        The rain.
    policy : str
        The missing-rain policy.
    n_series : int
        The number of series (the product of the dims before time).

    Returns
    -------
    numpy.ndarray of bool, shape (series, time)
        True where a day is missing.
    """
    n_time = data.tenths.shape[-1]
    if policy == "zero":
        return np.zeros((n_series, n_time), dtype=bool)
    return data.missing.reshape(n_series, n_time)


def _drop_incomplete_years(rain: DailyRain, years: np.ndarray, layout: JobLayout) -> None:
    """Preset ``INSUFFICIENT_DATA`` for every (year, cell) with a missing day anywhere in that year (the ``"drop"`` policy).

    ``layout.preset`` is changed in place.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The observed rain with its missing flags.
    years : numpy.ndarray of int64, shape (year,)
        The years of the output.
    layout : JobLayout
        The layout of the observed jobs; its ``preset`` reshapes to ``(year, season, series)``.
    """
    flat = rain.missing.reshape(layout.n_series, -1)
    n_seasons = len(layout.season_names)
    preset = layout.preset.reshape(len(years), n_seasons, layout.n_series)
    for i, year in enumerate(years):
        cols = np.flatnonzero(rain.year == year)
        if len(cols):
            preset[i, :, flat[:, cols].any(axis=1)] = Status.INSUFFICIENT_DATA


def _stack_starts(jobs: Jobs, preset: np.ndarray, starts: list[np.ndarray]) -> tuple[Jobs, np.ndarray]:
    """Repeat the jobs once per search start (base first).

    For variant ``v`` the start is ``max(job start, entry)`` and the origin is the job's own start. All variants then
    share one backend call and one set of prefix tables.

    Parameters
    ----------
    jobs : hcwf.onset.model.Jobs
        The base jobs.
    preset : numpy.ndarray of int8, shape (job,)
        The preset `~hcwf.onset.model.Status` codes of the base jobs.
    starts : list of numpy.ndarray
        One array per variant, each shape (job,), of earliest day indices (-1: no restriction).

    Returns
    -------
    tuple of (hcwf.onset.model.Jobs, numpy.ndarray of int8)
        The tiled jobs (base variant first) and the tiled presets, of length ``job * (len(starts) + 1)``.
    """
    n_jobs, n_var = len(jobs.series), len(starts) + 1
    start = np.stack([jobs.start, *[np.where(jobs.end >= jobs.start, np.maximum(jobs.start, e), jobs.start)
                                    for e in starts]])
    origin = np.broadcast_to(jobs.start if jobs.origin is None else jobs.origin, (n_var, n_jobs))
    tiled = Jobs(np.tile(jobs.series, n_var), start.reshape(-1), np.tile(jobs.end, n_var),
                 np.tile(jobs.data_end, n_var), origin.reshape(-1).copy())
    return tiled, np.tile(preset, n_var)


def _date_keys(values, calendar: str) -> np.ndarray:
    """Return the label keys (``year*10000 + month*100 + day``) of ``datetime64`` dates, -1 for ``NaT``.

    Parameters
    ----------
    values : array_like of datetime64
        The dates.
    calendar : str
        The calendar of the data; must be a standard one.

    Returns
    -------
    numpy.ndarray of int64
        The label keys, same shape as ``values``.

    Raises
    ------
    ValueError
        ``values`` are not ``datetime64`` or the calendar is not standard.
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError("search_start dates must be datetime64 (NaT for no restriction)")
    if calendar not in _STANDARD_CALENDARS:
        raise ValueError(f"search_start dates need a standard calendar, not {calendar!r}")
    days = values.astype("datetime64[D]")
    months = days.astype("datetime64[M]")
    year = days.astype("datetime64[Y]").astype(np.int64) + 1970
    keys = cal.label_keys(year, months.astype(np.int64) % 12 + 1, (days - months).astype(np.int64) + 1)
    return np.where(np.isnat(values), -1, keys)


def _entry_keys(entry, rows: np.ndarray, row_dim: str, spatial_dims: tuple[str, ...], spatial_shape: tuple[int, ...],
                calendar: str) -> np.ndarray:
    """Return the label keys of a ``search_start`` entry as (row, cell).

    An ``"MM-DD"`` string means that day of each row's year; a `xarray.DataArray` holds dates over ``row_dim`` (optional)
    and the spatial dims (optional), ``NaT`` for no restriction.

    Parameters
    ----------
    entry : str or xarray.DataArray
        The ``"MM-DD"`` string or the array of dates.
    rows : numpy.ndarray of int64, shape (row,)
        The years (or init years) of the output.
    row_dim : str
        The dim of ``entry`` that runs over ``rows``.
    spatial_dims : tuple of str
        The spatial dim names.
    spatial_shape : tuple of int
        The sizes of the spatial dims.
    calendar : str
        The calendar of the data.

    Returns
    -------
    numpy.ndarray of int64, shape (row, cell)
        The label keys, -1 where there is no restriction.

    Raises
    ------
    TypeError
        ``entry`` is neither a string nor a `xarray.DataArray`.
    ValueError
        ``entry`` has dims other than ``row_dim`` and the spatial dims, or holds invalid dates.
    """
    n_cell = int(np.prod(spatial_shape, dtype=np.int64))
    if isinstance(entry, str):
        month, day = int(entry[:2]), int(entry[3:])
        return np.broadcast_to((rows * 10000 + month * 100 + day)[:, None], (len(rows), n_cell))
    if not isinstance(entry, xr.DataArray):
        raise TypeError(f"a search_start entry is an 'MM-DD' string or a DataArray of dates, got {type(entry).__name__}")
    extra = set(entry.dims) - {row_dim, *spatial_dims}
    if extra:
        raise ValueError(f"search_start dims {sorted(extra)} are not {row_dim} or the spatial dims {spatial_dims}")
    dims = tuple(d for d in (row_dim, *spatial_dims) if d in entry.dims)
    keys = _date_keys(entry.transpose(*dims).values, calendar)
    shape = tuple(entry.sizes[d] if d in entry.dims else 1 for d in (row_dim, *spatial_dims))
    return np.broadcast_to(keys.reshape(shape), (len(rows), *spatial_shape)).reshape(len(rows), n_cell)


def _observed_starts(search_start: Mapping, rain: DailyRain, years: np.ndarray, n_seasons: int) -> list[np.ndarray]:
    """Return, per variant, the earliest day index of every observed job.

    Jobs are ordered (year, season, cell); an entry after the end of the data gives the time axis length.

    Parameters
    ----------
    search_start : collections.abc.Mapping
        The ``search_start`` entries by name (see `detect_observed`).
    rain : hcwf.onset.model.DailyRain
        The observed rain.
    years : numpy.ndarray of int64, shape (year,)
        The years of the output.
    n_seasons : int
        The number of seasons.

    Returns
    -------
    list of numpy.ndarray of int64
        One array of shape (year * season * cell,) per entry, -1 where there is no restriction.
    """
    keys = cal.label_keys(rain.year, rain.month, rain.day)
    spatial_dims, spatial_shape = rain.dims[:-1], rain.tenths.shape[:-1]
    out = []
    for entry in search_start.values():
        wanted = _entry_keys(entry, years, "year", spatial_dims, spatial_shape, rain.calendar)
        index = np.where(wanted >= 0, np.searchsorted(keys, wanted), -1)
        out.append(np.broadcast_to(index[:, None, :], (len(years), n_seasons, index.shape[1])).reshape(-1))
    return out


def _check_search_start(search_start: Mapping | None) -> None:
    """Reject ``search_start`` names that clash with the base start.

    Parameters
    ----------
    search_start : collections.abc.Mapping, optional
        The ``search_start`` entries by name.

    Raises
    ------
    ValueError
        A name equals ``BASE_START``.
    """
    if search_start and BASE_START in search_start:
        raise ValueError(f"'{BASE_START}' is the season's own start; name the search_start entry differently")


def detect_observed(definition: Definition, rain: DailyRain, *, thresholds=None, years=None,
                    season_climatology: xr.Dataset | None = None, backend: str = "numba",
                    attrs: dict | None = None, horizon: str = DEFAULT_HORIZON, missing: str | None = None,
                    search_start: Mapping[str, str | xr.DataArray] | None = None) -> xr.Dataset:
    """Detect onset in every (year, season, cell).

    Returns a Dataset with dims ``(year, season, *spatial)`` and variables ``onset_date`` (``datetime64[ns]``, or
    ``cftime`` objects for non-standard calendars), ``onset_index`` (index on the time axis, -1 if none) and ``status``
    (int8 `~hcwf.onset.model.Status` codes).

    Run options (not part of the definition; recorded in attrs): ``horizon`` is ``"censor"`` (a span past the data ends
    the search ``CENSORED``), ``"truncate"`` or ``"reject"`` (see `hcwf.onset.plan.compile`); ``missing`` overrides the
    policy ingest recorded on ``rain``, and ``"drop"`` marks every (year, cell) with a missing day in that year
    ``INSUFFICIENT_DATA``; ``rain_resolution`` is the one of ``rain``. ``search_start`` maps a name to dates (``"MM-DD"``,
    or a `xarray.DataArray` of dates over year and the spatial dims, ``NaT`` for none): each adds a variant whose
    search starts at ``max(season start, date)``, and the result gets a ``"start"`` dim, ``"base"`` first, then the
    names. Variants share one backend call.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition.
    rain : hcwf.onset.model.DailyRain
        The observed daily rain, dims ``(*spatial, time)``.
    thresholds : dict of str to (xarray.DataArray or numpy.ndarray or float), optional
        Threshold values by name, in mm per cell (a `xarray.DataArray` over the spatial dims, or an array).
    years : array_like of int, optional
        The years to search; default the unique years of ``rain``.
    season_climatology : xarray.Dataset, optional
        Required for `~hcwf.onset.model.DerivedSeasons`: ``n_seasons``, ``onset_doy``, ``cessation_doy`` over ``(season, *spatial)``.
    backend : {"reference", "numba", "gpu"}, default "numba"
        The detection backend.
    attrs : dict, optional
        Merged into the dataset attrs.
    horizon : {"censor", "truncate", "reject"}, default "censor"
        What to do with a span that runs past the data.
    missing : str, optional
        The missing-rain policy; default the one recorded on ``rain``.
    search_start : dict of str to (str or xarray.DataArray), optional
        Extra search-start variants by name.

    Returns
    -------
    xarray.Dataset
        ``onset_date``, ``onset_index`` and ``status`` over ``(year, season, [start], *spatial)``.

    """
    _check_search_start(search_start)
    policy = _missing_policy(rain, missing)
    plan = plan_mod.compile(definition, horizon=horizon, missing=policy)
    jobs, layout = build_observed_jobs(definition, rain, years, season_climatology)
    years = np.unique(rain.year) if years is None else np.asarray(years, dtype=np.int64)
    if policy == "drop":
        _drop_incomplete_years(rain, years, layout)
    preset = layout.preset
    if search_start:
        jobs, preset = _stack_starts(jobs, preset, _observed_starts(search_start, rain, years, len(layout.season_names)))
    spatial_dims = rain.dims[:-1]
    spatial_shape = rain.tenths.shape[:-1]
    n_time = rain.tenths.shape[-1]
    per_series = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape,
                                    first_window_only=True)
    day, status = _run(plan, rain.tenths.reshape(layout.n_series, n_time), _flat_missing(rain, policy, layout.n_series),
                       per_series, jobs, preset, backend)
    out_shape = (len(years), len(layout.season_names), *spatial_shape)
    dims = ("year", "season", *spatial_dims)
    coords = {"year": years, "season": layout.season_names, **_spatial_coords(spatial_dims, rain.coords)}
    if search_start:
        out_shape = (len(search_start) + 1, *out_shape)
        dims = ("year", "season", "start", *spatial_dims)
        coords["start"] = [BASE_START, *search_start]
        day, status = (np.moveaxis(a.reshape(out_shape), 0, 2) for a in (day, status))
    index = day.reshape(out_shape) if not search_start else day
    safe = np.where(index >= 0, index, 0)
    dates = _dates_from_labels(rain.calendar, rain.year[safe], rain.month[safe], rain.day[safe], index >= 0)
    ds = xr.Dataset(
        {"onset_date": (dims, dates), "onset_index": (dims, index.astype(np.int64)),
         "status": _status_var(status.reshape(index.shape), dims)}, coords=coords)
    ds.attrs.update({"definition_id": definition.id, "definition_version": definition.version, "backend": backend,
                     **_option_attrs(rain, horizon, policy, search_start)})
    ds.attrs.update(attrs or {})
    return ds


def _option_attrs(data: DailyRain | ForecastRain, horizon: str, policy: str, search_start: Mapping | None) -> dict:
    """Return the dataset attrs recording the run options (``rain_resolution``, ``missing``, ``horizon``, ``search_start``).

    Parameters
    ----------
    data : hcwf.onset.model.DailyRain or hcwf.onset.model.ForecastRain
        The rain, whose resolution is recorded.
    horizon : str
        The horizon option.
    policy : str
        The missing-rain policy.
    search_start : collections.abc.Mapping, optional
        The ``search_start`` entries by name.

    Returns
    -------
    dict
        The attrs; ``search_start`` is ``"name=MM-DD; name=per-cell dates"`` and appears only when entries are given.
    """
    out = {"rain_resolution": resolution_of(data.tenths), "missing": policy, "horizon": horizon}
    if search_start:
        out["search_start"] = "; ".join(f"{name}={entry if isinstance(entry, str) else 'per-cell dates'}"
                                        for name, entry in search_start.items())
    return out


def _bound_key(calendar: str, year: np.ndarray, ref) -> np.ndarray:
    """Return the label key of a month-day season bound plus its day offset in each of ``year``.

    Parameters
    ----------
    calendar : str
        The calendar name.
    year : numpy.ndarray of int64, shape (n,)
        The years.
    ref : object
        The season bound, with ``month``, ``day`` and ``offset_days`` attributes.

    Returns
    -------
    numpy.ndarray of int64, shape (n, 1)
        The label keys (``year*10000 + month*100 + day``).
    """
    if ref.offset_days == 0:
        return (year * 10000 + ref.month * 100 + ref.day)[:, None]
    y, m, d = _add_days(calendar, year, np.full_like(year, ref.month), np.full_like(year, ref.day),
                        np.array([ref.offset_days]))
    return cal.label_keys(y, m, d)


def _derived_windows(definition: Definition, calendar: str, init_year: np.ndarray, climatology: xr.Dataset,
                     spatial_dims: tuple[str, ...], n_cell: int):
    """Return the label-key windows of each derived season per init and cell, for the init year minus one, the year and the year after.

    The bounds follow `hcwf.onset.calendar.doy_bounds`: a season ending before its start day-of-year ends in the
    following year.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        A definition with `~hcwf.onset.model.DerivedSeasons`.
    calendar : str
        The calendar name.
    init_year : array_like of int
        The year of each init, shape (n_init,).
    climatology : xarray.Dataset
        The season climatology (see `_climatology_doys`).
    spatial_dims : tuple of str
        The spatial dim names.
    n_cell : int
        The number of cells.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, list of str)
        ``low`` and ``high`` (int64, shape (3, K, init, cell)) hold the ``year*10000 + month*100 + day`` key of the first
        and last day of the season that starts in year ``init_year + shift`` (shift -1, 0, 1); ``present`` (bool, shape
        (K, cell)) is False where the cell lacks that season; and the season names.
    """
    seasons = definition.seasons
    onset, cessation, n_seasons, names = _climatology_doys(definition, climatology, spatial_dims, n_cell)
    n_k = onset.shape[0]
    present = np.isfinite(onset) & np.isfinite(cessation)
    if n_seasons is not None:
        present &= (n_seasons[None, :] > np.arange(n_k)[:, None])
    init_year = np.asarray(init_year, dtype=np.int64)
    n_init = len(init_year)
    low = np.zeros((3, n_k, n_init, n_cell), dtype=np.int64)
    high = np.zeros_like(low)

    def shifted_keys(year: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Return the label keys of Jan 1 of each year plus per-cell day offsets, shaped (len(year), n_cell)."""
        uniq, inverse = np.unique(offsets, return_inverse=True)
        ones = np.ones_like(year)
        y, m, d = _add_days(calendar, year, ones, ones, uniq)
        return cal.label_keys(y, m, d)[:, inverse.reshape(-1)]

    for k in range(n_k):
        start_off, end_off, wraps, _ = cal.doy_offsets(onset[k], cessation[k], seasons.start.offset_days,
                                                       seasons.end.offset_days)
        wraps = wraps[None, :]
        for i, shift in enumerate((-1, 0, 1)):
            year = init_year + shift
            low[i, k] = shifted_keys(year, start_off)
            high[i, k] = np.where(wraps, shifted_keys(year + 1, end_off), shifted_keys(year, end_off))
    return low, high, present, names


def _pick_derived(keys: np.ndarray, allowed: np.ndarray, low: np.ndarray, high: np.ndarray, present: np.ndarray):
    """Choose each (init, cell)'s season year and the first and last qualifying lead of every derived season.

    A season takes the earliest of the three years having a lead of the window inside it.

    Parameters
    ----------
    keys : numpy.ndarray of int64, shape (init, lead)
        The label keys of the leads.
    allowed : numpy.ndarray of bool, shape (lead,)
        The leads of the lead window.
    low : numpy.ndarray of int64, shape (3, K, init, cell)
        The first-day keys of the season windows from `_derived_windows`.
    high : numpy.ndarray of int64, shape (3, K, init, cell)
        The last-day keys of the season windows from `_derived_windows`.
    present : numpy.ndarray of bool, shape (K, cell)
        Whether each cell has that season.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        ``first`` and ``last`` lead index (int64, shape (K, init, cell); -1 for ``last`` where none), ``has`` (bool,
        same shape) and ``take`` (bool, shape (3, K, init, cell)), the shift each (init, cell) chose.
    """
    _, n_k, n_init, n_cell = low.shape
    first = np.zeros((n_k, n_init, n_cell), dtype=np.int64)
    last = np.full((n_k, n_init, n_cell), -1, dtype=np.int64)
    has = np.zeros((n_k, n_init, n_cell), dtype=bool)
    chosen = np.zeros((3, n_k, n_init, n_cell), dtype=bool)
    n_leads = keys.shape[1]
    for k in range(n_k):
        done = ~np.broadcast_to(present[k][None, :], (n_init, n_cell))
        for i in range(3):
            inside = ((keys[:, :, None] >= low[i, k][:, None, :]) & (keys[:, :, None] <= high[i, k][:, None, :])
                      & allowed[None, :, None])
            take = inside.any(axis=1) & ~done
            first[k] = np.where(take, np.argmax(inside, axis=1), first[k])
            last[k] = np.where(take, n_leads - 1 - np.argmax(inside[:, ::-1, :], axis=1), last[k])
            has[k] |= take
            chosen[i, k] = take
            done |= take
    return first, last, has, chosen


def _label_table(forecast: ForecastRain, n_leads: int):
    """Return the date labels of the leads of a forecast.

    The label of lead ``k`` is ``init + k - 1`` days for ``lead_label`` ``"start"`` and ``init + k`` for ``"end"``.

    Parameters
    ----------
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    n_leads : int
        The number of leads.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The year, month and day of every lead, each int64 of shape (init, lead).
    """
    lead_offsets = np.arange(n_leads) + (1 if forecast.lead_label == "end" else 0)
    return _add_days(forecast.calendar, forecast.init_year, forecast.init_month, forecast.init_day, lead_offsets)


def _derived_dims(forecast: ForecastRain):
    """Return the spatial dims, member count and cell count of a forecast's series dims ``(init, [member], *spatial)``.

    Parameters
    ----------
    forecast : hcwf.onset.model.ForecastRain
        The forecast.

    Returns
    -------
    tuple of (tuple of str, int, int)
        The spatial dim names, the number of members (1 without a member dim) and the number of cells.
    """
    n_lead_dims = 2 if "member" in forecast.dims[:-1] else 1
    spatial_dims = forecast.dims[n_lead_dims:-1]
    n_cell = int(np.prod(forecast.tenths.shape[n_lead_dims:-1], dtype=np.int64))
    n_member = forecast.tenths.shape[1] if n_lead_dims == 2 else 1
    return spatial_dims, n_member, n_cell


def _expand_rows(a: np.ndarray, n_member: int, n_cell: int) -> np.ndarray:
    """Flatten (season, init) or (season, init, cell) values to (season, init, member, cell) order.

    Parameters
    ----------
    a : numpy.ndarray, shape (season, init) or (season, init, cell)
        The values.
    n_member : int
        The number of members (1 without members).
    n_cell : int
        The number of cells.

    Returns
    -------
    numpy.ndarray, shape (season * init * member * cell,)
        The values broadcast over member (and cell) and flattened.
    """
    k, n_init = a.shape[:2]
    shaped = a.reshape(k, n_init, 1, -1) if a.ndim == 3 else a.reshape(k, n_init, 1, 1)
    return np.broadcast_to(shaped, (k, n_init, n_member, n_cell)).ravel()


def _require_climatology(season_climatology: xr.Dataset | None) -> xr.Dataset:
    """Return ``season_climatology``, or raise when it is missing.

    Parameters
    ----------
    season_climatology : xarray.Dataset, optional
        The season climatology.

    Returns
    -------
    xarray.Dataset
        ``season_climatology`` itself.

    Raises
    ------
    ValueError
        ``season_climatology`` is ``None`` (derived seasons need one).
    """
    if season_climatology is None:
        raise ValueError("derived seasons need season_climatology (n_seasons, onset_doy, cessation_doy)")
    return season_climatology


def _derived_forecast_jobs(definition: Definition, forecast: ForecastRain, allowed: np.ndarray,
                           season_climatology: xr.Dataset | None) -> tuple[Jobs, JobLayout]:
    """Build forecast jobs for `~hcwf.onset.model.DerivedSeasons`, restricted to each cell's own season.

    A missing ``season_climatology`` raises `ValueError` (in `_require_climatology`).

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        A definition with `~hcwf.onset.model.DerivedSeasons`.
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    allowed : numpy.ndarray of bool, shape (lead,)
        The leads of the lead window.
    season_climatology : xarray.Dataset, optional
        The season climatology; required.

    Returns
    -------
    tuple of (hcwf.onset.model.Jobs, JobLayout)
        The jobs, ordered (season, init, [member], cell), and their layout.

    """
    climatology = _require_climatology(season_climatology)
    n_leads, n_init = forecast.tenths.shape[-1], forecast.tenths.shape[0]
    spatial_dims, n_member, n_cell = _derived_dims(forecast)
    n_series = n_init * n_member * n_cell
    low, high, present, names = _derived_windows(definition, forecast.calendar, forecast.init_year, climatology,
                                                 spatial_dims, n_cell)
    keys = cal.label_keys(*_label_table(forecast, n_leads))
    first, last, has, _ = _pick_derived(keys, allowed, low, high, present)
    k = len(names)
    preset = np.where(has, NO_PRESET, Status.NOT_FOUND).astype(np.int8)
    series = np.broadcast_to(np.arange(n_series, dtype=np.int64), (k, n_series)).ravel().copy()
    jobs = Jobs(series, _expand_rows(np.where(has, first, 0), n_member, n_cell),
                _expand_rows(np.where(has, last, -1), n_member, n_cell), np.full(k * n_series, n_leads, dtype=np.int64))
    return jobs, JobLayout((k,), _expand_rows(preset, n_member, n_cell), names, n_series)


def _lead_window_bounds(lead_window: tuple[int, int], n_leads: int, from_first: bool = False) -> tuple[int, int]:
    """Return the first and last candidate lead (1-based, inclusive) of a lead window, the last cut to the data.

    Parameters
    ----------
    lead_window : tuple of (int, int)
        The first and last candidate lead.
    n_leads : int
        The number of leads of the forecast.
    from_first : bool, default False
        Require the window to start at lead 1 (stitched detection).

    Returns
    -------
    tuple of (int, int)
        The first lead and the last lead, cut to ``n_leads``.

    Raises
    ------
    ValueError
        The window starts before lead 1, or after it when ``from_first``.
    """
    first, last = int(lead_window[0]), int(lead_window[1])
    if first < 1 or (from_first and first != 1):
        need = "start at 1" if from_first else "start at 1 or later"
        raise ValueError(f"lead_window must {need}, got {tuple(lead_window)}")
    return first, min(last, n_leads)


def build_forecast_jobs(definition: Definition, forecast: ForecastRain, lead_window: tuple[int, int] = (1, 30),
                        restrict_to_season: bool = True, season_climatology: xr.Dataset | None = None
                        ) -> tuple[Jobs, JobLayout]:
    """Build one job per (season, series) with series = (init, [member], cell) rows over the lead axis.

    Candidates are the leads in ``lead_window`` (1-based, inclusive) whose date label falls in a window of the season,
    taken from the earliest of the init's year minus one, the init's year and the year after that has any such lead;
    the qualifying leads are contiguous. With none, the job is ``NOT_FOUND``. With ``restrict_to_season`` false there
    is one season, ``"window"``, and every lead of ``lead_window`` is a candidate. `~hcwf.onset.model.DerivedSeasons` with
    ``restrict_to_season`` true need ``season_climatology`` (as for `build_observed_jobs`): each cell's windows come from
    its own season bounds, the season year is chosen per (init, cell) as above, and a cell without that season is
    ``NOT_FOUND``, so first, last and the status vary by series row. ``data_end`` is the number of leads.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition.
    forecast : hcwf.onset.model.ForecastRain
        The forecast, dims ``(init_time, [member], *spatial, lead)``.
    lead_window : tuple of (int, int), default (1, 30)
        The first and last candidate lead, 1-based and inclusive; the first must be at least 1 (a `ValueError` otherwise).
    restrict_to_season : bool, default True
        Limit the candidates to the seasons of the definition.
    season_climatology : xarray.Dataset, optional
        Required for `~hcwf.onset.model.DerivedSeasons` when ``restrict_to_season`` is true.

    Returns
    -------
    tuple of (hcwf.onset.model.Jobs, JobLayout)
        The jobs, ordered (season, init, [member], cell), and their layout.
    """
    n_leads = forecast.tenths.shape[-1]
    n_init = forecast.tenths.shape[0]
    n_series = int(np.prod(forecast.tenths.shape[:-1], dtype=np.int64))
    rest = n_series // max(n_init, 1)
    lo, hi = _lead_window_bounds(lead_window, n_leads)
    allowed = np.zeros(n_leads, dtype=bool)
    allowed[lo - 1:hi] = True
    if not restrict_to_season:
        names = ["window"]
        masks = [np.broadcast_to(allowed, (n_init, n_leads))]
    else:
        if isinstance(definition.seasons, DerivedSeasons):
            return _derived_forecast_jobs(definition, forecast, allowed, season_climatology)
        names = [s.name for s in definition.seasons]
        label_y, label_m, label_d = _label_table(forecast, n_leads)
        keys = cal.label_keys(label_y, label_m, label_d)
        masks = []
        for season in definition.seasons:
            wraps = (season.end.month, season.end.day) < (season.start.month, season.start.day)
            chosen = np.zeros((n_init, n_leads), dtype=bool)
            done = np.zeros(n_init, dtype=bool)
            for shift in (-1, 0, 1):
                year = np.asarray(forecast.init_year, dtype=np.int64) + shift
                low = _bound_key(forecast.calendar, year, season.start)
                high = _bound_key(forecast.calendar, year + wraps, season.end)
                inside = (keys >= low) & (keys <= high) & allowed[None, :]
                take = inside.any(axis=1) & ~done
                chosen[take] = inside[take]
                done |= take
            masks.append(chosen)
    k = len(names)
    first = np.zeros((k, n_init), dtype=np.int64)
    last = np.full((k, n_init), -1, dtype=np.int64)
    preset = np.full((k, n_init), NO_PRESET, dtype=np.int8)
    for i, mask in enumerate(masks):
        has = mask.any(axis=1)
        first[i] = np.where(has, np.argmax(mask, axis=1), 0)
        last[i] = np.where(has, n_leads - 1 - np.argmax(mask[:, ::-1], axis=1), -1)
        preset[i, ~has] = Status.NOT_FOUND

    def expand(a: np.ndarray) -> np.ndarray:
        """Repeat per-init values over the member and cell rows of each init."""
        return np.repeat(a, rest, axis=1).ravel()

    series = np.broadcast_to(np.arange(n_series, dtype=np.int64), (k, n_series)).ravel().copy()
    jobs = Jobs(series, expand(first), expand(last), np.full(k * n_series, n_leads, dtype=np.int64))
    return jobs, JobLayout((k,), expand(preset), names, n_series)


def _check_same_grid(history: DailyRain, forecast: ForecastRain, spatial_dims: tuple[str, ...]) -> None:
    """Check that the history and the forecast share dims, calendar and cell coordinates (tolerance 1e-6).

    Parameters
    ----------
    history : hcwf.onset.model.DailyRain
        The observed history.
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    spatial_dims : tuple of str
        The spatial dim names of the forecast.

    Raises
    ------
    ValueError
        The spatial dims, the calendars, the grid sizes, the ids of a string-valued dim or the numeric coordinates
        (to 1e-6) differ.
    """
    if tuple(history.dims[:-1]) != tuple(spatial_dims):
        raise ValueError(f"history dims {history.dims[:-1]} differ from the forecast's spatial dims {tuple(spatial_dims)}")
    if history.calendar != forecast.calendar:
        raise ValueError(f"history calendar {history.calendar!r} differs from the forecast's {forecast.calendar!r}")
    for i, dim in enumerate(spatial_dims):
        h, f = history.tenths.shape[i], forecast.tenths.shape[forecast.dims.index(dim)]
        if h != f:
            raise ValueError(f"history and forecast grids differ along {dim}: {h} vs {f} cells")
        if dim in history.coords and dim in forecast.coords:
            h_coords, f_coords = np.asarray(history.coords[dim]), np.asarray(forecast.coords[dim])
            if h_coords.dtype.kind in "USO":   # unit ids: the same ids in the same order
                if not (h_coords == f_coords).all():
                    raise ValueError(f"history and forecast differ along {dim}: the ids do not match")
            elif not np.allclose(h_coords.astype(float), f_coords.astype(float), rtol=0, atol=1e-6):
                raise ValueError(f"history and forecast grids differ along {dim}: coordinates do not match to 1e-6")


def build_stitched_jobs(definition: Definition, history: DailyRain, forecast: ForecastRain,
                        lead_window: tuple[int, int] = (1, 30), restrict_to_season: bool = True,
                        season_climatology: xr.Dataset | None = None) -> tuple[StitchedJobs, StitchedLayout]:
    """Build one stitched job per (season, init, [member], cell) over observed history followed by the forecast.

    The forecast row of a job is the flat (init, [member], cell) series index and its history row the cell. ``split``
    is the history index of lead 1's date label (init for ``lead_label`` ``"start"``, init + 1 for ``"end"``); the
    history must contain that label. The season year is chosen as in `build_forecast_jobs`, with the window needing
    one lead of ``lead_window`` inside it; candidates then run from the season start (in the history, or lead 1's day
    onward when the season starts later) to the earlier of the season end and the last lead of ``lead_window``. A season
    start before the history gives ``INSUFFICIENT_DATA``, a season missing from the init's years ``NOT_FOUND``. With
    ``restrict_to_season`` false there is one season, ``"window"``, whose candidates are the leads of ``lead_window``.
    `~hcwf.onset.model.DerivedSeasons` with ``restrict_to_season`` true need ``season_climatology``: the same rules apply with each cell's
    own season window (the season start is then per cell, and a cell without that season is ``NOT_FOUND``).
    ``lead_window`` must start at 1. ``data_end`` is ``split`` plus the number of leads. See spec section 13.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition.
    history : hcwf.onset.model.DailyRain
        The observed rain on the forecast grid; its date labels must be strictly increasing.
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    lead_window : tuple of (int, int), default (1, 30)
        The first and last candidate lead, 1-based; the first must be 1.
    restrict_to_season : bool, default True
        Limit the candidates to the seasons of the definition.
    season_climatology : xarray.Dataset, optional
        Required for `~hcwf.onset.model.DerivedSeasons` when ``restrict_to_season`` is true.

    Returns
    -------
    tuple of (hcwf.onset.model.StitchedJobs, StitchedLayout)
        The jobs, ordered (season, init, [member], cell), and their layout.

    Raises
    ------
    ValueError
        ``lead_window`` does not start at 1 (see `build_forecast_jobs`), the history and forecast grids differ, the history dates are not strictly
        increasing, or the history does not reach the date of lead 1 of some init.
    """
    n_leads = forecast.tenths.shape[-1]
    _, hi = _lead_window_bounds(lead_window, n_leads, from_first=True)
    n_init = forecast.tenths.shape[0]
    series_dims = forecast.dims[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims = series_dims[n_lead_dims:]
    _check_same_grid(history, forecast, spatial_dims)
    n_cell = int(np.prod(forecast.tenths.shape[n_lead_dims:-1], dtype=np.int64))
    n_member = forecast.tenths.shape[1] if n_lead_dims == 2 else 1
    n_series = n_init * n_member * n_cell
    hist_keys = cal.label_keys(history.year, history.month, history.day)
    if np.any(np.diff(hist_keys) <= 0):
        raise ValueError("history date labels must be strictly increasing")
    label_y, label_m, label_d = _label_table(forecast, n_leads)
    keys = cal.label_keys(label_y, label_m, label_d)
    pos = np.searchsorted(hist_keys, keys[:, 0])
    reached = (pos < len(hist_keys)) & (hist_keys[np.minimum(pos, len(hist_keys) - 1)] == keys[:, 0])
    if not reached.all():
        bad = int(np.argmax(~reached))
        raise ValueError(f"history does not reach {int(keys[bad, 0])}, the date label of lead 1 of init {bad}")
    split = pos.astype(np.int64)
    allowed = np.zeros(n_leads, dtype=bool)
    allowed[:hi] = True
    if not restrict_to_season:
        names = ["window"]
        first = np.zeros((1, n_init), dtype=np.int64)
        last = np.full((1, n_init), hi - 1, dtype=np.int64)
        has = np.ones((1, n_init), dtype=bool)
        start_hist = np.full((1, n_init), -1, dtype=np.int64)
    else:
        if isinstance(definition.seasons, DerivedSeasons):
            names, first, last, has, start_hist = _derived_stitched_windows(
                definition, forecast, hist_keys, keys, allowed, season_climatology)
        else:
            names = [season.name for season in definition.seasons]
            k = len(names)
            first = np.zeros((k, n_init), dtype=np.int64)
            last = np.full((k, n_init), -1, dtype=np.int64)
            has = np.zeros((k, n_init), dtype=bool)
            start_hist = np.full((k, n_init), -1, dtype=np.int64)
            for i, season in enumerate(definition.seasons):
                wraps = (season.end.month, season.end.day) < (season.start.month, season.start.day)
                done = np.zeros(n_init, dtype=bool)
                for shift in (-1, 0, 1):
                    year = np.asarray(forecast.init_year, dtype=np.int64) + shift
                    low = _bound_key(forecast.calendar, year, season.start)
                    high = _bound_key(forecast.calendar, year + wraps, season.end)
                    inside = (keys >= low) & (keys <= high) & allowed[None, :]
                    take = inside.any(axis=1) & ~done
                    first[i, take] = np.argmax(inside, axis=1)[take]
                    last[i, take] = (n_leads - 1 - np.argmax(inside[:, ::-1], axis=1))[take]
                    has[i, take] = True
                    before = take & (low[:, 0] <= keys[:, 0])
                    at = np.searchsorted(hist_keys, low[:, 0])
                    found = (at < len(hist_keys)) & (hist_keys[np.minimum(at, len(hist_keys) - 1)] == low[:, 0])
                    start_hist[i, before] = np.where(found, at, -2)[before]
                    done |= take
    k = len(names)
    split_b = split[None, :, None] if first.ndim == 3 else split[None, :]
    in_history = start_hist != -1
    start = np.where(in_history, start_hist, split_b + first)
    end = split_b + last
    preset = np.full(first.shape, NO_PRESET, dtype=np.int8)
    preset[~has] = Status.NOT_FOUND
    preset[has & (start_hist == -2)] = Status.INSUFFICIENT_DATA
    skip = preset >= 0
    start = np.where(skip, 0, start)
    end = np.where(skip, -1, end)

    def expand(a: np.ndarray) -> np.ndarray:
        """Repeat per-init (or per-init-and-cell) values over the member and cell rows of each init."""
        return _expand_rows(a, n_member, n_cell)

    series = np.broadcast_to(np.arange(n_series, dtype=np.int64), (k, n_series)).ravel().copy()
    jobs = StitchedJobs(cell=series % n_cell, series=series, split=expand(np.broadcast_to(split, (k, n_init))),
                        start=expand(start), end=expand(end),
                        data_end=expand(np.broadcast_to(split + n_leads, (k, n_init))))
    layout = StitchedLayout((k,), expand(preset), names, n_series, n_init, n_member, n_cell)
    return jobs, layout


def _derived_stitched_windows(definition: Definition, forecast: ForecastRain, hist_keys: np.ndarray, keys: np.ndarray,
                              allowed: np.ndarray, season_climatology: xr.Dataset | None):
    """Return the first and last lead, presence and history start of the derived seasons of a stitched search.

    ``start_hist`` is the history index of the season start when it is on or before lead 1's day (-2 when the history
    lacks that day), else -1. A missing ``season_climatology`` raises `ValueError` (in `_require_climatology`).

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        A definition with `~hcwf.onset.model.DerivedSeasons`.
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    hist_keys : numpy.ndarray of int64, shape (time,)
        The label keys of the history.
    keys : numpy.ndarray of int64, shape (init, lead)
        The label keys of the leads.
    allowed : numpy.ndarray of bool, shape (lead,)
        The leads of the lead window.
    season_climatology : xarray.Dataset, optional
        The season climatology; required.

    Returns
    -------
    tuple of (list of str, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The season names, then ``first``, ``last`` (int64), ``has`` (bool) and ``start_hist`` (int64), each shape
        (K, init, cell).

    """
    climatology = _require_climatology(season_climatology)
    spatial_dims, _, n_cell = _derived_dims(forecast)
    low, high, present, names = _derived_windows(definition, forecast.calendar, forecast.init_year, climatology,
                                                 spatial_dims, n_cell)
    first, last, has, chosen = _pick_derived(keys, allowed, low, high, present)
    start_hist = np.full(first.shape, -1, dtype=np.int64)
    for i in range(3):
        before = chosen[i] & (low[i] <= keys[None, :, :1])
        at = np.searchsorted(hist_keys, low[i])
        found = (at < len(hist_keys)) & (hist_keys[np.minimum(at, len(hist_keys) - 1)] == low[i])
        start_hist[before] = np.where(found, at, -2)[before]
    return names, first, last, has, start_hist


def _prior_search(plan: plan_mod.Plan, history: DailyRain, jobs: StitchedJobs, layout: StitchedLayout,
                  per_cell: dict[str, np.ndarray], backend: str,
                  policy: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Search the history alone per (season, init, cell) and return the onset day and whether it is ``ONSET``.

    Candidates run to ``min(season end, split - 1)`` with ``data_end = split``.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled plan.
    history : hcwf.onset.model.DailyRain
        The observed history.
    jobs : hcwf.onset.model.StitchedJobs
        The stitched jobs.
    layout : StitchedLayout
        The layout of ``jobs``.
    per_cell : dict of str to numpy.ndarray of float64
        The history-only thresholds (first window), shape (cell,).
    backend : str
        The backend name.
    policy : str, optional
        The run's missing-rain policy; default the one ingest recorded on the history. ``"zero"`` ignores the missing
        flags.

    Returns
    -------
    tuple of (numpy.ndarray of int64, numpy.ndarray of bool)
        The onset day index in the history (-1 for none) and whether the search ended in ``ONSET``, each shape
        (season, init, cell).
    """
    policy = _missing_policy(history, policy)
    k, n_init, n_cell = len(layout.season_names), layout.n_init, layout.n_cell
    rest = layout.n_series // max(n_init, 1)

    def per_init_cell(a: np.ndarray) -> np.ndarray:
        """Pick the first member's row of every (season, init) and flatten."""
        return a.reshape(k, n_init, rest)[:, :, :n_cell].ravel()

    split = per_init_cell(jobs.split)
    start, end = per_init_cell(jobs.start), np.minimum(per_init_cell(jobs.end), split - 1)
    preset = per_init_cell(layout.preset).copy()
    preset[(preset < 0) & (end < start)] = Status.NOT_FOUND
    pjobs = Jobs(per_init_cell(jobs.cell), start, end, split)
    n_time = history.tenths.shape[-1]
    day, status = _run(plan, history.tenths.reshape(n_cell, n_time), _flat_missing(history, policy, n_cell),
                       per_cell, pjobs, preset, backend)
    return day.reshape(k, n_init, n_cell), (status == Status.ONSET).reshape(k, n_init, n_cell)


def _run_stitched(plan: plan_mod.Plan, history: DailyRain, forecast: ForecastRain, thresholds: dict[str, np.ndarray],
                  per_cell: dict[str, np.ndarray], jobs: StitchedJobs, layout: StitchedLayout,
                  backend: str, windows: np.ndarray | None = None,
                  policy: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decide ``PRIOR`` from the history alone, then run the stitched search for the other jobs.

    ``policy`` is the run's missing-rain policy for the history and the forecast alike (default: the forecast's recorded
    one); under ``"zero"`` the days already hold 0 and no flag is passed, as in the plain search. The ``PRIOR`` search
    is skipped for curve rules, whose onset is a minimum over the whole window and so can move when the window grows
    (spec section 13).

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled plan.
    history : hcwf.onset.model.DailyRain
        The observed history.
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    thresholds : dict of str to numpy.ndarray of float64
        The per-series thresholds, shape (series,) or (series, W).
    per_cell : dict of str to numpy.ndarray of float64
        The history-only thresholds (first window), shape (cell,).
    jobs : hcwf.onset.model.StitchedJobs
        The stitched jobs.
    layout : StitchedLayout
        The layout of ``jobs``.
    backend : str
        The backend name.
    windows : numpy.ndarray of int64, optional
        The start leads of ``(series, W)`` thresholds.
    policy : str, optional
        The missing-rain policy; default the forecast's recorded one.

    Returns
    -------
    tuple of (numpy.ndarray of int64, numpy.ndarray of int8)
        The onset day index of each job (-1 where none; an index into history followed by forecast) and its `~hcwf.onset.model.Status`
        code, each shape (job,).
    """
    module = _load_backend(backend)
    extra = {} if windows is None else {"amount_windows": windows}
    n_cell, n_time = layout.n_cell, history.tenths.shape[-1]
    n_leads = forecast.tenths.shape[-1]
    n_series = layout.n_series
    policy = _missing_policy(forecast, policy)
    flat_h = history.tenths.reshape(n_cell, n_time), _flat_missing(history, policy, n_cell)
    flat_f = forecast.tenths.reshape(-1, n_leads), _flat_missing(forecast, policy, n_series)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=flat_f[0].shape[0],
                                                exact=resolution_of(flat_f[0]) == "exact")
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.where(layout.preset >= 0, layout.preset, Status.NOT_FOUND).astype(np.int8)
    active = layout.preset < 0
    bad = active & invalid[jobs.series]
    status[bad] = Status.INSUFFICIENT_DATA
    active &= ~bad
    if plan.family == plan_mod.FIRST_PASSAGE and active.any():
        k, n_init, n_member = len(layout.season_names), layout.n_init, layout.n_member
        prior_day, prior_hit = _prior_search(plan, history, jobs, layout, per_cell, backend, policy)
        shape = (k, n_init, n_member, n_cell)
        hit = np.broadcast_to(prior_hit[:, :, None, :], shape).ravel() & active
        day[hit] = np.broadcast_to(prior_day[:, :, None, :], shape).ravel()[hit]
        status[hit] = Status.PRIOR
        active &= ~hit
    idx = np.nonzero(active)[0]
    if len(idx) == 0:
        return day, status
    idx = idx[np.argsort(jobs.series[idx], kind="stable")]
    sorted_series = jobs.series[idx]
    per_chunk = max(1, MAX_TABLE_CELLS // max(n_leads, 1))
    edges = np.arange(0, flat_f[0].shape[0] + per_chunk, per_chunk)
    bounds = np.searchsorted(sorted_series, edges)
    for c in range(len(edges) - 1):
        sel = idx[bounds[c]:bounds[c + 1]]
        if len(sel) == 0:
            continue
        s0, s1 = edges[c], min(edges[c + 1], flat_f[0].shape[0])
        sub = StitchedJobs(jobs.cell[sel], jobs.series[sel] - s0, jobs.split[sel], jobs.start[sel], jobs.end[sel],
                           jobs.data_end[sel])
        sub_amounts = {key: value[s0:s1] for key, value in amounts.items()}
        out = module.detect_stitched(plan, flat_h[0], flat_h[1], flat_f[0][s0:s1], flat_f[1][s0:s1], sub_amounts, sub,
                                     **extra)
        day[sel] = out.day
        status[sel] = out.status
    return day, status


def lead_counts(lead: np.ndarray, max_lead: int, axis: int = 0) -> np.ndarray:
    """Count how many of the leads equal each lead ``0..max_lead`` along an axis.

    Leads at or below 0 count at lead 0 and ``NaN`` counts nowhere; this is the engine's ``onset_count`` of an
    ensemble.

    Parameters
    ----------
    lead : numpy.ndarray of float64
        The onset leads (``NaN`` where none).
    max_lead : int
        The largest lead counted.
    axis : int, default 0
        The axis of ``lead`` to count over (for example the member axis); it is replaced by the lead axis, last.

    Returns
    -------
    numpy.ndarray of int64
        The counts, with the lead axis ``0..max_lead`` last.
    """
    leads = np.arange(max_lead + 1)
    return (np.maximum(np.asarray(lead, dtype=float), 0)[..., None] == leads).sum(axis=axis)


def _ensemble_summary(lead: np.ndarray, status: np.ndarray, dims: tuple[str, ...], coords: dict,
                      max_lead: int, onset_count: bool = False) -> dict[str, xr.DataArray]:
    """Return ``members_by_status`` and ``onset_probability`` of an ensemble, optionally ``onset_count`` and ``members``.

    With ``onset_count``, also ``onset_count`` (members with their onset on each lead ``0..max_lead``) and ``members``.

    Parameters
    ----------
    lead : numpy.ndarray of float64, shape (init, member, season, [start], *spatial)
        The onset leads (``NaN`` where none).
    status : numpy.ndarray of int8
        The `~hcwf.onset.model.Status` codes, same shape as ``lead``.
    dims : tuple of str
        The dim names of ``lead``.
    coords : dict
        The coordinates of the dataset; those of the output dims are kept.
    max_lead : int
        The largest lead of the probability curve.
    onset_count : bool, default False
        Also return the count of members per lead and the number of members.

    Returns
    -------
    dict of str to xarray.DataArray
        ``members_by_status``, ``onset_probability`` and, with ``onset_count``, ``onset_count`` and ``members``.
    """
    codes = np.array([s.value for s in Status], dtype=np.int8)
    counts = np.stack([(status == c).sum(axis=1) for c in codes]).astype(np.int64)
    out_dims = ("init_time", *dims[2:])
    head = 3 + ("start" in dims)
    shared = {d: v for d, v in coords.items() if d in out_dims}
    by_status = xr.DataArray(
        counts, dims=("status_code", *out_dims),
        coords={"status_code": codes, "status_name": ("status_code", [s.name for s in Status]), **shared},
        attrs={"long_name": "number of members in each onset status"})
    leads = np.arange(max_lead + 1)
    reached = (lead[..., None] <= leads).sum(axis=1)
    usable = (status != Status.INSUFFICIENT_DATA).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(usable[..., None] > 0, reached / usable[..., None], np.nan)
    share = np.moveaxis(share, -1, head - 1)
    probability = xr.DataArray(
        share, dims=("init_time", *dims[2:head], "lead", *dims[head:]),
        coords={"lead": leads, **shared},
        attrs={"long_name": "share of members whose onset is on or before this lead",
               "comment": "PRIOR and back-dated onsets (onset_lead <= 0) count at lead 0; members with "
                          "INSUFFICIENT_DATA are excluded from the denominator; CENSORED and NOT_FOUND members "
                          "stay in it and never count as reached"})
    out = {"members_by_status": by_status, "onset_probability": probability}
    if onset_count:
        on_lead = np.moveaxis(lead_counts(lead, max_lead, axis=1), -1, head - 1)
        out["onset_count"] = xr.DataArray(
            on_lead.astype(np.int64), dims=probability.dims, coords={"lead": leads, **shared},
            attrs={"long_name": "number of members whose onset is on this lead",
                   "comment": "PRIOR and back-dated onsets (onset_lead <= 0) count at lead 0"})
        out["members"] = xr.DataArray(counts.sum(axis=0), dims=out_dims, coords=shared,
                                      attrs={"long_name": "number of members"})
    return out


def _forecast_starts(search_start: Mapping, forecast: ForecastRain, n_leads: int, n_season: int) -> list[np.ndarray]:
    """Return, per variant, the first lead index on or after the entry's date for every forecast job.

    Jobs are ordered (season, init, [member], cell).

    Parameters
    ----------
    search_start : collections.abc.Mapping
        The ``search_start`` entries by name (see `detect_forecast`).
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    n_leads : int
        The number of leads.
    n_season : int
        The number of seasons.

    Returns
    -------
    list of numpy.ndarray of int64
        One array of shape (season * init * member * cell,) per entry, -1 where there is no restriction.
    """
    series_dims = forecast.dims[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims, spatial_shape = series_dims[n_lead_dims:], forecast.tenths.shape[n_lead_dims:-1]
    n_cell = int(np.prod(spatial_shape, dtype=np.int64))
    n_member = forecast.tenths.shape[1] if n_lead_dims == 2 else 1
    lead_keys = cal.label_keys(*_label_table(forecast, n_leads))
    out = []
    for entry in search_start.values():
        wanted = _entry_keys(entry, np.asarray(forecast.init_year, dtype=np.int64), "init_time", spatial_dims,
                             spatial_shape, forecast.calendar)
        first = np.where(wanted >= 0, (lead_keys[:, :, None] < wanted[:, None, :]).sum(axis=1), -1)
        per_series = np.broadcast_to(first[:, None, :], (len(first), n_member, n_cell)).reshape(-1)
        out.append(np.tile(per_series, n_season))
    return out


def detect_forecast(definition: Definition, forecast: ForecastRain, *, thresholds=None,
                    lead_window: tuple[int, int] = (1, 30), restrict_to_season: bool = True,
                    backend: str = "numba", attrs: dict | None = None,
                    history: DailyRain | None = None, threshold_windows=None,
                    season_climatology: xr.Dataset | None = None, horizon: str = DEFAULT_HORIZON,
                    missing: str | None = None, search_start: Mapping[str, str | xr.DataArray] | None = None,
                    onset_count: bool = False) -> xr.Dataset:
    """Detect onset in every (init, [member], season, cell) of a forecast.

    Returns dims ``(init_time, [member], season, *spatial)`` with ``onset_lead`` (float64: the 1-based lead of the
    onset, ``NaN`` if none), ``onset_date`` (the date label of that day in the forecast's calendar) and ``status``. See
    `build_forecast_jobs` for how candidates are chosen; thresholds are per cell and broadcast over inits and members.

    With ``history`` (observed rainfall on the forecast grid) the searched series is the history up to the day before
    lead 1 followed by the forecast, as in spec section 13: `build_stitched_jobs` chooses the candidates, onsets
    decided in the history are ``PRIOR``, and ``onset_lead`` is 0 or negative for onsets before lead 1. Without it the
    forecast is searched alone and ``onset_lead`` is at least 1. ``attrs["forecast_history"]`` is ``"observed"`` or
    ``"none"``. When the forecast has members, ``members_by_status`` and ``onset_probability`` summarise them per
    init.

    A threshold with a ``"window"`` dim (`~hcwf.onset.climatology.compute_thresholds_from_forecasts` with windows) is
    applied per candidate by the window its lead falls in; the window start leads come from ``threshold_windows`` or the
    threshold's ``window_first`` coordinate (spec section 13).

    `~hcwf.onset.model.DerivedSeasons` with ``restrict_to_season`` true need ``season_climatology`` (``n_seasons``, ``onset_doy``,
    ``cessation_doy`` with dims ``(season, *spatial)`` on the forecast's grid, as for `detect_observed`): candidates
    are then limited to each cell's own season, the season coordinate is the climatology's (1, 2, ...), and a cell
    without that season is ``NOT_FOUND``. Stitched, curve rules accumulate from the cell's season start.

    Run options (not part of the definition; recorded in attrs), as for `detect_observed`: ``horizon`` ``"truncate"``
    (what blending does with forecasts) lets a veto see only the leads that exist; ``missing`` overrides the policy
    ingest recorded (``"drop"`` is for observed rain only); ``search_start`` adds variants (dates over ``init_time`` and
    the spatial dims, or ``"MM-DD"`` of each init's year) searching from the first lead on or after the date, with a
    ``"start"`` dim after season. They need ``forecast_history`` ``"none"``. ``onset_count`` adds, for an ensemble,
    ``onset_count`` (members with their onset on each lead ``0..lead_window[1]``, per start) and ``members``.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition.
    forecast : hcwf.onset.model.ForecastRain
        The forecast, dims ``(init_time, [member], *spatial, lead)``.
    thresholds : dict of str to (xarray.DataArray or numpy.ndarray or float), optional
        Threshold values by name, in mm per cell; a `xarray.DataArray` may have a ``"window"`` dim.
    lead_window : tuple of (int, int), default (1, 30)
        The first and last candidate lead, 1-based and inclusive.
    restrict_to_season : bool, default True
        Limit the candidates to the seasons of the definition.
    backend : {"reference", "numba", "gpu"}, default "numba"
        The detection backend.
    attrs : dict, optional
        Merged into the dataset attrs.
    history : hcwf.onset.model.DailyRain, optional
        Observed rain on the forecast grid to stitch before the forecast.
    threshold_windows : sequence of int, optional
        The start lead of each window of a windowed threshold.
    season_climatology : xarray.Dataset, optional
        Required for `~hcwf.onset.model.DerivedSeasons` when ``restrict_to_season`` is true.
    horizon : {"censor", "truncate", "reject"}, default "censor"
        What to do with a span that runs past the data.
    missing : str, optional
        The missing-rain policy; default the one recorded on ``forecast``.
    search_start : dict of str to (str or xarray.DataArray), optional
        Extra search-start variants by name.
    onset_count : bool, default False
        Add ``onset_count`` and ``members`` for an ensemble.

    Returns
    -------
    xarray.Dataset
        ``onset_lead``, ``onset_date`` and ``status`` over ``(init_time, [member], season, [start], *spatial)``, plus
        the ensemble summaries when the forecast has members.

    Raises
    ------
    ValueError
        The policy is ``"drop"``, ``search_start`` is combined with ``history``, ``onset_count`` is set without a member
        dim, or a ``search_start`` name is ``"base"``.
    """
    _check_search_start(search_start)
    policy = _missing_policy(forecast, missing)
    if policy == "drop":
        raise ValueError("missing 'drop' removes whole cell-years of observed rain; use 'zero', 'keep' or 'flag' "
                         "for forecasts")
    if search_start and history is not None:
        raise ValueError("search_start needs forecast_history 'none' (no history)")
    if onset_count and "member" not in forecast.dims:
        raise ValueError("onset_count needs a forecast with a member dim")
    plan = plan_mod.compile(definition, horizon=horizon, missing=policy)
    n_leads = forecast.tenths.shape[-1]
    series_dims = forecast.dims[:-1]
    series_shape = forecast.tenths.shape[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims, spatial_shape = series_dims[n_lead_dims:], series_shape[n_lead_dims:]
    per_series = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, series_shape)
    windows = _threshold_windows(plan, thresholds, threshold_windows)
    n_variants = 1
    if history is None:
        jobs, layout = build_forecast_jobs(definition, forecast, lead_window, restrict_to_season,
                                            season_climatology)
        preset = layout.preset
        if search_start:
            jobs, preset = _stack_starts(jobs, preset, _forecast_starts(
                search_start, forecast, n_leads, len(layout.season_names)))
            n_variants = len(search_start) + 1
        day, status = _run(plan, forecast.tenths.reshape(layout.n_series, n_leads),
                           _flat_missing(forecast, policy, layout.n_series), per_series, jobs, preset, backend,
                           windows)
        split = np.zeros(len(day), dtype=np.int64)
        hist_labels = None
    else:
        jobs, layout = build_stitched_jobs(definition, history, forecast, lead_window, restrict_to_season,
                                              season_climatology)
        per_cell = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape,
                                      first_window_only=True)
        day, status = _run_stitched(plan, history, forecast, per_series, per_cell, jobs, layout, backend, windows,
                                    policy)
        split = jobs.split
        hist_labels = (history.year, history.month, history.day)
    k = len(layout.season_names)
    full = (k, *series_shape)
    if search_start:
        full, to = (n_variants, *full), (n_lead_dims + 1, n_lead_dims)
        day, status, split = (np.moveaxis(a.reshape(full), (0, 1), to) for a in (day, status, split))
    else:
        day, status, split = (np.moveaxis(a.reshape(full), 0, n_lead_dims) for a in (day, status, split))
    found = day >= 0
    lead = np.where(found, day - split + 1, np.nan).astype(np.float64)
    label_y, label_m, label_d = _label_table(forecast, n_leads)
    init_index = np.arange(series_shape[0]).reshape((-1,) + (1,) * (day.ndim - 1))
    col = np.clip(np.where(found, day - split, 0), 0, n_leads - 1)
    year, month, dayn = label_y[init_index, col], label_m[init_index, col], label_d[init_index, col]
    if hist_labels is not None:
        in_hist = found & (day < split)
        h = np.where(in_hist, day, 0)
        year, month, dayn = (np.where(in_hist, labels[h], fc) for labels, fc in zip(hist_labels, (year, month, dayn)))
    dates = _dates_from_labels(forecast.calendar, year, month, dayn, found)
    dims = (*series_dims[:n_lead_dims], "season", *(("start",) if search_start else ()), *spatial_dims)
    coords = {"season": layout.season_names, **_spatial_coords(dims, forecast.coords)}
    if search_start:
        coords["start"] = [BASE_START, *search_start]
    variables = {"onset_lead": (dims, lead), "onset_date": (dims, dates), "status": _status_var(status, dims)}
    ds = xr.Dataset(variables, coords=coords)
    if "member" in series_dims:
        summary = _ensemble_summary(lead, status, dims, coords, lead_window[1], onset_count)
        for name, var in summary.items():
            ds[name] = var
    ds.attrs.update({"definition_id": definition.id, "definition_version": definition.version, "backend": backend,
                     "lead_label": forecast.lead_label,
                     "forecast_history": "none" if history is None else "observed",
                     **_option_attrs(forecast, horizon, policy, search_start)})
    ds.attrs.update(attrs or {})
    return ds

