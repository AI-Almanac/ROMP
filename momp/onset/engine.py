"""Turn seasons and forecast leads into onset jobs, run a backend, and package the results as xarray Datasets."""
from __future__ import annotations

import datetime as _dt
import importlib
import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from momp.onset import calendar as cal
from momp.onset import plan as plan_mod
from momp.onset.model import DailyRain, Definition, DerivedSeasons, ForecastRain, Jobs, Status, StitchedJobs

NO_PRESET = -1
MAX_TABLE_CELLS = 20_000_000
"""Series x time cells whose prefix tables are built at one time."""

_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
_BACKEND_MODULES = {"reference": "reference", "numba": "numba_cpu", "gpu": "gpu"}
_CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


@dataclass
class JobLayout:
    """How a flat Jobs array maps back to output dims.

    shape is the outer shape of the jobs (the job array reshapes to
    shape + (n_series,)); preset holds a Status code for jobs decided without
    a search (NO_PRESET otherwise); season_names label the season axis.
    """

    shape: tuple[int, ...]
    preset: np.ndarray
    season_names: list[str]
    n_series: int


@dataclass
class StitchedLayout(JobLayout):
    """JobLayout of stitched jobs, which are ordered (season, init, [member], cell).

    n_init, n_member (1 without members) and n_cell give the sizes of those axes;
    n_series = n_init * n_member * n_cell.
    """

    n_init: int = 0
    n_member: int = 1
    n_cell: int = 0


def _load_backend(name: str):
    """Import momp.onset.backends.<module> for a backend name (reference, numba, gpu)."""
    if name not in _BACKEND_MODULES:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(_BACKEND_MODULES)}")
    module = f"momp.onset.backends.{_BACKEND_MODULES[name]}"
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"backend {name!r} is not available ({module}): {exc}") from exc


def _threshold_windows(plan: plan_mod.Plan, thresholds, threshold_windows) -> np.ndarray | None:
    """Window start leads for thresholds that have a "window" dim (None when none do).

    threshold_windows overrides the starts; otherwise they come from the
    window_first coordinate that compute_thresholds_from_forecasts writes.
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
    """Per-series float arrays for the thresholds the plan needs, broadcast from per-cell values.

    A threshold with a "window" dim becomes (series, W); first_window_only keeps
    just its first column (used for history-only searches, whose candidates all
    lie before lead 1).
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
    """Run the backend over the jobs that need a search, chunked over series; return (day, status) per job.

    windows are the start leads of (series, W) thresholds; a job's lead is its day index plus 1.
    """
    extra = {} if windows is None else {"amount_windows": windows}
    module = _load_backend(backend)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=tenths.shape[0])
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
        sub = Jobs(jobs.series[sel] - s0, jobs.start[sel], jobs.end[sel], jobs.data_end[sel])
        sub_amounts = {key: value[s0:s1] for key, value in amounts.items()}
        out = module.detect(plan, tenths[s0:s1], missing[s0:s1], sub_amounts, sub, **extra)
        day[sel] = out.day
        status[sel] = out.status
    return day, status


def _climatology_doys(definition: Definition, climatology: xr.Dataset, spatial_dims: tuple[str, ...], n_series: int):
    """Per-cell season bounds of a season climatology: (start doy, end doy) as (K, cells), n_seasons (cells) or None, names.

    The start and end variables are the ones the definition's DerivedSeasons bounds refer to.
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
    """First and last day indices (K, Y, S) plus a "no such season" mask for DerivedSeasons."""
    seasons = definition.seasons
    onset, cessation, n_seasons, names = _climatology_doys(definition, climatology, spatial_dims, n_series)
    n_k = onset.shape[0]
    jan1 = cal.find_days(keys, years, 1, 1)
    firsts, lasts, none = [], [], []
    for k in range(n_k):
        s_doy = onset[k]
        e_doy = cessation[k]
        first, last = cal.doy_bounds(keys, years, s_doy, e_doy, seasons.start.offset_days, seasons.end.offset_days)
        absent = ~(np.isfinite(s_doy) & np.isfinite(e_doy))
        if n_seasons is not None:
            absent |= n_seasons <= k
        rescue = (first < 0) & ~absent[None, :] & (jan1[:, None] >= 0)
        start_rescued = jan1[:, None] + np.rint(np.nan_to_num(s_doy))[None, :].astype(np.int64) - 1 + seasons.start.offset_days
        first = np.where(rescue, start_rescued, first)
        last = np.where(rescue & (last < 0), np.iinfo(np.int64).max // 4, last)
        firsts.append(first)
        lasts.append(last)
        none.append(np.broadcast_to(absent[None, :], first.shape))
    return np.stack(firsts), np.stack(lasts), np.stack(none), names


def build_observed_jobs(definition: Definition, rain: DailyRain, years=None,
                        season_climatology: xr.Dataset | None = None) -> tuple[Jobs, JobLayout]:
    """One job per (year, season, cell), in that C order, with the cells as backend series.

    Fixed seasons use calendar.fixed_bounds; DerivedSeasons use calendar.doy_bounds
    on season_climatology (n_seasons, onset_doy, cessation_doy with dims (season,
    *spatial)), which is then required. A season start outside the data presets the
    job to INSUFFICIENT_DATA; a cell without that derived season presets NOT_FOUND.
    A season end past the data is clamped to the last day (the horizon flag then
    decides). data_end is the length of the time axis.
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
            f, l = cal.fixed_bounds(keys, years, season.start, season.end)
            firsts.append(np.broadcast_to(f[:, None], (len(years), n_series)))
            lasts.append(np.broadcast_to(l[:, None], (len(years), n_series)))
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


def _is_leap(calendar: str, year: np.ndarray) -> np.ndarray:
    """Whether each year has a Feb 29 in the given calendar."""
    year = np.asarray(year, dtype=np.int64)
    if calendar in ("noleap", "365_day", "360_day"):
        return np.zeros(year.shape, dtype=bool)
    if calendar in ("all_leap", "366_day"):
        return np.ones(year.shape, dtype=bool)
    if calendar == "julian":
        return year % 4 == 0
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def _day_of_year(calendar: str, year, month, day) -> np.ndarray:
    """1-based day of year of year-month-day labels in the calendar."""
    month = np.asarray(month, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    if calendar == "360_day":
        return (month - 1) * 30 + day
    return _CUM_DAYS[month - 1] + day + (_is_leap(calendar, year) & (month > 2))


def _year_length(calendar: str, year) -> np.ndarray:
    """Number of days in each year of the calendar."""
    if calendar == "360_day":
        return np.full(np.shape(year), 360)
    return 365 + _is_leap(calendar, year).astype(np.int64)


def _add_days(calendar: str, year, month, day, offsets: np.ndarray):
    """Labels of (year, month, day) + offsets days in the calendar: three int arrays shaped (n_dates, n_offsets)."""
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
    """datetime64[ns] (NaT where not valid) for standard calendars, cftime objects (None) otherwise."""
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
    """Status codes as an int8 DataArray with CF flag_values and flag_meanings."""
    return xr.DataArray(values.astype(np.int8), dims=dims, attrs={
        "long_name": "onset search outcome",
        "flag_values": np.array([s.value for s in Status], dtype=np.int8),
        "flag_meanings": " ".join(s.name for s in Status)})


def _spatial_coords(dims: tuple[str, ...], coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The coordinate arrays known for the given dims."""
    return {d: coords[d] for d in dims if d in coords}


def detect_observed(definition: Definition, rain: DailyRain, *, thresholds=None, years=None,
                    season_climatology: xr.Dataset | None = None, backend: str = "numba",
                    attrs: dict | None = None) -> xr.Dataset:
    """Detect onset in every (year, season, cell).

    Returns a Dataset with dims (year, season, *spatial) and variables onset_date
    (datetime64[ns], or cftime objects for non-standard calendars), onset_index
    (index on the time axis, -1 if none) and status (int8 Status codes).
    thresholds map names to per-cell mm values (DataArray over the spatial dims,
    or an ndarray). attrs are merged into the dataset attrs.
    """
    plan = plan_mod.compile(definition)
    jobs, layout = build_observed_jobs(definition, rain, years, season_climatology)
    years = np.unique(rain.year) if years is None else np.asarray(years, dtype=np.int64)
    spatial_dims = rain.dims[:-1]
    spatial_shape = rain.tenths.shape[:-1]
    n_time = rain.tenths.shape[-1]
    per_series = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape,
                                    first_window_only=True)
    day, status = _run(plan, rain.tenths.reshape(layout.n_series, n_time),
                       rain.missing.reshape(layout.n_series, n_time), per_series, jobs, layout.preset, backend)
    out_shape = (len(years), len(layout.season_names), *spatial_shape)
    index = day.reshape(out_shape)
    safe = np.where(index >= 0, index, 0)
    dates = _dates_from_labels(rain.calendar, rain.year[safe], rain.month[safe], rain.day[safe], index >= 0)
    dims = ("year", "season", *spatial_dims)
    ds = xr.Dataset(
        {"onset_date": (dims, dates), "onset_index": (dims, index.astype(np.int64)),
         "status": _status_var(status.reshape(out_shape), dims)},
        coords={"year": years, "season": layout.season_names, **_spatial_coords(spatial_dims, rain.coords)})
    ds.attrs.update({"definition_id": definition.id, "definition_version": definition.version, "backend": backend})
    ds.attrs.update(attrs or {})
    return ds


def _bound_key(calendar: str, year: np.ndarray, ref) -> np.ndarray:
    """Label key, shaped (n, 1), of a month-day season bound plus its day offset in each of `year`."""
    if ref.offset_days == 0:
        return (year * 10000 + ref.month * 100 + ref.day)[:, None]
    y, m, d = _add_days(calendar, year, np.full_like(year, ref.month), np.full_like(year, ref.day),
                        np.array([ref.offset_days]))
    return cal.label_keys(y, m, d)


def _derived_windows(definition: Definition, calendar: str, init_year: np.ndarray, climatology: xr.Dataset,
                     spatial_dims: tuple[str, ...], n_cell: int):
    """Label-key windows of each derived season per init and cell, for the init's year minus one, the year and the year after.

    Returns (low, high, present, names): low and high are int64 arrays shaped (3, K, n_init, n_cell) holding
    the year*10000 + month*100 + day key of the first and last day of the season that starts in year
    init_year + shift (shift -1, 0, 1); present is (K, n_cell), False where the cell lacks that season.
    The bounds follow calendar.doy_bounds: a season ending before its start day-of-year ends in the following year.
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
        """Label keys of Jan 1 of each year plus per-cell day offsets, shaped (len(year), n_cell)."""
        uniq, inverse = np.unique(offsets, return_inverse=True)
        ones = np.ones_like(year)
        y, m, d = _add_days(calendar, year, ones, ones, uniq)
        return cal.label_keys(y, m, d)[:, inverse.reshape(-1)]

    for k in range(n_k):
        s_doy = np.rint(np.nan_to_num(onset[k], nan=1.0)).astype(np.int64)
        e_doy = np.rint(np.nan_to_num(cessation[k], nan=1.0)).astype(np.int64)
        wraps = (e_doy < s_doy)[None, :]
        start_off = s_doy - 1 + seasons.start.offset_days
        end_off = e_doy - 1 + seasons.end.offset_days
        for i, shift in enumerate((-1, 0, 1)):
            year = init_year + shift
            low[i, k] = shifted_keys(year, start_off)
            high[i, k] = np.where(wraps, shifted_keys(year + 1, end_off), shifted_keys(year, end_off))
    return low, high, present, names


def _pick_derived(keys: np.ndarray, allowed: np.ndarray, low: np.ndarray, high: np.ndarray, present: np.ndarray):
    """Choose each (init, cell)'s season year and the first and last qualifying lead of every derived season.

    keys are the lead label keys (n_init, n_leads); allowed marks the leads of the lead window; low, high and
    present come from _derived_windows. A season takes the earliest of the three years having a lead of the window
    inside it. Returns first, last (K, n_init, n_cell; -1 where none), has (K, n_init, n_cell) and take (3, K, n_init, n_cell),
    the shift each (init, cell) chose.
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
    """Date labels (year, month, day), each (n_init, n_leads): init + k - 1 days for lead_label "start", init + k for "end"."""
    lead_offsets = np.arange(n_leads) + (1 if forecast.lead_label == "end" else 0)
    return _add_days(forecast.calendar, forecast.init_year, forecast.init_month, forecast.init_day, lead_offsets)


def _derived_dims(forecast: ForecastRain):
    """(spatial dims, n_member, n_cell) of a forecast's series dims (init, [member], *spatial)."""
    n_lead_dims = 2 if "member" in forecast.dims[:-1] else 1
    spatial_dims = forecast.dims[n_lead_dims:-1]
    n_cell = int(np.prod(forecast.tenths.shape[n_lead_dims:-1], dtype=np.int64))
    n_member = forecast.tenths.shape[1] if n_lead_dims == 2 else 1
    return spatial_dims, n_member, n_cell


def _expand_rows(a: np.ndarray, n_member: int, n_cell: int) -> np.ndarray:
    """Flatten (season, init) or (season, init, cell) values to (season, init, member, cell) order."""
    k, n_init = a.shape[:2]
    shaped = a.reshape(k, n_init, 1, -1) if a.ndim == 3 else a.reshape(k, n_init, 1, 1)
    return np.broadcast_to(shaped, (k, n_init, n_member, n_cell)).ravel()


def _require_climatology(season_climatology: xr.Dataset | None) -> xr.Dataset:
    """season_climatology, or a ValueError saying derived seasons need one."""
    if season_climatology is None:
        raise ValueError("derived seasons need season_climatology (n_seasons, onset_doy, cessation_doy)")
    return season_climatology


def _derived_forecast_jobs(definition: Definition, forecast: ForecastRain, allowed: np.ndarray,
                           season_climatology: xr.Dataset | None) -> tuple[Jobs, JobLayout]:
    """build_forecast_jobs for DerivedSeasons restricted to each cell's own season."""
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


def build_forecast_jobs(definition: Definition, forecast: ForecastRain, lead_window: tuple[int, int] = (1, 30),
                        restrict_to_season: bool = True, season_climatology: xr.Dataset | None = None
                        ) -> tuple[Jobs, JobLayout]:
    """One job per (season, series) with series = (init, [member], cell) rows over the lead axis.

    Candidates are the leads in lead_window (1-based, inclusive) whose date label
    falls in a window of the season, taken from the earliest of the init's year
    minus one, the init's year and the year after that has any such lead; the
    qualifying leads are contiguous. With none, the job is NOT_FOUND. With
    restrict_to_season=False there is one season, "window", and every lead of
    lead_window is a candidate. DerivedSeasons with restrict_to_season=True need
    season_climatology (as for build_observed_jobs): each cell's windows come from
    its own season bounds, the season year is chosen per (init, cell) as above, and
    a cell without that season is NOT_FOUND, so first, last and the status vary
    by series row. data_end is the number of leads.
    """
    n_leads = forecast.tenths.shape[-1]
    n_init = forecast.tenths.shape[0]
    n_series = int(np.prod(forecast.tenths.shape[:-1], dtype=np.int64))
    rest = n_series // max(n_init, 1)
    lo, hi = max(lead_window[0], 1), min(lead_window[1], n_leads)
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
    """Raise ValueError unless history and forecast share dims, calendar and cell coordinates (tolerance 1e-6)."""
    if tuple(history.dims[:-1]) != tuple(spatial_dims):
        raise ValueError(f"history dims {history.dims[:-1]} differ from the forecast's spatial dims {tuple(spatial_dims)}")
    if history.calendar != forecast.calendar:
        raise ValueError(f"history calendar {history.calendar!r} differs from the forecast's {forecast.calendar!r}")
    for i, dim in enumerate(spatial_dims):
        h, f = history.tenths.shape[i], forecast.tenths.shape[forecast.dims.index(dim)]
        if h != f:
            raise ValueError(f"history and forecast grids differ along {dim}: {h} vs {f} cells")
        if dim in history.coords and dim in forecast.coords:
            if not np.allclose(np.asarray(history.coords[dim], dtype=float), np.asarray(forecast.coords[dim], dtype=float),
                               rtol=0, atol=1e-6):
                raise ValueError(f"history and forecast grids differ along {dim}: coordinates do not match to 1e-6")


def build_stitched_jobs(definition: Definition, history: DailyRain, forecast: ForecastRain,
                        lead_window: tuple[int, int] = (1, 30), restrict_to_season: bool = True,
                        season_climatology: xr.Dataset | None = None) -> tuple[StitchedJobs, StitchedLayout]:
    """One stitched job per (season, init, [member], cell) over observed history followed by the forecast (spec section 13).

    The forecast row of a job is the flat (init, [member], cell) series index and
    its history row the cell. split is the history index of lead 1's date label
    (init for lead_label "start", init + 1 for "end"); the history must contain
    that label. The season year is chosen as in build_forecast_jobs, with the
    window needing one lead of lead_window inside it; candidates then run from the
    season start (in the history, or lead 1's day onward when the season starts
    later) to the earlier of the season end and the last lead of lead_window.
    A season start before the history gives INSUFFICIENT_DATA, a season missing
    from the init's years NOT_FOUND. With restrict_to_season=False there is one
    season, "window", whose candidates are the leads of lead_window. DerivedSeasons
    with restrict_to_season=True need season_climatology: the same rules apply with
    each cell's own season window (the season start is then per cell, and a cell
    without that season is NOT_FOUND). lead_window must start at 1. data_end is
    split plus the number of leads.
    """
    if lead_window[0] != 1:
        raise ValueError(f"stitched detection needs lead_window to start at 1, got {lead_window}")
    n_leads = forecast.tenths.shape[-1]
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
    hi = min(lead_window[1], n_leads)
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
    """Per (season, init, cell) first and last lead, presence and history start of the derived seasons of a stitched search.

    Returns names, first, last, has and start_hist, each (K, n_init, n_cell) except names: start_hist is the history
    index of the season start when it is on or before lead 1's day (-2 when the history lacks that day), else -1.
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
                  per_cell: dict[str, np.ndarray], backend: str) -> tuple[np.ndarray, np.ndarray]:
    """History-only search per (season, init, cell): the onset day (-1 none) and whether it is ONSET.

    Candidates run to min(season end, split - 1) with data_end = split. Returns
    arrays shaped (n_season, n_init, n_cell).
    """
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
    day, status = _run(plan, history.tenths.reshape(n_cell, n_time), history.missing.reshape(n_cell, n_time),
                       per_cell, pjobs, preset, backend)
    return day.reshape(k, n_init, n_cell), (status == Status.ONSET).reshape(k, n_init, n_cell)


def _run_stitched(plan: plan_mod.Plan, history: DailyRain, forecast: ForecastRain, thresholds: dict[str, np.ndarray],
                  per_cell: dict[str, np.ndarray], jobs: StitchedJobs, layout: StitchedLayout,
                  backend: str, windows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decide PRIOR from the history alone, then run the stitched search for the other jobs; return (day, status) per job.

    The PRIOR search is skipped for curve rules, whose onset is a minimum over the whole
    window and so can move when the window grows (spec section 13). per_cell holds
    the history-only thresholds (first window); windows are the start leads of
    (series, W) thresholds.
    """
    module = _load_backend(backend)
    extra = {} if windows is None else {"amount_windows": windows}
    n_cell, n_time = layout.n_cell, history.tenths.shape[-1]
    n_leads = forecast.tenths.shape[-1]
    n_series = layout.n_series
    flat_h = history.tenths.reshape(n_cell, n_time), history.missing.reshape(n_cell, n_time)
    flat_f = forecast.tenths.reshape(-1, n_leads), forecast.missing.reshape(-1, n_leads)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=flat_f[0].shape[0])
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.where(layout.preset >= 0, layout.preset, Status.NOT_FOUND).astype(np.int8)
    active = layout.preset < 0
    bad = active & invalid[jobs.series]
    status[bad] = Status.INSUFFICIENT_DATA
    active &= ~bad
    if plan.family == plan_mod.FIRST_PASSAGE and active.any():
        k, n_init, n_member = len(layout.season_names), layout.n_init, layout.n_member
        prior_day, prior_hit = _prior_search(plan, history, jobs, layout, per_cell, backend)
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


def _ensemble_summary(lead: np.ndarray, status: np.ndarray, dims: tuple[str, ...], coords: dict,
                      max_lead: int) -> dict[str, xr.DataArray]:
    """members_by_status and onset_probability from (init, member, season, *spatial) lead and status arrays."""
    codes = np.array([s.value for s in Status], dtype=np.int8)
    counts = np.stack([(status == c).sum(axis=1) for c in codes]).astype(np.int64)
    out_dims = ("init_time", *dims[2:])
    by_status = xr.DataArray(
        counts, dims=("status_code", *out_dims),
        coords={"status_code": codes, "status_name": ("status_code", [s.name for s in Status]), **{
            d: v for d, v in coords.items() if d in out_dims}},
        attrs={"long_name": "number of members in each onset status"})
    leads = np.arange(max_lead + 1)
    reached = (lead[..., None] <= leads).sum(axis=1)
    usable = (status != Status.INSUFFICIENT_DATA).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(usable[..., None] > 0, reached / usable[..., None], np.nan)
    share = np.moveaxis(share, -1, 2)
    probability = xr.DataArray(
        share, dims=("init_time", "season", "lead", *dims[3:]),
        coords={"lead": leads, **{d: v for d, v in coords.items() if d in out_dims}},
        attrs={"long_name": "share of members whose onset is on or before this lead",
               "comment": "PRIOR and back-dated onsets (onset_lead <= 0) count at lead 0; members with "
                          "INSUFFICIENT_DATA are excluded from the denominator; CENSORED and NOT_FOUND members "
                          "stay in it and never count as reached"})
    return {"members_by_status": by_status, "onset_probability": probability}


def detect_forecast(definition: Definition, forecast: ForecastRain, *, thresholds=None,
                    lead_window: tuple[int, int] = (1, 30), restrict_to_season: bool = True,
                    backend: str = "numba", attrs: dict | None = None,
                    history: DailyRain | None = None, threshold_windows=None,
                    season_climatology: xr.Dataset | None = None) -> xr.Dataset:
    """Detect onset in every (init, [member], season, cell) of a forecast.

    Returns dims (init_time, [member], season, *spatial) with onset_lead (float64:
    the 1-based lead of the onset, NaN if none), onset_date (the date label of
    that day in the forecast's calendar) and status. See build_forecast_jobs for
    how candidates are chosen; thresholds are per cell and broadcast over inits
    and members.

    With `history` (observed rainfall on the forecast grid) the searched series
    is the history up to the day before lead 1 followed by the forecast, as in
    spec section 13: build_stitched_jobs chooses the candidates, onsets decided
    in the history are PRIOR, and onset_lead is 0 or negative for onsets before
    lead 1. Without it the forecast is searched alone and onset_lead is at least
    1. attrs["forecast_history"] is "observed" or "none". When the forecast has
    members, members_by_status and onset_probability summarise them per init.

    A threshold with a "window" dim (compute_thresholds_from_forecasts with
    windows) is applied per candidate by the window its lead falls in; the window
    start leads come from threshold_windows or the threshold's window_first
    coordinate (spec section 13).

    DerivedSeasons with restrict_to_season=True need season_climatology (n_seasons,
    onset_doy, cessation_doy with dims (season, *spatial) on the forecast's grid, as
    for detect_observed): candidates are then limited to each cell's own season, the
    season coordinate is the climatology's (1, 2, ...), and a cell without that season
    is NOT_FOUND. Stitched, curve rules accumulate from the cell's season start.
    """
    plan = plan_mod.compile(definition)
    n_leads = forecast.tenths.shape[-1]
    series_dims = forecast.dims[:-1]
    series_shape = forecast.tenths.shape[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims, spatial_shape = series_dims[n_lead_dims:], series_shape[n_lead_dims:]
    per_series = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, series_shape)
    windows = _threshold_windows(plan, thresholds, threshold_windows)
    if history is None:
        jobs, layout = build_forecast_jobs(definition, forecast, lead_window, restrict_to_season,
                                            season_climatology)
        day, status = _run(plan, forecast.tenths.reshape(layout.n_series, n_leads),
                           forecast.missing.reshape(layout.n_series, n_leads), per_series, jobs, layout.preset, backend, windows)
        split = np.zeros(len(day), dtype=np.int64)
        hist_labels = None
    else:
        jobs, layout = build_stitched_jobs(definition, history, forecast, lead_window, restrict_to_season,
                                              season_climatology)
        per_cell = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape,
                                      first_window_only=True)
        day, status = _run_stitched(plan, history, forecast, per_series, per_cell, jobs, layout, backend, windows)
        split = jobs.split
        hist_labels = (history.year, history.month, history.day)
    k = len(layout.season_names)
    full = (k, *series_shape)
    day = np.moveaxis(day.reshape(full), 0, n_lead_dims)
    status = np.moveaxis(status.reshape(full), 0, n_lead_dims)
    split = np.moveaxis(split.reshape(full), 0, n_lead_dims)
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
    dims = (*series_dims[:n_lead_dims], "season", *spatial_dims)
    coords = {"season": layout.season_names, **_spatial_coords(dims, forecast.coords)}
    variables = {"onset_lead": (dims, lead), "onset_date": (dims, dates), "status": _status_var(status, dims)}
    ds = xr.Dataset(variables, coords=coords)
    if "member" in series_dims:
        summary = _ensemble_summary(lead, status, dims, coords, lead_window[1])
        for name, var in summary.items():
            ds[name] = var
    ds.attrs.update({"definition_id": definition.id, "definition_version": definition.version, "backend": backend,
                     "lead_label": forecast.lead_label,
                     "forecast_history": "none" if history is None else "observed"})
    ds.attrs.update(attrs or {})
    return ds


def regional(rain: DailyRain, mask, weights: str = "cos_lat") -> DailyRain:
    """Average the cells inside mask into one series (dims ("time",)) and re-round to tenths.

    mask is a boolean array (or DataArray) over the spatial dims. weights is
    "equal" or "cos_lat" (needs a "lat" dim with coordinates). Rain is averaged
    in float mm and rounded half to even. A day is missing only if every cell
    in the mask is missing on that day; with partial missing, the average runs
    over the cells that have data, with their weights renormalised.
    """
    spatial_dims = rain.dims[:-1]
    if isinstance(mask, xr.DataArray):
        mask = mask.transpose(*spatial_dims).values
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rain.tenths.shape[:-1]:
        raise ValueError(f"mask shape {mask.shape} does not match the spatial shape {rain.tenths.shape[:-1]}")
    if not mask.any():
        raise ValueError("mask selects no cells")
    if weights == "equal":
        w = np.ones(mask.shape)
    elif weights == "cos_lat":
        if "lat" not in spatial_dims or "lat" not in rain.coords:
            raise ValueError("cos_lat weights need a lat dim with coordinates")
        shape = [1] * len(spatial_dims)
        shape[spatial_dims.index("lat")] = -1
        w = np.broadcast_to(np.cos(np.deg2rad(np.asarray(rain.coords["lat"], dtype=float))).reshape(shape), mask.shape)
    else:
        raise ValueError(f"weights must be 'cos_lat' or 'equal', got {weights!r}")
    n_time = rain.tenths.shape[-1]
    w = np.where(mask, w, 0.0).reshape(-1, 1)
    valid = ~rain.missing.reshape(-1, n_time)
    weight_sum = (w * valid).sum(axis=0)
    total = (w * valid * rain.tenths.reshape(-1, n_time)).sum(axis=0)
    all_missing = weight_sum == 0
    mean = np.divide(total, weight_sum, out=np.zeros(n_time), where=~all_missing)
    tenths = np.rint(mean).astype(np.int32)
    coords = {"time": rain.coords["time"]} if "time" in rain.coords else {}
    return DailyRain(tenths, all_missing, ("time",), coords, rain.year, rain.month, rain.day, rain.calendar,
                     rain.day_boundary, rain.source)


def climatological_onset(definition: Definition, rain: DailyRain, *, thresholds=None, years,
                         season_climatology: xr.Dataset | None = None, backend: str = "numba") -> xr.Dataset:
    """Median onset day of year per season and cell over `years`, and how many years had an onset.

    Only years with status ONSET count. The day of year is counted from 1 Jan of
    the season's year, so onsets of a season that runs into the next year exceed
    365. onset_doy is NaN where fewer than half of `years` have an onset.
    Variables: onset_doy (float) and n_onset_years (int), dims (season, *spatial).
    """
    ds = detect_observed(definition, rain, thresholds=thresholds, years=years,
                         season_climatology=season_climatology, backend=backend)
    index = ds["onset_index"].values
    found = ds["status"].values == Status.ONSET
    safe = np.where(found, index, 0)
    year, month, day = rain.year[safe], rain.month[safe], rain.day[safe]
    season_year = ds["year"].values.reshape((-1,) + (1,) * (found.ndim - 1))
    doy = (_day_of_year(rain.calendar, year, month, day)
           + (year - season_year) * _year_length(rain.calendar, season_year)).astype(float)
    doy = np.where(found, doy, np.nan)
    n_found = found.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(doy, axis=0)
    median = np.where(n_found * 2 >= len(ds["year"]), median, np.nan)
    dims = ds["status"].dims[1:]
    out = xr.Dataset({"onset_doy": (dims, median), "n_onset_years": (dims, n_found.astype(np.int64))},
                     coords={d: ds.coords[d] for d in dims if d in ds.coords})
    out.attrs.update({"definition_id": definition.id, "definition_version": definition.version,
                      "years": [int(y) for y in ds["year"].values], "backend": backend})
    return out


def save(ds: xr.Dataset, path) -> None:
    """Write a result Dataset to a zarr store, replacing what is there.

    cftime onset dates (non-standard calendars) cannot hold "no onset" in a CF
    time encoding, so they are stored as YYYY-MM-DD strings, empty where there
    is no onset.
    """
    for name, var in ds.data_vars.items():
        if var.dtype == object:
            text = np.array([f"{d.year:04d}-{d.month:02d}-{d.day:02d}" if d is not None else "" for d in var.values.ravel()])
            ds = ds.assign({name: (var.dims, text.reshape(var.shape), var.attrs)})
    ds.to_zarr(path, mode="w")
