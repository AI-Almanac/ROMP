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
from momp.onset.model import DailyRain, Definition, DerivedSeasons, ForecastRain, Jobs, Status

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


def _load_backend(name: str):
    """Import momp.onset.backends.<module> for a backend name (reference, numba, gpu)."""
    if name not in _BACKEND_MODULES:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(_BACKEND_MODULES)}")
    module = f"momp.onset.backends.{_BACKEND_MODULES[name]}"
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"backend {name!r} is not available ({module}): {exc}") from exc


def _series_thresholds(plan: plan_mod.Plan, thresholds, spatial_dims: tuple[str, ...],
                       spatial_shape: tuple[int, ...], series_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Per-series float arrays for the thresholds the plan needs, broadcast from per-cell values."""
    out = {}
    for spec in plan.amount_specs:
        if spec.kind == "fixed" or spec.name in out:
            continue
        if not thresholds or spec.name not in thresholds:
            raise KeyError(f"threshold {spec.name!r} is needed by the definition but was not supplied")
        value = thresholds[spec.name]
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
         jobs: Jobs, preset: np.ndarray, backend: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the backend over the jobs that need a search, chunked over series; return (day, status) per job."""
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
        out = module.detect(plan, tenths[s0:s1], missing[s0:s1], sub_amounts, sub)
        day[sel] = out.day
        status[sel] = out.status
    return day, status


def _derived_bounds(definition: Definition, keys: np.ndarray, years: np.ndarray, climatology: xr.Dataset,
                    spatial_dims: tuple[str, ...], n_series: int):
    """First and last day indices (K, Y, S) plus a "no such season" mask for DerivedSeasons."""
    seasons = definition.seasons
    for ref in (seasons.start, seasons.end):
        if ref.climatological is None:
            raise ValueError("derived seasons need climatological bounds")
    var = {"onset": "onset_doy", "cessation": "cessation_doy"}
    n_k = climatology.sizes["season"]
    onset = climatology[var[seasons.start.climatological]].transpose("season", *spatial_dims).values.reshape(n_k, n_series)
    cessation = climatology[var[seasons.end.climatological]].transpose("season", *spatial_dims).values.reshape(n_k, n_series)
    n_seasons = None
    if "n_seasons" in climatology:
        n_seasons = climatology["n_seasons"].transpose(*spatial_dims).values.reshape(n_series)
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
    names = [str(s) for s in climatology["season"].values] if "season" in climatology.coords \
        else [f"season_{k + 1}" for k in range(n_k)]
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
    per_series = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape)
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


def _label_table(forecast: ForecastRain, n_leads: int):
    """Date labels (year, month, day), each (n_init, n_leads): init + k - 1 days for lead_label "start", init + k for "end"."""
    lead_offsets = np.arange(n_leads) + (1 if forecast.lead_label == "end" else 0)
    return _add_days(forecast.calendar, forecast.init_year, forecast.init_month, forecast.init_day, lead_offsets)


def build_forecast_jobs(definition: Definition, forecast: ForecastRain, lead_window: tuple[int, int] = (1, 30),
                        restrict_to_season: bool = True) -> tuple[Jobs, JobLayout]:
    """One job per (season, series) with series = (init, [member], cell) rows over the lead axis.

    Candidates are the leads in lead_window (1-based, inclusive) whose date label
    falls in a window of the season, taken from the earliest of the init's year
    minus one, the init's year and the year after that has any such lead; the
    qualifying leads are contiguous. With none, the job is NOT_FOUND. With
    restrict_to_season=False there is one season, "window", and every lead of
    lead_window is a candidate (this also allows derived seasons). data_end is
    the number of leads.
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
            raise NotImplementedError("derived seasons are not supported for forecasts with restrict_to_season=True")
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


def detect_forecast(definition: Definition, forecast: ForecastRain, *, thresholds=None,
                    lead_window: tuple[int, int] = (1, 30), restrict_to_season: bool = True,
                    backend: str = "numba", attrs: dict | None = None) -> xr.Dataset:
    """Detect onset in every (init, [member], season, cell) of a forecast.

    Returns dims (init_time, [member], season, *spatial) with onset_lead
    (1-based lead of the onset, -1 if none), onset_date (the date label of that
    lead in the forecast's calendar) and status. See build_forecast_jobs for how
    candidates are chosen; thresholds are per cell and broadcast over inits and
    members.
    """
    plan = plan_mod.compile(definition)
    jobs, layout = build_forecast_jobs(definition, forecast, lead_window, restrict_to_season)
    n_leads = forecast.tenths.shape[-1]
    series_dims = forecast.dims[:-1]
    series_shape = forecast.tenths.shape[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims, spatial_shape = series_dims[n_lead_dims:], series_shape[n_lead_dims:]
    per_series = _series_thresholds(plan, thresholds, spatial_dims, spatial_shape, series_shape)
    day, status = _run(plan, forecast.tenths.reshape(layout.n_series, n_leads),
                       forecast.missing.reshape(layout.n_series, n_leads), per_series, jobs, layout.preset, backend)
    k = len(layout.season_names)
    day = np.moveaxis(day.reshape((k, *series_shape)), 0, n_lead_dims)
    status = np.moveaxis(status.reshape((k, *series_shape)), 0, n_lead_dims)
    lead = np.where(day >= 0, day + 1, -1)
    label_y, label_m, label_d = _label_table(forecast, n_leads)
    init_index = np.arange(series_shape[0]).reshape((-1,) + (1,) * (day.ndim - 1))
    safe = np.where(day >= 0, day, 0)
    dates = _dates_from_labels(forecast.calendar, label_y[init_index, safe], label_m[init_index, safe],
                               label_d[init_index, safe], day >= 0)
    dims = (*series_dims[:n_lead_dims], "season", *spatial_dims)
    coords = {"season": layout.season_names, **_spatial_coords(dims, forecast.coords)}
    ds = xr.Dataset({"onset_lead": (dims, lead.astype(np.int64)), "onset_date": (dims, dates),
                     "status": _status_var(status, dims)}, coords=coords)
    ds.attrs.update({"definition_id": definition.id, "definition_version": definition.version, "backend": backend,
                     "lead_label": forecast.lead_label})
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
