"""A fast, sampled check of the day boundary and lead label between observations and a forecast model.

``hcwf-run --check --alignment`` and the lead-label detection of zarr datasets use this instead of correlating a
whole year. It reads a small, deterministic sample once and gets every lag from one vectorised step:

* inits: a few dozen, evenly spread over the evaluated period (the model's inits inside the run's dates);
* cells: up to a few thousand, a regular subgrid of the cells the observations and forecasts share inside the
  region box, without the cells the observations do not cover (the sea);
* the forecast leads 1-3 (members averaged on sqrt rain) and, for each init, the observed days from three days
  before the init to the last lead plus three days, all read in one selection per file or store;
* `hcwf.onset.alignment.lagged_correlations` pairs the forecast with a stacked (lag, ...) view of the
  observations and gives the pooled correlation of every lag and lead at once.

The verdicts are those of `hcwf.onset.alignment.check_forecast_alignment` (spec section 8); the report type and the
comparison itself live in `hcwf.onset.alignment`, and the reading uses `hcwf.onset.ingest` for the forecast days and
`hcwf.onset.calendar` for date keys. The result is cached next to the detection cache (a `hcwf.cache.Store` of kind
``"alignment"``) under the parts of both datasets (their files), the lead label, the day boundaries and the sample
specification; `check_models` runs several models in parallel threads. Files are found as the run finds them
(`hcwf.io.stores`), so datasets given by ``paths`` are read.

The keyword arguments (``obs_dir``, ``obs_file_pattern``, ``obs_var``, ``obs_unit_cvt``, ``model_dir``,
``model_var``, ``unit_cvt``, ``file_pattern``, ``model_dims``, ``probabilistic``, ``members``, ``accumulation``,
``daily_boundary``, ``region``, ``start_date``, ``end_date``, ``init_days``, ``years``) and, for datasets given as
lists of files, ``obs_paths``, ``obs_files_years``, ``model_paths`` and ``model_files_years``, are those of
`sample_spec` (a run's model, from its typed specs); the lead-label detection of zarr datasets
(`hcwf.config.detect`) writes them before a run exists.
"""
from __future__ import annotations

import os
import threading
import warnings
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import Store, file_identity
from hcwf.io.stores import forecast_sources as _year_sources
from hcwf.io.stores import observation_source, open_store
from hcwf.config.datasets import file_year, is_zarr, resolve_paths
from hcwf.onset import ingest as ing
from hcwf.onset.alignment import (ForecastAlignmentReport, _match_axis, _sqrt_field,
                                  forecast_alignment_from_samples)
from hcwf.onset.calendar import label_keys
from hcwf.errors import DatasetError

SAMPLE_INITS = 40
"""Most inits in the sample."""
SAMPLE_CELLS = 3000
"""Most cells in the sample (a regular subgrid of the shared cells)."""
SAMPLE_MEMBERS = 10
"""Most ensemble members averaged."""
LEADS = (1, 2, 3)
"""Forecast leads compared."""
MAX_LAG = 3
"""Lags run from -MAX_LAG to +MAX_LAG days."""
_MIN_COVER = 0.5
"""A cell is kept when at least this share of its sampled values are present in both datasets."""
_NETCDF_LOCK = threading.Lock()
"""Held while NetCDF files are read: the NetCDF/HDF5 libraries are not safe to use from several threads."""


@dataclass
class Sample:
    """The arrays of one sampled read: observations (init, day, cell), forecast (init, lead, cell), in sqrt(mm)."""

    obs_days: np.ndarray
    """Observed sqrt-rain, ``float32`` with shape (init, day, cell); day ``j`` is ``j - pad`` days after the init."""
    forecast: np.ndarray
    """Member-mean forecast sqrt-rain, ``float64`` with shape (init, lead, cell)."""
    leads: tuple
    """The forecast leads of the second axis of `forecast`."""
    pad: int
    """The number of days before the init date at the start of the day axis of `obs_days`."""
    n_inits: int
    """The number of inits in the sample."""
    n_cells: int
    """The number of cells in the sample."""
    notes: list = field(default_factory=list)
    """Notes for the report, as strings."""


def _patterns(value) -> tuple:
    """Return a file pattern or a sequence of them as a tuple.

    Parameters
    ----------
    value : str or sequence of str, optional
        The pattern(s), ``{}`` standing for the year; ``None`` gives the default ``"{}.nc"``.

    Returns
    -------
    tuple of str
        The patterns.
    """
    if value is None:
        return ("{}.nc",)
    return (value,) if isinstance(value, str) else tuple(value)


def _years_in(folder, pattern: str) -> tuple:
    """Return the years of the files in a folder that match a file pattern, sorted.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder to look in.
    pattern : str
        The file pattern, ``{}`` standing for the year.

    Returns
    -------
    tuple of int
        The years found, increasing.
    """
    import glob

    return tuple(sorted({y for y in (file_year(f, whole_number=True)
                                     for f in glob.glob(os.path.join(str(folder), pattern.replace("{}", "*"))))
                         if y is not None}))


def _listed_years(paths, base, years_filter) -> tuple:
    """Return the years of the files a dataset lists in `paths`, sorted.

    Parameters
    ----------
    paths : sequence of str
        The listed files or globs, as in the dataset specification.
    base : str, optional
        The folder relative paths are resolved against.
    years_filter : sequence of int, optional
        Keep only the files of these years.

    Returns
    -------
    tuple of int
        The years found, increasing.

    Raises
    ------
    ValueError
        The files cannot be resolved (the message of the `hcwf.errors.DatasetError`).
    """
    try:
        files = resolve_paths(paths, base, years_filter)
    except DatasetError as exc:
        raise ValueError(str(exc)) from None
    return tuple(sorted({y for y in map(file_year, files) if y is not None}))


_SIDES = {"obs": ("obs_dir", "obs_paths", "obs_files_years", "obs_file_pattern"),
          "model": ("model_dir", "model_paths", "model_files_years", "file_pattern")}
"""The kwargs of each side: folder, listed files, years filter of the listed files, file pattern."""


def sample_spec(rc, model: str) -> dict:
    """Return the keyword arguments of the sampled check for one model of a run, from its typed specs.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run configuration: its ``observations``, ``evaluation`` and ``area`` specs and the model specs are read.
    model : str
        The name of the model in the run.

    Returns
    -------
    dict
        The keyword arguments described in the module text (``obs_dir``, ``model_dir``, ``region``, ``years``, ...).
    """
    obs, ev, source = rc.observations, rc.evaluation, rc.model(model)
    return {"obs_dir": obs.dir, "obs_file_pattern": obs.file_pattern, "obs_var": obs.var, "obs_unit_cvt": obs.unit_cvt,
            "obs_paths": obs.paths, "model_dir": source.dir, "file_pattern": source.file_pattern,
            "model_var": source.var, "unit_cvt": source.unit_cvt, "model_dims": source.dims_map(),
            "accumulation": source.accumulation, "daily_boundary": source.daily_boundary,
            "probabilistic": ev.probabilistic, "members": ev.members, "region": rc.area.region,
            "start_date": ev.start_date, "end_date": ev.end_date, "init_days": ev.init_days, "years": ev.years}


def _is_store(kw: dict, side: str) -> bool:
    """Return whether the side's dataset is a zarr store.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    side : {"obs", "model"}
        The dataset to test.

    Returns
    -------
    bool
        True for a zarr folder; a dataset given by paths is files.
    """
    folder, paths, _, _ = _SIDES[side]
    return not kw.get(paths) and is_zarr(kw.get(folder))


def _dataset_years(kw: dict, side: str) -> tuple:
    """Return the years of a side's files.

    These are the years of its listed files, or those found in its folder by the file pattern.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    side : {"obs", "model"}
        The dataset.

    Returns
    -------
    tuple of int
        The years, increasing.
    """
    folder, paths, years_filter, pattern = _SIDES[side]
    if kw.get(paths):
        return _listed_years(kw[paths], kw.get(folder) or None, kw.get(years_filter))
    return _years_in(kw[folder], _patterns(kw.get(pattern))[0])


def _box_indices(values: np.ndarray, lo: float, hi: float, wrap: bool = False) -> np.ndarray:
    """Return the indices of the coordinate values inside a closed interval.

    Parameters
    ----------
    values : numpy.ndarray
        The coordinate values (latitudes or longitudes), converted to ``float64``.
    lo : float
        The lower end of the interval.
    hi : float
        The upper end of the interval.
    wrap : bool, default False
        Compare longitudes modulo 360, so an interval may cross the antimeridian (``lo > hi`` after the modulo).

    Returns
    -------
    numpy.ndarray of int64
        The indices of the values with ``lo <= value <= hi``.
    """
    values = np.asarray(values, dtype=float)
    if not wrap:
        return np.nonzero((values >= lo) & (values <= hi))[0]
    v, lo, hi = values % 360, lo % 360, hi % 360
    keep = (v >= lo) & (v <= hi) if lo <= hi else (v >= lo) | (v <= hi)
    return np.nonzero(keep)[0]


def _region_box(region):
    """Return the bounding box of a region, or ``None`` when the region is unknown.

    Parameters
    ----------
    region : str
        The region name, as for `hcwf.onset.regions.box`.

    Returns
    -------
    tuple of (float, float, float, float) or None
        ``(lats, latn, lonw, lone)`` in degrees; ``None`` when the box cannot be obtained.
    """
    from hcwf.onset.regions import box

    try:
        return box(region)
    except Exception:  # noqa: BLE001 - an unknown region just means no box
        return None


def _scale(da: xr.DataArray, factor) -> xr.DataArray:
    """Return the loaded array multiplied by a unit factor, as the run's loaders do.

    The factor is cast to the array's own dtype, so the dtype is kept.

    Parameters
    ----------
    da : xarray.DataArray
        The loaded data.
    factor : float, optional
        The unit conversion factor; ``None`` or 0 leaves the data unchanged.

    Returns
    -------
    xarray.DataArray
        The scaled data.
    """
    if factor:
        da = da * np.asarray(factor, dtype=da.dtype)
    return da


def _standardise(ds: xr.Dataset, kw: dict) -> xr.Dataset:
    """Return a forecast dataset with HCWF's names (``init_time``, ``step``, ``member``, ``lat``, ``lon``).

    Parameters
    ----------
    ds : xarray.Dataset
        The forecast as opened.
    kw : dict
        The keyword arguments of the sampled check: ``probabilistic`` (keep the member dim) and ``model_dims`` (the
        dimension-name mapping of the model).

    Returns
    -------
    xarray.Dataset
        The dataset with the standard dimension names, as the run reads it.
    """
    from hcwf.io.dims import dim_fmt_model, dim_fmt_model_ensemble

    fix = dim_fmt_model_ensemble if kw.get("probabilistic") else dim_fmt_model
    return fix(ds, kw.get("model_dims"))


def forecast_sources(kw: dict) -> list:
    """Return the standardised forecast datasets to read.

    These are the zarr store, or the files of the run's years (any number a year); a year without files is skipped.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.

    Returns
    -------
    list of tuple
        One ``(year, dataset, close)`` per store or file: the year (``None`` for a store), the standardised
        `xarray.Dataset`, and a callable that closes it.
    """
    folder = kw.get("model_dir")
    if _is_store(kw, "model"):
        return [(None, _standardise(open_store(folder), kw), lambda: None)]
    out = []
    pattern = _patterns(kw.get("file_pattern"))[0]
    for year in kw.get("years") or _dataset_years(kw, "model"):
        try:
            paths = _year_sources(year, folder, pattern, kw.get("model_paths"), kw.get("model_files_years"))
        except FileNotFoundError:
            continue
        for path in paths:
            raw = xr.open_dataset(path)
            out.append((year, _standardise(raw, kw), raw.close))
    return out


def _candidate_inits(times, kw: dict) -> np.ndarray:
    """Return the indices of the init times inside the run's years, month-day window and init weekdays.

    When no init qualifies, or the times cannot be converted to dates, all of them are returned.

    Parameters
    ----------
    times : numpy.ndarray
        The init times (``datetime64`` or convertible by `pandas.to_datetime`).
    kw : dict
        The keyword arguments of the sampled check: ``years``, ``start_date`` and ``end_date`` (``(year, month, day)``
        tuples, of which month and day are used) and ``init_days`` (weekdays, Monday is 0).

    Returns
    -------
    numpy.ndarray of int64
        The indices into `times`, increasing.
    """
    try:
        stamps = pd.to_datetime(np.asarray(times))
    except (TypeError, ValueError):
        return np.arange(len(times))
    keep = np.ones(len(stamps), dtype=bool)
    start, end = kw.get("start_date"), kw.get("end_date")
    years = kw.get("years")
    if years:
        keep &= stamps.year.isin(list(years))
    if start and end:
        md = stamps.month * 100 + stamps.day
        keep &= (md >= start[1] * 100 + start[2]) & (md <= end[1] * 100 + end[2])
    if kw.get("init_days") is not None:
        keep &= stamps.weekday.isin(list(kw["init_days"]))
    return np.nonzero(keep)[0] if keep.any() else np.arange(len(stamps))


def _spread(n_available: int, n_wanted: int) -> np.ndarray:
    """Return `n_wanted` indices spread evenly over ``range(n_available)``.

    Parameters
    ----------
    n_available : int
        The number of items to choose from.
    n_wanted : int
        The number of items wanted.

    Returns
    -------
    numpy.ndarray of int64
        All of ``range(n_available)`` when `n_available` is not above `n_wanted`, else the distinct rounded points of an
        even spacing from the first to the last index.
    """
    if n_available <= n_wanted:
        return np.arange(n_available)
    return np.unique(np.round(np.linspace(0, n_available - 1, n_wanted)).astype(int))


def _stride(n: int, step: int) -> np.ndarray:
    """Return every `step`-th index of ``range(n)``, starting half a step in so the subgrid is centred.

    Parameters
    ----------
    n : int
        The length of the axis.
    step : int
        The stride; 1 gives all indices.

    Returns
    -------
    numpy.ndarray of int64
        The chosen indices.
    """
    return np.arange(step // 2 if step > 1 else 0, n, step)


def _shared_grid(obs_ds: xr.Dataset, fc_ds: xr.Dataset, kw: dict) -> tuple:
    """Return the matched, subsampled grid indices inside the region box.

    The subgrid takes every s-th shared latitude and longitude, s the smallest stride that leaves about ``SAMPLE_CELLS``
    cells (a little more, as the sea is dropped after reading).

    Parameters
    ----------
    obs_ds : xarray.Dataset
        The observations, with ``lat`` and ``lon`` coordinates.
    fc_ds : xarray.Dataset
        The forecasts, with ``lat`` and ``lon`` coordinates.
    kw : dict
        The keyword arguments of the sampled check; ``region`` selects the box (no box when it is unknown or absent).

    Returns
    -------
    obs_lat : numpy.ndarray of int64
        Indices into the observations' latitudes.
    fc_lat : numpy.ndarray of int64
        The matching indices into the forecasts' latitudes.
    obs_lon : numpy.ndarray of int64
        Indices into the observations' longitudes.
    fc_lon : numpy.ndarray of int64
        The matching indices into the forecasts' longitudes.

    Raises
    ------
    ValueError
        The two grids share no cell inside the region box.
    """
    obs_lat, obs_lon = (np.asarray(obs_ds[c].values, float) for c in ("lat", "lon"))
    fc_lat, fc_lon = (np.asarray(fc_ds[c].values, float) for c in ("lat", "lon"))
    box = _region_box(kw.get("region"))
    lat_in = _box_indices(obs_lat, box[0], box[1]) if box else np.arange(len(obs_lat))
    lon_in = _box_indices(obs_lon, box[2], box[3], wrap=True) if box else np.arange(len(obs_lon))
    ra, fa = _match_axis(obs_lat[lat_in], fc_lat)
    rb, fb = _match_axis(obs_lon[lon_in], fc_lon)
    ol, fl, olon, flon = lat_in[ra], fa, lon_in[rb], fb
    if len(ol) == 0 or len(olon) == 0:
        raise ValueError("no shared cells inside the region box")
    step = max(1, int(np.ceil(np.sqrt(len(ol) * len(olon) / (SAMPLE_CELLS * 1.3)))))
    pick_lat, pick_lon = _stride(len(ol), step), _stride(len(olon), step)
    return ol[pick_lat], fl[pick_lat], olon[pick_lon], flon[pick_lon]


def _member_count(da: xr.DataArray) -> int:
    """Return how many members to average: up to ``SAMPLE_MEMBERS``, but no more than one file chunk along the member dim.

    Reading a second chunk of members multiplies the decoding of compressed NetCDF files several times over, and a few
    members already give most of the benefit of averaging; at least 4 are kept.

    Parameters
    ----------
    da : xarray.DataArray
        The forecast variable; its ``encoding`` holds the file chunk sizes.

    Returns
    -------
    int
        The number of members to read; ``SAMPLE_MEMBERS`` when `da` has no member dim or no chunk information.
    """
    chunks = da.encoding.get("chunksizes") or da.encoding.get("chunks")
    if "member" not in da.dims or not chunks or len(chunks) != da.ndim:
        return SAMPLE_MEMBERS
    return int(max(min(SAMPLE_MEMBERS, chunks[da.dims.index("member")]), min(4, SAMPLE_MEMBERS)))


def _read_forecast(kw: dict, grid: tuple, sources: list):
    """Return the sampled forecast: spread-out inits, leads 1-3, the subgrid, the members cut.

    The inits are chosen over all the sources, the data are read for the bounding box of the subgrid and cut to it, units
    are converted, and the days are made with `hcwf.onset.ingest.ingest_forecast_info` (``"00 UTC, labelled by start"``,
    mm; the ingest warnings are silenced).

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    grid : tuple of numpy.ndarray
        The indices of `_shared_grid`; the forecast's latitude and longitude indices (items 1 and 3) are used.
    sources : list of tuple
        The ``(year, dataset, close)`` items of `forecast_sources`.

    Returns
    -------
    hcwf.onset.model.ForecastRain
        The sampled forecast, with the leads 1 to ``max(LEADS)``.

    Raises
    ------
    ValueError
        No source has an init in the run's period.
    """
    fl, flon = grid[1], grid[3]
    chosen = []
    for k, (_, ds, _) in enumerate(sources):
        for i in _candidate_inits(ds["init_time"].values, kw):
            chosen.append((k, int(i), ds["init_time"].values[i]))
    if not chosen:
        raise ValueError("the forecasts have no init in the run's period")
    chosen.sort(key=lambda c: c[2])
    chosen = [chosen[i] for i in _spread(len(chosen), SAMPLE_INITS)]
    span_lat, span_lon = slice(int(fl.min()), int(fl.max()) + 1), slice(int(flon.min()), int(flon.max()) + 1)
    parts = []
    for k, (_, ds, _) in enumerate(sources):
        picks = [i for kk, i, _ in chosen if kk == k]
        if not picks:
            continue
        sub = ds
        if float(sub["step"][0]) == 0:
            sub = sub.sel(step=slice(1, None))
        sub = sub.sel(step=slice(None, max(LEADS)))
        da = sub[kw["model_var"]]
        n_members = _member_count(da)
        da = da.isel(init_time=picks)
        if "member" in da.dims:
            if kw.get("probabilistic") and kw.get("members"):
                da = da.sel(member=list(kw["members"])[:n_members])
            else:
                da = da.isel(member=slice(0, n_members))
        da = da.isel(lat=span_lat, lon=span_lon).load()
        da = _scale(da, kw.get("unit_cvt"))
        da = da.isel(lat=fl - span_lat.start, lon=flon - span_lon.start)
        parts.append(da)
    cube = parts[0] if len(parts) == 1 else xr.concat(parts, dim="init_time")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast, _ = ing.ingest_forecast_info(
            cube, day_boundary="00 UTC, labelled by start", lead_label="start", units="mm", source="forecast",
            init_dim="init_time", lead_dim="step", member_dim="member" if "member" in cube.dims else None,
            accumulation=kw.get("accumulation") or "daily", daily_boundary=kw.get("daily_boundary") or None)
    return forecast


def _forecast_field(forecast, leads) -> np.ndarray:
    """Return the member-mean sqrt-rain of the given leads.

    Parameters
    ----------
    forecast : hcwf.onset.model.ForecastRain
        The sampled forecast.
    leads : sequence of int
        The leads to take, each in the forecast's lead coordinate.

    Returns
    -------
    numpy.ndarray of float64, shape (init, lead, cell)
        Sqrt of the rain in mm, ``NaN`` where missing, with the cells in ``(lat, lon)`` row-major order.

    Raises
    ------
    ValueError
        A lead is not in the forecast's leads.
    """
    lead_axis = forecast.dims.index("lead")
    values = np.asarray(forecast.coords["lead"]).tolist()
    for k in leads:
        if k not in values:
            raise ValueError(f"lead {k} is not in the forecast leads")
    pick = [values.index(k) for k in leads]
    f = _sqrt_field(np.take(forecast.tenths, pick, axis=lead_axis), np.take(forecast.missing, pick, axis=lead_axis))
    dims = forecast.dims
    if "member" in dims:
        axis = dims.index("member")
        with np.errstate(all="ignore"):
            f = np.nanmean(f, axis=axis)
        dims = tuple(d for d in dims if d != "member")
    f = np.transpose(f, [dims.index(d) for d in ("init_time", "lat", "lon", "lead")])
    return f.reshape(f.shape[0], -1, f.shape[-1]).transpose(0, 2, 1)


def _obs_segments(kw: dict, years) -> list:
    """Return the standardised observation datasets to read.

    These are the zarr store, or the files with data for the years; a year without a file is skipped.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    years : sequence of int
        The years to find files for; empty means the first year found in the dataset.

    Returns
    -------
    list of tuple
        One ``(dataset, close)`` per store or file: the standardised `xarray.Dataset` and a callable that closes it.
    """
    from hcwf.io.dims import dim_fmt

    folder = kw.get("obs_dir")
    if _is_store(kw, "obs"):
        return [(dim_fmt(open_store(folder)), lambda: None)]
    out = []
    if not years:
        years = _dataset_years(kw, "obs")[:1]
    for year in sorted(years):
        try:
            _, found = observation_source(year, folder, _patterns(kw.get("obs_file_pattern")), kw.get("obs_paths"),
                                   kw.get("obs_files_years"))
        except FileNotFoundError:
            continue
        for path in (found if isinstance(found, tuple) else (found,)):
            raw = xr.open_dataset(path)
            out.append((dim_fmt(raw), raw.close))
    return out


def _read_obs(kw: dict, grid: tuple, forecast, pad: int, n_days: int):
    """Return the observed sqrt-rain for each init's days, read once per file or store.

    For every init of `forecast` found in the observations, the days ``-pad .. n_days - pad - 1`` around its init date
    are read (rain rounded to tenths of mm; values below -0.05 mm are missing).

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    grid : tuple of numpy.ndarray
        The indices of `_shared_grid`; the observations' latitude and longitude indices (items 0 and 2) are used.
    forecast : hcwf.onset.model.ForecastRain
        The sampled forecast whose init dates are looked up.
    pad : int
        The number of days before the init date that are read.
    n_days : int
        The number of days read for each init.

    Returns
    -------
    obs : numpy.ndarray of float32, shape (init, day, cell)
        Sqrt-rain in sqrt(mm), ``NaN`` where missing and for every init not in the observations.
    found : numpy.ndarray of bool, shape (init,)
        Whether each init date is in the observations.

    Raises
    ------
    ValueError
        No observation file is found, the files use different calendars or a calendar other than the forecast's, or no
        init date is in the observations.
    """
    ol, _, olon, _ = grid
    init_keys = label_keys(forecast.init_year, forecast.init_month, forecast.init_day)
    year_span = sorted({int(y) for y in forecast.init_year} | {int(y) - 1 for y in forecast.init_year}
                       | {int(y) + 1 for y in forecast.init_year})
    segments = _obs_segments(kw, year_span)
    if not segments:
        raise ValueError("no observation file covers the forecast inits")
    try:
        times = []
        calendar = None
        for ds, _ in segments:
            cal, ords = ing._read_time(ds["time"].values, constant_time_of_day=True)
            if calendar not in (None, cal):
                raise ValueError("the observation files use different calendars")
            calendar = cal
            times.append(ords)
        if calendar != forecast.calendar:
            raise ValueError(f"calendars differ ({calendar} vs {forecast.calendar}); cannot compare")
        low, high = min(int(t.min()) for t in times), max(int(t.max()) for t in times)
        full = np.arange(low, high + 1)
        _, y, m, d = ing._labels(full, calendar)
        full_keys = label_keys(y, m, d)
        pos = np.searchsorted(full_keys, init_keys)
        found = (pos < len(full_keys)) & (full_keys[np.minimum(pos, len(full_keys) - 1)] == init_keys)
        if not found.any():
            raise ValueError("no forecast init date is in the observations")
        init_ord = full[np.minimum(pos, len(full) - 1)]
        need = init_ord[:, None] + (np.arange(n_days) - pad)[None, :]
        wanted = np.unique(need[found])
        values = np.full((len(wanted), len(ol) * len(olon)), np.nan, dtype=np.float32)
        span_lat, span_lon = slice(int(ol.min()), int(ol.max()) + 1), slice(int(olon.min()), int(olon.max()) + 1)
        for (ds, _), ords in zip(segments, times):
            slot = np.searchsorted(ords, wanted)
            hit = (slot < len(ords)) & (ords[np.minimum(slot, len(ords) - 1)] == wanted)
            if not hit.any():
                continue
            order = np.argsort(slot[hit])
            idx = slot[hit][order]
            da = ds[kw["obs_var"]].isel(time=idx, lat=span_lat, lon=span_lon).load()
            da = _scale(da, kw.get("obs_unit_cvt")).transpose("time", "lat", "lon")
            block = da.values[:, ol - span_lat.start][:, :, olon - span_lon.start]
            values[np.nonzero(hit)[0][order]] = block.reshape(len(idx), -1)
    finally:
        for _, close in segments:
            close()
    tenths = np.round(values.astype(np.float64) * 10.0) / 10.0
    rain = np.where(np.isfinite(tenths) & (tenths >= -0.05), np.sqrt(np.maximum(tenths, 0.0)), np.nan)
    lookup = np.searchsorted(wanted, np.where(found[:, None], need, wanted[0]))
    obs = rain[lookup]
    obs[~found] = np.nan
    return obs.astype(np.float32), found


def read_sample(kw: dict) -> Sample:
    """Read the sample of one model against the observations (see the module text).

    Reading NetCDF files is serialised across threads (zarr stores are read concurrently). `ValueError` is raised on
    trouble, as in `_read_sample`.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check (see `sample_spec`).

    Returns
    -------
    Sample
        The sampled observations and forecast.
    """
    both_zarr = _is_store(kw, "obs") and _is_store(kw, "model")
    with nullcontext() if both_zarr else _NETCDF_LOCK:
        return _read_sample(kw)


def _read_sample(kw: dict) -> Sample:
    """Read the sample: the body of `read_sample`, run under the NetCDF lock when files are read.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.

    Returns
    -------
    Sample
        The sampled arrays, restricted to the inits found in the observations and to the cells with at least
        ``_MIN_COVER`` of their values present in both datasets (at most ``SAMPLE_CELLS``, spread evenly).

    Raises
    ------
    ValueError
        No forecast or observation is found, no init is in the observations, or no sampled cell has data in both.
    """
    sources = forecast_sources(kw)
    if not sources:
        raise ValueError("no forecast file or store found")
    try:
        obs_segments = _obs_segments(kw, kw.get("years") or ())
        if not obs_segments:
            raise ValueError("no observation file found")
        grid = _shared_grid(obs_segments[0][0], sources[0][1], kw)
        for _, close in obs_segments:
            close()
        forecast = _read_forecast(kw, grid, sources)
    finally:
        for _, _, close in sources:
            close()
    pad = MAX_LAG
    n_days = pad + max(LEADS) + MAX_LAG + 1
    obs_days, found = _read_obs(kw, grid, forecast, pad, n_days)
    fc = _forecast_field(forecast, LEADS)
    notes = []
    obs_days, fc = obs_days[found], fc[found]
    if not len(fc):
        raise ValueError("no forecast init date is in the observations")
    cover_obs = np.isfinite(obs_days).mean(axis=(0, 1))
    cover_fc = np.isfinite(fc).mean(axis=(0, 1))
    keep = np.nonzero((cover_obs >= _MIN_COVER) & (cover_fc >= _MIN_COVER))[0]
    if len(keep) == 0:
        raise ValueError("no sampled cell has data in both the observations and the forecasts")
    if len(keep) > SAMPLE_CELLS:
        keep = keep[_spread(len(keep), SAMPLE_CELLS)]
    return Sample(obs_days[:, :, keep], fc[:, :, keep], LEADS, pad, int(len(fc)), int(len(keep)), notes)


def sample_report(sample: Sample, lead_label: str, *, obs_boundary: str = "", forecast_boundary: str = "",
                  source: str = "forecast") -> ForecastAlignmentReport:
    """Return the alignment report of a sample under one lead label (cheap: no reading).

    Parameters
    ----------
    sample : Sample
        The sampled arrays.
    lead_label : {"start", "end"}
        The lead label to test.
    obs_boundary : str, default ""
        The observations' declared day boundary text, for the notes of the report.
    forecast_boundary : str, default ""
        The forecast's declared day boundary text, for the notes of the report.
    source : str, default "forecast"
        The name the report gives the forecast.

    Returns
    -------
    hcwf.onset.alignment.ForecastAlignmentReport
        The per-lead report and verdict, with the notes of the sample first.
    """
    report = forecast_alignment_from_samples(
        sample.obs_days, sample.forecast, leads=sample.leads, pad=sample.pad, lead_label=lead_label,
        obs="observations", source=source, n_inits=sample.n_inits, obs_boundary=obs_boundary,
        forecast_boundary=forecast_boundary, max_lag=MAX_LAG)
    report.notes = sample.notes + report.notes
    for lead_report in report.per_lead.values():
        lead_report.other = source
    return report


# --- the cache ---------------------------------------------------------------------------------------------------

_KEYS = ("obs_dir", "obs_var", "obs_unit_cvt", "obs_paths", "obs_files_years", "model_dir", "model_var", "unit_cvt",
         "model_paths", "model_files_years", "model_dims", "region", "probabilistic", "members", "accumulation",
         "daily_boundary", "start_date", "end_date", "init_days", "years")
"""The kwargs the verdict depends on, beside the files and file patterns."""


def _store() -> Store:
    """Return the verdict cache.

    It is a `hcwf.cache.Store` of kind ``"alignment"`` next to the detection cache (fresh copies, no memory layer).

    Returns
    -------
    hcwf.cache.Store
        The cache store.
    """
    from hcwf.config.detect import cache_dir as detect_dir

    return Store("alignment", detect_dir().parent, label="Alignment", memory=False)


def _files(kw: dict, side: str) -> list:
    """Return what identifies a side's content.

    This is the identity of its listed files, its zarr store, or its NetCDF files.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    side : {"obs", "model"}
        The dataset.

    Returns
    -------
    list
        One `hcwf.cache.file_identity` entry per file or store; ``[folder, None, None]`` when the folder cannot be read.
    """
    folder, paths, _, _ = _SIDES[side]
    if kw.get(paths):
        return [file_identity(f) for f in kw[paths]]
    if _is_store(kw, side):
        return [file_identity(kw[folder])]
    try:
        return [file_identity(f) for f in sorted(os.scandir(kw[folder]), key=lambda e: e.name) if ".nc" in f.name]
    except OSError:
        return [[str(kw.get(folder)), None, None]]


def cache_parts(kw: dict, lead_label: str, obs_boundary: str, forecast_boundary: str) -> dict:
    """Return what a cached verdict depends on: both datasets (their files), the lead label, the day boundaries, the sample.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check.
    lead_label : str
        The lead label tested.
    obs_boundary : str
        The observations' declared day boundary text.
    forecast_boundary : str
        The forecast's declared day boundary text.

    Returns
    -------
    dict
        The key parts for `hcwf.cache.Store`.
    """
    return {"lead_label": lead_label, "boundaries": [obs_boundary, forecast_boundary],
            "keys": {k: kw.get(k) for k in _KEYS},
            "patterns": [_patterns(kw.get("obs_file_pattern")), _patterns(kw.get("file_pattern"))],
            "obs_files": _files(kw, "obs"), "model_files": _files(kw, "model"),
            "sample": [SAMPLE_INITS, SAMPLE_CELLS, SAMPLE_MEMBERS, LEADS, MAX_LAG]}


def check_labels(kw: dict, labels=("start",), *, obs_boundary: str = "", forecast_boundary_format: str = "{label}",
                 source: str = "forecast", use_cache: bool = True) -> dict:
    """Return the sampled alignment reports of one model under each lead label.

    The sample is read once, and not at all when every label is cached. `ValueError` (or the reader's error) is raised
    when the sample cannot be made.

    Parameters
    ----------
    kw : dict
        The keyword arguments of the sampled check (see `sample_spec`).
    labels : sequence of str, default ``("start",)``
        The lead labels to test.
    obs_boundary : str, default ""
        The observations' declared day boundary text.
    forecast_boundary_format : str, default "{label}"
        A format string turning a label into the forecast's declared day boundary text (used only for the note on
        declared boundaries); it is formatted with ``label=``.
    source : str, default "forecast"
        The name the reports give the forecast.
    use_cache : bool, default True
        Read cached reports and store new ones.

    Returns
    -------
    dict of str to tuple of (hcwf.onset.alignment.ForecastAlignmentReport, bool)
        For each label the report and whether it came from the cache.
    """
    out, todo = {}, []
    store = _store()
    parts = {label: cache_parts(kw, label, obs_boundary, forecast_boundary_format.format(label=label))
             for label in labels} if use_cache else {}
    for label in labels:
        hit = store.get(parts[label], quiet=True) if use_cache else None
        if isinstance(hit, ForecastAlignmentReport):
            out[label] = (hit, True)
        else:
            todo.append(label)
    if todo:
        sample = read_sample(kw)
        for label in todo:
            report = sample_report(sample, label, obs_boundary=obs_boundary, source=source,
                                   forecast_boundary=forecast_boundary_format.format(label=label))
            if use_cache:
                store.put(parts[label], report)
            out[label] = (report, False)
    return out


def check_models(jobs: dict, *, workers: int = 8, use_cache: bool = True) -> dict:
    """Run `check_labels` for several models in parallel threads.

    Parameters
    ----------
    jobs : dict
        Maps a name to ``(kw, labels, options)`` where `options` are the keyword arguments of `check_labels`.
    workers : int, default 8
        The most threads; fewer when there are fewer jobs (one job runs in the calling thread).
    use_cache : bool, default True
        Passed on to `check_labels`.

    Returns
    -------
    dict
        ``{name: {label: (report, cached)}}``, or ``{name: exception}`` for a model whose sample could not be made.
    """
    def run(item):
        """Run one model's check; a failure is returned, not raised."""
        name, (kw, labels, options) = item
        try:
            return name, check_labels(kw, labels, use_cache=use_cache, **options)
        except Exception as exc:  # noqa: BLE001 - reported per model
            return name, exc

    if len(jobs) <= 1:
        return dict(run(item) for item in jobs.items())
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return dict(pool.map(run, jobs.items()))
