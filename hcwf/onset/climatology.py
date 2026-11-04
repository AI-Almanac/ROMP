"""Reference-climatology products: per-cell thresholds, harmonic season analysis and a threshold cache.

The module computes what a reference climatology gives an onset definition. `compute_threshold` and
`compute_thresholds` turn the `hcwf.onset.model.ThresholdSpec` entries of a definition into per-series thresholds in mm
from ingested rain (`hcwf.onset.model.DailyRain`); `compute_thresholds_from_forecasts` does the same from forecast days
(the model's own climatology); `ThresholdCache` keeps computed fields in a `hcwf.cache.Store`. `harmonic_seasons`
derives seasons from the mean annual cycle. `ReferenceOnset` is the reference (climatological) onset date, with
`reduce_onset_days` for its computed form. `hcwf.onset.engine` and `hcwf.onset.detect` consume these products.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import NETCDF, Store
from hcwf.onset.calendar import fixed_bounds, label_keys
from hcwf.onset.model import Definition, ThresholdSpec, resolution_of
from hcwf.onset.units import DAYS_IN_MONTH
from hcwf.spatial import ids

__all__ = [
    "ReferenceOnset", "ThresholdCache", "compute_thresholds", "compute_thresholds_from_forecasts",
    "harmonic_seasons", "reduce_onset_days",
]


MAX_MISSING_FRACTION = 0.10
"""The largest fraction of reference days that may be missing before a threshold is NaN."""
_CHUNK_VALUES = 1 << 24
"""The most values of rain held as float64 at once when thresholds are computed in chunks of series."""
_DAYS_IN_MONTH = np.array(DAYS_IN_MONTH)
"""The days of each month of a 365-day year."""
_CUMULATIVE_DAYS = np.concatenate([[0], np.cumsum(_DAYS_IN_MONTH)])
"""The days of the year before each month (13 entries, the last is 365)."""
_YEAR_LENGTH = 365
"""The days of the year the harmonic analysis uses (Feb 29 is dropped)."""


def _year_windows(spec: ThresholdSpec, rain, years: tuple[int, int], partial_years: bool = False) -> list[tuple[int, int]]:
    """Return the inclusive ``(first, last)`` time indices of the ``during`` window in each reference year.

    A window crossing the year end belongs to the year of its start. Years whose window is not fully inside the data
    are skipped, or with ``partial_years`` used for the days the data has (a year with none of them is still skipped).
    Without a ``during`` window the window is the whole of each year that has days.

    Parameters
    ----------
    spec : hcwf.onset.model.ThresholdSpec
        The threshold, whose ``during`` bounds give the window (None: the whole year).
    rain : hcwf.onset.model.DailyRain
        The rain, whose ``year``, ``month`` and ``day`` label each day.
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    partial_years : bool, default False
        Use the days a window has in the data instead of skipping a year whose window is only partly there.

    Returns
    -------
    list of tuple of (int, int)
        One ``(first, last)`` pair of time indices, both inclusive, for each year that has a window.
    """
    year_list = np.arange(years[0], years[1] + 1)
    if spec.during is None:
        windows = []
        for y in year_list:
            idx = np.nonzero(rain.year == y)[0]
            if idx.size:
                windows.append((int(idx[0]), int(idx[-1])))
        return windows
    keys = label_keys(rain.year, rain.month, rain.day)
    if partial_years:
        return _partial_windows(keys, year_list, *spec.during)
    first, last = fixed_bounds(keys, year_list, spec.during[0], spec.during[1])
    return [(int(a), int(b)) for a, b in zip(first, last) if a >= 0 and b >= a]


def _partial_windows(keys: np.ndarray, year_list: np.ndarray, start, end) -> list[tuple[int, int]]:
    """Return the window of each year cut to the days the data holds.

    The first day is the first on or after the start, the last the last on or before the end.

    Parameters
    ----------
    keys : numpy.ndarray of int64, shape (time,)
        The sorted date key (``year * 10000 + month * 100 + day``) of each day of the data.
    year_list : numpy.ndarray of int64, shape (year,)
        The reference years.
    start : hcwf.onset.model.DayRef
        The start of the window, a month-day with an optional day offset.
    end : hcwf.onset.model.DayRef
        The end of the window; one that is earlier in the year than ``start`` wraps into the next year.

    Returns
    -------
    list of tuple of (int, int)
        The ``(first, last)`` time indices, both inclusive, of each year that has at least one day.

    Raises
    ------
    ValueError
        A bound is climatological (a per-cell day), which has to be resolved per cell first.
    """
    if start.climatological is not None or end.climatological is not None:
        raise ValueError("thresholds take month-day bounds; resolve climatological bounds per cell")
    wraps = (end.month, end.day) < (start.month, start.day)
    low = year_list * 10000 + start.month * 100 + start.day
    high = (year_list + wraps) * 10000 + end.month * 100 + end.day
    first = np.searchsorted(keys, low, side="left") + start.offset_days
    last = np.searchsorted(keys, high, side="right") - 1 + end.offset_days
    first, last = np.maximum(first, 0), np.minimum(last, len(keys) - 1)
    return [(int(a), int(b)) for a, b in zip(first, last) if b >= a]


def _window_sum(block: np.ndarray, sequential: bool) -> np.ndarray:
    """Sum a block over time: numpy's own order, or day by day as a ``(time, series)`` array reduces.

    The day-by-day order is the one blending's per-file totals take; it differs from the pairwise sum in the last bits.

    Parameters
    ----------
    block : numpy.ndarray of float64, shape (series, time)
        The values to sum.
    sequential : bool
        Add day by day instead of with numpy's pairwise sum.

    Returns
    -------
    numpy.ndarray of float64, shape (series,)
        The sum of each series.
    """
    if not sequential:
        return block.sum(axis=1)
    return np.add.reduce(np.ascontiguousarray(block.T), axis=0)


def _add_compensated(total: np.ndarray, compensation: np.ndarray, value: np.ndarray) -> None:
    """Add ``value`` to ``total`` with Kahan compensation, in place, as pandas' groupby sum adds (blending's totals over files).

    Parameters
    ----------
    total : numpy.ndarray of float64, shape (series,)
        The running sum; updated in place.
    compensation : numpy.ndarray of float64, shape (series,)
        The running compensation; updated in place.
    value : numpy.ndarray of float64, shape (series,)
        The value to add.
    """
    y = value - compensation
    t = total + y
    compensation[:] = (t - total) - y
    total[:] = t


def _threshold_rows(spec: ThresholdSpec, x: np.ndarray, valid: np.ndarray, windows: list[tuple[int, int]],
                    max_missing_fraction: float = MAX_MISSING_FRACTION, sequential: bool = False) -> np.ndarray:
    """Compute the threshold in mm of each series, NaN where too many reference days are missing.

    Parameters
    ----------
    spec : hcwf.onset.model.ThresholdSpec
        The threshold: its ``kind``, ``days`` and ``quantile``.
    x : numpy.ndarray of float64, shape (series, time)
        Daily rain in mm (tenths already divided by 10 unless the rain is exact).
    valid : numpy.ndarray of bool, shape (series, time)
        False for a missing day.
    windows : list of tuple of (int, int)
        The inclusive ``(first, last)`` time indices of each reference year's window.
    max_missing_fraction : float, default MAX_MISSING_FRACTION
        A series with a larger fraction of its window days missing is NaN.
    sequential : bool, default False
        Sum day by day per year and add the years with Kahan compensation, the order of blending's totals (used for
        exact rain).

    Returns
    -------
    numpy.ndarray of float64, shape (series,)
        The threshold of each series in mm.

    Raises
    ------
    ValueError
        The kind is ``quantile_total`` and the quantile is None or outside 0..1.
    """
    n_days = sum(b - a + 1 for a, b in windows)
    if n_days == 0:
        return np.full(x.shape[0], np.nan)
    if spec.kind in ("mean_total", "mean_daily"):
        total, compensation = np.zeros(x.shape[0]), np.zeros(x.shape[0])
        count = np.zeros(x.shape[0])
        for a, b in windows:
            part = _window_sum(np.where(valid[:, a:b + 1], x[:, a:b + 1], 0.0), sequential)
            if sequential:
                _add_compensated(total, compensation, part)
            else:
                total += part
            count += valid[:, a:b + 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
        mean = np.where(count > 0, mean, np.nan)
        out = mean * spec.days if spec.kind == "mean_total" else mean
    else:
        if spec.quantile is None or not 0 <= spec.quantile <= 1:
            raise ValueError("quantile_total needs a quantile between 0 and 1")
        sums, count = [], np.zeros(x.shape[0])
        for a, b in windows:
            xv = np.where(valid[:, a:b + 1], x[:, a:b + 1], 0.0)
            count += valid[:, a:b + 1].sum(axis=1)
            if xv.shape[1] < spec.days:
                continue
            cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(xv, axis=1)], axis=1)
            bad = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(~valid[:, a:b + 1], axis=1)], axis=1)
            totals = cs[:, spec.days:] - cs[:, :-spec.days]
            sums.append(np.where(bad[:, spec.days:] - bad[:, :-spec.days] == 0, totals, np.nan))
        if not sums:
            return np.full(x.shape[0], np.nan)
        with np.errstate(all="ignore"):
            out = np.nanquantile(np.concatenate(sums, axis=1), spec.quantile, axis=1)
    missing_days = n_days - count
    return np.where(missing_days / n_days > max_missing_fraction, np.nan, out)


def _cells_of(rain):
    """Return the cells a rain holds only some of.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The rain.

    Returns
    -------
    hcwf.onset.model.Cells or None
        The cells, or None for a rain over its whole grid.
    """
    return getattr(rain, "cells", None)


def compute_threshold(spec: ThresholdSpec, rain, years: tuple[int, int], *,
                      max_missing_fraction: float = MAX_MISSING_FRACTION, partial_years: bool = False) -> np.ndarray:
    """Compute the per-series threshold in mm (spatial shape) from the reference years, both inclusive.

    ``mean_total`` is ``days`` x the mean daily rain over the ``during`` days pooled across years; ``mean_daily`` is that
    mean; ``quantile_total`` is the quantile (linear interpolation) of all ``days``-day running totals lying wholly
    inside ``during``, pooled across years. Windows crossing the year end belong to the year of their start. A series
    (grid cell or unit: the spatial dims are whatever ``rain`` has, so a unit series gives a unit threshold) is NaN when
    more than ``max_missing_fraction`` (default 10 %; 1.0 never) of its reference days are missing.

    Options reproducing onset_blending's computed threshold: ``max_missing_fraction=1.0``, ``partial_years=True``
    (a year whose window is only partly in the data counts its days, where by default it is skipped) and float rain
    (``rain_resolution "exact"``: millimetres are used as they are, summed day by day per year and the years added with
    Kahan compensation, the order of blending's per-file totals and groupby sum). The caller sets ``days`` (blending
    defaults it to the trigger window).

    A rain that holds only some cells of its grid (``rain.cells``) gives the threshold on the whole grid: the cells it
    lacks have what a series of masked days gives.

    Parameters
    ----------
    spec : hcwf.onset.model.ThresholdSpec
        The threshold to compute.
    rain : hcwf.onset.model.DailyRain
        The reference rain, ``int32`` tenths or ``float64`` mm, with dims of the spatial dims and ``time`` last.
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    max_missing_fraction : float, default MAX_MISSING_FRACTION
        A series with a larger fraction of its reference days missing is NaN.
    partial_years : bool, default False
        Count the days of a year whose window is only partly in the data instead of skipping that year.

    Returns
    -------
    numpy.ndarray of float64, shape spatial
        The threshold of each series in mm; the shape is that of the spatial dims of ``rain`` (the whole grid for a rain
        of some cells).

    Notes
    -----
    A `ValueError` comes from the computation when the kind is ``quantile_total`` without a quantile between 0 and 1,
    or when a window bound is climatological with ``partial_years``.
    """
    cells = _cells_of(rain)
    kept = _threshold_values(spec, rain, years, max_missing_fraction, partial_years)
    if cells is None:
        return kept
    return cells.spread(kept, _threshold_values(spec, cells.masked(rain), years, max_missing_fraction, partial_years))


def _threshold_values(spec: ThresholdSpec, rain, years: tuple[int, int], max_missing_fraction: float,
                      partial_years: bool) -> np.ndarray:
    """Compute the threshold of the series ``rain`` holds, in chunks of series (see `compute_threshold`).

    Parameters
    ----------
    spec : hcwf.onset.model.ThresholdSpec
        The threshold to compute.
    rain : hcwf.onset.model.DailyRain
        The reference rain.
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    max_missing_fraction : float
        A series with a larger fraction of its reference days missing is NaN.
    partial_years : bool
        Count the days of a year whose window is only partly in the data.

    Returns
    -------
    numpy.ndarray of float64, shape spatial
        The threshold in mm of each series, in the shape of the spatial dims of ``rain``.
    """
    windows = _year_windows(spec, rain, years, partial_years)
    spatial_shape = rain.tenths.shape[:-1]
    n_time = rain.tenths.shape[-1]
    flat = rain.tenths.reshape(-1, n_time)
    flat_missing = rain.missing.reshape(-1, n_time)
    exact = resolution_of(flat) == "exact"
    out = np.empty(flat.shape[0])
    step = max(1, _CHUNK_VALUES // max(n_time, 1))
    for a in range(0, flat.shape[0], step):
        x = flat[a:a + step].astype(np.float64)
        if not exact:
            x = x / 10.0
        out[a:a + step] = _threshold_rows(spec, x, ~flat_missing[a:a + step], windows, max_missing_fraction, exact)
    return out.reshape(spatial_shape)


def _spatial_axes(rain) -> tuple[tuple[str, ...], dict]:
    """Return the dims and coordinates of a per-series result over ``rain``.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The rain.

    Returns
    -------
    dims : tuple of str
        The spatial dims of ``rain``, or the whole grid's for a rain of some cells.
    coords : dict
        The coordinates of those dims.
    """
    cells = _cells_of(rain)
    if cells is not None:
        return cells.dims, cells.coords()
    dims = rain.dims[:-1]
    return dims, {d: rain.coords[d] for d in dims}


def _threshold_array(name: str, values: np.ndarray, spec: ThresholdSpec, rain, years: tuple[int, int]) -> xr.DataArray:
    """Wrap a threshold field as a DataArray whose attrs record its provenance.

    Parameters
    ----------
    name : str
        The name of the threshold.
    values : numpy.ndarray of float64, shape spatial
        The threshold in mm.
    spec : hcwf.onset.model.ThresholdSpec
        The threshold specification, recorded as ``spec`` (its ``repr``).
    rain : hcwf.onset.model.DailyRain
        The reference rain, which gives the dims, coordinates and ``source``.
    years : tuple of (int, int)
        The reference years, recorded as ``years``.

    Returns
    -------
    xarray.DataArray
        The field, with attrs ``units`` (``"mm"``), ``source``, ``years`` and ``spec``.
    """
    dims, coords = _spatial_axes(rain)
    return xr.DataArray(values, dims=dims, coords=coords, name=name,
                        attrs={"units": "mm", "source": rain.source, "years": [int(years[0]), int(years[1])],
                               "spec": repr(spec)})


def compute_thresholds(definition: Definition, rain, years: tuple[int, int], **options) -> dict[str, xr.DataArray]:
    """Compute every threshold the definition names, as DataArrays in mm.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition, whose ``thresholds`` are computed.
    rain : hcwf.onset.model.DailyRain
        The reference rain.
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    **options : dict
        Passed on to `compute_threshold` (``max_missing_fraction``, ``partial_years``).

    Returns
    -------
    dict of str to xarray.DataArray
        The field of each threshold by name, in mm, with attrs ``source``, ``years`` and ``spec``.
    """
    return {name: _threshold_array(name, compute_threshold(spec, rain, years, **options), spec, rain, years)
            for name, spec in definition.thresholds}


def _during_mask(forecast, spec: ThresholdSpec, years: tuple[int, int], first_lead: int, last_lead: int) -> np.ndarray:
    """Find which ``(init, lead)`` cells of a forecast count towards a threshold.

    A lead counts when its date label lies in the ``during`` window (the whole year when None) of a year in ``years``,
    and its lead number lies in ``first_lead`` to ``last_lead``. A window crossing the year end belongs to the year of its
    start.

    Parameters
    ----------
    forecast : hcwf.onset.model.ForecastRain
        The forecast, which gives the date label of each lead.
    spec : hcwf.onset.model.ThresholdSpec
        The threshold, whose ``during`` bounds give the window.
    years : tuple of (int, int)
        The first and last year, inclusive, of the window.
    first_lead : int
        The first lead counted (1-based).
    last_lead : int
        The last lead counted, inclusive.

    Returns
    -------
    numpy.ndarray of bool, shape (init, lead)
        True for a cell that counts.
    """
    from hcwf.onset.engine import _bound_key, _label_table

    n_leads = forecast.tenths.shape[-1]
    label_y, label_m, label_d = _label_table(forecast, n_leads)
    keys = label_keys(label_y, label_m, label_d)
    leads = np.zeros(n_leads, dtype=bool)
    leads[max(first_lead, 1) - 1:min(last_lead, n_leads)] = True
    if spec.during is None:
        hit = (label_y >= years[0]) & (label_y <= years[1])
        return hit & leads[None, :]
    start_ref, end_ref = spec.during
    wraps = (end_ref.month, end_ref.day) < (start_ref.month, start_ref.day)
    hit = np.zeros(keys.shape, dtype=bool)
    for shift in (-1, 0, 1):
        year = (label_y + shift).ravel()
        low = _bound_key(forecast.calendar, year, start_ref).reshape(keys.shape)
        high = _bound_key(forecast.calendar, year + wraps, end_ref).reshape(keys.shape)
        year = year.reshape(keys.shape)
        hit |= (year >= years[0]) & (year <= years[1]) & (keys >= low) & (keys <= high)
    return hit & leads[None, :]


def compute_thresholds_from_forecasts(definition: Definition, forecasts: Iterable, years: tuple[int, int],
                                      lead_window: tuple[int, int] = (1, 30),
                                      windows: Sequence[tuple[int, int]] | None = None) -> dict[str, xr.DataArray]:
    """Compute the model's own climatology: every threshold of the definition from forecast days, in mm.

    Streams through the forecasts (``ForecastRain`` objects on one grid), summing per cell the valid forecast days whose
    date label (from ``lead_label``) lies in the threshold's ``during`` window of a year in ``years`` (inclusive), over the
    leads of ``lead_window``, with members pooled. ``mean_total`` is ``days`` x that mean and ``mean_daily`` the mean;
    ``quantile_total`` is not supported. A cell is NaN when more than 10 % of the expected days are missing. The attrs
    record ``source`` (``"model:<source of the first forecast>"``), ``years``, ``lead_window`` and ``spec``.

    With ``windows`` (inclusive ``(first, last)`` lead pairs, each cut to ``lead_window``) every window is pooled separately
    and each DataArray gains a leading ``window`` dim with coordinates ``window_first`` and ``window_last`` (spec
    section 13). A window with no forecast days in it is NaN.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The onset definition, whose ``thresholds`` are computed.
    forecasts : iterable of hcwf.onset.model.ForecastRain
        The forecasts, all on one grid with the same dims and the same member and spatial shape.
    years : tuple of (int, int)
        The first and last year of the ``during`` windows, inclusive.
    lead_window : tuple of (int, int), default (1, 30)
        The first and last lead counted, inclusive and 1-based.
    windows : sequence of tuple of (int, int), optional
        Verification windows of leads to pool separately; None pools one window, ``lead_window``.

    Returns
    -------
    dict of str to xarray.DataArray
        The field of each threshold by name, in mm, over the spatial dims (``window`` first with ``windows``).

    Raises
    ------
    NotImplementedError
        A threshold is of the kind ``quantile_total``.
    ValueError
        The forecasts differ in dims or in member or spatial shape, or none was given.
    """
    spans = [(max(a, lead_window[0]), min(b, lead_window[1])) for a, b in (windows or [lead_window])]
    n_win = len(spans)
    specs = definition.thresholds
    for name, spec in specs:
        if spec.kind not in ("mean_total", "mean_daily"):
            raise NotImplementedError(
                f"threshold {name!r}: the model climatology supports only mean_total and mean_daily, not {spec.kind}")
    sums = {name: None for name, _ in specs}
    counts = {name: None for name, _ in specs}
    expected = {name: np.zeros(n_win, dtype=np.int64) for name, _ in specs}
    template = None
    for forecast in forecasts:
        if template is None:
            template = forecast
            shape = forecast.tenths.shape[2 if "member" in forecast.dims else 1:-1]
            for name, _ in specs:
                sums[name] = np.zeros((n_win, *shape), dtype=np.int64)
                counts[name] = np.zeros((n_win, *shape), dtype=np.int64)
        elif forecast.tenths.shape[1:-1] != template.tenths.shape[1:-1] or forecast.dims != template.dims:
            raise ValueError("forecasts must share dims and their member and spatial shape")
        valid = (~forecast.missing).astype(np.int64)
        x = np.where(forecast.missing, 0, forecast.tenths).astype(np.int64)
        letters = "abcdefgh"[:forecast.tenths.ndim - 2]
        kept = letters[1:] if "member" in forecast.dims else letters
        n_member = forecast.tenths.shape[1] if "member" in forecast.dims else 1
        for name, spec in specs:
            for w, (first, last) in enumerate(spans):
                mask = _during_mask(forecast, spec, years, first, last).astype(np.int64)
                sums[name][w] += np.einsum(f"i{letters}l,il->{kept}", x, mask)
                counts[name][w] += np.einsum(f"i{letters}l,il->{kept}", valid, mask)
                expected[name][w] += int(mask.sum()) * n_member
    if template is None:
        raise ValueError("no forecasts given")
    dims = tuple(d for d in template.dims if d not in ("init_time", "member", "lead"))
    out = {}
    for name, spec in specs:
        n = expected[name].reshape((-1,) + (1,) * (sums[name].ndim - 1))
        valid_days = counts[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(valid_days > 0, sums[name] / 10.0 / np.maximum(valid_days, 1), np.nan)
            values = mean * spec.days if spec.kind == "mean_total" else mean
            values = np.where(n == 0, np.nan, np.where((n - valid_days) / np.maximum(n, 1) > MAX_MISSING_FRACTION,
                                                       np.nan, values))
        coords = {d: template.coords[d] for d in dims if d in template.coords}
        attrs = {"units": "mm", "source": f"model:{template.source}", "years": [int(years[0]), int(years[1])],
                 "lead_window": [int(lead_window[0]), int(lead_window[1])], "spec": repr(spec)}
        if windows is None:
            out[name] = xr.DataArray(values[0], dims=dims, coords=coords, name=name, attrs=attrs)
        else:
            coords["window_first"] = ("window", np.array([a for a, _ in spans], dtype=np.int64))
            coords["window_last"] = ("window", np.array([b for _, b in spans], dtype=np.int64))
            out[name] = xr.DataArray(values, dims=("window", *dims), coords=coords, name=name, attrs=attrs)
    return out


def _doy_index(month: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Return the 0-based day of a 365-day year for each label.

    Parameters
    ----------
    month : numpy.ndarray of int, shape (time,)
        The month label of each day.
    day : numpy.ndarray of int, shape (time,)
        The day-of-month label of each day.

    Returns
    -------
    numpy.ndarray of int64, shape (time,)
        The index, or -1 for Feb 29 and for a label outside the 365-day year.
    """
    month = np.asarray(month, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    valid = (month >= 1) & (month <= 12) & (day >= 1) & (day <= _DAYS_IN_MONTH[np.clip(month - 1, 0, 11)])
    return np.where(valid, _CUMULATIVE_DAYS[np.clip(month - 1, 0, 11)] + day - 1, -1)


def _mean_cycle(rain, years: tuple[int, int]) -> np.ndarray:
    """Compute the mean annual cycle of the reference years.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The rain, in tenths (``int32``) or mm.
    years : tuple of (int, int)
        The first and last reference year, inclusive.

    Returns
    -------
    numpy.ndarray of float64, shape (series, 365)
        The mean daily rain in mm of each day of the 365-day year, NaN for a day with no valid data.

    Raises
    ------
    ValueError
        No day of the data falls on some day of the 365-day year in ``years``.
    """
    n_time = rain.tenths.shape[-1]
    flat = rain.tenths.reshape(-1, n_time)
    flat_missing = rain.missing.reshape(-1, n_time)
    doy = _doy_index(rain.month, rain.day)
    in_years = (rain.year >= years[0]) & (rain.year <= years[1])
    cycle = np.full((flat.shape[0], _YEAR_LENGTH), np.nan)
    for d in range(_YEAR_LENGTH):
        cols = np.nonzero((doy == d) & in_years)[0]
        if cols.size == 0:
            raise ValueError(
                f"No data for day {d + 1} of a 365-day year in {years}; "
                "harmonic_seasons needs a calendar with 365/366-day years")
        x = flat[:, cols] / 10.0
        valid = ~flat_missing[:, cols]
        count = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cycle[:, d] = np.where(count > 0, np.where(valid, x, 0.0).sum(axis=1) / count, np.nan)
    return cycle


def _anomaly_extrema(cycle: np.ndarray, start: np.ndarray, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the onset and cessation day of year for one segment of the cycle per cell.

    The segment begins at 0-based day ``start`` and runs ``length`` days (both per cell). The cumulative sum of its anomaly
    about its own mean is formed; cessation is the day of its maximum and onset the day after its minimum.

    Parameters
    ----------
    cycle : numpy.ndarray of float64, shape (cell, 365)
        The mean annual cycle.
    start : numpy.ndarray of int64, shape (cell,)
        The 0-based first day of the segment.
    length : numpy.ndarray of int64, shape (cell,)
        The length of the segment in days.

    Returns
    -------
    onset : numpy.ndarray of float64, shape (cell,)
        The 1-based onset day of the 365-day year.
    cessation : numpy.ndarray of float64, shape (cell,)
        The 1-based cessation day of the 365-day year.
    """
    n = cycle.shape[1]
    j = np.arange(n)
    rotated = np.take_along_axis(cycle, (start[:, None] + j[None, :]) % n, axis=1)
    inside = j[None, :] < length[:, None]
    mean = np.where(inside, rotated, 0.0).sum(axis=1) / length
    cum = np.cumsum(np.where(inside, rotated - mean[:, None], 0.0), axis=1)
    j_min = np.argmin(np.where(inside, cum, np.inf), axis=1)
    j_max = np.argmax(np.where(inside, cum, -np.inf), axis=1)
    onset = (start + j_min + 1) % n + 1
    cessation = (start + j_max) % n + 1
    return onset.astype(float), cessation.astype(float)


def harmonic_seasons(rain, years: tuple[int, int], max_seasons: int = 2) -> xr.Dataset:
    """Derive seasons per cell from harmonic analysis of the mean annual cycle (draft reading of Dunning et al. 2016).

    The mean daily cycle over ``years`` (Feb 29 dropped, 365 days) is decomposed with a Fourier transform. When the
    amplitude of the second harmonic exceeds that of the first, the cell has two seasons, otherwise one. For one season
    the cycle is rotated to start at the minimum of the first harmonic; for two, the year is split at the two minima of
    the second harmonic. Within each segment the anomaly about the segment mean is accumulated; onset is the day after
    the minimum of that sum and cessation the day of its maximum. This is a working interpretation, to be checked
    against the paper.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The reference rain.
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    max_seasons : {1, 2}, default 2
        The most seasons a cell can have; 1 gives a single season everywhere.

    Returns
    -------
    xarray.Dataset
        ``n_seasons`` (``int8``, over the spatial dims) and ``onset_doy`` / ``cessation_doy`` (``float64``, over
        ``season`` and the spatial dims) as 1-based days of the 365-day year, NaN where a cell has fewer seasons (or no
        valid data, when ``n_seasons`` is 0). Seasons are ordered by onset day of year. The attrs record ``source``,
        ``years`` and ``method``.

    Raises
    ------
    ValueError
        ``max_seasons`` is not 1 or 2, or the data lacks a day of the 365-day year.
    """
    if max_seasons not in (1, 2):
        raise ValueError("max_seasons must be 1 or 2")
    cells = _cells_of(rain)
    n_seasons, onset, cessation = _harmonic_values(rain, years, max_seasons)
    if cells is not None:   # the cells the rain lacks have what a series of masked days gives
        fill = _harmonic_values(cells.masked(rain), years, max_seasons)
        n_seasons, onset, cessation = (cells.spread(kept, one) for kept, one in zip((n_seasons, onset, cessation), fill))
    else:
        spatial_shape = rain.tenths.shape[:-1]
        n_seasons, onset, cessation = (n_seasons.reshape(spatial_shape), onset.reshape((max_seasons,) + spatial_shape),
                                       cessation.reshape((max_seasons,) + spatial_shape))
    dims, coords = _spatial_axes(rain)
    coords_season = {**coords, "season": np.arange(1, max_seasons + 1)}
    return xr.Dataset(
        {"n_seasons": (dims, n_seasons),
         "onset_doy": (("season",) + dims, onset),
         "cessation_doy": (("season",) + dims, cessation)},
        coords=coords_season,
        attrs={"source": rain.source, "years": [int(years[0]), int(years[1])], "method": "harmonic (draft)"})


def _harmonic_values(rain, years: tuple[int, int], max_seasons: int):
    """Compute the seasons of the series ``rain`` holds, flat (see `harmonic_seasons`).

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The reference rain.
    years : tuple of (int, int)
        The first and last reference year, inclusive.
    max_seasons : int
        The most seasons a cell can have, 1 or 2.

    Returns
    -------
    n_seasons : numpy.ndarray of int8, shape (series,)
        The number of seasons of each series; 0 without valid data.
    onset : numpy.ndarray of float64, shape (season, series)
        The 1-based onset day of year of each season, NaN where there is none.
    cessation : numpy.ndarray of float64, shape (season, series)
        The 1-based cessation day of year of each season, NaN where there is none.
    """
    cycle = _mean_cycle(rain, years)
    ok = np.isfinite(cycle).all(axis=1)
    n_cells = cycle.shape[0]
    onset = np.full((max_seasons, n_cells), np.nan)
    cessation = np.full((max_seasons, n_cells), np.nan)
    n_seasons = np.zeros(n_cells, dtype=np.int8)
    if ok.any():
        q = cycle[ok]
        spectrum = np.fft.rfft(q, axis=1)
        a1, a2 = np.abs(spectrum[:, 1]), np.abs(spectrum[:, 2])
        two = (a2 > a1) if max_seasons == 2 else np.zeros(len(q), dtype=bool)
        n = _YEAR_LENGTH
        t_min1 = ((np.pi - np.angle(spectrum[:, 1])) * n / (2 * np.pi)) % n
        t_min2 = ((np.pi - np.angle(spectrum[:, 2])) * n / (4 * np.pi)) % (n / 2)
        start1 = np.rint(t_min1).astype(np.int64) % n
        full = np.full(len(q), n)
        on1, ce1 = _anomaly_extrema(q, start1, full)
        start2 = np.rint(t_min2).astype(np.int64) % n
        mid = np.rint(t_min2 + n / 2).astype(np.int64) % n
        len_a = (mid - start2) % n
        on_a, ce_a = _anomaly_extrema(q, start2, len_a)
        on_b, ce_b = _anomaly_extrema(q, mid, n - len_a)
        first_is_a = on_a <= on_b
        on = np.stack([np.where(first_is_a, on_a, on_b), np.where(first_is_a, on_b, on_a)])
        ce = np.stack([np.where(first_is_a, ce_a, ce_b), np.where(first_is_a, ce_b, ce_a)])
        idx = np.nonzero(ok)[0]
        onset[0, idx] = np.where(two, on[0], on1)
        cessation[0, idx] = np.where(two, ce[0], ce1)
        if max_seasons == 2:
            onset[1, idx] = np.where(two, on[1], np.nan)
            cessation[1, idx] = np.where(two, ce[1], np.nan)
        n_seasons[idx] = np.where(two, 2, 1)
    return n_seasons, onset, cessation


class ThresholdCache:
    """A cache of computed threshold fields, so each is computed once.

    The fields are kept in a `hcwf.cache.Store` of kind ``"thresholds"``. A field is keyed by its name, the spec, the
    reference years, the data source label, the grid coordinates (unit ids by value) and `files`, the identities
    (`hcwf.cache.file_identity`) of the files the reference is read from, so changed data under the same ``source`` is
    recomputed. Give ingested data a source.

    Parameters
    ----------
    directory : str or pathlib.Path or None
        The folder of the entries, created on first write; None keeps them in memory.
    files : iterable, default ()
        The identities of the files the reference data is read from.

    Attributes
    ----------
    store : hcwf.cache.Store
        The store of the threshold fields.
    files : list
        The identities of the files the reference data is read from.
    """

    def __init__(self, directory, files=()):
        self.store = Store("thresholds", directory, codec=NETCDF, label="Thresholds")
        self.files = list(files)

    def parts(self, spec: ThresholdSpec, rain, years: tuple[int, int], name: str = "", **options) -> dict:
        """Return the parts that identify one threshold computation.

        Parameters
        ----------
        spec : hcwf.onset.model.ThresholdSpec
            The threshold.
        rain : hcwf.onset.model.DailyRain
            The reference rain, which gives the source, grid and resolution.
        years : tuple of (int, int)
            The reference years.
        name : str, default ""
            The name of the threshold.
        **options : dict
            The options of `compute_threshold`; they are part of the key only when given.

        Returns
        -------
        dict
            The parts: ``name``, ``source``, ``years``, ``spec``, ``grid``, ``files``, and ``options`` and ``resolution``
            when they apply.
        """
        parts = {"name": name, "source": rain.source, "years": [int(years[0]), int(years[1])], "spec": repr(spec),
                 "grid": {dim: rain.coords[dim] for dim in rain.dims[:-1]}, "files": self.files}
        if options:
            parts["options"] = {key: options[key] for key in sorted(options)}
        if rain.resolution == "exact":
            parts["resolution"] = "exact"
        return parts

    def get_or_compute(self, spec: ThresholdSpec, rain, years: tuple[int, int], name: str, **options) -> xr.DataArray:
        """Return the stored threshold if present, else compute it and store it.

        Parameters
        ----------
        spec : hcwf.onset.model.ThresholdSpec
            The threshold.
        rain : hcwf.onset.model.DailyRain
            The reference rain.
        years : tuple of (int, int)
            The reference years.
        name : str
            The name of the threshold.
        **options : dict
            Passed on to `compute_threshold`.

        Returns
        -------
        xarray.DataArray
            The threshold field in mm.
        """
        return self.store.get_or_compute(
            self.parts(spec, rain, years, name, **options),
            lambda: _threshold_array(name, compute_threshold(spec, rain, years, **options), spec, rain, years))


# ---------------------------------------------------------------------------
# Reference onset dates: one provider for a fixed month-day, a table, and the computed mean or median
# ---------------------------------------------------------------------------

ONSET_STATISTICS = ("mean", "median")
"""The statistics `reduce_onset_days` can take over the years."""


def reduce_onset_days(doy: xr.DataArray, how: str = "mean") -> xr.DataArray:
    """Reduce onset days of year over the ``year`` dim to the climatological onset day of year.

    ``"mean"`` is the mean over the years that have an onset, rounded half to even (HCWF's climatological onset);
    ``"median"`` is the median of those years, NaN where fewer than half of the years have an onset.

    Parameters
    ----------
    doy : xarray.DataArray
        The onset day of year, NaN for a year without an onset, with a ``year`` dim.
    how : {"mean", "median"}, default "mean"
        The statistic.

    Returns
    -------
    xarray.DataArray
        The climatological onset day of year, over the other dims of ``doy``.

    Raises
    ------
    ValueError
        ``how`` is not one of `ONSET_STATISTICS`.
    """
    if how == "mean":
        return np.round(doy.mean(dim="year"))
    if how != "median":
        raise ValueError(f"the onset statistic must be one of {ONSET_STATISTICS}, got {how!r}")
    values = doy.transpose("year", ...).values
    n_found = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median = np.where(n_found * 2 >= values.shape[0], median, np.nan)
    dims = tuple(d for d in doy.dims if d != "year")
    return xr.DataArray(median, dims=dims, coords={d: doy.coords[d] for d in dims if d in doy.coords})


@dataclass(frozen=True)
class ReferenceOnset:
    """The reference (climatological) onset date: one provider for every source.

    Sources: a fixed month-day (``"06-02"``) in every year and unit, a table of dates keyed by unit id and/or year
    (onset_blending's ``reference_onset``), or a computed field: the mean or median onset day of year over many years
    (`reduce_onset_days`), as the climatology reference forecast and the derived seasons use. Exactly one is set.

    Consumers: `ReferenceOnset.dates` (a date per ``(year, unit)``), `ReferenceOnset.search_start` (the entry for
    the ``search_start`` of the ``detect_*`` functions of `hcwf.onset.engine`, so an onset search never begins before the
    reference onset) and `ReferenceOnset.doy_field` (day of year per cell).
    """

    month_day: str | None = None
    """A month-day ``"MM-DD"`` used in every year and unit."""
    table: pd.DataFrame | None = field(default=None, compare=False)
    """A table with the key columns ``id`` and/or ``year`` and ``ref_onset_date`` (a ``datetime.date``)."""
    computed: xr.DataArray | None = field(default=None, compare=False)
    """A field of the onset day of year over the spatial dims (or ``unit``)."""

    def __post_init__(self):
        """Check that exactly one source is set.

        Raises
        ------
        ValueError
            Not exactly one of `month_day`, `table` and `computed` is set.
        """
        if sum(x is not None for x in (self.month_day, self.table, self.computed)) != 1:
            raise ValueError("a reference onset is exactly one of a month-day, a table or a computed day of year")

    @classmethod
    def read(cls, path, *, date_col: str | None = None, year_col: str | None = None, day_col: str | None = None,
             base_date: str | None = None, unit_col: str | None = None) -> "ReferenceOnset":
        """Read a table of reference onset dates from a CSV.

        The dates come from a date column, or from a year and a day offset from ``base_date``. The table is keyed by the
        unit-id column ``unit_col`` and/or the ``year_col`` (at least one); dates are ``datetime.date``.

        Parameters
        ----------
        path : str or pathlib.Path
            The CSV file.
        date_col : str, optional
            The column holding a parseable date.
        year_col : str, optional
            The column holding the year; also a key of the table.
        day_col : str, optional
            The column holding a day offset from ``base_date`` in that year (with ``year_col`` and ``base_date``).
        base_date : str, optional
            The ``"MM-DD"`` the day offset counts from.
        unit_col : str, optional
            The column holding the unit id; also a key of the table.

        Returns
        -------
        ReferenceOnset
            A reference onset with the table as its source.

        Raises
        ------
        ValueError
            The file has no usable date columns, ``unit_col`` is not a column, or neither ``year_col`` nor ``unit_col``
            gives a key.
        """
        table = pd.read_csv(path, dtype={unit_col: str} if unit_col else None, float_precision="round_trip")
        if date_col and date_col in table.columns:
            table["ref_onset_date"] = pd.to_datetime(table[date_col], errors="coerce").dt.date
        elif year_col and day_col and base_date and year_col in table.columns and day_col in table.columns:
            table["ref_onset_date"] = table.apply(
                lambda row: (pd.to_datetime(f"{int(row[year_col])}-{base_date}")
                             + pd.Timedelta(days=int(row[day_col]))).date(), axis=1)
        else:
            raise ValueError("ref_onset file needs either a parseable 'date_col', or 'year_col'+'day_col'+'base_date'.")
        keys = []
        if unit_col:
            if unit_col not in table.columns:
                raise ValueError(f"ref_onset unit_col '{unit_col}' not in file columns {table.columns.tolist()}")
            table = table.rename(columns={unit_col: "id"})
            table["id"] = ids.normalize(table["id"], context="reference-onset IDs")
            keys.append("id")
        if year_col and year_col in table.columns:
            table = table.rename(columns={year_col: "year"})
            table["year"] = table["year"].astype(int)
            keys.append("year")
        if not keys:
            raise ValueError("ref_onset file needs a 'year_col' and/or 'unit_col' to key on.")
        return cls(table=table[keys + ["ref_onset_date"]].drop_duplicates())

    def dates(self, unit_ids, years) -> np.ndarray:
        """Return the reference onset date of each ``(unit, year)`` pair, row by row.

        A computed field needs ``unit_ids`` to be its unit coordinate (a ``unit`` dim) and gives each year's Jan 1 plus
        its day of year.

        Parameters
        ----------
        unit_ids : array_like of str
            The unit id of each row.
        years : array_like of int
            The year of each row.

        Returns
        -------
        numpy.ndarray of datetime64[ns]
            The date of each row, ``NaT`` without one; the shape of the broadcast ``years`` (and ``unit_ids`` for a table).

        Raises
        ------
        ValueError
            The source is a computed field that is not 1-D over a ``unit`` dim, or the table gives a unit and year more
            than one date.
        """
        years = np.asarray(years, dtype=int)
        none = np.full(len(years), np.datetime64("NaT"), dtype="datetime64[ns]")
        if self.month_day is not None:
            distinct, codes = np.unique(years, return_inverse=True)
            dates = pd.to_datetime(pd.Series([f"{y}-{self.month_day}" for y in distinct]), errors="coerce").to_numpy()
            return dates[codes].astype("datetime64[ns]")
        if self.computed is not None:
            field_ = self.computed
            if field_.ndim != 1 or "unit" not in field_.dims:
                raise ValueError("dates by unit id need a computed reference onset over a 'unit' dim")
            doy = field_.sel(unit=np.asarray(unit_ids)).values
            jan1 = (years - 1970).astype("datetime64[Y]").astype("datetime64[D]")
            with np.errstate(invalid="ignore"):
                offset = np.where(np.isfinite(doy), doy, 1).astype(np.int64) - 1
            out = (jan1 + offset.astype("timedelta64[D]")).astype("datetime64[ns]")
            return np.where(np.isfinite(doy), out, none)
        return self._table_dates(unit_ids, years)

    def _table_dates(self, unit_ids, years):
        """Look up the table's date for each unit and year.

        Parameters
        ----------
        unit_ids : array_like of str
            The unit ids; broadcast together with ``years``.
        years : array_like of int
            The years; broadcast together with ``unit_ids``.

        Returns
        -------
        numpy.ndarray of datetime64[ns]
            The dates in the broadcast shape, ``NaT`` where the table has none.

        Raises
        ------
        ValueError
            The table gives a unit and year more than one date.
        """
        keys = [k for k in ("id", "year") if k in self.table.columns]
        queries = {"id": np.asarray(unit_ids, dtype=str), "year": np.asarray(years, dtype=int)}
        shape = np.broadcast_shapes(queries["id"].shape, queries["year"].shape)
        if not keys or not len(self.table):
            return np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        table = self.table.drop_duplicates(keys + ["ref_onset_date"])
        if table.duplicated(keys).any():
            raise ValueError("The reference onset table gives a unit and year more than one date")
        # Each key as a position among the table's values (-1 when the table lacks it), then one code per pair.
        table_code = np.zeros(len(table), dtype=np.int64)
        query_code = np.zeros(shape, dtype=np.int64)
        for key in keys:
            values = table[key].to_numpy(dtype=str if key == "id" else int)
            known = np.unique(values)
            position = np.searchsorted(known, queries[key]).clip(max=len(known) - 1)
            found = known[position] == queries[key]
            table_code = table_code * len(known) + np.searchsorted(known, values)
            query_code = np.where(found & (query_code >= 0), query_code * len(known) + position, -1)
        order = np.argsort(table_code, kind="stable")
        sorted_code = table_code[order]
        at = np.searchsorted(sorted_code, query_code).clip(max=len(sorted_code) - 1)
        hit = (query_code >= 0) & (sorted_code[at] == query_code)
        onsets = pd.to_datetime(table["ref_onset_date"]).to_numpy().astype("datetime64[ns]")[order]
        return np.where(hit, onsets[at], np.datetime64("NaT")).astype("datetime64[ns]")

    def search_start(self, years, unit_ids=None, *, dims: tuple[str, ...] = ("unit",)):
        """Return the ``search_start`` entry of ``detect_observed`` / ``detect_forecast`` for these years.

        A month-day is the ``"MM-DD"`` string. Otherwise a DataArray of dates (``NaT`` for none) over ``year`` and, for a
        table keyed by unit, the ``dims`` of the units (default one ``unit`` dim holding ``unit_ids``); a computed field over
        its own spatial dims gives each year's date.

        Parameters
        ----------
        years : array_like of int
            The years to give dates for.
        unit_ids : array_like of str, optional
            The unit ids; needed for a table keyed by unit.
        dims : tuple of str, default ``("unit",)``
            The dims of the units; only the first is used, naming the dim that holds ``unit_ids``.

        Returns
        -------
        str or xarray.DataArray
            The ``"MM-DD"`` string, or the dates as ``datetime64[ns]`` over ``year`` and the unit (or spatial) dims.

        Raises
        ------
        ValueError
            A table keyed by unit is used without ``unit_ids``.
        """
        years = np.asarray(years, dtype=int)
        if self.month_day is not None:
            return self.month_day
        if self.computed is not None and (unit_ids is None or "unit" not in self.computed.dims):
            doy = self.computed
            jan1 = (years - 1970).astype("datetime64[Y]").astype("datetime64[D]")
            values = np.where(np.isfinite(doy.values)[None], jan1.reshape((-1,) + (1,) * doy.ndim)
                              + (np.nan_to_num(doy.values, nan=1.0).astype(np.int64) - 1)[None].astype("timedelta64[D]"),
                              np.datetime64("NaT")).astype("datetime64[ns]")
            return xr.DataArray(values, dims=("year", *doy.dims), coords={"year": years, **doy.coords})
        if unit_ids is None:
            keyed = [k for k in ("id", "year") if k in self.table.columns]
            if keyed != ["year"]:
                raise ValueError("a reference onset table keyed by unit needs the unit ids")
            unit_ids = [""]
        unit_ids = np.asarray(unit_ids)
        if self.table is not None:
            values = self._table_dates(unit_ids[None, :], years[:, None])
        else:
            grid_years = np.repeat(years, len(unit_ids))
            grid_units = np.tile(unit_ids, len(years))
            values = self.dates(grid_units, grid_years).reshape(len(years), len(unit_ids))
        return xr.DataArray(values, dims=("year", dims[0]), coords={"year": years, dims[0]: unit_ids})

    def doy_field(self, like: xr.DataArray) -> xr.DataArray:
        """Return the day of year of the reference onset on the grid of ``like`` (the climatology reference forecast's input).

        A computed field is returned as it is; a month-day is its day of year in a year of 365 days. A table varies by
        year (and unit), so it has no single day-of-year field.

        Parameters
        ----------
        like : xarray.DataArray
            The grid a month-day is spread over.

        Returns
        -------
        xarray.DataArray
            The day of year of the reference onset.

        Raises
        ------
        ValueError
            The source is a table.
        """
        if self.computed is not None:
            return self.computed
        if self.month_day is None:
            raise ValueError("a reference onset table has dates per year; it gives no day-of-year field")
        doy = pd.Timestamp(f"2001-{self.month_day}").dayofyear
        return xr.full_like(like.astype(float), float(doy))
