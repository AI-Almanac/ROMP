"""Map month-day and day-of-year search bounds onto a dataset's own calendar labels.

Days are located through integer keys ``year*10000 + month*100 + day``, which increase with time in every calendar
(standard, noleap, 360_day, ...), so no datetime arithmetic is needed. The labels come from `hcwf.onset.model.DailyRain`;
the bounds are used by `hcwf.onset.engine` to turn seasons into day indices.
"""
from __future__ import annotations

import numpy as np

from hcwf.onset.model import DayRef

__all__ = []


def label_keys(year, month, day) -> np.ndarray:
    """Return the ``int64`` key ``year*10000 + month*100 + day`` for each label.

    Parameters
    ----------
    year : array_like of int
        The year of each label.
    month : array_like of int
        The month of each label, broadcast against ``year``.
    day : array_like of int
        The day of the month of each label, broadcast against ``year``.

    Returns
    -------
    numpy.ndarray of int64
        The keys, with the broadcast shape of the inputs.
    """
    return (np.asarray(year, dtype=np.int64) * 10000
            + np.asarray(month, dtype=np.int64) * 100
            + np.asarray(day, dtype=np.int64))


def find_days(keys: np.ndarray, years, month: int, day: int) -> np.ndarray:
    """Find a month-day in each of several years among the sorted label keys.

    Parameters
    ----------
    keys : numpy.ndarray of int64, shape (time,)
        The sorted label keys (see `label_keys`).
    years : array_like of int
        The years to look in.
    month : int
        The month of the day to find.
    day : int
        The day of the month to find.

    Returns
    -------
    numpy.ndarray of int64, shape (year,)
        The index in ``keys`` of that day in each year, -1 where the calendar has no such day (for example 29 February
        in a ``noleap`` calendar) or the data does not reach it.
    """
    wanted = np.asarray(years, dtype=np.int64) * 10000 + month * 100 + day
    idx = np.searchsorted(keys, wanted)
    found = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == wanted)
    return np.where(found, idx, -1)


def fixed_bounds(keys: np.ndarray, years, start: DayRef, end: DayRef) -> tuple[np.ndarray, np.ndarray]:
    """Return the inclusive day indices of a month-day window in each year.

    An end month-day earlier than the start month-day falls in the following year. The ``offset_days`` of each
    bound shifts it by whole days.

    Parameters
    ----------
    keys : numpy.ndarray of int64, shape (time,)
        The sorted label keys (see `label_keys`).
    years : array_like of int
        The years of the windows' starts.
    start : hcwf.onset.model.DayRef
        The first day of the window, a fixed month-day.
    end : hcwf.onset.model.DayRef
        The last day of the window, a fixed month-day.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The first and last day index of the window in each year, both ``int64`` with shape (year,); -1 where a bound
        is outside the data.

    Raises
    ------
    ValueError
        A bound is climatological rather than a fixed month-day.
    """
    if start.climatological is not None or end.climatological is not None:
        raise ValueError("fixed_bounds takes month-day bounds; resolve climatological bounds per cell")
    years = np.asarray(years, dtype=np.int64)
    end_years = years + ((end.month, end.day) < (start.month, start.day))
    first = find_days(keys, years, start.month, start.day)
    last = find_days(keys, end_years, end.month, end.day)
    first = np.where(first >= 0, first + start.offset_days, -1)
    last = np.where(last >= 0, last + end.offset_days, -1)
    return first, last


def doy_offsets(start_doy, end_doy, start_offset: int = 0, end_offset: int = 0):
    """Return the day offsets from January 1 of a window given by 1-based day-of-year bounds.

    This is the one place that rounds the day-of-year bounds, applies the offsets and decides which windows end in
    the following year (an end day-of-year before the start). `doy_bounds` maps the offsets onto day indices of a data
    axis; `hcwf.onset.engine` maps them onto calendar labels for forecasts.

    Parameters
    ----------
    start_doy : array_like of float, shape (cell,)
        The first day of the window as a day of the year; NaN where the cell has no such season.
    end_doy : array_like of float, shape (cell,)
        The last day of the window as a day of the year; NaN where the cell has no such season.
    start_offset : int, default 0
        Days added to the first day.
    end_offset : int, default 0
        Days added to the last day.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The offset of the first day from January 1 of the start year and the offset of the last day from January 1
        of the start year, or of the following year where the window wraps (both ``int64``, shape (cell,)); the wrap
        mask (bool); and the validity mask (bool), False where either bound is NaN. NaN bounds count as day 1, so
        the offsets of invalid cells are not meaningful.
    """
    start_doy = np.asarray(start_doy, dtype=float)
    end_doy = np.asarray(end_doy, dtype=float)
    valid = np.isfinite(start_doy) & np.isfinite(end_doy)
    s = np.rint(np.nan_to_num(start_doy, nan=1.0)).astype(np.int64)
    e = np.rint(np.nan_to_num(end_doy, nan=1.0)).astype(np.int64)
    return s - 1 + start_offset, e - 1 + end_offset, e < s, valid


def doy_bounds(keys: np.ndarray, years, start_doy, end_doy, start_offset: int = 0,
               end_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return the inclusive day indices of windows given by 1-based day-of-year bounds, over years and cells.

    An end day-of-year before the start falls in the following year. Offsets shift the bounds by whole days.

    Parameters
    ----------
    keys : numpy.ndarray of int64, shape (time,)
        The sorted label keys (see `label_keys`).
    years : array_like of int, shape (year,)
        The years of the windows' starts.
    start_doy : array_like of float, shape (cell,)
        The first day of the window as a day of the year; NaN where the cell has no such season.
    end_doy : array_like of float, shape (cell,)
        The last day of the window as a day of the year; NaN where the cell has no such season.
    start_offset : int, default 0
        Days added to the first day.
    end_offset : int, default 0
        Days added to the last day.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The first and last day index, both ``int64`` with shape (year, cell); -1 where the season is missing or a
        January 1st is outside the data.
    """
    years = np.asarray(years, dtype=np.int64)
    jan1 = find_days(keys, years, 1, 1)
    jan1_next = find_days(keys, years + 1, 1, 1)
    start_off, end_off, wraps, valid = doy_offsets(start_doy, end_doy, start_offset, end_offset)
    first = jan1[:, None] + start_off[None, :]
    end_base = np.where(wraps[None, :], jan1_next[:, None], jan1[:, None])
    last = end_base + end_off[None, :]
    ok = valid[None, :] & (jan1[:, None] >= 0) & (end_base >= 0)
    return np.where(ok, first, -1), np.where(ok, last, -1)
