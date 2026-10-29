"""Map month-day and day-of-year search bounds onto a dataset's own calendar labels.

Days are located through integer keys year*10000 + month*100 + day, which
increase with time in every calendar (standard, noleap, 360_day, ...), so no
datetime arithmetic is needed.
"""
from __future__ import annotations

import numpy as np

from momp.onset.model import DayRef


def label_keys(year, month, day) -> np.ndarray:
    """Return the int64 key year*10000 + month*100 + day for each label."""
    return (np.asarray(year, dtype=np.int64) * 10000
            + np.asarray(month, dtype=np.int64) * 100
            + np.asarray(day, dtype=np.int64))


def find_days(keys: np.ndarray, years, month: int, day: int) -> np.ndarray:
    """Index of month-day in each of `years` within the sorted label keys, or -1 where that day is absent."""
    wanted = np.asarray(years, dtype=np.int64) * 10000 + month * 100 + day
    idx = np.searchsorted(keys, wanted)
    found = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == wanted)
    return np.where(found, idx, -1)


def fixed_bounds(keys: np.ndarray, years, start: DayRef, end: DayRef) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive (start, end) day indices of a month-day window in each year, -1 where a bound is outside the data.

    An end month-day earlier than the start month-day falls in the following year.
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


def doy_bounds(keys: np.ndarray, years, start_doy, end_doy, start_offset: int = 0,
               end_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive day indices from 1-based day-of-year bounds, broadcast over years x cells.

    years has shape (Y,); start_doy and end_doy have shape (C,) and may be NaN
    (no such season), giving -1. An end day-of-year before the start falls in the
    following year. Offsets shift the bounds by whole days.
    """
    years = np.asarray(years, dtype=np.int64)
    jan1 = find_days(keys, years, 1, 1)
    jan1_next = find_days(keys, years + 1, 1, 1)
    start_doy = np.asarray(start_doy, dtype=float)
    end_doy = np.asarray(end_doy, dtype=float)
    valid = np.isfinite(start_doy) & np.isfinite(end_doy)
    s = np.where(valid, np.rint(np.nan_to_num(start_doy)), 1).astype(np.int64)
    e = np.where(valid, np.rint(np.nan_to_num(end_doy)), 1).astype(np.int64)
    wraps = e < s
    first = jan1[:, None] + s[None, :] - 1 + start_offset
    end_base = np.where(wraps[None, :], jan1_next[:, None], jan1[:, None])
    last = end_base + e[None, :] - 1 + end_offset
    ok = valid[None, :] & (jan1[:, None] >= 0) & (end_base >= 0)
    return np.where(ok, first, -1), np.where(ok, last, -1)
