"""The day-of-year window rule is implemented once and both the observed and forecast paths use it."""
import numpy as np

from hcwf.onset import calendar as cal
from hcwf.onset import engine


def _old_doy_bounds(keys, years, start_doy, end_doy, start_offset, end_offset):
    """The pre-merge `doy_bounds` body."""
    years = np.asarray(years, dtype=np.int64)
    jan1 = cal.find_days(keys, years, 1, 1)
    jan1_next = cal.find_days(keys, years + 1, 1, 1)
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


def _old_offsets(onset, cessation, start_offset, end_offset):
    """The pre-merge arithmetic of `_derived_windows`."""
    s_doy = np.rint(np.nan_to_num(onset, nan=1.0)).astype(np.int64)
    e_doy = np.rint(np.nan_to_num(cessation, nan=1.0)).astype(np.int64)
    return s_doy - 1 + start_offset, e_doy - 1 + end_offset, e_doy < s_doy


def test_doy_bounds_matches_old_over_grid():
    dates = np.arange("2019-01-01", "2023-01-01", dtype="datetime64[D]")
    ymd = [(d.astype(object).year, d.astype(object).month, d.astype(object).day) for d in dates]
    keys = cal.label_keys(*np.array(ymd).T)
    start = np.array([1, 60, 100.4, 200, 300.5, 366, np.nan, 150, 10])
    end = np.array([30, 59, 250, 100, 20, 365, 100, np.nan, 10])
    for so in (-3, 0, 5):
        for eo in (-2, 0, 7):
            for years in ([2019, 2020, 2021], [2018, 2022, 2023]):
                new = cal.doy_bounds(keys, years, start, end, so, eo)
                old = _old_doy_bounds(keys, years, start, end, so, eo)
                assert np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1])


def test_doy_offsets_matches_old_windows_arithmetic():
    start = np.array([1, 60, 100.4, 200, 300.5, 366, np.nan, 150, 10])
    end = np.array([30, 59, 250, 100, 20, 365, 100, np.nan, 10])
    for so in (-3, 0, 5):
        for eo in (-2, 0, 7):
            s, e, wraps, valid = cal.doy_offsets(start, end, so, eo)
            os_, oe, ow = _old_offsets(start, end, so, eo)
            assert np.array_equal(s, os_) and np.array_equal(e, oe) and np.array_equal(wraps, ow)
            assert np.array_equal(valid, np.isfinite(start) & np.isfinite(end))


def test_engine_has_no_second_doy_rounding():
    import inspect
    assert "np.rint" not in inspect.getsource(engine._derived_windows)
    assert "np.rint" not in inspect.getsource(engine._derived_bounds)
