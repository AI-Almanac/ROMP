"""Tests for momp.onset.climatology: thresholds, harmonic seasons and the threshold cache."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.onset import climatology
from momp.onset.climatology import ThresholdCache, compute_threshold, compute_thresholds, harmonic_seasons
from momp.onset.ingest import ingest
from momp.onset.model import (Amount, Comparison, Curve, DayRef, Definition, Season, ThresholdSpec)

START = "00 UTC, labelled by start"


def make_rain(fn, years=(2001, 2003), n_cells=2, calendar="standard", source="syn"):
    """Ingest daily rain from fn(date index, cell) -> mm, over whole years."""
    if calendar == "standard":
        times = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31")
    else:
        last_day = 30 if calendar == "360_day" else 31
        times = xr.date_range(f"{years[0]}-01-01", f"{years[1]}-12-{last_day}", calendar=calendar,
                              use_cftime=True)
    values = np.array([[fn(i, t, c) for c in range(n_cells)] for i, t in enumerate(times)], dtype=float)
    da = xr.DataArray(values[:, :, None], dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": np.arange(n_cells) * 1.0, "lon": [0.0]}, attrs={"units": "mm"})
    return ingest(da, day_boundary=START, source=source)


def md(month, day):
    """A fixed month-day DayRef."""
    return DayRef(month=month, day=day)


def test_mean_daily_and_total_whole_year():
    rain = make_rain(lambda i, t, c: 2.0 if c == 0 else float(t.month))
    daily = compute_threshold(ThresholdSpec("mean_daily"), rain, (2001, 2003))
    assert daily.shape == (2, 1)
    assert daily[0, 0] == pytest.approx(2.0)
    months = np.array([t.month for t in pd.date_range("2001-01-01", "2003-12-31")])
    assert daily[1, 0] == pytest.approx(months.mean())
    total = compute_threshold(ThresholdSpec("mean_total", days=5), rain, (2001, 2003))
    assert total[0, 0] == pytest.approx(10.0)


def test_mean_total_during_season():
    def fn(i, t, c):
        return 1.0 if 6 <= t.month <= 9 else 10.0
    rain = make_rain(fn)
    spec = ThresholdSpec("mean_total", days=5, during=(md(6, 1), md(9, 30)))
    assert compute_threshold(spec, rain, (2001, 2003))[0, 0] == pytest.approx(5.0)


def test_quantile_total_hand_computed():
    rain = make_rain(lambda i, t, c: float(t.day) if t.month == 3 and t.day <= 10 else 0.0)
    spec = ThresholdSpec("quantile_total", days=3, during=(md(3, 1), md(3, 10)), quantile=0.5)
    sums = [3 * d + 3 for d in range(1, 9)] * 3
    assert compute_threshold(spec, rain, (2001, 2003))[0, 0] == pytest.approx(np.median(sums))
    spec_hi = ThresholdSpec("quantile_total", days=3, during=(md(3, 1), md(3, 10)), quantile=1.0)
    assert compute_threshold(spec_hi, rain, (2001, 2003))[0, 0] == pytest.approx(27.0)


def test_window_across_year_end_belongs_to_start_year():
    def fn(i, t, c):
        if t.month == 12 and t.day >= 25:
            return 1.0
        if t.month == 1 and t.day <= 5:
            return {2001: 5.0, 2002: 5.0, 2003: 9.0}[t.year]
        return 0.0
    rain = make_rain(fn)
    spec = ThresholdSpec("mean_daily", during=(md(12, 25), md(1, 5)))
    assert compute_threshold(spec, rain, (2001, 2002))[0, 0] == pytest.approx((7 + 25 + 7 + 45) / 24)
    assert compute_threshold(spec, rain, (2002, 2002))[0, 0] == pytest.approx((7 + 45) / 12)


def test_missing_reference_days():
    def fn(i, t, c):
        if c == 1 and i % 5 == 0:
            return np.nan
        if c == 0 and i % 25 == 0:
            return np.nan
        return 2.0
    rain = make_rain(fn)
    out = compute_threshold(ThresholdSpec("mean_daily"), rain, (2001, 2003))
    assert out[0, 0] == pytest.approx(2.0)
    assert np.isnan(out[1, 0])


def test_compute_thresholds_attrs():
    rain = make_rain(lambda i, t, c: 3.0)
    definition = Definition(
        id="d", version=1, title="", citation="", wording="", status="draft", notes="",
        wet_day=Comparison("ge", Amount(tenths=10)), seasons=(Season("s", md(6, 1), md(9, 30)),),
        rule=Curve(baseline="b"),
        thresholds=(("b", ThresholdSpec("mean_daily")), ("t5", ThresholdSpec("mean_total", days=5))))
    out = compute_thresholds(definition, rain, (2001, 2002))
    assert set(out) == {"b", "t5"}
    assert out["t5"].dims == ("lat", "lon") and float(out["t5"][0, 0]) == pytest.approx(15.0)
    assert out["b"].attrs["source"] == "syn" and list(out["b"].attrs["years"]) == [2001, 2002]
    assert "mean_daily" in out["b"].attrs["spec"]


def doy_of(t):
    """Day of a 365-day year (1-based) for a date."""
    return pd.Timestamp(year=2001, month=t.month, day=t.day).dayofyear


def cycle_rain(fn, **kw):
    """Rain whose mean annual cycle is fn(doy), identical every year."""
    return make_rain(lambda i, t, c: 0.0 if (t.month == 2 and t.day == 29) else fn(doy_of(t)), **kw)


def test_harmonic_one_season():
    rain = cycle_rain(lambda d: 5 + 4 * np.cos(2 * np.pi * (d - 200) / 365), years=(2000, 2003))
    out = harmonic_seasons(rain, (2000, 2003))
    assert out.n_seasons.values.tolist() == [[1], [1]]
    assert out.onset_doy.dims == ("season", "lat", "lon")
    assert abs(out.onset_doy.values[0, 0, 0] - 110) <= 3
    assert abs(out.cessation_doy.values[0, 0, 0] - 291) <= 3
    assert np.isnan(out.onset_doy.values[1]).all()


def test_harmonic_two_seasons_noleap():
    rain = cycle_rain(lambda d: 5 + 4 * np.cos(4 * np.pi * (d - 100) / 365), calendar="noleap")
    out = harmonic_seasons(rain, (2001, 2003))
    assert out.n_seasons.values.max() == 2 and out.n_seasons.values.min() == 2
    onset, cessation = out.onset_doy.values[:, 0, 0], out.cessation_doy.values[:, 0, 0]
    assert abs(onset[0] - 55) <= 3 and abs(cessation[0] - 146) <= 3
    assert abs(onset[1] - 237) <= 3 and abs(cessation[1] - 328) <= 3


def test_harmonic_max_seasons_one_and_bad_inputs():
    rain = cycle_rain(lambda d: 5 + 4 * np.cos(4 * np.pi * (d - 100) / 365), calendar="noleap")
    out = harmonic_seasons(rain, (2001, 2003), max_seasons=1)
    assert out.n_seasons.values.max() == 1 and out.onset_doy.shape[0] == 1
    with pytest.raises(ValueError):
        harmonic_seasons(rain, (2001, 2003), max_seasons=3)
    with pytest.raises(ValueError, match="365"):
        harmonic_seasons(make_rain(lambda i, t, c: 1.0, calendar="360_day"), (2001, 2003))


def test_harmonic_cell_without_data():
    rain = cycle_rain(lambda d: 5 + 4 * np.cos(2 * np.pi * (d - 200) / 365), calendar="noleap")
    rain.missing[1] = True
    out = harmonic_seasons(rain, (2001, 2003))
    assert out.n_seasons.values.ravel().tolist() == [1, 0]
    assert np.isnan(out.onset_doy.values[0, 1, 0])


def test_cache_returns_stored_field_without_recomputing(tmp_path, monkeypatch):
    rain = make_rain(lambda i, t, c: 2.0)
    spec = ThresholdSpec("mean_daily")
    calls = []
    original = climatology.compute_threshold
    monkeypatch.setattr(climatology, "compute_threshold",
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    cache = ThresholdCache(tmp_path / "cache")
    first = cache.get_or_compute(spec, rain, (2001, 2003), "mean_daily_rain")
    second = cache.get_or_compute(spec, rain, (2001, 2003), "mean_daily_rain")
    assert len(calls) == 1
    xr.testing.assert_allclose(first, second)
    assert float(second[0, 0]) == pytest.approx(2.0) and second.attrs["source"] == "syn"
    assert len(list((tmp_path / "cache").glob("*.nc"))) == 1
    cache.get_or_compute(spec, rain, (2001, 2002), "mean_daily_rain")
    cache.get_or_compute(ThresholdSpec("mean_total", days=2), rain, (2001, 2003), "mean_daily_rain")
    assert len(calls) == 3


def test_cache_key_depends_on_source_and_grid():
    spec = ThresholdSpec("mean_daily")
    a = make_rain(lambda i, t, c: 1.0, source="a")
    b = make_rain(lambda i, t, c: 1.0, source="b")
    c = make_rain(lambda i, t, c: 1.0, source="a", n_cells=3)
    keys = {ThresholdCache.key(spec, r, (2001, 2003)) for r in (a, b, c)}
    assert len(keys) == 3
    assert ThresholdCache.key(spec, a, (2001, 2003)) == ThresholdCache.key(spec, a, (2001, 2003))
