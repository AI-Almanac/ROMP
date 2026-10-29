"""Tests for momp.onset.alignment on synthetic rainfall with known shifts."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.onset.alignment import check_alignment, check_forecast_alignment
from momp.onset.ingest import ingest, ingest_forecast

START = "00 UTC, labelled by start"
N_DAYS = 200


def rain_series(seed=0, n_cells=40, n_days=N_DAYS + 10):
    """Temporally correlated, skewed rain (mm) with shape (n_days, lat=n_cells, lon=1)."""
    rng = np.random.default_rng(seed)
    z = np.zeros((n_days, n_cells))
    z[0] = rng.normal(size=n_cells)
    for t in range(1, n_days):
        z[t] = 0.6 * z[t - 1] + 0.8 * rng.normal(size=n_cells)
    return np.maximum(z - 0.3, 0.0) ** 2 * 5


def to_daily(values, start="2001-01-01", boundary=START, lat0=0.0, source="x"):
    """Ingest (time, lat, 1) values as DailyRain."""
    n_t, n_lat = values.shape
    da = xr.DataArray(values[:, :, None], dims=("time", "lat", "lon"),
                      coords={"time": pd.date_range(start, periods=n_t), "lat": lat0 + np.arange(n_lat) * 0.25,
                              "lon": [10.0]}, attrs={"units": "mm"})
    return ingest(da, day_boundary=boundary, source=source)


def test_unshifted_is_aligned():
    """Identical series are reported as aligned with lag 0 and correlation 1."""
    base = rain_series()
    report = check_alignment(to_daily(base), to_daily(base))
    assert report.verdict == "aligned" and report.best_lag == 0
    assert report.correlations[0] == pytest.approx(1.0)
    assert report.n_cells == 40 and "aligned" in report.summary()


@pytest.mark.parametrize("shift", [1, -1])
def test_one_day_shift_detected(shift):
    """A series shifted by one day in either direction is reported with that best lag."""
    base = rain_series()
    ref = to_daily(base[3:N_DAYS + 3])
    other = to_daily(np.roll(base, -shift, axis=0)[3:N_DAYS + 3])
    report = check_alignment(ref, other)
    assert report.best_lag == shift
    assert report.verdict == f"likely shifted by {shift:+d} day(s)"
    assert report.correlations[shift] > 0.95 > report.correlations[0]


def test_noise_is_inconclusive():
    """Two unrelated noise series give an inconclusive verdict."""
    report = check_alignment(to_daily(rain_series(1)), to_daily(rain_series(2)))
    assert report.verdict == "inconclusive"


def test_no_shared_cells_is_inconclusive():
    """Series on non-overlapping grids give an inconclusive verdict noting no shared cells."""
    base = rain_series()
    report = check_alignment(to_daily(base), to_daily(base, lat0=0.1))
    assert report.verdict == "inconclusive" and "no shared cells" in report.notes
    assert report.correlations == {}


def test_no_overlap_and_calendar_mismatch():
    """Non-overlapping dates and differing calendars are noted and give an inconclusive verdict."""
    base = rain_series()
    assert "no overlapping dates" in check_alignment(to_daily(base), to_daily(base, start="2005-01-01")).notes
    times = xr.date_range("2001-01-01", periods=len(base), calendar="noleap", use_cftime=True)
    da = xr.DataArray(base[:, :, None], dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": np.arange(40) * 0.25, "lon": [10.0]}, attrs={"units": "mm"})
    other = ingest(da, day_boundary=START)
    report = check_alignment(to_daily(base), other)
    assert report.verdict == "inconclusive" and any("calendars differ" in n for n in report.notes)


def test_declared_boundary_mismatch_noted():
    """Differing declared day boundaries are noted in the report, and matching ones are not."""
    base = rain_series()
    report = check_alignment(to_daily(base), to_daily(base, boundary="03 UTC, labelled by end"))
    assert any("day boundaries differ" in n and "-0.88" in n for n in report.notes)
    assert not any("differ" in n for n in check_alignment(to_daily(base), to_daily(base)).notes)


def build_forecast(base, label_shift, n_leads=4, members=None):
    """Forecast whose lead k equals the observed day init + k - 1 + label_shift (a perfect 'start' forecast if 0)."""
    inits = np.arange(5, 150, 3)
    data = np.stack([[base[i + k - 1 + label_shift] for k in range(1, n_leads + 1)] for i in inits])
    data = data[:, :, :, None]
    dims = ("time", "step", "lat", "lon")
    if members:
        rng = np.random.default_rng(3)
        data = data[:, None] * rng.uniform(0.8, 1.2, size=(1, members, 1, 1, 1))
        dims = ("time", "number", "step", "lat", "lon")
    coords = {"time": pd.Timestamp("2001-01-01") + pd.to_timedelta(inits, unit="D"), "step": np.arange(1, n_leads + 1),
              "lat": np.arange(base.shape[1]) * 0.25, "lon": [10.0]}
    if members:
        coords["number"] = np.arange(members)
    return xr.DataArray(data, dims=dims, coords=coords, attrs={"units": "mm"})


@pytest.mark.parametrize("label,label_shift,expected_lag", [("start", 0, 0), ("end", 0, -1), ("end", 1, 0),
                                                            ("start", 1, 1)])
def test_forecast_label_shift(label, label_shift, expected_lag):
    """The forecast alignment check finds the lag implied by the lead label and any label shift."""
    base = rain_series()
    obs = to_daily(base)
    fc = ingest_forecast(build_forecast(base, label_shift), day_boundary=START, lead_label=label)
    report = check_forecast_alignment(obs, fc)
    assert sorted(report.per_lead) == [1, 2, 3]
    assert {r.best_lag for r in report.per_lead.values()} == {expected_lag}
    if expected_lag == 0:
        assert report.verdict == "aligned"
    else:
        assert report.verdict.startswith(f"likely shifted by {expected_lag:+d}")
    assert "lead 1" in report.summary()


def test_forecast_label_hint_and_members():
    """A forecast with members labelled as start but shifted suggests lead_label='end' and honours the chosen leads."""
    base = rain_series()
    obs = to_daily(base)
    fc = ingest_forecast(build_forecast(base, 1, members=3), day_boundary=START, lead_label="start")
    report = check_forecast_alignment(obs, fc, leads=(1, 4))
    assert "lead_label='end'" in report.verdict
    assert sorted(report.per_lead) == [1, 4]


def test_forecast_unknown_lead_and_no_cells():
    """An unknown lead raises ValueError, and a forecast sharing no cells with the observations is inconclusive."""
    base = rain_series()
    fc = ingest_forecast(build_forecast(base, 0), day_boundary=START, lead_label="start")
    with pytest.raises(ValueError, match="lead 9"):
        check_forecast_alignment(to_daily(base), fc, leads=(9,))
    report = check_forecast_alignment(to_daily(base, lat0=0.1), fc)
    assert report.verdict == "inconclusive"


# Optional real-data checks: point ROMP_TEST_DATA at a folder with ethiopia/ and india/ obs and models.
DATA = Path(os.environ.get("ROMP_TEST_DATA", "/nonexistent"))


@pytest.mark.skipif(not DATA.exists(), reason="set ROMP_TEST_DATA to run real-data checks")
@pytest.mark.parametrize("obs_path,var,boundary,model_path,good_label", [
    ("ethiopia/obs/2020.nc", "total_precipitation_24hr", "00 UTC, labelled by start",
     "ethiopia/models/AIFS_v2/2020.nc", "start"),
    ("india/obs/data_2020.nc", "RAINFALL", "03 UTC, labelled by end", "india/models/AIFS_v2/2020.nc", "end"),
])
def test_real_data_lead_label(obs_path, var, boundary, model_path, good_label):
    """On real observations and AIFS forecasts, the correct lead label is aligned and the other is shifted."""
    obs = ingest(xr.open_dataset(DATA / obs_path)[var], day_boundary=boundary, source="obs")
    tp = xr.open_dataset(DATA / model_path)["tp"].isel(time=slice(0, 20))
    reports = {label: check_forecast_alignment(obs, ingest_forecast(tp, day_boundary=START, lead_label=label,
                                                                    source="AIFS_v2"))
               for label in ("start", "end")}
    assert reports[good_label].verdict == "aligned"
    assert reports["end" if good_label == "start" else "start"].verdict.startswith("likely shifted")
