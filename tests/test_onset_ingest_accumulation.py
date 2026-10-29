"""Tests for forecast de-accumulation, sub-daily steps, daily boundaries and unit checks in momp.onset.ingest."""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.onset.ingest import (IngestInfo, detect_accumulation, detect_units, ingest_forecast,
                               ingest_forecast_info)

BOUNDARY = "00 UTC, labelled by start"
DATA = Path(os.environ.get("ROMP_TEST_DATA", "/nonexistent"))


def make(per_interval, step_hours, inits=("2001-05-01",), members=0, kind="since_init", units="mm",
         steps=None):
    """A forecast whose per-interval amounts (mm) are per_interval[k] for the k-th step of step_hours.

    kind "since_init" stores the running total, "daily" the amounts. `steps` overrides the step hours.
    """
    amounts = np.asarray(per_interval, dtype=float)
    values = np.cumsum(amounts) if kind == "since_init" else amounts
    hours = np.arange(1, len(amounts) + 1) * step_hours if steps is None else np.asarray(steps)
    if steps is not None:
        values = values[:len(hours)]
    shape = (len(inits), len(hours))
    data = np.broadcast_to(values, shape).copy()
    dims, coords = ["time", "step"], {"time": pd.to_datetime(list(inits)),
                                      "step": pd.to_timedelta(hours, unit="h")}
    if members:
        data = np.stack([data * (m + 1) for m in range(members)], axis=1)
        dims.insert(1, "number")
        coords["number"] = np.arange(members)
    return xr.DataArray(data, dims=dims, coords=coords, attrs={"units": units})


def tenths(fc):
    """The int tenths of the first init (and member), lead last."""
    return fc.tenths.reshape(-1, fc.tenths.shape[-1])[0].tolist()


@pytest.mark.parametrize("kind", ["since_init", "daily"])
def test_24h_steps(kind):
    """24 h steps give the daily totals exactly, with members scaled independently."""
    da = make([2.0, 0.0, 5.5], 24, members=2, kind=kind)
    fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation=kind)
    assert fc.dims == ("init_time", "member", "lead")
    assert fc.tenths[0, 0].tolist() == [20, 0, 55] and fc.tenths[0, 1].tolist() == [40, 0, 110]
    assert fc.ingest_info.accumulation == kind and fc.ingest_info.resolution == "daily"


@pytest.mark.parametrize("kind", ["since_init", "daily"])
@pytest.mark.parametrize("step", [6, 3])
def test_subdaily_steps_exact(kind, step):
    """Sub-daily steps give exact day totals (6 h amounts 0.5 k: days 5, 13, 21 mm), with members."""
    n = 3 * 24 // step
    per_day = 24 // step
    amounts = 0.5 * np.arange(1, n + 1) * (6 / step)
    da = make(amounts, step, members=2, kind=kind)
    fc, info = ingest_forecast_info(da, day_boundary=BOUNDARY, lead_label="start", accumulation=kind)
    expect = [round(10 * amounts[i * per_day:(i + 1) * per_day].sum()) for i in range(3)]
    assert fc.tenths[0, 0].tolist() == expect and fc.tenths[0, 1].tolist() == [2 * e for e in expect]
    assert fc.coords["lead"].tolist() == [1, 2, 3]
    assert info.step_hours == step and info.resolution == f"{step}-hourly"
    if step == 6:
        assert expect == [50, 130, 210]


def test_step_zero_and_integer_hours():
    """A step 0 is ignored, and integer-hour leads with a units attribute are read as hours."""
    da = make([0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], 6, steps=[0, 6, 12, 18, 24, 30, 36, 42, 48])
    fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="since_init")
    assert tenths(fc) == [40, 80]
    hours = da.assign_coords(step=[0, 6, 12, 18, 24, 30, 36, 42, 48])
    hours["step"].attrs["units"] = "hours"
    assert tenths(ingest_forecast(hours, day_boundary=BOUNDARY, lead_label="start",
                                  accumulation="since_init")) == [40, 80]


def test_noise_negatives_clipped():
    """A since-init dip of -0.02 mm becomes 0 with the usual warning (count and lowest value)."""
    acc = np.array([[1.0, 0.98, 4.0]])
    da = xr.DataArray(acc, dims=("time", "step"), coords={"time": pd.to_datetime(["2001-05-01"]),
                                                          "step": pd.to_timedelta([24, 48, 72], unit="h")},
                      attrs={"units": "mm"})
    with pytest.warns(UserWarning, match=r"1 negative values \(lowest -0.02 mm\)"):
        fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="since_init")
    assert tenths(fc) == [10, 0, 30]


def test_auto_detects_both_ways_and_reports():
    """auto picks since_init or daily, and emits an IngestInfo with the kind, evidence and resolution."""
    rng = np.random.default_rng(0)
    amounts = rng.gamma(0.5, 4.0, size=(40, 20)) + 0.1
    acc = np.cumsum(amounts, axis=1)
    coords = {"time": pd.date_range("2001-05-01", periods=40), "step": pd.to_timedelta(np.arange(1, 21) * 24, unit="h")}
    cumulative = xr.DataArray(acc, dims=("time", "step"), coords=coords, attrs={"units": "mm"})
    daily = xr.DataArray(amounts, dims=("time", "step"), coords=coords, attrs={"units": "mm"})
    assert detect_accumulation(cumulative, "step")[0] == "since_init"
    assert detect_accumulation(daily, "step")[0] == "daily"
    with pytest.warns(IngestInfo, match=r"accumulation: since_init \[non-decreasing along lead in 100.0 % "
                                         r"of series; mean .* -> .*\]; steps: daily"):
        fc = ingest_forecast(cumulative, day_boundary=BOUNDARY, lead_label="start")
    assert fc.ingest_info.accumulation == "since_init"
    assert tenths(fc)[:3] == [round(10 * v) for v in amounts[0, :3]]
    with pytest.warns(IngestInfo, match="accumulation: daily"):
        fc = ingest_forecast(daily, day_boundary=BOUNDARY, lead_label="start")
    assert fc.ingest_info.accumulation == "daily"


def test_auto_ambiguous_errors():
    """Non-decreasing data without growth cannot be classified and asks for an explicit setting."""
    flat = np.tile(1.0 + 0.01 * np.arange(10), (30, 1))
    da = xr.DataArray(flat, dims=("time", "step"), coords={"time": pd.date_range("2001-05-01", periods=30),
                                                          "step": pd.to_timedelta(np.arange(1, 11) * 24, unit="h")},
                      attrs={"units": "mm"})
    with pytest.raises(ValueError, match="set accumulation"):
        ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start")
    assert ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start",
                           accumulation="daily").tenths.shape[-1] == 10


def test_missing_boundary_step_errors():
    """A missing 24 h boundary (since_init) or interval (daily) is an error naming it."""
    steps = [6, 12, 18, 30, 36, 42, 48]
    since = make(np.ones(7), 6, steps=steps)
    with pytest.raises(ValueError, match="24 h"):
        ingest_forecast(since, day_boundary=BOUNDARY, lead_label="start", accumulation="since_init")
    per = make(np.ones(7), 6, steps=steps, kind="daily")
    with pytest.raises(ValueError, match="Missing step 24 h"):
        ingest_forecast(per, day_boundary=BOUNDARY, lead_label="start", accumulation="daily")


def test_incomplete_final_day_dropped_with_note():
    """Steps past the last complete day are dropped and noted."""
    da = make(np.ones(10), 6, kind="daily")
    with pytest.warns(IngestInfo, match="dropped the incomplete final day"):
        fc, info = ingest_forecast_info(da, day_boundary=BOUNDARY, lead_label="start", accumulation="daily")
    assert tenths(fc) == [40, 40] and info.notes


def test_unit_contradictions():
    """Declared units that contradict the magnitudes raise with the right suggestion."""
    mm_values = make([20.0, 0.0, 55.0], 24, kind="daily", units="m")
    with pytest.raises(ValueError, match="units='mm'"):
        ingest_forecast(mm_values, day_boundary=BOUNDARY, lead_label="start", accumulation="daily")
    metre_values = make([0.02, 0.0, 0.055], 24, kind="daily", units="mm")
    with pytest.raises(ValueError, match="units='m'"):
        ingest_forecast(metre_values, day_boundary=BOUNDARY, lead_label="start", accumulation="daily")
    fc = ingest_forecast(metre_values, day_boundary=BOUNDARY, lead_label="start", accumulation="daily", units="m")
    assert tenths(fc) == [200, 0, 550]
    assert detect_units(metre_values, "daily")[0] == "m" and detect_units(mm_values, "daily")[0] == "mm"
    since = make([0.02, 0.0, 0.055], 24, units="m")
    assert detect_units(since, "since_init")[0] == "m"


@pytest.mark.parametrize("kind", ["since_init", "daily"])
def test_daily_boundary_12utc_init(kind):
    """A 12 UTC init with 00 UTC days drops the first 12 h, labels from the next date, and keeps members."""
    amounts = np.arange(1, 17, dtype=float)
    da = make(amounts, 6, inits=("2001-05-01T12", "2001-05-02T12"), members=2, kind=kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IngestInfo)
        fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation=kind,
                             daily_boundary="00 UTC")
    assert fc.tenths[0, 0].tolist() == [180, 340, 500] and fc.tenths[1, 1].tolist() == [360, 680, 1000]
    assert fc.init_day.tolist() == [2, 3] and fc.init_month.tolist() == [5, 5]
    assert fc.ingest_info.offsets_hours == (12.0,)
    assert fc.dims[:2] == ("init_time", "member")


def test_daily_boundary_3h_steps_03utc():
    """3 h steps with a 00 UTC init and a 03 UTC day: first interval dropped, init date unchanged."""
    da = make(np.arange(1, 25, dtype=float), 3, kind="daily")
    with pytest.warns(IngestInfo, match="days cut at 03 UTC"):
        fc = ingest_forecast(da, day_boundary="03 UTC, labelled by end", lead_label="end", accumulation="daily",
                             daily_boundary="03 UTC")
    assert tenths(fc) == [440, 1080] and fc.init_day.tolist() == [1]
    since = make(np.arange(1, 25, dtype=float), 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IngestInfo)
        assert tenths(ingest_forecast(since, day_boundary=BOUNDARY, lead_label="end", accumulation="since_init",
                                      daily_boundary="03 UTC")) == [440, 1080]


def test_daily_boundary_without_it_is_24h_from_init():
    """Without daily_boundary a 12 UTC init keeps days of 24 h from the init and the init date."""
    da = make(np.arange(1, 9, dtype=float), 6, inits=("2001-05-01T12",), kind="daily")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IngestInfo)
        fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="daily")
    assert tenths(fc) == [100, 260] and fc.init_day.tolist() == [1]


def test_unreachable_boundary_and_snapping():
    """A boundary off the step grid errors naming the nearest hours; a tolerance snaps with a warning."""
    da = make(np.ones(12), 6, kind="daily")
    with pytest.raises(ValueError, match=r"00 UTC and 06 UTC"):
        ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="daily",
                        daily_boundary="03 UTC")
    with pytest.warns(UserWarning, match=r"snapped by 3 h"):
        fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", accumulation="daily",
                             daily_boundary="03 UTC", boundary_tolerance_hours=3)
    assert tenths(fc) == [40, 40, 40]


@pytest.mark.skipif(not DATA.exists(), reason="set ROMP_TEST_DATA to run real-data checks")
def test_real_aifs_v2_is_daily():
    """The AIFS_v2 archive is detected as daily totals with daily steps."""
    path = DATA / "india/models/AIFS_v2/2020.nc"
    if not path.exists():
        pytest.skip("no AIFS_v2 file")
    da = xr.open_dataset(path)["tp"].isel(time=slice(0, 4), lat=slice(0, 8), lon=slice(0, 8))
    assert detect_accumulation(da, "prediction_timedelta_daily")[0] == "daily"
    with pytest.warns(IngestInfo, match="steps: daily"):
        fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="end")
    assert fc.ingest_info.accumulation == "daily" and fc.ingest_info.resolution == "daily"
