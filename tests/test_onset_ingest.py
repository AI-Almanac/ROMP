"""Tests for momp.onset.ingest: units, rounding, gaps, calendars, day boundaries and forecast leads."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.onset.ingest import ingest, ingest_forecast, parse_day_boundary

BOUNDARY = "00 UTC, labelled by start"
# Optional real-data checks: point ROMP_TEST_DATA at a folder with ethiopia/ and india/ obs and models.
DATA = Path(os.environ.get("ROMP_TEST_DATA", "/nonexistent"))


def make(values, units="mm", times=None, dims=("time", "lat", "lon")):
    """A small DataArray with a daily time axis from 2001-01-01."""
    values = np.asarray(values, dtype=float)
    if times is None:
        times = pd.date_range("2001-01-01", periods=values.shape[0])
    coords = {dims[0]: times}
    for dim, size in zip(dims[1:], values.shape[1:]):
        coords[dim] = np.arange(size, dtype=float)
    return xr.DataArray(values, dims=dims, coords=coords, attrs={} if units is None else {"units": units})


def test_parse_day_boundary():
    """parse_day_boundary returns the UTC hour and label for valid strings and rejects invalid ones."""
    assert parse_day_boundary("00 UTC, labelled by start") == (0.0, "start")
    assert parse_day_boundary("03 UTC, labelled by end") == (3.0, "end")
    assert parse_day_boundary("08:30 IST, labelled by end") == (3.0, "end")
    assert parse_day_boundary("06 EAT, labelled by start") == (3.0, "start")
    for bad in ("00 XYZ, labelled by start", "25 UTC, labelled by start", "00 UTC", "00 UTC, labelled by middle"):
        with pytest.raises(ValueError):
            parse_day_boundary(bad)


def test_basic_ingest_renames_and_orders_dims():
    """Ingest renames the dimensions, orders them, and fills the tenths, calendar and date labels."""
    da = make(np.ones((4, 2, 3)), dims=("TIME", "LATITUDE", "LONGITUDE"))
    rain = ingest(da.transpose("LONGITUDE", "TIME", "LATITUDE"), day_boundary=BOUNDARY, source="x")
    assert rain.dims == ("lon", "lat", "time")
    assert rain.tenths.shape == (3, 2, 4) and rain.tenths.dtype == np.int32
    assert rain.source == "x" and rain.calendar == "standard"
    assert list(rain.coords) == ["lon", "lat", "time"]
    assert list(rain.day) == [1, 2, 3, 4] and set(rain.month) == {1} and set(rain.year) == {2001}


@pytest.mark.parametrize("units,factor", [("mm", 1), ("mm/day", 1), ("mm day-1", 1), ("mm d-1", 1),
                                          ("kg m-2", 1), ("kg m-2 day-1", 1), ("m", 1000), ("m/day", 1000)])
def test_units(units, factor):
    """Supported rainfall units are converted to tenths of a millimetre."""
    rain = ingest(make([[[0.0012]], [[2.5]]], units=units), day_boundary=BOUNDARY)
    assert rain.tenths[0, 0, 1] == round(2.5 * factor * 10)


def test_metres_conversion_values():
    """Rainfall in metres is converted to tenths of a millimetre."""
    rain = ingest(make([[[0.0123]], [[0.0]]], units="m"), day_boundary=BOUNDARY)
    assert rain.tenths[0, 0, 0] == 123


def test_units_argument_overrides_and_missing_units_error():
    """The units argument overrides the data's units, and missing or unsupported units raise ValueError."""
    assert ingest(make([[[0.01]]], units="mm"), day_boundary=BOUNDARY, units="m").tenths[0, 0, 0] == 100
    with pytest.raises(ValueError, match="units"):
        ingest(make([[[1.0]]], units=None), day_boundary=BOUNDARY)
    with pytest.raises(ValueError, match="Unsupported"):
        ingest(make([[[1.0]]], units="mm/s"), day_boundary=BOUNDARY)


def test_rounding_half_even():
    """Values are rounded to tenths with half-even rounding."""
    values = np.array([0.05, 0.15, 0.25, 0.35, 1.04, 1.06]).reshape(-1, 1, 1)
    rain = ingest(make(values), day_boundary=BOUNDARY)
    assert rain.tenths[0, 0].tolist() == [0, 2, 2, 4, 10, 11]


def test_nan_and_negative_handling():
    """NaN and values below the negative tolerance become missing, with a warning for the negatives."""
    values = np.array([np.nan, -0.04, -0.05, -0.2, 3.0]).reshape(-1, 1, 1)
    with pytest.warns(UserWarning, match=r"1 values below -0.05 mm \(lowest -0.20 mm\) were set to missing"):
        rain = ingest(make(values), day_boundary=BOUNDARY)
    assert rain.missing[0, 0].tolist() == [True, False, False, True, False]
    assert rain.tenths[0, 0].tolist() == [0, 0, 0, 0, 30]


def test_gaps_filled_with_warning():
    """Missing days in the time axis are filled as missing values with a warning."""
    da = make(np.arange(1, 6, dtype=float).reshape(-1, 1, 1)).isel(time=[0, 1, 4])
    with pytest.warns(UserWarning, match="Filled 2"):
        rain = ingest(da, day_boundary=BOUNDARY)
    assert rain.tenths.shape[-1] == 5
    assert rain.missing[0, 0].tolist() == [False, False, True, True, False]
    assert rain.tenths[0, 0].tolist() == [10, 20, 0, 0, 50]
    assert rain.day.tolist() == [1, 2, 3, 4, 5]


def test_duplicates_and_decreasing_rejected():
    """Duplicate or decreasing times raise ValueError."""
    da = make(np.ones((3, 1, 1)))
    dup = da.assign_coords(time=pd.to_datetime(["2001-01-01", "2001-01-01", "2001-01-02"]))
    with pytest.raises(ValueError, match="Duplicate"):
        ingest(dup, day_boundary=BOUNDARY)
    rev = da.assign_coords(time=pd.to_datetime(["2001-01-03", "2001-01-02", "2001-01-01"]))
    with pytest.raises(ValueError, match="increasing"):
        ingest(rev, day_boundary=BOUNDARY)


def test_non_daily_rejected():
    """Times that are not once-daily at the same time of day raise ValueError."""
    da = make(np.ones((3, 1, 1)), times=pd.to_datetime(["2001-01-01 00:00", "2001-01-02 06:00",
                                                       "2001-01-03 00:00"]))
    with pytest.raises(ValueError, match="time of day"):
        ingest(da, day_boundary=BOUNDARY)


def test_noleap_calendar_labels():
    """A noleap calendar gives the correct month and day labels."""
    times = xr.date_range("2001-02-27", periods=4, calendar="noleap", use_cftime=True)
    rain = ingest(make(np.ones((4, 1, 1)), times=times), day_boundary=BOUNDARY)
    assert rain.calendar == "noleap"
    assert list(zip(rain.month, rain.day)) == [(2, 27), (2, 28), (3, 1), (3, 2)]


def test_360_day_calendar_labels_and_gap():
    """A 360-day calendar gives the correct labels and gap filling."""
    times = xr.date_range("2001-02-28", periods=5, calendar="360_day", use_cftime=True)
    rain = ingest(make(np.ones((5, 1, 1)), times=times), day_boundary=BOUNDARY)
    assert rain.calendar == "360_day"
    assert list(zip(rain.month, rain.day)) == [(2, 28), (2, 29), (2, 30), (3, 1), (3, 2)]
    with pytest.warns(UserWarning):
        gappy = ingest(make(np.ones((5, 1, 1)), times=times).isel(time=[0, 4]), day_boundary=BOUNDARY)
    assert list(zip(gappy.month, gappy.day)) == [(2, 28), (2, 29), (2, 30), (3, 1), (3, 2)]


def test_day_boundary_validated():
    """An invalid day_boundary raises ValueError."""
    with pytest.raises(ValueError):
        ingest(make(np.ones((2, 1, 1))), day_boundary="midnight")


def test_one_dimensional_series():
    """A one-dimensional time series is ingested."""
    da = make(np.ones((3, 1, 1)))[:, 0, 0]
    assert ingest(da, day_boundary=BOUNDARY).tenths.tolist() == [10, 10, 10]


def make_forecast(leads, n_init=2, members=None, units="mm"):
    """A forecast array with dims (time, [number], lead, lat, lon)."""
    shape = (n_init,) + ((members,) if members else ()) + (len(leads), 2, 2)
    dims = ("time",) + (("number",) if members else ()) + ("step", "lat", "lon")
    coords = {"time": pd.date_range("2001-05-01", periods=n_init, freq="3D"), "step": leads,
              "lat": [10.0, 9.0], "lon": [0.0, 1.0]}
    if members:
        coords["number"] = np.arange(members)
    return xr.DataArray(np.ones(shape), dims=dims, coords=coords, attrs={"units": units})


def test_forecast_dims_and_labels():
    """Forecast ingest gives the expected dims, shape, lead labels and init days."""
    leads = pd.to_timedelta([1, 2, 3], unit="D")
    fc = ingest_forecast(make_forecast(leads, members=3), day_boundary=BOUNDARY, lead_label="end", source="m")
    assert fc.dims == ("init_time", "member", "lat", "lon", "lead")
    assert fc.tenths.shape == (2, 3, 2, 2, 3)
    assert fc.coords["lead"].tolist() == [1, 2, 3]
    assert fc.init_day.tolist() == [1, 4] and fc.lead_label == "end"


def test_forecast_integer_leads_and_explicit_dims():
    """Forecasts with integer leads and explicit dimension names are ingested, and unknown dimension names raise ValueError."""
    da = make_forecast([1, 2]).rename(time="t0", step="k")
    fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", init_dim="t0", lead_dim="k")
    assert fc.dims == ("init_time", "lat", "lon", "lead")
    with pytest.raises(ValueError, match="not in the data dims"):
        ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", init_dim="nope", lead_dim="k")


def test_forecast_lead_checks():
    """Leads that are not consecutive whole days starting at 1 raise ValueError."""
    for leads in ([0, 1, 2], [2, 3, 4], [1, 3, 5], [1, 2, 4]):
        with pytest.raises(ValueError, match="Leads"):
            ingest_forecast(make_forecast(leads), day_boundary=BOUNDARY, lead_label="start")
    with pytest.raises(ValueError, match="Leads"):
        ingest_forecast(make_forecast(pd.to_timedelta([6, 12, 18], unit="h")), day_boundary=BOUNDARY,
                        lead_label="start")


def test_forecast_lead_label_required_and_validated():
    """lead_label is required and must be a valid value."""
    with pytest.raises(TypeError):
        ingest_forecast(make_forecast([1, 2]), day_boundary=BOUNDARY)
    with pytest.raises(ValueError, match="lead_label"):
        ingest_forecast(make_forecast([1, 2]), day_boundary=BOUNDARY, lead_label="middle")


def test_forecast_duplicate_inits_rejected_and_sorted():
    """Duplicate init times raise ValueError and unsorted inits are sorted."""
    da = make_forecast([1, 2])
    dup = da.assign_coords(time=pd.to_datetime(["2001-05-01", "2001-05-01"]))
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_forecast(dup, day_boundary=BOUNDARY, lead_label="start")
    swapped = da.isel(time=[1, 0])
    fc = ingest_forecast(swapped, day_boundary=BOUNDARY, lead_label="start")
    assert fc.init_day.tolist() == [1, 4]


def test_forecast_metres():
    """Forecast rain in metres is converted to tenths of a millimetre."""
    fc = ingest_forecast(make_forecast([1, 2], units="m"), day_boundary=BOUNDARY, lead_label="start")
    assert set(fc.tenths.ravel().tolist()) == {10000}


def test_forecast_negative_rain_is_clipped_to_zero_not_missing():
    """Model rain below zero, however far, becomes 0 mm with a warning; NaN stays missing."""
    da = make_forecast([1, 2], units="mm")
    values = da.values.copy()
    values[0, 0, 0, 0], values[0, 1, 0, 0], values[0, 0, 1, 1] = -0.03, -20.3, np.nan
    da = da.copy(data=values)
    with pytest.warns(UserWarning, match=r"2 negative values \(lowest -20.30 mm\) were set to 0"):
        fc = ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start")
    assert fc.dims == ("init_time", "lat", "lon", "lead")
    assert fc.tenths[0, 0, 0, 0] == 0 and fc.tenths[0, 0, 0, 1] == 0
    assert not fc.missing[0, 0, 0, 0] and not fc.missing[0, 0, 0, 1] and fc.missing[0, 1, 1, 0]


needs_data = pytest.mark.skipif(not DATA.exists(), reason="set ROMP_TEST_DATA to run real-data checks")


@needs_data
def test_real_imd_obs():
    """Real IMD observations ingest with the expected dims, length and dates."""
    da = xr.open_dataset(DATA / "india/obs/data_2020.nc")["RAINFALL"].isel(LATITUDE=slice(0, 6),
                                                                              LONGITUDE=slice(0, 6))
    rain = ingest(da, day_boundary="03 UTC, labelled by end", source="imd")
    assert rain.dims == ("lat", "lon", "time") and rain.tenths.shape[-1] == 366
    assert rain.year[0] == 2020 and (rain.month[-1], rain.day[-1]) == (12, 31)


@needs_data
def test_real_chirps_obs():
    """Real CHIRPS observations ingest with the expected dims and length."""
    da = xr.open_dataset(DATA / "ethiopia/obs/2020.nc")["total_precipitation_24hr"].isel(
        latitude=slice(0, 5), longitude=slice(0, 5))
    rain = ingest(da, day_boundary="00 UTC, labelled by start")
    assert rain.dims == ("lat", "lon", "time") and rain.tenths.shape[-1] == 366
    assert rain.tenths.max() > 0


@needs_data
@pytest.mark.parametrize("path,has_members", [("ethiopia/models/AIFS_ENS_v2/2020.nc", True),
                                              ("ethiopia/models/AIFS_v2/2020.nc", False)])
def test_real_aifs_forecast(path, has_members):
    """Real AIFS forecasts, with and without members, ingest with the expected dims."""
    da = xr.open_dataset(DATA / path)["tp"].isel(time=slice(0, 3), lat=slice(0, 4), lon=slice(0, 4))
    if has_members:
        da = da.isel(number=slice(0, 3))
    fc = ingest_forecast(da, day_boundary="00 UTC, labelled by start", lead_label="end")
    expected = ("init_time",) + (("member",) if has_members else ()) + ("lat", "lon", "lead")
    assert fc.dims == expected
    assert fc.tenths.shape[0] == 3 and fc.tenths.shape[-1] == 50
    assert fc.coords["lead"][0] == 1 and fc.tenths.max() > 0
