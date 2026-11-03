"""One provider of reference onset dates: a month-day or a table (blending's reference_onset; oracle:
tests/blend/_oracle/nc_utils_references.py) and the computed mean or median onset day; all feed the engine's
search_start, and the climatology reference forecast."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from _oracle import nc_utils_references as legacy
from hcwf.onset import definition as D
from hcwf.onset import engine
from hcwf.onset.climatology import ReferenceOnset, reduce_onset_days
from hcwf.onset.ingest import ingest
from hcwf.references.climatology import compute_climatology_as_forecast
from test_onset_options import BOUNDARY, ENGINE_RULE, HEAD, observed_case

UNITS = np.array(["10.00_70.00", "10.00_70.25", "10.25_70.00", "10.25_70.25"])


@pytest.mark.parametrize("reference_onset", [
    None,
    {"mode": "constant_month_day", "month_day": "06-01"},
    pd.DataFrame({"year": [2001, 2002], "ref_onset_date": [dt.date(2001, 6, 3), dt.date(2002, 5, 29)]}),
    pd.DataFrame({"id": list(UNITS), "ref_onset_date": [dt.date(2000, 6, d) for d in (1, 2, 3, 4)]}),
    pd.DataFrame({"id": list(UNITS) * 2, "year": [2001] * 4 + [2002] * 4,
                  "ref_onset_date": [dt.date(2001 + (i >= 4), 6, 1 + i % 4) for i in range(8)]}),
])
def test_reference_dates_are_the_legacy_joins(reference_onset):
    if reference_onset is None:
        provider = None
    elif isinstance(reference_onset, dict):
        provider = ReferenceOnset(month_day=reference_onset["month_day"])
    else:
        provider = ReferenceOnset(table=reference_onset)
    units, years = np.repeat(UNITS, 2), np.tile([2001, 2002], len(UNITS))

    old = legacy.attach_ref_onset(pd.DataFrame({"id": units, "year": years}), reference_onset)["ref_onset_date"]

    new = provider.dates(units, years) if provider is not None else np.full(len(units), np.datetime64("NaT"), "datetime64[ns]")
    np.testing.assert_array_equal(new, pd.to_datetime(old).to_numpy().astype("datetime64[ns]"))


def test_a_table_is_read_from_a_csv_by_date_or_by_year_and_day(tmp_path):
    (tmp_path / "date.csv").write_text("unit,when\n10.00_70.00,2001-06-03\n10.00_70.25,2001-06-04\n")
    (tmp_path / "yd.csv").write_text("year,day\n2001,5\n2002,9\n")
    by_date = ReferenceOnset.read(tmp_path / "date.csv", date_col="when", unit_col="unit")
    np.testing.assert_array_equal(by_date.dates(["10.00_70.25", "10.00_70.00"], [2001, 2001]),
                                  np.array(["2001-06-04", "2001-06-03"], dtype="datetime64[ns]"))
    by_year = ReferenceOnset.read(tmp_path / "yd.csv", year_col="year", day_col="day", base_date="05-01")
    np.testing.assert_array_equal(by_year.dates(["x", "x"], [2001, 2002]),
                                  np.array(["2001-05-06", "2002-05-10"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="date_col"):
        ReferenceOnset.read(tmp_path / "yd.csv", year_col="year")
    with pytest.raises(ValueError, match="exactly one"):
        ReferenceOnset()


def test_mean_is_rounded_and_median_needs_half_the_years():
    doy = xr.DataArray([[100.0, 200.0, np.nan], [101.0, np.nan, np.nan], [110.0, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                       dims=("year", "lat"), coords={"year": [1, 2, 3, 4]})
    mean = reduce_onset_days(doy, "mean")
    np.testing.assert_array_equal(mean.values, np.round(doy.mean(dim="year", skipna=True).values), strict=False)
    assert mean.values[0] == 104.0 and mean.values[1] == 200.0 and np.isnan(mean.values[2])
    median = reduce_onset_days(doy, "median")
    assert median.values[0] == 101.0 and np.isnan(median.values[1]) and np.isnan(median.values[2])
    with pytest.raises(ValueError, match="statistic"):
        reduce_onset_days(doy, "mode")


def _engine_case(seed=33):
    rng = np.random.default_rng(seed)
    da, thr = observed_case(rng, nan=0.0)
    return D.loads(HEAD + ENGINE_RULE), ingest(da, day_boundary=BOUNDARY, rain_resolution="exact", missing="keep"), thr


def _starts(ds):
    return ds["onset_index"].isel(season=0).values, ds["status"].isel(season=0).values


def test_a_month_day_is_a_search_start_entry_as_blending_starts_after_its_cutoff():
    defn, rain, thr = _engine_case()
    entry = ReferenceOnset(month_day="05-20").search_start([2001, 2002])
    assert entry == "05-20"
    ds = engine.detect_observed(defn, rain, thresholds={"thr": thr}, horizon="reject", search_start={"ref": entry})
    _, status = _starts(ds.sel(start="ref"))
    assert (status == engine.Status.ONSET).any()
    first = np.array([pd.Timestamp(f"{y}-05-20") for y in (2001, 2002)])
    for i, year in enumerate((2001, 2002)):
        ok = status[i] == engine.Status.ONSET
        found = pd.to_datetime(ds.sel(start="ref")["onset_date"].isel(season=0).values[i][ok])
        assert (found >= first[i]).all()


def test_a_table_and_a_computed_field_give_per_unit_search_starts():
    defn, rain, thr = _engine_case()
    cells = [str(i) for i in range(6)]
    table = pd.DataFrame({"id": cells * 2, "year": [2001] * 6 + [2002] * 6,
                          "ref_onset_date": [dt.date(y, m, 15) for y in (2001, 2002) for m in (3, 4, 5, 6, 3, 4)]})
    entry = ReferenceOnset(table=table).search_start([2001, 2002], cells, dims=("lat",))
    assert entry.dims == ("year", "lat") and entry.sizes == {"year": 2, "lat": 6}
    manual = xr.DataArray(np.array([[f"{y}-0{m}-15" for m in (3, 4, 5, 6, 3, 4)] for y in (2001, 2002)],
                                   dtype="datetime64[ns]"), dims=("year", "lat"))
    np.testing.assert_array_equal(entry.values, manual.values)
    base = dict(thresholds={"thr": thr}, horizon="reject")
    via_provider = engine.detect_observed(defn, rain, search_start={"ref": entry}, **base)
    via_dates = engine.detect_observed(defn, rain, search_start={"ref": manual}, **base)
    xr.testing.assert_identical(via_provider.drop_attrs(), via_dates.drop_attrs())

    doy = xr.DataArray(np.array([[100.0, 120.0, np.nan, 150.0, 90.0, 80.0], [102.0, 118.0, np.nan, 148.0, 91.0, 79.0]]),
                       dims=("year", "lat"), coords={"year": [1, 2]})
    computed = ReferenceOnset.from_onsets(doy, "mean")
    entry = computed.search_start([2001, 2002])
    assert np.isnat(entry.values[:, 2]).all()
    assert pd.Timestamp(entry.values[0, 0]) == pd.Timestamp("2001-01-01") + pd.Timedelta(days=100)  # mean 101, doy 101
    assert pd.Timestamp(entry.values[1, 1]) == pd.Timestamp("2002-01-01") + pd.Timedelta(days=118)  # mean 119
    engine.detect_observed(defn, rain, thresholds={"thr": thr}, horizon="reject", search_start={"ref": entry})


def test_the_climatology_forecast_takes_the_provider_or_its_field():
    doy = xr.DataArray(np.array([[150.0, 152.0], [155.0, np.nan]]), dims=("lat", "lon"),
                       coords={"lat": [10.0, 10.5], "lon": [70.0, 70.5]})
    observed = xr.DataArray(np.array([["2013-06-10", "2013-06-12"], ["2013-06-20", "2013-06-22"]], dtype="datetime64[ns]"),
                            dims=("lat", "lon"), coords=doy.coords)
    inits = pd.date_range("2013-05-20", periods=5, freq="2D")
    direct = compute_climatology_as_forecast(doy, 2013, inits, observed, max_forecast_day=20)
    stacked = xr.concat([doy, doy], dim="year").assign_coords(year=[1, 2])
    via = compute_climatology_as_forecast(ReferenceOnset(computed=reduce_onset_days(stacked, "mean")), 2013, inits,
                                          observed, max_forecast_day=20)
    pd.testing.assert_frame_equal(direct, via)
    fixed = compute_climatology_as_forecast(ReferenceOnset(month_day="05-30"), 2013, inits, observed,
                                            max_forecast_day=20)
    assert set(fixed["onset_date"].dropna()) == {"2013-05-30"}
    with pytest.raises(ValueError, match="no day-of-year"):
        ReferenceOnset(table=pd.DataFrame({"year": [2001], "ref_onset_date": [dt.date(2001, 6, 1)]})).doy_field(doy)
