"""Thresholds: compute_threshold under blending's options reproduces its computed threshold exactly on unit series;
the one reader of threshold files and the join to units are blending's (oracle: tests/blend/_oracle)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io
import xarray as xr

from _oracle import nc_utils_references as legacy
from _oracle import thresholds as oracle
from hcwf.onset.climatology import ThresholdCache, compute_threshold
from hcwf.onset.model import DailyRain, DayRef, ThresholdSpec
from hcwf.onset.thresholds import (ThresholdScale, read_threshold_grid, read_threshold_table, scale_values,
                                   unit_thresholds)
from hcwf.spatial import ids as spatial_ids

BOUNDARY = "00 UTC, labelled by start"
N_UNITS = 30
IDS = np.array([f"u{i:03d}" for i in range(N_UNITS)])
YEARS = list(range(2001, 2010))


def md(month, day):
    return DayRef(month=month, day=day)


def series(seed=0, missing=0.1, partial=True):
    """Per-year (dates, rain[time, unit]) like blending's observation files, float mm with NaN; 2001 and 2009 partial."""
    rng = np.random.default_rng(seed)
    files = []
    for year in YEARS:
        start = "05-20" if partial and year == 2001 else "01-01"
        end = "08-15" if partial and year == 2009 else "12-31"
        dates = pd.date_range(f"{year}-{start}", f"{year}-{end}")
        rain = rng.gamma(0.4, 8, size=(len(dates), N_UNITS)) * (rng.random((len(dates), N_UNITS)) < 0.5)
        rain[rng.random(rain.shape) < missing] = np.nan
        rain[:, 5] = np.nan  # a unit with no data at all
        files.append((dates, rain))
    return files


def unit_rain(files, exact=True):
    """The files one after the other as one DailyRain over a "unit" dim (the days the files hold, nothing more)."""
    full = pd.DatetimeIndex(np.concatenate([dates.to_numpy() for dates, _ in files]))
    values = np.concatenate([rain for _, rain in files])
    missing = np.isnan(values)
    tenths = np.where(missing, 0.0, values).T.copy() if exact else np.rint(np.where(missing, 0.0, values) * 10).astype(np.int32).T.copy()
    return DailyRain(tenths, missing.T.copy(), ("unit", "time"), {"unit": IDS}, full.year.to_numpy(),
                     full.month.to_numpy(), full.day.to_numpy(), "standard", BOUNDARY, "synthetic")


SPEC = ThresholdSpec("mean_total", days=5, during=(md(5, 1), md(9, 30)))
BLENDING = {"max_missing_fraction": 1.0, "partial_years": True}


@pytest.mark.parametrize("seed,missing", [(0, 0.0), (1, 0.1), (2, 0.2), (3, 0.5)])
def test_blending_options_reproduce_its_computed_threshold_exactly(seed, missing):
    files = series(seed, missing)
    expected = oracle.compute_thresholds(files, IDS, 5, YEARS[0], YEARS[-1], "05-01", "09-30")

    got = compute_threshold(SPEC, unit_rain(files), (YEARS[0], YEARS[-1]), **BLENDING)

    assert got.shape == (N_UNITS,)
    np.testing.assert_array_equal(got, expected.set_index("id")["onset_thresh"].reindex(IDS).to_numpy())
    assert np.isnan(got[5])


def test_each_cells_threshold_is_its_mean_rain_over_the_window():
    files = series(4, 0.15, partial=False)
    got = compute_threshold(SPEC, unit_rain(files), (2001, 2008), **BLENDING)
    totals, days = np.zeros(N_UNITS), np.zeros(N_UNITS)
    for dates, rain in files[:8]:
        inside = (dates >= f"{dates.year[0]}-05-01") & (dates <= f"{dates.year[0]}-09-30")
        totals += np.nansum(rain[inside], axis=0)
        days += np.isfinite(rain[inside]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = 5 * totals / days
    np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)


def test_hcwf_defaults_cap_missing_days_and_skip_partial_years():
    files = series(5, 0.2)
    rain = unit_rain(files)
    default = compute_threshold(SPEC, rain, (YEARS[0], YEARS[-1]))
    # more than 10 % of the window days missing: NaN by default, a value without the cap
    assert np.isnan(default).sum() > np.isnan(compute_threshold(SPEC, rain, (YEARS[0], YEARS[-1]), **BLENDING)).sum()
    no_cap = compute_threshold(SPEC, rain, (YEARS[0], YEARS[-1]), max_missing_fraction=1.0)
    assert np.isfinite(no_cap[np.isfinite(default)]).all()
    # partial years only add the days the data has
    skipped = compute_threshold(SPEC, rain, (2001, 2009), max_missing_fraction=1.0)
    inner = compute_threshold(SPEC, rain, (2002, 2008), max_missing_fraction=1.0)
    np.testing.assert_array_equal(skipped, inner)
    assert not np.array_equal(skipped, compute_threshold(SPEC, rain, (2001, 2009), **BLENDING), equal_nan=True)


def test_tenths_rain_is_divided_by_ten_and_float_rain_used_as_it_is():
    files = series(6, 0.0, partial=False)
    exact = compute_threshold(SPEC, unit_rain(files), (2001, 2009), **BLENDING)
    tenths = compute_threshold(SPEC, unit_rain(files, exact=False), (2001, 2009), **BLENDING)
    assert np.nanmax(np.abs(exact - tenths)) < 0.05 * 5 + 1e-9
    assert not np.array_equal(exact, tenths, equal_nan=True)


def test_threshold_cache_keys_on_the_options(tmp_path):
    rain = unit_rain(series(7, 0.2))
    cache = ThresholdCache(tmp_path)
    plain = cache.get_or_compute(SPEC, rain, (2001, 2009), "thr")
    blend = cache.get_or_compute(SPEC, rain, (2001, 2009), "thr", **BLENDING)
    assert not np.array_equal(plain.values, blend.values, equal_nan=True)
    assert len(list((tmp_path / "thresholds").iterdir())) == 2
    again = cache.get_or_compute(SPEC, rain, (2001, 2009), "thr", **BLENDING)
    np.testing.assert_array_equal(again.values, blend.values)
    assert cache.parts(SPEC, rain, (2001, 2009), "thr").keys() <= cache.parts(SPEC, rain, (2001, 2009), "thr", **BLENDING).keys()


# -- the reader and the join -------------------------------------------------


def test_scale_is_clipped_factor_offset():
    values = np.array([1.0, 8.0, np.nan, 30.0])
    out = scale_values(values, ThresholdScale(factor=0.5, offset=1.0, min=2.0, max=10.0))
    np.testing.assert_array_equal(out, [2.0, 5.0, np.nan, 10.0])
    assert scale_values(values, None) is values
    da = xr.DataArray(values, dims="x", attrs={"units": "mm"})
    assert scale_values(da, ThresholdScale(factor=2.0)).attrs == {"units": "mm"}


def test_scaled_computed_thresholds_are_clipped():
    got = compute_threshold(SPEC, unit_rain(series(8)), (2001, 2009), **BLENDING)
    scaled = scale_values(got, ThresholdScale(factor=0.5, max=10))
    assert (scaled[np.isfinite(scaled)] <= 10).all()


def test_csv_tables_by_id_and_by_coordinates(tmp_path):
    (tmp_path / "id.csv").write_text("unit,thr\nb,2\na,1\n")
    (tmp_path / "xy.csv").write_text("lat,lon,onset_thresh\n10.0,70.0,11\n10.0,70.25,12\n")
    (tmp_path / "adm.csv").write_text("adm3_name,onset_thresh\nx,3\n")
    by_id = read_threshold_table(tmp_path / "id.csv", id_col="unit", value_col="thr")
    assert by_id.to_dict("list") == {"id": ["b", "a"], "onset_thresh": ["2", "1"]}
    assert list(read_threshold_table(tmp_path / "xy.csv").columns) == ["lat", "lon", "onset_thresh"]
    assert read_threshold_table(tmp_path / "adm.csv")["id"].tolist() == ["x"]
    with pytest.raises(ValueError, match="id_col"):
        read_threshold_table(tmp_path / "id.csv", id_col="nope")
    with pytest.raises(FileNotFoundError):
        read_threshold_table(tmp_path / "missing.csv")
    scaled = read_threshold_table(tmp_path / "xy.csv", scale=ThresholdScale(factor=2.0, max=22.0))
    np.testing.assert_array_equal(scaled["onset_thresh"], [22.0, 22.0])


def test_netcdf_and_mat_tables(tmp_path):
    lat, lon = np.array([10.0, 10.25]), np.array([70.0, 70.25])
    pairs = [(a, b) for a in lat for b in lon]
    xr.Dataset({"lat": ("u", [p[0] for p in pairs]), "lon": ("u", [p[1] for p in pairs]),
                "MWmean": ("u", [1.0, 2.0, 3.0, 4.0])}).to_netcdf(tmp_path / "t.nc")
    scipy.io.savemat(tmp_path / "t.mat", {"lat": lat, "lon": lon, "onset.day.thres": np.array([1.0, 2.0, 3.0, 4.0])})
    nc = read_threshold_table(tmp_path / "t.nc")
    mat = read_threshold_table(tmp_path / "t.mat")
    assert nc["onset_thresh"].tolist() == [1.0, 2.0, 3.0, 4.0]
    pd.testing.assert_frame_equal(nc, mat)


def test_grid_reader_is_hcwfs_netcdf_field(tmp_path):
    field = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("lat", "lon"),
                         coords={"lat": [1.0, 2.0], "lon": [5.0, 6.0, 7.0]}, name="thr")
    field.to_dataset().to_netcdf(tmp_path / "g.nc")
    with xr.open_dataset(tmp_path / "g.nc") as ds:
        expected = ds["thr"].sel(lat=[2.0]).load()
    got = read_threshold_grid(tmp_path / "g.nc", "thr", select=lambda da: da.sel(lat=[2.0]))
    xr.testing.assert_identical(got, expected)
    scaled = read_threshold_grid(tmp_path / "g.nc", "thr", scale=ThresholdScale(offset=1.0))
    np.testing.assert_array_equal(scaled.values, field.values + 1.0)


UNITS = np.array(["10.00_70.00", "10.00_70.25", "10.25_70.00", "10.25_70.25"])
CONVENTION = spatial_ids.GridIdConvention(2, "fixed", "spec")
SPEC_IDS = {"geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}


@pytest.mark.parametrize("thresholds", [
    20.0,
    pd.DataFrame({"lat": ["10.0", "10.0", "10.25", "10.25"], "lon": ["70.0", "70.25", "70.0", "70.25"],
                  "onset_thresh": ["11", "12", "13", "14"]}),
    pd.DataFrame({"id": list(UNITS[::-1]), "onset_thresh": [1.0, 2.0, 3.0, 4.0]}),
])
def test_unit_thresholds_are_the_legacy_joins(thresholds):
    old = legacy.attach_thresholds_id(pd.DataFrame({"id": UNITS}), thresholds, spec=SPEC_IDS)["onset_thresh"]
    np.testing.assert_array_equal(unit_thresholds(thresholds, UNITS, CONVENTION), old.to_numpy(dtype=float))


def test_a_threshold_table_must_cover_the_units():
    with pytest.raises(ValueError, match="missing 3 of 4"):
        unit_thresholds(pd.DataFrame({"id": ["10.00_70.00"], "onset_thresh": [1.0]}), UNITS, CONVENTION)
    assert np.isnan(unit_thresholds(None, UNITS)).all()
    with pytest.raises(ValueError, match="conflicting"):
        unit_thresholds(pd.DataFrame({"id": ["a", "a"], "onset_thresh": [1.0, 2.0]}), ["a"])
