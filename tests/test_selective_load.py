"""Loaders read only what a run needs (year, region box, leads, members) and mask and scale the array in place.

Each check compares against the straightforward full computation on the same synthetic data, for per-year NetCDF
files and for a zarr store of all years.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.io import input as romp_input
from momp.utils.region import mask_region_loaded, region_select

LAT = np.arange(2.0, 17.0, 1.0)
LON = np.arange(32.0, 50.0, 1.0)
CFG = dict(region="Ethiopia", shp_only=False, date_filter_year=2024, init_days=[0, 3], start_date=(2020, 5, 1),
           end_date=(2020, 6, 30), max_forecast_day=10, wet_spell=3, dry_extent=2, wet_init=1, wet_threshold=20,
           dry_spell=7, model="M", model_var="tp", unit_cvt=1000.0, file_pattern="{}.nc", members=None)


def obs_year(year, seed):
    """A year of daily observations (time, latitude, longitude) of float32 values."""
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    data = np.random.default_rng(seed).gamma(0.5, 4.0, (len(time), len(LAT), len(LON))).astype("float32")
    return xr.Dataset({"rain": (("time", "latitude", "longitude"), data)},
                      coords={"time": time, "latitude": LAT, "longitude": LON})


def forecast_year(year, seed, ensemble):
    """A year of daily forecasts in metres, daily inits May-July, 30 steps (starting at 0) and, if asked, 4 members."""
    init = pd.date_range(f"{year}-05-01", f"{year}-07-31")
    dims = ("time",) + (("number",) if ensemble else ()) + ("prediction_timedelta_daily", "lat", "lon")
    shape = (len(init),) + ((4,) if ensemble else ()) + (31, len(LAT), len(LON))
    data = np.random.default_rng(seed).gamma(0.5, 0.004, shape).astype("float32")
    coords = {"time": init, "prediction_timedelta_daily": pd.to_timedelta(np.arange(31), "D"), "lat": LAT[::-1],
              "lon": LON}
    if ensemble:
        coords["number"] = np.arange(4)
    return xr.Dataset({"tp": (dims, data[..., ::-1, :])}, coords=coords)


@pytest.fixture(autouse=True)
def fresh_cache():
    """Each test starts with no observation years held."""
    romp_input.clear_obs_cache()
    yield
    romp_input.clear_obs_cache()


def test_in_place_land_mask_keeps_the_cells_region_select_keeps():
    """Masking a loaded array in place leaves the same values and NaNs as region_select's lazy where."""
    ds = obs_year(2020, 1).rename(latitude="lat", longitude="lon").isel(time=slice(0, 5))
    expected = region_select(ds, region="Ethiopia", shp_only=False)["rain"]
    box = ds["rain"].sel(lat=slice(3, 15), lon=slice(33, 48)).load()
    got = mask_region_loaded(box, region="Ethiopia", shp_only=False)
    assert np.isnan(expected.values).any() and not np.isnan(expected.values).all()
    np.testing.assert_array_equal(got.values, expected.values)
    assert got is box


def test_obs_year_read_once_and_zarr_equals_netcdf(tmp_path):
    """A zarr store of all years gives the NetCDF year's values; a year is read from disk once per run."""
    years = {2019: obs_year(2019, 2), 2020: obs_year(2020, 3)}
    nc = tmp_path / "nc"
    nc.mkdir()
    for year, ds in years.items():
        ds.to_netcdf(nc / f"{year}.nc")
    store = tmp_path / "obs.zarr"
    xr.concat(list(years.values()), dim="time").to_zarr(store)
    kw = dict(obs_file_pattern=["{}.nc"], obs_var="rain", obs_unit_cvt=2.0, region="Ethiopia", shp_only=False, years=[2020])
    from_nc = romp_input.load_imd_rainfall(2020, obs_dir=str(nc), **kw)
    from_zarr = romp_input.load_imd_rainfall(2020, obs_dir=str(store), **kw)
    xr.testing.assert_identical(from_nc, from_zarr)
    assert romp_input.load_imd_rainfall(2020, obs_dir=str(store), **kw) is from_zarr
    box = years[2020]["rain"].sel(latitude=slice(3, 15), longitude=slice(33, 48))
    np.testing.assert_array_equal(from_nc.values[~np.isnan(from_nc.values)],
                                  (box.values * 2.0)[~np.isnan(from_nc.values)])
    with pytest.raises(FileNotFoundError):
        romp_input.load_imd_rainfall(2018, obs_dir=str(store), **kw)


@pytest.mark.parametrize("ensemble", [False, True])
def test_forecast_selects_before_reading_and_zarr_equals_netcdf(tmp_path, ensemble):
    """Inits, leads, members and the box are cut before reading; values are the old x * unit_cvt on the same cells."""
    full = {y: forecast_year(y, y, ensemble) for y in (2019, 2020)}
    nc = tmp_path / "nc"
    nc.mkdir()
    for year, ds in full.items():
        ds.to_netcdf(nc / f"{year}.nc")
    store = tmp_path / "m.zarr"
    xr.concat(list(full.values()), dim="time").to_zarr(store)
    loader = (romp_input.get_forecast_probabilistic_twice_weekly if ensemble
              else romp_input.get_forecast_deterministic_twice_weekly)
    kw = {**CFG, "members": [1, 3] if ensemble else None, "model_dims": {"member": "number"} if ensemble else None}
    kw = {k: v for k, v in kw.items() if k != "model_dims" or v}
    from_nc = loader(2020, model_dir=str(nc), **kw)
    from_zarr = loader(2020, model_dir=str(store), **kw)
    xr.testing.assert_identical(from_nc, from_zarr)
    steps = romp_input.onset_steps_needed(**kw)
    assert from_nc.sizes["step"] == steps and int(from_nc["step"][0]) == 1
    assert from_nc.sizes["init_time"] == len(pd.to_datetime(from_nc.init_time.values))
    assert list(pd.to_datetime(from_nc.init_time.values)) == list(romp_input.get_initialization_dates(2020, **kw))
    if ensemble:
        assert list(from_nc["member"].values) == [1, 3]
    reference = full[2020]["tp"].sortby("lat").sel(lat=slice(3, 15), lon=slice(33, 48), prediction_timedelta_daily=pd.to_timedelta(
        np.arange(1, steps + 1), "D"))
    reference = reference.sel(time=from_nc.init_time.values)
    if ensemble:
        reference = reference.sel(number=[1, 3])
    old = reference.transpose(*[d for d in ("time", "number", "prediction_timedelta_daily", "lat", "lon")
                                if d in reference.dims]) * 1000.0
    got = from_nc.transpose(*[d for d in from_nc.dims]).values
    expected = old.values
    keep = ~np.isnan(got)
    assert keep.any()
    np.testing.assert_array_equal(got[keep], expected[keep])
    with pytest.raises(FileNotFoundError):
        loader(2018, model_dir=str(store), **kw)
