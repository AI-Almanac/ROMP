"""Files are listed, found and opened by one implementation (hcwf.io.stores, hcwf.config.detect.open_data)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.cache import reset_run_state
from hcwf.config.detect import open_data
from hcwf.io import stores
from hcwf.runs import check


@pytest.fixture(autouse=True)
def fresh():
    """Hold no open store before or after a test."""
    reset_run_state()
    yield
    reset_run_state()


def _touch(folder, *names):
    """Make empty files with the names."""
    for name in names:
        (folder / name).write_text("")


def test_a_pattern_lists_the_same_files_however_it_is_read(tmp_path):
    """``{}`` is the year: ``2010_old.nc`` is not a file of ``{}.nc`` for the years, the sources or the one year."""
    _touch(tmp_path, "2010.nc", "2010_old.nc", "2011.nc", "readme.txt")
    assert stores.years_in_folder(tmp_path, "{}.nc") == (2010, 2011)
    assert [p.rsplit("/", 1)[1] for p in stores.model_sources(tmp_path, None, "{}.nc")] == ["2010.nc", "2011.nc"]
    assert stores.forecast_sources(2010, tmp_path, "{}.nc") == [str(tmp_path / "2010.nc")]
    assert stores.dataset_years(tmp_path, None, None, "{}.nc") == (2010, 2011)


def test_pattern_files_keep_the_text_around_the_year(tmp_path):
    """The text around ``{}`` is as written; the year is four digits."""
    _touch(tmp_path, "m_2010.nc", "m_20101.nc", "x_2011.nc", "m_2012.nc")
    assert [(y, p.rsplit("/", 1)[1]) for y, p in stores.pattern_files(tmp_path, "m_{}.nc")] == [
        (2010, "m_2010.nc"), (2012, "m_2012.nc")]


def test_data_files_are_netcdf_files_and_zarr_stores(tmp_path):
    """One predicate tells what a data file is."""
    _touch(tmp_path, "a.nc", "b.txt")
    (tmp_path / "c.zarr").mkdir()
    assert [p.name for p in stores.data_files(tmp_path)] == ["a.nc", "c.zarr"]
    assert stores.is_data_file("x.nc4") and not stores.is_data_file("x.csv")


def _zarr(path, **coords):
    """Write a small zarr store with a variable over the dims of the coordinates."""
    ds = xr.Dataset({"v": (tuple(coords), np.zeros(tuple(len(c) for c in coords.values())))}, coords=coords)
    ds.to_zarr(path)
    return str(path)


def test_open_data_shares_the_store_and_closing_it_leaves_the_store_open(tmp_path):
    """A zarr store opens once for the run; closing what ``open_data`` returns does not close the shared store."""
    path = _zarr(tmp_path / "s.zarr", time=pd.date_range("2001-01-01", periods=3), lat=[0.0])
    with open_data(path) as first:
        assert first.sizes["time"] == 3
    with open_data(path) as second:
        assert float(second["v"].isel(time=0, lat=0)) == 0.0
    assert stores.open_store(path) is stores.open_store(path)


def test_store_years_of_observations_and_forecasts(tmp_path):
    """A forecast store has its years in ``init_time``; a store without any time coordinate is an error."""
    obs = _zarr(tmp_path / "o.zarr", time=pd.to_datetime(["2001-05-01", "2002-05-01"]), lat=[0.0], lon=[0.0])
    fc = _zarr(tmp_path / "f.zarr", init_time=pd.to_datetime(["2003-05-01", "2004-05-01"]), step=[1, 2], lat=[0.0],
               lon=[0.0])
    bare = _zarr(tmp_path / "b.zarr", lat=[0.0], lon=[0.0, 1.0])
    assert stores.store_years(obs) == (2001, 2002)
    assert stores.store_years(fc, forecast=True) == (2003, 2004)
    with pytest.raises(ValueError, match="no time coordinate"):
        stores.store_years(bare)


def test_check_finds_files_as_a_run_does(tmp_path):
    """The check's file finder is the run's: patterns of a folder, the years of a store, an error for a store with no time."""
    _touch(tmp_path, "2010.nc", "2010_old.nc")
    assert check._find(tmp_path, ("{}.nc",), 2010) == tmp_path / "2010.nc"
    assert check._find(tmp_path, ("{}.nc",), 2011) is None
    assert check._find(tmp_path, ("{}.nc",), 2010, forecast=True) == tmp_path / "2010.nc"
    obs = _zarr(tmp_path / "o.zarr", time=pd.to_datetime(["2001-05-01"]), lat=[0.0], lon=[0.0])
    assert check._find(obs, ("{}.nc",), 2001) is not None and check._find(obs, ("{}.nc",), 2005) is None
    bare = _zarr(tmp_path / "b.zarr", lat=[0.0], lon=[0.0, 1.0])
    with pytest.raises(ValueError, match="no time coordinate"):
        check._find(bare, ("{}.nc",), 2001)


@pytest.mark.parametrize("lead_dim", ["step", "lead_days", "lead_day"])
def test_check_reads_the_leads_of_every_alias(tmp_path, lead_dim):
    """A lead axis named by any alias of the step role is read by the lead check, not skipped."""
    ds = xr.Dataset({"tp": (("init_time", lead_dim), np.zeros((1, 3)))},
                    coords={"init_time": pd.to_datetime(["2001-05-01"]), lead_dim: np.array([1.0, 2.0, 3.0])})
    path = tmp_path / "2001.nc"
    ds.to_netcdf(path)
    dic = {"onset_lead_lines": [], "verification_window_list": ((1, 3),), "tolerance_days_list": (1,),
           "max_forecast_day": 3, "day_bins": ((1, 3),), "model_list": ("M",)}
    report = check.CheckReport()
    check._leads(dic, report, {"M": path})
    assert "3 leads" in report.text()


def test_one_spatial_rename_for_datasets_and_arrays():
    """Datasets and arrays are renamed by one function: aliases, an optional fuzzy match, an error when required."""
    from hcwf.io.dims import dim_fmt, rename_spatial

    da = xr.DataArray(np.zeros((2, 3)), dims=("latitude", "lon"), coords={"latitude": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]})
    assert rename_spatial(da).dims == ("lat", "lon")
    odd = xr.Dataset({"v": (("t", "gridlat", "gridlon"), np.zeros((1, 2, 2)))},
                     coords={"t": [0], "gridlat": [0.0, 1.0], "gridlon": [0.0, 1.0]})
    assert set(rename_spatial(odd).dims) == {"t", "gridlat", "gridlon"}
    assert set(rename_spatial(odd, fuzzy=True).dims) == {"t", "lat", "lon"}
    with pytest.raises(IndexError):
        rename_spatial(odd, required=True)
    mask_style = xr.Dataset({"v": (("time", "nav_lat", "nav_lon"), np.zeros((1, 2, 2)))},
                            coords={"time": [0], "nav_lat": [0.0, 1.0], "nav_lon": [0.0, 1.0]})
    assert {"lat", "lon"} <= set(dim_fmt(mask_style).dims)
