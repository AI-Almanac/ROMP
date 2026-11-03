"""Datasets as a list of files, the one table of dimension names, units, and unit dimensions (the input parts of
onset_blending's test_io.py, test_pipeline.py and test_cell_transforms.py, against HCWF's single reader)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import typed_io
import xarray as xr

from _inputs import write_adm3, write_forecast, write_observations
from hcwf.config import detect
from hcwf.config.datasets import file_year, parse_dataset, parse_years, resolve_paths
from hcwf.onset import ingest as ing
from hcwf.io.dims import ROLE_ALIASES, dim_fmt, dim_fmt_model, dim_fmt_model_ensemble, find_dim
from hcwf.errors import DatasetError

BOUNDARY = "00 UTC, labelled by start"


# ------------------------------------------------------------------------------------------------ file lists

def test_the_year_of_a_file_is_the_first_19xx_or_20xx_in_its_name():
    assert file_year("/x/20200101.nc") == 2020 and file_year("/2012/ens_1999_v2013.nc") == 1999
    assert file_year("latest.nc") is None and file_year("era5_025deg.nc") is None
    # HCWF's own pattern detection needs a non-digit on both sides, so it never saw a year in 20200101.nc
    assert detect.find_pattern(["20200101.nc"])[0] is None


def test_paths_are_listed_by_year_then_name_and_filtered_by_years(tmp_path):
    for name in ("b_2004.nc", "a_2004.nc", "2003.nc", "x.nc"):
        (tmp_path / name).write_bytes(b"")
    listed = resolve_paths(["*.nc"], tmp_path)
    assert [f.name for f in listed] == ["x.nc", "2003.nc", "a_2004.nc", "b_2004.nc"]
    assert [f.name for f in resolve_paths([str(tmp_path / "2003.nc"), "a_2004.nc", "*2004*"], tmp_path)] == \
        ["2003.nc", "a_2004.nc", "b_2004.nc"]
    assert [f.name for f in resolve_paths("*.nc", tmp_path, years="2004:2004")] == ["a_2004.nc", "b_2004.nc"]
    assert [f.name for f in resolve_paths(["*.nc"], tmp_path, years=[2003])] == ["2003.nc"]
    with pytest.raises(DatasetError, match="no file for"):
        resolve_paths(["missing.nc"], tmp_path)
    with pytest.raises(DatasetError, match="no file matches"):
        resolve_paths(["*.nc"], tmp_path, years=[1990])
    assert parse_years("2010:2012") == (2010, 2011, 2012) and parse_years(2010) == (2010,)
    with pytest.raises(DatasetError):
        parse_years("later")


def test_a_dataset_entry_may_list_its_files():
    entry = {"paths": ["live/*.nc"], "years": "2010:2011", "variable": "tp", "unit_cvt": 1000,
             "lead_label": "start", "dims": {"unit": "adm3_name"}}
    data = parse_dataset(entry, "x", "forecast", require_id=False)
    assert data.paths == ("live/*.nc",) and data.years == (2010, 2011) and data.unit_cvt == 1000.0
    assert data.units is None and data.dims == {"unit": "adm3_name"}
    for bad, match in [({"file_pattern": "{}.nc"}, "both say"), ({"units": "mm"}, "both say how"),
                       ({"unit_cvt": -1}, "positive number"), ({"paths": []}, "list of file names")]:
        with pytest.raises(DatasetError, match=match):
            parse_dataset({**entry, **bad}, "x", "forecast", require_id=False)
    with pytest.raises(DatasetError, match="years needs paths"):
        parse_dataset({"path": "p", "variable": "v", "years": [2000], "day_boundary": BOUNDARY}, "x", "observations",
                      require_id=False)


def test_live_files_with_one_init_each_are_read_as_one_year(tmp_path):
    dates = [dt.date(2014, 6, 2), dt.date(2014, 6, 9), dt.date(2014, 6, 16)]
    for i, day in enumerate(dates):
        write_forecast(tmp_path / f"live_{day:%Y%m%d}.nc", [day], members=2, seed=i)
    write_forecast(tmp_path / "all.nc", dates, members=2, seed=0)
    other = write_forecast(tmp_path / "live_20130602.nc", [dt.date(2013, 6, 2)], members=2)
    kwargs = dict(model_var="tp", unit_cvt=None, region=None, date_filter_year=2014, init_days=list(range(7)),
                  start_date=(0, 5, 1), end_date=(0, 7, 31), members=None, model_paths=["live_*.nc"],
                  model_dir=str(tmp_path), model_files_years=[2014, 2013])
    got = typed_io.forecast(2014, kwargs, True)
    assert got.sizes["init_time"] == 3 and got.sizes["member"] == 2
    assert pd.to_datetime(got["init_time"].values).date.tolist() == dates
    # each file holds one init: the values are the ones in the file, not another file's
    for i, day in enumerate(dates):
        with xr.open_dataset(tmp_path / f"live_{day:%Y%m%d}.nc") as one:
            expect = one["tp"].transpose("time", "number", "day", "lat", "lon").values[0]
        np.testing.assert_array_equal(got.sel(init_time=str(day)).transpose("member", "step", "lat", "lon").values,
                                      expect)
    only = typed_io.forecast(2013, kwargs, True)
    assert only.sizes["init_time"] == 1 and other.exists()
    with pytest.raises(FileNotFoundError, match="among the listed paths"):
        typed_io.forecast(2012, kwargs, True)
    with pytest.raises(FileNotFoundError):
        typed_io.forecast(2014, {**kwargs, "model_files_years": [2013]}, True)


def test_observations_listed_by_path_are_read_for_their_year(tmp_path):
    write_observations(tmp_path, years=(2003, 2004), name=lambda y: f"obs_{y}_v2.nc")
    kwargs = dict(obs_var="precip", obs_unit_cvt=None, region=None, obs_paths=["obs_*.nc"], obs_dir=str(tmp_path))
    da = typed_io.observations(2004, kwargs)
    assert pd.DatetimeIndex(da["time"].values).year.unique().tolist() == [2004]
    with xr.open_dataset(tmp_path / "obs_2004_v2.nc") as ds:
        np.testing.assert_array_equal(da.values, ds["precip"].values)
    # the same year in two files is joined along time
    ds = xr.open_dataset(tmp_path / "obs_2004_v2.nc").load()
    ds.isel(time=slice(0, 10)).to_netcdf(tmp_path / "part_2004_a.nc")
    ds.isel(time=slice(10, 20)).to_netcdf(tmp_path / "part_2004_b.nc")
    both = typed_io.observations(2004, {**kwargs, "obs_paths": ["part_*.nc"]})
    np.testing.assert_array_equal(both.values, da.values)
    with pytest.raises(FileNotFoundError):
        typed_io.observations(1999, kwargs)


def test_detection_reads_the_first_listed_file(tmp_path):
    write_forecast(tmp_path / "20140602.nc", [dt.date(2014, 6, 2)], units="m")
    write_forecast(tmp_path / "20140609.nc", [dt.date(2014, 6, 9)], units="m")
    entry = {"path": str(tmp_path), "paths": ["2014*.nc"], "lead_label": "start", "accumulation": "daily",
             "variable": "tp"}
    res = detect.resolve_entry(entry, "forecast", detect_lead_label=False, use_cache=False)
    assert not res.problems, res.problems
    assert res.values["units"] == "m" and res.values["members"] == 3
    assert res.values["dims"] == {"init_time": "time", "step": "day", "member": "number", "lat": "lat", "lon": "lon"}
    assert any("2 file(s) listed in paths" in line for line in res.report())
    assert detect.resolve_entry({**entry, "paths": ["nothing*.nc"]}, "forecast", use_cache=False).problems["paths"]


# ---------------------------------------------------------------------------------- one table of dimension names

def test_the_three_alias_tables_are_one():
    from hcwf.config import detect as dd

    assert not any(hasattr(mod, name) for mod, name in [(ing, "_TIME_DIMS"), (ing, "_LAT_DIMS"), (ing, "_LON_DIMS"),
                                                        (ing, "_INIT_DIMS"), (ing, "_LEAD_DIMS"),
                                                        (ing, "_MEMBER_DIMS"), (dd, "_ROLE_ALIASES")])
    assert set(ROLE_ALIASES) == {"time", "init_time", "step", "member", "lat", "lon", "unit"}


@pytest.mark.parametrize("init, lead, member", [("initial_time", "lead_day", "members"),
                                                ("forecast_reference_time", "lead_days", "ensemble_member"),
                                                ("init_time", "step", "sample"), ("Time", "Day", "Number")])
def test_blendings_aliases_name_the_roles(init, lead, member):
    dims = [init, member, lead, "latitude", "Longitude"]
    assert find_dim(dims, "init_time") == init and find_dim(dims, "step") == lead
    assert find_dim(dims, "member") == member
    assert find_dim(dims, "lat") == "latitude" and find_dim(dims, "lon") == "Longitude"
    assert find_dim(["adm3", "time"], "unit") == "adm3" and find_dim(["ADM3_NAME"], "unit") == "ADM3_NAME"


def test_a_coordinate_in_days_is_the_lead():
    ds = xr.Dataset({"tp": (("t0", "horizon"), np.zeros((2, 3)))},
                    coords={"t0": pd.to_datetime(["2000-01-01", "2000-01-02"]),
                            "horizon": ("horizon", [1, 2, 3], {"units": "days"})})
    assert find_dim(list(ds.dims), "step", coords=ds.coords) == "horizon"
    assert find_dim(list(ds.dims), "step") is None
    out = ing.ingest_forecast(ds["tp"].rename(t0="time"), day_boundary=BOUNDARY, lead_label="start", units="mm",
                              accumulation="daily")
    assert out.dims == ("init_time", "lead") and out.coords["lead"].tolist() == [1, 2, 3]


def test_dim_fmt_renames_every_alias_to_the_canonical_names():
    times = pd.to_datetime(["2000-01-01", "2000-01-02"])
    ds = xr.Dataset({"tp": (("initial_time", "members", "lead_day", "latitude", "longitude"), np.zeros((2, 2, 3, 2, 2)))},
                    coords={"initial_time": times, "members": [0, 1], "lead_day": [1, 2, 3],
                            "latitude": [1.0, 2.0], "longitude": [3.0, 4.0]})
    out = dim_fmt_model_ensemble(ds)
    assert set(out["tp"].dims) == {"init_time", "member", "step", "lat", "lon"}
    assert set(dim_fmt_model(ds.isel(members=0, drop=True))["tp"].dims) == {"init_time", "step", "lat", "lon"}
    assert set(dim_fmt(xr.Dataset({"r": (("date", "y", "x"), np.zeros((1, 2, 2)))},
                                  coords={"date": times[:1], "y": [1.0, 2.0], "x": [3.0, 4.0]}))["r"].dims) \
        == {"time", "lat", "lon"}


def test_a_string_unit_dimension_is_read_as_unit(tmp_path):
    write_adm3(tmp_path / "adm3.nc", names=("north", " south ", "East ward"))
    ds = dim_fmt(xr.open_dataset(tmp_path / "adm3.nc"))
    assert ds["precip"].dims == ("time", "unit") and ds["unit"].values.tolist() == ["north", "south", "East ward"]
    raw = xr.open_dataset(tmp_path / "adm3.nc")["precip"]
    detected = detect._roles(raw, "observations", {})
    assert detected == {"time": "time", "unit": "adm3_name"}


# ------------------------------------------------------------------------------------------------------ units

@pytest.mark.parametrize("text, factor", [("metres", 1000.0), ("Meters", 1000.0), ("m", 1000.0), ("mm", 1.0)])
def test_metres_and_meters_are_units(text, factor):
    assert ing.unit_scale(text) == factor
    assert detect._units_from_attr(text) == ("m" if factor == 1000.0 else "mm")


def _forecast_entry(tmp_path, units):
    write_forecast(tmp_path / "f2014.nc", [dt.date(2014, 6, 2)], units=units, seed=3)
    return {"path": str(tmp_path), "file_pattern": "f{}.nc", "variable": "tp", "lead_label": "start",
            "accumulation": "daily"}


def test_a_numeric_unit_factor_serves_files_that_declare_no_units(tmp_path):
    entry = {**_forecast_entry(tmp_path, None), "unit_cvt": 0.5}
    res = detect.resolve_entry(entry, "forecast", detect_lead_label=False, use_cache=False)
    assert "units" not in res.problems and "units" not in res.values
    values = detect.finalize_units(res.values)
    assert values["unit_cvt"] == 0.5
    detect.check_entry(values, "forecast")
    assert any(line.startswith("unit_cvt: 0.5") for line in res.report())


def test_declared_units_win_over_the_numeric_factor(tmp_path):
    entry = {**_forecast_entry(tmp_path, "m"), "unit_cvt": 0.5}
    res = detect.resolve_entry(entry, "forecast", detect_lead_label=False, use_cache=False)
    values = detect.finalize_units(res.values)
    assert values["units"] == "m" and values["unit_cvt"] == 1000
    assert any("unit_cvt is only for files that declare no units" in line for line in res.report())


def test_a_forecast_without_declared_units_scales_like_onset_blending(tmp_path):
    """x * factor in float32, in place (onset_blending: block * scale)."""
    path = write_forecast(tmp_path / "f2014.nc", [dt.date(2014, 6, 2)], units=None, seed=4)
    kwargs = dict(model_var="tp", region=None, date_filter_year=2014, init_days=list(range(7)), start_date=(0, 5, 1),
                  end_date=(0, 7, 31), members=None, model_dir=str(tmp_path), file_pattern="f{}.nc")
    plain = typed_io.forecast(2014, {**kwargs, "unit_cvt": None}, True)
    scaled = typed_io.forecast(2014, {**kwargs, "unit_cvt": 0.1}, True)
    assert scaled.dtype == np.float32
    np.testing.assert_array_equal(scaled.values, plain.values * np.float32(0.1))
    assert path.exists()
