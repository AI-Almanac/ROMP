"""Detection and config writing on global files: samples where the data are, init paths, threshold years that are missing.

A global product's grid centre is often ocean (all missing); detection must find rain elsewhere, take the region's box
when it is known, and treat an all-missing sample as no evidence rather than as a wrong unit.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from hcwf.cache import reset_run_state
from hcwf.config import detect
from hcwf.config.init import _anchored, write_init
from hcwf.config.model import NO_SELECTION
from hcwf.io.thresholds import check_threshold_years
from test_decoded_once import obs_spec

LAT = np.arange(-60.0, 60.5, 1.0)
LON = np.arange(-180.0, 180.0, 1.0)
ROLES = {"time": "time", "lat": "lat", "lon": "lon"}


@pytest.fixture(autouse=True)
def fresh():
    """Start without cached samples and detections."""
    reset_run_state()
    yield
    reset_run_state()


def _global(folder, ocean_centre=True, all_missing=False, chunks=None, units="mm/day"):
    """Write a global observation file whose centre (the ocean of the grid) holds no values."""
    rng = np.random.default_rng(0)
    rain = rng.gamma(0.5, 6.0, (6, LAT.size, LON.size)).astype("float32")
    if ocean_centre:
        rain[:, :, np.abs(LON) < 60] = np.nan      # a band of 120 degrees: wider than the centred block of 96 cells
    if all_missing:
        rain[:] = np.nan
    ds = xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": units})},
                    coords={"time": pd.date_range("2020-01-01", periods=6), "lat": LAT, "lon": LON})
    folder.mkdir(parents=True, exist_ok=True)
    encoding = {"precip": {"chunksizes": chunks, "zlib": True}} if chunks else {}
    ds.to_netcdf(folder / "2020.nc", encoding=encoding)
    return folder


@pytest.mark.parametrize("chunks", [None, (3, 20, 40)])
def test_a_global_grid_whose_centre_is_missing_is_sampled_where_there_are_values(tmp_path, chunks):
    """The centre block is all NaN; the sample moves to a block with values, so the units are checked, not refused."""
    folder = _global(tmp_path / "obs", chunks=chunks)
    with xr.open_dataset(folder / "2020.nc") as ds:
        da = ds["precip"]
        sample = detect.load_sample(da, ROLES)
    assert np.isfinite(sample.values).any()
    res = detect.resolve_entry({"path": str(folder), "units": "mm", "day_boundary": "00 UTC, labelled by start"},
                               "observations", use_cache=False)
    assert "units" not in res.problems and res.values["units"] == "mm"
    assert not [p for p in res.problems.items() if "missing" in p[1]]


def test_the_sample_is_taken_around_the_regions_box_when_the_region_is_known(tmp_path):
    """With a region the sample block is around the centre of its box (Ethiopia: about 9N, 40E)."""
    folder = _global(tmp_path / "obs", ocean_centre=False)
    with xr.open_dataset(folder / "2020.nc") as ds:
        sample = detect.load_sample(ds["precip"], ROLES, region="ethiopia")
    assert abs(float(sample["lat"].mean()) - 9.0) < 5 and abs(float(sample["lon"].mean()) - 40.5) < 5


def test_an_all_missing_sample_is_a_warning_when_the_units_are_given_or_declared(tmp_path):
    """No value anywhere: units from the config or the attribute are kept, and the report only warns."""
    folder = _global(tmp_path / "obs", all_missing=True)
    for entry in ({"path": str(folder), "units": "mm"}, {"path": str(folder)}):
        res = detect.resolve_entry({**entry, "day_boundary": "00 UTC, labelled by start"}, "observations", use_cache=False)
        assert "units" not in res.problems
        assert res.values["units"] == "mm"
        assert any("all missing" in w and "units are taken as mm" in w for w in res.warnings)


def test_units_that_contradict_the_attribute_are_still_an_error_when_the_sample_is_all_missing(tmp_path):
    """Missing data excuse the magnitude check only: the given units must still match the declared ones."""
    folder = _global(tmp_path / "obs", all_missing=True, units="m")
    res = detect.resolve_entry({"path": str(folder), "units": "mm", "day_boundary": "00 UTC, labelled by start"},
                               "observations", use_cache=False)
    assert "units" in res.problems and "'units' attribute" in res.problems["units"]


# -- init writes paths relative to the config -------------------------------------------------------------------------

def test_init_writes_folder_paths_relative_to_the_config_file(tmp_path, monkeypatch):
    """A folder given relative to the working directory is written relative to where the config is written."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "obs").mkdir(parents=True)
    (tmp_path / "data" / "model").mkdir(parents=True)
    (tmp_path / "runs" / "one").mkdir(parents=True)
    target = write_init(tmp_path / "runs" / "one" / "run.yaml", obs="data/obs", models=["data/model"])
    config = yaml.safe_load(target.read_text())
    assert config["observations"]["path"] == "../../data/obs"
    assert config["models"][0]["path"] == "../../data/model"
    same = write_init(tmp_path / "run.yaml", obs="data/obs", models=["data/model"])
    assert yaml.safe_load(same.read_text())["observations"]["path"] == "data/obs"


def test_init_leaves_absolute_home_and_environment_paths_as_written(tmp_path):
    """Only a path relative to the working directory is rewritten."""
    assert _anchored("/abs/obs", tmp_path) == "/abs/obs"
    assert _anchored("~/obs", tmp_path) == "~/obs"
    assert _anchored("${DATA}/obs", tmp_path) == "${DATA}/obs"
    assert _anchored("obs", None) == "obs"


# -- threshold years without observation files ------------------------------------------------------------------------

def test_missing_observation_files_of_the_threshold_years_fail_with_a_fix(tmp_path):
    """The thresholds read every threshold year: files that do not exist stop the run early, naming the key to set."""
    folder = _global(tmp_path / "obs")
    (folder / "2020.nc").rename(folder / "2019.nc")
    spec = obs_spec(folder)
    check_threshold_years(spec, NO_SELECTION, (2019, 2019))
    with pytest.raises(FileNotFoundError) as error:
        check_threshold_years(spec, NO_SELECTION, (1998, 2019))
    text = str(error.value)
    assert "threshold years 1998-2019" in text and "no observation file for 1998, 1999" in text
    assert "Fix: set onset.threshold_years" in text
