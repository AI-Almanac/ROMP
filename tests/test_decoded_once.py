"""Every stored chunk of a compressed NetCDF file is decoded at most once by the readers of a run.

The files are small synthetic ones in the shape of the ai-almanac archive (zlib chunks over every dim, latitude flipped,
a region box that starts inside a chunk); ``_decode_count.count_decodes`` counts the chunks each read touches.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from _decode_count import count_decodes
from hcwf.cache import reset_run_state
from hcwf.config import detect
from hcwf.config.model import (AreaSpec, EvaluationSpec, ModelSpec, ObsSpec, OnsetOptions, Selection,
                               Window)
from hcwf.config.onset import onset_setup
from hcwf.io import forecasts as forecasts_io
from hcwf.io import observations as observations_io
from hcwf.onset import sample_alignment
from hcwf.spatial.units import UnitsSpec

LAT = np.arange(17.0, 2.0, -1.0)   # latitude runs south-ward in the file: every read goes against the stored order
LON = np.arange(32.0, 50.0, 1.0)
OBS_CHUNKS = (30, 4, 5)
MODEL_CHUNKS = (5, 2, 4, 4, 5)   # time, number, day, lat, lon
BOUNDARY = "00 UTC, labelled by start"
BOX = AreaSpec(region="Ethiopia", mask=None, nc_mask=None, shpfile_dir=None, polygon=False)
NO_MASKS = Selection(land_only=False, shp_only=False)


@pytest.fixture(autouse=True)
def fresh():
    """Hold no observation years before or after a test."""
    reset_run_state()
    yield
    reset_run_state()


def _gamma(seed, shape):
    """Return rain-like values (float32)."""
    return np.random.default_rng(seed).gamma(0.5, 4.0, shape).astype("float32")


def write_obs(folder, year, chunks=OBS_CHUNKS):
    """Write one year of daily observations ``rain(time, lat, lon)`` as chunked, compressed NetCDF."""
    folder.mkdir(exist_ok=True)
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), _gamma(year, (len(time), len(LAT), len(LON))))},
                    coords={"time": time, "lat": LAT, "lon": LON})
    ds.to_netcdf(folder / f"{year}.nc", encoding={"rain": {"chunksizes": chunks, "zlib": True}})
    return folder / f"{year}.nc"


def write_model(path, inits, steps=12, members=3, chunks=MODEL_CHUNKS, flip=True):
    """Write forecasts ``tp(time, number, day, lat, lon)`` of the inits as chunked, compressed NetCDF."""
    shape = (len(inits), members, steps, len(LAT), len(LON))
    lat = LAT if flip else LAT[::-1]
    ds = xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), _gamma(5, shape) * 0.001, {"units": "m"})},
                    coords={"time": pd.to_datetime(inits), "number": np.arange(members),
                            "day": np.arange(1, steps + 1), "lat": lat, "lon": LON})
    path.parent.mkdir(exist_ok=True)
    ds.to_netcdf(path, encoding={"tp": {"chunksizes": chunks, "zlib": True}})
    return path


def obs_spec(folder, **values) -> ObsSpec:
    """Return the ObsSpec of the folder's ``{year}.nc`` files."""
    fields = dict(name="obs", dir=str(folder), file_pattern=("{}.nc",), var="rain", unit_cvt=None, paths=None,
                  dims=None, day_boundary=None, units=None)
    return ObsSpec(**{**fields, **values})


def model_spec(folder, **values) -> ModelSpec:
    """Return the ModelSpec of the folder's ``{year}.nc`` files."""
    fields = dict(name="M", dir=str(folder), var="tp", unit_cvt=None, file_pattern="{}.nc", accumulation=None,
                  accumulation_detected=None, daily_boundary=None, dims=(("member", "number"), ("step", "day")),
                  paths=None, units=None)
    return ModelSpec(**{**fields, **values})


def options_spec() -> OnsetOptions:
    """Return OnsetOptions with every option unset."""
    return OnsetOptions(**{f.name: None for f in dataclasses.fields(OnsetOptions)} | {"percentage_threshold": 0.5})


def evaluation_spec() -> EvaluationSpec:
    """Return an EvaluationSpec of the 2020 season, ensemble members 0 and 2."""
    return EvaluationSpec(start_date=(2020, 5, 1), end_date=(2020, 7, 31), years=(2020,), climatology_years=(2020,),
                          init_days=(0, 3), date_filter_year=2024, windows=(Window(1, 10, 0),), max_forecast_day=10,
                          day_bins=((1, 10),), probabilistic=True, members=(0, 2), lead_window=None)


def only(found, name):
    """Return the counts of one variable, after checking that something of it was read."""
    counts = found.only(name)
    assert counts.chunks > 0
    return counts


# ------------------------------------------------------------------------------------------------ the helper itself

def test_the_counter_sees_every_touch_of_a_chunk(tmp_path):
    """A chunk read by two calls counts twice; a read of a chunked variable counts the chunks it touches."""
    path = write_obs(tmp_path, 2020)
    with xr.open_dataset(path) as ds:
        da = ds["rain"]
        with count_decodes() as found:
            da.isel(time=slice(0, 30), lat=slice(0, 4), lon=slice(0, 5)).load()   # exactly one chunk
            da.isel(time=slice(0, 30), lat=slice(0, 4), lon=slice(0, 5)).load()
            da.isel(time=slice(25, 35), lat=slice(0, 4), lon=slice(0, 5)).load()   # two time chunks
    counts = found.only("rain")
    assert counts.chunks == 2 and counts.total == 4 and counts.worst() == 3 and counts.ratio == 2.0


# ------------------------------------------------------------------------------------------------- observations

@pytest.mark.parametrize("selection", [NO_MASKS, Selection(shp_only=False)], ids=["box", "land mask"])
def test_an_observation_year_decodes_each_chunk_once(tmp_path, selection):
    """The region box (which starts inside a chunk, on a flipped latitude), with and without the cell mask."""
    write_obs(tmp_path, 2020)
    with count_decodes() as found:
        rain = observations_io.load_observations(2020, obs_spec(tmp_path), BOX, options_spec(), selection,
                                                 run_years=(2020,))
    assert rain.sizes["time"] == 366
    assert only(found, "rain").worst() == 1


def test_observation_years_read_together_decode_each_chunk_once(tmp_path):
    """Two years read through the run's iterator (the year before each is held too) decode every chunk once."""
    for year in (2019, 2020):
        write_obs(tmp_path, year)
    with count_decodes() as found:
        years = [rain for rain, error in observations_io.read_obs_years(
            [2019, 2020], obs_spec(tmp_path), BOX, options_spec(), NO_MASKS, run_years=(2020,))]
    assert len(years) == 2
    assert only(found, "rain").worst() == 1


def test_the_threshold_years_decode_each_chunk_once_however_often_thresholds_are_asked_for(tmp_path):
    """The first threshold year is read once for the probe and the stream together, and not again for another key."""
    from types import SimpleNamespace

    from hcwf.io import thresholds as thresholds_io

    for year in (2018, 2019):
        write_obs(tmp_path / "obs", year)
    onset = onset_setup({"onset_definition": "moron_robertson", "obs_day_boundary": BOUNDARY,
                         "model_lead_label": "start", "max_forecast_day": 10, "threshold_years": (2018, 2019)})
    ev = evaluation_spec()
    output = SimpleNamespace(data_dir=str(tmp_path / "out"), figure_dir=None)
    with count_decodes() as found:
        first = thresholds_io.load_thresholds(onset, obs_spec(tmp_path / "obs"), BOX, options_spec(), ev, output,
                                              select=NO_MASKS, cache_dir=tmp_path / "cache")
        again = thresholds_io.load_thresholds(onset, obs_spec(tmp_path / "obs"), BOX, options_spec(),
                                              dataclasses.replace(ev, max_forecast_day=9), output, select=NO_MASKS,
                                              cache_dir=tmp_path / "cache")
    assert first is not again
    assert only(found, "rain").worst() == 1


# --------------------------------------------------------------------------------------------------- forecasts

def test_a_forecast_year_decodes_each_chunk_once(tmp_path):
    """Chosen inits, the first leads, two of three members and the region box of a flipped, chunked file."""
    write_model(tmp_path / "2020.nc", pd.date_range("2020-05-01", "2020-07-31"))
    onset = onset_setup({"onset_definition": "icpac", "obs_day_boundary": BOUNDARY, "model_lead_label": "start",
                         "max_forecast_day": 10})
    with count_decodes() as found:
        da = forecasts_io.load_forecast(2020, model_spec(tmp_path), evaluation_spec(), BOX, options_spec(), onset,
                                        ensemble=True, select=NO_MASKS)
    assert da.sizes["member"] == 2
    assert only(found, "tp").worst() == 1


def test_a_unit_forecast_decodes_each_chunk_once(tmp_path):
    """The live forecast's reader (`UnitForecast`: all runs of a file, every issue) decodes each chunk once."""
    path = write_model(tmp_path / "live.nc", pd.date_range("2020-06-01", periods=23))
    with count_decodes() as found:
        with forecasts_io.UnitForecast(path, model_spec(tmp_path), UnitsSpec(mode="cells"),
                                       select=Selection(first_lead=1, last_lead=10)) as file:
            runs = [rain for _, _, rain in file.runs()]
    assert sum(len(rain["init_time"]) for rain in runs) == 23
    assert only(found, "tp").worst() == 1


def test_a_live_forecast_issue_decodes_each_chunk_once(tmp_path):
    """One issue of a file of several (the live-forecast case) reads its chunks once, however the runs are cut."""
    path = write_model(tmp_path / "live.nc", pd.date_range("2020-06-01", periods=23))
    with count_decodes() as found:
        with forecasts_io.UnitForecast(path, model_spec(tmp_path), UnitsSpec(mode="cells"),
                                       issue_dates=[pd.Timestamp("2020-06-12").date()],
                                       select=Selection(first_lead=1, last_lead=10)) as file:
            runs = [rain for _, _, rain in file.runs()]
    assert sum(len(rain["init_time"]) for rain in runs) == 1
    assert only(found, "tp").worst() == 1


def test_the_detection_sample_decodes_each_chunk_once(tmp_path):
    """The sample of a forecast file (the live-forecast check reads it) decodes each chunk once."""
    path = write_model(tmp_path / "2020.nc", pd.date_range("2020-05-01", "2020-07-31"))
    roles = {"init_time": "time", "step": "day", "member": "number", "lat": "lat", "lon": "lon"}
    with xr.open_dataset(path) as ds:
        with count_decodes() as found:
            detect.load_sample(ds["tp"], roles)
    assert only(found, "tp").worst() == 1


# ----------------------------------------------------------------------------------------------- the alignment check

def test_the_alignment_sample_decodes_each_chunk_once(tmp_path):
    """The sampled alignment check (observations and forecasts opened lazily, bounding-box reads) decodes each chunk once."""
    write_obs(tmp_path / "obs", 2020)
    write_model(tmp_path / "model" / "2020.nc", pd.date_range("2020-02-01", "2020-10-31", freq="2D"), steps=6)
    kw = dict(obs_dir=str(tmp_path / "obs"), obs_file_pattern=["{}.nc"], obs_var="rain", obs_unit_cvt=None,
              model_dir=str(tmp_path / "model"), model_var="tp", unit_cvt=1000.0, file_pattern="{}.nc",
              model_dims={"member": "number", "step": "day"}, probabilistic=True, region=None,
              start_date=(2020, 2, 1), end_date=(2020, 10, 31), init_days=None, years=(2020,))
    with count_decodes() as found:
        sample = sample_alignment.read_sample(kw)
    assert sample.n_inits > 0
    assert only(found, "rain").worst() == 1
    assert only(found, "tp").worst() == 1


# ----------------------------------------------------------------------------------------------------------- warm-up

def test_the_warm_up_reads_no_data(tmp_path):
    """Looking at a grid for its masks (what the warm-up does with the files it opens) decodes nothing."""
    from hcwf.io.warmup import _warm_masks

    path = write_obs(tmp_path, 2020)
    with xr.open_dataset(path) as lazy:
        with count_decodes() as found:
            _warm_masks(lazy, "rain", "obs", BOX, NO_MASKS, None)
    assert found.only("rain").total == 0

