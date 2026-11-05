"""A run whose sources are on different grids: the finer model is averaged onto the coarser grid, scored as the same data.

The synthetic model DETF holds the rain of DET on cells half as wide (each DET cell is four DETF cells of the same
values), so averaging it conservatively onto the grid of DET must give DET's forecasts, onsets and scores back.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hcwf
from hcwf.cache import reset_run_state
from hcwf.config.build import resolve
from hcwf.runs import check as check_mod
from test_run_experience import (CLIM_YEARS, LAT, LON, STEPS, _obs_rain, raw_config, write_config,
                                 write_forecast, write_obs)

FINE_LAT = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
FINE_LON = [37.5, 38.5, 39.5, 40.5, 41.5, 42.5]


def write_fine_forecast(folder, years=(2019, 2020)):
    """Write DET's forecasts on a grid of half the cell size: every DET cell repeated over its four fine cells."""
    folder.mkdir(parents=True)
    for year in years:
        time, rain = _obs_rain(year)
        inits = pd.date_range(f"{year}-05-01", f"{year}-05-31")
        noise = np.random.default_rng(year + 7)
        data = np.stack([rain[i:i + STEPS] for i in range(len(inits))])
        data = data * noise.uniform(0.5, 1.6, size=data.shape)
        data = np.repeat(np.repeat(data, 2, axis=2), 2, axis=3)
        xr.Dataset({"tp": (("time", "step", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": inits, "step": np.arange(1, STEPS + 1), "lat": FINE_LAT, "lon": FINE_LON}
                   ).to_netcdf(folder / f"{year}.nc")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Observations on the 2 degree grid, DET on it, and DETF on the 1 degree grid holding the same rain."""
    base = tmp_path_factory.mktemp("evalgrid_world")
    write_obs(base / "obs")
    write_forecast(base / "DET")
    write_fine_forecast(base / "DETF")
    assert len(list((base / "obs").glob("*.nc"))) == len(CLIM_YEARS)
    return base


@pytest.fixture(autouse=True)
def fresh():
    """Start every test without the state of a previous run."""
    reset_run_state()
    yield
    reset_run_state()


def test_the_coarser_models_grid_is_chosen_and_the_finer_model_is_averaged_onto_it(world, tmp_path):
    """DETF (finer) is averaged onto the grid of the observations and DET; its scores equal DET's."""
    path = write_config(tmp_path, raw_config(world, ("DET", "DETF")))
    rc = resolve(path, write=False)
    grid = rc.area.evaluation_grid
    assert grid is not None and grid.source == "observations" and grid.regrid == ("DETF",)
    np.testing.assert_allclose(grid.lat, LAT)
    np.testing.assert_allclose(grid.lon, LON)
    result = hcwf.run(path, quiet=True)
    spatial = result.scores["spatial"]
    det = spatial[next(k for k in spatial if k[0] == "DET")]
    fine = spatial[next(k for k in spatial if k[0] == "DETF")]
    assert det.sizes["lat"] == 3 and det.sizes["lon"] == 3 and fine.sizes == det.sizes
    assert det["false_alarm_rate"].notnull().sum() > 0 and fine["false_alarm_rate"].notnull().sum() > 0
    for name in det.data_vars:
        np.testing.assert_allclose(fine[name].values, det[name].values, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_the_override_scores_on_the_finer_grid_only_when_it_is_the_observations_or_a_model_of_the_run(world, tmp_path):
    """evaluation_grid names a source; an unknown name is a config problem that lists the sources."""
    from hcwf.errors import HCWFConfigError

    raw = raw_config(world, ("DET", "DETF"))
    raw["evaluation_grid"] = "DETF"
    rc = resolve(write_config(tmp_path, raw), write=False)
    grid = rc.area.evaluation_grid
    assert grid.source == "DETF" and set(grid.regrid) == {"observations", "DET"} and len(grid.lat) == 6
    raw["evaluation_grid"] = "NOPE"
    with pytest.raises(HCWFConfigError, match="evaluation_grid: 'NOPE' is not a source of the run"):
        resolve(write_config(tmp_path, raw, "bad.yaml"), write=False)


def test_check_reports_the_evaluation_grid_and_why(world, tmp_path):
    """--check names the grid, the reason and what is averaged onto it, and no longer says the data are never regridded."""
    rc, found = resolve(write_config(tmp_path, raw_config(world, ("DET", "DETF"))), check=True, write=False)
    text = check_mod.build_report(rc, found).text()
    assert "evaluation grid: the grid of observations" in text and "coarsest of the sources" in text
    assert "DETF: " in text and "averaged conservatively onto it" in text
    assert "never regridded" not in text and "cdo remapcon" not in text


def test_sources_on_one_grid_have_no_evaluation_grid(world, tmp_path):
    """Matching grids are read as before: nothing is attached and the check says so."""
    rc, found = resolve(write_config(tmp_path, raw_config(world, ("DET",))), check=True, write=False)
    assert rc.area.evaluation_grid is None
    assert "evaluation grid: the sources are on one grid inside the region box; nothing is regridded" in \
        check_mod.build_report(rc, found).text()
