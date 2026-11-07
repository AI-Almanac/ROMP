"""A blend and the evaluation grid: a blend on cell units trains on the evaluation grid (finer sources are averaged onto
the coarser grid, as benchmarks do), stores that grid, and puts live forecasts on it; listed forecast files have their
grids looked at."""
import datetime as dt
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _recipes import converted
from test_blend_training import (FORECAST_YEARS, LEAD_DAYS, OBS_YEARS, config_dict, onset_date, rain)

from hcwf.blend import TrainedBlend
from hcwf.blend.training import train
from hcwf.errors import IncompatibleForecast
from hcwf.io.grids import attach, source_grids

LATS, LONS = [10.0, 10.25], [77.0, 77.25]
FINE_LATS, FINE_LONS = [9.9375, 10.0625, 10.1875, 10.3125], [76.9375, 77.0625, 77.1875, 77.3125]


def _write_observations(folder):
    folder.mkdir(parents=True)
    for year in OBS_YEARS:
        dates = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(200)]
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def _write_forecasts(folder, fine, members=3):
    """Write the forecasts on the observations' cells, or on cells half their size that repeat each one."""
    folder.mkdir(parents=True)
    lats, lons = (FINE_LATS, FINE_LONS) if fine else (LATS, LONS)
    for year in FORECAST_YEARS:
        issues = [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), members, LEAD_DAYS, len(lats), len(lons)), dtype="float32")
        for t, issue in enumerate(issues):
            leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
            for m in range(members):
                for i in range(len(lats)):
                    for j in range(len(lons)):
                        cell = (i // 2 if fine else i) * 2 + (j // 2 if fine else j)
                        data[t, m, :, i, j] = rain(leads, onset_date(year, cell, shift=3 * (m - 1)))
        xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                           "day": np.arange(1, LEAD_DAYS + 1), "lat": lats, "lon": lons}).to_netcdf(folder / f"{year}.nc")


@pytest.fixture(scope="module")
def finer_model(tmp_path_factory):
    """A blend trained on observations (0.25 degrees) and a model on cells of half the size: saved, and loaded."""
    root = tmp_path_factory.mktemp("finer")
    _write_observations(root / "obs")
    _write_forecasts(root / "ens", fine=True)
    rc = attach(converted(config_dict(root), cache_dir=root / "cache").rc())
    result = train(rc)
    path = result.save(root / "trained") / "trained.blend"
    return root, rc, result, TrainedBlend.load(path), path


def _write_live(folder, lats, lons, year=2013):
    """Write one season of live forecasts on the given grid; every cell has the onsets of the cell it lies in."""
    folder.mkdir(parents=True, exist_ok=True)
    issues = [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
    lat0, lon0 = LATS[0], LONS[0]
    data = np.zeros((len(issues), 3, LEAD_DAYS, len(lats), len(lons)), dtype="float32")
    for t, issue in enumerate(issues):
        leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
        for m in range(3):
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    cell = min(max(int(np.floor((lat - lat0 + 0.125) / 0.25)), 0), 1) * 2 \
                        + min(max(int(np.floor((lon - lon0 + 0.125) / 0.25)), 0), 1)
                    data[t, m, :, i, j] = rain(leads, onset_date(year, cell, shift=3 * (m - 1)))
    xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
               coords={"time": pd.to_datetime(issues), "number": np.arange(3), "day": np.arange(1, LEAD_DAYS + 1),
                       "lat": lats, "lon": lons}).to_netcdf(folder / f"{year}.nc")
    return folder / f"{year}.nc"


def _same_numbers(got, expected):
    """The tables have the same rows and, to 1e-9 relative, the same numbers."""
    assert got["id"].tolist() == expected["id"].tolist()
    for column in expected.columns:
        if expected[column].dtype.kind == "f":
            np.testing.assert_allclose(got[column], expected[column], rtol=1e-9, atol=1e-12, err_msg=column)


def test_a_blend_on_sources_of_different_grids_trains_on_the_evaluation_grid(finer_model):
    """Cell units of sources on different grids are averaged onto the coarser grid, as a benchmark does."""
    _, rc, result, _, _ = finer_model
    grid = rc.area.evaluation_grid
    assert grid.source == "observations" and grid.regrid == ("observations", "ens")   # no region: every source is read from the area
    assert rc.models[0].units.grid is grid and rc.observations.units.grid is grid   # the observations are on it: their own cells
    assert result.inputs.units == ("10.00_77.00", "10.00_77.25", "10.25_77.00", "10.25_77.25")


def test_the_trained_blend_stores_the_grid_it_was_trained_on(finer_model):
    """The grid is in the file: centres, the source it came from, the area convention, the cells used."""
    _, rc, _, blend, path = finer_model
    stored = blend.grid
    assert stored.kind == "cells" and stored.grid.source == "observations" and stored.grid.regrid == ("observations", "ens")
    assert stored.grid.lat.tolist() == LATS and stored.grid.lon.tolist() == LONS
    assert stored.ids == blend.units and stored.kept.tolist() == [0, 1, 2, 3]
    assert blend.frozen_units["ens"].grid is stored.grid
    with zipfile.ZipFile(path) as archive:
        text = json.loads(archive.read("grid.json"))
    assert text["lat_axis"] == {"start": 10.0, "step": 0.25, "count": 2} and text["lon_convention"] == "-180 to 180"
    assert text["cell_area"] == "sphere" and text["steps"][1][0] == "ens" and text["steps"][1][1:] == [0.125, 0.125]


def test_a_finer_live_forecast_is_averaged_onto_the_stored_grid(finer_model, tmp_path):
    """A forecast on cells of half the size gives what the same forecast on the trained cells does."""
    _, _, _, blend, _ = finer_model
    fine = _write_live(tmp_path / "fine", FINE_LATS, FINE_LONS)
    coarse = _write_live(tmp_path / "coarse", LATS, LONS)
    finer = blend.forecast({"ens": fine}, workers=1)
    native = blend.forecast({"ens": coarse}, workers=1)
    assert len(native.weekly_table) > 0
    _same_numbers(finer.weekly_table, native.weekly_table)


@pytest.mark.parametrize("what, lats, lons, why", [
    ("shifted", [10.125, 10.375], [77.125, 77.375], "shifted"),
    ("coarser", [9.75, 10.25, 10.75], [76.75, 77.25, 77.75], "coarser"),
    ("not covering", [9.9375, 10.0625], FINE_LONS, "does not cover"),
])
def test_a_live_forecast_that_cannot_be_put_on_the_stored_grid_is_refused(finer_model, tmp_path, what, lats, lons, why):
    """A shifted, a coarser and a finer grid that misses trained cells are refused, naming both grids."""
    _, _, _, blend, _ = finer_model
    live = _write_live(tmp_path / what.replace(" ", "_"), lats, lons)
    with pytest.raises(IncompatibleForecast) as raised:
        blend.forecast({"ens": live}, workers=1)
    text = str(raised.value)
    assert "Fix:" in text and why in text and "trained on (2 lats 10 to 10.25 x 2 lons 77 to 77.25" in text
    assert "the live file's grid (" in text


@pytest.fixture(scope="module")
def one_grid(tmp_path_factory):
    """A blend trained on sources that share one grid, saved and loaded."""
    root = tmp_path_factory.mktemp("one")
    _write_observations(root / "obs")
    _write_forecasts(root / "ens", fine=False)
    rc = attach(converted(config_dict(root), cache_dir=root / "cache").rc())
    path = train(rc).save(root / "trained") / "trained.blend"
    return root, rc, TrainedBlend.load(path), path


def test_a_blend_on_one_grid_stores_that_grid_and_refuses_a_finer_one_it_was_not_given(one_grid, tmp_path):
    """Sources on one grid have no evaluation grid, yet the blend stores the observations' grid."""
    _, rc, blend, _ = one_grid
    assert rc.area.evaluation_grid is None and blend.grid.grid.regrid == ()
    assert blend.grid.grid.lat.tolist() == LATS and blend.frozen_units["ens"].grid is blend.grid.grid
    coarse = _write_live(tmp_path / "coarse", LATS, LONS)
    assert len(blend.forecast({"ens": coarse}, workers=1).weekly_table) > 0
    fine = _write_live(tmp_path / "fine", FINE_LATS, FINE_LONS)   # finer is averaged onto the stored grid
    _same_numbers(blend.forecast({"ens": fine}, workers=1).weekly_table, blend.forecast({"ens": coarse}, workers=1).weekly_table)


def test_a_blend_saved_before_the_grid_was_stored_still_loads_and_checks_the_files_own_grid(one_grid, tmp_path):
    """Without grid.json the grid is unknown: the live file's own cells are read, and its grid must be the training
    files' (the signature): a finer file is refused as another resolution, which is the safer reading."""
    _, _, new, path = one_grid
    old = tmp_path / "old.blend"
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {n: archive.read(n) for n in manifest["members"] if n != "grid.json"}
    manifest["members"] = {n: hashlib.sha256(d).hexdigest() for n, d in members.items()}
    with zipfile.ZipFile(old, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in members.items():
            archive.writestr(name, data)
    blend = TrainedBlend.load(old)
    assert blend.grid is None and blend.frozen_units["ens"].grid is None
    coarse = _write_live(tmp_path / "coarse", LATS, LONS)
    pd.testing.assert_frame_equal(blend.forecast({"ens": coarse}, workers=1).weekly_table,
                                  new.forecast({"ens": coarse}, workers=1).weekly_table)
    fine = _write_live(tmp_path / "fine", FINE_LATS, FINE_LONS)
    with pytest.raises(IncompatibleForecast, match="ens: 2013.nc: .*another resolution"):
        blend.forecast({"ens": fine}, workers=1)


def test_listed_forecast_files_have_their_grids_looked_at(tmp_path):
    """Sources on one grid have no evaluation grid, and the grids of listed files can be looked at (attach)."""
    _write_observations(tmp_path / "obs")
    _write_forecasts(tmp_path / "ens", fine=False)
    rc = converted(config_dict(tmp_path), cache_dir=tmp_path / "cache").rc()
    assert rc.models[0].paths and attach(rc).area.evaluation_grid is None
    grids = source_grids(rc)
    assert set(grids) == {"observations", "ens"} and grids["ens"][0].tolist() == LATS


def test_with_a_region_only_the_finer_source_is_averaged_and_the_observations_are_read_as_they_are(tmp_path):
    """With a region the grid is the chosen source's grid in the box: the observations are not given the grid."""
    _write_observations(tmp_path / "obs")
    _write_forecasts(tmp_path / "ens", fine=True)
    rc = attach(converted(config_dict(tmp_path), cache_dir=tmp_path / "cache", region="India").rc())
    assert rc.area.evaluation_grid.regrid == ("ens",)
    assert rc.observations.units.grid is None and rc.models[0].units.grid is rc.area.evaluation_grid
    result = train(rc)
    assert result.inputs.units == ("10.00_77.00", "10.00_77.25", "10.25_77.00", "10.25_77.25")
    assert TrainedBlend.from_training(result).grid.region == "India"
