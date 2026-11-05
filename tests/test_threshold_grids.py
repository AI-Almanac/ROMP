"""Threshold grid files meet the run's grid: used as is, averaged onto the evaluation grid, or onto the units.

Synthetic, small grids: a finer file is averaged onto a coarser evaluation grid (a missing cell renormalises), a file
on the evaluation grid is returned untouched, a file that does not cover the grid leaves NaN and a warning, and on
polygon units the file goes to the units from its own grid with the sphere areas of the rain.
"""
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
import xarray as xr

from hcwf.config.model import AreaSpec
from hcwf.config.onset import onset_setup
from hcwf.io import thresholds as thresholds_io
from hcwf.runs import check as check_mod
from hcwf.spatial import evalgrid, regrid
from hcwf.spatial.units import UnitsSpec

COARSE = (np.array([5.0, 7.0, 9.0]), np.array([34.0, 36.0, 38.0]))                 # 2 degree cells
FINE = (np.arange(4.5, 10.0, 1.0), np.arange(33.5, 39.0, 1.0))                     # 1 degree cells over the same area


def _grid(lat=COARSE[0], lon=COARSE[1]):
    """Return an evaluation grid of the cells given."""
    return evalgrid.EvaluationGrid("observations", np.asarray(lat, float), np.asarray(lon, float), (), (), "test")


def _area(grid):
    """Return the area of a run without a region, with the evaluation grid."""
    return AreaSpec(region=None, mask=None, nc_mask=None, shpfile_dir=None, polygon=False, evaluation_grid=grid)


def _field(lat, lon, fill=None):
    """Return a threshold field whose value is its latitude, or ``fill``."""
    values = np.broadcast_to(np.asarray(lat, float)[:, None], (len(lat), len(lon))).copy() if fill is None \
        else np.full((len(lat), len(lon)), fill)
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="thr")


def _file(tmp_path, field, name="thr.nc"):
    """Write the field as a one-variable NetCDF file and return the ``path:variable`` reference."""
    path = tmp_path / name
    field.to_dataset(name="thr").to_netcdf(path)
    return f"{path}:thr"


def _onset(reference):
    """Return the onset setup whose only named threshold is the file ``reference``."""
    return onset_setup({"onset_definition": "moron_robertson", "obs_day_boundary": "00 UTC, labelled by start",
                        "model_lead_label": "start", "max_forecast_day": 10, "threshold_years": (2018, 2019),
                        "thresholds": {"wet_spell_amount": reference}})


def _load(onset, grid, obs_units=None):
    """Load the thresholds of ``onset`` as a run on ``grid`` (or units) does."""
    obs = SimpleNamespace(units=obs_units)
    options = SimpleNamespace(threshold_scale=None, threshold_table=None)
    return thresholds_io.load_onset_thresholds(onset, obs, _area(grid), options, None, SimpleNamespace(figure_dir=None))


def test_a_finer_threshold_grid_is_averaged_onto_a_coarser_evaluation_grid(tmp_path):
    """Each coarse cell takes the area-weighted mean of the fine cells; a missing fine cell does not pull it down."""
    field = _field(*FINE)
    field.values[0, 0] = np.nan                                  # one missing cell in the first coarse cell
    got = _load(_onset(_file(tmp_path, field)), _grid())["wet_spell_amount"]
    assert got.shape == (3, 3) and np.array_equal(got["lat"].values, COARSE[0])
    # the value is the latitude: the mean over a coarse cell's rows is about its centre; the first cell lost one row-0 cell
    np.testing.assert_allclose(got.values[1:], np.repeat(COARSE[0][1:, None], 3, axis=1), atol=2e-3)
    assert got.values[0, 0] == pytest.approx(np.average([4.5, 5.5, 5.5], weights=[1, 1, 1]), abs=2e-2)
    assert np.isfinite(got.values).all()


def test_a_threshold_grid_on_the_evaluation_grid_is_used_as_it_is(tmp_path):
    """The same grid gives back the values untouched (also when the file stores its latitudes the other way round)."""
    field = _field(*COARSE)
    got = _load(_onset(_file(tmp_path, field)), _grid())["wet_spell_amount"]
    np.testing.assert_array_equal(got.values, field.values)
    flipped = _field(COARSE[0][::-1], COARSE[1])
    again = _load(_onset(_file(tmp_path, flipped, "flipped.nc")), _grid())["wet_spell_amount"]
    assert again.shape == (3, 3)
    assert evalgrid.same_grid(again["lat"].values, again["lon"].values, *COARSE)


def test_a_threshold_grid_that_does_not_cover_the_grid_leaves_nan_and_warns(tmp_path, caplog):
    """Cells outside the file's extent are NaN and counted; the loader warns and the check hints."""
    part = _field(COARSE[0][:2], COARSE[1])                      # the two southern rows of cells
    onset = _onset(_file(tmp_path, part))
    with caplog.at_level(logging.WARNING):
        got = _load(onset, _grid())["wet_spell_amount"]
    assert np.isnan(got.values[2]).all() and np.isfinite(got.values[:2]).all()
    assert any("does not cover 3 of the 9 evaluation cells" in r.message for r in caplog.records)
    obs = SimpleNamespace(units=None)
    (info,) = thresholds_io.threshold_grids(onset, obs, _area(_grid()), SimpleNamespace(), None,
                                            SimpleNamespace(figure_dir=None))
    assert (info.how, info.uncovered, info.lat, info.lon) == ("averaged", 3, 2, 3)
    report = check_mod.CheckReport()
    rc = SimpleNamespace(onset=onset, observations=obs, area=_area(_grid()), onset_options=SimpleNamespace(),
                         evaluation=None, output=SimpleNamespace(figure_dir=None), parallel=None)
    check_mod._threshold_grids(report, rc)
    text = "\n".join(report.lines)
    assert "averaged conservatively onto it" in text and "not the threshold of the averaged rain" in text
    assert "onset.threshold_years" in text and any("does not cover 3 cells" in h for h in report.warnings)


def test_the_check_says_when_a_threshold_is_used_as_is(tmp_path):
    """A threshold on the evaluation grid gets a line and no note about averaging."""
    onset = _onset(_file(tmp_path, _field(*COARSE)))
    rc = SimpleNamespace(onset=onset, observations=SimpleNamespace(units=None), area=_area(_grid()),
                         onset_options=SimpleNamespace(), evaluation=None, output=SimpleNamespace(figure_dir=None),
                         parallel=None)
    report = check_mod.CheckReport()
    check_mod._threshold_grids(report, rc)
    text = "\n".join(report.lines)
    assert "on the evaluation grid, used as is" in text and "averaged" not in text and not report.warnings


def test_a_threshold_grid_goes_to_polygon_units_from_its_own_grid(tmp_path):
    """On admin units the file is averaged onto the polygons with the sphere areas the rain uses."""
    import geopandas as gpd

    polygon = shapely.box(34.5, 5.5, 37.5, 8.5)
    path = tmp_path / "units.geojson"
    gpd.GeoDataFrame({"code": ["a"]}, geometry=[polygon], crs="EPSG:4326").to_file(path, driver="GeoJSON")
    spec = UnitsSpec(shapefile=str(path), unit_id_col="code")
    field = _field(*FINE)
    got = _load(_onset(_file(tmp_path, field)), None, obs_units=spec)["wet_spell_amount"]
    assert got.dims == ("unit",) and got["unit"].values.tolist() == ["a"]
    table = regrid.polygon_weights([polygon], ["a"], *FINE)
    expected = np.average(table["lat"], weights=table["weight"])
    assert got.values[0] == pytest.approx(expected, rel=1e-9)
    planar = regrid.polygon_weights([polygon], ["a"], *FINE, cell_area="planar")
    assert expected != pytest.approx(np.average(planar["lat"], weights=planar["weight"]), rel=1e-6)
