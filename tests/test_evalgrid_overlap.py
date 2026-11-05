"""The evaluation grid of a run without a region: the chosen grid cut to the area every source covers.

Synthetic, small grids: sources of different resolutions and extents are scored on the coarser grid over the overlap,
non-overlapping sources and a seam crossing fail with a Fix, the override works, and matching grids have none.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.cache import reset_run_state
from hcwf.config.model import AreaSpec, Selection
from hcwf.errors import EvaluationGridError
from hcwf.spatial import evalgrid

NO_MASKS = Selection(land_only=False, shp_only=False)


@pytest.fixture(autouse=True)
def fresh():
    """Hold no cached units or masks before or after a test."""
    reset_run_state()
    yield
    reset_run_state()


def _centres(first, last, step):
    """Return cell centres ``first, first + step, ...`` up to ``last``."""
    return first + step * np.arange(int(round((last - first) / step)) + 1)


FINE = (_centres(3.025, 14.975, 0.05), _centres(33.025, 47.975, 0.05))      # 3..15 N, 33..48 E, 0.05 degrees
COARSE = (_centres(0.125, 19.875, 0.25), _centres(30.125, 49.875, 0.25))    # 0..20 N, 30..50 E, 0.25 degrees


def _area(grid):
    """Return the area of a run without a region, with the evaluation grid ``grid``."""
    return AreaSpec(region=None, mask=None, nc_mask=None, shpfile_dir=None, polygon=False, evaluation_grid=grid)


def _dataset(grid, value=5.0, lon_shift=0.0):
    """Return a constant ``rain`` field on a grid, with longitudes shifted by ``lon_shift`` degrees."""
    lat, lon = grid[0], grid[1] + lon_shift
    return xr.Dataset({"rain": (("time", "lat", "lon"), np.full((2, lat.size, lon.size), value, dtype="float32"))},
                      coords={"time": pd.date_range("2020-01-01", periods=2), "lat": lat, "lon": lon})


def test_mismatched_grids_without_a_region_are_scored_on_the_coarser_grid_over_the_overlap():
    """The grid is the coarser source's cells inside the finer source's extent; every source is read there."""
    grid = evalgrid.build({"observations": FINE, "M": COARSE}, cut=True)
    assert grid.source == "M" and not grid.lon_global and set(grid.regrid) == {"observations", "M"}
    assert grid.box == pytest.approx((3.025 - 0.025, 14.975 + 0.025, 33.0, 48.0))
    np.testing.assert_allclose(grid.lat, _centres(3.125, 14.875, 0.25))
    np.testing.assert_allclose(grid.lon, _centres(33.125, 47.875, 0.25))
    text = "\n".join(grid.lines())
    assert "area every source covers" in text and "overlap of the sources" in text and "latitudes 3 to 15" in text
    for name, source in (("observations", FINE), ("M", COARSE)):
        out = evalgrid.region_read(_dataset(source), "rain", area=_area(grid), select=NO_MASKS, source=name,
                                   read_by="time", dataset_name=name, dir_fig=None)
        assert out.shape == (2, grid.lat.size, grid.lon.size)
        np.testing.assert_allclose(out["lat"].values, grid.lat)
        np.testing.assert_allclose(out.values, 5.0, rtol=1e-6)


def test_a_finer_source_is_averaged_over_each_coarse_cell_in_the_overlap():
    """A gradient in latitude: each coarse cell takes the area-weighted mean of the fine cells inside it."""
    grid = evalgrid.build({"observations": FINE, "M": COARSE}, cut=True)
    data = _dataset(FINE)
    data["rain"] = data["rain"] * 0 + xr.DataArray(FINE[0], dims="lat")
    out = evalgrid.region_read(data, "rain", area=_area(grid), select=NO_MASKS, source="observations",
                               read_by="time", dataset_name="obs", dir_fig=None)
    np.testing.assert_allclose(out.values[0, :, 0], grid.lat, atol=2e-3)   # the mean latitude of a cell is its centre


def test_the_override_works_without_a_region():
    """``evaluation_grid`` names the finer source: its cells inside the coarser source's extent are the grid."""
    grid = evalgrid.build({"observations": FINE, "M": COARSE}, "observations", cut=True)
    assert grid.source == "observations" and "evaluation_grid" in grid.reason
    assert grid.lat.size == FINE[0].size and grid.lon.size == FINE[1].size     # the coarse source covers all of it
    other = evalgrid.build({"observations": (FINE[0], FINE[1][:100]), "M": COARSE}, "M", cut=True)
    assert other.source == "M" and other.lon.max() < FINE[1][99] + 0.125


def test_sources_that_do_not_overlap_fail_with_a_fix():
    """Disjoint latitudes, then disjoint longitudes: the error says what to do."""
    north = (_centres(25.125, 29.875, 0.25), COARSE[1])
    with pytest.raises(EvaluationGridError, match=r"latitudes of the sources do not overlap.*Fix: set region"):
        evalgrid.build({"observations": FINE, "M": north}, cut=True)
    east = (COARSE[0], _centres(60.125, 69.875, 0.25))
    with pytest.raises(EvaluationGridError, match=r"no longitudes in common.*Fix: set region"):
        evalgrid.build({"observations": FINE, "M": east}, cut=True)


def test_an_overlap_across_the_longitude_seam_of_a_file_fails_with_a_fix():
    """A global 0-360 source and a regional one that straddles 0 degrees cannot be cut by a box."""
    regional = (_centres(0.5, 9.5, 1.0), _centres(-19.5, 9.5, 1.0))
    world = (_centres(-89.5, 89.5, 1.0), _centres(0.5, 359.5, 1.0))
    with pytest.raises(EvaluationGridError, match=r"cross the longitude seam.*Fix"):
        evalgrid.build({"observations": regional, "M": world}, "M", cut=True)


def test_global_sources_in_two_conventions_meet_on_all_longitudes():
    """A 0-360 coarse and a -180-180 finer global source: all longitudes, only the latitudes they share."""
    obs = (_centres(-59.75, 59.75, 0.5), _centres(-179.75, 179.75, 0.5))
    model = (_centres(-89.5, 89.5, 1.0), _centres(0.5, 359.5, 1.0))
    grid = evalgrid.build({"observations": obs, "M": model}, cut=True)
    assert grid.source == "M" and grid.lon_global and grid.box[:2] == (-60.0, 60.0)
    assert grid.lat.min() == -59.5 and grid.lat.max() == 59.5 and grid.lon.size == 360
    for name, source in (("observations", obs), ("M", model)):
        out = evalgrid.region_read(_dataset(source), "rain", area=_area(grid), select=NO_MASKS, source=name,
                                   read_by="time", dataset_name=name, dir_fig=None)
        assert out.shape == (2, grid.lat.size, grid.lon.size)
        np.testing.assert_allclose(out.values, 5.0, rtol=1e-6)


def test_matching_grids_have_no_evaluation_grid_without_a_region():
    """Sources on one grid are read as they always were."""
    assert evalgrid.build({"observations": COARSE, "M": COARSE}, cut=True) is None
    wrapped = (COARSE[0], COARSE[1] + 360.0)
    assert evalgrid.build({"observations": COARSE, "M": wrapped}, cut=True) is None
