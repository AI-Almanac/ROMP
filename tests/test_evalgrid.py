"""The evaluation grid: conservative averaging of finer sources onto the coarsest grid, chosen once per run.

Synthetic, small grids: the weights are exact areas on the sphere, a finer source is averaged (never sampled), missing
cells renormalise, 0-360 descending and -180-180 ascending grids meet, polygon units are built per source grid from its
native resolution, and a box is read from a global-like file with each stored chunk decoded once.
"""
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

from _decode_count import count_decodes
from hcwf.cache import reset_run_state
from hcwf.config.model import AreaSpec, Selection
from hcwf.spatial import evalgrid, regrid
from hcwf.spatial.units import EvaluationUnits

NO_MASKS = Selection(land_only=False, shp_only=False)


@pytest.fixture(autouse=True)
def fresh():
    """Hold no cached units or masks before or after a test."""
    reset_run_state()
    yield
    reset_run_state()


def _area(grid=None):
    """Return the area of a run in a region box with the evaluation grid ``grid``."""
    return AreaSpec(region="Ethiopia", mask=None, nc_mask=None, shpfile_dir=None, polygon=False, evaluation_grid=grid)


def _grid(lat, lon, source="MODEL", regrid_sources=("observations",)):
    """Return an `EvaluationGrid` of the given cells that averages the given sources onto it."""
    return evalgrid.EvaluationGrid(source, np.asarray(lat, float), np.asarray(lon, float), tuple(regrid_sources), (), "test")


def _sphere(lo, hi, dlon):
    """Return the area on the sphere (longitude degrees times the difference of the sines) of cells."""
    return dlon * (np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)))


def _centres(first, last, step):
    """Return cell centres ``first, first + step, ...`` up to ``last``."""
    return first + step * np.arange(int(round((last - first) / step)) + 1)


# -- exact areas on the sphere ----------------------------------------------------------------------------------------

FINE_LAT, FINE_LON = _centres(5.025, 8.975, 0.05), _centres(34.025, 37.975, 0.05)     # edges on multiples of 0.05
COARSE_LAT, COARSE_LON = _centres(4.75, 9.25, 0.25), _centres(33.75, 38.25, 0.25)     # edges on 0.125 + 0.25 k


def test_sphere_weights_are_exact_areas_and_conserve_on_non_nesting_grids():
    """Coarse cell areas sum from the fine cells they cover, and the total of a field is the same on both grids."""
    w = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON, cell_area="sphere")
    fine_area = (_sphere(FINE_LAT - 0.025, FINE_LAT + 0.025, 1.0)[:, None] * np.full(FINE_LON.size, 0.05)[None, :]).ravel()
    # every fine cell lies inside the coarse extent: its area is split among the coarse cells it overlaps, none lost
    np.testing.assert_allclose(np.asarray(w.sum(axis=0)).ravel(), fine_area, rtol=1e-12)
    # a coarse cell inside the fine extent is covered by fine cells whose shared areas add up to its own area
    covered = np.asarray(w.sum(axis=1)).reshape(COARSE_LAT.size, COARSE_LON.size)
    own = _sphere(COARSE_LAT - 0.125, COARSE_LAT + 0.125, 1.0)[:, None] * np.full(COARSE_LON.size, 0.25)[None, :]
    inner = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(covered[inner], own[inner], rtol=1e-12)
    assert 0 < covered[1, 1] < own[1, 1]      # an edge cell is only partly covered by the fine cells
    field = np.random.default_rng(3).gamma(0.5, 4.0, fine_area.size)
    transform = regrid.SparseTransform(w, None, tuple(str(i) for i in range(w.shape[0])), keys="coordinate",
                                       grid=(COARSE_LAT, COARSE_LON))
    mean = np.nan_to_num(transform.observed_mean(field[:, None])[:, 0])   # NaN where no fine cell reaches
    assert np.isclose((np.asarray(w.sum(axis=1)).ravel() * mean).sum(), (fine_area * field).sum(), rtol=1e-12)


def test_a_coarse_cell_is_the_area_weighted_mean_of_the_fine_cells_by_brute_force():
    """One partially overlapping coarse cell checked against an independent, loop-built overlap."""
    w = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON, cell_area="sphere").toarray()
    t_lat, t_lon = 3, 4                          # a coarse cell with centre 5.5, 34.75: edges 5.375..5.625 and 34.625..34.875
    lo_lat, hi_lat, lo_lon, hi_lon = COARSE_LAT[t_lat] - 0.125, COARSE_LAT[t_lat] + 0.125, \
        COARSE_LON[t_lon] - 0.125, COARSE_LON[t_lon] + 0.125
    expected = np.zeros((FINE_LAT.size, FINE_LON.size))
    for i, la in enumerate(FINE_LAT):
        for j, lo in enumerate(FINE_LON):
            dlat = min(hi_lat, la + 0.025) - max(lo_lat, la - 0.025)
            dlon = min(hi_lon, lo + 0.025) - max(lo_lon, lo - 0.025)
            if dlat > 0 and dlon > 0:
                expected[i, j] = dlon * (np.sin(np.deg2rad(min(hi_lat, la + 0.025)))
                                         - np.sin(np.deg2rad(max(lo_lat, la - 0.025))))
    np.testing.assert_allclose(w[t_lat * COARSE_LON.size + t_lon], expected.ravel(), rtol=1e-9, atol=1e-15)


def test_planar_is_still_the_default_and_polygon_weights_accept_the_sphere():
    """The default area stays planar (existing weight tables depend on it); the sphere differs by the latitude."""
    planar = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON)
    default = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON, cell_area="planar")
    assert (planar != default).nnz == 0
    sphere = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON, cell_area="sphere")
    ratio = sphere.sum() / planar.sum()
    assert np.isclose(ratio, np.cos(np.deg2rad(7.0)) * np.pi / 180, rtol=2e-3)   # about cos(latitude), per radian
    poly = [shapely.box(34.0, 5.0, 36.0, 7.0)]
    table = regrid.polygon_weights(poly, ["p"], FINE_LAT, FINE_LON, cell_area="sphere")
    assert np.isclose(table["weight"].sum(), _sphere(5.0, 7.0, 2.0), rtol=1e-12)


# -- the cells a source holds that have no value renormalise --------------------------------------------------------

def _fine_dataset(values, lat=FINE_LAT, lon=FINE_LON):
    """Return a lazy-looking dataset ``rain(time, lat, lon)`` of the fine grid."""
    time = pd.date_range("2020-01-01", periods=values.shape[0])
    return xr.Dataset({"rain": (("time", "lat", "lon"), values.astype("float32"))},
                      coords={"time": time, "lat": lat, "lon": lon})


def test_missing_fine_cells_renormalise_over_the_observed_area():
    """NaN fine cells (ocean in a land-only product) drop out of the mean; an all-NaN coarse cell is NaN."""
    rng = np.random.default_rng(1)
    values = rng.gamma(0.5, 4.0, (3, FINE_LAT.size, FINE_LON.size))
    values[:, :, :FINE_LON.size // 2] = np.nan          # the whole west half is missing
    values[:, 40:42, 60:62] = np.nan                    # and a hole in the east half
    grid = _grid(COARSE_LAT, COARSE_LON)
    out = evalgrid.region_read(_fine_dataset(values), "rain", area=_area(grid), select=NO_MASKS, source="observations",
                               read_by="time", dataset_name="obs", dir_fig=None)
    assert out.dims == ("time", "lat", "lon") and out.shape == (3, COARSE_LAT.size, COARSE_LON.size)
    np.testing.assert_array_equal(out["lat"].values, COARSE_LAT)
    w = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON, cell_area="sphere").toarray()
    flat = values.reshape(3, -1)
    for cell in (0, 20, 100, 130, 170):
        have = np.isfinite(flat[0])
        weight = w[cell] * have
        expected = (weight * np.nan_to_num(flat)).sum(axis=1) / weight.sum() if weight.sum() else np.full(3, np.nan)
        np.testing.assert_allclose(out.values.reshape(3, -1)[:, cell], expected, rtol=1e-6, equal_nan=True)
    assert np.isnan(out.values[:, :, 0]).all()          # west edge cells see only missing data
    assert np.isfinite(out.values[:, 3:-3, -3]).all()


def test_a_constant_field_stays_constant_and_an_unmasked_source_is_left_alone():
    """Averaging a constant gives the constant (weights are normalised); a source on the grid is read as it was."""
    values = np.full((2, FINE_LAT.size, FINE_LON.size), 7.0)
    grid = _grid(COARSE_LAT, COARSE_LON)
    out = evalgrid.region_read(_fine_dataset(values), "rain", area=_area(grid), select=NO_MASKS, source="observations",
                               read_by="time", dataset_name="obs", dir_fig=None)
    np.testing.assert_allclose(out.values[:, 1:-1, 1:-1], 7.0, rtol=1e-6)
    # a source that is not averaged (it is the evaluation grid's own) goes through region_select untouched
    native = _fine_dataset(np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5), lat=[5.0, 6.0, 7.0, 8.0],
                           lon=[34.0, 35.0, 36.0, 37.0, 38.0])
    same = evalgrid.region_read(native, "rain", area=_area(_grid([5.0, 6.0], [34.0, 35.0], regrid_sources=())),
                                select=NO_MASKS, source="MODEL", read_by="time", dataset_name="m", dir_fig=None)
    assert same.shape == (2, 4, 5)


# -- which grid is the evaluation grid ----------------------------------------------------------------------------

OBS = (_centres(3.025, 14.975, 0.05), _centres(33.025, 47.975, 0.05))
COARSE = (_centres(3.0, 15.0, 0.25), _centres(33.0, 48.0, 0.25))
COARSER = (_centres(3.0, 15.0, 0.5), _centres(33.0, 48.0, 0.5))


def test_the_coarsest_grid_is_chosen_by_default_and_the_first_wins_a_tie():
    """The coarsest source wins; with models of several sizes one grid serves the whole run."""
    assert evalgrid.choose({"observations": OBS, "A": COARSE})[0] == "A"
    source, reason = evalgrid.choose({"observations": OBS, "A": COARSE, "B": COARSER})
    assert source == "B" and "coarsest" in reason
    grid = evalgrid.build({"observations": OBS, "A": COARSE, "B": COARSER})
    assert grid.source == "B" and set(grid.regrid) == {"observations", "A"}
    assert np.array_equal(grid.lat, COARSER[0]) and np.array_equal(grid.lon, COARSER[1])
    assert evalgrid.choose({"observations": COARSE, "A": COARSE})[0] == "observations"
    assert evalgrid.choose({"A": COARSE, "observations": COARSE})[0] == "A"


def test_the_override_names_a_source_and_an_unknown_one_is_an_error():
    """``evaluation_grid`` picks the observations or a model, even a finer one; a typo names the sources."""
    grid = evalgrid.build({"observations": OBS, "A": COARSE, "B": COARSER}, "observations")
    assert grid.source == "observations" and set(grid.regrid) == {"A", "B"} and "evaluation_grid" in grid.reason
    assert evalgrid.build({"observations": OBS, "A": COARSE}, "A").source == "A"
    with pytest.raises(ValueError, match="not a source of the run"):
        evalgrid.choose({"observations": OBS, "A": COARSE}, "C")


def test_sources_on_one_grid_have_no_evaluation_grid_even_in_another_convention():
    """Grids equal to a tolerance (also 0-360 against -180-180) are one grid: nothing is averaged."""
    same = (OBS[0], OBS[1] + 1e-5)
    assert evalgrid.build({"observations": OBS, "A": same}) is None
    wrapped = (OBS[0], ((OBS[1] + 180.0) % 360.0) - 180.0 + 360.0)
    assert evalgrid.same_grid(*OBS, *wrapped)
    assert evalgrid.build({"observations": OBS, "A": (OBS[0][:-1], OBS[1])}) is not None


def test_the_grid_describes_itself_and_keys_the_caches():
    """The lines name the source, the reason and what is averaged; the key follows the cells and the sources."""
    grid = evalgrid.build({"observations": OBS, "A": COARSE})
    text = "\n".join(grid.lines())
    assert "the grid of A" in text and "coarsest" in text and "averaged conservatively" in text
    assert grid.key() != evalgrid.build({"observations": OBS, "A": COARSER}).key()
    assert grid.key() == evalgrid.build({"observations": OBS, "A": COARSE}).key()


# -- longitude conventions and orientation -----------------------------------------------------------------------------

def _model(lon_start, lat_desc=True):
    """Return a 1 degree global-like model on 0-360 longitudes with a field that tells longitude and latitude blocks."""
    lat = np.arange(90.0, -91.0, -1.0) if lat_desc else np.arange(-90.0, 91.0, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    shifted = ((lon + 180.0) % 360.0) - 180.0                 # the longitude as -180..180
    block = np.where((lat >= 4) & (lat <= 5), 3.0, np.where((lat >= 6) & (lat <= 7), 7.0, 0.0))
    values = shifted[None, None, :] * 100.0 + block[None, :, None]
    time = pd.date_range("2020-01-01", periods=2)
    return xr.Dataset({"tp": (("time", "lat", "lon"), np.broadcast_to(values, (2, lat.size, lon.size)).astype("float64"))},
                      coords={"time": time, "lat": lat, "lon": lon})


@pytest.mark.parametrize("lat_desc", [True, False])
@pytest.mark.parametrize("west", [33.0, -10.0])
def test_a_descending_0_360_model_meets_an_ascending_minus_180_180_grid(west, lat_desc):
    """A box in positive or negative longitudes cuts a 0-360 grid, latitude either way round, onto 2 degree cells."""
    lat = np.array([4.5, 6.5])                                  # cells 3.5-5.5 and 5.5-7.5 hold whole model cells
    lon = west + 1.0 + 2.0 * np.arange(3)                       # cells 2 degrees wide, centres west + 1, +3, +5
    grid = _grid(lat, lon, source="OBS", regrid_sources=("M",))
    out = evalgrid.region_read(_model(0.0, lat_desc), "tp", area=_area(grid), select=NO_MASKS, source="M",
                               read_by="time", dataset_name="M", dir_fig=None)
    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out["lat"].values, lat)
    np.testing.assert_array_equal(out["lon"].values, lon)
    expected = lon[None, :] * 100.0 + np.array([[3.0], [7.0]])    # the longitude (uniform in a cell) and the block
    np.testing.assert_allclose(out.values[0], expected, rtol=1e-9)


def test_a_box_across_the_longitude_seam_of_the_file_is_refused_clearly():
    """The seam is not supported (as for a region's box); the error says so instead of returning a wrong cut."""
    grid = _grid([4.0, 6.0], [-1.0, 1.0, 3.0], source="OBS", regrid_sources=("M",))
    with pytest.raises(ValueError, match="crosses the longitude seam"):
        evalgrid.region_read(_model(0.0), "tp", area=_area(grid), select=NO_MASKS, source="M", read_by="time",
                             dataset_name="M", dir_fig=None)


# -- polygon units are built per source grid ---------------------------------------------------------------------------

def test_polygon_units_are_built_for_each_source_grid_from_its_native_resolution():
    """One transform per source grid, each from its own cells (never one source via the coarser grid)."""
    polygon = shapely.Polygon([(34.1, 5.1), (36.3, 5.4), (36.0, 7.3), (34.4, 7.0)])
    units = EvaluationUnits(pd.DataFrame({"id": ["adm1"]}), "polygon", geometry=[polygon])
    fine, coarse = units.transform(FINE_LAT, FINE_LON), units.transform(COARSE_LAT, COARSE_LON)
    assert fine is not coarse
    assert fine.shape == (1, FINE_LAT.size * FINE_LON.size) and coarse.shape == (1, COARSE_LAT.size * COARSE_LON.size)
    assert units.transform(FINE_LAT, FINE_LON) is fine and units.transform(COARSE_LAT, COARSE_LON) is coarse
    direct = regrid.polygon_weights([polygon], ["adm1"], FINE_LAT, FINE_LON)
    assert fine.weighted_matrix.nnz == len(direct) and fine.weighted_matrix.nnz > 20 * coarse.weighted_matrix.nnz
    assert np.isclose(fine.weighted_matrix.sum(), direct["weight"].sum(), rtol=1e-9)   # areas on the sphere by default
    assert np.isclose(coarse.weighted_matrix.sum(), fine.weighted_matrix.sum(), rtol=1e-2)
    # a field with a gradient: the polygon mean from the fine cells is not the mean via the coarse cells
    field = (FINE_LAT[:, None] * 3.0 + FINE_LON[None, :] ** 2 * 0.1) * np.ones((FINE_LAT.size, FINE_LON.size))
    from_fine = fine.observed_mean(field.reshape(-1, 1))[0, 0]
    cells = regrid.grid_weights(FINE_LAT, FINE_LON, COARSE_LAT, COARSE_LON, cell_area="sphere")
    via = regrid.SparseTransform(cells, None, tuple(map(str, range(cells.shape[0]))), keys="coordinate",
                                 grid=(COARSE_LAT, COARSE_LON)).observed_mean(field.reshape(-1, 1))
    via_coarse = coarse.observed_mean(via)[0, 0]
    assert abs(from_fine - via_coarse) > 1e-3
    w = direct.assign(w=direct["weight"])
    assert np.isclose(from_fine, (w["w"] * (direct["lat"] * 3.0 + direct["lon"] ** 2 * 0.1)).sum() / w["w"].sum(),
                      rtol=1e-6)


# -- reading a box from a global-like file ----------------------------------------------------------------------------

def _write_global(path, n_init=3, n_step=2):
    """Write a 1 degree global model, one whole field per (init, step) chunk, latitude flipped, zlib."""
    lat = np.arange(90.0, -91.0, -1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    rng = np.random.default_rng(2)
    ds = xr.Dataset({"tp": (("time", "step", "lat", "lon"),
                            rng.gamma(0.5, 4.0, (n_init, n_step, lat.size, lon.size)).astype("float32"),
                            {"units": "mm"})},
                    coords={"time": pd.date_range("2020-05-01", periods=n_init), "step": np.arange(1, n_step + 1),
                            "lat": lat, "lon": lon})
    ds.to_netcdf(path, encoding={"tp": {"chunksizes": (1, 1, lat.size, lon.size), "zlib": True}})
    return path


def test_a_global_field_is_decoded_once_and_cut_to_the_box(tmp_path):
    """Every needed (init, step) field is decoded exactly once; no other chunk is touched."""
    path = _write_global(tmp_path / "global.nc")
    grid = _grid(_centres(4.0, 14.0, 2.0), _centres(34.0, 46.0, 2.0), source="OBS", regrid_sources=("M",))
    with xr.open_dataset(path) as ds, count_decodes() as decodes:
        out = evalgrid.region_read(ds, "tp", area=_area(grid), select=NO_MASKS, source="M", read_by="time",
                                   dataset_name="M", dir_fig=None)
    seen = decodes.only("tp")
    assert out.shape == (3, 2, 6, 7)
    assert seen.chunks == 6 and seen.worst() == 1
    with xr.open_dataset(path) as ds:
        field = ds["tp"].isel(time=1, step=0).load()
    # the cell centred 4N, 34E (edges 3-5N, 33-35E) by hand: half of the 3N and 5N rows and columns, all of 4N and 34E
    total = weight = 0.0
    for la in (3.0, 4.0, 5.0):
        for lo in (33.0, 34.0, 35.0):
            lat_cover = min(5.0, la + 0.5) - max(3.0, la - 0.5)
            lon_cover = min(35.0, lo + 0.5) - max(33.0, lo - 0.5)
            area = lon_cover * (np.sin(np.deg2rad(min(5.0, la + 0.5))) - np.sin(np.deg2rad(max(3.0, la - 0.5))))
            total += area * float(field.sel(lat=la, lon=lo))
            weight += area
            assert lat_cover > 0 and lon_cover > 0
    assert np.isclose(float(out.isel(time=1, step=0, lat=0, lon=0)), total / weight, rtol=1e-6)


def test_only_the_tiles_of_a_chunked_grid_that_feed_the_box_are_decoded_once(tmp_path):
    """A land-style product chunked over time and tiles: the tiles of the box are read once, the others not at all."""
    lat, lon = _centres(-9.75, 29.75, 0.5), _centres(20.25, 59.75, 0.5)
    rng = np.random.default_rng(4)
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), rng.gamma(0.5, 4.0, (40, lat.size, lon.size)).astype("float32"))},
                    coords={"time": pd.date_range("2020-01-01", periods=40), "lat": lat, "lon": lon})
    path = tmp_path / "obs.nc"
    ds.to_netcdf(path, encoding={"rain": {"chunksizes": (20, 16, 16), "zlib": True}})
    grid = _grid(_centres(4.0, 14.0, 2.0), _centres(34.0, 46.0, 2.0), source="M", regrid_sources=("observations",))
    with xr.open_dataset(path) as opened, count_decodes() as decodes:
        out = evalgrid.region_read(opened, "rain", area=_area(grid), select=NO_MASKS, source="observations",
                                   read_by="time", dataset_name="obs", dir_fig=None)
    seen = decodes.only("rain")
    assert out.shape == (40, 6, 7)
    assert seen.worst() == 1 and 0 < seen.chunks < 2 * 5 * 5
