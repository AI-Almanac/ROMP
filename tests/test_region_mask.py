"""Region masks: shapes, mask files and conditions are built on each dataset's own grid, with hand-computed answers."""

import logging
import json

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box, mapping

from momp.lib.config_schema import ConfigSchemaError
from momp.lib.config_yaml import build_config
from momp.utils import mask as mask_mod
from momp.utils.mask import MaskError, MaskSpec, align_raster_mask, build_mask, rasterise

LAT = np.array([0.5, 1.5, 2.5, 3.5])
LON = np.array([0.5, 1.5, 2.5, 3.5])


@pytest.fixture(autouse=True)
def fresh_masks():
    """Each test starts with no built masks or records."""
    mask_mod.reset()
    yield
    mask_mod.reset()


def write_mask(path, lat, lon, values, name="mask", lat_name="lat", lon_name="lon"):
    """Write a 2-D field on the given axes as NetCDF and return the path."""
    xr.Dataset({name: ((lat_name, lon_name), np.asarray(values))}, coords={lat_name: lat, lon_name: lon}).to_netcdf(path)
    return str(path)


def cells(keep, lat=LAT, lon=LON):
    """The kept cells as a set of (lat, lon) centres."""
    return {(float(lat[i]), float(lon[j])) for i, j in zip(*np.nonzero(keep))}


def finer(step, lo=0.0, hi=4.0):
    """Cell centres of a grid of the given step between lo and hi."""
    return np.arange(lo + step / 2, hi, step)


# ---------------------------------------------------------------------------------------------- shapes

def test_centre_rule_by_hand():
    """A box from lon 0.7 to 2.3 and lat 0 to 2 holds only the centre (1.5) of the three longitudes, in 2 rows."""
    keep = rasterise(box(0.7, 0.0, 2.3, 2.0), LAT, LON, "centre")
    assert cells(keep) == {(0.5, 1.5), (1.5, 1.5)}


def test_overlap_rule_by_hand():
    """The same box covers 0.3 of the outer columns and all of the middle one: 0.25 keeps 3 columns, 0.5 keeps 1."""
    shape = box(0.7, 0.0, 2.3, 2.0)
    assert cells(rasterise(shape, LAT, LON, "overlap_fraction", 0.25)) == {
        (la, lo) for la in (0.5, 1.5) for lo in (0.5, 1.5, 2.5)}
    assert cells(rasterise(shape, LAT, LON, "overlap_fraction", 0.5)) == {(0.5, 1.5), (1.5, 1.5)}


def test_overlap_rule_on_a_triangle():
    """The triangle (0,0) (2,0) (0,2): cell [0,1]^2 is inside, the two next to it are half inside, the 4th outside."""
    from shapely.geometry import Polygon

    triangle = Polygon([(0, 0), (2, 0), (0, 2)])
    assert cells(rasterise(triangle, LAT, LON, "overlap_fraction", 0.5)) == {(0.5, 0.5), (0.5, 1.5), (1.5, 0.5)}
    assert cells(rasterise(triangle, LAT, LON, "overlap_fraction", 0.51)) == {(0.5, 0.5)}
    assert cells(rasterise(triangle, LAT, LON, "overlap_fraction", 1.0)) == {(0.5, 0.5)}


def test_shape_on_a_0_360_grid():
    """Cell longitudes are wrapped into -180..180 for the shape: the grid 190..193 holds the shape at -170..-167."""
    lon = np.array([190.5, 191.5, 192.5, 193.5])
    keep = rasterise(box(-169.0, 0.0, -167.0, 2.0), LAT, lon, "centre")
    assert cells(keep, LAT, lon) == {(0.5, 191.5), (0.5, 192.5), (1.5, 191.5), (1.5, 192.5)}


# ---------------------------------------------------------------------------------------------- same mask, three ways

SHAPE = box(1.0, 0.0, 3.0, 2.0)
WANT = {(0.5, 1.5), (0.5, 2.5), (1.5, 1.5), (1.5, 2.5)}


def test_same_mask_as_shape_file_on_the_data_grid_and_file_on_a_finer_grid(tmp_path):
    """The shape, a 0/1 file on the data grid, a GeoJSON and a file on a 4 times finer grid keep the same cells."""
    import geopandas as gpd

    by_shape = rasterise(SHAPE, LAT, LON, "centre")
    assert cells(by_shape) == WANT

    on_grid = write_mask(tmp_path / "grid.nc", LAT, LON, by_shape.astype(int))
    fine_lat, fine_lon = finer(0.25), finer(0.25)
    on_fine = write_mask(tmp_path / "fine.nc", fine_lat, fine_lon,
                         rasterise(SHAPE, fine_lat, fine_lon, "centre").astype(int))
    geojson = tmp_path / "shape.geojson"
    gpd.GeoDataFrame(geometry=[SHAPE], crs="EPSG:4326").to_file(geojson, driver="GeoJSON")

    for spec in (MaskSpec(file=on_grid), MaskSpec(file=on_fine), MaskSpec(file=str(geojson))):
        assert cells(build_mask(spec, LAT, LON).keep) == WANT, spec


def test_same_grid_is_used_directly_and_a_finer_grid_is_regridded(tmp_path):
    """The printed lines say which: direct use on the same grid, area-weighted fraction for a finer mask."""
    keep = rasterise(SHAPE, LAT, LON, "centre")
    on_grid = write_mask(tmp_path / "grid.nc", LAT, LON, keep.astype(int))
    fine = finer(0.5)
    on_fine = write_mask(tmp_path / "fine.nc", fine, fine, rasterise(SHAPE, fine, fine, "centre").astype(int))
    same = "\n".join(build_mask(MaskSpec(file=on_grid), LAT, LON).lines)
    regridded = "\n".join(build_mask(MaskSpec(file=on_fine), LAT, LON).lines)
    assert "same grid" in same and "used directly" in same
    assert "area-weighted" in regridded and "0.5 x 0.5 deg" in regridded and "kept cells: 4 of 16" in regridded


def test_finer_mask_fraction_by_hand(tmp_path):
    """A 0.5 degree mask keeping 3 of the 4 sub-cells of a data cell keeps it at 0.5 and 0.75, not at 0.8."""
    fine = finer(0.5)
    values = np.zeros((8, 8), dtype=int)
    values[0:2, 0:2] = 1
    values[0, 0] = 0                       # data cell (0.5, 0.5): 3 of 4 sub-cells kept
    path = write_mask(tmp_path / "fine.nc", fine, fine, values)
    for fraction, expected in ((0.5, {(0.5, 0.5)}), (0.75, {(0.5, 0.5)}), (0.8, set())):
        spec = MaskSpec(file=path, overlap_fraction=fraction)
        if expected:
            assert cells(build_mask(spec, LAT, LON).keep) == expected
        else:
            with pytest.raises(MaskError, match="keeps no cell"):
                build_mask(spec, LAT, LON)


def test_coarser_mask_takes_the_nearest_cell(tmp_path):
    """A 2 degree mask [[1, 0], [0, 0]] gives the 4 data cells of the south-west quadrant."""
    path = write_mask(tmp_path / "coarse.nc", [1.0, 3.0], [1.0, 3.0], [[1, 0], [0, 0]])
    result = build_mask(MaskSpec(file=path), LAT, LON)
    assert cells(result.keep) == {(la, lo) for la in (0.5, 1.5) for lo in (0.5, 1.5)}
    assert "nearest mask cell" in "\n".join(result.lines)


def test_offset_grid_is_regridded_by_the_nearest_cell(tmp_path):
    """A mask of the same resolution but with centres 0.2 degrees off is not 'the same grid': nearest cell."""
    values = np.zeros((6, 6), dtype=int)
    values[1, 1] = 1                                         # the mask cell centred on (1.5, 1.5)
    path = write_mask(tmp_path / "offset.nc", np.arange(0.5, 6.0), np.arange(0.5, 6.0), values)
    lat = lon = np.array([1.3, 2.3])
    result = build_mask(MaskSpec(file=path), lat, lon)
    assert cells(result.keep, lat, lon) == {(1.3, 1.3)}
    text = "\n".join(result.lines)
    assert "nearest mask cell" in text and "largest distance" in text and "same grid" not in text


def test_latitude_descending_and_0_360_give_identical_results(tmp_path):
    """The same mask with latitudes north to south and longitudes 0..360 gives the cells of the plain file."""
    lon = np.array([-1.5, -0.5, 0.5, 1.5])
    values = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]])
    plain = write_mask(tmp_path / "plain.nc", LAT, lon, values)
    flipped = write_mask(tmp_path / "flipped.nc", LAT[::-1], lon, values[::-1])
    lon360 = np.array([358.5, 359.5, 0.5, 1.5])
    shifted = write_mask(tmp_path / "shifted.nc", LAT[::-1], lon360, values[::-1])
    reference = cells(build_mask(MaskSpec(file=plain), LAT, lon).keep, LAT, lon)
    assert len(reference) == 6
    for path in (flipped, shifted):
        result = build_mask(MaskSpec(file=path), LAT, lon)
        assert cells(result.keep, LAT, lon) == reference
    text = "\n".join(build_mask(MaskSpec(file=shifted), LAT, lon).lines)
    assert "flipped" in text and "shifted the mask's longitudes" in text


def test_data_in_0_360_and_mask_in_minus_180_180(tmp_path):
    """Data on 190..193 east and a mask file on -170..-167 give the same cells as the file on the data grid."""
    lon = np.array([190.5, 191.5, 192.5, 193.5])
    values = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    same = write_mask(tmp_path / "same.nc", LAT, lon, values)
    wrapped = write_mask(tmp_path / "wrapped.nc", LAT, lon - 360, values)
    assert cells(build_mask(MaskSpec(file=wrapped), LAT, lon).keep, LAT, lon) == cells(
        build_mask(MaskSpec(file=same), LAT, lon).keep, LAT, lon)


def test_unordered_data_axes_keep_the_data_order(tmp_path):
    """A data grid with latitudes north to south gets a mask in that same order."""
    path = write_mask(tmp_path / "m.nc", LAT, LON, rasterise(SHAPE, LAT, LON, "centre").astype(int))
    keep = build_mask(MaskSpec(file=path), LAT[::-1], LON).keep
    assert cells(keep, LAT[::-1], LON) == WANT


def test_mask_that_does_not_cover_the_data_is_an_error_with_both_extents(tmp_path):
    """A mask over longitude 0..2 cannot serve data over 0..4; the message gives both extents."""
    path = write_mask(tmp_path / "small.nc", np.array([0.5, 1.5, 2.5, 3.5]), np.array([0.5, 1.5]),
                      np.ones((4, 2), dtype=int))
    with pytest.raises(MaskError) as error:
        build_mask(MaskSpec(file=path), LAT, LON)
    text = str(error.value)
    assert "does not cover the data" in text
    assert "longitude 0..2" in text and "longitude 0..4" in text and "latitude 0..4" in text


# ---------------------------------------------------------------------------------------------- conditions

def test_condition_above_below_and_both(tmp_path):
    """A field of 0, 1000, 2000, 3000 by column: above 1500 keeps the last two columns, below 2500 the first three."""
    field = np.tile([0.0, 1000.0, 2000.0, 3000.0], (4, 1))
    path = write_mask(tmp_path / "dem.nc", LAT, LON, field, name="elevation")
    above = build_mask(MaskSpec(condition={"file": path, "variable": "elevation", "above": 1500}), LAT, LON).keep
    assert cells(above) == {(la, lo) for la in LAT for lo in (2.5, 3.5)}
    below = build_mask(MaskSpec(condition={"file": path, "variable": "elevation", "below": 2500}), LAT, LON).keep
    assert cells(below) == {(la, lo) for la in LAT for lo in (0.5, 1.5, 2.5)}
    band = build_mask(MaskSpec(condition={"file": path, "variable": "elevation", "above": 500, "below": 2500}),
                      LAT, LON).keep
    assert cells(band) == {(la, lo) for la in LAT for lo in (1.5, 2.5)}


def test_condition_on_a_finer_grid_and_intersected_with_a_shape(tmp_path):
    """A 0.5 degree field above 1500 in the east half is regridded by the fraction rule, then meets the shape."""
    fine = finer(0.5)
    field = np.tile(np.where(fine > 2.0, 2000.0, 0.0), (8, 1))
    path = write_mask(tmp_path / "dem.nc", fine, fine, field, name="elevation", lat_name="latitude",
                      lon_name="longitude")
    spec = MaskSpec(condition={"file": path, "variable": "elevation", "above": 1500}, file=None)
    assert cells(build_mask(spec, LAT, LON).keep) == {(la, lo) for la in LAT for lo in (2.5, 3.5)}
    both = MaskSpec(condition={"file": path, "variable": "elevation", "above": 1500},
                    file=str(_geojson(tmp_path, box(0.0, 0.0, 4.0, 2.0))))
    assert cells(build_mask(both, LAT, LON).keep) == {(la, lo) for la in (0.5, 1.5) for lo in (2.5, 3.5)}


def _geojson(tmp_path, shape):
    """Write a shape as GeoJSON and return its path."""
    path = tmp_path / "shape.geojson"
    path.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(shape)}))
    return path


def test_mask_variable_is_auto_detected_or_named(tmp_path):
    """One 2-D variable is picked; several need the name; an unknown name is an error with the choices."""
    one = write_mask(tmp_path / "one.nc", LAT, LON, np.ones((4, 4)), name="anything")
    assert build_mask(MaskSpec(file=one), LAT, LON).record["kept"] == 16
    two = tmp_path / "two.nc"
    xr.Dataset({"a": (("lat", "lon"), np.ones((4, 4))), "b": (("lat", "lon"), np.zeros((4, 4)))},
               coords={"lat": LAT, "lon": LON}).to_netcdf(two)
    with pytest.raises(MaskError, match="several variables"):
        build_mask(MaskSpec(file=str(two)), LAT, LON)
    assert build_mask(MaskSpec(file=str(two), variable="a"), LAT, LON).record["kept"] == 16
    with pytest.raises(MaskError, match="no variable 'c'"):
        build_mask(MaskSpec(file=str(two), variable="c"), LAT, LON)


def test_keep_value(tmp_path):
    """keep_value picks which value of the file is kept."""
    path = write_mask(tmp_path / "m.nc", LAT, LON, np.tile([1, 2, 2, 1], (4, 1)))
    assert cells(build_mask(MaskSpec(file=path, keep_value=2), LAT, LON).keep) == {
        (la, lo) for la in LAT for lo in (1.5, 2.5)}


# ---------------------------------------------------------------------------------------------- pipeline pieces

def test_region_mask_prints_once_saves_the_png_and_records(tmp_path, caplog):
    """The first use of a grid prints and saves mask_<dataset>.png; the second is silent; records carry the hash."""
    spec = {"file": str(_geojson(tmp_path, SHAPE)), "cell_rule": "centre"}
    caplog.set_level(logging.INFO)
    first = mask_mod.region_mask(LAT, LON, spec, "obs", tmp_path)
    out = caplog.text
    assert "Mask [obs]" in out and "kept cells: 4 of 16" in out
    assert (tmp_path / "mask_obs.png").stat().st_size > 1000
    again = mask_mod.region_mask(LAT, LON, spec, "obs", tmp_path)
    assert caplog.text == out and (first == again).all()
    mask_mod.region_mask(np.array([0.5, 1.5]), np.array([0.5, 1.5, 2.5]), spec, "AIFS", tmp_path)
    assert (tmp_path / "mask_AIFS.png").exists()
    records = mask_mod.records()
    assert set(records) == {"obs", "AIFS"} and records["obs"]["kept"] == 4
    assert len(records["obs"]["sources"][0]["sha256"]) == 64
    attrs = mask_mod.mask_attrs()
    assert attrs["mask_cell_rule"] == "centre" and attrs["mask_kept_cells_obs"] == "4 of 16"
    assert "sha256:" in attrs["mask_sources"]
    assert "kept 4 of 16 cells" in "\n".join(mask_mod.record_lines(records["obs"]))


def test_apply_region_mask_masks_only_gridded_variables(tmp_path):
    """Cells outside the mask become NaN in gridded variables; other variables are untouched."""
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), np.ones((2, 4, 4))), "stamp": (("time",), [1, 2])},
                    coords={"time": [0, 1], "lat": LAT, "lon": LON})
    out = mask_mod.apply_region_mask(ds, {"file": str(_geojson(tmp_path, SHAPE))}, "d")
    assert int(out["rain"].isel(time=0).notnull().sum()) == 4
    assert out["stamp"].dims == ("time",) and out["stamp"].values.tolist() == [1, 2]


def test_mask_without_cells_is_an_error(tmp_path):
    """A mask that keeps nothing of the grid says so instead of producing an all-NaN run."""
    with pytest.raises(MaskError, match="keeps no cell"):
        build_mask(MaskSpec(file=str(_geojson(tmp_path, box(10, 10, 11, 11)))), LAT, LON)


def test_a_shape_file_in_another_crs_is_reprojected(tmp_path):
    """A shapefile in a projected CRS lands in the same cells as the same shape in degrees."""
    import geopandas as gpd

    path = tmp_path / "shape.gpkg"
    gpd.GeoDataFrame(geometry=[SHAPE], crs="EPSG:4326").to_crs("EPSG:3857").to_file(path, driver="GPKG")
    assert cells(build_mask(MaskSpec(file=str(path)), LAT, LON).keep) == WANT


# ---------------------------------------------------------------------------------------------- Natural Earth

def _natural_earth_or_skip(resolution="50m"):
    """Skip the test with a reason when Natural Earth cannot be loaded (offline and not cached)."""
    try:
        mask_mod.natural_earth_records(resolution, "country")
        mask_mod.natural_earth_records(resolution, "land")
    except MaskError as exc:
        pytest.skip(f"Natural Earth {resolution} data not available offline: {str(exc)[:120]}")


CHIRPS_LAT = np.arange(3.0, 15.01, 0.25)
CHIRPS_LON = np.arange(33.0, 48.01, 0.25)


def test_country_by_any_name_and_unknown_country():
    """NAME, NAME_LONG and ADM0_A3 match in any case; an unknown name lists the closest ones."""
    _natural_earth_or_skip()
    base = mask_mod.region_mask(CHIRPS_LAT, CHIRPS_LON, {"country": "Ethiopia"}, "chirps")
    assert base[(CHIRPS_LAT == 9.0).argmax(), (CHIRPS_LON == 39.0).argmax()]        # Addis Ababa region
    assert not base[(CHIRPS_LAT == 3.0).argmax(), (CHIRPS_LON == 33.0).argmax()]    # South Sudan / Kenya corner
    for name in ("ETHIOPIA", "eth", "Ethiopia"):
        mask_mod.reset()
        assert (mask_mod.region_mask(CHIRPS_LAT, CHIRPS_LON, {"country": [name]}, "chirps") == base).all()
    with pytest.raises(MaskError, match="Ethiopia"):
        mask_mod.build_mask(MaskSpec(country=("Ethiopa",)), CHIRPS_LAT, CHIRPS_LON)


def test_several_countries_union_and_land_intersection():
    """Two countries give the union of their cells; land adds nothing to a landlocked-by-land selection."""
    _natural_earth_or_skip()
    one = mask_mod.build_mask(MaskSpec(country=("Ethiopia",)), CHIRPS_LAT, CHIRPS_LON).keep
    two = mask_mod.build_mask(MaskSpec(country=("Ethiopia", "Kenya")), CHIRPS_LAT, CHIRPS_LON).keep
    assert two.sum() > one.sum() and (two | one).sum() == two.sum() and (two & one).sum() == one.sum()
    both = mask_mod.build_mask(MaskSpec(country=("Ethiopia",), land=True), CHIRPS_LAT, CHIRPS_LON).keep
    assert (both == one).all()


def test_land_rule_changes_coastal_cells_only():
    """Cells at least 99 % on land have their centre on land, so that rule keeps a subset of the centre rule's cells."""
    _natural_earth_or_skip()
    lat, lon = np.arange(-1.0, 6.01, 0.5), np.arange(-5.0, 3.01, 0.5)      # Gulf of Guinea coast
    centre = mask_mod.build_mask(MaskSpec(land=True), lat, lon).keep
    strict = mask_mod.build_mask(MaskSpec(land=True, cell_rule="overlap_fraction", overlap_fraction=0.99), lat, lon).keep
    assert 0 < strict.sum() <= centre.sum() and not (strict & ~centre).any()
    assert not centre[0, 0]                                              # 1 degree south, 5 degrees west: the sea


def test_offline_error_is_clear(monkeypatch):
    """When Natural Earth cannot be downloaded the error says how to get it."""
    import cartopy.io.shapereader as shapereader

    def fail(**kwargs):
        raise OSError("network is unreachable")

    monkeypatch.setattr(shapereader, "natural_earth", fail)
    monkeypatch.setattr(mask_mod, "_NE_CACHE", {})
    with pytest.raises(MaskError, match="prefetch_cartopy_data"):
        mask_mod.country_geometry(("Ethiopia",), "50m")


# ---------------------------------------------------------------------------------------------- config

@pytest.fixture
def data_dirs(tmp_path):
    """A tiny observation folder and forecast folder, so a config can be resolved."""
    from tests.synthetic_data import write_forecast, write_obs

    write_obs(tmp_path / "obs")
    write_forecast(tmp_path / "fc")
    return tmp_path


def minimal(root, **extra):
    """The smallest valid config mapping over the folders of data_dirs, plus extra keys."""
    return {"region": "Ethiopia", "observations": {"path": str(root / "obs"), "variable": "precip", "units": "mm"},
            "models": [{"name": "M", "path": str(root / "fc"), "variable": "tp", "units": "m"}],
            "evaluation": {"start_date": [2020, 5, 1], "end_date": [2020, 7, 31],
                           "start_year_clim": 2000, "end_year_clim": 2010}, **extra}


def mask_errors(root, mask):
    """The ConfigSchemaError text of a config with this mask section, or '' if it is valid."""
    try:
        build_config(minimal(root, mask=mask))
    except ConfigSchemaError as exc:
        return str(exc)
    return ""


def test_mask_section_is_validated_with_help(data_dirs):
    """The mask section accepts the documented keys; mistakes name the key and suggest."""
    from momp.lib.config_schema import describe_mask_schema, describe_schema

    flat = build_config(minimal(data_dirs, mask={"land": True, "country": "Ethiopia",
                                      "condition": {"file": "/d.nc", "variable": "elevation", "above": 1500}}))
    assert flat["mask"] == {"land": True, "country": ["Ethiopia"],
                            "condition": {"file": "/d.nc", "variable": "elevation", "above": 1500}}
    assert "Did you mean \"country\"" in mask_errors(data_dirs, {"contry": "Ethiopia"})
    assert "cell_rule must be" in mask_errors(data_dirs, {"land": True, "cell_rule": "middle"})
    assert "overlap_fraction must be" in mask_errors(data_dirs, {"land": True, "overlap_fraction": 1.5})
    assert "give at least one of" in mask_errors(data_dirs, {"cell_rule": "centre"})
    assert "needs above and/or below" in mask_errors(data_dirs, {"condition": {"file": "/d.nc"}})
    assert "set file too" in mask_errors(data_dirs, {"land": True, "variable": "m"})
    assert mask_errors(data_dirs, {"land": True}) == ""
    helped = {row[0] for row in describe_schema() + describe_mask_schema()}
    assert {"mask", "mask.land", "mask.country", "mask.condition", "mask.resolution"} <= helped


def test_mask_and_nc_mask_cannot_be_combined(data_dirs):
    """The old flat nc_mask and the new mask are alternatives."""
    with pytest.raises(ConfigSchemaError, match="'mask' and 'nc_mask' are both set"):
        build_config(minimal(data_dirs, mask={"land": True}, nc_mask="/m.nc"))


def test_config_object_and_yaml_roundtrip(tmp_path):
    """Config(mask=Mask(...)) writes the mask section to YAML and reads it back."""
    from momp.config import Config, Mask

    cfg = Config(region="Ethiopia", mask=Mask(country="Ethiopia", condition={"file": "/d.nc", "variable": "e",
                                                                              "above": 1500}))
    path = tmp_path / "run.yaml"
    cfg.to_yaml(path)
    assert Config.from_yaml(path).to_mapping()["mask"] == {"country": "Ethiopia", "condition": {
        "file": "/d.nc", "variable": "e", "above": 1500}}


def test_relative_mask_paths_are_made_absolute(tmp_path, monkeypatch):
    """The loader resolves mask.file and mask.condition.file like nc_mask."""
    from momp.lib import loader

    monkeypatch.setattr(loader, "set_dir", lambda p: tmp_path / p)
    dic = {"work_dir": ".", "pkg_dir": ".", "mask": {"file": "m.nc", "condition": {"file": "d.nc", "above": 1}}}
    out = loader._resolve_paths(dic)["mask"]
    assert out["file"] == str(tmp_path / "m.nc") and out["condition"]["file"] == str(tmp_path / "d.nc")


# ---------------------------------------------------------------------------------------------- legacy path

def test_legacy_nc_mask_path_is_unchanged(tmp_path):
    """Without `mask`, region_select does what it did: box, nc_mask, land and country masks, byte for byte."""
    from momp.utils.land_mask import apply_nc_mask, mask_land, shp_mask
    from momp.utils.region import coords_fmt, region_select

    lat, lon = np.arange(6.5, 35.6, 2.0), np.arange(68.0, 91.5, 2.0)
    rng = np.random.default_rng(0)
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), rng.random((3, lat.size, lon.size)))},
                    coords={"time": [0, 1, 2], "lat": lat, "lon": lon})
    keep = (np.add.outer(lat, lon * 0) > 15).astype(float)
    mask_file = write_mask(tmp_path / "m.nc", lat, lon, keep)

    lats, latn, lonw, lone, prepared = coords_fmt(ds, region="India")
    expected = prepared.sel(lat=slice(lats, latn), lon=slice(lonw, lone))
    expected = shp_mask(mask_land(apply_nc_mask(expected, mask_file)), region="India")
    actual = region_select(ds, region="India", nc_mask=mask_file, mask=None, dataset_name="obs", dir_fig=str(tmp_path))

    xr.testing.assert_identical(actual, expected)
    expected.to_netcdf(tmp_path / "a.nc")
    actual.to_netcdf(tmp_path / "b.nc")
    assert (tmp_path / "a.nc").read_bytes() == (tmp_path / "b.nc").read_bytes()
    assert not mask_mod.records() and not list(tmp_path.glob("mask_*.png"))


def test_region_select_with_mask_replaces_the_implicit_masks_and_rejects_nc_mask(tmp_path):
    """With `mask` only the mask limits the cells; with both mask and nc_mask the run stops."""
    from momp.utils.region import region_select

    lat, lon = np.arange(6.5, 35.6, 2.0), np.arange(68.0, 91.5, 2.0)
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), np.ones((2, lat.size, lon.size)))},
                    coords={"time": [0, 1], "lat": lat, "lon": lon})
    shape = _geojson(tmp_path, box(70.0, 10.0, 80.0, 20.0))
    out = region_select(ds, region="India", nc_mask=None, mask={"file": str(shape)}, dataset_name="obs")
    kept = out["rain"].isel(time=0).notnull()
    assert float(out["rain"].isel(time=0).sum()) == int(kept.sum()) == 5 * 6      # lats 10.5..18.5, lons 70..80 (the centre on 80 is on the boundary)
    with pytest.raises(MaskError, match="both `mask` and `nc_mask`"):
        region_select(ds, region="India", nc_mask="x.nc", mask={"file": str(shape)})
