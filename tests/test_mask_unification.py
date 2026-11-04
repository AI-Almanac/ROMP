"""The implicit land and country masks keep exactly the cells the old land_mask module kept.

tests/_oracle_land_mask.py is a verbatim copy of the old module (TEST-ONLY). Every check compares the boolean
cells (or the NaN pattern and values of masked data) of the pipeline's functions with the oracle's, on the real
regression grids (Ethiopia and India observations and models), synthetic grids (0..360 longitudes, descending
latitudes, a grid crossing 0 degrees, one-row and one-column grids) and the box of every registered region,
for every combination of land_only, shp_only, nc_mask and the polygon (core monsoon zone) mask.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import _oracle_land_mask as old
from hcwf.cache import reset_run_state
from hcwf.spatial import masks as new_mask
from hcwf.config.model import AreaSpec, Selection
from hcwf.spatial import select as new_region

DATA = Path(__file__).resolve().parents[1] / "data"
def _area(region, **values):
    """The AreaSpec of a region (no mask section) and, if given, an nc_mask."""
    return AreaSpec(region=region, mask=None, nc_mask=values.get("nc_mask"), shpfile_dir=None, polygon=False)


REGION_IDS = sorted(p.stem for p in (Path(new_region.__file__).resolve().parents[1] / "onset" / "regions").glob("*.yaml"))
# registry ids (any case) are the region names; countries match by id or title
REGION_NAMES = REGION_IDS + ["Ethiopia", "India", "Kenya"]


def _real_grids():
    out = {}
    for path in sorted(DATA.glob("*/obs.zarr")) + sorted(DATA.glob("*/models/*.zarr")):
        try:
            ds = xr.open_zarr(path)
        except Exception:  # noqa: BLE001 - data not present on this machine
            continue
        lat = next((ds[k].values for k in ("lat", "latitude", "LATITUDE") if k in ds.coords), None)
        lon = next((ds[k].values for k in ("lon", "longitude", "LONGITUDE") if k in ds.coords), None)
        if lat is not None and lon is not None:
            out[f"{path.parent.parent.name if path.parent.name == 'models' else path.parent.name}/{path.stem}"] = (np.asarray(lat, float), np.asarray(lon, float))
    return out


def _grids():
    grids = _real_grids()
    grids.update({
        "global_0_360": (np.arange(-30.0, 30.1, 2.5), np.arange(0.0, 360.0, 2.5)),
        "desc_lat": (np.arange(40.0, -20.1, -1.0), np.arange(-20.0, 110.0, 1.0)),
        "cross_zero": (np.arange(-10.0, 12.1, 0.5), np.arange(-20.0, 30.1, 0.5)),
        "one_row": (np.array([9.0]), np.arange(36.0, 42.1, 0.5)),
        "one_col": (np.arange(5.0, 14.1, 0.5), np.array([39.0])),
        "coarse_2deg": (np.arange(6.0, 40.1, 2.0), np.arange(66.0, 102.1, 2.0)),
        "coarse_4deg": (np.arange(6.0, 42.1, 4.0), np.arange(66.0, 102.1, 4.0)),
        "fine_asc_0p1": (np.arange(3.0, 15.01, 0.1), np.arange(33.0, 48.01, 0.1)),
    })
    return grids


GRIDS = _grids()


def _boxed(lat, lon, region):
    """The grid after the region's box cut, as the pipeline does it, or None when nothing is left."""
    ds = xr.Dataset({"v": (("lat", "lon"), np.zeros((len(lat), len(lon)), "float32"))},
                    coords={"lat": lat, "lon": lon})
    out = new_region.region_box(ds, region=region)
    if out.sizes["lat"] == 0 or out.sizes["lon"] == 0:
        return None
    return out["lat"].values, out["lon"].values


def _cases():
    for gname, (lat, lon) in GRIDS.items():
        for region in REGION_NAMES:
            yield pytest.param(gname, region, id=f"{gname}-{region}")


def _ones(lat, lon):
    return xr.DataArray(np.ones((len(lat), len(lon)), dtype="float32"), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})


def _field(lat, lon, seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(rng.random((len(lat), len(lon))).astype("float32"), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})


def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_grids_cover_the_regression_data_when_present():
    assert len(GRIDS) >= 8
    if (DATA / "india" / "obs.zarr").exists():
        assert GRIDS["india/obs"][1].size == 135 and GRIDS["ethiopia/obs"][0].size == 49 and "india/aurora1p5" in GRIDS


@pytest.mark.parametrize("gname,region", list(_cases()))
def test_kept_cells_are_the_old_cells(gname, region):
    boxed = _boxed(*GRIDS[gname], region)
    if boxed is None:
        pytest.skip("the region box holds no cell of this grid")
    lat, lon = boxed
    old_land = old.land_mask_for(lat, lon)
    old_shp = ~np.isnan(old.shp_mask(_ones(lat, lon), region=region).values)

    # the primitives
    _same(new_mask.land_mask_for(lat, lon), old_land)
    _same(new_mask.shp_mask(_ones(lat, lon), region=region).values, old.shp_mask(_ones(lat, lon), region=region).values)
    da = _field(lat, lon)
    _same(new_mask.mask_land(da).values, old.mask_land(da).values)
    _same(new_mask.mask_land(da, land=False).values, old.mask_land(da, land=False).values)
    _same(new_mask.create_land_sea_mask(da).values, old.create_land_sea_mask(da).values)
    _same(new_mask.create_land_sea_mask(da, as_boolean=True).values, old.create_land_sea_mask(da, as_boolean=True).values)

    # the pipeline's keep cells for every option combination
    for land_only in (True, False):
        for shp_only in (True, False):
            expect = (old_land if land_only else np.ones_like(old_land)) & (old_shp if shp_only else True)
            _same(new_region._keep_cells(lat, lon, region, land_only, shp_only), expect)

    # the lazy path (where) on data, as region_select_masks applies it
    for land_only, shp_only in ((True, True), (True, False), (False, True)):
        expect = da
        if land_only:
            expect = old.mask_land(expect)
        if shp_only:
            expect = old.shp_mask(expect, region=region)
        got = new_region.region_select_masks(da.to_dataset(name="v"), area=_area(region),
                                             select=Selection(land_only=land_only, shp_only=shp_only))["v"]
        _same(got.values, expect.values)

    # the loaded path (in place)
    loaded = da.to_dataset(name="v").load().copy(deep=True)
    got = new_region.mask_region_loaded(loaded, area=_area(region))
    expect = old.shp_mask(old.mask_land(da), region=region)
    _same(got["v"].values, expect.values)


@pytest.mark.parametrize("gname", sorted(GRIDS))
def test_netcdf_mask_and_polygon_mask_are_the_old_ones(gname, tmp_path):
    lat, lon = GRIDS[gname]
    if lat.size < 2 or lon.size < 2:
        pytest.skip("needs a 2-D grid")
    da = _field(lat, lon, seed=1)
    ds = da.to_dataset(name="v")
    keep = (np.add.outer(np.arange(len(lat)), np.arange(len(lon))) % 3 != 0).astype("int8")
    file = tmp_path / "m.nc"
    xr.Dataset({"m": (("lat", "lon"), keep)}, coords={"lat": lat, "lon": lon}).to_netcdf(file)
    _same(new_mask.apply_nc_mask(ds.copy(deep=True), file)["v"].values, old.apply_nc_mask(ds.copy(deep=True), file)["v"].values)
    got = new_region.region_select_masks(ds, area=_area("Ethiopia", nc_mask=str(file)),
                                         select=Selection(land_only=False, shp_only=False))
    _same(got["v"].values, old.apply_nc_mask(ds.copy(deep=True), file)["v"].values)
    # polygon mask: only defined on 1, 2 and 4 degree grids
    step = abs(lat[1] - lat[0])
    if min(abs(step - 1.0), abs(step - 2.0), abs(step - 4.0)) < 0.1:
        _same(new_mask.polygon_mask(da).values, old.polygon_mask(da).values)


def test_points_inside_polygon_and_shp_geometry_are_the_old_ones():
    lon, lat = np.arange(60.0, 100.0, 1.0), np.arange(5.0, 40.0, 1.0)
    px, py = [70, 90, 90, 70], [10, 10, 30, 30]
    for a, b in zip(new_mask.points_inside_polygon(px, py, lon, lat), old.points_inside_polygon(px, py, lon, lat)):
        _same(a, b)
    for name in ("Ethiopia", "India", "ethiopia", "Nowhere"):
        a, b = new_mask.get_shp(region=name), old.get_shp(region=name)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.equals(b)


def test_land_mask_for_is_cached_read_only_and_listed_for_workers():
    reset_run_state()
    lat, lon = np.arange(3.0, 15.0, 0.5), np.arange(33.0, 48.0, 0.5)
    a = new_mask.land_mask_for(lat, lon)
    assert new_mask.land_mask_for(lat.copy(), lon.copy()) is a
    assert not a.flags.writeable
    assert len(new_mask._RAW_MASKS) == 1 and len(new_mask._LAND_BOOL) == 1
