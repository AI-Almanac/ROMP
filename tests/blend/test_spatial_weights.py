"""Weights built from shapes and grids: analytic areas, the legacy overlay table, conservation."""
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon, box

from _oracle import geometry_utils as legacy
from hcwf.spatial import SparseTransform, regrid
from hcwf.spatial.masks import cell_shares, rasterise

LAT = np.array([0.5, 1.5, 2.5, 3.5])
LON = np.array([0.5, 1.5, 2.5, 3.5, 4.5])


def table(geometries, ids, lat=LAT, lon=LON, **options):
    return regrid.polygon_weights(geometries, ids, lat, lon, **options)


def weight(frame, target, lat, lon):
    row = frame[(frame["target_id"] == target) & (frame["lat"] == lat) & (frame["lon"] == lon)]
    return float(row["weight"].iloc[0]) if len(row) else 0.0


def test_a_rectangle_overlaps_cells_by_hand():
    """Lon 0.7 to 2.3 (0.3, 1, 0.3 of the three cells' width), lat 0.25 to 1.5 (0.75 and 0.25 of two rows)."""
    frame = table([box(0.7, 0.25, 2.3, 1.5)], ["a"])
    expected = {(0.5, 0.5): 0.3 * 0.75, (0.5, 1.5): 1.0 * 0.75, (0.5, 2.5): 0.3 * 0.75,
                (1.5, 0.5): 0.3 * 0.5, (1.5, 1.5): 1.0 * 0.5, (1.5, 2.5): 0.3 * 0.5}
    assert len(frame) == len(expected)
    for (la, lo), area in expected.items():
        assert weight(frame, "a", la, lo) == pytest.approx(area)
    assert frame["weight"].sum() == pytest.approx(box(0.7, 0.25, 2.3, 1.5).area)


def test_shares_and_cos_lat_variants():
    shape = box(0.7, 0.25, 2.3, 1.5)
    area = table([shape], ["a"])
    share = table([shape], ["a"], measure="share")
    np.testing.assert_allclose(share["weight"], area["weight"])  # unit cells: share equals area
    cos = table([shape], ["a"], cell_area="cos_lat")
    np.testing.assert_allclose(cos["weight"], area["weight"] * np.cos(np.deg2rad(area["lat"])))
    coarse = table([shape], ["a"], lat=np.array([1.0, 3.0]), lon=np.array([1.0, 3.0]), measure="share")
    assert weight(coarse, "a", 1.0, 1.0) == pytest.approx(shape.intersection(box(0, 0, 2, 2)).area / 4)


def test_the_weights_hold_the_cells_values_in_the_table_exactly_as_given():
    """Unsorted axes and a 0..360 grid keep the grid's own coordinates in the table."""
    lat, lon = LAT[::-1], np.array([358.5, 359.5, 0.5, 1.5])
    shape = box(-1.5, 0.0, 0.5, 2.0)
    frame = table([shape], ["a"], lat=lat, lon=lon)
    assert set(frame["lon"]) == {358.5, 359.5, 0.5}
    assert weight(frame, "a", 1.5, 359.5) == pytest.approx(1.0)
    assert weight(frame, "a", 1.5, 358.5) == pytest.approx(0.5)
    assert weight(frame, "a", 1.5, 0.5) == pytest.approx(0.5)


def test_a_polygon_clipped_to_the_coverage():
    keep = np.ones((4, 5), dtype=bool)
    keep[:, 2:] = False
    coverage = regrid.coverage_geometry(LAT, LON, keep)
    assert coverage.area == pytest.approx(8.0)
    frame = table([box(0, 0, 4, 4)], ["a"], clip_to=coverage)
    assert set(frame["lon"]) == {0.5, 1.5} and frame["weight"].sum() == pytest.approx(8.0)


def test_polygons_with_no_overlap_are_left_out_and_none_at_all_is_an_error():
    frame = table([box(0, 0, 2, 2), box(10, 10, 11, 11)], ["in", "out"])
    assert set(frame["target_id"]) == {"in"}
    with pytest.raises(ValueError, match="No polygon overlaps"):
        table([box(10, 10, 11, 11)], ["out"])


def test_cell_shares_are_what_rasterise_thresholds():
    triangle = Polygon([(0, 0), (4, 0), (0, 3)])
    rows, cols, share = cell_shares(triangle, LAT, LON)
    full = np.zeros((LAT.size, LON.size))
    full[np.ix_(rows, cols)] = share
    for fraction in (0.1, 0.5, 0.9):
        np.testing.assert_array_equal(full >= fraction - 1e-9, rasterise(triangle, LAT, LON, "overlap_fraction", fraction))


def test_weights_equal_the_legacy_overlay_table(tmp_path):
    """The legacy builder (geopandas overlay, share of the cell, shares above 1e-5) gives the same table."""
    import geopandas as gpd

    lat, lon = np.arange(10.125, 12, 0.25), np.arange(70.125, 72, 0.25)
    xr.Dataset(coords={"lat": lat, "lon": lon}).to_netcdf(tmp_path / "grid.nc")
    shapes = [Polygon([(70.3, 10.2), (71.7, 10.4), (71.1, 11.8)]), box(70.0, 11.0, 72.0, 12.0).difference(box(70.5, 11.4, 71, 11.6))]
    admin = gpd.GeoDataFrame({"adm3_name": ["tri", "ring"], "geometry": shapes}, crs="EPSG:4326")
    old = legacy.build_grid_to_admin_weights(str(tmp_path / "grid.nc"), {"shapefile": None}, admin_gdf=admin)

    new = table(shapes, ["tri", "ring"], lat=lat, lon=lon, measure="share", min_share=1e-5)
    old = old.rename(columns={"adm3_name": "target_id"}).sort_values(["target_id", "latitude", "longitude"]).reset_index(drop=True)
    new = new.rename(columns={"lat": "latitude", "lon": "longitude"}).sort_values(
        ["target_id", "latitude", "longitude"]).reset_index(drop=True)
    assert len(new) == len(old) > 10
    np.testing.assert_allclose(new["latitude"], old["latitude"])
    np.testing.assert_allclose(new["longitude"], old["longitude"])
    np.testing.assert_allclose(new["weight"], old["weight"], rtol=1e-9)


def test_the_polygon_mean_of_a_field_is_the_area_weighted_mean():
    shape = box(0.7, 0.25, 2.3, 1.5)
    transform = SparseTransform.from_weights(table([shape], ["a"]), LAT, LON)
    field = (LAT[:, None] * 10 + LON[None, :]).reshape(-1, 1)
    area = table([shape], ["a"])
    expected = np.average(area["lat"] * 10 + area["lon"], weights=area["weight"])
    assert transform.observed_mean(field)[0, 0] == pytest.approx(expected)
    field[(LAT[:, None] == 0.5).repeat(5, 1).reshape(-1)] = np.nan  # a missing row is left out
    sub = area[area["lat"] != 0.5]
    assert transform.observed_mean(field)[0, 0] == pytest.approx(np.average(sub["lat"] * 10 + sub["lon"], weights=sub["weight"]))


# -- grid to grid -------------------------------------------------------------------------------------------

def test_conservative_weights_from_fine_to_coarse_are_block_means():
    fine_lat, fine_lon = np.arange(0.25, 4, 0.5), np.arange(0.25, 6, 0.5)
    matrix = regrid.grid_weights(fine_lat, fine_lon, LAT[:3], LON[:5])
    assert matrix.shape == (15, fine_lat.size * fine_lon.size)
    np.testing.assert_allclose(matrix.sum(axis=1).A.ravel(), 1.0)  # a coarse cell is 1 x 1 deg
    rng = np.random.default_rng(0)
    field = rng.normal(size=(fine_lat.size, fine_lon.size))
    got = (matrix @ field.reshape(-1)).reshape(3, 5)
    expected = field[:6, :10].reshape(3, 2, 5, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(got, expected)


def test_conservative_weights_from_coarse_to_fine_and_offset_grids():
    fine_lat, fine_lon = np.arange(0.25, 4, 0.5), np.arange(0.25, 5, 0.5)
    matrix = regrid.grid_weights(LAT, LON, fine_lat, fine_lon)
    field = np.arange(20.0).reshape(4, 5)
    np.testing.assert_allclose(matrix @ field.reshape(-1), np.repeat(np.repeat(field, 2, 0), 2, 1).reshape(-1) * 0.25)
    # a target cell straddling two source cells takes their area-weighted mean
    shifted = regrid.grid_weights(LAT, LON, np.array([1.0]), np.array([1.5]))
    mean = (shifted @ field.reshape(-1))[0] / shifted.sum()
    assert mean == pytest.approx((field[0, 1] + field[1, 1]) / 2)


def test_grid_weights_follow_the_sources_through_longitude_wraps_and_orders():
    lat = np.array([1.0, 0.0])  # descending
    src_lon = np.array([359.0, 0.0, 1.0])
    dst_lon = np.array([-1.0, 0.0, 1.0])
    matrix = regrid.grid_weights(lat, src_lon, lat, dst_lon)
    np.testing.assert_allclose(matrix.toarray(), np.eye(6))
    cos = regrid.grid_weights(lat, src_lon, lat, dst_lon, cell_area="cos_lat")
    np.testing.assert_allclose(cos.diagonal(), np.repeat(np.cos(np.deg2rad(lat)), 3))
    with pytest.raises(ValueError, match="cell_area"):
        regrid.grid_weights(lat, src_lon, lat, dst_lon, cell_area="bogus")


def test_weights_are_read_exactly_as_written(tmp_path):
    """Weights round-trip through CSV to the last bit (pandas' default float parser does not always)."""
    import numpy as np
    import pandas as pd
    from hcwf.spatial.regrid import read_weights

    rng = np.random.default_rng(3)
    weights = rng.random(5000) / 7.0
    table = pd.DataFrame({"target_id": [f"u{i % 50}" for i in range(5000)], "weight": weights,
                          "source_id": [f"s{i}" for i in range(5000)]})
    path = tmp_path / "w.csv"
    table.to_csv(path, index=False, float_format="%.17g")
    read = read_weights(str(path))
    np.testing.assert_array_equal(np.sort(read["weight"].to_numpy()), np.sort(weights))
