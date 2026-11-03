"""EvaluationUnits: grid, polygon and region units, subsets, and the transform onto a source grid."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from hcwf.spatial import EvaluationUnits, GridIdConvention
from hcwf.spatial import masks as mask_mod

LAT = np.array([0.5, 1.5, 2.5, 3.5])
LON = np.array([0.5, 1.5, 2.5, 3.5, 4.5])


@pytest.fixture(autouse=True)
def fresh_masks():
    mask_mod.reset()
    yield
    mask_mod.reset()


def write_shapes(tmp_path, name="admin.geojson", crs="EPSG:4326", **columns):
    frame = gpd.GeoDataFrame(
        {"code": ["A1", "B2", "C3"], "label": ["Alpha", "Beta", "Gamma"], "region": ["R", "R", "S"], **columns,
         "geometry": [box(0, 0, 2, 2), box(2, 0, 4, 2), box(0, 2, 4, 4)]}, crs="EPSG:4326").to_crs(crs)
    path = tmp_path / name
    frame.to_file(path)
    return str(path)


# -- grid units ---------------------------------------------------------------------------------------------

def test_grid_units_are_the_kept_cells_in_latitude_major_order():
    keep = np.zeros((4, 5), dtype=bool)
    keep[0, 1] = keep[2, 0] = keep[2, 4] = True
    units = EvaluationUnits.from_grid(LAT, LON, keep)
    assert units.kind == "grid" and len(units) == 3
    assert units.ids.tolist() == ["0.5_1.5", "2.5_0.5", "2.5_4.5"]
    assert units.lat.tolist() == [0.5, 2.5, 2.5] and units.lon.tolist() == [1.5, 0.5, 4.5]
    fixed = EvaluationUnits.from_grid(LAT, LON, keep, convention=GridIdConvention(2, "fixed"))
    assert fixed.ids[0] == "0.50_1.50"
    with pytest.raises(ValueError, match="No grid cell"):
        EvaluationUnits.from_grid(LAT, LON, np.zeros((4, 5), dtype=bool))


def test_grid_units_from_hcwf_mask(tmp_path):
    units = EvaluationUnits.from_mask({"file": write_shapes(tmp_path)}, LAT, LON, label="obs")
    assert len(units) == 16 and units.mask_record is not None
    path = tmp_path / "half.geojson"
    gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 4)]}, crs="EPSG:4326").to_file(path)
    kept = EvaluationUnits.from_mask({"file": str(path)}, LAT, LON)
    assert sorted(set(kept.lon)) == [0.5, 1.5] and len(kept) == 8
    assert len(EvaluationUnits.from_mask(None, LAT, LON)) == 20


def test_grid_selection_on_the_same_and_a_larger_grid():
    keep = np.zeros((4, 5), dtype=bool)
    keep[1, 2] = keep[3, 4] = True
    units = EvaluationUnits.from_grid(LAT, LON, keep)
    field = np.arange(20.0).reshape(-1, 1)
    same = units.transform(LAT, LON)
    assert same.observed_mean(field).ravel().tolist() == [7.0, 19.0]
    wider_lat, wider_lon = np.arange(-0.5, 5, 1.0), np.arange(-1.5, 7, 1.0)
    wide = units.transform(wider_lat, wider_lon)
    big = np.arange(wider_lat.size * wider_lon.size, dtype=float).reshape(-1, 1)
    expected = [big[(1 + 1) * wider_lon.size + (2 + 2), 0], big[(3 + 1) * wider_lon.size + (4 + 2), 0]]
    assert wide.observed_mean(big).ravel().tolist() == expected
    assert wide.weighted_matrix.nnz == 2
    near = units.transform(LAT + 1e-9, LON)  # inside HCWF's coordinate tolerance
    assert near.weighted_matrix.nnz == 2


def test_grid_units_on_a_finer_model_grid_are_conservative_means():
    units = EvaluationUnits.from_grid(LAT, LON)
    fine_lat, fine_lon = np.arange(0.25, 4, 0.5), np.arange(0.25, 5, 0.5)
    transform = units.transform(fine_lat, fine_lon)
    field = np.random.default_rng(0).normal(size=(fine_lat.size, fine_lon.size))
    got = transform.observed_mean(field.reshape(-1, 1)).reshape(4, 5)
    np.testing.assert_allclose(got, field.reshape(4, 2, 5, 2).mean(axis=(1, 3)))


# -- polygon units ------------------------------------------------------------------------------------------

def test_polygon_units_from_a_shapefile(tmp_path):
    units = EvaluationUnits.from_shapefile(write_shapes(tmp_path), "code", "label", "region")
    assert units.kind == "polygon" and units.ids.tolist() == ["A1", "B2", "C3"]
    assert units.column("name").tolist() == ["Alpha", "Beta", "Gamma"] and units.column("parent").tolist() == ["R", "R", "S"]
    assert units.lat.tolist() == [1.0, 1.0, 3.0] and units.lon.tolist() == [1.0, 3.0, 2.0]
    transform = units.transform(LAT, LON)
    assert transform.target_ids == ("A1", "B2", "C3")
    field = (LAT[:, None] * 0 + LON[None, :] + 10 * LAT[:, None]).reshape(-1, 1)
    got = transform.observed_mean(field).ravel()
    # A1 holds cells lat {0.5, 1.5} x lon {0.5, 1.5} wholly, so the mean of their values
    assert got[0] == pytest.approx(np.mean([10 * a + b for a in (0.5, 1.5) for b in (0.5, 1.5)]))
    assert got[2] == pytest.approx(np.mean([10 * a + b for a in (2.5, 3.5) for b in (0.5, 1.5, 2.5, 3.5)]))
    assert units.transform(LAT, LON) is transform


def test_polygon_weights_options_reach_the_transform(tmp_path):
    path = write_shapes(tmp_path)
    plain = EvaluationUnits.from_shapefile(path, "code").transform(LAT[:2], LON)
    cos = EvaluationUnits.from_shapefile(path, "code", cell_area="cos_lat").transform(LAT[:2], LON)
    a1 = plain.weighted_matrix[0].toarray().ravel()
    np.testing.assert_allclose(cos.weighted_matrix[0].toarray().ravel(),
                               a1 * np.cos(np.deg2rad(np.repeat(LAT[:2], LON.size))))


def test_shapefile_in_another_crs_is_brought_to_degrees(tmp_path):
    units = EvaluationUnits.from_shapefile(write_shapes(tmp_path, "utm.shp", crs="EPSG:3857"), "code")
    np.testing.assert_allclose(units.lat, [1.0, 1.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(units.lon, [1.0, 3.0, 2.0], atol=1e-6)


def test_shapefile_errors(tmp_path):
    path = write_shapes(tmp_path)
    with pytest.raises(ValueError, match="no id column 'nope'"):
        EvaluationUnits.from_shapefile(path, "nope")
    with pytest.raises(ValueError, match="no name column"):
        EvaluationUnits.from_shapefile(path, "code", "nope")
    with pytest.raises(ValueError, match="must be unique"):
        EvaluationUnits.from_shapefile(path, "region")
    with pytest.raises(FileNotFoundError):
        EvaluationUnits.from_shapefile(str(tmp_path / "none.shp"), "code")


def test_units_from_a_weights_csv_and_centroids_csv(tmp_path):
    pd.DataFrame({"target_id": ["u2", "u1", "u1"], "source_id": ["1_1", "1_1", "2_2"], "weight": [1.0, 1.0, 3.0]}).to_csv(
        tmp_path / "w.csv", index=False)
    pd.DataFrame({"adm3_name": ["u1", "u2"], "center_lat": [5.0, 6.0], "center_lon": [7.0, 8.0]}).to_csv(
        tmp_path / "c.csv", index=False)
    units = EvaluationUnits.from_weights(tmp_path / "w.csv", tmp_path / "c.csv")
    assert units.kind == "polygon" and units.ids.tolist() == ["u1", "u2"]
    assert units.lat.tolist() == [5.0, 6.0] and units.lon.tolist() == [7.0, 8.0]
    transform = units.transform(np.array([1.0, 2.0]), np.array([1.0, 2.0]), convention=GridIdConvention(0))
    assert transform.target_ids == ("u1", "u2") and transform.source_ids == ("1_1", "2_2")
    assert np.isnan(EvaluationUnits.from_weights(tmp_path / "w.csv").lat).all()


def test_grid_ids_as_targets_give_grid_units():
    weights = pd.DataFrame({"target_id": ["1.5_2.5", "0.5_2.5"], "lat": [1.0, 0.0], "lon": [1.0, 1.0], "weight": [1.0, 1.0]})
    units = EvaluationUnits.from_weights(weights)
    assert units.kind == "grid" and units.lat.tolist() == [0.5, 1.5] and units.lon.tolist() == [2.5, 2.5]


# -- subsets and restriction --------------------------------------------------------------------------------

def test_restricting_keeps_order_subsets_and_weights(tmp_path):
    units = EvaluationUnits.from_shapefile(write_shapes(tmp_path), "code", "label")
    for name, ids in (("training", ["B2", "C3"]), ("dissemination", ["A1", "C3"])):
        pd.DataFrame({"id": ids}).to_csv(tmp_path / f"{name}.csv", index=False)
        units = units.subset_from_csv(name, tmp_path / f"{name}.csv", "id")
    kept = units.restrict("training")
    assert kept.ids.tolist() == ["B2", "C3"] and kept.subsets["dissemination"].tolist() == [False, True]
    assert kept.column("name").tolist() == ["Beta", "Gamma"] and len(kept.geometry) == 2
    assert kept.transform(LAT, LON).target_ids == ("B2", "C3")
    assert len(units) == 3  # the original is untouched


def test_in_box_dict_and_region(tmp_path):
    units = EvaluationUnits(pd.DataFrame({"id": list("abcd"), "lat": [5.0, 9.0, 14.0, 20.0], "lon": [35.0, 40.0, 47.0, 40.0]}),
                            "polygon")
    assert units.in_box({"lat_min": 4, "lon_max": 40}).tolist() == [True, True, False, True]
    assert units.in_box({}).all()
    assert units.in_box("ethiopia").tolist() == [True, True, True, False]
    assert units.in_box("Ethiopia").tolist() == [True, True, True, False]


def test_region_box_handles_longitudes_on_the_0_360_convention():
    units = EvaluationUnits(pd.DataFrame({"id": list("ab"), "lat": [5.0, 5.0], "lon": [35.0, 100.0]}), "polygon")
    wide = EvaluationUnits(pd.DataFrame({"id": list("abc"), "lat": [5.0, 5.0, 5.0], "lon": [35.0, 100.0, 359.0]}), "polygon")
    assert units.in_box("ethiopia").tolist() == [True, False]
    assert wide.in_box("ethiopia").tolist() == [True, False, False]


# -- regions ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("weights", ["cos_lat", "equal"])
def test_a_region_is_a_one_row_transform_of_the_weighted_cell_mean(weights):
    """A region's one-row transform gives the (cos-lat or equal) weighted mean of its cells' present values."""
    rng = np.random.default_rng(7)
    lat = np.array([0.0, 20.0, 40.0, 60.0])
    lon = np.array([1.0, 2.0, 3.0])
    shape = (4, 3, 40)
    tenths = rng.integers(0, 400, shape).astype(float)
    missing = rng.random(shape) < 0.3
    missing[:, :, 5] = True
    keep = rng.random((4, 3)) < 0.7
    keep[0, 0] = True

    cell_weight = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 3)) if weights == "cos_lat" else np.ones((4, 3))
    w = (cell_weight * keep)[:, :, None] * ~missing
    with np.errstate(invalid="ignore"):
        expected = (w * tenths).sum(axis=(0, 1)) / w.sum(axis=(0, 1))

    region = EvaluationUnits.from_region(keep, lat, lon, weights=weights)
    transform = region.transform(lat, lon)
    values = np.where(missing, np.nan, tenths).reshape(-1, shape[-1])
    mean = transform.observed_mean(values)[0]
    np.testing.assert_array_equal(np.isnan(mean), np.isnan(expected))
    np.testing.assert_allclose(mean[~np.isnan(mean)], expected[~np.isnan(expected)], rtol=1e-9)


def test_a_region_on_another_grid_is_the_conservative_mean_of_its_cells():
    keep = np.zeros((4, 5), dtype=bool)
    keep[:2, :2] = True
    region = EvaluationUnits.from_region(keep, LAT, LON, id="box", weights="equal")
    assert region.kind == "region" and region.ids.tolist() == ["box"]
    assert region.lat.tolist() == [1.0] and region.lon.tolist() == [1.0]
    fine_lat, fine_lon = np.arange(0.25, 4, 0.5), np.arange(0.25, 5, 0.5)
    field = np.random.default_rng(1).normal(size=(fine_lat.size, fine_lon.size))
    got = region.transform(fine_lat, fine_lon).observed_mean(field.reshape(-1, 1))[0, 0]
    assert got == pytest.approx(field[:4, :4].mean())
    with pytest.raises(ValueError, match="selects no cells"):
        EvaluationUnits.from_region(np.zeros((4, 5), dtype=bool), LAT, LON)
    with pytest.raises(ValueError, match="weights must be"):
        EvaluationUnits.from_region(keep, LAT, LON, weights="bogus")


def test_unit_ids_must_be_unique():
    with pytest.raises(ValueError, match="must be unique"):
        EvaluationUnits(pd.DataFrame({"id": ["a", "a"]}), "polygon")
    with pytest.raises(ValueError, match="kind"):
        EvaluationUnits(pd.DataFrame({"id": ["a"]}), "blob")
