"""Area, dissemination and regrid filters are the legacy nc_utils / sparse transforms' (onset_blending
tests/test_domain.py, rebuilt on hcwf.spatial)."""
import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from _oracle import nc_utils_filters as nc_utils
from _oracle.geometry_utils import normalize_regrid_weights
from _oracle.sparse_transform_utils import compile_sparse_cell_transform, sparse_observed_weighted_mean
from hcwf.spatial import EvaluationUnits, GridIdConvention, regrid
from hcwf.errors import AreaError

CONVENTION = GridIdConvention(2, "fixed", "spec")
GEOMETRY = {"geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}
LATS, LONS = np.arange(10.0, 12.01, 0.25), np.arange(70.0, 72.01, 0.25)


def grid():
    return pd.DataFrame([(a, b) for a in LATS for b in LONS], columns=["lat", "lon"])


def units():
    return EvaluationUnits.from_grid(LATS, LONS, convention=CONVENTION)


def test_a_box_keeps_the_legacy_filters_rows():
    box_ = {"lat_min": 10.4, "lat_max": 11.6, "lon_max": 71.3}
    cells = units()
    kept = cells.restrict(cells.in_area(box=box_))

    old = nc_utils.filter_by_dissemination_cells(grid(), {"filter": {"bbox": box_}, **GEOMETRY})
    assert kept.ids.tolist() == old["id"].tolist()
    np.testing.assert_array_equal(kept.lat, old["lat"].to_numpy())
    np.testing.assert_array_equal(kept.lon, old["lon"].to_numpy())


def test_dissemination_units_keep_the_legacy_rows(tmp_path):
    sample = grid().sample(20, random_state=1)
    pd.DataFrame({"cell": [f"{a:.2f}_{b:.2f}" for a, b in zip(sample["lat"], sample["lon"])]}).to_csv(
        tmp_path / "cells.csv", index=False)
    cells = units().subset_from_csv("dissemination", tmp_path / "cells.csv", "cell", explicit=CONVENTION)

    old = nc_utils.filter_by_dissemination_cells(
        grid(), {"filter": {"dissemination_cells_file": str(tmp_path / "cells.csv"), "id_col": "cell"}, **GEOMETRY})
    assert cells.restrict("dissemination").ids.tolist() == old["id"].tolist()
    assert len(old) == 20 and cells.subsets["dissemination"].sum() == 20


@pytest.mark.parametrize("columns", ["lat_lon", "first_column"])
def test_dissemination_lists_without_an_id_column(tmp_path, columns):
    sample = grid().sample(7, random_state=3)
    if columns == "lat_lon":
        sample.to_csv(tmp_path / "cells.csv", index=False)
    else:
        pd.DataFrame({"anything": [f"{a:.2f}_{b:.2f}" for a, b in zip(sample["lat"], sample["lon"])]}).to_csv(
            tmp_path / "cells.csv", index=False)
    cells = units().subset_from_csv("dissemination", tmp_path / "cells.csv", explicit=CONVENTION)
    old = nc_utils.filter_by_dissemination_cells(
        grid(), {"filter": {"dissemination_cells_file": str(tmp_path / "cells.csv")}, **GEOMETRY})
    assert cells.restrict("dissemination").ids.tolist() == old["id"].tolist()


def test_a_subset_can_be_named_by_ids_and_refuses_unknown_ones():
    cells = units().with_subset("training", ["10.00_70.00", "10.25_70.25"])
    assert cells.restrict("training").ids.tolist() == ["10.00_70.00", "10.25_70.25"]
    with pytest.raises(ValueError, match="not units"):
        cells.with_subset("x", ["nowhere"])


def named_units(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "lat": [10.5, 11.5, 13.0], "lon": [70.5, 71.5, 70.0]}).to_csv(
        tmp_path / "centroids.csv", index=False)
    weights = pd.DataFrame({"target_id": ["a", "b", "c"], "lat": [10.5, 11.5, 13.0], "lon": [70.5, 71.5, 70.0],
                            "weight": 1.0})
    return EvaluationUnits.from_weights(weights, tmp_path / "centroids.csv")


def test_outlines_locate_named_units_by_their_centroids(tmp_path):
    square = {"type": "Polygon", "coordinates": [[[70, 10], [72, 10], [72, 12], [70, 12], [70, 10]]]}
    (tmp_path / "area.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": square}]}))
    cells = named_units(tmp_path)

    kept = cells.restrict(cells.in_area(outline=tmp_path / "area.geojson"))
    assert kept.ids.tolist() == ["a", "b"]
    with pytest.raises(AreaError, match="No cells fall inside"):
        cells.restrict(cells.ids == "c").in_area(outline=tmp_path / "area.geojson")


def test_outline_boundary_semantics():
    """contains drops a centroid on the edge (onset_blending), intersects keeps it (HCWF's centre rule)."""
    cells = EvaluationUnits(pd.DataFrame({"id": ["edge", "in"], "lat": [10.0, 10.5], "lon": [70.5, 70.5]}), "polygon")
    outline = box(70, 10, 72, 12)
    assert cells.in_outline(outline).tolist() == [False, True]
    assert cells.in_outline(outline, "intersects").tolist() == [True, True]


def test_bounds_need_centroids():
    cells = EvaluationUnits(pd.DataFrame({"id": ["a", "b"]}), "polygon")
    with pytest.raises(ValueError, match="no centroids"):
        cells.in_box({"lat_min": 0})
    assert units().in_area().all()


def test_regridding_is_the_legacy_sparse_mean():
    rng = np.random.default_rng(0)
    cells = grid()
    weights = cells.sample(40, random_state=0).assign(
        adm3_name=lambda f: rng.choice(["x", "y", "z"], len(f)), weight=lambda f: rng.uniform(0.1, 1, len(f)))
    table = weights.rename(columns={"lat": "latitude", "lon": "longitude"})
    values = rng.normal(size=(len(cells), 6))
    values[rng.random(values.shape) < 0.2] = np.nan

    transform = EvaluationUnits.from_weights(table).transform(*regrid.grid_axes(cells))

    mapping = normalize_regrid_weights(table).rename(columns={"latitude": "lat", "longitude": "lon"})
    mapping["source_id"] = list(zip(mapping["lat"].astype(float), mapping["lon"].astype(float)))
    sources = tuple((float(a), float(b)) for a in LATS for b in LONS)
    legacy = compile_sparse_cell_transform(mapping, source_ids=sources, target_ids=("x", "y", "z"))
    np.testing.assert_array_equal(transform.observed_mean(values), sparse_observed_weighted_mean(legacy, values))
    assert transform.target_ids == ("x", "y", "z") and transform.source_ids == sources
    np.testing.assert_array_equal(transform.weighted_matrix.toarray(), legacy.weighted_matrix.toarray())


def test_weights_keyed_by_coordinates_or_by_ids_are_one_transform():
    """blending's regrid (float lat/lon) and cell_transform (string ids) are two spellings of one transform."""
    rng = np.random.default_rng(1)
    chosen = grid().sample(30, random_state=2).assign(
        target_id=lambda f: rng.choice(["x", "y"], len(f)), weight=lambda f: rng.uniform(0.1, 1, len(f)))
    by_ids = chosen.assign(source_id=[f"{a:.2f}_{b:.2f}" for a, b in zip(chosen["lat"], chosen["lon"])]).drop(
        columns=["lat", "lon"])
    values = rng.normal(size=(len(grid()), 4))
    values[rng.random(values.shape) < 0.2] = np.nan

    by_coordinates = EvaluationUnits.from_weights(chosen).transform(LATS, LONS)
    id_transform = EvaluationUnits.from_weights(by_ids).transform(LATS, LONS, convention=CONVENTION)
    positions = id_transform.positions(grid()["lat"], grid()["lon"])
    got = id_transform.observed_mean(id_transform.scatter(values, positions))

    np.testing.assert_allclose(got, by_coordinates.observed_mean(values), rtol=1e-12, equal_nan=True)


def test_coordinate_weights_match_exactly_unless_a_tolerance_is_given():
    weights = pd.DataFrame({"target_id": ["x", "x"], "lat": [10.0, 10.25 + 1e-9], "lon": [70.0, 70.0], "weight": [1.0, 1.0]})
    cells = EvaluationUnits.from_weights(weights)
    with pytest.raises(ValueError, match="outside the compiled axes"):
        cells.transform(LATS, LONS)
    cells = EvaluationUnits.from_weights(weights)
    transform = cells.transform(LATS, LONS, tolerance=1e-6)
    assert transform.weighted_matrix.nnz == 2


def test_transforms_are_compiled_once_per_grid():
    cells = units()
    first = cells.transform(LATS, LONS)
    assert cells.transform(LATS, LONS) is first
    assert cells.transform(LATS[:-1], LONS) is not first
