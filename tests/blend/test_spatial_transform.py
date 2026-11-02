"""Cell transforms are the legacy sparse transforms (onset_blending tests/test_cell_transforms.py and the
transform parts of test_reference.py, rebuilt on momp.spatial)."""
import numpy as np
import pandas as pd
import pytest

from _oracle import nc_utils_filters as nc_utils
from _oracle.spatial_id_utils import validate_expected_source_ids
from _oracle.sparse_transform_utils import (compile_sparse_cell_transform, sparse_observed_weighted_mean,
                                            sparse_target_support)
from momp.spatial import GridIdConvention, SparseTransform, regrid

WEIGHTS = pd.DataFrame({
    "source_id": ["10.00_70.00", "10.00_70.25", "10.25_70.00", "10.25_70.25", "10.25_70.25"],
    "target_id": ["north", "north", "south", "south", "north"],
    "weight": [0.5, 0.25, 1.0, 0.5, 0.25],
})
SPEC = {"options": {"cell_transform_enabled": True}}
POINTS = pd.DataFrame([(a, b) for a in (10.0, 10.25) for b in (70.0, 70.25)], columns=["lat", "lon"])


def legacy_table(tmp_path, table):
    path = tmp_path / "weights.csv"
    table.to_csv(path, index=False)
    return nc_utils.read_cell_transform({"options": {**SPEC["options"], "cell_transform_file": str(path)},
                                         "geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}), path


def test_read_weights_is_the_legacy_cell_transform_table(tmp_path):
    old, path = legacy_table(tmp_path, WEIGHTS)
    new = regrid.read_weights(path)
    pd.testing.assert_frame_equal(new[["source_id", "target_id", "weight"]], old[["source_id", "target_id", "weight"]])


def test_zero_weights_are_dropped(tmp_path):
    table = pd.concat([WEIGHTS, pd.DataFrame({"source_id": ["10.00_70.00"], "target_id": ["south"], "weight": [0.0]})])
    old, path = legacy_table(tmp_path, table)
    new = regrid.read_weights(path)
    assert len(new) == len(old) == 5


@pytest.mark.parametrize("bad", [
    WEIGHTS.assign(weight=[0.5, -1, 1, 1, 1]),
    WEIGHTS.assign(weight=[0.5, np.nan, 1, 1, 1]),
    pd.concat([WEIGHTS, WEIGHTS.iloc[:1]]),
    WEIGHTS.assign(weight=[0, 0, 1, 1, 0]),
    WEIGHTS.iloc[:0],
    WEIGHTS.drop(columns="weight"),
])
def test_invalid_tables_are_refused_by_both(tmp_path, bad):
    with pytest.raises(ValueError):
        legacy_table(tmp_path, bad)
    with pytest.raises(ValueError):
        regrid.read_weights(bad)


def test_regrid_tables_are_validated_like_the_legacy_ones():
    ok = pd.DataFrame({"latitude": [10.0, 10.25], "longitude": [70.0, 70.0], "adm3_name": ["a", "a"], "weight": [1, 2]})
    table = regrid.read_weights(ok)
    assert table.columns.tolist() == ["target_id", "lat", "lon", "weight"]
    with pytest.raises(ValueError, match="target_id' or 'adm3_name"):
        regrid.read_weights(ok.drop(columns="adm3_name"))
    with pytest.raises(ValueError, match="duplicate"):
        regrid.read_weights(pd.concat([ok, ok]))
    with pytest.raises(ValueError, match="disagree"):
        regrid.read_weights(ok.assign(target_id=["b", "b"]))
    with pytest.raises(ValueError, match="latitude"):
        regrid.read_weights(ok.drop(columns="longitude"))
    with pytest.raises(FileNotFoundError):
        regrid.read_weights("nowhere.csv")


def compiled():
    old = compile_sparse_cell_transform(
        WEIGHTS, source_ids=tuple(sorted(WEIGHTS["source_id"].unique())),
        target_ids=tuple(sorted(WEIGHTS["target_id"].unique())))
    new = SparseTransform.from_weights(WEIGHTS)
    return old, new


def test_compiled_matrix_and_axes_are_the_legacy_ones():
    old, new = compiled()
    assert new.source_ids == old.source_ids and new.target_ids == old.target_ids
    assert new.source_index == old.source_index
    np.testing.assert_array_equal(new.weighted_matrix.toarray(), old.weighted_matrix.toarray())


@pytest.mark.parametrize("chunk", [1, 3, 256])
def test_observed_mean_and_support_are_the_legacy_ones(chunk):
    old, new = compiled()
    rng = np.random.default_rng(4)
    values = rng.normal(size=(4, 7))
    values[rng.random(values.shape) < 0.3] = np.nan
    values[:, 0] = np.nan
    np.testing.assert_array_equal(new.observed_mean(values, chunk), sparse_observed_weighted_mean(old, values, chunk))
    present = rng.random((4, 5)) < 0.5
    np.testing.assert_array_equal(new.supported(present), sparse_target_support(old, present))
    with pytest.raises(ValueError, match="source axis"):
        new.observed_mean(values[:3])


def test_points_are_found_by_the_grid_ids_their_coordinates_make():
    _, new = compiled()
    assert new.convention is not None and new.convention.decimal_digits == 2 and new.convention.number_format == "fixed"
    positions = new.positions(POINTS["lat"], POINTS["lon"])
    assert positions.tolist() == [0, 1, 2, 3]
    other = POINTS.iloc[::-1]
    assert new.positions(other["lat"], other["lon"]).tolist() == [3, 2, 1, 0]
    extra = pd.concat([POINTS, pd.DataFrame({"lat": [11.0], "lon": [70.0]})])
    assert new.positions(extra["lat"], extra["lon"]).tolist() == [0, 1, 2, 3, -1]


def test_missing_source_points_are_the_legacy_error():
    _, new = compiled()
    short = POINTS.iloc[:3]
    with pytest.raises(ValueError, match="expected weight source IDs are absent") as new_error:
        new.positions(short["lat"], short["lon"], require_all=True)
    with pytest.raises(ValueError, match="expected weight source IDs are absent") as old_error:
        validate_expected_source_ids(short, WEIGHTS["source_id"].drop_duplicates(),
                                     convention=GridIdConvention(2, "fixed", "spec"))
    assert str(new_error.value).split(": ")[-1] == str(old_error.value).split(": ")[-1]


def test_two_points_on_one_source_are_refused():
    _, new = compiled()
    twice = pd.concat([POINTS, POINTS.iloc[:1]])
    with pytest.raises(ValueError, match="share"):
        new.positions(twice["lat"], twice["lon"])


def test_names_that_are_not_grid_ids_match_by_id():
    table = pd.DataFrame({"source_id": ["a", "b"], "target_id": ["t", "t"], "weight": [1.0, 3.0]})
    transform = SparseTransform.from_weights(table)
    assert transform.convention is None
    with pytest.raises(ValueError, match="not grid ids"):
        transform.positions([1.0], [1.0])
    positions = transform.positions_of_ids(["b", "a", "zzz"])
    assert positions.tolist() == [1, 0, -1]
    out = transform.observed_mean(transform.scatter(np.array([[2.0], [6.0], [9.0]]), positions))
    assert out.tolist() == [[(3 * 2.0 + 6.0) / 4]]


def test_a_tolerance_matches_nearby_coordinates():
    table = WEIGHTS.assign(lat=[10.0, 10.0, 10.25, 10.25, 10.25], lon=[70.0, 70.25, 70.0, 70.25, 70.25]).drop(
        columns="source_id").drop_duplicates(["lat", "lon", "target_id"])
    transform = SparseTransform.from_weights(table, [10.0, 10.25], [70.0, 70.25])
    noisy = POINTS + 1e-8
    assert transform.positions(noisy["lat"], noisy["lon"]).tolist() == [-1] * 4
    assert transform.positions(noisy["lat"], noisy["lon"], tolerance=1e-6).tolist() == [0, 1, 2, 3]


def test_sources_of_gives_each_targets_cells():
    _, new = compiled()
    north = new.target_ids.index("north")
    assert new.sources_of([north]).tolist() == [0, 1, 3]
