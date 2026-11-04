"""hcwf.spatial.ids is the legacy spatial id code (oracle: tests/blend/_oracle/spatial_id_utils.py)."""
import numpy as np
import pandas as pd
import pytest

from _oracle import spatial_id_utils as old
from hcwf.spatial import ids as new

LAT = np.array([10.0, 10.25, 10.5, -0.25, 0.0])
LON = np.array([70.0, 70.25, 70.5, 0.5, -0.0])


def same(convention, legacy):
    return (convention.decimal_digits, convention.number_format, convention.source) == (
        legacy.decimal_digits, legacy.number_format, legacy.source)


@pytest.mark.parametrize("digits,fmt", [(0, "trimmed"), (2, "trimmed"), (2, "fixed"), (4, "fixed")])
def test_formatting_is_the_legacy_formatting(digits, fmt):
    ids = new.format_grid_ids(LAT, LON, new.GridIdConvention(digits, fmt, "spec"))
    assert ids == old.format_grid_ids(LAT, LON, old.GridIdConvention(digits, fmt, "spec"))


def test_inference_and_established_conventions_are_the_legacy_ones():
    lat, lon = np.arange(10, 12.01, 0.25), np.arange(70, 71.01, 0.25)
    pairs = np.array([(a, b) for a in lat for b in lon])
    assert same(new.infer_convention(pairs[:, 0], pairs[:, 1]), old.infer_grid_id_convention(pairs[:, 0], pairs[:, 1]))
    for fmt in ("trimmed", "fixed"):
        ids = pd.Series(new.format_grid_ids(pairs[:, 0], pairs[:, 1], new.GridIdConvention(2, fmt)))
        assert same(new.established_convention(ids), old.infer_grid_id_convention_from_ids(ids))


def test_resolve_agrees_with_the_legacy_resolution():
    ids = pd.Series(["10.00_70.00", "10.25_70.25"])
    explicit = new.GridIdConvention(2, "fixed", "spec")
    got = new.resolve(explicit, authoritative_ids=ids)
    want = old.resolve_grid_id_convention(
        spec={"geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}, authoritative_ids=ids)
    assert same(got, want)
    with pytest.raises(ValueError, match="conflicts"):
        new.resolve(new.GridIdConvention(1, "fixed", "spec"), authoritative_ids=ids)
    assert same(new.resolve(lat=LAT, lon=LON), old.resolve_grid_id_convention(lat=LAT, lon=LON))


def test_configured_convention():
    assert new.configured(None) is None
    assert same(new.configured(2, "fixed"),
                old.explicit_grid_id_convention({"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}))


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"lat": LAT, "lon": LON}),
    pd.DataFrame({"id": [" a ", "b", "c"]}),
    pd.DataFrame({"adm3_name": ["x", "y"]}),
])
def test_id_columns_are_the_legacy_ones(frame):
    pd.testing.assert_frame_equal(new.with_id_column(frame), old.ensure_spatial_id_col(frame))


@pytest.mark.parametrize("bad", [[None, "a"], ["a", ""], ["a", " "]])
def test_missing_or_empty_ids_are_refused(bad):
    with pytest.raises(ValueError, match="missing or empty"):
        new.normalize(bad)
    with pytest.raises(ValueError, match="missing or empty"):
        old.normalize_id_series(bad)


def test_one_id_for_two_coordinates_is_refused():
    frame = pd.DataFrame({"id": ["a", "a"], "lat": [1.0, 2.0], "lon": [1.0, 1.0]})
    with pytest.raises(ValueError, match="multiple coordinate pairs"):
        new.check_coordinates(frame)
    with pytest.raises(ValueError, match="multiple coordinate pairs"):
        old.validate_id_coordinate_consistency(frame)


def test_colliding_grid_ids_are_refused():
    frame = pd.DataFrame({"lat": [1.001, 1.002], "lon": [1.0, 1.0]})
    with pytest.raises(ValueError, match="represent multiple coordinates"):
        new.with_id_column(frame, convention=new.GridIdConvention(1, "fixed"))


def test_parse_grid_id():
    assert new.parse_grid_id("10.25_-70") == (10.25, -70.0)
    assert new.parse_grid_id("Abaala") is None and new.parse_grid_id(None) is None


def test_unit_id_column_prefers_id_over_adm3_name(tmp_path):
    from hcwf.onset import thresholds
    from hcwf.spatial import ids as spatial_ids
    assert spatial_ids.unit_id_column(["adm3_name", "id"]) == "id"
    assert spatial_ids.unit_id_column(["adm3_name"]) == "adm3_name"
    assert spatial_ids.unit_id_column(["lat", "lon"]) is None
    path = tmp_path / "t.csv"
    path.write_text("adm3_name,id,onset_thresh\\nname,u1,20\\n".replace("\\n", "\n"))
    assert thresholds.read_threshold_table(path)["id"].tolist() == ["u1"]
