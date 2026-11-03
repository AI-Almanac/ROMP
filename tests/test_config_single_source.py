"""Run-config paths of the `units:` section start at the config file; the settings of a case have one key per schema key."""

from hcwf.config import schema
from hcwf.config import build as loader
from flat_values import CASE_FIELDS, _setting_part


def _dic(**values):
    return dict(values)


def test_relative_units_files_start_at_the_folder_of_the_config(tmp_path, monkeypatch):
    config = tmp_path / "runs" / "run.yaml"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    dic = loader._resolve_paths(_dic(
        config_file=str(config), units_shapefile="geo/units.shp", units_centroids="../c.csv",
        units_dissemination="d.csv", units_outline=str(tmp_path / "abs.geojson"),
        units_weights={"obs": "w/obs.csv", "*": "w/all.csv"}))
    assert dic["units_shapefile"] == str(tmp_path / "runs" / "geo" / "units.shp")
    assert dic["units_centroids"] == str(tmp_path / "c.csv")
    assert dic["units_dissemination"] == str(tmp_path / "runs" / "d.csv")
    assert dic["units_outline"] == str(tmp_path / "abs.geojson")
    assert dic["units_weights"] == {"obs": str(tmp_path / "runs" / "w" / "obs.csv"),
                                    "*": str(tmp_path / "runs" / "w" / "all.csv")}


def test_units_files_are_left_alone_without_a_config_file_or_a_value(tmp_path):
    dic = loader._resolve_paths(_dic(units_shapefile="units.shp", units_weights=None))
    assert dic["units_shapefile"] == "units.shp" and dic["units_weights"] is None


def test_setting_has_a_key_for_every_schema_key_that_a_case_does_not_own():
    """A new schema key reaches the benchmark's keyword arguments without a second declaration."""
    names = set(_setting_part({}))
    assert {f.key for f in schema.FIELDS} - set(CASE_FIELDS) <= names
    assert not names & set(CASE_FIELDS)   # the case's own values must not be overwritten by a setting
    assert {"cache_dir", "short_names", "lead_window"} <= names
    given = _setting_part({"units_mode": "regrid", "missing": "zero", "not_a_key": 1})
    assert (given["units_mode"], given["missing"], given["cache_dir"]) == ("regrid", "zero", None)
    assert "not_a_key" not in given
