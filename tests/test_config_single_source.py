"""Run-config paths of the `units:` section start at the config file; Setting has one field per schema key."""
import dataclasses

from hcwf.lib import config_schema as schema
from hcwf.lib import loader
from hcwf.lib.control import init_dataclass
from hcwf.lib.convention import Case, Setting


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


def test_setting_has_a_field_for_every_schema_key_that_a_case_does_not_own():
    """A new schema key reaches the benchmark's keyword arguments without a second declaration."""
    names = {f.name for f in dataclasses.fields(Setting)}
    case_names = {f.name for f in dataclasses.fields(Case)}
    assert {f.key for f in schema.FIELDS} - case_names <= names
    assert not names & case_names   # a Setting is merged over a Case: it must not overwrite the case's own values
    assert {"cache_dir", "short_names", "lead_window"} <= names
    given = init_dataclass(Setting, {"units_mode": "regrid", "missing": "zero", "not_a_key": 1})
    assert (given.units_mode, given.missing, given.cache_dir) == ("regrid", "zero", None)
    given.update({"short_names": True, "not_a_key": 2})
    assert given.short_names is True and not hasattr(given, "not_a_key")
