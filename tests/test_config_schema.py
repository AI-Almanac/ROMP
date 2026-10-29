"""The YAML run config: schema errors, flattening, normalisation, defaults and the Config objects."""

import pytest

from momp.config import Config, Dataset, Evaluation, Onset
from momp.lib.config_schema import ConfigSchemaError, FIELDS, describe_schema
from momp.lib.config_yaml import build_config
from tests.synthetic_data import write_forecast, write_obs


@pytest.fixture
def data(tmp_path):
    """Synthetic observation and forecast folders."""
    write_obs(tmp_path / "obs")
    write_forecast(tmp_path / "fc")
    return tmp_path


def _raw(data, **over):
    """A complete nested run config over the synthetic data."""
    raw = {"region": "ethiopia",
           "observations": {"path": str(data / "obs"), "units": "mm", "day_boundary": "00 UTC, labelled by start"},
           "models": [{"name": "M", "path": str(data / "fc"), "units": "m", "lead_label": "start"}],
           "evaluation": {"start_date": [2020, 5, 1], "end_date": "2020-07-31"}}
    raw.update(over)
    return raw


def test_sections_flatten_to_the_flat_keys_with_tuples(data):
    """Nested keys become the flat keys; lists become tuples; dates become (y, m, d)."""
    cfg = build_config(_raw(data, evaluation={"start_date": [2020, 5, 1], "end_date": "2020-07-31",
                                              "verification_windows": [[1, 2], [3, 3]], "init_days": [0, 3]}))
    assert cfg["start_date"] == (2020, 5, 1) and cfg["end_date"] == (2020, 7, 31)
    assert cfg["verification_window_list"] == ((1, 2), (3, 3))
    assert cfg["init_days"] == (0, 3)
    assert cfg["model_list"] == ("M",) and cfg["unit_cvt_list"] == (1000,) and cfg["unit_cvt_list"] != (None,)
    assert cfg["obs_unit_cvt"] is None and cfg["obs_file_pattern"] == ("{}.nc",)
    assert cfg["obs_var"] == "precip" and cfg["model_var_list"] == ("tp",)


def test_defaults_are_filled_and_region_values_are_inherited(data):
    """Schema defaults appear; climatology years come from the region; the boundary stays the region's (None)."""
    cfg = build_config(_raw(data))
    assert cfg["max_forecast_day"] == 30 and cfg["onset_backend"] == "numba" and cfg["climatology_cv"] == "leave_one_year_out"
    assert (cfg["start_year_clim"], cfg["end_year_clim"]) == (1998, 2024)
    assert cfg["date_filter_year"] == 2020
    assert cfg["ref_model"] == "climatology" and cfg["ref_model_var"] == "precip"
    assert "onset_definition" not in cfg


def test_unknown_keys_get_suggestions_and_all_errors_are_listed(data):
    """A misspelt section key and a misspelt top-level key are both reported, with did-you-mean."""
    raw = _raw(data, evaluaton={"x": 1})
    raw["evaluation"]["max_forcast_day"] = 10
    with pytest.raises(ConfigSchemaError) as err:
        build_config(raw)
    text = str(err.value)
    assert 'unknown key \'evaluaton\'. Did you mean "evaluation"?' in text
    assert "evaluation.max_forcast_day" in text and 'Did you mean "max_forecast_day"?' in text


def test_type_errors_name_the_key_and_the_expected_form(data):
    """Wrong types say which key and what was expected."""
    raw = _raw(data)
    raw["evaluation"].update(start_date="May 1", init_days="monday", max_forecast_day="thirty")
    raw["onset"] = {"backend": "cuda"}
    with pytest.raises(ConfigSchemaError) as err:
        build_config(raw)
    text = str(err.value)
    assert "'evaluation.start_date': expected a date" in text
    assert "'evaluation.init_days': expected a list of whole numbers" in text
    assert "'evaluation.max_forecast_day': expected a whole number" in text
    assert "'onset.backend': expected one of 'numba', 'reference', 'gpu'" in text


def test_a_key_written_twice_is_an_error(data):
    """The same key given flat and nested is rejected."""
    with pytest.raises(ConfigSchemaError, match="sets obs_dir again"):
        build_config(_raw(data, obs_dir="/x"))


def test_explicit_values_override_a_dataset_file_entry(data):
    """models referencing a dataset-file entry take its values; inline keys win; the source is printed."""
    (data / "ds.yaml").write_text(
        f"id: mine\nkind: forecast\npath: {data / 'fc'}\nvariable: tp\nunits: mm\nlead_label: end\nname: FROMFILE\n")
    raw = _raw(data, dataset_files=[str(data / "ds.yaml")],
               models=[{"dataset": "mine", "name": "OVER", "units": "m", "lead_label": "start"}])
    cfg = build_config(raw)
    assert cfg["model_list"] == ("OVER",) and cfg["unit_cvt_list"] == (1000,)
    assert any("dataset 'mine'" in line and line.startswith("  variable") for line in cfg["dataset_report"])


def test_models_disagreeing_on_lead_label_is_an_error(data):
    """Two models with different lead labels cannot share one run."""
    models = [{"name": "A", "path": str(data / "fc"), "units": "m", "lead_label": "start"},
              {"name": "B", "path": str(data / "fc"), "units": "m", "lead_label": "end"}]
    with pytest.raises(ConfigSchemaError, match="disagree on the lead label"):
        build_config(_raw(data, models=models))


def test_the_report_names_every_detected_value(data):
    """Left-out dataset keys are detected and the dataset_report says so."""
    raw = _raw(data)
    del raw["models"][0]["lead_label"]
    cfg = build_config(raw)
    assert cfg["model_lead_label"] in (None, "start")
    text = "\n".join(cfg["dataset_report"])
    assert "[detected:" in text and "lead_label: start" in text


def test_config_objects_round_trip_through_yaml_and_flatten(data, tmp_path):
    """A Config built in Python writes YAML that reads back to the same flat config."""
    cfg = Config(region="ethiopia",
                 observations=Dataset(path=str(data / "obs"), units="mm", day_boundary="00 UTC, labelled by start"),
                 models=[Dataset(name="M", path=str(data / "fc"), units="m", lead_label="start")],
                 onset=Onset(definition="icpac", override={"wet_day": "at least 2 mm"}),
                 evaluation=Evaluation(start_date=(2020, 5, 1), end_date=(2020, 7, 31)))
    path = tmp_path / "run.yaml"
    cfg.to_yaml(path)
    flat_a, flat_b = cfg.to_flat(), Config.from_yaml(path).to_flat()
    for key in ("model_list", "obs_dir", "start_date", "onset_definition", "onset_override", "region"):
        assert flat_a[key] == flat_b[key]
    assert flat_a["onset_override"] == {"wet_day": "at least 2 mm"}


def test_every_schema_key_has_help_and_a_unique_flat_key():
    """The schema is documented: help text everywhere, no duplicate keys."""
    keys = [f.key for f in FIELDS]
    assert len(keys) == len(set(keys)) and all(f.help for f in FIELDS)
    assert len(describe_schema()) == len(FIELDS)
