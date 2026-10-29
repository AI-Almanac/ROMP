"""Dataset fields are specified by the user or detected from the files, and detection is never silent."""

import pytest

from momp.lib import dataset_detect as detect
from momp.lib.dataset_schema import DatasetError, TEMPLATES, expand_path, get_dataset, load_datasets, parse_dataset
from tests.synthetic_data import write_forecast, write_obs


def _obs(tmp_path, **kw):
    """Write an observation folder and return its path text."""
    write_obs(tmp_path / "obs", **kw)
    return str(tmp_path / "obs")


def test_observation_fields_are_detected_with_evidence(tmp_path):
    """file_pattern, variable, units and dims are found and each line says how."""
    res = detect.resolve_entry({"path": _obs(tmp_path, name="data_2020.nc")}, "observations")
    assert res.values["file_pattern"] == "data_{}.nc"
    assert res.values["variable"] == "precip"
    assert res.values["units"] == "mm"
    assert res.values["dims"] == {"time": "time", "lat": "lat", "lon": "lon"}
    text = res.report()
    assert any(line.startswith("variable: precip  [detected: only gridded variable in data_2020.nc]") for line in text)
    assert any("'units' attribute" in line for line in text)
    assert "day_boundary" in res.problems


def test_specified_values_win_and_are_reported_as_specified(tmp_path):
    """User values are kept and labelled."""
    res = detect.resolve_entry({"path": _obs(tmp_path), "variable": "precip", "file_pattern": "{}.nc", "units": "mm",
                                "day_boundary": "00 UTC, labelled by start"}, "observations")
    assert not res.problems
    assert "variable: precip  [specified]" in res.report()


def test_day_boundary_is_read_only_from_an_explicit_attribute(tmp_path):
    """A day_boundary attribute is used; without it the user must say."""
    res = detect.resolve_entry({"path": _obs(tmp_path, boundary_attr="03 UTC, labelled by end")}, "observations")
    assert res.values["day_boundary"] == "03 UTC, labelled by end"
    assert "attribute" in "\n".join(res.report())


def test_several_candidate_variables_is_an_error_naming_the_key(tmp_path):
    """Two gridded variables, neither precipitation-like, cannot be told apart."""
    path = _obs(tmp_path, var="a", extra_vars=("b",))
    res = detect.resolve_entry({"path": path}, "observations")
    assert "a, b" in res.problems["variable"] and "Fix: set variable" in res.problems["variable"]


def test_a_precipitation_like_name_breaks_the_tie(tmp_path):
    """Among several gridded variables the only precipitation-like one is chosen, and the evidence says so."""
    res = detect.resolve_entry({"path": _obs(tmp_path, var="tp", extra_vars=("t2m",))}, "observations")
    assert res.values["variable"] == "tp"
    assert "only precipitation-like variable" in "\n".join(res.report())


def test_missing_units_attribute_is_an_error_with_a_hint(tmp_path):
    """No units attribute: the key must be set, and the message guesses from the values."""
    res = detect.resolve_entry({"path": _obs(tmp_path, units=None)}, "observations")
    assert "no units attribute" in res.problems["units"] and "units: mm" in res.problems["units"]


def test_units_that_contradict_the_values_or_the_attribute_are_errors(tmp_path):
    """units mm on values that look like metres, or against the file's own attribute, is an error naming units."""
    write_forecast(tmp_path / "fc", units="m", scale=1e-3)
    res = detect.resolve_entry({"path": str(tmp_path / "fc"), "units": "mm", "lead_label": "start"}, "forecast")
    assert "units" in res.problems and "'m'" in res.problems["units"]
    write_forecast(tmp_path / "fc2", units=None, scale=1e-3)
    res = detect.resolve_entry({"path": str(tmp_path / "fc2"), "units": "mm", "lead_label": "start"}, "forecast")
    assert "looks like metres" in res.problems["units"]
    res = detect.resolve_entry({"path": str(tmp_path / "fc"), "units": "m", "lead_label": "start"}, "forecast")
    assert "units" not in res.problems


def test_specified_dim_that_does_not_exist_is_an_error(tmp_path):
    """A dims entry naming a dim absent from the file is an error naming the role."""
    res = detect.resolve_entry({"path": _obs(tmp_path), "dims": {"lat": "latitude"}}, "observations")
    assert "dims.lat" in res.problems


def test_forecast_members_and_dims_are_detected(tmp_path):
    """Members come from the member dim; step and init dims from known names."""
    write_forecast(tmp_path / "fc", members=4, label="start")
    res = detect.resolve_entry({"path": str(tmp_path / "fc"), "lead_label": "start"}, "forecast")
    assert res.values["members"] == 4
    assert res.values["dims"]["member"] == "number" and res.values["dims"]["step"] == "step"
    assert "member dim 'number' has 4 values" in "\n".join(res.report())


@pytest.mark.parametrize("label", ["start", "end"])
def test_lead_label_is_detected_by_aligning_with_the_observations(tmp_path, label):
    """The forecast built as lead k = day init + k - 1 (or + k) is recognised, with evidence."""
    obs = detect.resolve_entry({"path": _obs(tmp_path), "day_boundary": "00 UTC, labelled by start"}, "observations")
    write_forecast(tmp_path / "fc", label=label)
    res = detect.resolve_entry({"path": str(tmp_path / "fc")}, "forecast", obs=obs.values)
    assert res.values["lead_label"] == label
    assert f'lines up with observations at lag 0 under "{label}"' in "\n".join(res.report())


def test_lead_label_without_observations_is_an_error(tmp_path):
    """Nothing to compare against: the key must be set, and the message says how."""
    write_forecast(tmp_path / "fc")
    res = detect.resolve_entry({"path": str(tmp_path / "fc")}, "forecast")
    assert "Fix: set lead_label" in res.problems["lead_label"]


def test_wrong_kind_is_an_error(tmp_path):
    """A forecast folder described as observations is caught."""
    write_forecast(tmp_path / "fc")
    res = detect.resolve_entry({"path": str(tmp_path / "fc")}, "observations")
    assert "looks like forecast" in res.problems["kind"]


def test_ambiguous_file_names_are_an_error(tmp_path):
    """Two name patterns in the folder cannot be resolved."""
    write_obs(tmp_path / "obs", name="2020.nc")
    write_obs(tmp_path / "obs", name="old_1999.nc")
    res = detect.resolve_entry({"path": str(tmp_path / "obs")}, "observations")
    assert "several name patterns" in res.problems["file_pattern"]


def test_region_is_detected_from_the_extent():
    """The Ethiopia box is recognised; a box over the ocean is unknown."""
    assert detect.detect_region((3, 15, 33, 48))[0] == "ethiopia"
    assert detect.detect_region((-60, -50, -150, -140))[0] == "unknown"


def test_env_var_expansion_and_dataset_files(tmp_path, monkeypatch):
    """${VAR} and ${VAR:-default} expand; entries of a dataset file are found by id, with a helpful unknown-id error."""
    monkeypatch.setenv("ROMP_TEST_ROOT", str(tmp_path))
    assert expand_path("${ROMP_TEST_ROOT}/x") == f"{tmp_path}/x"
    assert expand_path("${ROMP_NOPE_X:-/fallback}/y") == "/fallback/y"
    with pytest.raises(DatasetError, match="ROMP_NOPE_X"):
        expand_path("${ROMP_NOPE_X}/y")
    (tmp_path / "ds.yaml").write_text(TEMPLATES["obs"].replace("my_obs", "mine"))
    datasets = load_datasets([str(tmp_path / "ds.yaml")])
    assert get_dataset("mine", datasets).variable == "rainfall"
    with pytest.raises(DatasetError, match="Did you mean"):
        get_dataset("mime", datasets)


@pytest.mark.parametrize("kind", ["obs", "forecast"])
def test_templates_are_valid_entries(kind):
    """The blank entries printed by init --dataset-template pass the dataset schema."""
    import yaml

    parse_dataset(yaml.safe_load(TEMPLATES[kind]), "template")


def test_dataset_schema_rejects_unknown_keys_with_a_suggestion():
    """A misspelt key names the likely one."""
    with pytest.raises(DatasetError, match='Did you mean "variable"'):
        parse_dataset({"id": "a", "kind": "forecast", "path": "/x", "varible": "tp", "units": "m", "lead_label": "end"}, "t")


def _to_zarr(folder, name):
    """Convert the one NetCDF file written into folder to a sibling zarr store and return its path."""
    import xarray as xr

    store = folder.parent / name
    with xr.open_dataset(next(folder.glob("*.nc"))) as ds:
        ds.to_zarr(store)
    return str(store)


def test_zarr_store_is_one_dataset_without_a_file_pattern(tmp_path):
    """A zarr store needs no file_pattern; variable, units, dims and the years come from its metadata."""
    write_obs(tmp_path / "obs_nc")
    store = _to_zarr(tmp_path / "obs_nc", "obs.zarr")
    res = detect.resolve_entry({"path": store, "day_boundary": "00 UTC, labelled by start"}, "observations",
                               use_cache=False)
    assert not res.problems and res.checked
    assert res.values["variable"] == "precip" and res.values["units"] == "mm"
    assert res.values["dims"] == {"time": "time", "lat": "lat", "lon": "lon"}
    assert "file_pattern" not in res.values
    text = "\n".join(res.report())
    assert "file_pattern: not needed  [detected: zarr store: one dataset holding every year (2020-2020" in text
    assert "no NetCDF files" not in text


def test_zarr_forecast_members_and_the_lead_label_from_the_sampled_check(tmp_path):
    """A zarr forecast gets members from the store and its lead_label from the sampled alignment check."""
    write_obs(tmp_path / "obs_nc")
    obs_store = _to_zarr(tmp_path / "obs_nc", "obs.zarr")
    write_forecast(tmp_path / "fc_nc", members=4, label="end")
    fc_store = _to_zarr(tmp_path / "fc_nc", "fc.zarr")
    obs = detect.resolve_entry({"path": obs_store, "day_boundary": "00 UTC, labelled by start"}, "observations",
                               use_cache=False)
    res = detect.resolve_entry({"path": fc_store}, "forecast", obs=obs.values, use_cache=False)
    assert res.values["members"] == 4 and res.values["lead_label"] == "end"
    assert 'lines up with observations at lag 0 under "end" (sampled check' in "\n".join(res.report())


def test_zarr_detection_is_cached_and_follows_the_store(tmp_path):
    """The detection cache keys on the store's metadata, so a rewritten store is a miss."""
    write_obs(tmp_path / "obs_nc")
    store = _to_zarr(tmp_path / "obs_nc", "obs.zarr")
    entry = {"path": store, "day_boundary": "00 UTC, labelled by start"}
    first = detect._fingerprint(entry, "observations", False)
    assert first and first == detect._fingerprint(entry, "observations", False)
    import shutil
    shutil.rmtree(store)
    write_obs(tmp_path / "obs_nc", name="2020.nc", var="rain")
    store = _to_zarr(tmp_path / "obs_nc", "obs.zarr")
    assert detect._fingerprint(entry, "observations", False) != first
