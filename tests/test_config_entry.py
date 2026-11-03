"""The entry points resolve a config to the same record: a path, a Config, a mapping and a flat dict, with command-line
flags and keyword overrides (the flags and the overrides record their source), `--model_list`, and the fingerprint of the
run folder."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from config_world import _norm, _same, configs, world  # noqa: F401  (fixtures: the fixture world and its configs)
from hcwf.runs import export
from hcwf.config.model import RunConfig
from hcwf.config import Config
from typed_io import flat_config, flat_command_line
from hcwf.errors import HCWFConfigError
from hcwf.runs.folder import fingerprint
from hcwf.runs.resolved import resolved_config_text


def test_a_command_line_flag_overrides_the_config_and_is_recorded(configs):
    path = configs["det_climatology"]
    plain = flat_command_line(["-p", str(path)])
    new = flat_command_line(["-p", str(path), "--max_forecast_day", "12"])
    assert new["max_forecast_day"] == 12 != plain["max_forecast_day"]
    assert new["config_sources"]["max_forecast_day"] == "command line"
    assert plain["config_sources"]["max_forecast_day"] != "command line"
    for key in plain:
        if key not in ("max_forecast_day", "config_sources", "run_paths", "dataset_report", "onset_lead_lines",
                       "accumulation_detected", "dir_out", "dir_fig", "lead_window"):
            assert _norm(new[key]) == _norm(plain[key]), key


def test_the_command_line_and_the_python_entry_resolve_a_path_alike(configs, tmp_path):
    for name, path in configs.items():
        _same(flat_command_line(["-p", str(path)]), flat_config(path))
        assert flat_config(path, write=False, tmp=tmp_path)["run_paths"].write is False


def test_every_source_gives_the_same_dict(configs):
    """A path, a Config, a nested mapping and a flat dict resolve to equal dicts."""
    path = configs["det_climatology"]
    from_path = flat_config(path)
    mapping = yaml.safe_load(Path(path).read_text())
    # a Config or mapping has no file: the paths that start at the config file (runs/) start at the working directory
    from_mapping = flat_config(mapping)
    from_config = flat_config(Config.from_mapping(mapping))
    _same(from_mapping, from_config)
    for key in from_path:
        if key not in ("run_paths", "config_file", "dataset_report", "config_sources", "dir_out", "dir_fig", "cache_dir"):
            assert from_path[key] == from_mapping[key], key
    again = flat_config(from_path)                       # a finished flat dict resolves to itself
    _same(from_path, again)
    _same(from_path, flat_config(SimpleNamespace(**from_path)))


def test_cli_flags_equal_run_overrides(configs):
    """Every override flag gives what hcwf.run's keyword overrides give, recorded as 'command line' (CLI) or 'override'."""
    path = configs["two_models_windows"]
    args = ["-p", str(path), "--max_forecast_day", "10", "--verification_window_list", "1,5", "--tolerance_days_list", "2",
            "--start_date", "2019", "6", "1", "--probabilistic", "false", "--parallel", "no", "--day_bins", "1,5",
            "--region", "ethiopia", "--onset_percentage_threshold", "0.4", "--show_plot", "false"]
    cli = flat_command_line(args)
    overrides = {"max_forecast_day": 10, "verification_window_list": ((1, 5),), "tolerance_days_list": (2,),
                 "start_date": (2019, 6, 1), "probabilistic": False, "parallel": False, "day_bins": ((1, 5),),
                 "region": "ethiopia", "onset_percentage_threshold": 0.4, "show_plot": False}
    api_like = flat_config(path, overrides)
    for key in api_like:
        if key != "config_sources":
            assert _norm(cli[key]) == _norm(api_like[key]), key
    assert {k for k, v in cli["config_sources"].items() if v == "command line"} == set(overrides)
    assert {k for k, v in api_like["config_sources"].items() if v == "override"} == set(overrides)
    assert not [v for k, v in cli["config_sources"].items() if v == "override"]
    from hcwf.runs.resolved import resolved_config_text

    assert "command line" in resolved_config_text(cli)
    assert cli["run_paths"].fingerprint == api_like["run_paths"].fingerprint != flat_config(path)["run_paths"].fingerprint



def test_bad_cli_values_are_validated(configs):
    path = str(configs["det_climatology"])
    with pytest.raises(HCWFConfigError, match="max_forecast_day"):
        flat_config(path, {"max_forecast_day": "soon"}, override_source="command line")
    with pytest.raises(HCWFConfigError, match="tolerance_days_list"):   # one tolerance for each window
        flat_command_line(["-p", path, "--verification_window_list", "1,5 6,10"])
    with pytest.raises(TypeError, match="unknown override"):
        flat_config(path, {"max_forecast_dya": 3})



def test_model_list_selects_models(configs):
    """--model_list keeps the named models (in the order given) with each model's own entries; an unknown name fails."""
    path = configs["two_models_windows"]
    full = flat_config(path)
    assert full["model_list"] == ("DET", "DETC")
    only = flat_command_line(["-p", str(path), "--model_list", "DETC"])
    assert only["model_list"] == ("DETC",) and only["config_sources"]["model_list"] == "command line"
    for key in ("model_dir_list", "model_var_list", "unit_cvt_list", "file_pattern_list", "accumulation_list",
                "daily_boundary_list", "accumulation_detected"):
        if key in full:
            assert only[key] == (full[key][1],), key
    flipped = flat_config(path, {"model_list": ("DETC", "DET")})
    assert flipped["model_dir_list"] == full["model_dir_list"][::-1]
    with pytest.raises(HCWFConfigError, match="unknown model 'NOPE'"):
        flat_command_line(["-p", str(path), "--model_list", "NOPE"])



def test_export_on_a_dict_validates_and_plans(configs):
    """export_onsets resolves a mapping like every entry: it plans the run folder and rejects an invalid config."""
    raw = yaml.safe_load(Path(configs["det_climatology"]).read_text())
    bad = {**raw, "evaluation": {**raw.get("evaluation", {}), "verification_windows": [[1, 5]], "tolerances": [3, 2]}}
    with pytest.raises(HCWFConfigError):
        export.export_onsets(bad)
    dic = flat_config(raw)
    assert dic["run_paths"] is not None and export.onsets_folder(RunConfig.from_flat(dic)) == dic["run_paths"].onsets


def test_fingerprint_and_resolved_text_are_the_same_through_both_entries(configs):
    """The run folder fingerprint and the resolved config text of a resolved config."""
    for name, path in configs.items():
        a = flat_command_line(["-p", str(path)])
        b = flat_config(path)
        assert fingerprint(a) == fingerprint(b) == a["run_paths"].fingerprint
        assert resolved_config_text(a).replace(" (cached)", "") == resolved_config_text(b).replace(" (cached)", "")
