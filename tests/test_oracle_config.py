"""The frozen pre-R2 configuration code (tests/_oracle_config.py) imports and agrees with the live code.

Step S0 of docs/review/runconfig_design.md. The oracle's `Setting`/`Case`/`make_case`/`case_kwargs` and its four finishing
variants (loader, api, export, driver) are compared with the live ones over fixture configs on synthetic data (a
deterministic run with the climatology reference, two models and two windows with a model reference, an ensemble) and
over the four regression configs when they are present. Later steps compare the new code with the oracle the same way.
"""
import flat_values
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import _oracle_config as old
from hcwf import export
from hcwf.config.model import RunConfig
from hcwf.config import Config
from typed_io import flat_config, flat_command_line
from hcwf.lib.assertion import HCWFConfigError
from hcwf.lib.run_folder import fingerprint
from hcwf.lib.resolved import resolved_config_text

sys.path.insert(0, str(Path(__file__).parent))
from test_run_experience import MAX_LEAD, _write_ensemble, raw_config, write_config, write_forecast, write_obs  # noqa: E402

REGRESSION_CFG = Path("/tmp/claude-1000/-home-hcwf-code-ROMP/f3fbeea6-493c-485b-bfb6-ad02edb09411/scratchpad/out_base/cfg")


def build_world(base: Path) -> Path:
    """Observations, DET, DETC (accumulated) and ENS forecasts under base; returns base."""
    write_obs(base / "obs")
    write_forecast(base / "DET")
    write_forecast(base / "DETC", cumulative=True)
    _write_ensemble(base / "ENS")
    return base


def fixture_configs(base: Path, folder: Path) -> dict[str, Path]:
    """{name: YAML path} of the fixture run configs (written into folder, pointing at the data in base)."""
    folder.mkdir(parents=True, exist_ok=True)
    two = {"verification_windows": [[1, MAX_LEAD], [1, 5]], "tolerances": [3, 2]}
    raws = {
        "det_climatology": raw_config(base, ("DET",)),
        "two_models_windows": raw_config(base, ("DET", "DETC"), evaluation=two),
        "model_reference": raw_config(base, ("DET", "DETC"), evaluation=two,
                                      reference={"model": "DETC", "path": str(base / "DETC"), "variable": "tp",
                                                 "units": "mm"}),
        "ensemble": raw_config(base, ("ENS",), evaluation={"probabilistic": True}),
    }
    return {name: write_config(folder, raw, f"{name}.yaml") for name, raw in raws.items()}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("oracle_world"))


@pytest.fixture(scope="module")
def configs(world, tmp_path_factory):
    out = fixture_configs(world, tmp_path_factory.mktemp("oracle_cfgs"))
    if REGRESSION_CFG.is_dir():
        out.update({p.stem: p for p in sorted(REGRESSION_CFG.glob("*.yaml"))})
    return out


def _keys(configs):
    return sorted(configs)


def _norm(value):
    """value without the " (cached)" the dataset detection adds to its notes the second time it reads the same files."""
    if isinstance(value, str):
        return value.replace(" (cached)", "")
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_norm(v) for v in value)
    return value


def _same(a, b):
    """Resolved flat dicts equal (run_paths compared by their fields; they carry the fingerprint)."""
    assert a.keys() == b.keys()
    for key in a:
        assert _norm(a[key]) == _norm(b[key]), key


def test_oracle_has_the_frozen_classes():
    """Setting has one field per schema key outside Case plus the run's own; Case has the 23 fields."""
    assert len(dataclasses.fields(old.Case)) == 23
    assert [f.name for f in dataclasses.fields(old.Case)] == list(flat_values.CASE_FIELDS)
    assert [f.name for f in dataclasses.fields(old.Setting)] == list(flat_values._setting_part({}))
    assert old.LAYOUT == ("model", "verification_window")


def test_loader_variant_agrees(configs):
    """The oracle's build_cfg / load_run_config and the live command-line resolve give every config the same dict."""
    for name, path in configs.items():
        args = ["-p", str(path)]
        _same(old.build_cfg(args), flat_command_line(args))
        _same(old.load_run_config(args), flat_config(path))


def test_loader_variant_agrees_with_cli_override(configs):
    """A command-line flag gives the oracle's values; only its record differs ('command line', which the oracle lacks)."""
    path = configs["det_climatology"]
    args = ["-p", str(path), "--max_forecast_day", "12"]
    new, ref = flat_command_line(args), old.load_run_config(args)
    assert new["max_forecast_day"] == 12
    assert new["config_sources"]["max_forecast_day"] == "command line"
    assert ref["config_sources"]["max_forecast_day"] != "command line"
    for key in ref:
        if key not in ("config_sources", "run_paths", "dataset_report"):
            assert new[key] == ref[key], key


def test_api_variant_agrees(configs, tmp_path):
    """The oracle's api._prepare and resolve: a path with overrides, write on and off, and a mapping."""
    for name, path in configs.items():
        for write in (True, False):
            _same(old._prepare(path, {}, write, tmp_path), flat_config(path, write=write, tmp=tmp_path))
    path = configs["two_models_windows"]
    _same(old._prepare(path, {"max_forecast_day": 12}, True, None), flat_config(path, {"max_forecast_day": 12}))
    mapping = yaml.safe_load(Path(configs["det_climatology"]).read_text())
    _same(old._prepare(mapping, {}, False, tmp_path), flat_config(mapping, write=False, tmp=tmp_path))


def test_every_source_gives_the_same_dict(configs, tmp_path):
    """A path, a Config, a nested mapping and a flat dict resolve to equal dicts (and equal the oracle's)."""
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
    _same(old._as_dict(path), from_path)
    _same(old._as_dict(dict(from_path)), again)
    _same(old._complete(dict(from_path)), again)


def test_export_variant_agrees(configs):
    """The oracle's export.load_cfg and resolve of a path."""
    for name, path in configs.items():
        _same(old.load_cfg(path), flat_config(path))


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
    from hcwf.lib.resolved import resolved_config_text

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


def test_driver_variant_finishes_like_the_loader(configs):
    """The oracle's driver branch: from the CLI, and from an already resolved dict (nothing is planned twice)."""
    for name, path in configs.items():
        args = ["-p", str(path)]
        dic = old.driver_finish(None, args)
        _same(dic, flat_command_line(args))
        _same(old.driver_finish(dic), dic)
        _same(flat_config(dic), dic)


def _case_dicts(case_cfg):
    return {key: case_cfg[key] for key in sorted(case_cfg)}


def test_cases_and_case_kwargs_agree(configs):
    """The oracle's Setting part and case_kwargs equal the bridge's for every model and the reference (first window)."""
    for name, path in configs.items():
        dic = flat_config(path)
        rc = RunConfig.from_flat(dic)
        frozen_setting = dataclasses.asdict(old.init_dataclass(old.Setting, dic))
        assert flat_values._setting_part(dic) == frozen_setting and list(flat_values._setting_part(dic)) == list(frozen_setting)
        for model in dic["model_list"]:
            assert old.case_kwargs(dic, model) == flat_values.model_kwargs(rc, model)
        if dic["ref_model"]:
            # export.case_kwargs also set model_list to (reference,); nothing reads it for a reference, and the flows
            # keep the forecast models' (the bridge follows the flows)
            frozen = old.case_kwargs(dic, dic["ref_model"], reference=True)
            assert frozen.pop("model_list") == (dic["ref_model"],)
            live = flat_values.model_kwargs(rc, dic["ref_model"], reference_as=dic["ref_model"])
            assert live.pop("model_list") == dic["model_list"] and {**frozen, **flat_values.reference_settings(dic)} == live


def test_oracles_two_reference_rules_are_the_ones_replaced(configs):
    """What the oracle (the pre-R2 flows) gave the reference, which the live code no longer does: the deterministic flow
    gave it models[0]'s accumulation and boundary and no model_dims; the ensemble flow the current model's and
    ref_model_dims. The live reference takes its own (tests/test_runconfig.py, tests/test_reference_own_settings.py)."""
    dic = flat_config(configs["model_reference"])
    det = old.deterministic_reference_cases(dic)
    assert len(det) == len(dic["verification_window_list"])
    for combi, kw in det:
        assert kw["model"] == dic["ref_model"] and kw["model_dims"] is None
        assert kw["accumulation"] == dic["accumulation_list"][0] and kw["daily_boundary"] == dic["daily_boundary_list"][0]
        assert kw["model_dir"] == dic["ref_model_dir"] and kw["unit_cvt"] == dic["ref_model_unit_cvt"]
    kw = old.case_kwargs(dic, dic["ref_model"], reference=True)
    assert kw["accumulation"] == dic["accumulation_list"][0] and kw["model_dims"] is None
    ens = old.ensemble_cases(dic)
    refs = old.ensemble_reference_cases(dic)
    assert len(ens) == len(refs) == len(dic["model_list"]) * len(dic["verification_window_list"])
    for (combi, prep, _), (_, ref) in zip(ens, refs):
        assert ref["model"] == dic["ref_model"] and ref["model_dims"] == dic.get("ref_model_dims")
        assert ref["accumulation"] == prep["accumulation"] and ref["daily_boundary"] == prep["daily_boundary"]
        assert prep["model"] == combi[0]


def test_flow_case_dicts_have_the_documented_shape(configs):
    """The flows' case dicts hold 150 keys (Setting 127, the schema's other keys, + Case 23); the setting wins in the deterministic flow and
    the case in the ensemble flow, which agree because the field sets do not overlap."""
    dic = flat_config(configs["two_models_windows"])
    for combi, kw in old.deterministic_cases(dic):
        assert len(kw) == 150
        assert kw["model"] == combi[0] and kw["verification_window"] == combi[1]
    for (combi, prep, kw), (_, det) in zip(old.ensemble_cases(dic), old.deterministic_cases(dic)):
        assert prep.keys() == kw.keys() == det.keys()
        assert {k: v for k, v in kw.items() if k != "day_bins"} == {k: v for k, v in det.items() if k != "day_bins"}


def test_fingerprints_agree(configs):
    """The run folder fingerprint and the resolved config text are the same through both resolve variants."""
    for name, path in configs.items():
        a = old.load_run_config(["-p", str(path)])
        b = flat_config(path)
        assert fingerprint(a) == fingerprint(b) == a["run_paths"].fingerprint
        assert resolved_config_text(a).replace(" (cached)", "") == resolved_config_text(b).replace(" (cached)", "")
