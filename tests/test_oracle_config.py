"""The frozen pre-R2 configuration code (tests/_oracle_config.py) imports and agrees with the live code.

Step S0 of docs/review/runconfig_design.md. The oracle's `Setting`/`Case`/`make_case`/`case_kwargs` and its four finishing
variants (loader, api, export, driver) are compared with the live ones over fixture configs on synthetic data (a
deterministic run with the climatology reference, two models and two windows with a model reference, an ensemble) and
over the four regression configs when they are present. Later steps compare the new code with the oracle the same way.
"""
import dataclasses
import sys
from pathlib import Path

import pytest
import yaml

import _oracle_config as old
from hcwf import api, export
from hcwf.lib import control, convention, loader
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
    assert [f.name for f in dataclasses.fields(old.Case)] == [f.name for f in dataclasses.fields(convention.Case)]
    assert [f.name for f in dataclasses.fields(old.Setting)] == [f.name for f in dataclasses.fields(convention.Setting)]
    assert old.LAYOUT == control.LAYOUT == ("model", "verification_window")


def test_loader_variant_agrees(configs):
    """loader.build_cfg / load_run_config of the oracle and of the live code resolve every config to the same dict."""
    for name, path in configs.items():
        args = ["-p", str(path)]
        _same(old.build_cfg(args), loader.build_cfg(args))
        _same(old.load_run_config(args), loader.load_run_config(args))


def test_loader_variant_agrees_with_cli_override(configs):
    path = configs["det_climatology"]
    args = ["-p", str(path), "--max_forecast_day", "12"]
    new, ref = loader.load_run_config(args), old.load_run_config(args)
    _same(ref, new)
    assert new["max_forecast_day"] == 12


def test_api_variant_agrees(configs, tmp_path):
    """api._prepare of the oracle and of the live code: a path with overrides, write on and off, and a mapping."""
    for name, path in configs.items():
        for write in (True, False):
            _same(old._prepare(path, {}, write, tmp_path), api._prepare(path, {}, write, tmp_path))
    path = configs["two_models_windows"]
    _same(old._prepare(path, {"max_forecast_day": 12}, True, None), api._prepare(path, {"max_forecast_day": 12}, True, None))
    mapping = yaml.safe_load(Path(configs["det_climatology"]).read_text())
    _same(old._prepare(mapping, {}, False, tmp_path), api._prepare(mapping, {}, False, tmp_path))


def test_export_variant_agrees(configs):
    """export.load_cfg / _complete / _as_dict of the oracle and of the live code."""
    for name, path in configs.items():
        live, frozen = export.load_cfg(path), old.load_cfg(path)
        _same(frozen, live)
        _same(old._as_dict(path), export._as_dict(path))
        flat = loader.load_run_config(["-p", str(path)])
        _same(old._as_dict(dict(flat)), export._as_dict(dict(flat)))
        _same(old._complete(dict(flat)), export._complete(dict(flat)))


def test_driver_variant_finishes_like_the_loader(configs):
    """The oracle's driver branch: from the CLI, and from an already resolved dict (nothing is planned twice)."""
    for name, path in configs.items():
        args = ["-p", str(path)]
        dic = old.driver_finish(None, args)
        _same(dic, loader.load_run_config(args))
        _same(old.driver_finish(dic), dic)


def _case_dicts(case_cfg):
    return {key: case_cfg[key] for key in sorted(case_cfg)}


def test_cases_and_case_kwargs_agree(configs):
    """The oracle's make_case / case_kwargs equal the live ones for every model, window and reference rule."""
    for name, path in configs.items():
        dic = loader.load_run_config(["-p", str(path)])
        assert old.iter_list(dic) == control.iter_list(dic)
        setting = convention.Setting
        live_setting = control.init_dataclass(setting, dic)
        frozen_setting = old.init_dataclass(old.Setting, dic)
        assert dataclasses.asdict(live_setting) == dataclasses.asdict(frozen_setting)
        for combi in old.product(*old.iter_list(dic)):
            frozen = old.make_case(old.Case, combi, dic)
            live = control.make_case(convention.Case, combi, dic)
            assert dataclasses.asdict(frozen) == dataclasses.asdict(live)
        for model in dic["model_list"]:
            assert old.case_kwargs(dic, model) == export.case_kwargs(dic, model)
        if dic["ref_model"]:
            assert old.case_kwargs(dic, dic["ref_model"], reference=True) == \
                export.case_kwargs(dic, dic["ref_model"], reference=True)


def test_reference_rules_are_the_two_documented_ones(configs):
    """Deterministic flow: the reference inherits models[0]'s accumulation and boundary and has no model_dims.
    Ensemble flow: the current model's own, and ref_model_dims (design section 1.5)."""
    dic = loader.load_run_config(["-p", str(configs["model_reference"])])
    first = dic["model_list"][0]
    det = old.deterministic_reference_cases(dic)
    assert len(det) == len(dic["verification_window_list"])
    for combi, kw in det:
        assert kw["model"] == dic["ref_model"] and kw["model_dims"] is None
        assert kw["accumulation"] == dic["accumulation_list"][0] and kw["daily_boundary"] == dic["daily_boundary_list"][0]
        assert kw["model_dir"] == dic["ref_model_dir"] and kw["unit_cvt"] == dic["ref_model_unit_cvt"]
    # case_kwargs(reference=True) follows the deterministic rule
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
    """The flows' case dicts hold 147 keys (Setting 124 + Case 23); the setting wins in the deterministic flow and
    the case in the ensemble flow, which agree because the field sets do not overlap."""
    dic = loader.load_run_config(["-p", str(configs["two_models_windows"])])
    for combi, kw in old.deterministic_cases(dic):
        assert len(kw) == 147
        assert kw["model"] == combi[0] and kw["verification_window"] == combi[1]
    for (combi, prep, kw), (_, det) in zip(old.ensemble_cases(dic), old.deterministic_cases(dic)):
        assert prep.keys() == kw.keys() == det.keys()
        assert {k: v for k, v in kw.items() if k != "day_bins"} == {k: v for k, v in det.items() if k != "day_bins"}


def test_fingerprints_agree(configs):
    """The run folder fingerprint and the resolved config text are the same through both resolve variants."""
    for name, path in configs.items():
        a = old.load_run_config(["-p", str(path)])
        b = loader.load_run_config(["-p", str(path)])
        assert fingerprint(a) == fingerprint(b) == a["run_paths"].fingerprint
        assert resolved_config_text(a).replace(" (cached)", "") == resolved_config_text(b).replace(" (cached)", "")
