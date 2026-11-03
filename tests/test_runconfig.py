"""RunConfig (hcwf/config/model.py) against the flat names of the old code (tests/flat_values.py): step S3 of
docs/review/runconfig_design.md. Parity is checked against the frozen pre-R2 code in tests/_oracle_config.py over the
fixture configs and, when present, the four regression configs: `case_kwargs` equals the oracle's `{**Setting, **Case}`
for every case and both reference rules, the typed inputs round-trip, and a RunConfig pickles small and equal."""
import typed_io
import flat_values
import dataclasses
import pickle
import subprocess
import sys
from dataclasses import asdict, FrozenInstanceError
from itertools import product

import pytest

import _oracle_config as old
from typed_io import flat_config
from hcwf.config.model import MetricsSpec, ModelSpec, OutputSpec, RunConfig, Selection
from hcwf.lib import config_schema as schema
from hcwf.lib.run_folder import fingerprint
from test_oracle_config import _norm, _same, configs, world  # noqa: F401  (fixtures: the fixture world and its configs)

FLOWS = flat_values.FLOWS


@pytest.fixture(scope="module")
def runs(configs):
    """{name: (resolved flat dict, RunConfig)} of every fixture and regression config."""
    out = {}
    for name, path in configs.items():
        dic = flat_config(path)
        out[name] = (dic, RunConfig.from_flat(dic))
    return out


def _oracle_case(dic, combi):
    return old.make_case(old.Case, combi, dic)


def _merge(dic, case):
    """Today's {**asdict(Setting), **asdict(Case)}."""
    return {**asdict(old.init_dataclass(old.Setting, dic)), **asdict(case)}


def test_cases_are_the_flows_loop(runs):
    """rc.cases() is the product of the models and windows, with the oracle's names, tolerances and years."""
    for name, (dic, rc) in runs.items():
        combis = list(product(*old.iter_list(dic)))
        assert [c.key for c in rc.cases()] == combis, name
        for case, combi in zip(rc.cases(), combis):
            frozen = _oracle_case(dic, combi)
            assert case.name == frozen.case_name and case.window.tolerance == frozen.tolerance_days
            assert case.window.span == frozen.verification_window and not case.reference
            assert case.window.bins(rc.evaluation.day_bins) == old.filter_bins_in_window(frozen.day_bins,
                                                                                         frozen.verification_window)
        assert rc.evaluation.years == old.evaluation_years(dic)
        assert rc.evaluation.climatology_years == old.climatology_years(dic)
        assert rc.evaluation.members == old.take_ensemble_members(dic["members"])


@pytest.mark.parametrize("flow", FLOWS)
def test_case_kwargs_equal_the_oracle_merge(runs, flow):
    """case_kwargs(rc, case, flow) is {**asdict(Setting), **asdict(Case)} of the oracle: same keys in the same order, equal
    values, for every model and window, and for the reference cases (the oracle's, with the reference's own accumulation,
    daily boundary and dims in place of the two old rules)."""
    for name, (dic, rc) in runs.items():
        for case, (combi, flow_kw) in zip(rc.cases(), old.deterministic_cases(dic)):
            expected = _merge(dic, _oracle_case(dic, combi))
            kw = flat_values.case_kwargs(rc, case, flow)
            assert kw == expected == flow_kw, (name, combi)
            assert list(kw) == list(expected), (name, combi)
        own = flat_values.reference_settings(dic)
        if flow == "deterministic":
            pairs = [(c, {**kw, **own}) for c, (_, kw) in zip(rc.reference_cases(), old.deterministic_reference_cases(dic))]
        else:
            pairs = []
            per_model = iter(kw for _, kw in old.ensemble_reference_cases(dic))
            for model in rc.models:
                pairs += [(c, {**next(per_model), **own}) for c in rc.reference_cases(for_model=model)]
        assert pairs, name
        for case, expected in pairs:
            assert case.reference
            assert flat_values.case_kwargs(rc, case, flow) == expected, (name, flow, case.name)


def test_ensemble_cases_have_the_oracles_prepare_dict(runs):
    """The ensemble flow's prep_cfg is the case_kwargs (full day bins); create_score_results narrows them to the window."""
    for name, (dic, rc) in runs.items():
        for case, (combi, prep, narrowed) in zip(rc.cases(), old.ensemble_cases(dic)):
            assert flat_values.case_kwargs(rc, case, "ensemble") == prep
            assert case.window.bins(rc.evaluation.day_bins) == narrowed["day_bins"]


def test_reference_settings_are_its_own(runs):
    """One rule in both flows: the reference is read with its own accumulation, daily boundary and dims, not a forecast
    model's. Only the ensemble flow's case name follows the forecast model."""
    dic, rc = runs["model_reference"]
    first, second = rc.models
    assert first.accumulation != second.accumulation == rc.reference.accumulation   # DETC, accumulated, is the reference
    assert rc.reference.accumulation_detected and rc.reference.daily_boundary is None
    for case in rc.reference_cases():
        assert case.reference and case.for_model is None and case.model.name == "DETC"
        assert case.model == rc.reference.as_model() and case.model.dir == rc.reference.dir
        assert case.model.accumulation == rc.reference.accumulation != first.accumulation
        assert case.model.dims == rc.reference.dims and case.name.startswith("DETC_")
    for model in rc.models:
        for case, plain in zip(rc.reference_cases(for_model=model), rc.reference_cases()):
            assert case.model == plain.model and case.window == plain.window
            assert case.for_model == model.name and case.name.startswith(model.name + "_")
            assert case.model.name == "DETC" and case.key[0] == "DETC"
    assert rc.reference.kind == "model"
    clim = runs["det_climatology"][1].reference
    assert clim.kind == "climatology" and clim.accumulation is None and clim.daily_boundary is None


@pytest.mark.parametrize("flow", FLOWS)
def test_from_case_kwargs_round_trips(runs, flow):
    """from_case_kwargs(case_kwargs(...)) gives typed inputs whose case_kwargs are the same dict."""
    for name, (dic, rc) in runs.items():
        cases = list(rc.cases())
        cases += rc.reference_cases() if flow == "deterministic" else \
            [c for m in rc.models for c in rc.reference_cases(for_model=m)]
        for case in cases:
            kw = flat_values.case_kwargs(rc, case, flow)
            rc2, case2, selection = typed_io.from_case_kwargs(kw)
            assert selection == Selection()
            assert case2.reference == (case.reference and case.model.name not in dic["model_list"])
            assert case2.name == case.name and case2.window == case.window
            assert flat_values.case_kwargs(rc2, case2, flow) == kw, (name, case.name)
            assert list(flat_values.case_kwargs(rc2, case2, flow)) == list(kw)


def test_from_case_kwargs_keeps_edits_and_selections(runs):
    """A value edited in place (years, max_forecast_day, the reference keys) is what the rebuilt inputs hold; the
    selection keys come back as a Selection; anything else is refused."""
    dic, rc = runs["model_reference"]
    kw = flat_values.case_kwargs(rc, rc.cases()[0], "deterministic")
    kw.update(years=(2019,), max_forecast_day=7, ref_model_var="other", keep_unit_ids=["a"], lat_select=(1, 2),
              model_files_years=(2019,))
    rc2, case2, selection = typed_io.from_case_kwargs(kw)
    assert rc2.evaluation.years == (2019,) and rc2.evaluation.max_forecast_day == 7 and rc2.reference.var == "other"
    assert selection.keep_unit_ids == ["a"] and selection.lat == (1, 2) and selection.model_files_years == (2019,)
    assert selection.grid_point is None
    assert flat_values.case_kwargs(rc2, case2, "deterministic", selection) == kw
    with pytest.raises(ValueError, match="not config keys or selections: surprise"):
        typed_io.from_case_kwargs({**kw, "surprise": 1})


def test_a_copy_of_the_values_is_independent(runs):
    for name, (dic, rc) in runs.items():
        flat = dict(rc.values)
        assert flat == dic and flat is not dic
        flat["max_forecast_day"] = -1
        assert rc.values["max_forecast_day"] == dic["max_forecast_day"]
        assert rc.values["run_paths"] is dic["run_paths"]


def test_pickle_is_small_and_equal(runs):
    for name, (dic, rc) in runs.items():
        data = pickle.dumps(rc)
        assert len(data) < 32 * 1024, (name, len(data))
        back = pickle.loads(data)
        assert back == rc and back is not rc and hash(back) == hash(rc)
        assert back.models == rc.models and back.evaluation == rc.evaluation and back.onset == rc.onset
        assert back.output == rc.output and back.reference == rc.reference and back.observations == rc.observations
        assert flat_values.case_kwargs(back, back.cases()[0], "ensemble") == \
            flat_values.case_kwargs(rc, rc.cases()[0], "ensemble")
    assert RunConfig.from_flat({**dic, "max_forecast_day": 12}, check=False) != rc


def test_specs_are_frozen(runs):
    dic, rc = runs["two_models_windows"]
    for obj, attr in [(rc, "project_name"), (rc.models[0], "name"), (rc.observations, "dir"), (rc.reference, "model"),
                      (rc.evaluation, "years"), (rc.evaluation.windows[0], "end"), (rc.area, "region"),
                      (rc.parallel, "enabled"), (rc.output, "data_dir"),
                      (rc.metrics, "FAR"), (rc.onset_options, "missing"), (rc.cases()[0], "window"),
                      (rc.figure_spec(), "polygon"), (Selection(), "lat")]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, 1)
        assert not hasattr(obj, "__dict__")                      # slots
    with pytest.raises(TypeError):
        rc.values["max_forecast_day"] = 1


def test_run_paths_fingerprint_is_the_flat_one(runs):
    for name, (dic, rc) in runs.items():
        assert rc.paths is dic["run_paths"] and rc.paths.fingerprint == fingerprint(dict(rc.values)), name


def test_without_files_is_the_write_false_resolve(configs, runs, tmp_path):
    """without_files(tmp) holds what flat_config(write=False, tmp=tmp) gives (outputs off, folders in tmp, no disk cache)."""
    for name, (dic, rc) in runs.items():
        quiet = rc.without_files(tmp_path)
        _same(dict(quiet.values), flat_config(configs[name], write=False, tmp=tmp_path))
        assert not quiet.paths.write and rc.paths.write
        assert quiet.output.save_fig is False and quiet.output.data_dir == str(tmp_path) and quiet.cache_dir is None
        assert quiet.onset_options.explain_cells is None and quiet.paths.fingerprint == rc.paths.fingerprint
        assert quiet.models == rc.models and quiet.evaluation == rc.evaluation
    assert rc.output.data_dir != str(tmp_path)                   # the original is untouched


def test_the_projections_match_the_flat_record(runs):
    """Spot checks of what each spec holds, against the flat keys."""
    dic, rc = runs["two_models_windows"]
    assert [m.name for m in rc.models] == list(dic["model_list"])
    assert [m.dir for m in rc.models] == list(dic["model_dir_list"])
    assert [m.accumulation for m in rc.models] == list(dic["accumulation_list"])
    assert [m.accumulation_detected for m in rc.models] == list(dic["accumulation_detected"])
    assert rc.models[0].dims_map() == dic["model_dims_list"][0] and rc.models[0].dims[0] == ("init_time", "time")
    assert [w.span for w in rc.evaluation.windows] == list(dic["verification_window_list"])
    assert [w.tolerance for w in rc.evaluation.windows] == list(dic["tolerance_days_list"])
    assert [w.label for w in rc.evaluation.windows] == ["1-10", "1-5"]
    obs = rc.observations
    assert (obs.name, obs.dir, obs.var, obs.file_pattern, obs.day_boundary) == \
        (dic["obs"], dic["obs_dir"], dic["obs_var"], dic["obs_file_pattern"], dic["obs_day_boundary"])
    assert rc.area.region == dic["region"] == "ethiopia" and rc.area.mask is None and rc.area.polygon is False
    assert rc.parallel.enabled is dic["parallel"] and rc.parallel.workers is None
    assert rc.output.data_dir == dic["dir_out"] and rc.output.figure_dir == dic["dir_fig"]
    assert rc.output.save_csv_score == dic["save_csv_score"] and rc.metrics.skill_score is dic["skill_score"]
    assert rc.evaluation.lead_window == dic["lead_window"] and rc.onset.backend == dic["onset_backend"]
    assert rc.onset_options.percentage_threshold == dic["onset_percentage_threshold"]
    assert rc.blend is None and rc.paths.name == dic["run_paths"].name and rc.short_names is dic["short_names"]
    assert dict(rc.sources) == dic["config_sources"] and rc.report == tuple(dic["dataset_report"])
    assert rc.config_file == dic["config_file"] and rc.lead_lines == tuple(dic["onset_lead_lines"])
    fig = rc.figure_spec()
    assert fig.dir_fig == dic["dir_fig"] and fig.models == tuple(dic["model_list"]) and fig.short_names
    assert rc.model("DETC") is rc.models[1]
    with pytest.raises(KeyError):
        rc.model("nope")


def test_generated_specs_follow_the_schema():
    for cls, section in ((OutputSpec, "output"), (MetricsSpec, "metrics")):
        assert [f.name for f in dataclasses.fields(cls)] == [f.nested_name for f in schema.FIELDS if f.section == section]
        assert cls.__module__ == "hcwf.config.model"


def test_model_spec_fields_correspond_to_the_list_keys():
    """Each per-model *_list key maps to a ModelSpec attribute (x_list and model_x_list to x; model_list to name)."""
    attrs = {f.name for f in dataclasses.fields(ModelSpec)}
    for field in schema.FIELDS:
        if field.internal and field.key.endswith("_list"):
            stem = field.key[:-5].removeprefix("model_")
            assert ("name" if stem == "model" else stem) in attrs, field.key
    assert {"accumulation_detected", "units"} <= attrs


def test_invariants(runs):
    dic, rc = runs["two_models_windows"]
    with pytest.raises(ValueError, match="unique"):
        RunConfig.from_flat({**dic, "model_list": ("DET", "DET")})
    with pytest.raises(ValueError, match="no spaces"):
        RunConfig.from_flat({**dic, "model_list": ("DET", "D ETC")})
    with pytest.raises(ValueError, match="max_forecast_day"):
        RunConfig.from_flat({**dic, "max_forecast_day": 4})
    RunConfig.from_flat({**dic, "max_forecast_day": 4}, check=False)
    with pytest.raises(ValueError, match="blend"):
        RunConfig.from_flat({**dic, "blend_years": {"training": [2019]}})


def test_a_blend_config_has_its_recipe(world, tmp_path):
    from test_run_experience import raw_config, write_config

    raw = raw_config(world, ("DET",))
    raw["blend"] = {"years": {"training": [2019], "cv_holdout": [2020]},
                    "formulas": {"f": "outcome ~ prob_clim_qx"}}
    dic = flat_config(write_config(tmp_path, raw, "blend.yaml"))
    rc = RunConfig.from_flat(dic)
    assert rc.blend is not None and [m.name for m in rc.blend.models] == ["DET"]
    assert rc.paths.fingerprint == fingerprint(dict(rc.values)) and pickle.loads(pickle.dumps(rc)) == rc
    assert rc.without_files(tmp_path).blend == rc.blend


def test_importing_the_config_package_stays_light():
    code = ("import sys, hcwf.config, hcwf.config.model, hcwf.config.build\n"
            "heavy = [m for m in ('hcwf.io.input', 'hcwf.blend.recipe', 'hcwf.utils.mask') "
            "if m in sys.modules]\n"
            "assert not heavy, heavy\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
