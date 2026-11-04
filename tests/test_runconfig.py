"""RunConfig (hcwf/config/model.py) against the flat record it is projected from (step S3 of
docs/review/runconfig_design.md), over the fixture configs and, when present, the four regression configs: the cases are
the product of the models and windows, `case_kwargs` (tests/typed_io.py) gives each case its settings and own values for
both reference rules, the typed inputs round-trip, and a RunConfig pickles small and equal."""
import typed_io
import dataclasses
import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from typed_io import FLOWS, CASE_FIELDS, case_kwargs, flat_config
from hcwf.config.model import MetricsSpec, ModelSpec, OutputSpec, RunConfig, Selection
from hcwf.config import schema
from hcwf.runs.folder import fingerprint
from config_world import _same, configs, world  # noqa: F401  (fixtures: the fixture world and its configs)


@pytest.fixture(scope="module")
def runs(configs):
    """{name: (resolved flat dict, RunConfig)} of every fixture and regression config."""
    out = {}
    for name, path in configs.items():
        dic = flat_config(path)
        out[name] = (dic, RunConfig.from_flat(dic))
    return out


def test_cases_are_the_flows_loop(runs):
    """rc.cases() is the product of the models and windows, with the case names, tolerances and years of the flat record."""
    for name, (dic, rc) in runs.items():
        combis = list(product(dic["model_list"], dic["verification_window_list"]))
        assert [c.key for c in rc.cases()] == combis, name
        tolerances = dict(zip(dic["verification_window_list"], dic["tolerance_days_list"]))
        for case, combi in zip(rc.cases(), combis):
            assert case.name == f"{combi[0]}_{combi[1][0]}-{combi[1][1]}"
            assert case.window.tolerance == tolerances[combi[1]] and case.window.span == combi[1] and not case.reference
            assert case.window.bins(rc.evaluation.day_bins) == tuple(
                b for b in dic["day_bins"] if combi[1][0] <= b[0] and b[1] <= combi[1][1])
        assert set(rc.evaluation.years) <= set(range(dic["start_date"][0], dic["end_date"][0] + 1)) and rc.evaluation.years
        assert set(rc.evaluation.climatology_years) <= set(range(dic["start_year_clim"], dic["end_year_clim"] + 1))


@pytest.mark.parametrize("flow", FLOWS)
def test_case_kwargs_hold_the_settings_and_the_cases_own_values(runs, flow):
    """case_kwargs(rc, case, flow) has every schema key (the case's own ones from the case, the others from the run's
    record), in the order of the case fields, for every model and window and for the reference cases."""
    for name, (dic, rc) in runs.items():
        cases = list(rc.cases())
        cases += rc.reference_cases() if flow == "deterministic" else \
            [c for m in rc.models for c in rc.reference_cases(for_model=m)]
        for case in cases:
            kw = case_kwargs(rc, case, flow)
            assert list(kw)[-len(CASE_FIELDS):] == list(CASE_FIELDS), (name, case.name)
            assert {f.key for f in schema.FIELDS} <= set(kw)
            assert (kw["case_name"], kw["verification_window"], kw["tolerance_days"]) == \
                (case.name, case.window.span, case.window.tolerance)
            model = case.model
            assert (kw["model"], kw["model_dir"], kw["accumulation"], kw["daily_boundary"], kw["model_dims"]) == \
                (model.name, model.dir, model.accumulation, model.daily_boundary, model.dims_map())
            assert (kw["years"], kw["years_clim"], kw["max_forecast_day"]) == \
                (rc.evaluation.years, rc.evaluation.climatology_years, rc.evaluation.max_forecast_day)
            assert kw["day_bins"] == rc.evaluation.day_bins   # the run's whole bins, in both flows
            for key in kw.keys() - set(CASE_FIELDS):
                if key in dic:
                    assert kw[key] == dic[key], (name, key)


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
            kw = case_kwargs(rc, case, flow)
            rc2, case2, selection = typed_io.from_case_kwargs(kw)
            assert selection == Selection()
            assert case2.reference == (case.reference and case.model.name not in dic["model_list"])
            assert case2.name == case.name and case2.window == case.window
            assert case_kwargs(rc2, case2, flow) == kw, (name, case.name)
            assert list(case_kwargs(rc2, case2, flow)) == list(kw)


def test_from_case_kwargs_keeps_edits_and_selections(runs):
    """A value edited in place (years, max_forecast_day, the reference keys) is what the rebuilt inputs hold; the
    selection keys come back as a Selection; anything else is refused."""
    dic, rc = runs["model_reference"]
    kw = case_kwargs(rc, rc.cases()[0], "deterministic")
    kw.update(years=(2019,), max_forecast_day=7, ref_model_var="other", keep_unit_ids=["a"], lat_select=(1, 2),
              model_files_years=(2019,))
    rc2, case2, selection = typed_io.from_case_kwargs(kw)
    assert rc2.evaluation.years == (2019,) and rc2.evaluation.max_forecast_day == 7 and rc2.reference.var == "other"
    assert selection.keep_unit_ids == ["a"] and selection.lat == (1, 2) and selection.model_files_years == (2019,)
    assert selection.grid_point is None
    assert case_kwargs(rc2, case2, "deterministic", selection) == kw
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
        assert case_kwargs(back, back.cases()[0], "ensemble") == \
            case_kwargs(rc, rc.cases()[0], "ensemble")
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


def test_importing_the_config_package_stays_light():
    code = ("import sys, hcwf.config, hcwf.config.model, hcwf.config.build\n"
            "heavy = [m for m in ('hcwf.io.observations', 'hcwf.blend.recipe', 'hcwf.spatial.masks') "
            "if m in sys.modules]\n"
            "assert not heavy, heavy\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
