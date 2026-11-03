"""The scoring flows loop over rc.cases() / rc.reference_cases() (step S5 of docs/review/runconfig_design.md).

Per flow, the cases given to compute_metrics (deterministic flow) and to prepare_pairs / create_score_results (ensemble
flow) are recorded and compared with rc.cases() / rc.reference_cases(), on 2 models x 2 windows with a model reference
and with a climatology reference."""
import dataclasses

import pandas as pd
import pytest

from hcwf.scoring import probabilistic as bin_skill_score
from hcwf.scoring import deterministic as spatial_far_mr_mae
from typed_io import case_kwargs, flat_config
from hcwf.config.model import RunConfig
from hcwf.scoring import pairs as skill
from hcwf.parallel import Pending
from config_world import configs, world  # noqa: F401  (fixtures: the fixture world and its configs)

REFERENCES = ("two_models_windows", "model_reference")   # climatology reference, model reference; 2 models x 2 windows
SWITCHES_OFF = ("save_nc_spatial_far_mr_mae", "plot_spatial_far_mr_mae", "save_nc_climatology", "plot_climatology_onset",
                "plot_panel_heatmap_error", "save_csv_score", "plot_heatmap_bss_auc", "plot_reliability",
                "plot_panel_heatmap_skill", "plot_bar_bss_rpss_auc")


def _run(configs, name, probabilistic):
    dic = {**flat_config(configs[name]), "probabilistic": probabilistic, **{k: False for k in SWITCHES_OFF}}
    return dic, RunConfig.from_flat(dic, check=False)


@pytest.mark.parametrize("name", REFERENCES)
def test_deterministic_flow_scores_the_cases_and_the_reference_cases(configs, monkeypatch, name):
    dic, rc = _run(configs, name, False)
    seen = []
    monkeypatch.setattr(spatial_far_mr_mae, "compute_metrics", lambda rc_, case: seen.append((rc_, case)) or ({}, {}))
    monkeypatch.setattr(spatial_far_mr_mae, "create_spatial_far_mr_mae", lambda *a: {})
    monkeypatch.setattr(spatial_far_mr_mae, "add_back_dated_map", lambda *a: None)
    spatial_far_mr_mae.spatial_far_mr_mae_map(rc)
    expected = list(rc.cases()) + list(rc.reference_cases())
    assert len(seen) == len(expected) == 2 * 2 + 2
    assert [case for _, case in seen] == expected
    assert [case.reference for _, case in seen] == [False] * 4 + [True] * 2
    for rc_, case in seen:
        assert rc_ is rc
        kw = case_kwargs(rc, case, "deterministic")
        assert kw["model"] == case.model.name and kw["verification_window"] == case.window.span
        assert kw["case_name"] == case.name and kw["tolerance_days"] == case.window.tolerance


@pytest.mark.parametrize("name", REFERENCES)
def test_ensemble_flow_scores_the_cases_with_the_windows_bins(configs, monkeypatch, name):
    dic, rc = _run(configs, name, True)
    prepared, created = [], []
    pairs = object()
    table = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(bin_skill_score, "prepare_pairs", lambda rc_, model: prepared.append((rc_, model)) or pairs)
    monkeypatch.setattr(bin_skill_score, "create_score_results",
                        lambda got, metrics, ev: created.append((got, metrics, ev)) or {})
    monkeypatch.setattr(bin_skill_score, "score_tables", lambda *a: (table, table, {}, {}))
    tables = bin_skill_score.skill_score_in_bins(rc)
    cases = list(rc.cases())
    assert len(created) == len(cases) == 2 * 2 and len(tables["overall"]) == 4
    for (got, metrics, ev), want in zip(created, cases):
        assert got is pairs and metrics is rc.metrics
        assert list(ev.day_bins) == list(want.window.bins(rc.evaluation.day_bins))
        assert ev == dataclasses.replace(rc.evaluation, day_bins=ev.day_bins)
    # the pair tables are prepared once per model
    assert [model.name for _, model in prepared] == [m.name for m in rc.models]
    assert all(rc_ is rc for rc_, _ in prepared)


def test_prepared_reference_is_the_case_builders(configs, monkeypatch):
    """The reference each model is scored against is RunConfig.reference_cases(for_model=model), the reference's own
    settings: the rule lives there only (scoring/pairs.py asks for it)."""
    dic, rc = _run(configs, "model_reference", True)
    read = []
    monkeypatch.setattr(skill, "start_pairs", lambda rc_, model, select=None: read.append(model) or Pending.finished(table))
    monkeypatch.setattr(skill.reference_io, "release_reference", lambda: None)
    table = pd.DataFrame({"a": [1]})
    for model in rc.models:
        read.clear()
        prepared = skill.prepare_pairs(rc, model)
        assert read == [model, rc.reference_cases(for_model=model)[0].model]
        assert prepared.reference is table and prepared.climatology is None
    # the ensemble flow names the reference's cases after the forecast model; its source is the reference's own
    for model in rc.models:
        for case in rc.reference_cases(for_model=model):
            assert case.reference and case.for_model == model.name and case.model == rc.reference.as_model()
            assert case_kwargs(rc, case, "ensemble")["model"] == rc.reference.model


def test_ensemble_flow_does_nothing_for_a_deterministic_run(configs):
    _, rc = _run(configs, "two_models_windows", False)
    assert bin_skill_score.skill_score_in_bins(rc) is None


@pytest.mark.parametrize("name", REFERENCES)
def test_one_process_frees_the_reference_rain_before_the_years_are_read(configs, monkeypatch, name):
    """Without worker processes the years run inside start_pairs, so the thresholds' reference rain is used (by the
    climatological onsets, or by nothing for a model reference) and released before the first of them starts."""
    dic, rc = _run(configs, name, True)
    rc = RunConfig.from_flat({**dic, "parallel": False}, check=False)
    steps, table = [], pd.DataFrame({"a": [1]})
    monkeypatch.setattr(skill, "start_pairs", lambda rc_, model, select=None: steps.append("pairs") or Pending.finished(table))
    monkeypatch.setattr(skill.reference_io, "release_reference", lambda: steps.append("release"))
    monkeypatch.setattr(skill, "climatological_onset_dataset", lambda *a: steps.append("climatology") or "onsets")
    monkeypatch.setattr(skill.thresholds_io, "thresholds_of", lambda *a: steps.append("thresholds"))
    monkeypatch.setattr(skill, "_climatology_pairs", lambda *a: table)
    skill.prepare_pairs(rc, rc.models[0])
    assert steps.index("release") < steps.index("pairs")
