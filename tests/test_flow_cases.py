"""The scoring flows loop over rc.cases() / rc.reference_cases() (step S5 of docs/review/runconfig_design.md).

Per flow, the sequence of keyword arguments given to compute_metrics_multiple_years (deterministic flow) and to
prepare_score_cache / create_score_results (ensemble flow) is recorded and compared with the frozen pre-R2 flows of
tests/_oracle_config.py, on 2 models x 2 windows with a model reference and with a climatology reference."""
import pandas as pd
import pytest

import _oracle_config as old
from hcwf.app import bin_skill_score, spatial_far_mr_mae
from hcwf.config import flat_bridge
from hcwf.config.build import resolve_flat
from hcwf.config.model import RunConfig
from hcwf.metrics import skill
from hcwf.stats.parallel import Pending
from test_oracle_config import configs, world  # noqa: F401  (fixtures: the fixture world and its configs)

REFERENCES = ("two_models_windows", "model_reference")   # climatology reference, model reference; 2 models x 2 windows
SWITCHES_OFF = ("save_nc_spatial_far_mr_mae", "plot_spatial_far_mr_mae", "save_nc_climatology", "plot_climatology_onset",
                "plot_panel_heatmap_error", "save_csv_score", "plot_heatmap_bss_auc", "plot_reliability",
                "plot_panel_heatmap_skill", "plot_bar_bss_rpss_auc")


def _run(configs, name, probabilistic):
    dic = {**resolve_flat(configs[name]), "probabilistic": probabilistic, **{k: False for k in SWITCHES_OFF}}
    return dic, RunConfig.from_flat(dic, check=False)


@pytest.mark.parametrize("name", REFERENCES)
def test_deterministic_flow_gives_the_oracles_kwargs(configs, monkeypatch, name):
    dic, rc = _run(configs, name, False)
    seen = []
    monkeypatch.setattr(spatial_far_mr_mae, "compute_metrics_multiple_years", lambda **kw: seen.append(kw) or ({}, {}))
    monkeypatch.setattr(spatial_far_mr_mae, "create_spatial_far_mr_mae", lambda *a: {})
    monkeypatch.setattr(spatial_far_mr_mae, "add_back_dated_map", lambda *a: None)
    spatial_far_mr_mae.spatial_far_mr_mae_map(rc)
    expected = [kw for _, kw in old.deterministic_cases(dic) + old.deterministic_reference_cases(dic)]
    assert len(seen) == len(expected) == 2 * 2 + 2
    for got, want in zip(seen, expected):
        assert got == want
        assert list(got) == list(want) or sorted(got) == sorted(want)


@pytest.mark.parametrize("name", REFERENCES)
def test_ensemble_flow_gives_the_oracles_kwargs(configs, monkeypatch, name):
    dic, rc = _run(configs, name, True)
    prepared, created = [], []
    cache = {"_forecast_obs_df_all": "forecast", "_ref_obs_df_all": "ref", "_clim_onset": None,
             "_climatology_obs_df_all": None}
    table = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(bin_skill_score, "prepare_score_cache", lambda **kw: prepared.append(kw) or dict(cache))
    monkeypatch.setattr(bin_skill_score, "create_score_results", lambda **kw: created.append(kw) or {})
    monkeypatch.setattr(bin_skill_score, "score_tables", lambda *a: (table, table, {}, {}))
    tables = bin_skill_score.skill_score_in_bins(rc)
    cases = old.ensemble_cases(dic)
    assert len(created) == len(cases) == 2 * 2 and len(tables["overall"]) == 4
    for got, (_, _, want) in zip(created, cases):
        assert got == {**want, **cache}
    # the pair tables are prepared once per model, from its first window, with every day bin
    firsts = {}
    for combi, prep, _ in cases:
        firsts.setdefault(combi[0], prep)
    assert [kw["model"] for kw in prepared] == list(firsts) == [m.name for m in rc.models]
    assert prepared == list(firsts.values())


def test_prepared_reference_is_the_case_builders(configs, monkeypatch):
    """The reference each model is scored against is RunConfig.reference_cases(like=model), with the ensemble rule:
    the rule lives there only (metrics/skill.py asks for it)."""
    dic, rc = _run(configs, "model_reference", True)
    read = []
    monkeypatch.setattr(skill, "start_pairs", lambda rc_, model, select=None: read.append(model) or Pending.finished(table))
    monkeypatch.setattr(skill.io_input, "release_reference", lambda: None)
    table = pd.DataFrame({"a": [1]})
    for model in rc.models:
        read.clear()
        prepared = skill.prepare_pairs(rc, model)
        assert read == [model, rc.reference_cases(like=model)[0].model]
        assert prepared.reference is table and prepared.climatology is None
    # ... and it is what the oracle's frozen reference_model_kwargs gives for each model's first window
    firsts = {}
    for combi, kw in old.ensemble_reference_cases(dic):
        firsts.setdefault(combi[0], kw)
    for model in rc.models:
        assert flat_bridge.case_kwargs(rc, rc.reference_cases(like=model)[0], "ensemble") == firsts[model.name]


def test_ensemble_flow_does_nothing_for_a_deterministic_run(configs):
    _, rc = _run(configs, "two_models_windows", False)
    assert bin_skill_score.skill_score_in_bins(rc) is None
