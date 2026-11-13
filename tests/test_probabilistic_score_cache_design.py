import time
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

import momp.app.bin_skill_score as bin_skill_score
import momp.metrics.skill as skill
from momp.lib.convention import Setting


FULL_DAY_BINS = ((1, 15), (16, 30))
WINDOWS = (((1, 15),), ((16, 30),))


def _score_kwargs(day_bins):
    return {
        "BS": True,
        "RPS": True,
        "AUC": True,
        "skill_score": True,
        "ref_model": "climatology",
        "ref_model_dir": "unused-ref-dir",
        "ref_model_var": "unused-ref-var",
        "ref_model_file_pattern": "unused-ref-pattern",
        "ref_model_unit_cvt": None,
        "years": (2001, 2002),
        "years_clim": (1991, 1992),
        "obs_dir": "unused-obs-dir",
        "obs_file_pattern": "unused-obs-pattern",
        "obs_var": "unused-obs-var",
        "thresh_file": "unused-thresh-file",
        "thresh_var": "unused-thresh-var",
        "wet_threshold": 20.0,
        "date_filter_year": 2002,
        "init_days": (0,),
        "start_date": (2001, 5, 1),
        "end_date": (2002, 10, 31),
        "model_dir": "unused-model-dir",
        "model_var": "unused-model-var",
        "unit_cvt": None,
        "file_pattern": "unused-model-pattern",
        "wet_init": 1.0,
        "wet_spell": 3,
        "dry_spell": 0,
        "dry_threshold": 1.0,
        "dry_extent": 0,
        "fallback_date": None,
        "mok": None,
        "members": (1, 2, 3, 4),
        "onset_percentage_threshold": 0.5,
        "max_forecast_day": 30,
        "day_bins": day_bins,
        "parallel": False,
    }


def _synthetic_pairs(model_probs):
    rows = []
    forecasts = (
        ("2001-05-01", 10.0, 20.0, (1, 0)),
        ("2002-05-01", 10.0, 20.0, (0, 1)),
    )
    for forecast_index, (init_time, lat, lon, observations) in enumerate(forecasts):
        for bin_index, (bin_start, bin_end) in enumerate(FULL_DAY_BINS):
            rows.append(
                {
                    "init_time": pd.Timestamp(init_time),
                    "lat": lat,
                    "lon": lon,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "bin_index": bin_index,
                    "bin_label": f"Days {bin_start}-{bin_end}",
                    "predicted_prob": model_probs[forecast_index][bin_index],
                    "observed_onset": observations[bin_index],
                    "total_members": 4,
                    "total_members_with_onset": 4,
                }
            )
    return pd.DataFrame(rows)


def _result_summary(results):
    return {
        "forecast_bs": results["BS"]["fair_brier_score"],
        "forecast_rps": results["RPS"]["fair_rps"],
        "forecast_auc": results["AUC"]["auc"],
        "clim_bs": results["BS_ref"]["fair_brier_score"],
        "clim_rps": results["RPS_ref"]["fair_rps"],
        "clim_auc": results["AUC_ref"]["auc"],
        "bss": results["skill_results"]["fair_brier_skill_score"],
        "rpss": results["skill_results"]["fair_rps_skill_score"],
        "bin_bss": results["skill_results"]["bin_fair_brier_skill_scores"],
    }


@pytest.fixture
def synthetic_score_inputs():
    return {
        "forecast": _synthetic_pairs(((0.70, 0.30), (0.20, 0.80))),
        "climatology": _synthetic_pairs(((0.45, 0.55), (0.55, 0.45))),
        "clim_onset": object(),
    }


def test_skill_score_in_bins_reuses_full_bin_score_cache(monkeypatch):
    cfg = SimpleNamespace(
        probabilistic=True,
        layout=["model", "verification_window"],
        model_list=("model-a",),
        verification_window_list=((1, 15), (16, 30)),
        tolerance_days_list=(3, 3),
        day_bins=FULL_DAY_BINS,
        model_dir_list=("model-dir",),
        model_var_list=("model-var",),
        file_pattern_list=("model-pattern",),
        unit_cvt_list=(None,),
        members=(1, 2, 3, 4),
        years=(2001, 2002),
        years_clim=(1991, 1992),
        obs="obs",
        obs_var="obs-var",
        thresh_file="threshold.nc",
        thresh_var="threshold",
        wet_threshold=20.0,
        wet_init=1.0,
        wet_spell=3,
        dry_spell=0,
        dry_threshold=1.0,
        dry_extent=0,
        fallback_date=None,
        mok=None,
        onset_percentage_threshold=0.5,
        max_forecast_day=30,
        ref_model="climatology",
        ref_model_var=None,
        case_name=None,
        region=None,
    )
    setting = Setting(
        obs_dir="obs-dir",
        obs_file_pattern="obs-pattern",
        ref_model_dir="ref-dir",
        ref_model_file_pattern="ref-pattern",
        ref_model_unit_cvt=None,
        date_filter_year=2002,
        init_days=(0,),
        start_date=(2001, 5, 1),
        end_date=(2002, 10, 31),
        BS=True,
        RPS=True,
        AUC=True,
        skill_score=True,
        save_csv_score=True,
        plot_heatmap_bss_auc=False,
        plot_reliability=False,
        plot_panel_heatmap_skill=False,
        plot_bar_bss_rpss_auc=False,
        parallel=False,
    )

    prepared = []
    scored = []

    def prepare_cache(**kwargs):
        prepared.append(kwargs)
        return {"_forecast_obs_df_all": object()}

    def create_results(**kwargs):
        scored.append(kwargs)
        assert "_forecast_obs_df_all" in kwargs
        return {"forecast_obs_df": pd.DataFrame()}

    monkeypatch.setattr(bin_skill_score, "prepare_score_cache", prepare_cache)
    monkeypatch.setattr(bin_skill_score, "create_score_results", create_results)
    monkeypatch.setattr(
        bin_skill_score,
        "save_score_results",
        lambda score_results, **kwargs: ({}, {}),
    )

    bin_skill_score.skill_score_in_bins(cfg=cfg, setting=setting)

    assert len(prepared) == 1
    assert prepared[0]["day_bins"] == FULL_DAY_BINS
    assert [call["day_bins"] for call in scored] == [((1, 15),), ((16, 30),)]


def test_cached_score_inputs_preserve_windowed_results(monkeypatch, synthetic_score_inputs):
    """A cached full-bin dataframe should score exactly like recomputing per window."""
    call_counts = Counter()

    def forecast_pairs(**kwargs):
        call_counts["forecast"] += 1
        return synthetic_score_inputs["forecast"].copy()

    def clim_onset(**kwargs):
        call_counts["clim_onset"] += 1
        return synthetic_score_inputs["clim_onset"]

    def climatology_pairs(clim_onset, **kwargs):
        call_counts["climatology"] += 1
        assert clim_onset is synthetic_score_inputs["clim_onset"]
        return synthetic_score_inputs["climatology"].copy()

    monkeypatch.setattr(skill, "multi_year_forecast_obs_pairs", forecast_pairs)
    monkeypatch.setattr(skill, "compute_climatological_onset_dataset", clim_onset)
    monkeypatch.setattr(
        skill, "multi_year_climatological_forecast_obs_pairs", climatology_pairs
    )

    uncached = [
        _result_summary(skill.create_score_results(**_score_kwargs(window)))
        for window in WINDOWS
    ]

    assert call_counts == {"forecast": 2, "clim_onset": 2, "climatology": 2}

    cached_call_counts = Counter()

    def cached_forecast_pairs(**kwargs):
        cached_call_counts["forecast"] += 1
        return synthetic_score_inputs["forecast"].copy()

    def cached_clim_onset(**kwargs):
        cached_call_counts["clim_onset"] += 1
        return synthetic_score_inputs["clim_onset"]

    def cached_climatology_pairs(clim_onset, **kwargs):
        cached_call_counts["climatology"] += 1
        assert clim_onset is synthetic_score_inputs["clim_onset"]
        return synthetic_score_inputs["climatology"].copy()

    monkeypatch.setattr(skill, "multi_year_forecast_obs_pairs", cached_forecast_pairs)
    monkeypatch.setattr(skill, "compute_climatological_onset_dataset", cached_clim_onset)
    monkeypatch.setattr(
        skill, "multi_year_climatological_forecast_obs_pairs", cached_climatology_pairs
    )

    cached_inputs = skill.prepare_score_cache(**_score_kwargs(FULL_DAY_BINS))

    def fail_forecast_pairs(**kwargs):
        raise AssertionError("forecast pairs should come from the score input cache")

    def fail_clim_onset(**kwargs):
        raise AssertionError("climatological onset should come from the score input cache")

    def fail_climatology_pairs(clim_onset, **kwargs):
        raise AssertionError("climatology pairs should come from the score input cache")

    monkeypatch.setattr(skill, "multi_year_forecast_obs_pairs", fail_forecast_pairs)
    monkeypatch.setattr(skill, "compute_climatological_onset_dataset", fail_clim_onset)
    monkeypatch.setattr(
        skill, "multi_year_climatological_forecast_obs_pairs", fail_climatology_pairs
    )

    cached = [
        _result_summary(
            skill.create_score_results(**_score_kwargs(window), **cached_inputs)
        )
        for window in WINDOWS
    ]

    assert cached == uncached
    assert cached_call_counts == {"forecast": 1, "clim_onset": 1, "climatology": 1}


def test_cached_score_inputs_report_end_to_end_speedup(
    monkeypatch, synthetic_score_inputs, record_property
):
    """Report speedup for one full-bin preparation reused across two windows."""
    producer_delay_seconds = 0.02

    def delayed_forecast_pairs(**kwargs):
        time.sleep(producer_delay_seconds)
        return synthetic_score_inputs["forecast"].copy()

    def delayed_clim_onset(**kwargs):
        time.sleep(producer_delay_seconds)
        return synthetic_score_inputs["clim_onset"]

    def delayed_climatology_pairs(clim_onset, **kwargs):
        time.sleep(producer_delay_seconds)
        return synthetic_score_inputs["climatology"].copy()

    monkeypatch.setattr(skill, "multi_year_forecast_obs_pairs", delayed_forecast_pairs)
    monkeypatch.setattr(skill, "compute_climatological_onset_dataset", delayed_clim_onset)
    monkeypatch.setattr(
        skill, "multi_year_climatological_forecast_obs_pairs", delayed_climatology_pairs
    )

    start = time.perf_counter()
    uncached = [
        _result_summary(skill.create_score_results(**_score_kwargs(window)))
        for window in WINDOWS
    ]
    uncached_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    cached_inputs = skill.prepare_score_cache(**_score_kwargs(FULL_DAY_BINS))

    cached = [
        _result_summary(
            skill.create_score_results(**_score_kwargs(window), **cached_inputs)
        )
        for window in WINDOWS
    ]
    cached_elapsed = time.perf_counter() - start
    speedup = uncached_elapsed / cached_elapsed

    record_property("uncached_elapsed_seconds", uncached_elapsed)
    record_property("cached_elapsed_seconds", cached_elapsed)
    record_property("speedup", speedup)

    print(
        "\nprobabilistic score cache speedup: "
        f"{speedup:.2f}x "
        f"(uncached={uncached_elapsed:.4f}s, cached={cached_elapsed:.4f}s)"
    )

    assert cached == uncached
    assert speedup > 1.25
