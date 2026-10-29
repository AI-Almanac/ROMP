"""ROMP scores a conditional-climatology reference end to end on tiny synthetic data, and leaves the climatology reference alone."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import momp.io.input as model_input
from momp.io.dict import extract_pd_bins
from momp.metrics.skill import create_score_results, prepare_score_cache
from momp.stats.benchmark import compute_metrics_multiple_years
from momp.stats.bins import multi_year_climatological_forecast_obs_pairs
from momp.stats.climatology import compute_climatological_onset_dataset, compute_climatology_as_forecast
from momp.stats.conditional_climatology import (compute_conditional_climatology_as_forecast,
                                                create_conditional_climatological_forecast_obs_pairs)
from momp.stats.score import calculate_auc_climatology, calculate_brier_score_climatology, calculate_rps

CLIM_YEARS = tuple(range(2000, 2014))
TARGET = 2013
LAT, LON = [10.0, 10.5], [70.0, 70.5]
DAY_BINS = ((1, 5), (6, 10))


def _write_obs(path):
    """Daily rain for May-July of each year, one file per year, wet enough that every cell has an onset."""
    path.mkdir()
    for year in CLIM_YEARS:
        rng = np.random.default_rng(year)
        time = pd.date_range(f"{year}-05-01", f"{year}-07-31")
        wet = np.where(time.dayofyear.to_numpy()[:, None, None] > 135 + rng.integers(0, 25), 3.0, 0.3)
        rain = rng.gamma(0.8, 10.0, size=(len(time), 2, 2)) * wet
        xr.Dataset({"rain": (("time", "lat", "lon"), rain)}, coords={"time": time, "lat": LAT, "lon": LON}) \
            .to_netcdf(path / f"obs_{year}.nc")


def _write_model(path, steps=14):
    """A tiny ensemble archive (3 members, daily leads) for the target year."""
    path.mkdir()
    rng = np.random.default_rng(99)
    init = pd.date_range(f"{TARGET}-05-01", f"{TARGET}-05-31", freq="D")
    shape = (len(init), 3, steps, 2, 2)
    xr.Dataset({"tp": (("time", "number", "step", "lat", "lon"), rng.gamma(0.8, 12.0, size=shape))},
               coords={"time": init, "number": [0, 1, 2], "step": np.arange(1, steps + 1),
                       "lat": LAT, "lon": LON}).to_netcdf(path / f"m_{TARGET}.nc")


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Keyword arguments of the scoring functions for forecast model A against the conditional climatology."""
    monkeypatch.setattr(model_input, "region_select", lambda ds, **kwargs: ds)
    _write_obs(tmp_path / "obs")
    _write_model(tmp_path / "a")
    return dict(
        BS=True, RPS=True, AUC=True, skill_score=True, probabilistic=True, parallel=False, onset_legacy=True,
        model="A", model_dir=str(tmp_path / "a"), model_var="tp", file_pattern="m_{}.nc", unit_cvt=None,
        model_dims={"init_time": "time", "step": "step", "member": "number"},
        ref_model="conditional_climatology", ref_model_dir=str(tmp_path / "obs"), ref_model_var="rain",
        ref_model_file_pattern="obs_{}.nc", ref_model_unit_cvt=None,
        years=(TARGET,), years_clim=CLIM_YEARS, obs_dir=str(tmp_path / "obs"), obs_file_pattern=["obs_{}.nc"],
        obs_var="rain", obs_unit_cvt=None, thresh_file=None, thresh_var=None, wet_threshold=20.0,
        date_filter_year=2000, init_days=(0, 2, 4), start_date=(2000, 5, 1), end_date=(2000, 5, 31),
        wet_init=1.0, wet_spell=2, dry_spell=1, dry_threshold=1.0, dry_extent=1, fallback_date=None,
        earliest_onset_date=None, members=None, onset_percentage_threshold=0.3, max_forecast_day=10,
        day_bins=DAY_BINS, region="x", nc_mask=None, check_alignment=False, grid_point=False,
        save_nc_climatology=False, tolerance_days=3, verification_window=(1, 10),
    )


def test_probabilistic_reference_gives_skill_scores(config):
    """The conditional reference yields BS/RPS/AUC for the reference and skill scores against it."""
    results = create_score_results(**config)

    table = results["climatology_obs_df"]
    assert len(table) and set(table["bin_label"]) == {"Days 1-5", "Days 6-10"}
    assert np.isfinite(results["BS_ref"]["fair_brier_score"])
    assert np.isfinite(results["RPS_ref"]["fair_rps"])
    assert np.isfinite(results["AUC_ref"]["auc"])
    skill = results["skill_results"]
    assert np.isfinite(skill["fair_brier_skill_score"]) and np.isfinite(skill["fair_rps_skill_score"])
    assert prepare_score_cache(**config)["_ref_obs_df_all"] is None


def test_pair_table_probabilities_are_conditional(config):
    """Each case's probabilities (Before, bins, After) sum to one, Before is zero, and only inits before the onset are scored."""
    clim_onset = compute_climatological_onset_dataset(**config)
    inits = pd.date_range(f"{TARGET}-05-01", f"{TARGET}-07-30", freq="3D")
    table = create_conditional_climatological_forecast_obs_pairs(clim_onset, TARGET, inits, day_bins=DAY_BINS)

    assert len(table)
    assert set(table["bin_label"]) == {"Before day 1", "Days 1-5", "Days 6-10", "After day 10"}
    by_case = table.groupby(["init_time", "lat", "lon"])
    np.testing.assert_allclose(by_case["predicted_prob"].sum().to_numpy(), 1.0)
    assert (table.loc[table["bin_label"] == "Before day 1", "predicted_prob"] == 0).all()
    assert (pd.to_datetime(table["init_time"]) < pd.to_datetime(table["obs_onset_date"])).all()
    assert (by_case["observed_onset"].sum() == 1).all()
    assert np.isinf(table["total_members"]).all()


def test_leave_one_year_out_excludes_the_test_year(config):
    """The leave-one-year-out probabilities differ from the full fit, which saw the test year's onset."""
    clim_onset = compute_climatological_onset_dataset(**config)
    inits = pd.date_range(f"{TARGET}-05-10", f"{TARGET}-06-10", freq="4D")
    loyo = create_conditional_climatological_forecast_obs_pairs(
        clim_onset, TARGET, inits, day_bins=DAY_BINS, climatology_cv="leave_one_year_out")
    full = create_conditional_climatological_forecast_obs_pairs(
        clim_onset, TARGET, inits, day_bins=DAY_BINS, climatology_cv="none")
    assert loyo[["init_time", "lat", "lon", "bin_label"]].equals(full[["init_time", "lat", "lon", "bin_label"]])
    assert not np.allclose(loyo["predicted_prob"], full["predicted_prob"])
    with pytest.raises(ValueError, match="climatology_cv"):
        create_conditional_climatological_forecast_obs_pairs(
            clim_onset, TARGET, inits, day_bins=DAY_BINS, climatology_cv="sometimes")


def test_deterministic_reference_is_the_conditional_median(config):
    """The deterministic forecast is the first lead whose conditional CDF reaches 0.5, scored by the FAR/MR/MAE code."""
    config = dict(config, probabilistic=False, model="conditional_climatology")
    metrics, onsets = compute_metrics_multiple_years(**config)
    frame = metrics[TARGET]
    assert len(frame) and {"true_positive", "false_positive", "false_negative", "mae_combined"} <= set(frame.columns)
    assert TARGET in onsets and onsets[TARGET].dtype.kind == "M"

    clim_onset = compute_climatological_onset_dataset(**config)
    inits = pd.date_range(f"{TARGET}-05-01", f"{TARGET}-07-30", freq="5D")
    table = compute_conditional_climatology_as_forecast(
        clim_onset, TARGET, inits, onsets[TARGET], max_forecast_day=10, earliest_onset_date=None)
    forecast = table.dropna(subset=["onset_day"])
    assert len(forecast) and forecast["onset_day"].between(1, 10).all()
    pairs = create_conditional_climatological_forecast_obs_pairs(clim_onset, TARGET, inits, onsets[TARGET],
                                                                  day_bins=((1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                                                                            (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)))
    one_day = pairs[pairs["bin_label"].str.startswith("Days")]
    cdf = one_day.groupby(["init_time", "lat", "lon"], sort=False)["predicted_prob"].apply(
        lambda p: int(np.argmax(np.cumsum(p.to_numpy()) >= 0.5) + 1) if (np.cumsum(p.to_numpy()) >= 0.5).any() else np.nan)
    row = forecast.iloc[0]
    key = (pd.Timestamp(row["init_time"]).strftime("%Y-%m-%d"), row["lat"], row["lon"])
    assert cdf.loc[key] == row["onset_day"]


def test_climatology_reference_is_unchanged(config):
    """ref_model "climatology" gives the same tables and scores as the climatology functions directly, with or without a climatology_cv."""
    config = dict(config, ref_model="climatology")
    results = create_score_results(**config)
    clim_onset = compute_climatological_onset_dataset(**config)
    pairs = extract_pd_bins(multi_year_climatological_forecast_obs_pairs(clim_onset, **config), DAY_BINS)

    assert results["climatology_obs_df"].to_csv() == pairs.to_csv()
    assert repr(results["BS_ref"]) == repr(calculate_brier_score_climatology(pairs))
    assert repr(results["RPS_ref"]) == repr(calculate_rps(pairs))
    assert repr(results["AUC_ref"]) == repr(calculate_auc_climatology(pairs))
    again = create_score_results(**dict(config, climatology_cv="none"))
    assert again["climatology_obs_df"].to_csv() == results["climatology_obs_df"].to_csv()

    det = dict(config, probabilistic=False, model="climatology")
    metrics, onsets = compute_metrics_multiple_years(**det)
    init_dates = model_input.get_initialization_dates(TARGET, **det)
    direct = compute_climatology_as_forecast(onsets[TARGET], TARGET, init_dates,
                                             _observed(det), max_forecast_day=10, earliest_onset_date=None)
    assert len(metrics[TARGET]) == direct[["lat", "lon"]].drop_duplicates().shape[0]


def _observed(config):
    """The target year's observed onsets (lat, lon)."""
    from momp.stats.detect import detect_observed_onset
    return detect_observed_onset(model_input.load_imd_rainfall(TARGET, **config), model_input.load_thresholds(**config),
                                 TARGET, **config)


def test_conditional_reference_scores_better_than_unconditional(config):
    """Conditioning on no onset yet gives a lower Brier and RPS than the unconditional climatology on the same cases."""
    cond = create_score_results(**config)
    plain = create_score_results(**dict(config, ref_model="climatology"))
    print(cond["BS_ref"]["fair_brier_score"], plain["BS_ref"]["fair_brier_score"],
          cond["RPS_ref"]["fair_rps"], plain["RPS_ref"]["fair_rps"])
    assert cond["BS_ref"]["fair_brier_score"] < plain["BS_ref"]["fair_brier_score"]
    assert cond["RPS_ref"]["fair_rps"] < plain["RPS_ref"]["fair_rps"]
