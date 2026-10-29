"""create_score_results scores a non-climatology reference from the reference model's own data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import momp.io.input as model_input
from momp.io.dict import extract_pd_bins
from momp.metrics.skill import create_score_results, prepare_score_cache
from momp.stats.bins import multi_year_climatological_forecast_obs_pairs
from momp.stats.climatology import compute_climatological_onset_dataset
from momp.stats.score import calculate_auc_climatology, calculate_brier_score_climatology, calculate_rps

YEARS = (2000, 2001, 2002)
LAT, LON = [10.0, 10.5], [70.0, 70.5]
DAY_BINS = ((1, 5), (6, 10))


def _write_obs(path):
    """Daily rain for May-July of each year, one file per year."""
    path.mkdir()
    for year in YEARS:
        rng = np.random.default_rng(year)
        time = pd.date_range(f"{year}-05-01", f"{year}-07-31")
        rain = rng.gamma(0.6, 12.0, size=(len(time), 2, 2))
        xr.Dataset({"rain": (("time", "lat", "lon"), rain)}, coords={"time": time, "lat": LAT, "lon": LON}) \
            .to_netcdf(path / f"obs_{year}.nc")


def _write_model(path, seed, steps=14):
    """A tiny ensemble archive (3 members, daily leads) with its own random rain, one file per year."""
    path.mkdir()
    for year in YEARS:
        rng = np.random.default_rng(seed * 1000 + year)
        init = pd.date_range(f"{year}-05-01", f"{year}-05-31", freq="D")
        shape = (len(init), 3, steps, 2, 2)
        xr.Dataset({"tp": (("time", "number", "step", "lat", "lon"), rng.gamma(0.6, 12.0, size=shape))},
                   coords={"time": init, "number": [0, 1, 2], "step": np.arange(1, steps + 1),
                           "lat": LAT, "lon": LON}).to_netcdf(path / f"m_{year}.nc")


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Keyword arguments of create_score_results for forecast model A, ready to point at model B or climatology."""
    monkeypatch.setattr(model_input, "region_select", lambda ds, **kwargs: ds)
    _write_obs(tmp_path / "obs")
    _write_model(tmp_path / "a", seed=1)
    _write_model(tmp_path / "b", seed=2)
    return dict(
        BS=True, RPS=True, AUC=True, skill_score=True, probabilistic=True, parallel=False, onset_legacy=True,
        model="A", model_dir=str(tmp_path / "a"), model_var="tp", file_pattern="m_{}.nc", unit_cvt=None,
        model_dims={"init_time": "time", "step": "step", "member": "number"},
        ref_model="B", ref_model_dir=str(tmp_path / "b"), ref_model_var="tp",
        ref_model_file_pattern="m_{}.nc", ref_model_unit_cvt=None,
        years=(2002,), years_clim=YEARS, obs_dir=str(tmp_path / "obs"), obs_file_pattern=["obs_{}.nc"],
        obs_var="rain", obs_unit_cvt=None, thresh_file=None, thresh_var=None, wet_threshold=20.0,
        date_filter_year=2000, init_days=(0, 2, 4), start_date=(2000, 5, 1), end_date=(2000, 5, 31),
        wet_init=1.0, wet_spell=2, dry_spell=1, dry_threshold=1.0, dry_extent=1, fallback_date=None,
        earliest_onset_date=None, members=None, onset_percentage_threshold=0.3, max_forecast_day=10,
        day_bins=DAY_BINS, region="x", nc_mask=None, check_alignment=False, grid_point=False,
    )


def _as_forecast(config, model):
    """The config with model "B" (or "A") as the forecast, so its own scores can be compared with a reference run."""
    if model == "A":
        return dict(config)
    return dict(config, model="B", model_dir=config["ref_model_dir"], ref_model_dir=config["model_dir"])


def _text(value):
    """CSV text of a table, or the repr of a score dict."""
    return value.to_csv() if hasattr(value, "to_csv") else repr(value)


def _scores(results, suffix=""):
    """BS, RPS and AUC of a results dict as comparable text."""
    return {name: _text(results[name + suffix]) for name in ("BS", "RPS", "AUC")}


def test_reference_scores_come_from_the_reference_model(config):
    """BS/RPS/AUC_ref differ from the forecast's own and equal the scores of the reference run as the forecast."""
    results = create_score_results(**config)
    as_forecast = create_score_results(**_as_forecast(config, "B"))

    assert _scores(results, "_ref") == _scores(as_forecast)
    assert _scores(results, "_ref") != _scores(results)
    assert results["climatology_obs_df"] is None


def test_cached_reference_pairs_are_the_reference_models(config):
    """prepare_score_cache stores the reference's own pairs, and create_score_results gives the same scores from it."""
    cache = prepare_score_cache(**config)

    assert not cache["_ref_obs_df_all"].equals(cache["_forecast_obs_df_all"])
    cached = create_score_results(**config, **cache)
    assert _scores(cached, "_ref") == _scores(create_score_results(**config), "_ref")


def test_forecast_model_dims_are_not_applied_to_the_reference(config, tmp_path):
    """A reference archive with other dim names is read by name inference, not with the forecast's model_dims."""
    renamed = tmp_path / "b_renamed"
    renamed.mkdir()
    for year in YEARS:
        with xr.open_dataset(tmp_path / "b" / f"m_{year}.nc") as ds:
            ds.rename({"time": "issued", "number": "realization"}).to_netcdf(renamed / f"m_{year}.nc")
    plain = create_score_results(**config)
    with_dims = create_score_results(**dict(config, ref_model_dir=str(renamed),
                                            ref_model_dims={"init_time": "issued", "member": "realization"}))
    assert _scores(with_dims, "_ref") == _scores(plain, "_ref")


def test_climatology_reference_is_unchanged(config):
    """The climatology reference equals the scores computed straight from the climatology pair tables."""
    config = dict(config, ref_model="climatology")
    results = create_score_results(**config)

    clim_onset = compute_climatological_onset_dataset(**config)
    pairs = extract_pd_bins(multi_year_climatological_forecast_obs_pairs(clim_onset, **config), DAY_BINS)

    assert results["climatology_obs_df"].to_csv() == pairs.to_csv()
    assert _text(results["BS_ref"]) == _text(calculate_brier_score_climatology(pairs))
    assert _text(results["RPS_ref"]) == _text(calculate_rps(pairs))
    assert _text(results["AUC_ref"]) == _text(calculate_auc_climatology(pairs))
    assert prepare_score_cache(**config)["_ref_obs_df_all"] is None
