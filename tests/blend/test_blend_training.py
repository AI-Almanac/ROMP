"""momp.blend.training: the whole in-memory pipeline on a small synthetic region, from files to the fitted model.

The training parts of onset_blending's tests/test_pipeline.py (the trained-blend file and forecasting are later
steps); the end-to-end comparison with onset_blending's `train` is tests/blend/test_blend_training_differential.py.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _recipes import blending_config, translate

from momp.blend import logit
from momp.blend.predictions import Predictions
from momp.blend.samples import Weekly
from momp.blend.training import STAGE_TABLES, TrainingResult, prepare, resources, train

LATS, LONS = [10.0, 10.25], [70.0, 70.25]
OBS_YEARS = range(2000, 2014)
FORECAST_YEARS = range(2010, 2014)
LEAD_DAYS = 40


def onset_date(year: int, cell: int, shift: int = 0) -> dt.date:
    """Onsets between late May and early July, varying by year and cell."""
    return dt.date(year, 5, 25) + dt.timedelta(days=(year * 7 + cell * 5) % 40 + shift)


def rain(dates, onset: dt.date) -> np.ndarray:
    """Dry before the onset, 10 mm a day for its first 3 days, then 6 mm a day."""
    days = np.array([(d - onset).days for d in dates])
    return np.where(days < 0, 0.0, np.where(days < 3, 10.0, 6.0)).astype("float32")


def write_observations(folder):
    folder.mkdir()
    for year in OBS_YEARS:
        dates = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(200)]
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def write_forecasts(folder, years=FORECAST_YEARS, members=3):
    folder.mkdir(exist_ok=True)
    for year in years:
        issues = [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), members, LEAD_DAYS, len(LATS), len(LONS)), dtype="float32")
        for t, issue in enumerate(issues):
            leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
            for m in range(members):
                for i in range(len(LATS)):
                    for j in range(len(LONS)):
                        data[t, m, :, i, j] = rain(leads, onset_date(year, i * 2 + j, shift=3 * (m - 1)))
        xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                           "day": np.arange(1, LEAD_DAYS + 1), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def config_dict(root, **evaluation):
    return {
        "name": "synthetic",
        "data_root": str(root),
        "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}},
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                    "rain_predictors": [{"agg": "diff"}]}],
        "onset": {"threshold": {"mm": 20}, "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
        "climatology": {"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": False},
        "evaluation": {"years": {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013},
                       "formulas": {"blended_model": "outcome ~ prob_clim_qx + diff_ens_qx"}, **evaluation},
    }


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("region")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    translation = translate(blending_config(config_dict(root)), cache_dir=root / "cache")
    return root, translation, train(translation.cfg, translation.recipe, translation.setup)


def test_training_scores_the_blend_against_climatology(region):
    _, _, result = region
    pooled = result.evaluation.pooled.set_index("model")

    assert isinstance(result, TrainingResult)
    assert {"unc_clim_raw", "clim_raw", "blended_model"} <= set(pooled.index)
    assert result.evaluation.yearly["year"].nunique() == 4


def test_the_result_keeps_what_a_trained_blend_is_made_from(region):
    _, t, result = region

    assert result.recipe is t.recipe and result.cfg is t.cfg and result.setup is t.setup
    assert result.model is result.evaluation.final_model and result.model.formula.startswith("outcome ~")
    assert result.forest is None
    assert result.inputs.units == ("10.00_70.00", "10.00_70.25", "10.25_70.00", "10.25_70.25")
    assert result.inputs.observed.data is not None and result.inputs.thresholds is not None
    assert result.inputs.reference is None
    assert (result.inputs.climatology.train_years, result.inputs.climatology.test_years) == ((2000, 2009), (2010, 2013))
    assert set(result.inputs.timings) >= {"observed onsets", "ens forecast onsets", "climatology", "weekly features",
                                          "evaluation and final fit"}


def test_stage_tables_are_kept_only_when_asked(region):
    _, t, result = region

    assert all(f.data is None for f in result.inputs.forecasts.values())  # train releases them
    assert result.inputs.climatology.data is None
    kept = prepare(t.cfg, t.recipe, t.setup, keep=STAGE_TABLES)
    assert all(f.data is not None for f in kept.forecasts.values())
    assert kept.climatology.data is not None


def test_a_training_run_saves_netcdf_and_csv_artifacts(region, tmp_path):
    _, _, result = region
    out = result.save(tmp_path / "trained")
    tag = result.evaluation.output_tag

    assert not list(out.glob("*.pkl"))
    assert tag == "_fixed_cutoff_2010_2013"
    assert {p.name for p in out.iterdir()} == {
        f"formula_sample_support{tag}.json", f"cv_preds_blended_model_global{tag}.nc",
        f"coefs_blended_model_global{tag}.nc", f"summary_models{tag}.csv", f"summary_models_pooled{tag}.csv",
        f"yearly_metrics_global{tag}.csv", "weekly_features.nc"}
    preds = Predictions.open(out / f"cv_preds_blended_model_global{tag}.nc")
    xr.testing.assert_identical(preds.data, result.evaluation.cv_preds[("blended_model", "global")].data)
    with xr.open_dataset(out / f"coefs_blended_model_global{tag}.nc") as stored:
        folds = logit.fold_models(stored.load())
    assert set(folds) == set(result.evaluation.coefficients[("blended_model", "global")])
    weekly = Weekly.open(out / "weekly_features.nc")
    pd.testing.assert_frame_equal(weekly.frame(), result.inputs.weekly.frame(), check_exact=True)
    assert weekly.attrs == result.inputs.weekly.attrs
    pooled = pd.read_csv(out / f"summary_models_pooled{tag}.csv")
    assert list(pooled.columns[:8]) == ["id", "brier", "rps", "auc", "n", "lat", "lon", "pietra"]


def test_saving_is_all_or_nothing_and_keeps_chosen_intermediates(region, tmp_path):
    _, _, result = region
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x")

    with pytest.raises(FileExistsError):
        result.save(tmp_path / "taken")
    with pytest.raises(ValueError, match="unknown intermediate 'forecasts'"):
        result.save(tmp_path / "new", intermediates=("forecasts",))
    assert not (tmp_path / "new").exists()
    out = result.save(tmp_path / "empty-dir-is-fine", intermediates=("observed",))
    assert (out / "observed_onsets.nc").exists() and not (out / "weekly_features.nc").exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".") or ".tmp" in p.name] == []


def test_the_random_forest_blend_is_scored_and_fitted(region):
    root, _, _ = region
    translation = translate(blending_config(config_dict(root, random_forest={"trees": 5, "min_samples_leaf": 5})),
                            cache_dir=root / "cache")

    result = train(translation.cfg, translation.recipe, translation.setup)

    pooled = result.evaluation.pooled.set_index("model")
    assert {"blended_model", "blended_forest"} <= set(pooled.index)
    preds = result.evaluation.cv_preds[("blended_forest", "global")]
    assert np.allclose(preds.probabilities.sum(axis=1), 1.0)
    assert np.allclose(preds.data["daily"].sum("lead_day"), 1.0)
    assert result.forest is not None and result.inputs.forest_inputs is not None


def test_resources_follow_the_runs_parallel_settings():
    assert resources({"parallel": False})[0] == 1
    assert resources({"parallel_workers": 3, "memory_budget": 10})[0] == 3
    assert resources({"memory_budget": 10})[1] == 10
