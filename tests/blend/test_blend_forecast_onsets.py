"""Forecast onsets of a model's files are the same however the runs of issues are split into tasks, and `needed_only` keeps
exactly what the weekly features read of them."""
import differential_onsets as synthetic
import numpy as np
import pytest
import xarray as xr
from _recipes import converted

from hcwf.blend import features
from hcwf.io import forecasts as forecasts_io
from hcwf.io.thresholds import thresholds_of


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    synthetic.write_data(root)
    loaded = converted(synthetic.configs(root)["plain"])
    rc, recipe = loaded.rc(), loaded.recipe
    thresholds = thresholds_of(rc, rc.models[0])
    return rc, recipe, thresholds, features.reference_onset(recipe)


def test_runs_of_one_issue_give_what_whole_files_give(setup, monkeypatch):
    rc, recipe, thresholds, reference = setup
    name = recipe.models[0].name
    whole = features.forecast(rc, recipe, name, thresholds, reference).data
    monkeypatch.setattr(forecasts_io, "_RUN_VALUES", 1)
    split = features.forecast(rc, recipe, name, thresholds, reference).data
    xr.testing.assert_identical(whole, split)


def test_needed_only_keeps_what_the_features_read(setup):
    rc, recipe, thresholds, reference = setup
    name = recipe.models[0].name
    full = features.forecast(rc, recipe, name, thresholds, reference)
    kept = features.forecast(rc, recipe, name, thresholds, reference, needed_only=True)
    series, days = features._forecast_needs(recipe, recipe.model(name), True)
    assert set(kept.data) == {"members", "onset_threshold", "year", "rain_mean", *(f"count_{v}" for v in series)}
    n_day = kept.data.sizes["day"]
    assert n_day == (full.data.sizes["day"] if days is None else min(full.data.sizes["day"], days))
    for variable in kept.data:
        np.testing.assert_array_equal(kept.data[variable].values,
                                      full.data[variable].isel(day=slice(0, n_day)).values
                                      if "day" in full.data[variable].dims else full.data[variable].values)
