"""momp.blend.random_forest: the CRPS random forest blend, on arrays, with onset-forest's numerics (a port of
onset_blending@1e6d696 tests/test_random_forest.py)."""
import numpy as np
import pandas as pd
import pytest

from momp.blend import random_forest
from momp.blend._crps_tree import CRPSForest

PARAMS = random_forest.Params(trees=6, min_samples_leaf=5, max_day=28, n_jobs=1)


def samples(n=400, seed=0):
    """Features whose first column moves the onset lead day, with gaps."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[rng.random((n, 6)) < 0.05] = np.nan
    lead = np.clip(np.round(10 + 6 * np.nan_to_num(X[:, 0]) + rng.normal(0, 2, n)), -3, 40)
    lead[rng.random(n) < 0.1] = np.nan  # no onset
    return X, lead


def test_the_target_is_the_onset_lead_day_after_the_issue():
    lead = np.array([5.0, 0.0, -2.0, np.nan, 31.0])

    keep, y = random_forest.target(lead, PARAMS)
    assert keep.tolist() == [True, False, False, True, True]
    assert y.tolist() == [5.0, 29.0, 31.0]  # no onset: past max_day

    keep, y = random_forest.target(lead, random_forest.Params(missing_onset="drop"))
    assert keep.tolist() == [True, False, False, False, True]


def test_probabilities_are_shares_of_lead_days_and_weeks_their_sums():
    X, lead = samples()
    fitted = random_forest.fit(X, lead, [f"x{i}" for i in range(6)], PARAMS)

    daily = random_forest.predict(fitted, X[:50])
    weeks = random_forest.weekly_from_daily(daily, 7)

    assert daily.shape == (50, 29) and np.allclose(daily.sum(axis=1), 1.0)
    assert np.allclose(weeks[:, 0], daily[:, :7].sum(axis=1))
    assert np.array_equal(weeks[:, -1], daily[:, -1])


def test_a_column_no_training_sample_observes_is_left_out():
    X, lead = samples()
    X[:, 3] = np.nan

    fitted = random_forest.fit(X, lead, [f"x{i}" for i in range(6)], PARAMS)

    assert fitted.features == ("x0", "x1", "x2", "x4", "x5")


def test_a_forest_as_arrays_predicts_the_same():
    X, lead = samples()
    fitted = random_forest.fit(X, lead, [f"x{i}" for i in range(6)], PARAMS)

    meta, arrays = fitted.to_arrays()
    again = random_forest.FittedForest.from_arrays(meta, arrays)

    assert np.array_equal(random_forest.predict(again, X), random_forest.predict(fitted, X))
    assert again.params == fitted.params


def test_folds_hold_out_true_holdouts_then_blocks_of_cv_years():
    folds = random_forest.folds(range(2000, 2008), [2005, 2004, 2006], [2007], years_per_fold=2)

    assert folds == [
        ((2007,), tuple(range(2000, 2007))),
        ((2004, 2005), (2000, 2001, 2002, 2003, 2006)),
        ((2006,), (2000, 2001, 2002, 2003, 2004, 2005)),
    ]


def test_cross_validation_seeds_each_fold_after_the_base_seed():
    X, lead = samples(600)
    year = np.repeat(np.arange(2000, 2006), 100)
    every = np.ones(len(X), dtype=bool)
    names = [f"x{i}" for i in range(6)]
    fold_years = random_forest.folds(range(2000, 2006), [2004, 2005])

    folds = random_forest.cross_validate(X, lead, names, year, every, every, fold_years, PARAMS)

    for index, ((test, daily), (test_years, train_years)) in enumerate(zip(folds, fold_years)):
        train = np.isin(year, train_years)
        seeded = random_forest.Params(**{**vars(PARAMS), "random_state": PARAMS.random_state + index + 1})
        expected = random_forest.predict(random_forest.fit(X[train], lead[train], names, seeded), X[test])
        assert np.array_equal(daily, expected)
        assert set(year[test]) == set(test_years)


def test_the_forest_is_onset_forests():
    onset_forest = pytest.importorskip("onset_forest")
    X, lead = samples()
    issue = pd.Timestamp("2020-06-01")
    frame = pd.DataFrame(X[:, :2], columns=["lat", "lon"])
    for d in (1, 2):
        frame[f"m_rain_mean_day_{d}"] = X[:, 1 + d]
    for d in range(1, 29):
        frame[f"clim_p_onset_day_{d}"] = X[:, 4] + d
    frame["time"] = issue
    frame["true_onset_date"] = [issue + pd.Timedelta(days=v) if np.isfinite(v) else pd.NaT for v in lead]
    legacy = onset_forest.fit(frame, ["m"], onset_forest.ForestParams(n_estimators=6, min_samples_leaf=5, n_jobs=1))

    names = list(frame.columns[:-2])
    fitted = random_forest.fit(frame[names].to_numpy(dtype=float), lead, names, PARAMS)

    expected = onset_forest.predict(legacy, frame)[list(onset_forest.DAILY_COLUMNS)].to_numpy()
    assert np.array_equal(random_forest.predict(fitted, frame[names].to_numpy(dtype=float)), expected)


def test_a_crps_forest_needs_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        CRPSForest(n_estimators=2).fit(np.array([[np.nan], [1.0]]), np.array([1.0, 2.0]))


def test_trees_grown_in_worker_processes_are_the_same_trees():
    X, lead = samples()
    names = [f"x{i}" for i in range(6)]

    alone = random_forest.fit(X, lead, names, PARAMS)
    pooled = random_forest.fit(X, lead, names, random_forest.Params(**{**vars(PARAMS), "n_jobs": 3}))

    assert np.array_equal(random_forest.predict(pooled, X), random_forest.predict(alone, X))
    (_, trees), (_, again) = alone.forest.to_arrays(), pooled.forest.to_arrays()
    for name, values in trees.items():
        np.testing.assert_array_equal(again[name], values)


def test_tree_workers_fit_the_memory_budget():
    assert random_forest.workers(1_000_000, 20, None) == 20
    per_worker = 300 * 2**20 + 1_000_000 * 96
    assert random_forest.workers(1_000_000, 20, 4 * per_worker) == 4
    assert random_forest.workers(1_000_000, 20, 1) == 1


class Inputs:
    """DailyInputs for two models' rainfall (lead days in a scrambled order) and the climatology."""

    def __init__(self, n=5):
        import xarray as xr

        rng = np.random.default_rng(3)
        self.time = np.datetime64("2001-06-01") + np.arange(n)
        self.data = xr.Dataset({"time": ("sample", self.time)})
        self.days = {"a": np.array([3, 1, 2]), "b": np.array([2, 1])}
        self.rains = {m: rng.random((n, len(d))) for m, d in self.days.items()}
        self.climatology = rng.random((n, 4))

    def rain(self, model):
        return self.days[model], self.rains[model]


class Samples(dict):
    def __init__(self, time):
        super().__init__(time=time, lat=np.arange(5.0), lon=np.arange(5.0) + 100)

    def take(self, name, rows):
        return self[name][rows]


def test_the_feature_matrix_orders_lat_lon_rain_days_then_climatology():
    inputs = Inputs()
    weekly = Samples(inputs.time)
    rows = np.array([4, 0, 2])

    names = random_forest.feature_names(inputs, ["a", "b"])
    X = random_forest.feature_matrix(weekly, inputs, rows, ["a", "b"])

    assert names == ["lat", "lon", "a_rain_mean_day_1", "a_rain_mean_day_2", "a_rain_mean_day_3",
                     "b_rain_mean_day_1", "b_rain_mean_day_2", *[f"clim_p_onset_day_{d}" for d in range(1, 5)]]
    assert X.shape == (3, len(names))
    np.testing.assert_array_equal(X[:, 0], [4.0, 0.0, 2.0])
    np.testing.assert_array_equal(X[:, 2], inputs.rains["a"][rows, 1])   # a's day 1 is its second column
    np.testing.assert_array_equal(X[:, 5], inputs.rains["b"][rows, 1])
    np.testing.assert_array_equal(X[:, -4:], inputs.climatology[rows])
    with pytest.raises(ValueError, match="not for these weekly samples"):
        random_forest.feature_matrix(Samples(inputs.time + 1), inputs, rows, ["a"])


def test_the_lead_day_histogram_counts_days_then_later():
    from momp.blend import _crps_tree

    counts = _crps_tree.lead_day_histogram(np.array([1.0, 1.0, 3.0, 5.0, 9.0]), 4)

    assert counts.tolist() == [2.0, 0.0, 1.0, 0.0, 2.0]
