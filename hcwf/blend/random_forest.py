"""The CRPS random forest blend: daily onset probabilities from each model's daily rainfall and the climatology.

Its samples are the weekly samples (`hcwf.blend.samples.Weekly`), and its inputs per sample
(`hcwf.blend.samples.DailyInputs`) are the unit's lat and lon, each model's ensemble-mean rainfall for every lead
day, and the conditional climatology's onset probability for lead days ``1..max_day``. It learns the observed
onset's lead day after the issue date (seasons without an onset count as a lead day past ``max_day``, or are left
out), and predicts, per sample, the share of its trees' leaf members on each lead day ``1..max_day`` and later; the
week bins are their sums.

Everything is arrays. The numerics are onset-forest's (its ``fit``, ``predict`` and ``cross_validate``,
``github.com/AI-Almanac/onset-forest`` at 61a7f5a): the same feature columns in the same order, the same median
imputation and targets, and the same trees (`hcwf.blend._crps_tree`), so a seeded forest reproduces its predictions
exactly. `hcwf.blend.evaluation` cross-validates and fits it, and `hcwf.blend.forecast` applies it to live files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hcwf import parallel as pool

from ._crps_tree import CRPSForest
from .samples import DailyInputs, Weekly

__all__ = ["FittedForest", "Params", "cross_validate", "feature_matrix", "fit", "folds", "predict", "weekly_from_daily"]


@dataclass(frozen=True)
class Params:
    """How the forest blend is fitted (onset-forest's ``ForestParams``) and cross-validated."""

    name: str = "blended_forest"
    """The name of the blend in the evaluation's outputs."""
    trees: int = 100
    """The number of trees."""
    min_samples_leaf: int = 20
    """The fewest training samples in a leaf."""
    max_features: str | None = "sqrt"
    """The features tried at each split (``"sqrt"``, ``"log2"``, ...); ``None`` tries all."""
    max_samples: float = 1.0
    """The share of the training samples each tree is grown on."""
    max_depth: int | None = None
    """The deepest a tree may grow; ``None`` is unlimited."""
    random_state: int = 123
    """The seed; cross-validation fold ``i`` uses ``random_state + i + 1``."""
    max_day: int = 28
    """The last lead day with its own probability; later onsets share the final ``later`` column."""
    days_per_week: int = 7
    """The lead days that make one week bin."""
    missing_onset: str = "after_week4"
    """What a season without an onset is: ``"after_week4"`` (lead day ``max_day + 1``) or ``"drop"``."""
    years_per_fold: int = 1
    """The consecutive years held out in each cross-validation fold."""
    n_jobs: int = -1
    """The worker processes to grow trees in; -1 takes what the run allows."""


def feature_names(inputs: DailyInputs, model_names) -> list[str]:
    """Return the names of the forest's features, in column order.

    Parameters
    ----------
    inputs : hcwf.blend.samples.DailyInputs
        The forest's inputs: each model's rain lead days and the climatology matrix.
    model_names : iterable of str
        The models, in the order of the feature columns.

    Returns
    -------
    list of str
        ``lat``, ``lon``, each model's ``<model>_rain_mean_day_<d>`` with the days ascending, then
        ``clim_p_onset_day_<d>`` for every climatology day.
    """
    names = ["lat", "lon"]
    for model in model_names:
        days, _ = inputs.rain(model)
        names += [f"{model}_rain_mean_day_{int(d)}" for d in np.sort(days)]
    names += [f"clim_p_onset_day_{d}" for d in range(1, inputs.climatology.shape[1] + 1)]
    return names


def feature_matrix(weekly: Weekly, inputs: DailyInputs, rows: np.ndarray, model_names) -> np.ndarray:
    """Return the features of some samples, in the column order of `feature_names`.

    Parameters
    ----------
    weekly : hcwf.blend.samples.Weekly
        The weekly samples (``lat``, ``lon`` and ``time`` are read).
    inputs : hcwf.blend.samples.DailyInputs
        The forest's inputs, one row per sample of `weekly`.
    rows : numpy.ndarray of int, shape (row,)
        The positions of the samples to take.
    model_names : iterable of str
        The models, in the order of the feature columns.

    Returns
    -------
    numpy.ndarray of float64, shape (row, feature)
        The features, ``NaN`` where missing.

    Raises
    ------
    ValueError
        The times of `inputs` differ from those of `weekly`.
    """
    if not np.array_equal(inputs.data["time"].values, weekly["time"]):
        raise ValueError("the forest's inputs are not for these weekly samples")
    blocks = [weekly.take("lat", rows)[:, None], weekly.take("lon", rows)[:, None]]
    for model in model_names:
        days, rain = inputs.rain(model)
        blocks.append(rain[rows][:, np.argsort(days, kind="stable")])
    blocks.append(inputs.climatology[rows])
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], axis=1)


def target(onset_lead: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """Find the samples the forest learns from and their target lead day.

    The samples are those with an onset after the issue date; a season without an onset counts as lead day
    ``max_day + 1`` (``missing_onset="after_week4"``) or is left out (``"drop"``).

    Parameters
    ----------
    onset_lead : numpy.ndarray of float64, shape (sample,)
        The observed onset's lead day after the issue date, ``NaN`` for a season without an onset.
    params : Params
        The settings (``max_day`` and ``missing_onset``).

    Returns
    -------
    keep : numpy.ndarray of bool, shape (sample,)
        The samples to learn from.
    lead : numpy.ndarray of float64, shape (n,)
        Their lead days, ``n = keep.sum()``.

    Raises
    ------
    ValueError
        ``params.missing_onset`` is neither ``"after_week4"`` nor ``"drop"``.
    """
    lead = np.asarray(onset_lead, dtype=np.float64).copy()
    if params.missing_onset == "after_week4":
        lead[np.isnan(lead)] = float(params.max_day + 1)
    elif params.missing_onset != "drop":
        raise ValueError("missing_onset must be 'after_week4' or 'drop'.")
    keep = ~np.isnan(lead) & (lead > 0)
    return keep, lead[keep]


@dataclass(frozen=True, eq=False)
class FittedForest:
    """A fitted forest: the features it reads (those observed in training), their training medians, and the trees."""

    features: tuple[str, ...]
    """The names of the features the forest reads, those observed in training."""
    columns: np.ndarray
    """The positions of `features` among the full feature names, ``int64`` of shape ``(feature,)``."""
    medians: np.ndarray
    """The training median of each feature (imputing missing values), ``float64`` of shape ``(feature,)``."""
    forest: CRPSForest
    """The fitted trees."""
    params: Params
    """The settings the forest was fitted with."""

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Split the forest into JSON-ready metadata and named arrays, for saving.

        Returns
        -------
        meta : dict
            ``features`` (list of names), ``forest`` (the trees' metadata) and ``params`` (a dict).
        arrays : dict of str to numpy.ndarray
            The trees' arrays, each as ``tree_<name>``, with ``columns`` and ``medians``.
        """
        meta, arrays = self.forest.to_arrays()
        params = {k: v for k, v in vars(self.params).items()}
        return (
            {"features": list(self.features), "forest": meta, "params": params},
            {**{f"tree_{k}": v for k, v in arrays.items()}, "columns": self.columns, "medians": self.medians},
        )

    @classmethod
    def from_arrays(cls, meta: dict, arrays) -> FittedForest:
        """Rebuild a forest from what `FittedForest.to_arrays` returned.

        Parameters
        ----------
        meta : dict
            The metadata.
        arrays : mapping of str to numpy.ndarray
            The arrays, with the trees' under ``tree_<name>``.

        Returns
        -------
        FittedForest
            The forest.
        """
        trees = {k[len("tree_"):]: arrays[k] for k in arrays if k.startswith("tree_")}
        return cls(
            features=tuple(meta["features"]),
            columns=np.asarray(arrays["columns"], dtype=np.int64),
            medians=np.asarray(arrays["medians"], dtype=np.float64),
            forest=CRPSForest.from_arrays(meta["forest"], trees),
            params=Params(**meta["params"]),
        )


_WORKER_BASE_BYTES = 300 * 2**20
"""The memory of a tree-growing worker that does not depend on the data: its interpreter and compiled kernels."""
_WORKER_BYTES_PER_ROW = 96
"""The memory of a tree-growing worker per training row: the node row sets, masks and targets of a tree's deepest path."""


def workers(train_rows: int, workers: int, budget_bytes: int | None) -> int:
    """Return the number of processes to grow trees in.

    Parameters
    ----------
    train_rows : int
        The number of training rows, which sets each worker's memory.
    workers : int
        The most processes wanted.
    budget_bytes : int or None
        The memory the processes may use; ``None`` for no limit.

    Returns
    -------
    int
        Up to `workers`, as many as the budget holds (`hcwf.parallel.workers_within`).
    """
    return pool.workers_within([_WORKER_BASE_BYTES + train_rows * _WORKER_BYTES_PER_ROW], workers, budget_bytes)


def fit(X: np.ndarray, onset_lead: np.ndarray, names, params: Params) -> FittedForest:
    """Fit the forest on a feature matrix, as ``onset_forest.fit`` does.

    The training samples are those of `target`; the columns they observe (not all ``NaN``) are kept, and missing
    values are imputed with the column median.

    Parameters
    ----------
    X : numpy.ndarray of float64, shape (sample, feature)
        The features of `feature_matrix`.
    onset_lead : numpy.ndarray of float64, shape (sample,)
        The observed onset's lead day after the issue date, ``NaN`` for a season without an onset.
    names : sequence of str
        The feature names, one per column of `X`.
    params : Params
        The settings.

    Returns
    -------
    FittedForest
        The fitted forest.

    Raises
    ------
    ValueError
        No sample has an onset after its issue date.
    """
    keep, y = target(onset_lead, params)
    if not keep.any():
        raise ValueError("No training rows have an onset after their issue date.")
    rows = X[keep]
    observed = np.flatnonzero(~np.isnan(rows).all(axis=0))
    rows = rows[:, observed]
    medians = np.nanmedian(rows, axis=0)
    filled = np.where(np.isnan(rows), medians, rows)
    forest = CRPSForest(
        n_estimators=params.trees,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        max_samples=params.max_samples,
        max_depth=params.max_depth,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
    ).fit(filled, y)
    return FittedForest(tuple(names[i] for i in observed), observed, medians, forest, params)


def predict(fitted: FittedForest, X: np.ndarray) -> np.ndarray:
    """Predict the daily onset probabilities.

    Parameters
    ----------
    fitted : FittedForest
        The fitted forest.
    X : numpy.ndarray of float64, shape (sample, feature)
        The features, with all the columns of `feature_matrix`; ``NaN`` is imputed with the training median.

    Returns
    -------
    numpy.ndarray of float64, shape (sample, max_day + 1)
        The probability of each lead day ``1..max_day``, then of a later onset.
    """
    rows = X[:, fitted.columns]
    filled = np.where(np.isnan(rows), fitted.medians, rows)
    return fitted.forest.predict_lead_day_distribution(filled, max_day=fitted.params.max_day)


def weekly_from_daily(daily: np.ndarray, days_per_week: int) -> np.ndarray:
    """Sum daily probabilities into week bins.

    Week ``w`` sums lead days ``days_per_week * (w - 1) + 1 .. days_per_week * w``; the last column (later) stays.

    Parameters
    ----------
    daily : numpy.ndarray of float64, shape (sample, day + 1)
        The daily probabilities, the last column being later.
    days_per_week : int
        The lead days in a week.

    Returns
    -------
    numpy.ndarray of float64, shape (sample, week + 1)
        The week bins, then later.

    Raises
    ------
    ValueError
        The daily columns (without later) do not divide into weeks.
    """
    n_days = daily.shape[1] - 1
    if n_days % days_per_week != 0:
        raise ValueError(f"{n_days} daily columns do not divide into {days_per_week}-day weeks.")
    weeks = daily[:, :n_days].reshape(len(daily), n_days // days_per_week, days_per_week).sum(axis=2)
    return np.column_stack([weeks, daily[:, n_days]])


def folds(training_years, cv_holdout_years, true_holdout_years=(), years_per_fold: int = 1):
    """Lay out the cross-validation folds.

    The true holdouts come first, one fold each, then the CV holdouts in contiguous blocks of up to
    `years_per_fold` (sorted first when that is above 1); each fold trains on the training years minus its test years
    and every true holdout.

    Parameters
    ----------
    training_years : iterable of int
        The years that may be trained on.
    cv_holdout_years : iterable of int
        The years held out in turn by cross-validation.
    true_holdout_years : iterable of int, default ()
        The years never trained on, each tested alone.
    years_per_fold : int, default 1
        The consecutive CV holdout years per fold.

    Returns
    -------
    list of tuple of (tuple of int, tuple of int)
        For each fold, the test years and the sorted training years.
    """
    true_holdouts = [int(year) for year in true_holdout_years or ()]
    cv_years = [int(year) for year in cv_holdout_years]
    if years_per_fold > 1:
        cv_years = sorted(cv_years)
    blocks = [(year,) for year in true_holdouts] + [
        tuple(cv_years[start:start + years_per_fold]) for start in range(0, len(cv_years), years_per_fold)
    ]
    return [
        (block, tuple(sorted({int(y) for y in training_years if int(y) not in block and int(y) not in true_holdouts})))
        for block in blocks
    ]


def cross_validate(X: np.ndarray, onset_lead: np.ndarray, names, year: np.ndarray, trainable: np.ndarray,
                   scoreable: np.ndarray, fold_years, params: Params) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute out-of-fold daily probabilities.

    Each fold's forest is fitted on the trainable samples of its training years and predicts the scoreable samples of
    its test years. Fold ``i`` is seeded ``random_state + i + 1``; folds without test samples are skipped.

    Parameters
    ----------
    X : numpy.ndarray of float64, shape (sample, feature)
        The features of `feature_matrix`.
    onset_lead : numpy.ndarray of float64, shape (sample,)
        The observed onset's lead day after the issue date, ``NaN`` for a season without an onset.
    names : sequence of str
        The feature names, one per column of `X`.
    year : numpy.ndarray of int, shape (sample,)
        The year of each sample.
    trainable : numpy.ndarray of bool, shape (sample,)
        The samples a fit may train on.
    scoreable : numpy.ndarray of bool, shape (sample,)
        The samples to predict.
    fold_years : iterable of tuple of (sequence of int, sequence of int)
        The ``(test years, train years)`` of each fold, as `folds` returns.
    params : Params
        The settings.

    Returns
    -------
    list of tuple of (numpy.ndarray, numpy.ndarray)
        For each fold with test samples, the positions of its samples (``int64``, shape ``(n,)``) and their daily
        probabilities (``float64``, shape ``(n, max_day + 1)``).

    Raises
    ------
    ValueError
        No scoreable sample falls in the test years of any fold.
    """
    out = []
    for index, (test_years, train_years) in enumerate(fold_years):
        test = np.flatnonzero(scoreable & np.isin(year, test_years))
        if not len(test):
            continue
        train = np.flatnonzero(trainable & np.isin(year, train_years))
        fitted = fit(X[train], onset_lead[train], names, replace(params, random_state=params.random_state + index + 1))
        out.append((test, predict(fitted, X[test])))
    if not out:
        raise ValueError("No predict_on rows fall in any fold's test year.")
    return out
