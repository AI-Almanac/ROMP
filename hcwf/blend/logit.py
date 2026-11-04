"""The multinomial logistic blend: formula features, standardised, into a multinomial logistic regression.

The blend predicts onset probabilities over the week bins. `FittedModel` is a fitted blend (it converts to and from
the coefficient bundle the legacy scripts pickled, and to plain arrays and JSON-able metadata; `fold_coefficients`
stacks CV folds' models into one `xarray.Dataset`); `fit_samples` fits one on weekly samples
(`hcwf.blend.samples.Weekly`), from the arrays when the formula has a design plan (`hcwf.blend.design.Plan`).

Cross-validation, on samples of the weekly features:

* `global_cv`: leave one year out, pooling every unit;
* `local_cv`: leave one year out within each unit;
* `cluster_cv`: leave one year out within each unit's cluster (one fit per cluster and year, predicting each of its
  units).

A formula without a design plan (functions, categorical terms) builds a Patsy design per fit instead, from the fit's
own training rows, as the legacy CV did. Global CV and the final fit share one design matrix: a formula's design for
every evaluation sample is built once, column-major, in shared memory that the fold workers and the final fit attach
to. Each fits its rows as the legacy ``compute_cv_global`` and ``fit_final`` did, bit for bit, while holding one copy
of its rows rather than two or three:

* a fold computes its scaler statistics a column at a time (a design column is contiguous, so the sums run as on the
  fold's own column-major design) and writes its standardised rows straight into the row-major matrix the solver
  reads;
* the final fit gathers its rows row-major, as ``fit()`` did, and standardises them in place.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
import sklearn
import xarray as xr
from packaging.version import Version
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hcwf.parallel import map_items
from hcwf import parallel as pool

from . import design
from .design import Plan
from .samples import Weekly

__all__ = []


log = logging.getLogger(__name__)

_SCALER_ARRAYS = ("scaler_mean", "scaler_scale", "scaler_var")
"""Names of the three arrays that hold a fitted ``StandardScaler``'s statistics."""


@dataclass(frozen=True, eq=False)
class FittedModel:
    """A multinomial logistic model on standardised formula features.

    Without a scaler (coefficient pickles from before the bundle format), the coefficients apply to the unscaled
    features. The arrays are copied to read-only C-ordered ``float64`` arrays when the model is made.
    """

    formula: str
    """The expanded formula that was fitted."""
    features: tuple[str, ...]
    """The design-matrix columns, in coefficient order."""
    classes: tuple[str, ...]
    """The outcome classes, in coefficient order."""
    coef: np.ndarray
    """The coefficients, ``float64`` of shape (class, feature)."""
    intercept: np.ndarray
    """The intercepts, ``float64`` of shape (class,)."""
    scaler_mean: np.ndarray | None = None
    """The scaler's column means, shape (feature,); None for a model without a scaler."""
    scaler_scale: np.ndarray | None = None
    """The scaler's column scales (standard deviations), shape (feature,)."""
    scaler_var: np.ndarray | None = None
    """The scaler's column variances, shape (feature,); kept so the legacy scaler is rebuilt exactly."""
    n_samples: int | None = None
    """The number of rows the scaler was fitted on; set with, and only with, a scaler."""

    def __post_init__(self):
        """Validate the fields and freeze them as read-only arrays and tuples.

        Raises
        ------
        ValueError
            The scaler arrays are only partly given, an array has the wrong shape, or `n_samples` is set without a
            scaler (or missing with one).
        """
        set_ = object.__setattr__
        set_(self, "features", tuple(self.features))
        set_(self, "classes", tuple(self.classes))
        n_classes, n_features = len(self.classes), len(self.features)
        shapes = {"coef": (n_classes, n_features), "intercept": (n_classes,)}
        has_scaler = self.scaler_mean is not None
        if has_scaler:
            shapes.update({name: (n_features,) for name in _SCALER_ARRAYS})
        elif any(getattr(self, name) is not None for name in _SCALER_ARRAYS):
            raise ValueError("scaler_mean, scaler_scale and scaler_var go together.")
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None or np.shape(value) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {np.shape(value)}.")
            # C order, like apply_blend_model.py's coefficient matrix, so that
            # predict's matrix product runs the same BLAS kernel bit for bit.
            array = np.array(value, dtype=float, order="C")
            array.flags.writeable = False
            set_(self, name, array)
        if has_scaler != (self.n_samples is not None):
            raise ValueError("n_samples is required with, and only with, a scaler.")
        if has_scaler:
            set_(self, "n_samples", int(self.n_samples))

    def predict(self, data: Mapping | pd.DataFrame) -> np.ndarray:
        """Return the class probabilities for the rows of ``data``.

        Rows with a non-finite feature get NaN. As ``apply_blend_model.py``: the design is built from the saved formula
        and must have exactly the saved features, at least one of them finite.

        Parameters
        ----------
        data : collections.abc.Mapping or pandas.DataFrame
            The feature columns: a table or a mapping of feature arrays, read by the formula.

        Returns
        -------
        numpy.ndarray of float64, shape (row, class)
            The probabilities, in `classes` order; ``NaN`` for a row with a non-finite feature.

        Raises
        ------
        ValueError
            The design matrix does not have exactly the saved features, or a saved feature has no finite value in
            ``data``.
        """
        features = list(self.features)
        names, X = design_matrix(self.formula, data)
        missing = [f for f in features if f not in names]
        unexpected = [f for f in names if f not in features]
        if missing or unexpected:
            raise ValueError(
                "Applied design matrix does not match the saved training schema. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )
        X = X[:, [names.index(f) for f in features]]
        all_nonfinite = [f for f, column in zip(features, X.T) if not np.isfinite(column).any()]
        if all_nonfinite:
            raise ValueError(
                "Saved model features have no finite values in the application "
                f"data: {all_nonfinite}."
            )
        if self.scaler_mean is not None:
            X = self._scaler().transform(X)

        scores = X @ self.coef.T + self.intercept
        bad = ~np.isfinite(X).all(axis=1)
        probs = np.full((len(X), len(self.classes)), np.nan)
        if (~bad).any():
            s = scores[~bad]
            s -= s.max(axis=1, keepdims=True)
            exp_s = np.exp(s)
            probs[~bad] = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probs

    @classmethod
    def from_estimators(cls, formula: str, features, clf, scaler) -> FittedModel:
        """Make a model from the fitted classifier and the scaler of its features.

        Parameters
        ----------
        formula : str
            The expanded formula that was fitted.
        features : sequence of str
            The design-matrix columns, in the order the classifier was fitted on.
        clf : sklearn.linear_model.LogisticRegression
            The fitted classifier (its ``classes_``, ``coef_`` and ``intercept_`` are used).
        scaler : sklearn.preprocessing.StandardScaler
            The scaler fitted on the training features.

        Returns
        -------
        FittedModel
            The model with the classifier's coefficients and the scaler's statistics.
        """
        return cls(
            formula=formula,
            features=features,
            classes=clf.classes_,
            coef=clf.coef_,
            intercept=clf.intercept_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            scaler_var=scaler.var_,
            n_samples=int(scaler.n_samples_seen_),
        )

    def to_legacy_bundle(self) -> dict:
        """Return the coefficient bundle ``3_fit_final_model.py`` pickles.

        Returns
        -------
        dict
            ``coefs`` (a `pandas.DataFrame` with the columns ``class``, ``feature``, ``coefficient`` and
            ``intercept``, one row per class and feature), ``scaler`` (a ``StandardScaler``, or None without
            scaler), ``features`` (list of str) and ``formula``.
        """
        rows = [
            {
                "class": cls,
                "feature": feature,
                "coefficient": float(self.coef[i, j]),
                "intercept": float(self.intercept[i]),
            }
            for i, cls in enumerate(self.classes)
            for j, feature in enumerate(self.features)
        ]
        return {
            "coefs": pd.DataFrame(rows),
            "scaler": self._scaler() if self.scaler_mean is not None else None,
            "features": list(self.features),
            "formula": self.formula,
        }

    @classmethod
    def from_legacy_bundle(cls, bundle: dict, formula: str | None = None) -> FittedModel:
        """Make a model from a final or per-year coefficient bundle.

        As in ``apply_blend_model.py``, classes are the sorted bundle classes and a feature missing from a class's
        coefficients counts as 0 (with a warning).

        Parameters
        ----------
        bundle : dict
            The bundle of `to_legacy_bundle`: ``coefs``, ``features``, ``formula`` (optional) and ``scaler``
            (optional).
        formula : str, optional
            The formula, used only when the bundle has none (older bundles).

        Returns
        -------
        FittedModel
            The model the bundle describes.

        Raises
        ------
        ValueError
            The bundle has no formula and none is given, or its scaler does not centre and scale.
        """
        formula = bundle.get("formula") or formula
        if not formula:
            raise ValueError(
                "The coefficient bundle has no saved formula; pass the formula it was fitted with."
            )
        coef_df = bundle["coefs"]
        features = list(bundle["features"])
        classes = sorted(coef_df["class"].unique())
        coef = np.zeros((len(classes), len(features)))
        intercept = np.zeros(len(classes))
        for i, cls_ in enumerate(classes):
            cls_rows = coef_df[coef_df["class"] == cls_]
            intercept[i] = cls_rows["intercept"].iloc[0]
            for j, feature in enumerate(features):
                feature_row = cls_rows[cls_rows["feature"] == feature]
                if feature_row.empty:
                    warnings.warn(
                        f"Feature '{feature}' not found in coefs for class '{cls_}' — using 0"
                    )
                else:
                    coef[i, j] = float(feature_row["coefficient"].iloc[0])

        scaler = bundle.get("scaler")
        stats = {}
        if scaler is not None:
            if not (scaler.with_mean and scaler.with_std):
                raise ValueError("Only a centring and scaling StandardScaler is supported.")
            stats = dict(
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,
                scaler_var=scaler.var_,
                n_samples=int(scaler.n_samples_seen_),
            )
        return cls(formula, features, classes, coef, intercept, **stats)

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Return the model as JSON-able metadata and named float arrays, without pickles.

        Returns
        -------
        meta : dict
            ``formula``, ``features``, ``classes`` and ``n_samples``.
        arrays : dict of str to numpy.ndarray
            ``coef`` and ``intercept``, and the three scaler arrays when the model has a scaler.
        """
        meta = {
            "formula": self.formula,
            "features": list(self.features),
            "classes": list(self.classes),
            "n_samples": self.n_samples,
        }
        names = ("coef", "intercept") + (_SCALER_ARRAYS if self.scaler_mean is not None else ())
        return meta, {name: getattr(self, name) for name in names}

    @classmethod
    def from_arrays(cls, meta: dict, arrays: dict[str, np.ndarray]) -> FittedModel:
        """Make the model that `to_arrays` describes.

        Parameters
        ----------
        meta : dict
            The metadata of `to_arrays`.
        arrays : dict of str to numpy.ndarray
            The named arrays of `to_arrays` (the scaler arrays may be missing).

        Returns
        -------
        FittedModel
            The model.
        """
        return cls(
            formula=meta["formula"],
            features=meta["features"],
            classes=meta["classes"],
            n_samples=meta["n_samples"],
            **{name: arrays.get(name) for name in ("coef", "intercept", *_SCALER_ARRAYS)},
        )

    def _scaler(self) -> StandardScaler:
        """Rebuild the fitted ``StandardScaler`` these statistics came from.

        Returns
        -------
        sklearn.preprocessing.StandardScaler
            A scaler with sklearn's own attribute set and the model's mean, variance, scale and sample count.
        """
        # Fitting a placeholder gives sklearn's own attribute set, order and
        # types; the statistics are then swapped in.
        scaler = StandardScaler().fit(np.zeros((1, len(self.features))))
        scaler.n_samples_seen_ = type(scaler.n_samples_seen_)(self.n_samples)
        scaler.mean_ = self.scaler_mean.copy()
        scaler.var_ = self.scaler_var.copy()
        scaler.scale_ = self.scaler_scale.copy()
        return scaler


def fold_coefficients(models: Mapping[int, FittedModel]) -> xr.Dataset:
    """Stack the CV folds' models over their holdout year.

    The folds share the formula, features and classes.

    Parameters
    ----------
    models : collections.abc.Mapping of int to FittedModel
        The model of each holdout year; every model has a scaler.

    Returns
    -------
    xarray.Dataset
        ``coef`` (year, class, feature), ``intercept`` (year, class), the scaler statistics (year, feature),
        ``n_samples`` (year), and the formula as the attribute ``formula``.

    Raises
    ------
    ValueError
        A fold has another formula, features or classes than the first, or has no scaler.
    """
    first = next(iter(models.values()))
    for year, model in models.items():
        if (model.formula, model.features, model.classes) != (first.formula, first.features, first.classes):
            raise ValueError(f"fold {year} has another formula, features or classes than the first fold")
        if model.scaler_mean is None:
            raise ValueError(f"fold {year} has no scaler")

    def stack(name):
        """Stack one attribute of the models along a new first axis."""
        return np.stack([getattr(m, name) for m in models.values()])

    return xr.Dataset(
        {
            "coef": (("year", "class", "feature"), stack("coef")),
            "intercept": (("year", "class"), stack("intercept")),
            **{name: (("year", "feature"), stack(name)) for name in _SCALER_ARRAYS},
            "n_samples": ("year", np.array([m.n_samples for m in models.values()], dtype=np.int64)),
        },
        coords={"year": np.array(list(models), dtype=np.int64), "class": list(first.classes),
                "feature": list(first.features)},
        attrs={"formula": first.formula},
    )


def fold_models(data: xr.Dataset) -> dict[int, FittedModel]:
    """Unstack the models that `fold_coefficients` stacked.

    Parameters
    ----------
    data : xarray.Dataset
        The dataset of `fold_coefficients`.

    Returns
    -------
    dict of int to FittedModel
        The model of each holdout year.
    """
    features = [str(f) for f in data["feature"].values]
    classes = [str(c) for c in data["class"].values]
    out = {}
    for k, year in enumerate(data["year"].values.tolist()):
        out[int(year)] = FittedModel(
            formula=str(data.attrs["formula"]),
            features=features,
            classes=classes,
            coef=data["coef"].values[k],
            intercept=data["intercept"].values[k],
            n_samples=int(data["n_samples"].values[k]),
            **{name: data[name].values[k] for name in _SCALER_ARRAYS},
        )
    return out


def design_matrix(formula: str, data: Mapping | pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Build the formula's right-hand side as a Patsy design.

    The intercept column is left out and rows with missing values are kept.

    Parameters
    ----------
    formula : str
        The formula; only its right-hand side is used.
    data : collections.abc.Mapping or pandas.DataFrame
        The columns the formula reads.

    Returns
    -------
    names : list of str
        The design columns.
    matrix : numpy.ndarray of float64, shape (row, column)
        The design matrix, ``NaN`` where a feature is missing.
    """
    matrix = patsy.dmatrix(
        design.right_hand_side(formula), data, NA_action=patsy.NAAction(NA_types=[])
    )
    return _without_intercept(matrix)


def _without_intercept(matrix) -> tuple[list[str], np.ndarray]:
    """Split a Patsy design into column names and a float matrix, dropping the intercept.

    Parameters
    ----------
    matrix : patsy.DesignMatrix
        The design.

    Returns
    -------
    names : list of str
        The column names without ``Intercept``.
    matrix : numpy.ndarray of float64, shape (row, column)
        The design without the intercept column.
    """
    names = list(matrix.design_info.column_names)
    keep = [j for j, name in enumerate(names) if name != "Intercept"]
    return [names[j] for j in keep], np.asarray(matrix, dtype=float)[:, keep]


def fit(formula: str, train: Mapping | pd.DataFrame) -> FittedModel:
    """Fit the formula on the rows with a known outcome and finite features.

    Parameters
    ----------
    formula : str
        The expanded formula.
    train : collections.abc.Mapping or pandas.DataFrame
        The training columns, with ``outcome`` (a class label per row, anything but a string for none).

    Returns
    -------
    FittedModel
        The model on standardised features.

    Raises
    ------
    ValueError
        The usable rows have fewer than two outcome classes.
    """
    features, X = design_matrix(formula, train)
    y = np.asarray(train["outcome"])
    valid = np.array([isinstance(v, str) for v in y])
    mask = valid & np.isfinite(X).all(axis=1)
    X_fit, y_fit = X[mask], y[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Fewer than 2 outcome classes in training data — cannot fit model.")
    log_fit(int(mask.sum()), len(y), np.unique(y_fit), features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    clf = multinomial()
    clf.fit(X_scaled, y_fit)
    return FittedModel.from_estimators(formula, features, clf, scaler)


def multinomial() -> LogisticRegression:
    """Make the blend's multinomial logistic regression (lbfgs, effectively unpenalised).

    sklearn before 1.5 has to be asked for ``multinomial``.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        An unfitted classifier with ``C=1e9``, ``max_iter=5000`` and ``tol=1e-5``.
    """
    kwargs = dict(solver="lbfgs", C=1e9, max_iter=5000, tol=1e-5)
    if Version(sklearn.__version__) < Version("1.5"):
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


def log_fit(used: int, rows: int, classes, features) -> None:
    """Log what a fit used: rows, outcome classes and feature columns.

    Parameters
    ----------
    used : int
        The training rows used.
    rows : int
        The candidate rows.
    classes : iterable
        The outcome classes of the training rows.
    features : sequence of str
        The feature columns.
    """
    log.info("  Training rows used : %d / %d", used, rows)
    log.info("  Outcome classes    : %s", sorted(classes))
    log.info("  Feature columns    : %d", len(features))


def fit_samples(formula: str, weekly: Weekly, rows: np.ndarray) -> FittedModel:
    """Fit the formula on samples of the weekly features, with the same numerics as `fit`.

    The fit runs from the arrays when the formula has a design plan, and on a Patsy design otherwise.

    Parameters
    ----------
    formula : str
        The expanded formula.
    weekly : hcwf.blend.samples.Weekly
        The weekly features and outcomes.
    rows : numpy.ndarray of int, shape (sample,)
        The positions in ``weekly`` to train on.

    Returns
    -------
    FittedModel
        The model.
    """
    plan = design.plan(formula, weekly.column_dtypes())
    if plan is None:
        return fit(formula, weekly.columns_of(rows))
    clf, scaler, used, total = fit_final_rows(
        lambda j: plan.column(j, weekly.take, rows), len(plan.columns), weekly.take("outcome", rows)
    )
    log_fit(used, total, clf.classes_, plan.columns)
    return FittedModel.from_estimators(formula, plan.columns, clf, scaler)


# ---------------------------------------------------------------------------
# Global cross-validation and the final fit, on one shared design
# ---------------------------------------------------------------------------

# A worker's interpreter and libraries, beyond the rows it fits.
_WORKER_BASE_BYTES = 350 * 2**20


def global_cv(plan: Plan, values, rows: np.ndarray, trainable: np.ndarray, year: np.ndarray,
              outcome: np.ndarray, *, holdout_years, training_years, true_holdout_years=(),
              required_classes=None, final_years=None, workers=1, budget_bytes=None):
    """Run global leave-one-year-out CV and the final fit on one shared design.

    The design of every evaluation sample is built once, column-major, in shared memory, and the folds and the final
    fit (the largest task, first) run on it, in the worker pool when the memory budget allows more than one worker.

    Parameters
    ----------
    plan : hcwf.blend.design.Plan
        The formula's design plan.
    values : callable
        ``values(name)`` gives the feature ``name`` per sample.
    rows : numpy.ndarray of int, shape (sample,)
        The evaluation's samples (the design's rows).
    trainable : numpy.ndarray of bool, shape (row,)
        The rows models train on (the dissemination cells).
    year : numpy.ndarray, shape (row,)
        The year of each row.
    outcome : numpy.ndarray of object, shape (row,)
        The outcome class of each row (a string, or anything else for none).
    holdout_years : iterable of int
        The years to hold out, one fold each.
    training_years : iterable of int
        The years a fold may train on.
    true_holdout_years : iterable of int, default ()
        The years no fold trains on.
    required_classes : iterable of str, optional
        The classes every fold's training rows must have.
    final_years : iterable of int, optional
        The years of the final fit; None for no final fit.
    workers : int, default 1
        The most worker processes.
    budget_bytes : int, optional
        The memory budget that limits the workers; None for no limit.

    Returns
    -------
    folds : list of tuple
        ``(test_year, test_rows, probabilities, model)`` per holdout year: the test row positions, the probabilities
        (``float64``, shape (test row, class), in the model's class order) and the `FittedModel`.
    final : tuple or None
        ``(classifier, scaler, rows used, rows)`` of the ``final_years`` fit, or None without ``final_years``.
    """
    tasks = [("fold", int(y)) for y in holdout_years]
    if final_years is not None:   # the largest task first, so that it does not start in the last round
        tasks.insert(0, ("final", tuple(int(y) for y in final_years)))
    context = {
        "formula_str": plan.formula,
        "columns": list(plan.columns),
        "year": np.asarray(year),
        "trainable": np.asarray(trainable, dtype=bool),
        "outcome": np.asarray(outcome, dtype=object),
        "training_years": tuple(training_years),
        "true_holdout_years": tuple(true_holdout_years),
        "required_classes": tuple(required_classes) if required_classes is not None else None,
    }
    shape = (len(rows), len(plan.columns))
    with pool.SharedArrays() as shared:
        design = plan.build(values, rows, out=shared.empty("design", shape, order="F"))
        context["design"] = shared.specs["design"]
        count = _workers(len(tasks), workers, shape, int(context["trainable"].sum()), budget_bytes)
        if count <= 1:
            results = [_run(task, context, design) for task in tasks]
        else:
            results = _run_pool(tasks, context, count)
        del design
    final = results.pop(0) if final_years is not None else None
    return results, final


def global_cv_frames(formula: str, weekly: Weekly, rows: np.ndarray, trainable: np.ndarray, *,
                     holdout_years, training_years, true_holdout_years=(), required_classes=None):
    """Run global CV for a formula without a design plan: each fold builds its Patsy design from its own training rows.

    Parameters
    ----------
    formula : str
        The expanded formula.
    weekly : hcwf.blend.samples.Weekly
        The weekly features and outcomes.
    rows : numpy.ndarray of int, shape (sample,)
        The evaluation's samples.
    trainable : numpy.ndarray of bool, shape (sample,)
        The samples models train on.
    holdout_years : iterable of int
        The years to hold out, one fold each.
    training_years : iterable of int
        The years a fold may train on.
    true_holdout_years : iterable of int, default ()
        The years no fold trains on.
    required_classes : iterable of str, optional
        The classes every fold's training rows must have.

    Returns
    -------
    list of tuple
        The folds as `global_cv` returns them: ``(test_year, test_rows, probabilities, model)``.

    Raises
    ------
    ValueError
        A fold has no test or training rows, lacks a required class, or its fit fails.
    """
    year = weekly.take("year", rows)
    outcome = weekly.take("outcome", rows)
    columns = _frame_columns(formula, weekly)
    out = []
    for test_year in holdout_years:
        train = np.flatnonzero(
            trainable & np.isin(year, training_years) & (year != test_year) & ~np.isin(year, true_holdout_years)
        )
        test = np.flatnonzero(year == test_year)
        _check_fold(test_year, train, test, outcome[train], training_years, required_classes)
        fitted = _fit_frames(formula, weekly.columns_of(rows[train], columns), outcome[train])
        if fitted is None:
            raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
        clf, scaler, info = fitted
        probabilities = _predict_frame(clf, scaler, info, weekly.columns_of(rows[test], columns))
        model = FittedModel.from_estimators(formula, info[1], clf, scaler)
        out.append((int(test_year), test, probabilities, model))
    return out


def _check_fold(test_year, train, test, outcome, training_years, required_classes):
    """Check that a global CV fold can be fitted.

    Parameters
    ----------
    test_year : int
        The holdout year.
    train : numpy.ndarray of int
        The training row positions.
    test : numpy.ndarray of int
        The test row positions.
    outcome : numpy.ndarray of object
        The outcome of the training rows.
    training_years : iterable of int
        The years a fold may train on (named in the message).
    required_classes : iterable of str or None
        The classes the training rows must have, or None.

    Raises
    ------
    ValueError
        There are no test rows, no training rows, a required class is missing, or fewer than two classes are present.
    """
    if not len(test):
        raise ValueError(f"Global CV holdout year {test_year} has no test rows.")
    if not len(train):
        raise ValueError(
            f"Global CV holdout year {test_year} has no training rows in "
            f"training_years={sorted(set(training_years))}."
        )
    present = {value for value in pd.unique(outcome) if isinstance(value, str)}
    if required_classes is not None:
        missing = [value for value in required_classes if value not in present]
        if missing:
            raise ValueError(
                f"Global CV holdout year {test_year} training rows are missing "
                f"outcome classes: {', '.join(missing)}."
            )
    if len(present) < 2:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")


def _workers(n_tasks: int, workers: int, shape, train_rows: int, budget_bytes) -> int:
    """Choose the number of workers within the memory budget (after the shared design), one per task at most.

    Parameters
    ----------
    n_tasks : int
        The number of tasks to run.
    workers : int
        The most workers wanted.
    shape : tuple of (int, int)
        The shape (row, column) of the shared design.
    train_rows : int
        The most rows a task trains on.
    budget_bytes : int or None
        The memory budget, or None for no limit.

    Returns
    -------
    int
        The number of workers, at least 1.
    """
    n_rows, n_columns = shape
    # The row-major fit matrix, the solver's per-row class arrays, the interpreter.
    per_worker = train_rows * 8 * (n_columns + 24) + _WORKER_BASE_BYTES
    budget = None if budget_bytes is None else max(budget_bytes - n_rows * n_columns * 8, 0)
    count = min(pool.workers_within([per_worker], workers, budget), n_tasks)
    return max(count, 1)


def _run_pool(tasks, context, count):
    """Run the CV tasks on the worker pool.

    Parameters
    ----------
    tasks : list of tuple
        The ``(kind, years)`` tasks of `global_cv`.
    context : dict
        What every task needs (the shared design's specs and the sample arrays).
    count : int
        The number of workers.

    Returns
    -------
    list
        The result of each task, in order.
    """
    log.info("  Fitting %d global CV tasks with %d workers.", len(tasks), count)
    return map_items(_run, tasks, (context,), workers=count)


def _run(task, context, design=None):
    """Run one CV task: a holdout fold or the final fit.

    Parameters
    ----------
    task : tuple
        ``("fold", year)`` or ``("final", years)``.
    context : dict
        What every task needs (see `global_cv`).
    design : numpy.ndarray of float64, shape (row, column), optional
        The shared design; None in a worker, which attaches it from the context.

    Returns
    -------
    tuple
        The result of ``_fold`` or ``_final``.
    """
    kind, years = task
    if design is None:  # in a worker
        design = pool.attach({"design": context["design"]})["design"]
    if kind == "fold":
        return _fold(years, design, context)
    return _final(years, design, context)


def _fold(test_year: int, design: np.ndarray, context: dict):
    """Fit one holdout year, as ``_compute_shared_fold`` fits it.

    Parameters
    ----------
    test_year : int
        The holdout year.
    design : numpy.ndarray of float64, shape (row, column)
        The shared column-major design.
    context : dict
        What every task needs (see `global_cv`).

    Returns
    -------
    tuple
        ``(test_year, test_rows, probabilities, model)``.

    Raises
    ------
    ValueError
        The fold cannot be fitted (see ``_check_fold``), or its fit fails.
    """
    year, trainable = context["year"], context["trainable"]
    train = np.flatnonzero(
        trainable
        & np.isin(year, context["training_years"])
        & (year != test_year)
        & ~np.isin(year, context["true_holdout_years"])
    )
    test = np.flatnonzero(year == test_year)
    outcome = context["outcome"][train]
    _check_fold(test_year, train, test, outcome, context["training_years"], context["required_classes"])
    fitted = fit_fold(design, train, outcome, test)
    if fitted is None:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
    probabilities, clf, scaler = fitted
    return test_year, test, probabilities, FittedModel.from_estimators(
        context["formula_str"], context["columns"], clf, scaler
    )


def fit_fold(design: np.ndarray, train: np.ndarray, y: np.ndarray, test: np.ndarray):
    """Fit and predict on rows of a column-major design.

    The scaler statistics are computed a column at a time and the standardised rows are written straight into a
    row-major matrix. Training rows with a ``NaN`` fall back to `fit_predict`.

    Parameters
    ----------
    design : numpy.ndarray of float64, shape (row, column)
        The shared column-major design.
    train : numpy.ndarray of int
        The training row positions.
    y : numpy.ndarray of object
        The outcome of the training rows.
    test : numpy.ndarray of int
        The test row positions.

    Returns
    -------
    tuple or None
        ``(probabilities, classifier, scaler)`` with the probabilities of shape (test row, class), or None when there
        is nothing to fit.
    """
    n_columns = design.shape[1]
    stats, bad = [], np.array([not isinstance(v, str) for v in y])
    for j in range(n_columns):
        values = design[:, j][train]
        if np.isnan(values).any():
            # NaN-aware statistics count per column: the legacy route.
            return fit_predict(_rows(design, train), y, _rows(design, test))
        stat = StandardScaler().fit(values[:, None])
        stats.append(stat)
        bad |= ~np.isfinite((values - stat.mean_[0]) / stat.scale_[0])
    scaler = _joined_scaler(stats, n_columns)
    keep = train[~bad]
    X = np.empty((len(keep), n_columns))
    for j in range(n_columns):
        X[:, j] = (design[:, j][keep] - scaler.mean_[j]) / scaler.scale_[j]
    X_test = scaler.transform(_rows(design, test))
    scaler.set_params(copy=True)  # as the legacy scaler is saved
    clf = _fit(X, y[~bad])
    return None if clf is None else (_predict(clf, X_test), clf, scaler)


def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray):
    """Fit and predict as the legacy CV did (``_fit_predict_design``).

    The training rows are standardised (in place) before the rows with a non-finite feature or no outcome are dropped,
    then the multinomial fit runs.

    Parameters
    ----------
    X_tr : numpy.ndarray of float64, shape (row, feature)
        The training design; overwritten by its standardised values.
    y_tr : numpy.ndarray of object, shape (row,)
        The outcome of the training rows.
    X_te : numpy.ndarray of float64, shape (row, feature)
        The test design.

    Returns
    -------
    tuple or None
        ``(probabilities, classifier, scaler)``, the probabilities of shape (test row, class) with ``NaN`` where a row
        has a non-finite feature; None when there is nothing to fit.
    """
    fitted = _fit_scaled(X_tr, y_tr)
    if fitted is None:
        return None
    clf, scaler = fitted
    return _predict(clf, scaler.transform(X_te)), clf, scaler


def _fit_scaled(X_tr: np.ndarray, y_tr: np.ndarray):
    """Fit the scaler and the classifier as `fit_predict` fits them.

    Parameters
    ----------
    X_tr : numpy.ndarray of float64, shape (row, feature)
        The training design; overwritten by its standardised values.
    y_tr : numpy.ndarray of object, shape (row,)
        The outcome of the training rows.

    Returns
    -------
    tuple or None
        ``(classifier, scaler)``, or None when there is nothing to fit.
    """
    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    scaler.set_params(copy=True)  # as saved before: later transforms copy
    bad = ~np.isfinite(X_tr).all(axis=1) | np.array([not isinstance(v, str) for v in y_tr], dtype=bool)
    clf = _fit(X_tr[~bad], y_tr[~bad])
    return None if clf is None else (clf, scaler)


def _fit(X: np.ndarray, y: np.ndarray):
    """Fit the multinomial model on prepared rows.

    Parameters
    ----------
    X : numpy.ndarray of float64, shape (row, feature)
        The standardised, finite training rows.
    y : numpy.ndarray of object, shape (row,)
        The outcome class of each row.

    Returns
    -------
    sklearn.linear_model.LogisticRegression or None
        The fitted classifier, or None with no rows, one class, or a solver that cannot fit (which warns).

    Notes
    -----
    Only the solver's `ValueError` and ``numpy.linalg.LinAlgError`` count as a fold that cannot be fitted; any other
    failure is a bug and propagates.
    """
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    clf = multinomial()
    try:
        clf.fit(X, y)
    except (ValueError, np.linalg.LinAlgError) as e:
        warnings.warn(f"_fit_predict_multinom fit failed: {e}")
        return None
    return clf


def _predict(clf, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities, ``NaN`` for rows with a non-finite feature.

    Parameters
    ----------
    clf : sklearn.linear_model.LogisticRegression
        The fitted classifier.
    X : numpy.ndarray of float64, shape (row, feature)
        The standardised rows.

    Returns
    -------
    numpy.ndarray of float64, shape (row, class)
        The probabilities.
    """
    bad = ~np.isfinite(X).all(axis=1)
    probs = np.full((len(X), len(clf.classes_)), np.nan)
    if (~bad).any():
        probs[~bad] = clf.predict_proba(X[~bad])
    return probs


# -- local and cluster CV ------------------------------------------------------


def local_cv(formula: str, weekly: Weekly, rows: np.ndarray, holdout_years):
    """Run each unit's own leave-one-year-out CV (``compute_cv_local``).

    For each unit and each holdout year it has, a fit on its other years' samples predicts that year's. A fit with
    fewer than two outcome classes predicts nothing.

    Parameters
    ----------
    formula : str
        The expanded formula.
    weekly : hcwf.blend.samples.Weekly
        The weekly features and outcomes.
    rows : numpy.ndarray of int, shape (sample,)
        The evaluation's samples, in every unit and year.
    holdout_years : iterable of int
        The years to hold out.

    Returns
    -------
    list of tuple
        ``(test, probabilities, classes)`` per unit and holdout year, by unit in order of first appearance, then
        holdout year: the test positions into ``rows``, the probabilities (``float64``, shape (test, class)) and the
        class labels.
    """
    year = weekly.take("year", rows)
    outcome = weekly.take("outcome", rows)
    fits = _Fits(formula, weekly, rows, outcome)
    out = []
    for mine in _by_unit(weekly, rows):
        present = set(year[mine].tolist())
        for test_year in (y for y in holdout_years if y in present):
            train = mine[year[mine] != test_year]
            test = mine[year[mine] == test_year]
            if not len(train) or not len(test):
                continue
            predicted = fits.predict(fits.fit(train), test)
            if predicted is not None:
                out.append((test, *predicted))
    return out


def cluster_cv(formula: str, weekly: Weekly, rows: np.ndarray, holdout_years, cluster_var: str):
    """Run leave-one-year-out CV within clusters (``compute_cv_clusters``).

    A unit's holdout year is predicted by the fit on the other years' samples of every unit in its cluster (its first
    sample's ``cluster_var``). The fit is the same for every unit of a cluster, so it runs once per cluster and year.

    Parameters
    ----------
    formula : str
        The expanded formula.
    weekly : hcwf.blend.samples.Weekly
        The weekly features and outcomes.
    rows : numpy.ndarray of int, shape (sample,)
        The evaluation's samples, in every unit and year.
    holdout_years : iterable of int
        The years to hold out.
    cluster_var : str
        The weekly column that gives a unit's cluster.

    Returns
    -------
    list of tuple
        As `local_cv`: ``(test, probabilities, classes)``.
    """
    year = weekly.take("year", rows)
    outcome = weekly.take("outcome", rows)
    cluster = weekly.take(cluster_var, rows)
    fits = _Fits(formula, weekly, rows, outcome)
    cache = {}
    out = []
    for mine in _by_unit(weekly, rows):
        here = cluster[mine[0]]
        present = set(year[mine].tolist())
        for test_year in (y for y in holdout_years if y in present):
            key = (here, test_year)
            if key not in cache:
                train = np.flatnonzero((year != test_year) & (cluster == here))
                cache[key] = fits.fit(train) if len(train) else None
            test = mine[year[mine] == test_year]
            predicted = fits.predict(cache[key], test)
            if predicted is not None:
                out.append((test, *predicted))
    return out


def _by_unit(weekly: Weekly, rows: np.ndarray):
    """Group the samples by unit.

    Parameters
    ----------
    weekly : hcwf.blend.samples.Weekly
        The weekly features and outcomes (``id`` names the unit).
    rows : numpy.ndarray of int, shape (sample,)
        The evaluation's samples.

    Returns
    -------
    list of numpy.ndarray of int
        The positions into `rows` of each unit's samples, units in order of first appearance.
    """
    codes, units = pd.factorize(weekly.take("id", rows))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(units) + 1))
    return [order[bounds[k]:bounds[k + 1]] for k in range(len(units))]


class _Fits:
    """CV fits of one formula on positions into rows.

    The fits run from the arrays when the formula has a design plan and with Patsy designs otherwise, as
    ``_fit_predict_multinom`` fitted them.

    Parameters
    ----------
    formula : str
        The expanded formula.
    weekly : hcwf.blend.samples.Weekly
        The weekly features and outcomes.
    rows : numpy.ndarray of int, shape (sample,)
        The evaluation's samples.
    outcome : numpy.ndarray of object, shape (sample,)
        The outcome class of each sample.

    Attributes
    ----------
    plan : hcwf.blend.design.Plan or None
        The formula's design plan, None when it has none.
    columns : list of str or None
        The weekly columns a Patsy design reads; None when there is a plan.
    """

    def __init__(self, formula: str, weekly: Weekly, rows: np.ndarray, outcome: np.ndarray):
        self.formula, self.weekly, self.rows, self.outcome = formula, weekly, rows, outcome
        self.plan = design.plan(formula, weekly.column_dtypes())
        self.columns = None if self.plan is not None else _frame_columns(formula, weekly)

    def fit(self, train: np.ndarray):
        """Fit the model on training samples.

        Parameters
        ----------
        train : numpy.ndarray of int
            The training positions into the rows.

        Returns
        -------
        tuple or None
            ``(classifier, scaler, info)`` with the Patsy design info (None with a design plan), or None when there
            is nothing to fit.
        """
        y = self.outcome[train]
        if len({v for v in pd.unique(y) if isinstance(v, str)}) < 2:
            return None
        if self.plan is not None:
            fitted = _fit_scaled(self.plan.build(self.weekly.take, self.rows[train], order="F"), y)
            return None if fitted is None else (*fitted, None)
        return _fit_frames(self.formula, self.weekly.columns_of(self.rows[train], self.columns), y)

    def predict(self, fitted, test: np.ndarray):
        """Predict the test samples.

        Parameters
        ----------
        fitted : tuple or None
            The result of `fit`.
        test : numpy.ndarray of int
            The test positions into the rows.

        Returns
        -------
        tuple or None
            ``(probabilities, classes)`` with the probabilities of shape (test, class), or None without a fit.
        """
        if fitted is None:
            return None
        clf, scaler, info = fitted
        if info is not None:
            frame = self.weekly.columns_of(self.rows[test], self.columns)
            return _predict_frame(clf, scaler, info, frame), tuple(clf.classes_)
        X = scaler.transform(self.plan.build(self.weekly.take, self.rows[test], order="F"))
        return _predict(clf, X), tuple(clf.classes_)


def _frame_columns(formula: str, weekly: Weekly) -> list[str]:
    """List the weekly columns a Patsy design of the formula may read.

    Parameters
    ----------
    formula : str
        The expanded formula.
    weekly : hcwf.blend.samples.Weekly
        The weekly features.

    Returns
    -------
    list of str
        The formula's columns that `weekly` has, without repeats.
    """
    return [c for c in dict.fromkeys(design.formula_columns(formula)) if c in weekly.columns]


def _fit_frames(formula: str, train: Mapping, y: np.ndarray):
    """Fit as `fit_predict` does, on a Patsy design of the training table.

    Parameters
    ----------
    formula : str
        The expanded formula.
    train : collections.abc.Mapping
        The training columns.
    y : numpy.ndarray of object, shape (row,)
        The outcome of the training rows.

    Returns
    -------
    tuple or None
        ``(classifier, scaler, (design info, design columns))``, or None when there is nothing to fit.
    """
    matrix = patsy.dmatrix(design.right_hand_side(formula), train)
    columns, X = _without_intercept(matrix)
    info = (matrix.design_info, columns)
    fitted = _fit_scaled(X, y)
    return None if fitted is None else (*fitted, info)


def _predict_frame(clf, scaler, info, frame: Mapping) -> np.ndarray:
    """Predict a test table, its design built with the training design's info, as ``_fit_predict_multinom`` did.

    Columns absent from the test design are zero and extra ones are dropped.

    Parameters
    ----------
    clf : sklearn.linear_model.LogisticRegression
        The fitted classifier.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler.
    info : tuple
        ``(design info, design columns)`` of the training design.
    frame : collections.abc.Mapping
        The test columns.

    Returns
    -------
    numpy.ndarray of float64, shape (row, class)
        The probabilities.
    """
    design_info, columns = info
    names, X = _without_intercept(patsy.dmatrix(design_info, frame))
    if names != columns:   # absent columns are zero, extra ones dropped
        position = {name: j for j, name in enumerate(names)}
        X = np.column_stack([X[:, position[c]] if c in position else np.zeros(len(X)) for c in columns])
    return _predict(clf, scaler.transform(X))


def _rows(design: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Gather rows of the design, column-major as the fold's own design would be.

    Parameters
    ----------
    design : numpy.ndarray of float64, shape (row, column)
        The shared design.
    rows : numpy.ndarray of int
        The positions to take.

    Returns
    -------
    numpy.ndarray of float64, shape (len(rows), column)
        The rows, in Fortran order.
    """
    out = np.empty((len(rows), design.shape[1]), order="F")
    for j in range(design.shape[1]):
        out[:, j] = design[:, j][rows]
    return out


def _joined_scaler(stats: list[StandardScaler], n_columns: int) -> StandardScaler:
    """Join per-column scalers into the scaler fitted on every column at once.

    Parameters
    ----------
    stats : list of sklearn.preprocessing.StandardScaler
        One scaler per column, each fitted on a single column.
    n_columns : int
        The number of columns.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        A scaler with the columns' means, variances and scales.
    """
    # A placeholder fit sets sklearn's own attributes in their order; the
    # statistics are then swapped in.
    scaler = StandardScaler(copy=False).fit(np.zeros((1, n_columns)))
    scaler.n_samples_seen_ = stats[0].n_samples_seen_
    scaler.mean_ = np.concatenate([s.mean_ for s in stats])
    scaler.var_ = np.concatenate([s.var_ for s in stats])
    scaler.scale_ = np.concatenate([s.scale_ for s in stats])
    return scaler


def _final(years, design, context):
    """Fit the final model on the trainable rows of the given years.

    Parameters
    ----------
    years : tuple of int
        The years of the final fit.
    design : numpy.ndarray of float64, shape (row, column)
        The shared column-major design.
    context : dict
        What every task needs (see `global_cv`).

    Returns
    -------
    tuple
        ``(classifier, scaler, rows used, rows)`` as `fit_final_rows`.
    """
    rows = np.flatnonzero(context["trainable"] & np.isin(context["year"], years))
    return fit_final_rows(lambda j: design[:, j][rows], design.shape[1], context["outcome"][rows])


def fit_final_rows(column, n_columns: int, y: np.ndarray):
    """Fit as ``fit()`` in ``model.py`` does, on design columns given per candidate row.

    The rows with an outcome and finite features are gathered row-major and standardised in place.

    Parameters
    ----------
    column : callable
        ``column(j)`` gives design column `j` for every candidate row (``float64``).
    n_columns : int
        The number of design columns.
    y : numpy.ndarray of object, shape (row,)
        The outcome class of each candidate row (a string, or anything else for none).

    Returns
    -------
    tuple
        ``(classifier, scaler, rows used, rows)``: the fitted `sklearn.linear_model.LogisticRegression` and
        ``StandardScaler``, the number of rows fitted and the number of candidates.

    Raises
    ------
    ValueError
        The usable rows have fewer than two outcome classes.
    """
    valid = np.array([isinstance(v, str) for v in y])
    finite = np.ones(len(y), dtype=bool)
    for j in range(n_columns):
        finite &= np.isfinite(column(j))
    mask = valid & finite
    y_fit = y[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Fewer than 2 outcome classes in training data — cannot fit model.")
    X = np.empty((int(mask.sum()), n_columns))
    for j in range(n_columns):
        X[:, j] = column(j)[mask]
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X)
    clf = multinomial()
    clf.fit(X, y_fit)
    return clf, scaler, int(mask.sum()), len(y)
