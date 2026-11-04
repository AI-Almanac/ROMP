"""Cross-validated evaluation of the blends and their baselines, on arrays.

The weekly features (`hcwf.blend.samples.Weekly`) and the recipe in; every model's out-of-fold probabilities
(`hcwf.blend.predictions.Predictions`) and their scores out, in an `Evaluation`. In order, which is also the row order
of the summary tables:

1. each enabled formula, cross-validated by each CV method (`hcwf.blend.logit`: global, local, clusters), and the
   operational fit of the final formula;
2. the random forest blend, when configured (`hcwf.blend.random_forest`);
3. external CV predictions;
4. the climatology baselines: ``unc_clim_raw`` (the skill baseline), ``clim_raw`` and its year windows;
5. each model's raw and Platt-calibrated probabilities (`hcwf.blend.calibration`);
6. the multi-model ensemble (`hcwf.blend.mme`).

Models train on the dissemination units and predict every unit; scores are over the dissemination units
(`hcwf.metrics.probabilistic`). The numerics are those of the legacy ``1_blend_evaluation.py``, operation for operation;
the artifacts are NetCDF (predictions, coefficients), CSV (scores, weights) and JSON, under blending's output
contract (names, columns and formats): the contract's names are kept as they are, whatever HCWF names its own files.

The blend-only settings are the `~hcwf.blend.recipe.Recipe`'s (years, CV methods, formulas, baselines, scoring, MME, external
predictions, forest). The dissemination units are given as ids (`dissemination_ids` reads them from the units
specification). Parallel work runs on the run's worker pool (`hcwf.parallel`).

Changed from the legacy script: the MME aligns any number of sources on their samples (id and issue date), where the
script's merge failed for two or more; ``evaluation.mme.optimize: always`` optimises whatever the holdout years.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from hcwf import parallel as pool
from hcwf.blend import calibration, design, logit, mme, random_forest
from hcwf.blend.predictions import Predictions
from hcwf.blend.recipe import (CLIMATOLOGY_FEATURE, ExternalPrediction, Recipe, probability_feature_base,
                               window_tag)
from hcwf.blend.samples import DailyInputs, Weekly
from hcwf.metrics import probabilistic as metrics
from hcwf.spatial import ids as unit_ids
from hcwf.spatial.units import read_id_list

__all__ = ["Evaluation", "RunSettings", "dissemination_ids", "evaluate", "final_rows", "fit_final",
           "load_external_predictions", "prepare_inputs"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSettings:
    """The years and tags of an evaluation."""

    cutoff_mode: str
    """The recipe's ``evaluation.cutoff_mode``."""
    training_years: tuple[int, ...]
    """The training years."""
    true_holdout_years: tuple[int, ...]
    """The true holdout years, which no fit trains on."""
    holdout_years: tuple[int, ...]
    """The true holdouts, then the CV holdouts: the CV folds, in order."""
    cutoff_tag: str
    """The recipe's cutoff tag, a part of every artifact name."""
    year_tag: str
    """The recipe's year tag, a part of every artifact name."""

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> RunSettings:
        """Read the years and tags from a recipe.

        Parameters
        ----------
        recipe : hcwf.blend.recipe.Recipe
            The blend recipe.

        Returns
        -------
        RunSettings
            The settings, with the true holdouts first in ``holdout_years``.
        """
        years = recipe.evaluation.years
        return cls(
            cutoff_mode=recipe.evaluation.cutoff_mode,
            training_years=tuple(years.training),
            true_holdout_years=tuple(years.true_holdout),
            holdout_years=tuple([*years.true_holdout, *years.cv_holdout]),
            cutoff_tag=recipe.cutoff_tag(),
            year_tag=recipe.year_tag(),
        )

    @property
    def output_tag(self) -> str:
        """The suffix on every evaluation artifact, e.g. ``"_fixed_cutoff_2023_2025"``."""
        return f"{self.cutoff_tag}{self.year_tag}"

    @property
    def fit_years(self) -> list[int]:
        """The final fit's years: the training years that are not true holdouts."""
        held_out = set(self.true_holdout_years)
        return [year for year in self.training_years if year not in held_out]


def dissemination_ids(units, weekly: Weekly | pd.DataFrame) -> np.ndarray:
    """Return the ids of the units the blend trains on and is scored on.

    These are the units specification's dissemination list (`hcwf.spatial.units.read_id_list`, grid ids under the weekly
    units' convention), or every unit of the weekly features when there is no list.

    Parameters
    ----------
    units : hcwf.spatial.units.UnitsSpec or None
        The units specification; None (or one without a dissemination list) means every unit.
    weekly : hcwf.blend.samples.Weekly or pandas.DataFrame
        The weekly features (a legacy table is taken as a ``Weekly``).

    Returns
    -------
    numpy.ndarray of object, shape (unit,)
        The normalised unit ids.
    """
    ids = _normalised(Weekly.of(weekly).units)
    if units is None or not units.dissemination:
        return np.asarray(sorted(set(ids)), dtype=object)
    listed = read_id_list(units.resolve_path(units.dissemination), units.dissemination_id_col,
                          explicit=units.convention, authoritative_ids=pd.Series(ids, dtype=object))
    return np.asarray(listed, dtype=object)


def _normalised(units) -> np.ndarray:
    """Normalise unit ids (each id is mapped on its own, so repeated ids give the same result).

    Parameters
    ----------
    units : sequence of str
        The unit ids, one per sample or per unit.

    Returns
    -------
    numpy.ndarray of object, shape (n,)
        The normalised ids, in the order given.
    """
    return unit_ids.normalize(pd.Series(units, dtype=object), context="blend input IDs").to_numpy(dtype=object)


def prepare_inputs(weekly: Weekly | pd.DataFrame, recipe: Recipe, dissemination=None) -> tuple[Weekly, np.ndarray]:
    """Return the weekly features with normalised ids, and the dissemination unit ids.

    Parameters
    ----------
    weekly : hcwf.blend.samples.Weekly or pandas.DataFrame
        The weekly features (a legacy table is taken as a ``Weekly``).
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe; its number of bins must match the climatology weeks of ``weekly``.
    dissemination : sequence of str, optional
        The dissemination unit ids; None means every unit.

    Returns
    -------
    weekly : hcwf.blend.samples.Weekly
        The features, with normalised unit ids.
    cells : numpy.ndarray of object, shape (cell,)
        The dissemination unit ids (sorted when they are every unit).

    Raises
    ------
    ValueError
        The climatology weeks present in ``weekly`` are not ``1..n`` for the recipe's ``n`` bins.
    """
    if isinstance(weekly, Weekly):
        weekly = weekly.with_units(_normalised(weekly.units))
    else:
        weekly = Weekly.from_frame(unit_ids.with_id_column(weekly, context="blend input IDs"))
    cells = np.asarray(sorted(set(weekly.units)) if dissemination is None else pd.unique(pd.Series(dissemination)),
                       dtype=object)
    weeks = _present_weeks(weekly.columns, CLIMATOLOGY_FEATURE)
    n_bins = recipe.features.count
    if weeks != list(range(1, n_bins + 1)):
        raise ValueError(
            f"The weekly features have climatology weeks {weeks} "
            f"({CLIMATOLOGY_FEATURE}_week<n>); the recipe bins {n_bins}."
        )
    return weekly, cells


def log_sample_support(support: dict) -> None:
    """Log the rows the formulas' common sample support kept, as the scripts printed it.

    Parameters
    ----------
    support : dict
        The support summary of `hcwf.blend.design.sample_support`: ``policy``, ``rows_before``, ``rows_after``,
        ``rows_excluded`` and ``nonfinite_by_column``.
    """
    log.info(
        "Formula sample support (%s): %s -> %s rows (%s excluded)",
        support["policy"], support["rows_before"], support["rows_after"], support["rows_excluded"],
    )
    if support["nonfinite_by_column"]:
        log.info("Non-finite formula predictors: %s", support["nonfinite_by_column"])


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    """Every model's CV predictions and scores, and the fitted weights.

    `save` writes them under the legacy script's artifact names, as NetCDF (predictions, coefficients), CSV (scores,
    weights) and JSON.
    """

    output_tag: str
    """The suffix of every artifact name (`RunSettings.output_tag`)."""
    sample_support: dict
    """The formulas' common sample support, with the formula resolution (``formula_sample_support{tag}.json``)."""
    formulas: dict[str, str]
    """The resolved formula that was cross-validated, by model name."""
    cell_metrics: pd.DataFrame | None = None
    """The per-unit metrics of every scored model (``summary_models{tag}``)."""
    pooled: pd.DataFrame | None = None
    """The pooled skill summary of `cell_metrics` (``summary_models_pooled{tag}``)."""
    yearly: pd.DataFrame | None = None
    """The yearly metrics of the models with CV predictions (``yearly_metrics_global{tag}``)."""
    cv_preds: dict[tuple[str, str], Predictions] = field(default_factory=dict)
    """The CV predictions, by ``(model, method)``."""
    coefficients: dict[tuple[str, str], dict[int, logit.FittedModel]] = field(default_factory=dict)
    """The fitted models of the CV folds, by ``(model, method)`` and then by holdout year."""
    platt_weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    """The Platt weights, by file stem."""
    mme_weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    """The MME weights, by file stem."""
    final_model: logit.FittedModel | None = None
    """The final formula fitted on every training year (set by ``evaluate(final=True)``; saved in the trained blend)."""
    final_forest: random_forest.FittedForest | None = None
    """The final random forest, likewise."""

    def save(self, directory: str | os.PathLike, *, pickles: bool = False) -> list[Path]:
        """Write the artifacts into a folder.

        Parameters
        ----------
        directory : str or pathlib.Path
            The folder; created when it does not exist.
        pickles : bool, default False
            Also write ``.pkl`` twins of the per-unit and yearly score tables.

        Returns
        -------
        list of pathlib.Path
            The paths written, in order.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tag = self.output_tag
        written = []

        def wrote(path):
            """Record and log a written path."""
            written.append(path)
            log.info("Wrote %s", path)

        path = directory / f"formula_sample_support{tag}.json"
        path.write_text(json.dumps(self.sample_support, indent=2))
        wrote(path)
        for (model, method), preds in self.cv_preds.items():
            path = directory / f"cv_preds_{model}_{method}{tag}.nc"
            preds.to_netcdf(path)
            wrote(path)
            folds = self.coefficients.get((model, method))
            if folds:
                path = directory / f"coefs_{model}_{method}{tag}.nc"
                logit.fold_coefficients(folds).to_netcdf(path)
                wrote(path)
        for stem, weights in {**self.platt_weights, **self.mme_weights}.items():
            path = directory / f"{stem}.csv"
            weights.to_csv(path, index=False)
            wrote(path)
        tables = [(f"summary_models{tag}", self.cell_metrics, True), (f"summary_models_pooled{tag}", self.pooled, False),
                  (f"yearly_metrics_global{tag}", self.yearly, True)]
        for stem, table, twin in tables:
            if table is None:
                continue
            path = directory / f"{stem}.csv"
            table.to_csv(path, index=False)
            wrote(path)
            if pickles and twin:
                path = directory / f"{stem}.pkl"
                table.to_pickle(path)
                wrote(path)
        return written


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


def evaluate(
    weekly: Weekly | pd.DataFrame,
    recipe: Recipe,
    dissemination=None,
    *,
    workers: int = 1,
    results_dir: str | os.PathLike | None = None,
    budget_bytes: int | None = None,
    final: bool = False,
    forest_inputs: DailyInputs | None = None,
) -> Evaluation:
    """Cross-validate the recipe's blends and score its baselines.

    Up to ``workers`` processes fit the global CV folds (as many as ``budget_bytes`` holds) and the Platt calibration;
    results do not depend on either. The final formula is fitted alongside the global CV folds, on their design matrix.
    Formula and CV errors raise; climatology, forecast and MME source problems warn and skip that model, as the script
    did.

    Parameters
    ----------
    weekly : hcwf.blend.samples.Weekly or pandas.DataFrame
        The weekly features (a legacy table is taken as a ``Weekly``).
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    dissemination : sequence of str, optional
        The dissemination unit ids (see `dissemination_ids`); None means every unit.
    workers : int, default 1
        The most processes for the global CV folds and the Platt calibration.
    results_dir : str or os.PathLike, optional
        Where relative ``external_predictions`` files are read from.
    budget_bytes : int, optional
        The memory the parallel fits may hold; None for no bound.
    final : bool, default False
        Also fit the final formula (and the forest) on every training year, into `Evaluation.final_model`
        (`Evaluation.final_forest`).
    forest_inputs : hcwf.blend.samples.DailyInputs, optional
        The daily inputs of these weekly samples (`hcwf.blend.features.daily_inputs`); needed when the recipe has a
        random forest.

    Returns
    -------
    Evaluation
        The predictions, scores and weights.

    Raises
    ------
    ValueError
        ``workers`` is below 1, the recipe has a random forest but ``forest_inputs`` is None, or an input is invalid.
        (A formula's CV failure surfaces as `RuntimeError` from the CV stage.)
    """
    if workers < 1:
        raise ValueError("workers must be a positive integer.")
    run = RunSettings.from_recipe(recipe)
    weekly, cells = prepare_inputs(weekly, recipe, dissemination)
    labels = recipe.outcome_labels()
    externals = _load_external_predictions(recipe.evaluation.external_predictions, results_dir, labels)

    # Formulas come before sample support: only predictors that enabled formulas reference may decide the common
    # evaluation rows.
    formulas, resolution = _formulas(recipe, weekly, labels, run)
    rows, support = design.sample_support(weekly, formulas)
    support["formula_resolution"] = resolution
    log_sample_support(support)
    log.info("Formulas: %s", list(formulas))

    ctx = _Context(recipe, run, weekly, rows, cells, labels, workers=workers, budget_bytes=budget_bytes)
    result = Evaluation(output_tag=run.output_tag, sample_support=support, formulas=formulas)
    final_name = recipe.final_formula() if final else None
    train_final = final_rows(ctx) if final else None
    scores = _Scores(ctx)
    _cross_validate(ctx, formulas, result, scores, final_name)
    forest = recipe.evaluation.random_forest
    if forest is not None:
        if forest_inputs is None:
            raise ValueError("the random forest blend needs its inputs (features.daily_inputs)")
        _forest(ctx, forest, forest_inputs, result, scores, with_final=final)
    if final and result.final_model is None:
        log.info("\nFitting %s on all %d training years...", final_name, len(run.fit_years))
        result.final_model = logit.fit_samples(formulas[final_name], weekly, train_final)
    for external, preds in externals:
        log.info("\nExternal CV predictions: %s (%s)", external.name, external.file)
        scores.add(preds, external.name, external.method, yearly=True)
    clim_preds = _climatology_baselines(ctx, scores)
    calibrated_preds = _forecast_baselines(ctx, result, scores)
    _mme(ctx, clim_preds, calibrated_preds, result, scores)

    scores.finish()
    if scores.cells:
        result.cell_metrics = pd.concat(scores.cells, ignore_index=True)
        try:
            result.pooled = metrics.pooled_skill(result.cell_metrics)
        except Exception as e:
            warnings.warn(f"Could not compute pooled summary: {e}")
    if scores.yearly:
        result.yearly = pd.DataFrame(scores.yearly)
    return result


def _formulas(recipe: Recipe, weekly: Weekly, labels, run: RunSettings):
    """Resolve the recipe's formulas for these weekly features.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    weekly : hcwf.blend.samples.Weekly
        The weekly features (their unavailable rain predictors are dropped).
    labels : list of str
        The outcome bins (``week1..weekN``, ``later``).
    run : RunSettings
        The years and cutoff mode.

    Returns
    -------
    tuple
        What `hcwf.blend.design.formulas` returns: the resolved formulas by model name and their resolution report.
    """
    evaluation = recipe.evaluation
    return design.formulas(evaluation.formula_map(), evaluation.window_variants,
                           weekly.attrs.get("unavailable_rain_predictors", {}), len(labels) - 1, run.cutoff_mode)


def fit_final(weekly: Weekly | pd.DataFrame, recipe: Recipe, dissemination=None) -> logit.FittedModel:
    """Fit the final formula on every training year that is not a true holdout.

    The fit is on the dissemination units, after the evaluation's sample support, and without the CV
    (``evaluate(..., final=True)`` fits it alongside).

    Parameters
    ----------
    weekly : hcwf.blend.samples.Weekly or pandas.DataFrame
        The weekly features (a legacy table is taken as a ``Weekly``).
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.
    dissemination : sequence of str, optional
        The dissemination unit ids; None means every unit.

    Returns
    -------
    hcwf.blend.logit.FittedModel
        The fitted final formula.
    """
    run = RunSettings.from_recipe(recipe)
    weekly, cells = prepare_inputs(weekly, recipe, dissemination)
    labels = recipe.outcome_labels()
    formulas, _ = _formulas(recipe, weekly, labels, run)
    rows, support = design.sample_support(weekly, formulas)
    log_sample_support(support)
    name = recipe.final_formula()
    train = final_rows(_Context(recipe, run, weekly, rows, cells, labels))
    log.info("\nFormula : %s", formulas[name])
    log.info("\nFitting %s on all %d training years...", name, len(run.fit_years))
    return logit.fit_samples(formulas[name], weekly, train)


@dataclass
class _Context:
    """What the stages of an evaluation share: the recipe, the years, the samples and the parallel budget."""

    recipe: Recipe
    """The blend recipe."""
    run: RunSettings
    """The years and tags."""
    weekly: Weekly
    """Every sample."""
    rows: np.ndarray
    """The formula-supported samples (positions into `weekly`), of every unit."""
    cells: np.ndarray
    """The dissemination unit ids."""
    labels: list[str]
    """The outcome bins: ``week1..weekN``, ``later``."""
    workers: int = 1
    """The most processes for the parallel stages."""
    budget_bytes: int | None = None
    """The memory the parallel fits may hold; None for no bound."""

    @functools.cached_property
    def year(self) -> np.ndarray:
        """The year of each supported sample.

        Returns
        -------
        numpy.ndarray of int, shape (row,)
            The years.
        """
        return self.weekly.take("year", self.rows)

    @functools.cached_property
    def unit_allowed(self) -> np.ndarray:
        """Per unit of the weekly features: whether it is a dissemination unit.

        Returns
        -------
        numpy.ndarray of bool, shape (unit,)
            True for a dissemination unit.
        """
        cells = set(self.cells.tolist())
        return np.fromiter((unit in cells for unit in self.weekly.units), dtype=bool, count=len(self.weekly.units))

    @functools.cached_property
    def allowed(self) -> np.ndarray:
        """Per supported sample: whether it is in a dissemination unit (models train on those).

        Returns
        -------
        numpy.ndarray of bool, shape (row,)
            True for a sample of a dissemination unit.
        """
        return self.unit_allowed[self.weekly.sample_unit[self.rows]]

    def in_years(self, years) -> np.ndarray:
        """Return the mask of the supported samples in some years.

        Parameters
        ----------
        years : iterable of int
            The years.

        Returns
        -------
        numpy.ndarray of bool, shape (row,)
            True where the sample's year is in `years`.
        """
        return np.isin(self.year, list(years))

    def allowed_ids(self, ids: np.ndarray) -> np.ndarray:
        """Return, per id, whether it is a dissemination unit.

        Parameters
        ----------
        ids : numpy.ndarray of object, shape (n,)
            Unit ids.

        Returns
        -------
        numpy.ndarray of bool, shape (n,)
            True for a dissemination unit.
        """
        return _allowed_ids(ids, self.cells)


def _allowed_ids(ids: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return, per id, whether it is one of `cells`, tested once per distinct id.

    Parameters
    ----------
    ids : numpy.ndarray of object, shape (n,)
        Unit ids.
    cells : numpy.ndarray of object, shape (cell,)
        The dissemination unit ids.

    Returns
    -------
    numpy.ndarray of bool, shape (n,)
        True where the id is in `cells`.
    """
    codes, units = pd.factorize(np.asarray(ids))
    wanted = set(cells.tolist())
    return np.fromiter((unit in wanted for unit in units.tolist()), dtype=bool, count=len(units))[codes]


def final_rows(ctx: _Context) -> np.ndarray:
    """Return the final fit's samples: the supported samples in the dissemination units and the fit years.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.

    Returns
    -------
    numpy.ndarray of int, shape (train,)
        Positions into the weekly features.

    Raises
    ------
    ValueError
        There are no such rows, or an outcome class has no row among them.
    """
    fit_years = ctx.run.fit_years
    candidates = ctx.rows[ctx.allowed]
    train = candidates[np.isin(ctx.weekly.take("year", candidates), fit_years)]
    if not len(train):
        raise ValueError("Final fit has no rows in the configured training years and dissemination cells.")
    present = {value for value in pd.unique(ctx.weekly.take("outcome", train)) if isinstance(value, str)}
    missing = [value for value in ctx.labels if value not in present]
    if missing:
        raise ValueError("Final-fit training rows are missing outcome classes: " + ", ".join(missing))
    log.info("  Training years        : %s", sorted(fit_years))
    log.info("  Training rows (allowed cells) : %d", len(train))
    return train


@dataclass
class _Scores:
    """Per-unit metric frames and yearly metric rows, in the order models are scored.

    `add` only queues a model; `finish` scores them all, in the run's worker pool when there are workers (each model
    is independent), and fills `cells` and `yearly` in queue order.
    """

    ctx: _Context
    """The evaluation's shared state."""
    cells: list = field(default_factory=list)
    """The per-unit metric tables, one per scored model."""
    yearly: list = field(default_factory=list)
    """The yearly metric rows of every scored model that asked for them."""
    queued: list = field(default_factory=list)
    """The models waiting to be scored, as ``(preds, model, method, yearly, warn_as, raise_as)``."""

    def add(self, preds: Predictions, model: str, method: str, *, yearly: bool, warn_as: str | None = None,
            raise_as: str | None = None) -> None:
        """Queue a model for scoring.

        The metrics are per unit over the dissemination units and, for CV predictions, per year over every unit. A
        failure to score the model raises (as `RuntimeError` ``"<raise_as>: ..."`` if `raise_as` is given) or, with
        `warn_as`, warns ``"<warn_as>: ..."`` and leaves the model out.

        Parameters
        ----------
        preds : hcwf.blend.predictions.Predictions
            The model's probabilities.
        model : str
            The model name.
        method : str
            The CV method (or ``"raw"``, ``"calibrated"``, ``"mme"``).
        yearly : bool
            Also compute the yearly metrics.
        warn_as : str, optional
            The warning prefix for a failure; None to raise instead.
        raise_as : str, optional
            The `RuntimeError` message prefix for a failure; None to re-raise the error itself.
        """
        self.queued.append((preds, model, method, yearly, warn_as, raise_as))

    def finish(self) -> None:
        """Score the queued models (see `add`) and fill `cells` and `yearly`.

        Raises
        ------
        RuntimeError
            A model failed to score and was queued with `raise_as`.
        result
            Whatever scoring a model raised (an `Exception`), when it was queued with neither `warn_as` nor `raise_as`.
        """
        queued, self.queued = self.queued, []
        if not queued:
            return
        jobs = [(preds, model, method, yearly) for preds, model, method, yearly, _, _ in queued]
        start = time.perf_counter()
        results = pool.map_items(_score, jobs, (self.ctx.cells,), workers=min(self.ctx.workers, len(jobs)))
        log.info("\nScored %d models in %.1fs", len(jobs), time.perf_counter() - start)
        for (_, _, _, _, warn_as, raise_as), result in zip(queued, results):
            if isinstance(result, Exception):
                if warn_as is not None:
                    warnings.warn(f"{warn_as}: {result}")
                    continue
                if raise_as is not None:
                    raise RuntimeError(f"{raise_as}: {result}") from result
                raise result
            cells, yearly = result
            self.cells.append(cells)
            self.yearly.extend(yearly)


def _score(job, cells: np.ndarray):
    """Score one model, returning the exception that scoring raised instead of raising it.

    Parameters
    ----------
    job : tuple
        ``(preds, model, method, yearly)``: the `hcwf.blend.predictions.Predictions`, the model name, the CV method and
        whether to compute the yearly metrics.
    cells : numpy.ndarray of object, shape (cell,)
        The dissemination unit ids.

    Returns
    -------
    tuple or Exception
        ``(per-unit metrics table, yearly metric rows)``, or the exception.
    """
    preds, model, method, yearly = job
    try:
        P, outcome = preds.probabilities, preds.outcome
        table = metrics.cell_metrics(P, outcome, preds.ids, preds.labels, allowed=_allowed_ids(preds.ids, cells),
                                     earlier=preds.earlier)
        table["model"] = model
        table["cv_method"] = method
        rows = metrics.yearly_metrics(P, outcome, preds.year, model, method, auc_method="sklearn") if yearly else []
        return table, rows
    except Exception as exc:
        return exc


# ---------------------------------------------------------------------------
# 1-3: the blends and external predictions
# ---------------------------------------------------------------------------


def _cv_methods(recipe: Recipe) -> list[str]:
    """Return the CV methods, under the names the artifacts carry.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.

    Returns
    -------
    list of str
        The recipe's ``evaluation.cv_methods``, with ``"clusters"`` renamed ``"cluster"``.
    """
    return ["cluster" if m == "clusters" else m for m in recipe.evaluation.cv_methods]


def _cross_validate(ctx: _Context, formulas: dict[str, str], result: Evaluation, scores: _Scores,
                    final_model: str | None = None) -> None:
    """Cross-validate each formula under each CV method; the global method also fits the final model.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    formulas : dict of str to str
        The resolved formulas by model name.
    result : Evaluation
        Receives the CV predictions, the fold coefficients and, from the global method, the final model.
    scores : _Scores
        Receives each model's predictions for scoring.
    final_model : str, optional
        The name of the formula to fit on every training year; None for none.

    Raises
    ------
    RuntimeError
        A formula's CV fails for any reason (the message names the model and method).
    ValueError
        A CV method produced no predictions; this is raised inside the guard and so reaches the caller as the
        `RuntimeError` above.
    """
    run = ctx.run
    dtypes = ctx.weekly.column_dtypes()
    for model, formula in formulas.items():
        log.info("\nCV: %s", model)
        for method in _cv_methods(ctx.recipe):
            log.info("  Method: %s", method)
            try:
                folds = {}
                if method == "global":
                    plan = design.plan(formula, dtypes)
                    if plan is not None:
                        parts, fitted = _global_cv(ctx, plan, model == final_model)
                        if fitted is not None:
                            result.final_model = fitted
                    else:
                        parts = logit.global_cv_frames(
                            formula, ctx.weekly, ctx.rows, ctx.allowed,
                            holdout_years=run.holdout_years, training_years=run.training_years,
                            true_holdout_years=run.true_holdout_years, required_classes=ctx.labels,
                        )
                        if not parts:
                            raise ValueError("Global CV produced no fold predictions.")
                    folds = {year: fold_model for year, _, _, fold_model in parts}
                    preds = _fold_predictions(ctx, [(test, P, m.classes) for _, test, P, m in parts])
                elif method == "local":
                    preds = _fold_predictions(ctx, logit.local_cv(formula, ctx.weekly, ctx.rows, run.holdout_years))
                elif method == "cluster":
                    parts = logit.cluster_cv(formula, ctx.weekly, ctx.rows, run.holdout_years,
                                             ctx.recipe.evaluation.cluster_var)
                    preds = _fold_predictions(ctx, parts)
                else:
                    warnings.warn(f"Unknown CV method: {method}. Skipping.")
                    continue
                if preds is None or not len(preds):
                    raise ValueError(f"No predictions for {model}/{method}.")
                scores.add(preds, model, method, yearly=True, raise_as=f"Error in {model}/{method}")
                result.cv_preds[(model, method)] = preds
                if folds:
                    result.coefficients[(model, method)] = folds
            except Exception as e:
                raise RuntimeError(f"Error in {model}/{method}: {e}") from e


def _fold_predictions(ctx: _Context, parts) -> Predictions | None:
    """Assemble CV folds' predictions as `~hcwf.blend.predictions.Predictions` over the outcome bins.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    parts : iterable of tuple
        One ``(positions into rows, probabilities, classes)`` per fold: the test positions, the probabilities
        ``(test, class)`` and the class labels of its columns.

    Returns
    -------
    hcwf.blend.predictions.Predictions or None
        The predictions, ``NaN`` for a class a fold lacked; None when there are no folds.
    """
    if not parts:
        return None
    index = {label: j for j, label in enumerate(ctx.labels)}
    positions = np.concatenate([test for test, _, _ in parts])
    P = np.full((len(positions), len(ctx.labels)), np.nan)
    start = 0
    for test, probabilities, classes in parts:
        for k, label in enumerate(classes):
            if label in index:
                P[start:start + len(test), index[label]] = probabilities[:, k]
        start += len(test)
    return Predictions.of_samples(ctx.weekly, ctx.rows[positions], P, ctx.labels)


def _global_cv(ctx: _Context, plan: design.Plan, with_final: bool):
    """Run the global CV on the shared design matrix.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    plan : hcwf.blend.design.Plan
        The design plan of the formula.
    with_final : bool
        Also fit the final model on the fit years.

    Returns
    -------
    folds : list of tuple
        The folds as `hcwf.blend.logit.global_cv` returns them.
    fitted : hcwf.blend.logit.FittedModel or None
        The final model, None unless `with_final`.

    Raises
    ------
    ValueError
        The CV produced no fold predictions.
    """
    run = ctx.run
    folds, final = logit.global_cv(
        plan,
        ctx.weekly.take,
        ctx.rows,
        ctx.allowed,
        ctx.year,
        ctx.weekly.take("outcome", ctx.rows),
        holdout_years=run.holdout_years,
        training_years=run.training_years,
        true_holdout_years=run.true_holdout_years,
        required_classes=ctx.labels,
        final_years=run.fit_years if with_final else None,
        workers=ctx.workers,
        budget_bytes=ctx.budget_bytes,
    )
    if not folds:
        raise ValueError("Global CV produced no fold predictions.")
    fitted = None
    if final is not None:
        clf, scaler, used, total = final
        logit.log_fit(used, total, clf.classes_, plan.columns)
        fitted = logit.FittedModel.from_estimators(plan.formula, plan.columns, clf, scaler)
    return folds, fitted


def _forest(ctx: _Context, params: random_forest.Params, inputs: DailyInputs, result: Evaluation, scores: _Scores,
            *, with_final: bool) -> None:
    """Cross-validate the random forest blend and, if asked, fit its final forest.

    The out-of-fold weekly probabilities are for the samples the formulas score, from forests fitted on every sample
    of the dissemination units in each fold's training years.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    params : hcwf.blend.random_forest.Params
        The forest's settings.
    inputs : hcwf.blend.samples.DailyInputs
        The daily inputs of the weekly samples.
    result : Evaluation
        Receives the CV predictions and the final forest.
    scores : _Scores
        Receives the predictions for scoring.
    with_final : bool
        Also fit the final forest on every training year.
    """
    run = ctx.run
    weekly = ctx.weekly
    models = inputs.model_names
    X = random_forest.feature_matrix(weekly, inputs, np.arange(len(weekly)), models)
    params = replace(params, n_jobs=random_forest.workers(len(X), ctx.workers, ctx.budget_bytes))
    names = random_forest.feature_names(inputs, models)
    year = weekly["year"]
    trainable = ctx.unit_allowed[weekly.sample_unit]
    scoreable = np.zeros(len(weekly), dtype=bool)
    scoreable[ctx.rows] = True
    true_holdouts = set(run.true_holdout_years)
    fold_years = random_forest.folds(
        run.training_years,
        [y for y in run.holdout_years if y not in true_holdouts],
        run.true_holdout_years,
        params.years_per_fold,
    )
    log.info("\nCV: %s (%d folds)", params.name, len(fold_years))
    folds = random_forest.cross_validate(X, inputs.onset_lead, names, year, trainable, scoreable, fold_years, params)
    test = np.concatenate([rows for rows, _ in folds])
    daily = np.concatenate([probabilities for _, probabilities in folds])
    weekly_probs = random_forest.weekly_from_daily(daily, params.days_per_week)
    preds = Predictions.of_samples(weekly, test, weekly_probs, ctx.labels, daily=daily)
    scores.add(preds, params.name, "global", yearly=True)
    result.cv_preds[(params.name, "global")] = preds
    if with_final:
        train = np.flatnonzero(trainable & np.isin(year, run.fit_years))
        log.info("\nFitting %s on all %d training years...", params.name, len(run.fit_years))
        result.final_forest = random_forest.fit(X[train], inputs.onset_lead[train], names, params)


def _load_external_predictions(externals, results_dir, labels) -> list[tuple[ExternalPrediction, Predictions]]:
    """Load the recipe's ``evaluation.external_predictions``.

    Parameters
    ----------
    externals : list of hcwf.blend.recipe.ExternalPrediction
        The external prediction entries.
    results_dir : str or os.PathLike or None
        Where relative files resolve; None is allowed only when every file is absolute.
    labels : list of str
        The outcome bins the predictions must have.

    Returns
    -------
    list of tuple of (hcwf.blend.recipe.ExternalPrediction, hcwf.blend.predictions.Predictions)
        Each entry with its loaded predictions.

    Raises
    ------
    ValueError
        A file is relative and `results_dir` is None, or a file is invalid (see `load_external_predictions`).
    """
    if results_dir is None:
        relative = [e.file for e in externals if not os.path.isabs(e.file)]
        if relative:
            raise ValueError(
                f"extras.external_predictions files {relative} are relative, "
                "but no results_dir was given to resolve them against."
            )
        results_dir = ""
    return [(external, load_external_predictions(external.name, os.path.join(results_dir, external.file), labels))
            for external in externals]


def load_external_predictions(name: str, path: str, labels) -> Predictions:
    """Load one model's external CV predictions, checked against the outcome bins.

    The file is NetCDF (`~hcwf.blend.predictions.Predictions.to_netcdf`) or a pickled legacy table (``id``, ``year``,
    ``outcome`` and ``cv_<bin>`` columns).

    Parameters
    ----------
    name : str
        The model name, used in error messages.
    path : str
        The file; ``.nc`` is read as NetCDF, anything else as a pickled table.
    labels : list of str
        The outcome bins the predictions must have.

    Returns
    -------
    hcwf.blend.predictions.Predictions
        The predictions.

    Raises
    ------
    ValueError
        The bins differ, columns are missing or beyond the bins, the table is not a ``pandas.DataFrame`` or there are no
        rows.
    """
    where = f"External predictions '{name}' ({path})"
    if path.endswith(".nc"):
        preds = Predictions.open(path)
        if preds.labels != list(labels):
            raise ValueError(f"{where}: bins {preds.labels}, expected {list(labels)}.")
        if not len(preds):
            raise ValueError(f"{where} has no rows.")
        return preds
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{where} must be a pandas DataFrame, got {type(frame).__name__}.")
    required = ["id", "year", "outcome"] + [f"cv_{b}" for b in labels]
    missing = [c for c in required if c not in frame.columns]
    n_bins = len(labels) - 1
    unexpected = [f"cv_week{w}" for w in _present_weeks(frame.columns, "cv") if w > n_bins]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing columns {missing}")
        if unexpected:
            problems.append(f"unexpected columns {unexpected} beyond {n_bins} week bins")
        raise ValueError(f"{where}: {'; '.join(problems)}. Expected {required}.")
    if frame.empty:
        raise ValueError(f"{where} has no rows.")
    return Predictions.from_frame(frame.reset_index(drop=True), labels)


# ---------------------------------------------------------------------------
# 4-5: climatology and forecast baselines
# ---------------------------------------------------------------------------


def _present_weeks(columns, base: str) -> list[int]:
    """Return the sorted week numbers present as ``<base>_week<n>`` columns.

    Parameters
    ----------
    columns : iterable of str
        The column names.
    base : str
        The column prefix.

    Returns
    -------
    list of int
        The week numbers, ascending.
    """
    pattern = re.compile(rf"^{re.escape(base)}_week(\d+)$")
    return sorted(int(m.group(1)) for c in columns for m in [pattern.match(c)] if m)


def _row_sum(columns: list[np.ndarray]) -> np.ndarray:
    """Return the per-row sum of columns, a missing value counting as 0.

    The columns are added left to right, as the legacy table's row sums are.

    Parameters
    ----------
    columns : list of numpy.ndarray of float64, shape (row,)
        The columns; at least one.

    Returns
    -------
    numpy.ndarray of float64, shape (row,)
        The sums.
    """
    total = np.where(np.isnan(columns[0]), 0.0, columns[0])
    for values in columns[1:]:
        total = total + np.where(np.isnan(values), 0.0, values)
    return total


def _week_columns(ctx: _Context, rows: np.ndarray, base: str, *, logits: bool):
    """Return the week labels present as ``<base>_week<n>`` and their values for some rows.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    rows : numpy.ndarray of int, shape (row,)
        Positions into the weekly features.
    base : str
        The column prefix.
    logits : bool
        The columns hold logits; return probabilities (``expit``).

    Returns
    -------
    labels : list of str
        ``week1``, ``week2``, ... for the weeks present.
    values : list of numpy.ndarray of float64, shape (row,)
        The probabilities of each week.
    """
    weeks = _present_weeks(ctx.weekly.columns, base)
    values = [np.asarray(ctx.weekly.take(f"{base}_week{w}", rows), dtype=float) for w in weeks]
    if logits:
        values = [expit(v) for v in values]
    return [f"week{w}" for w in weeks], values


def _holdout_rows(ctx: _Context) -> np.ndarray:
    """Return the supported samples of the holdout years.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.

    Returns
    -------
    numpy.ndarray of int, shape (row,)
        Positions into the weekly features.
    """
    return ctx.rows[ctx.in_years(ctx.run.holdout_years)]


def _climatology_models(recipe: Recipe) -> list[tuple[str, str]]:
    """Return the climatology baselines to score.

    Parameters
    ----------
    recipe : hcwf.blend.recipe.Recipe
        The blend recipe.

    Returns
    -------
    list of tuple of (str, str)
        The ``(model, column prefix)`` of each baseline, in scoring order.
    """
    baselines = recipe.evaluation.baselines
    out = [("unc_clim_raw", f"{CLIMATOLOGY_FEATURE}_unc")] if baselines.unconditional else []
    if baselines.conditional:
        out.append(("clim_raw", CLIMATOLOGY_FEATURE))
    out += [(f"clim_raw_{window_tag(w)}", f"{CLIMATOLOGY_FEATURE}_{window_tag(w)}") for w in baselines.windows]
    return out


def _climatology_baselines(ctx: _Context, scores: _Scores) -> dict[str, Predictions]:
    """Turn the climatology logits into probabilities for the holdout years and queue them for scoring.

    A baseline that fails warns and is skipped.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    scores : _Scores
        Receives each baseline for scoring.

    Returns
    -------
    dict of str to hcwf.blend.predictions.Predictions
        The predictions by model name, for the MME.

    Raises
    ------
    ValueError
        ``unc_clim_raw`` lacks its earlier column; raised inside the per-model guard, so it surfaces as a warning.
    """
    rows = _holdout_rows(ctx)
    out = {}
    if not len(rows):
        return out
    for model, prefix in _climatology_models(ctx.recipe):
        try:
            labels, weeks = _week_columns(ctx, rows, prefix, logits=True)
            if not labels:
                warnings.warn(f"Skipping clim logit raw for prefix={prefix}; no _week* columns found")
                continue
            if model == "unc_clim_raw":
                earlier_col = f"{prefix}_earlier"
                if earlier_col not in ctx.weekly.columns:
                    raise ValueError("unc_clim_raw requires earlier_col and it must exist in wide_df.")
                earlier = np.clip(expit(np.asarray(ctx.weekly.take(earlier_col, rows), dtype=float)), 0.0, 1.0)
                later = np.clip(1.0 - (earlier + _row_sum(weeks)), 0.0, 1.0)
                # The earlier bin is renormalised with the rest.
                every = [earlier, *weeks, later]
                total = _row_sum(every)
                good = total > 0
                earlier, *weeks, later = [np.where(good, v / np.where(good, total, 1.0), v) for v in every]
            else:
                earlier = None
                later = np.clip(1.0 - _row_sum(weeks), 0.0, 1.0)
            preds = Predictions.of_samples(ctx.weekly, rows, np.column_stack([*weeks, later]), labels + ["later"],
                                           earlier=earlier)
            out[model] = preds
            scores.add(preds, model, "raw", yearly=False, warn_as=f"Error in clim logit '{model}'")
        except Exception as e:
            warnings.warn(f"Error in clim logit '{model}': {e}")
    return out


def _forecast_baselines(ctx: _Context, result: Evaluation, scores: _Scores) -> dict[str, Predictions]:
    """Queue each model's probabilities, raw and Platt-calibrated, for scoring.

    A model that fails warns and is skipped.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    result : Evaluation
        Receives the exported Platt weights.
    scores : _Scores
        Receives the predictions for scoring.

    Returns
    -------
    dict of str to hcwf.blend.predictions.Predictions
        The calibrated predictions by MME key.
    """
    run = ctx.run
    calibrated_by_key = {}
    for scoring in ctx.recipe.forecast_scoring():
        name = scoring.model
        variant = scoring.variant or ctx.recipe.cutoff_variant()
        label = name if variant == "base" else f"{name}_{variant}"
        base = probability_feature_base(name, variant)
        if scoring.raw:
            try:
                rows = _holdout_rows(ctx)
                labels, weeks = _week_columns(ctx, rows, base, logits=False)
                if not labels:
                    warnings.warn(f"Skipping raw for {name} ({variant}); no {base}_week* columns found")
                elif len(rows):
                    later = np.clip(1.0 - _row_sum(weeks), 0.0, 1.0)
                    preds = Predictions.of_samples(ctx.weekly, rows, np.column_stack([*weeks, later]), labels + ["later"])
                    scores.add(preds, f"{label}_raw", "raw", yearly=False,
                               warn_as=f"Error in raw forecast '{name}'")
            except Exception as e:
                warnings.warn(f"Error in raw forecast '{name}': {e}")

        if scoring.calibrated:
            try:
                preds, weights = _calibrated(ctx, base, name, variant, export=scoring.platt_weights)
                if preds is not None and len(preds):
                    key = name if variant == "base" else f"{name}_calibrated_{variant}"
                    calibrated_by_key[key] = preds
                    scores.add(preds, f"{name}_calibrated" if variant == "base" else key, "calibrated", yearly=False,
                               warn_as=f"Error in calibrated forecast '{name}'")
                    if weights is not None:
                        result.platt_weights[f"platt_weights_{label}_calibrated_df{run.output_tag}"] = weights
            except Exception as e:
                warnings.warn(f"Error in calibrated forecast '{name}': {e}")
    return calibrated_by_key


def _calibrated(ctx: _Context, base: str, name: str, variant: str, *, export: bool):
    """Return Platt-calibrated CV predictions for the holdout years.

    The fits use the training and holdout years' samples (outside the true holdouts, in the dissemination units).

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    base : str
        The column prefix of the model's probabilities.
    name : str
        The model name.
    variant : str
        The cutoff variant.
    export : bool
        Also fit the weights on the training years' samples.

    Returns
    -------
    preds : hcwf.blend.predictions.Predictions or None
        The calibrated predictions; None without the model's columns or folds.
    weights : pandas.DataFrame or None
        The weights, None unless `export` and there are predictions.
    """
    run = ctx.run
    rows = ctx.rows[ctx.in_years([*run.training_years, *run.holdout_years])]
    weeks, values = _week_columns(ctx, rows, base, logits=False)
    if not weeks:
        warnings.warn(f"Skipping calibrated for {name} ({variant}); no {base}_week* columns found")
        return None, None
    labels = weeks + ["later"]
    P = np.column_stack([*values, np.maximum(0.0, 1.0 - _row_sum(values))])
    outcome = ctx.weekly.outcome_codes(rows, labels)
    year = ctx.weekly.take("year", rows)
    allowed = ctx.allowed[ctx.in_years([*run.training_years, *run.holdout_years])]
    trainable = ~np.isin(year, run.true_holdout_years) & allowed
    folds = calibration.cross_validated(P, outcome, year, trainable, run.holdout_years, workers=ctx.workers)
    preds = None
    if folds:
        positions = np.concatenate([test for _, test, _ in folds])
        preds = Predictions.of_samples(ctx.weekly, rows[positions], np.concatenate([c for _, _, c in folds]), labels)
    weights = None
    if export and preds is not None:
        train = np.isin(year, run.training_years)
        weights = calibration.export_weights(P[train], outcome[train], labels)
    return preds, weights


# ---------------------------------------------------------------------------
# 6: the multi-model ensemble
# ---------------------------------------------------------------------------


def _mme(ctx: _Context, clim_preds, calibrated_preds, result: Evaluation, scores: _Scores) -> None:
    """Queue the MME of climatology and calibrated forecasts for scoring.

    The weights are RPS-optimised, or the given ones (see ``Mme.optimize`` and ``Mme.weights`` of
    `hcwf.blend.recipe`). A variant whose sources are missing warns and is skipped.

    Parameters
    ----------
    ctx : _Context
        The evaluation's shared state.
    clim_preds : dict of str to hcwf.blend.predictions.Predictions
        The climatology predictions by model name.
    calibrated_preds : dict of str to hcwf.blend.predictions.Predictions
        The calibrated predictions by MME key.
    result : Evaluation
        Receives the optimised weights.
    scores : _Scores
        Receives the MME predictions for scoring.
    """
    settings = ctx.recipe.evaluation.mme
    if not (settings.enabled and clim_preds):
        return
    run = ctx.run
    names = [source.name for source in settings.sources]
    weight_cols = [f"w_{name}" for name in names]
    optimise = settings.weights is None and (settings.optimize == "always" or (
        settings.optimize == "auto" and sorted(run.holdout_years) == list(range(2000, 2025))
    ))
    clim_model = next((s.name for s in settings.sources if s.source == "clim"), names[0])
    for variant in settings.variants:
        log.info("\n=== MME optimization (RPS-only) variant: %s ===", variant)
        sources = _mme_sources(settings.sources, clim_preds, calibrated_preds)
        if sources is None:
            continue
        P_full, common = mme.align(sources)
        if any(source.labels != ctx.labels for source in sources):
            warnings.warn("MME sources must all have the outcome bins as their labels. Skipping.")
            continue
        base = sources[0]
        ids = base.ids[common]
        allowed = ctx.allowed_ids(ids)
        P_allowed = [metrics.clean(P[allowed]) for P in P_full]
        Y_allowed = metrics.one_hot(base.outcome[common][allowed], len(ctx.labels))

        stem = f"mme_optimized_weights{run.cutoff_tag}_{clim_model}_{variant}"
        if optimise:
            log.info("  Running MME optimization...")
            weights, rps = mme.optimise_weights(P_allowed, Y_allowed)
            result.mme_weights[f"{stem}{run.year_tag}"] = pd.DataFrame(
                [{"objective": "rps", **dict(zip(weight_cols, weights)), "rps": rps}]
            )
        else:
            weights = _given_mme_weights(settings.weights, names)
            if weights is None:
                warnings.warn("MME weights were not given (blend.mme.weights) and the holdout years are not "
                              "2000-2024, so they are not optimised. Skipping.")
                continue
            log.info("  Using the given MME weights...")
            rps = mme.mixed_rps(weights, P_allowed, Y_allowed)

        log.info("  MME weights: %s", dict(zip(names, np.round(weights, 4))))
        log.info("  RPS (pooled, allowed): %.6f", rps)
        preds = base.take(common).with_probabilities(mme.mix(weights, P_full))
        model = f"mme_{variant}_{clim_model}_opt_rps"
        scores.add(preds, model, "mme", yearly=False)
        log.info("  Added MME model: %s", model)


def _mme_sources(sources, clim_preds, calibrated_preds) -> list[Predictions] | None:
    """Return each MME source's CV predictions, in order.

    Parameters
    ----------
    sources : list
        The recipe's MME sources.
    clim_preds : dict of str to hcwf.blend.predictions.Predictions
        The climatology predictions by model name.
    calibrated_preds : dict of str to hcwf.blend.predictions.Predictions
        The calibrated predictions by MME key.

    Returns
    -------
    list of hcwf.blend.predictions.Predictions or None
        The predictions, or None (with a warning) if any source is missing.
    """
    out = []
    for source in sources:
        if source.source == "clim":
            preds = clim_preds.get(source.name)
            if preds is None:
                warnings.warn(f"MME blend_model '{source.name}' (source: clim) not found.")
                return None
        else:
            key = source.name if source.cal_variant == "base" else f"{source.name}_calibrated_{source.cal_variant}"
            preds = calibrated_preds.get(key)
            if preds is None:
                warnings.warn(f"MME blend_model '{source.name}' (key: {key}) not found.")
                return None
        out.append(preds)
    return out


def _given_mme_weights(given, names) -> np.ndarray | None:
    """Return the weights of ``blend.mme.weights`` as an array in source order.

    Parameters
    ----------
    given : dict of str to float or str or os.PathLike or None
        A mapping of source name to weight, or a weights CSV (as a run writes them, read at its ``rps`` row, else its
        first row); None when none were given.
    names : list of str
        The source names, in order.

    Returns
    -------
    numpy.ndarray of float64, shape (source,) or None
        The weights; None when none were given.

    Raises
    ------
    ValueError
        The CSV lacks a ``w_<name>`` column.
    """
    if given is None:
        return None
    if isinstance(given, dict):
        return np.array([given[name] for name in names], dtype=float)
    table = pd.read_csv(given, float_precision="round_trip")  # the weights as they were written
    missing = [f"w_{name}" for name in names if f"w_{name}" not in table.columns]
    if missing:
        raise ValueError(f"blend.mme.weights: {given} has no column {', '.join(missing)}")
    row = table[table["objective"] == "rps"].iloc[0] if "objective" in table.columns else table.iloc[0]
    return np.array([row[f"w_{name}"] for name in names], dtype=float)
