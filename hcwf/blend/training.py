"""Training a blend: every stage in memory, from input files to the fitted model.

`train(rc)` runs the stages in order: the thresholds (`hcwf.io.input.thresholds_of`), the observed onsets, the forecast
onsets of each model (units the climatology keeps), the climatology and the weekly features (`hcwf.blend.features`, on
HCWF's one detection adapter), then the cross-validated evaluation and the final fit (`hcwf.blend.evaluation`), and
returns a `TrainingResult`. `rc` is the run's RunConfig: it holds the blend-only settings (`rc.blend`, a
`hcwf.blend.recipe.Recipe`) and the resolved onset setup (`rc.onset`).

`TrainingResult.save` writes blending's output contract (docs/pipeline-redesign.md in onset_blending): the evaluation's
artifacts under the contract's own names (not HCWF's short ones), plus the chosen intermediates. The trained blend
(`trained.blend`, `hcwf.blend.trained.TrainedBlend.from_training`) is written beside them: it keeps the run's resolved
config, the recipe, the fitted model and forest, the observed onsets, the units, the thresholds and reference onsets,
and the climatology's years, which is all a forecast needs.

Workers are the run's (`rc.parallel`; `hcwf.parallel`), the memory budget is half of the memory available now;
every stage runs on the run's worker pool.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from hcwf.blend import evaluation as evaluation_stage
from hcwf.blend import features
from hcwf.blend.evaluation import Evaluation
from hcwf.blend.recipe import Recipe
from hcwf.blend.samples import DailyInputs, ForecastOnsets, ObservedOnsets, Weekly
from hcwf.cache import atomic_write
from hcwf.config.model import RunConfig
from hcwf.io import thresholds as thresholds_io
from hcwf.io.reference import release_reference
from hcwf.onset.climatology import ReferenceOnset
from hcwf.references.kde import Climatology
from hcwf.parallel import spec_workers
from hcwf import parallel as pool

log = logging.getLogger(__name__)

__all__ = ["Inputs", "STAGE_TABLES", "TrainingResult", "prepare", "resources", "train"]


def resources(rc: RunConfig, memory_budget: int | None = None) -> tuple[int, int | None]:
    """Return the workers and the memory the run allows.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run; its ``parallel`` setting gives the workers: one with ``parallel: false``, else ``parallel_workers``
        or the cores available.
    memory_budget : int, optional
        The budget in bytes; ``None`` takes half of the memory available now.

    Returns
    -------
    workers : int
        The number of worker processes.
    budget : int or None
        The memory budget in bytes.
    """
    return spec_workers(rc.parallel, 1 << 20), memory_budget or pool.memory_budget()


@dataclass
class Inputs:
    """Everything the blend is fitted on, stage by stage."""

    thresholds: object
    """The onset thresholds, as ``hcwf.io.input.thresholds_of(rc, model)`` returns them."""
    reference: ReferenceOnset | None
    """The reference onset (``cutoff_mode`` ``ref_onset``), if the recipe has one."""
    observed: ObservedOnsets
    """The observed onsets."""
    forecasts: dict[str, ForecastOnsets]
    """The forecast onsets of each model, by model name."""
    climatology: Climatology
    """The conditional climatology."""
    weekly: Weekly
    """The weekly samples and features."""
    timings: dict[str, float] = field(default_factory=dict)
    """The seconds each stage took, by stage name."""
    forest_inputs: DailyInputs | None = None
    """The random forest blend's inputs, when it is configured."""

    @property
    def units(self) -> tuple[str, ...]:
        """The units with samples, sorted."""
        return tuple(sorted({str(unit) for unit in self.weekly.units}))


STAGE_TABLES = ("observed", "forecasts", "climatology", "weekly")
"""The stage tables `prepare` can keep; the default keeps all of them."""


def prepare(rc: RunConfig, *, test_years: tuple[int, int] | None = None,
            keep: tuple[str, ...] = STAGE_TABLES) -> Inputs:
    """Compute the stages of a run: observed and forecast onsets, the climatology and the weekly features.

    The random forest's inputs are added when the recipe has one. Forecast and climatology arrays not in `keep` are
    released once the features are built; the observed onsets and weekly features are always kept.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run, with a ``blend`` section.
    test_years : tuple of (int, int), optional
        Overrides the climatology's test window (a live season).
    keep : tuple of str, default STAGE_TABLES
        The stage tables to keep: any of ``"observed"``, ``"forecasts"``, ``"climatology"``, ``"weekly"``.

    Returns
    -------
    Inputs
        The stages' arrays and how long each took.
    """
    recipe, setup = _recipe(rc), rc.onset
    workers, budget = resources(rc)
    timings: dict[str, float] = {}

    def lap(stage: str, started: float) -> None:
        """Record and log how long a stage took."""
        timings[stage] = time.perf_counter() - started
        log.info("%s in %.1fs", stage, timings[stage])

    started = time.perf_counter()
    # The blend reads its observations in the workers: the threshold years are read here without holding them, and their
    # reference rainfall goes once the thresholds are computed.
    thresholds = thresholds_io.thresholds_of(replace(rc, evaluation=replace(rc.evaluation, years=())), rc.models[0])
    release_reference()
    reference = features.reference_onset(recipe)
    observed = features.observed(rc, recipe, thresholds, reference, workers=workers)
    lap("observed onsets", started)

    # Only units the climatology keeps make samples, so the forecasts are read for those alone.
    blendable = observed.blendable(recipe.climatology.min_onset_years)
    forecasts = {}
    for model in recipe.models:
        started = time.perf_counter()
        forecasts[model.name] = features.forecast(rc, recipe, model.name, thresholds, reference,
                                                  keep_ids=blendable, workers=workers, budget_bytes=budget,
                                                  needed_only=True)
        lap(f"{model.name} forecast onsets", started)

    started = time.perf_counter()
    forecast_years = sorted({y for f in forecasts.values() for y in f.years})
    climatology = features.climatology(recipe, setup, observed, forecast_years, test_years=test_years,
                                       workers=workers)
    lap("climatology", started)

    started = time.perf_counter()
    weekly = features.weekly(recipe, setup.definition, climatology, observed, forecasts)
    lap("weekly features", started)
    forest_inputs = None
    forest = recipe.evaluation.random_forest
    if forest is not None:
        started = time.perf_counter()
        forest_inputs = features.daily_inputs(recipe, climatology, observed, forecasts, max_day=forest.max_day)
        lap("random forest inputs", started)
    if "forecasts" not in keep:
        for model_onsets in forecasts.values():
            model_onsets.release()
    if "climatology" not in keep:
        climatology.release()
    return Inputs(thresholds=thresholds, reference=reference, observed=observed, forecasts=forecasts,
                  climatology=climatology, weekly=weekly, timings=timings, forest_inputs=forest_inputs)


@dataclass
class TrainingResult:
    """A training run: its settings, the stages' arrays and the evaluation (with the fitted model and forest)."""

    rc: RunConfig
    """The run's configuration."""
    inputs: Inputs
    """The stages' arrays."""
    evaluation: Evaluation
    """The cross-validated evaluation, with the fitted model and forest."""

    @property
    def recipe(self) -> Recipe:
        """The blend's settings (the run's ``blend`` section)."""
        return self.rc.blend

    @property
    def setup(self):
        """The resolved onset setup (`hcwf.config.onset.OnsetSetup`)."""
        return self.rc.onset

    @property
    def model(self):
        """The operational model: the final formula fitted on every training year."""
        return self.evaluation.final_model

    @property
    def forest(self):
        """The operational random forest, when the recipe has one."""
        return self.evaluation.final_forest

    def save(self, directory, *, intermediates=()) -> Path:
        """Write the evaluation's artifacts, ``trained.blend`` and the chosen intermediates into a new directory.

        All or nothing: the files are written to a staging directory that is moved into place at the end.
        Intermediates are opt-in: the weekly features (`hcwf.blend.samples.Weekly.open` reads them) and the observed
        onsets, both as NetCDF.

        Parameters
        ----------
        directory : str or pathlib.Path
            The directory to create; it may exist if it is empty.
        intermediates : iterable of {"observed", "weekly"}, default ()
            The intermediates to write: ``observed_onsets.nc`` and ``weekly_features.nc``.

        Returns
        -------
        pathlib.Path
            The directory written.

        Raises
        ------
        FileExistsError
            `directory` exists and is not empty.
        ValueError
            An intermediate is not ``"observed"`` or ``"weekly"``.
        """
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        unknown = [name for name in intermediates if name not in ("observed", "weekly")]
        if unknown:
            raise ValueError(f"unknown intermediate {unknown[0]!r}")

        def write(staging: Path) -> None:
            """Write every file into the staging directory."""
            staging.mkdir()
            self.evaluation.save(staging, pickles=self.recipe.outputs.pickles)
            from hcwf.blend.trained import TRAINED_FILENAME, TrainedBlend

            TrainedBlend.from_training(self).save(staging / TRAINED_FILENAME)
            for name in intermediates:
                if name == "observed":
                    self.inputs.observed.data.to_netcdf(staging / "observed_onsets.nc")
                else:
                    self.inputs.weekly.to_netcdf(staging / "weekly_features.nc")

        if directory.exists():
            directory.rmdir()
        atomic_write(directory, write)
        return directory


def _recipe(rc: RunConfig) -> Recipe:
    """Return the run's blend settings.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run.

    Returns
    -------
    hcwf.blend.recipe.Recipe
        ``rc.blend``.

    Raises
    ------
    ValueError
        The run has no ``blend`` section, so it trains nothing.
    """
    if rc.blend is None:
        raise ValueError("the run has no `blend:` section, so there is no blend to train")
    return rc.blend


def train(rc: RunConfig) -> TrainingResult:
    """Train a blend: every stage in memory.

    The stages are the inputs, the cross-validated evaluation, and the operational model (and the random forest, when
    configured) fitted on every training year.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The run, with a ``blend`` section.

    Returns
    -------
    TrainingResult
        The settings, the stages' arrays and the evaluation.

    """
    recipe = _recipe(rc)
    workers, budget = resources(rc)
    inputs = prepare(rc, keep=("observed", "weekly"))

    # The final fit runs alongside the global CV folds, on their design matrix.
    started = time.perf_counter()
    dissemination = evaluation_stage.dissemination_ids(rc.observations.units, inputs.weekly)
    evaluation = evaluation_stage.evaluate(
        inputs.weekly, recipe, dissemination, workers=workers, budget_bytes=_budget_now(budget), final=True,
        forest_inputs=inputs.forest_inputs,
    )
    inputs.timings["evaluation and final fit"] = time.perf_counter() - started
    log.info("evaluation and final fit in %.1fs", inputs.timings["evaluation and final fit"])
    return TrainingResult(rc=rc, inputs=inputs, evaluation=evaluation)


def _budget_now(budget: int | None) -> int | None:
    """Return the memory budget, within half of what is available now that the stage arrays exist.

    Parameters
    ----------
    budget : int or None
        The budget set when the run started, in bytes.

    Returns
    -------
    int or None
        The smaller of `budget` and the budget available now; ``None`` if neither is limited.
    """
    budgets = [b for b in (budget, pool.memory_budget()) if b is not None]
    return min(budgets) if budgets else None
