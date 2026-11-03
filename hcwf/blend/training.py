"""Training a blend: every stage in memory, from input files to the fitted model.

`train(cfg, recipe, setup)` runs the stages in order: the thresholds (`hcwf.io.input.load_thresholds`), the observed
onsets, the forecast onsets of each model (units the climatology keeps), the climatology and the weekly features
(`hcwf.blend.features`, on HCWF's one detection adapter), then the cross-validated evaluation and the final fit
(`hcwf.blend.evaluation`), and returns a `TrainingResult`. `cfg` is HCWF's flat run configuration, `recipe` the
blend-only settings (`hcwf.blend.recipe.Recipe`) and `setup` the resolved onset setup (`OnsetSetup`).

`TrainingResult.save` writes blending's output contract (docs/pipeline-redesign.md in onset_blending): the evaluation's
artifacts under the contract's own names (not HCWF's short ones), plus the chosen intermediates. The trained blend
(`trained.blend`, `hcwf.blend.trained.TrainedBlend.from_training`) is written beside them: it keeps the run's resolved
config, the recipe, the fitted model and forest, the observed onsets, the units, the thresholds and reference onsets,
and the climatology's years, which is all a forecast needs.

Workers and the memory budget are the run's: `parallel` / `parallel_workers` / `memory_budget` of `cfg`
(`hcwf.utils.workers`); every stage runs on the run's worker pool.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from hcwf.blend import evaluation as evaluation_stage
from hcwf.blend import features
from hcwf.blend.evaluation import Evaluation
from hcwf.blend.recipe import Recipe
from hcwf.blend.samples import DailyInputs, ForecastOnsets, ObservedOnsets, Weekly
from hcwf.cache import atomic_write
from hcwf.io import input as io_input
from hcwf.onset.climatology import ReferenceOnset
from hcwf.onset.climatology_kde import Climatology
from hcwf.utils import workers as pool

log = logging.getLogger(__name__)

__all__ = ["Inputs", "STAGE_TABLES", "TrainingResult", "prepare", "resources", "train"]


def resources(cfg: dict) -> tuple[int, int | None]:
    """(workers, memory budget in bytes) the run allows: one worker with `parallel: false`, else `parallel_workers`
    or the cores available; the budget is `memory_budget` or half of the memory available now."""
    workers = pool.worker_count(cfg, tasks=1 << 20)
    budget = cfg.get("memory_budget") or pool.memory_budget()
    return workers, budget


@dataclass
class Inputs:
    """Everything the blend is fitted on, stage by stage."""

    thresholds: object                      # hcwf.io.input.load_thresholds(**cfg)
    reference: ReferenceOnset | None        # the reference onset (cutoff_mode ref_onset), if the recipe has one
    observed: ObservedOnsets
    forecasts: dict[str, ForecastOnsets]
    climatology: Climatology
    weekly: Weekly
    timings: dict[str, float] = field(default_factory=dict)
    forest_inputs: DailyInputs | None = None   # the random forest blend's, when it is configured

    @property
    def units(self) -> tuple[str, ...]:
        """The units with samples, sorted."""
        return tuple(sorted({str(unit) for unit in self.weekly.units}))


STAGE_TABLES = ("observed", "forecasts", "climatology", "weekly")


def prepare(cfg: dict, recipe: Recipe, setup, *, test_years: tuple[int, int] | None = None,
            keep: tuple[str, ...] = STAGE_TABLES) -> Inputs:
    """Observed and forecast onsets, the climatology and the weekly features of a run (and the random forest's
    inputs when it has one).

    Forecast and climatology arrays not in `keep` are released once the features are built; the observed onsets and
    weekly features are always kept. test_years overrides the climatology's test window (a live season)."""
    workers, budget = resources(cfg)
    timings: dict[str, float] = {}

    def lap(stage: str, started: float) -> None:
        timings[stage] = time.perf_counter() - started
        log.info("%s in %.1fs", stage, timings[stage])

    started = time.perf_counter()
    thresholds = io_input.load_thresholds(**cfg)
    reference = features.reference_onset(recipe)
    observed = features.observed(cfg, recipe, setup, thresholds, reference, workers=workers)
    lap("observed onsets", started)

    # Only units the climatology keeps make samples, so the forecasts are read for those alone.
    blendable = observed.blendable(recipe.climatology.min_onset_years)
    forecasts = {}
    for model in recipe.models:
        started = time.perf_counter()
        forecasts[model.name] = features.forecast(cfg, recipe, setup, model.name, thresholds, reference,
                                                  keep_ids=blendable, workers=workers, budget_bytes=budget)
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

    cfg: dict
    recipe: Recipe
    setup: object           # hcwf.lib.onset_config.OnsetSetup
    inputs: Inputs
    evaluation: Evaluation

    @property
    def model(self):
        """The operational model: the final formula fitted on every training year."""
        return self.evaluation.final_model

    @property
    def forest(self):
        """The operational random forest, when the recipe has one."""
        return self.evaluation.final_forest

    def save(self, directory, *, intermediates=()) -> Path:
        """Write the evaluation's artifacts, `trained.blend` and the chosen intermediates into a new directory, all or
        nothing. Intermediates are opt-in (default none): the weekly features (`Weekly.open` reads them) and the
        observed onsets as NetCDF."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        unknown = [name for name in intermediates if name not in ("observed", "weekly")]
        if unknown:
            raise ValueError(f"unknown intermediate {unknown[0]!r}")

        def write(staging: Path) -> None:
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


def train(cfg: dict, recipe: Recipe, setup) -> TrainingResult:
    """Every stage of a blend, in memory: inputs, cross-validated evaluation, and the operational model (and the
    random forest, when configured) fitted on every training year."""
    workers, budget = resources(cfg)
    inputs = prepare(cfg, recipe, setup, keep=("observed", "weekly"))

    # The final fit runs alongside the global CV folds, on their design matrix.
    started = time.perf_counter()
    dissemination = evaluation_stage.dissemination_ids(io_input.units_spec(cfg, "observations"), inputs.weekly)
    evaluation = evaluation_stage.evaluate(
        inputs.weekly, recipe, dissemination, workers=workers, budget_bytes=_budget_now(budget), final=True,
        forest_inputs=inputs.forest_inputs,
    )
    inputs.timings["evaluation and final fit"] = time.perf_counter() - started
    log.info("evaluation and final fit in %.1fs", inputs.timings["evaluation and final fit"])
    return TrainingResult(cfg=cfg, recipe=recipe, setup=setup, inputs=inputs, evaluation=evaluation)


def _budget_now(budget: int | None) -> int | None:
    """The memory budget, within half of what is available now that the stage arrays exist."""
    budgets = [b for b in (budget, pool.memory_budget()) if b is not None]
    return min(budgets) if budgets else None
