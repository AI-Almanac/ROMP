"""The blend-specific settings of a run: what the `blend:` section of the run YAML holds, and nothing HCWF's config has.

HCWF's resolved run config (`hcwf.config`, `hcwf.lib.config_schema`, `hcwf.lib.onset_config.OnsetSetup`) already says
where the observations and forecasts are (obs_*, model_* keys, `dataset_schema` entries), the onset definition and its
run options (onset_definition / onset_rule, rain_resolution, missing, negatives, gaps, onset_horizon), the backend,
years and workers. A `Recipe` is used *together with* that config: it holds only what blending adds, each field naming
its BlendConfig origin. `recipe_from_config(flat)` builds it from the flat config's `blend_*` keys (the `blend:` section
of the YAML; `hcwf-run convert` writes that section from a BlendConfig or a legacy spec set).

Pure data with validation: no file is read, no YAML parsed. Rules that need the onset definition (rain-window tokens,
the formulas' feature names) are checked by `Recipe.check(definition)`. The derived names (feature columns,
probability variants, labels, artifact tags) are computed here once.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

from hcwf.lib.dataset_schema import DatasetError, parse_years, suggest

from hcwf.blend.design import Formula, WindowVariant
from hcwf.blend.random_forest import Params as ForestParams
from hcwf.onset.model import Definition
from hcwf.stats.bins import LeadBins

__all__ = [
    "Baselines", "ClimatologySpec", "Evaluation", "ExternalPrediction", "Features",
    "ForecastScoring", "Mme", "MmeSource", "Model", "OnsetExtras", "Outputs", "RainHorizon", "RainPredictor", "Recipe",
    "RecipeError", "ReferenceSpec", "Years", "recipe_from_config",
]

CLIMATOLOGY_FEATURE = "prob_clim"
"""Stem of the climatology's feature columns (blending `config.CLIMATOLOGY_FEATURE`)."""
VARIANTS = ("base", "fixed_cutoff", "ref_onset")
"""The forecast probability variants: the search starts at lead 1, at the fixed cutoff date, at the reference onset."""
VARIANT_SUFFIX = {"base": "", "fixed_cutoff": "_fixed_cutoff", "ref_onset": "_ref", "no_ref_filter": ""}
CUTOFF_MODES = ("fixed_cutoff", "ref_onset", "no_ref_filter")
CV_METHODS = ("global", "local", "clusters")
WINDOW_TOKENS = ("trigger", "window", "dry_spell", "sum_window", "follow", "min_dry")
_MONTH_DAY = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
_RAIN_TRANSFORMS = ("identity", "none", "", "sqrt", "fourth_root", "log1p")
_MODEL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)*")


class RecipeError(ValueError):
    """A recipe that cannot be used; lists every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Invalid blend recipe:\n" + "\n".join(f"  - {p}" for p in self.problems))


# ---------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RainHorizon:
    """How far forecast rainfall is trusted for rain predictors (models[].rain_horizon)."""

    max_day: int | None = None
    """rain_horizon.max_day: the last rainfall lead day; None is the legacy probability_days + 10."""
    policy: Literal["legacy", "exact", "truncate"] | None = None
    """rain_horizon.policy: "exact" rejects later days, "truncate" ignores them."""

    @property
    def resolved_policy(self) -> str:
        if self.policy is not None:
            return self.policy
        return "legacy" if self.max_day is None else "exact"


@dataclass(frozen=True)
class RainPredictor:
    """A weekly rain feature: the max, min or (diff) excess over the threshold of rolling sums (models[].rain_predictors)."""

    agg: Literal["diff", "max", "min"]
    window: int | str = "trigger"
    """The window in days, or a token of the onset definition (`Definition.windows()`): trigger, dry_spell
    (the dry window), follow, min_dry (the dry run's length)."""


@dataclass(frozen=True)
class Model:
    """What blending adds to one forecast model of HCWF's model_list (BlendConfig.models[]): its files, variable, dims and
    unit_cvt are HCWF's model_* keys."""

    name: str
    """models[].name; prefixes every column derived from it."""
    lead_days: tuple[int, int] = (1, 45)
    """models[].lead_days min and max."""
    max_member: int | None = None
    """models[].max_member: members whose index is at most this are read."""
    rain_horizon: RainHorizon = field(default_factory=RainHorizon)
    rain_predictors: tuple[RainPredictor, ...] = ()
    rain_transform: str | float | None = None
    """models[].rain_transform; None uses features.rain_transform."""


# ---------------------------------------------------------------------------------------------------------------
# The reference onset and what the onset search adds
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceSpec:
    """The reference (climatological) onset (onset.reference_onset): a month-day every year, or a table of dates
    per year and/or unit (`hcwf.onset.climatology.ReferenceOnset.read`)."""

    month_day: str | None = None
    file: str | None = None
    date_col: str | None = None
    year_col: str | None = None
    day_col: str | None = None
    base_date: str | None = None
    unit_col: str | None = None

    def __post_init__(self):
        if (self.month_day is None) == (self.file is None):
            raise RecipeError(["reference onset: give exactly one of a month-day or a file"])
        if self.file is not None and self.date_col is None and not (self.year_col and self.day_col and self.base_date):
            raise RecipeError(["reference onset: a table needs a date column, or year, day and base date"])


@dataclass(frozen=True)
class OnsetExtras:
    """What the onset search needs besides HCWF's onset keys (BlendConfig.onset, observations). The onset rule, its
    thresholds (`thresholds`, `threshold_scale`, `threshold_table`, `threshold_years`), the units (`units_*`) and the
    reading options are HCWF config keys."""

    fixed_cutoff: str = "06-02"
    """onset.fixed_cutoff: the start of the fixed_cutoff forecast probabilities."""
    reference: ReferenceSpec | None = None
    """onset.reference_onset."""
    forecast_missing: Literal["keep", "flag", "zero"] = "keep"
    """The missing-rain policy of forecasts; observations take HCWF's `missing` (observations.missing_rain)."""
    observed_horizon: Literal["reject", "truncate", "censor"] = "reject"
    """A follow-up past the end of a season's observations (HCWF's onset_horizon is the forecasts')."""


# ---------------------------------------------------------------------------------------------------------------
# Climatology, features
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClimatologySpec:
    """The KDE climatology (BlendConfig.climatology), the arguments of `hcwf.onset.climatology_kde`."""

    issue_end: str = "07-31"
    forecast_window: int | None = None
    """None is features.probability_days."""
    horizons: tuple[tuple[str, str, int], ...] = ()
    """climatology.horizons as (start, end, forecast_window) month-days; exclusive with forecast_window."""
    cv_by_year: bool = True
    min_onset_years: int = 10
    train_years: tuple[int, int] | None = None
    test_years: tuple[int, int] | None = None
    windows: tuple[tuple[int, int], ...] = ()
    workers: int | None = None

    def horizon_dicts(self) -> list[dict] | None:
        """The horizons as `issue_grid` takes them (None without)."""
        return [{"start_md": a, "end_md": b, "forecast_window": w} for a, b, w in self.horizons] or None


@dataclass(frozen=True)
class Features:
    """The weekly features (BlendConfig.features)."""

    bins: LeadBins = field(default_factory=lambda: LeadBins.uniform(4, 7))
    """features.bins.count and days: `LeadBins.uniform(count, days)` (outcomes week1..weekN, later)."""
    probability_days: int = 28
    trim_after_true_onset: bool = True
    rain_transform: str | float = "identity"

    @property
    def count(self) -> int:
        return self.bins.n_inner

    @property
    def days(self) -> int:
        """Days per bin (bins are uniform)."""
        return self.bins.inner[0][1] - self.bins.inner[0][0] + 1


# ---------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------


def _years(values) -> tuple[int, ...]:
    return tuple(int(y) for y in values)


@dataclass(frozen=True)
class Years:
    """evaluation.years."""

    training: tuple[int, ...]
    cv_holdout: tuple[int, ...]
    true_holdout: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("training", "cv_holdout", "true_holdout"):
            object.__setattr__(self, name, _years(getattr(self, name)))

    @property
    def holdout(self) -> tuple[int, ...]:
        """Every year scored: true holdouts first, as the evaluation orders them."""
        return tuple(dict.fromkeys([*self.true_holdout, *self.cv_holdout]))


@dataclass(frozen=True)
class Baselines:
    """evaluation.climatology_baselines."""

    unconditional: bool = True
    conditional: bool = True
    windows: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ForecastScoring:
    """evaluation.forecasts[]: raw and calibrated scoring of a model's own probabilities."""

    model: str
    variant: Literal["base", "fixed_cutoff", "ref_onset"] | None = None
    raw: bool = True
    calibrated: bool = True
    platt_weights: bool = False


@dataclass(frozen=True)
class MmeSource:
    """evaluation.mme.sources[]."""

    name: str
    source: Literal["clim", "forecast"] = "clim"
    cal_variant: Literal["base", "fixed_cutoff", "ref_onset"] = "base"


@dataclass(frozen=True)
class Mme:
    """evaluation.mme: a multi-model ensemble with RPS-optimised weights."""

    enabled: bool = False
    variants: tuple[str, ...] = ("fixed_cutoff",)
    sources: tuple[MmeSource, ...] = ()
    optimize: Literal["auto", "always", "never"] = "auto"
    weights: dict[str, float] | str | None = None
    """Fixed weights instead of optimised ones: a mapping of source name to weight (non-negative, summing to 1), or
    the path of a weights CSV an earlier run wrote (columns objective, w_<source>..., rps; the "rps" row is used)."""


@dataclass(frozen=True)
class ExternalPrediction:
    """evaluation.external_predictions[]: CV predictions made elsewhere, scored with the rest."""

    name: str
    file: str
    method: str = "global"


@dataclass(frozen=True)
class Evaluation:
    """How the blend is cross-validated and scored (BlendConfig.evaluation)."""

    years: Years
    cutoff_mode: Literal["fixed_cutoff", "ref_onset", "no_ref_filter"] = "fixed_cutoff"
    cv_methods: tuple[Literal["global", "local", "clusters"], ...] = ("global",)
    cluster_var: str = "cluster"
    formulas: tuple[tuple[str, Formula], ...] = ()
    """evaluation.formulas in order (name, Formula)."""
    final_formula: str | None = None
    window_variants: tuple[WindowVariant, ...] = ()
    baselines: Baselines = field(default_factory=Baselines)
    forecasts: tuple[ForecastScoring, ...] | None = None
    """None scores every model, raw and calibrated."""
    mme: Mme = field(default_factory=Mme)
    external_predictions: tuple[ExternalPrediction, ...] = ()
    random_forest: ForestParams | None = None
    """evaluation.random_forest as the forest's Params (days_per_week is the bins' days)."""

    def formula_map(self) -> dict[str, Formula]:
        return dict(self.formulas)


@dataclass(frozen=True)
class Outputs:
    """outputs."""

    dir: str | None = None
    intermediates: tuple[Literal["observations", "forecasts", "climatology", "features"], ...] = ()
    pickles: bool = False


# ---------------------------------------------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """The blend settings of a run (BlendConfig, minus what HCWF's config holds)."""

    name: str
    models: tuple[Model, ...]
    onset: OnsetExtras
    evaluation: Evaluation
    climatology: ClimatologySpec = field(default_factory=ClimatologySpec)
    features: Features = field(default_factory=Features)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        self.validate()

    # -- lookups ---------------------------------------------------------------------------------------------

    def model(self, name: str) -> Model:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(name)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self.models)

    def forecast_scoring(self) -> tuple[ForecastScoring, ...]:
        """The configured raw/calibrated scoring, or every model's by default."""
        if self.evaluation.forecasts is not None:
            return self.evaluation.forecasts
        return tuple(ForecastScoring(model=name) for name in self.model_names)

    def final_formula(self) -> str:
        """The formula the operational model is fitted with."""
        if self.evaluation.final_formula is not None:
            return self.evaluation.final_formula
        return next(name for name, f in self.evaluation.formulas if f.enabled)

    def climatology_forecast_window(self) -> int:
        if self.climatology.forecast_window is not None:
            return self.climatology.forecast_window
        return self.features.probability_days

    def window_days(self, predictor: RainPredictor, definition: Definition) -> int:
        """A rain predictor's window in days; a token is read from the onset definition (`Definition.windows()`)."""
        window = predictor.window
        if isinstance(window, int):
            return window
        key = {"trigger": "trigger", "window": "trigger", "dry_spell": "dry_window", "sum_window": "dry_window",
               "follow": "follow", "min_dry": "dry_run"}[window.lower()]
        days = definition.windows()[key]
        if days is None:
            raise RecipeError([f"rain window {window!r}: the onset definition has no {key.replace('_', ' ')}"])
        return int(days)

    def rain_horizon_days(self, model: Model) -> int:
        """The last rainfall lead day a model's rain predictors read."""
        horizon = model.rain_horizon
        return horizon.max_day if horizon.max_day is not None else self.features.probability_days + 10

    # -- derived names -----------------------------------------------------------------------------------------

    def rain_feature_base(self, model: str, predictor: RainPredictor, definition: Definition) -> str:
        if predictor.agg == "diff":
            return f"diff_{model}"
        return f"{predictor.agg}_{model}_{self.window_days(predictor, definition)}day"

    def cutoff_variant(self) -> str:
        """The forecast probability variant the cutoff mode scores by default."""
        return {"fixed_cutoff": "fixed_cutoff", "ref_onset": "ref_onset"}.get(self.evaluation.cutoff_mode, "base")

    def probability_variants(self) -> list[str]:
        """The variants anything reads (scoring, MME sources, formulas), besides the base series the features always bin."""
        evaluation = self.evaluation
        variants = {scoring.variant or self.cutoff_variant() for scoring in self.forecast_scoring()}
        variants |= {s.cal_variant for s in evaluation.mme.sources if s.source == "forecast"}
        texts = [f.text for _, f in evaluation.formulas if f.enabled]
        for variant, suffix in (("fixed_cutoff", "_fixed_cutoff_"), ("ref_onset", "_ref_")):
            if any(f"_p_onset{suffix}" in text for text in texts):
                variants.add(variant)
        return [v for v in ("fixed_cutoff", "ref_onset") if v in variants]

    def feature_bases(self, definition: Definition) -> set[str]:
        """Every feature a formula may use: `<base>_qx` (one per week) or `<base>_week<n>`."""
        bases = {CLIMATOLOGY_FEATURE, f"{CLIMATOLOGY_FEATURE}_unc"}
        bases |= {f"{CLIMATOLOGY_FEATURE}_{window_tag(w)}" for w in self.climatology.windows}
        for model in self.models:
            bases |= {probability_feature_base(model.name, v) for v in VARIANTS}
            bases |= {self.rain_feature_base(model.name, p, definition) for p in model.rain_predictors}
        expanded = {f"{base}_qx" for base in bases}
        expanded |= {f"{base}_week{w}" for base in bases for w in range(1, self.features.count + 1)}
        expanded |= {f"{CLIMATOLOGY_FEATURE}_unc_earlier", "onset_threshold", "lat", "lon"}
        return expanded

    def climatology_baseline_names(self) -> list[str]:
        baselines = self.evaluation.baselines
        names = ["unc_clim_raw"] if baselines.unconditional else []
        if baselines.conditional:
            names.append("clim_raw")
        return names + [f"clim_raw_{window_tag(w)}" for w in baselines.windows]

    def outcome_labels(self) -> list[str]:
        return self.features.bins.labels("week", earlier=False)

    def cutoff_tag(self) -> str:
        return {"fixed_cutoff": "_fixed_cutoff", "no_ref_filter": "_no_ref_filter"}.get(self.evaluation.cutoff_mode, "")

    def year_tag(self) -> str:
        years = self.evaluation.years.holdout
        if not years:
            return ""
        return f"_{years[0]}" if len(years) == 1 else f"_{min(years)}_{max(years)}"

    def output_tag(self) -> str:
        """The `{cutoff}{years}` suffix every evaluation artifact carries."""
        return f"{self.cutoff_tag()}{self.year_tag()}"

    # -- validation --------------------------------------------------------------------------------------------

    def check(self, definition: Definition) -> None:
        """The rules that need the onset definition: rain-window tokens it can resolve, formulas that name only
        features this recipe builds; raises RecipeError."""
        problems: list[str] = []
        for model in self.models:
            for index, predictor in enumerate(model.rain_predictors):
                try:
                    self.window_days(predictor, definition)
                except RecipeError as error:
                    problems.extend(f"models[{model.name}].rain_predictors[{index}]: {p}" for p in error.problems)
        if not problems:
            available = self.feature_bases(definition)
            for name, formula in self.evaluation.formulas:
                if not formula.enabled:
                    continue
                for token in formula_terms(formula.text):
                    if token not in available:
                        import difflib

                        close = difflib.get_close_matches(token, sorted(available), n=2)
                        hint = f" (did you mean {' or '.join(close)}?)" if close else ""
                        problems.append(f"evaluation.formulas.{name}: {token!r} is not a feature this recipe "
                                        f"builds{hint}")
        if problems:
            raise RecipeError(problems)

    def validate(self) -> None:
        """Every rule that ties sections together (blending's `BlendConfig.validate`); raises RecipeError."""
        problems: list[str] = []
        names = self.model_names
        if not names:
            problems.append("models: at least one forecast model is required")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            problems.append(f"models: names must be unique (repeated: {', '.join(duplicates)})")
        for name in names:
            if not _MODEL_NAME.fullmatch(name):
                problems.append(f"models: name {name!r} must be letters, digits and single underscores")

        features, bins = self.features, self.features.bins
        try:
            bins.labels("week", earlier=False)
        except ValueError as error:
            problems.append(f"features.bins: {error}")
        else:
            if len({b - a for a, b in bins.inner}) != 1:
                problems.append("features.bins: the bins must have equal days")
            elif bins.last_day > features.probability_days:
                problems.append(f"features.bins: {bins.n_inner} bins of {features.days} days need {bins.last_day} "
                                f"probability days, but features.probability_days is {features.probability_days}")
        _check_rain_transform(features.rain_transform, "features.rain_transform", problems)

        for index, model in enumerate(self.models):
            self._check_model(model, f"models[{index}] ({model.name})", problems)

        for field_name, value in (("onset.fixed_cutoff", self.onset.fixed_cutoff),
                                  ("climatology.issue_end", self.climatology.issue_end)):
            if not _MONTH_DAY.match(str(value)):
                problems.append(f"{field_name}: expected a month-day like '05-01', got {value!r}")

        climatology = self.climatology
        if climatology.horizons and climatology.forecast_window is not None:
            problems.append("climatology: give 'forecast_window' or 'horizons', not both")
        if not climatology.horizons and self.climatology_forecast_window() < features.probability_days:
            problems.append(f"climatology.forecast_window ({self.climatology_forecast_window()}) is below "
                            f"features.probability_days ({features.probability_days})")
        for window in climatology.windows:
            if window[1] < window[0]:
                problems.append(f"climatology.windows: {window} ends before it starts")

        self._check_evaluation(problems)
        if problems:
            raise RecipeError(problems)

    def _check_model(self, model: Model, where: str, problems: list[str]) -> None:
        features = self.features
        low, high = model.lead_days
        if low < 0 or high < low:
            problems.append(f"{where}.lead_days: need 0 <= min <= max")
        if high < features.probability_days:
            problems.append(f"{where}.lead_days.max is {high}, below features.probability_days "
                            f"({features.probability_days})")
        horizon = model.rain_horizon
        if horizon.max_day is not None:
            if horizon.max_day <= 0:
                problems.append(f"{where}.rain_horizon.max_day must be positive")
            if horizon.policy == "legacy":
                problems.append(f"{where}.rain_horizon: 'legacy' cannot have a max_day")
            if low != 1:
                problems.append(f"{where}: a rain_horizon.max_day needs lead_days.min of 1")
            if high < horizon.max_day:
                problems.append(f"{where}.lead_days.max ({high}) is below rain_horizon.max_day ({horizon.max_day})")
        elif horizon.policy not in (None, "legacy"):
            problems.append(f"{where}.rain_horizon: policy {horizon.policy!r} needs a max_day")
        elif model.rain_predictors and high < features.probability_days + 10:
            problems.append(f"{where}: rain predictors without a rain_horizon.max_day read "
                            f"{features.probability_days + 10} lead days (probability_days + 10), "
                            f"but lead_days.max is {high}")
        if len([p for p in model.rain_predictors if p.agg == "diff"]) > 1:
            problems.append(f"{where}.rain_predictors: one 'diff' predictor per model (their columns share one name)")
        for index, predictor in enumerate(model.rain_predictors):
            window = predictor.window
            if isinstance(window, str):
                if window.lower() not in WINDOW_TOKENS:
                    problems.append(f"{where}.rain_predictors[{index}].window: {window!r} is not a number or one of "
                                    f"{', '.join(WINDOW_TOKENS)}")
            elif window < 1:
                problems.append(f"{where}.rain_predictors[{index}].window must be >= 1")
        if model.rain_transform is not None:
            _check_rain_transform(model.rain_transform, f"{where}.rain_transform", problems)

    def _check_evaluation(self, problems: list[str]) -> None:
        evaluation, features = self.evaluation, self.features
        years = evaluation.years
        if not years.training:
            problems.append("evaluation.years.training is empty")
        if not years.holdout:
            problems.append("evaluation.years: no cv_holdout or true_holdout years to score")
        if evaluation.cutoff_mode not in CUTOFF_MODES:
            problems.append(f"evaluation.cutoff_mode {evaluation.cutoff_mode!r} is not one of {CUTOFF_MODES}")
        for method in evaluation.cv_methods:
            if method not in CV_METHODS:
                problems.append(f"evaluation.cv_methods: {method!r} is not one of {CV_METHODS}")
        if evaluation.cutoff_mode == "ref_onset" and self.onset.reference is None:
            problems.append("evaluation.cutoff_mode 'ref_onset' needs onset.reference_onset")
        forest = evaluation.random_forest
        if forest is not None:
            if forest.max_day != features.bins.last_day:
                problems.append(f"evaluation.random_forest.max_day must span the week bins ({features.count} x "
                                f"{features.days} days)")
            if forest.trees < 1 or forest.min_samples_leaf < 1 or forest.years_per_fold < 1:
                problems.append("evaluation.random_forest: trees, min_samples_leaf and years_per_fold must be at least 1")
            if not 0.0 < forest.max_samples <= 1.0:
                problems.append("evaluation.random_forest.max_samples must be in (0, 1]")
            if forest.days_per_week != features.days:
                problems.append("evaluation.random_forest.days_per_week must be the bins' days")
        enabled = {name: f for name, f in evaluation.formulas if f.enabled}
        if not enabled:
            problems.append("evaluation.formulas: at least one enabled formula is required")
        if evaluation.final_formula is not None and evaluation.final_formula not in enabled:
            problems.append(f"evaluation.final_formula: {evaluation.final_formula!r} is not an enabled formula")
        elif evaluation.final_formula is None and len(enabled) > 1:
            problems.append("evaluation.final_formula: name the formula operations uses "
                            f"(one of {', '.join(sorted(enabled))})")
        for v_index, variant in enumerate(evaluation.window_variants):
            if variant.base not in enabled:
                problems.append(f"evaluation.window_variants[{v_index}].base: {variant.base!r} is not an enabled formula")
            for window in variant.windows:
                if tuple(window) not in self.climatology.windows:
                    problems.append(f"evaluation.window_variants[{v_index}]: window {window} is not in climatology.windows")
        for window in evaluation.baselines.windows:
            if tuple(window) not in self.climatology.windows:
                problems.append(f"evaluation.climatology_baselines.windows: {window} is not in climatology.windows")
        evaluated = set()
        for index, scoring in enumerate(self.forecast_scoring()):
            if scoring.model not in self.model_names:
                problems.append(f"evaluation.forecasts[{index}].model: {scoring.model!r} is not a model")
            evaluated.add((scoring.model, scoring.variant or evaluation.cutoff_mode, scoring.calibrated))
        mme = evaluation.mme
        if mme.enabled:
            if len(mme.sources) < 2:
                problems.append("evaluation.mme: an ensemble needs at least two sources")
            baselines = set(self.climatology_baseline_names())
            if mme.weights is not None:
                names = [source.name for source in mme.sources]
                if mme.optimize == "always":
                    problems.append("evaluation.mme.weights: given weights are used as they are, so optimize cannot "
                                    "be 'always'")
                if isinstance(mme.weights, dict):
                    if sorted(mme.weights) != sorted(names):
                        problems.append(f"evaluation.mme.weights: expected one weight for each source "
                                        f"({', '.join(names)}), got {', '.join(map(str, mme.weights)) or 'none'}")
                    elif any(w is None or w < 0 for w in mme.weights.values()) or abs(sum(mme.weights.values()) - 1) > 1e-6:
                        problems.append("evaluation.mme.weights: weights must be non-negative and sum to 1")
            for index, source in enumerate(mme.sources):
                if source.source == "clim" and source.name not in baselines:
                    problems.append(f"evaluation.mme.sources[{index}]: {source.name!r} is not a scored climatology "
                                    f"baseline ({', '.join(sorted(baselines))})")
                if source.source == "forecast" and (source.name, source.cal_variant, True) not in evaluated:
                    problems.append(f"evaluation.mme.sources[{index}]: needs evaluation.forecasts to calibrate "
                                    f"{source.name!r} ({source.cal_variant})")
        if not evaluation.baselines.unconditional:
            problems.append("evaluation.climatology_baselines.unconditional: unc_clim_raw is the baseline every skill "
                            "score is measured against and cannot be turned off")


# ---------------------------------------------------------------------------------------------------------------
# Derived names (module level: the features need them without a recipe at hand)
# ---------------------------------------------------------------------------------------------------------------


def window_tag(window: tuple[int, int]) -> str:
    return f"{window[0]}_{window[1]}"


def probability_feature_base(model: str, variant: str) -> str:
    return f"{model}_p_onset{VARIANT_SUFFIX[variant]}"


_FORMULA_FUNCTIONS = re.compile(r"^(?:I|C|Q|center|standardize|scale|np|numpy)(?:\..+)?$")


def formula_terms(text: str) -> list[str]:
    """The feature names a formula's right-hand side refers to."""
    rhs = text.split("~", 1)[1] if "~" in text else text
    tokens = [t.strip() for t in re.split(r"[+\-*:(),]", rhs)]
    return [t for t in tokens if t and not re.fullmatch(r"\d+(\.\d+)?", t) and not _FORMULA_FUNCTIONS.match(t)]


def _check_rain_transform(value, path: str, problems: list[str]) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return
    text = str(value).strip().lower()
    if text in _RAIN_TRANSFORMS or re.fullmatch(r"power:\s*-?\d+(\.\d+)?", text):
        return
    problems.append(f"{path}: {value!r} is not identity, sqrt, fourth_root, log1p or power:<p>")


# ---------------------------------------------------------------------------------------------------------------
# The Recipe of a run config: the `blend:` section
# ---------------------------------------------------------------------------------------------------------------


def _strict(value, allowed, where: str, problems: list[str]) -> dict:
    """A mapping with only the keys allowed; others are problems with a 'did you mean'."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{where}: expected a mapping of {', '.join(allowed)}, got {value!r}")
        return {}
    for key in value:
        if key not in allowed:
            problems.append(f"{where}.{key}: unknown key.{suggest(key, allowed)} Keys: {', '.join(allowed)}")
    return {k: v for k, v in value.items() if k in allowed}


def _list(value, where: str, problems: list[str]) -> list:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        problems.append(f"{where}: expected a list, got {value!r}")
        return []
    return list(value)


def _check_type(value, kind: str, where: str, problems: list[str]):
    """value if it is of the kind ('int', 'number', 'bool', 'str'), else a problem and None."""
    ok = {"int": isinstance(value, int) and not isinstance(value, bool),
          "number": isinstance(value, (int, float)) and not isinstance(value, bool),
          "bool": isinstance(value, bool), "str": isinstance(value, str)}[kind]
    if not ok:
        what = {"int": "a whole number", "number": "a number", "bool": "true or false", "str": "text"}[kind]
        problems.append(f"{where}: expected {what}, got {value!r}")
        return None
    return float(value) if kind == "number" else value


def _choice(value, choices, where: str, problems: list[str]):
    if value not in choices:
        problems.append(f"{where}: {value!r} is not one of {', '.join(map(repr, choices))}.{suggest(value, choices)}")
        return None
    return value


def years_of(value, where: str, problems: list[str]) -> tuple[int, ...]:
    """Years from a list, a year or text like '2000:2024,2026' (sorted, each once)."""
    if value is None:
        return ()
    try:
        items = value if isinstance(value, (list, tuple)) else [value]
        years = [y for item in items for y in parse_years(item, where)]
    except DatasetError as error:
        problems.append(str(error))
        return ()
    return tuple(sorted(dict.fromkeys(years)))


def year_window(value, where: str, problems: list[str]) -> tuple[int, int] | None:
    years = years_of(value, where, problems)
    if not years:
        problems.append(f"{where}: expected a first and last year")
        return None
    return years[0], years[-1]


def year_windows(value, where: str, problems: list[str]) -> tuple[tuple[int, int], ...]:
    windows = [year_window(item, f"{where}[{i}]", problems) for i, item in enumerate(_list(value, where, problems))]
    return tuple(w for w in windows if w is not None)


def _rain_model(name: str, value, problems: list[str]) -> Model:
    where = f"blend.models.{name}"
    node = _strict(value, ("lead_days", "max_member", "rain_horizon", "rain_predictors", "rain_transform"), where,
                   problems)
    lead = node.get("lead_days", (1, 45))
    if isinstance(lead, dict):
        lead = _strict(lead, ("min", "max"), f"{where}.lead_days", problems)
        lead = (lead.get("min", 1), lead.get("max", 45))
    if (not isinstance(lead, (list, tuple)) or len(lead) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in lead)):
        problems.append(f"{where}.lead_days: expected [min, max] (whole numbers) or {{min, max}}, got {lead!r}")
        lead = (1, 45)
    horizon = _strict(node.get("rain_horizon"), ("max_day", "policy"), f"{where}.rain_horizon", problems)
    if horizon.get("max_day") is not None:
        _check_type(horizon["max_day"], "int", f"{where}.rain_horizon.max_day", problems)
    if horizon.get("policy") is not None:
        _choice(horizon["policy"], ("legacy", "exact", "truncate"), f"{where}.rain_horizon.policy", problems)
    predictors = []
    for index, item in enumerate(_list(node.get("rain_predictors"), f"{where}.rain_predictors", problems)):
        at = f"{where}.rain_predictors[{index}]"
        item = _strict(item, ("agg", "window"), at, problems)
        if "agg" not in item:
            problems.append(f"{at}.agg: required (diff, max or min)")
            continue
        window = item.get("window", "trigger")
        if not isinstance(window, str) and not (isinstance(window, int) and not isinstance(window, bool)):
            problems.append(f"{at}.window: expected days or one of {', '.join(WINDOW_TOKENS)}, got {window!r}")
            continue
        agg = _choice(item["agg"], ("diff", "max", "min"), f"{at}.agg", problems)
        if agg is not None:
            predictors.append(RainPredictor(agg, window))
    max_member = node.get("max_member")
    if max_member is not None:
        _check_type(max_member, "int", f"{where}.max_member", problems)
    return Model(name=name, lead_days=(lead[0], lead[1]), max_member=max_member,
                 rain_horizon=RainHorizon(horizon.get("max_day"), horizon.get("policy")),
                 rain_predictors=tuple(predictors), rain_transform=node.get("rain_transform"))


def _reference(value, problems: list[str]) -> ReferenceSpec | None:
    if value is None:
        return None
    where = "blend.reference_onset"
    node = _strict(value, ("month_day", "file", "date_col", "year_col", "day_col", "base_date", "unit_col"), where,
                   problems)
    for key in ("month_day", "base_date"):
        if node.get(key) is not None:
            node[key] = str(node[key]).strip()
            if not _MONTH_DAY.match(node[key]):
                problems.append(f"{where}.{key}: expected a month-day like '05-01', got {node[key]!r}")
    try:
        return ReferenceSpec(**node)
    except RecipeError as error:
        problems.extend(f"{where}: {p.split(': ', 1)[-1]}" for p in error.problems)
        return None


def _climatology(value, problems: list[str]) -> ClimatologySpec:
    where = "blend.climatology"
    node = _strict(value, ("issue_end", "forecast_window", "horizons", "cv_by_year", "min_onset_years", "train_years",
                           "test_years", "windows", "workers"), where, problems)
    out: dict = {}
    if "issue_end" in node:
        out["issue_end"] = str(node["issue_end"]).strip()
    for key, kind in (("forecast_window", "int"), ("min_onset_years", "int"), ("workers", "int"),
                      ("cv_by_year", "bool")):
        if node.get(key) is not None:
            out[key] = _check_type(node[key], kind, f"{where}.{key}", problems)
    horizons = []
    for index, item in enumerate(_list(node.get("horizons"), f"{where}.horizons", problems)):
        at = f"{where}.horizons[{index}]"
        item = _strict(item, ("start", "end", "forecast_window"), at, problems)
        if "start" not in item or "end" not in item:
            problems.append(f"{at}: start and end month-days are required")
            continue
        for key in ("start", "end"):
            if not _MONTH_DAY.match(str(item[key]).strip()):
                problems.append(f"{at}.{key}: expected a month-day like '05-01', got {item[key]!r}")
        horizons.append((str(item["start"]).strip(), str(item["end"]).strip(), item.get("forecast_window", 0)))
    out["horizons"] = tuple(horizons)
    for key in ("train_years", "test_years"):
        if node.get(key) is not None:
            out[key] = year_window(node[key], f"{where}.{key}", problems)
    out["windows"] = year_windows(node.get("windows"), f"{where}.windows", problems)
    return ClimatologySpec(**out)


def _features(cfg: dict, problems: list[str]) -> Features:
    bins = _strict(cfg.get("blend_bins"), ("count", "days"), "blend.bins", problems)
    count, days = bins.get("count", 4), bins.get("days", 7)
    out: dict = {}
    if all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in (count, days)):
        out["bins"] = LeadBins.uniform(count, days)
    else:
        problems.append("blend.bins: count and days must be positive whole numbers")
    for key in ("probability_days", "trim_after_true_onset", "rain_transform"):
        if cfg.get(f"blend_{key}") is not None:
            out[key] = cfg[f"blend_{key}"]
    return Features(**out)


def _forest(value, days: int, problems: list[str]) -> ForestParams | None:
    if value is None:
        return None
    where = "blend.random_forest"
    node = _strict(value, ("name", "trees", "min_samples_leaf", "max_depth", "max_samples", "max_features",
                           "random_state", "max_day", "missing_onset", "years_per_fold"), where, problems)
    if node.get("max_features", "sqrt") not in ("sqrt", "all"):
        problems.append(f"{where}.max_features: {node['max_features']!r} is not 'sqrt' or 'all'")
    if node.get("missing_onset", "after_week4") not in ("after_week4", "drop"):
        problems.append(f"{where}.missing_onset: {node['missing_onset']!r} is not 'after_week4' or 'drop'")
    kinds = {"name": "str", "trees": "int", "min_samples_leaf": "int", "max_depth": "int", "max_samples": "number",
             "random_state": "int", "max_day": "int", "years_per_fold": "int"}
    for key, kind in kinds.items():
        if node.get(key) is not None:
            _check_type(node[key], kind, f"{where}.{key}", problems)
    max_features = None if node.get("max_features", "sqrt") == "all" else node.get("max_features", "sqrt")
    try:
        return ForestParams(**{**{k: v for k, v in node.items() if k != "max_features"}, "max_features": max_features,
                               "days_per_week": days})
    except TypeError as error:
        problems.append(f"{where}: {error}")
        return None


def _evaluation(cfg: dict, bins_days: int, problems: list[str]) -> Evaluation:
    years = _strict(cfg.get("blend_years"), ("training", "cv_holdout", "true_holdout"), "blend.years", problems)
    if cfg.get("blend_years") is None:
        problems.append("blend.years: required (training and cv_holdout and/or true_holdout years)")
    out: dict = {"years": Years(*(years_of(years.get(k), f"blend.years.{k}", problems)
                                  for k in ("training", "cv_holdout", "true_holdout")))}
    if cfg.get("blend_cutoff_mode") is not None:
        out["cutoff_mode"] = cfg["blend_cutoff_mode"]
    if cfg.get("blend_cv_methods") is not None:
        out["cv_methods"] = tuple(_choice(m, CV_METHODS, "blend.cv_methods", problems) for m in cfg["blend_cv_methods"])
    for key in ("cluster_var", "final_formula"):
        if cfg.get(f"blend_{key}") is not None:
            out[key] = cfg[f"blend_{key}"]
    formulas = cfg.get("blend_formulas") or {}
    if not isinstance(formulas, dict):
        problems.append(f"blend.formulas: expected a mapping of name to formula text, got {formulas!r}")
        formulas = {}
    parsed = []
    for name, item in formulas.items():
        at = f"blend.formulas.{name}"
        if isinstance(item, str):
            parsed.append((str(name), Formula(text=item)))
            continue
        item = _strict(item, ("text", "enabled"), at, problems)
        if not isinstance(item.get("text"), str):
            problems.append(f"{at}.text: required (the formula text)")
            continue
        parsed.append((str(name), Formula(text=item["text"], enabled=item.get("enabled", True))))
    out["formulas"] = tuple(parsed)
    variants = []
    for index, item in enumerate(_list(cfg.get("blend_window_variants"), "blend.window_variants", problems)):
        at = f"blend.window_variants[{index}]"
        item = _strict(item, ("base", "windows", "replace", "only_if_cutoff_mode"), at, problems)
        if not isinstance(item.get("base"), str):
            problems.append(f"{at}.base: required (the formula to vary)")
            continue
        variants.append(WindowVariant(base=item["base"], windows=year_windows(item.get("windows"),
                        f"{at}.windows", problems), replace=item.get("replace", "prob_clim"),
                        only_if_cutoff_mode=item.get("only_if_cutoff_mode")))
    out["window_variants"] = tuple(variants)
    base = _strict(cfg.get("blend_baselines"), ("unconditional", "conditional", "windows"), "blend.baselines", problems)
    out["baselines"] = Baselines(base.get("unconditional", True), base.get("conditional", True),
                                 year_windows(base.get("windows"), "blend.baselines.windows", problems))
    if cfg.get("blend_forecasts") is not None:
        scoring = []
        for index, item in enumerate(_list(cfg["blend_forecasts"], "blend.forecasts", problems)):
            at = f"blend.forecasts[{index}]"
            item = _strict(item, ("model", "variant", "raw", "calibrated", "platt_weights"), at, problems)
            if "model" not in item:
                problems.append(f"{at}.model: required")
                continue
            if item.get("variant") is not None:
                _choice(item["variant"], VARIANTS, f"{at}.variant", problems)
            scoring.append(ForecastScoring(**item))
        out["forecasts"] = tuple(scoring)
    mme = _strict(cfg.get("blend_mme"), ("enabled", "variants", "sources", "optimize", "weights"), "blend.mme",
                  problems)
    sources = []
    for index, item in enumerate(_list(mme.get("sources"), "blend.mme.sources", problems)):
        at = f"blend.mme.sources[{index}]"
        item = _strict(item, ("name", "source", "cal_variant"), at, problems)
        if "name" not in item:
            problems.append(f"{at}.name: required")
            continue
        if "source" in item:
            _choice(item["source"], ("clim", "forecast"), f"{at}.source", problems)
        if "cal_variant" in item:
            _choice(item["cal_variant"], VARIANTS, f"{at}.cal_variant", problems)
        sources.append(MmeSource(**item))
    if "optimize" in mme:
        _choice(mme["optimize"], ("auto", "always", "never"), "blend.mme.optimize", problems)
    weights = mme.get("weights")
    if isinstance(weights, dict):
        weights = {str(k): _check_type(v, "number", f"blend.mme.weights.{k}", problems) for k, v in weights.items()}
    elif weights is not None and not isinstance(weights, (str, os.PathLike)):
        problems.append(f"blend.mme.weights: expected a mapping of source name to weight or the path of a weights "
                        f"file, got {weights!r}")
        weights = None
    elif weights is not None:
        weights = str(weights)
    mme = {**mme, "weights": weights} if weights is not None else {k: v for k, v in mme.items() if k != "weights"}
    out["mme"] = Mme(**{**{k: v for k, v in mme.items() if k not in ("sources", "variants")},
                        "sources": tuple(sources),
                        **({"variants": tuple(mme["variants"])} if "variants" in mme else {})})
    external = []
    for index, item in enumerate(_list(cfg.get("blend_external_predictions"), "blend.external_predictions", problems)):
        at = f"blend.external_predictions[{index}]"
        item = _strict(item, ("name", "file", "method"), at, problems)
        if "name" not in item or "file" not in item:
            problems.append(f"{at}: name and file are required")
            continue
        external.append(ExternalPrediction(**item))
    out["external_predictions"] = tuple(external)
    out["random_forest"] = _forest(cfg.get("blend_random_forest"), bins_days, problems)
    return Evaluation(**out)


def recipe_from_config(cfg: dict, setup=None) -> Recipe:
    """The Recipe a resolved run config asks for: its `blend:` section (the flat `blend_*` keys), with the model names of
    `model_list` and the run's `project_name`. Every problem found is reported together as one RecipeError, nested keys
    with a 'did you mean'; unset keys take the defaults of the dataclasses above (blending's own).

    With `setup` (the run's OnsetSetup) the rules that need the onset definition are checked too (`Recipe.check`).
    """
    problems: list[str] = []
    names = tuple(cfg.get("model_list") or ())
    settings = cfg.get("blend_models")
    if settings is None:
        settings = {}
    elif not isinstance(settings, dict):
        problems.append(f"blend.models: expected a mapping of model name to settings, got {settings!r}")
        settings = {}
    for key in settings:
        if key not in names:
            problems.append(f"blend.models.{key}: not one of the models.{suggest(key, names)} Models: "
                            f"{', '.join(names)}")
    models = tuple(_rain_model(name, settings.get(name), problems) for name in names)
    features = _features(cfg, problems)
    evaluation = _evaluation(cfg, features.bins.inner[0][1] - features.bins.inner[0][0] + 1, problems)
    extras: dict = {}
    if cfg.get("blend_fixed_cutoff") is not None:
        extras["fixed_cutoff"] = str(cfg["blend_fixed_cutoff"]).strip()
    outputs = _strict(cfg.get("blend_outputs"), ("dir", "intermediates", "pickles"), "blend.outputs", problems)
    intermediates = tuple(_list(outputs.get("intermediates"), "blend.outputs.intermediates", problems))
    for item in intermediates:
        _choice(item, ("observations", "forecasts", "climatology", "features"), "blend.outputs.intermediates",
                problems)
    reference = _reference(cfg.get("blend_reference_onset"), problems)
    climatology = _climatology(cfg.get("blend_climatology"), problems)
    if problems:
        raise RecipeError(problems)
    recipe = Recipe(name=str(cfg.get("project_name") or "blend"), models=models,
                    onset=OnsetExtras(reference=reference, **extras), climatology=climatology, features=features,
                    evaluation=evaluation,
                    outputs=Outputs(outputs.get("dir"), intermediates, bool(outputs.get("pickles", False))))
    if setup is not None:
        recipe.check(setup.definition)
    return recipe
