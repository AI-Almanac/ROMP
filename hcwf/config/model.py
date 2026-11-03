"""RunConfig: a resolved run as typed, frozen specs (design: docs/review/runconfig_design.md 2.2, 2.3, 2.8, 2.10).

    rc = RunConfig.from_flat(resolve_flat("run.yaml"))
    for case in rc.cases():                     # every model with every verification window
        case.model.dir, case.window.span, case.window.tolerance
    rc.reference_cases()                        # the reference, deterministic rule (models[0]'s accumulation)
    rc.reference_cases(like=model)              # the reference, ensemble rule (that model's, and reference.dims)

``RunConfig`` keeps the resolved flat record (``rc.values``, read-only): run.yaml, the fingerprint and
RunResult.config are computed from it. The specs are projections of that record, built once in ``from_flat``. They
declare attribute names and types only: every key, default, help text and choice is declared once, in
hcwf/lib/config_schema.py, and ``OutputSpec`` and ``MetricsSpec`` are generated from its ``output`` and ``metrics``
sections. tests/test_runconfig_coverage.py checks that every schema key is read by a projection.

Imports of the heavy modules (io.input, onset_config, blend.recipe, control) are inside the functions, so that
``import hcwf.config.model`` stays light.
"""
from __future__ import annotations

from dataclasses import dataclass, make_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

from hcwf.lib import config_schema as schema

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.blend.recipe import Recipe
    from hcwf.lib.onset_config import OnsetSetup
    from hcwf.lib.run_folder import RunPaths
    from hcwf.spatial.units import UnitsSpec
    from hcwf.utils.mask import MaskSpec

Dims = tuple[tuple[str, str], ...]
"""Canonical dimension name -> name in the files, as sorted-as-given pairs (a dict is not hashable)."""
CLIMATOLOGY_REFERENCES = ("climatology", "conditional_climatology")


@dataclass(frozen=True, slots=True)
class Window:
    """One verification window and its tolerance (verification_window_list[i] and tolerance_days_list[i])."""

    start: int
    end: int
    tolerance: int

    @property
    def span(self) -> tuple[int, int]:
        """(start, end), today's `verification_window`."""
        return (self.start, self.end)

    @property
    def label(self) -> str:
        """The window as text in file names, e.g. "1-15"."""
        return f"{self.start}-{self.end}"

    def bins(self, day_bins) -> tuple[tuple[int, int], ...]:
        """The day bins that lie completely inside the window."""
        from hcwf.lib.control import filter_bins_in_window

        return filter_bins_in_window(day_bins, self.span)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One forecast model: the `*_list` keys of the flat record at its position."""

    name: str
    dir: Path | str
    var: str
    unit_cvt: float | None
    file_pattern: str
    accumulation: str | None
    accumulation_detected: bool | None
    daily_boundary: str | None
    dims: Dims | None
    paths: tuple | None
    units: UnitsSpec | None

    def dims_map(self) -> dict | None:
        """The dimension names as a dict, as the readers take them (None: inferred)."""
        return None if self.dims is None else dict(self.dims)


@dataclass(frozen=True, slots=True)
class ObsSpec:
    """The observations: the observations section and the obs_* keys."""

    name: str
    dir: Path | str
    file_pattern: tuple[str, ...]
    var: str
    unit_cvt: float | None
    paths: tuple | None
    dims: Dims | None
    day_boundary: str | None
    units: UnitsSpec | None

    def dims_map(self) -> dict | None:
        """The dimension names as a dict (None: inferred)."""
        return None if self.dims is None else dict(self.dims)


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    """The reference of the skill scores: the reference section and ref_model_unit_cvt."""

    model: str
    kind: Literal["climatology", "conditional_climatology", "model"]
    climatology_cv: str
    dir: Path | str | None
    file_pattern: str | None
    var: str | None
    unit_cvt: float | None
    dims: Dims | None
    units: UnitsSpec | None

    def as_model(self, like: ModelSpec, *, dims: Dims | None) -> ModelSpec:
        """The reference as a source to read: its own name, data and unit conversion, and `like`'s accumulation and daily
        boundary (the two rules of today's flows, written once: the deterministic flow passes models[0] and no dims, the
        ensemble flow the current model and `self.dims`; `RunConfig.reference_cases` does both)."""
        return ModelSpec(name=self.model, dir=self.dir, var=self.var, unit_cvt=self.unit_cvt,
                         file_pattern=self.file_pattern, accumulation=like.accumulation,
                         accumulation_detected=like.accumulation_detected, daily_boundary=like.daily_boundary,
                         dims=dims, paths=None, units=self.units)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """The evaluation section and lead_window, with the years resolved."""

    start_date: tuple[int, int, int]
    end_date: tuple[int, int, int]
    years: tuple[int, ...]
    climatology_years: tuple[int, ...]
    init_days: tuple[int, ...]
    date_filter_year: int | None
    windows: tuple[Window, ...]
    max_forecast_day: int
    day_bins: tuple[tuple[int, int], ...]
    probabilistic: bool
    members: tuple[int, ...] | None
    lead_window: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class OnsetOptions:
    """The onset-section keys that OnsetSetup does not hold (None: the option's own default)."""

    rain_resolution: str | None
    missing: str | None
    negatives: str | None
    gaps: str | None
    horizon: str | None
    threshold_scale: dict | None
    threshold_table: dict | None
    threshold_max_missing: float | None
    threshold_partial_years: bool | None
    percentage_threshold: float
    explain_cells: tuple | None


@dataclass(frozen=True, slots=True)
class AreaSpec:
    """Where the evaluation happens: the region, its masks and the boundary shapes of the maps."""

    region: str | None
    mask: MaskSpec | None
    nc_mask: Path | str | None
    shpfile_dir: Path | str | None
    polygon: bool


@dataclass(frozen=True, slots=True)
class ParallelSpec:
    """Whether and how many worker processes compute the years."""

    enabled: bool
    workers: int | None


def _generated(name: str, section: str, doc: str):
    """A frozen spec with one attribute (Field.nested_name) for each key of a schema section."""
    fields = [(f.nested_name, "Any", None) for f in schema.FIELDS if f.section == section]
    cls = make_dataclass(name, [(n, t) for n, t, _ in fields], frozen=True, slots=True, module=__name__)
    cls.__doc__ = doc
    return cls


OutputSpec = _generated("OutputSpec", "output", "The output section: folders and which files and figures to write "
                                                "(generated from the schema's output keys).")
MetricsSpec = _generated("MetricsSpec", "metrics", "The metrics section: which metrics to compute (generated from the "
                                                   "schema's metrics keys).")


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """What the figures read of a run (built by RunConfig.figure_spec)."""

    dir_fig: Path | str
    region: str | None
    shpfile_dir: Path | str | None
    polygon: bool
    save_fig: bool
    show_plot: bool
    show_panel: bool
    date_filter_year: int | None
    short_names: bool
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Case:
    """One (source, window) pair: what today's `convention.Case` holds, as typed values.

    For a reference case `model` is the reference as a source (`ReferenceSpec.as_model`) and for_model is, in the ensemble
    flow, the forecast model it is scored against (that flow names the reference's case after that model).
    """

    model: ModelSpec
    window: Window
    reference: bool = False
    for_model: str | None = None

    @property
    def name(self) -> str:
        """Today's case_name, e.g. "AIFS_1-15"."""
        from hcwf.utils.printing import combi_to_str

        return combi_to_str((self.for_model or self.model.name, self.window.span)).replace(" ", "_")

    @property
    def key(self) -> tuple[str, tuple[int, int]]:
        """(model name, window span), the combination the flows loop over."""
        return (self.model.name, self.window.span)


@dataclass(frozen=True, slots=True)
class Selection:
    """The per-call choices that are not run config (what blend and export put into the kwargs): None is not set."""

    keep_unit_ids: Any = None
    member_ids: Any = None
    max_member: Any = None
    issue_dates: Any = None
    last_lead: Any = None
    model_files_years: Any = None
    obs_files_years: Any = None
    model_paths: Any = None
    grid_point: Any = None
    lat: Any = None
    lon: Any = None
    land_only: Any = None
    shp_only: Any = None
    reference_onset: Any = None
    rain_horizon: Any = None


# ----------------------------------------------------------------------------------------------------------------------
# projections: each one reads the keys it owns, once, from the flat record
# ----------------------------------------------------------------------------------------------------------------------

_FILLED = {f.key: f.fill for f in schema.FIELDS}


def _get(values: Mapping, key: str):
    """values[key], or None for a key that is left out when not given (fill=False) or is not in the schema."""
    return values[key] if _FILLED.get(key, False) else values.get(key)


def _pairs(dims) -> Dims | None:
    """A dims mapping as hashable pairs (None stays None)."""
    return None if dims is None else tuple((str(k), str(v)) for k, v in dict(dims).items())


def _tuple(value) -> tuple | None:
    """A list as a tuple, one level deep (the files of each year); None stays None."""
    return None if value is None else tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in value)


def _at(values: Mapping, key: str, i: int):
    """Entry i of a per-model list key (None when the list is absent or shorter)."""
    seq = _get(values, key)
    return seq[i] if seq is not None and i < len(seq) else None


def _project_run(v) -> dict:
    """The run-level keys that no spec holds."""
    sources = _get(v, "config_sources")
    return {"project_name": _get(v, "project_name"), "dataset_files": tuple(_get(v, "dataset_files") or ()),
            "catalog_dirs": tuple(_get(v, "catalog_dirs") or ()), "config_file": _get(v, "config_file"),
            "sources": tuple((str(k), str(s)) for k, s in (sources or {}).items()),
            "report": tuple(_get(v, "dataset_report") or ()), "lead_lines": tuple(_get(v, "onset_lead_lines") or ()),
            "paths": _get(v, "run_paths"), "cache_dir": _get(v, "cache_dir"),
            "short_names": bool(_get(v, "short_names"))}


def _project_units(v) -> dict:
    """{source name: UnitsSpec or None} of the observations, each model and the reference, resolved once per source
    (io.input.units_spec is pure)."""
    from hcwf.io.input import units_spec

    names = ("observations", *_get(v, "model_list"), _get(v, "ref_model"))
    return {name: units_spec(v, name) for name in names}


def _project_onset(v):
    """The resolved onset setup."""
    from hcwf.lib.onset_config import onset_setup

    return onset_setup(v)


def _project_blend(v, onset):
    """The blend's Recipe, or None when the config trains no blend."""
    if not schema.has_blend(v):
        return None
    from hcwf.blend.recipe import recipe_from_config

    return recipe_from_config(v, onset)


def _project_models(v, units) -> tuple[ModelSpec, ...]:
    """The forecast models, one per entry of the *_list keys."""
    from hcwf.utils.practical import set_dir

    out = []
    for i, name in enumerate(_get(v, "model_list")):
        folder = _at(v, "model_dir_list", i)
        out.append(ModelSpec(
            name=name, dir=folder if Path(folder).is_absolute() else set_dir(folder), var=_at(v, "model_var_list", i),
            unit_cvt=_at(v, "unit_cvt_list", i), file_pattern=_at(v, "file_pattern_list", i),
            accumulation=_at(v, "accumulation_list", i), accumulation_detected=_at(v, "accumulation_detected", i),
            daily_boundary=_at(v, "daily_boundary_list", i), dims=_pairs(_at(v, "model_dims_list", i)),
            paths=None if _at(v, "model_paths_list", i) is None else tuple(_at(v, "model_paths_list", i)),
            units=units[name]))
    return tuple(out)


def _project_obs(v, units) -> ObsSpec:
    """The observations."""
    return ObsSpec(name=_get(v, "obs"), dir=_get(v, "obs_dir"), file_pattern=tuple(_get(v, "obs_file_pattern")),
                   var=_get(v, "obs_var"), unit_cvt=_get(v, "obs_unit_cvt"), paths=_tuple(_get(v, "obs_paths")),
                   dims=_pairs(_get(v, "obs_dims")), day_boundary=_get(v, "obs_day_boundary"),
                   units=units["observations"])


def _project_reference(v, units) -> ReferenceSpec:
    """The reference."""
    name = _get(v, "ref_model")
    return ReferenceSpec(model=name, kind=name if name in CLIMATOLOGY_REFERENCES else "model",
                         climatology_cv=_get(v, "climatology_cv"), dir=_get(v, "ref_model_dir"),
                         file_pattern=_get(v, "ref_model_file_pattern"), var=_get(v, "ref_model_var"),
                         unit_cvt=_get(v, "ref_model_unit_cvt"), dims=_pairs(_get(v, "ref_model_dims")),
                         units=units[name])


def _project_evaluation(v) -> EvaluationSpec:
    """The evaluation, with the years and members resolved."""
    from hcwf.lib.control import climatology_years, evaluation_years, take_ensemble_members

    spans, tolerances = _get(v, "verification_window_list"), _get(v, "tolerance_days_list")
    # a tolerance belongs to the first window of that span (control.make_case looks it up by span)
    windows = tuple(Window(s[0], s[1], tolerances[list(spans).index(s)]) for s in spans)
    lead = _get(v, "lead_window")
    return EvaluationSpec(
        start_date=tuple(_get(v, "start_date")), end_date=tuple(_get(v, "end_date")),
        years=evaluation_years(v), climatology_years=climatology_years(v), init_days=tuple(_get(v, "init_days")),
        date_filter_year=_get(v, "date_filter_year"), windows=windows, max_forecast_day=_get(v, "max_forecast_day"),
        day_bins=tuple(tuple(b) for b in _get(v, "day_bins")), probabilistic=_get(v, "probabilistic"),
        members=take_ensemble_members(_get(v, "members")), lead_window=None if lead is None else tuple(lead))


def _project_onset_options(v) -> OnsetOptions:
    """The onset options outside OnsetSetup."""
    cells = _get(v, "explain_cells")
    return OnsetOptions(
        rain_resolution=_get(v, "rain_resolution"), missing=_get(v, "missing"), negatives=_get(v, "negatives"),
        gaps=_get(v, "gaps"), horizon=_get(v, "onset_horizon"), threshold_scale=_get(v, "threshold_scale"),
        threshold_table=_get(v, "threshold_table"), threshold_max_missing=_get(v, "threshold_max_missing"),
        threshold_partial_years=_get(v, "threshold_partial_years"),
        percentage_threshold=_get(v, "onset_percentage_threshold"),
        explain_cells=None if cells is None else tuple(tuple(c) for c in cells))


def _project_area(v) -> AreaSpec:
    """The region and its masks."""
    mask = _get(v, "mask")
    if mask:
        from hcwf.utils.mask import MaskSpec

        mask = MaskSpec.from_mapping(mask)
    return AreaSpec(region=_get(v, "region"), mask=mask or None, nc_mask=_get(v, "nc_mask"),
                    shpfile_dir=_get(v, "shpfile_dir"), polygon=_get(v, "polygon"))


def _project_parallel(v) -> ParallelSpec:
    """The worker processes."""
    return ParallelSpec(enabled=_get(v, "parallel"), workers=_get(v, "parallel_workers"))


def _project_generated(v, cls, section: str):
    """A generated spec (output, metrics): one value per key of the section."""
    return cls(**{f.nested_name: _get(v, f.key) for f in schema.FIELDS if f.section == section})


@dataclass(frozen=True, eq=False, slots=True)
class RunConfig:
    """A resolved run: typed specs plus the resolved flat record they were built from (``values``)."""

    project_name: str
    area: AreaSpec
    observations: ObsSpec
    reference: ReferenceSpec
    models: tuple[ModelSpec, ...]
    evaluation: EvaluationSpec
    onset: OnsetSetup
    onset_options: OnsetOptions
    metrics: MetricsSpec
    output: OutputSpec
    blend: Recipe | None
    parallel: ParallelSpec
    paths: RunPaths | None
    cache_dir: Path | str | None
    short_names: bool
    dataset_files: tuple[str, ...]
    catalog_dirs: tuple[str, ...]
    config_file: str | None
    sources: tuple[tuple[str, str], ...]
    report: tuple[str, ...]
    lead_lines: tuple[str, ...]
    _values: dict

    @classmethod
    def from_flat(cls, values: Mapping, *, check: bool = True) -> RunConfig:
        """The RunConfig of a resolved flat record (hcwf.config.build.resolve_flat); the record is copied. check=False skips
        the invariants (for records edited in place, as the bridge rebuilds them)."""
        run = _project_run(values)
        units = _project_units(values)
        onset = _project_onset(values)
        models = _project_models(values, units)
        evaluation = _project_evaluation(values)
        rc = cls(area=_project_area(values), observations=_project_obs(values, units),
                 reference=_project_reference(values, units), models=models, evaluation=evaluation, onset=onset,
                 onset_options=_project_onset_options(values), metrics=_project_generated(values, MetricsSpec, "metrics"),
                 output=_project_generated(values, OutputSpec, "output"), blend=_project_blend(values, onset),
                 parallel=_project_parallel(values), _values=dict(values), **run)
        if check:
            rc._check()
        return rc

    def _check(self) -> None:
        """The invariants of a resolved config."""
        names = [m.name for m in self.models]
        if not names:
            raise ValueError("a run needs at least one model")
        if len(set(names)) != len(names) or any(" " in n for n in names):
            raise ValueError(f"model names must be unique and contain no spaces, got {names}")
        late = [w.span for w in self.evaluation.windows if w.end > self.evaluation.max_forecast_day]
        if late:
            raise ValueError(f"windows {late} end after max_forecast_day {self.evaluation.max_forecast_day}")
        if (self.blend is not None) != schema.has_blend(self._values):
            raise ValueError("the blend recipe must exist exactly when the config has a blend section")

    # -- the flat record ---------------------------------------------------------------------------------------------

    @property
    def values(self) -> Mapping:
        """The resolved flat record (read-only; ``to_flat`` in hcwf.config.flat_bridge copies it)."""
        return MappingProxyType(self._values)

    def __eq__(self, other) -> bool:
        """Equal when built from equal records (the specs are projections of it)."""
        return isinstance(other, RunConfig) and self._values == other._values

    def __hash__(self) -> int:
        """Hash of the project name and run fingerprint."""
        return hash((self.project_name, None if self.paths is None else self.paths.fingerprint))

    # -- cases -------------------------------------------------------------------------------------------------------

    def model(self, name: str) -> ModelSpec:
        """The model called name."""
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(name)

    def cases(self) -> tuple[Case, ...]:
        """Every model with every window, in the order of product(model_list, verification_window_list)."""
        return tuple(Case(m, w) for m in self.models for w in self.evaluation.windows)

    def reference_cases(self, like: ModelSpec | None = None) -> tuple[Case, ...]:
        """The reference with every window. Two rules, kept for parity: with no `like` (the deterministic flow) the
        reference has models[0]'s accumulation and daily boundary and no model dims; with `like` (the ensemble flow, one
        model at a time) it has that model's accumulation and daily boundary and `reference.dims`."""
        if like is None:
            source, for_model = self.reference.as_model(self.models[0], dims=None), None
        else:
            source, for_model = self.reference.as_model(like, dims=self.reference.dims), like.name
        return tuple(Case(source, w, reference=True, for_model=for_model) for w in self.evaluation.windows)

    def figure_spec(self) -> FigureSpec:
        """What the figures read."""
        return FigureSpec(dir_fig=self.output.figure_dir, region=self.area.region, shpfile_dir=self.area.shpfile_dir,
                          polygon=self.area.polygon, save_fig=self.output.save_fig, show_plot=self.output.show_plot,
                          show_panel=self.output.show_panel, date_filter_year=self.evaluation.date_filter_year,
                          short_names=self.short_names, models=tuple(m.name for m in self.models))

    def without_files(self, tmp: Path | None = None) -> RunConfig:
        """The same run writing no files (write=False): every output off, with tmp as the output folders and no disk cache."""
        from hcwf.config.build import _NO_FILES

        values = dict(self._values)
        for key, value in _NO_FILES.items():
            if key in values or key == "explain_cells":
                values[key] = value
        if tmp is not None:
            values["dir_out"] = values["dir_fig"] = str(tmp)
            values["cache_dir"] = None
        if values.get("run_paths") is not None:
            values["run_paths"] = replace(values["run_paths"], write=False)
        return type(self).from_flat(values)
