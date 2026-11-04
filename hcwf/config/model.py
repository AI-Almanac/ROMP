"""A resolved run as typed, frozen specs (design: ``docs/review/runconfig_design.md`` 2.2, 2.3, 2.8, 2.10).

This module holds `RunConfig` and the frozen specs it is made of (`ModelSpec`, `ObsSpec`, `ReferenceSpec`,
`EvaluationSpec`, ...), the `Case` pairs the flows loop over and the helpers on years, members and day bins. It is
built from the flat record that `hcwf.config.mapping` and `hcwf.config.build` resolve, and read by the pipeline::

    rc = resolve("run.yaml")
    for case in rc.cases():                     # every model with every verification window
        case.model.dir, case.window.span, case.window.tolerance
    rc.reference_cases()                        # the reference, with its own accumulation, daily boundary and dims
    rc.reference_cases(for_model=model)         # the same source; the cases are named after that forecast model

`RunConfig` keeps the resolved flat record (``rc.values``, read-only): ``run.yaml``, the fingerprint and
``RunResult.config`` are computed from it. The specs are projections of that record, built once in
`RunConfig.from_flat`. They declare attribute names and types only: every key, default, help text and choice is
declared once, in `hcwf.config.schema`, and `OutputSpec` and `MetricsSpec` are generated from its ``output`` and
``metrics`` sections. ``tests/test_runconfig_coverage.py`` checks that every schema key is read by a projection.

Imports of the heavy modules (``hcwf.io.input``, `hcwf.config.onset`, ``hcwf.blend.recipe``) are inside the
functions, so that ``import hcwf.config.model`` stays light.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, make_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

from hcwf.config import schema

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.blend.recipe import Recipe
    from hcwf.config.onset import OnsetSetup
    from hcwf.runs.folder import RunPaths
    from hcwf.spatial.units import UnitsSpec
    from hcwf.spatial.masks import MaskSpec

Dims = tuple[tuple[str, str], ...]
"""Canonical dimension name -> name in the files, as sorted-as-given pairs (a dict is not hashable)."""
CLIMATOLOGY_REFERENCES = ("climatology", "conditional_climatology")
"""The names of the reference sources that are climatologies, not a forecast model."""


# ----------------------------------------------------------------------------------------------------------------------
# helpers on years, members and day bins (the flat record and the typed specs both use them)
# ----------------------------------------------------------------------------------------------------------------------

def years_tuple_clim(year_start: int, year_end: int) -> tuple[int, ...]:
    """Create the tuple of the integers from `year_start` to `year_end` inclusive.

    Parameters
    ----------
    year_start : int
        The first year.
    year_end : int
        The last year, included.

    Returns
    -------
    tuple of int
        The years, ascending; empty when `year_end` is before `year_start`.
    """
    return tuple(range(year_start, year_end + 1))


def years_tuple_model(start_date: tuple[int, int, int], end_date: tuple[int, int, int]) -> tuple[int, ...]:
    """Create the tuple of the years from `start_date` to `end_date` inclusive.

    Parameters
    ----------
    start_date : tuple of (int, int, int)
        The first date as ``(year, month, day)``; only its year is used.
    end_date : tuple of (int, int, int)
        The last date as ``(year, month, day)``; only its year is used.

    Returns
    -------
    tuple of int
        The years, ascending.
    """
    return tuple(range(start_date[0], end_date[0] + 1))


def evaluation_years(dic: Mapping) -> tuple[int, ...]:
    """Return the years evaluated.

    Parameters
    ----------
    dic : collections.abc.Mapping
        The flat record, with ``years`` (``evaluation.years``) or else ``start_date`` and ``end_date``.

    Returns
    -------
    tuple of int
        ``years`` if given, else every year from ``start_date`` to ``end_date``.
    """
    return tuple(dic["years"]) if dic.get("years") else years_tuple_model(dic["start_date"], dic["end_date"])


def climatology_years(dic: Mapping) -> tuple[int, ...]:
    """Return the climatology years.

    Parameters
    ----------
    dic : collections.abc.Mapping
        The flat record, with ``years_clim`` (``evaluation.climatology_years``) or else ``start_year_clim`` and
        ``end_year_clim``.

    Returns
    -------
    tuple of int
        ``years_clim`` if given, else every year from ``start_year_clim`` to ``end_year_clim``.
    """
    return tuple(dic["years_clim"]) if dic.get("years_clim") else years_tuple_clim(dic["start_year_clim"],
                                                                                   dic["end_year_clim"])


def take_ensemble_members(members) -> tuple[int, ...] | None:
    """Normalize `members` into a tuple of ints (or None).

    Accepts a list or tuple of ints (returned as a tuple), a string ``"start-end"`` (expanded, end included), and
    None, empty or ``"All"`` (all members: None).

    Parameters
    ----------
    members : list of int or tuple of int or str or None
        The ensemble members as the config gives them.

    Returns
    -------
    tuple of int or None
        The members, or None for all of them.

    Raises
    ------
    TypeError
        `members` is none of the accepted forms.
    """
    if not members or members == "All":
        return None
    if isinstance(members, (list, tuple)):
        return tuple(members)
    if isinstance(members, str) and "-" in members:
        start, end = map(int, members.split("-", 1))
        return tuple(range(start, end + 1))
    raise TypeError("members must be list[int], tuple[int, ...], 'start-end' string, or None")


def filter_bins_in_window(day_bins, verification_window) -> tuple[tuple[int, int], ...]:
    """Return the day bins that lie completely within the verification window.

    For example ``((1, 5), (6, 10))`` in ``(6, 20)`` is ``((6, 10),)``.

    Parameters
    ----------
    day_bins : iterable of tuple of (int, int)
        The ``(first, last)`` lead days of each bin.
    verification_window : tuple of (int, int)
        The ``(start, end)`` lead days of the window, both included.

    Returns
    -------
    tuple of tuple of (int, int)
        The bins with ``start <= first`` and ``last <= end``, in the order given.
    """
    start_win, end_win = verification_window
    return tuple(b for b in day_bins if start_win <= b[0] and b[1] <= end_win)


@dataclass(frozen=True, slots=True)
class Window:
    """One verification window and its tolerance (``verification_window_list[i]`` and ``tolerance_days_list[i]``)."""

    start: int
    """The first lead day of the window."""
    end: int
    """The last lead day of the window, included."""
    tolerance: int
    """The tolerance of the window, in days."""

    @property
    def span(self) -> tuple[int, int]:
        """The ``(start, end)`` pair, the flat record's ``verification_window``."""
        return (self.start, self.end)

    @property
    def label(self) -> str:
        """The window as text in file names, e.g. ``"1-15"``."""
        return f"{self.start}-{self.end}"

    def bins(self, day_bins) -> tuple[tuple[int, int], ...]:
        """Return the day bins that lie completely inside the window.

        Parameters
        ----------
        day_bins : iterable of tuple of (int, int)
            The ``(first, last)`` lead days of each bin.

        Returns
        -------
        tuple of tuple of (int, int)
            The bins inside `span`, see `filter_bins_in_window`.
        """
        return filter_bins_in_window(day_bins, self.span)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One forecast model: the ``*_list`` keys of the flat record at its position."""

    name: str
    """The model name."""
    dir: Path | str
    """The folder of the model's files."""
    var: str
    """The name of the rainfall variable in the files."""
    unit_cvt: float | None
    """The multiplier to mm of files that declare no units, or None."""
    file_pattern: str
    """The file name pattern of the yearly files."""
    accumulation: str | None
    """Whether the rain is accumulated over the lead, or None when not known."""
    accumulation_detected: bool | None
    """Whether `accumulation` was detected from the files (not set in the config), or None."""
    daily_boundary: str | None
    """The boundary of the forecast day, or None."""
    dims: Dims | None
    """The dimension names in the files as `Dims` pairs, or None to infer them."""
    paths: tuple | None
    """The files listed in the config (one tuple per year), or None to find them with `file_pattern`."""
    units: UnitsSpec | None
    """The evaluation units of the model, or None when not resolved."""

    def dims_map(self) -> dict | None:
        """Return the dimension names as a dict, as the readers take them.

        Returns
        -------
        dict or None
            Canonical dimension name to name in the files, or None when they are inferred.
        """
        return None if self.dims is None else dict(self.dims)


@dataclass(frozen=True, slots=True)
class ObsSpec:
    """The observations: the observations section and the ``obs_*`` keys."""

    name: str
    """The observations' name."""
    dir: Path | str
    """The folder of the files."""
    file_pattern: tuple[str, ...]
    """The file name patterns of the yearly files."""
    var: str
    """The name of the rainfall variable in the files."""
    unit_cvt: float | None
    """The multiplier to mm of files that declare no units, or None."""
    paths: tuple | None
    """The files listed in the config (one tuple per year), or None to find them with `file_pattern`."""
    dims: Dims | None
    """The dimension names in the files as `Dims` pairs, or None to infer them."""
    day_boundary: str | None
    """The boundary of the observed day, or None."""
    units: UnitsSpec | None
    """The evaluation units of the observations, or None when not resolved."""

    def dims_map(self) -> dict | None:
        """Return the dimension names as a dict.

        Returns
        -------
        dict or None
            Canonical dimension name to name in the files, or None when they are inferred.
        """
        return None if self.dims is None else dict(self.dims)


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    """The reference of the skill scores: the reference section and ``ref_model_unit_cvt``."""

    model: str
    """The reference's name: ``"climatology"``, ``"conditional_climatology"`` or a model name."""
    kind: Literal["climatology", "conditional_climatology", "model"]
    """What the reference is: one of the two climatologies, or ``"model"`` for a forecast model."""
    climatology_cv: str
    """The cross-validation scheme of a climatology reference."""
    dir: Path | str | None
    """The folder of the reference model's files (a model reference only)."""
    file_pattern: str | None
    """The file name pattern of the reference model's files."""
    var: str | None
    """The name of the rainfall variable in the reference model's files."""
    unit_cvt: float | None
    """The multiplier to mm of reference files that declare no units, or None."""
    dims: Dims | None
    """The dimension names in the reference files as `Dims` pairs, or None to infer them."""
    units: UnitsSpec | None
    """The evaluation units of the reference, or None when not resolved."""
    accumulation: str | None = None
    """Whether the reference rain is accumulated over the lead (None: a climatology has none)."""
    accumulation_detected: bool | None = None
    """Whether `accumulation` was detected from the files, or None."""
    daily_boundary: str | None = None
    """The boundary of the reference forecast day (None: a climatology has none)."""

    def as_model(self) -> ModelSpec:
        """Return the reference as a source to read.

        It has its own name, data, unit conversion, dims, accumulation and daily boundary (a climatology reference has
        no accumulation or boundary of its own: None).

        Returns
        -------
        ModelSpec
            The reference as a model spec, without ``paths``.
        """
        return ModelSpec(name=self.model, dir=self.dir, var=self.var, unit_cvt=self.unit_cvt,
                         file_pattern=self.file_pattern, accumulation=self.accumulation,
                         accumulation_detected=self.accumulation_detected, daily_boundary=self.daily_boundary,
                         dims=self.dims, paths=None, units=self.units)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """The evaluation section and lead_window, with the years resolved."""

    start_date: tuple[int, int, int]
    """The first forecast date as ``(year, month, day)``."""
    end_date: tuple[int, int, int]
    """The last forecast date as ``(year, month, day)``."""
    years: tuple[int, ...]
    """The years evaluated, see `evaluation_years`."""
    climatology_years: tuple[int, ...]
    """The years of the climatology, see `climatology_years`."""
    init_days: tuple[int, ...]
    """The weekdays on which forecasts are initialised (0 is Monday)."""
    date_filter_year: int | None
    """The year for the init-day calendar alignment, or None for the start year."""
    windows: tuple[Window, ...]
    """The verification windows, each with its tolerance."""
    max_forecast_day: int
    """The longest lead, in days."""
    day_bins: tuple[tuple[int, int], ...]
    """The ``(first, last)`` lead days of each bin of the binned scores."""
    probabilistic: bool
    """Whether the ensemble flow is run."""
    members: tuple[int, ...] | None
    """The ensemble members used, or None for all."""
    lead_window: tuple[int, int] | None
    """The ``(first, last)`` lead days the forecasts are cut to, or None for all leads (see `hcwf.config.leads`)."""


@dataclass(frozen=True, slots=True)
class OnsetOptions:
    """The onset-section keys that `OnsetSetup` does not hold (None: the option's own default)."""

    rain_resolution: str | None
    """``"tenths"`` or ``"exact"``: the resolution rain is compared at."""
    missing: str | None
    """The missing-rain policy: ``"flag"``, ``"keep"``, ``"zero"`` or ``"drop"``."""
    negatives: str | None
    """``"default"`` or ``"keep"``: what happens to negative rain."""
    gaps: str | None
    """``"fill"`` or ``"refuse"``: what happens to gaps in the observations."""
    horizon: str | None
    """A follow-up past the end of the data: ``"censor"``, ``"truncate"`` or ``"reject"``; the engine uses ``"censor"`` when None."""
    threshold_scale: dict | None
    """The ``factor``, ``offset``, ``min`` and ``max`` mapping that rescales the thresholds."""
    threshold_table: dict | None
    """How a threshold table is read (a mapping of column and variable names)."""
    threshold_max_missing: float | None
    """The share of missing reference days above which a computed threshold series has none."""
    threshold_partial_years: bool | None
    """Whether the days of a year only partly in the data count towards computed thresholds."""
    percentage_threshold: float
    """The share of members (0-1) that must have an onset."""
    explain_cells: tuple | None
    """The ``(lat, lon)`` cells to explain, or None."""

    def ingest(self, forecast: bool, units, *, check_units=None) -> dict:
        """Return the parity options that `ingest` (or, with `forecast`, `ingest_forecast_info`) takes, as keywords.

        The options are ``rain_resolution`` (``"tenths"`` or ``"exact"``), ``missing``, ``negatives`` (``"default"``
        or ``"keep"``), ``gaps`` (observations: ``"fill"`` or ``"refuse"``) and ``check_units`` (forecasts; a per-call
        choice, not config). An option that is None is left out, so HCWF's own behaviour is what none gives. A cell
        transform applies the missing policy to the source cells as it reads them, so the observations are then
        ingested with ``"keep"``.

        Parameters
        ----------
        forecast : bool
            True for the options of a forecast ingest (no ``gaps``), false for the observations'.
        units : UnitsSpec or None
            The observations' units spec; its ``how`` is read when `forecast` is false.
        check_units : optional
            The forecasts' ``check_units`` choice, passed on when `forecast` is true and it is not None.

        Returns
        -------
        dict
            The keyword arguments, only the ones that are set.
        """
        out = {key: getattr(self, key) for key in (("rain_resolution", "missing", "negatives") if forecast
                                                   else ("rain_resolution", "missing", "negatives", "gaps"))
               if getattr(self, key) is not None}
        if forecast and check_units is not None:
            out["check_units"] = check_units
        if not forecast and units is not None and units.how == "cell_transform":
            out["missing"] = "keep"
        return out

    def engine(self, units, forecast: bool = False) -> dict:
        """Return the run options of the onset engine (horizon and missing-rain policy); none given is HCWF's own.

        Parameters
        ----------
        units : UnitsSpec or None
            The observations' units spec, as for `ingest`.
        forecast : bool, default False
            Whether the options are those of a forecast ingest.

        Returns
        -------
        dict
            ``{"horizon": ..., "missing": ...}``; the horizon is ``"censor"`` when none is set.
        """
        return {"horizon": self.horizon or "censor",
                "missing": self.ingest(forecast, units).get("missing")}


@dataclass(frozen=True, slots=True)
class AreaSpec:
    """Where the evaluation happens: the region, its masks and the boundary shapes of the maps."""

    region: str | None
    """The region name, or None."""
    mask: MaskSpec | None
    """The cell mask, or None."""
    nc_mask: Path | str | None
    """A NetCDF mask file of the evaluation, or None."""
    shpfile_dir: Path | str | None
    """The folder of the boundary shapes of the maps, or None."""
    polygon: bool
    """Whether the maps draw polygons."""
    mask_given: dict | None = field(default=None, compare=False, hash=False)
    """The `mask:` mapping exactly as the config gave it (a key written with its default value stays): what the
    observed-onset cache key hashes. None: not given (a spec built by hand then falls back to its non-default keys)."""


@dataclass(frozen=True, slots=True)
class ParallelSpec:
    """Whether and how many worker processes compute the years."""

    enabled: bool
    """Whether the years are computed in worker processes."""
    workers: int | None
    """The number of worker processes, or None for the default."""


def _generated(name: str, section: str, doc: str):
    """Build a frozen spec with one attribute (``Field.nested_name``) for each key of a schema section.

    Parameters
    ----------
    name : str
        The name of the class.
    section : str
        The schema section whose keys become the attributes.
    doc : str
        The ``__doc__`` of the class.

    Returns
    -------
    type
        A frozen, slotted dataclass, with `hcwf.config.model` as its module.
    """
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
    """What the figures read of a run (built by `RunConfig.figure_spec`)."""

    dir_fig: Path | str
    """The folder the figures are written to."""
    region: str | None
    """The region name, or None."""
    shpfile_dir: Path | str | None
    """The folder of the boundary shapes of the maps, or None."""
    polygon: bool
    """Whether the maps draw polygons."""
    save_fig: bool
    """Whether figures are saved."""
    show_plot: bool
    """Whether figures are shown."""
    show_panel: bool
    """Whether the panel figure is made."""
    date_filter_year: int | None
    """The year whose dates filter the maps, or None."""
    short_names: bool
    """Whether short model names are used in file names."""
    models: tuple[str, ...]
    """The names of the forecast models."""
    max_forecast_day: int
    """The longest lead, in days."""
    units: UnitsSpec | None
    """The observations' evaluation units: the maps draw a field over "unit" on their polygons or centroids."""


@dataclass(frozen=True, slots=True)
class Case:
    """One (source, window) pair: what today's ``convention.Case`` holds, as typed values.

    For a reference case `model` is the reference as a source (`ReferenceSpec.as_model`) and `for_model` is, in the
    ensemble flow, the forecast model it is scored against (that flow names the reference's case after that model).
    """

    model: ModelSpec
    """The source of the case (the reference as a model for a reference case)."""
    window: Window
    """The verification window of the case."""
    reference: bool = False
    """Whether the case is the reference's."""
    for_model: str | None = None
    """The forecast model a reference case is named after (ensemble flow), or None."""

    @property
    def name(self) -> str:
        """The flat record's ``case_name``, e.g. ``"AIFS_1-15"``."""
        from hcwf.naming import combi_to_str

        return combi_to_str((self.for_model or self.model.name, self.window.span)).replace(" ", "_")

    @property
    def key(self) -> tuple[str, tuple[int, int]]:
        """The ``(model name, window span)`` pair, the combination the flows loop over."""
        return (self.model.name, self.window.span)


@dataclass(frozen=True, slots=True)
class Selection:
    """The per-call choices that are not run config (what blend and export put into the kwargs): None is not set."""

    keep_unit_ids: Any = None
    """The evaluation unit ids to keep (the forecasts are read for these units only)."""
    member_ids: Any = None
    """The ensemble members to read, among those the files hold."""
    max_member: Any = None
    """The most members to read."""
    issue_dates: Any = None
    """The forecast issue dates to read."""
    first_lead: Any = None
    """The first lead day to select from the forecast files."""
    last_lead: Any = None
    """The last lead day to select from the forecast files."""
    model_files_years: Any = None
    """The years of the forecast files to use."""
    obs_files_years: Any = None
    """The years of the observation files to use."""
    model_paths: Any = None
    """The forecast files to read, in place of the configured ones."""
    grid_point: Any = None
    """Whether one grid point (at `lat` and `lon`) is selected."""
    lat: Any = None
    """The latitude of the grid point."""
    lon: Any = None
    """The longitude of the grid point."""
    land_only: Any = None
    """Whether the thresholds use land cells only."""
    shp_only: Any = None
    """Whether the thresholds use the cells inside the shapefile only."""
    reference_onset: Any = None
    """A `hcwf.onset.climatology.ReferenceOnset` to use in place of the configured reference, or None."""
    rain_horizon: Any = None
    """The ``(max_day, policy)`` rain horizon applied when the leads are selected."""
    check_units: Any = None
    """Whether the forecasts' units are checked (a per-call choice, not config)."""


NO_SELECTION = Selection()
"""The `Selection` that chooses nothing: the default of every reader's ``select``."""


def complete_flat(kw: Mapping[str, Any]) -> dict:
    """Return the flat values with what a partial dict leaves out filled in.

    A blend's frozen config and unit tests have fewer keys than a run holds: each missing schema key a resolved record
    always holds takes its default (None when it is required or computed), and the case's keys and the per-model
    lists a missing source needs take placeholders that no reader of the partial dict looks at.

    Parameters
    ----------
    kw : collections.abc.Mapping
        The partial flat record.

    Returns
    -------
    dict
        A copy of `kw` with the missing keys filled in.
    """
    out = dict(kw)
    for item in schema.FIELDS:
        if item.fill and item.key not in out:
            out[item.key] = None if callable(item.default) or item.default is schema.REQUIRED else copy.deepcopy(item.default)
    if not out.get("years_clim") and None in (out.get("start_year_clim"), out.get("end_year_clim")):
        out["start_year_clim"], out["end_year_clim"] = 1, 0   # no climatology years
    for key, placeholder in (("start_date", (1, 1, 1)), ("end_date", (1, 1, 1)), ("init_days", ()), ("day_bins", ())):
        if out.get(key) is None:
            out[key] = placeholder
    out.setdefault("model", (out.get("model_list") or ["model"])[0])
    out.setdefault("model_dir", "")
    for key in ("model_var", "unit_cvt", "file_pattern", "accumulation", "daily_boundary", "model_dims"):
        out.setdefault(key, None)
    span = out.get("verification_window") or (1, out.get("max_forecast_day") or 1)
    out["verification_window"] = span
    out.setdefault("tolerance_days", 0)
    out.setdefault("case_name", Case(ModelSpec(out["model"], "", None, None, None, None, None, None, None, None, None),
                                     Window(*span, out["tolerance_days"])).name)
    for key, value in (("model_list", [out["model"]]), ("model_dir_list", [out["model_dir"]]),
                       ("model_var_list", [out["model_var"]]), ("unit_cvt_list", [out["unit_cvt"]]),
                       ("file_pattern_list", [out["file_pattern"]]), ("verification_window_list", [tuple(span)]),
                       ("tolerance_days_list", [out["tolerance_days"]])):
        if out.get(key) is None:
            out[key] = value
    return out


# ----------------------------------------------------------------------------------------------------------------------
# projections: each one reads the keys it owns, once, from the flat record
# ----------------------------------------------------------------------------------------------------------------------

_FILLED = {f.key: f.fill for f in schema.FIELDS}


def _get(values: Mapping, key: str):
    """Return ``values[key]``, or None for a key that is left out when not given or is not in the schema.

    Parameters
    ----------
    values : collections.abc.Mapping
        The flat record.
    key : str
        The flat key.

    Returns
    -------
    object
        The value; None when the key is absent and its schema field has ``fill=False`` (or it is not in the schema).
        A key that the schema always fills (``fill=True``) and that is missing in `values` raises ``KeyError``.
    """
    return values[key] if _FILLED.get(key, False) else values.get(key)


def _pairs(dims) -> Dims | None:
    """Return a dims mapping as hashable pairs.

    Parameters
    ----------
    dims : collections.abc.Mapping or None
        Canonical dimension name to name in the files.

    Returns
    -------
    tuple of tuple of (str, str) or None
        The ``(name, file name)`` pairs in the mapping's order; None stays None.
    """
    return None if dims is None else tuple((str(k), str(v)) for k, v in dict(dims).items())


def _tuple(value) -> tuple | None:
    """Return a list as a tuple, one level deep (the files of each year).

    Parameters
    ----------
    value : iterable or None
        A list whose items may be lists or tuples.

    Returns
    -------
    tuple or None
        The items, those that are lists or tuples converted to tuples; None stays None.
    """
    return None if value is None else tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in value)


def _at(values: Mapping, key: str, i: int):
    """Return entry `i` of a per-model list key.

    Parameters
    ----------
    values : collections.abc.Mapping
        The flat record.
    key : str
        The key of the per-model list (``model_dir_list``, ...).
    i : int
        The model's position.

    Returns
    -------
    object
        The entry, or None when the list is absent or shorter.
    """
    seq = _get(values, key)
    return seq[i] if seq is not None and i < len(seq) else None


def _project_run(v) -> dict:
    """Project the run-level keys that no spec holds.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    dict
        The `RunConfig` field values ``project_name``, ``dataset_files``, ``catalog_dirs``, ``config_file``,
        ``sources``, ``report``, ``lead_lines``, ``paths``, ``cache_dir`` and ``short_names``.
    """
    sources = _get(v, "config_sources")
    return {"project_name": _get(v, "project_name"), "dataset_files": tuple(_get(v, "dataset_files") or ()),
            "catalog_dirs": tuple(_get(v, "catalog_dirs") or ()), "config_file": _get(v, "config_file"),
            "sources": tuple((str(k), str(s)) for k, s in (sources or {}).items()),
            "report": tuple(_get(v, "dataset_report") or ()), "lead_lines": tuple(_get(v, "onset_lead_lines") or ()),
            "paths": _get(v, "run_paths"), "cache_dir": _get(v, "cache_dir"),
            "short_names": bool(_get(v, "short_names"))}


def _project_units(v) -> dict:
    """Project the units spec of the observations, each model and the reference, resolved once per source.

    ``UnitsSpec.from_flat`` is pure, so one call per source name is enough.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    dict of str to UnitsSpec
        The spec by source name (``"observations"``, each model, the reference). An invalid ``units:`` section
        raises ``ValueError``.
    """
    from hcwf.spatial.units import UnitsSpec

    names = ("observations", *_get(v, "model_list"), _get(v, "ref_model"))
    return {name: UnitsSpec.from_flat(v, name) for name in names}


def _project_onset(v):
    """Project the resolved onset setup.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    hcwf.config.onset.OnsetSetup
        The setup, see `hcwf.config.onset.onset_setup`.
    """
    from hcwf.config.onset import onset_setup

    return onset_setup(v)


def _project_blend(v, onset, lenient: bool = False):
    """Project the blend's recipe, or None when the config trains no blend.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.
    onset : hcwf.config.onset.OnsetSetup
        The resolved onset setup.
    lenient : bool, default False
        Return None instead of raising when the recipe is invalid.

    Returns
    -------
    hcwf.blend.recipe.Recipe or None
        The recipe, or None when there is no ``blend:`` section (or, lenient, when its recipe is invalid).

    Raises
    ------
    ValueError
        The recipe is invalid and `lenient` is false (``RecipeError`` is a ``ValueError``).
    """
    if not schema.has_blend(v):
        return None
    from hcwf.blend.recipe import recipe_from_config

    try:
        return recipe_from_config(v, onset)
    except ValueError:   # RecipeError is an HCWFConfigError, a ValueError
        if not lenient:
            raise
        return None


def _project_models(v, units) -> tuple[ModelSpec, ...]:
    """Project the forecast models, one per entry of the ``*_list`` keys.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.
    units : dict
        The units spec by source name, from `_project_units`.

    Returns
    -------
    tuple of ModelSpec
        The models in ``model_list`` order; a relative ``model_dir`` is resolved against the data folder.
    """
    from hcwf.io.paths import set_dir

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
    """Project the observations.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.
    units : dict
        The units spec by source name, from `_project_units`.

    Returns
    -------
    ObsSpec
        The observations' spec.
    """
    return ObsSpec(name=_get(v, "obs"), dir=_get(v, "obs_dir"), file_pattern=_tuple(_get(v, "obs_file_pattern")),
                   var=_get(v, "obs_var"), unit_cvt=_get(v, "obs_unit_cvt"), paths=_tuple(_get(v, "obs_paths")),
                   dims=_pairs(_get(v, "obs_dims")), day_boundary=_get(v, "obs_day_boundary"),
                   units=units["observations"])


def _project_reference(v, units) -> ReferenceSpec:
    """Project the reference of the skill scores.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.
    units : dict
        The units spec by source name, from `_project_units`.

    Returns
    -------
    ReferenceSpec
        The reference's spec; its ``kind`` is ``"model"`` unless ``ref_model`` is a climatology.
    """
    name = _get(v, "ref_model")
    return ReferenceSpec(model=name, kind=name if name in CLIMATOLOGY_REFERENCES else "model",
                         climatology_cv=_get(v, "climatology_cv"), dir=_get(v, "ref_model_dir"),
                         file_pattern=_get(v, "ref_model_file_pattern"), var=_get(v, "ref_model_var"),
                         unit_cvt=_get(v, "ref_model_unit_cvt"), dims=_pairs(_get(v, "ref_model_dims")),
                         units=units[name], accumulation=_get(v, "ref_model_accumulation"),
                         accumulation_detected=_get(v, "ref_model_accumulation_detected"),
                         daily_boundary=_get(v, "ref_model_daily_boundary"))


def _project_evaluation(v) -> EvaluationSpec:
    """Project the evaluation, with the years and members resolved.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    EvaluationSpec
        The evaluation's spec; each window takes the tolerance of the first window of the same span.
    """
    spans, tolerances = _get(v, "verification_window_list"), _get(v, "tolerance_days_list")
    # a tolerance belongs to the first window of that span (as make_case did: looked up by span)
    windows = tuple(Window(s[0], s[1], tolerances[list(spans).index(s)]) for s in spans)
    lead = _get(v, "lead_window")
    return EvaluationSpec(
        start_date=tuple(_get(v, "start_date")), end_date=tuple(_get(v, "end_date")),
        years=evaluation_years(v), climatology_years=climatology_years(v), init_days=tuple(_get(v, "init_days")),
        date_filter_year=_get(v, "date_filter_year"), windows=windows, max_forecast_day=_get(v, "max_forecast_day"),
        day_bins=tuple(tuple(b) for b in _get(v, "day_bins")), probabilistic=_get(v, "probabilistic"),
        members=take_ensemble_members(_get(v, "members")), lead_window=None if lead is None else tuple(lead))


def _project_onset_options(v) -> OnsetOptions:
    """Project the onset options outside `OnsetSetup`.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    OnsetOptions
        The options; ``None`` where the config leaves them unset.
    """
    cells = _get(v, "explain_cells")
    return OnsetOptions(
        rain_resolution=_get(v, "rain_resolution"), missing=_get(v, "missing"), negatives=_get(v, "negatives"),
        gaps=_get(v, "gaps"), horizon=_get(v, "onset_horizon"), threshold_scale=_get(v, "threshold_scale"),
        threshold_table=_get(v, "threshold_table"), threshold_max_missing=_get(v, "threshold_max_missing"),
        threshold_partial_years=_get(v, "threshold_partial_years"),
        percentage_threshold=_get(v, "onset_percentage_threshold"),
        explain_cells=None if cells is None else tuple(tuple(c) for c in cells))


def _project_area(v) -> AreaSpec:
    """Project the region and its masks.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    AreaSpec
        The area's spec; its ``mask`` is None when the config has none.
    """
    mask = _get(v, "mask")
    if mask:
        from hcwf.spatial.masks import MaskSpec

        mask = MaskSpec.from_mapping(mask)
    given = _get(v, "mask")
    return AreaSpec(region=_get(v, "region"), mask=mask or None, nc_mask=_get(v, "nc_mask"),
                    shpfile_dir=_get(v, "shpfile_dir"), polygon=_get(v, "polygon"),
                    mask_given=dict(given) if isinstance(given, Mapping) else None)


def _project_parallel(v) -> ParallelSpec:
    """Project the worker processes.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.

    Returns
    -------
    ParallelSpec
        The parallel spec.
    """
    return ParallelSpec(enabled=_get(v, "parallel"), workers=_get(v, "parallel_workers"))


def _project_generated(v, cls, section: str):
    """Project a generated spec (output, metrics): one value per key of the section.

    Parameters
    ----------
    v : collections.abc.Mapping
        The flat record.
    cls : type
        The generated class (`OutputSpec` or `MetricsSpec`).
    section : str
        The schema section whose keys are read.

    Returns
    -------
    object
        An instance of `cls`.
    """
    return cls(**{f.nested_name: _get(v, f.key) for f in schema.FIELDS if f.section == section})


@dataclass(frozen=True, eq=False, slots=True)
class RunConfig:
    """A resolved run: typed specs plus the resolved flat record they were built from (``values``)."""

    project_name: str
    """The project's name."""
    area: AreaSpec
    """The region and its masks."""
    observations: ObsSpec
    """The observations."""
    reference: ReferenceSpec
    """The reference of the skill scores."""
    models: tuple[ModelSpec, ...]
    """The forecast models."""
    evaluation: EvaluationSpec
    """The evaluation section, with the years resolved."""
    onset: OnsetSetup
    """The resolved onset setup."""
    onset_options: OnsetOptions
    """The onset options that `onset` does not hold."""
    metrics: MetricsSpec
    """Which metrics to compute."""
    output: OutputSpec
    """The folders and which files and figures to write."""
    blend: Recipe | None
    """The blend's recipe, or None when the config trains no blend."""
    parallel: ParallelSpec
    """Whether and how many worker processes compute the years."""
    paths: RunPaths | None
    """The run folder's paths, or None when nothing is written."""
    cache_dir: Path | str | None
    """The cache folder, or None for the default."""
    short_names: bool
    """Whether short model names are used in file names."""
    dataset_files: tuple[str, ...]
    """The dataset files the config names."""
    catalog_dirs: tuple[str, ...]
    """The dataset catalog folders."""
    config_file: str | None
    """The path of the run file, or None."""
    sources: tuple[tuple[str, str], ...]
    """The ``(key, source)`` pairs telling where each config value came from."""
    report: tuple[str, ...]
    """The dataset detection report lines."""
    lead_lines: tuple[str, ...]
    """The lines reporting the lead handling of the onset."""
    _values: dict
    """The resolved flat record (read it through `values`)."""

    @classmethod
    def from_flat(cls, values: Mapping, *, check: bool = True, lenient: bool = False) -> RunConfig:
        """Build the `RunConfig` of a resolved flat record (`hcwf.config.build.resolve` builds it).

        Parameters
        ----------
        values : collections.abc.Mapping
            The resolved flat record; it is copied.
        check : bool, default True
            Check the invariants (``False`` for records edited in place).
        lenient : bool, default False
            Build a config whose ``units:`` or ``blend:`` section is invalid too (for ``--check``): the specs of an
            invalid section are None, and the invariants are not checked.

        Returns
        -------
        RunConfig
            The config.

        Raises
        ------
        ValueError
            The ``units:`` section or the blend recipe is invalid and `lenient` is false, or an invariant fails.
        """
        run = _project_run(values)
        try:
            units = _project_units(values)
        except ValueError:
            if not lenient:
                raise
            units = {name: None for name in ("observations", *_get(values, "model_list"), _get(values, "ref_model"))}
        onset = _project_onset(values)
        models = _project_models(values, units)
        evaluation = _project_evaluation(values)
        rc = cls(area=_project_area(values), observations=_project_obs(values, units),
                 reference=_project_reference(values, units), models=models, evaluation=evaluation, onset=onset,
                 onset_options=_project_onset_options(values), metrics=_project_generated(values, MetricsSpec, "metrics"),
                 output=_project_generated(values, OutputSpec, "output"),
                 blend=_project_blend(values, onset, lenient), parallel=_project_parallel(values),
                 _values=dict(values), **run)
        if check and not lenient:
            rc._check()
        return rc

    def _check(self) -> None:
        """Check the invariants of a resolved config.

        Raises
        ------
        ValueError
            There is no model, model names are not unique or contain spaces, a window ends after
            ``max_forecast_day``, or the blend recipe does not exist exactly when the config has a ``blend:``
            section.
        """
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
        """The resolved flat record (read-only; copy it with ``dict(rc.values)``)."""
        return MappingProxyType(self._values)

    def __eq__(self, other) -> bool:
        """Compare with another config: equal when built from equal records (the specs are projections of it).

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True when `other` is a `RunConfig` with an equal flat record.
        """
        return isinstance(other, RunConfig) and self._values == other._values

    def __hash__(self) -> int:
        """Return the hash of the project name and run fingerprint.

        Returns
        -------
        int
            The hash.
        """
        return hash((self.project_name, None if self.paths is None else self.paths.fingerprint))

    # -- cases -------------------------------------------------------------------------------------------------------

    def model(self, name: str) -> ModelSpec:
        """Return the model called `name`.

        Parameters
        ----------
        name : str
            The model name.

        Returns
        -------
        ModelSpec
            The model's spec.

        Raises
        ------
        KeyError
            There is no model of that name.
        """
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(name)

    def cases(self) -> tuple[Case, ...]:
        """Return every model with every window, in the order of ``product(model_list, verification_window_list)``.

        Returns
        -------
        tuple of Case
            The cases of the forecast models.
        """
        return tuple(Case(m, w) for m in self.models for w in self.evaluation.windows)

    def reference_cases(self, for_model: ModelSpec | None = None) -> tuple[Case, ...]:
        """Return the reference with every window.

        The reference is read with its own settings (`ReferenceSpec.as_model`: its data, accumulation, daily boundary
        and dims) in both flows. `for_model` (the ensemble flow, one forecast model at a time) only names the cases
        after that model, so that each model's results keep their own file names.

        Parameters
        ----------
        for_model : ModelSpec, optional
            The forecast model the cases are named after; None names them after the reference.

        Returns
        -------
        tuple of Case
            One reference case per window.
        """
        source = self.reference.as_model()
        name = None if for_model is None else for_model.name
        return tuple(Case(source, w, reference=True, for_model=name) for w in self.evaluation.windows)

    def first_case(self, model: str, *, reference_as: str | None = None) -> Case:
        """Return the first window's case of one model.

        The detection does not depend on the window, so that case stands for all. With `reference_as` (a source name:
        ``"climatology"`` or ``"conditional_climatology"``) the case is the reference's, under that name, built by
        `reference_cases`; `model` is then ignored.

        Parameters
        ----------
        model : str
            The model name.
        reference_as : str, optional
            The source name to give the reference's case; None returns the case of `model`.

        Returns
        -------
        Case
            The case. A `model` that is not in the config (with `reference_as` None) raises ``StopIteration``.
        """
        if reference_as is None:
            return next(c for c in self.cases() if c.model.name == model)
        case = self.reference_cases()[0]
        return replace(case, model=replace(case.model, name=reference_as))

    def figure_spec(self) -> FigureSpec:
        """Return what the figures read of the run.

        Returns
        -------
        FigureSpec
            The figures' spec.
        """
        return FigureSpec(dir_fig=self.output.figure_dir, region=self.area.region, shpfile_dir=self.area.shpfile_dir,
                          polygon=self.area.polygon, save_fig=self.output.save_fig, show_plot=self.output.show_plot,
                          show_panel=self.output.show_panel, date_filter_year=self.evaluation.date_filter_year,
                          short_names=self.short_names, models=tuple(m.name for m in self.models),
                          max_forecast_day=self.evaluation.max_forecast_day, units=self.observations.units)
