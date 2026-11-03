"""Resolve HCWF's onset settings (definition, season, years, day boundary, lead label, backend) from a config dict.

Order of precedence for every setting: the explicit config value, then the
region's default (hcwf.onset.regions), then an error naming the missing key.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

from hcwf.lib.assertion import HCWFConfigError
from hcwf.onset import definition as definition_mod
from hcwf.onset import plan as plan_mod, regions
from hcwf.onset.ingest import parse_day_boundary
from hcwf.onset.model import Definition, DerivedSeasons, FirstPassage

BACKENDS = ("numba", "reference", "gpu")
FORECAST_BOUNDARY = "00 UTC, labelled by {label}"
"""Declared day boundary of forecast accumulations: 24 h after a 00 UTC init, labelled per lead_label."""


class OnsetConfigError(HCWFConfigError):
    """The onset settings are missing, inconsistent or unknown; the message says how to fix them."""


class Thresholds(dict):
    """Named per-cell thresholds ({name: DataArray(lat, lon)}), plus the season climatology of derived seasons."""

    season_climatology = None


@dataclass(frozen=True)
class OnsetSetup:
    """The resolved onset settings; sources says where each value came from."""

    definition: Definition
    season: str
    threshold_years: tuple[int, int] | None
    threshold_files: tuple[tuple[str, str, str], ...]
    obs_day_boundary: str | None
    lead_label: str
    backend: str
    check_alignment: bool
    dry_check: str = "on"
    overrides: tuple[str, ...] = field(default=(), compare=False)
    forecast_history: str = "none"
    threshold_source: str = "observations"
    sources: dict = field(default_factory=dict, compare=False)
    threshold_constants: tuple[tuple[str, float], ...] = ()
    """Thresholds given as a number of mm: (name, mm). `threshold_files` entries without a variable are tables."""

    @property
    def uses_thresholds(self) -> bool:
        """Whether the definition needs per-cell thresholds or derived seasons computed or loaded."""
        return bool(self.definition.thresholds) or isinstance(self.definition.seasons, DerivedSeasons)

    @property
    def derived_seasons(self) -> bool:
        """Whether the search windows come from a harmonic season analysis."""
        return isinstance(self.definition.seasons, DerivedSeasons)

    @property
    def ingest_boundary(self) -> str:
        """The day boundary handed to ingestion."""
        return self.obs_day_boundary

    @property
    def forecast_boundary(self) -> str:
        """The day boundary declared for forecasts."""
        return FORECAST_BOUNDARY.format(label=self.lead_label)

    def lines(self) -> list[str]:
        """Printable summary: one line per setting with where it came from."""
        d = self.definition
        src = self.sources
        out = []
        changed = f", overridden: {', '.join(self.overrides)}" if self.overrides else ""
        out.append(f"Onset definition : {d.id} v{d.version} ({d.status}){changed}  [{src.get('definition')}]")
        out.append(f"Season           : {self.season}  [{src.get('season')}]")
        years = f"{self.threshold_years[0]}-{self.threshold_years[1]}" if self.threshold_years else "not used"
        out.append(f"Threshold years  : {years}  [{src.get('threshold_years')}]")
        out.append(f"Day boundary     : {self.obs_day_boundary or 'unknown'}  [{src.get('obs_day_boundary')}]")
        out.append(f"Lead label       : {self.lead_label}  [{src.get('lead_label')}]")
        out.append(f"Dry-spell check  : {self.dry_check}  [{src.get('dry_check')}]")
        out.append(f"Forecast history : {self.forecast_history}  [{src.get('forecast_history')}]")
        out.append(f"Threshold source : {self.threshold_source}  [{src.get('threshold_source')}]")
        out.append(f"Backend          : {self.backend}  [{src.get('backend')}]")
        return out


def _chain(cfg):
    """The region named by cfg['region'] and its ancestors (nearest first), or ([], error text)."""
    name = cfg.get("region")
    if not isinstance(name, str):
        return [], "no region is set"
    try:
        region = regions.load_region(name)
    except regions.RegionError as exc:
        return [], str(exc)
    return [region, *regions.ancestors(region)], None


def _region_value(chain, attr):
    """The nearest non-None attribute of the region chain, with the region's id."""
    for region in chain:
        value = getattr(region, attr)
        if value is not None:
            return value, region.id
    return None, None


def _years(value, key: str) -> tuple[int, int]:
    """Validate a (first, last) year pair."""
    try:
        first, last = (int(v) for v in value)
    except (TypeError, ValueError):
        raise OnsetConfigError(f"{key} = {value!r} must be a (first, last) pair of years, e.g. (1998, 2024)") from None
    if first > last:
        raise OnsetConfigError(f"{key} = {value!r} has first year after last year")
    return first, last


def _threshold_files(value, definition: Definition):
    """Validate the thresholds mapping against the definition: (files, constants).

    A value is "path.nc:variable" (a grid), a table file without a variable (by unit id), or a number of mm.
    files are (name, path, variable or None for a table); constants are (name, mm).
    """
    if not value:
        return (), ()
    if not isinstance(value, dict):
        raise OnsetConfigError(f"thresholds = {value!r} must be a mapping like {{'name': 'path.nc:variable'}}")
    known = {name for name, _ in definition.thresholds}
    out, constants = [], []
    for name, ref in value.items():
        if name not in known:
            raise OnsetConfigError(f"thresholds names '{name}', which definition '{definition.id}' does not use "
                                   f"(it uses: {', '.join(sorted(known)) or 'no thresholds'})")
        if not isinstance(ref, str):
            constants.append((name, float(ref)))
            continue
        path, sep, var = ref.rpartition(":")
        if not sep:
            out.append((name, ref, None))
        elif not path or not var:
            raise OnsetConfigError(f"thresholds['{name}'] = {ref!r} must look like 'path.nc:variable'")
        else:
            out.append((name, path, var))
    return tuple(out), tuple(constants)


def _backend(requested):
    """Validate the backend name and fall back to 'reference' (with a warning) when it cannot be loaded."""
    if requested not in BACKENDS:
        raise OnsetConfigError(f"onset_backend = {requested!r} is unknown. Fix: use one of {', '.join(BACKENDS)}")
    if requested == "reference":
        return requested, None
    from hcwf.onset import engine
    try:
        module = engine._load_backend("numba" if requested == "numba" else "gpu")
        ok = module.available() if hasattr(module, "available") else True
    except ImportError:
        ok = False
    if ok:
        return requested, None
    reason = f"onset_backend = {requested!r} is not available here; using 'reference' (slower, same results)"
    warnings.warn(reason, UserWarning, stacklevel=3)
    return "reference", reason


_cache: dict = {}


_SETUP_KEYS = ("onset_definition", "onset_season", "threshold_years", "thresholds", "obs_day_boundary",
               "model_lead_label", "check_alignment", "onset_backend", "region", "dry_spell_check",
               "forecast_history", "threshold_source", "onset_override", "onset_rule", "rain_resolution")
"""The config keys the onset setup (definition mode) is resolved from."""


def setup_key(cfg) -> str:
    """Text identifying the onset setup of a config (definition mode): equal for equal settings, in any process."""
    return repr([cfg.get(k) for k in _SETUP_KEYS])


def onset_setup(cfg) -> OnsetSetup:
    """Resolve the onset settings of a config mapping (memoized)."""
    key = setup_key(cfg)
    if key not in _cache:
        _cache[key] = _definition_setup(cfg)
    return _cache[key]


def _dry_check(definition: Definition, cfg) -> tuple[Definition, str, str]:
    """Apply dry_spell_check: the definition to use, "on", "off" or "none in this definition", and its source."""
    wanted = cfg.get("dry_spell_check", True)
    if not isinstance(wanted, bool):
        raise OnsetConfigError(f"dry_spell_check = {wanted!r} must be True or False")
    has_vetoes = isinstance(definition.rule, FirstPassage) and bool(definition.rule.reject_if)
    source = "config dry_spell_check" if "dry_spell_check" in cfg else "default"
    if not has_vetoes:
        return definition, "none in this definition", source
    if wanted:
        return definition, "on", source
    if definition.source_yaml:
        return definition_mod.without_dry_spell_check(definition), "off", source
    return replace(definition, rule=replace(definition.rule, reject_if=())), "off", source


def _choices(cfg, key, allowed, default):
    """Validate forecast_history / threshold_source and say where the value came from."""
    value = cfg.get(key)
    if value is None:
        return default, "default"
    if value not in allowed:
        raise OnsetConfigError(f"{key} = {value!r} must be one of: {', '.join(repr(a) for a in allowed)}")
    return value, f"config {key}"


def history_lookback(definition: Definition) -> int:
    """How many days before the candidate any requirement or veto of the rule reads (0 if none)."""
    compiled = plan_mod.compile(definition)
    firsts = [c.first for c in compiled.conditions if not c.since_start] + [v.first for v in compiled.vetoes]
    return max(0, -min(firsts, default=0))


def history_needs_previous_year(setup: OnsetSetup) -> bool:
    """Whether the observed history of a year may start in the previous year: a wrapping season, or a season start
    closer to 1 January than the rule's look-back."""
    import datetime

    seasons = setup.definition.seasons
    if isinstance(seasons, DerivedSeasons) or setup.forecast_history != "observed":
        return False
    season = next(s for s in seasons if s.name == setup.season)
    start, end = season.start, season.end
    if (end.month, end.day) < (start.month, start.day):
        return True
    doy = datetime.date(2001, start.month, start.day).timetuple().tm_yday + start.offset_days
    return doy - history_lookback(setup.definition) < 1


def _changed_keys(changes: dict) -> list[str]:
    """The keys an override changes: top-level names, or the names inside a changed block (e.g. 'reject if')."""
    out = []
    for key, value in changes.items():
        out.extend(str(k) for k in value) if isinstance(value, dict) and value else out.append(str(key))
    return out


def _load_definition(cfg, ref, sources):
    """Load the definition (registry id, file path or inline rule) and apply onset_override; returns it with
    the list of changed keys."""
    import yaml

    fine = cfg.get("rain_resolution") == "exact"   # exact rain lets a definition state amounts finer than 0.1 mm
    try:
        rule = cfg.get("onset_rule")
        if rule is not None:
            if not isinstance(rule, dict):
                raise OnsetConfigError(f"onset.rule = {rule!r} must be a mapping holding a complete definition")
            definition = definition_mod.loads(yaml.safe_dump(rule, sort_keys=False), fine_amounts=fine)
        else:
            definition = definition_mod.load(str(Path(str(ref)).expanduser())
                                             if str(ref) not in definition_mod.registry_ids() else str(ref),
                                             fine_amounts=fine)
        changes = cfg.get("onset_override")
        if changes is None:
            return definition, ()
        if not isinstance(changes, dict):
            raise OnsetConfigError(f"onset.override = {changes!r} must be a mapping of definition keys to change")
        return definition_mod.override(definition, changes, fine_amounts=fine), tuple(_changed_keys(changes))
    except definition_mod.DefinitionError as exc:
        raise OnsetConfigError(f"onset_definition = {ref!r}: {exc}") from None


def _definition_setup(cfg) -> OnsetSetup:
    """Resolve every setting: explicit value, else region default, else an error."""
    chain, region_problem = _chain(cfg)
    sources: dict = {}
    ref = cfg.get("onset_definition")
    if cfg.get("onset_rule") is not None:
        if ref is not None:
            raise OnsetConfigError("onset.rule (a definition written inline) cannot be combined with "
                                   "onset.definition. Fix: keep one; use onset.override to change a registry definition")
        ref, sources["definition"] = "<inline>", "config onset.rule (inline definition)"
    elif ref is not None:
        sources["definition"] = "config onset_definition"
    else:
        default = regions.definitions_for(chain[0].id)[0] if chain else None
        if default is None:
            why = region_problem if not chain else f"region '{chain[0].title}' has no default definition"
            raise OnsetConfigError(
                f"onset_definition is not set and there is no region default ({why}). "
                "Fix: set onset_definition = \"icpac\" (an id from `hcwf-run onset list`) or the path of a YAML file")
        ref, sources["definition"] = default, f"region default ({chain[0].id})"
    definition, overrides = _load_definition(cfg, ref, sources)
    if chain:
        warning = regions.check_definition_region(definition, chain[0].id)
        if warning:
            warnings.warn(warning, UserWarning, stacklevel=3)

    seasons = definition.seasons
    names = [str(k) for k in range(1, seasons.max_seasons + 1)] if isinstance(seasons, DerivedSeasons) \
        else [s.name for s in seasons]
    season = cfg.get("onset_season")
    if season is None:
        if len(names) != 1:
            raise OnsetConfigError(f"definition '{definition.id}' has several seasons ({', '.join(names)}) and "
                                   f"onset_season is not set. Fix: set onset_season = \"{names[0]}\" (or another)")
        season, sources["season"] = names[0], "the definition's only season"
    else:
        season = str(season)
        sources["season"] = "config onset_season"
        if season not in names:
            raise OnsetConfigError(f"onset_season = {season!r} is not a season of '{definition.id}' "
                                   f"(its seasons: {', '.join(names)})")

    files, constants = _threshold_files(cfg.get("thresholds"), definition)
    years = cfg.get("threshold_years")
    if years is not None:
        years, sources["threshold_years"] = _years(years, "threshold_years"), "config threshold_years"
    else:
        value, region_id = _region_value(chain, "climatology_years")
        if value is not None:
            years, sources["threshold_years"] = tuple(value), f"region default ({region_id})"
    covered = {name for name, _, _ in files} | {name for name, _ in constants}
    needs_years = isinstance(seasons, DerivedSeasons) or any(n not in covered for n, _ in definition.thresholds)
    if years is None:
        if needs_years:
            raise OnsetConfigError(
                f"definition '{definition.id}' needs thresholds computed from observations but threshold_years is "
                "not set and the region declares none. Fix: set threshold_years = (1998, 2024), or give "
                "precomputed thresholds with thresholds = {'name': 'path.nc:variable'}")
        sources["threshold_years"] = "not needed"

    boundary = cfg.get("obs_day_boundary")
    if boundary is not None:
        sources["obs_day_boundary"] = "config obs_day_boundary"
    else:
        boundary, region_id = _region_value(chain, "day_boundary")
        if boundary is None:
            raise OnsetConfigError(
                "obs_day_boundary is not set and the region declares none. "
                "Fix: set obs_day_boundary = \"00 UTC, labelled by start\" (when the observation day starts and "
                "which date labels it)")
        sources["obs_day_boundary"] = f"region default ({region_id})"
    try:
        _, boundary_label = parse_day_boundary(boundary)
    except ValueError as exc:
        raise OnsetConfigError(f"obs_day_boundary: {exc}") from None

    label = cfg.get("model_lead_label")
    if label is None:
        label, sources["lead_label"] = boundary_label, "derived from obs_day_boundary"
    else:
        sources["lead_label"] = "config model_lead_label"
        if label not in ("start", "end"):
            raise OnsetConfigError(f"model_lead_label = {label!r} must be \"start\" or \"end\"")

    definition, dry_check, sources["dry_check"] = _dry_check(definition, cfg)
    if dry_check == "off":
        overrides = (*overrides, "reject if (dry_spell_check = false)")
    history, sources["forecast_history"] = _choices(cfg, "forecast_history", ("observed", "none"), "observed")
    source, sources["threshold_source"] = _choices(cfg, "threshold_source", ("observations", "model"),
                                                   "observations")
    if source == "model":
        for name, spec in definition.thresholds:
            if name not in covered and spec.kind not in ("mean_total", "mean_daily"):
                raise OnsetConfigError(
                    f"threshold_source = 'model' cannot compute threshold '{name}' of '{definition.id}' "
                    f"({spec.kind}); the model's climatology supports only mean-based thresholds. "
                    "Fix: set threshold_source = 'observations' or give that threshold in thresholds = {...}")
    backend, note = _backend(cfg.get("onset_backend") or "numba")
    sources["backend"] = note or "config onset_backend"
    return OnsetSetup(definition, season, years, files, boundary, label, backend,
                      bool(cfg.get("check_alignment", False)), dry_check, tuple(overrides), history, source, sources,
                      constants)


def check_config(dic: dict) -> None:
    """Validate the onset part of a raw config dict with a full resolution."""
    onset_setup({"onset_definition": None, **dic})


def forecast_reach(setup: OnsetSetup) -> int:
    """The largest number of days after the candidate (candidate included) any requirement or veto reads."""
    compiled = plan_mod.compile(setup.definition)
    reach = 1
    for cond in compiled.conditions:
        reach = max(reach, cond.last)
    for veto in compiled.vetoes:
        extra = veto.length - 1 if veto.placement in (plan_mod.STARTS_INSIDE, plan_mod.BEGINS) else 0
        reach = max(reach, veto.last + extra)
    if compiled.report_span is not None:
        reach = max(reach, compiled.report_span[1])
    return reach
