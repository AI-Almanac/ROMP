"""Resolve ROMP's onset settings (definition, season, years, day boundary, lead label, backend) from a config dict.

Order of precedence for every setting: the explicit config value, then the
region's default (momp.onset.regions), then an error naming the missing key. A
config with no ``onset_definition`` key but the old flat keys (wet_init,
wet_spell, ...) is in legacy mode: the flat keys are translated by
momp.onset.legacy and lead k is dated init + k.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

from momp.lib.assertion import ROMPConfigError
from momp.onset import definition as definition_mod
from momp.onset import legacy, plan as plan_mod, regions
from momp.onset.ingest import parse_day_boundary
from momp.onset.model import Definition, DerivedSeasons, FirstPassage

FLAT_KEYS = ("wet_init", "wet_threshold", "wet_spell", "dry_threshold", "dry_spell", "dry_extent",
             "thresh_file", "thresh_var", "earliest_onset_date")
"""The old flat onset keys; they conflict with onset_definition."""

BACKENDS = ("numba", "reference", "gpu")
LEGACY_BOUNDARY = "00 UTC, labelled by start"
"""Placeholder day boundary for legacy ingestion; detection ignores it."""
FORECAST_BOUNDARY = "00 UTC, labelled by {label}"
"""Declared day boundary of forecast accumulations: 24 h after a 00 UTC init, labelled per lead_label."""


class OnsetConfigError(ROMPConfigError):
    """The onset settings are missing, inconsistent or unknown; the message says how to fix them."""


class Thresholds(dict):
    """Named per-cell thresholds ({name: DataArray(lat, lon)}), plus the season climatology of derived seasons."""

    season_climatology = None


@dataclass(frozen=True)
class OnsetSetup:
    """The resolved onset settings; sources says where each value came from."""

    legacy: bool
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

    @property
    def uses_thresholds(self) -> bool:
        """Whether the definition needs per-cell thresholds or derived seasons computed or loaded."""
        return not self.legacy and (bool(self.definition.thresholds)
                                    or isinstance(self.definition.seasons, DerivedSeasons))

    @property
    def derived_seasons(self) -> bool:
        """Whether the search windows come from a harmonic season analysis."""
        return isinstance(self.definition.seasons, DerivedSeasons)

    @property
    def ingest_boundary(self) -> str:
        """The day boundary handed to ingestion."""
        return self.obs_day_boundary or LEGACY_BOUNDARY

    @property
    def forecast_boundary(self) -> str:
        """The day boundary declared for forecasts."""
        return FORECAST_BOUNDARY.format(label=self.lead_label)

    def lines(self) -> list[str]:
        """Printable summary: one line per setting with where it came from."""
        d = self.definition
        src = self.sources
        out = []
        if self.legacy:
            out.append(f"Onset definition : legacy flat keys ({d.id}, deprecated)  [{src.get('definition')}]")
        else:
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


def is_legacy(cfg) -> bool:
    """Whether cfg uses the old flat keys: the loader's onset_legacy flag, else no onset_definition key at all."""
    flag = cfg.get("onset_legacy")
    return bool(flag) if flag is not None else "onset_definition" not in cfg


def detect_legacy(dic: dict) -> bool:
    """For a raw config dict: legacy iff there is no onset_definition key and an old flat key is set."""
    if "onset_definition" in dic:
        return False
    return any(key in dic and (key != "earliest_onset_date" or dic[key]) for key in FLAT_KEYS)


def check_conflicts(dic: dict) -> None:
    """Raise if onset_definition is combined with an old flat key."""
    if "onset_definition" not in dic:
        return
    clash = [key for key in FLAT_KEYS if dic.get(key) is not None]
    if clash:
        raise OnsetConfigError(
            f"{', '.join(clash)} cannot be combined with onset_definition. Fix: remove "
            f"{'it' if len(clash) == 1 else 'them'} from the config and put the rule in a definition "
            "(a YAML file with 'extends: <id>' and the changed values; see docs/onset_in_romp.md)")


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


def _threshold_files(value, definition: Definition) -> tuple[tuple[str, str, str], ...]:
    """Validate the thresholds mapping {name: "path.nc:variable"} against the definition."""
    if not value:
        return ()
    if not isinstance(value, dict):
        raise OnsetConfigError(f"thresholds = {value!r} must be a mapping like {{'name': 'path.nc:variable'}}")
    known = {name for name, _ in definition.thresholds}
    out = []
    for name, ref in value.items():
        if name not in known:
            raise OnsetConfigError(f"thresholds names '{name}', which definition '{definition.id}' does not use "
                                   f"(it uses: {', '.join(sorted(known)) or 'no thresholds'})")
        path, sep, var = str(ref).rpartition(":")
        if not sep or not path or not var:
            raise OnsetConfigError(f"thresholds['{name}'] = {ref!r} must look like 'path.nc:variable'")
        out.append((name, path, var))
    return tuple(out)


def _backend(requested):
    """Validate the backend name and fall back to 'reference' (with a warning) when it cannot be loaded."""
    if requested not in BACKENDS:
        raise OnsetConfigError(f"onset_backend = {requested!r} is unknown. Fix: use one of {', '.join(BACKENDS)}")
    if requested == "reference":
        return requested, None
    from momp.onset import engine
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


def onset_setup(cfg, thresh_slice=None) -> OnsetSetup:
    """Resolve the onset settings of a config mapping (memoized in definition mode).

    In legacy mode thresh_slice, when given, is the threshold the call site loaded:
    a number of mm, or a per-cell DataArray (the "legacy_threshold").
    """
    if is_legacy(cfg):
        return _legacy_setup(cfg, thresh_slice)
    keys = ("onset_definition", "onset_season", "threshold_years", "thresholds", "obs_day_boundary",
            "model_lead_label", "check_alignment", "onset_backend", "region", "dry_spell_check",
            "forecast_history", "threshold_source", "onset_override", "onset_rule")
    key = repr([cfg.get(k) for k in keys])
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


def _choices(cfg, key, allowed, default, legacy_mode):
    """Validate forecast_history / threshold_source; legacy mode only allows the previous behaviour."""
    value = cfg.get(key)
    if value is None:
        return default, "default"
    if value not in allowed:
        raise OnsetConfigError(f"{key} = {value!r} must be one of: {', '.join(repr(a) for a in allowed)}")
    if legacy_mode and value != default:
        raise OnsetConfigError(f"{key} = {value!r} needs an onset_definition; the old flat keys only work with "
                               f"{key} = {default!r}. Fix: set onset_definition, or remove {key}")
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


def _legacy_setup(cfg, thresh_slice) -> OnsetSetup:
    """Translate the flat keys; the day boundary is only known from an explicit key or the region."""
    flat = dict(cfg)
    if thresh_slice is not None:
        if getattr(thresh_slice, "ndim", 0):
            flat["thresh_file"] = "<array>"
        else:
            flat["wet_threshold"], flat["thresh_file"] = float(thresh_slice), None
    try:
        definition = legacy.from_flat_config(flat)
    except legacy.LegacyConfigError as exc:
        raise OnsetConfigError(str(exc)) from None
    chain, _ = _chain(cfg)
    boundary, origin = cfg.get("obs_day_boundary"), "config obs_day_boundary"
    if boundary is None:
        boundary, region_id = _region_value(chain, "day_boundary")
        origin = f"region default ({region_id})" if boundary else "unknown"
    definition, dry_check, dry_source = _dry_check(definition, cfg)
    history, history_source = _choices(cfg, "forecast_history", ("observed", "none"), "none", True)
    source, source_source = _choices(cfg, "threshold_source", ("observations", "model"), "observations", True)
    backend, note = _backend(cfg.get("onset_backend") or "numba")
    sources = {"dry_check": dry_source, "forecast_history": "legacy mode", "threshold_source": "legacy mode",
               "definition": "old flat keys", "season": "from start_date/earliest_onset_date to end_date + 47 days",
               "threshold_years": "legacy mode", "obs_day_boundary": origin,
               "lead_label": "legacy convention: lead k is init + k",
               "backend": note or "config onset_backend"}
    return OnsetSetup(True, definition, legacy.LEGACY_SEASON, None, (), boundary, "end", backend,
                      bool(cfg.get("check_alignment", True)), dry_check, (), history, source, sources)


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

    try:
        rule = cfg.get("onset_rule")
        if rule is not None:
            if not isinstance(rule, dict):
                raise OnsetConfigError(f"onset.rule = {rule!r} must be a mapping holding a complete definition")
            definition = definition_mod.loads(yaml.safe_dump(rule, sort_keys=False))
        else:
            definition = definition_mod.load(str(Path(str(ref)).expanduser())
                                             if str(ref) not in definition_mod.registry_ids() else str(ref))
        changes = cfg.get("onset_override")
        if changes is None:
            return definition, ()
        if not isinstance(changes, dict):
            raise OnsetConfigError(f"onset.override = {changes!r} must be a mapping of definition keys to change")
        return definition_mod.override(definition, changes), tuple(_changed_keys(changes))
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
                "Fix: set onset_definition = \"icpac\" (an id from `python -m momp.onset list`) or the path of a YAML file")
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

    files = _threshold_files(cfg.get("thresholds"), definition)
    years = cfg.get("threshold_years")
    if years is not None:
        years, sources["threshold_years"] = _years(years, "threshold_years"), "config threshold_years"
    else:
        value, region_id = _region_value(chain, "climatology_years")
        if value is not None:
            years, sources["threshold_years"] = tuple(value), f"region default ({region_id})"
    covered = {name for name, _, _ in files}
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
    history, sources["forecast_history"] = _choices(cfg, "forecast_history", ("observed", "none"), "observed", False)
    source, sources["threshold_source"] = _choices(cfg, "threshold_source", ("observations", "model"),
                                                   "observations", False)
    if source == "model":
        for name, spec in definition.thresholds:
            if name not in covered and spec.kind not in ("mean_total", "mean_daily"):
                raise OnsetConfigError(
                    f"threshold_source = 'model' cannot compute threshold '{name}' of '{definition.id}' "
                    f"({spec.kind}); the model's climatology supports only mean-based thresholds. "
                    "Fix: set threshold_source = 'observations' or give that threshold in thresholds = {...}")
    backend, note = _backend(cfg.get("onset_backend") or "numba")
    sources["backend"] = note or "config onset_backend"
    return OnsetSetup(False, definition, season, years, files, boundary, label, backend,
                      bool(cfg.get("check_alignment", True)), dry_check, tuple(overrides), history, source, sources)


def check_config(dic: dict) -> None:
    """Validate the onset part of a raw config dict: conflicts, then a full resolution."""
    check_conflicts(dic)
    cfg = dict(dic)
    cfg["onset_legacy"] = detect_legacy(dic)
    if not cfg["onset_legacy"]:
        cfg.setdefault("onset_definition", None)
    onset_setup(cfg)


def forecast_reach(setup: OnsetSetup) -> int:
    """The largest number of days after the candidate (candidate included) any requirement or veto reads."""
    compiled = plan_mod.compile(setup.definition)
    reach = 1
    for cond in compiled.conditions:
        reach = max(reach, cond.last)
    for veto in compiled.vetoes:
        extra = veto.length - 1 if veto.placement == plan_mod.STARTS_INSIDE else 0
        reach = max(reach, veto.last + extra)
    if compiled.report_span is not None:
        reach = max(reach, compiled.report_span[1])
    return reach
