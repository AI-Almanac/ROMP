"""Load, validate and normalize onset definition YAML into momp.onset.model.Definition.

Validation is strict: unknown keys and every ambiguity listed in
docs/onset_spec.md section 3.7 are errors, and each message names the YAML path
and says how to fix it. `extends` resolution, the registry and the semantic
definition hash also live here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import yaml

from momp.onset import units
from momp.onset.model import (
    Comparison, Condition, Curve, DayRef, Definition, DerivedSeasons, FirstPassage,
    Season, Span, ThresholdSpec, Total, Veto, WetDays,
)

REGISTRY_DIR = Path(__file__).parent / "registry"

_TOP_KEYS = ("id", "version", "title", "citation", "wording", "status", "notes", "extends", "wet_day",
             "season", "seasons", "thresholds", "onset", "curve", "regions")
_REQUIRED_KEYS = ("id", "version", "title", "citation", "wording", "status", "notes", "wet_day")
_RESTATED_KEYS = ("id", "version", "title", "status")
_EXCLUSIVE_PAIRS = (("season", "seasons"), ("onset", "curve"))
_ID_RE = re.compile(r"^[a-z0-9_]+$")
_REQUIRE_KEYS = ("first day", "all days", "wet days", "total")


class DefinitionError(ValueError):
    """An onset definition that is malformed or ambiguous; the message names the YAML path and the fix."""


def _fail(path: str, message: str):
    """Raise a DefinitionError for the YAML path."""
    raise DefinitionError(f"{path}: {message}" if path else message)


def _join(path: str, key: str) -> str:
    """Return the YAML path of a child key."""
    return f"{path}.{key}" if path else key


def _units(path: str, func, *args):
    """Call a units parser, re-raising its UnitsError as a DefinitionError at the YAML path."""
    try:
        return func(*args)
    except units.UnitsError as exc:
        raise DefinitionError(f"{path}: {exc}") from None


def _mapping(value, path: str, allowed: tuple[str, ...], required: tuple[str, ...] = ()) -> dict:
    """Check value is a mapping with only allowed keys and all required ones."""
    if not isinstance(value, dict):
        _fail(path, f"expected a mapping with keys {', '.join(allowed)} but found {value!r}. "
                    "Fix: write 'key: value' lines")
    for key in value:
        if key not in allowed:
            _fail(_join(path, str(key)), f"unknown key '{key}'. Fix: remove it or correct the spelling; "
                                         f"allowed keys are {', '.join(allowed)}")
    for key in required:
        if key not in value:
            _fail(path, f"missing required key '{key}'. Fix: add '{key}:'")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that rejects repeated keys in one mapping instead of keeping the last."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    """Build a mapping, raising on a duplicated key."""
    loader.flatten_mapping(node)
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key '{key}'. Fix: keep only one", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _read_mapping(text: str, origin: str) -> dict:
    """Parse YAML text into a mapping."""
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise DefinitionError(f"{origin}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise DefinitionError(f"{origin}: expected a mapping of definition keys at the top level. "
                              "Fix: write 'id: ...', 'wet_day: ...' and so on")
    return data


def registry_ids() -> list[str]:
    """Return the ids of the bundled definitions, sorted."""
    return sorted(p.stem for p in REGISTRY_DIR.glob("*.yaml"))


def _locate(ref: str, base_dir: Path | None) -> tuple[Path, str]:
    """Find a registry id or a relative file path, returning (resolved file, text)."""
    if ref in registry_ids():
        file = REGISTRY_DIR / f"{ref}.yaml"
    else:
        file = Path(ref)
        if not file.is_absolute():
            file = (base_dir if base_dir is not None else Path.cwd()) / file
    if not file.is_file():
        _fail("extends", f"'{ref}' is neither a registry id ({', '.join(registry_ids())}) nor a file "
                         f"relative to {base_dir if base_dir is not None else Path.cwd()}. "
                         "Fix: use a registry id or the path of a .yaml file")
    return file.resolve(), file.read_text()


def _merge(parent: dict, child: dict) -> dict:
    """Deep-merge two mappings: nested mappings merge key by key, everything else is replaced."""
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_extends(mapping: dict, base_dir: Path | None, stack: tuple[Path, ...]) -> dict:
    """Return the mapping with its `extends` chain merged in and the `extends` key removed."""
    if "extends" not in mapping:
        return mapping
    ref = mapping["extends"]
    if not isinstance(ref, str):
        _fail("extends", f"expected a registry id or file path but found {ref!r}. Fix: write e.g. 'extends: icpac'")
    for key in _RESTATED_KEYS:
        if key not in mapping:
            _fail("extends", f"'{key}' must be restated in an extending definition. Fix: add '{key}:'")
    file, text = _locate(ref, base_dir)
    if file in stack:
        chain = " -> ".join(p.stem for p in (*stack, file))
        _fail("extends", f"cycle in the extends chain ({chain}). Fix: remove one of the extends")
    parent = _resolve_extends(_read_mapping(text, str(file)), file.parent, (*stack, file))
    child = {k: v for k, v in mapping.items() if k != "extends"}
    for first, second in _EXCLUSIVE_PAIRS:
        for mine, other in ((first, second), (second, first)):
            if mine in child and other not in child:
                parent = {k: v for k, v in parent.items() if k != other}
    return _merge(parent, child)


def loads(text: str, *, base_dir: Path | str | None = None) -> Definition:
    """Parse and validate definition YAML text; a relative `extends` path is taken from base_dir."""
    base = Path(base_dir) if base_dir is not None else None
    merged = _resolve_extends(_read_mapping(text, "definition"), base, ())
    return _build(merged)


def load(id_or_path: str | Path) -> Definition:
    """Load a registry definition by id, or a definition from the path of a .yaml file."""
    if isinstance(id_or_path, str) and id_or_path in registry_ids():
        file = REGISTRY_DIR / f"{id_or_path}.yaml"
    else:
        file = Path(id_or_path)
        if not file.is_file():
            raise DefinitionError(
                f"'{id_or_path}' is neither a registry id ({', '.join(registry_ids())}) nor a file. "
                "Fix: use a registry id or the path of a .yaml file")
    file = file.resolve()
    merged = _resolve_extends(_read_mapping(file.read_text(), str(file)), file.parent, (file,))
    return _build(merged)


def override(defn: Definition, changes: dict) -> Definition:
    """Return the definition with `changes` merged into its stored YAML mapping and validated again.

    The merge follows `extends`: mappings merge key by key, lists and scalars replace.
    source_yaml of the result reflects the change, and the hash changes with the rule.
    """
    merged = _merge(yaml.safe_load(defn.source_yaml), changes)
    for first, second in _EXCLUSIVE_PAIRS:
        for mine, other in ((first, second), (second, first)):
            if mine in changes and other not in changes:
                merged.pop(other, None)
    return _build(merged)


def without_dry_spell_check(defn: Definition) -> Definition:
    """The definition without its `reject if` vetoes; unchanged for curve rules and rules without vetoes."""
    if not isinstance(defn.rule, FirstPassage) or not defn.rule.reject_if:
        return defn
    return override(defn, {
        "title": f"{defn.title} (no dry-spell check)",
        "notes": f"{defn.notes.rstrip()} Dry-spell check switched off by the run configuration.",
        "onset": {"reject if": []},
    })


def _canonical(obj):
    """Turn model objects into plain JSON values, tagging each dataclass with its class name."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: _canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.compare}
        return {"_type": type(obj).__name__, **fields}
    if isinstance(obj, (tuple, list)):
        return [_canonical(item) for item in obj]
    return obj


def definition_hash(defn: Definition) -> str:
    """sha256 of the canonical JSON of wet_day, seasons, rule and thresholds; provenance does not count."""
    semantic = {
        "wet_day": _canonical(defn.wet_day),
        "seasons": _canonical(defn.seasons),
        "rule": _canonical(defn.rule),
        "thresholds": {name: _canonical(spec) for name, spec in sorted(defn.thresholds)},
    }
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _build(m: dict) -> Definition:
    """Validate a fully merged mapping and normalize it into a Definition."""
    _mapping(m, "", _TOP_KEYS, _REQUIRED_KEYS)
    ident = m["id"]
    if not isinstance(ident, str) or not _ID_RE.match(ident):
        _fail("id", f"{ident!r} is not a valid id. Fix: use lowercase letters, digits and underscores, e.g. 'my_onset'")
    version = m["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        _fail("version", f"{version!r} is not a version number. Fix: write a whole number such as 1")
    if m["status"] not in ("draft", "verified"):
        _fail("status", f"{m['status']!r} is not a status. Fix: write 'draft' or 'verified'")
    provenance = {}
    for key in ("title", "citation", "wording", "notes"):
        value = m[key]
        if not isinstance(value, str) or not value.strip():
            _fail(key, "expected non-empty text. Fix: write the text, or 'TODO' while status is draft")
        if m["status"] == "verified" and value.strip().startswith("TODO"):
            _fail(key, "is still TODO but status is verified. Fix: fill it in or set 'status: draft'")
        provenance[key] = value

    regions = _regions(m.get("regions", []))
    thresholds = _thresholds(m.get("thresholds", {}))
    names = frozenset(name for name, _ in thresholds)
    wet_day = _wet_day(m["wet_day"])
    seasons = _seasons(m)
    if ("onset" in m) == ("curve" in m):
        _fail("onset", "exactly one of 'onset' (first-passage rule) or 'curve' (cumulative anomaly) is "
                       "needed. Fix: keep one of them")
    rule = _onset(m["onset"], names) if "onset" in m else _curve(m["curve"], names)
    return Definition(
        id=ident, version=version, status=m["status"], wet_day=wet_day, seasons=seasons, rule=rule,
        thresholds=thresholds, regions=regions, source_yaml=yaml.safe_dump(m, sort_keys=False), **provenance)


def _regions(value) -> tuple[str, ...]:
    """Parse the optional list of region ids, checking each exists in the region registry."""
    from momp.onset.regions import region_ids

    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail("regions", f"expected a list of region ids but found {value!r}. Fix: write e.g. 'regions: [east_africa]'")
    known = region_ids()
    for i, name in enumerate(value):
        if name not in known:
            _fail(f"regions[{i}]", f"'{name}' is not a region (known: {', '.join(known)}). "
                                   "Fix: use an id from momp/onset/regions or add that entry")
    return tuple(value)


def _comparison(value, path: str, names: frozenset[str]) -> Comparison:
    """Parse a comparison and check any threshold name it uses is defined."""
    comparison = _units(path, units.parse_comparison, value)
    name = comparison.amount.threshold
    if name is not None and name not in names:
        defined = ", ".join(sorted(names)) or "none defined"
        _fail(path, f"threshold '{name}' is not defined (defined: {defined}). "
                    f"Fix: add it under 'thresholds:' or use a fixed amount such as '20 mm'")
    return comparison


def _wet_day(value) -> Comparison:
    """Parse wet_day, which must be 'at least X mm' or 'more than X mm'."""
    comparison = _comparison(value, "wet_day", frozenset())
    if comparison.op not in ("ge", "gt") or comparison.amount.tenths is None:
        _fail("wet_day", f"{value!r} is not allowed. Fix: write 'at least X mm' or 'more than X mm' "
                         "with a fixed amount")
    return comparison


def _dayref(value, path: str, *, climatological: bool) -> DayRef:
    """Parse a day reference; climatological references are only valid in derived seasons."""
    ref = _units(path, units.parse_dayref, value)
    if climatological and ref.climatological is None:
        _fail(path, f"{value!r} is a fixed date but derived seasons need climatological bounds. "
                    "Fix: write e.g. 'climatological onset - 50 days'")
    if not climatological and ref.climatological is not None:
        _fail(path, f"{value!r} is climatological but this season is fixed. Fix: write a date as MM-DD, "
                    "or use 'derive: harmonic regime' under 'seasons'")
    return ref


def _seasons(m: dict):
    """Parse season / seasons into a tuple of Season or a DerivedSeasons."""
    if ("season" in m) == ("seasons" in m):
        _fail("season", "exactly one of 'season' (one window) or 'seasons' (several or derived) is needed. "
                        "Fix: keep one of them")
    if "season" in m:
        window = _mapping(m["season"], "season", ("from", "to"), ("from", "to"))
        return (_season("main", window, "season"),)
    block = m["seasons"]
    if not isinstance(block, dict) or not block:
        _fail("seasons", "expected named seasons or 'derive: harmonic regime'. Fix: write e.g. "
                         "'long_rains: {from: 02-01, to: 05-31}'")
    if "derive" in block:
        _mapping(block, "seasons", ("derive", "from", "to"), ("derive", "from", "to"))
        if block["derive"] != "harmonic regime":
            _fail("seasons.derive", f"{block['derive']!r} is not a known method. Fix: write 'harmonic regime'")
        return DerivedSeasons(
            method="harmonic",
            start=_dayref(block["from"], "seasons.from", climatological=True),
            end=_dayref(block["to"], "seasons.to", climatological=True))
    seasons = []
    for name, window in block.items():
        path = f"seasons.{name}"
        if not isinstance(name, str) or not _ID_RE.match(name):
            _fail(path, "is not a valid season name. Fix: use lowercase letters, digits and underscores")
        _mapping(window, path, ("from", "to"), ("from", "to"))
        seasons.append(_season(name, window, path))
    return tuple(seasons)


def _season(name: str, window: dict, path: str) -> Season:
    """Build a fixed Season from a from/to mapping."""
    return Season(name,
                  _dayref(window["from"], f"{path}.from", climatological=False),
                  _dayref(window["to"], f"{path}.to", climatological=False))


def _thresholds(block) -> tuple[tuple[str, ThresholdSpec], ...]:
    """Parse the thresholds mapping."""
    if not isinstance(block, dict):
        _fail("thresholds", "expected a mapping of threshold names. Fix: write e.g. "
                            "'wet_spell_amount: {mean total over: 5 days}'")
    specs = []
    for name, body in block.items():
        path = f"thresholds.{name}"
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            _fail(path, "is not a valid threshold name. Fix: start with a lowercase letter, then use "
                        "lowercase letters, digits and underscores")
        specs.append((name, _threshold(body, path)))
    return tuple(specs)


def _threshold(body, path: str) -> ThresholdSpec:
    """Parse one threshold specification."""
    kinds = ("mean total over", "mean daily rain", "quantile of totals over")
    _mapping(body, path, (*kinds, "quantile", "during"))
    present = [k for k in kinds if k in body]
    if len(present) != 1:
        _fail(path, f"needs exactly one of {', '.join(kinds)}. Fix: keep one of them")
    during = None
    if "during" in body:
        during = _units(_join(path, "during"), units.parse_during, body["during"])
    kind = present[0]
    if kind != "quantile of totals over" and "quantile" in body:
        _fail(_join(path, "quantile"), "only applies to 'quantile of totals over'. Fix: remove it")
    if kind == "mean daily rain":
        if body[kind] is not True:
            _fail(_join(path, kind), f"{body[kind]!r} is not allowed. Fix: write 'mean daily rain: true'")
        return ThresholdSpec("mean_daily", days=1, during=during)
    days = _units(_join(path, kind), units.parse_duration, body[kind])
    if kind == "mean total over":
        return ThresholdSpec("mean_total", days=days, during=during)
    q = body.get("quantile")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q < 1:
        _fail(_join(path, "quantile"), f"{q!r} is not a quantile. Fix: add 'quantile:' with a number "
                                       "between 0 and 1, e.g. 0.8")
    return ThresholdSpec("quantile_total", days=days, during=during, quantile=float(q))


def _curve(body, names: frozenset[str]) -> Curve:
    """Parse the curve rule."""
    _mapping(body, "curve", ("accumulate", "onset"), ("accumulate", "onset"))
    text = body["accumulate"]
    match = re.fullmatch(r"rain minus ([a-z][a-z0-9_]*)", text) if isinstance(text, str) else None
    if match is None:
        _fail("curve.accumulate", f"{text!r} is not understood. Fix: write 'rain minus <threshold name>'")
    baseline = match.group(1)
    if baseline not in names:
        _fail("curve.accumulate", f"threshold '{baseline}' is not defined (defined: "
                                  f"{', '.join(sorted(names)) or 'none defined'}). "
                                  "Fix: add it under 'thresholds:' with 'mean daily rain: true'")
    reports = {"day after minimum": "day_after_minimum", "day of minimum": "day_of_minimum"}
    if body["onset"] not in reports:
        _fail("curve.onset", f"{body['onset']!r} is not understood. Fix: write 'day after minimum' "
                             "or 'day of minimum'")
    return Curve(baseline=baseline, report=reports[body["onset"]])


def _span(value, path: str, window: int | None, *, allow_season_to_date: bool = False) -> Span:
    """Parse a span, optionally allowing 'season to date'."""
    span = _units(path, units.parse_span, value, window)
    if span.since_search_start and not allow_season_to_date:
        _fail(path, f"{value!r} is not allowed here. Fix: use 'days A to B' or 'N days after onset'")
    return span


def _onset(body, names: frozenset[str]) -> FirstPassage:
    """Parse the first-passage rule."""
    _mapping(body, "onset", ("window", "require", "reject if", "report"), ("require",))
    window = _units("onset.window", units.parse_duration, body["window"]) if "window" in body else None
    items = body["require"]
    if not isinstance(items, list) or not items:
        _fail("onset.require", "expected a non-empty list. Fix: add items such as '- total: at least 20 mm'")
    require = tuple(_requirement(item, f"onset.require[{i}]", window, names) for i, item in enumerate(items))
    vetoes_raw = body.get("reject if", [])
    if not isinstance(vetoes_raw, list):
        _fail("onset.reject if", "expected a list. Fix: write '- dry run: at least 7 days' items, or '[]' for none")
    vetoes = tuple(_veto(item, f"onset.reject if[{i}]", window, names) for i, item in enumerate(vetoes_raw))

    report, report_span = "candidate", None
    if "report" in body:
        if body["report"] == "first day":
            pass
        elif body["report"] == "first wet day":
            if window is None:
                _fail("onset.report", "'first wet day' needs a trigger window. Fix: add 'window: N days' "
                                      "under onset, or use 'report: first day'")
            report, report_span = "first_wet_day", Span(1, window)
        else:
            _fail("onset.report", f"{body['report']!r} is not understood. Fix: write 'first day' "
                                  "or 'first wet day'")
    return FirstPassage(require=require, reject_if=vetoes, report=report, report_span=report_span)


def _requirement(item, path: str, window: int | None, names: frozenset[str]) -> Condition:
    """Parse one `require` item into a Total or WetDays condition."""
    _mapping(item, path, (*_REQUIRE_KEYS, "over"))
    present = [k for k in _REQUIRE_KEYS if k in item]
    if len(present) != 1:
        _fail(path, f"needs exactly one of {', '.join(_REQUIRE_KEYS)}. Fix: keep one of them per item "
                    "(plus an optional 'over:')")
    key = present[0]
    value_path = _join(path, key)
    if key == "first day":
        if item[key] != "wet" or "over" in item:
            _fail(value_path, "must be 'first day: wet' without 'over'. Fix: write 'first day: wet'")
        return WetDays(Span(1, 1), "ge", 1)

    if "over" in item:
        span = _span(item["over"], _join(path, "over"), window, allow_season_to_date=(key == "total"))
    elif window is None:
        _fail(value_path, f"'{key}' applies to the trigger window but none is set. Fix: add "
                          "'window: N days' under onset, or give this item its own 'over:'")
    else:
        span = Span(1, window)

    if key == "total":
        return Total(span, _comparison(item[key], value_path, names))
    length = span.last - span.first + 1
    if key == "all days":
        if item[key] != "wet":
            _fail(value_path, f"{item[key]!r} is not allowed. Fix: write 'all days: wet'")
        return WetDays(span, "ge", length)
    op, count = _units(value_path, units.parse_count, item[key])
    return WetDays(span, op, count)


def _veto(item, path: str, window: int | None, names: frozenset[str]) -> Veto:
    """Parse one `reject if` item into a Veto."""
    _mapping(item, path, ("dry run", "dry window", "within", "starting within"))
    spells = [k for k in ("dry run", "dry window") if k in item]
    if len(spells) != 1:
        _fail(path, "needs exactly one of 'dry run' or 'dry window'. Fix: write e.g. 'dry run: at least 7 days'")
    places = [k for k in ("within", "starting within") if k in item]
    if len(places) != 1:
        _fail(path, "needs exactly one of 'within' or 'starting within' to say where the spell must "
                    "lie. Fix: add e.g. 'within: 21 days after onset'")
    kind, place = spells[0], places[0]
    if kind == "dry run":
        spell = _units(_join(path, kind), units.parse_dry_run, item[kind])
        length = spell.length
    else:
        spell = _units(_join(path, kind), units.parse_dry_window, item[kind])
        length = spell.days
        threshold = spell.comparison.amount.threshold
        if threshold is not None and threshold not in names:
            _fail(_join(path, kind), f"threshold '{threshold}' is not defined (defined: "
                                     f"{', '.join(sorted(names)) or 'none defined'}). "
                                     "Fix: add it under 'thresholds:' or use a fixed amount such as '5 mm'")
    span = _span(item[place], _join(path, place), window)
    placement = "inside" if place == "within" else "starts_inside"
    span_length = span.last - span.first + 1
    if placement == "inside" and length > span_length:
        _fail(_join(path, place), f"the {length}-day spell cannot fit inside a {span_length}-day span. "
                                  "Fix: lengthen the span, shorten the spell, or use 'starting within'")
    return Veto(spell, span, placement)

