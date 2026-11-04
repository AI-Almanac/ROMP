"""Load, validate and normalize onset definition YAML into `hcwf.onset.model.Definition`.

Validation is strict: unknown keys and every ambiguity listed in ``docs/onset_spec.md`` section 3.7 are errors, and
each message names the YAML path and says how to fix it. ``extends`` resolution, the registry of bundled definitions
(``hcwf/onset/registry/*.yaml``) and the semantic definition hash also live here.

The text parsers for amounts, spans and counts are in `hcwf.onset.units`; the typed result classes are in
`hcwf.onset.model`; region ids are checked against `hcwf.onset.regions`. The entry points are `load` (a registry id
or a file), `loads` (YAML text), `override` (a change to a loaded definition) and `definition_hash`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import yaml

from hcwf.errors import DefinitionError
from hcwf.onset import units
from hcwf.onset.model import (
    Comparison, Condition, Curve, DayRef, Definition, DerivedSeasons, FirstPassage,
    Season, Span, ThresholdSpec, Total, Veto, WetDays,
)
from hcwf.errors import UnitsError

REGISTRY_DIR = Path(__file__).parent / "registry"
"""The folder of the bundled definitions, ``hcwf/onset/registry``."""

_TOP_KEYS = ("id", "version", "title", "citation", "wording", "status", "notes", "extends", "wet_day",
             "dry_day", "season", "seasons", "thresholds", "onset", "curve", "regions")
_REQUIRED_KEYS = ("id", "version", "title", "citation", "wording", "status", "notes", "wet_day")
_RESTATED_KEYS = ("id", "version", "title", "status")
_EXCLUSIVE_PAIRS = (("season", "seasons"), ("onset", "curve"))
_ID_RE = re.compile(r"^[a-z0-9_]+$")
_REQUIRE_KEYS = ("first day", "all days", "wet days", "total")


def _fail(path: str, message: str):
    """Raise a `~hcwf.errors.DefinitionError` for a YAML path.

    Parameters
    ----------
    path : str
        The dotted YAML path of the problem; empty for the whole document.
    message : str
        What is wrong and how to fix it.

    Raises
    ------
    DefinitionError
        Always, with the message prefixed by ``path`` when it is not empty.
    """
    raise DefinitionError(f"{path}: {message}" if path else message)


def _join(path: str, key: str) -> str:
    """Return the YAML path of a child key.

    Parameters
    ----------
    path : str
        The parent path; empty for the top level.
    key : str
        The child key.

    Returns
    -------
    str
        ``"path.key"``, or ``key`` alone at the top level.
    """
    return f"{path}.{key}" if path else key


def _units(path: str, func, *args):
    """Call a units parser, turning its `~hcwf.errors.UnitsError` into a `~hcwf.errors.DefinitionError` at the YAML path.

    Parameters
    ----------
    path : str
        The YAML path of the value being parsed.
    func : callable
        The parser from `hcwf.onset.units`.
    *args : tuple
        Passed on to ``func``.

    Returns
    -------
    object
        Whatever ``func`` returns.

    Raises
    ------
    DefinitionError
        ``func`` raised `~hcwf.errors.UnitsError`; the message starts with ``path``.
    """
    try:
        return func(*args)
    except UnitsError as exc:
        raise DefinitionError(f"{path}: {exc}") from None


def _mapping(value, path: str, allowed: tuple[str, ...], required: tuple[str, ...] = ()) -> dict:
    """Check that a value is a mapping with only the allowed keys and all the required ones.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when ``value`` is not a mapping, has an unknown key or lacks a required key.

    Parameters
    ----------
    value : object
        The parsed YAML value.
    path : str
        Its YAML path.
    allowed : tuple of str
        The permitted keys.
    required : tuple of str, default ()
        The keys that must be present.

    Returns
    -------
    dict
        ``value``, unchanged.
    """
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
    """A safe YAML loader that rejects a repeated key in one mapping instead of keeping the last."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    """Build a mapping from a YAML node, rejecting a duplicated key.

    Registered as the mapping constructor of `_UniqueKeyLoader`.

    Parameters
    ----------
    loader : yaml.SafeLoader
        The loader that is constructing the document.
    node : yaml.MappingNode
        The mapping node.
    deep : bool, default False
        Whether to construct nested objects at once.

    Returns
    -------
    dict
        The mapping, with merge keys (``<<``) flattened.

    Raises
    ------
    yaml.constructor.ConstructorError
        A key occurs twice in the mapping.
    """
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
    """Parse YAML text into a mapping.

    Parameters
    ----------
    text : str
        The YAML text.
    origin : str
        The file name or label used in error messages.

    Returns
    -------
    dict
        The top-level mapping, with repeated keys rejected.

    Raises
    ------
    DefinitionError
        The text is not valid YAML, repeats a key, or is not a mapping at the top level.
    """
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise DefinitionError(f"{origin}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise DefinitionError(f"{origin}: expected a mapping of definition keys at the top level. "
                              "Fix: write 'id: ...', 'wet_day: ...' and so on")
    return data


def registry_ids() -> list[str]:
    """Return the ids of the bundled definitions.

    Returns
    -------
    list of str
        The file stems of ``hcwf/onset/registry/*.yaml``, sorted.
    """
    return sorted(p.stem for p in REGISTRY_DIR.glob("*.yaml"))


def _locate(ref: str, base_dir: Path | None) -> tuple[Path, str]:
    """Find a registry id or a relative file path and read it.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when ``ref`` is neither a registry id nor an existing file.

    Parameters
    ----------
    ref : str
        A registry id, or a file path (relative paths are taken from ``base_dir``).
    base_dir : pathlib.Path or None
        The folder of the extending file; None means the current working directory.

    Returns
    -------
    tuple of (pathlib.Path, str)
        The resolved file and its text.
    """
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
    """Deep-merge two mappings.

    Nested mappings merge key by key; everything else (lists, scalars) is replaced by the child's value.

    Parameters
    ----------
    parent : dict
        The base mapping; not changed.
    child : dict
        The mapping whose values win.

    Returns
    -------
    dict
        A new merged mapping.
    """
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_extends(mapping: dict, base_dir: Path | None, stack: tuple[Path, ...]) -> dict:
    """Merge the ``extends`` chain of a mapping into it.

    The parent is loaded from a registry id or a file, resolved in turn, and merged under the child. When the child
    defines one of an exclusive pair (``season``/``seasons``, ``onset``/``curve``) and not the other, the parent's
    other key is dropped first. ``id``, ``version``, ``title`` and ``status`` must be restated in the extending
    definition.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when ``extends`` is not a string, a key is not restated, the parent is not found or invalid, or the chain is a cycle.

    Parameters
    ----------
    mapping : dict
        The parsed definition.
    base_dir : pathlib.Path or None
        Where a relative ``extends`` path is taken from; None means the current working directory.
    stack : tuple of pathlib.Path
        The files already being resolved, to detect a cycle.

    Returns
    -------
    dict
        The mapping with its chain merged in and the ``extends`` key removed; ``mapping`` itself when it has none.
    """
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


def loads(text: str, *, base_dir: Path | str | None = None, fine_amounts: bool = False) -> Definition:
    """Parse and validate definition YAML text.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the text is invalid (the message names the YAML path and the fix).

    Parameters
    ----------
    text : str
        The YAML text of one definition.
    base_dir : pathlib.Path or str, optional
        Where a relative ``extends`` path is taken from; None means the current working directory.
    fine_amounts : bool, default False
        Allow fixed amounts finer than 0.1 mm (they need ``rain_resolution`` ``"exact"``).

    Returns
    -------
    hcwf.onset.model.Definition
        The validated definition.
    """
    base = Path(base_dir) if base_dir is not None else None
    merged = _resolve_extends(_read_mapping(text, "definition"), base, ())
    return _build(merged, fine_amounts)


def load(id_or_path: str | Path, *, fine_amounts: bool = False) -> Definition:
    """Load a registry definition by id, or a definition from the path of a ``.yaml`` file.

    Parameters
    ----------
    id_or_path : str or pathlib.Path
        A registry id (see `registry_ids`) or the path of a definition file; a relative ``extends`` path is taken from
        the file's folder.
    fine_amounts : bool, default False
        Allow fixed amounts finer than 0.1 mm (they need ``rain_resolution`` ``"exact"``).

    Returns
    -------
    hcwf.onset.model.Definition
        The validated definition.

    Raises
    ------
    DefinitionError
        ``id_or_path`` is neither a registry id nor a file, or the definition is invalid.
    """
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
    return _build(merged, fine_amounts)


def override(defn: Definition, changes: dict, *, fine_amounts: bool = False) -> Definition:
    """Return a definition with changes merged into its stored YAML mapping and validated again.

    The merge follows ``extends``: mappings merge key by key, lists and scalars replace. ``source_yaml`` of the result
    reflects the change, and the hash changes with the rule.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the changed definition is invalid.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition to change.
    changes : dict
        The YAML keys to merge in; changing one of an exclusive pair (``season``/``seasons``, ``onset``/``curve``)
        removes the other.
    fine_amounts : bool, default False
        Allow fixed amounts finer than 0.1 mm.

    Returns
    -------
    hcwf.onset.model.Definition
        The new definition.
    """
    merged = _merge(yaml.safe_load(defn.source_yaml), changes)
    for first, second in _EXCLUSIVE_PAIRS:
        for mine, other in ((first, second), (second, first)):
            if mine in changes and other not in changes:
                merged.pop(other, None)
    return _build(merged, fine_amounts)


def without_dry_spell_check(defn: Definition) -> Definition:
    """Return the definition without its ``reject if`` vetoes.

    Curve rules and rules without vetoes are returned unchanged. Otherwise the title and notes say that the check is
    off, and the semantic hash changes.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition.

    Returns
    -------
    hcwf.onset.model.Definition
        ``defn`` itself, or a copy with ``reject if: []``.
    """
    if not isinstance(defn.rule, FirstPassage) or not defn.rule.reject_if:
        return defn
    return override(defn, {
        "title": f"{defn.title} (no dry-spell check)",
        "notes": f"{defn.notes.rstrip()} Dry-spell check switched off by the run configuration.",
        "onset": {"reject if": []},
    })


_OMITTED_WHEN_NONE = ("mm",)
"""Fields added after hashes were first recorded; leaving them out when unset keeps those hashes."""


def _canonical(obj):
    """Turn model objects into plain JSON values, tagging each dataclass with its class name.

    Fields with ``compare=False`` are left out, and so are the fields of ``_OMITTED_WHEN_NONE`` when unset.

    Parameters
    ----------
    obj : object
        A dataclass, a tuple or list of them, or a plain value.

    Returns
    -------
    object
        A dict (with a ``_type`` key) for a dataclass, a list for a tuple or list, else ``obj``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: _canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                  if f.compare and not (f.name in _OMITTED_WHEN_NONE and getattr(obj, f.name) is None)}
        return {"_type": type(obj).__name__, **fields}
    if isinstance(obj, (tuple, list)):
        return [_canonical(item) for item in obj]
    return obj


def definition_hash(defn: Definition) -> str:
    """Return the sha256 of the semantic content of a definition.

    The hash is taken over the canonical JSON of ``wet_day``, ``dry_day`` (if set), seasons, rule and thresholds;
    provenance (title, citation, wording, notes) does not count.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition.

    Returns
    -------
    str
        The 64-character hexadecimal digest.
    """
    semantic = {
        "wet_day": _canonical(defn.wet_day),
        "seasons": _canonical(defn.seasons),
        "rule": _canonical(defn.rule),
        "thresholds": {name: _canonical(spec) for name, spec in sorted(defn.thresholds)},
    }
    if defn.dry_day is not None:
        semantic["dry_day"] = _canonical(defn.dry_day)
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _build(m: dict, fine_amounts: bool = False) -> Definition:
    """Validate a fully merged mapping and normalize it into a `~hcwf.onset.model.Definition`.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the mapping is invalid.

    Parameters
    ----------
    m : dict
        The merged mapping (``extends`` already resolved).
    fine_amounts : bool, default False
        Allow fixed amounts finer than 0.1 mm while parsing.

    Returns
    -------
    hcwf.onset.model.Definition
        The validated definition.
    """
    with units.fine_amounts(fine_amounts):
        return _build_checked(m)


def _build_checked(m: dict) -> Definition:
    """Validate and normalize a mapping, inside the amount-precision context of `_build`.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when a key, id, version, status or text is invalid, or not exactly one of ``onset`` and ``curve`` is given.

    Parameters
    ----------
    m : dict
        The merged mapping.

    Returns
    -------
    hcwf.onset.model.Definition
        The validated definition, with ``source_yaml`` the dumped mapping.
    """
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
    dry_day = _dry_day(m["dry_day"]) if "dry_day" in m else None
    seasons = _seasons(m)
    if ("onset" in m) == ("curve" in m):
        _fail("onset", "exactly one of 'onset' (first-passage rule) or 'curve' (cumulative anomaly) is "
                       "needed. Fix: keep one of them")
    rule = _onset(m["onset"], names) if "onset" in m else _curve(m["curve"], names)
    return Definition(
        id=ident, version=version, status=m["status"], wet_day=wet_day, seasons=seasons, rule=rule,
        thresholds=thresholds, regions=regions, source_yaml=yaml.safe_dump(m, sort_keys=False), dry_day=dry_day,
        **provenance)


def _regions(value) -> tuple[str, ...]:
    """Parse the optional list of region ids, checking that each exists in the region registry.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when ``value`` is not a list of strings, or an id is not a known region.

    Parameters
    ----------
    value : object
        The ``regions`` value.

    Returns
    -------
    tuple of str
        The region ids.
    """
    from hcwf.onset.regions import region_ids

    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail("regions", f"expected a list of region ids but found {value!r}. Fix: write e.g. 'regions: [east_africa]'")
    known = region_ids()
    for i, name in enumerate(value):
        if name not in known:
            _fail(f"regions[{i}]", f"'{name}' is not a region (known: {', '.join(known)}). "
                                   "Fix: use an id from hcwf/onset/regions or add that entry")
    return tuple(value)


def _comparison(value, path: str, names: frozenset[str]) -> Comparison:
    """Parse a comparison and check that any threshold name it uses is defined.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the text cannot be parsed, or uses a threshold that is not in ``names``.

    Parameters
    ----------
    value : str
        The comparison text, such as ``"at least 1 mm"``.
    path : str
        Its YAML path.
    names : frozenset of str
        The defined threshold names.

    Returns
    -------
    hcwf.onset.model.Comparison
        The parsed comparison.
    """
    comparison = _units(path, units.parse_comparison, value)
    name = comparison.amount.threshold
    if name is not None and name not in names:
        defined = ", ".join(sorted(names)) or "none defined"
        _fail(path, f"threshold '{name}' is not defined (defined: {defined}). "
                    f"Fix: add it under 'thresholds:' or use a fixed amount such as '20 mm'")
    return comparison


def _wet_day(value) -> Comparison:
    """Parse ``wet_day``, which must be ``"at least X mm"`` or ``"more than X mm"``.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the text is not of that form.

    Parameters
    ----------
    value : str
        The ``wet_day`` text.

    Returns
    -------
    hcwf.onset.model.Comparison
        The comparison, with a fixed amount and operator ``ge`` or ``gt``.
    """
    comparison = _comparison(value, "wet_day", frozenset())
    if comparison.op not in ("ge", "gt") or comparison.amount.threshold is not None:
        _fail("wet_day", f"{value!r} is not allowed. Fix: write 'at least X mm' or 'more than X mm' "
                         "with a fixed amount")
    return comparison


def _dry_day(value) -> Comparison:
    """Parse ``dry_day``, which must be ``"less than X mm"`` or ``"at most X mm"``.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the text is not of that form.

    Parameters
    ----------
    value : str
        The ``dry_day`` text.

    Returns
    -------
    hcwf.onset.model.Comparison
        The comparison, with a fixed amount and operator ``lt`` or ``le``.
    """
    comparison = _comparison(value, "dry_day", frozenset())
    if comparison.op not in ("lt", "le") or comparison.amount.threshold is not None:
        _fail("dry_day", f"{value!r} is not allowed. Fix: write 'less than X mm' or 'at most X mm' "
                         "with a fixed amount, or leave dry_day out so a dry day is a valid day that is not wet")
    return comparison


def _dayref(value, path: str, *, climatological: bool) -> DayRef:
    """Parse a day reference.

    Climatological references are only valid in derived seasons, and fixed dates only in fixed seasons.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the text cannot be parsed, or its kind does not match ``climatological``.

    Parameters
    ----------
    value : str
        The text, such as ``"02-01"`` or ``"climatological onset - 50 days"``.
    path : str
        Its YAML path.
    climatological : bool
        Whether a climatological reference is required (True) or forbidden (False).

    Returns
    -------
    hcwf.onset.model.DayRef
        The parsed reference.
    """
    ref = _units(path, units.parse_dayref, value)
    if climatological and ref.climatological is None:
        _fail(path, f"{value!r} is a fixed date but derived seasons need climatological bounds. "
                    "Fix: write e.g. 'climatological onset - 50 days'")
    if not climatological and ref.climatological is not None:
        _fail(path, f"{value!r} is climatological but this season is fixed. Fix: write a date as MM-DD, "
                    "or use 'derive: harmonic regime' under 'seasons'")
    return ref


def _seasons(m: dict):
    """Parse ``season`` / ``seasons`` into fixed seasons or a derived-seasons rule.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when both or neither key is given, or the block, a season name, a window or the ``derive`` method is invalid.

    Parameters
    ----------
    m : dict
        The merged definition mapping; exactly one of ``season`` and ``seasons`` is read.

    Returns
    -------
    tuple of hcwf.onset.model.Season or hcwf.onset.model.DerivedSeasons
        A tuple of seasons (a single ``"main"`` season for ``season``), or the derived-seasons rule for
        ``derive: harmonic regime``.
    """
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
    """Build a fixed `~hcwf.onset.model.Season` from a ``from``/``to`` mapping.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when a date is invalid or climatological.

    Parameters
    ----------
    name : str
        The season name.
    window : dict
        The mapping with ``from`` and ``to`` dates (``MM-DD``).
    path : str
        The YAML path of the season.

    Returns
    -------
    hcwf.onset.model.Season
        The season.
    """
    return Season(name,
                  _dayref(window["from"], f"{path}.from", climatological=False),
                  _dayref(window["to"], f"{path}.to", climatological=False))


def _thresholds(block) -> tuple[tuple[str, ThresholdSpec], ...]:
    """Parse the ``thresholds`` mapping.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when ``block`` is not a mapping, a name is not valid, or a specification is invalid.

    Parameters
    ----------
    block : dict
        The mapping of threshold names to their specifications.

    Returns
    -------
    tuple of tuple of (str, hcwf.onset.model.ThresholdSpec)
        The ``(name, spec)`` pairs in the order written.
    """
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
    """Parse one threshold specification.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when a key is unknown or misplaced, a duration or quantile is invalid, or not exactly one kind is given.

    Parameters
    ----------
    body : dict
        The specification: exactly one of ``mean total over``, ``mean daily rain`` and ``quantile of totals over``,
        plus ``quantile`` for the last and an optional ``during``.
    path : str
        Its YAML path.

    Returns
    -------
    hcwf.onset.model.ThresholdSpec
        The parsed specification.
    """
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
    """Parse the ``curve`` rule.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when a key is missing or unknown, the text is not understood, or the threshold is not defined.

    Parameters
    ----------
    body : dict
        The mapping with ``accumulate`` (``"rain minus <threshold name>"``) and ``onset`` (``"day after minimum"`` or
        ``"day of minimum"``).
    names : frozenset of str
        The defined threshold names.

    Returns
    -------
    hcwf.onset.model.Curve
        The cumulative-anomaly rule.
    """
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
    """Parse a span, optionally allowing ``"season to date"``.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the text cannot be parsed, or is ``"season to date"`` where that is not allowed.

    Parameters
    ----------
    value : str
        The span text, such as ``"days 1 to 5"`` or ``"5 days after onset"``.
    path : str
        Its YAML path.
    window : int or None
        The trigger window in days, for spans given relative to it.
    allow_season_to_date : bool, default False
        Whether ``"season to date"`` is accepted.

    Returns
    -------
    hcwf.onset.model.Span
        The parsed span.
    """
    span = _units(path, units.parse_span, value, window)
    if span.since_search_start and not allow_season_to_date:
        _fail(path, f"{value!r} is not allowed here. Fix: use 'days A to B' or 'N days after onset'")
    return span


def _onset(body, names: frozenset[str]) -> FirstPassage:
    """Parse the first-passage rule.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when a key is unknown or missing, a list is invalid, or ``report`` is not understood or needs a window that is not set.

    Parameters
    ----------
    body : dict
        The ``onset`` mapping: ``require`` (a non-empty list), and optionally ``window``, ``reject if`` and ``report``.
    names : frozenset of str
        The defined threshold names.

    Returns
    -------
    hcwf.onset.model.FirstPassage
        The rule.
    """
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
    """Parse one ``require`` item into a `~hcwf.onset.model.Total` or `~hcwf.onset.model.WetDays` condition.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the keys are invalid, the span is missing or invalid, or the value is not allowed.

    Parameters
    ----------
    item : dict
        The mapping with exactly one of ``first day``, ``all days``, ``wet days`` and ``total``, plus an optional
        ``over``.
    path : str
        Its YAML path.
    window : int or None
        The trigger window in days; the span of an item without ``over``.
    names : frozenset of str
        The defined threshold names.

    Returns
    -------
    hcwf.onset.model.Condition
        A `~hcwf.onset.model.WetDays` or `~hcwf.onset.model.Total` condition.
    """
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
    """Parse one ``reject if`` item into a `~hcwf.onset.model.Veto`.

    Raises `~hcwf.errors.DefinitionError` (through `_fail`) when the keys are invalid, a spell or span cannot be parsed, a threshold is not defined, or the spell cannot fit inside the span.

    Parameters
    ----------
    item : dict
        The mapping with exactly one of ``dry run`` and ``dry window`` and exactly one of ``within``,
        ``starting within`` and ``beginning within``.
    path : str
        Its YAML path.
    window : int or None
        The trigger window in days, for a span given relative to it.
    names : frozenset of str
        The defined threshold names.

    Returns
    -------
    hcwf.onset.model.Veto
        The veto.
    """
    _mapping(item, path, ("dry run", "dry window", "within", "starting within", "beginning within"))
    spells = [k for k in ("dry run", "dry window") if k in item]
    if len(spells) != 1:
        _fail(path, "needs exactly one of 'dry run' or 'dry window'. Fix: write e.g. 'dry run: at least 7 days'")
    places = [k for k in ("within", "starting within", "beginning within") if k in item]
    if len(places) != 1:
        _fail(path, "needs exactly one of 'within', 'starting within' or 'beginning within' to say where "
                    "the spell must lie. Fix: add e.g. 'within: 21 days after onset'")
    kind, place = spells[0], places[0]
    if place == "beginning within" and kind != "dry run":
        _fail(_join(path, place), "only applies to a 'dry run'. Fix: use 'within' or 'starting within'")
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
    placement = {"within": "inside", "starting within": "starts_inside", "beginning within": "begins"}[place]
    span_length = span.last - span.first + 1
    if placement == "inside" and length > span_length:
        _fail(_join(path, place), f"the {length}-day spell cannot fit inside a {span_length}-day span. "
                                  "Fix: lengthen the span, shorten the spell, or use 'starting within'")
    return Veto(spell, span, placement)

