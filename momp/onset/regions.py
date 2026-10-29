"""Regions and countries: where each onset definition applies, and each area's reference datasets.

Entries live in momp/onset/regions/*.yaml (schema in docs/onset_spec.md,
"Regions and countries"). Areas nest through `part_of`, so a country inherits
the definitions of the regions that contain it.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from momp.onset.definition import registry_ids as definition_ids
from momp.onset.ingest import parse_day_boundary
from momp.onset.model import Definition

REGIONS_DIR = Path(__file__).parent / "regions"

_KEYS = ("id", "title", "kind", "iso", "aliases", "part_of", "domain", "observations", "climatology",
         "definitions", "notes")
_ID_RE = re.compile(r"^[a-z0-9_]+$")


class RegionError(ValueError):
    """A malformed region entry or an unknown region; the message names the YAML path and the fix."""


@dataclass(frozen=True)
class Region:
    """A country or region with its reference datasets and recommended onset definitions."""

    id: str
    title: str
    kind: str
    iso: str | None = None
    aliases: tuple[str, ...] = ()
    part_of: tuple[str, ...] = ()
    lat: tuple[float, float] | None = None
    lon: tuple[float, float] | None = None
    observations_dataset: str | None = None
    day_boundary: str | None = None
    climatology_years: tuple[int, int] | None = None
    default_definition: str | None = None
    also_definitions: tuple[str, ...] = ()
    notes: str = ""


def _fail(origin: str, path: str, message: str):
    """Raise a RegionError naming the file and YAML path."""
    raise RegionError(f"{origin}: {path}: {message}" if path else f"{origin}: {message}")


def _mapping(value, origin: str, path: str, allowed: tuple[str, ...]) -> dict:
    """Check value is a mapping with only allowed keys."""
    if not isinstance(value, dict):
        _fail(origin, path, f"expected a mapping but found {value!r}. Fix: write 'key: value' lines")
    for key in value:
        if key not in allowed:
            _fail(origin, f"{path}.{key}" if path else str(key),
                  f"unknown key '{key}'. Fix: remove it or correct the spelling; allowed keys are {', '.join(allowed)}")
    return value


def _strings(value, origin: str, path: str) -> tuple[str, ...]:
    """Check value is a list of strings."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail(origin, path, f"expected a list of text but found {value!r}. Fix: write e.g. '[a, b]'")
    return tuple(value)


def _span(value, origin: str, path: str) -> tuple[float, float]:
    """Check value is a [low, high] pair of numbers."""
    ok = (isinstance(value, list) and len(value) == 2
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value) and value[0] < value[1])
    if not ok:
        _fail(origin, path, f"{value!r} is not a range. Fix: write [low, high] in degrees with low < high")
    return float(value[0]), float(value[1])


def _parse(text: str, origin: str) -> Region:
    """Validate one region entry."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegionError(f"{origin}: invalid YAML: {exc}") from None
    data = _mapping(data, origin, "", _KEYS)
    for key in ("id", "title", "kind"):
        if key not in data:
            _fail(origin, "", f"missing required key '{key}'. Fix: add '{key}:'")
    ident = data["id"]
    if not isinstance(ident, str) or not _ID_RE.match(ident):
        _fail(origin, "id", f"{ident!r} is not a valid id. Fix: use lowercase letters, digits and underscores")
    if not isinstance(data["title"], str) or not data["title"].strip():
        _fail(origin, "title", "expected non-empty text. Fix: write the area's name")
    kind = data["kind"]
    if kind not in ("country", "region"):
        _fail(origin, "kind", f"{kind!r} is not a kind. Fix: write 'country' or 'region'")
    iso = data.get("iso")
    if kind == "country" and not (isinstance(iso, str) and re.fullmatch(r"[A-Z]{3}", iso)):
        _fail(origin, "iso", f"{iso!r} is not an ISO 3166-1 alpha-3 code. Fix: add e.g. 'iso: ETH' to a country")
    if kind == "region" and "iso" in data:
        _fail(origin, "iso", "is only for countries. Fix: remove it")
    aliases = _strings(data.get("aliases", []), origin, "aliases")
    part_of = _strings(data.get("part_of", []), origin, "part_of")
    for parent in part_of:
        if parent == ident or not (REGIONS_DIR / f"{parent}.yaml").is_file():
            _fail(origin, "part_of", f"'{parent}' is not another region (known: {', '.join(region_ids())}). "
                                     "Fix: use the id of an existing region entry")
    lat = lon = None
    if "domain" in data:
        box = _mapping(data["domain"], origin, "domain", ("lat", "lon"))
        for key in ("lat", "lon"):
            if key not in box:
                _fail(origin, "domain", f"missing '{key}'. Fix: write 'domain: {{lat: [3, 15], lon: [33, 48]}}'")
        lat, lon = _span(box["lat"], origin, "domain.lat"), _span(box["lon"], origin, "domain.lon")
    dataset = boundary = None
    if "observations" in data:
        obs = _mapping(data["observations"], origin, "observations", ("dataset", "day_boundary"))
        for key in ("dataset", "day_boundary"):
            if not isinstance(obs.get(key), str):
                _fail(origin, f"observations.{key}", "is required text. Fix: add it, e.g. "
                                                     "'day_boundary: 00 UTC, labelled by start'")
        dataset, boundary = obs["dataset"], obs["day_boundary"]
        try:
            parse_day_boundary(boundary)
        except ValueError as exc:
            _fail(origin, "observations.day_boundary", f"{exc}. Fix: use the form '00 UTC, labelled by start'")
    years = None
    if "climatology" in data:
        clim = _mapping(data["climatology"], origin, "climatology", ("years",))
        match = re.fullmatch(r"(\d{4})-(\d{4})", str(clim.get("years", "")))
        if match is None or int(match.group(1)) > int(match.group(2)):
            _fail(origin, "climatology.years", f"{clim.get('years')!r} is not a year range. "
                                               "Fix: write 'YYYY-YYYY' with the start not after the end, e.g. 1998-2024")
        years = (int(match.group(1)), int(match.group(2)))
    default, also = None, ()
    if "definitions" in data:
        block = _mapping(data["definitions"], origin, "definitions", ("default", "also"))
        known = definition_ids()
        default = block.get("default")
        also = _strings(block.get("also", []), origin, "definitions.also")
        named = ([("definitions.default", default)] if default is not None else []) + [
            (f"definitions.also[{i}]", d) for i, d in enumerate(also)]
        for path, name in named:
            if name not in known:
                _fail(origin, path, f"'{name}' is not a registry definition (known: {', '.join(known)}). "
                                    "Fix: use the id of a file in momp/onset/registry")
    return Region(id=ident, title=data["title"], kind=kind, iso=iso, aliases=aliases, part_of=part_of,
                  lat=lat, lon=lon, observations_dataset=dataset, day_boundary=boundary,
                  climatology_years=years, default_definition=default, also_definitions=also,
                  notes=str(data.get("notes", "")))


def region_ids() -> list[str]:
    """Return the ids of the bundled regions, sorted."""
    return sorted(p.stem for p in REGIONS_DIR.glob("*.yaml"))


@functools.lru_cache(maxsize=None)
def _index() -> dict[str, Region]:
    """Load every region, keyed by lower-cased id and alias."""
    index: dict[str, Region] = {}
    for file in sorted(REGIONS_DIR.glob("*.yaml")):
        region = _parse(file.read_text(), file.name)
        if region.id != file.stem:
            _fail(file.name, "id", f"'{region.id}' differs from the file name. Fix: rename the file to {region.id}.yaml")
        for name in (region.id, *region.aliases):
            key = name.lower()
            if key in index and index[key].id != region.id:
                _fail(file.name, "aliases", f"'{name}' is already used by {index[key].id}. Fix: remove the alias")
            index[key] = region
    return index


def load_region(name: str) -> Region:
    """Find a region by id or alias, ignoring case."""
    try:
        return _index()[name.strip().lower()]
    except KeyError:
        raise RegionError(
            f"unknown region '{name}' (known: {', '.join(region_ids())}). "
            "Fix: use a region id or alias, or add momp/onset/regions/<id>.yaml") from None


def ancestors(region: Region) -> list[Region]:
    """Return the regions containing this one through part_of, nearest first; cycles are cut."""
    seen = {region.id}
    out: list[Region] = []
    queue = list(region.part_of)
    while queue:
        parent = load_region(queue.pop(0))
        if parent.id in seen:
            continue
        seen.add(parent.id)
        out.append(parent)
        queue.extend(parent.part_of)
    return out


def definitions_for(name: str) -> tuple[str | None, list[str]]:
    """Return (default definition id or None, other recommended ids) for a region or country.

    The nearest area in the part_of chain that names a default wins; the others
    are the union of every `also` along the chain, without the default.
    """
    region = load_region(name)
    chain = [region, *ancestors(region)]
    default = next((r.default_definition for r in chain if r.default_definition), None)
    also: list[str] = []
    for r in chain:
        for ident in (r.default_definition, *r.also_definitions):
            if ident and ident != default and ident not in also:
                also.append(ident)
    return default, also


def check_definition_region(defn: Definition, name: str) -> str | None:
    """Return a warning when the definition is not meant for the region, else None.

    There is no warning if the definition lists no regions, or if the region or
    any region containing it is listed.
    """
    if not defn.regions:
        return None
    region = load_region(name)
    covered = {r.id for r in (region, *ancestors(region))}
    if covered & set(defn.regions):
        return None
    return (f"definition '{defn.id}' lists regions {', '.join(defn.regions)}, which do not include "
            f"{region.title} ({', '.join(sorted(covered))}); check it applies there")
