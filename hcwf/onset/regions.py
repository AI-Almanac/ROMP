"""Regions and countries: where each onset definition applies, and each area's reference datasets.

Entries live in ``hcwf/onset/regions/*.yaml`` (schema in ``docs/onset_spec.md``, "Regions and countries"). Areas nest
through ``part_of``, so a country inherits the definitions of the regions that contain it. `box` is the one lookup of a
region's lat/lon box, and `polygon_boundary` the India CMZ outline for the grids that have one. Definitions are named
by the ids of `hcwf.onset.definition`'s registry; the units of an area are `hcwf.spatial.units.UnitsSpec`.
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from hcwf.errors import RegionError
from hcwf.onset.definition import registry_ids as definition_ids
from hcwf.onset.ingest import parse_day_boundary
from hcwf.onset.model import Definition
from hcwf.spatial.units import UnitsSpec
from hcwf.errors import UnitsSpecError

log = logging.getLogger(__name__)

REGIONS_DIR = Path(__file__).parent / "regions"
"""The folder of the bundled region YAML files."""

_KEYS = ("id", "title", "kind", "iso", "aliases", "part_of", "domain", "units", "observations",
         "climatology", "definitions", "notes")
_UNITS_KEYS = ("shapefile", "unit_id_col", "unit_name_col", "parent_col", "weights", "centroids", "centroid_id_col",
               "dissemination", "dissemination_id_col")
_ID_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Region:
    """A country or region with its reference datasets and recommended onset definitions."""

    id: str
    """The identifier, also the name of its YAML file."""
    title: str
    """The area's name."""
    kind: str
    """``"country"`` or ``"region"``."""
    iso: str | None = None
    """The ISO 3166-1 alpha-3 code of a country."""
    aliases: tuple[str, ...] = ()
    """Other names the area is found by."""
    part_of: tuple[str, ...] = ()
    """The ids of the regions that contain this one."""
    lat: tuple[float, float] | None = None
    """The latitude range ``(low, high)`` of the domain box, in degrees."""
    lon: tuple[float, float] | None = None
    """The longitude range ``(low, high)`` of the domain box, in degrees."""
    units: UnitsSpec | None = None
    """The spatial units of the area (see `hcwf.spatial.units.UnitsSpec`)."""
    observations_dataset: str | None = None
    """The name of the reference observation dataset."""
    day_boundary: str | None = None
    """The day boundary of that dataset, for example ``"00 UTC, labelled by start"``."""
    climatology_years: tuple[int, int] | None = None
    """The first and last year of the reference climatology."""
    default_definition: str | None = None
    """The id of the definition recommended for the area."""
    also_definitions: tuple[str, ...] = ()
    """The ids of other recommended definitions."""
    notes: str = ""
    """Free-text notes."""


def _fail(origin: str, path: str, message: str):
    """Raise a `~hcwf.errors.RegionError` naming the file and YAML path.

    Parameters
    ----------
    origin : str
        The file name or other source of the YAML.
    path : str
        The dotted YAML path of the fault; empty for the whole entry.
    message : str
        What is wrong and how to fix it.

    Raises
    ------
    RegionError
        Always.
    """
    raise RegionError(f"{origin}: {path}: {message}" if path else f"{origin}: {message}")


def _mapping(value, origin: str, path: str, allowed: tuple[str, ...]) -> dict:
    """Check that a value is a mapping with only allowed keys.

    A `~hcwf.errors.RegionError` is raised (by `_fail`) when ``value`` is not a mapping or has a key that is not allowed.

    Parameters
    ----------
    value : object
        The YAML value.
    origin : str
        The file name, for error messages.
    path : str
        The YAML path of ``value``, for error messages.
    allowed : tuple of str
        The keys the mapping may have.

    Returns
    -------
    dict
        ``value`` itself.
    """
    if not isinstance(value, dict):
        _fail(origin, path, f"expected a mapping but found {value!r}. Fix: write 'key: value' lines")
    for key in value:
        if key not in allowed:
            _fail(origin, f"{path}.{key}" if path else str(key),
                  f"unknown key '{key}'. Fix: remove it or correct the spelling; allowed keys are {', '.join(allowed)}")
    return value


def _strings(value, origin: str, path: str) -> tuple[str, ...]:
    """Check that a value is a list of strings.

    A `~hcwf.errors.RegionError` is raised (by `_fail`) when it is not.

    Parameters
    ----------
    value : object
        The YAML value.
    origin : str
        The file name, for error messages.
    path : str
        The YAML path of ``value``, for error messages.

    Returns
    -------
    tuple of str
        The strings.
    """
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _fail(origin, path, f"expected a list of text but found {value!r}. Fix: write e.g. '[a, b]'")
    return tuple(value)


def _span(value, origin: str, path: str) -> tuple[float, float]:
    """Check that a value is a ``[low, high]`` pair of numbers with ``low < high``.

    A `~hcwf.errors.RegionError` is raised (by `_fail`) when it is not.

    Parameters
    ----------
    value : object
        The YAML value.
    origin : str
        The file name, for error messages.
    path : str
        The YAML path of ``value``, for error messages.

    Returns
    -------
    tuple of (float, float)
        The range.
    """
    ok = (isinstance(value, list) and len(value) == 2
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value) and value[0] < value[1])
    if not ok:
        _fail(origin, path, f"{value!r} is not a range. Fix: write [low, high] in degrees with low < high")
    return float(value[0]), float(value[1])


def _parse(text: str, origin: str, base: Path | None = None) -> Region:
    """Validate one region entry.

    Parameters
    ----------
    text : str
        The YAML text of the entry.
    origin : str
        The file name, for error messages.
    base : pathlib.Path, optional
        The folder of the entry's file, where relative ``units`` paths start.

    Returns
    -------
    Region
        The validated region.

    Raises
    ------
    RegionError
        The YAML is invalid, a key is unknown, missing or malformed, or a name it refers to does not exist.
    """
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
    units = _parse_units(data["units"], origin, base) if "units" in data else None
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
                                    "Fix: use the id of a file in hcwf/onset/registry")
    return Region(id=ident, title=data["title"], kind=kind, iso=iso, aliases=aliases, part_of=part_of,
                  lat=lat, lon=lon, units=units, observations_dataset=dataset, day_boundary=boundary,
                  climatology_years=years, default_definition=default, also_definitions=also,
                  notes=str(data.get("notes", "")))


def _parse_units(value, origin: str, base: Path | None) -> UnitsSpec:
    """Validate a ``units:`` block: a shapefile (with its id column) or a weights table, and optional extras.

    A `~hcwf.errors.RegionError` is raised (by `_fail`) when the block has an unknown key, a value that is not text, a wrong
    combination of keys, or is rejected by `hcwf.spatial.units.UnitsSpec`.

    Parameters
    ----------
    value : object
        The YAML value of the block.
    origin : str
        The file name, for error messages.
    base : pathlib.Path or None
        The folder where relative paths start; None means `REGIONS_DIR`.

    Returns
    -------
    hcwf.spatial.units.UnitsSpec
        The units specification.
    """
    block = _mapping(value, origin, "units", _UNITS_KEYS)
    for key, text in block.items():
        if not isinstance(text, str) or not text.strip():
            _fail(origin, f"units.{key}", f"{text!r} is not text. Fix: write a path or a column name")
    if ("shapefile" in block) == ("weights" in block):
        _fail(origin, "units", "needs exactly one of 'shapefile' and 'weights'. Fix: name the polygons' file or the "
                               "weights CSV")
    if "shapefile" in block and "unit_id_col" not in block:
        _fail(origin, "units.unit_id_col", "is required with a shapefile. Fix: name the column holding the unit ids")
    wrong = [k for k in (("weights", "centroids", "centroid_id_col") if "shapefile" in block
                         else ("unit_id_col", "unit_name_col", "parent_col")) if k in block]
    if wrong:
        _fail(origin, f"units.{wrong[0]}", "does not go with this source. Fix: shapefile takes unit_*_col and "
                                           "parent_col, weights takes centroids and centroid_id_col")
    try:
        return UnitsSpec(base=base or REGIONS_DIR, **block)
    except UnitsSpecError as exc:
        _fail(origin, "units", str(exc))


def region_ids() -> list[str]:
    """Return the ids of the bundled regions, sorted.

    Returns
    -------
    list of str
        The stems of the YAML files of `REGIONS_DIR`.
    """
    return sorted(p.stem for p in REGIONS_DIR.glob("*.yaml"))


@functools.lru_cache(maxsize=None)
def _index() -> dict[str, Region]:
    """Load every region, keyed by lower-cased id and alias.

    The result is cached for the process. A `~hcwf.errors.RegionError` is raised (by `_fail`) when a file is invalid, its id differs
    from its file name, or an alias is used twice.

    Returns
    -------
    dict of str to Region
        The regions by lower-cased id and alias.
    """
    index: dict[str, Region] = {}
    for file in sorted(REGIONS_DIR.glob("*.yaml")):
        region = _parse(file.read_text(), file.name, file.parent)
        if region.id != file.stem:
            _fail(file.name, "id", f"'{region.id}' differs from the file name. Fix: rename the file to {region.id}.yaml")
        for name in (region.id, *region.aliases):
            key = name.lower()
            if key in index and index[key].id != region.id:
                _fail(file.name, "aliases", f"'{name}' is already used by {index[key].id}. Fix: remove the alias")
            index[key] = region
    return index


def load_region(name: str) -> Region:
    """Find a region by id or alias, ignoring case.

    Parameters
    ----------
    name : str
        The id or an alias of the region.

    Returns
    -------
    Region
        The region.

    Raises
    ------
    RegionError
        No region has that id or alias.
    """
    try:
        return _index()[name.strip().lower()]
    except KeyError:
        raise RegionError(
            f"unknown region '{name}' (known: {', '.join(region_ids())}). "
            "Fix: use a region id or alias, or add hcwf/onset/regions/<id>.yaml") from None


def ancestors(region: Region) -> list[Region]:
    """Return the regions containing this one through ``part_of``, nearest first; cycles are cut.

    Parameters
    ----------
    region : Region
        The region whose containers are wanted.

    Returns
    -------
    list of Region
        The ancestors, breadth first, each once.
    """
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
    """Return the default and the other recommended definition ids of a region or country.

    The nearest area in the ``part_of`` chain that names a default wins; the others are the union of every ``also``
    along the chain, without the default.

    Parameters
    ----------
    name : str
        The id or alias of the region.

    Returns
    -------
    tuple of (str or None, list of str)
        The default definition id (None if no area names one) and the other recommended ids.
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

    There is no warning if the definition lists no regions, or if the region or any region containing it is listed.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition.
    name : str
        The id or alias of the region.

    Returns
    -------
    str or None
        The warning text, or None.
    """
    if not defn.regions:
        return None
    region = load_region(name)
    covered = {r.id for r in (region, *ancestors(region))}
    if covered & set(defn.regions):
        return None
    return (f"definition '{defn.id}' lists regions {', '.join(defn.regions)}, which do not include "
            f"{region.title} ({', '.join(sorted(covered))}); check it applies there")


def box(name: str) -> tuple[float, float, float, float]:
    """Return the ``domain`` box of a region; the name matches by id or alias, ignoring case.

    A `~hcwf.errors.RegionError` is raised (by `_fail`) when the region has no domain box.

    Parameters
    ----------
    name : str
        The id or alias of the region.

    Returns
    -------
    tuple of (float, float, float, float)
        The south and north latitude and the west and east longitude, in degrees.
    """
    region = load_region(str(name).strip())
    if region.lat is None or region.lon is None:
        _fail(f"{region.id}.yaml", "domain", "the region has no domain (lat/lon box). Fix: add "
                                              "'domain: {lat: [low, high], lon: [low, high]}'")
    return region.lat[0], region.lat[1], region.lon[0], region.lon[1]


def polygon_boundary(da):
    """Return the India CMZ polygon that matches the grid spacing of an array (1, 2 or 4 degrees).

    Parameters
    ----------
    da : xarray.DataArray
        An array with a ``lat`` coordinate of at least two values; the spacing is read from the first two.

    Returns
    -------
    tuple of (numpy.ndarray or None, numpy.ndarray or None)
        The latitudes and longitudes of the polygon vertices in degrees, or ``(None, None)`` for another spacing.
    """
    polygon_lat, polygon_lon = None, None
    lat = da.lat.values
    lat_diff = abs(lat[1] - lat[0])

    if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
        polygon_lon = np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83])
        polygon_lat = np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17])
        log.debug("Using 2-degree CMZ polygon coordinates")

    elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
        polygon_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        polygon_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        log.debug("Using 4-degree CMZ polygon coordinates")

    elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
        polygon_lon = np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74])
        polygon_lat = np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18])
        log.debug("Using 1-degree CMZ polygon coordinates")

    return polygon_lat, polygon_lon


def units_spec(name: str) -> UnitsSpec | None:
    """Return the ``units`` block of the nearest area in the ``part_of`` chain that declares one.

    Parameters
    ----------
    name : str
        The id or alias of the region.

    Returns
    -------
    hcwf.spatial.units.UnitsSpec or None
        The specification, or None if no area in the chain declares one.
    """
    region = load_region(name)
    return next((r.units for r in (region, *ancestors(region)) if r.units), None)
