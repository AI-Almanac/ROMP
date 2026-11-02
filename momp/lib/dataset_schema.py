"""The schema of a dataset description: where observations or forecasts are and how to read them.

ROMP ships no datasets; you describe your own, either inline in the run config::

    observations: {path: /data/obs, file_pattern: "{}.nc", variable: rain, units: mm,
                   day_boundary: "00 UTC, labelled by start"}
    models:
      - {name: AIFS, path: /data/aifs, variable: tp, units: m, lead_label: start,
         dims: {init_time: time, step: lead}}

or once in a dataset file of yours (listed under ``dataset_files:`` or, as a folder of
<id>.yaml files, ``catalog_dirs:`` in the run config) and referenced by id::

    observations: {dataset: my_obs}
    models: [{dataset: my_aifs, name: AIFS}]

Keys of an entry: id (files only), title, kind (observations or forecast; implied inline),
name (short name used in output files), path, file_pattern (default "{}.nc", {} is the year),
variable, units (mm or m; detected from the file when left out), dims (names of the
file's dimensions: time/lat/lon for observations, init_time/step/member/lat/lon for
forecasts), day_boundary (observations: when the day starts and which date labels it, e.g.
"00 UTC, labelled by start") or lead_label (forecast: "start" = lead k is dated init + k - 1,
"end" = init + k), accumulation (forecast: daily, since_init or auto), daily_boundary (forecast: cut the
days at this UTC clock time, e.g. "03 UTC", instead of 24 h from the init), members (ensemble size), notes. ``path`` may use ``~`` and ``${ENV_VAR}``
or ``${ENV_VAR:-default}``.

Instead of ``file_pattern`` a dataset may list its files in ``paths`` (file names or globs, relative to ``path`` when
that is given), with any number of files per year (live operational files hold one init each); ``years`` keeps the
files of those years (a list, or "2010:2013"). A file's year is the first 19xx or 20xx in its name
(`file_year`). ``unit_cvt`` (a number, the multiplier to mm) replaces ``units`` for files that declare none; a
file that declares units is converted by them. `momp-run init --dataset-template obs` prints a blank entry.
"""
from __future__ import annotations

import difflib
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from momp.lib.assertion import ROMPConfigError

KINDS = ("observations", "forecast")
DATASET_KEYS = ("id", "title", "kind", "name", "path", "paths", "years", "file_pattern", "variable", "units",
                "unit_cvt", "dims", "day_boundary", "lead_label", "accumulation", "daily_boundary", "members", "notes")
_ID_RE = re.compile(r"^[a-z0-9_]+$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
OBS_DIMS = ("time", "lat", "lon", "unit")
FORECAST_DIMS = ("init_time", "step", "member", "lat", "lon", "unit")
FILE_YEAR_RE = re.compile(r"(19|20)\d{2}")
"""The year of a file named in ``paths``: the first 19xx or 20xx anywhere in its name (as onset_blending reads it)."""
UNIT_FACTORS = {"mm": None, "m": 1000}
"""Factor taking a ``units`` value to mm (None: already mm)."""

TEMPLATES = {
    "obs": """\
# One observation dataset. Save in a file listed under dataset_files: in your run config,
# or put the same keys inline under observations:
id: my_obs                     # lowercase letters, digits, underscores (files only)
title: My daily rainfall observations
kind: observations
name: OBS                      # short name used in output files
path: /path/to/observations    # folder of the yearly files; ~ and ${ENV_VAR} work
file_pattern: "{}.nc"          # {} is the year; quote it
variable: rainfall             # rainfall variable in the files
units: mm                      # mm or m; detected from the file if left out
dims: {time: time, lat: lat, lon: lon}   # the file's names, if not the usual ones
day_boundary: 00 UTC, labelled by start  # when the day starts, which date labels it
notes: free text
""",
    "forecast": """\
# One forecast dataset. Save in a file listed under dataset_files: in your run config,
# or put the same keys inline as an item of models:
id: my_model                   # lowercase letters, digits, underscores (files only)
title: My model's daily rainfall forecasts
kind: forecast
name: MODEL                    # model name used in output files (no spaces)
path: /path/to/forecasts       # folder of the yearly files; ~ and ${ENV_VAR} work
file_pattern: "{}.nc"          # {} is the year; quote it
variable: tp                   # rainfall variable in the files
units: m                       # mm or m; detected from the file if left out
dims: {init_time: time, step: step}      # add member: number for an ensemble
lead_label: start              # start: lead k is dated init + k - 1; end: init + k
accumulation: auto             # daily, since_init or auto (detected)
# daily_boundary: 03 UTC       # cut days at this UTC time instead of 24 h from the init (sub-daily steps)
members: 1                     # ensemble size, if an ensemble
notes: free text
""",
}
"""Blank dataset entries, as printed by `momp-run init --dataset-template`."""


class DatasetError(ROMPConfigError):
    """A malformed dataset entry or an unknown dataset; the message names the file or key and the fix."""


@dataclass(frozen=True)
class Dataset:
    """One dataset description. unit_cvt is the multiplier to mm (None for none); path is not yet expanded."""

    kind: str
    path: str
    file_pattern: str
    variable: str
    unit_cvt: float | None
    units: str | None = None
    accumulation: str = "auto"
    daily_boundary: str | None = None
    id: str | None = None
    title: str = ""
    name: str | None = None
    dims: dict | None = None
    day_boundary: str | None = None
    lead_label: str | None = None
    members: int | None = None
    notes: str = ""
    source: str = ""
    paths: tuple | None = None
    years: tuple | None = None

    @property
    def short_name(self) -> str:
        """The name used for output files: ``name`` if the entry has one, else its id."""
        return self.name or self.id or "dataset"

    @property
    def expanded_path(self) -> str:
        """The path with ``~`` and ${ENV_VAR} expanded; raises DatasetError if a variable is unset."""
        return expand_path(self.path, origin=self.source or "dataset")

    def files(self) -> list[Path]:
        """The files of a dataset given by ``paths`` (see `resolve_paths`); raises DatasetError for any other dataset."""
        if self.paths is None:
            raise DatasetError(f"{self.source or 'dataset'}: the dataset has no 'paths'")
        return resolve_paths(self.paths, self.expanded_path if self.path else None, self.years,
                             origin=self.source or "dataset")


def suggest(name: str, candidates) -> str:
    """Return ' Did you mean "x"?' for the closest candidate to name, or an empty string."""
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=1, cutoff=0.6)
    return f' Did you mean "{close[0]}"?' if close else ""


def expand_env(text: str, origin: str = "") -> str:
    """Replace ${VAR} and ${VAR:-default} in text by the environment; an unset VAR without default is an error."""
    def sub(match):
        """Value of one ${...} reference."""
        value = os.environ.get(match.group(1))
        if value:
            return value
        if match.group(2) is not None:
            return match.group(2)
        where = f" in {origin}" if origin else ""
        raise DatasetError(f"environment variable {match.group(1)} is not set{where}. "
                           f"Fix: export {match.group(1)}=... or write ${{{match.group(1)}:-default}}")
    return _ENV_RE.sub(sub, text)


def expand_path(text: str, origin: str = "") -> str:
    """Expand ${ENV_VAR} references and a leading ``~`` in a path string."""
    return str(Path(expand_env(str(text), origin)).expanduser())


def file_year(path) -> int | None:
    """The year a file holds: the first 19xx or 20xx in its name, or None."""
    match = FILE_YEAR_RE.search(Path(str(path)).name)
    return int(match.group()) if match else None


def parse_years(value, origin: str = "years") -> tuple[int, ...]:
    """A years filter: a list of years, one year, or text such as "2010:2013" (both ends included)."""
    try:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, int):
            return (value,)
        if isinstance(value, str):
            parts = [p for p in re.split(r"\s*[:\-]\s*", value.strip()) if p]
            if len(parts) == 2:
                first, last = int(parts[0]), int(parts[1])
                if last < first:
                    raise ValueError
                return tuple(range(first, last + 1))
            if len(parts) == 1:
                return (int(parts[0]),)
            raise ValueError
        years = tuple(int(y) for y in value)
        if not years or any(isinstance(y, bool) for y in value):
            raise ValueError
        return years
    except (TypeError, ValueError):
        raise DatasetError(f"{origin}: expected a list of years or text such as '2010:2013' but found {value!r}") from None


def resolve_paths(paths, base=None, years=None, origin: str = "paths") -> list[Path]:
    """The files named by ``paths`` (names or globs, ``~`` and ${ENV_VAR} expanded, relative ones under `base`).

    The result is sorted by (year in the name, path), as onset_blending lists files, and holds each file once;
    `years` keeps only the files of those years (a file whose name has no year is dropped then). A name that is
    not a glob must exist; at least one file must remain.
    """
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    found: dict[str, Path] = {}
    missing = []
    for item in items:
        text = expand_path(str(item), origin)
        path = Path(text)
        if not path.is_absolute() and base:
            path = Path(base) / path
        if any(c in text for c in "*?["):
            hits = sorted(Path(p) for p in glob.glob(str(path)) if Path(p).is_file())
            if not hits:
                missing.append(str(path))
        elif path.is_file():
            hits = [path]
        else:
            hits = []
            missing.append(str(path))
        for hit in hits:
            found.setdefault(str(hit), hit)
    if missing:
        raise DatasetError(f"{origin}: no file for {', '.join(missing)}. Fix: correct paths")
    files = list(found.values())
    if years is not None:
        wanted = set(parse_years(years, f"{origin} years"))
        files = [f for f in files if file_year(f) in wanted]
    if not files:
        raise DatasetError(f"{origin}: no file matches paths {items!r}" + (f" for the years {sorted(wanted)}" if years is not None else ""))
    return sorted(files, key=lambda f: (file_year(f) or 0, str(f)))


def unit_factor(units: str, origin: str) -> float | None:
    """The multiplier to mm for a units name; unknown names are an error that asks for unit_cvt."""
    if units not in UNIT_FACTORS:
        raise DatasetError(f"{origin}: units {units!r} is not known (known: {', '.join(UNIT_FACTORS)}). "
                           "Fix: convert the data to mm or m first and set units accordingly")
    return UNIT_FACTORS[units]


def _fail(origin: str, message: str):
    """Raise a DatasetError naming where the entry is."""
    raise DatasetError(f"{origin}: {message}")


def _text(data: dict, key: str, origin: str, required: bool = True):
    """Return data[key] as non-empty text; a missing key is an error when required, else None."""
    value = data.get(key)
    if value is None:
        if required:
            _fail(origin, f"missing required key '{key}'. Fix: add '{key}:'")
        return None
    if not isinstance(value, str) or not value.strip():
        _fail(origin, f"{key}: expected non-empty text but found {value!r}")
    return value


def parse_dataset(data, origin: str, kind: str | None = None, require_id: bool = True) -> Dataset:
    """Validate one dataset entry given as a mapping; origin names the file or key in error messages.

    kind forces the kind (inline entries); require_id is False for inline entries.
    """
    from momp.onset.ingest import parse_day_boundary

    if not isinstance(data, dict):
        _fail(origin, f"expected a mapping of keys but found {data!r}. Fix: write 'key: value' lines")
    for key in data:
        if key not in DATASET_KEYS:
            _fail(origin, f"unknown key '{key}'.{suggest(key, DATASET_KEYS)} Allowed keys: {', '.join(DATASET_KEYS)}")
    ident = _text(data, "id", origin, required=require_id)
    if ident is not None and not _ID_RE.match(ident):
        _fail(origin, f"id {ident!r} is not valid. Fix: use lowercase letters, digits and underscores")
    kind = kind or data.get("kind")
    if kind not in KINDS:
        _fail(origin, f"kind {kind!r} is not a kind. Fix: write 'observations' or 'forecast'")
    if data.get("kind") not in (None, kind):
        _fail(origin, f"kind is {data['kind']!r} but a {kind} dataset is needed here")
    paths = data.get("paths")
    if paths is not None:
        paths = (paths,) if isinstance(paths, str) else paths
        if not isinstance(paths, (list, tuple)) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
            _fail(origin, f"paths: expected a list of file names or globs but found {data.get('paths')!r}")
        if "file_pattern" in data:
            _fail(origin, "file_pattern and paths both say which files to read. Fix: keep one of them")
        paths = tuple(paths)
    path = _text(data, "path", origin, required=paths is None) or ""
    years = parse_years(data["years"], f"{origin}: years") if data.get("years") is not None else None
    if years is not None and paths is None:
        _fail(origin, "years needs paths: a dataset read by file_pattern is read for the years the run asks for")
    variable = _text(data, "variable", origin)
    pattern = _text(data, "file_pattern", origin, required=False) or "{}.nc"
    if "{}" not in pattern:
        _fail(origin, f"file_pattern {pattern!r} has no '{{}}' for the year. Fix: write e.g. '{{}}.nc' (quoted)")
    units = data.get("units")
    cvt = unit_factor(units, origin) if units is not None else None
    if data.get("unit_cvt") is not None:
        factor = data["unit_cvt"]
        if units is not None:
            _fail(origin, "units and unit_cvt both say how to convert to mm. Fix: keep one of them")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 < factor < float("inf"):
            _fail(origin, f"unit_cvt: expected a positive number (the multiplier to mm) but found {factor!r}")
        cvt = float(factor)
    dims = data.get("dims")
    allowed = OBS_DIMS if kind == "observations" else FORECAST_DIMS
    if dims is not None:
        if not isinstance(dims, dict) or not all(isinstance(v, str) and v for v in dims.values()):
            _fail(origin, f"dims: expected a mapping of names to the file's dimension names but found {dims!r}")
        for key in dims:
            if key not in allowed:
                _fail(origin, f"dims.{key}: unknown name.{suggest(key, allowed)} For {kind} use: {', '.join(allowed)}")
    boundary = label = members = daily_boundary = None
    accumulation = "auto"
    if kind == "observations":
        boundary = _text(data, "day_boundary", origin)
        try:
            parse_day_boundary(boundary)
        except ValueError as exc:
            _fail(origin, f"day_boundary: {exc}")
        for key in ("lead_label", "members", "daily_boundary"):
            if key in data:
                _fail(origin, f"{key} is only for forecast datasets. Fix: remove it")
    else:
        label = data.get("lead_label")
        if label not in ("start", "end"):
            _fail(origin, f"lead_label {label!r} must be 'start' (lead k is dated init + k - 1) or 'end' "
                          "(init + k)")
        if "day_boundary" in data:
            _fail(origin, "day_boundary is only for observation datasets. Fix: remove it")
        accumulation = data.get("accumulation", "auto")
        if accumulation not in ("daily", "since_init", "auto"):
            _fail(origin, f"accumulation {accumulation!r} must be 'daily', 'since_init' or 'auto'")
        daily_boundary = data.get("daily_boundary")
        if daily_boundary is not None:
            try:
                from momp.onset.ingest import _parse_clock

                _parse_clock(str(daily_boundary))
            except ValueError as exc:
                _fail(origin, f"daily_boundary: {exc}. Fix: write a UTC clock time such as '03 UTC'")
            daily_boundary = str(daily_boundary)
        members = data.get("members")
        if members is not None and (isinstance(members, bool) or not isinstance(members, int) or members < 1):
            _fail(origin, f"members: expected a positive whole number but found {members!r}")
    return Dataset(kind=kind, path=path, file_pattern=pattern, variable=variable, unit_cvt=cvt, units=units, accumulation=accumulation,
                   daily_boundary=daily_boundary, id=ident, title=str(data.get("title") or ""), name=_text(data, "name", origin, required=False),
                   dims=dims, day_boundary=boundary, lead_label=label, members=members,
                   notes=str(data.get("notes") or "").strip(), source=origin, paths=paths, years=years)


def _read_entries(file: Path) -> list[Dataset]:
    """The dataset entries of one YAML file: a single entry, or a list of entries."""
    try:
        data = yaml.safe_load(file.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(f"{file}: cannot read as YAML: {exc}") from None
    items = data if isinstance(data, list) else [data]
    return [parse_dataset(item, f"{file}" + (f"[{i}]" if isinstance(data, list) else ""))
            for i, item in enumerate(items)]


def load_datasets(dataset_files=(), catalog_dirs=()) -> dict[str, Dataset]:
    """The datasets described in dataset_files (YAML files) and catalog_dirs (folders of *.yaml), keyed by id.

    Later files win on a repeated id. Paths may use ~ and ${ENV_VAR}.
    """
    files = [Path(expand_path(f)) for f in dataset_files or ()]
    for directory in catalog_dirs or ():
        folder = Path(expand_path(directory))
        if not folder.is_dir():
            raise DatasetError(f"dataset folder not found: {folder}. Fix: correct catalog_dirs")
        files.extend(sorted(folder.glob("*.yaml")))
    found: dict[str, Dataset] = {}
    for file in files:
        if not file.is_file():
            raise DatasetError(f"dataset file not found: {file}. Fix: correct dataset_files")
        for entry in _read_entries(file):
            found[entry.id] = entry
    return found


def get_dataset(ident: str, datasets: dict[str, Dataset], kind: str | None = None) -> Dataset:
    """Find a dataset by id among the loaded ones; unknown ids and the wrong kind are errors that say what to do."""
    if ident not in datasets:
        known = ", ".join(sorted(datasets)) or "none: no dataset_files or catalog_dirs are set"
        raise DatasetError(f"unknown dataset '{ident}'.{suggest(ident, datasets)} Known datasets: {known}. "
                           "Fix: list the file under dataset_files: (or its folder under catalog_dirs:) "
                           "or describe the dataset inline")
    entry = datasets[ident]
    if kind and entry.kind != kind:
        raise DatasetError(f"dataset '{ident}' is a {entry.kind} dataset but a {kind} dataset is needed here")
    return entry
