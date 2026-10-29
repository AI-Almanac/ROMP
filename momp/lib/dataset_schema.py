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
"end" = init + k), accumulation (forecast: daily, since_init or auto), members (ensemble size), notes. ``path`` may use ``~`` and ``${ENV_VAR}``
or ``${ENV_VAR:-default}``. `momp-run init --dataset-template obs` prints a blank entry.
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from momp.lib.assertion import ROMPConfigError

KINDS = ("observations", "forecast")
DATASET_KEYS = ("id", "title", "kind", "name", "path", "file_pattern", "variable", "units", "dims",
                "day_boundary", "lead_label", "accumulation", "members", "notes")
_ID_RE = re.compile(r"^[a-z0-9_]+$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
OBS_DIMS = ("time", "lat", "lon")
FORECAST_DIMS = ("init_time", "step", "member", "lat", "lon")
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
    id: str | None = None
    title: str = ""
    name: str | None = None
    dims: dict | None = None
    day_boundary: str | None = None
    lead_label: str | None = None
    members: int | None = None
    notes: str = ""
    source: str = ""

    @property
    def short_name(self) -> str:
        """The name used for output files: ``name`` if the entry has one, else its id."""
        return self.name or self.id or "dataset"

    @property
    def expanded_path(self) -> str:
        """The path with ``~`` and ${ENV_VAR} expanded; raises DatasetError if a variable is unset."""
        return expand_path(self.path, origin=self.source or "dataset")


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
    path = _text(data, "path", origin)
    variable = _text(data, "variable", origin)
    pattern = _text(data, "file_pattern", origin, required=False) or "{}.nc"
    if "{}" not in pattern:
        _fail(origin, f"file_pattern {pattern!r} has no '{{}}' for the year. Fix: write e.g. '{{}}.nc' (quoted)")
    units = data.get("units")
    cvt = unit_factor(units, origin) if units is not None else None
    dims = data.get("dims")
    allowed = OBS_DIMS if kind == "observations" else FORECAST_DIMS
    if dims is not None:
        if not isinstance(dims, dict) or not all(isinstance(v, str) and v for v in dims.values()):
            _fail(origin, f"dims: expected a mapping of names to the file's dimension names but found {dims!r}")
        for key in dims:
            if key not in allowed:
                _fail(origin, f"dims.{key}: unknown name.{suggest(key, allowed)} For {kind} use: {', '.join(allowed)}")
    boundary = label = members = None
    accumulation = "auto"
    if kind == "observations":
        boundary = _text(data, "day_boundary", origin)
        try:
            parse_day_boundary(boundary)
        except ValueError as exc:
            _fail(origin, f"day_boundary: {exc}")
        for key in ("lead_label", "members"):
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
        members = data.get("members")
        if members is not None and (isinstance(members, bool) or not isinstance(members, int) or members < 1):
            _fail(origin, f"members: expected a positive whole number but found {members!r}")
    return Dataset(kind=kind, path=path, file_pattern=pattern, variable=variable, unit_cvt=cvt, units=units, accumulation=accumulation,
                   id=ident, title=str(data.get("title") or ""), name=_text(data, "name", origin, required=False),
                   dims=dims, day_boundary=boundary, lead_label=label, members=members,
                   notes=str(data.get("notes") or "").strip(), source=origin)


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
