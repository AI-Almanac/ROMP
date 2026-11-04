"""The schema of a dataset description: where observations or forecasts are and how to read them.

HCWF ships no datasets; you describe your own, either inline in the run config::

    observations: {path: /path/to/observations, file_pattern: "{}.nc", variable: rain, units: mm,
                   day_boundary: "00 UTC, labelled by start"}
    models:
      - {name: <model>, path: /path/to/forecasts, variable: tp, units: m, lead_label: start,
         dims: {init_time: time, step: lead}}

or once in a dataset file of yours (listed under ``dataset_files:`` or, as a folder of
``<id>.yaml`` files, ``catalog_dirs:`` in the run config) and referenced by id::

    observations: {dataset: my_obs}
    models: [{dataset: my_forecast, name: <model>}]

Keys of an entry: ``id`` (files only), ``title``, ``kind`` (``observations`` or ``forecast``; implied inline),
``name`` (short name used in output files), ``path``, ``file_pattern`` (default ``"{}.nc"``, ``{}`` is the year),
``variable``, ``units`` (``mm`` or ``m``; detected from the file when left out), ``dims`` (names of the
file's dimensions: ``time``/``lat``/``lon`` for observations, ``init_time``/``step``/``member``/``lat``/``lon`` for
forecasts), ``day_boundary`` (observations: when the day starts and which date labels it, e.g.
``"00 UTC, labelled by start"``) or ``lead_label`` (forecast: ``"start"`` = lead k is dated init + k - 1,
``"end"`` = init + k), ``accumulation`` (forecast: ``daily``, ``since_init`` or ``auto``), ``daily_boundary``
(forecast: cut the days at this UTC clock time, e.g. ``"03 UTC"``, instead of 24 h from the init), ``members``
(ensemble size), ``notes``. ``path`` may use ``~`` and ``${ENV_VAR}`` or ``${ENV_VAR:-default}``.

Instead of ``file_pattern`` a dataset may list its files in ``paths`` (file names or globs, relative to ``path`` when
that is given), with any number of files per year (live operational files hold one init each); ``years`` keeps the
files of those years (a list, or ``"2010:2013"``). A file's year is the first ``19xx`` or ``20xx`` in its name
(`file_year`). ``unit_cvt`` (a number, the multiplier to mm) replaces ``units`` for files that declare none; a
file that declares units is converted by them. ``hcwf-run init --dataset-template obs`` prints a blank entry.

The module holds the `Dataset` record, its parser (`parse_dataset`, `load_datasets`, `get_dataset`) and the helpers
that read a file year (`file_year`), a years filter (`parse_years`, `parse_year_list`), file lists
(`resolve_paths`) and paths with environment variables (`expand_path`).
"""
from __future__ import annotations

import difflib
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from hcwf.errors import DatasetError

KINDS = ("observations", "forecast")
"""The values of ``kind``."""
DATASET_KEYS = ("id", "title", "kind", "name", "path", "paths", "years", "file_pattern", "variable", "units",
                "unit_cvt", "dims", "day_boundary", "lead_label", "accumulation", "daily_boundary", "members", "notes")
"""The keys a dataset entry may hold."""
_ID_RE = re.compile(r"^[a-z0-9_]+$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
FORECAST_DIMS = ("init_time", "step", "member", "lat", "lon", "unit")
"""The dimension names a forecast dataset may rename in ``dims`` (``unit``: the dimension of evaluation units)."""
OBS_DIMS = ("time",) + tuple(d for d in FORECAST_DIMS if d not in ("init_time", "step", "member"))
"""The dimension names an observation dataset may rename in ``dims``."""
ALL_DIMS = FORECAST_DIMS + ("time",)
"""Every dimension name some ``dims`` mapping may rename (the union of `OBS_DIMS` and `FORECAST_DIMS`)."""
_YEAR_ANYWHERE = re.compile(r"(19|20)\d{2}")
_YEAR_WHOLE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")

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
"""Blank dataset entries, as printed by ``hcwf-run init --dataset-template``."""


@dataclass(frozen=True)
class Dataset:
    """One dataset description, validated by `parse_dataset`.

    ``path`` is not yet expanded (`Dataset.expanded_path` expands it).
    """

    kind: str
    """``"observations"`` or ``"forecast"``."""
    path: str
    """The folder of the files, as written (``~`` and ``${ENV_VAR}`` not expanded); ``""`` when ``paths`` gives them."""
    file_pattern: str
    """The file name pattern, ``{}`` standing for the year."""
    variable: str
    """The rainfall variable in the files."""
    unit_cvt: float | None
    """The multiplier to mm (None for none)."""
    units: str | None = None
    """The units name (``"mm"``, ``"m"``, ...), or None to detect it from the files."""
    accumulation: str = "auto"
    """Forecasts: ``"daily"``, ``"since_init"`` or ``"auto"``."""
    daily_boundary: str | None = None
    """Forecasts: the UTC clock time at which days are cut, or None for 24 h from the init."""
    id: str | None = None
    """The id of the entry in a dataset file (None inline)."""
    title: str = ""
    """A free title."""
    name: str | None = None
    """The short name used in output files."""
    dims: dict | None = None
    """Canonical dimension name to the name in the files, or None to infer them."""
    day_boundary: str | None = None
    """Observations: when the day starts and which date labels it."""
    lead_label: str | None = None
    """Forecasts: ``"start"`` (lead k is dated init + k - 1) or ``"end"`` (init + k)."""
    members: int | None = None
    """Forecasts: the ensemble size."""
    notes: str = ""
    """Free text."""
    source: str = ""
    """Where the entry was read from (file name or config key), for error messages."""
    paths: tuple | None = None
    """File names or globs that replace ``file_pattern``, or None."""
    years: tuple | None = None
    """The years of ``paths`` to keep, or None for all."""

    @property
    def short_name(self) -> str:
        """The name used for output files: ``name`` if the entry has one, else its id."""
        return self.name or self.id or "dataset"

    @property
    def expanded_path(self) -> str:
        """The path with ``~`` and ``${ENV_VAR}`` expanded (`~hcwf.errors.DatasetError` if a variable is unset and has no default)."""
        return expand_path(self.path, origin=self.source or "dataset")

    def files(self) -> list[Path]:
        """Return the files of a dataset given by ``paths`` (see `resolve_paths`).

        Returns
        -------
        list of pathlib.Path
            The files, sorted by year and path, each once.

        Raises
        ------
        DatasetError
            The dataset has no ``paths``, or `resolve_paths` finds no file.
        """
        if self.paths is None:
            raise DatasetError(f"{self.source or 'dataset'}: the dataset has no 'paths'")
        return resolve_paths(self.paths, self.expanded_path if self.path else None, self.years,
                             origin=self.source or "dataset")


def suggest(name: str, candidates) -> str:
    """Return a spelling hint for the candidate closest to a name.

    Parameters
    ----------
    name : str
        The unknown name.
    candidates : iterable
        The known names.

    Returns
    -------
    str
        `` Did you mean "x"?`` (with a leading space) for the closest candidate, or an empty string when none is close
        (similarity of at least 0.6).
    """
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=1, cutoff=0.6)
    return f' Did you mean "{close[0]}"?' if close else ""


def expand_env(text: str, origin: str = "") -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in a text by the environment.

    An empty variable counts as unset; an unset one without a default raises `~hcwf.errors.DatasetError` (from the replacement
    function, named with ``origin``).

    Parameters
    ----------
    text : str
        The text with the references.
    origin : str, default ""
        Where the text comes from, named in the error.

    Returns
    -------
    str
        ``text`` with every reference replaced.
    """
    def sub(match):
        """Return the value of one ``${...}`` reference."""
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
    """Expand ``${ENV_VAR}`` references and a leading ``~`` in a path string.

    An unset variable without a default raises `~hcwf.errors.DatasetError` (see `expand_env`).

    Parameters
    ----------
    text : str
        The path as written.
    origin : str, default ""
        Where the path comes from, named in the error.

    Returns
    -------
    str
        The expanded path.
    """
    return str(Path(expand_env(str(text), origin)).expanduser())


def year_match(name: str, whole_number: bool = False):
    """Find the year in a file name.

    By default the year is the first ``19xx`` or ``20xx`` anywhere in the name (as onset_blending reads it, so
    ``20200101.nc`` holds 2020); with ``whole_number``, the first ``18xx``, ``19xx`` or ``20xx`` that is not inside a
    longer run of digits (the year of a ``{}`` file pattern).

    Parameters
    ----------
    name : str
        The file name.
    whole_number : bool, default False
        Look for a year that is a whole number instead of anywhere.

    Returns
    -------
    re.Match or None
        The match (``group(0)`` is the year's text and ``span()`` its place), or None.
    """
    return (_YEAR_WHOLE if whole_number else _YEAR_ANYWHERE).search(name)


def file_year(path, whole_number: bool = False) -> int | None:
    """Return the year a file holds (see `year_match`).

    Parameters
    ----------
    path : str or pathlib.Path
        The file; only its name is read.
    whole_number : bool, default False
        Passed to `year_match`.

    Returns
    -------
    int or None
        The year, or None when the name has none.
    """
    match = year_match(Path(str(path)).name, whole_number)
    return int(match.group(0)) if match else None


def parse_years(value, origin: str = "years") -> tuple[int, ...]:
    """Parse a years filter.

    The filter is a list of years, one year, or text such as ``"2010:2013"`` or ``"2000:2010,2015"`` (ends included;
    ``..``, ``-`` and ``to`` also separate the ends).

    Parameters
    ----------
    value : int or str or sequence of int
        The filter. A bool is not a year.
    origin : str, default "years"
        Where the filter comes from, named in the error.

    Returns
    -------
    tuple of int
        The years in the order given (not sorted, not de-duplicated).

    Raises
    ------
    DatasetError
        The value is not a non-empty list of years or such a text, or a range runs backwards.
    ValueError
        Never escapes: it is raised inside the function and turned into the `~hcwf.errors.DatasetError`.
    """
    try:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, int):
            return (value,)
        if isinstance(value, str):
            years: list[int] = []
            for item in (p.strip() for p in value.split(",")):   # '2000:2010,2015': ranges and single years
                if not item:
                    continue
                parts = [p for p in re.split(r"\s*(?::|\.\.|-|to)\s*", item) if p]
                if len(parts) == 2:
                    first, last = int(parts[0]), int(parts[1])
                    if last < first:
                        raise ValueError
                    years.extend(range(first, last + 1))
                elif len(parts) == 1:
                    years.append(int(parts[0]))
                else:
                    raise ValueError
            if not years:
                raise ValueError
            return tuple(years)
        years = tuple(int(y) for y in value)
        if not years or any(isinstance(y, bool) for y in value):
            raise ValueError
        return years
    except (TypeError, ValueError):
        raise DatasetError(f"{origin}: expected a list of years or text such as '2010:2013' but found {value!r}") from None


def is_zarr(path):
    """Tell whether a path is a zarr store.

    A store is a name ending in ``.zarr``, or a directory holding ``.zarray``, ``.zgroup`` or ``zarr.json``.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to test.

    Returns
    -------
    bool
        True for a zarr store.
    """
    text = str(path).rstrip("/")
    if text.endswith(".zarr"):
        return True
    folder = Path(text)
    return folder.is_dir() and any((folder / name).exists() for name in ("zarr.json", ".zgroup", ".zarray"))


def parse_year_list(value, origin: str = "years") -> tuple[int, ...]:
    """Parse a years setting into sorted distinct years.

    The setting is a year, a list of years or of texts such as ``"2000:2010,2015"`` (or one such text); this is the one
    parser of ``blend.years``, ``evaluation.years`` and ``evaluation.climatology_years``. An item that is not a year
    or a range raises `~hcwf.errors.DatasetError` (see `parse_years`).

    Parameters
    ----------
    value : int or str or sequence of (int or str)
        The setting. An empty list or tuple gives no years.
    origin : str, default "years"
        Where the setting comes from, named in the error.

    Returns
    -------
    tuple of int
        The years, ascending, each once.

    """
    if isinstance(value, (list, tuple)) and not value:
        return ()
    items = value if isinstance(value, (list, tuple)) and any(isinstance(item, str) for item in value) else [value]
    return tuple(sorted({year for item in items for year in parse_years(item, origin)}))


def resolve_paths(paths, base=None, years=None, origin: str = "paths") -> list[Path]:
    """List the files named by ``paths`` (names or globs, ``~`` and ``${ENV_VAR}`` expanded).

    A zarr store (a directory, see `is_zarr`) counts as a file: it is one source holding every year.

    The result is sorted by (year in the name, path), as onset_blending lists files, and holds each file once;
    ``years`` keeps only the files of those years (a file whose name has no year is dropped then). A name that is
    not a glob must exist; at least one file must remain.

    Parameters
    ----------
    paths : str or pathlib.Path or sequence of (str or pathlib.Path)
        File names or globs; a relative one is looked up under ``base``.
    base : str or pathlib.Path, optional
        The folder of the relative names; None leaves them relative to the working directory.
    years : int or str or sequence, optional
        A years filter (see `parse_years`); None keeps every year.
    origin : str, default "paths"
        Where the setting comes from, named in the errors.

    Returns
    -------
    list of pathlib.Path
        The files, each once, ordered by year in the name (0 for none) and then by path.

    Raises
    ------
    DatasetError
        A name that is not a glob does not exist, a glob matches nothing, or no file remains.
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
            hits = sorted(Path(p) for p in glob.glob(str(path)) if Path(p).is_file() or Path(p).is_dir() and is_zarr(p))
            if not hits:
                missing.append(str(path))
        elif path.is_file() or path.is_dir() and is_zarr(path):
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
    """Return the multiplier to mm for a units name, from the table `hcwf.onset.ingest.MM_PER_UNIT`.

    Parameters
    ----------
    units : str
        The units name; case and repeated blanks do not matter.
    origin : str
        Where the name comes from, named in the error.

    Returns
    -------
    float or None
        The multiplier, or None when the units are already mm.

    Raises
    ------
    DatasetError
        The units name is not in the table.
    """
    from hcwf.onset.ingest import MM_PER_UNIT, mm_per_unit

    factor = mm_per_unit(units)
    if factor is None:
        raise DatasetError(f"{origin}: units {units!r} is not known (known: {', '.join(MM_PER_UNIT)}). "
                           "Fix: convert the data to mm or m first and set units accordingly")
    return None if factor == 1.0 else factor


def _fail(origin: str, message: str):
    """Raise a `hcwf.errors.DatasetError` naming where the entry is.

    Parameters
    ----------
    origin : str
        Where the entry comes from.
    message : str
        What is wrong.

    Raises
    ------
    DatasetError
        Always, with the message ``"<origin>: <message>"``.
    """
    raise DatasetError(f"{origin}: {message}")


def _text(data: dict, key: str, origin: str, required: bool = True):
    """Return a value of an entry as non-empty text.

    Parameters
    ----------
    data : dict
        The entry.
    key : str
        The key to read.
    origin : str
        Where the entry comes from, named in the errors.
    required : bool, default True
        Whether a missing key is an error.

    Returns
    -------
    str or None
        The text, or None for a missing key that is not required. A required key that is missing, or a value that is
        not non-empty text, raises `~hcwf.errors.DatasetError`.
    """
    value = data.get(key)
    if value is None:
        if required:
            _fail(origin, f"missing required key '{key}'. Fix: add '{key}:'")
        return None
    if not isinstance(value, str) or not value.strip():
        _fail(origin, f"{key}: expected non-empty text but found {value!r}")
    return value


def parse_dataset(data, origin: str, kind: str | None = None, require_id: bool = True) -> Dataset:
    """Validate one dataset entry given as a mapping and build its `Dataset`.

    Parameters
    ----------
    data : dict
        The entry, with the keys of `DATASET_KEYS`.
    origin : str
        The file or config key the entry comes from, named in error messages.
    kind : str, optional
        Forces the kind (inline entries); None takes ``data["kind"]``.
    require_id : bool, default True
        Whether ``id`` is required (False for inline entries).

    Returns
    -------
    Dataset
        The validated entry. An entry that is not a mapping, has an unknown or wrongly used key, or holds an invalid
        value raises `~hcwf.errors.DatasetError`.
    """
    from hcwf.onset.ingest import parse_day_boundary

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
                from hcwf.onset.ingest import _parse_clock

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
    """Read the dataset entries of one YAML file: a single entry, or a list of entries.

    Parameters
    ----------
    file : pathlib.Path
        The YAML file.

    Returns
    -------
    list of Dataset
        The entries in file order.

    Raises
    ------
    DatasetError
        The file cannot be read as YAML, or an entry is invalid.
    """
    try:
        data = yaml.safe_load(file.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(f"{file}: cannot read as YAML: {exc}") from None
    items = data if isinstance(data, list) else [data]
    return [parse_dataset(item, f"{file}" + (f"[{i}]" if isinstance(data, list) else ""))
            for i, item in enumerate(items)]


def load_datasets(dataset_files=(), catalog_dirs=()) -> dict[str, Dataset]:
    """Load the datasets described in dataset files and catalog folders.

    Later files win on a repeated id. Paths may use ``~`` and ``${ENV_VAR}``.

    Parameters
    ----------
    dataset_files : iterable of str, default ()
        YAML files, each with one entry or a list of entries.
    catalog_dirs : iterable of str, default ()
        Folders of ``*.yaml`` files (read in sorted order).

    Returns
    -------
    dict of str to Dataset
        The entries keyed by id.

    Raises
    ------
    DatasetError
        A file or folder is missing, or an entry is invalid.
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
    """Find a dataset by id among the loaded ones.

    Parameters
    ----------
    ident : str
        The id.
    datasets : dict of str to Dataset
        The loaded datasets (see `load_datasets`).
    kind : str, optional
        The kind needed; None accepts any.

    Returns
    -------
    Dataset
        The entry.

    Raises
    ------
    DatasetError
        The id is unknown (with a suggestion) or the dataset is of another kind.
    """
    if ident not in datasets:
        known = ", ".join(sorted(datasets)) or "none: no dataset_files or catalog_dirs are set"
        raise DatasetError(f"unknown dataset '{ident}'.{suggest(ident, datasets)} Known datasets: {known}. "
                           "Fix: list the file under dataset_files: (or its folder under catalog_dirs:) "
                           "or describe the dataset inline")
    entry = datasets[ident]
    if kind and entry.kind != kind:
        raise DatasetError(f"dataset '{ident}' is a {entry.kind} dataset but a {kind} dataset is needed here")
    return entry
