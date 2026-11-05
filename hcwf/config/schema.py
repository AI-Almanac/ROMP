"""The schema of an HCWF YAML run config: every key with its type, allowed values, default and help.

A YAML config is a mapping whose keys are either the flat config keys (e.g.
``obs_dir``) or sections that group them for readability::

    observations:  dataset, name, path, paths, years, file_pattern, variable, units, unit_cvt, dims, day_boundary
    reference:     dataset, model, path, file_pattern, variable, units, dims, climatology_cv
    models:        a list of {dataset, name, path, paths, years, file_pattern, variable, units, unit_cvt, dims,
                   lead_label, accumulation, daily_boundary, members}
    onset:         definition, override, rule, season, threshold_years, thresholds, threshold_scale,
                   threshold_table, threshold_max_missing, threshold_partial_years, rain_resolution, missing,
                   negatives, gaps, horizon, lead_label, check_alignment,
                   dry_spell_check, forecast_history, threshold_source, backend,
                   percentage_threshold, explain_cells
    units:         mode, shapefile, unit_id_col, unit_name_col, parent_col, weights, centroids, centroid_id_col,
                   dissemination, dissemination_id_col, dissemination_scope, box, outline, boundary, id_digits,
                   id_format, cell_area (evaluation units instead of grid cells; extends the region's ``units:``)
    blend:         years, cutoff_mode, cv_methods, cluster_var, formulas, final_formula, window_variants,
                   baselines, forecasts, mme, external_predictions, random_forest, fixed_cutoff, reference_onset,
                   bins, probability_days, trim_after_true_onset, rain_transform, climatology, models, outputs
                   (train an onset blend: `hcwf.blend`; see ``hcwf-run convert`` for blending configs)
    evaluation:    start_date, end_date, years, start_year_clim, end_year_clim, climatology_years, init_days,
                   date_filter_year, verification_windows, tolerances, max_forecast_day, day_bins, probabilistic, members
    metrics:       FAR, MAE, MR, BS, RPS, AUC, Reliability, skill_score
    output:        data_dir, figure_dir, save_* , plot_*, show_*

`flatten_sections` maps a section entry to its flat key (`FIELDS` holds the mapping),
`normalise` turns every value into the form the rest of HCWF expects (tuples, not lists;
dates as ``(year, month, day)``), and `apply_defaults` fills the keys left out. All problems
are collected and raised together as one `~hcwf.errors.ConfigSchemaError`: unknown keys come with a
"did you mean" suggestion, wrongly typed values name the key and the form expected.

The module holds the table of keys (`FIELDS`, built from `Field`), the value kinds that check and normalise a value
(`KINDS`) and the functions that turn a raw YAML mapping into a validated flat config. `hcwf.config.mapping` builds
on it to resolve a config, `hcwf.config.model` generates its output and metrics specs from it, and
`hcwf.config.datasets` supplies the years and path helpers it uses.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Any, Callable

from hcwf.errors import ConfigSchemaError, HCWFConfigError, UnitsError
from hcwf.onset import units
from hcwf.onset.model import DEFAULT_HORIZON, HORIZON_POLICIES, MISSING_POLICIES, RESOLUTIONS
from hcwf.config.datasets import ALL_DIMS, expand_env, parse_year_list, suggest
from hcwf.errors import DatasetError

__all__ = [
    "BY_KEY", "BY_PATH", "CV_MODES", "DEFAULT_CV", "FIELDS", "Field", "REQUIRED", "SECTIONS", "apply_defaults",
    "flatten_sections", "has_blend", "normalise", "section_names",
]


SECTIONS = {
    "observations": "the observation dataset",
    "reference": "the reference model of the skill scores (climatology by default)",
    "models": "the forecast models, a list",
    "onset": "what counts as onset and how it is searched",
    "units": "evaluation units (admin units, weighted cells) instead of grid cells; extends the region's units",
    "blend": "train an onset blend (cross-validated logit or forest over climatology and forecast onset features)",
    "evaluation": "the period, lead windows and metrics settings",
    "metrics": "which metrics to compute",
    "output": "output folders and which files and figures to write",
}
"""The sections a YAML config may use, with one-line help."""


class _Required:
    """The marker of a `Field` that has no default: the key must be given."""

    def __repr__(self):
        """Return the marker's name.

        Returns
        -------
        str
            ``"REQUIRED"``.
        """
        return "REQUIRED"


REQUIRED = _Required()
"""The default of a `Field` whose key must be given."""


@dataclass(frozen=True)
class Field:
    """One config key.

    ``key`` is the flat key used by HCWF; ``section`` and ``name`` give where it may be written nested
    (``name`` defaults to ``key``). Keys with ``fill=False`` are left out when not given (HCWF chooses the
    value itself, e.g. by mode); a callable default is called with the config built so far.
    """

    key: str
    """The flat key used by HCWF."""
    kind: str
    """A key of `KINDS`: how the value is checked and normalised."""
    help: str
    """What the key does, one line."""
    default: Any = None
    """The default; `REQUIRED` for none, or a callable that takes the config built so far."""
    section: str | None = None
    """The section the key may be written in, or None for a top-level key."""
    name: str | None = None
    """The name inside the section (None: the same as ``key``)."""
    choices: tuple | None = None
    """The allowed values, or None for any value of the kind."""
    nullable: bool = False
    """Whether ``null`` is accepted."""
    fill: bool = True
    """Whether the default is written when the key is left out (False: HCWF chooses the value itself)."""
    internal: bool = False
    """Whether the key is set by HCWF and refused in a YAML file."""

    @property
    def nested_name(self) -> str:
        """The key's name inside its section."""
        return self.name or self.key

    @property
    def path(self) -> str:
        """The key as written nested, e.g. ``evaluation.start_date``, or the flat key when it has no section."""
        return f"{self.section}.{self.nested_name}" if self.section else self.key

    @property
    def required(self) -> bool:
        """Whether the key must be given."""
        return self.default is REQUIRED

    @property
    def py_type(self) -> str:
        """The NumPy-style Python type of the value as a keyword argument, e.g. ``str or None``."""
        text = "{" + ", ".join(repr(c) for c in self.choices) + "}" if self.choices else KINDS[self.kind][2]
        return text + (" or None" if self.nullable and not text.endswith("None") else "")

    @property
    def note(self) -> str:
        """What the kind's wording says beyond the Python type (empty for none), as a sentence."""
        return KINDS[self.kind][3]

    @property
    def expected(self) -> str:
        """The form of value expected, for error messages and help."""
        text = KINDS[self.kind][0]
        if self.choices:
            text = "one of " + ", ".join(repr(c) for c in self.choices)
        return text + (" or null" if self.nullable else "")


def _is_int(value) -> bool:
    """Tell whether a value is a whole number that is not a bool.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True for an `int` that is not a `bool`.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    """Tell whether a value is a number that is not a bool.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True for an `int` or `float` that is not a `bool`.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value):
    """Check that a value is text.

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    str
        The value, unchanged.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, str):
        raise ValueError
    return value


def _path(value):
    """Check that a value is a path and expand its environment references.

    A value that is not text raises `ValueError` (from `_text`); an unset variable raises a `hcwf.errors.DatasetError`.

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    str
        The path with ``${ENV_VAR}`` references expanded (see `hcwf.config.datasets.expand_env`).
    """
    return expand_env(_text(value))


def _name(value):
    """Check that a value is a name: text, or a whole number (season names may be 1 or 2).

    A value of another type raises `ValueError` (from `_text`).

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    str
        The name as text.
    """
    if _is_int(value):
        return str(value)
    return _text(value)


def _integer(value):
    """Check that a value is a whole number (not a bool).

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    int
        The value, unchanged.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not _is_int(value):
        raise ValueError
    return value


def _number(value):
    """Check that a value is a number (not a bool).

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    int or float
        The value, unchanged.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not _is_number(value):
        raise ValueError
    return value


def _workers(value):
    """Check that a value is a number of workers: a whole number of 1 or more.

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    int
        The value, unchanged.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not _is_int(value) or value < 1:
        raise ValueError
    return value


def _boolean(value):
    """Check that a value is true or false.

    Parameters
    ----------
    value : object
        The raw value from the YAML file.

    Returns
    -------
    bool
        The value, unchanged.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, bool):
        raise ValueError
    return value


def _date(value):
    """Check a date and return it as a tuple.

    The date is written ``[year, month, day]``, ``"YYYY-MM-DD"`` or as a YAML date. A date that does not
    exist (30 February) is also a `ValueError`.

    Parameters
    ----------
    value : datetime.date or str or sequence of int
        The raw value from the YAML file.

    Returns
    -------
    tuple of (int, int, int)
        The date as ``(year, month, day)``.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if isinstance(value, datetime.date):
        return (value.year, value.month, value.day)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*", value)
        parts = tuple(int(g) for g in match.groups()) if match else None
    elif isinstance(value, (list, tuple)) and len(value) == 3 and all(_is_int(v) for v in value):
        parts = tuple(value)
    else:
        parts = None
    if parts is None:
        raise ValueError
    datetime.date(*parts)
    return parts


def _month_day(value):
    """Check a month and day and return them as a tuple.

    The day is written ``[month, day]`` or ``"MM-DD"``. A day that does not exist in a year (30 February) is
    also a `ValueError`; 29 February is accepted.

    Parameters
    ----------
    value : str or sequence of int
        The raw value from the YAML file.

    Returns
    -------
    tuple of (int, int)
        The day as ``(month, day)``.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_int(v) for v in value):
        text = f"{value[0]}-{value[1]}"
    else:
        raise ValueError
    try:
        return units.parse_month_day(text, leap_day=True, two_digit=False)
    except UnitsError:
        raise ValueError from None


def _years(value):
    """Check a years setting and return the years sorted.

    The setting is a list of years, or text such as ``"2000:2020,2022"`` (see
    `hcwf.config.datasets.parse_year_list`); it must not be empty.

    Parameters
    ----------
    value : int or str or sequence
        The raw value from the YAML file.

    Returns
    -------
    tuple of int
        The years, ascending, each once.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    try:
        years = parse_year_list(value)
    except DatasetError:
        raise ValueError from None
    if not years:
        raise ValueError
    return years


def _pair(value, sep="-"):
    """Check a pair of whole numbers and return it as a tuple.

    The pair is written ``[a, b]`` or as text ``"a-b"`` (with ``sep`` between the numbers).

    Parameters
    ----------
    value : str or sequence of int
        The raw value from the YAML file.
    sep : str, default "-"
        The separator of the text form (used inside a regular expression).

    Returns
    -------
    tuple of (int, int)
        The pair ``(a, b)``.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if isinstance(value, str):
        match = re.fullmatch(rf"\s*(\d+)\s*{sep}\s*(\d+)\s*", value)
        if match:
            return tuple(int(g) for g in match.groups())
    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_int(v) for v in value):
        return tuple(value)
    raise ValueError


def _windows(value):
    """Check a list of day windows and return it as a tuple of pairs.

    Each window is ``[first, last]`` or ``"first-last"`` (see `_pair`); the list must not be empty.

    Parameters
    ----------
    value : sequence
        The raw value from the YAML file.

    Returns
    -------
    tuple of tuple of (int, int)
        The ``(first, last)`` windows, in the order given.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError
    return tuple(_pair(v) for v in value)


def _integers(value):
    """Check a non-empty list of whole numbers and return it as a tuple.

    Parameters
    ----------
    value : sequence of int
        The raw value from the YAML file.

    Returns
    -------
    tuple of int
        The numbers, in the order given.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)) or not value or not all(_is_int(v) for v in value):
        raise ValueError
    return tuple(value)


def _texts(value):
    """Check a list of text and return it as a tuple.

    Parameters
    ----------
    value : sequence of str
        The raw value from the YAML file.

    Returns
    -------
    tuple of str
        The texts, in the order given (possibly none).

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ValueError
    return tuple(value)


def _paths(value):
    """Check a list of paths and return them expanded as a tuple.

    A value that is not a list of text raises `ValueError` (see `_texts`).

    Parameters
    ----------
    value : sequence of str
        The raw value from the YAML file.

    Returns
    -------
    tuple of str
        The paths with ``${ENV_VAR}`` references expanded.
    """
    return tuple(_path(v) for v in _texts(value))


def _patterns(value):
    """Check one file pattern or a list of them and return a tuple.

    A value that is not text or a list of text raises `ValueError` (see `_text`, `_texts`).

    Parameters
    ----------
    value : str or sequence of str
        The raw value from the YAML file.

    Returns
    -------
    tuple of str
        The patterns.
    """
    return (_text(value),) if isinstance(value, str) else _texts(value)


def _numbers_or_null(value):
    """Check a list of numbers or null (no conversion) and return it as a tuple.

    Parameters
    ----------
    value : sequence
        The raw value from the YAML file.

    Returns
    -------
    tuple of (int or float or None)
        The items, in the order given.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)) or not all(v is None or _is_number(v) for v in value):
        raise ValueError
    return tuple(value)


def _dims(value):
    """Check a mapping of canonical dimension names to the names in the files.

    The keys are ``init_time``, ``step``, ``member``, ``lat``, ``lon``, ``unit`` or ``time``; the values are
    non-empty text.

    Parameters
    ----------
    value : dict
        The raw value from the YAML file.

    Returns
    -------
    dict of str to str
        A copy of the mapping.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, dict) or not all(k in ALL_DIMS and isinstance(v, str) and v for k, v in value.items()):
        raise ValueError
    return dict(value)


def _paths_list(value):
    """Check a list with one list of paths per model and return a tuple of tuples.

    Parameters
    ----------
    value : sequence of sequence of str
        The raw value from the YAML file.

    Returns
    -------
    tuple of tuple of str
        The paths of each model, with ``${ENV_VAR}`` references expanded.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError
    return tuple(_paths(v) for v in value)


def _dims_list(value):
    """Check a list with one dims mapping (or null) per model and return a tuple.

    Parameters
    ----------
    value : sequence of (dict or None)
        The raw value from the YAML file.

    Returns
    -------
    tuple of (dict or None)
        The mappings (see `_dims`), None where the model has none.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError
    return tuple(None if v is None else _dims(v) for v in value)


def _members(value):
    """Check the ensemble members: ``"All"``, a range such as ``"0-9"``, or a list of whole numbers.

    Parameters
    ----------
    value : str or sequence of int
        The raw value from the YAML file.

    Returns
    -------
    str or tuple of int
        The text for ``"All"`` or a range (unchanged), else the numbers as a tuple.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if isinstance(value, str):
        if value == "All" or re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", value):
            return value
        raise ValueError
    return _integers(value)


def _cells(value):
    """Check a list of ``[lat, lon]`` pairs and return a list of tuples.

    Parameters
    ----------
    value : sequence of sequence of float
        The raw value from the YAML file.

    Returns
    -------
    list of tuple of (float, float)
        The cells, in the order given.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, (list, tuple)) or not all(
            isinstance(c, (list, tuple)) and len(c) == 2 and all(_is_number(v) for v in c) for c in value):
        raise ValueError
    return [tuple(c) for c in value]


def _thresholds(value):
    """Check a mapping of threshold names to ``"path.nc:variable"`` text, a table file, or a number of mm.

    Parameters
    ----------
    value : dict
        The raw value from the YAML file.

    Returns
    -------
    dict of str to (str or float)
        A copy of the mapping.

    Raises
    ------
    ValueError
        The value does not have the expected form.
    """
    if not isinstance(value, dict) or not all(isinstance(k, str) and (isinstance(v, str) or _is_number(v))
                                              for k, v in value.items()):
        raise ValueError
    return dict(value)


MASK_KEYS = {
    "land": ("true or false", "keep land cells only (Natural Earth land polygons from cartopy)", "bool", ""),
    "country": ("a country name or a list of them", "keep the cells of these countries (Natural Earth "
                "admin_0_countries; matched on NAME, NAME_LONG or ADM0_A3, any case)", "str or list of str", ""),
    "file": ("a path", "an existing mask: a 0/1 NetCDF file, or a shapefile / GeoJSON", "str", ""),
    "variable": ("text", "the mask variable of a NetCDF file (auto-detected if the file has only one)", "str", ""),
    "keep_value": ("a number", "the value of the cells to keep in the NetCDF mask (default 1)", "float", ""),
    "condition": ("a mapping {file, variable, above, below}", "a mask made from a field, e.g. "
                  "{file: dem.nc, variable: elevation, above: 1500}: keeps the cells above and/or below the numbers",
                  "dict", ""),
    "cell_rule": ("one of 'centre', 'overlap_fraction'", "how a shape picks cells: 'centre' (the cell centre is "
                  "inside; default) or 'overlap_fraction' (enough of the cell's area is inside)",
                  "{'centre', 'overlap_fraction'}", ""),
    "overlap_fraction": ("a number in (0, 1]", "the share of a cell's area that must be inside the shape (or, for a "
                         "mask file with finer cells, covered by kept mask cells); default 0.5", "float",
                         "Must be in (0, 1]."),
    "resolution": ("one of '10m', '50m', '110m'", "Natural Earth resolution of land and country shapes "
                   "(default '50m')", "{'10m', '50m', '110m'}", ""),
}
"""The keys of the ``mask:`` section: ``(wording, help, python_type, note)`` as in `KINDS`."""
_MASK_CONDITION_KEYS = ("file", "variable", "above", "below")


def _mask(value):
    """Check a ``mask:`` section and return the keys given.

    The section is a mapping of ``land``, ``country``, ``file``, ``variable``, ``keep_value``, ``condition``,
    ``cell_rule``, ``overlap_fraction`` and ``resolution`` (see `MASK_KEYS`); it must select something and
    ``variable`` and ``keep_value`` need ``file``.

    Parameters
    ----------
    value : dict
        The raw section.

    Returns
    -------
    dict
        The keys given, normalised (``country`` always a list, file paths with ``${ENV_VAR}`` expanded).

    Raises
    ------
    ValueError
        The section is not a mapping, has an unknown key or an invalid value, or selects nothing; the message says what
        is wrong.
    """
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping with {', '.join(MASK_KEYS)}")
    for key in value:
        if key not in MASK_KEYS:
            raise ValueError(f"unknown key {key!r}.{suggest(key, MASK_KEYS)} Keys: {', '.join(MASK_KEYS)}")
    out = {}
    for key, item in value.items():
        if key == "land":
            out[key] = _boolean(item)
        elif key == "country":
            names = [item] if isinstance(item, str) else item
            if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) and n for n in names):
                raise ValueError("country must be a country name or a non-empty list of names")
            out[key] = list(names)
        elif key in ("file", "variable"):
            out[key] = _path(item) if key == "file" else _text(item)
        elif key == "keep_value":
            if not (_is_number(item) or isinstance(item, bool)):
                raise ValueError("keep_value must be a number")
            out[key] = item
        elif key == "condition":
            if not isinstance(item, dict) or "file" not in item:
                raise ValueError("condition must be a mapping with file, variable and above and/or below")
            for sub in item:
                if sub not in _MASK_CONDITION_KEYS:
                    raise ValueError(f"unknown key condition.{sub}.{suggest(sub, _MASK_CONDITION_KEYS)} "
                                     f"Keys: {', '.join(_MASK_CONDITION_KEYS)}")
            if item.get("above") is None and item.get("below") is None:
                raise ValueError("condition needs above and/or below")
            if any(item.get(s) is not None and not _is_number(item[s]) for s in ("above", "below")):
                raise ValueError("condition.above and condition.below must be numbers")
            out[key] = {s: (_path(item[s]) if s == "file" else item[s]) for s in _MASK_CONDITION_KEYS
                        if item.get(s) is not None}
        elif key == "cell_rule":
            if item not in ("centre", "overlap_fraction"):
                raise ValueError(f"cell_rule must be 'centre' or 'overlap_fraction', got {item!r}")
            out[key] = item
        elif key == "overlap_fraction":
            if not _is_number(item) or not 0 < item <= 1:
                raise ValueError("overlap_fraction must be a number above 0 and at most 1")
            out[key] = item
        elif key == "resolution":
            if item not in ("10m", "50m", "110m"):
                raise ValueError(f"resolution must be '10m', '50m' or '110m', got {item!r}")
            out[key] = item
    if not any(out.get(k) for k in ("land", "country", "file", "condition")):
        raise ValueError("give at least one of land: true, country, file, condition (a mask section needs "
                         "something to select)")
    if ("variable" in out or "keep_value" in out) and "file" not in out:
        raise ValueError("variable and keep_value describe the mask file; set file too")
    return out


UNIT_MODES = ("cells", "regrid", "cell_transform")
"""How rain reaches the units (``units.mode``)."""


def _unit_mode(value):
    """Check a ``units.mode``: one of `UNIT_MODES`, or a mapping of source name to one.

    The source is ``"observations"``, a model's name or ``"*"``.

    Parameters
    ----------
    value : str or dict of str to str
        The raw mode.

    Returns
    -------
    str or dict of str to str
        The mode, or a copy of the mapping.

    Raises
    ------
    ValueError
        The value is not a mode or a non-empty mapping of names to modes.
    """
    if isinstance(value, str) and value in UNIT_MODES:
        return value
    if isinstance(value, dict) and value and all(isinstance(k, str) and v in UNIT_MODES for k, v in value.items()):
        return dict(value)
    raise ValueError


def _anything(value):
    """Accept any value.

    Parameters
    ----------
    value : object
        The raw value.

    Returns
    -------
    object
        The value, unchanged.
    """
    return value


KINDS: dict[str, tuple[str, Callable, str, str]] = {
    "str": ("text", _text,
        "str", ""),
    "name": ("a name (text or a whole number)", _name,
        "str or int", ""),
    "path": ("a path (text; ${ENV_VAR} is expanded)", _path,
        "str", "``${ENV_VAR}`` in it is expanded."),
    "int": ("a whole number", _integer,
        "int", ""),
    "number": ("a number", _number,
        "float", ""),
    "bool": ("true or false", _boolean,
        "bool", ""),
    "workers": ("a whole number of 1 or more", _workers,
        "int", "A whole number of 1 or more."),
    "date": ("a date, [year, month, day] or 'YYYY-MM-DD'", _date,
        "list of int or str", "Given as ``[year, month, day]`` or ``'YYYY-MM-DD'``."),
    "month_day": ("a month and day, [month, day] or 'MM-DD'", _month_day,
        "list of int or str", "Given as ``[month, day]`` or ``'MM-DD'``."),
    "years": ("a list of years or text such as '2000:2020,2022' (ends included)", _years,
        "list of int or str", "Text such as ``'2000:2020,2022'`` is accepted (ends included)."),
    "year_pair": ("a pair of years, [first, last] or 'first-last'", _pair,
        "list of int or str", "Given as ``[first, last]`` or ``'first-last'``."),
    "windows": ("a list of [first, last] day pairs, e.g. [[1, 15], [16, 30]]", _windows,
        "list of list of int", "Each item is a ``[first, last]`` day pair, e.g. ``[[1, 15], [16, 30]]``."),
    "ints": ("a list of whole numbers, e.g. [3, 5]", _integers,
        "list of int", ""),
    "strs": ("a list of text", _texts,
        "list of str", ""),
    "paths": ("a list of paths", _paths,
        "list of str", ""),
    "patterns": ("a file pattern such as '{}.nc' (quote it), or a list of them", _patterns,
        "str or list of str", "A file pattern such as ``'{}.nc'`` (quote it)."),
    "numbers_or_null": ("a list of numbers or null", _numbers_or_null,
        "list of float or None", ""),
    "dims": ("a mapping from init_time, step, member, lat, lon (or unit) to the file's dimension names", _dims,
        "dict", "Maps init_time, step, member, lat, lon (or unit) to the file's dimension names."),
    "dims_list": ("a list of dims mappings or null, one per model", _dims_list,
        "list of dict or None", "One mapping per model."),
    "paths_list": ("a list with one list of paths per model", _paths_list,
        "list of list of str", "One list of paths per model."),
    "members": ("'All', a range like '0-9', or a list of whole numbers", _members,
        "str or list of int", "Given as ``'All'``, a range like ``'0-9'``, or a list of whole numbers."),
    "cells": ("a list of [lat, lon] pairs, e.g. [[9.0, 38.75]]", _cells,
        "list of list of float", "Each item is a ``[lat, lon]`` pair, e.g. ``[[9.0, 38.75]]``."),
    "thresholds": ("a mapping of threshold names to 'path.nc:variable' (a grid), a table file (by unit id) or a "
                   "number of mm", _thresholds,
        "dict", "A threshold is ``'path.nc:variable'`` (a grid), a table file (by unit id) or a number of mm."),
    "mask": ("a mapping of land, country, file, variable, keep_value, condition, cell_rule, overlap_fraction, "
             "resolution", _mask,
        "dict", ""),
    "unit_mode": ("one of 'cells', 'regrid', 'cell_transform', or a mapping of source name to one of them", _unit_mode,
        "{'cells', 'regrid', 'cell_transform'} or dict of str to str", "A mapping gives one mode per source name."),
    "any": ("any value", _anything,
        "object", ""),
    "any_mapping": ("any value", _anything,
        "dict", ""),
}
"""Value kinds: ``(wording, checker, python_type, note)``.

The wording is the YAML form expected (error messages, the CLI, the config-key reference); the checker checks and
normalises a value; the Python type is the NumPy-style type of the value as a keyword argument; the note is what
the wording says beyond the type (empty for none)."""

_SIX_BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))
"""The default day bins: six of five days."""
_TOGGLES = (
    ("save_fig", "write figures (PNG/PDF)", True),
    ("save_nc_spatial_far_mr_mae", "save the spatial FAR, MR and MAE as NetCDF", True),
    ("save_csv_score", "save the skill scores as CSV", True),
    ("save_nc_climatology", "save the climatological onset as NetCDF", True),
    ("save_onsets", "write every source's onset dates as NetCDF to dir_out/onsets/ (repeats the detection)", False),
    ("plot_spatial_far_mr_mae", "map FAR, MR and MAE", True),
    ("plot_heatmap_bss_auc", "heatmaps of BSS and AUC", True),
    ("plot_reliability", "reliability diagrams", True),
    ("plot_climatology_onset", "map of the climatological onset", True),
    ("plot_panel_heatmap_error", "panel heatmap of MAE, MR and FAR", True),
    ("plot_panel_heatmap_skill", "panel heatmap of the skill scores", True),
    ("plot_bar_bss_rpss_auc", "bar plot of the skill scores", True),
    ("show_plot", "show each model's plot interactively", False),
    ("show_panel", "show the panel plot interactively", False),
)
"""The yes/no keys of the ``output`` section: ``(key, help, default)``."""


CV_MODES = ("none", "leave_one_year_out")
"""The values of ``climatology_cv`` (both climatology references: the conditional and the unconditional)."""
DEFAULT_CV = "leave_one_year_out"
"""The ``climatology_cv`` used when none is given."""


def _fields() -> tuple[Field, ...]:
    """Build the table of every config key.

    Returns
    -------
    tuple of Field
        Every key, in documentation order: top-level keys, then the observations, reference, onset, units, blend,
        evaluation, metrics and output sections.
    """
    top = [
        Field("project_name", "str", "name shown in the run banner", "HCWF run"),
        Field("dataset_files", "paths", "YAML files describing your datasets, referenced by id with `dataset:`", ()),
        Field("catalog_dirs", "paths", "folders of <id>.yaml dataset files, referenced by id with `dataset:`", ()),
        Field("region", "str", "target region id or alias, see `hcwf-run onset regions`; null: no region (no box, "
              "no region defaults; what a converted onset_blending config uses)", REQUIRED, nullable=True),
        Field("nc_mask", "path", "NetCDF mask of 1s and 0s limiting the evaluated cells", None, nullable=True),
        Field("mask", "mask", "cells to evaluate, built on the evaluation grid (see evaluation_grid); the parts given intersect: "
              + "; ".join(f"{k}: {h}" for k, (_, h, *_rest) in MASK_KEYS.items()), None, nullable=True, fill=False),
        Field("evaluation_grid", "str", "the grid the run is scored on when observations and models are on different "
              "grids: 'observations' or a model's name, whose grid (cut to the region box, mask built on it) every "
              "source is averaged conservatively onto; null: the coarsest of all the sources", None, nullable=True,
              fill=False),
        Field("shpfile_dir", "path", "shapefile for the region boundary in maps", None, nullable=True),
        Field("polygon", "bool", "mask and boundary from region polygons (hcwf.onset.regions.polygon_boundary)", False),
        Field("parallel", "bool", "compute years in several worker processes (false: one process); "
              "write `parallel: {workers: N}` to choose the number", True),
        Field("parallel_workers", "workers", "worker processes when parallel is on (parallel.workers); null: the "
              "cores available, never more than there are years", None, nullable=True, fill=False),
        # the models list, as flat keys (the `models:` section builds these)
        Field("model_list", "strs", "model names (use the `models:` section instead)", REQUIRED, internal=True),
        Field("model_dir_list", "paths", "forecast folder of each model", REQUIRED, internal=True),
        Field("model_var_list", "strs", "rainfall variable of each model", REQUIRED, internal=True),
        Field("unit_cvt_list", "numbers_or_null", "multiplier to mm of each model (derived from units)", REQUIRED, internal=True),
        Field("accumulation_list", "strs", "accumulation of each model (from models[].accumulation)", fill=False, internal=True),
        Field("daily_boundary_list", "strs", "daily boundary of each model, or null (from models[].daily_boundary)", fill=False, internal=True),
        Field("file_pattern_list", "strs", "file pattern of each model", REQUIRED, internal=True),
        Field("model_paths_list", "paths_list", "the files of each model, resolved from models[].paths and years "
              "(a blend reads every file of each model)", fill=False, internal=True),
        Field("model_dims_list", "dims_list", "dimension names of each model's files", None, fill=False, internal=True),
    ]
    obs = [
        Field("obs", "str", "observation dataset name (default: 'obs')", "obs",
              "observations", "name"),
        Field("obs_dir", "path", "folder of the observation files (or the base of observations.paths)", REQUIRED,
              "observations", "path"),
        Field("obs_file_pattern", "patterns", "file pattern, {} is the year", ("{}.nc",), "observations",
              "file_pattern"),
        Field("obs_var", "str", "rainfall variable in the files", REQUIRED, "observations", "variable"),
        Field("obs_unit_cvt", "number", "multiplier to mm (derived from units)", None, nullable=True,
              internal=True),
        Field("obs_paths", "paths", "the observation files, resolved from observations.paths and years (any number "
              "per year; instead of obs_dir and obs_file_pattern)", fill=False, internal=True),
        Field("obs_dims", "dims", "dimension names of the observation files", None, nullable=True, fill=False,
              internal=True),
        Field("obs_day_boundary", "str", "when the day starts and which date labels it, e.g. "
              "'00 UTC, labelled by start' (null: the region's)", None, "observations", "day_boundary",
              nullable=True),
    ]
    ref = [
        Field("ref_model", "str", "'climatology', 'conditional_climatology' (P(onset on lead k | no onset by the issue date) "
              "from a KDE of observed onsets over start_year_clim..end_year_clim; needs >= 10 onset years per "
              "cell, other cells are not scored by the reference) or a reference model's name", "climatology",
              "reference", "model"),
        Field("climatology_cv", "str", "cross-validation of the climatology references ('climatology' and "
              "'conditional_climatology'): 'leave_one_year_out' leaves the evaluated year out of its own climatology "
              "(members or KDE fit; a year outside the climatology years has nothing to leave out), 'none' keeps all "
              "climatology years; not used by a model reference", DEFAULT_CV,
              "reference", choices=CV_MODES),
        Field("ref_model_dir", "path", "reference data folder (default: the observations')", None, "reference",
              "path", nullable=True),
        Field("ref_model_file_pattern", "str", "reference file pattern (default: the observations')", None,
              "reference", "file_pattern", nullable=True),
        Field("ref_model_var", "str", "reference rainfall variable (default: the observations')", None,
              "reference", "variable", nullable=True),
        Field("ref_model_unit_cvt", "number", "reference multiplier to mm (derived from reference.units)", None,
              nullable=True, internal=True),
        Field("ref_model_dims", "dims", "dimension names of the reference files", None, "reference", "dims",
              nullable=True, fill=False),
        Field("ref_model_accumulation", "str", "accumulation of the reference model's files: 'daily', 'since_init' or "
              "'auto' (default: detected from the first file; only a reference model has one)", None, "reference",
              "accumulation", choices=("daily", "since_init", "auto"), nullable=True, fill=False),
        Field("ref_model_daily_boundary", "str", "UTC clock time at which the reference model's days are cut, e.g. "
              "'03 UTC' (default: none)", None, "reference", "daily_boundary", nullable=True, fill=False),
        Field("ref_model_accumulation_detected", "bool", "whether ref_model_accumulation was detected, not given",
              False, nullable=True, fill=False, internal=True),
    ]
    onset = [
        Field("onset_definition", "str", "registry id or path of a definition YAML (null: the region's)", None,
              "onset", "definition", nullable=True, fill=False),
        Field("onset_override", "any_mapping", "changes merged into the definition (same rules as `extends`), e.g. "
              "{season: {from: 05-15}}", None, "onset", "override", nullable=True, fill=False),
        Field("onset_rule", "any_mapping", "a complete definition written inline instead of a registry id", None, "onset",
              "rule", nullable=True, fill=False),
        Field("onset_season", "name", "season name, for definitions with several", None, "onset", "season",
              nullable=True),
        Field("threshold_years", "year_pair", "(first, last) years of the climatological thresholds "
              "(null: the region's)", None, "onset", nullable=True),
        Field("thresholds", "thresholds", "precomputed thresholds {name: 'path.nc:variable'}", None, "onset",
              nullable=True),
        Field("model_lead_label", "str", "date of lead k: 'start' = init + k - 1, 'end' = init + k "
              "(null: from the day boundary)", None, "onset", "lead_label", choices=("start", "end"),
              nullable=True),
        Field("check_alignment", "bool", "check that observations and forecasts are not shifted (an analysis: reads a year of data, so off by default; `--check --alignment` runs it once)", False, "onset"),
        Field("dry_spell_check", "bool", "false drops the definition's dry-spell condition", True, "onset"),
        Field("forecast_history", "str", "'observed' stitches observed rain before each forecast, 'none' "
              "(null: observed)", None, "onset", choices=("observed", "none"), nullable=True, fill=False),
        Field("threshold_source", "str", "thresholds from 'observations' or the 'model' climatology "
              "(null: observations)", None, "onset", choices=("observations", "model"), nullable=True, fill=False),
        Field("rain_resolution", "str", "'tenths' (HCWF: rain rounded to 0.1 mm) or 'exact' (float millimetres, as "
              "onset_blending reads them; null: tenths)", None, "onset", choices=RESOLUTIONS, nullable=True,
              fill=False),
        Field("missing", "str", "missing-rain policy: flag, keep, zero or drop (null: flag)", None, "onset",
              choices=MISSING_POLICIES, nullable=True, fill=False),
        Field("negatives", "str", "'default' flags negative observations and zeroes negative forecasts, 'keep' leaves "
              "them (null: default)", None, "onset", choices=("default", "keep"), nullable=True, fill=False),
        Field("gaps", "str", "days missing inside a year of observations: 'fill' them or 'refuse' (null: fill)", None,
              "onset", choices=("fill", "refuse"), nullable=True, fill=False),
        Field("onset_horizon", "str", "a follow-up past the end of the data: 'censor' the search, 'truncate' the "
              f"follow-up or 'reject' the onset (null: {DEFAULT_HORIZON})", None, "onset", "horizon",
              choices=HORIZON_POLICIES, nullable=True, fill=False),
        Field("threshold_scale", "any", "clip(t * factor + offset, min, max) applied to tables, grids and computed "
              "thresholds: a mapping of factor, offset, min, max (null: none)", None, "onset", nullable=True,
              fill=False),
        Field("threshold_table", "any", "how a threshold table is read: a mapping of id_col, value_col, lat_col, "
              "lon_col, lat_var, lon_var, value_var, mat_lat_name, mat_lon_name, mat_value_name", None, "onset",
              nullable=True, fill=False),
        Field("threshold_max_missing", "number", "computed thresholds: the share of missing reference days above "
              "which a series has none (null: 0.10; 1 never drops one)", None, "onset", nullable=True, fill=False),
        Field("threshold_partial_years", "bool", "computed thresholds: count the days of a year only partly in the "
              "data (null: false, such a year is skipped)", None, "onset", nullable=True, fill=False),
        Field("onset_backend", "str", "detection engine", "numba", "onset", "backend",
              choices=("numba", "reference", "gpu")),
        Field("onset_percentage_threshold", "number", "share of members (0-1) that must have an onset", 0.5,
              "onset", "percentage_threshold"),
        Field("explain_cells", "cells", "cells to explain, [[lat, lon]] (null: off)", None, "onset",
              nullable=True),
    ]
    units = [
        Field("units_mode", "unit_mode", "how rain reaches the units (one mode, or a mapping by source: 'observations', a model's name, '*'): 'cells' (the file's own cells or units), 'regrid' (weights from the file's grid; unit factors after averaging) or 'cell_transform' (a source-to-unit weights table; factors before averaging, missing rain judged on the source cells); null: cells, or regrid with a shapefile or weights", None, "units", "mode", nullable=True, fill=False),
        Field("units_shapefile", "path", "polygons of the units (with unit_id_col)", None, "units", "shapefile", nullable=True, fill=False),
        Field("units_unit_id_col", "str", "id column of the shapefile", None, "units", "unit_id_col", nullable=True, fill=False),
        Field("units_unit_name_col", "str", "name column of the shapefile", None, "units", "unit_name_col", nullable=True, fill=False),
        Field("units_parent_col", "str", "parent column of the shapefile", None, "units", "parent_col", nullable=True, fill=False),
        Field("units_weights", "any", "weights table (source_id, target_id, weight), or a mapping of source name ('observations' or a model's name) to its own table", None, "units", "weights", nullable=True, fill=False),
        Field("units_centroids", "path", "CSV of unit centroids (id, lat, lon)", None, "units", "centroids", nullable=True, fill=False),
        Field("units_centroid_id_col", "str", "id column of the centroids", None, "units", "centroid_id_col", nullable=True, fill=False),
        Field("units_dissemination", "path", "CSV list of the dissemination units", None, "units", "dissemination", nullable=True, fill=False),
        Field("units_dissemination_id_col", "str", "id column of the dissemination list", None, "units", "dissemination_id_col", nullable=True, fill=False),
        Field("units_dissemination_scope", "str", "'inputs' reads only the dissemination units, 'training' marks them as the subset 'dissemination'", None, "units", "dissemination_scope", choices=("inputs", "training"), nullable=True, fill=False),
        Field("units_box", "any", "only units whose centroid is inside: a mapping of lat_min, lat_max, lon_min, lon_max", None, "units", "box", nullable=True, fill=False),
        Field("units_outline", "path", "only units whose centroid is inside this GeoJSON or shapefile", None, "units", "outline", nullable=True, fill=False),
        Field("units_boundary", "str", "a centroid on the outline's edge is 'contains' (left out) or 'intersects' (kept)", None, "units", "boundary", choices=("contains", "intersects"), nullable=True, fill=False),
        Field("units_cell_area", "str", "how the weights of the shapefile's polygons measure a cell's area: 'sphere' (exact areas on the sphere), 'cos_lat' (degrees squared times cos(latitude)) or 'planar' (degrees squared, as onset_blending); null: sphere", None, "units", "cell_area", choices=("planar", "cos_lat", "sphere"), nullable=True, fill=False),
        Field("units_id_digits", "int", "decimal digits of grid unit ids (null: established or inferred)", None, "units", "id_digits", nullable=True, fill=False),
        Field("units_id_format", "str", "grid ids 'trimmed' of trailing zeros or 'fixed'", None, "units", "id_format", choices=("trimmed", "fixed"), nullable=True, fill=False),
    ]
    blend = [
        Field("blend_years", "any", "years of the blend, a mapping of training (fit on these), cv_holdout (scored by "
              "leave-out cross-validation) and true_holdout (scored, never fitted; optional); each a list of years or "
              "text like '2000:2024,2026'. Required to train a blend", None, "blend", "years", nullable=True,
              fill=False),
        Field("blend_cutoff_mode", "str", "which issue dates are scored: 'fixed_cutoff' (the issue on or after "
              "fixed_cutoff), 'ref_onset' (the issue on or after the reference onset) or 'no_ref_filter' "
              "(default 'fixed_cutoff')", None, "blend", "cutoff_mode", choices=("fixed_cutoff", "ref_onset",
              "no_ref_filter"), nullable=True, fill=False),
        Field("blend_cv_methods", "strs", "cross-validation methods: 'global', 'local' (one fit per unit) and/or "
              "'clusters' (default ['global'])", None, "blend", "cv_methods", nullable=True, fill=False),
        Field("blend_cluster_var", "str", "grouping column of the 'clusters' method (default 'cluster')", None,
              "blend", "cluster_var", nullable=True, fill=False),
        Field("blend_formulas", "any", "the blend models: a mapping of name to a formula text "
              "('outcome ~ prob_clim_qx + diff_<model>_qx') or to {text, enabled}; features are <base>_qx or "
              "<base>_week<n>", None, "blend", "formulas", nullable=True, fill=False),
        Field("blend_final_formula", "str", "the formula fitted on every training year for operations (default: the "
              "only enabled formula)", None, "blend", "final_formula", nullable=True, fill=False),
        Field("blend_window_variants", "any", "formula variants that swap a feature for its windowed climatology: a "
              "list of {base, windows: [[2000, 2010]], replace: prob_clim, only_if_cutoff_mode}", None, "blend",
              "window_variants", nullable=True, fill=False),
        Field("blend_baselines", "any", "climatology baselines scored: {unconditional: true, conditional: true, "
              "windows: [[2000, 2010]]} (unconditional is the skill baseline and cannot be turned off)", None,
              "blend", "baselines", nullable=True, fill=False),
        Field("blend_forecasts", "any", "raw and calibrated scoring of each model's own probabilities: a list of "
              "{model, variant: base|fixed_cutoff|ref_onset, raw: true, calibrated: true, platt_weights: false} "
              "(default: every model, raw and calibrated)", None, "blend", "forecasts", nullable=True, fill=False),
        Field("blend_mme", "any", "multi-model ensemble with RPS-optimised weights: {enabled, variants, sources: "
              "[{name, source: clim|forecast, cal_variant}], optimize: auto|always|never, weights: {source name: "
              "weight} or the path of a weights CSV (relative to the config file), used instead of optimising}", None,
              "blend", "mme", nullable=True, fill=False),
        Field("blend_external_predictions", "any", "cross-validated predictions made elsewhere, scored with the "
              "rest: a list of {name, file, method}", None, "blend", "external_predictions", nullable=True,
              fill=False),
        Field("blend_random_forest", "any", "also blend with the CRPS random forest: {name, trees, min_samples_leaf, "
              "max_depth, max_samples, max_features: sqrt|all, random_state, max_day, missing_onset: "
              "after_week4|drop, years_per_fold}", None, "blend", "random_forest", nullable=True, fill=False),
        Field("blend_fixed_cutoff", "str", "start of the fixed_cutoff forecast probabilities, 'MM-DD' (default "
              "'06-02')", None, "blend", "fixed_cutoff", nullable=True, fill=False),
        Field("blend_reference_onset", "any", "the reference (climatological) onset: {month_day: 'MM-DD'} every year, "
              "or {file, date_col | year_col + day_col + base_date, unit_col} a table of dates", None, "blend",
              "reference_onset", nullable=True, fill=False),
        Field("blend_bins", "any", "the weekly outcome bins: {count: 4, days: 7} (outcomes week1..weekN, later)", None,
              "blend", "bins", nullable=True, fill=False),
        Field("blend_probability_days", "int", "lead days of onset probability summed into the bins (default 28)",
              None, "blend", "probability_days", nullable=True, fill=False),
        Field("blend_trim_after_true_onset", "bool", "drop issue dates after the observed onset (default true)", None,
              "blend", "trim_after_true_onset", nullable=True, fill=False),
        Field("blend_rain_transform", "any", "transform of the rain features: identity, sqrt, fourth_root, log1p, "
              "power:<p> or a number (default identity)", None, "blend", "rain_transform", nullable=True,
              fill=False),
        Field("blend_climatology", "any", "the KDE climatology: {issue_end: '07-31', forecast_window | horizons: "
              "[{start, end, forecast_window}], cv_by_year: true, min_onset_years: 10, train_years, test_years, "
              "windows: [[2000, 2010]], workers}", None, "blend", "climatology", nullable=True, fill=False),
        Field("blend_models", "any", "per forecast model (by its name in `models:`): {lead_days: [1, 45] or "
              "{min, max}, max_member, rain_horizon: {max_day, policy: legacy|exact|truncate}, rain_predictors: "
              "[{agg: diff|max|min, window: days or a token of the definition}], rain_transform}", None, "blend",
              "models", nullable=True, fill=False),
        Field("blend_outputs", "any", "what a blend run writes besides its artifacts: {dir (where "
              "external_predictions files are read from), intermediates: [observations, features] (written as "
              "observed_onsets.nc and weekly_features.nc), pickles: false}", None, "blend", "outputs", nullable=True, fill=False),
    ]
    evaluation = [
        Field("start_date", "date", "first date; its month/day start the forecast init dates", REQUIRED,
              "evaluation"),
        Field("end_date", "date", "last date; its month/day end the forecast init dates", REQUIRED,
              "evaluation"),
        Field("start_year_clim", "int", "first year of the climatology (default: the region's)", REQUIRED,
              "evaluation"),
        Field("end_year_clim", "int", "last year of the climatology (default: the region's)", REQUIRED,
              "evaluation"),
        Field("years", "years", "the years evaluated, when not every year from start_date to end_date (may skip "
              "years; all must lie in that range)", None, "evaluation", nullable=True, fill=False),
        Field("years_clim", "years", "the climatology years, when not every year from start_year_clim to "
              "end_year_clim (may skip years; all must lie in that range)", None, "evaluation",
              "climatology_years", nullable=True, fill=False),
        Field("init_days", "ints", "forecast init weekdays, 0 = Monday", (0, 3), "evaluation"),
        Field("date_filter_year", "int", "year for init-day calendar alignment (default: start year)", None,
              "evaluation", nullable=True),
        Field("verification_window_list", "windows", "lead windows scored", ((1, 15), (16, 30)), "evaluation",
              "verification_windows"),
        Field("tolerance_days_list", "ints", "error margin in days, one per window", (3, 5), "evaluation",
              "tolerances"),
        Field("max_forecast_day", "int", "last lead day used", 30, "evaluation"),
        Field("day_bins", "windows", "lead bins for the statistics", _SIX_BINS, "evaluation"),
        Field("probabilistic", "bool", "evaluate ensembles", False, "evaluation"),
        Field("members", "members", "ensemble members used", "All", "evaluation"),
    ]
    metrics = [Field(name, "bool", f"{name} metric", True, "metrics")
               for name in ("FAR", "MAE", "MR", "BS", "RPS", "AUC", "Reliability", "skill_score")]
    output = [
        Field("dir_out", "path", "folder for NetCDF and CSV results", "output", "output", "data_dir"),
        Field("dir_fig", "path", "folder for figures", "figures", "output", "figure_dir"),
    ] + [Field(name, "bool", text, default, "output") for name, text, default in _TOGGLES] + [
        Field("plot_portrait", "bool", "portrait plots", False, "output", fill=False),
    ]
    return tuple(top + obs + ref + onset + units + blend + evaluation + metrics + output)


FIELDS = _fields()
"""Every config key, in documentation order."""
BY_KEY = {f.key: f for f in FIELDS}
"""The `Field` of each flat key."""
BY_PATH = {(f.section, f.nested_name): f for f in FIELDS if f.section}
"""The `Field` of each key written nested, by ``(section, name)``."""
SPECIAL_SUBKEYS = {"observations": ("dataset", "units", "unit_cvt", "dims", "paths", "years"),
                   "reference": ("dataset", "units")}
"""Section entries that are not config keys: the dataset file entry to take values from, units, dims, listed files."""
MODEL_ENTRY_KEYS = ("dataset", "name", "path", "paths", "years", "file_pattern", "variable", "units", "unit_cvt", "dims",
                    "lead_label", "accumulation", "daily_boundary", "members")
"""Keys of one entry of the ``models:`` list."""


BLEND_KEYS = tuple(f.key for f in FIELDS if f.section == "blend")
"""The flat keys of the ``blend:`` section; a config that sets any of them trains a blend."""


def has_blend(config) -> bool:
    """Tell whether a flat config has a ``blend:`` section, i.e. sets any of its keys.

    Parameters
    ----------
    config : collections.abc.Mapping
        A flat config, or any mapping of flat keys.

    Returns
    -------
    bool
        True when any key of `BLEND_KEYS` is present and not None.
    """
    return any(config.get(key) is not None for key in BLEND_KEYS)


def section_names(section: str) -> list[str]:
    """Return the names a section accepts.

    Parameters
    ----------
    section : str
        A key of `SECTIONS`.

    Returns
    -------
    list of str
        For ``models`` the keys of `MODEL_ENTRY_KEYS`; otherwise the names of the section's fields followed by its
        `SPECIAL_SUBKEYS`.
    """
    if section == "models":
        return list(MODEL_ENTRY_KEYS)
    return [f.nested_name for f in FIELDS if f.section == section] + list(SPECIAL_SUBKEYS.get(section, ()))


def _where(name: str) -> str:
    """Say where a bare key name belongs when it is written at the top level by mistake.

    Parameters
    ----------
    name : str
        The key.

    Returns
    -------
    str
        The sections that accept it, comma separated (``"observations, models"``), or an empty string.
    """
    homes = [f.section for f in FIELDS if f.section and f.nested_name == name]
    if name in MODEL_ENTRY_KEYS:
        homes.append("models")
    if name in SPECIAL_SUBKEYS.get("observations", ()):
        homes.append("observations")
    return ", ".join(dict.fromkeys(homes))


_ONSET_KEYS_REMOVED = ("The old flat onset keys (wet_init, wet_threshold, wet_spell, dry_threshold, dry_spell, dry_extent, "
                       "thresh_file, thresh_var, earliest_onset_date, mok) are no longer read. Fix: write the rule as an "
                       "onset definition; `hcwf-run convert old.yaml -o new.yaml` translates a config that still has them")
REMOVED_KEYS = {
    **dict.fromkeys(("wet_init", "wet_threshold", "wet_spell", "dry_threshold", "dry_spell", "dry_extent", "thresh_file",
                     "thresh_var", "earliest_onset_date", "mok"), _ONSET_KEYS_REMOVED),
    **dict.fromkeys(("layout", "work_dir", "pkg_dir", "debug", "fallback_date"),
                    "This key was removed: it had no effect. Fix: delete it"),
}
"""Top-level keys that earlier versions read; naming one gets a specific message instead of a spelling suggestion."""


def unknown_key_message(name: str, section: str | None = None) -> str:
    """Return the error text for an unknown key.

    The text carries a "did you mean" suggestion (see `hcwf.config.datasets.suggest`) and where the key may belong.

    Parameters
    ----------
    name : str
        The unknown key.
    section : str, optional
        The section it was written in; None for a top-level key.

    Returns
    -------
    str
        The message.
    """
    if section:
        candidates = section_names(section)
        return (f"unknown key '{section}.{name}'.{suggest(name, candidates)} "
                f"Keys of {section}: {', '.join(candidates)}")
    if name in REMOVED_KEYS:
        return f"unknown key '{name}'. {REMOVED_KEYS[name]}"
    candidates = [k for k, f in BY_KEY.items() if not f.internal] + list(SECTIONS)
    hint = suggest(name, candidates)
    home = _where(name)
    if home and not hint:
        hint = f' "{name}" belongs inside the section: {home}.'
    return f"unknown key '{name}'.{hint} Run `hcwf-run init` to see a config with the common keys"


def flatten_sections(raw: dict, errors: list[str]) -> tuple[dict, dict, dict]:
    """Flatten a YAML mapping to flat config keys.

    Parameters
    ----------
    raw : dict
        The mapping read from YAML: flat keys and sections.
    errors : list of str
        Receives a message for each problem (unknown keys, a key set twice, a section that is not a mapping).

    Returns
    -------
    flat : dict
        The raw values by flat key.
    origin : dict of str to str
        The path each flat key was written under, for messages.
    special : dict
        The catalog-aware blocks: ``{"observations": dict, "reference": dict, "models": list or None}``.
    """
    flat: dict = {}
    origin: dict = {}
    special: dict = {"observations": {}, "reference": {}, "models": None}

    def put(field: Field, value, written: str):
        """Store a raw value under its flat key, once; report a second setting in `errors`."""
        if field.key in flat:
            errors.append(f"'{written}' sets {field.key} again; it is already set by '{origin[field.key]}'. "
                          "Fix: give it once")
            return
        flat[field.key], origin[field.key] = value, written

    for key, value in raw.items():
        if key == "models":
            if value is not None:
                special["models"] = value
        elif key in SECTIONS:
            if value is None:
                continue
            if not isinstance(value, dict):
                errors.append(f"'{key}': expected a mapping of keys ({', '.join(section_names(key))}) "
                              f"but found {value!r}")
                continue
            for sub, subvalue in value.items():
                field = BY_PATH.get((key, sub))
                if field:
                    put(field, subvalue, f"{key}.{sub}")
                elif sub in SPECIAL_SUBKEYS.get(key, ()):
                    special[key][sub] = subvalue
                else:
                    errors.append(unknown_key_message(str(sub), key))
        elif key == "parallel" and isinstance(value, dict):
            for sub, subvalue in value.items():
                if sub == "workers":
                    put(BY_KEY["parallel_workers"], subvalue, "parallel.workers")
                elif sub == "enabled":
                    put(BY_KEY["parallel"], subvalue, "parallel.enabled")
                else:
                    errors.append(f"'parallel.{sub}': unknown key.{suggest(str(sub), ('workers', 'enabled'))} "
                                  "`parallel` is true, false, or a mapping of workers (and enabled)")
        elif key in BY_KEY and not BY_KEY[key].internal:
            put(BY_KEY[key], value, key)
        else:
            errors.append(unknown_key_message(str(key)))
    return flat, origin, special


def normalise_value(field: Field, value, written: str, errors: list[str]):
    """Check one value against its field and return it in HCWF's form.

    Parameters
    ----------
    field : Field
        The key's field.
    value : object
        The raw value.
    written : str
        The key as written, for messages.
    errors : list of str
        Receives a message when the value is not valid.

    Returns
    -------
    object
        The normalised value; None for a null value and, with a message in ``errors``, for an invalid one.
    """
    if value is None:
        if field.nullable or field.kind == "any":
            return None
        errors.append(f"'{written}': expected {field.expected}, got null")
        return None
    try:
        out = KINDS[field.kind][1](value)
    except HCWFConfigError as exc:   # carries its own message (e.g. an unset ${ENV_VAR}); say it as written
        errors.append(f"'{written}': {exc}")
        return None
    except (ValueError, TypeError) as exc:
        detail = f" ({exc})" if field.kind == "mask" and str(exc) else ""
        errors.append(f"'{written}': expected {field.expected}, got {value!r}{detail}")
        return None
    if field.choices and out not in field.choices:
        errors.append(f"'{written}': expected {field.expected}, got {value!r}.{suggest(out, field.choices)}")
        return None
    return out


def normalise(flat: dict, origin: dict, errors: list[str]) -> dict:
    """Normalise every given value.

    Parameters
    ----------
    flat : dict
        The raw values by flat key.
    origin : dict of str to str
        The path each key was written under (see `flatten_sections`).
    errors : list of str
        Receives a message for each invalid value.

    Returns
    -------
    dict
        The normalised values; a key that failed is dropped.
    """
    out = {}
    for key, value in flat.items():
        field = BY_KEY[key]
        before = len(errors)
        checked = normalise_value(field, value, origin.get(key, key), errors)
        if len(errors) == before:
            out[key] = checked
    return out


def apply_defaults(config: dict, errors: list[str], skip=()) -> dict:
    """Fill the defaults into a config.

    Parameters
    ----------
    config : dict
        The normalised values by flat key.
    errors : list of str
        Receives a message for each missing required key.
    skip : iterable of str, default ()
        Required keys that may stay missing without an error.

    Returns
    -------
    dict
        A new dict with the default of every key left out (callable defaults are called with the dict built so far);
        keys with ``fill=False`` stay out.
    """
    out = dict(config)
    for field in FIELDS:
        if field.key in out:
            continue
        if field.required:
            if field.key in skip:
                continue
            errors.append(f"missing required key '{field.path}': {field.help}")
        elif field.fill:
            default = field.default(out) if callable(field.default) else field.default
            out[field.key] = default
    return out


def raise_errors(errors: list[str], source: str) -> None:
    """Raise one error listing every problem found, if there are any.

    Parameters
    ----------
    errors : list of str
        The messages collected.
    source : str
        What was read (a file name), named in the message.

    Raises
    ------
    ConfigSchemaError
        ``errors`` is not empty.
    """
    if errors:
        report = "\n".join(f"  • {e}" for e in errors)
        raise ConfigSchemaError(f"Invalid config {source}:\n{report}")
