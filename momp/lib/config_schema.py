"""The schema of a ROMP YAML run config: every key with its type, allowed values, default and help.

A YAML config is a mapping whose keys are either the flat config keys (the names used in
config.in, e.g. ``obs_dir``) or sections that group them for readability::

    observations:  dataset, name, path, file_pattern, variable, units, dims, day_boundary
    reference:     dataset, model, path, file_pattern, variable, units, dims, climatology_cv
    models:        a list of {dataset, name, path, file_pattern, variable, units, dims,
                   lead_label, accumulation, daily_boundary, members}
    onset:         definition, override, rule, season, threshold_years, thresholds, lead_label, check_alignment,
                   dry_spell_check, forecast_history, threshold_source, backend,
                   percentage_threshold, explain_cells
    evaluation:    start_date, end_date, start_year_clim, end_year_clim, init_days, date_filter_year,
                   verification_windows, tolerances, max_forecast_day, day_bins, probabilistic, members
    metrics:       FAR, MAE, MR, BS, RPS, AUC, Reliability, skill_score
    output:        data_dir, figure_dir, save_* , plot_*, show_*

``flatten_sections`` maps a section entry to its flat key (``FIELDS`` holds the mapping),
``normalise`` turns every value into the form the rest of ROMP expects (tuples, not lists;
dates as (year, month, day)), and ``apply_defaults`` fills the keys left out. All problems
are collected and raised together as one ConfigSchemaError: unknown keys come with a
"did you mean" suggestion, wrongly typed values name the key and the form expected.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Any, Callable

from momp.lib.assertion import ROMPConfigError
from momp.lib.dataset_schema import expand_env, suggest

DIM_NAMES = ("init_time", "step", "member", "lat", "lon")
"""Canonical forecast dimension names that ``dims`` may rename."""
SECTIONS = {
    "observations": "the observation dataset",
    "reference": "the reference model of the skill scores (climatology by default)",
    "models": "the forecast models, a list",
    "onset": "what counts as onset and how it is searched",
    "evaluation": "the period, lead windows and metrics settings",
    "metrics": "which metrics to compute",
    "output": "output folders and which files and figures to write",
}
"""The sections a YAML config may use, with one-line help."""
LIST_SECTIONS = ("models",)
"""Sections that are a list of entries, not a mapping of keys."""


class ConfigSchemaError(ROMPConfigError):
    """A YAML config has unknown keys, wrongly typed values or missing required keys; all are listed."""


class _Required:
    """Marker for 'no default: the key must be given'."""

    def __repr__(self):
        """Print as REQUIRED."""
        return "REQUIRED"


REQUIRED = _Required()


@dataclass(frozen=True)
class Field:
    """One config key.

    key is the flat key used by ROMP; section/name give where it may be written nested
    (name defaults to key). fill=False keys are left out when not given (ROMP chooses the
    value itself, e.g. by mode); a callable default is called with the config built so far.
    """

    key: str
    kind: str
    help: str
    default: Any = None
    section: str | None = None
    name: str | None = None
    choices: tuple | None = None
    nullable: bool = False
    fill: bool = True
    internal: bool = False

    @property
    def nested_name(self) -> str:
        """The key's name inside its section."""
        return self.name or self.key

    @property
    def path(self) -> str:
        """The key as written nested, e.g. 'evaluation.start_date', or the flat key when it has no section."""
        return f"{self.section}.{self.nested_name}" if self.section else self.key

    @property
    def required(self) -> bool:
        """Whether the key must be given."""
        return self.default is REQUIRED

    @property
    def expected(self) -> str:
        """The form of value expected, for error messages and help."""
        text = KINDS[self.kind][0]
        if self.choices:
            text = "one of " + ", ".join(repr(c) for c in self.choices)
        return text + (" or null" if self.nullable else "")


def _is_int(value) -> bool:
    """True for a whole number that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value):
    """Text."""
    if not isinstance(value, str):
        raise ValueError
    return value


def _path(value):
    """A path, with ${ENV_VAR} references expanded."""
    return expand_env(_text(value))


def _name(value):
    """A name written as text or a whole number (season names may be 1 or 2), returned as text."""
    if _is_int(value):
        return str(value)
    return _text(value)


def _integer(value):
    """A whole number."""
    if not _is_int(value):
        raise ValueError
    return value


def _number(value):
    """A number."""
    if not _is_number(value):
        raise ValueError
    return value


def _workers(value):
    """A number of workers: a whole number of 1 or more."""
    if not _is_int(value) or value < 1:
        raise ValueError
    return value


def _boolean(value):
    """True or false."""
    if not isinstance(value, bool):
        raise ValueError
    return value


def _date(value):
    """A date as [year, month, day], 'YYYY-MM-DD' or a YAML date, returned as a (year, month, day) tuple."""
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
    """A month and day as [month, day] or 'MM-DD', returned as a (month, day) tuple."""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d{1,2})-(\d{1,2})\s*", value)
        parts = tuple(int(g) for g in match.groups()) if match else None
    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_int(v) for v in value):
        parts = tuple(value)
    else:
        parts = None
    if parts is None:
        raise ValueError
    datetime.date(2001, *parts)
    return parts


def _pair(value, sep="-"):
    """A pair of whole numbers as [a, b] or 'a-b', returned as a tuple."""
    if isinstance(value, str):
        match = re.fullmatch(rf"\s*(\d+)\s*{sep}\s*(\d+)\s*", value)
        if match:
            return tuple(int(g) for g in match.groups())
    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_int(v) for v in value):
        return tuple(value)
    raise ValueError


def _windows(value):
    """A list of [first, last] day pairs (or 'first-last' strings), returned as a tuple of tuples."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError
    return tuple(_pair(v) for v in value)


def _integers(value):
    """A non-empty list of whole numbers, returned as a tuple."""
    if not isinstance(value, (list, tuple)) or not value or not all(_is_int(v) for v in value):
        raise ValueError
    return tuple(value)


def _texts(value):
    """A list of text, returned as a tuple."""
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ValueError
    return tuple(value)


def _paths(value):
    """A list of paths with ${ENV_VAR} expanded, returned as a tuple."""
    return tuple(_path(v) for v in _texts(value))


def _patterns(value):
    """One file pattern or a list of them, returned as a tuple."""
    return (_text(value),) if isinstance(value, str) else _texts(value)


def _numbers_or_null(value):
    """A list of numbers or null (no conversion), returned as a tuple."""
    if not isinstance(value, (list, tuple)) or not all(v is None or _is_number(v) for v in value):
        raise ValueError
    return tuple(value)


def _dims(value):
    """A mapping of canonical dimension names (init_time, step, member, lat, lon) to the file's names."""
    if not isinstance(value, dict) or not all(k in DIM_NAMES + ("time",) and isinstance(v, str) and v for k, v in value.items()):
        raise ValueError
    return dict(value)


def _dims_list(value):
    """A list with one dims mapping (or null) per model, returned as a tuple."""
    if not isinstance(value, (list, tuple)):
        raise ValueError
    return tuple(None if v is None else _dims(v) for v in value)


def _members(value):
    """'All', a range such as '0-9', or a list of whole numbers (returned as a tuple)."""
    if isinstance(value, str):
        if value == "All" or re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", value):
            return value
        raise ValueError
    return _integers(value)


def _cells(value):
    """A list of [lat, lon] pairs, returned as a list of tuples."""
    if not isinstance(value, (list, tuple)) or not all(
            isinstance(c, (list, tuple)) and len(c) == 2 and all(_is_number(v) for v in c) for c in value):
        raise ValueError
    return [tuple(c) for c in value]


def _thresholds(value):
    """A mapping of threshold names to 'path.nc:variable' text."""
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError
    return dict(value)


def _years(value):
    """'All', a list of years (returned as a tuple), or null."""
    if value == "All":
        return value
    return _integers(value)


MASK_KEYS = {
    "land": ("true or false", "keep land cells only (Natural Earth land polygons from cartopy)"),
    "country": ("a country name or a list of them", "keep the cells of these countries (Natural Earth "
                "admin_0_countries; matched on NAME, NAME_LONG or ADM0_A3, any case)"),
    "file": ("a path", "an existing mask: a 0/1 NetCDF file, or a shapefile / GeoJSON"),
    "variable": ("text", "the mask variable of a NetCDF file (auto-detected if the file has only one)"),
    "keep_value": ("a number", "the value of the cells to keep in the NetCDF mask (default 1)"),
    "condition": ("a mapping {file, variable, above, below}", "a mask made from a field, e.g. "
                  "{file: dem.nc, variable: elevation, above: 1500}: keeps the cells above and/or below the numbers"),
    "cell_rule": ("one of 'centre', 'overlap_fraction'", "how a shape picks cells: 'centre' (the cell centre is "
                  "inside; default) or 'overlap_fraction' (enough of the cell's area is inside)"),
    "overlap_fraction": ("a number in (0, 1]", "the share of a cell's area that must be inside the shape (or, for a "
                         "mask file with finer cells, covered by kept mask cells); default 0.5"),
    "resolution": ("one of '10m', '50m', '110m'", "Natural Earth resolution of land and country shapes "
                   "(default '50m')"),
}
"""The keys of the ``mask:`` section: expected form and help."""
_MASK_CONDITION_KEYS = ("file", "variable", "above", "below")


def _mask(value):
    """A mask section: a mapping of land, country, file, variable, keep_value, condition, cell_rule, ...

    Returns a plain mapping of the keys given (country always as a list, file paths with ${ENV_VAR} expanded).
    A bare ValueError message explains what is wrong.
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


def _anything(value):
    """Any value."""
    return value


KINDS: dict[str, tuple[str, Callable]] = {
    "str": ("text", _text),
    "name": ("a name (text or a whole number)", _name),
    "path": ("a path (text; ${ENV_VAR} is expanded)", _path),
    "int": ("a whole number", _integer),
    "number": ("a number", _number),
    "bool": ("true or false", _boolean),
    "workers": ("a whole number of 1 or more", _workers),
    "date": ("a date, [year, month, day] or 'YYYY-MM-DD'", _date),
    "month_day": ("a month and day, [month, day] or 'MM-DD'", _month_day),
    "year_pair": ("a pair of years, [first, last] or 'first-last'", _pair),
    "windows": ("a list of [first, last] day pairs, e.g. [[1, 15], [16, 30]]", _windows),
    "ints": ("a list of whole numbers, e.g. [3, 5]", _integers),
    "strs": ("a list of text", _texts),
    "paths": ("a list of paths", _paths),
    "patterns": ("a file pattern such as '{}.nc' (quote it), or a list of them", _patterns),
    "numbers_or_null": ("a list of numbers or null", _numbers_or_null),
    "dims": ("a mapping from init_time, step, member, lat, lon to the file's dimension names", _dims),
    "dims_list": ("a list of dims mappings or null, one per model", _dims_list),
    "members": ("'All', a range like '0-9', or a list of whole numbers", _members),
    "cells": ("a list of [lat, lon] pairs, e.g. [[9.0, 38.75]]", _cells),
    "thresholds": ("a mapping of threshold names to 'path.nc:variable'", _thresholds),
    "years": ("'All' or a list of years", _years),
    "mask": ("a mapping of land, country, file, variable, keep_value, condition, cell_rule, overlap_fraction, "
             "resolution", _mask),
    "any": ("any value", _anything),
}
"""Value kinds: a description of the form expected and the function that checks and normalises a value."""

_SIX_BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))
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


def _default_pkg_dir(_config) -> str:
    """The folder that contains the momp package (informational only)."""
    from pathlib import Path

    import momp

    return str(Path(momp.__file__).resolve().parent.parent)


def _fields() -> tuple[Field, ...]:
    """Build the table of every config key."""
    top = [
        Field("project_name", "str", "name shown in the run banner", "ROMP run"),
        Field("work_dir", "path", "folder relative paths are meant from (informational)", "."),
        Field("pkg_dir", "path", "your ROMP checkout (informational)", _default_pkg_dir),
        Field("layout", "strs", "dimensions of the processed cases; leave as is", ("model", "verification_window")),
        Field("dataset_files", "paths", "YAML files describing your datasets, referenced by id with `dataset:`", ()),
        Field("catalog_dirs", "paths", "folders of <id>.yaml dataset files, referenced by id with `dataset:`", ()),
        Field("region", "str", "target region id or alias, see `python -m momp.onset regions`", REQUIRED),
        Field("nc_mask", "path", "NetCDF mask of 1s and 0s limiting the evaluated cells", None, nullable=True),
        Field("mask", "mask", "cells to evaluate, on each dataset's own grid; the parts given intersect: "
              + "; ".join(f"{k}: {h}" for k, (_, h) in MASK_KEYS.items()), None, nullable=True, fill=False),
        Field("shpfile_dir", "path", "shapefile for the region boundary in maps", None, nullable=True),
        Field("polygon", "bool", "mask and boundary from region polygons (params/region_def)", False),
        Field("parallel", "bool", "compute years in several worker processes (false: one process); "
              "write `parallel: {workers: N}` to choose the number", True),
        Field("parallel_workers", "workers", "worker processes when parallel is on (parallel.workers); null: the "
              "cores available, never more than there are years", None, nullable=True, fill=False),
        Field("debug", "bool", "developer mode: keep going on invalid config", False),
        Field("years", "years", "reserved: years to evaluate instead of start/end dates", None, nullable=True),
        Field("years_clim", "years", "reserved: climatology years instead of start/end", None, nullable=True),
        Field("fallback_date", "month_day", "reserved: date buffer for onsets before the start date", None,
              nullable=True),
        # the models list, as flat keys (the `models:` section builds these)
        Field("model_list", "strs", "model names (use the `models:` section instead)", REQUIRED, internal=True),
        Field("model_dir_list", "paths", "forecast folder of each model", REQUIRED, internal=True),
        Field("model_var_list", "strs", "rainfall variable of each model", REQUIRED, internal=True),
        Field("unit_cvt_list", "numbers_or_null", "multiplier to mm of each model (derived from units)", REQUIRED, internal=True),
        Field("accumulation_list", "strs", "accumulation of each model (from models[].accumulation)", fill=False, internal=True),
        Field("daily_boundary_list", "strs", "daily boundary of each model, or null (from models[].daily_boundary)", fill=False, internal=True),
        Field("file_pattern_list", "strs", "file pattern of each model", REQUIRED, internal=True),
        Field("model_dims_list", "dims_list", "dimension names of each model's files", None, fill=False, internal=True),
        # old flat onset keys (legacy mode; they cannot be combined with onset.definition)
        Field("wet_init", "number", "legacy: rain (mm) of the first potential wet day", fill=False, nullable=True),
        Field("wet_threshold", "number", "legacy: total rain (mm) of the wet spell", fill=False, nullable=True),
        Field("wet_spell", "int", "legacy: days of the wet spell", fill=False, nullable=True),
        Field("dry_threshold", "number", "legacy: rain (mm) below which a day is dry", fill=False, nullable=True),
        Field("dry_spell", "int", "legacy: dry run (days) that rejects an onset", fill=False, nullable=True),
        Field("dry_extent", "int", "legacy: days after onset searched for the dry run", fill=False, nullable=True),
        Field("thresh_file", "path", "legacy: NetCDF of per-cell thresholds", fill=False, nullable=True),
        Field("thresh_var", "str", "legacy: variable in thresh_file", fill=False, nullable=True),
        Field("earliest_onset_date", "month_day", "legacy: earliest onset, [month, day]", fill=False, nullable=True),
    ]
    obs = [
        Field("obs", "str", "observation dataset name (default: 'obs')", "obs",
              "observations", "name"),
        Field("obs_dir", "path", "folder of the observation files", REQUIRED, "observations", "path"),
        Field("obs_file_pattern", "patterns", "file pattern, {} is the year", ("{}.nc",), "observations",
              "file_pattern"),
        Field("obs_var", "str", "rainfall variable in the files", REQUIRED, "observations", "variable"),
        Field("obs_unit_cvt", "number", "multiplier to mm (derived from units)", None, nullable=True,
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
        Field("climatology_cv", "str", "cross-validation of the conditional climatology: 'none' or "
              "'leave_one_year_out' (refit the KDE without the evaluated year); only used by ref_model "
              "'conditional_climatology'", "leave_one_year_out",
              "reference", choices=("none", "leave_one_year_out")),
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
    ]
    onset = [
        Field("onset_definition", "str", "registry id or path of a definition YAML (null: the region's)", None,
              "onset", "definition", nullable=True, fill=False),
        Field("onset_override", "any", "changes merged into the definition (same rules as `extends`), e.g. "
              "{season: {from: 05-15}}", None, "onset", "override", nullable=True, fill=False),
        Field("onset_rule", "any", "a complete definition written inline instead of a registry id", None, "onset",
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
        Field("onset_backend", "str", "detection engine", "numba", "onset", "backend",
              choices=("numba", "reference", "gpu")),
        Field("onset_percentage_threshold", "number", "share of members (0-1) that must have an onset", 0.5,
              "onset", "percentage_threshold"),
        Field("explain_cells", "cells", "cells to explain, [[lat, lon]] (null: off)", None, "onset",
              nullable=True),
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
    return tuple(top + obs + ref + onset + evaluation + metrics + output)


FIELDS = _fields()
"""Every config key, in documentation order."""
BY_KEY = {f.key: f for f in FIELDS}
BY_PATH = {(f.section, f.nested_name): f for f in FIELDS if f.section}
SPECIAL_SUBKEYS = {"observations": ("dataset", "units", "dims"), "reference": ("dataset", "units")}
"""Section entries that are not config keys: the dataset file entry to take values from, units, dims."""
MODEL_ENTRY_KEYS = ("dataset", "name", "path", "file_pattern", "variable", "units", "dims", "lead_label",
                    "accumulation", "daily_boundary", "members")
"""Keys of one entry of the `models:` list."""


def section_names(section: str) -> list[str]:
    """The names a section accepts."""
    if section == "models":
        return list(MODEL_ENTRY_KEYS)
    return [f.nested_name for f in FIELDS if f.section == section] + list(SPECIAL_SUBKEYS.get(section, ()))


def _where(name: str) -> str:
    """Where a bare key name belongs when written at the top level by mistake, e.g. 'section observations'."""
    homes = [f.section for f in FIELDS if f.section and f.nested_name == name]
    if name in MODEL_ENTRY_KEYS:
        homes.append("models")
    if name in SPECIAL_SUBKEYS.get("observations", ()):
        homes.append("observations")
    return ", ".join(dict.fromkeys(homes))


def unknown_key_message(name: str, section: str | None = None) -> str:
    """Error text for an unknown key, with a 'did you mean' suggestion and where it may belong."""
    if section:
        candidates = section_names(section)
        return (f"unknown key '{section}.{name}'.{suggest(name, candidates)} "
                f"Keys of {section}: {', '.join(candidates)}")
    candidates = [k for k, f in BY_KEY.items() if not f.internal] + list(SECTIONS)
    hint = suggest(name, candidates)
    home = _where(name)
    if home and not hint:
        hint = f' "{name}" belongs inside the section: {home}.'
    return f"unknown key '{name}'.{hint} Run `momp-run init` to see a config with the common keys"


def flatten_sections(raw: dict, errors: list[str]) -> tuple[dict, dict, dict]:
    """Flatten a YAML mapping to flat config keys.

    Returns (flat, origin, special): flat holds the raw values by flat key, origin the path each
    was written under (for messages), special the catalog-aware blocks {"observations": {...},
    "reference": {...}, "models": [...] or None}. Problems are appended to errors.
    """
    flat: dict = {}
    origin: dict = {}
    special: dict = {"observations": {}, "reference": {}, "models": None}

    def put(field: Field, value, written: str):
        """Store a raw value under its flat key, once."""
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
    """Check one value against its field and return it in ROMP's form; a problem is appended to errors."""
    if value is None:
        if field.nullable or field.kind == "any":
            return None
        errors.append(f"'{written}': expected {field.expected}, got null")
        return None
    try:
        out = KINDS[field.kind][1](value)
    except (ValueError, TypeError) as exc:
        detail = f" ({exc})" if field.kind == "mask" and str(exc) else ""
        errors.append(f"'{written}': expected {field.expected}, got {value!r}{detail}")
        return None
    if field.choices and out not in field.choices:
        errors.append(f"'{written}': expected {field.expected}, got {value!r}.{suggest(out, field.choices)}")
        return None
    return out


def normalise(flat: dict, origin: dict, errors: list[str]) -> dict:
    """Normalise every given value; keys that failed are dropped (the error is in errors)."""
    out = {}
    for key, value in flat.items():
        field = BY_KEY[key]
        before = len(errors)
        checked = normalise_value(field, value, origin.get(key, key), errors)
        if len(errors) == before:
            out[key] = checked
    return out


def apply_defaults(config: dict, errors: list[str], skip=()) -> dict:
    """Fill defaults into config (a new dict); missing required keys (except those in skip) go to errors."""
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
    """Raise one ConfigSchemaError listing every error, if there are any."""
    if errors:
        report = "\n".join(f"  • {e}" for e in errors)
        raise ConfigSchemaError(f"Invalid config {source}:\n{report}")


def describe_schema() -> list[tuple[str, str, str, str]]:
    """Rows (path, expected form, default, help) for every key, for documentation and `--help` text."""
    rows = []
    for f in FIELDS:
        default = "required" if f.required else ("computed" if callable(f.default) else repr(f.default))
        if not f.fill and not f.required:
            default = "ROMP chooses"
        rows.append((f.path, f.expected, default, f.help))
    return rows


def describe_mask_schema() -> list[tuple[str, str, str, str]]:
    """Rows (path, expected form, default, help) for the keys inside the ``mask:`` section."""
    return [(f"mask.{key}", expected, "unset", help_) for key, (expected, help_) in MASK_KEYS.items()]
