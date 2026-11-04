"""Convert onset_blending configs to HCWF run YAML: a BlendConfig YAML (``configs/*.yml``) or a legacy spec set.

``hcwf-run convert blend.yml -o run.yaml`` and ``hcwf-run convert --specs ... -o run.yaml`` write the equivalent run
config: the sources, thresholds, units and onset options into HCWF's own sections, the settings only a blend has into
the ``blend:`` section (`hcwf.blend.recipe.recipe_from_config` reads it back), and a comment for everything that is not
carried over. Nothing of onset_blending is imported and no BlendConfig object is kept: the converter reads the YAML
mapping, fills blending's defaults and writes the HCWF mapping.

What each blending part becomes (the parity options make HCWF read the data exactly as blending did):

* ``observations.files`` / ``models[].files``: ``paths`` (a regular expression in a folder becomes the equivalent glob;
  one a glob cannot say lists the files that match now) and ``years``; ``unit_scale`` is ``unit_cvt`` (files that declare
  units are converted by them, as before), observations are millimetres; ``dims`` map file dimensions to roles.
* the onset rule: an inline ``onset.rule`` (blending's wet-day, trigger, follow-up and dry-spell options in HCWF's
  definition language, `rule_body`); ``rain_resolution: exact``, ``negatives: keep``, ``gaps: refuse``,
  ``horizon: truncate``, ``missing`` = ``observations.missing_rain``, ``lead_label: end``,
  ``forecast_history: none``; ``observations.date_marks`` is the day boundary. The registry ids
  ``moron_robertson_blend`` and ``icpac_blend`` are these readings of the two papers.
* thresholds: a number (``thresholds: {thr: mm}``), a table (``threshold_table`` names its columns), a scale
  (``threshold_scale``), or computed from the observations (the rule's ``mean total over`` with
  ``threshold_max_missing: 1`` and ``threshold_partial_years``, as blending counts partial years).
* ``domain``: the ``units:`` section (box, outline, dissemination, centroids, shapefile, per-source modes and weights).
* rain-window tokens (``follow``): blending's number is kept exactly; a token whose reading in the HCWF definition
  differs becomes the number (blending's ``follow`` is ``follow_days``, the definition's span is one day longer for
  ``followup_anchor: onset_day``).
* the rest (years, formulas, CV, baselines, MME, forest, climatology, features, reference onset, fixed cutoff, per-model
  lead days, rain horizon and predictors): the ``blend:`` section.

The first half of the module converts a BlendConfig (`convert_blend`, `convert_blend_file`, the private `_Convert`);
the second half ports the legacy spec sets of ``onset_blending.legacy.specs`` (`convert_specs`, `convert_spec_files`,
the private `_Converter`), which build a BlendConfig mapping and hand it to `convert_blend`. The result is a
`Conversion`, which ``hcwf-run convert`` writes out. `validate_conversion` checks a conversion against
`hcwf.config.schema` without reading any data file.
"""
from __future__ import annotations

import os
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hcwf.blend.recipe import CLIMATOLOGY_FEATURE, years_of, year_windows
from hcwf.errors import ConvertError
from hcwf.config.datasets import suggest

__all__ = ["ConvertError", "Conversion", "convert_blend", "convert_blend_file", "convert_spec_files", "convert_specs",
           "is_blend_config", "regex_to_glob", "rule_body", "resolve_rule"]

DERIVED = "derived from the config"
"""Reason given for dropping a key that the new config derives itself."""
UNUSED = "no code reads it"
"""Reason given for dropping a key that no code of the old pipeline read."""
RUNTIME = "a run setting, not part of the blend"
"""Reason given for dropping a run setting (workers, parallel) that is not part of the blend."""


@dataclass
class Conversion:
    """The converted run: its mapping (the YAML), a comment per key path, and what was not carried over."""

    mapping: dict
    """The run config as a nested mapping, in the order it is written to YAML."""
    notes: tuple[str, ...] = ()
    """What was not carried over, one sentence each (written as ``# Not carried over`` comments)."""
    comments: dict = field(default_factory=dict)
    """A comment per key path: ``(key,)`` for a top-level key or ``(section, key)`` for a key of a section."""

    def to_yaml(self, sources: dict[str, str] | None = None, origin: str = "onset_blending") -> str:
        """Render the run YAML, headed by where it came from and what was not carried over.

        Parameters
        ----------
        sources : dict of str to str, optional
            Labelled paths of the files converted (for example ``{"BlendConfig": "configs/a.yml"}``), listed in the
            header; None lists none.
        origin : str, default "onset_blending"
            Name of the project the config came from, in the first header line.

        Returns
        -------
        str
            The YAML text, ending in a newline.
        """
        lines = [f"# Converted from {origin} by `hcwf-run convert`. Check it, then run:  hcwf-run -p <this file>"]
        lines += [f"#   {label}: {path}" for label, path in (sources or {}).items()]
        if self.notes:
            lines.append("# Not carried over:")
            lines += [f"#   - {note}" for note in self.notes]
        return _render(self.mapping, self.comments, "\n".join(lines))


def _flow(value) -> str:
    """Dump one value as single-line YAML, keys in the order written.

    Parameters
    ----------
    value : object
        A YAML-serialisable value.

    Returns
    -------
    str
        The flow-style YAML of `value`, without the document end marker or the trailing newline.
    """
    text = yaml.safe_dump(value, default_flow_style=True, width=10 ** 9, allow_unicode=True, sort_keys=False)
    return (text[:-5] if text.endswith("\n...\n") else text).strip()


def _simple(value) -> bool:
    """Tell whether a value reads well on one line: a scalar, or a short list or mapping of scalars.

    Parameters
    ----------
    value : object
        The value to be written.

    Returns
    -------
    bool
        True for a scalar, a mapping of at most 6 scalars, or a list of at most 40 items that are scalars or lists
        of at most 2 items.
    """
    if isinstance(value, dict):
        return len(value) <= 6 and all(not isinstance(v, (dict, list)) for v in value.values())
    if isinstance(value, list):
        return len(value) <= 40 and all(not isinstance(v, (dict, list)) or (isinstance(v, list) and len(v) <= 2)
                                        for v in value)
    return True


def _render(mapping: dict, comments: dict, header: str) -> str:
    """Render the run mapping as YAML: a key per line, nested values in block style, a ``# comment`` where there is one.

    The sections ``observations``, ``onset``, ``units``, ``blend`` and ``evaluation`` are written one key per line under
    the section name; every other top-level key is written on its own line.

    Parameters
    ----------
    mapping : dict
        The run mapping.
    comments : dict
        Comments by key path: ``(key,)`` for a top-level key, ``(section, key)`` for a key of a section.
    header : str
        Comment lines written first; empty for none.

    Returns
    -------
    str
        The YAML text, ending in a newline.
    """
    out = [header.rstrip()] if header else []

    def put(indent: str, key, value, note: str | None) -> None:
        """Append one key (after its comment, if any) to `out`."""
        if note:
            out.append(f"{indent}# {note}")
        if _simple(value):
            out.append(f"{indent}{key}: {_flow(value)}")
        else:
            body = yaml.safe_dump({key: value}, default_flow_style=False, sort_keys=False, width=110,
                                  allow_unicode=True)
            out.extend(indent + line for line in body.rstrip("\n").split("\n"))

    for key, value in mapping.items():
        if isinstance(value, dict) and key in ("observations", "onset", "units", "blend", "evaluation"):
            out.append(f"{key}:")
            for sub, item in value.items():
                put("  ", sub, item, comments.get((key, sub)))
        else:
            put("", key, value, comments.get((key,)))
    return "\n".join(out) + "\n"


# -- the keys of a BlendConfig (unknown keys are refused, with a suggestion) -----------------------------------------

_FILES = {"folder": None, "pattern": None, "paths": None, "years": None}
_TABLE = {"file": None, "id_col": None}
_BLEND_KEYS = {
    "name": None, "data_root": None,
    "domain": {
        "grid_ids": {"decimal_digits": None, "format": None},
        "regrid": {"shapefile": None, "unit_id_col": None, "unit_name_col": None, "parent_id_col": None, "crs": None,
                   "grid_lat_var": None, "grid_lon_var": None, "clip_to_coverage": None, "weights": None},
        "dissemination": _TABLE, "dissemination_scope": None, "centroids": _TABLE,
        "area_of_interest": {"bbox": {"lat_min": None, "lat_max": None, "lon_min": None, "lon_max": None},
                             "geometry": None}},
    "observations": {"files": _FILES, "variable": None, "dims": None, "missing_rain": None, "cell_transform": None,
                     "date_marks": None},
    "models": [{"name": None, "files": _FILES, "variable": None, "dims": None, "unit_scale": None, "max_member": None,
                "lead_days": {"min": None, "max": None}, "rain_horizon": {"max_day": None, "policy": None},
                "rain_predictors": [{"agg": None, "window": None}], "rain_transform": None, "cell_transform": None,
                "regrid_weights": None}],
    "onset": {
        "definition": {"window": None, "trigger_rule": None, "min_wet_days": None, "wet_day_min_mm": None,
                       "wet_day_comparison": None, "threshold_comparison": None, "follow_days": None,
                       "followup_anchor": None,
                       "dry_spell": {"mode": None, "min_dry_days": None, "dry_day_min_mm": None, "contained": None,
                                     "sum_window": None, "sum_min_mm": None}},
        "threshold": {"mm": None, "file": None,
                      "climatology": {"days": None, "start": None, "end": None, "years": None},
                      "id_col": None, "value_col": None, "lat_col": None, "lon_col": None, "lat_var": None,
                      "lon_var": None, "value_var": None, "mat_lat_name": None, "mat_lon_name": None,
                      "mat_value_name": None, "scale": {"factor": None, "offset": None, "min": None, "max": None}},
        "season_start": None, "fixed_cutoff": None,
        "reference_onset": {"month_day": None, "file": None, "date_col": None, "year_col": None, "day_col": None,
                            "base_date": None, "unit_col": None}},
    "climatology": {"issue_end": None, "forecast_window": None, "horizons": None, "cv_by_year": None,
                    "min_onset_years": None, "train_years": None, "test_years": None, "windows": None,
                    "workers": None},
    "features": {"bins": {"count": None, "days": None}, "probability_days": None, "join": None,
                 "trim_after_true_onset": None, "rain_transform": None},
    "evaluation": {"years": {"training": None, "cv_holdout": None, "true_holdout": None}, "cutoff_mode": None,
                   "cv_methods": None, "cluster_var": None, "formulas": None, "final_formula": None,
                   "window_variants": None, "climatology_baselines": None, "forecasts": None, "mme": None,
                   "external_predictions": None, "random_forest": None},
    "outputs": {"dir": None, "intermediates": None, "pickles": None},
}

_WEIGHTS_NOTE = ("evaluation.mme.weights_year_tag {tag!r}: saved MME weights are no longer found by guessing a results folder; "
                 "give them with blend.mme.weights (a mapping of source name to weight, or the path of the "
                 "mme_optimized_weights_*.csv a run wrote)")

_RULE_DEFAULTS = {"window": 5, "trigger_rule": "all_days_wet", "min_wet_days": None, "wet_day_min_mm": 1.0,
                  "wet_day_comparison": "gte", "threshold_comparison": "gt", "follow_days": 21,
                  "followup_anchor": "after_trigger"}
_SPELL_DEFAULTS = {"mode": "consecutive_dry", "min_dry_days": 7, "dry_day_min_mm": None, "contained": False,
                   "sum_window": 10, "sum_min_mm": 5.0}
_THRESHOLD_DEFAULTS = {"id_col": None, "value_col": "onset_thresh", "lat_col": "lat", "lon_col": "lon",
                       "lat_var": "lat", "lon_var": "lon", "value_var": "MWmean", "mat_lat_name": "lat",
                       "mat_lon_name": "lon", "mat_value_name": "onset.day.thres"}
_OBS_ROLES = {"time": "time", "lat": "lat", "lon": "lon", "adm3_name": "unit"}
_MODEL_ROLES = {"time": "init_time", "day": "step", "number": "member", "lat": "lat", "lon": "lon", "adm3_name": "unit"}
_WINDOW_TOKENS = ("trigger", "window", "dry_spell", "sum_window", "follow", "min_dry")
_DEFINITION_TOKEN = {"trigger": "trigger", "window": "trigger", "dry_spell": "dry_window", "sum_window": "dry_window",
                     "follow": "follow", "min_dry": "dry_run"}


def _unknown(node, spec, path: str, problems: list[str]) -> None:
    """Report the keys of `node` that the spec of a BlendConfig section does not have.

    Parameters
    ----------
    node : object
        The loaded YAML value of the section (a mapping, a list of mappings or anything else, which is skipped).
    spec : dict or list or None
        The known keys: a dict of key to the spec of its value, a one-item list for a list of mappings.
    path : str
        The dotted path of `node`, empty at the top; prefixes the problem messages.
    problems : list of str
        Appended with one message per unknown key, with a suggestion when there is a close key.
    """
    if isinstance(spec, list):
        if node is None:
            return
        for index, item in enumerate(node if isinstance(node, (list, tuple)) else [node]):
            _unknown(item, spec[0], f"{path}[{index}]", problems)
        return
    if not isinstance(spec, dict) or not isinstance(node, dict):
        return
    for key, value in node.items():
        if key not in spec:
            problems.append(f"{path + '.' if path else ''}{key}: unknown key.{suggest(key, spec)}")
        else:
            _unknown(value, spec[key], f"{path + '.' if path else ''}{key}", problems)


def _dig(node, path: str, default=None):
    """Read ``node[a][b]...`` for the dotted path ``"a.b..."``.

    Parameters
    ----------
    node : object
        The mapping to read from.
    path : str
        Dotted keys.
    default : object, optional
        What to return when a step is not a mapping, is missing or is null.

    Returns
    -------
    object
        The value at `path`, or `default`.
    """
    for part in path.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            return default
        node = node[part]
    return node


# -- the onset rule ------------------------------------------------------------------------------------------------


def resolve_rule(definition: dict | None) -> types.SimpleNamespace:
    """Resolve the onset rule of a BlendConfig ``onset.definition`` mapping, with blending's defaults filled in.

    A null value counts as missing and takes the default.

    Parameters
    ----------
    definition : dict, optional
        The ``onset.definition`` mapping (with its ``dry_spell`` sub-mapping); None or empty is all defaults.

    Returns
    -------
    types.SimpleNamespace
        The rule, with attributes ``window``, ``wet_day_min_mm`` (mm), ``trigger_rule``, ``min_wet_days``,
        ``wet_day_comparison``, ``threshold_comparison``, ``follow_days``, ``followup_anchor``, ``mode``,
        ``min_dry_days``, ``dry_day_min_mm`` (mm; the wet-day limit unless given), ``dry_run_contained``,
        ``sum_window`` and ``sum_min_mm`` (mm). Days are ``int``, amounts ``float``.
    """
    definition = definition or {}
    spell = {**_SPELL_DEFAULTS, **{k: v for k, v in (definition.get("dry_spell") or {}).items() if v is not None}}
    rule = {**_RULE_DEFAULTS, **{k: v for k, v in definition.items() if k != "dry_spell" and v is not None}}
    wet = float(rule["wet_day_min_mm"])
    return types.SimpleNamespace(
        window=int(rule["window"]), wet_day_min_mm=wet, trigger_rule=rule["trigger_rule"],
        min_wet_days=rule["min_wet_days"], wet_day_comparison=rule["wet_day_comparison"],
        threshold_comparison=rule["threshold_comparison"], follow_days=int(rule["follow_days"]),
        followup_anchor=rule["followup_anchor"], mode=spell["mode"], min_dry_days=int(spell["min_dry_days"]),
        dry_day_min_mm=float(spell["dry_day_min_mm"] if spell["dry_day_min_mm"] is not None else wet),
        dry_run_contained=bool(spell["contained"]), sum_window=int(spell["sum_window"]),
        sum_min_mm=float(spell["sum_min_mm"]))


def resolve_token(token: str, rule) -> int:
    """Resolve a rain predictor's window token to days, as blending reads it from the rule.

    Parameters
    ----------
    token : str
        One of ``trigger``, ``window`` (the rule's ``window``), ``dry_spell``, ``sum_window`` (its ``sum_window``),
        ``follow`` (its ``follow_days``) or ``min_dry`` (its ``min_dry_days``); case is ignored.
    rule : types.SimpleNamespace
        The rule as `resolve_rule` returns it.

    Returns
    -------
    int
        The window in days.

    Raises
    ------
    ValueError
        `token` is not a known token.
    """
    token = token.lower()
    if token in ("trigger", "window"):
        return rule.window
    if token in ("dry_spell", "sum_window"):
        return rule.sum_window
    if token == "follow":
        return rule.follow_days
    if token == "min_dry":
        return rule.min_dry_days
    raise ValueError(f"unknown rain window token {token!r}")


def rule_body(r, threshold: tuple | None = None) -> str:
    """Write the wet-day, dry-day, thresholds and onset parts of an HCWF definition for the rule `r`.

    The definition searches as onset_blending's ``find_onsets`` does. The total the trigger needs is the threshold
    ``thr``, the mean total over the trigger window unless `threshold` says over which days of each year it is computed
    from the observations.

    Parameters
    ----------
    r : types.SimpleNamespace
        The rule, as `resolve_rule` returns it.
    threshold : tuple of (int, str, str), optional
        ``(days, first month-day, last month-day)``: ``thr`` is the mean total over ``days`` days during those dates of
        each year; None is the mean total over the trigger window.

    Returns
    -------
    str
        The YAML text of the definition body (``wet_day``, ``dry_day``, ``thresholds`` and ``onset``), ending in a
        newline.
    """
    wet = f"{'at least' if r.wet_day_comparison == 'gte' else 'more than'} {r.wet_day_min_mm} mm"
    cmp_ = "at least" if r.threshold_comparison == "gte" else "more than"
    total = f"{cmp_} thr"
    require = ["{all days: wet}"] if r.trigger_rule == "all_days_wet" else ["{first day: wet}"]
    if r.min_wet_days is not None:
        require.append(f"{{wet days: at least {r.min_wet_days}}}")
    require.append(f"{{total: {total}}}")
    if r.followup_anchor == "after_trigger":
        span = f"{r.follow_days} days after window"
    else:
        span = f"days 1 to {r.follow_days + 1}"
    length = r.min_dry_days if r.mode == "consecutive_dry" else r.sum_window
    contained = r.mode == "window_sum" or r.dry_run_contained
    if r.follow_days == 0 or (contained and r.follow_days < length):
        veto = "[]"
    elif r.mode == "window_sum":
        veto = f"[{{dry window: {r.sum_window} days with less than {r.sum_min_mm} mm, within: {span}}}]"
    else:
        place = "within" if r.dry_run_contained else "beginning within"
        veto = f"[{{dry run: at least {r.min_dry_days} days, {place}: {span}}}]"
    if threshold is None:
        thresholds = "thresholds: {thr: {mean total over: %d days}}\n" % r.window
    else:
        thresholds = "thresholds: {thr: {mean total over: %d days, during: %s to %s}}\n" % threshold
    return (f"wet_day: {wet}\ndry_day: less than {r.dry_day_min_mm} mm\n{thresholds}"
            f"onset:\n  window: {r.window} days\n  require: [{', '.join(require)}]\n  reject if: {veto}\n")


def _rule_mapping(rule, name: str, season_start: str, threshold: tuple | None) -> dict:
    """Build the inline ``onset.rule`` of a converted config: a complete definition (see `rule_body`).

    Parameters
    ----------
    rule : types.SimpleNamespace
        The rule, as `resolve_rule` returns it.
    name : str
        The name of the BlendConfig; it makes the rule id and title.
    season_start : str
        The first ``MM-DD`` of the season (``onset.season_start``).
    threshold : tuple of (int, str, str), optional
        The computed-threshold window, as `rule_body` takes it.

    Returns
    -------
    dict
        The definition, parsed from its YAML text.
    """
    head = (f"id: {re.sub(r'[^a-z0-9_]+', '_', name.lower()).strip('_') or 'blend'}_rule\nversion: 1\n"
            f"title: The onset rule of the onset_blending config {name}\ncitation: onset_blending\n"
            "wording: Converted from onset_blending's onset.definition; see the config it came from.\n"
            "status: draft\nnotes: Written by `hcwf-run convert` so that HCWF searches exactly as onset_blending did "
            "(to the end of each season's data; season.from is onset.season_start).\n"
            f"season: {{from: {season_start}, to: 12-31}}\n")
    return yaml.safe_load(head + rule_body(rule, threshold=threshold))


# -- files: folder and regular expression -> glob -----------------------------------------------------------------


_YEAR4 = re.compile(r"\(19\|20\)\\d\{2\}")
_DIGITS = re.compile(r"(?:\\d|\[0-9\])\{(\d+)\}")


def regex_to_glob(pattern: str) -> str | None:
    r"""Convert a regular expression on file names to the shell glob that matches the same names.

    Handles literal text (with escaped dots), ``.*``, ``.``, ``^``, ``$``, ``\d{n}``, ``[0-9]{n}`` and
    ``(19|20)\d{2}``. The glob matches the same names as ``re.search(pattern, name)``.

    Parameters
    ----------
    pattern : str
        The regular expression.

    Returns
    -------
    str or None
        The glob, or None when the regular expression says more than a glob can (alternatives, repeats).
    """
    text = pattern
    start = text.startswith("^")
    text = text[1:] if start else text
    end = text.endswith("$") and not text.endswith("\\$")
    text = text[:-1] if end else text
    out, i = [], 0
    while i < len(text):
        rest = text[i:]
        if match := _YEAR4.match(rest):
            out.append("[0-9][0-9][0-9][0-9]")
        elif match := _DIGITS.match(rest):
            out.append("[0-9]" * int(match.group(1)))
        elif rest.startswith(".*") and rest[2:3] not in ("?", "*", "+"):
            out.append("*")
            i += 2
            continue
        elif rest[0] == "\\" and len(rest) > 1 and rest[1] in ".-_/":
            out.append(rest[1])
            i += 2
            continue
        elif rest[0] == "." and rest[1:2] not in ("*", "+", "?", "{"):
            out.append("?")
            i += 1
            continue
        elif rest[0].isalnum() or rest[0] in "_-":
            if rest[1:2] in ("*", "+", "?", "{"):
                return None
            out.append(rest[0])
            i += 1
            continue
        else:
            return None
        i += match.end()
    glob = ("" if start else "*") + "".join(out) + ("" if end else "*")
    return re.sub(r"\*+", "*", glob)


def _years_value(years):
    """Write a years list as the compact text that ``parse_years`` reads, or as a single year.

    Parameters
    ----------
    years : sequence of int
        The years, ascending and each once; not empty.

    Returns
    -------
    int or str
        The year itself for one year, else runs as text such as ``"2000:2024"`` or ``"2000:2010,2015"``.
    """
    if len(years) == 1:
        return years[0]
    runs, first = [], years[0]
    for before, year in zip(years, [*years[1:], None]):
        if year != before + 1 if year is not None else True:
            runs.append((first, before))
            first = year
    return ",".join(f"{a}:{b}" if a != b else f"{a}" for a, b in runs)


# -- the converter -------------------------------------------------------------------------------------------------


class _Convert:
    """One conversion: reads a BlendConfig mapping, keeps the problems and notes found on the way.

    Parameters
    ----------
    data : dict
        The BlendConfig mapping (what its YAML holds).
    root : pathlib.Path, optional
        The folder relative paths start at; None is the working directory.
    region : str, optional
        An HCWF region id for the run; None is no region.

    Attributes
    ----------
    problems : list of str
        What cannot be converted; `_Convert.convert` raises `ConvertError` with them.
    notes : list of str
        What is not carried over.
    comments : dict
        Comments by key path, filled by `_Convert.convert`.
    definition_windows : dict
        The windows of the HCWF definition written from the onset rule, set by `_Convert._onset_with_windows` (empty
        when the definition cannot be read).
    """

    def __init__(self, data: dict, root: Path | None, region: str | None):
        self.data, self.root, self.region = data, root, region
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.comments: dict = {}

    # -- small helpers -----------------------------------------------------------------------------------------

    def get(self, path: str, default=None):
        """Read a dotted path of the BlendConfig mapping.

        Parameters
        ----------
        path : str
            Dotted keys, for example ``"evaluation.years.training"``.
        default : object, optional
            What to return when a step is missing or null.

        Returns
        -------
        object
            The value, or `default`.
        """
        return _dig(self.data, path, default)

    def path(self, value) -> str:
        """Anchor a configured path at the data root (absolute, as blending's ``config.path`` joined it).

        Parameters
        ----------
        value : str or pathlib.Path
            The path as written in the config.

        Returns
        -------
        str
            The normalised absolute path; a relative one starts at the data root, or the working directory without one.
        """
        value = Path(str(value))
        if not value.is_absolute():
            value = (self.root or Path.cwd()) / value
        return os.path.normpath(str(value))

    def note(self, text: str) -> None:
        """Record something that is not carried over.

        Parameters
        ----------
        text : str
            The sentence for the ``# Not carried over`` header.
        """
        self.notes.append(text)

    def years(self, value, where: str) -> tuple[int, ...]:
        """Parse a years value, recording a problem when it is invalid.

        Parameters
        ----------
        value : object
            A list, a year or text such as ``"2000:2024,2026"``; None is no years.
        where : str
            The config path named in a problem message.

        Returns
        -------
        tuple of int
            The years, sorted and each once; empty for None or an invalid value.
        """
        problems: list[str] = []
        years = years_of(value, where, problems)
        self.problems += problems
        return years

    # -- files ---------------------------------------------------------------------------------------------------

    def files(self, node: dict | None, what: str) -> dict:
        """Convert a blending FileSet to ``paths`` and ``years``: the files, in blending's (year, name) order when listed.

        Exactly one of ``folder`` (with ``pattern``, by default any ``.nc`` file, converted by `regex_to_glob`; a pattern
        a glob cannot say lists the files that match now) and ``paths`` must be given.

        Parameters
        ----------
        node : dict, optional
            The FileSet mapping (``folder``, ``pattern``, ``paths``, ``years``).
        what : str
            The source named in problem and note messages, for example ``"observations"``.

        Returns
        -------
        dict
            ``paths`` (a list of str) and, when given, ``years``; empty when the FileSet is invalid (a problem is
            recorded).
        """
        node = node or {}
        folder, listed, pattern = node.get("folder"), node.get("paths"), node.get("pattern", r"\.nc$")
        out: dict = {}
        if bool(folder) == bool(listed):
            self.problems.append(f"{what}.files: give either 'folder' (with 'pattern') or 'paths', not both")
            return out
        if listed:
            out["paths"] = [self.path(p) for p in listed]
        else:
            try:
                re.compile(pattern)
            except re.error as exc:
                self.problems.append(f"{what}.files.pattern: not a regular expression ({exc})")
                return out
            directory = self.path(folder)
            glob = regex_to_glob(pattern)
            if glob is not None:
                out["paths"] = [os.path.join(directory, glob)]
            else:   # the regular expression says more than a glob can: list what matches now
                found = sorted(p.name for p in Path(directory).iterdir() if p.is_file() and re.search(pattern, p.name)) \
                    if Path(directory).is_dir() else []
                self.note(f"{what}.files.pattern {pattern!r} is not a glob: the files that match it in {directory} "
                          f"{'were listed' if found else 'could not be listed (no such folder), so every file of it is read'}"
                          "; edit paths if the folder changes")
                out["paths"] = [os.path.join(directory, n) for n in found] or [os.path.join(directory, "*")]
        if node.get("years") not in (None, "", []):
            years = self.years(node["years"], f"{what}.files.years")
            if years:
                out["years"] = _years_value(years)
        return out

    @staticmethod
    def dims(rename, roles: dict) -> dict | None:
        """Convert blending's ``{file dimension: role}`` to HCWF's ``{role: file dimension}`` (roles blending reads only).

        Parameters
        ----------
        rename : dict, optional
            Blending's rename mapping of a file dimension to its role.
        roles : dict
            The role names blending reads (``_OBS_ROLES`` or ``_MODEL_ROLES``) mapped to HCWF's dimension names.

        Returns
        -------
        dict or None
            HCWF dimension name to the name in the files; None when there is none.
        """
        out = {roles[role]: name for name, role in (rename or {}).items() if role in roles}
        return out or None

    # -- onset ---------------------------------------------------------------------------------------------------

    def onset(self) -> tuple[dict, tuple, "types.SimpleNamespace"]:
        """Convert the ``onset`` section: the ``onset:`` mapping of the run, the threshold years and the rule.

        Exactly one of ``mm``, ``file`` and ``climatology`` of ``onset.threshold`` must be given (else a problem is
        recorded). A computed threshold (``climatology``) uses its own years; the others use the training years.

        Returns
        -------
        tuple of (dict, tuple of int or None, types.SimpleNamespace)
            The ``onset:`` mapping of the run, the ``(first, last)`` years the thresholds are computed from (None when
            there are none), and the rule as `resolve_rule` returns it.
        """
        node = self.get("onset", {})
        rule = resolve_rule(node.get("definition"))
        season_start = str(node.get("season_start", "05-01"))
        threshold = node.get("threshold") or {}
        obs = self.get("observations", {})
        out: dict = {}
        computed = threshold.get("climatology")
        given = [k for k in ("mm", "file", "climatology") if threshold.get(k) is not None]
        if len(given) != 1:
            self.problems.append("onset.threshold: give exactly one of 'mm', 'file' or 'climatology'")
        if threshold.get("mm") is not None and threshold.get("scale") is not None:
            self.problems.append("onset.threshold.scale: only applies to a threshold table")
        window = None
        if computed is not None:
            years = _year_pair(self.years(computed.get("years"), "onset.threshold.climatology.years"), self.problems,
                               "onset.threshold.climatology.years")
            window = (int(computed.get("days") or rule.window), str(computed.get("start", "04-01")),
                      str(computed.get("end", "10-31")))
        else:
            training = self.years(self.get("evaluation.years.training"), "evaluation.years.training")
            years = (min(training), max(training)) if training else None
        out["rule"] = _rule_mapping(rule, str(self.data.get("name", "blend")), season_start, window)
        if threshold.get("scale") is not None:
            scale = threshold["scale"]
            out["threshold_scale"] = {k: scale.get(k) for k in ("factor", "offset", "min", "max")
                                      if scale.get(k) is not None}
            out["threshold_scale"].setdefault("factor", 1.0)
            out["threshold_scale"].setdefault("offset", 0.0)
        if threshold.get("mm") is not None:
            out["thresholds"] = {"thr": float(threshold["mm"])}
        elif computed is not None:
            out["threshold_max_missing"] = 1.0   # blending never drops a unit for missing days
            out["threshold_partial_years"] = True   # and counts a year only partly in the data
        elif threshold.get("file") is not None:
            out["thresholds"] = {"thr": self.path(threshold["file"])}
            out["threshold_table"] = {k: threshold.get(k, default) for k, default in _THRESHOLD_DEFAULTS.items()}
        if years is not None:
            out["threshold_years"] = list(years)
        out.update({"rain_resolution": "exact", "missing": str(obs.get("missing_rain", "keep")).lower(),
                    "negatives": "keep", "gaps": "refuse", "horizon": "truncate", "lead_label": "end",
                    "forecast_history": "none"})
        return out, years, rule

    # -- units ---------------------------------------------------------------------------------------------------

    def units(self, models: list) -> dict:
        """Convert the ``domain`` section to the ``units:`` section of the run.

        Parameters
        ----------
        models : list of dict
            The BlendConfig ``models`` entries; each source gets a mode (``cells``, ``regrid`` or ``cell_transform``).

        Returns
        -------
        dict
            The ``units:`` mapping (box or outline, id convention, dissemination and centroids files, shapefile columns,
            per-source ``mode`` and ``weights``).
        """
        domain = self.get("domain", {})
        grid = domain.get("grid_ids") or {}
        out: dict = {}
        regrid = domain.get("regrid")
        area = domain.get("area_of_interest")
        if area is not None and (area.get("bbox") is None) == (area.get("geometry") is None):
            self.problems.append("domain.area_of_interest: give exactly one of 'bbox' or 'geometry'")
        if area is not None and area.get("bbox") is not None:
            out["box"] = {k: float(v) for k, v in area["bbox"].items() if v is not None}
        elif area is not None:
            out["outline"] = self.path(area["geometry"])
        if grid.get("decimal_digits") is not None:
            out["id_digits"] = grid["decimal_digits"]
        out["id_format"] = grid.get("format", "trimmed")
        out["dissemination_scope"] = domain.get("dissemination_scope", "inputs")
        for key, label in (("dissemination", "dissemination"), ("centroids", "centroids")):
            table = domain.get(key)
            if table is not None:
                if "file" not in table:
                    self.problems.append(f"domain.{key}.file: required")
                    continue
                out[label] = self.path(table["file"])
                if table.get("id_col") is not None:
                    out["dissemination_id_col" if key == "dissemination" else "centroid_id_col"] = table["id_col"]
        if regrid is not None:
            if regrid.get("shapefile") is not None:
                out["shapefile"] = self.path(regrid["shapefile"])
                out["unit_id_col"] = regrid.get("unit_id_col", "adm3_name")
                for key, label in (("unit_name_col", "unit_name_col"), ("parent_id_col", "parent_col")):
                    if regrid.get(key) is not None:
                        out[label] = regrid[key]
            if regrid.get("crs", "EPSG:4326") != "EPSG:4326":
                self.note(f"domain.regrid.crs {regrid['crs']!r} (HCWF reads shapefiles in EPSG:4326 after converting)")
            if regrid.get("clip_to_coverage", True) is not True:
                self.note("domain.regrid.clip_to_coverage false (HCWF always restricts and renormalises the weights "
                          "to the observed cells)")
        modes, weights = {}, {}

        def source(name, transform, model=None):
            """Record the mode and weights of one source in `modes` and `weights`."""
            if transform is not None:
                modes[name], weights[name] = "cell_transform", self.path(transform)
            elif regrid is None:
                modes[name] = "cells"
            else:
                modes[name] = "regrid"
                supplied = (model.get("regrid_weights") if model is not None else None) or regrid.get("weights")
                if supplied is not None:
                    weights[name] = self.path(supplied)

        source("observations", self.get("observations.cell_transform"))
        for model in models:
            source(model["name"], model.get("cell_transform"), model)
        out["mode"] = modes
        if weights:
            out["weights"] = weights
        for key in ("grid_lat_var", "grid_lon_var"):
            if regrid is not None and regrid.get(key) is not None:
                self.note(f"domain.regrid.{key} (HCWF finds the grid's coordinates itself)")
        return out

    # -- sources -------------------------------------------------------------------------------------------------

    def observations(self) -> dict:
        """Convert the ``observations`` section.

        Returns
        -------
        dict
            The ``observations:`` mapping of the run: ``paths`` and ``years``, ``variable``, ``unit_cvt`` (1.0: blending
            reads millimetres), ``dims`` and ``day_boundary``.
        """
        obs = self.get("observations", {})
        for key in ("files", "variable"):
            if key not in obs:
                self.problems.append(f"observations.{key}: required")
        out = self.files(obs.get("files"), "observations")
        out["variable"] = obs.get("variable")
        out["unit_cvt"] = 1.0   # blending reads observations as millimetres
        dims = self.dims(obs.get("dims"), _OBS_ROLES)
        if dims:
            out["dims"] = dims
        out["day_boundary"] = f"00 UTC, labelled by {obs.get('date_marks', 'end')}"
        if obs.get("date_marks", "end") not in ("start", "end"):
            self.problems.append(f"observations.date_marks: {obs.get('date_marks')!r} is not 'start' or 'end'")
        return out

    def model_entry(self, model: dict, index: int) -> dict:
        """Convert one ``models`` entry to a model of the run.

        Parameters
        ----------
        model : dict
            The BlendConfig model entry.
        index : int
            Its position in ``models``, for the problem messages.

        Returns
        -------
        dict
            ``name``, ``paths`` and ``years``, ``variable`` (default ``tp``), ``unit_cvt`` (from ``unit_scale``),
            ``dims`` and ``accumulation`` (``daily``).
        """
        where = f"models[{index}]"
        if "name" not in model:
            self.problems.append(f"{where}.name: required")
        out = {"name": model.get("name"), **self.files(model.get("files"), f"models[{model.get('name')}]"),
               "variable": model.get("variable", "tp"), "unit_cvt": float(model.get("unit_scale", 1.0))}
        dims = self.dims(model.get("dims"), _MODEL_ROLES)
        if dims:
            out["dims"] = dims
        out["accumulation"] = "daily"   # blending reads lead-day totals
        return out

    def model_blend(self, model: dict, rule) -> dict:
        """Convert the ``blend.models.<name>`` settings of one model: lead days, member limit, rain horizon, predictors.

        A window token of a rain predictor stays a token only where the HCWF definition (``definition_windows``) reads
        it to the same number of days; otherwise it becomes that number.

        Parameters
        ----------
        model : dict
            The BlendConfig model entry.
        rule : types.SimpleNamespace
            The rule, as `resolve_rule` returns it.

        Returns
        -------
        dict
            The keys that are set (``lead_days``, ``max_member``, ``rain_horizon``, ``rain_predictors``,
            ``rain_transform``); empty when none is.
        """
        out: dict = {}
        lead = model.get("lead_days") or {}
        if lead:
            out["lead_days"] = [lead.get("min", 1), lead.get("max", 45)]
        if model.get("max_member") is not None:
            out["max_member"] = model["max_member"]
        if model.get("rain_horizon"):
            out["rain_horizon"] = dict(model["rain_horizon"])
        predictors = []
        definition = self.definition_windows
        for item in model.get("rain_predictors") or []:
            window = item.get("window", "trigger")
            if isinstance(window, str) and window.lower() in _WINDOW_TOKENS:
                days = resolve_token(window, rule)
                token = _DEFINITION_TOKEN[window.lower()]
                # keep the token only where the HCWF definition reads it to the same number of days
                window = window.lower() if definition.get(token) == days else days
            predictors.append({"agg": item.get("agg"), "window": window})
        if predictors:
            out["rain_predictors"] = predictors
        if model.get("rain_transform") is not None:
            out["rain_transform"] = model["rain_transform"]
        return out

    # -- the blend section ---------------------------------------------------------------------------------------

    def blend(self, models: list, rule) -> dict:
        """Convert what only a blend has to the ``blend:`` section of the run.

        Parameters
        ----------
        models : list of dict
            The BlendConfig ``models`` entries (their per-model settings go to ``blend.models``).
        rule : types.SimpleNamespace
            The rule, as `resolve_rule` returns it.

        Returns
        -------
        dict
            The ``blend:`` mapping (years, formulas, cross-validation, baselines, MME, climatology, features,
            reference onset, fixed cutoff, outputs).
        """
        out: dict = {}
        evaluation = self.get("evaluation", {})
        years = evaluation.get("years") or {}
        parsed = {k: self.years(years.get(k), f"evaluation.years.{k}") for k in ("training", "cv_holdout", "true_holdout")}
        out["years"] = {k: _years_value(v) for k, v in parsed.items() if v}
        for key in ("cutoff_mode", "cv_methods", "cluster_var", "formulas", "final_formula", "forecasts", "mme",
                    "external_predictions", "random_forest"):
            if evaluation.get(key) is not None:
                out[key] = evaluation[key]
        if isinstance(out.get("mme"), dict) and "weights_year_tag" in out["mme"]:
            out["mme"] = {k: v for k, v in out["mme"].items() if k != "weights_year_tag"}
            self.note(_WEIGHTS_NOTE.format(tag=evaluation["mme"]["weights_year_tag"]))
        for item in out.get("forecasts") or []:
            if isinstance(item, dict) and "model" not in item:
                self.problems.append("evaluation.forecasts[].model: required")
        if evaluation.get("window_variants") is not None:
            out["window_variants"] = [{**v, "windows": [list(w) for w in year_windows(v.get("windows"), "", [])]}
                                      if isinstance(v, dict) and "windows" in v else v
                                      for v in evaluation["window_variants"]]
        if evaluation.get("climatology_baselines") is not None:
            baselines = dict(evaluation["climatology_baselines"])
            if "windows" in baselines:
                baselines["windows"] = [list(w) for w in year_windows(baselines["windows"], "", [])]
            out["baselines"] = baselines
        onset = self.get("onset", {})
        if onset.get("fixed_cutoff") is not None:
            out["fixed_cutoff"] = str(onset["fixed_cutoff"])
        if onset.get("reference_onset") is not None:
            reference = dict(onset["reference_onset"])
            if reference.get("file") is not None:
                reference["file"] = self.path(reference["file"])
            if (reference.get("month_day") is None) == (reference.get("file") is None):
                self.problems.append("onset.reference_onset: give exactly one of 'month_day' or 'file'")
            out["reference_onset"] = {k: (str(v) if k in ("month_day", "base_date") else v) for k, v in reference.items()}
        features = self.get("features", {})
        if features.get("bins") is not None:
            out["bins"] = dict(features["bins"])
        for key in ("probability_days", "trim_after_true_onset", "rain_transform"):
            if features.get(key) is not None:
                out[key] = features[key]
        climatology = self.get("climatology")
        if climatology:
            clim = dict(climatology)
            for key in ("train_years", "test_years"):
                if clim.get(key) is not None:
                    clim[key] = list(_year_pair(self.years(clim[key], f"climatology.{key}"), self.problems,
                                                f"climatology.{key}"))
            if clim.get("windows") is not None:
                clim["windows"] = [list(w) for w in year_windows(clim["windows"], "climatology.windows", self.problems)]
            if clim.get("horizons") is not None:
                clim["horizons"] = [{k: (str(v) if k in ("start", "end") else v) for k, v in h.items()}
                                    for h in clim["horizons"]]
            out["climatology"] = clim
        per_model = {m["name"]: self.model_blend(m, rule) for m in models}
        per_model = {k: v for k, v in per_model.items() if v}
        if per_model:
            out["models"] = per_model
        outputs = dict(self.get("outputs", {}))
        if outputs.get("dir") is not None:
            outputs["dir"] = self.path(outputs["dir"])
        wanted = outputs.get("intermediates")
        if wanted:   # HCWF writes only these two intermediates; the others are left out of the run file
            written = [name for name in wanted if name in ("observations", "features")]
            if len(written) != len(wanted):
                self.note(f"outputs.intermediates {[n for n in wanted if n not in written]} are not written by HCWF "
                          "(only observations and features are): left out")
            outputs["intermediates"] = written
            if not written:
                del outputs["intermediates"]
        if outputs:
            out["outputs"] = outputs
        if features.get("join", "inner") != "inner":
            self.problems.append("features.join: only 'inner' exists")
        return out

    # -- assembly ------------------------------------------------------------------------------------------------

    def convert(self) -> Conversion:
        """Convert the whole BlendConfig.

        Returns
        -------
        Conversion
            The run mapping with its comments and the notes.

        Raises
        ------
        ConvertError
            The config has unknown keys, misses a required key or cannot be converted; the error lists every problem.
        """
        data = self.data
        _unknown(data, _BLEND_KEYS, "", self.problems)
        for key in ("name", "observations", "models", "onset", "evaluation"):
            if key not in data:
                self.problems.append(f"{key}: required")
        if self.problems:
            raise ConvertError(self.problems)
        models = [m for m in data.get("models") or [] if isinstance(m, dict)]
        onset, years, rule = self._onset_with_windows()
        mapping: dict = {"project_name": str(data["name"]), "region": self.region}
        mapping["observations"] = self.observations()
        mapping["models"] = [self.model_entry(m, i) for i, m in enumerate(models)]
        mapping["onset"] = onset
        mapping["units"] = self.units(models)
        mapping["blend"] = self.blend(models, rule)
        if self.get("outputs.dir") is not None:
            self.note("outputs.dir is kept as blend.outputs.dir (where external predictions are read from); a run writes "
                      "its artifacts to the run folder's blend/ subfolder (output.data_dir chooses the folder)")
        if data.get("data_root"):
            self.note(f"data_root {data['data_root']!r}: every relative path is now absolute")
        if self.problems:
            raise ConvertError(self.problems)
        self.comments.update({
            ("region",): "no region: the whole domain of the data, as onset_blending reads it; name a region to use its box "
                         "and defaults",
            ("observations", "unit_cvt"): "onset_blending reads observations as millimetres (declared units still win)",
            ("onset", "rule"): "onset_blending's onset.definition as an HCWF definition; registry ids moron_robertson_blend "
                               "and icpac_blend are its readings of the two papers",
            ("onset", "rain_resolution"): "the parity options: exact rain, the missing-rain policy of observations, "
                                          "negatives and gaps as read, follow-ups past the data truncated",
        })
        return Conversion(mapping, tuple(self.notes), self.comments)

    def _onset_with_windows(self):
        """Convert the onset section and read the windows of the HCWF definition it makes.

        The windows are what the rain-window tokens of the predictors are checked against. Sets ``definition_windows``;
        when HCWF cannot read the rule written from the config, a problem is recorded and the windows are empty, so
        that tokens become numbers.

        Returns
        -------
        tuple of (dict, tuple of int or None, types.SimpleNamespace)
            What `_Convert.onset` returns: the ``onset:`` mapping, the threshold years and the rule.
        """
        from hcwf.onset import definition as definition_mod

        onset, years, rule = self.onset()
        try:
            self.definition_windows = definition_mod.loads(yaml.safe_dump(onset["rule"], sort_keys=False),
                                                           fine_amounts=True).windows()
        except Exception as exc:  # noqa: BLE001 - the rule is reported with the problem; tokens then become numbers
            self.problems.append(f"onset.definition: HCWF cannot read the rule written from it: {exc}")
            self.definition_windows = {}
        return onset, years, rule


def _year_pair(years, problems: list[str], where: str) -> tuple[int, int] | None:
    """Take the first and last of a years value.

    Parameters
    ----------
    years : tuple of int
        The years, sorted; may be empty.
    problems : list of str
        Appended with a problem when there are no years.
    where : str
        The config path named in that problem.

    Returns
    -------
    tuple of (int, int) or None
        ``(first, last)``; None when there are no years.
    """
    if not years:
        problems.append(f"{where}: expected a first and last year")
        return None
    return years[0], years[-1]


def validate_conversion(conversion: Conversion):
    """Check a conversion without reading any data file.

    Its sections are flattened and typed as a run config's are, the ``blend:`` section builds a Recipe and the onset
    settings resolve.

    Parameters
    ----------
    conversion : Conversion
        The conversion to check.

    Returns
    -------
    tuple of (dict, hcwf.blend.recipe.Recipe, hcwf.config.onset.OnsetSetup)
        The flat run config, the recipe built from the ``blend:`` section and the resolved onset settings.

    Raises
    ------
    ConvertError
        The config does not validate, the recipe is invalid or the onset settings are; the error lists the problems.
    """
    from hcwf.blend.recipe import recipe_from_config
    from hcwf.errors import RecipeError
    from hcwf.config import schema
    from hcwf.config.onset import onset_setup
    from hcwf.errors import OnsetConfigError

    mapping, errors = conversion.mapping, []
    flat, origin, _ = schema.flatten_sections(mapping, errors)
    cfg = schema.normalise(flat, origin, errors)
    if errors:
        raise ConvertError(errors)
    cfg["model_list"] = tuple(m["name"] for m in mapping["models"])
    try:
        setup = onset_setup(cfg)
        return cfg, recipe_from_config(cfg, setup), setup
    except RecipeError as exc:
        raise ConvertError([f"blend: {p}" for p in exc.problems]) from None
    except OnsetConfigError as exc:
        raise ConvertError([str(exc)]) from None


def is_blend_config(data) -> bool:
    """Tell whether a loaded YAML mapping looks like a BlendConfig.

    Parameters
    ----------
    data : object
        The loaded YAML.

    Returns
    -------
    bool
        True for a mapping with ``evaluation``, ``onset`` and a mapping ``observations.files``.
    """
    return (isinstance(data, dict) and "evaluation" in data and "onset" in data
            and isinstance((data.get("observations") or {}).get("files"), dict))


def convert_blend(data: dict, *, data_root=None, region: str | None = None) -> Conversion:
    """Convert a BlendConfig given as a mapping (what its YAML holds) to an HCWF run.

    Relative paths start at `data_root` (the config's own ``data_root`` key unless given; the working directory without
    either) and are written absolute. A problem with the config raises `ConvertError`, which lists every problem.

    Parameters
    ----------
    data : dict
        The BlendConfig mapping.
    data_root : str or pathlib.Path, optional
        Where relative paths start; None uses the config's ``data_root``.
    region : str, optional
        An HCWF region id for the run; None is no region, the data's whole domain, as blending uses it.

    Returns
    -------
    Conversion
        The run mapping, its comments and the notes.
    """
    root = data_root if data_root is not None else data.get("data_root")
    return _Convert(dict(data), Path(root).resolve() if root else None, region).convert()


def convert_blend_file(path, *, region: str | None = None) -> Conversion:
    """Run `convert_blend` on a BlendConfig YAML file.

    Relative paths start at the config's ``data_root``, itself relative to the file (the file's folder by default).

    Parameters
    ----------
    path : str or pathlib.Path
        The BlendConfig YAML file.
    region : str, optional
        An HCWF region id for the run; None is no region.

    Returns
    -------
    Conversion
        The run mapping, its comments and the notes.

    Raises
    ------
    ConvertError
        The file does not hold a mapping (`convert_blend` raises it too for a config that cannot be converted).
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConvertError([f"{path}: expected a mapping of keys"])
    root = path.parent.resolve() / (data.get("data_root") or ".")
    return convert_blend(data, data_root=root.resolve(), region=region)


# -- the legacy spec sets (ported from onset_blending.legacy.specs) -------------------------------------------------

# Old defaults that differ from the new config's, written out explicitly.
_OLD_TRAINING_YEARS = list(range(2000, 2025))
_OLD_FIXED_CUTOFF = "06-02"
_OLD_FORECAST_VARIANTS = {"base": "", "ref_onset": "_ref", "fixed_cutoff": "_fixed_cutoff"}
_CONNECT_VARIANTS = {"fixed_cutoff": "fixed_cutoff", "ref": "ref_onset"}
# Combine daily columns the connect stage reads, by raw output column.
_DAILY_OUT = {
    "predicted_prob": "p_onset",
    "predicted_prob_fixed_cutoff": "p_onset_fixed_cutoff",
    "predicted_prob_ref": "p_onset_ref",
    "forecast_rain": "rain_mean",
}




class _Spec:
    """One old spec that remembers which keys the conversion has read.

    Parameters
    ----------
    label : str
        The spec's name in notes, for example ``"ground_truth"`` or ``"forecast[AIFS]"``.
    data : dict, optional
        The loaded spec; None is empty.
    notes : list of str
        The shared list that `_Spec.drop` appends its notes to.

    Attributes
    ----------
    label : str
        The spec's name in notes.
    data : dict
        The loaded spec.
    read : set of str
        The dotted paths read so far.
    """

    def __init__(self, label: str, data: dict | None, notes: list[str]):
        self.label = label
        self.data = data or {}
        self.read: set[str] = set()
        self._notes = notes

    def get(self, path: str, default=None):
        """Read a dotted path of the spec and mark it as read.

        Parameters
        ----------
        path : str
            Dotted keys, for example ``"options.min_day"``.
        default : object, optional
            What to return when the path is missing or null.

        Returns
        -------
        object
            The value, or `default`.
        """
        self.read.add(path)
        node = self._lookup(path)
        return default if node is None else node

    def has(self, path: str) -> bool:
        """Tell whether the spec has a non-null value at a dotted path (without marking it as read).

        Parameters
        ----------
        path : str
            Dotted keys.

        Returns
        -------
        bool
            True when the value exists and is not None.
        """
        return self._lookup(path) is not None

    def drop(self, path: str, reason: str) -> None:
        """Mark a key as read and record that it is not carried over, if it is set.

        Parameters
        ----------
        path : str
            Dotted keys.
        reason : str
            Why the key is dropped, written in the note.
        """
        if self.has(path):
            self.read.add(path)
            self._notes.append(f"{self.label}: {path} ({reason})")

    def unread(self) -> list[str]:
        """List the leaf keys nothing read; reading a key covers everything below it.

        Returns
        -------
        list of str
            Dotted paths of the leaves of the spec that were not read.
        """
        leaves: list[str] = []

        def walk(node, path):
            """Collect the unread leaves below `node` into `leaves`."""
            if isinstance(node, dict) and node:
                for key, value in node.items():
                    walk(value, f"{path}.{key}" if path else str(key))
            elif path and not self._covered(path):
                leaves.append(path)

        walk(self.data, "")
        return leaves

    def _covered(self, path: str) -> bool:
        """Tell whether a path, or one of its parents, has been read.

        Parameters
        ----------
        path : str
            Dotted keys.

        Returns
        -------
        bool
            True when `path` or a prefix of it is in ``read``.
        """
        parts = path.split(".")
        return any(".".join(parts[:i]) in self.read for i in range(1, len(parts) + 1))

    def _lookup(self, path: str):
        """Look up a dotted path without marking it as read.

        Parameters
        ----------
        path : str
            Dotted keys.

        Returns
        -------
        object
            The value, or None when a step is missing.
        """
        node = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node


def _same_path(a: str | None, b: str | None) -> bool:
    """Tell whether two paths are the same after normalising.

    Parameters
    ----------
    a : str, optional
        A path.
    b : str, optional
        Another path.

    Returns
    -------
    bool
        True when both are given and equal under ``os.path.normpath``.
    """
    return a is not None and b is not None and os.path.normpath(a) == os.path.normpath(b)


def _without_none(mapping: dict) -> dict:
    """Drop the null values of a mapping.

    Parameters
    ----------
    mapping : dict
        Any mapping.

    Returns
    -------
    dict
        A new dict with the items whose value is not None.
    """
    return {k: v for k, v in mapping.items() if v is not None}


def _grid_id_convention(spec: _Spec | None) -> tuple | None:
    """Read the grid-id convention a spec states (``grid_id_decimal_digits``, ``grid_id_format``).

    The keys are read under ``geometry`` when the spec has a ``geometry`` mapping (the regrid spec), else at the top.
    Reading marks them as read in the spec.

    Parameters
    ----------
    spec : _Spec, optional
        A spec; None states nothing.

    Returns
    -------
    tuple of (int or None, str) or None
        ``(decimal digits or None, lower-case format, ``"trimmed"`` when not given)``; None when the spec states
        neither.
    """
    if spec is None:
        return None
    base = "geometry." if isinstance(spec.data.get("geometry"), dict) else ""
    digits = spec.get(base + "grid_id_decimal_digits")
    fmt = spec.get(base + "grid_id_format")
    if digits is None and fmt is None:
        return None
    return (digits, str(fmt).lower() if fmt is not None else "trimmed")


def _train_years(run: dict) -> tuple:
    """Read the training years of a climatology run.

    Parameters
    ----------
    run : dict
        A ``climatologies`` run of the ground-truth spec.

    Returns
    -------
    tuple of (int or None, int or None)
        ``(train_year_min, train_year_max)``; None for a limit that is not given.
    """
    return (run.get("train_year_min"), run.get("train_year_max"))


def _climatology_settings(run: dict) -> dict:
    """Read the settings of a climatology run that connect's conditional and unconditional runs must share.

    Parameters
    ----------
    run : dict
        A ``climatologies`` run of the ground-truth spec.

    Returns
    -------
    dict
        ``onset_col`` (default ``onset_day``), ``season_start`` and ``issue_end`` (``MM-DD`` text), ``forecast_window``,
        ``horizons`` (a list of ``{start, end, forecast_window}`` or None), ``cv_by_year`` (default True),
        ``min_onset_years`` (default 10) and ``test_years`` (a ``(min, max)`` tuple, None for a limit not given).
    """
    horizons = run.get("horizons")
    return {
        "onset_col": run.get("onset_col", "onset_day"),
        "season_start": str(run.get("season_start_md")),
        "issue_end": str(run.get("issue_end_md")),
        "forecast_window": run.get("forecast_window"),
        "horizons": [
            {"start": str(h["start_md"]), "end": str(h["end_md"]), "forecast_window": int(h["forecast_window"])}
            for h in horizons
        ] if horizons else None,
        "cv_by_year": bool(run.get("cv_by_year", True)),
        "min_onset_years": int(run.get("min_onset_years", 10)),
        "test_years": (run.get("test_year_min"), run.get("test_year_max")),
    }


class _Converter:
    """One conversion of a legacy spec set (onset_blending's stage specs) to a BlendConfig mapping, then to a run.

    Every spec is wrapped in a `_Spec` so that keys nothing read are reported at the end. Facts stated in several specs
    must agree (`_Converter.agree`); disagreements are problems. `_Converter.convert` builds the BlendConfig mapping
    and passes it to `convert_blend`.

    Parameters
    ----------
    name : str
        The project name.
    ground_truth : dict, optional
        The ground-truth ``raw_data`` spec.
    forecasts : dict of str to dict
        The ``raw_data`` spec of each model, by model name.
    combine : dict, optional
        The ``combine`` spec.
    connect : dict, optional
        The ``connect`` spec.
    cv_models : dict, optional
        The ``cv_models`` spec.
    regrid : dict, optional
        The ``regrid`` spec; None when the data is not regridded.
    connect_onset_spec : dict, optional
        The ``raw_data`` spec that connect's ``onset_spec`` names when that is not the ground truth.
    data_root : str, optional
        The folder the old scripts ran from; relative paths of the specs start there.
    region : str, optional
        An HCWF region id for the run.

    Attributes
    ----------
    problems : list of str
        What cannot be converted.
    notes : list of str
        What is not carried over.
    gt : _Spec
        The wrapped ground-truth spec.
    forecasts : dict of str to _Spec
        The wrapped model specs, by model name.
    combine : _Spec
        The wrapped ``combine`` spec.
    connect : _Spec
        The wrapped ``connect`` spec.
    cv : _Spec
        The wrapped ``cv_models`` spec.
    regrid : _Spec or None
        The wrapped regrid spec.
    definition : dict or None
        The agreed onset definition, set by `_Converter.onset`.
    """

    def __init__(self, *, name, ground_truth, forecasts, combine, connect, cv_models, regrid, connect_onset_spec,
                 data_root=None, region=None):
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.name = name
        self.data_root = data_root
        self.region = region
        self.gt = _Spec("ground_truth", ground_truth, self.notes)
        self.forecasts = {m: _Spec(f"forecast[{m}]", spec, self.notes) for m, spec in forecasts.items()}
        self.combine = _Spec("combine", combine, self.notes)
        self.connect = _Spec("connect", connect, self.notes)
        self.cv = _Spec("cv_models", cv_models, self.notes)
        self.regrid = _Spec("regrid", regrid, self.notes) if regrid is not None else None
        self.connect_onset = connect_onset_spec

    # -- helpers -----------------------------------------------------------

    def agree(self, what: str, values: dict[str, Any]):
        """Check that one fact stated in several specs has the same value everywhere.

        Parameters
        ----------
        what : str
            The fact, named in the problem.
        values : dict of str to object
            The value stated in each spec, by spec label; None means not stated.

        Returns
        -------
        object
            The agreed value, or None when no spec states it (when they disagree, a problem is recorded and the first
            distinct value is returned).
        """
        stated = {where: v for where, v in values.items() if v is not None}
        distinct = []
        for value in stated.values():
            if value not in distinct:
                distinct.append(value)
        if len(distinct) > 1:
            listed = "; ".join(f"{where} = {value!r}" for where, value in stated.items())
            self.problems.append(f"{what} disagrees between specs: {listed}")
        return distinct[0] if distinct else None

    def years_between(self, spec: _Spec) -> tuple[int, ...]:
        """Read the years from ``options.min_year`` to ``options.max_year`` of a spec.

        Parameters
        ----------
        spec : _Spec
            A ``raw_data`` spec.

        Returns
        -------
        tuple of int
            The years, inclusive; empty when neither limit is given or only one is (a problem is recorded then).
        """
        low, high = spec.get("options.min_year"), spec.get("options.max_year")
        if low is None and high is None:
            return ()
        if low is None or high is None:
            self.problems.append(
                f"{spec.label}: options.min_year and max_year must be given together to convert"
            )
            return ()
        return tuple(range(int(low), int(high) + 1))

    @property
    def raw_specs(self) -> list[_Spec]:
        """The ground-truth spec followed by the forecast specs."""
        return [self.gt, *self.forecasts.values()]

    # -- onset -------------------------------------------------------------

    @staticmethod
    def onset_definition(spec: _Spec | dict) -> dict:
        """Resolve the onset definition as ``read_onset_params`` does, with its defaults.

        Parameters
        ----------
        spec : _Spec or dict
            A ``raw_data`` spec, or the loaded mapping of one.

        Returns
        -------
        dict
            The BlendConfig ``onset.definition`` mapping: ``window``, ``trigger_rule``, ``wet_day_min_mm``,
            ``wet_day_comparison``, ``follow_days``, ``followup_anchor`` and ``dry_spell``.
        """
        if isinstance(spec, dict):
            spec = _Spec("onset", spec, [])
        wet = float(spec.get("options.onset_definition.wet_day_min_mm", 1.0))
        base = "options.onset_definition"
        return {
            "window": int(spec.get("options.window", 5)),
            "trigger_rule": str(spec.get(f"{base}.trigger_rule", "all_days_wet")).lower(),
            "wet_day_min_mm": wet,
            "wet_day_comparison": str(spec.get(f"{base}.wet_day_comparison", "gte")).lower(),
            "follow_days": int(spec.get(f"{base}.follow_days", 21)),
            "followup_anchor": str(spec.get(f"{base}.followup_anchor", "after_trigger")).lower(),
            "dry_spell": {
                "mode": str(spec.get(f"{base}.dry_spell.mode", "consecutive_dry")),
                "min_dry_days": int(spec.get(f"{base}.dry_spell.min_dry_days", 7)),
                "dry_day_min_mm": float(spec.get(f"{base}.dry_spell.dry_day_min_mm", wet)),
                "sum_window": int(spec.get(f"{base}.dry_spell.sum_window", 10)),
                "sum_min_mm": float(spec.get(f"{base}.dry_spell.sum_min_mm", 5.0)),
            },
        }

    def threshold(self, spec: _Spec) -> dict | None:
        """Resolve the onset threshold as ``onset_utils`` does, in the new config's shape.

        A number (also a number given as the file) is ``mm`` (with a scale applied); otherwise the keys that apply to the
        file type (NetCDF, MAT or CSV) are read and the others are dropped with a note.

        Parameters
        ----------
        spec : _Spec
            A ``raw_data`` spec.

        Returns
        -------
        dict or None
            The ``onset.threshold`` mapping; None when the spec has no threshold.
        """
        rule = spec.get("thresholds.rule") or {}
        rule_type = str(rule.get("type", "")).lower()
        if spec.get("thresholds.constant") is not None:
            return {"mm": float(spec.get("thresholds.constant"))}
        if rule_type == "constant":
            return {"mm": float(rule["value"])}
        source = spec.get("thresholds.file")
        if source is None:
            return None
        scale = None
        if rule_type == "scale":
            scale = _without_none(
                {k: rule.get(k) for k in ("factor", "offset", "min", "max")}
            )
        try:
            value = float(source)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            if scale:
                value = value * float(scale.get("factor", 1.0)) + float(scale.get("offset", 0.0))
                value = max(value, scale["min"]) if scale.get("min") is not None else value
                value = min(value, scale["max"]) if scale.get("max") is not None else value
            return {"mm": value}
        out: dict[str, Any] = {"file": source}
        t = "thresholds."
        ext = os.path.splitext(str(source))[1].lower()
        before = set(spec.read)
        if ext in (".nc", ".nc4", ".netcdf"):
            out["lat_var"] = spec.get(t + "lat_var") or spec.get(t + "lat_col") or "lat"
            out["lon_var"] = spec.get(t + "lon_var") or spec.get(t + "lon_col") or "lon"
            out["value_var"] = spec.get(t + "thresh_var") or spec.get(t + "thresh_col") or "MWmean"
        elif ext == ".mat":
            out["mat_lat_name"] = spec.get(t + "mat_lat_name", "lat")
            out["mat_lon_name"] = spec.get(t + "mat_lon_name", "lon")
            out["mat_value_name"] = spec.get(t + "mat_thresh_name", "onset.day.thres")
        else:
            out["id_col"] = spec.get(t + "id_col") or spec.get(t + "adm3_col")
            out["value_col"] = spec.get(t + "thresh_col", "onset_thresh")
            out["lat_col"] = spec.get(t + "lat_col", "lat")
            out["lon_col"] = spec.get(t + "lon_col", "lon")
        used = spec.read - before
        for key in ("thresh_var", "lat_var", "lon_var", "mat_lat_name", "mat_lon_name",
                    "mat_thresh_name", "id_col", "adm3_col", "thresh_col", "lat_col", "lon_col"):
            if t + key not in used:
                spec.drop(t + key, f"does not apply to a {ext or 'CSV'} threshold file")
        if scale:
            out["scale"] = scale
        return _without_none(out)

    @staticmethod
    def reference_onset(spec: _Spec) -> dict | None:
        """Read the reference onset of a spec.

        A constant month-day takes precedence over a file; the file keys are then dropped with a note.

        Parameters
        ----------
        spec : _Spec
            A ``raw_data`` spec.

        Returns
        -------
        dict or None
            ``{"month_day": ...}``, or the file and its columns; None when the spec has none.
        """
        month_day = spec.get("ref_onset.constant_month_day") or spec.get("ref_onset.date")
        if month_day:
            for key in ("file", "date_col", "year_col", "day_col", "base_date", "unit_col"):
                spec.drop(f"ref_onset.{key}", "a constant month-day takes precedence")
            return {"month_day": str(month_day)}
        if not spec.get("ref_onset.file"):
            return None
        return _without_none(
            {
                "file": spec.get("ref_onset.file"),
                "date_col": spec.get("ref_onset.date_col"),
                "year_col": spec.get("ref_onset.year_col"),
                "day_col": spec.get("ref_onset.day_col"),
                "base_date": spec.get("ref_onset.base_date"),
                "unit_col": spec.get("ref_onset.unit_col"),
            }
        )

    def onset(self, uses_ref_series: bool) -> dict:
        """Convert the onset settings of all the specs to the BlendConfig ``onset`` section.

        The onset definition, the threshold, the reference onset and the fixed cutoff must agree between the specs;
        ``season_start`` is the ground truth's ``options.cutoff_month_day``. Sets ``definition``.

        Parameters
        ----------
        uses_ref_series : bool
            Whether the evaluation uses the ref series (see `_Converter.uses_ref_series`): then a forecast spec without
            a ``ref_onset`` is a problem rather than a note.

        Returns
        -------
        dict
            ``threshold``, ``season_start``, ``fixed_cutoff``, ``reference_onset`` and ``definition`` where known.
        """
        definitions = {s.label: self.onset_definition(s) for s in self.raw_specs}
        definition = self.agree("the onset definition (options.window, options.onset_definition)", definitions)
        self.definition = definition
        thresholds = {s.label: self.threshold(s) for s in self.raw_specs}
        threshold = self.agree("the onset threshold", thresholds)
        if threshold is None:
            self.problems.append(
                "no onset threshold in any raw_data spec: the old pipeline then finds no onsets at all"
            )

        references = {s.label: self.reference_onset(s) for s in self.raw_specs}
        reference = references[self.gt.label]
        stated = {k: v for k, v in references.items() if v is not None}
        if reference is None and stated:
            self.problems.append(
                "ref_onset is set in " + ", ".join(stated) + " but not in ground_truth; one reference "
                "onset now sets both the observed onset search start and the forecasts' ref series"
            )
        elif reference is not None:
            missing = [s.label for s in self.forecasts.values() if references[s.label] is None]
            if missing and uses_ref_series:
                self.problems.append(
                    f"ref_onset is missing from {', '.join(missing)}, whose ref series the evaluation "
                    "uses; there it equals the plain series, while the new config computes it"
                )
            elif missing:
                self.notes.append(
                    f"{', '.join(missing)}: no ref_onset, so their unused ref series was the plain series"
                )
            self.agree("ref_onset", {k: v for k, v in references.items() if v is not None})

        season_start = self.gt.get("options.cutoff_month_day")
        if season_start is None:
            self.problems.append("ground_truth: options.cutoff_month_day is required")
        for spec in self.forecasts.values():
            spec.drop("options.cutoff_month_day", "only ground-truth processing reads it")
        fixed = self.agree(
            "options.fixed_cutoff_month_day",
            {s.label: str(s.get("options.fixed_cutoff_month_day", _OLD_FIXED_CUTOFF)) for s in self.forecasts.values()},
        )
        gt_fixed = self.gt.get("options.fixed_cutoff_month_day")
        if gt_fixed is not None:
            self.notes.append(
                "ground_truth: options.fixed_cutoff_month_day (only forecast processing reads it"
                + (")" if str(gt_fixed) == fixed else f"; it said {gt_fixed!r}, the forecasts {fixed!r})")
            )

        out: dict[str, Any] = {"threshold": threshold, "season_start": season_start}
        if fixed is not None:
            out["fixed_cutoff"] = fixed
        if reference is not None:
            out["reference_onset"] = reference
        if definition is not None:
            definition = {**definition, "dry_spell": dict(definition["dry_spell"])}
            if definition["dry_spell"]["dry_day_min_mm"] == definition["wet_day_min_mm"]:
                del definition["dry_spell"]["dry_day_min_mm"]
            out["definition"] = definition
        return out

    # -- inputs ------------------------------------------------------------

    def files(self, spec: _Spec, regrid_source: dict | None, years: tuple[int, ...]) -> dict:
        """Build the BlendConfig ``files`` entry of a source.

        The regrid writes ``<stem>_adm3.nc`` beside each source file; the new pipeline regrids in memory, so with a
        regrid entry the sources are read.

        Parameters
        ----------
        spec : _Spec
            The ``raw_data`` spec of the source.
        regrid_source : dict, optional
            The regrid entry of the source (``nc_folder``, ``file_regex``); None when it is not regridded.
        years : tuple of int
            The years to read; empty for all.

        Returns
        -------
        dict
            ``folder``, ``pattern`` and, when there are years, ``years``.
        """
        folder = spec.get("input.nc_folder")
        pattern = spec.get("input.file_regex")
        if regrid_source is not None:
            # The regrid writes <stem>_adm3.nc beside each source file; the new
            # pipeline regrids in memory, so it reads the sources.
            folder = regrid_source.get("nc_folder")
            pattern = regrid_source.get("file_regex") or r"\.nc$"
        out = {"folder": folder, "pattern": pattern}
        if years:
            out["years"] = list(years)
        return out

    @staticmethod
    def dims(spec: _Spec) -> dict:
        """Read the dimension renames of a spec, with the wide day dimension renamed ``day``.

        Parameters
        ----------
        spec : _Spec
            A ``raw_data`` spec.

        Returns
        -------
        dict of str to str
            File dimension name to role.
        """
        rename = dict(spec.get("dimensions.rename", {}) or {})
        day_dim = spec.get("input.wide_day_dim", "day")
        if day_dim != "day":
            rename = {old: ("day" if new == day_dim else new) for old, new in rename.items()}
        return {str(k): str(v) for k, v in rename.items()}

    def cell_transform(self, spec: _Spec) -> str | None:
        """Read the cell transform file of a spec, if it is enabled.

        Parameters
        ----------
        spec : _Spec
            A ``raw_data`` spec.

        Returns
        -------
        str or None
            The file; None when the transform is not enabled (a problem is recorded when it is enabled without a file).
        """
        enabled = bool(spec.get("options.cell_transform_enabled", False))
        if enabled:
            path = spec.get("options.cell_transform_file")
            if path is None:
                self.problems.append(f"{spec.label}: options.cell_transform_enabled needs cell_transform_file")
            return path
        spec.drop("options.cell_transform_file", "cell_transform_enabled is false")
        return None

    def regrid_entry(self, folder: str | None) -> dict | None:
        """Find the entry of the regrid spec for a forecast folder.

        Parameters
        ----------
        folder : str, optional
            The source folder of the forecast.

        Returns
        -------
        dict or None
            The ``forecasts`` entry of the regrid spec with the same ``nc_folder``; None when there is none or no
            regrid spec.
        """
        if self.regrid is None:
            return None
        for entry in self.regrid.get("forecasts", []) or []:
            if _same_path(entry.get("nc_folder"), folder):
                return entry
        return None

    def observations(self) -> dict:
        """Convert the ground-truth spec to the BlendConfig ``observations`` section.

        Returns
        -------
        dict
            ``files``, ``variable``, ``dims``, ``missing_rain`` and ``cell_transform`` where set; the regrid spec must
            agree with the ground truth on folder and variable (a problem otherwise).
        """
        gt = self.gt
        regrid_source = None
        if self.regrid is not None:
            regrid_source = {
                "nc_folder": self.regrid.get("ground_truth.nc_folder"),
                "file_regex": self.regrid.get("ground_truth.file_regex"),
            }
            if not _same_path(regrid_source["nc_folder"], gt.get("input.nc_folder")):
                self.problems.append(
                    "regrid ground_truth.nc_folder and ground_truth input.nc_folder differ "
                    f"({regrid_source['nc_folder']!r} vs {gt.get('input.nc_folder')!r}); the raw_data "
                    "spec reads the regridded files beside the sources"
                )
            regrid_variable = self.regrid.get("ground_truth.value_col")
            if regrid_variable and str(regrid_variable).lower() != str(gt.get("input.value_col")).lower():
                self.problems.append(
                    f"regrid ground_truth.value_col {regrid_variable!r} is not ground_truth "
                    f"input.value_col {gt.get('input.value_col')!r}"
                )
        missing = gt.get("options.missing_rain_policy") or gt.get("input.missing_rain_policy") or "keep"
        out = {
            "files": self.files(gt, regrid_source, self.years_between(gt)),
            "variable": gt.get("input.value_col"),
            "dims": self.dims(gt),
            "missing_rain": str(missing).lower(),
            "cell_transform": self.cell_transform(gt),
        }
        return _without_none(out)

    def models(self, connect_models: list[dict], onset_definition: dict | None) -> list[dict]:
        """Convert the forecast specs to the BlendConfig ``models`` section.

        The models are those of connect's ``forecast_models``, in its order. Each needs a ``raw_data`` spec and a
        ``combine`` entry; ``rain_day_max`` and ``rain_horizon_policy`` must agree between raw data and connect.

        Parameters
        ----------
        connect_models : list of dict
            The ``forecast_models`` entries of the ``connect`` spec.
        onset_definition : dict, optional
            The agreed onset definition (`_Converter.onset`); a rain-predictor window token is kept only if it reads
            the same number of days under it as under connect's definition.

        Returns
        -------
        list of dict
            The BlendConfig model entries.
        """
        combine_models = self.combine.get("forecasts", {}) or {}
        names = [str(entry.get("name")) for entry in connect_models]
        for name in self.forecasts:
            if name not in names:
                self.problems.append(f"forecast[{name}] is not in connect forecast_models ({', '.join(names)})")
        for name in combine_models:
            if name not in names:
                self.notes.append(f"combine: forecasts.{name} (connect never reads it)")
        top_weights = self.regrid.get("weights_in") if self.regrid is not None else None
        token_source = self.connect_token_definition()
        models = []
        for index, entry in enumerate(connect_models):
            name = str(entry.get("name"))
            raw = self.forecasts.get(name)
            combined = combine_models.get(name)
            if raw is None:
                self.problems.append(f"connect forecast_models[{index}] {name!r} has no raw_data spec")
                continue
            if combined is None:
                self.problems.append(f"connect forecast_models[{index}] {name!r} has no combine forecasts entry")
                combined = {}
            model: dict[str, Any] = {"name": name}

            years = self.years_between(raw)
            sources = combined.get("sources") or []
            if len(sources) > 1:
                self.problems.append(
                    f"combine forecasts.{name}: {len(sources)} sources; the new config reads each "
                    "model from one set of files"
                )
            if sources and sources[0].get("years") not in (None, ""):
                parse_problems: list[str] = []
                combine_years = years_of(sources[0]["years"], f"combine forecasts.{name}.sources[0].years", parse_problems)
                self.problems += parse_problems
                if combine_years:
                    years = tuple(sorted(set(years) & set(combine_years))) if years else combine_years
            if any("prob_col" in source for source in sources):
                self.notes.append(f"combine: forecasts.{name}.sources[].prob_col ({UNUSED})")

            regrid_source = self.regrid_entry(raw.get("input.nc_folder"))
            if self.regrid is not None and regrid_source is None:
                self.notes.append(
                    f"forecast[{name}]: the regrid spec does not list its folder, so its files are "
                    "read as they are (already on the units)"
                )
            model["files"] = self.files(raw, regrid_source, years)
            model["variable"] = raw.get("input.value_col")
            model["dims"] = self.dims(raw)
            if raw.get("filter.max_number") is not None:
                model["max_member"] = int(raw.get("filter.max_number"))

            min_day, max_day = raw.get("options.min_day"), raw.get("options.max_day")
            if min_day is None or max_day is None:
                self.problems.append(f"forecast[{name}]: options.min_day and max_day are required")
            else:
                model["lead_days"] = {"min": int(min_day), "max": int(max_day)}
                combine_max = combined.get("max_day")
                if combine_max is not None and int(combine_max) < int(max_day):
                    self.problems.append(
                        f"combine forecasts.{name}.max_day ({combine_max}) is below forecast[{name}] "
                        f"options.max_day ({max_day}); the old combine stops on that"
                    )

            rain_day_max = self.agree(
                f"{name}: rain_day_max",
                {f"forecast[{name}] options": raw.get("options.rain_day_max"), "connect": entry.get("rain_day_max")},
            )
            policy = self.agree(
                f"{name}: rain_horizon_policy",
                {
                    f"forecast[{name}] options": _lower(raw.get("options.rain_horizon_policy")),
                    "connect": _lower(entry.get("rain_horizon_policy")),
                },
            )
            horizon = _without_none({"max_day": rain_day_max, "policy": policy})
            if horizon.get("policy") == ("exact" if rain_day_max is not None else "legacy"):
                del horizon["policy"]
            if horizon:
                model["rain_horizon"] = horizon

            predictors = []
            for p_index, predictor in enumerate(entry.get("rain_predictors") or []):
                parsed = self.rain_predictor(predictor, token_source, onset_definition, f"connect forecast_models[{index}].rain_predictors[{p_index}]")
                if parsed is not None:
                    predictors.append(parsed)
            if predictors:
                model["rain_predictors"] = predictors
            transform = entry.get("rain_transform")
            if transform is not None:
                model["rain_transform"] = _rain_transform(transform)
            for variant in entry.get("variants") or []:
                if variant not in _CONNECT_VARIANTS:
                    self.problems.append(
                        f"connect forecast_models[{index}].variants: {variant!r} is not one of "
                        f"{', '.join(_CONNECT_VARIANTS)}"
                    )
            model["cell_transform"] = self.cell_transform(raw)
            weights = (regrid_source or {}).get("weights_in") or top_weights
            if weights is not None:
                model["regrid_weights"] = weights
            self.check_combine_columns(name, combined, first=index == 0)
            models.append(_without_none(model))
        return models

    def check_combine_columns(self, name: str, combined: dict, first: bool) -> None:
        """Check the columns the ``combine`` stage writes for a model against what connect reads.

        Parameters
        ----------
        name : str
            The model name.
        combined : dict
            Its ``combine`` ``forecasts`` entry (``daily`` and ``constants``).
        first : bool
            Whether it is connect's first model, which must have an ``onset_thresh`` constant.
        """
        for item in combined.get("daily") or []:
            col, out = item.get("col"), item.get("out")
            expected = _DAILY_OUT.get(col)
            if expected is None:
                if out in _DAILY_OUT.values():
                    self.problems.append(
                        f"combine forecasts.{name}.daily: {col!r} is named {out!r}, which connect reads; "
                        "the new pipeline names each series by what it is"
                    )
                else:
                    self.notes.append(f"combine: forecasts.{name}.daily {col} -> {out} (connect never reads it)")
            elif out != expected:
                self.problems.append(
                    f"combine forecasts.{name}.daily maps {col!r} to {out!r}; connect reads it as {expected!r}"
                )
        constants = {item.get("col"): item.get("out") for item in combined.get("constants") or []}
        for col, out in constants.items():
            if col == "onset_thresh" and out != "onset_thresh":
                self.problems.append(f"combine forecasts.{name}.constants maps onset_thresh to {out!r}")
            elif col not in ("onset_thresh", "ref_onset_date"):
                self.notes.append(f"combine: forecasts.{name}.constants {col} (connect never reads it)")
        if first and "onset_thresh" not in constants:
            self.problems.append(
                f"combine forecasts.{name} (connect's first model) has no onset_thresh constant, "
                "which the old connect stage reads the threshold feature from"
            )
        if any(item.get("add_plus") for item in combined.get("daily") or []):
            self.notes.append(f"combine: forecasts.{name}.daily[].add_plus (connect never reads the plus columns)")

    def connect_token_definition(self) -> dict:
        """Resolve the onset definition connect resolved window tokens against.

        Returns
        -------
        dict
            The definition of the spec that connect's ``onset_spec`` names (the ground truth when it is not given
            separately), or the one in connect's own ``options``.
        """
        if self.connect.get("onset_spec"):
            if self.connect_onset is not None:
                return self.onset_definition(self.connect_onset)
            return self.onset_definition(_Spec("onset", self.gt.data, []))
        inline = {"options": _without_none({
            "window": self.connect.get("options.window"),
            "onset_definition": self.connect.get("options.onset_definition"),
        })}
        return self.onset_definition(inline)

    def rain_predictor(self, predictor, tokens: dict, onset_definition: dict | None, where: str) -> dict | None:
        """Convert one rain predictor of connect to a BlendConfig ``rain_predictors`` entry.

        Parameters
        ----------
        predictor : dict or str
            ``{agg, window}`` or text such as ``"max_5day"`` or ``"diff_follow"``.
        tokens : dict
            The onset definition that the window tokens of connect refer to.
        onset_definition : dict, optional
            The agreed onset definition; a token that reads a different number of days under it becomes that number.
        where : str
            The config path named in problems.

        Returns
        -------
        dict or None
            ``{"agg": ..., "window": ...}`` with a number of days or a token; None when it is invalid (a problem is
            recorded).
        """
        if isinstance(predictor, dict):
            agg = str(predictor.get("agg", "")).lower()
            window = predictor.get("window", "trigger")
        else:
            text = str(predictor)
            agg, _, rest = text.partition("_")
            last = text.rsplit("_", 1)[-1]
            digits = re.fullmatch(r"(\d+)day", last)
            window = int(digits.group(1)) if digits else last
        if agg not in ("diff", "max", "min"):
            self.problems.append(f"{where}: agg {agg!r} is not diff, max or min")
            return None
        if isinstance(window, str) and window.strip().isdigit():
            window = int(window)
        if isinstance(window, str):
            token = window.lower()
            try:
                resolved = resolve_token(token, resolve_rule(tokens))
            except ValueError:
                self.problems.append(f"{where}: window {window!r} is not a number or a known token")
                return None
            if onset_definition is not None:
                now = resolve_token(token, resolve_rule(onset_definition))
                if now != resolved:
                    return {"agg": agg, "window": resolved}
            return {"agg": agg, "window": token}
        return {"agg": agg, "window": int(window)}

    # -- domain ------------------------------------------------------------

    def grid_ids(self) -> dict:
        """Convert the grid-id convention, which every spec that states it must agree on.

        Returns
        -------
        dict
            ``decimal_digits`` and ``format`` (left out when ``trimmed``); empty when no spec states it.
        """
        stated = {s.label: _grid_id_convention(s) for s in [self.regrid, *self.raw_specs, self.connect, self.cv] if s is not None}
        agreed = self.agree("the grid-id convention (grid_id_decimal_digits, grid_id_format)", stated)
        if agreed is None:
            return {}
        return _without_none({"decimal_digits": agreed[0], "format": agreed[1] if agreed[1] != "trimmed" else None})

    def domain(self) -> dict:
        """Convert the filters of the specs and the regrid spec to the BlendConfig ``domain`` section.

        The dissemination units, the bounding box and the centroids file must agree between the specs.

        Returns
        -------
        dict
            ``grid_ids``, ``dissemination``, ``dissemination_scope``, ``area_of_interest``, ``centroids`` and
            ``regrid`` where set.
        """
        out: dict[str, Any] = {}
        grid_ids = self.grid_ids()
        if grid_ids:
            out["grid_ids"] = grid_ids

        raw_dissemination = {}
        for spec in self.raw_specs:
            path = spec.get("filter.dissemination_cells_file")
            if path is not None:
                id_col = spec.get("filter.id_col") or spec.get("filter.dissemination_id_col")
                raw_dissemination[spec.label] = _without_none({"file": path, "id_col": id_col})
        cv_file = self.cv.get("cell.dissemination")
        cv_dissemination = None
        if cv_file:
            cv_id = self.cv.get("cell.id_col") or self.cv.get("cell.dissemination_id_col")
            cv_dissemination = _without_none({"file": cv_file, "id_col": cv_id})
        else:
            self.cv.drop("cell", "cell.dissemination is empty")
        stated = dict(raw_dissemination)
        if cv_dissemination is not None:
            stated["cv_models cell"] = cv_dissemination
        dissemination = self.agree("the dissemination units", stated)
        if dissemination is not None:
            out["dissemination"] = dissemination
            if not raw_dissemination:
                out["dissemination_scope"] = "training"
        if self.regrid is not None:
            report_file = self.regrid.get("dissemination_cells_file")
            self.regrid.get("dissemination_id_col")
            if report_file is not None and not _same_path(report_file, (dissemination or {}).get("file")):
                self.notes.append(
                    f"regrid: dissemination_cells_file {report_file!r} (only the coverage report used it, "
                    "and it is not the blend's dissemination file)"
                )

        bboxes = {s.label: s.get("filter.bbox") for s in self.raw_specs}
        bbox = self.agree("filter.bbox", bboxes)
        if bbox:
            out["area_of_interest"] = {"bbox": {k: float(v) for k, v in bbox.items() if v is not None}}

        centroids = {}
        for spec in self.raw_specs:
            path = spec.get("filter.centroids_file")
            if path is not None:
                centroids[spec.label] = _without_none({"file": path, "id_col": spec.get("filter.centroids_id_col")})
        centroid_table = self.agree("filter.centroids_file", centroids)
        written = self.regrid.get("centroids_out") if self.regrid is not None else None
        if centroid_table is not None and _same_path(centroid_table["file"], written):
            self.notes.append("ground_truth: filter.centroids_file (the regrid writes it)")
        elif centroid_table is not None:
            out["centroids"] = centroid_table

        if self.regrid is not None:
            out["regrid"] = self.regrid_section()
        return out

    def regrid_section(self) -> dict:
        """Convert the regrid spec to the ``domain.regrid`` section.

        Returns
        -------
        dict
            The shapefile and its columns, ``crs``, the grid coordinate variables, ``clip_to_coverage`` and
            ``weights``, without the null values.
        """
        r = self.regrid
        out = {
            "shapefile": r.get("geometry.shapefile"),
            "unit_id_col": r.get("geometry.region_id_col") or r.get("geometry.region_key_col") or "adm3_name",
            "unit_name_col": r.get("geometry.region_name_col"),
            "parent_id_col": r.get("geometry.parent_key_col"),
            "crs": r.get("geometry.crs", "EPSG:4326"),
            "grid_lat_var": r.get("geometry.grid_lat_var"),
            "grid_lon_var": r.get("geometry.grid_lon_var"),
            "clip_to_coverage": bool(r.get("clip_to_coverage", True)),
            "weights": r.get("ground_truth.weights_in") or r.get("weights_in"),
        }
        r.drop("geometry.region_label", UNUSED)
        for key in ("weights_out", "centroids_out", "report_out"):
            r.drop(key, "written by result.save")
        if any("value_col" in entry for entry in r.get("forecasts", []) or []):
            self.notes.append("regrid: forecasts[].value_col (it only named a weights file)")
        return _without_none(out)

    # -- climatology and features -------------------------------------------

    def climatology(self, window_tags: list) -> tuple[dict, list[tuple[int, int]]]:
        """Convert the ``climatologies`` runs of the ground truth to the BlendConfig ``climatology`` section.

        Connect reads a conditional and an unconditional climatology, and one per window tag, through the prefixes the
        ``combine`` spec writes them under. The conditional and unconditional runs must agree on their settings and
        training years, each windowed run on its settings, and its training years must be the window of its tag.

        Parameters
        ----------
        window_tags : list
            The ``climatology.window_tags`` of connect, each ``"<start>_<end>"`` (an unquoted one is read by YAML as a
            number and is a problem).

        Returns
        -------
        dict
            The ``climatology`` section; empty when a run cannot be found.
        list of tuple of (int, int)
            The ``(start, end)`` years of the windowed climatologies.
        """
        runs = self.gt.get("climatologies", {}) or {}
        prefixes: dict[str, str] = {}
        for key, entry in (self.combine.get("input.climatologies", {}) or {}).items():
            prefixes[(entry or {}).get("out_prefix") or f"{key}_p_onset"] = key
        if not prefixes and self.combine.get("input.climatology_rds"):
            prefixes["clim_p_onset"] = "clim"
        base_prefix = self.connect.get("climatology.base_prefix", "clim")
        unc_prefix = self.connect.get("climatology.unconditional_prefix", "clim_unc")

        def run_for(prefix: str, what: str) -> tuple[str, dict] | None:
            """Find the ``climatologies`` run behind a connect prefix, recording a problem when there is none."""
            key = prefixes.get(f"{prefix}_p_onset")
            if key is None:
                self.problems.append(
                    f"connect reads the {what} climatology as {prefix}_p_onset_*, which no combine "
                    "input.climatologies entry writes"
                )
                return None
            if key not in runs:
                self.problems.append(f"combine climatology {key!r} has no ground_truth climatologies.{key} run")
                return None
            return key, runs[key]

        base = run_for(base_prefix, "conditional")
        unc = run_for(unc_prefix, "unconditional")
        used = {k for k, _ in (base, unc) if k}
        out: dict[str, Any] = {}
        if base is None or unc is None:
            return out, []
        base_run, unc_run = base[1], unc[1]
        if not base_run.get("conditional", True):
            self.problems.append(f"climatologies.{base[0]} feeds the conditional climatology but is unconditional")
        if unc_run.get("conditional", True):
            self.problems.append(f"climatologies.{unc[0]} feeds the unconditional climatology but is conditional")

        settings = self.agree(
            "climatology settings",
            {f"climatologies.{base[0]}": _climatology_settings(base_run), f"climatologies.{unc[0]}": _climatology_settings(unc_run)},
        )
        self.agree(
            "climatology training years",
            {f"climatologies.{base[0]}": _train_years(base_run), f"climatologies.{unc[0]}": _train_years(unc_run)},
        )
        windows: list[tuple[int, int]] = []
        for tag in window_tags:
            match = re.fullmatch(r"(\d{4})_(\d{4})", str(tag))
            if match is None:
                hint = " (an unquoted 2000_2024 in YAML reads as the number 20002024; quote it)" if isinstance(tag, int) else ""
                self.problems.append(f"connect climatology.window_tags: {tag!r} is not '<start>_<end>'{hint}")
                continue
            window = (int(match.group(1)), int(match.group(2)))
            found = run_for(f"{base_prefix}_{tag}", f"{tag} windowed")
            if found is None:
                continue
            used.add(found[0])
            self.agree(f"climatology settings for window {tag}", {"main": settings, f"climatologies.{found[0]}": _climatology_settings(found[1])})
            if _train_years(found[1]) != window:
                self.problems.append(
                    f"climatologies.{found[0]} trains on {_train_years(found[1])}, but its tag says {window}"
                )
            windows.append(window)
        for key in runs:
            if key not in used:
                self.notes.append(f"ground_truth: climatologies.{key} (connect never reads it)")
        for key in ("paths.climatology_out_dir", "paths.climatology_out_stem", "input.gt_path"):
            self.gt.drop(key, DERIVED)

        if settings is None:
            return out, windows
        if settings["onset_col"] != "onset_day":
            self.problems.append(f"climatologies use onset_col {settings['onset_col']!r}; only onset_day is supported")
        if settings["season_start"] != str(self.gt.get("options.cutoff_month_day")):
            self.problems.append(
                f"climatologies season_start_md {settings['season_start']!r} is not ground_truth "
                f"options.cutoff_month_day {self.gt.get('options.cutoff_month_day')!r} (both set the onset-day origin)"
            )
        out = {
            "issue_end": settings["issue_end"],
            # The default is features.probability_days; say it only when it differs.
            "forecast_window": None
            if settings["forecast_window"] == int(self.connect.get("day_max", 28))
            else settings["forecast_window"],
            "horizons": settings["horizons"],
            "cv_by_year": settings["cv_by_year"],
            "min_onset_years": settings["min_onset_years"],
            "train_years": list(_train_years(base_run)),
            "test_years": list(settings["test_years"]),
            "windows": [list(w) for w in windows] or None,
            "workers": self.gt.get("climatology_runtime.workers"),
        }
        return _without_none(out), windows

    def features(self) -> dict:
        """Convert the ``connect`` and ``combine`` options to the BlendConfig ``features`` section.

        Returns
        -------
        dict
            ``bins`` (``count`` and ``days``), ``probability_days``, ``trim_after_true_onset`` and ``rain_transform``.
        """
        c = self.connect
        n_bins = c.get("n_bins") if c.has("n_bins") else c.get("n_weeks", 4)
        cv_bins = self.cv.get("n_bins")
        self.agree("the number of week bins", {"connect": int(n_bins), "cv_models": cv_bins})
        days = c.get("days_per_bin") if c.has("days_per_bin") else c.get("days_per_week", 7)
        join = self.combine.get("options.join", "inner")
        if join == "full":
            self.problems.append(
                "combine: options.join 'full' (an outer join of the day-level table) has no counterpart: "
                "the samples are the units and issue dates every source covers"
            )
        elif join != "inner":
            self.notes.append(f"combine: options.join {join!r} (the old combine read anything but 'full' as inner)")
        return {
            "bins": {"count": int(n_bins), "days": int(days)},
            "probability_days": int(c.get("day_max", 28)),
            "trim_after_true_onset": bool(self.combine.get("options.trim_forecasts_after_true_onset", False)),
            "rain_transform": _rain_transform(c.get("rain_transform", "identity")),
        }

    # -- evaluation ----------------------------------------------------------

    def rename_prefix(self, text: str) -> str:
        """Rename the climatology feature prefix of connect to the one HCWF writes (``prob_clim``).

        Parameters
        ----------
        text : str
            A formula or column name.

        Returns
        -------
        str
            `text` with each use of connect's ``climatology.output_prefix`` before an underscore replaced.
        """
        prefix = self.connect.get("climatology.output_prefix", CLIMATOLOGY_FEATURE)
        if prefix == CLIMATOLOGY_FEATURE:
            return text
        return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(prefix)}(?=_)", CLIMATOLOGY_FEATURE, text)

    def evaluation(self, windows: list[tuple[int, int]]) -> dict:
        """Convert the ``cv_models`` spec to the BlendConfig ``evaluation`` section.

        Parameters
        ----------
        windows : list of tuple of (int, int)
            The windowed climatologies (`_Converter.climatology`), which the window variants must be among.

        Returns
        -------
        dict
            ``years``, ``cutoff_mode``, ``cv_methods``, ``cluster_var``, ``formulas``, ``window_variants``,
            ``climatology_baselines``, ``forecasts``, ``mme`` and, when given, ``external_predictions``.
        """
        cv = self.cv
        if cv.has("run.input_rds_override"):
            self.problems.append("cv_models: run.input_rds_override was removed (the old code rejects it)")
        cv.drop("run.MR", UNUSED)
        cv.drop("run.pipeline_input_dir", DERIVED)
        cv.drop("run.pipeline_output_dir", "set outputs.dir instead")
        years = {
            "training": list(cv.get("run.training_years", _OLD_TRAINING_YEARS)),
            "cv_holdout": list(cv.get("run.cv_holdout_years", [])),
            "true_holdout": list(cv.get("run.true_holdout_years", [])),
        }
        out: dict[str, Any] = {
            "years": years,
            "cutoff_mode": cv.get("run.cutoff_mode", "ref_onset"),
            "cv_methods": self.cv_methods(),
            "cluster_var": cv.get("cv.cluster_var", "cluster"),
        }
        formulas = {}
        for name, entry in (cv.get("models.formulas", {}) or {}).items():
            entry = entry or {}
            formulas[str(name)] = {
                "text": self.rename_prefix(str(entry.get("text", ""))),
                "enabled": bool(entry.get("enabled", True)),
            }
        out["formulas"] = formulas
        out["window_variants"] = self.window_variants(windows)
        out["climatology_baselines"] = self.climatology_baselines()
        out["forecasts"] = self.forecast_evaluations()
        out["mme"] = self.mme()
        external = cv.get("extras.external_predictions")
        if external:
            out["external_predictions"] = [_without_none(dict(e)) for e in external]
        return out

    def cv_methods(self) -> list[str]:
        """Convert ``cv.methods`` of the ``cv_models`` spec.

        Returns
        -------
        list of str
            The methods in HCWF's names (``global``, ``local``, ``clusters``); ``neighbors`` is a problem and an
            unknown method a note.
        """
        methods = []
        for method in self.cv.get("cv.methods", ["global"]):
            method = str(method)
            if method in ("global", "local"):
                methods.append(method)
            elif method.startswith("cluster"):
                methods.append("clusters")
            elif method == "neighbors":
                self.problems.append("cv_models cv.methods: 'neighbors' is not implemented (the old code raises)")
            else:
                self.notes.append(f"cv_models: cv.methods {method!r} (the old code warned and skipped it)")
        return methods

    def window_variants(self, windows: list[tuple[int, int]]) -> list[dict]:
        """Convert ``models.window_variants`` of the ``cv_models`` spec.

        Only the replacement ``X_qx`` to ``X{window}_qx`` is supported. A disabled variant is a note.

        Parameters
        ----------
        windows : list of tuple of (int, int)
            The windowed climatologies available; each window of a variant must be among them.

        Returns
        -------
        list of dict
            The BlendConfig ``window_variants`` entries.
        """
        raw = self.cv.get("models.window_variants")
        if not raw:
            return []
        entries = raw if isinstance(raw, list) else [raw]
        out = []
        for index, entry in enumerate(entries):
            where = f"cv_models models.window_variants[{index}]"
            noted = f"cv_models: models.window_variants[{index}]"
            if not entry.get("enabled", False):
                self.notes.append(f"{noted} (disabled)")
                continue
            replace = entry.get("replace") or {}
            source, target = str(replace.get("from", "")), str(replace.get("to", ""))
            if not source.endswith("_qx") or target != source[:-3] + "{window}_qx":
                self.problems.append(
                    f"{where}.replace: only 'X_qx' -> 'X{{window}}_qx' is supported, got {source!r} -> {target!r}"
                )
                continue
            end = entry.get("end_year")
            variant_windows = [[int(s), int(end)] for s in entry.get("start_years") or [] if int(s) != 1900]
            for window in variant_windows:
                if tuple(window) not in windows:
                    self.problems.append(
                        f"{where}: window {window[0]}_{window[1]} needs that climatology "
                        "(connect climatology.window_tags)"
                    )
            if entry.get("include_baseline_1900") is not None:
                self.notes.append(f"{noted}.include_baseline_1900 ({UNUSED})")
            self.notes.append(
                f"{noted}.replace (the old replace never matched, so the variant repeated its base formula; "
                "it now swaps the windowed climatology in)"
            )
            out.append(_without_none({
                "base": entry.get("base_name"),
                "windows": variant_windows,
                "replace": self.rename_prefix(source[:-3]),
                "only_if_cutoff_mode": entry.get("only_if_cutoff_mode"),
            }))
        return out

    def climatology_baselines(self) -> dict:
        """Convert the ``extras.clim_logits`` tasks of the ``cv_models`` spec to the climatology baselines.

        Returns
        -------
        dict
            ``unconditional`` (always True), ``conditional`` and ``windows`` (a list of ``[start, end]`` years).
        """
        prefix = self.connect.get("climatology.output_prefix", CLIMATOLOGY_FEATURE)
        tasks = self.cv.get("extras.clim_logits", []) or []
        out = {"unconditional": True, "conditional": False, "windows": []}
        names = [t.get("name") for t in tasks]
        if "unc_clim_raw" not in names:
            self.notes.append("cv_models: no unc_clim_raw clim_logits task; it is always scored now (the pooled summary needs it)")
        for index, task in enumerate(tasks):
            name, column = task.get("name"), task.get("base_col_prefix")
            where = f"cv_models extras.clim_logits[{index}]"
            if name == "unc_clim_raw":
                if column != f"{prefix}_unc" or task.get("earlier_col") != f"{prefix}_unc_earlier" or not task.get("earlier_is_logit", True):
                    self.problems.append(
                        f"{where}: unc_clim_raw must read {prefix}_unc with earlier_col {prefix}_unc_earlier "
                        "as logits, which connect writes"
                    )
            elif name == "clim_raw":
                if column != prefix:
                    self.problems.append(f"{where}: clim_raw must read {prefix}, which connect writes")
                starts = [int(y) for y in task.get("window_start_years") or []]
                end = task.get("window_end_year")
                if not starts or end is None:
                    out["conditional"] = True
                    continue
                for start in starts:
                    if start == 1900:
                        out["conditional"] = True
                    else:
                        out["windows"].append([start, int(end)])
            else:
                self.problems.append(
                    f"{where}: {name!r} is not unc_clim_raw or clim_raw; windowed climatologies are "
                    "clim_raw with window_start_years"
                )
        return out

    def forecast_evaluations(self) -> list[dict]:
        """Convert ``extras.forecasts`` of the ``cv_models`` spec.

        Returns
        -------
        list of dict
            One entry per forecast: ``model``, ``variant``, ``raw``, ``calibrated`` and ``platt_weights``.
        """
        variants = self.cv.get("extras.forecast_variants") or _OLD_FORECAST_VARIANTS
        for variant, suffix in variants.items():
            if _OLD_FORECAST_VARIANTS.get(variant) != suffix:
                self.problems.append(
                    f"cv_models extras.forecast_variants: {variant!r} -> {suffix!r}; only "
                    + ", ".join(f"{k} -> {v!r}" for k, v in _OLD_FORECAST_VARIANTS.items())
                    + " exist"
                )
        out = []
        for index, entry in enumerate(self.cv.get("extras.forecasts", []) or []):
            variant = entry.get("variant") or "base"
            if variant not in variants:
                self.problems.append(f"cv_models extras.forecasts[{index}]: variant {variant!r} is not in forecast_variants")
            if "fair_brier" in entry:
                self.notes.append(f"cv_models: extras.forecasts[{index}].fair_brier ({UNUSED})")
            out.append({
                "model": entry.get("name"),
                "variant": variant,
                "raw": bool(entry.get("raw", False)),
                "calibrated": bool(entry.get("calibrated", False)),
                "platt_weights": bool(entry.get("export_platt_weights", False)),
            })
        return out

    def mme(self) -> dict:
        """Convert the ``mme`` options of the ``cv_models`` spec.

        Returns
        -------
        dict
            ``enabled``, ``variants``, ``sources`` and ``optimize`` (``auto`` or ``never``).
        """
        cv = self.cv
        cv.drop("mme.mc_cores", UNUSED)
        sources = []
        for index, model in enumerate(cv.get("mme.blend_models", []) or []):
            if model.get("source") not in ("clim", "forecast"):
                self.notes.append(f"cv_models: mme.blend_models[{index}] (source {model.get('source')!r}; the old code skipped it)")
                continue
            sources.append({"name": model.get("name"), "source": model.get("source"),
                            "cal_variant": model.get("cal_variant", "base")})
        if cv.get("mme.weights_year_tag"):
            self.notes.append(_WEIGHTS_NOTE.format(tag=cv.get("mme.weights_year_tag")))
        return _without_none({
            "enabled": bool(cv.get("mme.enabled", False)),
            "variants": list(cv.get("mme.variants", ["fixed_cutoff"])),
            "sources": sources,
            "optimize": "auto" if cv.get("mme.optimize_if_holdout_is_full_2000_2024", True) else "never",
        })

    # -- assembly ------------------------------------------------------------

    def convert(self) -> Conversion:
        """Convert the whole spec set.

        Drops the keys that are derived or run settings, builds the BlendConfig mapping from the specs and converts it
        with `convert_blend`; the keys nothing read are listed as notes.

        Returns
        -------
        Conversion
            The run mapping, its comments and the notes (those of the spec set first, in pipeline order).

        Raises
        ------
        ConvertError
            The specs disagree or cannot be converted; the error lists every problem.
        """
        for spec in self.raw_specs:
            for key in ("input.parallel", "input.workers"):
                spec.drop(key, RUNTIME)
            for key in ("output", "input.wide_prefix", "id"):
                spec.drop(key, DERIVED)
            expected = "ground_truth_rainfall" if spec is self.gt else "rainfall_forecast"
            if spec.get("type", expected) != expected:
                self.problems.append(f"{spec.label}: type is {spec.get('type')!r}, expected {expected!r}")
        for key in ("input.ground_truth_wide_rds", "input.climatology_rds", "output"):
            self.combine.drop(key, DERIVED)
        for key in ("input_rds", "output_rds"):
            self.connect.drop(key, DERIVED)
        if self.connect.get("onset_spec"):
            self.notes.append(
                f"connect: onset_spec {self.connect.get('onset_spec')!r} (rain windows now resolve "
                "against the one onset definition)"
            )
        self.connect.drop("climatology.output_prefix", "the climatology features are always prob_clim")

        data: dict[str, Any] = {"name": self.name}
        if self.data_root is not None:
            data["data_root"] = self.data_root
        connect_models = list(self.connect.get("forecast_models", []) or [])
        if not connect_models:
            self.problems.append("connect: forecast_models is required")
        data["onset"] = self.onset(self.uses_ref_series())
        data["domain"] = self.domain()
        data["observations"] = self.observations()
        data["models"] = self.models(connect_models, self.definition)
        data["climatology"], windows = self.climatology(list(self.connect.get("climatology.window_tags", []) or []))
        data["features"] = self.features()
        data["evaluation"] = self.evaluation(windows)

        for spec in [self.regrid, *self.raw_specs, self.combine, self.connect, self.cv]:
            if spec is not None:
                self.notes += [f"{spec.label}: {path} ({UNUSED})" for path in spec.unread()]
        if self.problems:
            raise ConvertError(self.problems)
        converted = convert_blend(data, region=self.region)
        return Conversion(converted.mapping, (*self.sorted_notes(), *converted.notes), converted.comments)

    def sorted_notes(self) -> list[str]:
        """Sort the notes by spec, in pipeline order.

        Returns
        -------
        list of str
            The notes, grouped by the spec label they start with (regrid, ground truth, forecasts, combine, connect,
            ``cv_models``), others last.
        """
        order = ["regrid", "ground_truth", *(s.label for s in self.forecasts.values()), "combine", "connect", "cv_models"]
        rank = {label: i for i, label in enumerate(order)}
        return sorted(self.notes, key=lambda note: rank.get(note.split(":", 1)[0], len(order)))

    def uses_ref_series(self) -> bool:
        """Tell whether the evaluation of ``cv_models`` uses the ref series.

        Returns
        -------
        bool
            True for the ``ref_onset`` cutoff mode, a ``ref_onset`` forecast or MME variant, or a formula that reads a
            ``_p_onset_ref_`` column.
        """
        if self.cv.data.get("run", {}).get("cutoff_mode", "ref_onset") == "ref_onset":
            return True
        extras = self.cv.data.get("extras") or {}
        if any((f or {}).get("variant") == "ref_onset" for f in extras.get("forecasts") or []):
            return True
        if any((m or {}).get("cal_variant") == "ref_onset" for m in (self.cv.data.get("mme") or {}).get("blend_models") or []):
            return True
        texts = [str((f or {}).get("text", "")) for f in ((self.cv.data.get("models") or {}).get("formulas") or {}).values()]
        return any("_p_onset_ref_" in text for text in texts)




def _lower(value):
    """Lower-case and strip a value that may be None.

    Parameters
    ----------
    value : object
        A string, or None.

    Returns
    -------
    str or None
        The stripped lower-case text; None stays None.
    """
    return None if value is None else str(value).strip().lower()


def _rain_transform(value):
    """Write a rain transform in the BlendConfig form.

    Parameters
    ----------
    value : object
        A transform name, or a mapping with ``power``.

    Returns
    -------
    object
        ``"power:<p>"`` for a mapping with ``power``; otherwise `value` unchanged.
    """
    if isinstance(value, dict) and "power" in value:
        return f"power:{value['power']}"
    return value




def convert_specs(
    *,
    name: str,
    ground_truth: dict,
    forecasts: dict[str, dict],
    combine: dict,
    connect: dict,
    cv_models: dict,
    regrid: dict | None = None,
    connect_onset_spec: dict | None = None,
    data_root: str | None = None,
    region: str | None = None,
) -> Conversion:
    """Convert a set of stage specs (as loaded from YAML) to an HCWF run.

    The specs' paths are relative to the directory the old scripts ran from: give it as `data_root`.

    Parameters
    ----------
    name : str
        The project name.
    ground_truth : dict
        The ground-truth ``raw_data`` spec.
    forecasts : dict of str to dict
        The ``raw_data`` spec of each model, by model name (the combine and connect key).
    combine : dict
        The ``combine`` spec.
    connect : dict
        The ``connect`` spec.
    cv_models : dict
        The ``cv_models`` spec.
    regrid : dict, optional
        The ``regrid`` spec; None when the data is not regridded.
    connect_onset_spec : dict, optional
        The ``raw_data`` spec that connect's ``onset_spec`` names, when that is not the ground-truth spec.
    data_root : str, optional
        The folder the old scripts ran from.
    region : str, optional
        An HCWF region id for the run; None is no region.

    Returns
    -------
    Conversion
        The run mapping, its comments and the notes.
    """
    return _Converter(
        name=name, ground_truth=ground_truth, forecasts=forecasts, combine=combine,
        connect=connect, cv_models=cv_models, regrid=regrid, connect_onset_spec=connect_onset_spec,
        data_root=data_root, region=region,
    ).convert()


def _load(path: str | Path | None) -> dict | None:
    """Load a YAML spec file.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        The file; None loads nothing.

    Returns
    -------
    dict or None
        The mapping (empty for an empty file); None when `path` is None.
    """
    if path is None:
        return None
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def convert_spec_files(
    *,
    name: str,
    ground_truth: str | Path,
    forecasts: dict[str, str | Path],
    combine: str | Path,
    connect: str | Path,
    cv_models: str | Path,
    regrid: str | Path | None = None,
    data_root: str | None = None,
    region: str | None = None,
) -> Conversion:
    """Run `convert_specs` on YAML files.

    The spec that connect's ``onset_spec`` names is looked up beside the ground-truth spec (``<onset_spec>.yml``) when it
    is not the ground-truth spec itself.

    Parameters
    ----------
    name : str
        The project name.
    ground_truth : str or pathlib.Path
        The ground-truth ``raw_data`` spec file.
    forecasts : dict of str to (str or pathlib.Path)
        The ``raw_data`` spec file of each model, by model name.
    combine : str or pathlib.Path
        The ``combine`` spec file.
    connect : str or pathlib.Path
        The ``connect`` spec file.
    cv_models : str or pathlib.Path
        The ``cv_models`` spec file.
    regrid : str or pathlib.Path, optional
        The ``regrid`` spec file; None when the data is not regridded.
    data_root : str, optional
        The folder the old scripts ran from.
    region : str, optional
        An HCWF region id for the run; None is no region.

    Returns
    -------
    Conversion
        The run mapping, its comments and the notes.
    """
    connect_spec = _load(connect)
    onset_spec = None
    onset_id = (connect_spec or {}).get("onset_spec")
    if onset_id and onset_id != Path(ground_truth).stem:
        candidate = Path(ground_truth).parent / f"{onset_id}.yml"
        onset_spec = _load(candidate) if candidate.exists() else None
    return convert_specs(
        name=name,
        ground_truth=_load(ground_truth),
        forecasts={model: _load(path) for model, path in forecasts.items()},
        combine=_load(combine),
        connect=connect_spec,
        cv_models=_load(cv_models),
        regrid=_load(regrid),
        connect_onset_spec=onset_spec,
        data_root=data_root,
        region=region,
    )


