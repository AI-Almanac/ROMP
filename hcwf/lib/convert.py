"""Convert between the flat config dict and the nested YAML run config; translate the old flat onset keys.

``flat_to_mapping`` turns the flat dict the pipeline uses into the nested mapping of a run YAML
(sections, a `models:` list); ``render_yaml`` writes a mapping as readable YAML, optionally with
a comment per value saying where it came from (used for resolved_config.yaml). ``convert_flat_onset``
is the only reader of the flat onset keys (wet_init, wet_spell, ...) of earlier versions; `hcwf-run convert` uses it
to write the equivalent onset definition.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import yaml

from hcwf.lib import config_schema as schema
from hcwf.lib.config_schema import ConfigSchemaError
from hcwf.lib.dataset_schema import suggest

SECTION_ORDER = ("observations", "reference", "models", "onset", "units", "blend", "evaluation", "metrics", "output")
INTERNAL_KEYS = ("dataset_report", "config_sources", "lead_window", "onset_lead_lines", "run_paths",
                 "config_file", "cache_dir", "short_names", "run_overrides", "accumulation_detected")
"""Keys the loader adds; they are not part of a run config."""
_MODEL_LISTS = ("model_list", "model_dir_list", "model_var_list", "unit_cvt_list", "file_pattern_list",
                "model_dims_list", "model_paths_list")
_UNIT_KEYS = ("obs_unit_cvt", "ref_model_unit_cvt")
_OBS_LISTED = {"obs_paths": "paths", "obs_dims": "dims"}
_MODEL_EXTRA = ("accumulation_list", "daily_boundary_list", "accumulation_detected")


def units_from_cvt(cvt, where: str) -> str:
    """'mm' or 'm' for an old unit_cvt multiplier (None or 1 -> mm, 1000 -> m); anything else is an error."""
    if cvt is None or cvt == 1:
        return "mm"
    if cvt == 1000:
        return "m"
    raise ConfigSchemaError(f"{where} = {cvt!r} cannot be written as units (only mm or m). Fix: HCWF no longer "
                            "takes a free multiplier; convert the data to mm or m, or tell us what it is")


def plain(value):
    """Tuples to lists and Paths to text, recursively, so the value dumps as plain YAML."""
    if hasattr(value, "to_mapping"):
        return plain(value.to_mapping())
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def flat_to_mapping(flat: dict, sources: dict | None = None) -> tuple[dict, dict]:
    """The nested run-config mapping equivalent to a flat config dict, with a comment per path.

    Returns (mapping, comments) where comments maps a path tuple such as ('onset', 'definition') to the
    source text of its value. Keys that are not config keys raise ConfigSchemaError with a suggestion.
    """
    sources = sources or {}
    top: dict = {}
    sections: dict = {name: {} for name in SECTION_ORDER}
    comments: dict = {}
    errors = []
    for key, value in flat.items():
        if key in INTERNAL_KEYS:
            continue
        if key in _UNIT_KEYS or key in _MODEL_EXTRA:
            continue
        if (key == "obs_dir" and not value) or (key == "obs_file_pattern" and flat.get("obs_paths")):
            continue   # observations given as listed files have neither a folder nor a pattern
        if key in _OBS_LISTED:   # the listed files and dims of the observations, written in the observations section
            if value:
                sections["observations"][_OBS_LISTED[key]] = plain(value)
            continue
        field = schema.BY_KEY.get(key)
        if field is None:
            errors.append(f"unknown key '{key}'.{suggest(key, schema.BY_KEY)}")
            continue
        if key in _MODEL_LISTS:
            continue
        target = sections[field.section] if field.section else top
        target[field.nested_name] = plain(value)
        comments[(field.section, field.nested_name) if field.section else (key,)] = sources.get(key)
    if errors:
        raise ConfigSchemaError("Cannot write this config as YAML:\n" + "\n".join(f"  • {e}" for e in errors))
    blend = schema.has_blend(flat)
    if "model_list" in flat:
        entries = []
        for i, name in enumerate(flat["model_list"]):
            cvt = flat["unit_cvt_list"][i]
            entry = {"name": name}
            if flat["model_dir_list"][i]:
                entry["path"] = plain(flat["model_dir_list"][i])
            entry["variable"] = flat["model_var_list"][i]
            if blend:   # a blend keeps its own multiplier (declared units still win)
                entry["unit_cvt"] = 1.0 if cvt is None else cvt
            else:
                entry["units"] = units_from_cvt(cvt, f"unit_cvt_list[{i}]")
            listed = (flat.get("model_paths_list") or [()] * (i + 1))[i]
            if listed:
                entry["paths"] = plain(listed)
            else:
                entry["file_pattern"] = flat["file_pattern_list"][i]
            accumulation = (flat.get("accumulation_list") or ["auto"] * (i + 1))[i]
            detected = (flat.get("accumulation_detected") or [False] * (i + 1))[i]
            if accumulation != "auto" and not detected:   # a detected value is re-detected on the next run
                entry["accumulation"] = accumulation
            boundary = (flat.get("daily_boundary_list") or [None] * (i + 1))[i]
            if boundary:
                entry["daily_boundary"] = boundary
            dims = (flat.get("model_dims_list") or [None] * (i + 1))[i]
            if dims:
                entry["dims"] = plain(dims)
            entries.append(entry)
        sections["models"] = entries
    if "obs_dir" in flat:
        cvt = flat.get("obs_unit_cvt")
        if blend:
            sections["observations"]["unit_cvt"] = 1.0 if cvt is None else cvt
        else:
            sections["observations"]["units"] = units_from_cvt(cvt, "obs_unit_cvt")
    if "ref_model" in flat and "ref_model_dir" in flat:
        sections["reference"]["units"] = units_from_cvt(flat.get("ref_model_unit_cvt"), "ref_model_unit_cvt")
    mapping = dict(top)
    for name in SECTION_ORDER:
        if sections[name]:
            mapping[name] = sections[name]
    return mapping, comments


def _flow(value) -> str:
    """One value as single-line YAML."""
    text = yaml.safe_dump(value, default_flow_style=True, width=10 ** 9, allow_unicode=True)
    if text.endswith("\n...\n"):
        text = text[:-5]
    return text.strip()


def render_yaml(mapping: dict, comments: dict | None = None, header: str = "") -> str:
    """Write a run-config mapping as YAML text, one value per line, with `# source` comments where known."""
    comments = comments or {}
    lines = [header.rstrip()] if header else []

    def note(path):
        """The trailing comment for a path."""
        text = comments.get(path)
        return f"  # {text}" if text else ""

    for key, value in mapping.items():
        if key == "models":
            lines.append("models:")
            for entry in value:
                lines.append("  - " + _flow(entry))
        elif isinstance(value, dict) and key in schema.SECTIONS:
            lines.append(f"{key}:")
            for sub, subvalue in value.items():
                lines.append(f"  {sub}: {_flow(subvalue)}{note((key, sub))}")
        else:
            lines.append(f"{key}: {_flow(value)}{note((key,))}")
    return "\n".join(lines) + "\n"


FLAT_ONSET_KEYS = ("wet_init", "wet_threshold", "wet_spell", "dry_threshold", "dry_spell", "dry_extent",
                   "thresh_file", "thresh_var", "earliest_onset_date", "mok")
"""The flat onset keys earlier versions read at the top level of a run config; only `hcwf-run convert` reads them now."""
EXTEND_END_DAYS = 47
"""The old observed search read rain this many days past end_date; the converted season ends where the last wet spell fit."""
THRESHOLD_NAME = "legacy_threshold"
"""The name the converted definition gives the per-cell threshold of a thresh_file."""


def _tenths(flat: dict, key: str) -> int:
    """The value of `key` in whole tenths of a mm, or an error if it is finer than 0.1 mm."""
    value = flat[key]
    try:
        tenths = round(float(value) * 10)
        exact = abs(float(value) * 10 - tenths) < 1e-6
    except (TypeError, ValueError):
        raise ConfigSchemaError(f"{key} = {value!r} is not a number of mm. Fix: set {key} to a number") from None
    if not exact:
        raise ConfigSchemaError(f"{key} = {value!r} is not a whole number of tenths of a mm; the onset component "
                                f"works on 0.1 mm. Fix: round {key} to 0.1 mm, e.g. {tenths / 10}")
    return tenths


def _mm(tenths: int) -> str:
    """An amount in tenths of a mm as definition text, e.g. 'at least 20 mm'."""
    return f"{tenths // 10} mm" if tenths % 10 == 0 else f"{tenths / 10:.1f} mm"


def _days(flat: dict, key: str, minimum: int) -> int:
    """The value of `key` as a whole number of at least `minimum`."""
    value = flat[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < minimum:
        raise ConfigSchemaError(f"{key} = {value!r} must be a whole number of at least {minimum}")
    return int(value)


def _normalised(flat: dict) -> dict:
    """`flat` without null values and with its dates in the forms the old schema accepted ('YYYY-MM-DD', 'MM-DD',
    lists, YAML dates) turned into (year, month, day) and (month, day) tuples by the schema's own kinds."""
    flat = {key: value for key, value in flat.items() if value is not None}
    for key, kind in (("start_date", "date"), ("end_date", "date"), ("earliest_onset_date", "month_day"),
                      ("mok", "month_day")):
        if key in flat:
            try:
                flat[key] = schema.KINDS[kind][1](flat[key])
            except (ValueError, TypeError):
                raise ConfigSchemaError(f"{key} = {flat[key]!r} is not {schema.KINDS[kind][0]}") from None
    return flat


def _season_end(end_date, window: int, start: tuple[int, int]) -> datetime.date:
    """The last day of the season as a month-day (in 2001): end_date plus the old read-ahead, less the wet spell.

    The old search counted these days in each file's own calendar. A fixed month-day is the same only when no
    29 February lies in between, so an end_date from 15 January to 28 February is refused; a season that ends after
    the year turns is kept only when it ends before it starts (a season may wrap into the next year but not exceed it).
    """
    days = EXTEND_END_DAYS - (window - 1)
    try:
        ends = [datetime.date(year, *end_date[1:]) + datetime.timedelta(days=days) for year in (2001, 2004)]
    except ValueError:
        raise ConfigSchemaError(f"end_date {list(end_date)} is 29 February, which the season grammar does not allow. "
                                "Fix: use 28 February or 1 March") from None
    if (ends[0].month, ends[0].day) != (ends[1].month, ends[1].day):
        raise ConfigSchemaError(
            f"end_date {list(end_date)} plus {days} days passes 29 February, so the old season ended one day earlier in "
            "leap years, and a definition's season end is one fixed month-day. Fix: write the season of the "
            f"definition by hand (onset.rule.season.to: {ends[0]:%m-%d} for ordinary years, {ends[1]:%m-%d} for leap years)")
    end = ends[0]
    if end.year != 2001 and (end.month, end.day) >= start:
        raise ConfigSchemaError(
            f"the season from {start[0]:02d}-{start[1]:02d} to end_date {list(end_date)} plus {days} days ends on "
            f"{end:%m-%d} of the next year, after the day it started; a season cannot be longer than a year. "
            "Fix: set a later start (earliest_onset_date or start_date) or an earlier end_date")
    return end


def flat_onset_to_rule(flat: dict) -> tuple[dict, dict]:
    """The onset definition (a mapping, for `onset.rule`) and the `onset.thresholds` the old flat onset keys describe.

    Needs wet_init, wet_spell, dry_spell, dry_extent and end_date (a date in any form the old schema read) and, without
    thresh_file, wet_threshold; reads start_date (1 January if absent), earliest_onset_date (or its old name mok),
    thresh_file and thresh_var (required with thresh_file). The rule reproduces the old detection with these
    differences: dry_threshold was never used and is dropped; the old forecast dry window was one day shorter than the
    observed one, the observed one (days 2 to dry_extent + 1 after the candidate) is used for both; the season ends
    where the old observed search ended, the last candidate whose wet spell fit in the data read up to end_date + 47
    days (a season end after the year turns wraps into the next year). The old run also took forecasts as given
    (`forecast_history: none`) and labelled leads by `end`; `convert_flat_onset` writes both next to the rule.
    """
    flat = _normalised(flat)
    if "mok" in flat:
        if "earliest_onset_date" in flat:
            raise ConfigSchemaError("the config sets both 'mok' and 'earliest_onset_date'. Fix: remove 'mok'")
        flat["earliest_onset_date"] = flat.pop("mok")
    missing = [key for key in ("wet_init", "wet_spell", "dry_spell", "dry_extent", "end_date") if key not in flat]
    if "thresh_file" not in flat and "wet_threshold" not in flat:
        missing.append("wet_threshold")
    if missing:
        raise ConfigSchemaError(f"the flat onset keys lack {', '.join(missing)}. Fix: add them, then convert again")
    if "thresh_file" in flat and not flat.get("thresh_var"):
        raise ConfigSchemaError(f"thresh_file = {flat['thresh_file']!r} needs thresh_var, the variable holding the "
                                "grid (the old reader needed it too). Fix: set thresh_var")
    wet_init = _tenths(flat, "wet_init")
    window = _days(flat, "wet_spell", 1)
    dry_spell = _days(flat, "dry_spell", 1)
    dry_extent = _days(flat, "dry_extent", 0)
    amount = THRESHOLD_NAME if "thresh_file" in flat else _mm(_tenths(flat, "wet_threshold"))
    onset: dict = {"window": f"{window} days",
                   "require": [{"all days": "wet"}, {"total": f"more than {amount}"}]}
    onset["reject if"] = ([{"dry run": f"at least {dry_spell} days", "within": f"days 2 to {dry_extent + 1}"}]
                          if dry_extent > window and dry_extent >= dry_spell else [])
    start = flat.get("earliest_onset_date") or (flat.get("start_date") or (None, 1, 1))[1:]
    start = tuple(start)
    end = _season_end(flat["end_date"], window, start)
    rule = {"id": "converted_flat_onset", "version": 1, "title": "Converted from the flat onset keys",
            "citation": "TODO", "wording": "TODO", "status": "draft",
            "notes": "Written by `hcwf-run convert` from wet_init, wet_threshold, wet_spell, dry_spell, dry_extent, "
                     "thresh_file and earliest_onset_date.",
            "wet_day": f"at least {_mm(wet_init)}",
            "season": {"from": f"{start[0]:02d}-{start[1]:02d}", "to": f"{end.month:02d}-{end.day:02d}"}}
    thresholds = {}
    if "thresh_file" in flat:
        rule["thresholds"] = {THRESHOLD_NAME: {"mean total over": f"{window} days"}}
        thresholds[THRESHOLD_NAME] = f"{flat['thresh_file']}:{flat['thresh_var']}"
    rule["onset"] = onset
    return rule, thresholds


_OLD_RUN_SETTINGS = {"forecast_history": ("forecast_history", "none"), "lead_label": ("model_lead_label", "end")}
"""onset keys the old flat-key run always used: forecasts as given (no observed rain stitched before them) and leads
labelled by `end`; the second item is the flat key a config may also write them under."""
_OLD_YEARS = {"years": "years", "years_clim": "climatology_years"}
"""Old flat keys for explicit years and their `evaluation` names."""


def convert_flat_onset(source, target, force: bool = False) -> Path:
    """Write a copy of the run config `source` with its flat onset keys turned into an `onset: {rule: ...}` section.

    The keys are the ones earlier versions read at the top level (FLAT_ONSET_KEYS); the rule is checked by loading it
    as a definition. The old run took forecasts as given and labelled leads by `end`, so `onset.forecast_history: none`
    and `onset.lead_label: end` are written too unless the config sets them. Old `years` / `years_clim` become
    `evaluation.years` / `evaluation.climatology_years`. Refuses to overwrite target unless force.
    """
    from hcwf.onset import definition as definition_mod

    target = Path(target)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists. Pass --force to overwrite it")
    raw = yaml.safe_load(Path(source).read_text())
    if not isinstance(raw, dict):
        raise ConfigSchemaError(f"{source} is not a run config (expected a mapping of keys at the top level)")
    flat = {key: raw.pop(key) for key in FLAT_ONSET_KEYS if key in raw}
    if all(value is None for value in flat.values()):
        raise ConfigSchemaError(f"{source} has no flat onset keys ({', '.join(FLAT_ONSET_KEYS)}); nothing to convert")
    evaluation = raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else {}
    for key in ("start_date", "end_date"):
        flat[key] = evaluation.get(key, raw.get(key))
    onset = raw.setdefault("onset", {})
    clash = [key for key in ("definition", "rule") if onset.get(key) is not None]
    if clash:
        raise ConfigSchemaError(f"the config has flat onset keys and onset.{clash[0]}. Fix: keep one of them")
    dropped = [key for key in schema.REMOVED_KEYS if key in raw and key not in FLAT_ONSET_KEYS]
    for key in dropped:
        raw.pop(key)
    rule, thresholds = flat_onset_to_rule(flat)
    definition_mod.loads(yaml.safe_dump(rule, sort_keys=False))
    onset["rule"] = rule
    if thresholds:
        onset["thresholds"] = thresholds
    for name, (flat_key, value) in _OLD_RUN_SETTINGS.items():
        if onset.get(name) is None and raw.get(flat_key) is None:
            onset[name] = value
    notes = []
    years = {name: raw.pop(key) for key, name in _OLD_YEARS.items() if key in raw}
    years = {name: value for name, value in years.items() if value not in (None, "All")}   # 'All' was the default
    if years:
        raw["evaluation"] = {**(raw.get("evaluation") or {}), **years}
        notes.append(f"# Moved to evaluation: {', '.join(years)}")
    if dropped:
        notes.append(f"# Dropped keys that have no effect any more: {', '.join(dropped)}")
    if str(raw.get("region")) == "India":
        notes.append("# region 'India' now selects the registry's India region (its own box), not the old fixed box")
    notes.append("# Written for the old run: onset.forecast_history none and onset.lead_label end (unless set).")
    notes.append("# Known differences: the old earliest_onset_date also filtered the climatology reference, now only "
                 "the observed and")
    notes.append("# forecast searches start there; and the season end is a calendar date, where the old end counted "
                 "days in each model's own calendar.")
    header = (f"# Converted from {Path(source).name} by `hcwf-run convert`: the flat onset keys became onset.rule.\n"
              + "".join(note + "\n" for note in notes)
              + f"# Check it, then run: hcwf-run -p {target} --check\n")
    target.write_text(header + yaml.safe_dump(raw, sort_keys=False, width=120))
    return target
