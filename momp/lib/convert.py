"""Convert between the flat config dict and the nested YAML run config; read old config.in files.

``flat_to_mapping`` turns the flat dict the pipeline uses into the nested mapping of a run YAML
(sections, a `models:` list); ``render_yaml`` writes a mapping as readable YAML, optionally with
a comment per value saying where it came from (used for resolved_config.yaml). ``read_legacy``
exec()s an old-style config.in; it exists only for `momp-run convert`, which writes the
equivalent YAML. Nothing else in ROMP reads config.in any more.
"""
from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml

from momp.lib import config_schema as schema
from momp.lib.config_schema import ConfigSchemaError
from momp.lib.dataset_schema import suggest

SECTION_ORDER = ("observations", "reference", "models", "onset", "evaluation", "metrics", "output")
INTERNAL_KEYS = ("dataset_report", "config_sources", "onset_legacy", "lead_window", "onset_lead_lines", "run_paths",
                 "config_file", "cache_dir", "short_names", "run_overrides", "accumulation_detected")
"""Keys the loader adds; they are not part of a run config."""
_MODEL_LISTS = ("model_list", "model_dir_list", "model_var_list", "unit_cvt_list", "file_pattern_list",
                "model_dims_list")
_LEGACY_RENAMES = {"mok": "earliest_onset_date"}
_UNIT_KEYS = ("obs_unit_cvt", "ref_model_unit_cvt")
_MODEL_EXTRA = ("accumulation_list", "daily_boundary_list", "accumulation_detected")


def units_from_cvt(cvt, where: str) -> str:
    """'mm' or 'm' for an old unit_cvt multiplier (None or 1 -> mm, 1000 -> m); anything else is an error."""
    if cvt is None or cvt == 1:
        return "mm"
    if cvt == 1000:
        return "m"
    raise ConfigSchemaError(f"{where} = {cvt!r} cannot be written as units (only mm or m). Fix: ROMP no longer "
                            "takes a free multiplier; convert the data to mm or m, or tell us what it is")


def read_legacy(path) -> dict:
    """Read an old-style config.in (Python assignments) into a flat dict; the only place that exec()s a config."""
    from momp.utils.practical import set_dir

    item = set_dir(str(path))
    if not item.exists():
        raise FileNotFoundError(f"Could not find config file: {path}")
    text = item.read_text()
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    local: dict = {}
    exec(text, {}, local)  # noqa: S102 - old config.in files are Python by definition; this is the converter
    excluded = {"f", "config_file_path", "params_in"}
    flat = {k: v for k, v in local.items() if not k.startswith("__") and not callable(v) and k not in excluded}
    for old, new in _LEGACY_RENAMES.items():
        if old in flat:
            flat[new] = flat.pop(old)
    return flat


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
        if key in INTERNAL_KEYS or (key == "earliest_onset_date" and value is None):
            continue
        if key in _UNIT_KEYS or key in _MODEL_EXTRA:
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
    if "model_list" in flat:
        entries = []
        for i, name in enumerate(flat["model_list"]):
            entry = {"name": name, "path": plain(flat["model_dir_list"][i]), "variable": flat["model_var_list"][i],
                     "units": units_from_cvt(flat["unit_cvt_list"][i], f"unit_cvt_list[{i}]"),
                     "file_pattern": flat["file_pattern_list"][i]}
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
        sections["observations"]["units"] = units_from_cvt(flat.get("obs_unit_cvt"), "obs_unit_cvt")
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


def convert_file(source, target, force: bool = False) -> Path:
    """Convert an old config.in to a YAML run config at target; refuses to overwrite unless force."""
    flat = read_legacy(source)
    flat.setdefault("model_lead_label", flat.get("model_lead_label"))
    mapping, _ = flat_to_mapping(flat)
    target = Path(target)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists. Pass --force to overwrite it")
    header = (f"# Converted from {Path(source).name} by `momp-run convert`. Check it, then run:\n"
              f"#   momp-run -p {target.name}\n")
    text = render_yaml(mapping, header=header)
    schema_check = yaml.safe_load(text)
    from momp.lib.config_yaml import build_config

    build_config(schema_check, source=str(target))
    target.write_text(text)
    return target
