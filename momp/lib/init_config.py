"""Write a starting run config for `momp-run init`.

The file holds what only the user knows (region, years, datasets). Values ROMP can work out itself (the
region's onset definition, day boundary and climatology years, schema defaults) are NOT copied in; they are
resolved at run time. Dataset values detected from paths you pass with --obs/--model are written, each with
a comment showing the evidence, so you can review and edit them.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from momp.lib import dataset_detect as detect
from momp.lib.convert import _flow
from momp.lib.dataset_schema import DatasetError, expand_path
from momp.onset import regions

_OBS_PLACEHOLDERS = (("file_pattern", '"{}.nc"', "the year placeholder; detected from the files if left out"),
                     ("variable", "rainfall", "rainfall variable; detected if it is the only one"),
                     ("units", "mm", "mm, mm/day, m or cm; read from the file's attribute if left out"),
                     ("day_boundary", '"00 UTC, labelled by start"',
                      "when the day starts and which date labels it; the region's if left out"))
_MODEL_PLACEHOLDERS = (("file_pattern", '"{}.nc"', "the year placeholder; detected from the files if left out"),
                       ("variable", "tp", "rainfall variable; detected if it is the only one"),
                       ("units", "m", "mm, mm/day, m or cm; read from the file's attribute if left out"),
                       ("lead_label", "start", "start: lead k is dated init + k - 1; end: init + k. "
                                               "Detected against the observations if left out"))


def _season_bounds(region_name: str | None):
    """((month, day), (month, day)) of the region's default definition season, or the demo's May to July."""
    from momp.onset import definition as definition_mod
    from momp.onset.model import DerivedSeasons

    try:
        default = regions.definitions_for(region_name)[0]
        seasons = definition_mod.load(default).seasons
        if not isinstance(seasons, DerivedSeasons):
            s = seasons[0]
            if (s.end.month, s.end.day) > (s.start.month, s.start.day):
                return (s.start.month, s.start.day), (s.end.month, s.end.day), default
    except Exception:  # noqa: BLE001 - any failure falls back to the default window
        pass
    return (5, 1), (7, 31), None


def _entry_lines(res: detect.Resolution, placeholders, indent: str, first: str = "") -> list[str]:
    """YAML lines for one dataset: detected/specified values with evidence, TODOs for what cannot be detected."""
    lines = []
    shown = set()
    for line in res.lines:
        if line.key == "path" or line.key in shown:
            continue
        shown.add(line.key)
        note = f"{line.how}" + (f": {line.evidence}" if line.evidence else "")
        lines.append(f"{indent}{line.key}: {_flow(line.value)}  # [{note}]")
    for key, message in res.problems.items():
        name = key.split(".")[0]
        example = next((e for k, e, _ in placeholders if k == name), "...")
        lines.append(f"{indent}# {name}: {example}  # TODO: {message}")
    return lines


def render(region=None, obs=None, models=(), years=(2019, 2024), definition=None) -> str:
    """The text of a starting run config; obs and models are folders (or None for placeholders)."""
    notes = []
    obs_res = None
    obs_lines: list[str]
    if obs:
        obs_res = detect.resolve_entry({"path": obs}, "observations")
        obs_lines = [f"  path: {_flow(obs)}"] + _entry_lines(obs_res, _OBS_PLACEHOLDERS, "  ")
    else:
        obs_lines = ["  path: /path/to/observations  # TODO: folder of the yearly observation files"] + [
            f"  # {k}: {example}  # {text}" for k, example, text in _OBS_PLACEHOLDERS]
    region_line = None
    if region:
        region_line = f"region: {regions.load_region(region).title}"
    elif obs_res is not None and obs_res.extent:
        found, evidence = detect.detect_region(obs_res.extent)
        if found != "unknown":
            region_line = f"region: {regions.load_region(found).title}  # [detected: {evidence}]"
    region_line = region_line or "region: Ethiopia  # TODO: your region, see `python -m momp.onset regions`"
    region_name = region_line.split(":")[1].split("#")[0].strip()
    try:
        regions.load_region(region_name)
    except regions.RegionError:
        region_name = None
    start, end, default = _season_bounds(region_name)
    obs_for_labels = dict(obs_res.values) if obs_res is not None else None
    if obs_for_labels is not None and not obs_for_labels.get("day_boundary"):
        boundary = next((r.day_boundary for r in ([regions.load_region(region_name), *regions.ancestors(
            regions.load_region(region_name))] if region_name else []) if r.day_boundary), None)
        obs_for_labels["day_boundary"] = boundary
        if boundary and obs_res is not None:
            obs_res.problems.pop("day_boundary", None)
            obs_lines = [line for line in obs_lines if "day_boundary" not in line]
            obs_lines.append(f"  # day_boundary: from the region ({boundary}); set it here to override")
    model_blocks, probabilistic = [], False
    for i, path in enumerate(models or [None]):
        if path:
            res = detect.resolve_entry({"path": path}, "forecast", obs=obs_for_labels)
            name = Path(expand_path(path)).name or f"MODEL{i + 1}"
            block = [f"  - name: {_flow(name)}", f"    path: {_flow(path)}"] + _entry_lines(res, _MODEL_PLACEHOLDERS, "    ")
            probabilistic = probabilistic or (res.values.get("members") or 1) > 1
        else:
            block = ["  - name: MODEL  # TODO: model name used in output files (no spaces)",
                     "    path: /path/to/forecasts  # TODO: folder of the yearly forecast files"] + [
                f"    # {k}: {example}  # {text}" for k, example, text in _MODEL_PLACEHOLDERS]
        model_blocks.append("\n".join(block))
    out = ["# ROMP run config. Edit, then run:  momp-run -p run.yaml",
           "# Only what you must decide is here. The region supplies the onset definition, day boundary and",
           "# climatology years; unset keys take their defaults. Each run writes resolved_config.yaml (every value",
           "# and where it came from) next to its outputs. Keys: see momp/lib/config_schema.py.",
           "", region_line, ""]
    if definition:
        out += ["onset:", f"  definition: {definition}", ""]
    else:
        out += [f"# onset:  # the region's definition is used (default here: {default or 'none'});",
                "#   definition: <registry id or path>  # choose another, see `python -m momp.onset list`",
                "#   override: {season: {from: 05-15}}  # change part of a definition",
                ""]
    out += ["observations:"] + obs_lines + ["", "models:"] + model_blocks + [
        "", "evaluation:",
        f"  start_date: [{years[0]}, {start[0]}, {start[1]}]  # the years evaluated; month/day bound the init dates",
        f"  end_date: [{years[1]}, {end[0]}, {end[1]}]"]
    if probabilistic:
        out.append("  probabilistic: true  # ensemble forecasts detected")
    return "\n".join(out) + "\n"


def write_init(target, force=False, **kwargs) -> Path:
    """Render and write the starting config to target (or print it for '-'); refuses to overwrite unless force."""
    from momp.config import Config

    text = render(**kwargs)
    Config.from_mapping(yaml.safe_load(text))
    if str(target) == "-":
        print(text, end="")
        return Path("-")
    target = Path(target)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists. Pass --force to overwrite it")
    target.write_text(text)
    return target
