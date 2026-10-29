"""Write the fully resolved config of a run next to its outputs, so the run can be reproduced from one file.

``resolved_config.yaml`` holds every value after defaults, region defaults, auto-detection and overrides, each
with a comment saying where it came from; running ``momp-run -p resolved_config.yaml`` reproduces the run.
``resolved_definition.yaml`` holds the onset definition as used (after any override), with its hash.
"""
from __future__ import annotations

from pathlib import Path

from momp.lib.convert import flat_to_mapping, render_yaml
from momp.lib.onset_config import onset_setup
from momp.onset import definition as definition_mod
from momp.onset.describe import describe


def definition_lines(cfg: dict) -> list[str]:
    """Printable lines about the onset definition in use: its plain-words sentence, overrides and hash."""
    setup = onset_setup(cfg)
    d = setup.definition
    changed = f", overridden: {', '.join(setup.overrides)}" if setup.overrides else ""
    return [f"Definition      : {d.id} v{d.version}{changed}",
            f"Definition text : {describe(d)}",
            f"Definition hash : {definition_mod.definition_hash(d)}"]


def resolved_mapping(cfg: dict) -> tuple[dict, dict, object]:
    """The resolved flat values (onset settings filled from the region), their sources, and the OnsetSetup."""
    flat = {k: v for k, v in cfg.items()}
    sources = dict(cfg.get("config_sources") or {})
    setup = onset_setup(cfg)
    if not setup.legacy:
        resolved = {"onset_season": setup.season, "obs_day_boundary": setup.obs_day_boundary,
                    "model_lead_label": setup.lead_label}
        if setup.threshold_years:
            resolved["threshold_years"] = list(setup.threshold_years)
        if cfg.get("onset_rule") is None:
            resolved["onset_definition"] = cfg.get("onset_definition") or setup.definition.id
        for key, source_key in (("onset_definition", "definition"), ("onset_season", "season"),
                                ("threshold_years", "threshold_years"), ("obs_day_boundary", "obs_day_boundary"),
                                ("model_lead_label", "lead_label")):
            if key in resolved:
                flat[key] = resolved[key]
                sources[key] = setup.sources.get(source_key, "resolved")
        flat["forecast_history"] = setup.forecast_history
        flat["threshold_source"] = setup.threshold_source
        sources["forecast_history"] = setup.sources.get("forecast_history", "default")
        sources["threshold_source"] = setup.sources.get("threshold_source", "default")
    return flat, sources, setup


def write_resolved(cfg: dict, directory=None) -> tuple[Path, Path]:
    """Write resolved_config.yaml and resolved_definition.yaml into directory (default: dir_out)."""
    directory = Path(directory or cfg["dir_out"])
    directory.mkdir(parents=True, exist_ok=True)
    flat, sources, setup = resolved_mapping(cfg)
    mapping, comments = flat_to_mapping(flat, sources)
    header = ("# Resolved config of this run: every value after defaults, region defaults, auto-detection and\n"
              "# overrides, each with its source. Reproduce the run with:  momp-run -p resolved_config.yaml\n"
              f"# definition: {setup.definition.id} v{setup.definition.version}, hash "
              f"{definition_mod.definition_hash(setup.definition)}\n")
    config_path = directory / "resolved_config.yaml"
    config_path.write_text(render_yaml(mapping, comments, header=header))
    definition_path = directory / "resolved_definition.yaml"
    definition_path.write_text(
        f"# Onset definition as used by this run"
        f"{', overridden: ' + ', '.join(setup.overrides) if setup.overrides else ''}\n"
        f"# hash: {definition_mod.definition_hash(setup.definition)}\n{setup.definition.source_yaml}")
    return config_path, definition_path
