"""Write the fully resolved config of a run next to its outputs, so the run can be reproduced from one file.

`resolved_config_text` builds the text, `write_resolved` writes it; `hcwf.runs.folder` decides where the files go.

``resolved_config.yaml`` holds every value after defaults, region defaults, auto-detection and overrides, each
with a comment saying where it came from; running ``hcwf-run -p resolved_config.yaml`` reproduces the run.
When a region mask is in use, comments at the top record per dataset the mask sources with content hashes, the
cell rule, the alignment actions and the kept cell counts (known once each dataset's grid has been read).
``resolved_definition.yaml`` holds the onset definition as used (after any override), with its hash.
In a run folder (no explicit ``data_dir``) the files are ``run.yaml`` and ``definition.yaml`` in the run
folder, and the derived output folders are left out so the file can be run again anywhere.
"""
from __future__ import annotations

from pathlib import Path

from hcwf.config.convert import flat_to_mapping, render_yaml
from hcwf.config.onset import onset_setup
from hcwf.onset import definition as definition_mod
from hcwf.onset.describe import describe


def definition_lines(setup) -> list[str]:
    """Return printable lines about the onset definition in use: its identity, plain-words sentence and hash.

    Parameters
    ----------
    setup : hcwf.config.onset.OnsetSetup
        The onset setup of the run; its overrides are named when there are any.

    Returns
    -------
    list of str
        Three lines: ``Definition``, ``Definition text`` and ``Definition hash``.
    """
    d = setup.definition
    changed = f", overridden: {', '.join(setup.overrides)}" if setup.overrides else ""
    return [f"Definition      : {d.id} v{d.version}{changed}",
            f"Definition text : {describe(d)}",
            f"Definition hash : {definition_mod.definition_hash(d)}"]


def resolved_mapping(cfg: dict) -> tuple[dict, dict, object]:
    """Return the resolved flat values (onset settings filled from the region), their sources, and the onset setup.

    In a run folder the derived output folders are left out of the values so that the file can be run again anywhere.

    Parameters
    ----------
    cfg : dict
        The flat config, with ``config_sources`` and ``run_paths`` when known.

    Returns
    -------
    flat : dict
        The flat values with the resolved onset keys.
    sources : dict of str to str
        Where each value came from.
    setup : hcwf.config.onset.OnsetSetup
        The onset setup of `cfg`.
    """
    flat = {k: v for k, v in cfg.items()}
    sources = dict(cfg.get("config_sources") or {})
    paths = cfg.get("run_paths")
    if paths is not None and not paths.explicit:
        flat.pop("dir_out", None)
        if sources.get("dir_fig") != "run config":
            flat.pop("dir_fig", None)
    setup = onset_setup(cfg)
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


def resolved_config_text(cfg: dict) -> str:
    """Return the text of ``resolved_config.yaml`` for a flat config dict: a header, the values and their sources.

    The header also records, per dataset, the mask sources the run has loaded so far.

    Parameters
    ----------
    cfg : dict
        The flat config.

    Returns
    -------
    str
        The YAML text.
    """
    flat, sources, setup = resolved_mapping(cfg)
    mapping, comments = flat_to_mapping(flat, sources)
    paths = cfg.get("run_paths")
    name = paths.config_name if paths is not None else "resolved_config.yaml"
    header = ("# Resolved config of this run: every value after defaults, region defaults, auto-detection and\n"
              f"# overrides, each with its source. Reproduce the run with:  hcwf-run -p {name}\n"
              f"# definition: {setup.definition.id} v{setup.definition.version}, hash "
              f"{definition_mod.definition_hash(setup.definition)}\n")
    from hcwf.spatial.masks import record_lines, records

    for record in records().values():
        header += "".join(f"# {line}\n" for line in record_lines(record))
    return render_yaml(mapping, comments, header=header)


def resolved_config_text_of(rc) -> str:
    """Return `resolved_config_text` of a run config.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.

    Returns
    -------
    str
        The YAML text.
    """
    return resolved_config_text(dict(rc.values))


def write_resolved(rc, directory=None) -> tuple[Path, Path]:
    """Write the resolved config and definition of a run config into a folder.

    The files are ``resolved_config.yaml`` and ``resolved_definition.yaml``, or ``run.yaml`` and ``definition.yaml`` in a
    run folder.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    directory : str or pathlib.Path, optional
        The folder (made if missing); the run folder when None, else ``dir_out``.

    Returns
    -------
    config_path : pathlib.Path
        The resolved config file.
    definition_path : pathlib.Path
        The resolved definition file.
    """
    cfg = dict(rc.values)
    paths = cfg.get("run_paths")
    directory = Path(directory or (paths.root if paths is not None else cfg["dir_out"]))
    directory.mkdir(parents=True, exist_ok=True)
    setup = onset_setup(cfg)
    config_path = directory / (paths.config_name if paths is not None else "resolved_config.yaml")
    config_path.write_text(resolved_config_text(cfg))
    definition_path = directory / (paths.definition_name if paths is not None else "resolved_definition.yaml")
    definition_path.write_text(
        f"# Onset definition as used by this run"
        f"{', overridden: ' + ', '.join(setup.overrides) if setup.overrides else ''}\n"
        f"# hash: {definition_mod.definition_hash(setup.definition)}\n{setup.definition.source_yaml}")
    return config_path, definition_path
