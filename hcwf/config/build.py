"""The one way to turn a run config into the validated flat dict a run uses (design: docs/review/runconfig_design.md 2.4).

    flat = resolve_flat("run.yaml")                                  # a path, hcwf.config.Config, mapping or flat dict
    flat = resolve_flat(cfg, {"max_forecast_day": 12})               # with overrides (flat keys, validated)
    flat = resolve_cli(sys.argv[1:])                                 # the command line: -p path and override flags

Every entry (hcwf.run / hcwf.check, hcwf-run and --check, hcwf.export, the run-wide config of the apps) goes through
``resolve_flat``: read the source, apply the overrides (normalised against the schema and recorded in
``config_sources``), resolve the paths, plan the lead window and the run folder, then validate. Nothing is created
on disk: ``flat["run_paths"]`` says where the run will write.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from hcwf.lib import config_schema as schema
from hcwf.utils.practical import set_dir

_NO_FILES = {"save_fig": False, "save_nc_spatial_far_mr_mae": False, "save_csv_score": False,
             "save_nc_climatology": False, "save_onsets": False, "plot_spatial_far_mr_mae": False,
             "plot_heatmap_bss_auc": False, "plot_reliability": False, "plot_climatology_onset": False,
             "plot_panel_heatmap_error": False, "plot_panel_heatmap_skill": False, "plot_bar_bss_rpss_auc": False,
             "plot_portrait": False, "show_plot": False, "show_panel": False, "explain_cells": None}
"""Config values that turn every file-writing step off (write=False)."""

_MODEL_KEYS = tuple(f.key for f in schema.FIELDS if f.internal and f.key.endswith("_list")) + ("accumulation_detected",)
"""The flat keys that hold one entry per model (selecting models keeps the same entries of each)."""


def resolve_flat(source, overrides: Mapping[str, Any] | None = None, *, write: bool = True, tmp: Path | None = None,
                 override_source: str = "override", check: bool = False):
    """The validated flat config of a run, finished (paths, lead window, run folder plan); raises HCWFConfigError.

    source is the path of a run YAML, a hcwf.config.Config, a nested mapping (as the YAML), a SimpleNamespace or a
    flat dict (finishing a flat dict that is already finished changes nothing). overrides are flat config keys
    (`model_list` selects some of the configured models); each is normalised like a YAML value and recorded in
    ``config_sources`` as override_source. write=False (a run that writes no files) switches every writing step off
    and, with tmp, puts the outputs in that folder and keeps the observed-onset cache in memory.

    Validation is hcwf.config.validate.problems: any problem (the units' and the blend's too) raises one HCWFConfigError
    before a run folder exists. check=True returns (values, problems) instead of raising, for `--check`: values is None
    when a fatal problem means the config cannot be resolved, and a units or blend problem is left for the report.
    """
    from hcwf.config import validate

    values, config_file = _read(source)
    _apply_overrides(values, overrides or {}, override_source)
    values["config_file"] = config_file
    finish(values, config_file, write)
    found = validate.problems(values)
    if check:
        return (None if found.fatal else values), found
    found.raise_if_any()
    if not write:
        for key, value in _NO_FILES.items():
            if key in values or key == "explain_cells":
                values[key] = value
        if tmp is not None:
            values["dir_out"] = values["dir_fig"] = str(tmp)
            values["cache_dir"] = None
    return values


def resolve_cli(cli_args=None, **kwargs) -> dict:
    """The validated flat config named by -p/--param (or HCWF_CONFIG) of the command line, with its override flags
    applied and recorded as 'command line' (cli_args is sys.argv[1:] when None)."""
    from hcwf.lib.loader import _get_config_path_pre_parse
    from hcwf.lib.parser import parse_overrides

    return resolve_flat(_get_config_path_pre_parse(cli_args), parse_overrides(cli_args),
                        override_source="command line", **kwargs)


def _read(source) -> tuple[dict, str | None]:
    """(a new flat dict, the config file's path or None) of a path, Config, mapping, namespace or flat dict."""
    from hcwf.config import Config
    from hcwf.lib.assertion import HCWFConfigError
    from hcwf.lib.config_yaml import build_config, is_yaml_path
    from hcwf.lib.parser import ensure_config_exists

    if isinstance(source, (str, Path)):
        path = ensure_config_exists(source)
        if not is_yaml_path(path):
            raise HCWFConfigError(
                f"{source} is not a YAML config. hcwf-run reads YAML only. Start a new one with `hcwf-run init`.")
        return Config.from_yaml(path).to_flat(source=str(path)), str(path)
    if isinstance(source, Config):
        return source.to_flat(), None
    if isinstance(source, SimpleNamespace):
        source = vars(source)
    if isinstance(source, dict) and ("model_list" in source or "obs_dir" in source):   # already flat
        return dict(source), source.get("config_file")
    if isinstance(source, dict):
        return build_config(source, source="<mapping>"), None
    raise TypeError(f"a config is a YAML path, a hcwf.config.Config or a mapping, got {type(source).__name__}")


def _apply_overrides(values: dict, overrides: Mapping[str, Any], origin: str) -> None:
    """Put the overrides into values (normalised, recorded in config_sources as origin); `model_list` selects models."""
    errors: list[str] = []
    sources = dict(values.get("config_sources") or {})
    for key, value in overrides.items():
        if key == "model_list":
            _select_models(values, value, errors)
        else:
            field = schema.BY_KEY.get(key)
            if field is None or field.internal:
                raise TypeError(f"unknown override '{key}'.{schema.suggest(key, [k for k in schema.BY_KEY])} "
                                "Overrides are flat config keys, e.g. max_forecast_day, verification_window_list, "
                                "start_date")
            values[key] = schema.normalise_value(field, value, key, errors)
        sources[key] = origin
    schema.raise_errors(errors, "<overrides>")
    if overrides:
        values["config_sources"] = sources


def _select_models(values: dict, names, errors: list[str]) -> None:
    """Keep only the models `names` (in that order) of the configured models, with each of their per-model entries."""
    names = (names,) if isinstance(names, str) else tuple(names)
    have = list(values["model_list"])
    unknown = [n for n in names if n not in have]
    if unknown or not names or len(set(names)) != len(names):
        problem = (f"unknown model{'s' if len(unknown) > 1 else ''} {', '.join(map(repr, unknown))}" if unknown else
                   "no models given" if not names else "a model is named twice")
        errors.append(f"'model_list': {problem}. The config's models are: {', '.join(have)}"
                      f"{schema.suggest(unknown[0], have) if unknown else ''}")
        return
    index = [have.index(n) for n in names]
    for key in _MODEL_KEYS:
        if isinstance(values.get(key), (tuple, list)) and len(values[key]) == len(have):
            values[key] = tuple(values[key][i] for i in index)


def finish(values: dict, config_file: str | None, write: bool = True) -> dict:
    """Finish a flat config in place: resolve the paths, then the lead window and the run folder plan."""
    from hcwf.lib import run_folder

    _resolve_paths(values)
    if "lead_window" not in values:
        if schema.has_blend(values):   # a blend reads each model's leads itself (lead_days), not HCWF's benchmark window
            values["lead_window"], values["onset_lead_lines"] = (1, values["max_forecast_day"]), ()
        else:
            from hcwf.lib.lead_window import apply_lead_window

            apply_lead_window(values)
    return run_folder.apply(values, run_folder.plan(values, config_file, write))


def _resolve_paths(dic):
    """Resolve relative paths in config dict to absolute paths."""
    for key in ["ref_model_dir", "dir_out", "dir_fig", "obs_dir"]:
        if key in dic and not Path(str(dic[key])).is_absolute():
            dic[key] = set_dir(dic[key])

    for key in ["shpfile_dir", "nc_mask"]:
        if dic.get(key) is not None:
            if not Path(str(dic[key])).is_absolute():
                dic[key] = set_dir(dic[key])

    _anchor_units_paths(dic)
    _anchor_mme_weights(dic)

    if isinstance(dic.get("mask"), dict):
        mask = dict(dic["mask"])
        if mask.get("file") and not Path(str(mask["file"])).is_absolute():
            mask["file"] = str(set_dir(mask["file"]))
        condition = mask.get("condition")
        if condition and not Path(str(condition["file"])).is_absolute():
            mask["condition"] = {**condition, "file": str(set_dir(condition["file"]))}
        dic["mask"] = mask

    if isinstance(dic.get("thresholds"), dict):
        dic["thresholds"] = {
            name: _resolve_threshold_ref(ref) for name, ref in dic["thresholds"].items()
        }

    return dic


_UNITS_PATH_KEYS = ("units_shapefile", "units_weights", "units_centroids", "units_dissemination", "units_outline")
"""The `units:` keys that name files (units_weights may also map a source name to its file)."""


def _anchor_units_paths(dic):
    """Make the relative file paths of the `units:` section start at the folder of the config file (the working directory
    when the config is not from a file), as a region file's units start at the region file."""
    config_file = dic.get("config_file")
    if not config_file:
        return
    base = Path(str(config_file)).expanduser().resolve().parent

    def anchor(path):
        path = Path(str(path)).expanduser()
        return str(path if path.is_absolute() else (base / path).resolve())

    for key in _UNITS_PATH_KEYS:
        value = dic.get(key)
        if isinstance(value, dict):
            dic[key] = {source: anchor(path) for source, path in value.items()}
        elif value is not None:
            dic[key] = anchor(value)


def _anchor_mme_weights(dic):
    """Make the relative path of `blend.mme.weights` (a weights file; a mapping is inline weights) start at the folder
    of the config file, the working directory when the config is not from a file."""
    mme = dic.get("blend_mme")
    if not isinstance(mme, dict) or not isinstance(mme.get("weights"), (str, Path)):
        return
    base = Path(str(dic["config_file"])).expanduser().resolve().parent if dic.get("config_file") else Path.cwd()
    path = Path(str(mme["weights"])).expanduser()
    dic["blend_mme"] = {**mme, "weights": str(path if path.is_absolute() else (base / path).resolve())}


def _resolve_threshold_ref(ref):
    """Make the path part of a "path.nc:variable" threshold reference absolute."""
    path, sep, var = str(ref).rpartition(":")
    if not sep or not path or Path(path).is_absolute():
        return ref
    return f"{set_dir(path)}:{var}"
