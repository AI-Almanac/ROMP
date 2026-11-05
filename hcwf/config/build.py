"""The one way to turn a run config into the validated flat dict a run uses (design: ``docs/review/runconfig_design.md`` 2.4).

This module is the entry of the config layer: it reads a source with `hcwf.config.mapping` and `hcwf.config.Config`,
validates it with `hcwf.config.validate` and returns the `hcwf.config.model.RunConfig` that the pipeline reads::

    rc = resolve("run.yaml")                                         # a path, hcwf.config.Config, mapping or flat dict
    rc = resolve(cfg, {"max_forecast_day": 12})                      # with overrides (flat keys, validated)
    rc = resolve_command_line(sys.argv[1:])                          # the command line: -p path and override flags

`resolve` returns the `~hcwf.config.model.RunConfig`. Every entry (``hcwf.run`` / ``hcwf.check``, ``hcwf-run`` and ``--check``,
``hcwf.runs.export``, the run-wide config of the apps) goes through the same steps: read the source, apply the overrides
(normalised against the schema and recorded in ``config_sources``), resolve the paths, plan the lead window and the run
folder, then validate. Nothing is created on disk: ``flat["run_paths"]`` says where the run will write.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from hcwf.config import schema
from hcwf.io.paths import set_dir

__all__ = ["resolve", "resolve_command_line"]


_NO_FILES = {"save_fig": False, "save_nc_spatial_far_mr_mae": False, "save_csv_score": False,
             "save_nc_climatology": False, "save_onsets": False, "plot_spatial_far_mr_mae": False,
             "plot_heatmap_bss_auc": False, "plot_reliability": False, "plot_climatology_onset": False,
             "plot_panel_heatmap_error": False, "plot_panel_heatmap_skill": False, "plot_bar_bss_rpss_auc": False,
             "plot_portrait": False, "show_plot": False, "show_panel": False, "explain_cells": None}
"""Config values that turn every file-writing step off (write=False)."""

_MODEL_KEYS = tuple(f.key for f in schema.FIELDS if f.internal and f.key.endswith("_list")) + ("accumulation_detected",)
"""The flat keys that hold one entry per model (selecting models keeps the same entries of each)."""


def resolve(source, overrides: Mapping[str, Any] | None = None, *, write: bool = True, tmp: Path | None = None,
            override_source: str = "override", check: bool = False):
    """Return the validated `~hcwf.config.model.RunConfig` of a run, finished (paths, lead window, run folder plan).

    A config with a problem raises `~hcwf.errors.HCWFConfigError`, unless ``check`` is true.

    Parameters
    ----------
    source : hcwf.config.model.RunConfig or str or pathlib.Path or hcwf.config.Config or dict or types.SimpleNamespace
        A `~hcwf.config.model.RunConfig` (returned as it is; overrides then raise `TypeError`), the path of a run YAML, a
        `hcwf.config.Config`, a nested mapping, a namespace or a flat dict.
    overrides : collections.abc.Mapping, optional
        Flat config keys to override, see `_resolve_values`.
    write : bool, default True
        Whether the run writes files; false switches every writing step off, see `_resolve_values`.
    tmp : pathlib.Path, optional
        With ``write=False``, the folder that takes the outputs and the in-memory observed-onset cache.
    override_source : str, default "override"
        What the overrides are recorded as in ``config_sources``.
    check : bool, default False
        Return ``(RunConfig or None, problems)`` instead of raising, for ``--check``.

    Returns
    -------
    hcwf.config.model.RunConfig or tuple
        The config; with ``check``, the pair of the config (None when a fatal problem means it cannot be resolved)
        and the `hcwf.config.validate.Problems` found.

    Raises
    ------
    TypeError
        ``source`` is a `~hcwf.config.model.RunConfig` and ``overrides`` are given.
    """
    from hcwf.config.model import RunConfig

    if isinstance(source, RunConfig):
        if overrides:
            raise TypeError("a RunConfig is already resolved: overrides cannot be applied to it")
        if check:
            from hcwf.config import validate

            return source, validate.problems(source.values)
        return source
    from hcwf.io.grids import attach

    out = _resolve_values(source, overrides, write=write, tmp=tmp, override_source=override_source, check=check)
    if not check:
        return attach(RunConfig.from_flat(out))
    values, found = out
    return (None if values is None else attach(RunConfig.from_flat(values, lenient=True))), found


def resolve_command_line(cli_args=None, **kwargs):
    """Return the validated `~hcwf.config.model.RunConfig` named by ``-p/--param`` (or ``HCWF_CONFIG``) of the command line.

    Its override flags are applied and recorded as ``"command line"``.

    Parameters
    ----------
    cli_args : list of str, optional
        The command line without the program name; ``sys.argv[1:]`` when None.
    **kwargs : dict
        Passed on to `resolve` (``write``, ``tmp``, ``check``).

    Returns
    -------
    hcwf.config.model.RunConfig or tuple
        As `resolve`.
    """
    from hcwf.cli.parser import _get_config_path_pre_parse
    from hcwf.cli.parser import parse_overrides

    return resolve(_get_config_path_pre_parse(cli_args), parse_overrides(cli_args),
                   override_source="command line", **kwargs)


def _resolve_values(source, overrides, *, write, tmp, override_source, check):
    """Run the steps of `resolve`, to the validated flat dict.

    ``source`` is the path of a run YAML, a `hcwf.config.Config`, a nested mapping (as the YAML), a ``SimpleNamespace`` or a
    flat dict (finishing a flat dict that is already finished changes nothing). ``overrides`` are flat config keys
    (``model_list`` selects some of the configured models); each is normalised like a YAML value and recorded in
    ``config_sources`` as ``override_source``. ``write=False`` (a run that writes no files) switches every writing step
    off and, with ``tmp``, puts the outputs in that folder and keeps the observed-onset cache in memory.

    Validation is `hcwf.config.validate.problems`: any problem (the units' and the blend's too) raises one
    `~hcwf.errors.HCWFConfigError` before a run folder exists.

    Parameters
    ----------
    source : str or pathlib.Path or hcwf.config.Config or dict or types.SimpleNamespace
        What to read.
    overrides : collections.abc.Mapping or None
        Flat config keys to override.
    write : bool
        Whether the run writes files.
    tmp : pathlib.Path or None
        With ``write=False``, the folder that takes the outputs.
    override_source : str
        What the overrides are recorded as in ``config_sources``.
    check : bool
        Return the values and the problems instead of raising, for ``--check``.

    Returns
    -------
    dict or tuple
        The validated flat dict; with ``check``, ``(values, problems)`` where ``values`` is None when a fatal problem
        means the config cannot be resolved (a units or blend problem is left for the report).
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


def _read(source) -> tuple[dict, str | None]:
    """Read a config source into a new flat dict.

    Parameters
    ----------
    source : str or pathlib.Path or hcwf.config.Config or dict or types.SimpleNamespace
        A YAML path, a `hcwf.config.Config`, a nested mapping, a namespace or a flat dict (one with ``model_list``
        or ``obs_dir``).

    Returns
    -------
    tuple of (dict, str or None)
        The new flat dict and the config file's path (None when the source is not a file).

    Raises
    ------
    HCWFConfigError
        ``source`` is a path that is not a YAML file.
    TypeError
        ``source`` is none of the accepted kinds.
    """
    from hcwf.config import Config
    from hcwf.errors import HCWFConfigError
    from hcwf.config.mapping import build_config, is_yaml_path
    from hcwf.cli.parser import ensure_config_exists

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
    """Put the overrides into ``values`` (normalised, recorded in ``config_sources`` as ``origin``); ``model_list`` selects models.

    Parameters
    ----------
    values : dict
        The flat config, changed in place.
    overrides : collections.abc.Mapping
        Flat config keys to their new values.
    origin : str
        What each override is recorded as in ``config_sources``.

    Raises
    ------
    TypeError
        A key is not a flat config key, or is an internal one.
    """
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
    """Keep only the models ``names`` (in that order) of the configured models, with each of their per-model entries.

    Parameters
    ----------
    values : dict
        The flat config, changed in place (the ``*_list`` keys that hold one entry per model).
    names : str or iterable of str
        The models to keep, in the order wanted.
    errors : list of str
        Takes the message when a name is unknown, none is given or one is named twice; ``values`` is then left alone.
    """
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
    """Finish a flat config in place: resolve the paths, then the lead window and the run folder plan.

    Parameters
    ----------
    values : dict
        The flat config, changed in place.
    config_file : str or None
        The path of the run file, or None.
    write : bool, default True
        Whether the run writes files (planned in the run folder).

    Returns
    -------
    dict
        ``values``, with ``lead_window``, ``onset_lead_lines`` and the run folder plan added.
    """
    from hcwf.runs import folder as run_folder

    _resolve_paths(values)
    _default_blend_units(values)
    if "lead_window" not in values:
        if schema.has_blend(values):   # a blend reads each model's leads itself (lead_days), not HCWF's benchmark window
            values["lead_window"], values["onset_lead_lines"] = (1, values["max_forecast_day"]), ()
        else:
            from hcwf.config.leads import apply_lead_window

            apply_lead_window(values)
    return run_folder.apply(values, run_folder.plan(values, config_file, write))


def _default_blend_units(values: dict) -> None:
    """Default the units of a blend run that has no ``units:`` section.

    Such a run evaluates on the grid cells of every source, as ``hcwf-run convert`` writes it for a gridded blend config
    (mode ``cells``, ``trimmed`` grid ids, dissemination scope ``inputs``). A blend reads its onsets per unit, so it
    cannot go without units. Any ``units:`` key given is the user's own and nothing is added.

    Parameters
    ----------
    values : dict
        The flat config, changed in place.
    """
    if not schema.has_blend(values) or any(values.get(f.key) is not None for f in schema.FIELDS if f.section == "units"):
        return
    values.update(units_mode={"*": "cells"}, units_id_format="trimmed", units_dissemination_scope="inputs")
    sources = dict(values.get("config_sources") or {})
    sources.update({key: "default (blend: grid cells)" for key in ("units_mode", "units_id_format",
                                                                   "units_dissemination_scope")})
    values["config_sources"] = sources


def _resolve_paths(dic):
    """Resolve the relative paths of a flat config to absolute paths.

    Parameters
    ----------
    dic : dict
        The flat config, changed in place: the folders, the shapefile and NetCDF mask, the ``mask`` files, the
        ``units`` files, the blend's weights file and the named thresholds.

    Returns
    -------
    dict
        ``dic``.
    """
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
"""The ``units:`` keys that name files (``units_weights`` may also map a source name to its file)."""


def _anchor_units_paths(dic):
    """Make the relative file paths of the ``units:`` section start at the folder of the config file.

    A region file's units start at the region file. Nothing changes when the config is not from a file.

    Parameters
    ----------
    dic : dict
        The flat config, changed in place.

    Returns
    -------
    None
        Nothing; ``dic`` is changed in place.
    """
    config_file = dic.get("config_file")
    if not config_file:
        return
    base = Path(str(config_file)).expanduser().resolve().parent

    def anchor(path):
        """Return `path` as an absolute string, relative paths starting at the config file's folder."""
        path = Path(str(path)).expanduser()
        return str(path if path.is_absolute() else (base / path).resolve())

    for key in _UNITS_PATH_KEYS:
        value = dic.get(key)
        if isinstance(value, dict):
            dic[key] = {source: anchor(path) for source, path in value.items()}
        elif value is not None:
            dic[key] = anchor(value)


def _anchor_mme_weights(dic):
    """Make the relative path of ``blend.mme.weights`` start at the folder of the config file.

    A weights file is anchored (a mapping is inline weights and stays); the working directory is the base when the
    config is not from a file.

    Parameters
    ----------
    dic : dict
        The flat config, changed in place.

    Returns
    -------
    None
        Nothing; ``dic`` is changed in place.
    """
    mme = dic.get("blend_mme")
    if not isinstance(mme, dict) or not isinstance(mme.get("weights"), (str, Path)):
        return
    base = Path(str(dic["config_file"])).expanduser().resolve().parent if dic.get("config_file") else Path.cwd()
    path = Path(str(mme["weights"])).expanduser()
    dic["blend_mme"] = {**mme, "weights": str(path if path.is_absolute() else (base / path).resolve())}


def _resolve_threshold_ref(ref):
    """Make the path part of a ``"path.nc:variable"`` threshold reference absolute.

    Parameters
    ----------
    ref : str
        The reference, ``path:variable`` or a bare path.

    Returns
    -------
    str
        The reference with its path resolved against the data folder, or ``ref`` unchanged when it has no ``:`` part
        or is already absolute.
    """
    path, sep, var = str(ref).rpartition(":")
    if not sep or not path or Path(path).is_absolute():
        return ref
    return f"{set_dir(path)}:{var}"
