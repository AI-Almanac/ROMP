"""Load the YAML run config, apply command-line overrides and expose the config and Setting objects."""
import importlib.resources
import os
from pathlib import Path
import argparse
import sys
import warnings

from momp.utils.practical import set_dir
from momp.lib.control import init_dataclass
from momp.lib.convention import Setting
from .parser import create_parser
from types import SimpleNamespace
from momp.lib.assertion import ROMPValidator, ROMPConfigError
from momp.lib.parser import ensure_config_exists

package = "momp"

with importlib.resources.as_file(importlib.resources.files(package)) as p:
    base_dir = Path(p)

_cfg = None
_setting = None


def _get_config_path_pre_parse(cli_args=None):
    """Return the config path from -p/--param in cli_args, else the MOMP_CONFIG environment variable.

    A run must always say which config it uses; with neither set this raises,
    explaining how to give one (start one with `momp-run init`, or see examples/config.yaml).
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-p", "--param", default=None)
    args, _ = pre_parser.parse_known_args(cli_args)
    path = args.param or os.environ.get("MOMP_CONFIG")
    if not path:
        raise ROMPConfigError(
            "No config file given. Pass one with `momp-run -p path/to/run.yaml` (or set the "
            "MOMP_CONFIG environment variable). Start one with `momp-run init`; see `momp-run -h`.")
    return path


def _read_config(config_path):
    """Read a YAML run config into the flat config dict (validated, datasets described, defaults resolved)."""
    from momp.config import Config
    from momp.lib.config_yaml import is_yaml_path

    config_item = set_dir(config_path)
    if not config_item.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    if not is_yaml_path(config_item):
        raise ROMPConfigError(
            f"{config_path} is not a YAML config. momp-run reads YAML only. Convert an old config.in with "
            f"`momp-run convert {config_path} -o run.yaml`, or start a new one with `momp-run init`.")
    return Config.from_yaml(config_item).to_flat(source=str(config_item))


def _rename_deprecated_keys(dic):
    """Map the old India-specific ``mok`` key to ``earliest_onset_date``, which defaults to None (no filter)."""
    if "mok" in dic:
        if "earliest_onset_date" in dic:
            raise ROMPConfigError("Config sets both 'mok' and 'earliest_onset_date'; remove 'mok'.")
        warnings.warn("Config key 'mok' is deprecated; rename it to 'earliest_onset_date'.", FutureWarning)
        dic["earliest_onset_date"] = dic.pop("mok")
    dic.setdefault("earliest_onset_date", None)
    return dic


def _set_onset_mode(dic):
    """Flag configs that use the old flat onset keys; other configs get onset_definition = None if it is unset."""
    from momp.lib.onset_config import detect_legacy

    dic["onset_legacy"] = detect_legacy(dic)
    if not dic["onset_legacy"]:
        dic.setdefault("onset_definition", None)
    return dic


def _resolve_paths(dic):
    """Resolve relative paths in config dict to absolute paths."""
    dic["work_dir"] = Path(dic["work_dir"]).expanduser().resolve()
    dic["pkg_dir"] = Path(dic["pkg_dir"]).expanduser().resolve()

    for key in ["ref_model_dir", "dir_out", "dir_fig", "obs_dir"]:
        if key in dic and not Path(str(dic[key])).is_absolute():
            dic[key] = set_dir(dic[key])

    for key in ["thresh_file", "shpfile_dir", "nc_mask"]:
        if dic.get(key) is not None:
            if not Path(str(dic[key])).is_absolute():
                dic[key] = set_dir(dic[key])

    if isinstance(dic.get("thresholds"), dict):
        dic["thresholds"] = {
            name: _resolve_threshold_ref(ref) for name, ref in dic["thresholds"].items()
        }

    return dic


def _resolve_threshold_ref(ref):
    """Make the path part of a "path.nc:variable" threshold reference absolute."""
    path, sep, var = str(ref).rpartition(":")
    if not sep or not path or Path(path).is_absolute():
        return ref
    return f"{set_dir(path)}:{var}"


def build_cfg(cli_args=None):
    """Read the YAML run config, resolve paths, create output dirs and apply CLI overrides; return the config dict."""
    config_path = ensure_config_exists(_get_config_path_pre_parse(cli_args))
    dic = _read_config(config_path)
    dic = _rename_deprecated_keys(dic)
    dic = _set_onset_mode(dic)
    dic = _resolve_paths(dic)

    os.makedirs(dic["dir_fig"], exist_ok=True)
    os.makedirs(dic["dir_out"], exist_ok=True)

    args = create_parser(dic, cli_args=cli_args)
    args_dict = vars(args)
    overrides = {k: v for k, v in args_dict.items() if k in dic and v is not None}
    dic.update(overrides)

    return dic


def get_cfg(cli_args=None):
    """Return the validated config as a SimpleNamespace, built once and cached; exits on invalid config unless debug."""
    global _cfg
    if _cfg is None:
        _cfg = build_cfg(cli_args)

    try:
        validator = ROMPValidator(_cfg)
        validator.validate()
        if "lead_window" not in _cfg:
            from momp.lib.lead_window import apply_lead_window
            apply_lead_window(_cfg)
    except ROMPConfigError as e:
        print(f"Error: Invalid Config!!! {e}")
        if not _cfg.get('debug'):
            sys.exit(1)

    return SimpleNamespace(**_cfg)


def get_setting(cli_args=None):
    """Return the Setting dataclass built once from the config and cached."""
    global _setting
    if _setting is None:
        cfg = get_cfg(cli_args)
        _setting = init_dataclass(Setting, vars(cfg))
    return _setting
