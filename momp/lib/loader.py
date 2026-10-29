"""Load config.in, apply command-line overrides and expose the config and Setting objects."""
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
    """Return the -p/--param config path from cli_args (default "params/config.in")."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-p", "--param", default="params/config.in")
    args, _ = pre_parser.parse_known_args(cli_args)
    return args.param


def _read_config(config_path):
    """Read and exec config.in into an isolated namespace, return config dict."""
    config_item = set_dir(config_path)

    if not config_item.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    if isinstance(config_item, Path):
        with open(config_item, "r") as f:
            params_in = f.read()
    else:
        with importlib.resources.as_file(config_item) as actual_path:
            with open(actual_path, "r") as f:
                params_in = f.read()

    params_in = "\n".join(
        line for line in params_in.splitlines() if not line.strip().startswith("#")
    )

    # Execute config into an isolated namespace rather than module globals
    local_ns = {}
    exec(params_in, {}, local_ns)

    excluded = {"f", "config_file_path", "params_in"}
    dic = {
        k: v for k, v in local_ns.items()
        if not k.startswith("__")
        and not callable(v)
        and k not in excluded
    }
    return dic


def _rename_deprecated_keys(dic):
    """Map the old India-specific ``mok`` key to ``earliest_onset_date``, which defaults to None (no filter)."""
    if "mok" in dic:
        if "earliest_onset_date" in dic:
            raise ROMPConfigError("Config sets both 'mok' and 'earliest_onset_date'; remove 'mok'.")
        warnings.warn("Config key 'mok' is deprecated; rename it to 'earliest_onset_date'.", FutureWarning)
        dic["earliest_onset_date"] = dic.pop("mok")
    dic.setdefault("earliest_onset_date", None)
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

    return dic


def build_cfg(cli_args=None):
    """Read config.in, resolve paths, create output dirs and apply CLI overrides; return the config dict."""
    config_path = ensure_config_exists(_get_config_path_pre_parse(cli_args))
    dic = _read_config(config_path)
    dic = _rename_deprecated_keys(dic)
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
