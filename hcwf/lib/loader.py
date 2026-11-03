"""The run config's command-line path, and the process-wide config and Setting the flows read.

The config itself is resolved by hcwf.config.build (the one resolve path); this module only finds the config named on
the command line and caches the resolved config for the flows that still read it globally.
"""
import contextlib
import os
import argparse
import sys

from hcwf.lib.control import init_dataclass
from hcwf.lib.convention import Setting
from types import SimpleNamespace
from hcwf.lib.assertion import HCWFConfigError
import logging

log = logging.getLogger(__name__)

_cfg = None
_setting = None


def _get_config_path_pre_parse(cli_args=None):
    """Return the config path from -p/--param in cli_args, else the HCWF_CONFIG environment variable.

    A run must always say which config it uses; with neither set this raises,
    explaining how to give one (start one with `hcwf-run init`, or see examples/config.yaml).
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-p", "--param", default=None)
    args, _ = pre_parser.parse_known_args(cli_args)
    path = args.param or os.environ.get("HCWF_CONFIG")
    if not path:
        raise HCWFConfigError(
            "No config file given. Pass one with `hcwf-run -p path/to/run.yaml` (or set the "
            "HCWF_CONFIG environment variable). Start one with `hcwf-run init`; see `hcwf-run -h`.")
    return path


@contextlib.contextmanager
def use_config(dic):
    """Make dic the process-wide config (what get_cfg and get_setting return) for a with block, then restore."""
    global _cfg, _setting
    saved = (_cfg, _setting)
    _cfg = dic
    _setting = init_dataclass(Setting, dic)
    try:
        yield
    finally:
        _cfg, _setting = saved


def get_cfg(cli_args=None):
    """Return the validated config as a SimpleNamespace, built once and cached; exits on invalid config."""
    global _cfg
    if _cfg is None:
        from hcwf.config.build import resolve_cli

        try:
            _cfg = resolve_cli(cli_args)
        except HCWFConfigError as e:
            log.info(f"Error: Invalid Config!!! {e}")
            sys.exit(1)

    return SimpleNamespace(**_cfg)


def get_setting(cli_args=None):
    """Return the Setting dataclass built once from the config and cached."""
    global _setting
    if _setting is None:
        cfg = get_cfg(cli_args)
        _setting = init_dataclass(Setting, vars(cfg))
    return _setting
