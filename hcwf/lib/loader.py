"""The run config's command-line path.

The config itself is resolved by hcwf.config.build (the one resolve path); this module only finds the config named on
the command line.
"""
import os
import argparse

from hcwf.lib.assertion import HCWFConfigError


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
