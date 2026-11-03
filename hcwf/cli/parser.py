"""Command-line parser that overrides run-config values, plus argparse type helpers and the config path of the command line.

The config itself is resolved by hcwf.config.build (the one resolve path); this module only finds the config named on
the command line.
"""
import argparse
import ast
import os
from pathlib import Path

from hcwf.errors import HCWFConfigError


OVERRIDE_FLAGS = ("model_list", "verification_window_list", "max_forecast_day", "dry_spell_check", "forecast_history",
                  "threshold_source", "onset_percentage_threshold", "tolerance_days_list", "day_bins", "start_date",
                  "end_date", "show_plot", "show_panel", "probabilistic", "parallel", "region")
"""The command-line flags that override a config value (the flat key names; `model_list` selects models of `models:`)."""


def create_parser():
    """The command-line parser: every override flag defaults to None (not given), the run-control flags are flags."""
    parser = argparse.ArgumentParser(description="HCWF Package Parameter Loader")
    add = parser.add_argument
    add("--model_list", nargs="+", default=None, help="evaluate only these models of the config's models: section")
    add("--verification_window_list", type=parse_window_list, default=None,
        help="verification windows as start,end lead days (e.g., '1,15 16,20')")
    add("--max_forecast_day", type=int, default=None, help="max forecast day")
    add("--dry_spell_check", type=str2bool, default=None,
        help="keep the definition's dry-spell condition, True or False")
    for name, choices in (("forecast_history", ("observed", "none")), ("threshold_source", ("observations", "model"))):
        add(f"--{name}", choices=choices, default=None, help=f"{name}: {' or '.join(choices)}")
    add("--onset_percentage_threshold", type=float, default=None,
        help="probability threshold (0.0-1.0) for ensemble onset")
    add("--tolerance_days_list", nargs="+", type=int, action=ForceTupleAction, default=None,
        help="allowed error margins in days e.g., 3 5")
    add("--day_bins", type=parse_window_list, default=None, help="day bins for statistics e.g., '1,15 16,30'")
    add("--start_date", nargs=3, type=int, action=ForceTupleAction, default=None,
        help="evaluation start date as YYYY M D (e.g., 2024 5 1)")
    add("--end_date", nargs=3, type=int, action=ForceTupleAction, default=None,
        help="evaluation end date as YYYY M D (e.g., 2024 10 31)")
    add("--show_plot", type=str2bool, default=None, help="toggle individual model plot display, True or False")
    add("--show_panel", type=str2bool, default=None, help="toggle panel plot display, True or False")
    add("--probabilistic", type=str2bool, default=None, help="ensemble evaluation toggle, True or False")
    add("--parallel", type=str2bool, default=None, help="parallelization, True or False")
    add("--region", type=str, default=None, help="a region id or alias (hcwf/onset/regions/*.yaml)")

    # run control (handled by the driver, not config values)
    add("--check", action="store_true",
        help="validate and resolve the config, print the plan (settings, datasets, files, leads, "
             "hints, run folder, cache) and run nothing; exit 1 on errors")
    add("--alignment", action="store_true",
        help="with --check: also run the observation/forecast alignment check (off by default)")
    add("--overwrite", action="store_true",
        help="re-run a config whose run folder already exists (clears that folder first)")
    add("--quiet", action="store_true", help="show only warnings and errors")
    add("--verbose", action="store_true", help="show debug detail (run.log always has it)")
    return parser


def parse_overrides(cli_args=None) -> dict:
    """The config overrides given on the command line, {flat key: value}; unknown arguments are ignored.

    Only the flags actually given are returned; build.resolve validates and records them as 'command line'.
    cli_args is the argument list (sys.argv[1:] when None).
    """
    args, _ = create_parser().parse_known_args(cli_args)
    out = {key: getattr(args, key) for key in OVERRIDE_FLAGS if getattr(args, key) is not None}
    if "model_list" in out:
        out["model_list"] = tuple(out["model_list"])
    return out


class ForceTupleAction(argparse.Action):
    """Custom action to convert nargs list into a tuple automatically."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Store values on the namespace as a tuple."""
        setattr(namespace, self.dest, tuple(values))


def parse_window_list(string):
    """
    Converts a string input like '1,15 16,20' into ((1, 15), (16, 20))
    """
    try:
        # Split by space to get individual pairs, then split by comma
        pairs = [tuple(map(int, p.split(','))) for p in string.split()]
        return tuple(pairs)
    except Exception:
        raise argparse.ArgumentTypeError("Window list must be in format 'start,end start,end' (e.g., '1,15 16,20')")


def str2bool(v):
    """Convert a yes/no/true/false-style string (or a bool) to bool for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Takes a string like "(3, 5)" and turns it into a Python tuple (3, 5).
def parse_tuple(value):
    """
    Parses a string into its native Python type.
    Example: "(2019, 5, 1)" -> tuple
    Example: "((1, 5), (6, 10))" -> nested tuple
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Fallback for simple strings if eval fails
        return value


def parse_num_to_tuple(value):
    """
    Converts a space-separated string into a Python tuple of integers.
    CLI: "2024 5 1" -> Python: (2024, 5, 1)
    """
    try:
        # Split string by spaces, convert each part to int, then to tuple
        return tuple(map(int, value.split()))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{value}'. Expected space-separated integers like '2024 5 1'."
        )


def ensure_config_exists(param_path: str) -> Path:
    """Return param_path expanded and resolved, raising FileNotFoundError (with a hint) if it doesn't exist."""
    path = Path(param_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. Start one with `hcwf-run init`.")
    return path


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
