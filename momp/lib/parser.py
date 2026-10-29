"""Command-line parser that overrides run-config values, plus argparse type helpers."""
import argparse
import ast
from pathlib import Path


def create_parser(config, cli_args=None):
    """Parse CLI overrides, using the config dict values as defaults; unknown arguments are ignored."""

    parser = argparse.ArgumentParser(description="ROMP Package Parameter Loader")

    parser.add_argument(
        "--model_list",
        nargs="+",
        default=config["model_list"],
        help=f"Model list (default: {config['model_list']})"
    )

    parser.add_argument(
        "--verification_window_list",
        type=parse_window_list,
        default=config['verification_window_list'],
        help="Verification window as start end day (default: {config['verification_window_list']}), \
                (e.g., '1,15 16,20')"
    )

    parser.add_argument(
        "--max_forecast_day",
        type=int,
        default=config['max_forecast_day'],
        help="Max forecast day (default: {config['max_forecast_day']})"
    )

    # The flat onset keys exist only in legacy configs (see momp.lib.onset_config).
    flat_options = (
        ("wet_init", float, "Rain threshold (mm) for first potential wet day"),
        ("wet_threshold", float, "Rainfall threshold (mm) if no thresh_file provided"),
        ("wet_spell", int, "Min days rainfall must stay above threshold"),
        ("dry_threshold", float, "Max rainfall (mm) to define a dry day"),
        ("dry_spell", int, "Max consecutive dry days allowed after onset"),
        ("dry_extent", int, "Search window (days) after onset to check for dry spell"),
    )
    for name, kind, text in flat_options:
        if name in config:
            parser.add_argument(f"--{name}", type=kind, default=config[name],
                                help=f"{text} (default: {config[name]})")

    if "dry_spell_check" in config:
        parser.add_argument(
            "--dry_spell_check",
            type=str2bool,
            default=config["dry_spell_check"],
            help=f"Keep the definition's dry-spell condition, True or False (default: {config['dry_spell_check']})"
        )

    for name, choices in (("forecast_history", ("observed", "none")), ("threshold_source", ("observations", "model"))):
        if name in config:
            parser.add_argument(f"--{name}", choices=choices, default=config[name],
                                help=f"{name}: {' or '.join(choices)} (default: {config[name]})")

    parser.add_argument(
        "--onset_percentage_threshold",
        type=float,
        default=config['onset_percentage_threshold'],
        help=f"Probability threshold (0.0-1.0) for ensemble onset (default: {config['onset_percentage_threshold']})"
    )

    # Allowed error margin (in days)
    parser.add_argument(
        "--tolerance_days_list",
        nargs='+',
        type=int,
        action=ForceTupleAction,
        default=config['tolerance_days_list'],
        help=f"Allowed error margins in days e.g., 3 5 (default: {config['tolerance_days_list']})"
    )

    # Time windows for grouping statistics
    parser.add_argument(
        "--day_bins",
        type=parse_window_list,
        default=config['day_bins'],
        help=f"Day bins for statistics e.g., '1,15 16,30' (default: {config['day_bins']})"
    )

    # Date range for evaluation
    parser.add_argument(
        "--start_date",
        nargs=3,
        type=int,
        action=ForceTupleAction,
        default=config['start_date'],
        help="Start date as: YYYY M D (e.g., 2024 5 1)"
    )

    parser.add_argument(
        "--end_date",
        nargs=3,
        type=int,
        action=ForceTupleAction,
        default=config['end_date'],
        help=f"Evaluation end date as YYYY M D (e.g., 2024 10 31) (default: {config['end_date']})"
    )

    parser.add_argument(
        "--show_plot",
        type=str2bool,
        default=config['show_plot'],
        help="Toggle individual model plot display, True of False (default: {config['show_plot']})"
    )

    parser.add_argument(
        "--show_panel",
        type=str2bool,
        default=config['show_panel'],
        help="Toggle panel plot display, True of False (default: {config['show_panel']})"
    )

    parser.add_argument(
        "--probabilistic",
        type=str2bool,
        default=config['probabilistic'],
        help=f"Ensemble evaluation toggle, True or False (default: {config['probabilistic']})"
    )

    parser.add_argument(
        "--debug",
        type=str2bool,
        default=config['debug'],
        help="debug model for developers only, True of False (default: {config['debug']})"
    )

    parser.add_argument(
        "--parallel",
        type=str2bool,
        default=config['parallel'],
        help="parallelization, True of False (default: {config['parallel']})"
    )

    parser.add_argument(
        "--region",
        type=str,
        default=config['region'],
        help="region as defined in params.region_def (default: {config['region']})"
    )

    # run control (handled by the driver, not config values)
    parser.add_argument("--check", action="store_true",
                        help="validate and resolve the config, print the plan (settings, datasets, files, leads, "
                             "hints, run folder, cache) and run nothing; exit 1 on errors")
    parser.add_argument("--alignment", action="store_true",
                        help="with --check: also run the observation/forecast alignment check (off by default)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run a config whose run folder already exists (clears that folder first)")
    parser.add_argument("--quiet", action="store_true", help="show only warnings and errors")
    parser.add_argument("--verbose", action="store_true", help="show debug detail (run.log always has it)")

    # cli_args can be passed explicitly; if None, default to sys.argv
    if cli_args is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(cli_args)

    # ---- Convert list → tuple if needed ----
    # Only convert if user supplied; default already tuple
    if isinstance(args.model_list, list):
        args.model_list = tuple(args.model_list)

    return args


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
            f"Config file not found: {path}. Start one with `momp-run init`.")
    return path
