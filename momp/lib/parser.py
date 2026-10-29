"""Command-line parser that overrides config.in values, plus argparse type helpers."""
import argparse
import ast
from pathlib import Path
from importlib import resources


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

    parser.add_argument(
        "--wet_init",
        type=float,
        default=config['wet_init'],
        help=f"Rain threshold (mm) for first potential wet day (default: {config['wet_init']})"
    )

    parser.add_argument(
        "--wet_threshold",
        type=float,
        default=config['wet_threshold'],
        help=f"Rainfall threshold (mm) if no thresh_file provided (default: {config['wet_threshold']})"
    )

    parser.add_argument(
        "--wet_spell",
        type=int,
        default=config['wet_spell'],
        help=f"Min days rainfall must stay above threshold (default: {config['wet_spell']})"
    )

    parser.add_argument(
        "--dry_threshold",
        type=float,
        default=config['dry_threshold'],
        help=f"Max rainfall (mm) to define a dry day (default: {config['dry_threshold']})"
    )

    parser.add_argument(
        "--dry_spell",
        type=int,
        default=config['dry_spell'],
        help=f"Max consecutive dry days allowed after onset (default: {config['dry_spell']})"
    )

    parser.add_argument(
        "--dry_extent",
        type=int,
        default=config['dry_extent'],
        help=f"Search window (days) after onset to check for dry spell (default: {config['dry_extent']})"
    )

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
    """
    Return the config file path: the packaged momp/params/config.in for "params/config.in",
    otherwise param_path expanded and resolved (nothing is created).
    """
    if param_path == "params/config.in":
        # return a real path to the packaged file (read-only)
        with resources.as_file(resources.files("momp").joinpath("params/config.in")) as p:
            return Path(p)

    return Path(param_path).expanduser().resolve()
