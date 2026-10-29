"""
ROMP Main Entry Script
----------------------
This script serves as the primary execution point for the Rainy season Onset
Metrics Package (ROMP). It orchestrates the calculation of skill scores
or the generation of spatial performance maps.

Usage:
    momp-run -p run.yaml [--check] [--overwrite] [--quiet | --verbose]
    or
    python -m momp.driver -p run.yaml

``--check`` validates and resolves the config, prints the plan and runs nothing (exit 1 on errors). The run
itself is momp.api.execute, which ``momp.run`` (notebooks) shares.
"""

import argparse
import logging
import sys
import traceback
from types import SimpleNamespace

from momp.lib.assertion import ROMPConfigError
from momp.lib.config_schema import ConfigSchemaError
from momp.utils import log as logs

log = logging.getLogger(__name__)


def _flags(argv):
    """The run-control flags (--check, --overwrite, --quiet, --verbose) of the command line."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("check", "overwrite", "quiet", "verbose"):
        parser.add_argument(f"--{name}", action="store_true")
    return parser.parse_known_args(argv)[0]


def _check(flags) -> int:
    """`--check`: print the plan of the run (or why the config cannot be resolved); return the exit code."""
    from momp.lib.check import build_report
    from momp.lib.loader import load_run_config

    try:
        dic = load_run_config()
    except ConfigSchemaError as exc:
        print("\n".join(getattr(exc, "report", ())) or "(no datasets described)")
        print(f"\n{exc}\n\ncheck: the config cannot be resolved; nothing was run")
        return 1
    except (ROMPConfigError, FileNotFoundError) as exc:
        print(f"Error: {exc}\n\ncheck: the config cannot be resolved; nothing was run")
        return 1
    with logs.session(logs.QUIET):
        report = build_report(dic, overwrite=flags.overwrite)
    print(report.text())
    return 0 if report.ok else 1


def run_momp(cfg=None, setting=None):
    """
    Executes the standard ROMP evaluation workflow.

    This function triggers the bin-based skill score calculations
    or generates spatial maps for False Alarm Rate (FAR),
    Miss Rate (MR), and Mean Absolute Error (MAE). It reads the config named by -p (or MOMP_CONFIG),
    writes the run folder (or the config's data_dir) and exits with status 1 on any failure.

    Args:
        cfg: A loaded configuration (dict or namespace) to run instead of reading one from the command line.
        setting: Unused; kept for compatibility (the Setting is built from the config).
    """
    from momp.api import execute
    from momp.lib import loader
    from momp.lib.assertion import ROMPValidator
    from momp.lib.run_folder import RunExistsError

    if cfg is None and setting is None:
        from momp.cli import handled_without_run

        if handled_without_run(sys.argv[1:]):
            return

    flags = _flags(sys.argv[1:])
    if cfg is None and flags.check:
        sys.exit(_check(flags))
    try:
        if cfg is None:
            dic = loader.load_run_config()
        else:
            dic = dict(vars(cfg)) if isinstance(cfg, SimpleNamespace) else dict(cfg)
            if "run_paths" not in dic:
                loader.plan_run(dic, dic.get("config_file"))
            ROMPValidator(dic).validate()
    except (ROMPConfigError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        execute(dic, write=True, overwrite=flags.overwrite, verbosity=logs.verbosity_from(flags.quiet, flags.verbose))
    except RunExistsError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: ROMP failed during execution: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


# ------------------------------------------------------------------------------
# EXECUTION BLOCK
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    run_momp()
