"""
HCWF Main Entry Script
----------------------
This script serves as the primary execution point for HCWF (Human-Centered
Weather Forecasts). It orchestrates the calculation of skill scores
or the generation of spatial performance maps.

Usage:
    hcwf-run -p run.yaml [--check] [--overwrite] [--quiet | --verbose]
    hcwf-run -p a.yaml -p b.yaml     (or -p 'runs/*.yaml'): several run files, one after another
    or, with the interpreter, `python -m` and this module's name (the same as hcwf-run)

``--check`` validates and resolves the config, prints the plan and runs nothing (exit 1 on errors). The run
itself is hcwf.api.execute, which ``hcwf.run`` (notebooks) shares.
"""

import argparse
import glob
import logging
import os
import sys
import traceback

from hcwf.errors import HCWFConfigError
from hcwf import log as logs
from hcwf.errors import ConfigSchemaError

log = logging.getLogger(__name__)


def _flags(argv):
    """The run-control flags (--check, --overwrite, --quiet, --verbose) of the command line."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("check", "overwrite", "quiet", "verbose", "alignment"):
        parser.add_argument(f"--{name}", action="store_true")
    return parser.parse_known_args(argv)[0]


def config_paths(argv) -> list[str]:
    """The run files named by -p/--param in argv (repeatable; a glob names every file it matches), else HCWF_CONFIG."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-p", "--param", action="append", default=[])
    given = parser.parse_known_args(argv)[0].param or ([os.environ["HCWF_CONFIG"]] if os.environ.get("HCWF_CONFIG") else [])
    found: list[str] = []
    for item in given:
        text = os.path.expanduser(item)
        hits = sorted(glob.glob(text)) if any(c in text for c in "*?[") else [text]
        if not hits:
            raise HCWFConfigError(f"no run file matches '{item}'")
        found.extend(h for h in hits if h not in found)
    return found


def _argv_for(argv, path: str) -> list[str]:
    """argv with every -p/--param replaced by the one run file `path` (the other arguments are kept)."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ("-p", "--param"):
            skip = True
        elif not arg.startswith("--param="):
            out.append(arg)
    return [*out, "-p", path]


def _check(flags, cli_args=None) -> int:
    """`--check`: print the plan of the run (or why the config cannot be resolved); return the exit code."""
    from hcwf.cache import run_scope

    with run_scope():
        return _check_in_scope(flags, cli_args)


def _check_in_scope(flags, cli_args) -> int:
    from hcwf.config.build import resolve_command_line
    from hcwf.runs.check import build_report

    try:
        rc, found = resolve_command_line(cli_args, check=True)
        if rc is None:
            raise HCWFConfigError(found.message())
    except ConfigSchemaError as exc:
        print("\n".join(getattr(exc, "report", ())) or "(no datasets described)")
        print(f"\n{exc}\n\ncheck: the config cannot be resolved; nothing was run")
        return 1
    except (HCWFConfigError, FileNotFoundError) as exc:
        print(f"Error: {exc}\n\ncheck: the config cannot be resolved; nothing was run")
        return 1
    with logs.session(logs.QUIET):
        report = build_report(rc, found, overwrite=flags.overwrite, alignment=flags.alignment)
    print(report.text())
    return 0 if report.ok else 1


def _run(cfg, flags, cli_args=None) -> int:
    """Load (or take) a config and run it; the exit code: 0, or 1 after printing why it failed."""
    from hcwf.api import execute
    from hcwf.cache import run_scope
    from hcwf.config.build import resolve, resolve_command_line
    from hcwf.errors import RunExistsError

    with run_scope():   # one config's state, cleared before the next config of a batch
        try:
            rc = resolve_command_line(cli_args) if cfg is None else resolve(cfg)
        except (HCWFConfigError, FileNotFoundError) as e:
            print(f"Error: {e}")
            return 1

        try:
            execute(rc, write=True, overwrite=flags.overwrite,
                    verbosity=logs.verbosity_from(flags.quiet, flags.verbose))
        except (RunExistsError, HCWFConfigError) as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error: HCWF failed during execution: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
        return 0


def _batch(paths, argv, flags) -> int:
    """Run (or check) several run files one after another, one region per file; a failing file does not stop the
    others. The exit code is 1 when any failed."""
    failed = []
    for index, path in enumerate(paths, 1):
        print(f"\n=== {path} ({index} of {len(paths)}) ===", flush=True)
        cli_args = _argv_for(argv, path)
        code = _check(flags, cli_args) if flags.check else _run(None, flags, cli_args)
        if code:
            failed.append(path)
    print(f"\n{len(paths) - len(failed)} of {len(paths)} run files {'checked' if flags.check else 'finished'}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def run_hcwf(cfg=None):
    """
    Executes the standard HCWF evaluation workflow.

    This function triggers the bin-based skill score calculations
    or generates spatial maps for False Alarm Rate (FAR),
    Miss Rate (MR), and Mean Absolute Error (MAE); a config with a `blend:` section trains the blend instead. It reads
    the config named by -p (or HCWF_CONFIG; several -p run them one after another), writes the run folder (or the
    config's data_dir) and exits with status 1 on any failure.

    Args:
        cfg: A loaded configuration (dict or namespace) to run instead of reading one from the command line.
    """
    flags = _flags(sys.argv[1:])
    if cfg is None:
        try:
            paths = config_paths(sys.argv[1:])
        except HCWFConfigError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if len(paths) > 1:
            sys.exit(_batch(paths, sys.argv[1:], flags))
    if cfg is None and flags.check:
        sys.exit(_check(flags))
    code = _run(cfg, flags)
    if code:
        sys.exit(code)


# ------------------------------------------------------------------------------
# EXECUTION BLOCK
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from hcwf.cli import main

    main()
