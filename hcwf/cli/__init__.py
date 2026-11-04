"""The ``hcwf-run`` command line: run a config, start one (init), convert an old one, describe datasets.

The commands are::

    hcwf-run -p run.yaml            run (also `hcwf-run run -p run.yaml`; HCWF_CONFIG works instead of -p)
    hcwf-run -p run.yaml --check    validate and print the plan, run nothing (exit 1 on errors)
    hcwf-run -p a.yaml -p b.yaml    several run files (or a glob), one after another: one region per file
                                    (a run file with a `blend:` section trains the blend)
    hcwf-run init [...]             write a starting run.yaml, detecting dataset values from your paths
    hcwf-run datasets -p run.yaml   show what the config's datasets resolve to, with checks
    hcwf-run convert old.yaml -o run.yaml   convert a run YAML with the old flat onset keys (wet_init, ...)
    hcwf-run convert blend.yml -o run.yaml          convert an onset_blending config to a run YAML with a blend: section
    hcwf-run convert --specs ... -o run.yaml        the same from a legacy spec set
    hcwf-run forecast BLEND... --forecast MODEL=PATH [...]   apply trained blends (trained.blend) to live forecasts
    hcwf-run export -p run.yaml     write the gridded onset dates of each source as NetCDF (no scoring)
    hcwf-run cache info|clean       what the on-disk caches hold (kinds, entries, sizes, ages); delete entries
    hcwf-run onset list|describe|regions|schematic|explain|check-alignment   the onset definitions and regions
    hcwf-run -h                     this help (with a config, -h lists the run-time overrides)

This module holds the dispatcher `main` and the handlers of the subcommands other than a run. A run (the default
command) is `hcwf.cli.driver.run_hcwf`; the override flags are defined in `hcwf.cli.parser` and the ``onset``
subcommands in `hcwf.cli.onset`. The work itself is done by `hcwf.api`, `hcwf.runs`, `hcwf.config` and `hcwf.cache`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HELP = """usage: hcwf-run [-p run.yaml] [--check] [--overwrite] [--quiet|--verbose] [overrides]      run the config
       hcwf-run init [options]                  write a starting run.yaml
       hcwf-run datasets -p run.yaml            show how the config's datasets resolve
       hcwf-run convert old.yaml -o run.yaml    convert a run YAML with the old flat onset keys
       hcwf-run convert blend.yml -o run.yaml   convert an onset_blending config (or --specs ...) to run YAML
       hcwf-run forecast BLEND --forecast MODEL=PATH   apply a trained blend to live forecast files
       hcwf-run export -p run.yaml              write the gridded onset dates as NetCDF
       hcwf-run cache info|clean                list or delete cached results (runs/.cache and ~/.cache/hcwf)
       hcwf-run onset COMMAND                   onset definitions: list, describe, regions, schematic, explain, check-alignment

HCWF (Human-Centered Weather Forecasts) benchmarks rainy-season onset forecasts.
One YAML run config describes a run; point hcwf-run at it with -p (or HCWF_CONFIG).

commands:
  run        (default) run the config given with -p (repeat -p, or give a glob, to run several files in turn). With a config, `hcwf-run -p run.yaml -h` lists the
             command-line overrides of config values. --check validates and resolves everything and prints
             the plan (settings with sources, datasets and detections, mask, leads, files, alignment, hints,
             run folder, cache) without running; --overwrite re-runs a run folder that exists (clears it);
             --quiet shows only warnings, --verbose adds debug lines (run.log always has them).
  init       write a short commented run.yaml. Use --obs PATH and --model PATH (repeatable) to detect the
             datasets' variable, units, dims, members, file pattern and lead label from your files, with the
             evidence written next to each value. Also: --region, --years Y0 Y1, --definition ID, -o FILE,
             --force, --dataset-template obs|forecast.
  datasets   resolve the datasets of a config (detections with their evidence, warnings, problems).
  convert    read a run YAML with the old flat onset keys (wet_init, wet_spell, ...), an onset_blending config (YAML)
             or a legacy spec set (--specs) and write the
             equivalent run YAML (-o FILE, --force, --region ID for blending configs, which have none). Blending
             configs and specs become `onset: {rule: ...}` with the parity options, `units:` and a `blend:`
             section; what is not carried over is listed in comments at the top.
  forecast   apply trained blends (the trained.blend a blend run writes into its blend/ folder, or one of
             onset_blending's) to live forecast files: `hcwf-run forecast BLEND... --forecast MODEL=PATH
             [--forecast ...] [--issue-date YYYY-MM-DD] [--fewer-members] [--netcdf] [-o DIR]`. PATH is a file or a glob;
             give each trained model once (repeat for several files of a model). The live files may
             hold more than the blend was trained on (leads, members, a larger domain) but not less, and must have
             its units, accumulation, grid spacing and issue hour; every difference is reported before anything is
             read. The issue date is the one the files hold, else the latest, and must lie in the blend's issue
             window. Writes blended_forecast_probabilities.csv, and daily_onset_probabilities.csv for a random forest
             blend (--netcdf adds a .nc twin of each), into -o DIR (default: forecasts/<issue date>/ beside the blend). Several
             blends (one per region) are applied one after another to the same live files.
  export     run only the onset detection of a config and write one NetCDF per source (observed, each model,
             climatology, conditional_climatology) with the dates exactly as HCWF scores them. Options:
             --sources observed,AIFS,climatology (default: all the run uses), --out DIR (default: the run's onsets folder, dir_out/onsets with an explicit data_dir).
             `output.save_onsets: true` in the run file does the same before a normal run's scoring.
  cache      `hcwf-run cache info [--dir DIR ...]` lists each cache folder (default: runs/.cache here and the shared
             $HCWF_CACHE_DIR or ~/.cache/hcwf) by kind of result: entries, size, oldest and newest. `hcwf-run cache clean
             (--all | --kind KIND ... | --older-than 30d) [--dir DIR ...] [--dry-run]` deletes whole entries (what a
             run needs again is recomputed once); --dry-run lists what would go and deletes nothing.
  onset      the onset component: `list` (definition ids), `describe ID` (the rule in words, hash, status), `regions`,
             `schematic ID --png F`, `explain --obs ... --lat --lon --year` (why one cell got its date),
             `check-alignment` (obs vs forecast day lags). `hcwf-run onset COMMAND -h` lists each one's options.
"""
"""The text of ``hcwf-run -h`` (shown when no config is named)."""


def _init(argv) -> int:
    """Write a starting run config (``hcwf-run init``).

    Parameters
    ----------
    argv : list of str
        The command line after ``init``: ``--region``, ``--obs``, ``--model`` (repeatable), ``--years``,
        ``--definition``, ``--blend``, ``-o``, ``--force`` and ``--dataset-template``.

    Returns
    -------
    int
        The exit code: 0 on success (or after printing a dataset template), 1 when the file cannot be written or the
        region or datasets are wrong.
    """
    from hcwf.config import datasets as dataset_schema
    from hcwf.config.init import write_init

    p = argparse.ArgumentParser(prog="hcwf-run init", description="Write a starting run.yaml.")
    p.add_argument("--region", help="region id or alias (default: detected from --obs, else a placeholder)")
    p.add_argument("--obs", metavar="PATH", help="folder of the observation files")
    p.add_argument("--model", metavar="PATH", action="append", default=[], help="folder of a model's forecasts "
                                                                                "(repeat for several models)")
    p.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"), default=(2019, 2024))
    p.add_argument("--definition", help="registry id of an onset definition (default: the region's, not written)")
    p.add_argument("--blend", action="store_true", help="write a `blend:` section (a commented skeleton without it)")
    p.add_argument("-o", "--output", default="run.yaml", help="file to write, '-' for the screen (default run.yaml)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--dataset-template", choices=("obs", "forecast"),
                   help="print a blank dataset entry and exit")
    args = p.parse_args(argv)
    if args.dataset_template:
        print(dataset_schema.TEMPLATES[args.dataset_template], end="")
        return 0
    try:
        target = write_init(args.output, force=args.force, region=args.region, obs=args.obs, models=args.model,
                            years=tuple(args.years), definition=args.definition, blend=args.blend)
    except (FileExistsError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - region and dataset errors carry their own fix text
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if str(target) != "-":
        print(f"Wrote {target}. Review it (look for TODO), then run: hcwf-run -p {target}")
    return 0


def _datasets(argv) -> int:
    """Print the resolution of the config's datasets (``hcwf-run datasets -p run.yaml``).

    Parameters
    ----------
    argv : list of str
        The command line after ``datasets``: ``-p`` (the run config; ``$HCWF_CONFIG`` by default).

    Returns
    -------
    int
        The exit code: 0 when every dataset resolved, 1 when the config cannot be read or a dataset does not resolve.
    """
    import warnings

    import yaml

    from hcwf.errors import ConfigSchemaError
    from hcwf.config.mapping import build_config

    p = argparse.ArgumentParser(prog="hcwf-run datasets", description="Show how a config's datasets resolve.")
    p.add_argument("-p", "--param", default=os.environ.get("HCWF_CONFIG"), help="the run config (or HCWF_CONFIG)")
    args = p.parse_args(argv)
    if not args.param:
        p.error("give the run config with -p run.yaml (or set HCWF_CONFIG)")
    try:
        raw = yaml.safe_load(open(os.path.expanduser(args.param)).read())
    except (OSError, yaml.YAMLError) as exc:
        print(f"Error: cannot read {args.param}: {exc}", file=sys.stderr)
        return 1
    from hcwf.cache import run_scope

    with warnings.catch_warnings(), run_scope():
        warnings.simplefilter("ignore")
        try:
            cfg = build_config(raw, source=args.param)
            report, error = cfg["dataset_report"], None
        except ConfigSchemaError as exc:
            report, error = getattr(exc, "report", ()), exc
    print("\n".join(report) if report else "(no datasets described)")
    if error is not None:
        print(f"\n{error}")
        return 1
    print("\nAll datasets resolved.")
    return 0


def _convert(argv) -> int:
    """Convert an old config to a run YAML (``hcwf-run convert SOURCE -o run.yaml``).

    The source is a run YAML with flat onset keys, an onset_blending config YAML, or (with ``--specs``) a legacy spec
    set. Any failure of the conversion is printed and gives exit code 1.

    Parameters
    ----------
    argv : list of str
        The command line after ``convert``: the source file, ``-o``, ``--force``, ``--region`` and, for ``--specs``,
        ``--name``, ``--ground-truth``, ``--forecast``, ``--combine``, ``--connect``, ``--cv-models``, ``--regrid`` and
        ``--data-root``.

    Returns
    -------
    int
        The exit code: 0 on success, 1 when the conversion failed.

    Raises
    ------
    FileExistsError
        The output file exists and ``--force`` is not given. Raised inside the command and printed as an error with exit
        code 1; like every other error of the conversion it does not reach the caller.
    ValueError
        The source is not a YAML file. Printed as an error with exit code 1; it does not reach the caller either.
    """
    import yaml

    p = argparse.ArgumentParser(prog="hcwf-run convert", description="Convert a run YAML with the old flat onset keys, an onset_blending "
                                "config (YAML) or a legacy spec set (--specs) to a YAML run config.")
    p.add_argument("source", nargs="?", help="the run YAML with flat onset keys, or the onset_blending config YAML")
    p.add_argument("-o", "--output", default="run.yaml")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--region", help="HCWF region id of the converted blending config (default: none, the data's whole "
                                    "domain, as blending reads it)")
    specs = p.add_argument_group("legacy spec set (--specs)")
    specs.add_argument("--specs", action="store_true", help="convert a set of stage specs")
    specs.add_argument("--name", help="the blend's name")
    specs.add_argument("--ground-truth", help="raw_data spec of the observations and climatologies")
    specs.add_argument("--forecast", action="append", default=[], metavar="MODEL=SPEC",
                       help="a model's raw_data spec (repeat per model)")
    specs.add_argument("--combine")
    specs.add_argument("--connect")
    specs.add_argument("--cv-models")
    specs.add_argument("--regrid")
    specs.add_argument("--data-root", help="where the specs' relative paths start (default: the current folder)")
    args = p.parse_args(argv)
    try:
        target = Path(args.output)
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} exists. Pass --force to overwrite it")
        if args.specs:
            from hcwf.config.convert_specs import convert_spec_files, validate_conversion

            missing = [n for n in ("name", "ground_truth", "combine", "connect", "cv_models") if not getattr(args, n)]
            if missing or not args.forecast:
                p.error("--specs needs --name, --ground-truth, --forecast MODEL=SPEC (repeatable), --combine, "
                        "--connect and --cv-models")
            forecasts = dict(item.partition("=")[::2] for item in args.forecast)
            if "" in forecasts.values():
                p.error("--forecast expects MODEL=SPEC")
            conversion = convert_spec_files(name=args.name, ground_truth=args.ground_truth, forecasts=forecasts,
                                            combine=args.combine, connect=args.connect, cv_models=args.cv_models,
                                            regrid=args.regrid, data_root=args.data_root, region=args.region)
            sources = {"ground_truth": args.ground_truth, **{f"forecast[{m}]": f for m, f in forecasts.items()},
                       "combine": args.combine, "connect": args.connect, "cv_models": args.cv_models,
                       **({"regrid": args.regrid} if args.regrid else {})}
            validate_conversion(conversion)
            text = conversion.to_yaml(sources, origin="the stage specs")
        elif not args.source:
            p.error("give the file to convert (or --specs ...)")
        elif Path(args.source).suffix.lower() in (".yml", ".yaml"):
            from hcwf.config.convert_specs import convert_blend_file, is_blend_config, validate_conversion

            if not is_blend_config(yaml.safe_load(Path(args.source).read_text())):
                from hcwf.config.convert import convert_flat_onset

                if args.region:
                    p.error("--region applies to converting a blending config; a run YAML keeps its own region")
                return _converted(convert_flat_onset(args.source, args.output, force=args.force))
            conversion = convert_blend_file(args.source, region=args.region)
            validate_conversion(conversion)
            text = conversion.to_yaml({"config": args.source})
        else:
            raise ValueError(f"{args.source} is not a YAML file. Only YAML configs are read; start one with `hcwf-run init`")
        target.write_text(text)
    except Exception as exc:  # noqa: BLE001 - report any conversion problem with its message
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return _converted(target)


def _converted(target) -> int:
    """Print that the converted run YAML was written.

    Parameters
    ----------
    target : str or pathlib.Path
        The file written.

    Returns
    -------
    int
        The exit code, 0.
    """
    print(f"Wrote {target}. Check it, then run: hcwf-run -p {target}")
    return 0


def _forecast_targets(blends, output):
    """Return the output folder of each blend, or the text of the problem.

    With several blends and ``-o``, each gets a subfolder named after its folder, or after the run folder when that is
    ``blend`` (``runs/<run>/blend/trained.blend``); two blends of one name are refused up front, before anything is
    written.

    Parameters
    ----------
    blends : list of str
        The ``trained.blend`` files.
    output : str or None
        The ``-o`` folder.

    Returns
    -------
    list of str or None or str
        One entry per blend: the folder to write to (None: beside the blend). A text instead of a list says that
        several blends would write to the same subfolder.
    """
    if not output:
        return [None] * len(blends)
    if len(blends) == 1:
        return [output]
    names = []
    for index, path in enumerate(blends):
        folder = Path(path).resolve().parent
        names.append((folder.parent.name if folder.name == "blend" else folder.name) or f"blend{index + 1}")
    clashes = {name: [b for b, n in zip(blends, names) if n == name] for name in names if names.count(name) > 1}
    if clashes:
        return "several blends would write to the same subfolder of -o: " + "; ".join(
            f"'{name}' for {', '.join(files)}" for name, files in clashes.items()) + \
            ". Fix: give them to separate calls, each with its own -o"
    return [str(Path(output) / name) for name in names]


def _forecast(argv) -> int:
    """Apply trained blends to live forecast files (``hcwf-run forecast BLEND... --forecast MODEL=PATH [...]``).

    A blend that cannot be applied is reported on stderr and the others still run.

    Parameters
    ----------
    argv : list of str
        The command line after ``forecast``: the ``trained.blend`` files, ``--forecast`` (repeatable),
        ``--issue-date``, ``--fewer-members``, ``--netcdf`` and ``-o``.

    Returns
    -------
    int
        The exit code: 0, or 1 when a blend failed.
    """
    p = argparse.ArgumentParser(prog="hcwf-run forecast", description="Apply trained blends (trained.blend files) to "
                                "live forecast files and write the blended onset probabilities.")
    p.add_argument("blend", nargs="+", help="a trained.blend file (several: one region each, applied in turn)")
    p.add_argument("--forecast", action="append", required=True, metavar="MODEL=PATH",
                   help="the live forecast file(s) of a trained model: a file or glob (repeat per model, or per file)")
    p.add_argument("--issue-date", help="forecast this issue date only (YYYY-MM-DD); default the one the files hold")
    p.add_argument("--fewer-members", action="store_true",
                   help="allow live ensembles that hold only some of the trained members")
    p.add_argument("--netcdf", action="store_true", help="also write the probabilities as NetCDF (default: CSV only)")
    p.add_argument("-o", "--output", help="new folder for the forecast (default forecasts/<issue date>/ beside the "
                                          "blend; with several blends, one subfolder each)")
    args = p.parse_args(argv)
    from hcwf.api import forecast
    from hcwf.blend.forecast import parse_forecast_arguments
    from hcwf.errors import IncompatibleForecast
    from hcwf.errors import BlendFileError

    try:
        files = parse_forecast_arguments(args.forecast)
    except ValueError as exc:
        p.error(str(exc))
    targets = _forecast_targets(args.blend, args.output)
    if isinstance(targets, str):
        p.error(targets)
    failed = 0
    for path, target in zip(args.blend, targets):
        target = None if target is None else Path(target)
        try:
            written = forecast(path, files, issue_date=args.issue_date, fewer_members=args.fewer_members, output=target,
                               write=True, netcdf=args.netcdf).folder
        except (IncompatibleForecast, BlendFileError, FileExistsError, FileNotFoundError, ValueError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"Wrote {written}")
    return 1 if failed else 0


def _export(argv) -> int:
    """Write the gridded onset dates as NetCDF (``hcwf-run export -p run.yaml [--sources ...] [--out DIR]``).

    Parameters
    ----------
    argv : list of str
        The command line after ``export``: ``-p`` (``$HCWF_CONFIG`` by default), ``--sources`` and ``--out``.

    Returns
    -------
    int
        The exit code: 0 on success, 1 when the export failed.
    """
    from hcwf.runs.export import export_onsets

    p = argparse.ArgumentParser(prog="hcwf-run export", description="Write the gridded onset dates of a run as "
                                "NetCDF, one file per source, running only the detection.")
    p.add_argument("-p", "--param", default=os.environ.get("HCWF_CONFIG"), help="the run config (or HCWF_CONFIG)")
    p.add_argument("--sources", help="comma-separated: observed, model names, climatology, conditional_climatology "
                                     "(default: everything the run uses)")
    p.add_argument("--out", help="folder for the files (default: dir_out/onsets of the config)")
    args = p.parse_args(argv)
    if not args.param:
        p.error("give the run config with -p run.yaml (or set HCWF_CONFIG)")
    try:
        paths = export_onsets(args.param, sources=args.sources, out=args.out)
    except Exception as exc:  # noqa: BLE001 - config and data problems carry their own fix text
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for source, path in paths.items():
        print(f"{source}: {path}")
    return 0


def _onset(argv) -> int:
    """Run the ``onset`` commands (``hcwf-run onset ...``): definitions, regions, schematic and one-cell explanations.

    Parameters
    ----------
    argv : list of str
        The command line after ``onset``.

    Returns
    -------
    int
        The exit code of `hcwf.cli.onset.main`.
    """
    from hcwf.cache import run_scope
    from hcwf.cli.onset import main

    with run_scope():
        return main(argv)


def _age(seconds: float) -> str:
    """Return a time span as text: ``5m``, ``3h``, ``12d``.

    Parameters
    ----------
    seconds : float
        The span in seconds.

    Returns
    -------
    str
        The span in the largest of days, hours and minutes that it reaches, else in seconds (0 for a negative span).
    """
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds / size:.0f}{unit}"
    return f"{max(seconds, 0):.0f}s"


def _size(nbytes: float) -> str:
    """Return a size in bytes as text, e.g. ``3.2 MB``.

    Parameters
    ----------
    nbytes : float
        The size in bytes.

    Returns
    -------
    str
        The size in B, KB, MB or GB (1024 per step).
    """
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} B" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def _cache(argv) -> int:
    """Show what the on-disk caches hold, or delete entries (``hcwf-run cache info|clean``).

    Parameters
    ----------
    argv : list of str
        The command line after ``cache``: the action (``info`` or ``clean``), ``--dir`` (repeatable), ``--kind``
        (repeatable), ``--older-than``, ``--all`` and ``--dry-run``.

    Returns
    -------
    int
        The exit code, 0.
    """
    import time

    from hcwf import cache as cache_mod

    p = argparse.ArgumentParser(prog="hcwf-run cache", description="List or delete the cached results (observed onsets, "
                                "thresholds, dataset detection, ...) in the cache folders.")
    p.add_argument("action", choices=("info", "clean"))
    p.add_argument("--dir", action="append", default=[], metavar="DIR",
                   help="a cache folder (repeat for several; default: runs/.cache here and the shared cache, "
                        "$HCWF_CACHE_DIR or ~/.cache/hcwf)")
    p.add_argument("--kind", action="append", default=[], metavar="KIND", help="only this kind of result (repeat)")
    p.add_argument("--older-than", metavar="AGE", help="clean: only entries last written longer ago than this (12h, 30d, 2w)")
    p.add_argument("--all", action="store_true", help="clean: delete every entry of the chosen kinds")
    p.add_argument("--dry-run", action="store_true", help="clean: list what would be deleted, delete nothing")
    args = p.parse_args(argv)
    older = None
    if args.older_than is not None:
        try:
            older = cache_mod.parse_duration(args.older_than)
        except ValueError as exc:
            p.error(str(exc))
    if args.action == "clean" and not (args.all or args.kind or older is not None):
        p.error("clean needs --all, --kind KIND or --older-than AGE (nothing is deleted without one of them)")
    roots = [Path(d).expanduser() for d in args.dir] or [Path.cwd() / "runs" / ".cache", cache_mod.global_root()]
    now = time.time()
    total = 0
    for root in roots:
        stores = cache_mod.store_kinds(root, args.kind)
        print(f"{root}" + ("" if root.is_dir() else "  (not there)"))
        for store in stores:
            if args.action == "info":
                entries = store.entries()
                if not entries:
                    continue
                size = sum(e.size for e in entries)
                print(f"  {store.kind:<18} {len(entries):>6} entries  {_size(size):>10}  oldest {_age(now - entries[0].mtime)}"
                      f"  newest {_age(now - entries[-1].mtime)}")
                total += size
            else:
                gone = store.prune(older_than=older, dry_run=args.dry_run, now=now)
                if gone:
                    size = sum(e.size for e in gone)
                    print(f"  {store.kind:<18} {'would delete' if args.dry_run else 'deleted'} {len(gone)} entries ({_size(size)})")
                    total += size
    if args.action == "info":
        print(f"total {_size(total)}")
    else:
        print(f"{'would free' if args.dry_run else 'freed'} {_size(total)}")
    return 0


def _has_config(argv) -> bool:
    """Return whether a config is named by ``-p``/``--param`` in `argv` or by ``$HCWF_CONFIG``.

    Parameters
    ----------
    argv : list of str
        The command line.

    Returns
    -------
    bool
        True when a config is named.
    """
    return any(a in ("-p", "--param") or a.startswith("--param=") for a in argv) or bool(os.environ.get("HCWF_CONFIG"))


COMMANDS = {"init": _init, "datasets": _datasets, "convert": _convert, "export": _export, "forecast": _forecast,
            "cache": _cache, "onset": _onset}
"""The subcommands: name to handler, which takes the arguments after the name and returns an exit code."""


def main(argv=None) -> None:
    """Run `hcwf-run`: a subcommand, the top-level help, or (the default) a run of the config(s).

    Exits with the code of the subcommand or of the run when it is not 0.

    Parameters
    ----------
    argv : list of str, optional
        The arguments after the program name; ``sys.argv`` is used (and replaced by these when given) when None.
    """
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    args = sys.argv[1:]
    command = args[0] if args else None
    if command in COMMANDS:
        code = COMMANDS[command](args[1:])
        if code:
            sys.exit(code)
        return
    if command == "run":
        sys.argv = [sys.argv[0], *args[1:]]
    elif any(a in ("-h", "--help") for a in args) and not _has_config(args):
        print(HELP, end="")
        return
    from hcwf.cli.driver import run_hcwf

    run_hcwf()


if __name__ == "__main__":
    main()
