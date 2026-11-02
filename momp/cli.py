"""The `momp-run` command line: run a config, start one (init), convert an old one, describe datasets.

    momp-run -p run.yaml            run (also `momp-run run -p run.yaml`; MOMP_CONFIG works instead of -p)
    momp-run -p run.yaml --check    validate and print the plan, run nothing (exit 1 on errors)
    momp-run -p a.yaml -p b.yaml    several run files (or a glob), one after another: one region per file
                                    (a run file with a `blend:` section trains the blend)
    momp-run init [...]             write a starting run.yaml, detecting dataset values from your paths
    momp-run datasets -p run.yaml   show what the config's datasets resolve to, with checks
    momp-run convert old.in -o run.yaml   convert an old config.in to YAML
    momp-run convert blend.yml -o run.yaml          convert an onset_blending config to a run YAML with a blend: section
    momp-run convert --specs ... -o run.yaml        the same from a legacy spec set
    momp-run forecast ...           reserved (applying a trained blend to new forecasts; not implemented yet)
    momp-run export -p run.yaml     write the gridded onset dates of each source as NetCDF (no scoring)
    momp-run -h                     this help (with a config, -h lists the run-time overrides)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HELP = """usage: momp-run [-p run.yaml] [--check] [--overwrite] [--quiet|--verbose] [overrides]      run the config
       momp-run init [options]                  write a starting run.yaml
       momp-run datasets -p run.yaml            show how the config's datasets resolve
       momp-run convert old.in -o run.yaml      convert an old config.in to YAML
       momp-run convert blend.yml -o run.yaml   convert an onset_blending config (or --specs ...) to run YAML
       momp-run export -p run.yaml              write the gridded onset dates as NetCDF

ROMP (Rainy season Onset Metrics Package) benchmarks rainy-season onset forecasts.
One YAML run config describes a run; point momp-run at it with -p (or MOMP_CONFIG).

commands:
  run        (default) run the config given with -p (repeat -p, or give a glob, to run several files in turn). With a config, `momp-run -p run.yaml -h` lists the
             command-line overrides of config values. --check validates and resolves everything and prints
             the plan (settings with sources, datasets and detections, mask, leads, files, alignment, hints,
             run folder, cache) without running; --overwrite re-runs a run folder that exists (clears it);
             --quiet shows only warnings, --verbose adds debug lines (run.log always has them).
  init       write a short commented run.yaml. Use --obs PATH and --model PATH (repeatable) to detect the
             datasets' variable, units, dims, members, file pattern and lead label from your files, with the
             evidence written next to each value. Also: --region, --years Y0 Y1, --definition ID, -o FILE,
             --force, --dataset-template obs|forecast.
  datasets   resolve the datasets of a config (detections with their evidence, warnings, problems).
  convert    read an old config.in, an onset_blending config (YAML) or a legacy spec set (--specs) and write the
             equivalent run YAML (-o FILE, --force, --region ID for blending configs, which have none). Blending
             configs and specs become `onset: {rule: ...}` with the parity options, `units:` and a `blend:`
             section; what is not carried over is listed in comments at the top.
  forecast   reserved for applying a trained blend; not implemented yet.
  export     run only the onset detection of a config and write one NetCDF per source (observed, each model,
             climatology, conditional_climatology) with the dates exactly as ROMP scores them. Options:
             --sources observed,AIFS,climatology (default: all the run uses), --out DIR (default: the run's onsets folder, dir_out/onsets with an explicit data_dir).
             `output.save_onsets: true` in the run file does the same before a normal run's scoring.
"""


def _init(argv) -> int:
    """`momp-run init`: write a starting run config."""
    from momp.lib import dataset_schema
    from momp.lib.init_config import write_init

    p = argparse.ArgumentParser(prog="momp-run init", description="Write a starting run.yaml.")
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
        print(f"Wrote {target}. Review it (look for TODO), then run: momp-run -p {target}")
    return 0


def _datasets(argv) -> int:
    """`momp-run datasets -p run.yaml`: print the resolution of the config's datasets."""
    import warnings

    import yaml

    from momp.lib.config_schema import ConfigSchemaError
    from momp.lib.config_yaml import build_config

    p = argparse.ArgumentParser(prog="momp-run datasets", description="Show how a config's datasets resolve.")
    p.add_argument("-p", "--param", default=os.environ.get("MOMP_CONFIG"), help="the run config (or MOMP_CONFIG)")
    args = p.parse_args(argv)
    if not args.param:
        p.error("give the run config with -p run.yaml (or set MOMP_CONFIG)")
    try:
        raw = yaml.safe_load(open(os.path.expanduser(args.param)).read())
    except (OSError, yaml.YAMLError) as exc:
        print(f"Error: cannot read {args.param}: {exc}", file=sys.stderr)
        return 1
    with warnings.catch_warnings():
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
    """`momp-run convert SOURCE -o run.yaml`: an old config.in, a BlendConfig YAML, or (--specs) a legacy spec set."""
    import yaml

    p = argparse.ArgumentParser(prog="momp-run convert", description="Convert an old config.in, an onset_blending "
                                "config (YAML) or a legacy spec set (--specs) to a YAML run config.")
    p.add_argument("source", nargs="?", help="the old config.in, or the onset_blending config YAML")
    p.add_argument("-o", "--output", default="run.yaml")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--region", help="ROMP region id of the converted blending config (default: none, the data's whole "
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
            from momp.lib.convert_specs import convert_spec_files, validate_conversion

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
            from momp.lib.convert_specs import convert_blend_file, is_blend_config, validate_conversion

            if not is_blend_config(yaml.safe_load(Path(args.source).read_text())):
                raise ValueError(f"{args.source} is not an onset_blending config (no observations.files, onset and "
                                 "evaluation); a ROMP run config needs no conversion")
            conversion = convert_blend_file(args.source, region=args.region)
            validate_conversion(conversion)
            text = conversion.to_yaml({"config": args.source})
        else:
            from momp.lib.convert import convert_file

            return _converted(convert_file(args.source, args.output, force=args.force))
        target.write_text(text)
    except Exception as exc:  # noqa: BLE001 - report any conversion problem with its message
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return _converted(target)


def _converted(target) -> int:
    print(f"Wrote {target}. Check it, then run: momp-run -p {target}")
    return 0


def _forecast(argv) -> int:
    """`momp-run forecast`: reserved for applying a trained blend to new forecasts (not implemented yet)."""
    print("Error: `momp-run forecast` is reserved and not implemented yet. A trained blend is written to the run "
          "folder's blend/ subfolder by `momp-run -p run.yaml` for a config with a `blend:` section.", file=sys.stderr)
    return 2


def _export(argv) -> int:
    """`momp-run export -p run.yaml [--sources ...] [--out DIR]`: write the gridded onset dates as NetCDF."""
    from momp.export import export_onsets, load_cfg

    p = argparse.ArgumentParser(prog="momp-run export", description="Write the gridded onset dates of a run as "
                                "NetCDF, one file per source, running only the detection.")
    p.add_argument("-p", "--param", default=os.environ.get("MOMP_CONFIG"), help="the run config (or MOMP_CONFIG)")
    p.add_argument("--sources", help="comma-separated: observed, model names, climatology, conditional_climatology "
                                     "(default: everything the run uses)")
    p.add_argument("--out", help="folder for the files (default: dir_out/onsets of the config)")
    args = p.parse_args(argv)
    if not args.param:
        p.error("give the run config with -p run.yaml (or set MOMP_CONFIG)")
    try:
        paths = export_onsets(load_cfg(args.param), sources=args.sources, out=args.out)
    except Exception as exc:  # noqa: BLE001 - config and data problems carry their own fix text
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for source, path in paths.items():
        print(f"{source}: {path}")
    return 0


def _has_config(argv) -> bool:
    """Whether a config is named by -p/--param in argv or MOMP_CONFIG."""
    return any(a in ("-p", "--param") or a.startswith("--param=") for a in argv) or bool(os.environ.get("MOMP_CONFIG"))


def handled_without_run(argv) -> bool:
    """Run init/datasets/convert/export or print the top-level help; True if that was the whole job.

    `run` is dropped from sys.argv. -h with a config falls through so the run parser lists its overrides.
    """
    command = argv[0] if argv else None
    commands = {"init": _init, "datasets": _datasets, "convert": _convert, "export": _export, "forecast": _forecast}
    if command in commands:
        code = commands[command](argv[1:])
        if code:
            sys.exit(code)
        return True
    if command == "run":
        sys.argv = [sys.argv[0], *argv[1:]]
        return False
    if any(a in ("-h", "--help") for a in argv) and not _has_config(argv):
        print(HELP, end="")
        return True
    return False


def main(argv=None) -> None:
    """Entry point of `momp-run`."""
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    if handled_without_run(sys.argv[1:]):
        return
    from momp.driver import run_momp

    run_momp()


if __name__ == "__main__":
    main()
