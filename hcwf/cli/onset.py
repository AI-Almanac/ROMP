"""The ``hcwf-run onset`` commands: list, describe, regions, check-alignment, explain, schematic.

Each command prints to the screen; ``explain`` and ``schematic`` can also save a PNG. The commands are thin wrappers
over the onset component (definitions, regions, ingestion, alignment and plots); ``hcwf-run onset COMMAND -h`` lists the
options of each one.
"""
from __future__ import annotations

import argparse
import sys

import xarray as xr

from hcwf.onset import definition, regions
from hcwf.onset.alignment import check_forecast_alignment
from hcwf.onset.describe import describe
from hcwf.onset.ingest import ingest, ingest_forecast
from hcwf.errors import DefinitionError, RegionError
from hcwf.io.dims import find_dim

__all__ = []


def _list(args) -> int:
    """Print every registry definition with its title and regions.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line (no options).

    Returns
    -------
    int
        The exit code, 0.
    """
    for ident in definition.registry_ids():
        defn = definition.load(ident)
        where = ", ".join(defn.regions) or "any region"
        print(f"{ident:24s} {defn.title}  [{where}]")
    return 0


def _describe(args) -> int:
    """Print the sentence, hash, status and notes of a definition.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line: ``definition`` (a registry id or a YAML path) and ``no_dry_spell_check``.

    Returns
    -------
    int
        The exit code, 0.
    """
    defn = definition.load(args.definition)
    if args.no_dry_spell_check:
        defn = definition.without_dry_spell_check(defn)
    print(f"{defn.id} v{defn.version} ({defn.status})")
    print(defn.title)
    print()
    print(describe(defn))
    print()
    print(f"hash: {definition.definition_hash(defn)}")
    if defn.notes.strip():
        print(f"notes: {' '.join(defn.notes.split())}")
    return 0


def _regions(args) -> int:
    """Print every region with its default definition, day boundary and climatology years.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line (no options).

    Returns
    -------
    int
        The exit code, 0.
    """
    for ident in regions.region_ids():
        region = regions.load_region(ident)
        default, also = regions.definitions_for(ident)
        years = f"{region.climatology_years[0]}-{region.climatology_years[1]}" if region.climatology_years else "-"
        print(f"{ident:14s} {region.kind:8s} default: {default or '-':20s} "
              f"day boundary: {region.day_boundary or '-':28s} climatology years: {years}")
        if also:
            print(f"{'':14s} also: {', '.join(also)}")
    return 0


def _open(path: str, var: str) -> xr.DataArray:
    """Open one variable of a NetCDF file.

    Parameters
    ----------
    path : str
        The NetCDF file.
    var : str
        The variable.

    Returns
    -------
    xarray.DataArray
        The variable, read into memory block by block (`hcwf.spatial.select.load_blocks`).
    """
    from hcwf.config.detect import open_data
    from hcwf.spatial.select import load_blocks

    with open_data(path) as ds:
        da = ds[var]
        return load_blocks(da, find_dim([str(d) for d in da.dims], "time") or da.dims[0])


def _check_alignment(args) -> int:
    """Compare a forecast file with an observation file at day lags -3..+3 and print the verdict.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line: ``obs``, ``obs_var``, ``obs_day_boundary``, ``obs_units``, ``model``, ``model_var``,
        ``units`` and ``lead_label``.

    Returns
    -------
    int
        The exit code, 0.
    """
    obs = ingest(_open(args.obs, args.obs_var), day_boundary=args.obs_day_boundary, units=args.obs_units,
                 source=args.obs)
    forecast = ingest_forecast(_open(args.model, args.model_var), day_boundary=f"00 UTC, labelled by {args.lead_label}",
                               lead_label=args.lead_label, units=args.units, accumulation="daily", source=args.model)
    print(check_forecast_alignment(obs, forecast).summary())
    return 0


def _explain(args) -> int:
    """Print the readable log of one cell's onset search and optionally save its anatomy plot.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line: ``obs``, ``obs_var``, ``obs_units``, ``day_boundary``, ``definition``, ``lat``,
        ``lon``, ``year``, ``threshold_years``, ``no_dry_spell_check``, ``png`` and ``dark``.

    Returns
    -------
    int
        The exit code, 0.

    Raises
    ------
    ValueError
        The definition needs thresholds or has derived seasons and ``--threshold-years`` is not given.
    """
    from hcwf.onset import climatology, explain
    from hcwf.onset.model import DerivedSeasons

    defn = definition.load(args.definition)
    if args.no_dry_spell_check:
        defn = definition.without_dry_spell_check(defn)
    rain = ingest(_open(args.obs, args.obs_var), day_boundary=args.day_boundary, units=args.obs_units, source=args.obs)
    thresholds = clim = None
    years = tuple(args.threshold_years) if args.threshold_years else None
    if defn.thresholds:
        if years is None:
            raise ValueError(f"definition '{defn.id}' needs thresholds; give --threshold-years Y0 Y1")
        thresholds = climatology.compute_thresholds(defn, rain, years)
    if isinstance(defn.seasons, DerivedSeasons):
        if years is None:
            raise ValueError(f"definition '{defn.id}' has derived seasons; give --threshold-years Y0 Y1")
        clim = climatology.harmonic_seasons(rain, years)
    anatomy = explain.trace_observed(defn, rain, lat=args.lat, lon=args.lon, year=args.year, thresholds=thresholds,
                                     season_climatology=clim)
    print(explain.explain(anatomy))
    if args.png:
        from hcwf.onset.viz.anatomy import anatomy_plot

        anatomy_plot(anatomy, mode="dark" if args.dark else "light").savefig(args.png, dpi=150)
        print(f"saved {args.png}")
    return 0


def _schematic(args) -> int:
    """Save the schematic of a definition as a PNG.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line: ``definition``, ``png`` and ``dark``.

    Returns
    -------
    int
        The exit code, 0.
    """
    from hcwf.onset.viz.schematic import schematic

    schematic(definition.load(args.definition), mode="dark" if args.dark else "light").savefig(args.png, dpi=150)
    print(f"saved {args.png}")
    return 0


def main(argv=None) -> int:
    """Run the ``hcwf-run onset`` command line.

    An error of the definition, the region or a file is printed on stderr and gives exit status 1.

    Parameters
    ----------
    argv : list of str, optional
        The arguments after ``onset``; ``sys.argv`` when None.

    Returns
    -------
    int
        The exit status: 0, or 1 after an error.
    """
    parser = argparse.ArgumentParser(prog="hcwf-run onset", description=__doc__.replace("``", ""))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="definition ids with titles and regions").set_defaults(run=_list)
    p = sub.add_parser("describe", help="a definition in plain words, with hash, status and notes")
    p.add_argument("definition", help="registry id or path of a YAML file")
    p.add_argument("--no-dry-spell-check", action="store_true", help="show the variant without the 'reject if' vetoes")
    p.set_defaults(run=_describe)
    sub.add_parser("regions", help="region ids with their defaults").set_defaults(run=_regions)
    p = sub.add_parser("check-alignment", help="compare observations and forecasts at day lags -3..+3")
    p.add_argument("--obs", required=True, help="observation NetCDF file")
    p.add_argument("--obs-var", required=True)
    p.add_argument("--obs-day-boundary", required=True, help='e.g. "00 UTC, labelled by start"')
    p.add_argument("--obs-units", default="mm", help="observation units (default mm)")
    p.add_argument("--model", required=True, help="forecast NetCDF file")
    p.add_argument("--model-var", required=True)
    p.add_argument("--units", required=True, help='forecast units, e.g. "m" or "mm"')
    p.add_argument("--lead-label", required=True, choices=("start", "end"))
    p.set_defaults(run=_check_alignment)
    p = sub.add_parser("explain", help="why one cell got its observed onset date: a log and an optional anatomy PNG")
    p.add_argument("--obs", required=True, help="observation NetCDF file")
    p.add_argument("--obs-var", required=True)
    p.add_argument("--obs-units", default="mm", help="observation units (default mm)")
    p.add_argument("--day-boundary", required=True, help='e.g. "00 UTC, labelled by start"')
    p.add_argument("--definition", required=True, help="registry id or path of a YAML file")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--threshold-years", type=int, nargs=2, metavar=("Y0", "Y1"),
                   help="years of the observations that thresholds are computed from")
    p.add_argument("--no-dry-spell-check", action="store_true", help="explain the variant without the 'reject if' vetoes")
    p.add_argument("--png", help="save the anatomy plot here")
    p.add_argument("--dark", action="store_true", help="dark-mode plot")
    p.set_defaults(run=_explain)
    p = sub.add_parser("schematic", help="a picture of what a definition asks for")
    p.add_argument("definition", help="registry id or path of a YAML file")
    p.add_argument("--png", required=True, help="output file")
    p.add_argument("--dark", action="store_true", help="dark-mode plot")
    p.set_defaults(run=_schematic)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (DefinitionError, RegionError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
