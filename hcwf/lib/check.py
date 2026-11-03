"""``hcwf-run -p run.yaml --check``: show everything a run would do and run nothing.

``build_report(dic)`` takes the resolved flat config (`resolve(...).values`) and returns a
``CheckReport``: the settings with where each value came from, the datasets with what was detected and the
evidence, the mask, the lead windows, which files exist or are missing, the sampled alignment of each model with
the observations, sanity hints (each names the key to set), the run folder and the observed-onset cache that a run
would use. Reading is limited to a sample of the files; nothing is written. Errors (a run would fail) make
``CheckReport.ok`` False.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from hcwf.lib.config_schema import has_blend
from hcwf.config.model import climatology_years, evaluation_years
from hcwf.lib.onset_config import onset_setup

log = logging.getLogger(__name__)

_MAX_LISTED = 8
_GRID_TOLERANCE = 1e-3
_SETTING_KEYS = (
    ("region", "region"), ("start_date", "first date"), ("end_date", "last date"), ("init_days", "init weekdays"),
    ("start_year_clim", "climatology from"), ("end_year_clim", "climatology to"),
    ("verification_window_list", "lead windows"), ("tolerance_days_list", "tolerances (days)"),
    ("max_forecast_day", "max forecast day"), ("day_bins", "day bins"), ("probabilistic", "ensemble"),
    ("members", "members"), ("ref_model", "reference"), ("climatology_cv", "climatology_cv"),
    ("onset_percentage_threshold", "ensemble onset share"), ("forecast_history", "forecast history"))


@dataclass
class CheckReport:
    """The outcome of a check: printable lines, the errors (a run would fail) and the warnings (hints)."""

    lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there is no error."""
        return not self.errors

    def text(self) -> str:
        """The whole report as text."""
        return "\n".join(self.lines)

    def section(self, title: str) -> None:
        """Start a titled section."""
        self.lines += ["", f"== {title} =="]

    def add(self, text: str) -> None:
        """Add one line."""
        self.lines.append(text)

    def error(self, text: str) -> None:
        """Record an error (also listed in the summary)."""
        self.errors.append(text)
        self.lines.append(f"ERROR: {text}")

    def hint(self, key: str, text: str) -> None:
        """Record a warning naming the key to set."""
        self.warnings.append(f"[{key}] {text}")
        self.lines.append(f"HINT [{key}]: {text}")


def _short(values, limit=_MAX_LISTED) -> str:
    """A list of values as text, cut after `limit` items."""
    values = list(values)
    text = ", ".join(str(v) for v in values[:limit])
    return text + (f", ... ({len(values)} in all)" if len(values) > limit else "")


def _source(dic: dict, key: str) -> str:
    """Where a flat key's value came from, in brackets."""
    return f"[{(dic.get('config_sources') or {}).get(key, 'resolved')}]"


def _open(path):
    """The lazy Dataset of a NetCDF file or zarr store (a context manager that closes it)."""
    from hcwf.io.input import is_zarr

    return xr.open_zarr(str(path), chunks=None) if is_zarr(path) else xr.open_dataset(path)


def _store_has_year(store, year) -> bool:
    """True if the zarr store has a datetime coordinate value in the year."""
    ds = xr.open_zarr(str(store), chunks=None)
    for coord in ds.coords.values():
        if coord.ndim == 1 and np.issubdtype(coord.dtype, np.datetime64):
            return bool((coord.dt.year == year).any())
    return True


def _find(folder, patterns, year):
    """The path of the yearly file in folder matching one of the patterns, or None.

    A folder that is a zarr store holds all years: the store is returned if it has times in the year.
    """
    from hcwf.io.input import is_zarr

    if is_zarr(folder):
        return Path(str(folder)) if _store_has_year(folder, year) else None
    for pattern in patterns:
        path = Path(str(folder)) / pattern.format(year)
        if path.exists():
            return path
    return None


def _settings(dic: dict, report: CheckReport) -> None:
    """The settings section: the main keys with their sources, then the onset setup lines."""
    report.section("Settings")
    report.add(f"config file     : {dic.get('config_file', '(not from a file)')}")
    for key, label in _SETTING_KEYS:
        if key in dic:
            report.add(f"{label:<16}: {dic[key]}  {_source(dic, key)}")
    report.add(f"models          : {', '.join(dic['model_list'])}")
    for line in onset_setup(dic).lines():
        report.add(line)
    report.add(f"region box      : {_box_text(dic)}")
    mask = dic.get("mask")
    if mask:
        report.add("mask            : " + ", ".join(f"{k}={v}" for k, v in mask.items()) + f"  {_source(dic, 'mask')}")
    elif dic.get("nc_mask"):
        report.add(f"mask            : nc_mask {dic['nc_mask']} (old key; land and country masks also apply)")
    else:
        report.add("mask            : none (the region's box with the land and country masks of the old behaviour)")


def _box_text(dic: dict) -> str:
    """The box of the region a run selects cells with, and which region it is the domain of."""
    from hcwf.onset import regions

    name = dic.get("region")
    if name is None:
        return "none (no region: the whole domain of the data)"
    box = _box(dic)
    if box is None:
        return f"unknown (region {name!r} is not in the registry)"
    return (f"lat {box[0]:g} to {box[1]:g}, lon {box[2]:g} to {box[3]:g}  "
            f"[the domain of region {regions.load_region(str(name).strip()).id}]")


def _datasets(dic: dict, report: CheckReport) -> None:
    """The datasets section: what was detected or given for each dataset, with evidence."""
    report.section("Datasets")
    for line in dic.get("dataset_report") or ("(no datasets described)",):
        report.add(line)
        if line.strip().startswith("warning:"):
            text = line.strip()[len("warning:"):].strip()
            key = next((k for k in ("units", "members", "region") if k in text.split()[:3] or text.startswith(k)),
                       "datasets")
            report.warnings.append(f"[{key}] {text}")
        if "accumulation:" in line and "since_init" in line and "detected" in line:
            report.hint("accumulation", "detected as since_init (running totals); HCWF differences them into daily "
                                        "totals. If the files hold per-interval amounts set models[].accumulation: daily")


def _years(dic: dict) -> tuple[tuple, tuple]:
    """The evaluated years and the climatology years."""
    return evaluation_years(dic), climatology_years(dic)


def _files(dic: dict, report: CheckReport) -> dict:
    """List which yearly files exist; missing files a run needs are errors. Returns the first sample path per dataset."""
    report.section("Files")
    years, clim = _years(dic)
    setup = onset_setup(dic)
    needs_clim = dic.get("ref_model") in ("climatology", "conditional_climatology")
    obs_years = set(years) | (set(clim) if needs_clim else set())
    if setup.uses_thresholds and setup.threshold_source == "observations" and setup.threshold_years and not any(
            True for _ in (*setup.threshold_files, *setup.threshold_constants)):
        obs_years |= set(range(setup.threshold_years[0], setup.threshold_years[1] + 1))
    samples = {}
    if dic.get("obs_paths") or dic.get("model_paths_list"):
        return _listed_files(dic, report)
    patterns = tuple(dic.get("obs_file_pattern") or ("{}.nc",))
    found = {y: _find(dic["obs_dir"], patterns, y) for y in sorted(obs_years)}
    missing = [y for y, p in found.items() if p is None]
    samples["observations"] = next((p for y, p in found.items() if p is not None and y in years), None) \
        or next((p for p in found.values() if p is not None), None)
    report.add(f"observations: {len(found) - len(missing)} of {len(found)} yearly files found in {dic['obs_dir']} "
               f"(years {min(found)}-{max(found)})")
    if missing:
        report.error(f"observation files missing for {_short(missing)} in {dic['obs_dir']} "
                     f"(pattern {', '.join(patterns)})")
    for i, model in enumerate(dic["model_list"]):
        folder, pattern = dic["model_dir_list"][i], dic["file_pattern_list"][i]
        got = {y: _find(folder, (pattern,), y) for y in years}
        gone = [y for y, p in got.items() if p is None]
        samples[model] = next((p for p in got.values() if p is not None), None)
        report.add(f"{model}: {len(got) - len(gone)} of {len(got)} yearly files found in {folder}")
        if gone:
            report.error(f"{model} forecast files missing for {_short(gone)} in {folder} (pattern {pattern})")
    ref = dic.get("ref_model")
    if ref and ref not in ("climatology", "conditional_climatology") and dic.get("ref_model_dir"):
        got = {y: _find(dic["ref_model_dir"], (dic.get("ref_model_file_pattern") or "{}.nc",), y) for y in years}
        gone = [y for y, p in got.items() if p is None]
        report.add(f"reference {ref}: {len(got) - len(gone)} of {len(got)} yearly files found")
        if gone:
            report.error(f"reference {ref} files missing for {_short(gone)}")
    return samples


def _listed_files(dic: dict, report: CheckReport) -> dict:
    """The files of datasets given as lists (`paths`): how many exist, over which years; a missing one is an error."""
    from hcwf.lib.dataset_schema import file_year

    samples = {}

    def one(label, files):
        files = [Path(f) for f in files]
        gone = [f for f in files if not f.exists()]
        years = sorted({file_year(f) for f in files} - {None})
        span = f", years {years[0]}-{years[-1]}" if years else ""
        report.add(f"{label}: {len(files) - len(gone)} of {len(files)} listed file(s) found{span}")
        if gone:
            report.error(f"{label} files missing: {_short(gone)}")
        samples[label] = next((f for f in files if f.exists()), None)

    if dic.get("obs_paths"):
        one("observations", dic["obs_paths"])
    for i, model in enumerate(dic["model_list"]):
        listed = (dic.get("model_paths_list") or [()] * (i + 1))[i]
        if listed:
            one(model, listed)
    return samples


def _grid(path, dims=None, forecast=False):
    """(lat, lon, variable-lat-order) of the first file: the coordinate arrays after the usual renames."""
    from hcwf.utils.standard import dim_fmt, dim_fmt_model, rename_model_dims

    with _open(path) as ds:
        ds = dim_fmt_model(ds, dims) if forecast else dim_fmt(rename_model_dims(ds, dims))
        return np.asarray(ds["lat"].values, float), np.asarray(ds["lon"].values, float)


def _box(dic: dict):
    """The region's lat/lon box (lats, latn, lonw, lone), or None when the region is unknown."""
    from hcwf.params.region_def import domain

    try:
        return domain(dic["region"])
    except Exception:  # noqa: BLE001 - reported by the region checks
        return None


def _in_box(lat, lon, box):
    """Latitudes and longitudes (0-360 normalised) inside the box."""
    if box is None:
        return lat, lon % 360
    lats, latn, lonw, lone = box
    lon = lon % 360
    lat = lat[(lat >= lats) & (lat <= latn)]
    lo, hi = lonw % 360, lone % 360
    lon = lon[(lon >= lo) & (lon <= hi)] if lo <= hi else lon[(lon >= lo) | (lon <= hi)]
    return np.sort(lat), np.sort(lon)


def _matched(values, targets) -> float:
    """The share of values that have a target within the grid tolerance."""
    if len(values) == 0:
        return 1.0
    if len(targets) == 0:
        return 0.0
    nearest = np.abs(values[:, None] - targets[None, :]).min(axis=1)
    return float((nearest <= _GRID_TOLERANCE).mean())


def _grids(dic: dict, report: CheckReport, samples: dict) -> None:
    """Grid hints: latitude order and obs/model grid mismatches inside the region box."""
    report.section("Grids")
    box = _box(dic)
    obs_path = samples.get("observations")
    if obs_path is None:
        report.add("(no observation file to read)")
        return
    try:
        obs_lat, obs_lon = _grid(obs_path, (dic.get("obs_dims") or None))
    except Exception:  # noqa: BLE001 - observations on evaluation units (a "unit" dimension) have no grid
        report.add("(the observations are not on a lat/lon grid: evaluation units; see Units)")
        return
    obs_in = _in_box(obs_lat, obs_lon, box)
    report.add(f"observations: {len(obs_lat)} lats x {len(obs_lon)} lons in the file, {len(obs_in[0])} x "
               f"{len(obs_in[1])} inside the region box")
    if len(obs_lat) > 1 and obs_lat[0] > obs_lat[-1]:
        report.hint("observations", "latitudes run north to south; HCWF flips them to south to north, nothing to set")
    if len(obs_in[0]) == 0 or len(obs_in[1]) == 0:
        report.error(f"the observations have no cell inside the region box of '{dic['region']}'. "
                     "Fix: set region, or check the longitude convention")
    for i, model in enumerate(dic["model_list"]):
        path = samples.get(model)
        if path is None:
            continue
        dims = (dic.get("model_dims_list") or [None] * len(dic["model_list"]))[i]
        try:
            lat, lon = _grid(path, dims, forecast=True)
        except Exception as exc:  # noqa: BLE001 - reported as a hint; the datasets section has the detail
            report.hint(model, f"could not read the grid of {path.name}: {exc}")
            continue
        if len(lat) > 1 and lat[0] > lat[-1]:
            report.hint(model, "latitudes run north to south; HCWF flips them to south to north, nothing to set")
        inside = _in_box(lat, lon, box)
        report.add(f"{model}: {len(lat)} lats x {len(lon)} lons in the file, {len(inside[0])} x {len(inside[1])} "
                   "inside the region box")
        share = _matched(inside[0], obs_in[0]) * _matched(inside[1], obs_in[1])
        if share < 1.0:
            report.hint(model, f"only {share:.0%} of the model's cells have an observation cell within "
                               f"{_GRID_TOLERANCE:g} degrees, so the others are not scored (the data are never "
                               "regridded). Fix: put forecasts and observations on one grid, e.g. with cdo remapcon")


def _mask_summary(dic: dict, report: CheckReport, samples: dict) -> None:
    """The mask section: cells kept on the observations' grid (the mask is built on a sample file)."""
    mask, path = dic.get("mask"), samples.get("observations")
    if not mask or path is None:
        return
    report.section("Mask")
    from hcwf.config.model import AreaSpec
    from hcwf.utils import mask as mask_mod
    from hcwf.utils.region import region_select
    from hcwf.utils.standard import dim_fmt

    mask_mod.reset()
    try:
        with _open(path) as ds:
            ds = dim_fmt(ds).isel(time=slice(0, 1))
            area = AreaSpec(region=dic["region"], mask=mask_mod.MaskSpec.from_mapping(mask), nc_mask=dic.get("nc_mask"),
                            shpfile_dir=None, polygon=False)
            region_select(ds, area=area, dataset_name="obs", dir_fig=None)
        for record in mask_mod.records().values():
            for line in mask_mod.record_lines(record):
                report.add(line)
    except Exception as exc:  # noqa: BLE001 - the mask failing is an error a run would hit too
        report.error(f"the mask could not be built: {exc}")
    finally:
        mask_mod.reset()


def _units(dic: dict, report: CheckReport, found) -> None:
    """The evaluation units each source is read onto (the `units:` section and the region's): mode, tables, restriction;
    an invalid section (a problem of hcwf.config.validate) and a named file that is not there are errors."""
    if found.units:
        report.section("Units")
        for text in found.units:
            report.error(text)
        return
    specs = found.specs
    if all(spec is None for spec in specs.values()):
        return
    report.section("Units")
    for name, spec in specs.items():
        if spec is None:
            report.add(f"{name}: grid cells")
            continue
        parts = [f"{spec.how}"]
        for label in ("shapefile", "weights", "centroids", "dissemination", "outline"):
            value = getattr(spec, label)
            if value:
                where = spec.resolve_path(value)
                parts.append(f"{label} {where}")
                if not where.is_file():
                    report.error(f"{name}: the units' {label} file is missing: {where}")
        if spec.unit_id_col:
            parts.append(f"id column {spec.unit_id_col}")
        if spec.box:
            parts.append("box " + ", ".join(f"{k} {v:g}" for k, v in spec.box))
        if spec.dissemination:
            parts.append(f"dissemination scope {spec.dissemination_scope}")
        if spec.how == "cells" or spec.id_digits is not None:
            parts.append(f"grid ids {spec.id_format}" + (f" with {spec.id_digits} decimals" if spec.id_digits is not None else ""))
        report.add(f"{name}: " + "; ".join(parts))


def _blend(dic: dict, report: CheckReport, found) -> None:
    """The `blend:` section as the run reads it: models, years, bins, formulas, scoring, and the artifacts' folder; every
    problem of the section (and the rules that need the onset definition) is an error."""
    report.section("Blend")
    for text in found.blend:
        report.error(text)
    recipe = found.recipe
    if recipe is None:
        return
    ev, ft = recipe.evaluation, recipe.features
    report.add(f"blend           : {recipe.name}; outcomes {', '.join(recipe.outcome_labels())}; "
               f"{ft.count} bins of {ft.days} days from {ft.probability_days} probability days")
    for model in recipe.models:
        horizon = model.rain_horizon
        predictors = ", ".join(f"{p.agg}({p.window})" for p in model.rain_predictors) or "none"
        member = "" if model.max_member is None else f", members up to {model.max_member}"
        report.add(f"  {model.name:<14}: leads {model.lead_days[0]}-{model.lead_days[1]}{member}; rain horizon "
                   f"{f'to day {horizon.max_day}, ' if horizon.max_day is not None else ''}{horizon.resolved_policy}; "
                   f"rain predictors {predictors}")
    report.add(f"years           : training {_short(ev.years.training)}; cv holdout {_short(ev.years.cv_holdout)}; "
               f"true holdout {_short(ev.years.true_holdout) or 'none'}")
    report.add(f"cutoff / cv     : {ev.cutoff_mode}; {', '.join(ev.cv_methods)}; fixed cutoff {recipe.onset.fixed_cutoff}"
               + (f"; reference onset {ref.month_day or ref.file}" if (ref := recipe.onset.reference) else ""))
    for name, formula in ev.formulas:
        mark = "" if formula.enabled else " (disabled)"
        final = " (final)" if name == recipe.final_formula() else ""
        report.add(f"formula {name}{final}{mark}: {formula.text}")
    report.add(f"scored          : {', '.join(recipe.climatology_baseline_names())}"
               + (f"; forest {ev.random_forest.name}" if ev.random_forest else "")
               + ("; MME" if ev.mme.enabled else "") + f"; output tag '{recipe.output_tag()}'")
    clim = recipe.climatology
    report.add(f"climatology     : issues to {clim.issue_end}, {recipe.climatology_forecast_window()} lead days, "
               f"{'leave-one-year-out' if clim.cv_by_year else 'no CV'}, at least {clim.min_onset_years} onset years")
    paths = dic.get("run_paths")
    if paths is not None:
        report.add(f"artifacts       : {paths.blend}")
    report.add("note            : a run with a blend section trains the blend; the benchmark of the other sections "
               "is not run")


def _leads(dic: dict, report: CheckReport, samples: dict) -> None:
    """Lead windows and the leads of each model's files; leads that are not whole days are hinted."""
    report.section("Lead windows")
    for line in dic.get("onset_lead_lines") or ():
        report.add(line)
    report.add(f"windows {dic['verification_window_list']} with tolerances {dic['tolerance_days_list']} days; "
               f"max forecast day {dic['max_forecast_day']}; day bins {dic['day_bins']}")
    from hcwf.onset import ingest

    for i, model in enumerate(dic["model_list"]):
        path = samples.get(model)
        if path is None:
            continue
        try:
            with _open(path) as ds:
                names = [d for d in ds.dims if str(d).lower() in ("step", "lead", "prediction_timedelta", "day",
                                                                  "prediction_timedelta_daily")]
                lead = (dic.get("model_dims_list") or [None] * 99)[i]
                lead = (lead or {}).get("step") or (names[0] if names else None)
                if lead is None:
                    continue
                hours = ingest.lead_hours(ds, lead)
        except Exception as exc:  # noqa: BLE001
            report.add(f"{model}: leads could not be read ({exc})")
            continue
        positive = hours[hours > 0]
        if len(positive) == 0:
            continue
        dt, resolution = ingest._resolution(positive)
        report.add(f"{model}: {len(positive)} leads, {resolution} steps, last at {positive.max() / 24:g} days")
        whole = np.allclose(positive / 24.0, np.rint(positive / 24.0)) and abs(dt - 24.0) < 1e-6
        if not whole:
            report.hint(model, f"the leads are not whole days ({resolution} steps: "
                               f"{_short(np.round(positive[:6], 3).tolist(), 6)} h). HCWF reads lead steps as whole "
                               "days, so aggregate the forecasts to daily totals first (or set "
                               "models[].daily_boundary for days cut at a UTC hour)")


def _alignment(dic: dict, report: CheckReport) -> None:
    """The alignment of each model with the observations: a sampled check, run in parallel over the models.

    A few dozen inits spread over the period and a few thousand cells are read once per dataset and every lag is
    correlated in one step (hcwf.lib.align_sample); the verdict is cached with a fingerprint of both datasets.
    """
    report.section("Alignment")
    setup = onset_setup(dic)
    if not setup.check_alignment:
        report.add("onset.check_alignment is false: skipped")
        return
    from hcwf.config.model import RunConfig
    from hcwf.lib import align_sample
    from hcwf.lib.onset_config import FORECAST_BOUNDARY

    jobs = {}
    rc = RunConfig.from_flat(dic, check=False)
    for model in dic["model_list"]:
        try:
            kw = align_sample.sample_spec(rc, model)
        except Exception as exc:  # noqa: BLE001
            report.add(f"{model}: skipped ({exc})")
            continue
        jobs[model] = (kw, (setup.lead_label,), {"obs_boundary": setup.obs_day_boundary, "source": model,
                                                 "forecast_boundary_format": FORECAST_BOUNDARY})
    for model, result in align_sample.check_models(jobs).items():
        if isinstance(result, Exception):
            report.add(f"{model}: skipped ({result})")
            continue
        (result, cached), = result.values()
        n_inits = next((r.n_overlap_days for r in result.per_lead.values()), 0)
        n_cells = next((r.n_cells for r in result.per_lead.values()), 0)
        report.add(f"{model} (sample of {n_inits} inits x {n_cells} cells){' (cached)' if cached else ''}: "
                   + result.summary().replace("\n", "\n    "))
        if str(result.verdict).startswith("likely shifted"):
            report.hint("lead_label", f"{model} looks shifted against the observations ({result.verdict}); check "
                                      "onset.lead_label and observations.day_boundary")


def _short_climatology(dic: dict, report: CheckReport) -> None:
    """Hint when the climatology period is short."""
    n = len(climatology_years(dic))
    needs = 10 if dic.get("ref_model") == "conditional_climatology" else 15
    if n < needs:
        report.hint("evaluation.start_year_clim", f"the climatology has only {n} years "
                    f"({dic['start_year_clim']}-{dic['end_year_clim']}); the climatology reference and "
                    f"thresholds need about {needs} or more. Set start_year_clim and end_year_clim")
    setup = onset_setup(dic)
    if setup.threshold_years and setup.uses_thresholds:
        m = setup.threshold_years[1] - setup.threshold_years[0] + 1
        if m < 10:
            report.hint("onset.threshold_years", f"the thresholds come from only {m} years "
                        f"({setup.threshold_years[0]}-{setup.threshold_years[1]})")


def _run_folder(dic: dict, report: CheckReport, overwrite: bool = False) -> None:
    """Where the run writes, and whether it would be refused."""
    report.section("Run folder")
    paths = dic.get("run_paths")
    if paths is None:
        report.add("(no run folder planned: this config is not from a file)")
        return
    if paths.explicit:
        report.add(f"explicit data_dir: {paths.dir_out} (figures {paths.dir_fig}); files are overwritten")
    else:
        report.add(f"run folder: {paths.root}")
        report.add(f"  scores {paths.dir_out}; figures {paths.dir_fig}; onsets {paths.onsets}; cache {paths.cache_dir}")
        if paths.exists and overwrite:
            report.add("  exists: --overwrite is given, so a run would clear it first")
        elif paths.exists:
            report.error(f"this config already ran: {paths.root}. A run is refused unless --overwrite is given "
                         "(which clears that folder first)")
        else:
            report.add("  new: a run would create it")
    report.add(f"log: {paths.log_file}")


def _cache(dic: dict, report: CheckReport) -> None:
    """The observed-onset cache a run would use and which entries exist."""
    report.section("Cache")
    from hcwf.stats import observed_cache

    paths = dic.get("run_paths")
    folder = Path(dic["cache_dir"]) if dic.get("cache_dir") else Path(dic["dir_out"]) / ".onset_cache"
    entries = sorted((folder / observed_cache.SUBFOLDER).glob("obs-*")) if folder.is_dir() else []
    report.add(f"cache folder: {folder} ({len(entries)} observed-onset entries)")
    if has_blend(dic):
        report.add("observed onsets: a blend run detects them itself (search starts, thresholds and units are its own)")
        return
    setup = onset_setup(dic)
    if paths is None:
        return
    if setup.uses_thresholds and not setup.threshold_files and not setup.threshold_constants:
        report.add("observed onsets: the thresholds are computed from the observations (or cached) when the run "
                   "starts, so hit or miss is reported by the run; an earlier run with the same observations, "
                   "definition, cells and years is a hit")
        return
    try:
        from dataclasses import replace

        from hcwf.config.model import RunConfig
        from hcwf.io.input import observations_of, thresholds_of

        rc = RunConfig.from_flat(dic, check=False)
        read = replace(rc, output=replace(rc.output, figure_dir=None))   # a check draws no mask figure
        obs = rc.observations
        year = next(y for y in rc.evaluation.years if _find(obs.dir, tuple(obs.file_pattern), y))
        rain = observations_of(year, read)
        parts = observed_cache.parts_for(rain, thresholds_of(read, rc.models[0]), rc)
        fp = observed_cache.fingerprint(parts)
        if (folder / observed_cache.SUBFOLDER / fp / "onsets.nc").exists():
            report.add(f"observed onsets: expected cache hit {fp}")
        else:
            report.add(f"observed onsets: expected cache miss {fp} (detected once, then shared by all models, "
                       "windows and the reference)")
    except Exception as exc:  # noqa: BLE001
        report.add(f"observed onsets: expectation not available ({exc})")


def build_report(rc, found=None, overwrite: bool = False, alignment: bool = False) -> CheckReport:
    """Check a resolved RunConfig (see the module text); reads samples of the files and writes nothing.

    found is the hcwf.config.validate.Problems of the config (resolve(check=True) returns them); they are computed when
    not given. overwrite says whether --overwrite is given (an existing run folder is then not an error). alignment
    runs the alignment check once (`--check --alignment`).
    """
    dic = dict(rc.values)   # the report reads the flat record (it also edits a copy: check_alignment)
    if alignment:
        dic["check_alignment"] = True
    if found is None:
        from hcwf.config.validate import problems

        found = problems(dic)
    report = CheckReport()
    report.add(f"HCWF check of {dic.get('project_name', 'run')}: nothing will be run")
    _settings(dic, report)
    _datasets(dic, report)
    samples = _files(dic, report)
    _units(dic, report, found)
    _grids(dic, report, samples)
    _mask_summary(dic, report, samples)
    _leads(dic, report, samples)
    _alignment(dic, report)
    _short_climatology(dic, report)
    if has_blend(dic):
        _blend(dic, report, found)
    _run_folder(dic, report, overwrite)
    _cache(dic, report)
    report.section("Summary")
    report.add(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
               + ("" if report.errors else "; the run would start"))
    for text in report.errors:
        report.add(f"  ERROR {text}")
    for text in report.warnings:
        report.add(f"  HINT {text}")
    return report
