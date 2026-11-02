"""TEMPORARY TEST SCAFFOLDING: onset_blending's BlendConfig -> (ROMP flat config dict, Recipe).

The split. Keys ROMP's config already has stay ROMP keys: observations.files/variable/unit conversion/date marks ->
obs_paths, obs_var, obs_unit_cvt, obs_day_boundary; observations.missing_rain -> missing; the onset rule ->
onset_rule (+ onset_season, onset_backend, threshold_years); the thresholds -> thresholds (a number, a table file),
threshold_table, threshold_scale, threshold_max_missing, threshold_partial_years, and for computed ones the
threshold's days and `during` in the rule; the units (domain: box or outline, dissemination, centroids, grid ids, regrid,
cell transforms) -> the `units_*` keys, per source where weights differ; models[].files/variable/dims/unit_scale ->
model_list and the model_*_list keys (model_paths_list: the files of each model); lead_days.max ->
max_forecast_day; the forecast horizon -> onset_horizon; the parity options -> rain_resolution, negatives, gaps;
years -> years. What ROMP has no key for goes to the Recipe: models[].rain_*, max_member, lead days, fixed_cutoff,
reference_onset, climatology, features, evaluation, outputs.

The real converter (`momp-run convert`, a later step) maps old BlendConfig YAML onto ROMP's run YAML, and the run YAML
onto a Recipe. Until then the parity tests run blending's own configs (configs/*.yml, tests/_configs.py) through momp by
loading them with the onset_blending package itself and translating the loaded `BlendConfig` here. Replace this module
by the converter's tests when it exists.

The onset rule follows docs/blend_merge/overlap_onset.md section 1 (`definition_yaml`: a blending `OnsetRule` as a momp
definition text); `definition_yaml` is also what the engine parity tests build their definitions with.

onset_blending is imported from its checkout (ONSET_BLENDING_SRC, default /home/hcwf/code/onset_blending), which is
never modified; it only needs the packages ROMP's environment already has.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from momp.blend import recipe as R
from momp.blend.design import Formula, WindowVariant
from momp.blend.random_forest import Params as ForestParams
from momp.onset import definition as D
from momp.stats.bins import LeadBins

ORACLE_SRC = Path(os.environ.get("ONSET_BLENDING_SRC", "/home/hcwf/code/onset_blending"))

HEAD = """id: parity
version: 1
title: t
citation: t
wording: t
status: draft
notes: t
season: {from: 01-01, to: 12-31}
"""


def head(season_start: str = "01-01") -> str:
    return HEAD.replace("from: 01-01", f"from: {season_start}")


def definition_yaml(r, fixed_total: float | None = None, season_start: str = "01-01", threshold: tuple | None = None) -> str:
    """The momp definition text of a blending rule (a fixed total amount instead of the thr threshold if given; the
    thr threshold over `threshold` = (days, first month-day, last month-day) instead of the trigger window)."""
    wet = f"{'at least' if r.wet_day_comparison == 'gte' else 'more than'} {r.wet_day_min_mm} mm"
    cmp_ = "at least" if r.threshold_comparison == "gte" else "more than"
    total = f"{cmp_} {fixed_total} mm" if fixed_total is not None else f"{cmp_} thr"
    require = ["{all days: wet}"] if r.trigger_rule == "all_days_wet" else ["{first day: wet}"]
    if r.min_wet_days is not None:
        require.append(f"{{wet days: at least {r.min_wet_days}}}")
    require.append(f"{{total: {total}}}")
    if r.followup_anchor == "after_trigger":
        span = f"{r.follow_days} days after window"
    else:
        span = f"days 1 to {r.follow_days + 1}"
    length = r.min_dry_days if r.mode == "consecutive_dry" else r.sum_window
    contained = r.mode == "window_sum" or r.dry_run_contained
    if r.follow_days == 0 or (contained and r.follow_days < length):
        veto = "[]"
    elif r.mode == "window_sum":
        veto = f"[{{dry window: {r.sum_window} days with less than {r.sum_min_mm} mm, within: {span}}}]"
    else:
        place = "within" if r.dry_run_contained else "beginning within"
        veto = f"[{{dry run: at least {r.min_dry_days} days, {place}: {span}}}]"
    if threshold is None:
        thresholds = "thresholds: {thr: {mean total over: %d days}}\n" % r.window
    else:
        thresholds = "thresholds: {thr: {mean total over: %d days, during: %s to %s}}\n" % threshold
    return (head(season_start) + f"wet_day: {wet}\ndry_day: less than {r.dry_day_min_mm} mm\n{thresholds}"
            f"onset:\n  window: {r.window} days\n  require: [{', '.join(require)}]\n  reject if: {veto}\n")


# -- loading blending's configs ---------------------------------------------------------------------------------------


def oracle():
    """The onset_blending package (from its checkout), imported once."""
    if str(ORACLE_SRC) not in sys.path:
        sys.path.insert(0, str(ORACLE_SRC))
    import onset_blending

    return onset_blending


def load_blending_config(path):
    """A BlendConfig from a YAML file, with blending's own loader."""
    oracle()
    from onset_blending.config import load_config

    return load_config(path)


def blending_config(data: dict):
    """A BlendConfig from a dict, with blending's own loader."""
    oracle()
    from onset_blending.config import BlendConfig

    return BlendConfig.from_dict(data)


# -- the translation --------------------------------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402


@dataclass
class Translation:
    """What a BlendConfig stands for: ROMP's flat config dict, the blend Recipe, and the resolved OnsetSetup."""

    cfg: dict
    recipe: R.Recipe
    setup: object


def _files(files, config, what: str, strict: bool = True) -> list[str]:
    """The files of a blending FileSet in blending's order; with strict=False a configuration whose data is not here
    (the shipped configs) lists none."""
    from onset_blending.core import io

    try:
        return [str(p) for p in io.list_files(files, config, what)]
    except FileNotFoundError:
        if strict:
            raise
        return []


def _dims(rename: dict, roles: dict) -> dict | None:
    """blending's {file dimension: role} as momp's {canonical: file dimension}."""
    out = {roles[role]: name for name, role in rename.items() if role in roles}
    return out or None


_OBS_ROLES = {"time": "time", "lat": "lat", "lon": "lon", "adm3_name": "unit"}
_MODEL_ROLES = {"time": "init_time", "day": "step", "number": "member", "lat": "lat", "lon": "lon", "adm3_name": "unit"}


def _units(config) -> dict:
    """The `units_*` keys of a BlendConfig: the domain's box or outline, dissemination, centroids and grid ids for every
    source, and for each source how its cells become the units (observations.cell_transform, models[].cell_transform,
    domain.regrid and the weights each names)."""
    domain = config.domain
    out = {"units_id_digits": domain.grid_ids.decimal_digits, "units_id_format": domain.grid_ids.format,
           "units_dissemination_scope": domain.dissemination_scope}
    area = domain.area_of_interest
    if area is not None and area.bbox is not None:
        out["units_box"] = {k: float(v) for k, v in vars(area.bbox).items() if v is not None}
    elif area is not None:
        out["units_outline"] = str(config.path(area.geometry))
    if domain.dissemination is not None:
        out.update(units_dissemination=str(config.path(domain.dissemination.file)),
                   units_dissemination_id_col=domain.dissemination.id_col)
    if domain.centroids is not None:
        out.update(units_centroids=str(config.path(domain.centroids.file)),
                   units_centroid_id_col=domain.centroids.id_col)
    regrid = domain.regrid
    if regrid is not None and regrid.shapefile is not None:
        out.update(units_shapefile=str(config.path(regrid.shapefile)), units_unit_id_col=regrid.unit_id_col,
                   units_unit_name_col=regrid.unit_name_col, units_parent_col=regrid.parent_id_col)
    modes, weights = {}, {}

    def source(name, transform, model=None):
        if transform is not None:
            modes[name], weights[name] = "cell_transform", str(config.path(transform))
        elif regrid is None:
            modes[name] = "cells"
        else:
            modes[name] = "regrid"
            supplied = (model.regrid_weights if model is not None else None) or regrid.weights
            if supplied is not None:
                weights[name] = str(config.path(supplied))

    source("observations", config.observations.cell_transform)
    for m in config.models:
        source(m.name, m.cell_transform, m)
    return {**out, "units_mode": modes, "units_weights": weights or None}


def _threshold(config) -> dict:
    """The threshold keys of a BlendConfig: the `thresholds` mapping (a number or a table file), how a table is read, the
    scale, and the options of a computed threshold (blending counts a year only partly in the data and never drops a
    unit for missing days)."""
    threshold = config.onset.threshold
    out = {"thresholds": None, "threshold_table": None, "threshold_scale": None, "threshold_max_missing": None,
           "threshold_partial_years": None}
    if threshold.scale is not None:
        s = threshold.scale
        out["threshold_scale"] = {"factor": s.factor, "offset": s.offset, "min": s.min, "max": s.max}
    if threshold.mm is not None:
        out["thresholds"] = {"thr": float(threshold.mm)}
    elif threshold.climatology is not None:
        out.update(threshold_max_missing=1.0, threshold_partial_years=True)
    else:
        out["thresholds"] = {"thr": str(config.path(threshold.file))}
        out["threshold_table"] = {name: getattr(threshold, name) for name in (
            "id_col", "value_col", "lat_col", "lon_col", "lat_var", "lon_var", "value_var", "mat_lat_name",
            "mat_lon_name", "mat_value_name")}
    return out


def _reference(config) -> R.ReferenceSpec | None:
    ref = config.onset.reference_onset
    if ref is None:
        return None
    if ref.month_day is not None:
        return R.ReferenceSpec(month_day=str(ref.month_day))
    return R.ReferenceSpec(file=str(config.path(ref.file)), date_col=ref.date_col, year_col=ref.year_col,
                           day_col=ref.day_col, base_date=ref.base_date, unit_col=ref.unit_col)


def _rule(config) -> dict:
    import yaml

    from onset_blending.core.onset_rule import OnsetRule

    c = config.onset.threshold.climatology
    threshold = None if c is None else (c.days if c.days is not None else config.onset.definition.window, c.start, c.end)
    text = definition_yaml(OnsetRule.of(config.onset.definition), season_start=config.onset.season_start,
                           threshold=threshold)
    return yaml.safe_load(text)


def translate(config, strict: bool = True, cache_dir=None) -> Translation:
    """The (ROMP config dict, Recipe, OnsetSetup) a BlendConfig stands for (strict=False: its data need not exist).
    Computed thresholds are kept in cache_dir (a new temporary folder without one)."""
    import tempfile

    from onset_blending import config as cfg_mod

    from momp.lib.onset_config import onset_setup

    oracle()
    obs, onset, ev = config.observations, config.onset, config.evaluation
    computed = onset.threshold.climatology
    years = tuple(computed.years) if computed else (min(ev.years.training), max(ev.years.training))
    cache_dir = str(cache_dir or tempfile.mkdtemp(prefix="blend-cache-"))
    cfg = {
        "obs_paths": _files(obs.files, config, "observations", strict), "obs_var": obs.variable,
        "obs_unit_cvt": None, "obs_dims": _dims(obs.dims, _OBS_ROLES),
        "obs_day_boundary": f"00 UTC, labelled by {obs.date_marks}", "missing": str(obs.missing_rain).lower(),
        "onset_legacy": False, "model_lead_label": "end", "forecast_history": "none", "onset_backend": "numba", "onset_rule": _rule(config),
        "threshold_years": list(years), "rain_resolution": "exact", "negatives": "keep", "gaps": "refuse",
        "onset_horizon": "truncate", "years": sorted(set(ev.years.training) | set(ev.years.holdout)),
        "max_forecast_day": max(m.lead_days.max for m in config.models),
        "model_list": [m.name for m in config.models],
        "model_var_list": [m.variable for m in config.models],
        "unit_cvt_list": [None if m.unit_scale == 1.0 else m.unit_scale for m in config.models],
        "model_dims_list": [_dims(m.dims, _MODEL_ROLES) for m in config.models],
        "model_paths_list": [_files(m.files, config, f"model {m.name}", strict) for m in config.models],
        "region": None, "dir_out": cache_dir, "cache_dir": cache_dir, **_threshold(config), **_units(config),
    }
    setup = onset_setup(cfg)
    definition = setup.definition
    momp_windows = definition.windows()
    token_key = {"trigger": "trigger", "window": "trigger", "dry_spell": "dry_window", "sum_window": "dry_window",
                 "follow": "follow", "min_dry": "dry_run"}

    def predictor(p):
        days = cfg_mod.resolve_window(p, onset.definition)
        if isinstance(p.window, str) and momp_windows[token_key[p.window.lower()]] == days:
            return R.RainPredictor(p.agg, p.window)
        return R.RainPredictor(p.agg, int(days))

    models = tuple(
        R.Model(name=m.name, lead_days=(m.lead_days.min, m.lead_days.max), max_member=m.max_member,
                rain_horizon=R.RainHorizon(m.rain_horizon.max_day, m.rain_horizon.policy),
                rain_predictors=tuple(predictor(p) for p in m.rain_predictors), rain_transform=m.rain_transform)
        for m in config.models)
    clim = config.climatology
    climatology = R.ClimatologySpec(
        issue_end=clim.issue_end, forecast_window=clim.forecast_window,
        horizons=tuple((h.start, h.end, h.forecast_window) for h in clim.horizons), cv_by_year=clim.cv_by_year,
        min_onset_years=clim.min_onset_years, train_years=clim.train_years, test_years=clim.test_years,
        windows=tuple(tuple(w) for w in clim.windows), workers=clim.workers)
    features = R.Features(
        bins=LeadBins.uniform(config.features.bins.count, config.features.bins.days),
        probability_days=config.features.probability_days,
        trim_after_true_onset=config.features.trim_after_true_onset, rain_transform=config.features.rain_transform)
    forest = None
    if ev.random_forest is not None:
        f = ev.random_forest
        forest = ForestParams(
            name=f.name, trees=f.trees, min_samples_leaf=f.min_samples_leaf,
            max_features=None if f.max_features == "all" else f.max_features, max_samples=f.max_samples,
            max_depth=f.max_depth, random_state=f.random_state, max_day=f.max_day,
            days_per_week=config.features.bins.days, missing_onset=f.missing_onset, years_per_fold=f.years_per_fold)
    evaluation = R.Evaluation(
        years=R.Years(ev.years.training, ev.years.cv_holdout, ev.years.true_holdout),
        cutoff_mode=ev.cutoff_mode, cv_methods=tuple(ev.cv_methods), cluster_var=ev.cluster_var,
        formulas=tuple((name, Formula(text=f.text, enabled=f.enabled)) for name, f in ev.formulas.items()),
        final_formula=ev.final_formula,
        window_variants=tuple(WindowVariant(base=v.base, windows=tuple(tuple(w) for w in v.windows), replace=v.replace,
                                            only_if_cutoff_mode=v.only_if_cutoff_mode) for v in ev.window_variants),
        baselines=R.Baselines(ev.climatology_baselines.unconditional, ev.climatology_baselines.conditional,
                              tuple(tuple(w) for w in ev.climatology_baselines.windows)),
        forecasts=None if ev.forecasts is None else tuple(
            R.ForecastScoring(f.model, f.variant, f.raw, f.calibrated, f.platt_weights) for f in ev.forecasts),
        mme=R.Mme(ev.mme.enabled, tuple(ev.mme.variants),
                  tuple(R.MmeSource(s.name, s.source, s.cal_variant) for s in ev.mme.sources), ev.mme.optimize,
                  ev.mme.weights_year_tag),
        external_predictions=tuple(R.ExternalPrediction(e.name, e.file, e.method) for e in ev.external_predictions),
        random_forest=forest)
    recipe = R.Recipe(
        name=config.name, models=models,
        onset=R.OnsetExtras(fixed_cutoff=onset.fixed_cutoff, reference=_reference(config)),
        climatology=climatology, features=features, evaluation=evaluation,
        outputs=R.Outputs(config.outputs.dir, tuple(config.outputs.intermediates), config.outputs.pickles))
    recipe.check(definition)
    return Translation(cfg, recipe, setup)
