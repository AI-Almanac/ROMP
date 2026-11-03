"""NetCDF export of the gridded onset dates an HCWF run scores: what is collected here is written by hcwf.io.output.

    from hcwf.runs.export import export_onsets
    paths = export_onsets(cfg, sources=["observed", "AIFS"], out="onsets")   # {source: Path}

or ``hcwf-run export -p run.yaml [--sources observed,AIFS,climatology] [--out DIR]``, or ``output.save_onsets:
true`` in the run file (files go to the run's onsets/ folder, dir_out/onsets/ with an explicit dir_out, before scoring).

One file per source, each over all evaluated years: ``onsets_observed.nc``, ``onsets_<model>.nc`` for every
model, ``onsets_climatology.nc`` and ``onsets_conditional_climatology.nc``. The values are not re-detected
with other settings: every number comes from the functions the scoring code calls (hcwf.onset.detect,
hcwf.references.climatology, hcwf.references.conditional), with the keyword arguments the scoring builds for
the same case (Setting and Case dataclasses), so masks, lead window (after shortening), forecast history,
thresholds and definition are the scored ones. Unlike the scoring tables, which keep only the cases with an
observed onset and an init before it, the files cover every init and cell; the flag ``scored`` marks the cases
the scoring keeps.

Dates are CF time variables (int32 days since 1970-01-01 in the data's calendar, cftime for non-standard
calendars, the fill value meaning no onset); day numbers and leads are float32 with NaN for none. Every file
has CF-1.8 global attributes holding the definition, thresholds, masks, forecast history, lead window, data
paths, versions and the resolved run config (see docs/onset_in_hcwf.md, "Exporting onset dates").
"""
from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import run_scope
from hcwf.io.forecasts import forecast_of, initialization_dates
from hcwf.io.observations import history_of, observations_of
from hcwf.io.thresholds import thresholds_of
from hcwf.config.model import AreaSpec, Case, RunConfig
from hcwf.runs.folder import file_name
from hcwf.runs.resolved import resolved_config_text_of
from hcwf.io.output import (EPOCH, coord_attrs, date_var, day_of_year, days_of, float_var, json_text, lead_comment,
                            netcdf_attrs, standard_calendar, status_var, with_unit_labels, write_netcdf)
from hcwf.spatial import axes
from hcwf.onset import definition as definition_mod
from hcwf.onset.describe import describe
from hcwf.onset.model import Status
from hcwf.references.climatology import climatological_onset, climatological_onset_dataset
from hcwf.references.climatology import compute_climatology_as_forecast
from hcwf.onset.observed_cache import threshold_hash
from hcwf.references.conditional import (CONDITIONAL_CLIMATOLOGY, cv_mode,
                                                compute_conditional_climatology_as_forecast,
                                                conditional_lead_probabilities)
from hcwf.onset.detect import (_ensemble_status, detect_forecast, detect_observed, detect_observed_dataset,
                               ensemble_onset)
import logging

log = logging.getLogger(__name__)

OBSERVED = "observed"
CLIMATOLOGY = "climatology"
_NO_OBS = np.datetime64("2262-01-01", "ns")
"""Stand-in observed onset (after every init) that makes every cell a case for the climatology functions."""


# --- configuration ----------------------------------------------------------------------------------

def onsets_folder(rc: RunConfig) -> Path:
    """The folder of the onset files: the run's ``onsets`` folder, or dir_out/onsets."""
    return Path(rc.paths.onsets) if rc.paths is not None else Path(rc.output.data_dir) / "onsets"


def default_sources(rc: RunConfig) -> list[str]:
    """The sources a run uses: observed, every model and the climatology reference when it is one."""
    names = [OBSERVED, *(m.name for m in rc.models)]
    if rc.reference.model in (CLIMATOLOGY, CONDITIONAL_CLIMATOLOGY):
        names.append(rc.reference.model)
    return names


def _check_sources(rc: RunConfig, sources) -> list[str]:
    """The requested sources (a list or comma-separated text) checked against the ones the config can export."""
    if sources is None:
        return default_sources(rc)
    if isinstance(sources, str):
        sources = [s for s in sources.split(",") if s.strip()]
    names = [s.strip() for s in sources]
    allowed = [OBSERVED, *(m.name for m in rc.models), CLIMATOLOGY, CONDITIONAL_CLIMATOLOGY]
    bad = [n for n in names if n not in allowed]
    if bad:
        raise ValueError(f"unknown source(s) {bad}; choose from {allowed}")
    return names


# --- helpers for dates, attributes and encoding -------------------------------------------------------

def _mask_summary(area: AreaSpec, sample: xr.DataArray) -> dict:
    """The mask settings and how many cells (or units) of the loaded grid have no data at all in the sample variable."""
    spatial = [d for d in sample.dims if d in ("lat", "lon", "unit")]
    other = [d for d in sample.dims if d not in spatial]
    empty = sample.isnull().all(other) if other else sample.isnull()
    return {"region": area.region, "nc_mask": area.nc_mask, "shpfile_dir": area.shpfile_dir,
            "polygon": bool(area.polygon), "n_cells": int(empty.size), "n_cells_without_data": int(empty.sum())}


def _version() -> str:
    """The installed version of HCWF."""
    from importlib import metadata

    try:
        return metadata.version("hcwf")
    except metadata.PackageNotFoundError:
        return "unknown"


def _provenance(rc: RunConfig, case: Case, source: str, kind: str, thresholds, mask: dict, years, extra: dict) -> dict:
    """Global attributes of an export file: CF convention, definition, thresholds, mask, settings, paths, versions, config."""
    setup, ev, obs, model = rc.onset, rc.evaluation, rc.observations, case.model
    d = setup.definition
    window = ev.lead_window or (1, ev.max_forecast_day)
    try:
        config_yaml = resolved_config_text_of(rc)
    except Exception as exc:  # noqa: BLE001 - a config that cannot be rendered must not stop the export
        config_yaml = f"unavailable: {exc}"
    paths = {"observations": {"dir": str(obs.dir), "file_pattern": obs.file_pattern, "variable": obs.var}}
    if kind != "observed":
        paths["forecast"] = {"dir": str(model.dir), "file_pattern": model.file_pattern, "variable": model.var}
    if setup.threshold_files:
        paths["threshold_files"] = [list(f) for f in setup.threshold_files]
    attrs = {
        "Conventions": "CF-1.8", "title": f"HCWF rainy-season onset dates: {source}",
        "institution": "HCWF (hcwf)", "source": f"hcwf {_version()}, hcwf.runs.export",
        "history": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} created by hcwf.runs.export",
        "created": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "hcwf_version": _version(), "source_name": source, "source_kind": kind,
        "definition_id": d.id, "definition_version": d.version, "definition_hash": definition_mod.definition_hash(d),
        "definition_sentence": describe(d), "definition_yaml": d.source_yaml or "",
        "definition_overrides": ",".join(setup.overrides),
        "dry_spell_check": setup.dry_check, "onset_season": setup.season, "onset_backend": setup.backend,
        "threshold_source": setup.threshold_source,
        "threshold_years": f"{setup.threshold_years[0]}-{setup.threshold_years[1]}" if setup.threshold_years and setup.uses_thresholds else "",
        "threshold_files": json_text([list(f) for f in setup.threshold_files]),
        "threshold_values_hash": "none" if thresholds is None or (
            isinstance(thresholds, dict) and not thresholds and getattr(thresholds, "season_climatology", None) is None
        ) else threshold_hash(thresholds),
        "mask_summary": json_text(mask),
        "forecast_history": setup.forecast_history, "lead_label": setup.lead_label,
        "lead_window": f"{window[0]}-{window[1]}", "max_forecast_day": ev.max_forecast_day,
        "obs_day_boundary": setup.obs_day_boundary, "forecast_day_boundary": setup.forecast_boundary,
        "years": [int(y) for y in years], "years_clim": [int(y) for y in ev.climatology_years],
        "init_days": list(ev.init_days), "region": rc.area.region,
        "dataset_paths": json_text(paths), "resolved_config_yaml": config_yaml,
    }
    attrs.update(extra)
    return netcdf_attrs(attrs)


# --- observed -----------------------------------------------------------------------------------------

def _years_loop(rc: RunConfig, load):
    """Yield (year, loaded) for every evaluated year, skipping (with a warning) years whose files are missing."""
    for year in rc.evaluation.years:
        try:
            yield year, load(year)
        except FileNotFoundError as exc:
            warnings.warn(f"export skips year {year} (missing input file): {exc}", UserWarning)


def collect_observed(rc: RunConfig) -> xr.Dataset:
    """The observed onsets of every evaluated year, from hcwf.onset.detect.detect_observed_result (what is scored).

    Dims (year, season, lat, lon), or (year, season, unit) on evaluation units: onset_date, onset_doy and status.
    """
    case = rc.first_case(rc.models[0].name)
    thresholds = thresholds_of(rc, case.model)
    parts, sample, setup = [], None, None
    for year, rain in _years_loop(rc, lambda y: observations_of(y, rc)):
        result, setup = detect_observed_dataset(rain, thresholds, year, rc)
        parts.append(result)
        if sample is None:
            sample = rain.isel(time=slice(0, 400))
    if not parts:
        raise ValueError("no observation year could be read; nothing to export")
    result = xr.concat(parts, dim="year", join="exact", data_vars="minimal", coords="minimal")
    calendar = parts[0].attrs["calendar"]
    spatial = axes.spatial_dims(result["onset_date"].dims)
    dims = ("year", "season", *spatial)
    days = days_of(result["onset_date"].transpose(*dims).values, calendar)
    ds = xr.Dataset(
        {"onset_date": date_var(days, dims, calendar, "observed rainy-season onset date",
                                 "no onset: masked (see status)"),
         "onset_doy": float_var(day_of_year(days, calendar), dims, "day of the calendar year of the onset", "1",
                                 "1-366 in the onset date's own year; NaN where there is no onset"),
         "status": status_var(result["status"].transpose(*dims).values, dims)},
        coords={"year": result["year"].values, "season": result["season"].values,
                **{d: result[d].values for d in spatial}})
    ds = with_unit_labels(ds, rc.observations.units)
    coord_attrs(ds, None, "")
    years = [int(y) for y in ds["year"].values]
    mask = _mask_summary(rc.area, sample)
    ds.attrs = _provenance(rc, case, OBSERVED, "observed", thresholds, mask, years, {
        "calendar": calendar, "comment": "onset_date is NaT (masked) and onset_doy NaN where status is not ONSET"})
    return ds


# --- models -------------------------------------------------------------------------------------------

def collect_model(rc: RunConfig, model: str) -> xr.Dataset:
    """The onsets of one forecast model for every evaluated year, from hcwf.onset.detect._detect_forecast.

    Dims (init_time, [member], season, lat, lon), or with unit instead of lat and lon on evaluation units: onset_date, onset_doy, onset_lead (the engine's 1-based lead,
    <= 0 when back-dated into the observed history), onset_day (days from the init date to the onset date, as
    the scoring tables' onset_day), status and scored. Ensembles (the run's `probabilistic`) also have
    members_by_status, onset_probability and the ensemble onset the scoring uses: ensemble_onset_date,
    ensemble_onset_doy, ensemble_onset_day and ensemble_status.
    """
    case = rc.first_case(model)
    ev, threshold = rc.evaluation, rc.onset_options.percentage_threshold
    ensemble = bool(ev.probabilistic)
    thresholds = thresholds_of(rc, case.model)
    parts, sample, calendar, setup = [], None, "standard", None

    def load(year):
        """The year's rain, forecast and observed onsets, as the scoring loads them."""
        return observations_of(year, rc), forecast_of(year, rc, case.model, ensemble=ensemble)

    for year, (rain, p_model) in _years_loop(rc, load):
        onset_da = detect_observed(rain, thresholds, year, rc)
        out = detect_forecast(p_model, thresholds, onset_da, rc, case, members=ev.members if ensemble else None,
                              history=history_of(year, rain, rc))
        setup, calendar = out["setup"], out["calendar"]
        parts.append(_model_year(out, ensemble, threshold))
        if sample is None:
            sample = p_model.isel({d: 0 for d in p_model.dims if d not in ("lat", "lon", "unit")})
    if not parts:
        raise ValueError(f"no forecast year of {model} could be read; nothing to export")
    ds = xr.concat(parts, dim="init_time", join="exact", data_vars="minimal", coords="minimal")
    ds = with_unit_labels(ds, rc.observations.units)
    coord_attrs(ds, setup.lead_label, lead_comment(setup.lead_label))
    years = sorted({int(t.year) for t in pd.DatetimeIndex(ds["init_time"].values)})
    extra = {"calendar": calendar, "model": model, "ensemble": ensemble,
             "onset_percentage_threshold": threshold if ensemble else None,
             "members": ev.members,
             "comment": "scored is 1 where HCWF scores the case: the cell has an observed onset and the init is "
                        "before it; values are given for every init and cell"}
    ds.attrs = _provenance(rc, case, model, "ensemble forecast" if ensemble else "deterministic forecast",
                           thresholds, _mask_summary(rc.area, sample), years, extra)
    return ds


def _model_year(out: dict, ensemble: bool, threshold: float) -> xr.Dataset:
    """One year of a model as a Dataset from the dict detect_forecast returns."""
    res, calendar = out["result"], out["calendar"]
    dims = tuple(res["status"].dims)
    spatial = out["spatial"]
    status = res["status"].transpose(*dims).values
    found = status == Status.ONSET
    days = np.where(found, days_of(res["onset_date"].transpose(*dims).values, calendar), np.nan)
    init_days = days_of(out["init"], calendar)
    shape = (-1,) + (1,) * (len(dims) - 1)
    variables = {
        "onset_date": date_var(days, dims, calendar, "forecast rainy-season onset date",
                                "date label of the onset day; set only where status is ONSET (a PRIOR search is "
                                "reported as no onset, as scored); no onset: masked"),
        "onset_doy": float_var(day_of_year(days, calendar), dims, "day of the calendar year of the onset", "1",
                                "1-366 in the onset date's own year; NaN where there is no onset"),
        "onset_lead": float_var(np.where(found, res["onset_lead"].transpose(*dims).values, np.nan), dims,
                                 "lead of the onset day in days", "",
                                 "1-based forecast lead of the onset day (see the lead coordinate for lead_label); "
                                 "<= 0 means back-dated into the observed history; NaN where there is no onset"),
        "onset_day": float_var(days - init_days.reshape(shape), dims, "days from the init date to the onset date",
                                "", "as the onset_day column of the scored tables; NaN where there is no onset"),
        "status": status_var(status, dims),
        "scored": xr.DataArray(out["valid"].astype(np.int8), dims=("init_time", *spatial), attrs={
            "long_name": "case is scored by HCWF", "flag_values": np.array([0, 1], dtype=np.int8),
            "flag_meanings": "not_scored scored",
            "comment": "1: the cell has an observed onset and the init is before it"}),
    }
    if ensemble:
        _ensemble_variables(variables, out, res, threshold, calendar, init_days)
    return xr.Dataset(variables, coords={"init_time": res["init_time"].values, "season": res["season"].values,
                                         **{d: res[d].values for d in spatial},
                                         **({"member": res["member"].values} if "member" in res.coords else {})})


def _ensemble_variables(variables: dict, out: dict, res: xr.Dataset, threshold: float, calendar: str,
                        init_days: np.ndarray) -> None:
    """Add the ensemble summary (the engine's, and the ensemble onset HCWF scores) to variables, in place."""
    spatial = out["spatial"]
    dims = ("init_time", "season", *spatial)
    by_status = res["members_by_status"].transpose("status_code", *dims)
    variables["members_by_status"] = xr.DataArray(
        by_status.values.astype(np.int32), dims=by_status.dims,
        coords={"status_code": by_status["status_code"].values,
                "status_name": ("status_code", by_status["status_name"].values)},
        attrs={**by_status.attrs, "comment": "member counts per status code; the codes are the status_code flags"})
    probability = res["onset_probability"].transpose("init_time", "season", "lead", *spatial)
    variables["onset_probability"] = xr.DataArray(
        probability.values.astype(np.float32), dims=probability.dims, coords={"lead": probability["lead"].values},
        attrs=dict(probability.attrs))
    count, fraction, day, date = ensemble_onset(out["day"], out["init"], threshold)
    has_onset = ~np.isnan(day)
    names = _ensemble_status(out["status"], threshold, has_onset)
    codes = np.vectorize(lambda n: Status[n].value, otypes=[np.int8])(names)
    ensemble_days = days_of(date, calendar) if standard_calendar(calendar) else axes.broadcast_rows(init_days, day.ndim) + day
    ensemble_days = np.where(has_onset, ensemble_days, np.nan)
    comment = (f"as scored: the ensemble has an onset when at least {threshold:g} of the members have one; its "
               "day after the init is the mean member onset day rounded half to even (this is not the median)")
    variables["ensemble_onset_date"] = date_var(ensemble_days[:, None], dims, calendar, "ensemble onset date", comment)
    variables["ensemble_onset_doy"] = float_var(day_of_year(ensemble_days, calendar)[:, None], dims,
                                                 "day of the calendar year of the ensemble onset", "1")
    variables["ensemble_onset_day"] = float_var(day[:, None], dims, "days from the init date to the ensemble onset",
                                                 "", comment)
    variables["ensemble_status"] = status_var(codes[:, None], dims, "ensemble onset search outcome")


# --- climatology references -----------------------------------------------------------------------------

def _reference_inputs(rc: RunConfig, model: str):
    """The reference's case, thresholds and per-year (year, observed onsets, init dates, rain sample) of a climatology reference."""
    case = rc.first_case(model, reference_as=model)
    thresholds = thresholds_of(rc, case.model)
    years = []
    for year, rain in _years_loop(rc, lambda y: observations_of(y, rc)):
        years.append((year, detect_observed(rain, thresholds, year, rc), initialization_dates(year, rc.evaluation), rain))
    if not years:
        raise ValueError("no observation year could be read; nothing to export")
    return case, thresholds, years


def _scored(init_dates, obs_da: xr.DataArray) -> np.ndarray:
    """The cases HCWF scores, (init, lat, lon) or (init, unit) bool: the cell has an observed onset and the init is before it."""
    obs = obs_da.transpose(*axes.spatial_dims(obs_da.dims)).values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates]).values.astype("datetime64[ns]")
    return ~np.isnat(obs)[None] & (axes.broadcast_rows(inits, obs.ndim + 1) < obs[None])


def _stand_in(obs_da: xr.DataArray) -> xr.DataArray:
    """Observed onsets far in the future, so that the climatology functions return a row for every init and cell."""
    return xr.DataArray(np.full(obs_da.shape, _NO_OBS), coords=obs_da.coords, dims=obs_da.dims)


def _reference_year(table: pd.DataFrame, init_dates, obs_da: xr.DataArray, scored: np.ndarray, season: str) -> xr.Dataset:
    """One year of a climatology reference from the table of its forecast functions (a row per init, lat, lon or unit)."""
    spatial = axes.spatial_dims(obs_da.dims)
    shape = (len(init_dates), 1, *(obs_da.sizes[d] for d in spatial))
    dims = ("init_time", "season", *spatial)
    date = pd.to_datetime(pd.Series(table["onset_date"].to_numpy(dtype=object))).to_numpy().reshape(shape)
    day = table["onset_day"].to_numpy(dtype=float).reshape(shape)
    days = days_of(date.astype("datetime64[ns]"), "standard")
    found = ~np.isnan(days)
    return xr.Dataset(
        {"onset_date": date_var(days, dims, "standard", "climatological forecast onset date",
                                 "the forecast is a date within the lead window; none: masked"),
         "onset_doy": float_var(day_of_year(days, "standard"), dims, "day of the calendar year of the onset", "1"),
         "onset_lead": float_var(day, dims, "lead of the onset day in days", "",
                                  "days after the init date (lead k is init + k); NaN where there is no onset"),
         "onset_day": float_var(day, dims, "days from the init date to the onset date"),
         "status": status_var(np.where(found, Status.ONSET, Status.NOT_FOUND), dims),
         "scored": xr.DataArray(scored.astype(np.int8), dims=("init_time", *spatial), attrs={
             "long_name": "case is scored by HCWF", "flag_values": np.array([0, 1], dtype=np.int8),
             "flag_meanings": "not_scored scored",
             "comment": "1: the cell has an observed onset and the init is before it"})},
        coords={"init_time": pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates]).values,
                "season": [season], **{d: obs_da[d].values for d in spatial}})


def collect_climatology(rc: RunConfig) -> xr.Dataset:
    """The traditional climatology: its onset per cell and, per init, the forecast HCWF scores from it.

    climatological_onset_doy and climatological_onset_date (season, lat, lon) come from
    hcwf.references.climatology.climatological_onset (the rounded mean day of the climatology years);
    the per-init onset_date, onset_doy, onset_lead, onset_day, status and scored (init_time, season, lat, lon)
    from compute_climatology_as_forecast (a date within the lead window, else no onset).
    """
    case, thresholds, years = _reference_inputs(rc, CLIMATOLOGY)
    setup, ev = rc.onset, rc.evaluation
    clim_doy = climatological_onset(rc, case)
    parts = []
    for year, obs_da, init_dates, _ in years:
        if not len(init_dates):
            continue
        table = compute_climatology_as_forecast(clim_doy, year, init_dates, _stand_in(obs_da),
                                                max_forecast_day=ev.max_forecast_day)
        parts.append(_reference_year(table, init_dates, obs_da, _scored(init_dates, obs_da), setup.season))
    ds = xr.concat(parts, dim="init_time", join="exact", data_vars="minimal", coords="minimal")
    spatial = axes.spatial_dims(clim_doy.dims)
    doy = clim_doy.transpose(*spatial).values.astype(float)[None]
    reference_year = int(years[0][0])
    nominal = np.where(np.isnan(doy), np.nan, (np.datetime64(f"{reference_year}-01-01") - EPOCH)
                       / np.timedelta64(1, "D") + doy - 1)
    ds = with_unit_labels(ds, rc.observations.units)
    if "unit" in spatial:   # units are matched by id
        doy = clim_doy.reindex(unit=ds["unit"].values).transpose(*spatial).values.astype(float)[None]
        nominal = np.where(np.isnan(doy), np.nan, (np.datetime64(f"{reference_year}-01-01") - EPOCH)
                           / np.timedelta64(1, "D") + doy - 1)
    ds["climatological_onset_doy"] = float_var(doy, ("season", *spatial), "climatological onset, day of year",
                                               "1", "rounded mean day of the calendar year of the observed onsets "
                                               f"of years_clim {ev.climatology_years[0]}-{ev.climatology_years[-1]}")
    ds["climatological_onset_date"] = date_var(nominal, ("season", *spatial), "standard",
                                                "climatological onset date",
                                                f"the day of year placed in the nominal year {reference_year}")
    coord_attrs(ds, setup.lead_label, "days after the init date: the climatology forecast onset lies at lead k = init + k")
    ds.attrs = _provenance(rc, case, CLIMATOLOGY, "climatology reference", thresholds,
                           _mask_summary(rc.area, years[0][3].isel(time=slice(0, 400))), [y[0] for y in years],
                           {"calendar": "standard", "climatology_reference_year": reference_year,
                            "comment": "per-init variables are the climatology as a forecast, as scored; "
                                       "status is ONSET when the climatological date lies within the lead window, "
                                       "else NOT_FOUND; scored is 1 where HCWF scores the case"})
    return ds


def collect_conditional_climatology(rc: RunConfig) -> xr.Dataset:
    """The conditional climatology: per init its median onset and its full onset probability by lead.

    onset_date, onset_doy, onset_lead, onset_day, status and scored (init_time, season, lat, lon) come from
    compute_conditional_climatology_as_forecast (the first lead whose conditional CDF reaches 0.5; no onset
    when it does not or when the cell has no KDE). onset_probability
    (init_time, season, lead, lat, lon) is that conditional CDF, P(onset on or before the lead | no onset by
    the init), for leads 0 to max_forecast_day (0 at lead 0), NaN for cells without a KDE.
    """
    case, thresholds, years = _reference_inputs(rc, CONDITIONAL_CLIMATOLOGY)
    setup, ev = rc.onset, rc.evaluation
    clim_onset = climatological_onset_dataset(rc, case)
    horizon = int(ev.max_forecast_day)
    climatology_cv = rc.reference.climatology_cv
    parts = []
    for year, obs_da, init_dates, _ in years:
        if not len(init_dates):
            continue
        table = compute_conditional_climatology_as_forecast(clim_onset, year, init_dates, _stand_in(obs_da),
                                                            max_forecast_day=ev.max_forecast_day,
                                                            climatology_cv=climatology_cv)
        part = _reference_year(table, init_dates, obs_da, _scored(init_dates, obs_da), setup.season)
        probability = conditional_lead_probabilities(clim_onset, year, init_dates, obs_da, horizon, climatology_cv)
        cdf = np.cumsum(probability, axis=-1)
        cdf = np.concatenate([np.where(np.isnan(cdf[..., :1]), np.nan, 0.0), cdf], axis=-1)
        spatial = axes.spatial_dims(obs_da.dims)
        part["onset_probability"] = xr.DataArray(
            np.moveaxis(cdf, -1, 1)[:, None].astype(np.float32), dims=("init_time", "season", "lead", *spatial),
            coords={"lead": np.arange(horizon + 1)},
            attrs={"long_name": "probability that the onset is on or before this lead, given none by the init",
                   "comment": "conditional CDF of the KDE of observed onsets over the climatology years; 0 at "
                              "lead 0; NaN for cells with fewer than 10 onset years"})
        parts.append(part)
    ds = xr.concat(parts, dim="init_time", join="exact", data_vars="minimal", coords="minimal")
    ds = with_unit_labels(ds, rc.observations.units)
    coord_attrs(ds, setup.lead_label, "days after the init date: the conditional onset lies at lead k = init + k")
    ds.attrs = _provenance(rc, case, CONDITIONAL_CLIMATOLOGY, "conditional climatology reference", thresholds,
                           _mask_summary(rc.area, years[0][3].isel(time=slice(0, 400))), [y[0] for y in years],
                           {"calendar": "standard", "climatology_cv": cv_mode(climatology_cv),
                            "climatology_cv_configured": climatology_cv,
                            "comment": "per-init variables are the conditional climatology as scored (median "
                                       "onset = first lead with CDF >= 0.5); climatology_cv is the mode the "
                                       "scoring used; scored is 1 where HCWF scores the case"})
    return ds


# --- entry point --------------------------------------------------------------------------------------

def _collect(rc: RunConfig, source: str) -> xr.Dataset:
    """The Dataset of one source."""
    if source == OBSERVED:
        return collect_observed(rc)
    if source == CLIMATOLOGY:
        return collect_climatology(rc)
    if source == CONDITIONAL_CLIMATOLOGY:
        return collect_conditional_climatology(rc)
    return collect_model(rc, source)


def export_onsets(cfg, sources=None, out=None) -> dict:
    """Write the onset dates of each requested source to NetCDF; return {source: path}.

    cfg is the path of a run YAML, a hcwf.config.Config, a mapping, a flat config dict (resolved, validated and
    planned by hcwf.config.build.resolve) or a RunConfig.
    sources is a list or comma-separated text of "observed", model names, "climatology" and
    "conditional_climatology" (default: observed, every model, and the climatology reference the run uses).
    out is the folder (default: the run's onsets folder, dir_out/onsets for an explicit dir_out). Files are named
    onsets_<source>.nc, or <source>.nc inside a run folder. Only the detection is run, with the scoring's own
    functions and settings; nothing is scored.
    """
    from hcwf.config.build import resolve

    with run_scope():   # inside a run (output.save_onsets) this keeps the run's state; alone it starts and ends its own
        rc = resolve(cfg)
        names = _check_sources(rc, sources)
        folder = Path(out) if out is not None else onsets_folder(rc)
        paths = {}
        for source in names:
            log.info(f"Exporting onset dates of {source} ...")
            ds = _collect(rc, source)
            path = write_netcdf(ds, folder / file_name(rc.paths, "onset", source.replace(" ", "_")))
            log.info(f"  wrote {path}")
            paths[source] = path
        return paths
