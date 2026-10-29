"""NetCDF export of the gridded onset dates a ROMP run scores.

    from momp.export import export_onsets
    paths = export_onsets(cfg, sources=["observed", "AIFS"], out="onsets")   # {source: Path}

or ``momp-run export -p run.yaml [--sources observed,AIFS,climatology] [--out DIR]``, or ``output.save_onsets:
true`` in the run file (files go to dir_out/onsets/ before scoring).

One file per source, each over all evaluated years: ``onsets_observed.nc``, ``onsets_<model>.nc`` for every
model, ``onsets_climatology.nc`` and ``onsets_conditional_climatology.nc``. The values are not re-detected
with other settings: every number comes from the functions the scoring code calls (momp.stats.detect,
momp.stats.climatology, momp.stats.conditional_climatology), with the keyword arguments the scoring builds for
the same case (Setting and Case dataclasses), so masks, lead window (after shortening), forecast history,
thresholds and definition are the scored ones. Unlike the scoring tables, which keep only the cases with an
observed onset and an init before it, the files cover every init and cell; the flag ``scored`` marks the cases
the scoring keeps.

Dates are CF time variables (int32 days since 1970-01-01 in the data's calendar, cftime for non-standard
calendars, the fill value meaning no onset); day numbers and leads are float32 with NaN for none. Every file
has CF-1.8 global attributes holding the definition, thresholds, masks, forecast history, lead window, data
paths, versions and the resolved run config (see docs/onset_in_romp.md, "Exporting onset dates").
"""
from __future__ import annotations

import datetime
import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from momp.io.input import (get_forecast_deterministic_twice_weekly, get_forecast_probabilistic_twice_weekly,
                           get_initialization_dates, load_history, load_imd_rainfall, load_thresholds)
from momp.lib.control import init_dataclass, iter_list, make_case
from momp.lib.convention import Case, Setting
from momp.lib.onset_config import onset_setup
from momp.lib.resolved import resolved_config_text
from momp.onset import definition as definition_mod
from momp.onset.describe import describe
from momp.onset.model import Status
from momp.stats.climatology import compute_climatological_onset, compute_climatological_onset_dataset
from momp.stats.climatology import compute_climatology_as_forecast
from momp.stats.conditional_climatology import (CONDITIONAL_CLIMATOLOGY, climatology_cv_mode,
                                                compute_conditional_climatology_as_forecast,
                                                conditional_lead_probabilities)
from momp.stats.detect import (_detect_forecast, detect_observed_onset, detect_observed_result, ensemble_onset,
                               _ensemble_status)

OBSERVED = "observed"
CLIMATOLOGY = "climatology"
SOURCE_FILE = "onsets_{}.nc"
"""File name of a source's export; {} is the source name."""
DATE_UNITS = "days since 1970-01-01"
DATE_FILL = -2147483648
_EPOCH = np.datetime64("1970-01-01")
_NO_OBS = np.datetime64("2262-01-01", "ns")
"""Stand-in observed onset (after every init) that makes every cell a case for the climatology functions."""
_MAX_CHUNK = 2_000_000
_COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}


# --- configuration ----------------------------------------------------------------------------------

def load_cfg(path) -> dict:
    """The flat config dict of a run YAML as the run itself builds it (paths resolved, lead window shortened).

    Reads the file with the loader's own steps and the validator, but without its global cache.
    """
    from momp.lib.assertion import ROMPValidator
    from momp.lib.loader import build_cfg

    dic = build_cfg(["-p", str(path)])
    ROMPValidator(dic).validate()
    return _complete(dic)


def _complete(dic: dict) -> dict:
    """Finish a flat config the way the loader does (mode flag, paths, lead window); dicts that are done pass through."""
    from momp.lib import loader
    from momp.lib.lead_window import apply_lead_window

    if "onset_legacy" not in dic:
        loader._rename_deprecated_keys(dic)
        loader._set_onset_mode(dic)
        loader._resolve_paths(dic)
    if "lead_window" not in dic:
        apply_lead_window(dic)
    return dic


def _as_dict(cfg) -> dict:
    """A flat config dict from a dict, SimpleNamespace, momp.config.Config object or the path of a run YAML."""
    if isinstance(cfg, (str, Path)):
        return load_cfg(cfg)
    if hasattr(cfg, "to_flat"):
        cfg = cfg.to_flat()
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return _complete(dict(cfg))


def case_kwargs(dic: dict, model: str, *, reference: bool = False) -> dict:
    """The keyword arguments the scoring builds for one model: Setting and Case values of its first window.

    The detection does not depend on the verification window, so the first one stands for all. With
    reference=True the case is the climatology reference's, built as spatial_far_mr_mae_map builds it:
    the model name is the reference's and the data location the reference's.
    """
    dic = dict(dic)
    if reference:
        dic["model_list"] = (model,)
    pool = iter_list(dic)
    combi = tuple(model if field == "model" else values[0] for field, values in zip(dic["layout"], pool))
    case = make_case(Case, combi, dic)
    if reference:
        case.update({"model_dir": dic["ref_model_dir"], "model_var": case.ref_model_var,
                     "file_pattern": dic["ref_model_file_pattern"], "unit_cvt": dic["ref_model_unit_cvt"],
                     "model_dims": None})
    return {**asdict(init_dataclass(Setting, dic)), **asdict(case)}


def default_sources(dic: dict) -> list[str]:
    """The sources a run uses: observed, every model and the climatology reference when it is one."""
    names = [OBSERVED, *dic["model_list"]]
    if dic.get("ref_model") in (CLIMATOLOGY, CONDITIONAL_CLIMATOLOGY):
        names.append(dic["ref_model"])
    return names


def _check_sources(dic: dict, sources) -> list[str]:
    """The requested sources (a list or comma-separated text) checked against the ones the config can export."""
    if sources is None:
        return default_sources(dic)
    if isinstance(sources, str):
        sources = [s for s in sources.split(",") if s.strip()]
    names = [s.strip() for s in sources]
    allowed = [OBSERVED, *dic["model_list"], CLIMATOLOGY, CONDITIONAL_CLIMATOLOGY]
    bad = [n for n in names if n not in allowed]
    if bad:
        raise ValueError(f"unknown source(s) {bad}; choose from {allowed}")
    return names


# --- helpers for dates, attributes and encoding -------------------------------------------------------

def _days(values, calendar: str) -> np.ndarray:
    """Days since 1970-01-01 (float, NaN for none) of datetime64 values or cftime objects (None for none)."""
    values = np.asarray(values)
    if values.dtype.kind == "M":
        days = (values.astype("datetime64[D]") - _EPOCH) / np.timedelta64(1, "D")
        return np.where(np.isnat(values), np.nan, days).astype(float)
    import cftime

    out = np.full(values.shape, np.nan)
    present = np.array([v is not None for v in values.ravel()], dtype=bool).reshape(values.shape)
    if present.any():
        out[present] = cftime.date2num(list(values[present]), DATE_UNITS, calendar=calendar)
    return out


def _to_dates(days, calendar: str) -> np.ndarray:
    """datetime64[ns] (NaT for none) or, for a non-standard calendar, cftime objects (None) of day numbers."""
    days = np.asarray(days, dtype=float)
    present = ~np.isnan(days)
    if _standard(calendar):
        whole = np.where(present, days, 0).astype("int64").astype("timedelta64[D]")
        return np.where(present, (_EPOCH + whole).astype("datetime64[ns]"), np.datetime64("NaT", "ns"))
    import cftime

    out = np.full(days.shape, None, dtype=object)
    if present.any():
        out[present] = cftime.num2date(days[present], DATE_UNITS, calendar=calendar)
    return out


def _standard(calendar: str) -> bool:
    """Whether dates in this calendar are numpy datetimes."""
    return calendar in ("standard", "gregorian", "proleptic_gregorian")


def _doy(days, calendar: str) -> np.ndarray:
    """Day of the calendar year (1-366) of day numbers, NaN for none, counted in the date's own year."""
    dates = _to_dates(days, calendar)
    if _standard(calendar):
        return pd.DatetimeIndex(dates.ravel()).dayofyear.to_numpy(dtype=float).reshape(dates.shape)
    return np.array([np.nan if d is None else d.dayofyr for d in dates.ravel()], dtype=float).reshape(dates.shape)


def _date_var(days, dims, calendar: str, long_name: str, comment: str = "") -> xr.DataArray:
    """A CF date variable: day numbers with NaN for none, written as int32 with a fill value (so none is masked)."""
    attrs = {"long_name": long_name, "units": DATE_UNITS, "calendar": "standard" if _standard(calendar) else calendar}
    if comment:
        attrs["comment"] = comment
    return xr.DataArray(np.asarray(days, dtype=float), dims=dims, attrs=attrs)


def _float_var(values, dims, long_name: str, units: str = "", comment: str = "") -> xr.DataArray:
    """A float32 variable with NaN for none."""
    attrs = {"long_name": long_name}
    if units:
        attrs["units"] = units
    if comment:
        attrs["comment"] = comment
    return xr.DataArray(np.asarray(values, dtype=np.float32), dims=dims, attrs=attrs)


def status_var(codes, dims, long_name: str = "onset search outcome") -> xr.DataArray:
    """An int8 status variable with CF flag_values and flag_meanings (ONSET, NOT_FOUND, CENSORED, PRIOR, INSUFFICIENT_DATA)."""
    return xr.DataArray(np.asarray(codes).astype(np.int8), dims=dims, attrs={
        "long_name": long_name,
        "flag_values": np.array([s.value for s in Status], dtype=np.int8),
        "flag_meanings": " ".join(s.name for s in Status)})


def _coord_attrs(ds: xr.Dataset, lead_label: str | None, lead_comment: str) -> None:
    """Attach CF attributes to the coordinates present in ds."""
    attrs = {
        "lat": {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude", "axis": "Y"},
        "lon": {"units": "degrees_east", "standard_name": "longitude", "long_name": "longitude", "axis": "X"},
        "year": {"long_name": "year of the onset season (the season's start)"},
        "season": {"long_name": "onset season (name in the definition)"},
        "init_time": {"standard_name": "forecast_reference_time", "long_name": "forecast initialisation date"},
        "member": {"long_name": "ensemble member"},
        "status_code": {"long_name": "onset status code", "flag_values": np.array([s.value for s in Status], dtype=np.int8),
                        "flag_meanings": " ".join(s.name for s in Status)},
        "lead": {"units": "days", "long_name": "forecast lead", "comment": lead_comment, "lead_label": lead_label or ""},
    }
    for name, attrs_of in attrs.items():
        if name in ds.coords:
            ds[name].attrs.update(attrs_of)


def _lead_comment(label: str | None) -> str:
    """Describe which date a lead number stands for."""
    if label == "start":
        return ("lead_label 'start': lead k is the forecast day dated init + k - 1; lead 1 is the init day. "
                "Leads <= 0 are onsets back-dated into the observed history.")
    if label == "end":
        return ("lead_label 'end': lead k is the forecast day dated init + k. Leads <= 0 are onsets back-dated "
                "into the observed history.")
    return "lead k counted from the first forecast day (lead 1); leads <= 0 are onsets back-dated into the observed history."


def _json(value) -> str:
    """JSON text of a value (paths and unknown objects as text)."""
    return json.dumps(value, default=str, sort_keys=True)


def _clean(attrs: dict) -> dict:
    """Attributes NetCDF can hold: text, numbers and number arrays; bools become 0/1; None is dropped."""
    out = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        elif isinstance(value, dict):
            value = _json(value)
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        elif isinstance(value, Path):
            value = str(value)
        out[key] = value
    return out


def _threshold_hash(thresholds) -> str:
    """sha256 of the threshold values (names, shapes and numbers), or "none" when the definition has none."""
    if thresholds is None or (isinstance(thresholds, dict) and not thresholds and
                              getattr(thresholds, "season_climatology", None) is None):
        return "none"
    items = sorted(thresholds.items()) if isinstance(thresholds, dict) else [("legacy_threshold", thresholds)]
    climatology = getattr(thresholds, "season_climatology", None)
    if climatology is not None:
        items += [(f"season_climatology.{name}", climatology[name]) for name in sorted(climatology.data_vars)]
    digest = hashlib.sha256()
    for name, value in items:
        array = np.ascontiguousarray(np.asarray(getattr(value, "values", value), dtype=float))
        digest.update(f"{name}|{array.shape}|".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mask_summary(dic: dict, sample: xr.DataArray) -> dict:
    """The mask settings and how many cells of the loaded grid have no data at all in the sample variable."""
    spatial = [d for d in sample.dims if d in ("lat", "lon")]
    other = [d for d in sample.dims if d not in spatial]
    empty = sample.isnull().all(other) if other else sample.isnull()
    return {"region": dic.get("region"), "nc_mask": dic.get("nc_mask"), "shpfile_dir": dic.get("shpfile_dir"),
            "polygon": bool(dic.get("polygon")), "n_cells": int(empty.size), "n_cells_without_data": int(empty.sum())}


def _version() -> str:
    """The installed version of ROMP."""
    from importlib import metadata

    try:
        return metadata.version("momp")
    except metadata.PackageNotFoundError:
        return "unknown"


def _provenance(dic: dict, kw: dict, source: str, kind: str, thresholds, mask: dict, years, extra: dict) -> dict:
    """Global attributes of an export file: CF convention, definition, thresholds, mask, settings, paths, versions, config."""
    setup = onset_setup(kw)
    d = setup.definition
    window = dic.get("lead_window") or (1, kw.get("max_forecast_day"))
    try:
        config_yaml = resolved_config_text(dic)
    except Exception as exc:  # noqa: BLE001 - a config that cannot be rendered must not stop the export
        config_yaml = f"unavailable: {exc}"
    paths = {"observations": {"dir": str(kw.get("obs_dir")), "file_pattern": kw.get("obs_file_pattern"),
                              "variable": kw.get("obs_var")}}
    if kind != "observed":
        paths["forecast"] = {"dir": str(kw.get("model_dir")), "file_pattern": kw.get("file_pattern"),
                             "variable": kw.get("model_var")}
    if setup.threshold_files:
        paths["threshold_files"] = [list(f) for f in setup.threshold_files]
    attrs = {
        "Conventions": "CF-1.8", "title": f"ROMP rainy-season onset dates: {source}",
        "institution": "ROMP (momp)", "source": f"momp {_version()}, momp.export",
        "history": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} created by momp.export",
        "created": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "romp_version": _version(), "source_name": source, "source_kind": kind,
        "definition_id": d.id, "definition_version": d.version, "definition_hash": definition_mod.definition_hash(d),
        "definition_sentence": describe(d), "definition_yaml": d.source_yaml or "",
        "definition_overrides": ",".join(setup.overrides), "definition_legacy": setup.legacy,
        "dry_spell_check": setup.dry_check, "onset_season": setup.season, "onset_backend": setup.backend,
        "threshold_source": setup.threshold_source,
        "threshold_years": f"{setup.threshold_years[0]}-{setup.threshold_years[1]}" if setup.threshold_years and setup.uses_thresholds else "",
        "threshold_files": _json([list(f) for f in setup.threshold_files]),
        "threshold_values_hash": _threshold_hash(thresholds),
        "mask_summary": _json(mask),
        "forecast_history": setup.forecast_history, "lead_label": setup.lead_label,
        "lead_window": f"{window[0]}-{window[1]}", "max_forecast_day": kw.get("max_forecast_day"),
        "obs_day_boundary": setup.obs_day_boundary or "unknown", "forecast_day_boundary": setup.forecast_boundary,
        "years": [int(y) for y in years], "years_clim": [int(y) for y in kw.get("years_clim") or ()],
        "init_days": list(kw.get("init_days") or ()), "region": kw.get("region"),
        "dataset_paths": _json(paths), "resolved_config_yaml": config_yaml,
    }
    attrs.update(extra)
    return _clean(attrs)


def _chunks(var: xr.DataArray) -> tuple[int, ...]:
    """Chunk sizes: whole map, lead and status axes per chunk, one init, year, season (and member for big grids)."""
    whole = ("lat", "lon", "lead", "status_code", "member")
    chunk = [min(n, 512) if d in whole else 1 for d, n in zip(var.dims, var.shape)]
    while np.prod(chunk) > _MAX_CHUNK:
        for dim in ("member", "lead", "status_code", "lon", "lat"):
            if dim in var.dims and chunk[var.dims.index(dim)] > 1:
                chunk[var.dims.index(dim)] = -(-chunk[var.dims.index(dim)] // 2)
                break
        else:
            break
    return tuple(int(c) for c in chunk)


def write_netcdf(ds: xr.Dataset, path) -> Path:
    """Write ds as compressed NetCDF4: zlib level 4 with shuffle, sensible chunks, fill values for date variables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, var in ds.data_vars.items():
        enc = dict(_COMPRESSION) if var.size else {}
        if var.size:
            enc["chunksizes"] = _chunks(var)
        if str(var.attrs.get("units", "")).startswith("days since"):
            enc.update({"dtype": "int32", "_FillValue": DATE_FILL})
        elif var.dtype.kind in "iu":
            enc["_FillValue"] = None
        encoding[name] = enc
    ds.to_netcdf(path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    return path


def open_onsets(path) -> xr.Dataset:
    """Open an export file; dates on non-standard calendars come back as cftime objects with None for no onset.

    xarray cannot decode a cftime variable that has masked values, so such date variables are decoded here
    (momp.export._to_dates) and everything else by xarray. Standard calendars open as datetime64 with NaT. The
    `lead` coordinate has units "days": it is opened as plain numbers, not timedeltas.
    """
    raw = xr.open_dataset(path, decode_times=False, decode_timedelta=False)
    dates = [n for n, v in raw.data_vars.items() if str(v.attrs.get("units", "")).startswith("days since")
             and not _standard(v.attrs.get("calendar", "standard"))]
    ds = xr.decode_cf(raw.drop_vars(dates), decode_timedelta=False)
    for name in dates:
        ds[name] = (raw[name].dims, _to_dates(raw[name].values, raw[name].attrs["calendar"]), {
            k: v for k, v in raw[name].attrs.items() if k not in ("units", "calendar")})
    ds.attrs.update(raw.attrs)
    return ds


# --- observed -----------------------------------------------------------------------------------------

def _years_loop(kw: dict, load):
    """Yield (year, loaded) for every evaluated year, skipping (with a warning) years whose files are missing."""
    for year in kw["years"]:
        try:
            yield year, load(year)
        except FileNotFoundError as exc:
            warnings.warn(f"export skips year {year} (missing input file): {exc}", UserWarning)


def collect_observed(dic: dict) -> xr.Dataset:
    """The observed onsets of every evaluated year, from momp.stats.detect.detect_observed_result (what is scored).

    Dims (year, season, lat, lon): onset_date, onset_doy and status.
    """
    kw = case_kwargs(dic, dic["model_list"][0])
    thresholds = load_thresholds(**kw)
    parts, sample, setup = [], None, None
    for year, rain in _years_loop(kw, lambda y: load_imd_rainfall(y, **kw)):
        result, setup = detect_observed_result(rain, thresholds, year, **kw)
        parts.append(result)
        if sample is None:
            sample = rain.isel(time=slice(0, 400))
    if not parts:
        raise ValueError("no observation year could be read; nothing to export")
    result = xr.concat(parts, dim="year", join="exact", data_vars="minimal", coords="minimal")
    calendar = parts[0].attrs["calendar"]
    dims = ("year", "season", "lat", "lon")
    days = _days(result["onset_date"].transpose(*dims).values, calendar)
    ds = xr.Dataset(
        {"onset_date": _date_var(days, dims, calendar, "observed rainy-season onset date",
                                 "no onset: masked (see status)"),
         "onset_doy": _float_var(_doy(days, calendar), dims, "day of the calendar year of the onset", "1",
                                 "1-366 in the onset date's own year; NaN where there is no onset"),
         "status": status_var(result["status"].transpose(*dims).values, dims)},
        coords={"year": result["year"].values, "season": result["season"].values,
                "lat": result["lat"].values, "lon": result["lon"].values})
    _coord_attrs(ds, None, "")
    years = [int(y) for y in ds["year"].values]
    mask = _mask_summary(dic, sample)
    ds.attrs = _provenance(dic, kw, OBSERVED, "observed", thresholds, mask, years, {
        "calendar": calendar, "comment": "onset_date is NaT (masked) and onset_doy NaN where status is not ONSET"})
    return ds


# --- models -------------------------------------------------------------------------------------------

def collect_model(dic: dict, model: str) -> xr.Dataset:
    """The onsets of one forecast model for every evaluated year, from momp.stats.detect._detect_forecast.

    Dims (init_time, [member], season, lat, lon): onset_date, onset_doy, onset_lead (the engine's 1-based lead,
    <= 0 when back-dated into the observed history), onset_day (days from the init date to the onset date, as
    the scoring tables' onset_day), status and scored. Ensembles (the run's `probabilistic`) also have
    members_by_status, onset_probability and the ensemble onset the scoring uses: ensemble_onset_date,
    ensemble_onset_doy, ensemble_onset_day and ensemble_status.
    """
    kw = case_kwargs(dic, model)
    ensemble = bool(kw["probabilistic"])
    thresholds = load_thresholds(**kw)
    parts, sample, calendar, setup = [], None, "standard", None

    def load(year):
        """The year's rain, forecast and observed onsets, as the scoring loads them."""
        rain = load_imd_rainfall(year, **kw)
        fetch = get_forecast_probabilistic_twice_weekly if ensemble else get_forecast_deterministic_twice_weekly
        return rain, fetch(year, **kw)

    for year, (rain, p_model) in _years_loop(kw, load):
        onset_da = detect_observed_onset(rain, thresholds, year, **kw)
        out = _detect_forecast(p_model, thresholds, onset_da, kw, members=kw.get("members") if ensemble else None,
                               history=load_history(year, rain, **kw))
        setup, calendar = out["setup"], out["calendar"]
        parts.append(_model_year(out, ensemble, kw["onset_percentage_threshold"]))
        if sample is None:
            sample = p_model.isel({d: 0 for d in p_model.dims if d not in ("lat", "lon")})
    if not parts:
        raise ValueError(f"no forecast year of {model} could be read; nothing to export")
    ds = xr.concat(parts, dim="init_time", join="exact", data_vars="minimal", coords="minimal")
    _coord_attrs(ds, setup.lead_label, _lead_comment(setup.lead_label))
    years = sorted({int(t.year) for t in pd.DatetimeIndex(ds["init_time"].values)})
    extra = {"calendar": calendar, "model": model, "ensemble": ensemble,
             "onset_percentage_threshold": kw["onset_percentage_threshold"] if ensemble else None,
             "members": kw.get("members"),
             "comment": "scored is 1 where ROMP scores the case: the cell has an observed onset and the init is "
                        "before it; values are given for every init and cell"}
    ds.attrs = _provenance(dic, kw, model, "ensemble forecast" if ensemble else "deterministic forecast",
                           thresholds, _mask_summary(dic, sample), years, extra)
    return ds


def _model_year(out: dict, ensemble: bool, threshold: float) -> xr.Dataset:
    """One year of a model as a Dataset from the dict _detect_forecast returns."""
    res, calendar = out["result"], out["calendar"]
    dims = tuple(res["status"].dims)
    spatial = ("lat", "lon")
    status = res["status"].transpose(*dims).values
    found = status == Status.ONSET
    days = np.where(found, _days(res["onset_date"].transpose(*dims).values, calendar), np.nan)
    init_days = _days(out["init"], calendar)
    shape = (-1,) + (1,) * (len(dims) - 1)
    variables = {
        "onset_date": _date_var(days, dims, calendar, "forecast rainy-season onset date",
                                "date label of the onset day; set only where status is ONSET (a PRIOR search is "
                                "reported as no onset, as scored); no onset: masked"),
        "onset_doy": _float_var(_doy(days, calendar), dims, "day of the calendar year of the onset", "1",
                                "1-366 in the onset date's own year; NaN where there is no onset"),
        "onset_lead": _float_var(np.where(found, res["onset_lead"].transpose(*dims).values, np.nan), dims,
                                 "lead of the onset day in days", "",
                                 "1-based forecast lead of the onset day (see the lead coordinate for lead_label); "
                                 "<= 0 means back-dated into the observed history; NaN where there is no onset"),
        "onset_day": _float_var(days - init_days.reshape(shape), dims, "days from the init date to the onset date",
                                "", "as the onset_day column of the scored tables; NaN where there is no onset"),
        "status": status_var(status, dims),
        "scored": xr.DataArray(out["valid"].astype(np.int8), dims=("init_time", *spatial), attrs={
            "long_name": "case is scored by ROMP", "flag_values": np.array([0, 1], dtype=np.int8),
            "flag_meanings": "not_scored scored",
            "comment": "1: the cell has an observed onset and the init is before it"}),
    }
    if ensemble:
        _ensemble_variables(variables, out, res, threshold, calendar, init_days)
    return xr.Dataset(variables, coords={"init_time": res["init_time"].values, "season": res["season"].values,
                                         "lat": res["lat"].values, "lon": res["lon"].values,
                                         **({"member": res["member"].values} if "member" in res.coords else {})})


def _ensemble_variables(variables: dict, out: dict, res: xr.Dataset, threshold: float, calendar: str,
                        init_days: np.ndarray) -> None:
    """Add the ensemble summary (the engine's, and the ensemble onset ROMP scores) to variables, in place."""
    dims = ("init_time", "season", "lat", "lon")
    by_status = res["members_by_status"].transpose("status_code", *dims)
    variables["members_by_status"] = xr.DataArray(
        by_status.values.astype(np.int32), dims=by_status.dims,
        coords={"status_code": by_status["status_code"].values,
                "status_name": ("status_code", by_status["status_name"].values)},
        attrs={**by_status.attrs, "comment": "member counts per status code; the codes are the status_code flags"})
    probability = res["onset_probability"].transpose("init_time", "season", "lead", "lat", "lon")
    variables["onset_probability"] = xr.DataArray(
        probability.values.astype(np.float32), dims=probability.dims, coords={"lead": probability["lead"].values},
        attrs=dict(probability.attrs))
    count, fraction, day, date = ensemble_onset(out["day"], out["init"], threshold)
    has_onset = ~np.isnan(day)
    names = _ensemble_status(out["status"], threshold, has_onset)
    codes = np.vectorize(lambda n: Status[n].value, otypes=[np.int8])(names)
    ensemble_days = _days(date, calendar) if _standard(calendar) else init_days.reshape(-1, 1, 1) + day
    ensemble_days = np.where(has_onset, ensemble_days, np.nan)
    comment = (f"as scored: the ensemble has an onset when at least {threshold:g} of the members have one; its "
               "day after the init is the mean member onset day rounded half to even (this is not the median)")
    variables["ensemble_onset_date"] = _date_var(ensemble_days[:, None], dims, calendar, "ensemble onset date", comment)
    variables["ensemble_onset_doy"] = _float_var(_doy(ensemble_days, calendar)[:, None], dims,
                                                 "day of the calendar year of the ensemble onset", "1")
    variables["ensemble_onset_day"] = _float_var(day[:, None], dims, "days from the init date to the ensemble onset",
                                                 "", comment)
    variables["ensemble_status"] = status_var(codes[:, None], dims, "ensemble onset search outcome")


# --- climatology references -----------------------------------------------------------------------------

def _reference_inputs(dic: dict, model: str):
    """The case kwargs, thresholds and per-year (year, observed onsets, init dates, rain sample) of a climatology reference."""
    kw = case_kwargs(dic, model, reference=True)
    thresholds = load_thresholds(**kw)
    years = []
    for year, rain in _years_loop(kw, lambda y: load_imd_rainfall(y, **kw)):
        years.append((year, detect_observed_onset(rain, thresholds, year, **kw),
                      get_initialization_dates(year, **kw), rain))
    if not years:
        raise ValueError("no observation year could be read; nothing to export")
    return kw, thresholds, years


def _scored(init_dates, obs_da: xr.DataArray) -> np.ndarray:
    """The cases ROMP scores, (init, lat, lon) bool: the cell has an observed onset and the init is before it."""
    obs = obs_da.transpose("lat", "lon").values.astype("datetime64[ns]")
    inits = pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates]).values.astype("datetime64[ns]")
    return ~np.isnat(obs)[None] & (inits[:, None, None] < obs[None])


def _stand_in(obs_da: xr.DataArray) -> xr.DataArray:
    """Observed onsets far in the future, so that the climatology functions return a row for every init and cell."""
    return xr.DataArray(np.full(obs_da.shape, _NO_OBS), coords=obs_da.coords, dims=obs_da.dims)


def _reference_year(table: pd.DataFrame, init_dates, obs_da: xr.DataArray, scored: np.ndarray, season: str) -> xr.Dataset:
    """One year of a climatology reference from the table of its forecast functions (a row per init, lat, lon)."""
    lat, lon = obs_da["lat"].values, obs_da["lon"].values
    shape = (len(init_dates), 1, len(lat), len(lon))
    dims = ("init_time", "season", "lat", "lon")
    date = pd.to_datetime(pd.Series(table["onset_date"].to_numpy(dtype=object))).to_numpy().reshape(shape)
    day = table["onset_day"].to_numpy(dtype=float).reshape(shape)
    days = _days(date.astype("datetime64[ns]"), "standard")
    found = ~np.isnan(days)
    return xr.Dataset(
        {"onset_date": _date_var(days, dims, "standard", "climatological forecast onset date",
                                 "the forecast is a date within the lead window; none: masked"),
         "onset_doy": _float_var(_doy(days, "standard"), dims, "day of the calendar year of the onset", "1"),
         "onset_lead": _float_var(day, dims, "lead of the onset day in days", "",
                                  "days after the init date (lead k is init + k); NaN where there is no onset"),
         "onset_day": _float_var(day, dims, "days from the init date to the onset date"),
         "status": status_var(np.where(found, Status.ONSET, Status.NOT_FOUND), dims),
         "scored": xr.DataArray(scored.astype(np.int8), dims=("init_time", "lat", "lon"), attrs={
             "long_name": "case is scored by ROMP", "flag_values": np.array([0, 1], dtype=np.int8),
             "flag_meanings": "not_scored scored",
             "comment": "1: the cell has an observed onset and the init is before it"})},
        coords={"init_time": pd.DatetimeIndex([pd.to_datetime(t) for t in init_dates]).values,
                "season": [season], "lat": lat, "lon": lon})


def collect_climatology(dic: dict) -> xr.Dataset:
    """The traditional climatology: its onset per cell and, per init, the forecast ROMP scores from it.

    climatological_onset_doy and climatological_onset_date (season, lat, lon) come from
    momp.stats.climatology.compute_climatological_onset (the rounded mean day of the climatology years);
    the per-init onset_date, onset_doy, onset_lead, onset_day, status and scored (init_time, season, lat, lon)
    from compute_climatology_as_forecast (a date within the lead window, else no onset).
    """
    kw, thresholds, years = _reference_inputs(dic, CLIMATOLOGY)
    setup = onset_setup(kw)
    clim_doy = compute_climatological_onset(**kw)
    parts = []
    for year, obs_da, init_dates, _ in years:
        if not len(init_dates):
            continue
        table = compute_climatology_as_forecast(clim_doy, year, init_dates, _stand_in(obs_da), **kw)
        parts.append(_reference_year(table, init_dates, obs_da, _scored(init_dates, obs_da), setup.season))
    ds = xr.concat(parts, dim="init_time", join="exact", data_vars="minimal", coords="minimal")
    doy = clim_doy.transpose("lat", "lon").values.astype(float)[None]
    reference_year = int(years[0][0])
    nominal = np.where(np.isnan(doy), np.nan, (np.datetime64(f"{reference_year}-01-01") - _EPOCH)
                       / np.timedelta64(1, "D") + doy - 1)
    ds["climatological_onset_doy"] = _float_var(doy, ("season", "lat", "lon"), "climatological onset, day of year",
                                               "1", "rounded mean day of the calendar year of the observed onsets "
                                               f"of years_clim {kw['years_clim'][0]}-{kw['years_clim'][-1]}")
    ds["climatological_onset_date"] = _date_var(nominal, ("season", "lat", "lon"), "standard",
                                                "climatological onset date",
                                                f"the day of year placed in the nominal year {reference_year}")
    _coord_attrs(ds, setup.lead_label, "days after the init date: the climatology forecast onset lies at lead k = init + k")
    ds.attrs = _provenance(dic, kw, CLIMATOLOGY, "climatology reference", thresholds,
                           _mask_summary(dic, years[0][3].isel(time=slice(0, 400))), [y[0] for y in years],
                           {"calendar": "standard", "climatology_reference_year": reference_year,
                            "comment": "per-init variables are the climatology as a forecast, as scored; "
                                       "status is ONSET when the climatological date lies within the lead window, "
                                       "else NOT_FOUND; scored is 1 where ROMP scores the case"})
    return ds


def collect_conditional_climatology(dic: dict) -> xr.Dataset:
    """The conditional climatology: per init its median onset and its full onset probability by lead.

    onset_date, onset_doy, onset_lead, onset_day, status and scored (init_time, season, lat, lon) come from
    compute_conditional_climatology_as_forecast (the first lead whose conditional CDF reaches 0.5; no onset
    when it does not, when the cell has no KDE or the date is before earliest_onset_date). onset_probability
    (init_time, season, lead, lat, lon) is that conditional CDF, P(onset on or before the lead | no onset by
    the init), for leads 0 to max_forecast_day (0 at lead 0), NaN for cells without a KDE.
    """
    kw, thresholds, years = _reference_inputs(dic, CONDITIONAL_CLIMATOLOGY)
    setup = onset_setup(kw)
    clim_onset = compute_climatological_onset_dataset(**kw)
    horizon = int(kw["max_forecast_day"])
    parts = []
    for year, obs_da, init_dates, _ in years:
        if not len(init_dates):
            continue
        table = compute_conditional_climatology_as_forecast(clim_onset, year, init_dates, _stand_in(obs_da), **kw)
        part = _reference_year(table, init_dates, obs_da, _scored(init_dates, obs_da), setup.season)
        probability = conditional_lead_probabilities(clim_onset, year, init_dates, obs_da["lat"].values,
                                                     obs_da["lon"].values, horizon, kw.get("climatology_cv"))
        cdf = np.cumsum(probability, axis=-1)
        cdf = np.concatenate([np.where(np.isnan(cdf[..., :1]), np.nan, 0.0), cdf], axis=-1)
        part["onset_probability"] = xr.DataArray(
            cdf.transpose(0, 3, 1, 2)[:, None].astype(np.float32), dims=("init_time", "season", "lead", "lat", "lon"),
            coords={"lead": np.arange(horizon + 1)},
            attrs={"long_name": "probability that the onset is on or before this lead, given none by the init",
                   "comment": "conditional CDF of the KDE of observed onsets over the climatology years; 0 at "
                              "lead 0; NaN for cells with fewer than 10 onset years"})
        parts.append(part)
    ds = xr.concat(parts, dim="init_time", join="exact", data_vars="minimal", coords="minimal")
    _coord_attrs(ds, setup.lead_label, "days after the init date: the conditional onset lies at lead k = init + k")
    configured = dic.get("climatology_cv")
    ds.attrs = _provenance(dic, kw, CONDITIONAL_CLIMATOLOGY, "conditional climatology reference", thresholds,
                           _mask_summary(dic, years[0][3].isel(time=slice(0, 400))), [y[0] for y in years],
                           {"calendar": "standard", "climatology_cv": climatology_cv_mode(kw),
                            "climatology_cv_configured": configured,
                            "comment": "per-init variables are the conditional climatology as scored (median "
                                       "onset = first lead with CDF >= 0.5); climatology_cv is the mode the "
                                       "scoring used; scored is 1 where ROMP scores the case"})
    return ds


# --- entry point --------------------------------------------------------------------------------------

def _collect(dic: dict, source: str) -> xr.Dataset:
    """The Dataset of one source."""
    if source == OBSERVED:
        return collect_observed(dic)
    if source == CLIMATOLOGY:
        return collect_climatology(dic)
    if source == CONDITIONAL_CLIMATOLOGY:
        return collect_conditional_climatology(dic)
    return collect_model(dic, source)


def export_onsets(cfg, sources=None, out=None) -> dict:
    """Write the onset dates of each requested source to NetCDF; return {source: path}.

    cfg is a flat config dict (or SimpleNamespace, momp.config.Config, or the path of a run YAML).
    sources is a list or comma-separated text of "observed", model names, "climatology" and
    "conditional_climatology" (default: observed, every model, and the climatology reference the run uses).
    out is the folder (default dir_out/onsets). Only the detection is run, with the scoring's own functions
    and settings; nothing is scored.
    """
    dic = _as_dict(cfg)
    names = _check_sources(dic, sources)
    folder = Path(out) if out is not None else Path(dic["dir_out"]) / "onsets"
    paths = {}
    for source in names:
        print(f"Exporting onset dates of {source} ...")
        ds = _collect(dic, source)
        path = write_netcdf(ds, folder / SOURCE_FILE.format(source.replace(" ", "_")))
        print(f"  wrote {path}")
        paths[source] = path
    return paths
