"""Everything a run writes besides figures: the score tables (CSV), the spatial-metrics and climatological-onset NetCDF
files, the per-unit tables and GeoJSON, and the CF NetCDF files of onset dates (`runs.export` collects those).

The tables themselves are computed in hcwf.metrics.tables; this module only writes (and, for onset files, reads) them.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.metrics.tables import score_tables
from hcwf.naming import tuple_to_str
from hcwf.onset.model import Status
from hcwf.runs.folder import file_name

log = logging.getLogger(__name__)

DATE_UNITS = "days since 1970-01-01"
DATE_FILL = -2147483648
EPOCH = np.datetime64("1970-01-01")
_MAX_CHUNK = 2_000_000
_COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}


# --- score tables and spatial metrics -----------------------------------------------------------------

def save_score_results(score_results, case, paths, output):
    """Save overall and binned skill scores of one case to CSV files in output.data_dir (named by `paths`); return
    (binned_data, overall_scores_nested)."""
    model, verification_window = case.model.name, case.window.span
    overall_df, binned_df, binned_data, overall_scores_nested = score_tables(score_results, model,
                                                                             verification_window)
    window_str = tuple_to_str(verification_window)
    dir_out = output.data_dir
    overall_filename = os.path.join(dir_out, file_name(paths, "overall", model, window_str))
    overall_df.to_csv(overall_filename, index=False)
    log.info(f"Saved overall scores to '{overall_filename}'")
    log.info(overall_df)
    binned_filename = os.path.join(dir_out, file_name(paths, "binned", model, window_str))
    binned_df.to_csv(binned_filename, index=False)
    log.info(f"Saved binned scores to '{binned_filename}'")
    log.info(binned_df)
    return binned_data, overall_scores_nested


def with_unit_labels(obj, units):
    """A DataArray or Dataset over "unit" with each unit's name and centroid (unit_name, lat, lon) as coordinates.

    Returned as it is when there is no "unit" dimension. `units` is the observations' UnitsSpec (or their ObsSpec).
    The names, centroids and polygons are the run's own
    (hcwf.io.input.unit_table); a unit without a name is called by its id.
    """
    if "unit" not in obj.dims:
        return obj
    from hcwf.spatial.onto_units import unit_table

    table = unit_table(units, obj["unit"].values)
    names = table.column("name")
    return obj.assign_coords(unit_name=("unit", table.ids if names is None else np.asarray(names).astype(str)),
                             lat=("unit", table.lat), lon=("unit", table.lon))


def save_unit_tables(ds, units, stem):
    """Write a per-unit Dataset as the table `stem`.csv (unit_id, name, centroid lat and lon, one column per variable)
    and, for polygon units, as the GeoJSON `stem`.geojson (the same values on the polygons). Returns the paths."""
    table = ds.to_dataframe().reset_index().rename(columns={"unit": "unit_id", "unit_name": "name"})
    table = table[["unit_id", "name", "lat", "lon", *[c for c in table.columns if c not in ("unit_id", "name", "lat", "lon")]]]
    written = [f"{stem}.csv"]
    table.to_csv(written[0], index=False)
    from hcwf.spatial.onto_units import unit_table

    polygons = unit_table(units, ds["unit"].values)
    if polygons.geometry is not None:
        import geopandas as gpd

        frame = gpd.GeoDataFrame(table, geometry=polygons.geometry, crs="EPSG:4326")
        written.append(f"{stem}.geojson")
        frame.to_file(written[1], driver="GeoJSON")
    return written


def netcdf_attrs(attrs: dict, *, none: str = "drop") -> dict:
    """Attributes NetCDF can hold: text, numbers and number arrays; bools become 0/1, dicts JSON text, lists and tuples
    comma-joined text, paths text. A None is dropped (`none="drop"`) or written as the text "None" (`none="text"`)."""
    out = {}
    for key, value in attrs.items():
        if value is None:
            if none == "text":
                out[key] = str(value)
            continue
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        elif isinstance(value, dict):
            value = json_text(value)
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        elif isinstance(value, Path):
            value = str(value)
        out[key] = value
    return out


def save_metrics_to_netcdf(spatial_metrics, case, paths, output, ev, units, desc_dict=None, fname='spatial_metrics',
                           allowed_attrs=('model', 'years', 'tolerance_days', 'verification_window',
                                          'max_forecast_day', 'lead_window')):
    """
    Create an xarray Dataset from spatial_metrics, attach global attributes,
    and save to a NetCDF file.

    Parameters
    ----------
    spatial_metrics : dict
        Dictionary of data variables for xr.Dataset
    case : Case
        The case the metrics are of: its model, window and name are the file's attributes and name
    paths : RunPaths or None
        Names the file (short or historical)
    output : OutputSpec
        The file is saved in output.data_dir
    ev : EvaluationSpec
        Source of the years, max_forecast_day and lead_window attributes
    units : UnitsSpec or None
        The observations' evaluation units (labels and polygons of a field over "unit")
    desc_dict : dict, optional
        Extra global attributes added after the case-derived ones
    fname : str
        File name prefix: "spatial_metrics" is named by run_folder.file_name, any other as dir/{fname}_{case_name}.nc
    """

    ds = with_unit_labels(xr.Dataset(spatial_metrics), units)

    attrs = {'model': case.model.name, 'years': ev.years, 'tolerance_days': case.window.tolerance,
             'verification_window': case.window.span, 'max_forecast_day': ev.max_forecast_day,
             'lead_window': ev.lead_window}
    if fname == 'spatial_metrics':
        fout = os.path.join(output.data_dir, file_name(paths, "spatial", case.name))
    else:
        fout = os.path.join(output.data_dir, f"{fname}_{case.name}.nc")

    clean_attrs = netcdf_attrs({key: attrs.get(key) for key in allowed_attrs}, none="text")

    ds.attrs.update(clean_attrs)
    if 'n_back_dated' in ds:
        ds.attrs['n_back_dated_total'] = int(ds['n_back_dated'].sum())

    if desc_dict is not None:
        ds.attrs.update(desc_dict)

    from hcwf.spatial.masks import mask_attrs

    ds.attrs.update(mask_attrs())   # nothing unless a region mask is in use

    ds.to_netcdf(fout)

    log.info(f"{fname} saved to: {fout}")
    if "unit" in ds.dims:   # per evaluation unit: also a table keyed by unit id, and the polygons with the values
        for path in save_unit_tables(ds, units, os.path.splitext(fout)[0]):
            log.info(f"{fname} (per unit) saved to: {path}")


def save_climatology_onset(onset_da_dict, paths, output, units, years_clim):
    """Write the climatological onset (day of year) of the reference run to NetCDF, over lat and lon or over the units.

    onset_da_dict is the reference case's {year: onset}: every year holds the same climatological field.
    """
    from hcwf.naming import tuple_to_str_range
    from hcwf.spatial.masks import mask_attrs

    fout = os.path.join(output.data_dir, file_name(paths, "climatology", tuple_to_str_range(years_clim)))
    climatological_onset_doy = with_unit_labels(next(iter(onset_da_dict.values())), units)
    climatological_onset_doy.attrs["years"] = years_clim
    climatological_onset_doy.attrs.update(mask_attrs())   # nothing unless a region mask is in use
    climatological_onset_doy.to_netcdf(fout)


# --- CF NetCDF files of onset dates -------------------------------------------------------------------

def days_of(values, calendar: str) -> np.ndarray:
    """Days since 1970-01-01 (float, NaN for none) of datetime64 values or cftime objects (None for none)."""
    values = np.asarray(values)
    if values.dtype.kind == "M":
        days = (values.astype("datetime64[D]") - EPOCH) / np.timedelta64(1, "D")
        return np.where(np.isnat(values), np.nan, days).astype(float)
    import cftime

    out = np.full(values.shape, np.nan)
    present = np.array([v is not None for v in values.ravel()], dtype=bool).reshape(values.shape)
    if present.any():
        out[present] = cftime.date2num(list(values[present]), DATE_UNITS, calendar=calendar)
    return out


def to_dates(days, calendar: str) -> np.ndarray:
    """datetime64[ns] (NaT for none) or, for a non-standard calendar, cftime objects (None) of day numbers."""
    days = np.asarray(days, dtype=float)
    present = ~np.isnan(days)
    if standard_calendar(calendar):
        whole = np.where(present, days, 0).astype("int64").astype("timedelta64[D]")
        return np.where(present, (EPOCH + whole).astype("datetime64[ns]"), np.datetime64("NaT", "ns"))
    import cftime

    out = np.full(days.shape, None, dtype=object)
    if present.any():
        out[present] = cftime.num2date(days[present], DATE_UNITS, calendar=calendar)
    return out


def standard_calendar(calendar: str) -> bool:
    """Whether dates in this calendar are numpy datetimes."""
    return calendar in ("standard", "gregorian", "proleptic_gregorian")


def day_of_year(days, calendar: str) -> np.ndarray:
    """Day of the calendar year (1-366) of day numbers, NaN for none, counted in the date's own year."""
    dates = to_dates(days, calendar)
    if standard_calendar(calendar):
        return pd.DatetimeIndex(dates.ravel()).dayofyear.to_numpy(dtype=float).reshape(dates.shape)
    return np.array([np.nan if d is None else d.dayofyr for d in dates.ravel()], dtype=float).reshape(dates.shape)


def date_var(days, dims, calendar: str, long_name: str, comment: str = "") -> xr.DataArray:
    """A CF date variable: day numbers with NaN for none, written as int32 with a fill value (so none is masked)."""
    attrs = {"long_name": long_name, "units": DATE_UNITS, "calendar": "standard" if standard_calendar(calendar) else calendar}
    if comment:
        attrs["comment"] = comment
    return xr.DataArray(np.asarray(days, dtype=float), dims=dims, attrs=attrs)


def float_var(values, dims, long_name: str, units: str = "", comment: str = "") -> xr.DataArray:
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


def coord_attrs(ds: xr.Dataset, lead_label: str | None, lead_comment: str) -> None:
    """Attach CF attributes to the coordinates present in ds."""
    attrs = {
        "lat": {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude", "axis": "Y"},
        "lon": {"units": "degrees_east", "standard_name": "longitude", "long_name": "longitude", "axis": "X"},
        "unit": {"long_name": "evaluation unit id", "cf_role": "timeseries_id"},
        "unit_name": {"long_name": "evaluation unit name"},
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


def lead_comment(label: str | None) -> str:
    """Describe which date a lead number stands for."""
    if label == "start":
        return ("lead_label 'start': lead k is the forecast day dated init + k - 1; lead 1 is the init day. "
                "Leads <= 0 are onsets back-dated into the observed history.")
    if label == "end":
        return ("lead_label 'end': lead k is the forecast day dated init + k. Leads <= 0 are onsets back-dated "
                "into the observed history.")
    return "lead k counted from the first forecast day (lead 1); leads <= 0 are onsets back-dated into the observed history."


def json_text(value) -> str:
    """JSON text of a value (paths and unknown objects as text)."""
    return json.dumps(value, default=str, sort_keys=True)


def _chunks(var: xr.DataArray) -> tuple[int, ...]:
    """Chunk sizes: whole map, lead and status axes per chunk, one init, year, season (and member for big grids)."""
    whole = ("lat", "lon", "unit", "lead", "status_code", "member")
    chunk = [min(n, 512) if d in whole else 1 for d, n in zip(var.dims, var.shape)]
    while np.prod(chunk) > _MAX_CHUNK:
        for dim in ("member", "lead", "status_code", "unit", "lon", "lat"):
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
    (`to_dates`) and everything else by xarray. Standard calendars open as datetime64 with NaT. The
    `lead` coordinate has units "days": it is opened as plain numbers, not timedeltas.
    """
    raw = xr.open_dataset(path, decode_times=False, decode_timedelta=False)
    dates = [n for n, v in raw.data_vars.items() if str(v.attrs.get("units", "")).startswith("days since")
             and not standard_calendar(v.attrs.get("calendar", "standard"))]
    ds = xr.decode_cf(raw.drop_vars(dates), decode_timedelta=False)
    for name in dates:
        ds[name] = (raw[name].dims, to_dates(raw[name].values, raw[name].attrs["calendar"]), {
            k: v for k, v in raw[name].attrs.items() if k not in ("units", "calendar")})
    ds.attrs.update(raw.attrs)
    return ds
