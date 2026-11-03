"""Onset thresholds read from files: one reader for per-unit tables and for gridded NetCDF fields, and the join to units.

A threshold table is a pandas DataFrame with `onset_thresh` and either an `id` (a unit id) or `lat` and `lon`
(a grid cell). `read_threshold_table` reads one from a CSV (keyed by a unit id column, adm3_name, id, or lat/lon),
a NetCDF file (flattened lat, lon and value variables) or a MATLAB `.mat` file (value over the lat x lon product).
`read_threshold_grid` reads a NetCDF field on the grid, as HCWF's `threshold_files` do. Both apply the optional
`ThresholdScale` (clip(t * factor + offset, min, max)) through `scale_values`. `unit_thresholds` joins a table (or a
constant, or nothing) to the ids of the units, normalising ids and refusing a table that does not cover them.

Thresholds computed from the observations are `hcwf.onset.climatology.compute_threshold`; this module is the
file side. The readers port onset_blending's `core/reference.py` (`read_thresholds`, `_scaled`, `_threshold_csv`,
`Reference.unit_thresholds`) operation for operation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.spatial import ids

NETCDF_SUFFIXES = (".nc", ".nc4", ".netcdf")


@dataclass(frozen=True)
class ThresholdScale:
    """clip(threshold * factor + offset, min, max); a bound left None is not applied."""

    factor: float = 1.0
    offset: float = 0.0
    min: float | None = None
    max: float | None = None


def scale_values(values, scale: ThresholdScale | None):
    """The values as clip(v * factor + offset, min, max) (the values themselves without a scale).

    A DataArray keeps its dims, coords and attrs; NaN stays NaN (np.maximum and np.minimum propagate it).
    """
    if scale is None:
        return values
    out = np.asarray(values, dtype=float) * float(scale.factor) + float(scale.offset)
    if scale.min is not None:
        out = np.maximum(out, float(scale.min))
    if scale.max is not None:
        out = np.minimum(out, float(scale.max))
    if isinstance(values, xr.DataArray):
        return values.copy(data=out)
    return out


def scale_table(table: pd.DataFrame, scale: ThresholdScale | None) -> pd.DataFrame:
    """The table with its `onset_thresh` scaled (the table itself without a scale)."""
    if scale is None:
        return table
    return table.assign(onset_thresh=scale_values(table["onset_thresh"].astype(float).values, scale))


def read_threshold_grid(path, var: str, *, select=None, scale: ThresholdScale | None = None) -> xr.DataArray:
    """The threshold field `var` of a NetCDF file as a loaded DataArray, as HCWF's `threshold_files` read it.

    `select`, if given, is called on the DataArray before it is loaded (HCWF passes its region selection, so only the
    region is read); `scale` is applied to the loaded values.
    """
    with xr.open_dataset(path) as ds:
        field = ds[var]
        if select is not None:
            field = select(field)
        return scale_values(field.load(), scale)


def read_threshold_table(path, *, scale: ThresholdScale | None = None, id_col: str | None = None,
                         value_col: str = "onset_thresh", lat_col: str = "lat", lon_col: str = "lon",
                         lat_var: str = "lat", lon_var: str = "lon", value_var: str = "MWmean",
                         mat_lat_name: str = "lat", mat_lon_name: str = "lon",
                         mat_value_name: str = "onset.day.thres") -> pd.DataFrame:
    """A threshold table (id or lat/lon, and onset_thresh) from a CSV, NetCDF or MATLAB file, optionally scaled.

    CSV: the unit id is `id_col` when given (it must exist), else an `adm3_name` column, else `id`, else the table is
    keyed by `lat_col` and `lon_col`; `value_col` holds the threshold. Values stay text until a join reads them.
    NetCDF (.nc, .nc4, .netcdf): the variables `lat_var`, `lon_var` and `value_var`, each flattened (netCDF4 reading,
    so masked entries show their stored value); .mat: the vectors `mat_lat_name` and `mat_lon_name` give every
    (lat, lon) pair and `mat_value_name` the values flattened over them.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold file not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix in NETCDF_SUFFIXES:
        import netCDF4

        with netCDF4.Dataset(path) as ds:
            table = pd.DataFrame({
                "lat": np.array(ds[lat_var][:]).flatten(),
                "lon": np.array(ds[lon_var][:]).flatten(),
                "onset_thresh": np.array(ds[value_var][:]).flatten(),
            }).drop_duplicates()
    elif suffix == ".mat":
        import scipy.io

        content = scipy.io.loadmat(path)
        lats = np.array(content[mat_lat_name]).flatten()
        lons = np.array(content[mat_lon_name]).flatten()
        table = pd.DataFrame([(a, b) for a in lats for b in lons], columns=["lat", "lon"])
        table["onset_thresh"] = np.array(content[mat_value_name]).flatten()
        table = table.drop_duplicates()
    else:
        table = _read_csv(path, id_col, value_col, lat_col, lon_col)
    return scale_table(table, scale)


def _read_csv(path: str, id_col: str | None, value_col: str, lat_col: str, lon_col: str) -> pd.DataFrame:
    """The CSV's id (or lat, lon) and onset_thresh columns, as text, without repeated rows."""
    table = pd.read_csv(path, dtype=str)
    if id_col:
        if id_col not in table.columns:
            raise ValueError(f"Configured thresholds.id_col '{id_col}' not found. Found: {table.columns.tolist()}")
        return table.rename(columns={id_col: "id", value_col: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    if "adm3_name" in table.columns:
        return table.rename(columns={"adm3_name": "id", value_col: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    if "id" in table.columns:
        return table.rename(columns={value_col: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    table = table.rename(columns={lat_col: "lat", lon_col: "lon", value_col: "onset_thresh"})
    return table[["lat", "lon", "onset_thresh"]].drop_duplicates()


def unit_thresholds(thresholds, unit_ids, explicit: ids.GridIdConvention | None = None) -> np.ndarray:
    """Each unit's threshold (NaN without one): `thresholds` is None, a constant, a table or an array over "unit",
    joined by unit id.

    A table keyed by lat/lon is read under the units' grid-id convention and must cover them; ids are normalised
    (stripped text); one id with two values, or a unit the table lacks, is an error.
    """
    unit_ids = np.asarray(unit_ids)
    if thresholds is None:
        return np.full(len(unit_ids), np.nan)
    if isinstance(thresholds, xr.DataArray) and "unit" in thresholds.dims:
        thresholds = pd.DataFrame({"id": np.asarray(thresholds["unit"].values).astype(str),
                                   "onset_thresh": np.asarray(thresholds.values, dtype=float)})
    if not isinstance(thresholds, pd.DataFrame):
        return np.full(len(unit_ids), float(thresholds))
    convention = ids.resolve(explicit, authoritative_ids=pd.Series(unit_ids, dtype=str),
                             context="rainfall/threshold ID handoff")
    table = thresholds.copy()
    table.columns = table.columns.str.lower()
    if "id" not in table.columns:
        if "adm3_name" in table.columns:
            table["id"] = table["adm3_name"]
        elif {"lat", "lon"} <= set(table.columns):
            coords = ids.with_id_column(table[["lat", "lon"]], explicit=explicit, convention=convention,
                                        context="latitude/longitude IDs")
            table["id"] = coords["id"].to_numpy()
        else:
            raise ValueError("Thresholds table must have 'id', 'adm3_name', or ('lat', 'lon').")
    table["id"] = ids.normalize(table["id"], context="threshold IDs")
    table["onset_thresh"] = table["onset_thresh"].astype(float)
    table = table[["id", "onset_thresh"]].drop_duplicates()
    conflicting = table["id"].duplicated(keep=False)
    if conflicting.any():
        raise ValueError(f"Threshold IDs have conflicting values: {', '.join(table.loc[conflicting, 'id'].unique()[:10])}")
    by_id = table.set_index("id")["onset_thresh"]
    wanted = pd.Index(pd.unique(np.asarray(unit_ids, dtype=str)))
    missing = wanted[~wanted.isin(by_id.index)]
    if len(missing):
        raise ValueError(
            f"Threshold table is missing {len(missing)} of {len(wanted)} rainfall IDs after domain "
            f"filtering. Examples: {', '.join(map(str, missing[:10]))}"
        )
    return by_id.reindex(np.asarray(unit_ids, dtype=str)).to_numpy(dtype=float)
