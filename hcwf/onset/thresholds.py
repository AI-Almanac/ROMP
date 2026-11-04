"""Onset thresholds read from files: one reader for per-unit tables and for gridded NetCDF fields, and the join to units.

A threshold table is a `pandas.DataFrame` with ``onset_thresh`` and either an ``id`` (a unit id) or ``lat`` and ``lon``
(a grid cell). `read_threshold_table` reads one from a CSV (keyed by a unit id column, ``id``, ``adm3_name``, or
lat/lon), a NetCDF file (flattened lat, lon and value variables) or a MATLAB ``.mat`` file (value over the lat x lon
product). `read_threshold_grid` reads a NetCDF field on the grid, as HCWF's ``threshold_files`` do. Both apply the
optional `ThresholdScale` (``clip(t * factor + offset, min, max)``) through `scale_values`. `unit_thresholds` joins a
table (or a constant, or nothing) to the ids of the units, normalising ids and refusing a table that does not cover
them.

Thresholds computed from the observations are `hcwf.onset.climatology.compute_threshold`; this module is the file
side. The readers port onset_blending's ``core/reference.py`` (``read_thresholds``, ``_scaled``, ``_threshold_csv``,
``Reference.unit_thresholds``) operation for operation.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import memo
from hcwf.spatial import ids

NETCDF_SUFFIXES = (".nc", ".nc4", ".netcdf")
"""The file suffixes `read_threshold_table` reads as NetCDF."""


@dataclass(frozen=True)
class ThresholdScale:
    """A linear rescaling of thresholds, ``clip(threshold * factor + offset, min, max)``; a bound left None is not applied."""

    factor: float = 1.0
    """The factor the threshold is multiplied by."""
    offset: float = 0.0
    """The amount added after the multiplication."""
    min: float | None = None
    """The lower bound, or None for none."""
    max: float | None = None
    """The upper bound, or None for none."""


def scale_values(values, scale: ThresholdScale | None):
    """Return the values as ``clip(v * factor + offset, min, max)``, or the values themselves without a scale.

    A `xarray.DataArray` keeps its dims, coords and attrs; NaN stays NaN (`numpy.maximum` and `numpy.minimum`
    propagate it).

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        The thresholds, converted to ``float64`` when scaled.
    scale : ThresholdScale or None
        The scaling; None returns ``values`` unchanged.

    Returns
    -------
    numpy.ndarray of float64 or xarray.DataArray
        The scaled values, of the shape of ``values``; a ``DataArray`` when ``values`` is one.
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
    """Return the table with its ``onset_thresh`` scaled, or the table itself without a scale.

    Parameters
    ----------
    table : pandas.DataFrame
        The table, with an ``onset_thresh`` column.
    scale : ThresholdScale or None
        The scaling; None returns ``table`` unchanged.

    Returns
    -------
    pandas.DataFrame
        The table, with ``onset_thresh`` as ``float64`` after scaling.
    """
    if scale is None:
        return table
    return table.assign(onset_thresh=scale_values(table["onset_thresh"].astype(float).values, scale))


def read_threshold_grid(path, var: str, *, select=None, scale: ThresholdScale | None = None) -> xr.DataArray:
    """Read the threshold field of a NetCDF file as a loaded array, as HCWF's ``threshold_files`` read it.

    Parameters
    ----------
    path : str or pathlib.Path
        The NetCDF file.
    var : str
        The name of the threshold variable.
    select : callable, optional
        Called on the `xarray.DataArray` before it is loaded and returns the part to read (HCWF passes its region
        selection, so only the region is read).
    scale : ThresholdScale, optional
        Applied to the loaded values.

    Returns
    -------
    xarray.DataArray
        The loaded, scaled field.
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
    """Read a threshold table (id or lat/lon, and ``onset_thresh``) from a CSV, NetCDF or MATLAB file.

    CSV: the unit id is ``id_col`` when given (a `ValueError` is raised if the CSV has no such column), else an ``id`` column, else ``adm3_name``, else the table
    is keyed by ``lat_col`` and ``lon_col``; ``value_col`` holds the threshold. Values stay text until a join reads them.
    NetCDF (``.nc``, ``.nc4``, ``.netcdf``): the variables ``lat_var``, ``lon_var`` and ``value_var``, each flattened
    (netCDF4 reading, so masked entries show their stored value). ``.mat``: the vectors ``mat_lat_name`` and
    ``mat_lon_name`` give every (lat, lon) pair and ``mat_value_name`` the values flattened over them.

    Parameters
    ----------
    path : str or pathlib.Path
        The file; its suffix picks the reader.
    scale : ThresholdScale, optional
        Applied to ``onset_thresh``.
    id_col : str, optional
        The CSV column holding the unit id; None looks for ``id``, ``adm3_name`` or lat/lon.
    value_col : str, default "onset_thresh"
        The CSV column holding the threshold.
    lat_col : str, default "lat"
        The CSV latitude column, used when the table has no unit id.
    lon_col : str, default "lon"
        The CSV longitude column, used when the table has no unit id.
    lat_var : str, default "lat"
        The NetCDF latitude variable.
    lon_var : str, default "lon"
        The NetCDF longitude variable.
    value_var : str, default "MWmean"
        The NetCDF threshold variable.
    mat_lat_name : str, default "lat"
        The ``.mat`` latitude vector.
    mat_lon_name : str, default "lon"
        The ``.mat`` longitude vector.
    mat_value_name : str, default "onset.day.thres"
        The ``.mat`` threshold array.

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` (or ``lat`` and ``lon``) and ``onset_thresh``, without repeated rows; ``onset_thresh`` is text
        for a CSV that is not scaled, ``float64`` otherwise.

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist.
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
    """Read the id (or lat, lon) and ``onset_thresh`` columns of a CSV, as text, without repeated rows.

    Parameters
    ----------
    path : str
        The CSV file.
    id_col : str or None
        The column holding the unit id; None looks for ``id``, ``adm3_name`` or lat/lon.
    value_col : str
        The column holding the threshold.
    lat_col : str
        The latitude column, used when the table has no unit id.
    lon_col : str
        The longitude column, used when the table has no unit id.

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` (or ``lat`` and ``lon``) and ``onset_thresh``, all text.

    Raises
    ------
    ValueError
        ``id_col`` is not a column of the CSV.
    """
    table = pd.read_csv(path, dtype=str)
    if id_col:
        if id_col not in table.columns:
            raise ValueError(f"Configured thresholds.id_col '{id_col}' not found. Found: {table.columns.tolist()}")
        return table.rename(columns={id_col: "id", value_col: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    key = ids.unit_id_column(table.columns)
    if key is not None:
        return table.rename(columns={key: "id", value_col: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    table = table.rename(columns={lat_col: "lat", lon_col: "lon", value_col: "onset_thresh"})
    return table[["lat", "lon", "onset_thresh"]].drop_duplicates()


_joined = memo("onset.thresholds.joined", maxsize=16, share=True)
"""The tables and arrays already joined to units this run, as ``{fingerprint: thresholds}``.

Joining a table keyed by lat/lon formats every cell's grid id, which every year and file of a run would repeat; the
workers start with the parent's.
"""


def _fingerprint(thresholds, unit_ids, explicit) -> str:
    """Return a digest of what a join depends on: the table or array, the units and the explicit convention.

    Parameters
    ----------
    thresholds : pandas.DataFrame or xarray.DataArray
        The table, or the array over ``"unit"``.
    unit_ids : numpy.ndarray
        The unit ids.
    explicit : hcwf.spatial.ids.GridIdConvention or None
        The explicit grid-id convention.

    Returns
    -------
    str
        A hexadecimal digest.
    """
    digest = hashlib.blake2b(digest_size=16)
    if isinstance(thresholds, pd.DataFrame):
        digest.update(repr(list(thresholds.columns)).encode())
        digest.update(pd.util.hash_pandas_object(thresholds, index=False).to_numpy().tobytes())
    else:
        digest.update(np.asarray(thresholds["unit"].values).astype(str).tobytes())
        digest.update(np.ascontiguousarray(thresholds.values, dtype=float).tobytes())
    digest.update(str(unit_ids.shape).encode())
    digest.update(unit_ids.astype(str).tobytes())
    digest.update(repr(explicit).encode())
    return digest.hexdigest()


def unit_thresholds(thresholds, unit_ids, explicit: ids.GridIdConvention | None = None) -> np.ndarray:
    """Return each unit's threshold (NaN without one), joined by unit id.

    A table keyed by lat/lon is read under the units' grid-id convention and must cover them; ids are normalised
    (stripped text); one id with two values, or a unit the table lacks, is an error. A join is made once per run.

    Parameters
    ----------
    thresholds : None, float, pandas.DataFrame or xarray.DataArray
        No threshold, a constant, a table (see `read_threshold_table`) or an array over ``"unit"``.
    unit_ids : array_like of str, shape (unit,)
        The ids of the units.
    explicit : hcwf.spatial.ids.GridIdConvention, optional
        The grid-id convention to use for a table keyed by lat/lon; None detects it from ``unit_ids``.

    Returns
    -------
    numpy.ndarray of float64, shape (unit,)
        The threshold of each unit, in the order of ``unit_ids``; all NaN for None.
    """
    unit_ids = np.asarray(unit_ids)
    if isinstance(thresholds, pd.DataFrame) or (isinstance(thresholds, xr.DataArray) and "unit" in thresholds.dims):
        key = _fingerprint(thresholds, unit_ids, explicit)
        joined = _joined.get(key)
        if joined is None:
            joined = _joined.setdefault(key, _join_units(thresholds, unit_ids, explicit))
        return joined.copy()
    return _join_units(thresholds, unit_ids, explicit)


def _join_units(thresholds, unit_ids, explicit) -> np.ndarray:
    """Join thresholds to units, as `unit_thresholds` does without the memo.

    Parameters
    ----------
    thresholds : None, float, pandas.DataFrame or xarray.DataArray
        No threshold, a constant, a table or an array over ``"unit"``.
    unit_ids : numpy.ndarray of str, shape (unit,)
        The ids of the units.
    explicit : hcwf.spatial.ids.GridIdConvention or None
        The grid-id convention for a table keyed by lat/lon.

    Returns
    -------
    numpy.ndarray of float64, shape (unit,)
        The threshold of each unit, in the order of ``unit_ids``.

    Raises
    ------
    ValueError
        The table has no ``id``, ``adm3_name`` or lat/lon columns, an id has conflicting values, or the table lacks
        some of the units.
    """
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
        if ids.unit_id_column(table.columns) is not None:
            table["id"] = table[ids.unit_id_column(table.columns)]
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
