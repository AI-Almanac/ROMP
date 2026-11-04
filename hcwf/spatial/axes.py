"""The spatial axes of onset arrays and tables: lat and lon (grid cells) or unit (evaluation units).

Every array of the benchmark has either the two dims ("lat", "lon") or the one dim ("unit", string ids), and every
table has either the two columns lat and lon or the one column unit. These functions are the only place that tells the
two apart, so the pipeline from detection to scores is written once, for any number of spatial axes.

The module has no dependencies on other hcwf modules; `hcwf.spatial.units`, `hcwf.spatial.masks`, `hcwf.io` and the
onset, ensemble and scoring code all use it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "broadcast_rows", "columns_of", "coords_of", "date_strings", "flat_positions", "locate", "snap", "spatial_dims",
]

GRID = ("lat", "lon")
"""The spatial dims of a grid-cell array."""
UNIT = ("unit",)
"""The spatial dims of an evaluation-unit array."""
GRID_TOLERANCE = 0.01
"""How far (in degrees) a table's cell may be from a grid coordinate and still be that cell."""


def spatial_dims(names) -> tuple[str, ...]:
    """Return the spatial dims: ``("unit",)`` when ``names`` hold a unit axis, else ``("lat", "lon")``.

    Parameters
    ----------
    names : iterable of str
        Dims, columns or coordinate names.

    Returns
    -------
    tuple of str
        `UNIT` or `GRID`.
    """
    return UNIT if "unit" in set(names) else GRID


def coords_of(obj, dims=None) -> dict:
    """Return the coordinate values of the spatial axes of a DataArray or Dataset.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The object with the spatial coordinates.
    dims : tuple of str, optional
        The axes to read; by default its own spatial dims (`spatial_dims`).

    Returns
    -------
    dict of str to numpy.ndarray
        The values of each axis, keyed by dim.
    """
    dims = dims or spatial_dims(obj.dims)
    return {d: np.asarray(obj[d].values) for d in dims}


def first_close(values, targets, tolerance=GRID_TOLERANCE):
    """Find the index in ``values`` of the first entry within ``tolerance`` of each target, -1 where there is none.

    Parameters
    ----------
    values : numpy.ndarray of float64, shape (value,)
        The coordinate to search.
    targets : array_like of float, shape (target,)
        The coordinates to look for.
    tolerance : float, default GRID_TOLERANCE
        The largest distance (strictly below) that still counts as a match, in the units of the values (degrees).

    Returns
    -------
    numpy.ndarray of int64, shape (target,)
        The index of the first match of each target, -1 where there is none.
    """
    found = np.full(len(targets), -1, dtype=np.int64)
    for n, target in enumerate(targets):
        close = np.flatnonzero(np.abs(values - target) < tolerance)
        if len(close):
            found[n] = close[0]
    return found


def date_strings(values) -> np.ndarray:
    """Format datetime values as ``YYYY-MM-DD`` strings.

    Parameters
    ----------
    values : array_like of datetime64
        The dates.

    Returns
    -------
    numpy.ndarray of object, shape (date,)
        The strings, in the order of ``values``.
    """
    return pd.DatetimeIndex(values).strftime("%Y-%m-%d").to_numpy(dtype=object)


def columns_of(dims, coords, index) -> dict:
    """Return the table columns of the points at ``index``.

    Parameters
    ----------
    dims : tuple of str
        The spatial dims.
    coords : mapping of str to array_like
        The coordinate values of each dim.
    index : tuple of numpy.ndarray of int
        One index array per spatial dim, one entry per point.

    Returns
    -------
    dict of str to numpy.ndarray
        The coordinate values of the points, keyed by dim.
    """
    return {d: np.asarray(coords[d])[ix] for d, ix in zip(dims, index)}


def locate(da, columns: dict):
    """Find where table rows sit in a DataArray over spatial dims.

    Rows are located by ``lat`` and ``lon`` within `GRID_TOLERANCE` of a grid coordinate, or by ``unit`` equal to an id.

    Parameters
    ----------
    da : xarray.DataArray
        An array over the spatial dims (its coordinates are searched).
    columns : dict of str to array_like
        The values of each row for the dims of ``da``: ``{dim: values}``.

    Returns
    -------
    index : tuple of numpy.ndarray of int64
        One array per dim, -1 where a row has no cell.
    located : numpy.ndarray of bool
        Which rows have a cell on every dim.
    """
    dims = tuple(columns)
    if dims == UNIT:
        ids = pd.Index(np.asarray(da["unit"].values).astype(str))
        found = ids.get_indexer(np.asarray(columns["unit"]).astype(str))
        return (found,), found >= 0
    index = []
    for d in dims:
        values = np.asarray(columns[d], dtype=float)
        unique, back = np.unique(values, return_inverse=True)
        index.append(first_close(da[d].values, unique)[back])
    located = np.logical_and.reduce([ix >= 0 for ix in index])
    return tuple(index), located


def flat_positions(points: pd.DataFrame, dims, coords) -> np.ndarray:
    """Find the position of each row of ``points`` in the raveled array over ``dims`` with ``coords`` (exact match).

    Parameters
    ----------
    points : pandas.DataFrame
        A table with one column per dim.
    dims : tuple of str
        The spatial dims.
    coords : mapping of str to array_like
        The coordinate values of each dim.

    Returns
    -------
    numpy.ndarray of int64, shape (row,)
        The position in the C-order raveled array, -1 where the point is absent.
    """
    if dims == UNIT:
        return pd.Index(np.asarray(coords["unit"]).astype(str)).get_indexer(points["unit"].to_numpy().astype(str))
    grid = pd.MultiIndex.from_product([coords[d] for d in dims])
    return grid.get_indexer(pd.MultiIndex.from_frame(points[list(dims)]))


def snap(values, dims, coords, lead: int = 0):
    """Re-index the last ``len(dims)`` axes of ``values`` by the first coordinate within tolerance of each (grids only).

    The identity unless a coordinate repeats within the tolerance; units are matched by id and need no snapping.

    Parameters
    ----------
    values : numpy.ndarray
        An array whose last axes are the spatial dims.
    dims : tuple of str
        The spatial dims.
    coords : mapping of str to numpy.ndarray
        The coordinate values of each dim.
    lead : int, default 0
        The number of leading axes (such as year) left alone.

    Returns
    -------
    numpy.ndarray
        The snapped array (``values`` itself for units).
    """
    if dims == UNIT:
        return values
    index = [first_close(coords[d], coords[d]) for d in dims]
    if lead == 0:
        return values[np.ix_(*index)]
    out = values
    for n, ix in enumerate(index):
        out = np.take(out, ix, axis=lead + n)
    return out


def broadcast_rows(rows, ndim: int):
    """Shape one value per leading row to broadcast against arrays of ``ndim`` dims.

    Parameters
    ----------
    rows : array_like
        One value per leading row.
    ndim : int
        The number of dims of the arrays it will broadcast against.

    Returns
    -------
    numpy.ndarray, shape (row, 1, ..., 1)
        The values with ``ndim - 1`` trailing axes of length 1.
    """
    return np.asarray(rows).reshape((-1,) + (1,) * (ndim - 1))
