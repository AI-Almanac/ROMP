"""The spatial axes of onset arrays and tables: lat and lon (grid cells) or unit (evaluation units).

Every array of the benchmark has either the two dims ("lat", "lon") or the one dim ("unit", string ids), and every
table has either the two columns lat and lon or the one column unit. These functions are the only place that tells the
two apart, so the pipeline from detection to scores is written once, for any number of spatial axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRID = ("lat", "lon")
UNIT = ("unit",)
GRID_TOLERANCE = 0.01
"""How far (in degrees) a table's cell may be from a grid coordinate and still be that cell."""


def spatial_dims(names) -> tuple[str, ...]:
    """("unit",) when `names` (dims, columns or coordinates) hold a unit axis, else ("lat", "lon")."""
    return UNIT if "unit" in set(names) else GRID


def coords_of(obj, dims=None) -> dict:
    """{dim: values} of the spatial axes of a DataArray or Dataset (`dims` default: its own spatial dims)."""
    dims = dims or spatial_dims(obj.dims)
    return {d: np.asarray(obj[d].values) for d in dims}


def first_close(values, targets, tolerance=GRID_TOLERANCE):
    """Index in values of the first entry within tolerance of each target, -1 where there is none."""
    found = np.full(len(targets), -1, dtype=np.int64)
    for n, target in enumerate(targets):
        close = np.flatnonzero(np.abs(values - target) < tolerance)
        if len(close):
            found[n] = close[0]
    return found


def date_strings(values) -> np.ndarray:
    """The datetime values as an object array of YYYY-MM-DD strings."""
    return pd.DatetimeIndex(values).strftime("%Y-%m-%d").to_numpy(dtype=object)


def columns_of(dims, coords, index) -> dict:
    """The table columns of the points at `index` (a tuple with one index array per spatial dim): {dim: values}."""
    return {d: np.asarray(coords[d])[ix] for d, ix in zip(dims, index)}


def locate(da, columns: dict):
    """Where table rows sit in a DataArray over spatial dims: (index tuple, located mask).

    `columns` is {dim: values of each row} for the dims of `da` (lat and lon: within tolerance of a grid
    coordinate; unit: equal to an id). The index tuple has one array per dim, -1 where a row has no cell, and
    `located` marks the rows that have one.
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
    """The position of each row of `points` in the raveled array over `dims` with `coords` (exact match), -1 if absent."""
    if dims == UNIT:
        return pd.Index(np.asarray(coords["unit"]).astype(str)).get_indexer(points["unit"].to_numpy().astype(str))
    grid = pd.MultiIndex.from_product([coords[d] for d in dims])
    return grid.get_indexer(pd.MultiIndex.from_frame(points[list(dims)]))


def snap(values, dims, coords, lead: int = 0):
    """`values` with its last len(dims) axes re-indexed by the first coordinate within tolerance of each (grids only).

    The identity unless a coordinate repeats within the tolerance; units are matched by id and need no snapping.
    `lead` is the number of leading axes (such as year) left alone.
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
    """`rows` (one value per leading row) shaped to broadcast against arrays of `ndim` dims: (n, 1, ..., 1)."""
    return np.asarray(rows).reshape((-1,) + (1,) * (ndim - 1))
