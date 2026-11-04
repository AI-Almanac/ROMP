"""Source-to-unit weights and the sparse transform that applies them.

One concept: a CSR matrix ``W`` of shape (unit, source) built from weights, and `SparseTransform.observed_mean`, the
mean of the source values over each unit weighted by ``W`` and renormalised over the sources that have a value::

    observed_mean(V) = (W @ filled) / (W @ observed)        NaN where a unit has no observed source.

Where the weights come from:

* a table (`read_weights`): rows of (target_id, weight) and either float source coordinates (latitude/longitude,
  or lat/lon) or string source ids (source_id). Both key styles give the same transform;
* the polygons of a shapefile (`polygon_weights`): the overlap area of each polygon with each cell of a grid;
* another grid (`grid_weights`): conservative area-overlap weights between two regular lat/lon grids;
* a one-for-one selection of cells (`selection`), or a weighted set of cells in one unit (a region).

Cell areas are planar (degrees squared) by default; ``cell_area="cos_lat"`` multiplies by the cosine of the cell's
latitude, so a unit's weights follow the area on the sphere to first order.

Source cells are the cells of a grid in latitude-major order (latitude axis outer, longitude inner), the order
of ``ravel`` on a (lat, lon) array. Coordinates match exactly (float equality) by default, which reproduces
tables written by onset_blending; ``tolerance`` matches within a distance instead (HCWF's alignment tolerance).

The module builds on `hcwf.spatial.ids` (grid-id conventions) and `hcwf.spatial.masks` (cell bounds and overlaps);
`hcwf.spatial.units` builds the transform of each source grid onto the evaluation units with it.
"""
from __future__ import annotations

import os
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import sparse

from hcwf.io.dims import find_dim
from hcwf.spatial.masks import _nearest, _overlap, cell_bounds, cell_shares

from . import ids as unit_ids

__all__ = ["SparseTransform", "read_weights"]

CELL_AREAS = ("planar", "cos_lat")
"""The cell-area conventions: planar degrees squared, or weighted by the cosine of the latitude."""
MEASURES = ("area", "share")
"""What a polygon weight measures: the overlap ``"area"``, or the ``"share"`` of the cell inside the polygon."""
_WEIGHT_TYPES = {"target_id": str, "adm3_name": str, "source_id": str}
"""The dtypes the id columns of a weight CSV are read with (ids stay strings, never numbers)."""


def default_tolerance() -> float:
    """Return HCWF's coordinate tolerance for matching grids (`hcwf.onset.alignment`).

    Returns
    -------
    float
        The tolerance, in degrees.
    """
    from hcwf.onset.alignment import _COORD_TOL

    return float(_COORD_TOL)


def match_axis(wanted, axis, tolerance: float | None = None) -> np.ndarray:
    """Find the index into ``axis`` of each wanted coordinate, -1 where none matches.

    Exact float equality, or within ``tolerance`` of the nearest (HCWF's matching) when it is given.

    Parameters
    ----------
    wanted : array_like of float
        The coordinates to find (flattened).
    axis : array_like of float
        The axis to search (flattened), in any order.
    tolerance : float, optional
        How far (in degrees) the nearest axis value may be; None for exact equality.

    Returns
    -------
    numpy.ndarray of int64, shape (wanted,)
        The index in ``axis`` of each coordinate, -1 where none matches.
    """
    wanted = np.asarray(wanted, dtype=float).reshape(-1)
    axis = np.asarray(axis, dtype=float).reshape(-1)
    out = np.full(wanted.size, -1, dtype=np.int64)
    if not wanted.size or not axis.size:
        return out
    order = np.argsort(axis, kind="stable")
    if tolerance is None:
        pos = np.clip(np.searchsorted(axis[order], wanted), 0, axis.size - 1)
        found = axis[order][pos] == wanted
    else:
        pos, distance = _nearest(axis[order], wanted)
        found = distance <= tolerance
    out[found] = order[pos[found]]
    return out


def conform_longitudes(wanted, axis) -> np.ndarray:
    """Put longitudes in the convention of ``axis`` (-180 to 180 when it has a negative value, else 0 to 360).

    Only a value outside the axis' range is shifted by 360 (so a value already in range is returned exactly, as
    exact matching needs); a shifted value can carry float noise, which matching within a tolerance absorbs.

    Parameters
    ----------
    wanted : array_like of float
        The longitudes, in degrees (flattened).
    axis : array_like of float
        The longitude axis whose convention is followed.

    Returns
    -------
    numpy.ndarray of float64, shape (wanted,)
        The longitudes in the convention of ``axis``.
    """
    wanted = np.array(wanted, dtype=float).reshape(-1)
    if np.min(axis) < 0:
        wanted[wanted > 180.0] -= 360.0
    else:
        wanted[wanted < 0.0] += 360.0
    return wanted


# ----------------------------------------------------------------------------------------------------------
# weight tables
# ----------------------------------------------------------------------------------------------------------

def weights_keys(weights: pd.DataFrame) -> str:
    """Tell how a weight table names its sources.

    Parameters
    ----------
    weights : pandas.DataFrame
        A weight table.

    Returns
    -------
    {"id", "coordinate"}
        ``"id"`` when it has a ``source_id`` column, else ``"coordinate"`` (``lat`` / ``lon`` columns).
    """
    return "id" if "source_id" in weights.columns else "coordinate"


def read_weights(source, context: str = "weights") -> pd.DataFrame:
    """Read a weight table from a CSV path or copy it from a DataFrame, validate it and put it in one form.

    Columns in: ``target_id`` (or legacy ``adm3_name``), ``weight``, and the sources as ``source_id`` strings, or as
    ``latitude`` / ``longitude`` (or ``lat`` / ``lon``) floats. Out: ``target_id`` (normalised string), ``weight``, and
    ``source_id``, or ``lat`` and ``lon``. Refused: missing columns, an empty table, weights that are not finite and
    non-negative, duplicate (source, target) rows, a target without positive total weight. Rows of zero weight are
    dropped.

    Parameters
    ----------
    source : str or os.PathLike or pandas.DataFrame
        A CSV path or the table itself (copied).
    context : str, default "weights"
        What the table is, for error messages.

    Returns
    -------
    pandas.DataFrame
        The validated table, flagged ``attrs["spatial_weights"]`` so that it is not checked again.

    Raises
    ------
    FileNotFoundError
        ``source`` is a path that does not exist.
    ValueError
        The table breaks one of the rules above.
    """
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{context} file not found: {path}")
        table = pd.read_csv(path, dtype=_WEIGHT_TYPES, float_precision="round_trip")
    else:
        table = source.copy()
    if "target_id" not in table.columns and "adm3_name" not in table.columns:
        raise ValueError(f"{context} must contain 'target_id' or 'adm3_name'.")
    if "target_id" in table.columns:
        table["target_id"] = unit_ids.normalize(table["target_id"], context=f"{context} target IDs")
    if "adm3_name" in table.columns:
        table["adm3_name"] = unit_ids.normalize(table["adm3_name"], context=f"{context} legacy target IDs")
    if {"target_id", "adm3_name"} <= set(table.columns):
        if (table["target_id"] != table["adm3_name"]).any():
            raise ValueError(f"{context}: 'target_id' and 'adm3_name' disagree.")
    elif "adm3_name" in table.columns:
        table["target_id"] = table["adm3_name"]
    if "weight" not in table.columns:
        raise ValueError(f"{context} must contain a weight column.")
    if table.empty:
        raise ValueError(f"{context} must not be empty.")
    if "source_id" in table.columns:
        table["source_id"] = unit_ids.normalize(table["source_id"], context=f"{context} source IDs")
        keys = ["source_id"]
    else:
        lat = find_dim(list(table.columns), "lat")
        lon = find_dim(list(table.columns), "lon")
        if lat is None or lon is None:
            raise ValueError(f"{context} must contain source_id, or latitude and longitude.")
        table["lat"] = pd.to_numeric(table[lat], errors="raise").astype(float)
        table["lon"] = pd.to_numeric(table[lon], errors="raise").astype(float)
        keys = ["lat", "lon"]
    table["weight"] = pd.to_numeric(table["weight"], errors="coerce")
    if not np.isfinite(table["weight"]).all() or (table["weight"] < 0).any():
        raise ValueError(f"{context} must be finite and non-negative.")
    if table.duplicated(keys + ["target_id"]).any():
        raise ValueError(f"{context} contain duplicate source/target rows.")
    totals = table.groupby("target_id")["weight"].sum()
    if (totals <= 0).any():
        raise ValueError(f"{context} targets have zero total weight: "
                         f"{', '.join(totals[totals <= 0].index.astype(str)[:10])}")
    out = table.loc[table["weight"] > 0, ["target_id"] + keys + ["weight"]].reset_index(drop=True)
    out.attrs["spatial_weights"] = True
    return out


def _canonical(weights, context="weights") -> pd.DataFrame:
    """Return the weights as `read_weights` gives them (a table that already is one is not checked again).

    Parameters
    ----------
    weights : str or os.PathLike or pandas.DataFrame
        A weight table or a CSV path.
    context : str, default "weights"
        What the table is, for error messages.

    Returns
    -------
    pandas.DataFrame
        The validated table.
    """
    if isinstance(weights, pd.DataFrame) and weights.attrs.get("spatial_weights"):
        return weights
    return read_weights(weights, context)


# ----------------------------------------------------------------------------------------------------------
# the transform
# ----------------------------------------------------------------------------------------------------------

class SparseTransform:
    """Unit-by-source weights (CSR), for weighted means over each unit's observed sources.

    `keys` says how sources are named: ``"coordinate"`` (``source_ids`` are (lat, lon) float pairs, the cells of a
    grid in latitude-major order) or ``"id"`` (``source_ids`` are strings; `convention` is the grid-id convention they
    follow, if any).

    Parameters
    ----------
    weighted_matrix : scipy.sparse.csr_matrix of float64, shape (unit, source)
        The weight of each source in each unit.
    source_ids : sequence or None
        The id of each source (strings, or (lat, lon) pairs); None for a grid, whose cells are then the sources.
    target_ids : tuple of str
        The id of each unit, in the order of the matrix rows.
    keys : {"coordinate", "id"}, default "coordinate"
        How the sources are named.
    convention : hcwf.spatial.ids.GridIdConvention, optional
        The grid-id convention the source ids follow, if any.
    grid : tuple of (array_like of float, array_like of float), optional
        The ``(lat, lon)`` axes of the source grid, for a coordinate-keyed transform.

    Attributes
    ----------
    weighted_matrix : scipy.sparse.csr_matrix of float64, shape (unit, source)
        The weights.
    target_ids : tuple of str
        The unit ids.
    keys : str
        ``"coordinate"`` or ``"id"``.
    convention : hcwf.spatial.ids.GridIdConvention or None
        The grid-id convention of the source ids.
    grid : tuple of (numpy.ndarray, numpy.ndarray) or None
        The ``(lat, lon)`` axes of the source grid, float64.
    shape : tuple of (int, int)
        The shape of the matrix, ``(unit, source)``.
    """

    def __init__(self, weighted_matrix, source_ids, target_ids: tuple, *, keys: str = "coordinate",
                 convention: unit_ids.GridIdConvention | None = None, grid=None):
        self.weighted_matrix = weighted_matrix
        self.target_ids = tuple(target_ids)
        self.keys, self.convention = keys, convention
        self._source_ids = None if source_ids is None else tuple(source_ids)
        self.grid = None if grid is None else (np.asarray(grid[0], float), np.asarray(grid[1], float))
        self.shape = weighted_matrix.shape

    @cached_property
    def source_ids(self) -> tuple:
        """The id of each source: strings, or (lat, lon) pairs of the grid's cells in latitude-major order.

        Returns
        -------
        tuple
            The source ids, in the order of the matrix columns.
        """
        if self._source_ids is not None:
            return self._source_ids
        return tuple((float(a), float(b)) for a in self.grid[0] for b in self.grid[1])

    def select_targets(self, positions) -> "SparseTransform":
        """Return the transform restricted to some of the units.

        Parameters
        ----------
        positions : array_like of int
            The positions of the units to keep, in the order they are wanted.

        Returns
        -------
        hcwf.spatial.regrid.SparseTransform
            A transform with the same sources, keys, convention and grid and only those unit rows.
        """
        positions = np.asarray(positions, dtype=np.int64)
        return SparseTransform(self.weighted_matrix[positions], self._source_ids,
                               tuple(np.asarray(self.target_ids)[positions]), keys=self.keys,
                               convention=self.convention, grid=self.grid)

    @cached_property
    def source_index(self) -> dict:
        """The position of each source id in `source_ids`.

        Returns
        -------
        dict
            The column of each source id.
        """
        return {source: index for index, source in enumerate(self.source_ids)}

    @classmethod
    def compile(cls, weights: pd.DataFrame, *, source_ids=None, target_ids=None, source_col="source_id",
                target_col="target_id", weight_col="weight") -> "SparseTransform":
        """Compile a transform from a long (source, target, weight) table.

        Rows of zero weight are dropped; the transform is id-keyed.

        Parameters
        ----------
        weights : pandas.DataFrame
            The table, one row per (source, target) pair.
        source_ids : sequence, optional
            The sources in matrix order; by default in order of appearance.
        target_ids : sequence, optional
            The targets in matrix order; by default in order of appearance.
        source_col : str, default "source_id"
            The source column.
        target_col : str, default "target_id"
            The target column.
        weight_col : str, default "weight"
            The weight column.

        Returns
        -------
        SparseTransform
            The compiled transform.

        Raises
        ------
        ValueError
            A column is missing, the table is empty, a weight is not finite and non-negative, a target has no positive
            weight, or an id is outside the given axes.
        """
        missing = {source_col, target_col, weight_col} - set(weights.columns)
        if missing:
            raise ValueError(f"Sparse cell transform is missing columns: {sorted(missing)}")
        if weights.empty:
            raise ValueError("Sparse cell-transform weights must not be empty.")
        values = pd.to_numeric(weights[weight_col], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Compiled cell-transform weights must be finite and non-negative.")
        if (pd.Series(values).groupby(weights[target_col].to_numpy()).sum() <= 0).any():
            raise ValueError("Every compiled cell-transform target must have positive weight.")
        positive = values > 0
        rows = weights.loc[positive]
        source_ids = tuple(pd.unique(rows[source_col]) if source_ids is None else source_ids)
        target_ids = tuple(pd.unique(rows[target_col]) if target_ids is None else target_ids)
        source_pos = rows[source_col].map({s: i for i, s in enumerate(source_ids)})
        target_pos = rows[target_col].map({t: i for i, t in enumerate(target_ids)})
        if source_pos.isna().any() or target_pos.isna().any():
            raise ValueError("Cell-transform weights contain IDs outside the compiled axes.")
        matrix = sparse.csr_matrix(
            (values[positive], (target_pos.to_numpy(int), source_pos.to_numpy(int))),
            shape=(len(target_ids), len(source_ids)),
        )
        return cls(matrix, source_ids, target_ids, keys="id")

    @classmethod
    def from_weights(cls, weights, lat=None, lon=None, *, target_ids=None,
                     convention: unit_ids.GridIdConvention | None = None,
                     tolerance: float | None = None, wrap_longitude: bool = False) -> "SparseTransform":
        """Build the transform of a weight table (see `~hcwf.spatial.regrid.read_weights`; a raw table or CSV path is read first).

        Weights keyed by coordinates need the source grid (``lat``, ``lon`` axes): its cells, latitude-major, are
        the sources, and a weight's cell is found by exact coordinates, or within ``tolerance``; a weight for a
        cell the grid lacks is an error. Weights keyed by source ids have the sources they name (sorted), and
        the grid-id `convention` (explicit, else the one the ids establish) names grid points
        (see `positions`). Targets are the table's, sorted, unless `target_ids` gives their order.
        ``wrap_longitude`` puts the weights' longitudes in the convention of the grid before matching (a grid HCWF
        swapped to 0..360 or -180..180); use it with a ``tolerance`` when a shifted longitude carries float noise.

        Parameters
        ----------
        weights : str or os.PathLike or pandas.DataFrame
            The weight table or a CSV path.
        lat : array_like of float, shape (lat,), optional
            The latitudes of the source grid, in degrees; needed for coordinate-keyed weights.
        lon : array_like of float, shape (lon,), optional
            The longitudes of the source grid, in degrees; needed for coordinate-keyed weights.
        target_ids : sequence of str, optional
            The targets in matrix order; the table's, sorted, when None.
        convention : hcwf.spatial.ids.GridIdConvention, optional
            The explicit grid-id convention, for id-keyed weights.
        tolerance : float, optional
            How far (in degrees) a weight's cell may be from a grid cell; None for exact coordinates.
        wrap_longitude : bool, default False
            Put the weights' longitudes in the convention of the grid before matching.

        Returns
        -------
        SparseTransform
            The transform, coordinate-keyed (with the grid) or id-keyed.

        Raises
        ------
        ValueError
            Coordinate-keyed weights come without the grid, a weight's cell is not on the grid, a target is not among
            `target_ids`, or the weights are invalid (see `~hcwf.spatial.regrid.read_weights`).
        """
        weights = _canonical(weights)
        targets = tuple(sorted(weights["target_id"].unique())) if target_ids is None else tuple(target_ids)
        if weights_keys(weights) == "id":
            sources = tuple(sorted(weights["source_id"].unique()))
            transform = cls.compile(weights, source_ids=sources, target_ids=targets)
            transform.convention = unit_ids.resolve(convention, authoritative_ids=pd.Series(sources, dtype=str),
                                                    context="cell transform")
            return transform
        if lat is None or lon is None:
            raise ValueError("Weights keyed by coordinates need the source grid (lat and lon axes).")
        lat, lon = np.asarray(lat, dtype=float).reshape(-1), np.asarray(lon, dtype=float).reshape(-1)
        wlon = conform_longitudes(weights["lon"].to_numpy(), lon) if wrap_longitude else weights["lon"].to_numpy()
        lat_pos, lon_pos = _cell_axes(weights["lat"].to_numpy(), wlon, lat, lon, tolerance)
        target_pos = pd.Series(weights["target_id"]).map({t: i for i, t in enumerate(targets)})
        if target_pos.isna().any():
            raise ValueError("Cell-transform weights contain IDs outside the compiled axes.")
        values = weights["weight"].to_numpy(float)
        matrix = sparse.csr_matrix(
            (values, (target_pos.to_numpy(int), lat_pos * lon.size + lon_pos)), shape=(len(targets), lat.size * lon.size)
        )
        return cls(matrix, None, targets, keys="coordinate", grid=(lat, lon))

    def supported(self, present: np.ndarray) -> np.ndarray:
        """Tell whether any of a target's sources is present.

        Parameters
        ----------
        present : numpy.ndarray of bool, shape (source, column)
            Which sources are present in each column.

        Returns
        -------
        numpy.ndarray of bool, shape (target, column)
            True where at least one source of the target is present.
        """
        support = self.weighted_matrix.copy()
        support.data = np.ones_like(support.data)
        return np.asarray(support @ np.asarray(present, dtype=float)) > 0

    @cached_property
    def single_sources(self) -> np.ndarray | None:
        """Each target's one source position when every target has exactly one source of weight 1.0, else None.

        These are units that are the cells themselves. Then a target's observed mean is its source's value
        (``x * 1.0 / 1.0`` is ``x``), NaN where that is not finite.

        Returns
        -------
        numpy.ndarray of int64, shape (target,) or None
            The source position of each target; None when some target has not exactly one source of weight 1.
        """
        matrix = self.weighted_matrix.tocsr()
        if matrix.shape[0] == 0 or (np.diff(matrix.indptr) != 1).any() or (matrix.data != 1.0).any():
            return None
        return np.asarray(matrix.indices, dtype=np.int64)

    def observed_mean(self, source_values: np.ndarray, column_chunk_size: int = 256) -> np.ndarray:
        """Compute the weighted mean of the observed sources of each target: ``W @ filled / W @ observed``.

        Non-finite source values count as not observed; a target with no observed source is NaN.

        Parameters
        ----------
        source_values : numpy.ndarray of float64, shape (source, column)
            The value of each source in each column.
        column_chunk_size : int, default 256
            The number of columns handled at once, to bound memory.

        Returns
        -------
        numpy.ndarray of float64, shape (target, column)
            The mean of each target in each column.

        Raises
        ------
        ValueError
            ``source_values`` is not 2-D or its first axis is not the number of sources.
        """
        source_values = np.asarray(source_values, dtype=float)
        if source_values.ndim != 2 or source_values.shape[0] != self.shape[1]:
            raise ValueError("Sparse transform source axis does not match the compiled weight matrix.")
        out = np.full((self.shape[0], source_values.shape[1]), np.nan)
        for start in range(0, source_values.shape[1], int(column_chunk_size)):
            stop = min(source_values.shape[1], start + int(column_chunk_size))
            block = source_values[:, start:stop]
            valid = np.isfinite(block)
            numerator = np.asarray(self.weighted_matrix @ np.where(valid, block, 0.0))
            denominator = np.asarray(self.weighted_matrix @ valid.astype(float))
            np.divide(numerator, denominator, out=out[:, start:stop], where=denominator > 0)
        return out

    def sources_of(self, targets) -> np.ndarray:
        """Find the source positions with weight for the given target positions.

        Parameters
        ----------
        targets : array_like of int
            The target positions (matrix rows).

        Returns
        -------
        numpy.ndarray of int32 or int64
            The sorted distinct source positions with weight in any of the targets.
        """
        return np.unique(self.weighted_matrix[targets].indices)

    def positions(self, lat, lon, *, tolerance: float | None = None, require_all: bool = False) -> np.ndarray:
        """Find each point's position among the sources (-1 for a point the weights do not use).

        Coordinate-keyed sources match by exact coordinates, or within ``tolerance``. Id-keyed sources match the
        points' grid ids under the transform's convention (so it needs one). ``require_all`` refuses a source
        that no point matches. Two points on one source are refused.

        Parameters
        ----------
        lat : array_like of float
            The latitudes of the points, in degrees (flattened).
        lon : array_like of float
            The longitudes of the points, in degrees (flattened).
        tolerance : float, optional
            How far (in degrees) a point may be from a coordinate-keyed source; None for exact coordinates.
        require_all : bool, default False
            Refuse a source that no point matches.

        Returns
        -------
        numpy.ndarray of int64, shape (point,)
            The source position of each point, -1 where it is not a source.

        Raises
        ------
        ValueError
            The sources are ids without a grid-id convention, two points share a source, or ``require_all`` is true and a
            source has no point.
        """
        lat = np.asarray(lat, dtype=float).reshape(-1)
        lon = np.asarray(lon, dtype=float).reshape(-1)
        if self.keys == "id":
            if self.convention is None:
                raise ValueError("The transform's source ids are not grid ids; match points with positions_of_ids.")
            return self.positions_of_ids(unit_ids.format_grid_ids(lat, lon, self.convention), require_all=require_all)
        if self.grid is not None and tolerance is None:
            # sources are the cells of a grid: one axis lookup each, no tuple dictionary
            lat_pos, lon_pos = match_axis(lat, self.grid[0]), match_axis(lon, self.grid[1])
            out = np.where((lat_pos >= 0) & (lon_pos >= 0), lat_pos * self.grid[1].size + lon_pos, -1)
        elif tolerance is None:
            index = self.source_index
            out = np.array([index.get((float(a), float(b)), -1) for a, b in zip(lat, lon)], dtype=np.int64)
        else:
            slat = np.array(sorted({s[0] for s in self.source_ids}))
            slon = np.array(sorted({s[1] for s in self.source_ids}))
            lat_pos, lon_pos = match_axis(lat, slat, tolerance), match_axis(lon, slon, tolerance)
            index = self.source_index
            out = np.array([index.get((slat[i], slon[j]), -1) if i >= 0 and j >= 0 else -1
                            for i, j in zip(lat_pos, lon_pos)], dtype=np.int64)
        return self._checked(out, require_all)

    def positions_of_ids(self, ids, *, require_all: bool = False) -> np.ndarray:
        """Find each id's position among the (id-keyed) sources, -1 for one the weights do not use.

        Parameters
        ----------
        ids : iterable of str
            The ids of the points.
        require_all : bool, default False
            Refuse a source that no id names.

        Returns
        -------
        numpy.ndarray of int64, shape (point,)
            The source position of each id, -1 where it is not a source.

        Raises
        ------
        ValueError
            ``require_all`` is true and a source is named by none of ``ids``, or two ids share a source.
        """
        index = self.source_index
        out = np.array([index.get(str(i), -1) for i in ids], dtype=np.int64)
        if require_all:
            missing = sorted(set(self.source_ids) - set(map(str, ids)))
            if missing:
                raise ValueError(
                    f"{len(missing)} expected weight source IDs are absent from the raw NetCDF coordinates: "
                    f"{', '.join(map(str, missing[:10]))}"
                )
        return self._checked(out, False)

    def _checked(self, positions: np.ndarray, require_all: bool) -> np.ndarray:
        """Check that no two points share a source (and, with ``require_all``, that every source has a point).

        Parameters
        ----------
        positions : numpy.ndarray of int64, shape (point,)
            The source position of each point, -1 for none.
        require_all : bool
            Also require every source to have a point.

        Returns
        -------
        numpy.ndarray of int64, shape (point,)
            `positions`, unchanged.

        Raises
        ------
        ValueError
            Two points share a source, or ``require_all`` is true and a source has no point.
        """
        used = positions[positions >= 0]
        if len(np.unique(used)) != len(used):
            raise ValueError("points share cell-transform source ids")
        if require_all and len(used) != self.shape[1]:
            raise ValueError(f"{self.shape[1] - len(used)} transform sources are absent from the points")
        return positions

    def scatter(self, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Place (point, column) values on the source axis (NaN for sources without a point).

        Parameters
        ----------
        values : numpy.ndarray of float64, shape (point, column)
            The values of the points.
        positions : numpy.ndarray of int64, shape (point,)
            The source position of each point, -1 for none (such a point is dropped).

        Returns
        -------
        numpy.ndarray of float64, shape (source, column)
            The values on the sources, NaN where no point sits.
        """
        values = np.asarray(values, dtype=float)
        out = np.full((self.shape[1], values.shape[1]), np.nan)
        used = positions >= 0
        out[positions[used]] = values[used]
        return out


def point_sources(points: pd.DataFrame, transform: SparseTransform, tolerance: float | None = None) -> np.ndarray:
    """Find each point's position among the transform's sources (-1 for none).

    Parameters
    ----------
    points : pandas.DataFrame
        A table with ``lat`` and ``lon`` columns.
    transform : SparseTransform
        The transform whose sources are searched.
    tolerance : float, optional
        How far (in degrees) a point may be from a source; None for exact coordinates.

    Returns
    -------
    numpy.ndarray of int64, shape (row,)
        The source position of each point, -1 where it is not a source.
    """
    return transform.positions(points["lat"], points["lon"], tolerance=tolerance)


def grid_axes(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """List the distinct latitudes and longitudes of a grid's points, in file order.

    Parameters
    ----------
    points : pandas.DataFrame
        A table with ``lat`` and ``lon`` columns.

    Returns
    -------
    lat : numpy.ndarray
        The distinct latitudes.
    lon : numpy.ndarray
        The distinct longitudes.
    """
    return pd.unique(points["lat"].to_numpy()), pd.unique(points["lon"].to_numpy())


def _cell_axes(wlat, wlon, lat, lon, tolerance):
    """Find the axis positions of weight coordinates among the grid axes.

    Parameters
    ----------
    wlat : numpy.ndarray of float64, shape (weight,)
        The latitude of each weight.
    wlon : numpy.ndarray of float64, shape (weight,)
        The longitude of each weight.
    lat : numpy.ndarray of float64, shape (lat,)
        The latitude axis of the grid.
    lon : numpy.ndarray of float64, shape (lon,)
        The longitude axis of the grid.
    tolerance : float or None
        How far (in degrees) a weight may be from an axis value; None for exact equality.

    Returns
    -------
    lat_pos : numpy.ndarray of int64, shape (weight,)
        The latitude index of each weight.
    lon_pos : numpy.ndarray of int64, shape (weight,)
        The longitude index of each weight.

    Raises
    ------
    ValueError
        A coordinate is not on the grid.
    """
    lat_pos = np.empty(len(wlat), dtype=np.int64)
    lon_pos = np.empty(len(wlon), dtype=np.int64)
    for out, wanted, axis in ((lat_pos, wlat, lat), (lon_pos, wlon, lon)):
        distinct, inverse = np.unique(wanted, return_inverse=True)
        found = match_axis(distinct, axis, tolerance)
        if (found < 0).any():
            raise ValueError("Cell-transform weights contain IDs outside the compiled axes.")
        out[:] = found[inverse]
    return lat_pos, lon_pos


# ----------------------------------------------------------------------------------------------------------
# building weights
# ----------------------------------------------------------------------------------------------------------

def _check_options(cell_area, measure=None):
    """Check the ``cell_area`` and ``measure`` options.

    Parameters
    ----------
    cell_area : str
        The cell-area convention; one of `CELL_AREAS`.
    measure : str, optional
        What a weight measures; one of `MEASURES`, or None to skip the check.

    Raises
    ------
    ValueError
        An option is not one of its values.
    """
    if cell_area not in CELL_AREAS:
        raise ValueError(f"cell_area must be one of {CELL_AREAS}, got {cell_area!r}")
    if measure is not None and measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")


def polygon_weights(geometries, ids, lat, lon, *, cell_area: str = "planar", measure: str = "area",
                    min_share: float = 1e-9, clip_to=None) -> pd.DataFrame:
    """Compute the weights of the cells of a grid in each polygon: a table of (target_id, lat, lon, weight).

    The grid has centres ``lat``, ``lon`` (any order; longitudes are wrapped to -180..180 for the overlap, the
    table keeps the grid's own values). A cell's weight in a polygon is the area of their overlap (measure
    ``"area"``, planar degrees squared, times cos(latitude) when ``cell_area`` is ``"cos_lat"``) or the share of the cell's
    area inside the polygon (measure ``"share"``, onset_blending's legacy table: intersection / cell area). Cells
    with a share of at most ``min_share`` are left out. ``clip_to`` is a geometry each polygon is first cut to (for
    instance the area the observations cover, see `coverage_geometry`).

    The overlap is the one `hcwf.spatial.masks.rasterise` thresholds (`hcwf.spatial.masks.cell_shares`).

    Parameters
    ----------
    geometries : iterable of shapely.geometry.base.BaseGeometry or None
        The polygons, one per id; None or empty ones are skipped.
    ids : iterable of str
        The target id of each polygon.
    lat : array_like of float, shape (lat,)
        The latitudes of the cell centres, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes of the cell centres, in degrees.
    cell_area : {"planar", "cos_lat"}, default "planar"
        The cell-area convention.
    measure : {"area", "share"}, default "area"
        What a weight measures.
    min_share : float, default 1e-9
        Cells whose share of their area in the polygon is at most this are left out.
    clip_to : shapely.geometry.base.BaseGeometry, optional
        A geometry each polygon is cut to first.

    Returns
    -------
    pandas.DataFrame
        The table with ``target_id`` (str), ``lat``, ``lon`` and ``weight`` (float64), flagged as a weight table.

    Raises
    ------
    ValueError
        An option is invalid, or no polygon overlaps the grid.
    """
    _check_options(cell_area, measure)
    lat, lon = np.asarray(lat, dtype=float).reshape(-1), np.asarray(lon, dtype=float).reshape(-1)
    order_lat, order_lon = np.argsort(lat, kind="stable"), np.argsort(lon, kind="stable")
    slat, slon = lat[order_lat], lon[order_lon]
    lat_edges, lon_edges = cell_bounds(slat, slon)
    dlat, dlon = np.diff(lat_edges), np.diff(lon_edges)
    cos = np.cos(np.deg2rad(slat))
    parts = []
    for target, geom in zip(ids, geometries):
        if geom is None or geom.is_empty:
            continue
        if clip_to is not None:
            geom = geom.intersection(clip_to)
        rows, cols, share = cell_shares(geom, slat, slon)
        if not share.size:
            continue
        weight = share.copy()
        if measure == "area":
            weight *= dlat[rows][:, None] * dlon[cols][None, :]
        if cell_area == "cos_lat":
            weight *= cos[rows][:, None]
        i, j = np.nonzero(share > min_share)
        if i.size:
            parts.append(pd.DataFrame({
                "target_id": str(target), "lat": lat[order_lat[rows[i]]], "lon": lon[order_lon[cols[j]]],
                "weight": weight[i, j],
            }))
    if not parts:
        raise ValueError("No polygon overlaps the grid.")
    out = pd.concat(parts, ignore_index=True)
    out.attrs["spatial_weights"] = True
    return out


def coverage_geometry(lat, lon, keep=None):
    """Build the union of the boxes of the grid's cells (those where ``keep`` is True).

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes of the cell centres, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes of the cell centres, in degrees.
    keep : numpy.ndarray of bool, shape (lat, lon), optional
        The cells to include; all when None.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        The union of the boxes.

    Raises
    ------
    ValueError
        No cell is kept.
    """
    import shapely

    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    order_lat, order_lon = np.argsort(lat, kind="stable"), np.argsort(lon, kind="stable")
    lat_edges, lon_edges = cell_bounds(lat[order_lat], lon[order_lon])
    keep = np.ones((lat.size, lon.size), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    i, j = np.nonzero(keep[np.ix_(order_lat, order_lon)])
    if not i.size:
        raise ValueError("The coverage holds no cell.")
    return shapely.union_all(shapely.box(lon_edges[j], lat_edges[i], lon_edges[j + 1], lat_edges[i + 1]))


def _axis_intervals(centres, other):
    """Compute the cell (lo, hi) edges of each centre, in the order the centres are given (see `~hcwf.spatial.masks.cell_bounds`).

    Parameters
    ----------
    centres : array_like of float, shape (n,)
        The cell centres of one axis, in any order.
    other : array_like of float
        The centres of the other axis (`~hcwf.spatial.masks.cell_bounds` takes both).

    Returns
    -------
    lo : numpy.ndarray of float64, shape (n,)
        The lower edge of each cell.
    hi : numpy.ndarray of float64, shape (n,)
        The upper edge of each cell.
    """
    centres = np.asarray(centres, dtype=float)
    order = np.argsort(centres, kind="stable")
    edges = cell_bounds(centres[order], np.asarray(other, dtype=float)[np.argsort(other, kind="stable")])[0]
    lo, hi = np.empty(centres.size), np.empty(centres.size)
    lo[order], hi[order] = edges[:-1], edges[1:]
    return lo, hi


def _axis_overlaps(lo_a, hi_a, lo_b, hi_b, block_elements: int = 1 << 22) -> sparse.csr_matrix:
    """Compute the length two sets of intervals share, built a block of rows at a time.

    Parameters
    ----------
    lo_a : numpy.ndarray of float64, shape (a,)
        The lower edges of the first set.
    hi_a : numpy.ndarray of float64, shape (a,)
        The upper edges of the first set.
    lo_b : numpy.ndarray of float64, shape (b,)
        The lower edges of the second set.
    hi_b : numpy.ndarray of float64, shape (b,)
        The upper edges of the second set.
    block_elements : int, default 1 << 22
        The most elements of the dense block of overlaps computed at once.

    Returns
    -------
    scipy.sparse.csr_matrix of float64, shape (a, b)
        The overlap length of each pair, zero entries not stored.
    """
    step = max(1, block_elements // max(len(lo_b), 1))
    rows, cols, vals = [], [], []
    for start in range(0, len(lo_a), step):
        block = _overlap(lo_a[start:start + step], hi_a[start:start + step], lo_b, hi_b)
        i, j = np.nonzero(block > 0)
        rows.append(i + start), cols.append(j), vals.append(block[i, j])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(lo_a), len(lo_b)))


def grid_weights(source_lat, source_lon, lat, lon, *, cell_area: str = "planar") -> sparse.csr_matrix:
    """Compute conservative weights from the cells of one lat/lon grid to the cells of another.

    A matrix in latitude-major order (``(len(lat) * len(lon), len(source_lat) * len(source_lon))``), the area each
    target cell shares with each source cell (the separable overlaps `hcwf.spatial.masks.align_raster_mask` uses).
    Source longitudes are first shifted by whole turns to the target's convention. ``cell_area="cos_lat"``
    multiplies by the cosine of the source cell's latitude.

    A `ValueError` is raised (by `_check_options`) when ``cell_area`` is not one of `CELL_AREAS`.

    Parameters
    ----------
    source_lat : array_like of float, shape (source_lat,)
        The latitudes of the source cell centres, in degrees.
    source_lon : array_like of float, shape (source_lon,)
        The longitudes of the source cell centres, in degrees.
    lat : array_like of float, shape (lat,)
        The latitudes of the target cell centres, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes of the target cell centres, in degrees.
    cell_area : {"planar", "cos_lat"}, default "planar"
        The cell-area convention.

    Returns
    -------
    scipy.sparse.csr_matrix of float64, shape (target_cell, source_cell)
        The shared area (degrees squared) of each target and source cell.
    """
    _check_options(cell_area)
    source_lat, source_lon = np.asarray(source_lat, float), np.asarray(source_lon, float)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    start = (lon.min() + lon.max()) / 2 - 180.0
    source_lon = ((source_lon - start) % 360.0) + start
    s_lat, s_lon = _axis_intervals(source_lat, source_lon), _axis_intervals(source_lon, source_lat)
    t_lat, t_lon = _axis_intervals(lat, lon), _axis_intervals(lon, lat)
    matrix = sparse.kron(_axis_overlaps(*t_lat, *s_lat), _axis_overlaps(*t_lon, *s_lon), format="csr")
    if cell_area == "cos_lat":
        scale = np.repeat(np.cos(np.deg2rad(source_lat)), source_lon.size)
        matrix = (matrix @ sparse.diags(scale)).tocsr()
    return matrix


def selection(positions, n_sources: int, weights=None) -> sparse.csr_matrix:
    """Build a one-source-per-row matrix: row i weights source ``positions[i]`` (1 unless ``weights`` gives it).

    Parameters
    ----------
    positions : array_like of int, shape (row,)
        The source position of each row.
    n_sources : int
        The number of sources (columns).
    weights : array_like of float, shape (row,), optional
        The weight of each row's source; 1 for all when None.

    Returns
    -------
    scipy.sparse.csr_matrix of float64, shape (row, n_sources)
        The selection matrix.
    """
    positions = np.asarray(positions, dtype=np.int64)
    values = np.ones(positions.size) if weights is None else np.asarray(weights, dtype=float)
    return sparse.csr_matrix((values, (np.arange(positions.size), positions)), shape=(positions.size, n_sources))
