"""Define the evaluation units, the places onsets are computed, scored and blended for.

A unit is a grid cell, a polygon (an admin unit) or a region (a set of cells averaged into one series). One
table describes them, whatever the kind: an ordered, unique string ``id``, a centroid ``lat`` / ``lon``, and
optionally ``name``, ``parent``, the polygon ``geometry`` and named boolean subsets (for instance ``dissemination``,
``training``). `EvaluationUnits` is that table, `UnitsSpec` the configuration that says where it comes from, and
`read_centroids`, `read_id_list` and `read_shapes` read the tables and shapes it is built from. The weights
themselves are built in `hcwf.spatial.regrid`, the grid-id conventions live in `hcwf.spatial.ids`.

`EvaluationUnits.transform(source_lat, source_lon)` gives the `~hcwf.spatial.regrid.SparseTransform` from the cells
of a source grid (observations or a model) to the units, compiled once per grid. Rain is read on the source grid,
averaged with ``observed_mean`` onto the units in float mm and only then rounded to tenths
(`EvaluationUnits.from_region` is the one-unit case), so the engine runs on a ``("unit",)`` spatial axis unchanged.

How the transform comes about:

* grid units (from a grid and HCWF's mask): a 0/1 selection when the source grid holds every unit cell,
  otherwise conservative area weights from the source cells to the unit cells;
* polygon units: cell-polygon overlap areas from the shapes, or weights from a table;
* a region: cos-lat (or equal) weights on its cells, composed with the grid weights for another grid.

String ids appear only at the edges (`hcwf.spatial.ids`): reading and writing tables and lists.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from hcwf.config import schema
from hcwf.errors import AreaError, UnitsSpecError
from . import ids as unit_ids
from . import regrid
from .regrid import SparseTransform
from hcwf.errors import RegionError

log = logging.getLogger(__name__)

KINDS = ("grid", "polygon", "region")
"""The kinds of unit: a grid cell, a polygon and a region (a set of cells averaged into one series)."""
_GRID_ID = re.compile(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
"""The pattern of a ``<lat>_<lon>`` grid id, with the latitude and the longitude as groups."""
_LAT_NAMES = ("lat", "center_lat", "latitude")
"""The accepted names of a latitude column, in order of preference."""
_LON_NAMES = ("lon", "center_lon", "longitude")
"""The accepted names of a longitude column, in order of preference."""


def read_centroids(source, id_col: str | None = None) -> pd.DataFrame:
    """Read the unit centroids as a table of (id, lat, lon), from a CSV path or a table.

    The id column is `id_col`, else the first of ``target_id``, ``id``, ``adm3_name``, else the first column;
    latitude and longitude are ``lat`` / ``center_lat`` / ``latitude`` and ``lon`` / ``center_lon`` / ``longitude``.
    There is one row per id.

    Parameters
    ----------
    source : str or os.PathLike or pandas.DataFrame
        The path of a CSV file, or a table with the same columns (it is copied).
    id_col : str, optional
        The name of the id column; None picks it as described above.

    Returns
    -------
    pandas.DataFrame
        The columns ``id`` (normalised strings, unique), ``lat`` and ``lon`` (``float64``, degrees).

    Raises
    ------
    FileNotFoundError
        `source` is a path that does not exist.
    ValueError
        `id_col` is not a column of the table, or the table has no latitude or longitude column.
    """
    if isinstance(source, (str, os.PathLike)):
        if not os.path.exists(source):
            raise FileNotFoundError(f"centroids file not found: {source}")
        table = pd.read_csv(source, dtype=str)
    else:
        table = source.copy()
    if id_col and id_col not in table.columns:
        raise ValueError(f"centroids id_col '{id_col}' not found. Found: {table.columns.tolist()}")
    key = id_col or next((c for c in ("target_id", "id", "adm3_name") if c in table.columns), table.columns[0])
    lat = next((c for c in _LAT_NAMES if c in table.columns), None)
    lon = next((c for c in _LON_NAMES if c in table.columns), None)
    if not lat or not lon:
        raise ValueError("The centroids need lat/lon (or center_lat/center_lon) columns.")
    table = table.rename(columns={key: "id", lat: "lat", lon: "lon"})
    table["id"] = unit_ids.normalize(table["id"], context="centroid IDs")
    out = table[["id", "lat", "lon"]].drop_duplicates("id").reset_index(drop=True)
    out["lat"], out["lon"] = out["lat"].astype(float), out["lon"].astype(float)
    return out


def read_id_list(path, id_col: str | None = None, *, explicit=None, authoritative_ids=None) -> pd.Series:
    """Read the ids of a CSV list of units (for instance the dissemination units) as normalised strings.

    The id column is `id_col`, else ``id`` / ``adm3_name``, else grid ids made from ``lat`` and ``lon``, else (with a
    warning) the first column. Grid ids are written under the convention the `authoritative_ids` (the units')
    establish.

    Parameters
    ----------
    path : str or os.PathLike
        The CSV file to read.
    id_col : str, optional
        The name of the id column; None picks it as described above.
    explicit : hcwf.spatial.ids.GridIdConvention, optional
        A grid-id convention set by the configuration; None lets the units' ids decide.
    authoritative_ids : pandas.Series of str, optional
        The ids of the units, whose convention the listed grid ids follow.

    Returns
    -------
    pandas.Series of str
        The distinct ids, in file order, with a fresh index.

    Raises
    ------
    FileNotFoundError
        `path` does not exist.
    ValueError
        `id_col` is not a column of the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"unit list not found: {path}")
    table = pd.read_csv(path, dtype=str)
    if id_col:
        if id_col not in table.columns:
            raise ValueError(f"Configured id_col '{id_col}' not found. Found: {table.columns.tolist()}")
        table = table.rename(columns={id_col: "id"})
    elif not any(c in table.columns for c in ("id", "adm3_name")) and not {"lat", "lon"} <= set(table.columns):
        log.warning("%s has no explicit spatial key; using its first column %r", path, table.columns[0])
        table = table.rename(columns={table.columns[0]: "id"})
    convention = unit_ids.resolve(explicit, authoritative_ids=authoritative_ids, context="unit ID handoff")
    table = unit_ids.with_id_column(table, explicit=explicit, convention=convention, context="unit list IDs")
    unit_ids.check_coordinates(table, context="unit list IDs")
    return table["id"].drop_duplicates().reset_index(drop=True)


def read_shapes(path, unit_id_col: str, unit_name_col: str | None = None, parent_col: str | None = None):
    """Read the features of a shapefile or GeoJSON as (table of id [, name, parent], geometries) in EPSG:4326.

    Ids are normalised strings and must be unique. A file without a coordinate system is taken to be EPSG:4326.

    Parameters
    ----------
    path : str or os.PathLike
        The shapefile or GeoJSON to read.
    unit_id_col : str
        The attribute that holds the unit id.
    unit_name_col : str, optional
        The attribute that holds the unit name; None leaves the ``name`` column out.
    parent_col : str, optional
        The attribute that holds the parent unit; None leaves the ``parent`` column out.

    Returns
    -------
    table : pandas.DataFrame
        One row per feature, with ``id`` and, when asked for, ``name`` and ``parent``.
    geometries : list of shapely.Geometry
        The geometry of each feature, in the order of the rows, in EPSG:4326.

    Raises
    ------
    FileNotFoundError
        `path` does not exist.
    ValueError
        A requested attribute is not in the file, or the ids are not unique.
    """
    import geopandas as gpd

    if not os.path.exists(path):
        raise FileNotFoundError(f"Shapefile not found: {path}")
    frame = gpd.read_file(path)
    columns = {"id": unit_id_col, "name": unit_name_col, "parent": parent_col}
    for role, column in columns.items():
        if column and column not in frame.columns:
            raise ValueError(f"{path}: no {role} column {column!r}. Available: {frame.columns.tolist()}")
    if frame.crs is not None and not frame.crs.equals("EPSG:4326"):
        frame = frame.to_crs("EPSG:4326")
    table = pd.DataFrame({role: frame[column].to_numpy() for role, column in columns.items() if column})
    table["id"] = unit_ids.normalize(table["id"], context="shapefile unit IDs").to_numpy()
    duplicated = table["id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"Shapefile unit IDs must be unique; duplicates: {', '.join(table.loc[duplicated, 'id'].unique()[:10])}")
    return table, list(frame.geometry)


class _Cells:
    """The grid a grid or region unit lives on.

    It holds the axes, the flat (latitude-major) index of each unit's cell and, for a region, the weight of each of
    its cells.

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes of the grid, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes of the grid, in degrees.
    flat : array_like of int, shape (cell,)
        The flat latitude-major index (``i_lat * len(lon) + i_lon``) of each cell.
    weights : array_like of float, shape (cell,), optional
        The weight of each cell in a region; None for grid units.

    Attributes
    ----------
    lat : numpy.ndarray of float64, shape (lat,)
        The latitudes of the grid.
    lon : numpy.ndarray of float64, shape (lon,)
        The longitudes of the grid.
    flat : numpy.ndarray of int64, shape (cell,)
        The flat index of each cell.
    weights : numpy.ndarray of float64, shape (cell,) or None
        The weight of each cell, or None.
    """

    def __init__(self, lat, lon, flat, weights=None):
        self.lat, self.lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        self.flat = np.asarray(flat, dtype=np.int64)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def take(self, index) -> "_Cells":
        """Return the cells at the given positions, on the same grid.

        Parameters
        ----------
        index : array_like of int or array_like of bool
            The positions (or a boolean selection) among the cells.

        Returns
        -------
        _Cells
            The selected cells, with their weights.
        """
        return _Cells(self.lat, self.lon, self.flat[index], None if self.weights is None else self.weights[index])


class EvaluationUnits:
    """An ordered table of units and what is needed to aggregate source grids onto them.

    Build one with `from_grid`, `from_mask`, `from_shapefile`, `from_weights` or `from_region`. The columns are
    ``id`` (unique strings), ``lat``, ``lon`` (centroid, NaN when unknown), ``name`` and ``parent`` when known.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per unit with an ``id`` column and, optionally, ``lat``, ``lon``, ``name`` and ``parent``. The ids
        are normalised to strings and must be unique; a missing ``lat`` or ``lon`` becomes NaN.
    kind : {"grid", "polygon", "region"}
        What a unit is.
    geometry : sequence of shapely.Geometry, optional
        The polygon of each unit, for a shapefile.
    subsets : dict of str to array_like of bool, optional
        Named boolean subsets, each with one entry per unit.
    weights : pandas.DataFrame, optional
        A source-to-unit weights table, from which `transform` is built.
    cells : _Cells, optional
        The grid and cells of grid and region units.
    weight_options : dict, optional
        The options of `hcwf.spatial.regrid.polygon_weights` for polygon units.
    mask_record : object, optional
        The record of the mask the units were selected with, kept for the run record.

    Attributes
    ----------
    table : pandas.DataFrame
        The units, one row each.
    kind : str
        ``"grid"``, ``"polygon"`` or ``"region"``.
    geometry : list of shapely.Geometry or None
        The polygons of a shapefile.
    subsets : dict of str to numpy.ndarray of bool, shape (unit,)
        Maps a subset name to a boolean array over the units.
    mask_record : object or None
        The record of the mask the units were selected with.

    Raises
    ------
    ValueError
        `kind` is not one of `KINDS`, the ids are not unique, or a subset does not have one entry per unit.
    """

    def __init__(self, table: pd.DataFrame, kind: str, *, geometry=None, subsets=None, weights=None, cells=None,
                 weight_options=None, mask_record=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        table = table.reset_index(drop=True)
        table["id"] = unit_ids.normalize(table["id"], context="unit IDs").to_numpy()
        if table["id"].duplicated().any():
            raise ValueError(f"Unit IDs must be unique; duplicates: {', '.join(table.loc[table['id'].duplicated(), 'id'].unique()[:10])}")
        for column in ("lat", "lon"):
            table[column] = np.nan if column not in table.columns else table[column].astype(float)
        self.table, self.kind = table, kind
        self.geometry = None if geometry is None else list(geometry)
        self.subsets = {name: np.asarray(mask, dtype=bool) for name, mask in (subsets or {}).items()}
        for name, mask in self.subsets.items():
            if mask.shape != (len(table),):
                raise ValueError(f"subset {name!r} has {mask.size} entries for {len(table)} units")
        self._weights = weights
        self._cells = cells
        self._weight_options = dict(weight_options or {})
        self.mask_record = mask_record
        self._transforms: dict = {}

    # -- the table -----------------------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of units.

        Returns
        -------
        int
            The number of rows of the table.
        """
        return len(self.table)

    @property
    def ids(self) -> np.ndarray:
        """The unit ids, as a string array of shape ``(unit,)``."""
        return self.table["id"].to_numpy(dtype=str)

    @property
    def lat(self) -> np.ndarray:
        """The centroid latitudes in degrees, a ``float64`` array of shape ``(unit,)`` (NaN when unknown)."""
        return self.table["lat"].to_numpy(dtype=float)

    @property
    def lon(self) -> np.ndarray:
        """The centroid longitudes in degrees, a ``float64`` array of shape ``(unit,)`` (NaN when unknown)."""
        return self.table["lon"].to_numpy(dtype=float)

    def column(self, name: str) -> np.ndarray | None:
        """Return an optional column (``name``, ``parent``), None when the units have none.

        Parameters
        ----------
        name : str
            The column to read.

        Returns
        -------
        numpy.ndarray or None
            The values of the column, one per unit, or None when the table has no such column.
        """
        return self.table[name].to_numpy() if name in self.table.columns else None

    # -- constructors --------------------------------------------------------------------------------------

    @classmethod
    def from_grid(cls, lat, lon, keep=None, *, convention: unit_ids.GridIdConvention | None = None,
                  explicit: unit_ids.GridIdConvention | None = None, mask_record=None) -> "EvaluationUnits":
        """Build grid units from the cells of a (lat, lon) grid where `keep` is True.

        The units are in latitude-major order. Ids are ``<lat>_<lon>`` under `convention`, else the explicit or
        inferred one.

        Parameters
        ----------
        lat : array_like of float, shape (lat,)
            The latitudes of the grid, in degrees.
        lon : array_like of float, shape (lon,)
            The longitudes of the grid, in degrees.
        keep : array_like of bool, shape (lat, lon), optional
            The cells to keep; None keeps all.
        convention : hcwf.spatial.ids.GridIdConvention, optional
            The convention to write the ids under; None resolves one from `explicit` or the coordinates.
        explicit : hcwf.spatial.ids.GridIdConvention, optional
            A convention set by the configuration, used when `convention` is None.
        mask_record : object, optional
            The record of the mask that gave `keep`, stored on the units.

        Returns
        -------
        EvaluationUnits
            The grid units, of kind ``"grid"``.

        Raises
        ------
        ValueError
            `keep` has the wrong shape or keeps no cell.
        """
        lat, lon = np.asarray(lat, dtype=float).reshape(-1), np.asarray(lon, dtype=float).reshape(-1)
        keep = np.ones((lat.size, lon.size), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
        if keep.shape != (lat.size, lon.size):
            raise ValueError(f"keep has shape {keep.shape} for a grid of {lat.size} x {lon.size}")
        flat = np.flatnonzero(keep.reshape(-1))
        if not flat.size:
            raise ValueError("No grid cell is kept.")
        cell_lat, cell_lon = lat[flat // lon.size], lon[flat % lon.size]
        convention = convention or unit_ids.resolve(explicit, lat=cell_lat, lon=cell_lon, context="grid units")
        ids = unit_ids.format_grid_ids(cell_lat, cell_lon, convention)
        table = pd.DataFrame({"id": ids, "lat": cell_lat, "lon": cell_lon})
        return cls(table, "grid", cells=_Cells(lat, lon, flat), mask_record=mask_record)

    @classmethod
    def from_mask(cls, mask, lat, lon, *, label: str = "data", **kwargs) -> "EvaluationUnits":
        """Build grid units from HCWF's mask on a grid.

        Parameters
        ----------
        mask : hcwf.spatial.masks.MaskSpec or dict or None
            The mask, or its mapping; None keeps every cell.
        lat : array_like of float, shape (lat,)
            The latitudes of the grid, in degrees.
        lon : array_like of float, shape (lon,)
            The longitudes of the grid, in degrees.
        label : str, default "data"
            What the grid is, for the messages of the mask.
        **kwargs : dict
            Passed on to `from_grid` (``convention``, ``explicit``).

        Returns
        -------
        EvaluationUnits
            The grid units of the cells the mask keeps.
        """
        if mask is None:
            return cls.from_grid(lat, lon, **kwargs)
        from hcwf.spatial.masks import MaskSpec, build_mask

        result = build_mask(MaskSpec.from_mapping(mask), lat, lon, label)
        return cls.from_grid(lat, lon, result.keep, mask_record=result.record, **kwargs)

    @classmethod
    def from_shapefile(cls, path, unit_id_col: str, unit_name_col: str | None = None,
                       parent_col: str | None = None, *, cell_area: str = "planar", measure: str = "area",
                       min_share: float = 1e-9, clip_to=None) -> "EvaluationUnits":
        """Build polygon units from the features of a shapefile or GeoJSON.

        Centroids are the polygons' centroids; the weights onto a grid are built from the overlap areas when
        `transform` is asked for one (options as in `hcwf.spatial.regrid.polygon_weights`).

        Parameters
        ----------
        path : str or os.PathLike
            The shapefile or GeoJSON.
        unit_id_col : str
            The attribute that holds the unit id.
        unit_name_col : str, optional
            The attribute that holds the unit name.
        parent_col : str, optional
            The attribute that holds the parent unit.
        cell_area : str, default "planar"
            How the area of a cell is measured (see `hcwf.spatial.regrid.polygon_weights`).
        measure : str, default "area"
            What the overlap is measured in (see `hcwf.spatial.regrid.polygon_weights`).
        min_share : float, default 1e-9
            The smallest overlap share kept (see `hcwf.spatial.regrid.polygon_weights`).
        clip_to : object, optional
            An outline the cells are clipped to (see `hcwf.spatial.regrid.polygon_weights`).

        Returns
        -------
        EvaluationUnits
            The polygon units, in file order.
        """
        import warnings

        import shapely

        table, geometry = read_shapes(path, unit_id_col, unit_name_col, parent_col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a geographic-CRS centroid is fine here
            centroid = shapely.centroid(np.array(geometry, dtype=object))
        table["lat"], table["lon"] = shapely.get_y(centroid), shapely.get_x(centroid)
        options = {"cell_area": cell_area, "measure": measure, "min_share": min_share, "clip_to": clip_to}
        return cls(table, "polygon", geometry=geometry, weight_options=options)

    @classmethod
    def from_weights(cls, weights, centroids=None, *, centroid_id_col: str | None = None) -> "EvaluationUnits":
        """Build units named by the targets of a weights table or CSV.

        The units are sorted, as `~hcwf.spatial.regrid.SparseTransform.from_weights` has them, with centroids from a
        table or CSV, else from ``<lat>_<lon>`` ids. The kind is grid when every id is a grid id, else polygon. The
        weights give `transform`.

        Parameters
        ----------
        weights : str or os.PathLike or pandas.DataFrame
            The weights table, or the path of its CSV (see `hcwf.spatial.regrid.read_weights`); its ``target_id``
            column names the units.
        centroids : str or os.PathLike or pandas.DataFrame, optional
            The centroids, as for `read_centroids`; None takes them from the ids.
        centroid_id_col : str, optional
            The id column of `centroids`.

        Returns
        -------
        EvaluationUnits
            The units, with the weights kept for `transform`.
        """
        weights = regrid.read_weights(weights)
        ids = sorted(weights["target_id"].unique())
        table = pd.DataFrame({"id": ids})
        if centroids is not None:
            table = table.merge(read_centroids(centroids, centroid_id_col), on="id", how="left")
        else:
            parsed = table["id"].str.extract(_GRID_ID)
            table["lat"], table["lon"] = parsed[0].astype(float), parsed[1].astype(float)
        kind = "grid" if table["lat"].notna().all() and table["id"].str.match(_GRID_ID).all() else "polygon"
        return cls(table, kind, weights=weights)

    @classmethod
    def from_region(cls, keep, lat, lon, *, id: str = "region", name: str | None = None,
                    weights: str = "cos_lat") -> "EvaluationUnits":
        """Build one unit that averages the cells of a mask.

        The cells are weighted by cos(latitude), or equally. The centroid is the cells' mean position.

        Parameters
        ----------
        keep : array_like of bool, shape (lat, lon)
            The cells of the region.
        lat : array_like of float, shape (lat,)
            The latitudes of the grid, in degrees.
        lon : array_like of float, shape (lon,)
            The longitudes of the grid, in degrees.
        id : str, default "region"
            The id of the unit.
        name : str, optional
            The name of the unit; None leaves the ``name`` column out.
        weights : {"cos_lat", "equal"}, default "cos_lat"
            How the cells are weighted.

        Returns
        -------
        EvaluationUnits
            One unit of kind ``"region"``.

        Raises
        ------
        ValueError
            `weights` is not one of the choices, `keep` does not match the grid, or it selects no cell.
        """
        if weights not in ("cos_lat", "equal"):
            raise ValueError(f"weights must be 'cos_lat' or 'equal', got {weights!r}")
        lat, lon = np.asarray(lat, dtype=float).reshape(-1), np.asarray(lon, dtype=float).reshape(-1)
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (lat.size, lon.size):
            raise ValueError(f"mask shape {keep.shape} does not match the grid {lat.size} x {lon.size}")
        flat = np.flatnonzero(keep.reshape(-1))
        if not flat.size:
            raise ValueError("mask selects no cells")
        cell_lat = lat[flat // lon.size]
        cell_weight = np.cos(np.deg2rad(cell_lat)) if weights == "cos_lat" else np.ones(flat.size)
        table = pd.DataFrame({"id": [id], "lat": [cell_lat.mean()], "lon": [lon[flat % lon.size].mean()]})
        if name is not None:
            table["name"] = name
        return cls(table, "region", cells=_Cells(lat, lon, flat, cell_weight))

    # -- subsets and area of interest ----------------------------------------------------------------------

    def subset_from_csv(self, name: str, path, id_col: str | None = None, *, explicit=None) -> "EvaluationUnits":
        """Return a copy with a subset named by a CSV list of unit ids (see `read_id_list`).

        The listed ids that are not units are ignored, as listing a cell outside the data is harmless.

        Parameters
        ----------
        name : str
            The name of the subset.
        path : str or os.PathLike
            The CSV list of ids.
        id_col : str, optional
            The id column of the list.
        explicit : hcwf.spatial.ids.GridIdConvention, optional
            A grid-id convention set by the configuration.

        Returns
        -------
        EvaluationUnits
            The same units with the subset added.
        """
        listed = read_id_list(path, id_col, explicit=explicit, authoritative_ids=pd.Series(self.ids, dtype=str))
        out = self._copy(self.table)
        out.subsets = {**self.subsets, name: np.isin(self.ids, listed.to_numpy())}
        log.info("subset %s: %d of %d units", name, int(out.subsets[name].sum()), len(self))
        return out

    def restrict(self, keep) -> "EvaluationUnits":
        """Return the units where `keep` is True (or in the named subset), in order, with their subsets and weights.

        Parameters
        ----------
        keep : str or array_like of bool, shape (unit,)
            A subset name or a boolean selection.

        Returns
        -------
        EvaluationUnits
            The selected units.
        """
        keep = self.subsets[keep] if isinstance(keep, str) else np.asarray(keep, dtype=bool)
        index = np.flatnonzero(keep)
        out = self._copy(self.table.iloc[index])
        out.geometry = None if self.geometry is None else [self.geometry[i] for i in index]
        out.subsets = {name: mask[index] for name, mask in self.subsets.items()}
        out._cells = None if self._cells is None else self._cells.take(index)
        if self._weights is not None:
            out._weights = self._weights[self._weights["target_id"].isin(set(out.ids))]
            out._weights.attrs["spatial_weights"] = True
        return out

    def in_box(self, box) -> np.ndarray:
        """Tell whether each unit's centroid is inside a box.

        A dict with any of ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max`` (inclusive; a missing bound is open)
        compares the raw longitudes, as onset_blending does. A region name (or alias) takes the region's box as HCWF
        does (`hcwf.spatial.select.region_box`: longitude conventions and the seam are handled). Units without a
        centroid are outside; no centroid at all is an error.

        Parameters
        ----------
        box : dict or str
            The bounds in degrees, or a region name.

        Returns
        -------
        numpy.ndarray of bool, shape (unit,)
            True for the units inside the box.

        Raises
        ------
        ValueError
            No unit has a centroid.
        """
        lat, lon = self.lat, self.lon
        if np.isnan(lat).all() or np.isnan(lon).all():
            raise ValueError("A box needs unit lat/lon: the units have no centroids (and no '<lat>_<lon>' ids).")
        if isinstance(box, str):
            return self._in_region_box(box)
        keep = np.ones(len(self), dtype=bool)
        for key, test in (("lat_min", lat.__ge__), ("lat_max", lat.__le__),
                          ("lon_min", lon.__ge__), ("lon_max", lon.__le__)):
            if box.get(key) is not None:
                keep &= test(float(box[key]))
        return keep

    def _in_region_box(self, region: str) -> np.ndarray:
        """Tell whether each unit's centroid is inside HCWF's box of a region.

        The box is separable, so cutting the distinct latitudes and longitudes with
        `~hcwf.spatial.select.region_box` (which also conforms the longitude convention) cuts the units.

        Parameters
        ----------
        region : str
            A region name or alias.

        Returns
        -------
        numpy.ndarray of bool, shape (unit,)
            True for the units inside the box.
        """
        import xarray as xr

        from hcwf.spatial.select import region_box

        lat_axis, lon_axis = np.unique(self.lat[np.isfinite(self.lat)]), np.unique(self.lon[np.isfinite(self.lon)])
        axes = xr.Dataset({"axis_lat": ("lat", lat_axis), "axis_lon": ("lon", lon_axis)},
                          coords={"lat": lat_axis, "lon": lon_axis})
        cut = region_box(axes, region=region)
        return np.isin(self.lat, cut["axis_lat"].values) & np.isin(self.lon, cut["axis_lon"].values)

    def in_outline(self, outline, boundary: str = "contains") -> np.ndarray:
        """Tell whether each unit's centroid is inside an outline.

        The outline is a shapely geometry or a GeoJSON / shapefile path (all its features are unioned). Centroid
        longitudes are wrapped to -180..180.

        Parameters
        ----------
        outline : shapely.Geometry or str or os.PathLike
            The outline, or the file that holds it.
        boundary : {"contains", "intersects"}, default "contains"
            ``"contains"`` leaves out a centroid on the edge (onset_blending); ``"intersects"`` keeps it (HCWF's
            centre rule for masks).

        Returns
        -------
        numpy.ndarray of bool, shape (unit,)
            True for the units inside the outline.

        Raises
        ------
        ValueError
            `boundary` is not one of the choices.
        """
        import shapely

        if boundary not in ("contains", "intersects"):
            raise ValueError("boundary must be 'contains' or 'intersects'")
        if isinstance(outline, (str, os.PathLike)):
            from hcwf.spatial.masks import shape_file_geometry

            outline = shape_file_geometry(str(outline))
        lat, lon = self.lat, ((self.lon + 180.0) % 360.0) - 180.0
        test = shapely.contains_xy if boundary == "contains" else shapely.intersects_xy
        return np.asarray(test(outline, lon, lat), dtype=bool)

    def in_area(self, *, box=None, outline=None, boundary: str = "contains") -> np.ndarray:
        """Tell which units are inside the area of interest: the box, else the outline.

        See `in_box` and `in_outline`. All units are inside when neither is given.

        Parameters
        ----------
        box : dict or str, optional
            The box, as for `in_box`.
        outline : shapely.Geometry or str or os.PathLike, optional
            The outline, as for `in_outline`; used when `box` is None.
        boundary : {"contains", "intersects"}, default "contains"
            How a centroid on the edge of the outline counts.

        Returns
        -------
        numpy.ndarray of bool, shape (unit,)
            True for the units inside the area.

        Raises
        ------
        AreaError
            No unit is inside.
        """
        if box is not None:
            keep = self.in_box(box)
        elif outline is not None:
            keep = self.in_outline(outline, boundary)
        else:
            return np.ones(len(self), dtype=bool)
        if not keep.any():
            raise AreaError("No cells fall inside the area of interest")
        log.info("area of interest: %d -> %d units", len(self), int(keep.sum()))
        return keep

    def _copy(self, table) -> "EvaluationUnits":
        """Return units with the given table and the same kind, geometry, weights, cells and subsets.

        Parameters
        ----------
        table : pandas.DataFrame
            The table of the new units (copied).

        Returns
        -------
        EvaluationUnits
            The copy; it shares the geometry, weights and cells with `self` and has its own transform cache.
        """
        out = EvaluationUnits(table.copy(), self.kind, geometry=self.geometry, weights=self._weights,
                              cells=self._cells, weight_options=self._weight_options, mask_record=self.mask_record)
        out.subsets = dict(self.subsets)
        return out

    # -- transforms ----------------------------------------------------------------------------------------

    def transform(self, source_lat, source_lon, *, tolerance: float | None = None,
                  convention: unit_ids.GridIdConvention | None = None, wrap_longitude: bool = False) -> SparseTransform:
        """Return the transform from the cells of a source grid to the units, compiled once per grid and tolerance.

        Weights from a table are matched to the grid's cells by exact coordinates, or within `tolerance` (the
        parity path for tables written by onset_blending is the exact one). Grid and region units use a
        selection when the grid holds all their cells within `tolerance` (default: HCWF's coordinate tolerance)
        and conservative area weights otherwise.

        Parameters
        ----------
        source_lat : array_like of float, shape (lat,)
            The latitudes of the source grid, in degrees.
        source_lon : array_like of float, shape (lon,)
            The longitudes of the source grid, in degrees.
        tolerance : float, optional
            The coordinate tolerance in degrees; None takes HCWF's default.
        convention : hcwf.spatial.ids.GridIdConvention, optional
            The grid-id convention used to match a weights table's ids.
        wrap_longitude : bool, default False
            Put a weight table's longitudes in the source grid's convention before matching (see
            `~hcwf.spatial.regrid.SparseTransform.from_weights`).

        Returns
        -------
        hcwf.spatial.regrid.SparseTransform
            The sparse transform from the ``lat * lon`` source cells to the units.
        """
        source_lat, source_lon = np.asarray(source_lat, dtype=float), np.asarray(source_lon, dtype=float)
        key = (source_lat.tobytes(), source_lon.tobytes(), tolerance, convention, wrap_longitude)
        if key not in self._transforms:
            self._transforms[key] = self._compile(source_lat, source_lon, tolerance, convention, wrap_longitude)
        return self._transforms[key]

    def _compile(self, lat, lon, tolerance, convention, wrap_longitude=False) -> SparseTransform:
        """Build the transform of `transform` for one source grid.

        Parameters
        ----------
        lat : numpy.ndarray of float64, shape (lat,)
            The latitudes of the source grid.
        lon : numpy.ndarray of float64, shape (lon,)
            The longitudes of the source grid.
        tolerance : float or None
            The coordinate tolerance in degrees; None takes HCWF's default.
        convention : hcwf.spatial.ids.GridIdConvention or None
            The grid-id convention used to match a weights table's ids.
        wrap_longitude : bool, default False
            Whether to wrap a weights table's longitudes to the grid's convention.

        Returns
        -------
        hcwf.spatial.regrid.SparseTransform
            The transform, with one target per unit.
        """
        targets = tuple(self.ids)
        if self._weights is not None:
            return SparseTransform.from_weights(self._weights, lat, lon, target_ids=targets, convention=convention,
                                                tolerance=tolerance, wrap_longitude=wrap_longitude)
        if self.kind == "polygon":
            table = regrid.polygon_weights(self.geometry, self.ids, lat, lon, **self._weight_options)
            return SparseTransform.from_weights(table, lat, lon, target_ids=targets, tolerance=tolerance)
        cells = self._cells
        n_cells = lat.size * lon.size
        tol = regrid.default_tolerance() if tolerance is None else tolerance
        if np.array_equal(lat, cells.lat) and np.array_equal(lon, cells.lon):  # the units' own grid: nothing to match
            matrix = regrid.selection(cells.flat, n_cells, cells.weights)
        else:
            lat_pos = regrid.match_axis(cells.lat[cells.flat // cells.lon.size], lat, tol)
            lon_pos = regrid.match_axis(cells.lon[cells.flat % cells.lon.size], lon, tol)
            if (lat_pos >= 0).all() and (lon_pos >= 0).all():
                matrix = regrid.selection(lat_pos * lon.size + lon_pos, n_cells, cells.weights)
            else:
                to_cells = regrid.grid_weights(lat, lon, cells.lat, cells.lon)[cells.flat]
                if cells.weights is not None:
                    to_cells = sparse.diags(cells.weights) @ to_cells
                matrix = to_cells.tocsr()
        if self.kind == "region":
            matrix = sparse.csr_matrix(matrix.sum(axis=0))
        return SparseTransform(matrix, None, targets, keys="coordinate", grid=(lat, lon))


# ---------------------------------------------------------------------------------------------------------------
# Where the units come from: one specification for a region's `units:` block and a run's `units:` section
# ---------------------------------------------------------------------------------------------------------------

UNITS_KEYS = {f.key: f.nested_name for f in schema.FIELDS if f.section == "units"}
"""The flat config keys of the ``units:`` section and the `UnitsSpec` field each sets (declared once, in the schema)."""
_PER_SOURCE = ("units_mode", "units_weights")
"""Keys that may be a mapping from a source (``"observations"`` or a model's name; ``"*"`` for the rest) to a value."""
MODES = ("cells", "regrid", "cell_transform")
"""The values of ``units.mode``."""
BOUNDARIES = ("contains", "intersects")
"""The values of ``units.boundary``."""
DISSEMINATION_SCOPES = ("inputs", "training")
"""The values of ``units.dissemination_scope``."""


@dataclass(frozen=True)
class UnitsSpec:
    """How a source's cells (or pre-aggregated units) become evaluation units, and which of them are used.

    Source of the units: a ``shapefile`` (polygons, with ``unit_id_col``; names and parents optional) or a ``weights``
    table (units named by its targets, with optional ``centroids``), else the cells of the source grid. ``mode`` says
    how rain reaches them: ``"cells"`` (the file's own cells, or its own units when it has a ``"unit"`` dimension),
    ``"regrid"`` (weights from the source grid; millimetre factors applied after averaging, missing rain judged on the
    units) or ``"cell_transform"`` (a source-to-unit weights table; factors applied before averaging, missing rain
    judged on the source cells); None is ``"cells"`` without a shapefile or weights, else ``"regrid"``.

    The rest restricts or labels the units: ``box`` (sorted (key, value) pairs of ``lat_min`` / ``lat_max`` /
    ``lon_min`` / ``lon_max``) or an ``outline`` (``boundary`` ``"contains"`` leaves out a centroid on the edge,
    ``"intersects"`` keeps it), the ``dissemination`` list (``dissemination_scope`` ``"inputs"`` restricts the units
    read to it, ``"training"`` only marks it as the subset ``dissemination``), and how grid ids are written
    (``id_digits``, ``id_format``).

    Paths are as written; `resolve_path` expands ``${ENV}`` and ``~`` and anchors a relative one at ``base`` (the
    folder of the YAML that wrote it).
    """

    base: Path = Path(".")
    """The folder a relative path is anchored at."""
    shapefile: str | None = None
    """The shapefile or GeoJSON of polygon units."""
    unit_id_col: str | None = None
    """The attribute of the shapefile that holds the unit id."""
    unit_name_col: str | None = None
    """The attribute of the shapefile that holds the unit name."""
    parent_col: str | None = None
    """The attribute of the shapefile that holds the parent unit."""
    weights: str | None = None
    """A source-to-unit weights table (CSV) that names the units."""
    centroids: str | None = None
    """A table of unit centroids (CSV)."""
    centroid_id_col: str | None = None
    """The id column of the centroids table."""
    dissemination: str | None = None
    """A CSV list of the dissemination units."""
    dissemination_id_col: str | None = None
    """The id column of the dissemination list."""
    mode: str | None = None
    """How rain reaches the units, one of `MODES`; None picks it from the other fields (see `how`)."""
    box: tuple | None = None
    """The area of interest as sorted (key, value) pairs; a mapping given is converted."""
    outline: str | None = None
    """A GeoJSON or shapefile outline of the area of interest."""
    boundary: str = "contains"
    """Whether a centroid on the edge of the outline is inside, one of `BOUNDARIES`."""
    dissemination_scope: str = "inputs"
    """What the dissemination list does, one of `DISSEMINATION_SCOPES`."""
    id_digits: int | None = None
    """The number of decimals of the coordinates in grid ids."""
    id_format: str = "trimmed"
    """How grid ids are written (``"trimmed"`` or fixed width)."""

    def __post_init__(self):
        """Check the fields and turn a mapping `box` into sorted pairs.

        Raises
        ------
        UnitsSpecError
            A choice is not one of its values, a shapefile and a weights table are both named, the mode does not
            match them, or a shapefile comes without ``unit_id_col``.
        """
        if self.mode is not None and self.mode not in MODES:
            raise UnitsSpecError(f"units mode must be one of {MODES}, got {self.mode!r}")
        if self.boundary not in BOUNDARIES:
            raise UnitsSpecError(f"units boundary must be one of {BOUNDARIES}, got {self.boundary!r}")
        if self.dissemination_scope not in DISSEMINATION_SCOPES:
            raise UnitsSpecError(f"units dissemination_scope must be one of {DISSEMINATION_SCOPES}, "
                                 f"got {self.dissemination_scope!r}")
        if self.shapefile and self.weights:
            raise UnitsSpecError("units name a shapefile or a weights table, not both")
        if self.how == "cells" and (self.shapefile or self.weights):
            raise UnitsSpecError("units mode 'cells' reads the file's own cells; it takes no shapefile or weights")
        if self.how != "cells" and not (self.shapefile or self.weights):
            raise UnitsSpecError(f"units mode {self.how!r} needs a shapefile or a weights table")
        if self.shapefile and not self.unit_id_col:
            raise UnitsSpecError("units with a shapefile need unit_id_col")
        if isinstance(self.box, dict):
            object.__setattr__(self, "box", tuple(sorted((k, float(v)) for k, v in self.box.items() if v is not None)))

    @classmethod
    def from_flat(cls, values, source: str = "observations") -> "UnitsSpec | None":
        """Build the evaluation units specification of a source from a flat config record.

        This is the one place the flat ``units_*`` keys (declared in the schema's ``units`` section, field =
        ``nested_name``) become a `UnitsSpec`; ``RunConfig.from_flat`` calls it once per source.

        The keys given extend the region's declared units (its shapefile or weights, replaced when the config names
        the other): a config with only ``units_box`` still reads the region's polygons. ``units_mode`` and
        ``units_weights`` may be mappings by source, as models on different grids have their own weights (the source
        is ``"observations"`` or a model's name).

        Parameters
        ----------
        values : collections.abc.Mapping
            The flat config record.
        source : str, default "observations"
            The source the specification is for: ``"observations"`` or a model's name.

        Returns
        -------
        UnitsSpec or None
            The specification, or None when the record has no ``units_*`` keys for the source (the grid cells are
            used).
        """
        given = {}
        for key, field in UNITS_KEYS.items():
            value = values.get(key)
            if isinstance(value, dict) and key in _PER_SOURCE:
                value = value.get(source, value.get("*"))
            if value is not None:
                given[field] = value
        if not given:
            return None
        if "weights" in given:   # a source with its own weights table does not use the shared shapefile
            for field in ("shapefile", "unit_id_col", "unit_name_col", "parent_col"):
                given.pop(field, None)
        base = None
        if values.get("region"):
            from hcwf.onset import regions

            try:
                base = regions.units_spec(str(values["region"]))
            except RegionError:
                base = None
        if base is None:
            return cls(**given)
        if "weights" in given:
            base = replace(base, shapefile=None, unit_id_col=None, unit_name_col=None, parent_col=None)
        elif "shapefile" in given:
            base = replace(base, weights=None)
        return replace(base, **given)

    @property
    def how(self) -> str:
        """The mode in force: ``"cells"``, ``"regrid"`` or ``"cell_transform"``."""
        if self.mode is not None:
            return self.mode
        return "regrid" if (self.shapefile or self.weights) else "cells"

    @property
    def convention(self):
        """The configured grid-id convention, or None (established or inferred per table)."""
        return unit_ids.configured(self.id_digits, self.id_format)

    def resolve_path(self, value: str) -> Path:
        """Return the path with ``${ENV}`` and ``~`` expanded, relative to `base`.

        Parameters
        ----------
        value : str
            The path as written.

        Returns
        -------
        pathlib.Path
            An absolute path stays as it is; a relative one is anchored at `base`.

        Raises
        ------
        UnitsSpecError
            The path uses an environment variable that is not set.
        """
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        if re.search(r"\$\{?\w+\}?", expanded):
            raise UnitsSpecError(f"units path '{value}' uses an environment variable that is not set. "
                                 "Fix: export it or write the path out")
        return Path(self.base) / expanded if not os.path.isabs(expanded) else Path(expanded)

    def files(self) -> list[Path]:
        """Return the files this specification names that exist (what a cache key must follow).

        Returns
        -------
        list of pathlib.Path
            The existing shapefile, weights, centroids, dissemination and outline files, sorted.
        """
        named = [self.shapefile, self.weights, self.centroids, self.dissemination, self.outline]
        found = [self.resolve_path(n) for n in named if n]
        return sorted(p for p in found if p.is_file())

    def build(self, lat=None, lon=None, ids=None) -> EvaluationUnits:
        """Build the units of a source.

        They are the units of the file's `ids` when it has a ``"unit"`` dimension (centroids from the spec's table),
        else of the source grid (`lat`, `lon`) in ``"cells"`` mode, else the shapefile's or the weights' units. The
        dissemination list is a boolean subset of that name.

        Parameters
        ----------
        lat : array_like of float, shape (lat,), optional
            The latitudes of the source grid, in degrees; needed in ``"cells"`` mode.
        lon : array_like of float, shape (lon,), optional
            The longitudes of the source grid, in degrees; needed in ``"cells"`` mode.
        ids : array_like of str, shape (unit,), optional
            The unit ids of a file that is already aggregated.

        Returns
        -------
        EvaluationUnits
            The units, with the ``dissemination`` subset when the specification has a list.
        """
        convention = self.convention
        if ids is not None:
            table = pd.DataFrame({"id": np.asarray(ids).astype(str)})
            if self.centroids:
                table = table.merge(read_centroids(self.resolve_path(self.centroids), self.centroid_id_col),
                                    on="id", how="left")
            units = EvaluationUnits(table, "polygon")
        elif self.how == "cells":
            units = EvaluationUnits.from_grid(lat, lon, explicit=convention)
        elif self.shapefile:
            units = EvaluationUnits.from_shapefile(self.resolve_path(self.shapefile), self.unit_id_col,
                                                   self.unit_name_col, self.parent_col)
        else:
            centroids = self.resolve_path(self.centroids) if self.centroids else None
            units = EvaluationUnits.from_weights(self.resolve_path(self.weights), centroids,
                                                 centroid_id_col=self.centroid_id_col)
        if self.dissemination:
            units = units.subset_from_csv("dissemination", self.resolve_path(self.dissemination),
                                          self.dissemination_id_col, explicit=convention)
        return units

    def kept(self, units: EvaluationUnits, keep_ids=None) -> np.ndarray:
        """Return the positions of the units used, ordered by id.

        A unit is used when it is inside the box or outline, in the dissemination list (scope ``"inputs"``) and among
        `keep_ids`.

        Parameters
        ----------
        units : EvaluationUnits
            The units to choose from.
        keep_ids : collections.abc.Iterable of str, optional
            The ids that may be used; None keeps every id.

        Returns
        -------
        numpy.ndarray of int64, shape (kept,)
            The positions in `units`, sorted by id.

        Raises
        ------
        ValueError
            No unit is kept.
        """
        keep = units.in_area(box=None if self.box is None else dict(self.box),
                             outline=None if self.outline is None else self.resolve_path(self.outline),
                             boundary=self.boundary)
        if self.dissemination and self.dissemination_scope == "inputs":
            keep = keep & units.subsets["dissemination"]
        if keep_ids is not None:
            keep = keep & np.isin(units.ids, np.asarray(sorted(keep_ids), dtype=str))
        positions = np.flatnonzero(keep)
        if not len(positions):
            raise ValueError("No unit of the file is kept")
        return positions[np.argsort(units.ids[positions], kind="stable")]

    def select(self, lat=None, lon=None, ids=None, keep_ids=None) -> tuple[EvaluationUnits, np.ndarray]:
        """Build the units and choose those used: `build` then `kept`.

        Parameters
        ----------
        lat : array_like of float, shape (lat,), optional
            The latitudes of the source grid, as for `build`.
        lon : array_like of float, shape (lon,), optional
            The longitudes of the source grid, as for `build`.
        ids : array_like of str, shape (unit,), optional
            The unit ids of an aggregated file, as for `build`.
        keep_ids : collections.abc.Iterable of str, optional
            The ids that may be used, as for `kept`.

        Returns
        -------
        units : EvaluationUnits
            All the units of the source.
        positions : numpy.ndarray of int64, shape (kept,)
            The positions of those used, sorted by id.
        """
        units = self.build(lat, lon, ids)
        return units, self.kept(units, keep_ids)
