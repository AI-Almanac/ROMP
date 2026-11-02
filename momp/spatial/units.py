"""Evaluation units: the places onsets are computed, scored and blended for.

A unit is a grid cell, a polygon (an admin unit) or a region (a set of cells averaged into one series). One
table describes them, whatever the kind: an ordered, unique string `id`, a centroid `lat` / `lon`, and optionally
`name`, `parent`, the polygon `geometry` and named boolean subsets (for instance `dissemination`, `training`).

`EvaluationUnits.transform(source_lat, source_lon)` gives the `SparseTransform` from the cells of a source grid
(observations or a model) to the units, compiled once per grid. Rain is read on the source grid, averaged with
`observed_mean` onto the units in float mm and only then rounded to tenths (see `momp.onset.engine.regional`
for the one-unit case), so the engine runs on a ("unit",) spatial axis unchanged.

How the transform comes about:

* grid units (from a grid and ROMP's mask): a 0/1 selection when the source grid holds every unit cell,
  otherwise conservative area weights from the source cells to the unit cells;
* polygon units: cell-polygon overlap areas from the shapes, or weights from a table;
* a region: cos-lat (or equal) weights on its cells, composed with the grid weights for another grid.

String ids appear only at the edges (`momp.spatial.ids`): reading and writing tables and lists.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse

from . import ids as unit_ids
from . import regrid
from .regrid import SparseTransform

log = logging.getLogger(__name__)

KINDS = ("grid", "polygon", "region")
_GRID_ID = re.compile(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
_LAT_NAMES = ("lat", "center_lat", "latitude")
_LON_NAMES = ("lon", "center_lon", "longitude")


class AreaError(ValueError):
    """No unit falls inside the area of interest."""


def read_centroids(source, id_col: str | None = None) -> pd.DataFrame:
    """Unit centroids as a table of (id, lat, lon), from a CSV path or a DataFrame.

    The id column is `id_col`, else the first of target_id, id, adm3_name, else the first column; latitude and
    longitude are lat / center_lat / latitude and lon / center_lon / longitude. One row per id.
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
    """The ids of a CSV list of units (for instance the dissemination units), as normalised strings.

    The id column is `id_col`, else id / adm3_name, else grid ids made from lat and lon, else (with a warning) the
    first column. Grid ids are written under the convention the `authoritative_ids` (the units') establish.
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
    """The features of a shapefile or GeoJSON as (table of id [, name, parent], geometries) in EPSG:4326.

    Ids are normalised strings and must be unique. A file without a coordinate system is taken to be EPSG:4326.
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
    """The grid a grid or region unit lives on: axes, the flat (latitude-major) index of each unit's cell and, for a
    region, the weight of each of its cells."""

    def __init__(self, lat, lon, flat, weights=None):
        self.lat, self.lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        self.flat = np.asarray(flat, dtype=np.int64)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def take(self, index) -> "_Cells":
        return _Cells(self.lat, self.lon, self.flat[index], None if self.weights is None else self.weights[index])


class EvaluationUnits:
    """An ordered table of units and what is needed to aggregate source grids onto them.

    Build one with `from_grid`, `from_mask`, `from_shapefile`, `from_weights` or `from_region`. The columns are
    `id` (unique strings), `lat`, `lon` (centroid, NaN when unknown), `name` and `parent` when known; `kind` is
    "grid", "polygon" or "region"; `geometry` holds the polygons of a shapefile; `subsets` maps a name to a boolean
    array over the units.
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
        return len(self.table)

    @property
    def ids(self) -> np.ndarray:
        return self.table["id"].to_numpy(dtype=str)

    @property
    def lat(self) -> np.ndarray:
        return self.table["lat"].to_numpy(dtype=float)

    @property
    def lon(self) -> np.ndarray:
        return self.table["lon"].to_numpy(dtype=float)

    def column(self, name: str) -> np.ndarray | None:
        """An optional column (name, parent), None when the units have none."""
        return self.table[name].to_numpy() if name in self.table.columns else None

    def to_frame(self) -> pd.DataFrame:
        """The table with each subset as a boolean column."""
        out = self.table.copy()
        for name, mask in self.subsets.items():
            out[name] = mask
        return out

    # -- constructors --------------------------------------------------------------------------------------

    @classmethod
    def from_grid(cls, lat, lon, keep=None, *, convention: unit_ids.GridIdConvention | None = None,
                  explicit: unit_ids.GridIdConvention | None = None, mask_record=None) -> "EvaluationUnits":
        """The cells of a (lat, lon) grid where the boolean (nlat, nlon) `keep` is True (all when None), as grid
        units in latitude-major order. Ids are "<lat>_<lon>" under `convention`, else the explicit or inferred one."""
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
        """Grid units from ROMP's mask (a `MaskSpec` or its mapping; None keeps every cell) on a grid."""
        if mask is None:
            return cls.from_grid(lat, lon, **kwargs)
        from momp.utils.mask import MaskSpec, build_mask

        result = build_mask(MaskSpec.from_mapping(mask), lat, lon, label)
        return cls.from_grid(lat, lon, result.keep, mask_record=result.record, **kwargs)

    @classmethod
    def from_shapefile(cls, path, unit_id_col: str, unit_name_col: str | None = None,
                       parent_col: str | None = None, *, cell_area: str = "planar", measure: str = "area",
                       min_share: float = 1e-9, clip_to=None) -> "EvaluationUnits":
        """Polygon units from the features of a shapefile or GeoJSON. Centroids are the polygons' centroids; the
        weights onto a grid are built from the overlap areas when `transform` is asked for one (options as in
        `polygon_weights`)."""
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
        """Units named by the targets of a weights table or CSV (sorted, as `SparseTransform.from_weights` has
        them), with centroids from a table or CSV, else from "<lat>_<lon>" ids. Kind is grid when every id is a
        grid id, else polygon. The weights give `transform`."""
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
        """One unit averaging the cells of a boolean (nlat, nlon) mask: cos(latitude) weights, or equal ones.
        This is what `momp.onset.engine.regional` computes. The centroid is the cells' mean position."""
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

    def with_subset(self, name: str, selection) -> "EvaluationUnits":
        """A copy with the named subset: a boolean array over the units, or the ids it holds (ids the units lack
        are an error)."""
        selection = np.asarray(selection)
        if selection.dtype == bool:
            mask = selection
        else:
            wanted = pd.Index(selection.astype(str))
            missing = wanted[~wanted.isin(self.ids)]
            if len(missing):
                raise ValueError(f"subset {name!r}: {len(missing)} ids are not units, e.g. {', '.join(missing[:10])}")
            mask = np.isin(self.ids, wanted)
        out = self._copy(self.table)
        out.subsets = {**self.subsets, name: mask}
        return out

    def subset_from_csv(self, name: str, path, id_col: str | None = None, *, explicit=None) -> "EvaluationUnits":
        """A copy with a subset named by a CSV list of unit ids (see `read_id_list`); the listed ids that are
        not units are ignored, as listing a cell outside the data is harmless."""
        listed = read_id_list(path, id_col, explicit=explicit, authoritative_ids=pd.Series(self.ids, dtype=str))
        out = self._copy(self.table)
        out.subsets = {**self.subsets, name: np.isin(self.ids, listed.to_numpy())}
        log.info("subset %s: %d of %d units", name, int(out.subsets[name].sum()), len(self))
        return out

    def restrict(self, keep) -> "EvaluationUnits":
        """The units where `keep` is True or the named subset, in order, with their subsets and weights."""
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
        """Whether each unit's centroid is inside a box.

        A dict with any of lat_min, lat_max, lon_min, lon_max (inclusive; a missing bound is open) compares
        the raw longitudes, as onset_blending does. A region name (or alias) takes the region's box as ROMP
        does (`momp.utils.region.region_box`: longitude conventions and the seam are handled). Units without a
        centroid are outside; no centroid at all is an error.
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
        """ROMP's region box on the units' centroid axes: the box is separable, so cutting the distinct latitudes
        and longitudes with `region_box` (which also conforms the longitude convention) cuts the units."""
        import xarray as xr

        from momp.utils.region import region_box

        lat_axis, lon_axis = np.unique(self.lat[np.isfinite(self.lat)]), np.unique(self.lon[np.isfinite(self.lon)])
        axes = xr.Dataset({"axis_lat": ("lat", lat_axis), "axis_lon": ("lon", lon_axis)},
                          coords={"lat": lat_axis, "lon": lon_axis})
        cut = region_box(axes, region=region)
        return np.isin(self.lat, cut["axis_lat"].values) & np.isin(self.lon, cut["axis_lon"].values)

    def in_outline(self, outline, boundary: str = "contains") -> np.ndarray:
        """Whether each unit's centroid is inside an outline: a shapely geometry or a GeoJSON / shapefile path (all
        its features are unioned). boundary "contains" leaves out a centroid on the edge (onset_blending);
        "intersects" keeps it (ROMP's centre rule for masks). Centroid longitudes are wrapped to -180..180."""
        import shapely

        if boundary not in ("contains", "intersects"):
            raise ValueError("boundary must be 'contains' or 'intersects'")
        if isinstance(outline, (str, os.PathLike)):
            from momp.utils.mask import shape_file_geometry

            outline = shape_file_geometry(str(outline))
        lat, lon = self.lat, ((self.lon + 180.0) % 360.0) - 180.0
        test = shapely.contains_xy if boundary == "contains" else shapely.intersects_xy
        return np.asarray(test(outline, lon, lat), dtype=bool)

    def in_area(self, *, box=None, outline=None, boundary: str = "contains") -> np.ndarray:
        """Units inside the area of interest: the box, else the outline (see `in_box`, `in_outline`); all when
        neither is given. No unit inside is an `AreaError`."""
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
        out = EvaluationUnits(table.copy(), self.kind, geometry=self.geometry, weights=self._weights,
                              cells=self._cells, weight_options=self._weight_options, mask_record=self.mask_record)
        out.subsets = dict(self.subsets)
        return out

    # -- transforms ----------------------------------------------------------------------------------------

    def transform(self, source_lat, source_lon, *, tolerance: float | None = None,
                  convention: unit_ids.GridIdConvention | None = None, wrap_longitude: bool = False) -> SparseTransform:
        """The transform from the cells of a source grid (axes `source_lat`, `source_lon`) to the units, compiled
        once per grid (and tolerance).

        Weights from a table are matched to the grid's cells by exact coordinates, or within `tolerance` (the
        parity path for tables written by onset_blending is the exact one). Grid and region units use a
        selection when the grid holds all their cells within `tolerance` (default: ROMP's coordinate tolerance)
        and conservative area weights otherwise. `wrap_longitude` puts a weight table's longitudes in the source
        grid's convention before matching (see `SparseTransform.from_weights`).
        """
        source_lat, source_lon = np.asarray(source_lat, dtype=float), np.asarray(source_lon, dtype=float)
        key = (source_lat.tobytes(), source_lon.tobytes(), tolerance, convention, wrap_longitude)
        if key not in self._transforms:
            self._transforms[key] = self._compile(source_lat, source_lon, tolerance, convention, wrap_longitude)
        return self._transforms[key]

    def _compile(self, lat, lon, tolerance, convention, wrap_longitude=False) -> SparseTransform:
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

    def clear_transforms(self) -> None:
        """Forget the compiled transforms (they are kept per source grid)."""
        self._transforms.clear()
