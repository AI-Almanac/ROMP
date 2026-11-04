"""Build region masks on each dataset's own grid, never by regridding rainfall.

A run config's ``mask:`` section (see `MaskSpec`) names the cells to evaluate by one or more of

* ``land`` and ``country``: Natural Earth polygons (cartopy), rasterised on the data grid by the cell rule;
* ``file``: a 0/1 NetCDF mask, or a shapefile / GeoJSON;
* ``condition``: a threshold on a field (``above`` / ``below``), for example an elevation file.

All parts given intersect. The final mask is built separately for the observations and for each model,
on that dataset's own latitude/longitude grid:

* A shape is rasterised by the cell rule: ``centre`` keeps the cells whose centre is inside it,
  ``overlap_fraction`` keeps the cells with at least that share of their area (planar, in degrees squared)
  inside it. Longitudes of the cells are wrapped to -180..180 for the test.
* A mask file is first brought to the data's conventions (latitude order, longitude convention, axis
  names). If its coordinates then match the data grid within a tolerance (one hundredth of a cell) it is used
  directly. Otherwise the MASK (never the rainfall) is regridded onto the data grid: a mask with finer cells
  keeps a data cell when the area share of kept mask cells inside it is at least ``overlap_fraction``; a mask
  with coarser (or equal) cells gives each data cell the value of the nearest mask cell. A mask that does not
  cover the data extent is an error that names both extents.

Nothing is silent: `build_mask` returns the lines describing the sources, rule, alignment actions and kept
cell count, `region_mask` prints them once per dataset and grid, saves a small PNG of the kept cells to
``dir_fig/mask_<dataset>.png`` and keeps a record (sources with content hashes, rule, kept counts) that the
resolved config and the NetCDF outputs carry (`records`, `mask_attrs`).

The module also holds the implicit masks of a region run (the land and country masks, `default_mask`, `default_keep`,
`apply_default_mask`) and the older helpers that apply a country shapefile, a land mask or a NetCDF mask to a dataset.
`hcwf.spatial.select` cuts data to a region and calls these to apply the masks; `hcwf.spatial.units` builds grid units
from `build_mask`.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from typing import Union

import numpy as np
import xarray as xr

from hcwf.cache import memo
from hcwf.config.datasets import suggest
from hcwf.errors import MaskError
from hcwf.io.dims import ROLE_ALIASES, find_dim
import logging

__all__ = [
    "MaskSpec", "absorb_records", "get_shp", "mask_attrs", "polygon_mask", "record_lines", "records", "reset",
    "shp_mask",
]

log = logging.getLogger(__name__)

RESOLUTIONS = ("10m", "50m", "110m")
"""Natural Earth resolutions for ``land`` and ``country``."""
SHAPE_SUFFIXES = (".shp", ".geojson", ".json", ".gpkg")
"""File suffixes read as a shape (everything else is read as NetCDF)."""
_SIDECARS = (".dbf", ".shx", ".prj", ".cpg")
"""The files next to a ``.shp`` that `file_hash` covers."""
_NE_COUNTRIES = ("cultural", "admin_0_countries")
"""The Natural Earth (category, name) of the country shapes."""
_NE_LAND = ("physical", "land")
"""The Natural Earth (category, name) of the land shapes."""
_COUNTRY_FIELDS = ("NAME", "NAME_LONG", "ADM0_A3")
"""The attributes of a Natural Earth country that a ``mask.country`` name is matched against."""
_EPS = 1e-9
"""The slack of comparisons between areas, shares and coordinates."""


@dataclass(frozen=True)
class MaskSpec:
    """The ``mask:`` section of a run config.

    The parts given intersect. How a shape decides whether a data cell is kept is the cell rule (``cell_rule`` and
    ``overlap_fraction``), which also turns finer mask files into data cells.

    The last four fields (``land_resolution``, ``boundary``, ``regrid``, ``allow_empty``) are not read from a config;
    `default_mask` sets them for the implicit masks.
    """

    land: bool = False
    """Keep land cells only."""
    country: tuple = ()
    """Country names (Natural Earth ``NAME``, ``NAME_LONG`` or ``ADM0_A3``, any case)."""
    file: str | None = None
    """A 0/1 NetCDF mask (see ``variable``, ``keep_value``) or a shapefile / GeoJSON."""
    variable: str | None = None
    """The variable of the mask file; None picks the only one on the (lat, lon) axes."""
    keep_value: float = 1
    """The value of the mask variable that marks a kept cell."""
    condition: dict | None = None
    """A mask from a field: a mapping with ``file``, ``variable``, ``above`` and ``below``."""
    cell_rule: str = "centre"
    """How a shape is turned into data cells: ``"centre"`` or ``"overlap_fraction"``."""
    overlap_fraction: float = 0.5
    """The share of a cell's area that must be inside a shape (or kept mask cells) for the cell to be kept."""
    resolution: str = "50m"
    """The Natural Earth resolution of land and country shapes, one of `RESOLUTIONS`."""
    land_resolution: str | None = None
    """The resolution of the land shape if not ``resolution``."""
    boundary: str = "inclusive"
    """For the centre rule, ``"inclusive"`` (a cell centre on the shape's edge is inside) or ``"strict"``."""
    regrid: str = "area"
    """How a finer mask file is brought to the data grid: ``"area"`` (kept share) or ``"nearest"`` mask cell."""
    allow_empty: bool = False
    """Whether a mask that keeps no cell is not an error."""

    @classmethod
    def from_mapping(cls, raw) -> "MaskSpec":
        """Build the spec from the (normalised) mapping of the config.

        Parameters
        ----------
        raw : MaskSpec or collections.abc.Mapping
            The ``mask:`` section; a `MaskSpec` is returned as it is. A single country name is made a tuple.

        Returns
        -------
        MaskSpec
            The spec; the fields that a config does not set keep their defaults.
        """
        if isinstance(raw, MaskSpec):
            return raw
        raw = dict(raw)
        country = raw.get("country") or ()
        if isinstance(country, str):
            country = (country,)
        return cls(land=bool(raw.get("land", False)), country=tuple(country), file=raw.get("file"),
                   variable=raw.get("variable"), keep_value=raw.get("keep_value", 1),
                   condition=dict(raw["condition"]) if raw.get("condition") else None,
                   cell_rule=raw.get("cell_rule", "centre"),
                   overlap_fraction=float(raw.get("overlap_fraction", 0.5)),
                   resolution=raw.get("resolution", "50m"))

    def key(self) -> str:
        """Return a text identifying the spec, for caching.

        Returns
        -------
        str
            The fields as JSON with sorted keys.
        """
        return json.dumps(self.__dict__, sort_keys=True, default=str)


@dataclass
class MaskResult:
    """A mask on one dataset's grid: the boolean cells kept, the lines that describe it and its record."""

    keep: np.ndarray
    """The cells kept, ``bool`` of shape ``(lat, lon)``."""
    lines: list
    """The printable lines that describe the sources, rule, alignment and kept count."""
    record: dict
    """The record of the mask (sources with content hashes, rule, kept counts)."""
    outline: object = None
    """The shape of the mask, for the PNG; None when the mask has no shape."""
    lat: np.ndarray = field(default=None)
    """The latitudes of the grid, ``float64`` of shape ``(lat,)``."""
    lon: np.ndarray = field(default=None)
    """The longitudes of the grid, ``float64`` of shape ``(lon,)``."""


# ----------------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------------

def cell_edges(centres, fallback=None) -> np.ndarray:
    """Compute the cell edges of ascending cell centres: midpoints between centres, half a step beyond the ends.

    A single centre needs ``fallback``, the cell size to assume.

    Parameters
    ----------
    centres : array_like of float, shape (n,)
        The ascending cell centres, in degrees.
    fallback : float, optional
        The cell size in degrees to assume for a single centre.

    Returns
    -------
    numpy.ndarray of float64, shape (n + 1,)
        The cell edges.

    Raises
    ------
    MaskError
        There is a single centre and no ``fallback``.
    """
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        if not fallback:
            raise MaskError("cannot tell the cell size of a one-cell axis; the mask needs at least 2 cells in "
                            "latitude and in longitude")
        return np.array([c[0] - fallback / 2, c[0] + fallback / 2])
    mid = (c[1:] + c[:-1]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def cell_size(centres, fallback=None) -> float:
    """Compute the median step of ascending cell centres.

    Parameters
    ----------
    centres : array_like of float, shape (n,)
        The ascending cell centres, in degrees.
    fallback : float, optional
        The size to return for a single centre (0 when None).

    Returns
    -------
    float
        The cell size in degrees.
    """
    c = np.asarray(centres, dtype=float)
    return float(np.median(np.diff(c))) if c.size > 1 else float(fallback or 0.0)


def _deg(value: float) -> str:
    """Format a cell size or coordinate in degrees, short.

    Parameters
    ----------
    value : float
        The value in degrees.

    Returns
    -------
    str
        The value with 6 significant digits.
    """
    return f"{value:.6g}"


def _grid_text(lat, lon) -> str:
    """Describe a grid as ``'nlon x nlat cells of dlon x dlat deg'``.

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The ascending latitude centres, in degrees.
    lon : array_like of float, shape (lon,)
        The ascending longitude centres, in degrees.

    Returns
    -------
    str
        The description.
    """
    return (f"{len(lon)} x {len(lat)} cells of {_deg(cell_size(lon, cell_size(lat)))} x "
            f"{_deg(cell_size(lat, cell_size(lon)))} deg")


def _make_valid(geom):
    """Repair a geometry if it is invalid.

    Parameters
    ----------
    geom : shapely.Geometry
        The geometry.

    Returns
    -------
    shapely.Geometry
        ``geom``, or its repaired version when it was invalid.
    """
    import shapely

    return geom if shapely.is_valid(geom) else shapely.make_valid(geom)


def cell_bounds(lat, lon):
    """Compute the cell edges of a grid of ascending centres.

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The ascending latitude centres, in degrees.
    lon : array_like of float, shape (lon,)
        The ascending longitude centres, in degrees.

    Returns
    -------
    lat_edges : numpy.ndarray of float64, shape (lat + 1,)
        The latitude edges.
    lon_edges : numpy.ndarray of float64, shape (lon + 1,)
        The longitude edges.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    fallback = cell_size(lon, cell_size(lat, 1.0)) or 1.0
    return cell_edges(lat, cell_size(lon, fallback)), cell_edges(lon, cell_size(lat, fallback))


def _window(geom, lat, lon):
    """Prepare a shape and a grid for testing: the repaired shape, wrapped longitudes, cell edges, candidate cells.

    Parameters
    ----------
    geom : shapely.Geometry
        The shape, in longitude -180..180.
    lat : array_like of float, shape (lat,)
        The ascending latitude centres, in degrees.
    lon : array_like of float, shape (lon,)
        The ascending longitude centres, in degrees.

    Returns
    -------
    geom : shapely.Geometry or None
        The repaired shape; None when the shape is empty.
    wrapped : numpy.ndarray of float64, shape (lon,)
        The cell longitudes wrapped to -180..180.
    edges : tuple of numpy.ndarray
        ``(lat_lo, lat_hi, lon_lo, lon_hi)``: the lower and upper edge of every row and column, the longitudes wrapped
        like the centres.
    rows : numpy.ndarray of int64, shape (n_rows,)
        The rows whose cells can touch the shape's bounding box.
    cols : numpy.ndarray of int64, shape (n_cols,)
        The columns whose cells can touch the shape's bounding box.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    shift = wrapped - lon
    lat_edges, lon_edges = cell_bounds(lat, lon)
    lon_lo, lon_hi = lon_edges[:-1] + shift, lon_edges[1:] + shift
    lat_lo, lat_hi = lat_edges[:-1], lat_edges[1:]
    empty = np.zeros(0, dtype=int)
    geom = _make_valid(geom)
    if geom.is_empty:
        return None, wrapped, (lat_lo, lat_hi, lon_lo, lon_hi), empty, empty
    minx, miny, maxx, maxy = geom.bounds
    rows = np.nonzero((lat_hi >= miny) & (lat_lo <= maxy))[0]
    cols = np.nonzero((lon_hi >= minx) & (lon_lo <= maxx))[0]
    return geom, wrapped, (lat_lo, lat_hi, lon_lo, lon_hi), rows, cols


def cell_shares(geom, lat, lon):
    """Compute the share of each cell's area (planar, degrees squared) inside a shape.

    The shape is in longitude -180..180; cell longitudes are wrapped into that range. Cells outside the shape's
    bounding box are left out (share 0).

    Parameters
    ----------
    geom : shapely.Geometry
        The shape.
    lat : array_like of float, shape (lat,)
        The ascending latitude centres, in degrees.
    lon : array_like of float, shape (lon,)
        The ascending longitude centres, in degrees.

    Returns
    -------
    rows : numpy.ndarray of int64, shape (n_rows,)
        The indices of the rows that can touch the shape's bounding box.
    cols : numpy.ndarray of int64, shape (n_cols,)
        The indices of the columns that can touch the shape's bounding box.
    share : numpy.ndarray of float64, shape (n_rows, n_cols)
        The share of each of those cells that is inside the shape, 0 to 1.
    """
    import shapely

    geom, _, (lat_lo, lat_hi, lon_lo, lon_hi), rows, cols = _window(geom, lat, lon)
    if geom is None or rows.size == 0 or cols.size == 0:
        return rows[:0], cols[:0], np.zeros((0, 0))
    shapely.prepare(geom)
    x0, y0 = np.meshgrid(lon_lo[cols], lat_lo[rows])
    x1, y1 = np.meshgrid(lon_hi[cols], lat_hi[rows])
    boxes = shapely.box(x0.ravel(), y0.ravel(), x1.ravel(), y1.ravel())
    share = shapely.contains(geom, boxes).astype(float)
    partly = shapely.intersects(geom, boxes) & (share == 0)
    if partly.any():
        share[partly] = shapely.area(shapely.intersection(geom, boxes[partly])) / shapely.area(boxes[partly])
    return rows, cols, share.reshape(rows.size, cols.size)


def rasterise(geom, lat, lon, rule="centre", fraction=0.5, strict=False) -> np.ndarray:
    """Rasterise a shape onto a grid: the cells it keeps under the cell rule.

    With the rule ``"centre"`` the cell centre is inside the shape (on its boundary counts as inside unless ``strict``).
    With ``"overlap_fraction"`` the part of the cell's area (planar, in degrees squared) inside the shape is at least
    ``fraction`` (see `cell_shares`). The shape is in longitude -180..180; cell longitudes are wrapped into that range.

    Parameters
    ----------
    geom : shapely.Geometry
        The shape.
    lat : array_like of float, shape (lat,)
        The ascending latitude centres, in degrees.
    lon : array_like of float, shape (lon,)
        The ascending longitude centres, in degrees.
    rule : {"centre", "overlap_fraction"}, default "centre"
        The cell rule; any value other than ``"centre"`` means ``"overlap_fraction"``.
    fraction : float, default 0.5
        The least share of the cell's area inside the shape, for ``"overlap_fraction"``.
    strict : bool, default False
        For ``"centre"``, leave out a centre on the shape's boundary.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True for the cells kept.
    """
    import shapely

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out = np.zeros((lat.size, lon.size), dtype=bool)
    if rule != "centre":
        rows, cols, share = cell_shares(geom, lat, lon)
        if rows.size and cols.size:
            out[np.ix_(rows, cols)] = share >= fraction - _EPS
        return out
    geom, wrapped, _, rows, cols = _window(geom, lat, lon)
    if geom is None or rows.size == 0 or cols.size == 0:
        return out
    shapely.prepare(geom)
    xx, yy = np.meshgrid(wrapped[cols], lat[rows])
    out[np.ix_(rows, cols)] = (shapely.contains_xy if strict else shapely.intersects_xy)(geom, xx, yy)
    return out


# ----------------------------------------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------------------------------------

def file_hash(path) -> str:
    """Compute the SHA-256 of a file's content; a shapefile also covers its ``.dbf``, ``.shx``, ``.prj`` and ``.cpg``.

    Parameters
    ----------
    path : str or os.PathLike
        The file.

    Returns
    -------
    str
        The hex digest (of whichever of the files exist).
    """
    path = Path(path)
    digest = hashlib.sha256()
    names = [path] + ([path.with_suffix(s) for s in _SIDECARS] if path.suffix.lower() == ".shp" else [])
    for item in names:
        if item.exists():
            with open(item, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


_NE_CACHE = memo("spatial.masks.natural_earth", maxsize=4, persistent=True)
"""The Natural Earth records read, by (``resolution``, ``file``): fixed by the installed data, so kept across runs."""


def _natural_earth(resolution, category, name):
    """Return the path of a Natural Earth shapefile, downloaded on first use; offline errors say how to fix them.

    Parameters
    ----------
    resolution : {"10m", "50m", "110m"}
        The resolution.
    category : str
        The Natural Earth category (``"cultural"`` or ``"physical"``).
    name : str
        The data set (``"admin_0_countries"`` or ``"land"``).

    Returns
    -------
    str
        The path of the ``.shp`` file.

    Raises
    ------
    MaskError
        The data could not be loaded (for instance offline, before it was downloaded).
    """
    from cartopy.io import shapereader

    try:
        return shapereader.natural_earth(resolution=resolution, category=category, name=name)
    except Exception as exc:
        raise MaskError(
            f"Natural Earth data ({resolution} {category}/{name}) could not be loaded: {type(exc).__name__}: {exc}. "
            "cartopy downloads it on first use, so it needs a network connection once; to prepare an offline "
            "machine run `python scripts/prefetch_cartopy_data.py` while online (the Docker image does this at "
            "build time) or point CARTOPY_DATA_DIR at a folder that holds the shapefiles.") from exc


def natural_earth_records(resolution, which):
    """Return the Natural Earth country or land records, cached.

    Parameters
    ----------
    resolution : {"10m", "50m", "110m"}
        The resolution.
    which : {"country", "land"}
        The data set.

    Returns
    -------
    path : str
        The path of the shapefile.
    records : list
        The records read from it (cartopy ``Record`` objects).
    """
    from cartopy.io import shapereader

    category, name = _NE_COUNTRIES if which == "country" else _NE_LAND
    key = (resolution, name)

    def read():
        """Read the shapefile and its records, for the cache."""
        path = _natural_earth(resolution, category, name)
        return path, list(shapereader.Reader(path).records())

    return _NE_CACHE.get_or_make(key, read)


def country_geometry(names, resolution):
    """Return the union of the named countries' shapes, the matched names and the shapefile path.

    A name matches ``NAME``, ``NAME_LONG`` or ``ADM0_A3`` of Natural Earth ``admin_0_countries``, any case. An
    unknown name is an error that lists the closest matches.

    Parameters
    ----------
    names : sequence of str
        The country names.
    resolution : {"10m", "50m", "110m"}
        The Natural Earth resolution.

    Returns
    -------
    geometry : shapely.Geometry
        The union of the countries, in longitude/latitude degrees.
    matched : list of str
        The Natural Earth ``NAME`` of each country, in the order of ``names``.
    path : str
        The path of the shapefile.

    Raises
    ------
    MaskError
        A name matches no country.
    """
    import shapely

    path, records = natural_earth_records(resolution, "country")
    index: dict = {}
    for record in records:
        for key in _COUNTRY_FIELDS:
            value = record.attributes.get(key)
            if value:
                index.setdefault(str(value).lower(), record)
    chosen, matched = [], []
    for name in names:
        record = index.get(str(name).lower())
        if record is None:
            close = difflib.get_close_matches(str(name).lower(), list(index), n=5, cutoff=0.5)
            hint = f" Closest names: {', '.join(repr(c.title() if len(c) > 3 else c.upper()) for c in close)}." \
                if close else ""
            raise MaskError(f"mask.country: unknown country {name!r} (Natural Earth admin_0_countries, matched on "
                            f"NAME, NAME_LONG and ADM0_A3, any case).{hint}")
        chosen.append(_make_valid(record.geometry))
        matched.append(str(record.attributes.get("NAME")))
    return shapely.union_all(chosen), matched, path


_LAND_GEOMETRY = memo("spatial.masks.land_geometry", maxsize=4, persistent=True)
"""The union of the Natural Earth land shapes, by resolution: fixed by the installed data, so kept across runs."""


def land_geometry(resolution):
    """Return the union of the Natural Earth land polygons and the shapefile path, cached.

    Parameters
    ----------
    resolution : {"10m", "50m", "110m"}
        The Natural Earth resolution.

    Returns
    -------
    geometry : shapely.Geometry
        The union of the land shapes, in longitude/latitude degrees.
    path : str
        The path of the shapefile.
    """
    import shapely

    def build():
        """Union the land records, for the cache."""
        path, records = natural_earth_records(resolution, "land")
        return shapely.union_all([_make_valid(r.geometry) for r in records]), path

    return _LAND_GEOMETRY.get_or_make(resolution, build)


def shape_file_geometry(path):
    """Return the union of every geometry of a shapefile or GeoJSON, in longitude/latitude degrees (EPSG:4326).

    Parameters
    ----------
    path : str or os.PathLike
        The shapefile or GeoJSON.

    Returns
    -------
    shapely.Geometry
        The repaired union of the features, reprojected to EPSG:4326 when the file has another coordinate system.

    Raises
    ------
    MaskError
        The file cannot be read.
    """
    import geopandas as gpd

    try:
        frame = gpd.read_file(path)
    except Exception as exc:
        raise MaskError(f"mask.file: cannot read the shape file {path}: {type(exc).__name__}: {exc}") from exc
    if frame.crs is not None and not frame.crs.equals("EPSG:4326"):
        frame = frame.to_crs("EPSG:4326")
    return _make_valid(frame.geometry.union_all())


def _find_axis(ds, role, units, what, path):
    """Find the latitude or longitude coordinate of a dataset, by name, units and ``standard_name``.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset.
    role : {"lat", "lon"}
        The role whose names (`hcwf.io.dims.ROLE_ALIASES`) are accepted.
    units : tuple of str
        The accepted ``units`` attributes, lower case.
    what : {"latitude", "longitude"}
        The axis wanted (also the ``standard_name`` that is accepted).
    path : str or os.PathLike
        The file, for the message.

    Returns
    -------
    str
        The name of the coordinate.

    Raises
    ------
    MaskError
        No coordinate matches.
    """
    found = find_dim(list(ds.coords) + list(ds.dims), role)
    if found is not None:
        return found
    for name in ds.coords:
        attrs = ds[name].attrs
        if str(attrs.get("units", "")).lower() in units or str(attrs.get("standard_name", "")).lower() == what:
            return name
    raise MaskError(f"{path}: no {what} coordinate found (looked for {', '.join(ROLE_ALIASES[role])}); coordinates are "
                    f"{', '.join(map(str, ds.coords))}")


def read_raster_mask(path, variable=None, keep_value=1, condition=None):
    """Read a 0/1 mask (or a field with ``above`` / ``below``) from NetCDF.

    The result has dims (lat, lon) in the file's own order and longitude convention. A field is thresholded on its own
    grid first (``above`` is strict, ``below`` is strict, both give a band); cells without a value are not kept.

    Parameters
    ----------
    path : str or os.PathLike
        The NetCDF file.
    variable : str, optional
        The variable; None picks the only one on the (lat, lon) axes.
    keep_value : float, default 1
        The value that marks a kept cell, when there is no ``condition``.
    condition : dict, optional
        A threshold with ``above`` and / or ``below``, applied to the variable.

    Returns
    -------
    kept : numpy.ndarray of bool, shape (lat, lon)
        The cells kept.
    lat : numpy.ndarray of float64, shape (lat,)
        The latitudes of the file, in degrees.
    lon : numpy.ndarray of float64, shape (lon,)
        The longitudes of the file, in degrees.
    notes : str
        A text that names the axes and the variable found.

    Raises
    ------
    MaskError
        The file cannot be opened, has no (or no single) variable on the axes, has a variable that is not 2-D, or has
        axes that are not 1-D.
    """
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as exc:
        raise MaskError(f"cannot open the mask file {path}: {type(exc).__name__}: {exc}") from exc
    with ds:
        lat_name = _find_axis(ds, "lat", ("degrees_north", "degree_north", "degrees_n"), "latitude", path)
        lon_name = _find_axis(ds, "lon", ("degrees_east", "degree_east", "degrees_e"), "longitude", path)
        if variable is None:
            candidates = [v for v in ds.data_vars if lat_name in ds[v].dims and lon_name in ds[v].dims]
            if len(candidates) != 1:
                raise MaskError(f"{path}: {'no variable' if not candidates else 'several variables'} on "
                                f"({lat_name}, {lon_name}) to choose from ({', '.join(candidates) or 'none'}); set "
                                "the mask variable")
            variable = candidates[0]
        elif variable not in ds.data_vars:
            raise MaskError(f"{path}: no variable {variable!r}; the file has {', '.join(map(str, ds.data_vars))}"
                            f"{suggest(variable, ds.data_vars)}")
        da = ds[variable]
        extra = [d for d in da.dims if d not in (lat_name, lon_name)]
        for dim in extra:
            if da.sizes[dim] != 1:
                raise MaskError(f"{path}: variable {variable!r} has dimension {dim!r} of size {da.sizes[dim]}; a "
                                "mask must be 2-D (lat, lon)")
        da = da.squeeze(extra, drop=True).transpose(lat_name, lon_name)
        values = np.asarray(da.values)
        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise MaskError(f"{path}: latitude and longitude must be 1-D axes (curvilinear grids are not supported)")
    if condition:
        with np.errstate(invalid="ignore"):
            kept = np.ones(values.shape, dtype=bool)
            if condition.get("above") is not None:
                kept &= values > condition["above"]
            if condition.get("below") is not None:
                kept &= values < condition["below"]
    else:
        with np.errstate(invalid="ignore"):
            kept = values == keep_value
    notes = f"variable {variable!r}, axes lat={lat_name!r} lon={lon_name!r}"
    return np.asarray(kept, dtype=bool), lat, lon, notes


# ----------------------------------------------------------------------------------------------------------
# aligning a raster mask with a data grid
# ----------------------------------------------------------------------------------------------------------

def _overlap(lo_a, hi_a, lo_b, hi_b) -> np.ndarray:
    """Compute the length shared by each interval of one set and each interval of another.

    Parameters
    ----------
    lo_a : numpy.ndarray of float64, shape (a,)
        The lower ends of the first set.
    hi_a : numpy.ndarray of float64, shape (a,)
        The upper ends of the first set.
    lo_b : numpy.ndarray of float64, shape (b,)
        The lower ends of the second set.
    hi_b : numpy.ndarray of float64, shape (b,)
        The upper ends of the second set.

    Returns
    -------
    numpy.ndarray of float64, shape (a, b)
        The shared length of interval a and interval b, 0 when they do not meet.
    """
    return np.clip(np.minimum(hi_a[:, None], hi_b[None, :]) - np.maximum(lo_a[:, None], lo_b[None, :]), 0.0, None)


def _nearest(values, targets) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest of the ascending ``values`` to each target.

    Parameters
    ----------
    values : numpy.ndarray of float64, shape (n,)
        The ascending values.
    targets : numpy.ndarray of float64, shape (m,)
        The targets.

    Returns
    -------
    pick : numpy.ndarray of int64, shape (m,)
        The index in ``values`` of the nearest value to each target.
    distance : numpy.ndarray of float64, shape (m,)
        The distance of each target to that value.
    """
    right = np.clip(np.searchsorted(values, targets), 1, len(values) - 1) if len(values) > 1 else np.zeros(
        len(targets), dtype=int)
    left = np.clip(right - 1, 0, None)
    pick = np.where(np.abs(values[right] - targets) < np.abs(targets - values[left]), right, left)
    return pick, np.abs(values[pick] - targets)


def align_raster_mask(kept, mlat, mlon, lat, lon, fraction=0.5, label="mask file", regrid="area"):
    """Bring a boolean mask on its own grid onto the data grid (ascending centres lat, lon).

    The actions returned are the printable steps taken, in order: latitude flip, longitude wrap, then either direct use
    (coordinates equal within a hundredth of a cell) or the regridding of the mask (area-weighted fraction >=
    ``fraction`` for a finer mask, nearest cell for a coarser or equally fine one; with ``regrid="nearest"`` also for a
    finer one).

    Parameters
    ----------
    kept : array_like of bool, shape (mlat, mlon)
        The mask on its own grid.
    mlat : array_like of float, shape (mlat,)
        The latitudes of the mask, in degrees (any order).
    mlon : array_like of float, shape (mlon,)
        The longitudes of the mask, in degrees (any convention).
    lat : array_like of float, shape (lat,)
        The ascending latitudes of the data grid, in degrees.
    lon : array_like of float, shape (lon,)
        The ascending longitudes of the data grid, in degrees.
    fraction : float, default 0.5
        The least share of a data cell that kept mask cells must cover, for a finer mask.
    label : str, default "mask file"
        What the mask is, for the messages.
    regrid : {"area", "nearest"}, default "area"
        How a finer mask is brought to the data grid.

    Returns
    -------
    kept : numpy.ndarray of bool, shape (lat, lon)
        The mask on the data grid.
    actions : list of str
        The steps taken, in order.

    Raises
    ------
    MaskError
        The mask does not cover the data extent (the message names both extents).
    """
    actions = []
    kept = np.asarray(kept, dtype=bool)
    mlat = np.asarray(mlat, dtype=float)
    mlon = np.asarray(mlon, dtype=float)
    if mlat.size > 1 and mlat[0] > mlat[-1]:
        order = np.argsort(mlat)
        mlat, kept = mlat[order], kept[order]
        actions.append("flipped the mask's latitude order (north to south in the file) to south to north")
    elif mlat.size > 1 and np.any(np.diff(mlat) < 0):
        order = np.argsort(mlat)
        mlat, kept = mlat[order], kept[order]
        actions.append("sorted the mask's latitudes")
    centre = (lon.min() + lon.max()) / 2
    start = centre - 180.0
    wrapped = ((mlon - start) % 360.0) + start
    if np.any(np.abs(wrapped - mlon) > _EPS):
        actions.append(f"shifted the mask's longitudes from {_deg(mlon.min())}..{_deg(mlon.max())} to the data's "
                       f"convention ({_deg(wrapped.min())}..{_deg(wrapped.max())})")
    order = np.argsort(wrapped, kind="stable")
    mlon, kept = wrapped[order], kept[:, order]
    if mlon.size > 1:
        keep_cols = np.concatenate([[True], np.diff(mlon) > _EPS])
        mlon, kept = mlon[keep_cols], kept[:, keep_cols]
    dres = (cell_size(lat, cell_size(lon)), cell_size(lon, cell_size(lat)))
    mres = (cell_size(mlat, dres[0]), cell_size(mlon, dres[1]))
    tol = 0.01 * min(dres)
    actions.append(f"{label} grid: {_grid_text(mlat, mlon)}; data grid: {_grid_text(lat, lon)}")
    same_res = abs(mres[0] - dres[0]) <= tol and abs(mres[1] - dres[1]) <= tol
    ilat, dlat = _nearest(mlat, lat)
    ilon, dlon = _nearest(mlon, lon)
    if same_res and dlat.max() <= tol and dlon.max() <= tol:
        actions.append(f"same grid (coordinates equal within {_deg(tol)} deg): used directly")
        return kept[np.ix_(ilat, ilon)], actions
    elat, elon = cell_edges(lat, dres[0]), cell_edges(lon, dres[1])
    mlat_e, mlon_e = cell_edges(mlat, mres[0]), cell_edges(mlon, mres[1])
    ctol = 0.01 * min(dres + mres)
    if elat[0] < mlat_e[0] - ctol or elat[-1] > mlat_e[-1] + ctol or elon[0] < mlon_e[0] - ctol \
            or elon[-1] > mlon_e[-1] + ctol:
        raise MaskError(
            f"the {label} does not cover the data: it spans latitude {_deg(mlat_e[0])}..{_deg(mlat_e[-1])}, "
            f"longitude {_deg(mlon_e[0])}..{_deg(mlon_e[-1])} (cell edges, longitude wrapped to the data's "
            f"convention) but the data spans latitude {_deg(elat[0])}..{_deg(elat[-1])}, longitude "
            f"{_deg(elon[0])}..{_deg(elon[-1])}")
    offsets = f"largest distance from a data cell centre to the nearest mask centre: {_deg(dlat.max())} deg " \
              f"latitude, {_deg(dlon.max())} deg longitude"
    if regrid == "nearest" or (mres[0] >= dres[0] - tol and mres[1] >= dres[1] - tol):
        actions.append(f"regridded the mask onto the data grid by the nearest mask cell (mask cells "
                       f"{_deg(mres[1])} x {_deg(mres[0])} deg, data cells "
                       f"{_deg(dres[1])} x {_deg(dres[0])} deg); {offsets}")
        return kept[np.ix_(ilat, ilon)], actions
    rows = np.nonzero((mlat_e[1:] > elat[0] - ctol) & (mlat_e[:-1] < elat[-1] + ctol))[0]
    cols = np.nonzero((mlon_e[1:] > elon[0] - ctol) & (mlon_e[:-1] < elon[-1] + ctol))[0]
    w_lat = _overlap(elat[:-1], elat[1:], mlat_e[rows], mlat_e[rows + 1]) / np.diff(elat)[:, None]
    w_lon = _overlap(elon[:-1], elon[1:], mlon_e[cols], mlon_e[cols + 1]) / np.diff(elon)[:, None]
    share = w_lat @ kept[np.ix_(rows, cols)].astype(float) @ w_lon.T
    actions.append(f"regridded the mask onto the data grid: a data cell is kept when kept mask cells cover at "
                   f"least {fraction:g} of its area (area-weighted; mask cells {_deg(mres[1])} x {_deg(mres[0])} "
                   f"deg are finer than the data's {_deg(dres[1])} x {_deg(dres[0])} deg); {offsets}")
    return share >= fraction - _EPS, actions


# ----------------------------------------------------------------------------------------------------------
# building the mask of one dataset
# ----------------------------------------------------------------------------------------------------------

def _source(kind, name, path=None, extra=None) -> dict:
    """Make the record of what a mask part was made from.

    Parameters
    ----------
    kind : str
        The part: ``"land"``, ``"country"``, ``"file"`` or ``"condition"``.
    name : str
        What the part was made from (a name or a file name).
    path : str or os.PathLike, optional
        The file; its path and content hash are added when given.
    extra : dict, optional
        More entries for the record.

    Returns
    -------
    dict
        The record, with ``part``, ``source`` and, for a file, ``path`` and ``sha256``.
    """
    out = {"part": kind, "source": name}
    if path:
        out["path"] = str(path)
        out["sha256"] = file_hash(path)
    if extra:
        out.update(extra)
    return out


def _rule_text(spec: MaskSpec) -> str:
    """Describe the cell rule in words.

    Parameters
    ----------
    spec : MaskSpec
        The mask spec.

    Returns
    -------
    str
        The description.
    """
    if spec.cell_rule == "centre":
        return "centre (a cell is kept when its centre is inside the shape)"
    return f"overlap_fraction >= {spec.overlap_fraction:g} (a cell is kept when that share of its area is inside)"


def _rule_short(spec: MaskSpec) -> str:
    """Describe the cell rule in a few words.

    Parameters
    ----------
    spec : MaskSpec
        The mask spec.

    Returns
    -------
    str
        The description.
    """
    return "centre rule" if spec.cell_rule == "centre" else f"overlap_fraction >= {spec.overlap_fraction:g} rule"


def build_mask(spec: MaskSpec, lat, lon, label="data") -> MaskResult:
    """Build the mask of one dataset on its own grid, in the order of ``lat`` and ``lon`` given.

    Every part of the spec is applied and the parts intersect. The result carries the printable lines and the
    record (sources with content hashes, rule, kept counts).

    Parameters
    ----------
    spec : MaskSpec
        The mask to build.
    lat : array_like of float, shape (lat,)
        The latitudes of the dataset, in degrees (any order).
    lon : array_like of float, shape (lon,)
        The longitudes of the dataset, in degrees (any order).
    label : str, default "data"
        What the grid is, for the messages and the record.

    Returns
    -------
    MaskResult
        The mask (``bool`` of shape ``(lat, lon)``), its lines and its record.

    Raises
    ------
    MaskError
        A source is unusable, the grid is not covered, or no cell is kept (unless ``spec.allow_empty``).
    """
    import shapely

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ilat, ilon = np.argsort(lat, kind="stable"), np.argsort(lon, kind="stable")
    slat, slon = lat[ilat], lon[ilon]
    total = slat.size * slon.size
    lines, sources, shapes, rasters = [], [], [], []
    shape_names = []
    if spec.land:
        resolution = spec.land_resolution or spec.resolution
        geom, path = land_geometry(resolution)
        shapes.append(geom)
        shape_names.append("land")
        sources.append(_source("land", f"natural_earth {resolution} physical/land", path))
    if spec.country:
        geom, matched, path = country_geometry(spec.country, spec.resolution)
        shapes.append(geom)
        shape_names.append("country " + ", ".join(matched))
        sources.append(_source("country", f"natural_earth {spec.resolution} cultural/admin_0_countries: "
                                          + ", ".join(matched), path))
    if spec.file:
        if Path(spec.file).suffix.lower() in SHAPE_SUFFIXES:
            shapes.append(shape_file_geometry(spec.file))
            shape_names.append(f"shape file {spec.file}")
            sources.append(_source("file", "shape file", spec.file))
        else:
            kept, mlat, mlon, notes = read_raster_mask(spec.file, spec.variable, spec.keep_value)
            rasters.append((f"mask file {spec.file}", kept, mlat, mlon, notes))
            sources.append(_source("file", "mask file", spec.file, {"variable": spec.variable,
                                                                    "keep_value": spec.keep_value}))
    if spec.condition:
        cond = spec.condition
        kept, mlat, mlon, notes = read_raster_mask(cond["file"], cond.get("variable"), condition=cond)
        text = " and ".join(f"{cond.get('variable') or 'field'} {sign} {cond[key]:g}"
                            for key, sign in (("above", ">"), ("below", "<")) if cond.get(key) is not None)
        rasters.append((f"condition {text} from {cond['file']}", kept, mlat, mlon, notes))
        sources.append(_source("condition", text, cond["file"], {k: cond[k] for k in ("variable", "above", "below")
                                                                 if cond.get(k) is not None}))
    keep = np.ones((slat.size, slon.size), dtype=bool)
    outline = None
    parts = []
    if shapes:
        shape = shapes[0]
        for other in shapes[1:]:
            shape = _make_valid(shapely.intersection(shape, other))
        outline = shape
        raster = rasterise(shape, slat, slon, spec.cell_rule, spec.overlap_fraction, spec.boundary == "strict")
        keep &= raster
        parts.append((" + ".join(shape_names) + f" rasterised on the data grid by the {_rule_short(spec)}", int(raster.sum())))
    for text, kept, mlat, mlon, notes in rasters:
        lines.append(f"  {text}: {notes}")
        aligned, actions = align_raster_mask(kept, mlat, mlon, slat, slon, spec.overlap_fraction, label="mask",
                                          regrid=spec.regrid)
        lines.extend(f"    - {a}" for a in actions)
        keep &= aligned
        parts.append((text, int(aligned.sum())))
    count = int(keep.sum())
    out = np.zeros_like(keep)
    out[np.ix_(ilat, ilon)] = keep
    header = f"Mask [{label}]: " + " + ".join(s["source"] for s in sources)
    summary = [header, f"  data grid: {_grid_text(slat, slon)}; cell rule: {_rule_text(spec)}"]
    summary += lines
    for text, n in parts:
        summary.append(f"  {text}: {n} cells")
    summary.append(f"  kept cells: {count} of {total}")
    record = {"dataset": label, "rule": spec.cell_rule, "overlap_fraction": spec.overlap_fraction,
              "resolution": spec.resolution, "sources": sources, "kept": count, "total": total,
              "actions": [line.strip("- ").strip() for line in lines]}
    if count == 0 and not spec.allow_empty:
        raise MaskError("\n".join(summary) + f"\nthe mask keeps no cell of the {label} grid")
    return MaskResult(keep=out, lines=summary, record=record, outline=outline, lat=lat, lon=lon)


# ----------------------------------------------------------------------------------------------------------
# picture
# ----------------------------------------------------------------------------------------------------------

def save_mask_png(result: MaskResult, path, title="") -> Path:
    """Save a small PNG of the kept cells (and the shape outline if there is one), in the onset figure colours.

    Parameters
    ----------
    result : MaskResult
        The mask to draw.
    path : str or os.PathLike
        The PNG file; its folder is created.
    title : str, default ""
        The title of the figure, followed by the number of cells kept.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    from hcwf.onset.viz.style import palette

    p = palette("light")
    lat, lon = result.lat, result.lon
    ilat, ilon = np.argsort(lat), np.argsort(lon)
    slat, slon = lat[ilat], lon[ilon]
    keep = result.keep[np.ix_(ilat, ilon)]
    fig = Figure(figsize=(4.2, 4.0), dpi=110, facecolor=p.surface)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor=p.surface)
    fallback = (cell_size(slon, cell_size(slat, 1.0)), cell_size(slat, cell_size(slon, 1.0)))
    ax.pcolormesh(cell_edges(slon, fallback[1]), cell_edges(slat, fallback[0]), keep.astype(float),
                  cmap=ListedColormap([p.wash, p.observed]), vmin=0, vmax=1, shading="flat")
    if result.outline is not None:
        for geom in getattr(result.outline, "geoms", [result.outline]):
            rings = [geom.exterior] if hasattr(geom, "exterior") else []
            for ring in rings:
                x, y = np.asarray(ring.xy[0]), np.asarray(ring.xy[1])
                if slon.max() > 180:
                    x = x % 360
                ax.plot(x, y, color=p.ink_secondary, lw=0.6)
    ax.set_xlim(cell_edges(slon, fallback[1])[[0, -1]])
    ax.set_ylim(cell_edges(slat, fallback[0])[[0, -1]])
    ax.set_aspect("equal")
    ax.tick_params(colors=p.ink_secondary, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(p.axis)
    ax.set_xlabel("longitude", color=p.ink_secondary, fontsize=8)
    ax.set_ylabel("latitude", color=p.ink_secondary, fontsize=8)
    ax.set_title(f"{title}: {int(keep.sum())} of {keep.size} cells kept", color=p.ink, fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=p.surface)
    return path


# ----------------------------------------------------------------------------------------------------------
# use in the pipeline
# ----------------------------------------------------------------------------------------------------------

_BUILT = memo("spatial.masks.built", maxsize=64, share=True)
"""The mask of each dataset grid built for a ``mask:`` section, by (mask key, dataset, grid hash); the workers' copy."""
_RECORDS = memo("spatial.masks.records", share=True)
"""The record of each dataset's mask (what was kept, from which files), by dataset name."""


def records() -> dict:
    """Return the records of the masks built so far.

    Returns
    -------
    dict of str to dict
        The record of each dataset's mask (see `MaskResult`), by dataset name.
    """
    return dict(_RECORDS.items())


def absorb_records(found: dict) -> None:
    """Take the records a worker built into this process (a dataset already recorded here keeps its record).

    Parameters
    ----------
    found : dict of str to dict
        The records of the worker, by dataset name.
    """
    for name, record in found.items():
        _RECORDS.setdefault(name, record)


def reset() -> None:
    """Forget the built masks and records: ``--check`` reads the mask of one sample file on its own, between others."""
    _BUILT.clear()
    _RECORDS.clear()


def region_mask(lat, lon, mask, dataset="data", dir_fig=None) -> np.ndarray:
    """Return the mask of one dataset's grid for a config's ``mask`` mapping, built once per grid.

    The first time a dataset's grid is seen, the mask is built, its description is logged, the PNG is saved
    to ``dir_fig/mask_<dataset>.png`` (when ``dir_fig`` is given) and the record is stored for `records`. A mask that
    cannot be built raises ``MaskError`` (see `build_mask`).

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes of the dataset, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes of the dataset, in degrees.
    mask : MaskSpec or collections.abc.Mapping
        The ``mask:`` section.
    dataset : str, default "data"
        The name of the dataset, for the messages, the record and the PNG.
    dir_fig : str or os.PathLike, optional
        The folder of the PNG; None saves no figure.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True for the cells kept, in the order of ``lat`` and ``lon`` given.
    """
    spec = MaskSpec.from_mapping(mask)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    key = (spec.key(), str(dataset), hashlib.sha256(lat.tobytes() + lon.tobytes()).hexdigest())
    result = _BUILT.get(key)
    if result is None:
        result = build_mask(spec, lat, lon, label=str(dataset))
        log.info("\n".join(result.lines))
        if dir_fig:
            png = save_mask_png(result, Path(dir_fig) / f"mask_{dataset}.png", title=str(dataset))
            result.record["figure"] = str(png)
            log.info(f"  mask figure: {png}")
        _BUILT[key] = result
        _RECORDS[str(dataset)] = result.record
    return result.keep


def apply_region_mask(ds, mask, dataset="data", dir_fig=None):
    """Set the cells of ``ds`` outside the mask to NaN.

    The mask is built on this dataset's own grid, so the data are never regridded. Variables without both
    ``lat`` and ``lon`` dimensions are left as they are.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates.
    mask : MaskSpec or collections.abc.Mapping
        The ``mask:`` section.
    dataset : str, default "data"
        The name of the dataset, for the messages, the record and the PNG.
    dir_fig : str or os.PathLike, optional
        The folder of the PNG; None saves no figure.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The masked data (a copy).
    """
    return _where_keep(ds, region_mask(ds["lat"].values, ds["lon"].values, mask, dataset, dir_fig))


def _where_keep(ds, keep):
    """Set the cells of ``ds`` outside ``keep`` to NaN.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates; the variables of a Dataset without both dims are left as they
        are.
    keep : numpy.ndarray of bool, shape (lat, lon)
        The cells to keep.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The masked data (a copy).
    """
    cond = xr.DataArray(keep, dims=("lat", "lon"), coords={"lat": ds["lat"], "lon": ds["lon"]})
    if isinstance(ds, xr.DataArray):
        return ds.where(cond)
    out = ds.copy()
    for name, var in ds.data_vars.items():
        if "lat" in var.dims and "lon" in var.dims:
            out[name] = var.where(cond)
    return out


def record_lines(rec: dict) -> list[str]:
    """Describe one mask record as printable comment lines (sources with hashes, rule, kept cells).

    Parameters
    ----------
    rec : dict
        The record of a mask (see `MaskResult`).

    Returns
    -------
    list of str
        The lines.
    """
    lines = [f"mask [{rec['dataset']}]: cell rule {rec['rule']}"
             + (f" (overlap_fraction {rec['overlap_fraction']:g})" if rec["rule"] == "overlap_fraction" else "")
             + f", kept {rec['kept']} of {rec['total']} cells"]
    for src in rec["sources"]:
        text = f"  {src['part']}: {src['source']}"
        if src.get("path"):
            text += f" [{src['path']}, sha256 {src['sha256']}]"
        lines.append(text)
    for action in rec.get("actions", ()):
        lines.append(f"  {action}")
    return lines


def mask_attrs() -> dict:
    """Return the NetCDF global attributes that describe the masks built so far.

    The cell rule and the sources are those of the first record.

    Returns
    -------
    dict
        ``mask_cell_rule``, ``mask_sources``, ``mask_overlap_fraction`` (for that rule) and ``mask_kept_cells_<dataset>``
        for each dataset; empty when no mask is in use.
    """
    if not len(_RECORDS):
        return {}
    first = _RECORDS.values()[0]
    out = {"mask_cell_rule": first["rule"],
           "mask_sources": "; ".join(
               f"{s['part']}={s.get('path') or s['source']}" + (f" sha256:{s['sha256']}" if s.get("sha256") else "")
               for s in first["sources"])}
    if first["rule"] == "overlap_fraction":
        out["mask_overlap_fraction"] = first["overlap_fraction"]
    for name, rec in records().items():
        out[f"mask_kept_cells_{name}"] = f"{rec['kept']} of {rec['total']}"
    return out


# ----------------------------------------------------------------------------------------------------------
# the implicit masks (land_only, shp_only, nc_mask): a preset of the same algorithm
# ----------------------------------------------------------------------------------------------------------
# Used when a run sets no `mask:` section. `default_mask` states them as a MaskSpec and `build_mask` evaluates
# it: the cells whose centre is strictly inside the Natural Earth 110m land (land_only) and the 10m shape of the
# run's country (shp_only), and the nearest cell of the nc_mask file. The results are cached per grid.

_MAX_DEFAULT = 64
"""The most implicit masks kept in memory."""
_RAW_MASKS = memo("spatial.masks.implicit_results", maxsize=_MAX_DEFAULT)
"""The MaskResult of each implicit mask built, by cache key."""
_LAND_BOOL = memo("spatial.masks.implicit_keep", maxsize=_MAX_DEFAULT, share=True)
"""The read-only boolean cells kept by each implicit mask, by cache key; the workers' copy."""


def _grid_key(lat, lon) -> str:
    """Hash the lat and lon values (as float64), the cache key of a grid.

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes, in degrees.

    Returns
    -------
    str
        The hex digest, which also covers the shapes.
    """
    lat = np.ascontiguousarray(np.asarray(lat, dtype="float64"))
    lon = np.ascontiguousarray(np.asarray(lon, dtype="float64"))
    return hashlib.sha256(str(lat.shape).encode() + lat.tobytes() + str(lon.shape).encode() + lon.tobytes()).hexdigest()


def _registry_title(region):
    """Return the registry title of a region name in any case (``"ethiopia"`` -> ``"Ethiopia"``), else the name itself.

    Parameters
    ----------
    region : str
        The region name.

    Returns
    -------
    str
        The title in the region registry, or ``region`` when it is not a registry region.
    """
    try:
        from hcwf.onset import regions

        return regions.load_region(str(region)).title
    except Exception:  # noqa: BLE001 - not a registry region: the name is matched as written
        return region


def _country_record(region, resolution="10m"):
    """Return the Natural Earth country record named ``region`` (``NAME``, or the registry title of the name).

    Parameters
    ----------
    region : str
        The region or country name.
    resolution : {"10m", "50m", "110m"}, default "10m"
        The Natural Earth resolution.

    Returns
    -------
    object or None
        The cartopy ``Record`` of the country, None if there is none.
    """
    names = {region, _registry_title(region)}
    for country in natural_earth_records(resolution, "country")[1]:
        if country.attributes['NAME'] in names:
            return country


def get_shp(region, resolution='10m'):
    """Return the Natural Earth country shape named ``region`` (``NAME``, or the registry title of the name).

    Parameters
    ----------
    region : str
        The region or country name.
    resolution : {"10m", "50m", "110m"}, default "10m"
        The Natural Earth resolution.

    Returns
    -------
    shapely.Geometry or None
        The country's shape, None if there is none.
    """
    record = _country_record(region, resolution)
    return None if record is None else record.geometry


def default_mask(region=None, land_only=True, shp_only=True, nc_mask=None, resolution="10m") -> MaskSpec:
    """Return the `MaskSpec` of the implicit masks of a run with no ``mask:`` section.

    A cell is kept when its centre is strictly inside the shape. The resulting mask may be empty.

    Parameters
    ----------
    region : str, optional
        The country for ``shp_only``; a region that is not a country adds no part and a warning.
    land_only : bool, default True
        Keep land only (Natural Earth 110m).
    shp_only : bool, default True
        Keep the country ``region`` only (Natural Earth ``resolution``).
    nc_mask : str or os.PathLike, optional
        A 0/1 NetCDF file; each data cell takes the nearest mask cell.
    resolution : {"10m", "50m", "110m"}, default "10m"
        The Natural Earth resolution of the country shape.

    Returns
    -------
    MaskSpec
        The spec, with ``boundary="strict"``, ``regrid="nearest"`` and ``allow_empty=True``.
    """
    country = ()
    if shp_only:
        record = _country_record(region, resolution)
        if record is None:
            log.warning("specified region is not in cartopy.io.shapereader")
        else:
            country = (record.attributes["NAME"],)
    return MaskSpec(land=bool(land_only), country=country, file=str(nc_mask) if nc_mask else None,
                    resolution=resolution, land_resolution="110m", boundary="strict", regrid="nearest",
                    allow_empty=True)


def default_keep(lat, lon, region=None, land_only=True, shp_only=True, nc_mask=None) -> np.ndarray:
    """Return the cells kept by the implicit masks (`default_mask`) on a grid.

    Built by `build_mask` once per grid, region and options in this process (cached by a hash of the coordinates) and
    returned read-only, so callers can apply it in place on numpy arrays, e.g. ``arr[..., ~keep] = np.nan``.

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes, in degrees.
    region : str, optional
        The country for ``shp_only``.
    land_only : bool, default True
        Keep land only.
    shp_only : bool, default True
        Keep the country ``region`` only.
    nc_mask : str or os.PathLike, optional
        A 0/1 NetCDF file.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True where kept; read-only. All True when no part applies.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    key = (_grid_key(lat, lon), str(region) if shp_only else None, bool(land_only), bool(shp_only),
           str(nc_mask) if nc_mask else None)
    keep = _LAND_BOOL.get(key)
    if keep is None:
        spec = default_mask(region, land_only, shp_only, nc_mask)
        if spec.land or spec.country or spec.file:
            result = build_mask(spec, lat, lon, label="default")
            _RAW_MASKS[key] = result
            keep = np.array(result.keep)
        else:
            keep = np.ones((lat.size, lon.size), dtype=bool)
        keep.setflags(write=False)
        _LAND_BOOL[key] = keep
    return keep


def land_mask_for(lat, lon) -> np.ndarray:
    """Return the land mask for 1-D lat and lon values (see `default_keep`).

    Parameters
    ----------
    lat : array_like of float, shape (lat,)
        The latitudes, in degrees.
    lon : array_like of float, shape (lon,)
        The longitudes, in degrees.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True on land; read-only.
    """
    return default_keep(lat, lon, None, True, False)


def apply_default_mask(obj, region=None, land_only=True, shp_only=True, nc_mask=None):
    """Set the cells of ``obj`` outside the implicit masks to NaN.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates.
    region : str, optional
        The country for ``shp_only``.
    land_only : bool, default True
        Keep land only.
    shp_only : bool, default True
        Keep the country ``region`` only.
    nc_mask : str or os.PathLike, optional
        A 0/1 NetCDF file.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        ``obj`` itself when every cell is kept, else a masked copy.
    """
    keep = default_keep(obj["lat"].values, obj["lon"].values, region, land_only, shp_only, nc_mask)
    return obj if keep.all() else _where_keep(obj, keep)


def create_land_sea_mask(
    obj: Union[xr.Dataset, xr.DataArray],
    as_boolean: bool = False,
) -> xr.DataArray:
    """Make a land-sea mask on the grid of ``obj``.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates.
    as_boolean : bool, default False
        Return True / False instead of 1 / 0.

    Returns
    -------
    xarray.DataArray
        The mask over ``(lat, lon)``: 1 for land and 0 for sea (integers), or booleans.
    """
    keep = land_mask_for(obj["lat"].values, obj["lon"].values)
    out = xr.DataArray(np.array(keep), dims=("lat", "lon"), coords={"lat": obj["lat"], "lon": obj["lon"]})
    return out if as_boolean else out.astype(int)


def mask_land(da, land=True):
    """Set the sea cells (or the land cells) of ``da`` to NaN.

    Parameters
    ----------
    da : xarray.Dataset or xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates.
    land : bool, default True
        Keep the land (set the sea to NaN); False keeps the sea.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The masked data (a copy).
    """
    keep = land_mask_for(da["lat"].values, da["lon"].values)
    return _where_keep(da, keep if land else ~keep)


def shp_mask(da, region):
    """Set the cells of ``da`` whose centre is not strictly inside the country ``region`` to NaN.

    This is the implicit ``shp_only`` mask. A region that is not a country of the shapefile leaves ``da`` as it is.

    Parameters
    ----------
    da : xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates.
    region : str
        The country.

    Returns
    -------
    xarray.DataArray
        The masked data, or ``da`` itself when the region is not a country.
    """
    spec = default_mask(region, False, True)
    if not spec.country:
        return da
    lat, lon = da["lat"].values, da["lon"].values
    keep = build_mask(spec, lat, lon, label="default").keep
    mask_da = xr.DataArray(keep, dims=['lat', 'lon'], coords={'lat': da.lat, 'lon': da.lon})
    return da.where(mask_da)


def apply_nc_mask(ds, nc_mask, mask_var=None, keep_value=1):
    """Set the cells of ``ds`` where the 0/1 NetCDF mask ``nc_mask`` is not ``keep_value`` to NaN.

    Each data cell takes the nearest mask cell. A file that cannot be used as a mask raises ``MaskError`` (see
    `build_mask`).

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data, with ``lat`` and ``lon`` coordinates.
    nc_mask : str or os.PathLike
        The mask file.
    mask_var : str, optional
        The variable of the file; None picks it if the file has only one.
    keep_value : float, default 1
        The value that marks a kept cell.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The masked data (a copy).
    """
    spec = MaskSpec(file=str(nc_mask), variable=mask_var, keep_value=keep_value, regrid="nearest", allow_empty=True)
    keep = build_mask(spec, ds["lat"].values, ds["lon"].values, label="nc_mask").keep
    return _where_keep(ds, keep)


def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """Find the grid points that are inside a polygon.

    Parameters
    ----------
    polygon_lon : array_like of float, shape (vertex,)
        The longitudes of the polygon's vertices.
    polygon_lat : array_like of float, shape (vertex,)
        The latitudes of the polygon's vertices.
    grid_lons : numpy.ndarray of float64
        The longitudes of the grid points: a 1-D axis (shape ``(lon,)``) or a 2-D array (shape ``(lat, lon)``).
    grid_lats : numpy.ndarray of float64
        The latitudes of the grid points: a 1-D axis (shape ``(lat,)``) or a 2-D array (shape ``(lat, lon)``).

    Returns
    -------
    inside_mask : numpy.ndarray of bool, shape (lat, lon)
        True for the points inside the polygon.
    inside_lons : numpy.ndarray of float64, shape (inside,)
        The longitudes of the points inside the polygon.
    inside_lats : numpy.ndarray of float64, shape (inside,)
        The latitudes of the points inside the polygon.
    """
    from matplotlib.path import Path

    # Create polygon path
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    polygon_path = Path(polygon_vertices)

    # Create meshgrid if needed
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats

    # Flatten the grids to test each point
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Test which points are inside the polygon
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)

    # Get coordinates of points inside polygon
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]

    return inside_mask, inside_lons, inside_lats


def polygon_mask(da_model):
    """Mask data outside the polygon boundary of its region.

    The boundary comes from `hcwf.onset.regions.polygon_boundary`.

    Parameters
    ----------
    da_model : xarray.DataArray
        The data, with 1-D ``lat`` and ``lon`` coordinates.

    Returns
    -------
    xarray.DataArray
        The data with the cells outside the polygon set to NaN.
    """
    from hcwf.onset.regions import polygon_boundary

    orig_lat = da_model.lat.values
    orig_lon = da_model.lon.values

    polygon1_lat, polygon1_lon = polygon_boundary(da_model)

    inside_mask, inside_lons, inside_lats = points_inside_polygon(polygon1_lon, polygon1_lat, orig_lon, orig_lat)

    # Selecting with sel(lat=inside_lats, lon=inside_lons) would not work:
    # xarray matches all lats in inside_lats with all lons in inside_lons, forming a cartesian product.
    # so output values are repeated multiple times — every selected latitude is paired with
    # every selected longitude.

    da_model_slice = da_model.where(inside_mask)

    return da_model_slice
