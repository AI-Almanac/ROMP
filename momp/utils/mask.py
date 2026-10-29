"""Region masks built on each dataset's own grid, never by regridding rainfall.

A run config's ``mask:`` section (see ``MaskSpec``) names the cells to evaluate by one or more of

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

Nothing is silent: ``build_mask`` returns the lines describing the sources, rule, alignment actions and kept
cell count, ``region_mask`` prints them once per dataset and grid, saves a small PNG of the kept cells to
``dir_fig/mask_<dataset>.png`` and keeps a record (sources with content hashes, rule, kept counts) that the
resolved config and the NetCDF outputs carry (``records``, ``mask_attrs``).
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from momp.lib.assertion import ROMPConfigError
import logging

log = logging.getLogger(__name__)

CELL_RULES = ("centre", "overlap_fraction")
"""How a shape decides whether a data cell is kept."""
RESOLUTIONS = ("10m", "50m", "110m")
"""Natural Earth resolutions for ``land`` and ``country``."""
SHAPE_SUFFIXES = (".shp", ".geojson", ".json", ".gpkg")
"""File suffixes read as a shape (everything else is read as NetCDF)."""
_SIDECARS = (".dbf", ".shx", ".prj", ".cpg")
_NE_COUNTRIES = ("cultural", "admin_0_countries")
_NE_LAND = ("physical", "land")
_COUNTRY_FIELDS = ("NAME", "NAME_LONG", "ADM0_A3")
_LAT_NAMES = ("lat", "latitude", "y", "nav_lat", "lats")
_LON_NAMES = ("lon", "longitude", "x", "nav_lon", "lons", "long")
_EPS = 1e-9


class MaskError(ROMPConfigError):
    """A region mask cannot be built: bad specification, unreadable source, grid not covered, or no cell kept."""


@dataclass(frozen=True)
class MaskSpec:
    """The ``mask:`` section of a run config.

    land: keep land cells only. country: country names (Natural Earth NAME, NAME_LONG or ADM0_A3, any case).
    file: a 0/1 NetCDF mask (variable, keep_value) or a shapefile / GeoJSON. condition: {file, variable, above,
    below}, a mask from a field. cell_rule and overlap_fraction: how shapes (and finer mask files) are turned
    into data cells. resolution: Natural Earth resolution of land and country shapes.
    """

    land: bool = False
    country: tuple = ()
    file: str | None = None
    variable: str | None = None
    keep_value: float = 1
    condition: dict | None = None
    cell_rule: str = "centre"
    overlap_fraction: float = 0.5
    resolution: str = "50m"

    @classmethod
    def from_mapping(cls, raw) -> "MaskSpec":
        """Build the spec from the (normalised) mapping of the config; a MaskSpec is returned as it is."""
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
        """A text identifying the spec, for caching."""
        return json.dumps(self.__dict__, sort_keys=True, default=str)


@dataclass
class MaskResult:
    """A mask on one dataset's grid: the boolean cells kept, the lines that describe it and its record."""

    keep: np.ndarray
    lines: list
    record: dict
    outline: object = None
    lat: np.ndarray = field(default=None)
    lon: np.ndarray = field(default=None)


# ----------------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------------

def cell_edges(centres, fallback=None) -> np.ndarray:
    """Cell edges (n + 1) of ascending cell centres: midpoints between centres, half a step beyond the ends.

    A single centre needs `fallback`, the cell size to assume.
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
    """The median step of ascending cell centres (`fallback` for a single centre)."""
    c = np.asarray(centres, dtype=float)
    return float(np.median(np.diff(c))) if c.size > 1 else float(fallback or 0.0)


def _deg(value: float) -> str:
    """A cell size or coordinate in degrees, short."""
    return f"{value:.6g}"


def _grid_text(lat, lon) -> str:
    """'nlon x nlat cells of dlon x dlat deg' for ascending centres."""
    return (f"{len(lon)} x {len(lat)} cells of {_deg(cell_size(lon, cell_size(lat)))} x "
            f"{_deg(cell_size(lat, cell_size(lon)))} deg")


def _make_valid(geom):
    """The geometry repaired if it is invalid."""
    import shapely

    return geom if shapely.is_valid(geom) else shapely.make_valid(geom)


def rasterise(geom, lat, lon, rule="centre", fraction=0.5) -> np.ndarray:
    """The cells of the grid (ascending centres lat, lon) kept by a shape under the cell rule; shape (nlat, nlon).

    centre: the cell centre is inside the shape (on its boundary counts as inside). overlap_fraction: the part
    of the cell's area (planar, in degrees squared) inside the shape is at least `fraction`. The shape is in
    longitude -180..180; cell longitudes are wrapped into that range.
    """
    import shapely

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    fallback = cell_size(lon, cell_size(lat, 1.0)) or 1.0
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    shift = wrapped - lon
    lon_edges = cell_edges(lon, cell_size(lat, fallback))
    lon_lo, lon_hi = lon_edges[:-1] + shift, lon_edges[1:] + shift
    lat_edges = cell_edges(lat, cell_size(lon, fallback))
    lat_lo, lat_hi = lat_edges[:-1], lat_edges[1:]
    out = np.zeros((lat.size, lon.size), dtype=bool)
    geom = _make_valid(geom)
    if geom.is_empty:
        return out
    minx, miny, maxx, maxy = geom.bounds
    rows = np.nonzero((lat_hi >= miny) & (lat_lo <= maxy))[0]
    cols = np.nonzero((lon_hi >= minx) & (lon_lo <= maxx))[0]
    if rows.size == 0 or cols.size == 0:
        return out
    shapely.prepare(geom)
    if rule == "centre":
        xx, yy = np.meshgrid(wrapped[cols], lat[rows])
        out[np.ix_(rows, cols)] = shapely.intersects_xy(geom, xx, yy)
        return out
    x0, y0 = np.meshgrid(lon_lo[cols], lat_lo[rows])
    x1, y1 = np.meshgrid(lon_hi[cols], lat_hi[rows])
    boxes = shapely.box(x0.ravel(), y0.ravel(), x1.ravel(), y1.ravel())
    share = shapely.contains(geom, boxes).astype(float)
    partly = shapely.intersects(geom, boxes) & (share == 0)
    if partly.any():
        share[partly] = shapely.area(shapely.intersection(geom, boxes[partly])) / shapely.area(boxes[partly])
    out[np.ix_(rows, cols)] = (share >= fraction - _EPS).reshape(rows.size, cols.size)
    return out


# ----------------------------------------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------------------------------------

def file_hash(path) -> str:
    """The SHA-256 of a file's content; a shapefile also covers its .dbf, .shx, .prj and .cpg files."""
    path = Path(path)
    digest = hashlib.sha256()
    names = [path] + ([path.with_suffix(s) for s in _SIDECARS] if path.suffix.lower() == ".shp" else [])
    for item in names:
        if item.exists():
            with open(item, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


_NE_CACHE: dict = {}


def _natural_earth(resolution, category, name):
    """The path of a Natural Earth shapefile, downloaded on first use; offline errors say how to fix them."""
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
    """The (path, records) of Natural Earth countries (which='country') or land (which='land'), cached."""
    from cartopy.io import shapereader

    category, name = _NE_COUNTRIES if which == "country" else _NE_LAND
    key = (resolution, name)
    if key not in _NE_CACHE:
        path = _natural_earth(resolution, category, name)
        _NE_CACHE[key] = (path, list(shapereader.Reader(path).records()))
    return _NE_CACHE[key]


def country_geometry(names, resolution):
    """The union of the named countries' shapes and the (matched names, shapefile path).

    A name matches NAME, NAME_LONG or ADM0_A3 of Natural Earth admin_0_countries, any case. An unknown name is
    an error that lists the closest matches.
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


def land_geometry(resolution):
    """The union of the Natural Earth land polygons and the shapefile path."""
    import shapely

    path, records = natural_earth_records(resolution, "land")
    return shapely.union_all([_make_valid(r.geometry) for r in records]), path


def shape_file_geometry(path):
    """The union of every geometry of a shapefile or GeoJSON, in longitude/latitude degrees (EPSG:4326)."""
    import geopandas as gpd

    try:
        frame = gpd.read_file(path)
    except Exception as exc:
        raise MaskError(f"mask.file: cannot read the shape file {path}: {type(exc).__name__}: {exc}") from exc
    if frame.crs is not None and not frame.crs.equals("EPSG:4326"):
        frame = frame.to_crs("EPSG:4326")
    return _make_valid(frame.geometry.union_all())


def _find_axis(ds, names, units, what, path):
    """The name of the latitude or longitude coordinate of a dataset, by name, units and standard_name."""
    for name in list(ds.coords) + list(ds.dims):
        if str(name).lower() in names:
            return name
    for name in ds.coords:
        attrs = ds[name].attrs
        if str(attrs.get("units", "")).lower() in units or str(attrs.get("standard_name", "")).lower() == what:
            return name
    raise MaskError(f"{path}: no {what} coordinate found (looked for {', '.join(names)}); coordinates are "
                    f"{', '.join(map(str, ds.coords))}")


def read_raster_mask(path, variable=None, keep_value=1, condition=None):
    """Read a 0/1 mask (or a field with above / below) from NetCDF as (kept 2-D bool, lat, lon, notes).

    The result has dims (lat, lon) in the file's own order and longitude convention; `notes` names the axes
    and variable found. A field is thresholded on its own grid first (above is strict, below is strict, both
    give a band); cells without a value are not kept.
    """
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as exc:
        raise MaskError(f"cannot open the mask file {path}: {type(exc).__name__}: {exc}") from exc
    with ds:
        lat_name = _find_axis(ds, _LAT_NAMES, ("degrees_north", "degree_north", "degrees_n"), "latitude", path)
        lon_name = _find_axis(ds, _LON_NAMES, ("degrees_east", "degree_east", "degrees_e"), "longitude", path)
        if variable is None:
            candidates = [v for v in ds.data_vars if lat_name in ds[v].dims and lon_name in ds[v].dims]
            if len(candidates) != 1:
                raise MaskError(f"{path}: {'no variable' if not candidates else 'several variables'} on "
                                f"({lat_name}, {lon_name}) to choose from ({', '.join(candidates) or 'none'}); set "
                                "the mask variable")
            variable = candidates[0]
        elif variable not in ds.data_vars:
            raise MaskError(f"{path}: no variable {variable!r}; the file has {', '.join(map(str, ds.data_vars))}"
                            f"{_suggest(variable, ds.data_vars)}")
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


def _suggest(name, candidates) -> str:
    """' Did you mean "x"?' for the closest candidate."""
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=1, cutoff=0.6)
    return f' Did you mean "{close[0]}"?' if close else ""


# ----------------------------------------------------------------------------------------------------------
# aligning a raster mask with a data grid
# ----------------------------------------------------------------------------------------------------------

def _overlap(lo_a, hi_a, lo_b, hi_b) -> np.ndarray:
    """Matrix [a, b] of the length shared by interval a of the first set and interval b of the second."""
    return np.clip(np.minimum(hi_a[:, None], hi_b[None, :]) - np.maximum(lo_a[:, None], lo_b[None, :]), 0.0, None)


def _nearest(values, targets) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest ascending `values` to each target, and the distance."""
    right = np.clip(np.searchsorted(values, targets), 1, len(values) - 1) if len(values) > 1 else np.zeros(
        len(targets), dtype=int)
    left = np.clip(right - 1, 0, None)
    pick = np.where(np.abs(values[right] - targets) < np.abs(targets - values[left]), right, left)
    return pick, np.abs(values[pick] - targets)


def align_raster_mask(kept, mlat, mlon, lat, lon, fraction=0.5, label="mask file"):
    """Bring a boolean mask on its own grid onto the data grid (ascending centres lat, lon).

    Returns (kept on the data grid, actions): the actions are the printable steps taken, in order: latitude
    flip, longitude wrap, then either direct use (coordinates equal within a hundredth of a cell) or the
    regridding of the mask (area-weighted fraction >= `fraction` for a finer mask, nearest cell for a coarser
    or equally fine one). A mask that does not cover the data extent raises MaskError naming both extents.
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
    if mres[0] >= dres[0] - tol and mres[1] >= dres[1] - tol:
        actions.append(f"regridded the mask onto the data grid by the nearest mask cell (mask cells "
                       f"{_deg(mres[1])} x {_deg(mres[0])} deg are not finer than the data's "
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
    """One record of what a mask part was made from."""
    out = {"part": kind, "source": name}
    if path:
        out["path"] = str(path)
        out["sha256"] = file_hash(path)
    if extra:
        out.update(extra)
    return out


def _rule_text(spec: MaskSpec) -> str:
    """The cell rule in words."""
    if spec.cell_rule == "centre":
        return "centre (a cell is kept when its centre is inside the shape)"
    return f"overlap_fraction >= {spec.overlap_fraction:g} (a cell is kept when that share of its area is inside)"


def _rule_short(spec: MaskSpec) -> str:
    """The cell rule in a few words."""
    return "centre rule" if spec.cell_rule == "centre" else f"overlap_fraction >= {spec.overlap_fraction:g} rule"


def build_mask(spec: MaskSpec, lat, lon, label="data") -> MaskResult:
    """The mask of one dataset on its own grid: boolean (nlat, nlon) in the order of lat and lon given.

    Every part of the spec is applied and the parts intersect. The result carries the printable lines and the
    record (sources with content hashes, rule, kept counts). Raises MaskError if a source is unusable, the
    grid is not covered or no cell is kept.
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
        geom, path = land_geometry(spec.resolution)
        shapes.append(geom)
        shape_names.append("land")
        sources.append(_source("land", f"natural_earth {spec.resolution} physical/land", path))
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
        raster = rasterise(shape, slat, slon, spec.cell_rule, spec.overlap_fraction)
        keep &= raster
        parts.append((" + ".join(shape_names) + f" rasterised on the data grid by the {_rule_short(spec)}", int(raster.sum())))
    for text, kept, mlat, mlon, notes in rasters:
        lines.append(f"  {text}: {notes}")
        aligned, actions = align_raster_mask(kept, mlat, mlon, slat, slon, spec.overlap_fraction, label="mask")
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
    if count == 0:
        raise MaskError("\n".join(summary) + f"\nthe mask keeps no cell of the {label} grid")
    return MaskResult(keep=out, lines=summary, record=record, outline=outline, lat=lat, lon=lon)


# ----------------------------------------------------------------------------------------------------------
# picture
# ----------------------------------------------------------------------------------------------------------

def save_mask_png(result: MaskResult, path, title="") -> Path:
    """Save a small PNG of the kept cells (and the shape outline if there is one), in the onset figure colours."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    from momp.onset.viz.style import palette

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

_BUILT: dict = {}
_RECORDS: dict = {}


def records() -> dict:
    """The records of the masks built so far, by dataset name."""
    return dict(_RECORDS)


def reset() -> None:
    """Forget the built masks and records (for a new run in the same process)."""
    _BUILT.clear()
    _RECORDS.clear()


def region_mask(lat, lon, mask, dataset="data", dir_fig=None) -> np.ndarray:
    """The boolean mask (nlat, nlon) of one dataset's grid for a config's `mask` mapping, built once per grid.

    The first time a dataset's grid is seen, the mask is built, its description is printed, the PNG is saved
    to dir_fig/mask_<dataset>.png (when dir_fig is given) and the record is stored for `records`.
    """
    spec = MaskSpec.from_mapping(mask)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    key = (spec.key(), str(dataset), hashlib.sha256(lat.tobytes() + lon.tobytes()).hexdigest())
    if key not in _BUILT:
        result = build_mask(spec, lat, lon, label=str(dataset))
        log.info("\n".join(result.lines))
        if dir_fig:
            png = save_mask_png(result, Path(dir_fig) / f"mask_{dataset}.png", title=str(dataset))
            result.record["figure"] = str(png)
            log.info(f"  mask figure: {png}")
        _BUILT[key] = result
        _RECORDS[str(dataset)] = result.record
    return _BUILT[key].keep


def apply_region_mask(ds, mask, dataset="data", dir_fig=None):
    """`ds` (Dataset or DataArray with lat and lon) with the cells outside the mask set to NaN.

    The mask is built on this dataset's own grid, so the data are never regridded. Variables without both
    lat and lon dimensions are left as they are.
    """
    keep = region_mask(ds["lat"].values, ds["lon"].values, mask, dataset, dir_fig)
    cond = xr.DataArray(keep, dims=("lat", "lon"), coords={"lat": ds["lat"], "lon": ds["lon"]})
    if isinstance(ds, xr.DataArray):
        return ds.where(cond)
    out = ds.copy()
    for name, var in ds.data_vars.items():
        if "lat" in var.dims and "lon" in var.dims:
            out[name] = var.where(cond)
    return out


def record_lines(rec: dict) -> list[str]:
    """One mask record as printable comment lines (sources with hashes, rule, kept cells)."""
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
    """NetCDF global attributes describing the masks built so far (empty when no mask is in use)."""
    if not _RECORDS:
        return {}
    first = next(iter(_RECORDS.values()))
    out = {"mask_cell_rule": first["rule"],
           "mask_sources": "; ".join(
               f"{s['part']}={s.get('path') or s['source']}" + (f" sha256:{s['sha256']}" if s.get("sha256") else "")
               for s in first["sources"])}
    if first["rule"] == "overlap_fraction":
        out["mask_overlap_fraction"] = first["overlap_fraction"]
    for name, rec in _RECORDS.items():
        out[f"mask_kept_cells_{name}"] = f"{rec['kept']} of {rec['total']}"
    return out
