"""One way to draw a spatial field: colored grid cells, or colored polygons (admin units).

The maps of HCWF (spatial metrics, climatology onset, observed onset) all draw a field over the spatial axes of a
DataArray. Over (lat, lon) that is a pcolormesh; over "unit" it is a choropleth of the run's unit polygons, or
markers at the centroids for units that have no polygons (cells, weighted cells). The colour limits, colormaps,
norms and projection transform are the caller's, so the same styling serves both.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path

NO_DATA = "lightgrey"
"""Colour of a polygon without a value."""
_MARKER_AREA = 60


def colorbar_extend(mappable) -> str:
    """The `extend` of the colorbar of a drawn field: an arrow at each end where values lie beyond the colour limits.

    Values past a limit are drawn in the end colour, so the bar says so with an arrow. A BoundaryNorm that already
    extends keeps that end (its extension is part of the colour mapping), whether or not a value lies there.
    Returns "neither", "min", "max" or "both"."""
    mappable.autoscale_None()
    norm = mappable.norm
    if isinstance(norm, BoundaryNorm):
        low, high, fixed = norm.boundaries[0], norm.boundaries[-1], norm.extend
    else:
        low, high, fixed = norm.vmin, norm.vmax, "neither"
    values = np.ma.masked_invalid(np.ma.asarray(mappable.get_array(), dtype=float)).compressed()
    below = fixed in ("min", "both") or bool(values.size and values.min() < low)
    above = fixed in ("max", "both") or bool(values.size and values.max() > high)
    return "both" if below and above else "min" if below else "max" if above else "neither"


def units_of(da, kwargs):
    """The run's units (names, centroids, polygons) of a DataArray over "unit", or None for a grid field."""
    if "unit" not in da.dims:
        return None
    from hcwf.io.input import unit_table

    return unit_table(kwargs, da["unit"].values)


def extent(da, units=None):
    """(lats, lons) of a field: its axes for a grid, the min and max of the polygons (or centroids) for units."""
    if units is None:
        return da.lat.values, da.lon.values
    if units.geometry is not None:
        west, south, east, north = _bounds(units.geometry)
        return np.array([south, north]), np.array([west, east])
    return np.array([np.nanmin(units.lat), np.nanmax(units.lat)]), np.array([np.nanmin(units.lon), np.nanmax(units.lon)])


def _bounds(geometries):
    """(west, south, east, north) of a list of shapely geometries."""
    bounds = np.array([g.bounds for g in geometries])
    return bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()


def _path(geometry) -> Path:
    """The matplotlib Path of a shapely Polygon or MultiPolygon (holes included)."""
    parts = getattr(geometry, "geoms", [geometry])
    vertices, codes = [], []
    for part in parts:
        for ring in (part.exterior, *part.interiors):
            xy = np.asarray(ring.coords)[:, :2]
            vertices.append(xy)
            codes.append(np.r_[Path.MOVETO, np.full(len(xy) - 2, Path.LINETO), Path.CLOSEPOLY])
    return Path(np.concatenate(vertices), np.concatenate(codes))


def _draw_units(ax, da, units, style, edgecolor="grey", linewidth=0.3):
    """Polygons of the units coloured by the values of da; units without a value are grey. Returns the mappable."""
    values = np.asarray(da.transpose("unit").values, dtype=float)
    order = {unit: n for n, unit in enumerate(units.ids)}
    at = np.array([order[str(u)] for u in da["unit"].values])
    vmin, vmax = style.pop("vmin", None), style.pop("vmax", None)
    transform = style.get("transform")
    extra = {} if transform is None else {"transform": transform}
    has_value = np.isfinite(values)
    if units.geometry is None:   # no polygons: markers at the centroids
        mappable = ax.scatter(units.lon[at][has_value], units.lat[at][has_value], c=values[has_value], s=_MARKER_AREA,
                              edgecolors=edgecolor, linewidths=linewidth, **style, **extra)
        ax.scatter(units.lon[at][~has_value], units.lat[at][~has_value], c=NO_DATA, s=_MARKER_AREA,
                   edgecolors=edgecolor, linewidths=linewidth, **extra)
    else:
        patches = [PathPatch(_path(units.geometry[i])) for i in at]
        mappable = PatchCollection([p for p, ok in zip(patches, has_value) if ok], edgecolor=edgecolor,
                                   linewidth=linewidth, **style)
        mappable.set_array(values[has_value])
        ax.add_collection(mappable)
        if (~has_value).any():
            ax.add_collection(PatchCollection([p for p, ok in zip(patches, has_value) if not ok], facecolor=NO_DATA,
                                              edgecolor=edgecolor, linewidth=linewidth, **extra))
        ax.autoscale_view()
    if vmin is not None and vmax is not None:
        mappable.set_clim(vmin, vmax)
    return mappable


def draw_field(ax, da, units=None, *, shading="auto", **style):
    """Draw a field on `ax`; return the mappable to give a colorbar.

    `da` is a (lat, lon) DataArray (a pcolormesh: `shading` "auto" puts a cell on each coordinate, "flat" draws cell
    edges half a step around the coordinates and leaves NaN unfilled) or a (unit) DataArray, drawn on the polygons of
    `units` (hcwf.graphics.field.units_of). `style` is passed to the artist: cmap, norm, vmin, vmax, transform.
    """
    if "unit" in da.dims:
        return _draw_units(ax, da, units, dict(style))
    if shading == "flat":
        lons, lats = da.lon.values, da.lat.values
        lon_edges = np.concatenate([lons - (lons[1] - lons[0]) / 2, [lons[-1] + (lons[1] - lons[0]) / 2]])
        lat_edges = np.concatenate([lats - (lats[1] - lats[0]) / 2, [lats[-1] + (lats[1] - lats[0]) / 2]])
        edges = np.meshgrid(lon_edges, lat_edges)
        return ax.pcolormesh(*edges, np.ma.masked_invalid(da.values), shading="flat", **style)
    return ax.pcolormesh(da.lon, da.lat, da.values, shading="auto", **style)
