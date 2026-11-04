"""One way to draw a spatial field: colored grid cells, or colored polygons (admin units).

The maps of HCWF (spatial metrics, climatology onset, observed onset) all draw a field over the spatial axes of a
`xarray.DataArray`. Over ``(lat, lon)`` that is a pcolormesh; over ``"unit"`` it is a choropleth of the run's unit
polygons, or markers at the centroids for units that have no polygons (cells, weighted cells). The colour limits,
colormaps, norms and projection transform are the caller's, so the same styling serves both. The map modules
`hcwf.figures.maps`, `hcwf.figures.onset_map` and `hcwf.figures.func_map` call `draw_field`.
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
"""Marker area (points squared) of a unit drawn at its centroid."""


def colorbar_extend(mappable) -> str:
    """Return the ``extend`` of the colorbar of a drawn field.

    Values past a colour limit are drawn in the end colour, so the bar says so with an arrow at that end. A
    `matplotlib.colors.BoundaryNorm` that already extends keeps that end (its extension is part of the colour
    mapping), whether or not a value lies there.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The drawn field (a mesh, collection or scatter) whose norm and array are inspected; its norm limits are
        autoscaled when unset.

    Returns
    -------
    str
        One of ``"neither"``, ``"min"``, ``"max"`` or ``"both"``: the ends with an arrow.
    """
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


def units_of(da, spec):
    """Return the run's units (names, centroids, polygons) of a field over ``"unit"``, or None for a grid field.

    Parameters
    ----------
    da : xarray.DataArray
        The field; it has a ``unit`` dimension (units) or ``lat`` and ``lon`` (grid).
    spec : object
        The run's ``FigureSpec``, or the observations' ``UnitsSpec`` or ``ObsSpec``: anything with a ``units``
        attribute, or a units spec itself.

    Returns
    -------
    object or None
        The unit table of ``da["unit"]`` (see `hcwf.spatial.onto_units.unit_table`), or None for a grid field.
    """
    if "unit" not in da.dims:
        return None
    from hcwf.spatial.onto_units import unit_table

    return unit_table(getattr(spec, "units", spec), da["unit"].values)


def extent(da, units=None):
    """Return the latitudes and longitudes spanned by a field.

    Parameters
    ----------
    da : xarray.DataArray
        The field; used for its ``lat`` and ``lon`` axes when `units` is None.
    units : object, optional
        The unit table from `units_of`. None means a grid field.

    Returns
    -------
    lats : numpy.ndarray of float64, shape (lat,) or (2,)
        The latitude axis for a grid; ``[south, north]`` (polygon bounds, or centroid min and max) for units.
    lons : numpy.ndarray of float64, shape (lon,) or (2,)
        The longitude axis for a grid; ``[west, east]`` for units.
    """
    if units is None:
        return da.lat.values, da.lon.values
    if units.geometry is not None:
        west, south, east, north = _bounds(units.geometry)
        return np.array([south, north]), np.array([west, east])
    return np.array([np.nanmin(units.lat), np.nanmax(units.lat)]), np.array([np.nanmin(units.lon), np.nanmax(units.lon)])


def _bounds(geometries):
    """Return the joint bounding box of shapely geometries.

    Parameters
    ----------
    geometries : sequence of shapely.geometry.base.BaseGeometry
        The geometries.

    Returns
    -------
    tuple of (float, float, float, float)
        ``(west, south, east, north)``.
    """
    bounds = np.array([g.bounds for g in geometries])
    return bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()


def _path(geometry) -> Path:
    """Return the matplotlib path of a shapely polygon (holes included).

    Parameters
    ----------
    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The geometry; every exterior and interior ring becomes a closed sub-path.

    Returns
    -------
    matplotlib.path.Path
        The path, with ``MOVETO``, ``LINETO`` and ``CLOSEPOLY`` codes per ring.
    """
    parts = getattr(geometry, "geoms", [geometry])
    vertices, codes = [], []
    for part in parts:
        for ring in (part.exterior, *part.interiors):
            xy = np.asarray(ring.coords)[:, :2]
            vertices.append(xy)
            codes.append(np.r_[Path.MOVETO, np.full(len(xy) - 2, Path.LINETO), Path.CLOSEPOLY])
    return Path(np.concatenate(vertices), np.concatenate(codes))


def _draw_units(ax, da, units, style, edgecolor="grey", linewidth=0.3):
    """Draw the units coloured by the values of a field; units without a value are grey.

    Polygons are drawn as a patch collection; units without polygons (``units.geometry`` is None) are drawn as
    markers at their centroids.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    da : xarray.DataArray
        The field, with a ``unit`` dimension.
    units : object
        The unit table from `units_of` (``ids``, ``lat``, ``lon``, ``geometry``).
    style : dict
        Artist keywords (``cmap``, ``norm``, ``transform``, ...). ``vmin`` and ``vmax`` are removed from it and set
        as the colour limits when both are given.
    edgecolor : str, default "grey"
        The outline colour of units.
    linewidth : float, default 0.3
        The outline width of units.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        The collection of the units that have a value, to give a colorbar.
    """
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
    """Draw a field on `ax` and return the mappable to give a colorbar.

    A ``(lat, lon)`` field is a pcolormesh. A ``(unit)`` field is drawn on the polygons of `units` (see `units_of`).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    da : xarray.DataArray
        The field, with dims ``(lat, lon)`` or ``(unit,)``.
    units : object, optional
        The unit table from `units_of`; needed for a ``(unit)`` field.
    shading : {"auto", "flat"}, default "auto"
        For a grid field: ``"auto"`` puts a cell on each coordinate; ``"flat"`` draws cell edges half a step around
        the coordinates and leaves NaN unfilled.
    **style : dict
        Passed to the artist: ``cmap``, ``norm``, ``vmin``, ``vmax``, ``transform``.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        The drawn mesh or collection.
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
