"""Rain on units: source cells -> unit series (onset_blending's regrid and cell transform, as one transform).

`unit_series` and `read_on_units` aggregate cells onto units, reading only the storage tiles that feed them; `read_units` also
reads a file whose ``unit`` dimension is already there. The run's evaluation units come from `unit_table`, `units_spec`,
`units_files`.

The module builds on `hcwf.spatial.units` (the table of units and the transform of each source grid onto them),
`hcwf.spatial.select` (the block-wise reading of lazy arrays) and `hcwf.spatial.axes`; the readers of `hcwf.io` call it.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.cache import file_identity
from hcwf.onset.ingest import unit_scale
from hcwf.spatial import axes
from hcwf.spatial import ids as unit_ids
from hcwf.spatial.select import load_blocks, region_box, scale_in_place
from hcwf.spatial.units import EvaluationUnits, UnitsSpec

__all__ = ["declared_scale", "point_dims", "read_units", "select_units", "unit_table", "unit_transform", "units_files"]

log = logging.getLogger(__name__)

SOURCE_POLICIES = ("flag", "keep", "zero", "drop")
"""Missing-rain policies applied to the source cells before they are averaged onto units: "flag" and "keep" leave
NaN as it is (a unit averages its observed sources), "zero" counts NaN as 0 mm and "drop" removes every source cell
with a missing day in a calendar year from that year's means (see `unit_series`)."""


def unit_transform(units, lat, lon, positions=None, *, tolerance=None, wrap_longitude=False, convention=None):
    """Build the transform of ``units`` from the cells of a (lat, lon) grid, restricted to the units at ``positions``.

    The weights are matched to the grid's cells by exact coordinates (as onset_blending matches them) or within
    ``tolerance``; ``wrap_longitude`` first puts the weights' longitudes in the grid's convention.

    Parameters
    ----------
    units : hcwf.spatial.units.EvaluationUnits
        The evaluation units.
    lat : numpy.ndarray of float64, shape (lat,)
        The latitudes of the source grid, in degrees.
    lon : numpy.ndarray of float64, shape (lon,)
        The longitudes of the source grid, in degrees.
    positions : array_like of int, shape (kept,), optional
        The positions of the units to keep, in that order; all units when None.
    tolerance : float, optional
        How far (in degrees) a weight's cell may be from a grid cell; None for exact coordinates.
    wrap_longitude : bool, default False
        Put the weights' longitudes in the grid's convention first.
    convention : hcwf.spatial.ids.GridIdConvention, optional
        The grid-id convention of weights keyed by source ids.

    Returns
    -------
    hcwf.spatial.regrid.SparseTransform
        The transform from the grid's cells to the (kept) units.
    """
    full = units.transform(lat, lon, tolerance=tolerance, convention=convention, wrap_longitude=wrap_longitude)
    if positions is None:
        return full
    return full.select_targets(positions)


def _cell_positions(transform, lat, lon, flat=None):
    """Find each cell's position among the transform's sources (-1 for a cell that is none).

    The cells are those of the (lat, lon) grid in latitude-major order, or, with ``flat`` (each cell's position in the
    whole grid), the cells given one by one as lat and lon arrays. A coordinate-keyed transform has the grid's cells
    themselves as its sources, so a cell's position is its place in the grid.

    Parameters
    ----------
    transform : hcwf.spatial.regrid.SparseTransform
        The transform whose sources are searched.
    lat : numpy.ndarray of float64
        The latitudes: the grid's axis, or one per cell with ``flat``.
    lon : numpy.ndarray of float64
        The longitudes: the grid's axis, or one per cell with ``flat``.
    flat : array_like of int, optional
        The position of each cell in the whole (lat, lon) grid, for cells given one by one.

    Returns
    -------
    numpy.ndarray of int64, shape (cell,)
        The source position of each cell, -1 where the cell is not a source.
    """
    if transform.keys == "coordinate":
        return np.arange(len(lat) * len(lon), dtype=np.int64) if flat is None else np.asarray(flat, dtype=np.int64)
    if flat is None:
        lat, lon = (a.reshape(-1) for a in np.meshgrid(np.asarray(lat, float), np.asarray(lon, float), indexing="ij"))
    return transform.positions(lat, lon)


def feeding_cells(transform, lat, lon):
    """Find the grid cells that are a source of some unit of the transform (what a read must keep).

    Parameters
    ----------
    transform : hcwf.spatial.regrid.SparseTransform
        The transform from the grid's cells to units.
    lat : numpy.ndarray of float64, shape (lat,)
        The latitudes of the grid.
    lon : numpy.ndarray of float64, shape (lon,)
        The longitudes of the grid.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True where the cell is a source.
    """
    sources = np.unique(transform.weighted_matrix.indices)
    keep = np.zeros(len(lat) * len(lon), dtype=bool)
    if transform.keys == "coordinate":
        keep[sources] = True
    else:
        keep = np.isin(_cell_positions(transform, lat, lon), sources)
    return keep.reshape(len(lat), len(lon))


def unit_series(da, transform, *, missing="flag", post_scale=None, support=False, dtype=None):
    """Average a loaded DataArray with lat and lon dims onto the units of ``transform``: dims become (the others, unit).

    Each unit is the weighted mean of its observed (finite) source cells, in float64 mm, NaN where none is observed
    (`~hcwf.spatial.regrid.SparseTransform.observed_mean`). ``missing`` is applied to the sources first (``"zero"`` counts NaN as 0 mm;
    ``"drop"`` needs a ``time`` dim and removes a cell from the means of every calendar year in which it has a missing
    day, see `SOURCE_POLICIES`). ``post_scale`` multiplies the means (a unit factor applied after the averaging, as
    onset_blending does for a regrid; apply it to ``da`` first for a cell transform). The array is read in column blocks,
    so working memory stays one block beyond the result.

    The array may hold only some of the grid's cells (``load_blocks(..., compact=True)``: the dims others + ``cell``,
    with ``lat`` and ``lon`` over ``cell``); the cells it lacks are not observed. Units that are the cells themselves
    (one source of weight 1) take their cell's value directly, without the weighted sums. The result is float64, or
    ``dtype`` when that changes nothing but the storage (cells' own values, no ``post_scale``).

    Parameters
    ----------
    da : xarray.DataArray
        The rain in mm, with ``lat`` and ``lon`` dims (or a compacted ``cell`` dim) and any others, such as ``time``.
    transform : hcwf.spatial.regrid.SparseTransform
        The transform from the array's cells to the units.
    missing : {"flag", "keep", "zero", "drop"}, default "flag"
        The missing-rain policy for the sources (see `SOURCE_POLICIES`).
    post_scale : float, optional
        A factor multiplied onto the means after averaging.
    support : bool, default False
        Also return which units have a source each year, and the years.
    dtype : numpy.dtype or type, optional
        The dtype of the result where that changes nothing but the storage.

    Returns
    -------
    xarray.DataArray or tuple of (xarray.DataArray, numpy.ndarray, numpy.ndarray)
        The unit series with dims (the others, ``unit``), float64 or ``dtype``. With ``support`` a tuple of that, the
        ``bool`` array of shape (unit, year) of which units have a source that year, and the years (``int``, shape
        (year,); ``[0]`` when the array has no ``time``).

    Raises
    ------
    ValueError
        ``missing`` is not one of `SOURCE_POLICIES`, or is ``"drop"`` and the dims are not ``(time, lat, lon)``.
    """
    if missing not in SOURCE_POLICIES:
        raise ValueError(f"missing must be one of {SOURCE_POLICIES}, got {missing!r}")
    if "cell" in da.dims:
        spatial, lat, lon, flat = ["cell"], da["lat"].values, da["lon"].values, da["cell"].values
        place = _cell_positions(transform, lat, lon, flat)
    else:
        spatial, lat, lon = ["lat", "lon"], np.asarray(da["lat"].values), np.asarray(da["lon"].values)
        place = _cell_positions(transform, lat, lon)
    others = [d for d in da.dims if d not in spatial]
    if missing == "drop" and others != ["time"]:
        raise ValueError("missing 'drop' needs data with the dims (time, lat, lon)")
    view = np.moveaxis(da.values, [da.dims.index(d) for d in spatial], list(range(-len(spatial), 0)))
    rest_shape = view.shape[:-len(spatial)]
    cells = view.reshape(int(np.prod(rest_shape, dtype=np.int64)), place.size)
    n_unit, n_source = transform.shape
    used = np.flatnonzero(place >= 0)
    whole = len(used) == n_source == len(place) and bool((place == np.arange(n_source)).all())
    single = None if missing == "drop" else transform.single_sources
    exact = single is not None and (post_scale is None or post_scale == 1.0)
    out = np.empty((cells.shape[0], n_unit), dtype=dtype if exact and dtype is not None else float)

    def sources(rows):
        """Return the float64 values (source, row) of the given rows of `cells`."""
        block = cells[rows]
        if whole:
            return block.T.astype(float)
        values = np.full((n_source, block.shape[0]), np.nan)
        values[place[used]] = block[:, used].T.astype(float)
        return values

    if single is not None:
        column = np.full(n_source, -1, dtype=np.int64)
        column[place[used]] = used
        wanted = column[single]
        absent = (wanted < 0) | (not cells.shape[1])
        wanted = np.where(wanted < 0, 0, wanted)

    def own(rows):
        """Return the rows' values (row, unit) at the units' own cells: NaN where the cell is missing or not finite."""
        taken = cells[rows]
        block = taken[:, wanted] if taken.shape[1] else np.full((taken.shape[0], n_unit), np.nan)
        if not np.issubdtype(block.dtype, np.floating):
            block = block.astype(float)
        if absent.any():
            block[:, absent] = np.nan
        if missing == "zero":
            np.copyto(block, 0.0, where=np.isnan(block))
        block[~np.isfinite(block)] = np.nan
        return block

    years = present = None
    if missing == "drop":
        times = da["time"]
        years = np.asarray(times.dt.year.values)
        distinct = np.unique(years)
        present = np.ones((n_source, len(distinct)), dtype=bool)
        for y, year in enumerate(distinct):
            rows = np.flatnonzero(years == year)
            values = sources(rows)
            incomplete = np.isnan(values).any(axis=1)
            values[incomplete] = np.nan
            present[incomplete, y] = False
            out[rows] = transform.observed_mean(values).T
    else:
        for start in range(0, cells.shape[0], 256):
            rows = slice(start, min(cells.shape[0], start + 256))
            if single is not None:
                out[rows] = own(rows)
                continue
            values = sources(rows)
            if missing == "zero":
                values = np.where(np.isnan(values), 0.0, values)
            out[rows] = transform.observed_mean(values).T
    if post_scale is not None and post_scale != 1.0:
        out *= post_scale
    if dtype is not None and out.dtype != dtype:
        out = out.astype(dtype)
    coords = {d: da[d].values for d in others if d in da.coords}
    coords["unit"] = np.asarray(transform.target_ids, dtype=str)
    result = xr.DataArray(out.reshape(rest_shape + (n_unit,)), dims=tuple(others) + ("unit",), coords=coords,
                          attrs=da.attrs, name=da.name)
    if not support:
        return result
    if present is None:
        distinct = np.unique(np.asarray(da["time"].dt.year.values)) if "time" in da.coords else np.array([0])
        present = np.ones((n_source, len(distinct)), dtype=bool)
    return result, transform.supported(present), distinct


def read_on_units(da, units, *, block_dim, region=None, positions=None, tolerance=None, wrap_longitude=False,
                  convention=None, scale=None, post_scale=None, missing="flag", support=False, dtype=None, readers=None):
    """Read a lazy DataArray (dims named as HCWF's, lat and lon among them) onto ``units``, reading only what feeds them.

    The transform of ``units`` from the array's own cells is compiled (`unit_transform`; the array's lat and lon are
    matched as they are, so give it the cut or uncut array in the order you want matched), only the storage tiles
    holding a source cell of the wanted ``positions`` are read (block by block along ``block_dim``, `~hcwf.spatial.select.load_blocks`), ``scale``
    multiplies the cells in place before they are averaged (as for a cell transform), and `unit_series` averages them
    (``post_scale``, ``missing``, ``support`` and ``dtype`` as there). Only the cells that feed a unit are held in memory, not
    the whole grid.

    With a ``region`` the array is first cut to the region's box (latitude ascending, longitude in the region's
    convention, so use ``wrap_longitude`` with weights written in the other one); no cell mask is applied, the weights
    decide which cells count. Without a region the file's own order and extent are read.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy rain array, with ``lat`` and ``lon`` dims.
    units : hcwf.spatial.units.EvaluationUnits
        The evaluation units.
    block_dim : str
        The dim to read along, block by block.
    region : object, optional
        The region whose box the array is cut to first; None reads the file's own extent.
    positions : array_like of int, optional
        The positions of the units wanted; all when None.
    tolerance : float, optional
        How far (in degrees) a weight's cell may be from a grid cell; None for exact coordinates.
    wrap_longitude : bool, default False
        Put the weights' longitudes in the grid's convention first.
    convention : hcwf.spatial.ids.GridIdConvention, optional
        The grid-id convention of weights keyed by source ids.
    scale : float, optional
        A factor multiplied onto the cells in place before averaging.
    post_scale : float, optional
        A factor multiplied onto the means after averaging.
    missing : {"flag", "keep", "zero", "drop"}, default "flag"
        The missing-rain policy (see `SOURCE_POLICIES`).
    support : bool, default False
        Also return which units have a source each year, and the years (see `unit_series`).
    dtype : numpy.dtype or type, optional
        The dtype of the result where that changes nothing but the storage.
    readers : int, optional
        The most processes the blocks may be decoded by (see `~hcwf.spatial.select.load_blocks`); None reads in this
        process.

    Returns
    -------
    xarray.DataArray or tuple of (xarray.DataArray, numpy.ndarray, numpy.ndarray)
        The unit series, dims (the others, ``unit``); with ``support`` the tuple `unit_series` returns.
    """
    if region is not None:
        da = region_box(da, region=region)
    lat, lon = np.asarray(da["lat"].values), np.asarray(da["lon"].values)
    transform = unit_transform(units, lat, lon, positions, tolerance=tolerance, wrap_longitude=wrap_longitude,
                               convention=convention)
    if scale and da.variable._in_memory:
        da = da.copy(deep=True)  # the caller's array is not scaled in place
    loaded = scale_in_place(load_blocks(da, block_dim, keep=feeding_cells(transform, lat, lon), compact=True,
                                       readers=readers), scale)
    return unit_series(loaded, transform, missing=missing, post_scale=post_scale, support=support, dtype=dtype)


def read_unit_dim(da, *, block_dim, ids=None, scale=None):
    """Read the kept units of a lazy DataArray that already has a ``unit`` dimension (pre-aggregated admin units).

    Only the storage tiles that hold a kept unit are read. The array comes back with its units compacted to the kept
    ones.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy array with a ``unit`` dim.
    block_dim : str
        The dim to read along, block by block.
    ids : sequence of str, optional
        The units to keep, in the order wanted; all, in file order, when None.
    scale : float, optional
        A factor multiplied onto the values in place.

    Returns
    -------
    xarray.DataArray
        The array with its ``unit`` dim reduced to ``ids``, in that order.

    Raises
    ------
    KeyError
        A unit of ``ids`` is not in the data.
    """
    labels = np.asarray(da["unit"].values).astype(str)
    if ids is None:
        wanted = np.arange(labels.size)
    else:
        index = {label: i for i, label in enumerate(labels)}
        missing = [str(i) for i in ids if str(i) not in index]
        if missing:
            raise KeyError(f"units not in the data: {', '.join(missing[:10])}")
        wanted = np.array([index[str(i)] for i in ids], dtype=np.int64)
    keep = np.zeros(labels.size, dtype=bool)
    keep[wanted] = True
    loaded = load_blocks(da, block_dim, keep=keep, keep_dims=("unit",))
    return scale_in_place(loaded.isel(unit=wanted), scale)  # isel with positions copies: nothing of the caller's is scaled


def _units_of(source) -> UnitsSpec | None:
    """Return the evaluation units of a source: the `~hcwf.spatial.units.UnitsSpec` itself, or that of an observation, model or reference spec.

    Parameters
    ----------
    source : hcwf.spatial.units.UnitsSpec or object or None
        A ``UnitsSpec``, an ``ObsSpec``, ``ModelSpec`` or ``ReferenceSpec`` (anything with a ``units`` attribute), or
        None.

    Returns
    -------
    hcwf.spatial.units.UnitsSpec or None
        The specification; None means the grid cells are used.
    """
    return source if source is None or isinstance(source, UnitsSpec) else source.units


def units_spec(source) -> UnitsSpec | None:
    """Return the evaluation units of a source: its resolved `~hcwf.spatial.units.UnitsSpec`, or None when the config has no ``units`` keys.

    With none, the grid cells are used. `~hcwf.spatial.units.UnitsSpec.from_flat` made the specification once, when the run configuration
    was built.

    Parameters
    ----------
    source : object
        The run's ``ObsSpec``, ``ModelSpec`` or ``ReferenceSpec`` (or a ``UnitsSpec``).

    Returns
    -------
    hcwf.spatial.units.UnitsSpec or None
        The resolved specification, or None for the grid cells.
    """
    return _units_of(source)


def units_files(source) -> list:
    """List the identities (path, size, time) of the files the source's units specification names; none without units.

    Parameters
    ----------
    source : object
        The run's ``ObsSpec``, ``ModelSpec`` or ``ReferenceSpec`` (or a ``UnitsSpec``).

    Returns
    -------
    list
        The ``file_identity`` of each file; empty without units.
    """
    spec = _units_of(source)
    return [] if spec is None else [file_identity(p) for p in spec.files()]


def point_dims(source) -> tuple:
    """Return the spatial dims of a run's onset arrays: ``("unit",)`` on evaluation units, else ``("lat", "lon")``.

    Parameters
    ----------
    source : object
        The observations' ``ObsSpec`` (or its ``UnitsSpec``).

    Returns
    -------
    tuple of str
        `hcwf.spatial.axes.UNIT` or `hcwf.spatial.axes.GRID`.
    """
    return axes.UNIT if _units_of(source) is not None else axes.GRID


def unit_table(source, ids) -> EvaluationUnits:
    """Build the run's evaluation units of the string ``ids``, in that order: their names, centroids and, for polygons, geometry.

    Polygon (shapefile) and weights units come from the specification; ids of cells or of a pre-aggregated file are
    labelled with the centroids table, or with the cell they spell (``<lat>_<lon>``).

    Parameters
    ----------
    source : object
        The observations' ``ObsSpec`` (or its ``UnitsSpec``); it must have units.
    ids : array_like of str
        The unit ids wanted.

    Returns
    -------
    hcwf.spatial.units.EvaluationUnits
        The units, in the order of ``ids``.

    Raises
    ------
    ValueError
        An id is not among the run's units.
    """
    spec, ids = _units_of(source), np.asarray(ids).astype(str)
    units = spec.build(ids=None if (spec.shapefile or spec.weights) else ids)
    order = pd.Index(units.ids).get_indexer(ids)
    if (order < 0).any():
        raise ValueError(f"units not among the run's units: {', '.join(ids[order < 0][:10])}")
    table = units.table.iloc[order].reset_index(drop=True)
    for n in np.flatnonzero(table["lat"].isna().to_numpy()):
        parsed = unit_ids.parse_grid_id(table["id"][n])
        if parsed is not None:
            table.loc[n, ["lat", "lon"]] = parsed
    geometry = None if units.geometry is None else [units.geometry[i] for i in order]
    return EvaluationUnits(table, units.kind, geometry=geometry)


def declared_scale(da, unit_cvt) -> float | None:
    """Find the millimetres per unit of the array's rainfall: its declared ``units`` attribute, else ``unit_cvt``.

    Parameters
    ----------
    da : xarray.DataArray
        The rain array; its ``units`` attribute is read when non-empty.
    unit_cvt : float or None
        The configured conversion factor, used when the array declares no units (None means 1).

    Returns
    -------
    float or None
        The factor, None when it is 1 (no scaling needed).
    """
    declared = da.attrs.get("units")
    declared = str(declared).strip() if declared is not None and str(declared).strip() else None
    scale = unit_scale(declared) if declared is not None else (unit_cvt or 1.0)
    return None if scale == 1.0 else float(scale)


def select_units(da, spec: UnitsSpec, keep_ids=None):
    """Select the units for an array with ``lat`` and ``lon``, or a ``unit``, dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The array whose cells or unit ids are matched.
    spec : hcwf.spatial.units.UnitsSpec
        The units specification.
    keep_ids : collections.abc.Iterable of str, optional
        The ids that may be used.

    Returns
    -------
    units : hcwf.spatial.units.EvaluationUnits
        All the units of the source.
    positions : numpy.ndarray of int64, shape (kept,)
        The positions of those used, ordered by id.
    """
    if "unit" in da.dims:
        return spec.select(ids=np.asarray(da["unit"].values), keep_ids=keep_ids)
    return spec.select(lat=da["lat"].values, lon=da["lon"].values, keep_ids=keep_ids)


def read_units(da, spec: UnitsSpec, units, positions, *, block_dim, scale=None, missing="flag", support=False):
    """Read the array's rain on the used units, dims (the others, ``unit``).

    The result is float64 for averaged units and the file's dtype for its own cells or units. A regrid scales after
    averaging, a cell transform before (see `~hcwf.spatial.units.UnitsSpec`); with ``support`` a cell transform also returns which units
    have a source each year and the years.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy rain array, over ``lat`` and ``lon`` or ``unit``.
    spec : hcwf.spatial.units.UnitsSpec
        The units specification (its ``how`` and ``convention`` decide the transform).
    units : hcwf.spatial.units.EvaluationUnits
        All the units of the source.
    positions : numpy.ndarray of int64, shape (kept,)
        The positions of the units used.
    block_dim : str
        The dim to read along, block by block.
    scale : float, optional
        The unit conversion factor (mm per unit).
    missing : {"flag", "keep", "zero", "drop"}, default "flag"
        The missing-rain policy (see `SOURCE_POLICIES`).
    support : bool, default False
        Also return which units have a source each year, and the years (cell transform only).

    Returns
    -------
    xarray.DataArray or tuple of (xarray.DataArray, numpy.ndarray, numpy.ndarray)
        The rain on the used units; with ``support`` the tuple `unit_series` returns.
    """
    if "unit" in da.dims:
        return read_unit_dim(da, block_dim=block_dim, ids=units.ids[positions], scale=scale)
    common = dict(block_dim=block_dim, positions=positions, convention=spec.convention, missing=missing,
                  support=support)
    if spec.how == "regrid":
        return read_on_units(da, units, post_scale=scale, **common)
    if spec.how == "cells":
        return read_on_units(da, units, scale=scale, dtype=da.dtype, **common)
    return read_on_units(da, units, scale=scale, **common)
