"""Rain on units: source cells -> unit series (onset_blending's regrid and cell transform, as one transform).

`unit_series` and `read_on_units` aggregate cells onto units, reading only the storage tiles that feed them; `read_units` also
reads a file whose "unit" dimension is already there. The run's evaluation units come from `unit_table`, `units_spec`, `units_files`.
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

log = logging.getLogger(__name__)

SOURCE_POLICIES = ("flag", "keep", "zero", "drop")
"""Missing-rain policies applied to the source cells before they are averaged onto units: "flag" and "keep" leave
NaN as it is (a unit averages its observed sources), "zero" counts NaN as 0 mm and "drop" removes every source cell
with a missing day in a calendar year from that year's means (see `unit_series`)."""


def unit_transform(units, lat, lon, positions=None, *, tolerance=None, wrap_longitude=False, convention=None):
    """The transform of `units` (hcwf.spatial.EvaluationUnits) from the cells of a (lat, lon) grid, restricted to the
    units at `positions` (all when None).

    The weights are matched to the grid's cells by exact coordinates (as onset_blending matches them) or within
    `tolerance`; wrap_longitude first puts the weights' longitudes in the grid's convention. `convention` is the
    grid-id convention of weights keyed by source ids.
    """
    from hcwf.spatial import SparseTransform

    full = units.transform(lat, lon, tolerance=tolerance, convention=convention, wrap_longitude=wrap_longitude)
    if positions is None:
        return full
    positions = np.asarray(positions, dtype=np.int64)
    out = SparseTransform(full.weighted_matrix[positions], full._source_ids, tuple(np.asarray(full.target_ids)[positions]),
                          keys=full.keys, convention=full.convention, grid=full.grid)
    return out


def _cell_positions(transform, lat, lon):
    """Each cell of the (lat, lon) grid (latitude-major) as its position among the transform's sources, or None when the
    sources are the grid's cells themselves."""
    if transform.keys == "coordinate":
        return None
    cell_lat, cell_lon = np.meshgrid(np.asarray(lat, float), np.asarray(lon, float), indexing="ij")
    return transform.positions(cell_lat.reshape(-1), cell_lon.reshape(-1))


def feeding_cells(transform, lat, lon):
    """Boolean (nlat, nlon): the grid cells that are a source of some unit of the transform (what a read must keep)."""
    sources = np.unique(transform.weighted_matrix.indices)
    place = _cell_positions(transform, lat, lon)
    keep = np.zeros(len(lat) * len(lon), dtype=bool)
    if place is None:
        keep[sources] = True
    else:
        keep = np.isin(place, sources)
    return keep.reshape(len(lat), len(lon))


def unit_series(da, transform, *, missing="flag", post_scale=None, support=False):
    """Average a loaded DataArray with lat and lon dims onto the units of `transform`: dims become (the others, "unit").

    Each unit is the weighted mean of its observed (finite) source cells, in float64 mm, NaN where none is observed
    (SparseTransform.observed_mean). `missing` is applied to the sources first ("zero" counts NaN as 0 mm; "drop" needs
    a "time" dim and removes a cell from the means of every calendar year in which it has a missing day, see
    SOURCE_POLICIES). `post_scale` multiplies the means (a unit factor applied after the averaging, as onset_blending
    does for a regrid; apply it to `da` first for a cell transform). The array is read in column blocks, so working
    memory stays one block beyond the result. With `support` the result is (series, (unit, year) booleans of which
    units have a source that year, the years).
    """
    if missing not in SOURCE_POLICIES:
        raise ValueError(f"missing must be one of {SOURCE_POLICIES}, got {missing!r}")
    lat, lon = np.asarray(da["lat"].values), np.asarray(da["lon"].values)
    others = [d for d in da.dims if d not in ("lat", "lon")]
    if missing == "drop" and others != ["time"]:
        raise ValueError("missing 'drop' needs data with the dims (time, lat, lon)")
    view = np.moveaxis(da.values, [da.dims.index("lat"), da.dims.index("lon")], [-2, -1])
    rest_shape = view.shape[:-2]
    cells = view.reshape(-1, len(lat) * len(lon))
    place = _cell_positions(transform, lat, lon)
    n_unit, n_source = transform.shape
    used = None if place is None else np.flatnonzero(place >= 0)
    out = np.empty((cells.shape[0], n_unit))

    def sources(rows):
        """(source, column) float64 values of the given rows of `cells`."""
        block = cells[rows]
        if place is None:
            return block.T.astype(float)
        values = np.full((n_source, block.shape[0]), np.nan)
        values[place[used]] = block[:, used].T.astype(float)
        return values

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
            values = sources(rows)
            if missing == "zero":
                values = np.where(np.isnan(values), 0.0, values)
            out[rows] = transform.observed_mean(values).T
    if post_scale is not None and post_scale != 1.0:
        out *= post_scale
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
                  convention=None, scale=None, post_scale=None, missing="flag", support=False):
    """Read a lazy DataArray (dims named as HCWF's, lat and lon among them) onto `units`, reading only what feeds them.

    The transform of `units` from the array's own cells is compiled (`unit_transform`; the array's lat and lon are
    matched as they are, so give it the cut or uncut array in the order you want matched), only the storage tiles
    holding a source cell of the wanted `positions` are read (block by block along block_dim, `load_blocks`), `scale`
    multiplies the cells in place before they are averaged (as for a cell transform), and `unit_series` averages them
    (post_scale, missing and support as there). Returns the unit series, dims (the others, "unit").

    With a `region` the array is first cut to the region's box (latitude ascending, longitude in the region's
    convention, so use wrap_longitude with weights written in the other one); no cell mask is applied, the weights
    decide which cells count. Without a region the file's own order and extent are read.
    """
    if region is not None:
        da = region_box(da, region=region)
    lat, lon = np.asarray(da["lat"].values), np.asarray(da["lon"].values)
    transform = unit_transform(units, lat, lon, positions, tolerance=tolerance, wrap_longitude=wrap_longitude,
                               convention=convention)
    if scale and da.variable._in_memory:
        da = da.copy(deep=True)  # the caller's array is not scaled in place
    loaded = scale_in_place(load_blocks(da, block_dim, keep=feeding_cells(transform, lat, lon)), scale)
    return unit_series(loaded, transform, missing=missing, post_scale=post_scale, support=support)


def read_unit_dim(da, *, block_dim, ids=None, scale=None):
    """Read the kept units of a lazy DataArray that already has a "unit" dimension (pre-aggregated admin units).

    `ids` lists the units to keep, in the order wanted (default: all, in file order); only the storage tiles that hold
    one are read. `scale` multiplies in place. Returns the array with its units compacted to the kept ones.
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
    """The evaluation units of a source: the UnitsSpec itself, or that of an ObsSpec, ModelSpec or ReferenceSpec (None:
    the grid cells are used)."""
    return source if source is None or isinstance(source, UnitsSpec) else source.units


def units_spec(source) -> UnitsSpec | None:
    """The evaluation units of a source: its resolved UnitsSpec, or None when the config has no `units:` keys (the grid
    cells are used). `source` is the run's ObsSpec, ModelSpec or ReferenceSpec; `UnitsSpec.from_flat` made it once, when the
    RunConfig was built."""
    return _units_of(source)


def units_files(source) -> list:
    """The identities (path, size, time) of the files the source's units specification names; none without units."""
    spec = _units_of(source)
    return [] if spec is None else [file_identity(p) for p in spec.files()]


def point_dims(source) -> tuple:
    """The spatial dims of a run's onset arrays: ("unit",) on evaluation units, else ("lat", "lon") (see hcwf.spatial.axes).
    `source` is the observations' ObsSpec (or its UnitsSpec)."""
    return axes.UNIT if _units_of(source) is not None else axes.GRID


def unit_table(source, ids) -> EvaluationUnits:
    """The run's evaluation units of the string `ids`, in that order: their names, centroids and, for polygons, geometry.

    `source` is the observations' ObsSpec (or its UnitsSpec). Polygon (shapefile) and weights
    units come from the specification; ids of cells or of a pre-aggregated file are labelled with the centroids table,
    or with the cell they spell ("<lat>_<lon>").
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
    """Millimetres per unit of the array's rainfall: its declared `units` attribute, else unit_cvt; None for 1."""
    declared = da.attrs.get("units")
    declared = str(declared).strip() if declared is not None and str(declared).strip() else None
    scale = unit_scale(declared) if declared is not None else (unit_cvt or 1.0)
    return None if scale == 1.0 else float(scale)


def select_units(da, spec: UnitsSpec, keep_ids=None):
    """(units, positions of those used, ordered by id) for an array with lat and lon, or a "unit", dimension."""
    if "unit" in da.dims:
        return spec.select(ids=np.asarray(da["unit"].values), keep_ids=keep_ids)
    return spec.select(lat=da["lat"].values, lon=da["lon"].values, keep_ids=keep_ids)


def read_units(da, spec: UnitsSpec, units, positions, *, block_dim, scale=None, missing="flag", support=False):
    """The array's rain on the used units, dims (the others, "unit"): float64 for averaged units, the file's dtype for
    its own cells or units. A regrid scales after averaging, a cell transform before (see UnitsSpec); with `support`
    a cell transform also returns which units have a source each year and the years."""
    if "unit" in da.dims:
        return read_unit_dim(da, block_dim=block_dim, ids=units.ids[positions], scale=scale)
    common = dict(block_dim=block_dim, positions=positions, convention=spec.convention, missing=missing,
                  support=support)
    if spec.how == "regrid":
        return read_on_units(da, units, post_scale=scale, **common)
    out = read_on_units(da, units, scale=scale, **common)
    return out.astype(da.dtype, copy=False) if spec.how == "cells" else out
