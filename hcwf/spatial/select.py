"""Region helpers: longitude/latitude orientation fixes, regional subsetting with masks, and domain averaging.

The area of a run is an `AreaSpec` (region, mask, nc_mask); the implicit land and country masks are switched off per
call by a `Selection` (land_only, shp_only; both on when it does not say), as blend's readers do.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from hcwf.onset.regions import box
from hcwf.spatial.masks import apply_default_mask, default_keep

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.config.model import AreaSpec, Selection

log = logging.getLogger(__name__)


def implicit_masks(select: Selection | None) -> tuple[bool, bool]:
    """(land_only, shp_only) of a selection: the implicit land and country masks are on unless it turns them off."""
    land = True if select is None or select.land_only is None else select.land_only
    shp = True if select is None or select.shp_only is None else select.shp_only
    return land, shp


def lon_swap(ds_tag, *, region):
    """Rename coords to lat/lon, then return the region bounds and dataset in a consistent longitude convention.

    Returns
    -------
    tuple
        (lats, latn, lonw, lone, ds_tag); the bounds are conformed to the dataset's longitude range.
    """
    lats, latn, lonw, lone = box(region)

    coord_list = list(ds_tag.coords.keys())

    if "lon" not in coord_list:
        log.debug("lon NOT in coords for model --> ")
        lat_coords = [variable for variable in coord_list if "lat" in variable][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable][0]

        ds_tag = ds_tag.rename({lat_coords: "lat", lon_coords: "lon"})

    if np.min(ds_tag.lon) < 0:
        lonw = ((lonw + 180) % 360) - 180
        lone = ((lone + 180) % 360) - 180

    if lonw > lone:
        # a box across the dataset's longitude seam: the swap that would conform the data was never implemented
        raise ValueError(f"region {region!r}: its longitude box {lonw}..{lone} crosses the longitude seam of the data, "
                         "which is not supported")

    return lats, latn, lonw, lone, ds_tag


def lat_swap(ds_tag):
    """Flip the dataset along lat if latitudes are in descending order."""
    if ds_tag.lat[0] > ds_tag.lat[-1]:
        ds_tag = ds_tag.isel(lat=slice(None, None, -1))

    return ds_tag


def coords_fmt(ds_tag, *, region):
    """Standardize lon/lat orientation and return (lats, latn, lonw, lone, ds_tag) for the region."""
    lats, latn, lonw, lone, ds_tag = lon_swap(ds_tag, region=region)
    ds_tag = lat_swap(ds_tag)

    return lats, latn, lonw, lone, ds_tag



def has_box(ds, region) -> bool:
    """Whether `ds` is cut to a region's box: it has a lat/lon grid (not a "unit" dimension) and a region is named."""
    return region is not None and "unit" not in ds.dims


def region_box(ds, *, region):
    """Subset `ds` (Dataset or DataArray) to the region's lat/lon box, lazily: nothing is read from file here.

    Longitude and latitude are put in HCWF's orientation first (see coords_fmt).     Without a region, or for data on a string "unit" dimension (admin units), there is no box: `ds` is returned
    as it is, in the file's own order.
    """
    if not has_box(ds, region):
        return ds
    lats, latn, lonw, lone, ds = coords_fmt(ds, region=region)
    return ds.sel(lat=slice(lats, latn), lon=slice(lonw, lone))


def _keep_cells(lat, lon, region, land_only, shp_only):
    """Boolean (lat, lon) array of the cells the implicit land and country masks keep (hcwf.spatial.masks.default_keep, cached)."""
    return default_keep(lat, lon, region, land_only, shp_only)


_BLOCK_BYTES = 64 * 1024 * 1024


def _tiles(size: int, stored: int | None):
    """Index ranges (start, stop) of one axis in storage-chunk steps (the whole axis when it is not chunked)."""
    step = stored or size
    return [(a, min(size, a + step)) for a in range(0, size, max(1, step))]


def load_blocks(da, block_dim, keep=None, keep_dims=None):
    """Read a (lazy) DataArray into memory once, filled block by block along block_dim.

    The result is allocated once; each block (about 64 MB, a whole number of stored chunks when the source is
    chunked) is read and copied in, so the reader's temporaries for fancy-indexed reads stay block sized instead of
    the size of the whole selection. An array already in memory is returned as it is.

    keep is an optional boolean array over the spatial dims keep_dims (default: ("lat", "lon") for a 2-D mask, the
    one spatial dim ("unit", else "lat") for a 1-D mask). Only the storage tiles of those dims that hold a kept
    cell are read, a block at a time; every cell outside keep is NaN in the result (onset_blending reads global and
    admin-unit files this way). The array must be floating point; the result has the shape of da.
    """
    if keep is not None and not da.variable._in_memory and block_dim in da.dims and da.ndim:
        return _load_kept(da, block_dim, np.asarray(keep, dtype=bool), keep_dims)
    if da.variable._in_memory or block_dim not in da.dims or da.ndim == 0:
        return da.load()
    axis = da.dims.index(block_dim)
    chunks = da.encoding.get("chunks")
    chunks = chunks if chunks and len(chunks) == da.ndim else None
    row_bytes = max(1, da.dtype.itemsize * int(np.prod(da.shape[:axis] + da.shape[axis + 1:], dtype=np.int64)))
    rows = max(1, _BLOCK_BYTES // row_bytes)
    if chunks and rows < chunks[axis] and da.shape[axis] > rows:
        # a block along block_dim would be part of one stored chunk, which every block would decode again (an
        # archive chunked over the whole time axis): block along the axis with the most blocks instead
        others = [n for n in range(da.ndim) if n != axis]
        axis = max(others, key=lambda n: da.shape[n] / chunks[n])
        row_bytes = max(1, da.dtype.itemsize * int(np.prod(da.shape[:axis] + da.shape[axis + 1:], dtype=np.int64)))
        rows = max(1, _BLOCK_BYTES // row_bytes)
    stored = chunks[axis] if chunks else None
    if stored:
        rows = max(1, rows // stored) * stored
    out = np.empty(da.shape, dtype=da.dtype)
    for start in range(0, da.shape[axis], rows):
        index = [slice(None)] * da.ndim
        index[axis] = slice(start, start + rows)
        out[tuple(index)] = da.isel({da.dims[axis]: index[axis]}).values
    return xr.DataArray(out, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name)


def _load_kept(da, block_dim, keep, keep_dims):
    """load_blocks for a keep mask: read the (block, spatial tile) pieces that hold a kept cell, NaN elsewhere."""
    if not np.issubdtype(da.dtype, np.floating):
        raise ValueError(f"a keep mask needs floating point data, got {da.dtype}")
    if keep_dims is None:
        keep_dims = ("lat", "lon") if keep.ndim == 2 else (next(d for d in ("unit", "lat") if d in da.dims),)
    axes = [da.dims.index(d) for d in keep_dims]
    if keep.shape != tuple(da.shape[a] for a in axes):
        raise ValueError(f"keep has shape {keep.shape} for the dims {keep_dims} of size "
                         f"{tuple(da.shape[a] for a in axes)}")
    chunks = da.encoding.get("chunks") or da.encoding.get("chunksizes")  # zarr, or NetCDF4/HDF5 storage
    chunks = chunks if chunks and len(chunks) == da.ndim else None
    axis = da.dims.index(block_dim)
    if axis in axes:
        raise ValueError("block_dim cannot be one of the dims of keep")
    per_row = da.dtype.itemsize * int(np.prod([n for i, n in enumerate(da.shape) if i != axis], dtype=np.int64))
    rows = max(1, _BLOCK_BYTES // max(1, per_row))
    stored = chunks[axis] if chunks else None
    if stored:
        rows = max(1, rows // stored) * stored
    out = np.full(da.shape, np.nan, dtype=da.dtype)
    ranges = [_tiles(da.shape[a], chunks[a] if chunks else None) for a in axes]
    for corner in np.ndindex(*[len(r) for r in ranges]):
        bounds = [r[c] for r, c in zip(ranges, corner)]
        inside = keep[tuple(slice(a, b) for a, b in bounds)]
        if not inside.any():
            continue
        for start in range(0, da.shape[axis], rows):
            index = [slice(None)] * da.ndim
            index[axis] = slice(start, start + rows)
            for a, (lo, hi) in zip(axes, bounds):
                index[a] = slice(lo, hi)
            block = da.isel({d: index[i] for i, d in enumerate(da.dims) if index[i] != slice(None)}).values
            if not inside.all():
                moved = np.moveaxis(block, axes, list(range(-len(axes), 0)))
                moved[..., ~inside] = np.nan
            out[tuple(index)] = block
    return xr.DataArray(out, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name)


def in_memory(da):
    """The DataArray with its data in memory (a no-op for one already read)."""
    return da if da.variable._in_memory else da.load()


def scale_in_place(da, factor):
    """Multiply the loaded array by factor in its own memory (float32 stays float32), if there is a factor."""
    if factor:
        data = da.values
        np.multiply(data, factor, out=data)
    return da


def read_selection(obj, block_dim, keep=None):
    """Read a box-selected Dataset or DataArray into memory (see load_blocks); Dataset variables are read one by one."""
    if isinstance(obj, xr.DataArray):
        return load_blocks(obj, block_dim, keep)
    return obj.assign({name: load_blocks(var, block_dim, keep) for name, var in obj.data_vars.items()})


def _mask_in_place(da, keep):
    """Set the cells outside `keep` (a (lat, lon) boolean array) to NaN in the array's own memory; returns da."""
    if not np.issubdtype(da.dtype, np.floating):
        return da.where(xr.DataArray(keep, dims=("lat", "lon"), coords={"lat": da["lat"], "lon": da["lon"]}))
    data = da.values
    view = np.moveaxis(data, (da.dims.index("lat"), da.dims.index("lon")), (-2, -1))
    view[..., ~keep] = np.nan
    return da


def mask_region_loaded(obj, *, area: AreaSpec, select: Selection | None = None, dataset_name=None, dir_fig=None):
    """Limit an already loaded, box-selected Dataset or DataArray to the cells of the run's mask, others set to NaN.

    The same cells as region_select keeps. With the implicit land and country masks the NaNs are written into the
    array's own memory (one pass, no copy); a `mask` section or an nc_mask goes through hcwf.spatial.masks or
    apply_nc_mask as in region_select, which build a masked copy.
    """
    region = area.region
    if not has_box(obj, region):
        return obj
    land_only, shp_only = implicit_masks(select)
    if area.mask or area.nc_mask:
        return region_select_masks(obj, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
    if not (land_only or shp_only):
        return obj
    keep = _keep_cells(obj["lat"].values, obj["lon"].values, region, land_only, shp_only)
    if keep.all():
        return obj
    if isinstance(obj, xr.DataArray):
        return _mask_in_place(obj, keep)
    for name, var in obj.data_vars.items():
        if "lat" in var.dims and "lon" in var.dims:
            obj[name] = _mask_in_place(var, keep)
    return obj


def region_select_masks(ds_reg, *, area: AreaSpec, select: Selection | None = None, dataset_name=None, dir_fig=None):
    """Apply the cell masks of region_select to a box-selected dataset: `mask`, else nc_mask, land and country masks.

    Data without a box (no region, or admin units on a "unit" dimension) has no cells to mask.
    """
    if not has_box(ds_reg, area.region):
        return ds_reg
    if area.mask:
        if area.nc_mask:
            from hcwf.errors import MaskError

            raise MaskError("both `mask` and `nc_mask` are set; use one. nc_mask: <file> is the same as "
                            "mask: {file: <file>}, except that the old nc_mask also applies the land and country "
                            "masks implicitly")
        from hcwf.spatial.masks import apply_region_mask

        return apply_region_mask(ds_reg, area.mask, dataset_name or "data", dir_fig)

    land_only, shp_only = implicit_masks(select)
    return apply_default_mask(ds_reg, area.region, land_only, shp_only, area.nc_mask)


def region_select(ds, *, area: AreaSpec, select: Selection | None = None, dataset_name=None, dir_fig=None,
                  read_by=None, keep=None):
    """Subset `ds` to the region's lat/lon box, then limit it to the cells of the run's mask.

    With a mask in `area` (the config's ``mask:`` section) the mask is built on this dataset's own grid by
    hcwf.spatial.masks (dataset_name labels it in messages, records and the PNG saved to dir_fig); it replaces
    the implicit land and country masks below and cannot be combined with nc_mask. Without a mask the old
    behaviour is unchanged: the optional NetCDF mask `area.nc_mask`, then the land mask and the country-shapefile
    mask (each off when `select` says land_only / shp_only is False).

    Without read_by the result stays lazy and the masks are applied with where. With read_by (a dim name) the
    box is read into memory once, block by block along that dim, and the implicit masks are then written in place
    (see mask_region_loaded): slice inits, leads and members before calling so only those are read. `keep` (a
    boolean array over the box's cells, see load_blocks) then reads only the storage tiles holding a kept cell; the
    other cells are NaN.

    With no region, or on a "unit" dimension, there is no box and no cell mask: the data is read as it is.
    """
    ds_reg = region_box(ds, region=area.region)
    if read_by:
        ds_reg = read_selection(ds_reg, read_by, keep)
        return mask_region_loaded(ds_reg, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
    return region_select_masks(ds_reg, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
