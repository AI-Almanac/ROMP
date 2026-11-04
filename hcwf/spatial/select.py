"""Cut data to the region of a run: orientation fixes, the lat/lon box, the cell masks and the block-wise read.

The area of a run is an `~hcwf.config.model.AreaSpec` (``region``, ``mask``, ``nc_mask``); the implicit land and country
masks are switched off per call by a `~hcwf.config.model.Selection` (``land_only``, ``shp_only``; both on when it does
not say), as blend's readers do. `region_select` is the entry point: it takes the box (`region_box`), reads it into
memory block by block (`load_blocks`, which reads only the storage tiles that hold a kept cell) and applies the masks
that `hcwf.spatial.masks` builds. The region boxes themselves come from `hcwf.onset.regions`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

import numpy as np
import xarray as xr

from hcwf.onset.regions import box
from hcwf.spatial.masks import apply_default_mask, default_keep

if TYPE_CHECKING:  # pragma: no cover
    from hcwf.config.model import AreaSpec, Selection

log = logging.getLogger(__name__)


def implicit_masks(select: Selection | None) -> tuple[bool, bool]:
    """Return whether the implicit land and country masks are on for a selection.

    Both are on unless the selection turns them off.

    Parameters
    ----------
    select : hcwf.config.model.Selection or None
        The selection; None, or a field left unset, means on.

    Returns
    -------
    land_only : bool
        Whether the land mask applies.
    shp_only : bool
        Whether the country-shapefile mask applies.
    """
    land = True if select is None or select.land_only is None else select.land_only
    shp = True if select is None or select.shp_only is None else select.shp_only
    return land, shp


def lon_swap(ds_tag, *, region):
    """Rename coords to lat/lon, then return the region bounds and dataset in a consistent longitude convention.

    Parameters
    ----------
    ds_tag : xarray.Dataset or xarray.DataArray
        The data; its latitude and longitude coordinates (any name containing ``lat`` / ``lon``) are renamed to
        ``lat`` and ``lon`` when ``lon`` is not a coordinate.
    region : str
        The region name (or alias) whose box is wanted.

    Returns
    -------
    lats : float
        The southern bound in degrees.
    latn : float
        The northern bound in degrees.
    lonw : float
        The western bound in degrees, conformed to the dataset's longitude range (-180..180 when the data has
        negative longitudes).
    lone : float
        The eastern bound in degrees, conformed like ``lonw``.
    ds_tag : xarray.Dataset or xarray.DataArray
        The data with ``lat`` and ``lon`` coordinates.

    Raises
    ------
    ValueError
        The longitude box crosses the longitude seam of the data, which is not supported.
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
    """Flip the dataset along ``lat`` if latitudes are in descending order.

    Parameters
    ----------
    ds_tag : xarray.Dataset or xarray.DataArray
        The data, with a ``lat`` coordinate.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The data with ascending latitudes (the input when they already were).
    """
    if ds_tag.lat[0] > ds_tag.lat[-1]:
        ds_tag = ds_tag.isel(lat=slice(None, None, -1))

    return ds_tag


def coords_fmt(ds_tag, *, region):
    """Standardize lon/lat orientation and return the region's bounds and the data.

    A longitude box that crosses the longitude seam of the data is not supported (`lon_swap` raises `ValueError`).

    Parameters
    ----------
    ds_tag : xarray.Dataset or xarray.DataArray
        The data.
    region : str
        The region name (or alias).

    Returns
    -------
    lats : float
        The southern bound in degrees.
    latn : float
        The northern bound in degrees.
    lonw : float
        The western bound in degrees, in the data's longitude convention.
    lone : float
        The eastern bound in degrees, in the data's longitude convention.
    ds_tag : xarray.Dataset or xarray.DataArray
        The data with ``lat`` and ``lon`` coordinates and ascending latitudes (see `lon_swap`, `lat_swap`).
    """
    lats, latn, lonw, lone, ds_tag = lon_swap(ds_tag, region=region)
    ds_tag = lat_swap(ds_tag)

    return lats, latn, lonw, lone, ds_tag


def has_box(ds, region) -> bool:
    """Tell whether ``ds`` is cut to a region's box.

    That is the case when it has a lat/lon grid (not a ``"unit"`` dimension) and a region is named.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data.
    region : str or None
        The region name, or None for none.

    Returns
    -------
    bool
        True when the data is cut to the region's box.
    """
    return region is not None and "unit" not in ds.dims


def region_box(ds, *, region):
    """Subset ``ds`` to the region's lat/lon box, lazily: nothing is read from file here.

    Longitude and latitude are put in HCWF's orientation first (see `coords_fmt`). Without a region, or for data on a
    string ``"unit"`` dimension (admin units), there is no box: ``ds`` is returned as it is, in the file's own order.
    A longitude box that crosses the longitude seam of the data is not supported (`lon_swap` raises `ValueError`).

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data.
    region : str or None
        The region name (or alias), or None for no box.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The data inside the box, still lazy, with ascending ``lat``.
    """
    if not has_box(ds, region):
        return ds
    lats, latn, lonw, lone, ds = coords_fmt(ds, region=region)
    return ds.sel(lat=slice(lats, latn), lon=slice(lonw, lone))


def _keep_cells(lat, lon, region, land_only, shp_only):
    """Return the cells the implicit land and country masks keep (`hcwf.spatial.masks.default_keep`, cached).

    Parameters
    ----------
    lat : numpy.ndarray of float64, shape (lat,)
        The latitudes of the grid, in degrees.
    lon : numpy.ndarray of float64, shape (lon,)
        The longitudes of the grid, in degrees.
    region : str
        The region name (or alias).
    land_only : bool
        Whether the land mask applies.
    shp_only : bool
        Whether the country-shapefile mask applies.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon)
        True for the cells kept.
    """
    return default_keep(lat, lon, region, land_only, shp_only)


_BLOCK_BYTES = 64 * 1024 * 1024
"""The size in bytes of a block `load_blocks` reads at a time (64 MB)."""


def _tiles(size: int, stored: int | None):
    """Return the index ranges of one axis in storage-chunk steps.

    Parameters
    ----------
    size : int
        The length of the axis.
    stored : int or None
        The length of the storage chunks along the axis; None when it is not chunked (one range of the whole axis).

    Returns
    -------
    list of tuple of (int, int)
        The ``(start, stop)`` of each tile; the last one may be shorter.
    """
    step = stored or size
    return [(a, min(size, a + step)) for a in range(0, size, max(1, step))]


def _lazy_key(da):
    """Return the outer index a lazy selection applies to its file's variable, with that variable's shape.

    Parameters
    ----------
    da : xarray.DataArray
        The array, lazy or not.

    Returns
    -------
    tuple or None
        ``(key, shape)``: ``key`` has one slice or integer array per dim of the file's variable (which is the dims of ``da``
        when the selection drops none) and ``shape`` is that variable's shape. None when ``da`` is in memory or its
        selection is not an outer index xarray keeps (the callers then cut by position).
    """
    try:
        from xarray.core import indexing

        data = da.variable._data
        while not isinstance(data, indexing.LazilyIndexedArray):
            if tuple(data.shape) != tuple(da.shape):
                return None
            data = data.array
        key = data.key
        if not isinstance(key, (indexing.OuterIndexer, indexing.BasicIndexer)) or len(key.tuple) != da.ndim:
            return None
        shape = tuple(data.array.shape)
        key = tuple(key.tuple)
        for entry, n, size in zip(key, shape, da.shape):
            size_of = len(range(n)[entry]) if isinstance(entry, slice) else np.ndim(entry) == 1 and len(entry)
            if size_of != size:
                return None
        return key, shape
    except Exception:  # noqa: BLE001 - an internal layout this does not know is cut by position
        return None


def _chunk_runs(da, axis: int, stored: int | None):
    """Return the runs of positions along an axis that lie in one stored chunk of the file.

    The positions of ``da`` are the file's rows through the lazy selection (a region box starts anywhere in the file, a
    flipped latitude runs backwards), so the stored chunks are cut where the file rows cross a chunk boundary, not at
    multiples of the chunk length from the first position: a piece made of whole runs reads each chunk once.

    Parameters
    ----------
    da : xarray.DataArray
        The array.
    axis : int
        The axis.
    stored : int or None
        The length of the file's chunks along the axis; None when it is not chunked (one run of the whole axis).

    Returns
    -------
    list of tuple of (int, int)
        The ``(start, stop)`` of each run in positions of ``da``. Cut at multiples of ``stored`` from the first position
        (see `_tiles`) when the file's rows along the axis are not strictly monotonic or not known.
    """
    size = da.shape[axis]
    if not stored:
        return _tiles(size, None)
    found = _lazy_key(da)
    if found is not None and size:
        key, shape = found
        entry = key[axis]
        rows = np.arange(shape[axis])[entry] if isinstance(entry, slice) else np.asarray(entry)
        step = np.diff(rows)
        if len(step) == 0 or np.all(step > 0) or np.all(step < 0):
            edges = np.r_[0, np.flatnonzero(np.diff(rows // stored)) + 1, size].tolist()
            return list(zip(edges[:-1], edges[1:]))
    return _tiles(size, stored)


def _picked_rows(da, axis=None):
    """Unpick a lazy NetCDF4 selection that picks an increasing list of rows on one axis.

    The axis is the one whose entry of the lazy selection is the list (the inits of a year out of a file's whole time
    axis). The result is None for anything else (in memory, zarr, unchunked storage, a selection xarray cannot be
    unpicked, rows not increasing): the caller then reads with xarray's own indexing.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy selection.
    axis : int, optional
        The axis the list must be on; None accepts the one axis that has a list.

    Returns
    -------
    read : callable
        ``read(key)`` reads the same variable, with the same decoding, through another outer index ``key``.
    key : tuple
        The outer index of the lazy selection: one entry per dim of ``da``, a slice or an integer array.
    stored : int
        The length of the file's chunks along the axis.
    shape : tuple of int
        The shape of the file's variable.
    axis : int
        The axis of the list.
    None
        Returned alone (not as a tuple) when ``da`` is not such a selection.
    """
    try:
        import copy

        from xarray.core import indexing

        chunks = da.encoding.get("chunksizes")
        wrappers, data = [], da.variable._data
        while not isinstance(data, indexing.LazilyIndexedArray):
            if tuple(data.shape) != tuple(da.shape):
                return None
            wrappers.append(data)
            data = data.array
        root = data.array
        while hasattr(root, "array"):
            root = root.array
        key = data.key
        if type(root).__name__ != "NetCDF4ArrayWrapper" or not isinstance(key, indexing.OuterIndexer):
            return None
        key = key.tuple
        if not chunks or len(chunks) != da.ndim or len(key) != da.ndim:
            return None
        picked = [a for a, entry in enumerate(key)
                  if isinstance(entry, np.ndarray) and entry.ndim == 1 and entry.dtype.kind in "iu" and len(entry)
                  and entry[0] >= 0 and np.all(np.diff(entry) > 0)]
        if len(picked) != 1 or (axis is not None and picked[0] != axis):
            return None
        axis = picked[0]
        inner = [w for w in wrappers if not isinstance(w, (indexing.MemoryCachedArray, indexing.CopyOnWriteArray))]

        def read(new_key):
            """Read the variable through the outer index `new_key`, decoded like the original selection."""
            current = indexing.LazilyIndexedArray(data.array, indexing.OuterIndexer(new_key))
            for wrapper in reversed(inner):   # the file's decoding (fill values, scaling) over the new selection
                clone = copy.copy(wrapper)
                clone.array, current = current, clone
            return np.asarray(current)

        return read, key, int(chunks[axis]), tuple(data.array.shape), axis
    except Exception:  # noqa: BLE001 - an internal layout this does not know is read the ordinary way
        return None


def _along(entry, extent, part):
    """Narrow the index of one axis by a slice, as an index HDF5 reads in file order.

    Parameters
    ----------
    entry : slice or numpy.ndarray of int
        The index of the axis (a slice or an integer array, over an axis of length ``extent``).
    extent : int
        The length of the axis.
    part : slice
        The slice to take of what ``entry`` selects.

    Returns
    -------
    index : slice or numpy.ndarray of int
        The narrowed index; ``entry`` itself when ``part`` takes everything. A slice that runs backwards (a flipped
        latitude) is returned as the forward slice of the same rows.
    flip : bool
        Whether the rows read through ``index`` are to be reversed to get the narrowed selection.
    """
    if part == slice(None):
        picked = range(extent)[entry] if isinstance(entry, slice) else None
        if picked is None or picked.step > 0:
            return entry, False
    elif not isinstance(entry, slice):
        return entry[part], False
    else:
        picked = range(extent)[entry][part]
    if picked.step > 0:
        return slice(picked.start, picked.stop, picked.step), False
    forward = picked[::-1]
    return slice(forward.start, forward.stop, forward.step), True


def _read_slices(da, index, axis):
    """Read ``da[index]`` as an array, from the file the way HDF5 can decode it once.

    The rows a list picks along one axis (the inits of a year out of a file's whole time axis) read as a list make HDF5
    decode a stored chunk again for every run of the list, which costs several times the read when the chunks are larger
    than its cache. Here the rows are read one stored chunk at a time as a contiguous range and the wanted ones taken in
    memory: the same values, one chunk in memory more.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy array.
    index : sequence of slice
        One entry per dim of ``da``: a slice (``slice(None)`` for the whole dim).
    axis : int
        The axis whose selection is blocked by the caller (the rows are picked along the list's own axis, whichever
        it is).

    Returns
    -------
    numpy.ndarray
        The values, with the dtype ``da`` decodes to.
    """
    plan = _picked_rows(da)
    if plan is None:
        return da.isel({d: i for d, i in zip(da.dims, index) if i != slice(None)}).values
    read, key, stored, shape, axis = plan   # the rows of a run are picked along the list's own axis, whichever is blocked
    keys, flips = [], []
    for a, entry in enumerate(key):
        if a == axis:
            keys.append(None)
            continue
        narrowed, flip = _along(entry, shape[a], index[a])
        keys.append(narrowed)
        if flip:
            flips.append(a)
    rows = key[axis][index[axis]]
    out = None
    chunk_of = rows // max(1, stored)
    for first in np.flatnonzero(np.r_[True, np.diff(chunk_of) != 0]):
        stop = first + int(np.searchsorted(chunk_of[first:], chunk_of[first], side="right"))
        group = rows[first:stop]
        keys[axis] = slice(int(group[0]), int(group[-1]) + 1)
        piece = read(tuple(keys))
        if len(group) != piece.shape[axis]:
            piece = np.take(piece, group - group[0], axis=axis)
        if out is None:
            out = np.empty(piece.shape[:axis] + (len(rows),) + piece.shape[axis + 1:], dtype=piece.dtype)
        place = [slice(None)] * out.ndim
        place[axis] = slice(first, stop)
        out[tuple(place)] = piece
    for a in flips:   # the reversed axes were read in file order
        out = np.flip(out, axis=a)
    return out


_PARALLEL_READ_BYTES = 256 * 1024 * 1024
"""A read smaller than this is made in this process: starting reader processes would cost more than they save."""
_READ_BYTES_EACH = 32 * 1024 * 1024
"""The selection each reader process is given at least (a read of 740 MB uses 20 of them)."""


def _row_blocks(da, axis, rows):
    """Split an axis into blocks of about ``rows`` rows.

    When the file is chunked a block ends where a stored chunk's rows end (see `_chunk_runs`: the rows are the file's
    through the selection, so a region box that starts inside a chunk or a flipped latitude is cut at the chunk edges
    too), so a chunk is decoded for one block only.

    Parameters
    ----------
    da : xarray.DataArray
        The array.
    axis : int
        The axis to split.
    rows : int
        The wanted number of rows of a block.

    Returns
    -------
    list of tuple of (int, int)
        The ``(start, stop)`` of each block.
    """
    n = da.shape[axis]
    chunks = da.encoding.get("chunks") or da.encoding.get("chunksizes")
    stored = chunks[axis] if chunks and len(chunks) == da.ndim else None
    if not stored:
        return [(a, min(n, a + rows)) for a in range(0, n, rows)]
    blocks = []
    for first, end in _chunk_runs(da, axis, stored):   # join neighbouring chunks' rows while the block stays within `rows`
        if blocks and end - blocks[-1][0] <= rows:
            blocks[-1] = (blocks[-1][0], end)
        else:
            blocks.append((first, end))
    return blocks


def _block_index(ndim, axis, start, stop):
    """Return the index of rows ``start..stop`` along an axis and all of the others.

    Parameters
    ----------
    ndim : int
        The number of dims.
    axis : int
        The axis that is cut.
    start : int
        The first row.
    stop : int
        One past the last row.

    Returns
    -------
    list of slice
        One slice per dim.
    """
    index = [slice(None)] * ndim
    index[axis] = slice(start, stop)
    return index


def _read_task(da, index, axis):
    """Read one block of a parallel read, in a reader process: what `_read_slices` gives.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy array.
    index : sequence of slice
        One slice per dim of ``da``.
    axis : int
        The axis of the list, as for `_read_slices`.

    Returns
    -------
    numpy.ndarray
        The values of the block.
    """
    return _read_slices(da, index, axis)


def _read_workers(da, readers) -> int:
    """Return how many reader processes a read of ``da`` is split over.

    It is 1 (this process) unless the selection is big enough and an increasing list of rows is picked (see
    `_picked_rows`).

    Parameters
    ----------
    da : xarray.DataArray
        The lazy array.
    readers : int or None
        The most processes allowed; None means one.

    Returns
    -------
    int
        The number of processes, at least 1.
    """
    nbytes = int(da.size) * da.dtype.itemsize
    workers = min(int(readers or 1), nbytes // _READ_BYTES_EACH)
    return workers if workers >= 2 and nbytes >= _PARALLEL_READ_BYTES and _picked_rows(da) is not None else 1


def _read_ahead(da, indexes, axis, readers) -> Iterator[np.ndarray]:
    """Yield the blocks `_read_slices` gives for each index of ``indexes``, in order.

    With ``readers`` (the most processes allowed, see `hcwf.parallel.reader_count`) above one and a selection big enough,
    the blocks are decoded by reader processes (a NetCDF/HDF5 file is decoded by one thread at a time), a few ahead of
    the one being placed, so memory stays at those blocks; otherwise one at a time in this process. When the reader
    processes cannot start, the rest is read in this process.

    Parameters
    ----------
    da : xarray.DataArray
        The lazy array.
    indexes : list of sequence of slice
        The block indexes, one slice per dim of ``da`` each.
    axis : int
        The axis of the list, as for `_read_slices`.
    readers : int or None
        The most processes allowed; None reads in this process.

    Yields
    ------
    numpy.ndarray
        The values of the next block.
    """
    workers = min(_read_workers(da, readers), len(indexes))
    if workers < 2:
        for index in indexes:
            yield _read_slices(da, index, axis)
        return
    from collections import deque
    from concurrent.futures.process import BrokenProcessPool

    from hcwf import parallel

    done = 0
    try:
        pool = parallel.reader_pool(workers)
        waiting, ahead = deque(), 2 * workers
        queue = iter(indexes)
        try:
            for index in queue:
                waiting.append(pool.submit(_read_task, da, index, axis))
                if len(waiting) >= ahead:
                    break
            while waiting:
                block = waiting.popleft().result()
                for index in queue:
                    waiting.append(pool.submit(_read_task, da, index, axis))
                    break
                yield block
                done += 1
                del block
        finally:
            for future in waiting:
                future.cancel()
    except BrokenProcessPool:
        # the reader processes could not start (a script that runs its work on import, without `if __name__ == "__main__":`,
        # cannot spawn them): read the rest here
        parallel.shutdown_readers()
        log.warning("reader processes could not start (does the script guard its entry point with "
                    "`if __name__ == \"__main__\":`?); reading in this process")
        for index in indexes[done:]:
            yield _read_slices(da, index, axis)


def load_blocks(da, block_dim, keep=None, keep_dims=None, *, compact=False, readers=None):
    """Read a (lazy) DataArray into memory once, filled block by block along ``block_dim``.

    The result is allocated once; each block (about 64 MB, a whole number of stored chunks when the source is
    chunked) is read and copied in, so the reader's temporaries for fancy-indexed reads stay block sized instead of
    the size of the whole selection. An array already in memory is returned as it is.

    ``keep`` is an optional boolean array over the spatial dims ``keep_dims``. Only the storage tiles of those dims that
    hold a kept cell are read, a block at a time; every cell outside ``keep`` is NaN in the result (onset_blending reads
    global and admin-unit files this way). The array must be floating point; the result has the shape of ``da``.

    With ``compact`` (a 2-D ``keep``, an array not yet in memory) only the kept cells are allocated: the result has the
    dims of ``da`` without the two spatial ones, then ``"cell"`` (the kept cells in latitude-major order), with the
    cells' ``lat`` and ``lon`` as coordinates over ``"cell"`` and ``"cell"`` itself the cell's position in the whole
    grid.

    A `ValueError` is raised when ``keep`` is given and the array is not floating point, when ``keep`` does not match
    ``keep_dims``, when ``block_dim`` is one of ``keep_dims``, or when ``compact`` is asked with a ``keep`` that is not 2-D.

    Parameters
    ----------
    da : xarray.DataArray
        The array to read, lazy or in memory.
    block_dim : str
        The dim the blocks are cut along.
    keep : numpy.ndarray of bool, optional
        The cells to read, over ``keep_dims``; None reads everything.
    keep_dims : tuple of str, optional
        The dims ``keep`` is over; None means ``("lat", "lon")`` for a 2-D mask and the one spatial dim (``"unit"``,
        else ``"lat"``) for a 1-D mask.
    compact : bool, default False
        Allocate only the kept cells (see above).
    readers : int, optional
        The most processes the blocks may be decoded by (see `_read_ahead`); None reads in this process.

    Returns
    -------
    xarray.DataArray
        The loaded array, with the shape of ``da`` (or the compact shape above).
    """
    if keep is not None and not da.variable._in_memory and block_dim in da.dims and da.ndim:
        return _load_kept(da, block_dim, np.asarray(keep, dtype=bool), keep_dims, compact, readers)
    if da.variable._in_memory or block_dim not in da.dims or da.ndim == 0:
        return da.load()
    axis = da.dims.index(block_dim)
    chunks = da.encoding.get("chunks") or da.encoding.get("chunksizes")   # zarr, or NetCDF4/HDF5 storage
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
    indexes = [_block_index(da.ndim, axis, start, stop) for start, stop in _row_blocks(da, axis, rows)]
    for index, block in zip(indexes, _read_ahead(da, indexes, axis, readers)):
        out[tuple(index)] = block
    return xr.DataArray(out, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name)


def _load_kept(da, block_dim, keep, keep_dims, compact=False, readers=None):
    """Read a keep mask's cells for `load_blocks`: the (block, spatial tile) pieces that hold a kept cell.

    Every other cell is NaN (or, with ``compact``, not allocated; see `load_blocks`).

    Parameters
    ----------
    da : xarray.DataArray
        The lazy floating-point array.
    block_dim : str
        The dim the blocks are cut along.
    keep : numpy.ndarray of bool
        The cells to read, over ``keep_dims``.
    keep_dims : tuple of str or None
        The dims ``keep`` is over; None picks them as `load_blocks` does.
    compact : bool, default False
        Allocate only the kept cells.
    readers : int, optional
        The most processes the blocks may be decoded by; None reads in this process.

    Returns
    -------
    xarray.DataArray
        The loaded array, as `load_blocks` returns it.

    Raises
    ------
    ValueError
        The array is not floating point, ``keep`` does not match ``keep_dims``, ``block_dim`` is one of ``keep_dims``, or
        ``compact`` is asked with a ``keep`` that is not 2-D.
    """
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
    if compact:
        if len(axes) != 2:
            raise ValueError("a compact read needs a 2-D keep")
        others = [i for i in range(da.ndim) if i not in axes]
        out = np.empty(tuple(da.shape[i] for i in others) + (int(keep.sum()),), dtype=da.dtype)
        rank = (np.cumsum(keep.reshape(-1)) - 1).reshape(keep.shape)   # a kept cell's place among the kept cells
    else:
        out = np.full(da.shape, np.nan, dtype=da.dtype)
    ranges = [_chunk_runs(da, a, chunks[a] if chunks else None) for a in axes]
    split, cut = None, [(None, None)]
    if _read_workers(da, readers) > 1 and chunks:
        # in reader processes the pieces are one stored chunk's rows of one tile, and cut along the axis with the most
        # stored chunks too, so that there are enough of them to keep every reader busy
        rows = 1
        free = [i for i in range(da.ndim) if i != axis and i not in axes]
        if free:
            split = max(free, key=lambda i: -(-da.shape[i] // chunks[i]))
            cut = _chunk_runs(da, split, chunks[split])
    jobs = []   # (tile bounds, kept cells of the tile, index) of every piece to read
    for corner in np.ndindex(*[len(r) for r in ranges]):
        bounds = [r[c] for r, c in zip(ranges, corner)]
        inside = keep[tuple(slice(a, b) for a, b in bounds)]
        if not inside.any():
            continue
        for start, stop in _row_blocks(da, axis, rows):
            for lo, hi in cut:
                index = _block_index(da.ndim, axis, start, stop)
                for a, (first, last) in zip(axes, bounds):
                    index[a] = slice(first, last)
                if split is not None:
                    index[split] = slice(lo, hi)
                jobs.append((bounds, inside, index))
    for (bounds, inside, index), block in zip(jobs, _read_ahead(da, [job[2] for job in jobs], axis, readers)):
        if compact:
            moved = np.moveaxis(block, axes, [-2, -1])
            place = rank[tuple(slice(a, b) for a, b in bounds)][inside]
            out[tuple(index[i] for i in others) + (place,)] = moved[..., inside]
            continue
        if not inside.all():
            moved = np.moveaxis(block, axes, list(range(-len(axes), 0)))
            moved[..., ~inside] = np.nan
        out[tuple(index)] = block
    if compact:
        rows_at, cols_at = np.nonzero(keep)
        lat_dim, lon_dim = keep_dims
        coords = {k: v for k, v in da.coords.items() if not set(v.dims) & set(keep_dims)}
        coords.update({lat_dim: ("cell", np.asarray(da[lat_dim].values)[rows_at]),
                       lon_dim: ("cell", np.asarray(da[lon_dim].values)[cols_at]),
                       "cell": rows_at * keep.shape[1] + cols_at})
        return xr.DataArray(out, dims=tuple(da.dims[i] for i in others) + ("cell",), coords=coords, attrs=da.attrs,
                            name=da.name)
    return xr.DataArray(out, dims=da.dims, coords=da.coords, attrs=da.attrs, name=da.name)


def in_memory(da):
    """Return the DataArray with its data in memory (a no-op for one already read).

    Parameters
    ----------
    da : xarray.DataArray
        The array.

    Returns
    -------
    xarray.DataArray
        ``da`` when it is in memory, else the loaded array.
    """
    return da if da.variable._in_memory else da.load()


def scale_in_place(da, factor):
    """Multiply the loaded array by ``factor`` in its own memory (float32 stays float32), if there is a factor.

    Parameters
    ----------
    da : xarray.DataArray
        The array, in memory and floating point.
    factor : float or None
        The factor; None or 0 leaves the array as it is.

    Returns
    -------
    xarray.DataArray
        ``da``, changed in place.
    """
    if factor:
        data = da.values
        np.multiply(data, factor, out=data)
    return da


def read_selection(obj, block_dim, keep=None, readers=None):
    """Read a box-selected Dataset or DataArray into memory (see `load_blocks`); Dataset variables one by one.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The lazy data.
    block_dim : str
        The dim the blocks are cut along.
    keep : numpy.ndarray of bool, shape (lat, lon), optional
        The cells to read; None reads everything.
    readers : int, optional
        The most processes the blocks may be decoded by; None reads in this process.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The data in memory.
    """
    if isinstance(obj, xr.DataArray):
        return load_blocks(obj, block_dim, keep, readers=readers)
    return obj.assign({name: load_blocks(var, block_dim, keep, readers=readers) for name, var in obj.data_vars.items()})


def _mask_in_place(da, keep):
    """Set the cells outside ``keep`` to NaN in the array's own memory.

    An integer array cannot hold NaN: it is masked with ``where`` into a float copy instead.

    Parameters
    ----------
    da : xarray.DataArray
        The array, in memory, with ``lat`` and ``lon`` dims.
    keep : numpy.ndarray of bool, shape (lat, lon)
        The cells to keep.

    Returns
    -------
    xarray.DataArray
        ``da``, changed in place (a masked copy for a non-floating array).
    """
    if not np.issubdtype(da.dtype, np.floating):
        return da.where(xr.DataArray(keep, dims=("lat", "lon"), coords={"lat": da["lat"], "lon": da["lon"]}))
    data = da.values
    view = np.moveaxis(data, (da.dims.index("lat"), da.dims.index("lon")), (-2, -1))
    view[..., ~keep] = np.nan
    return da


def implicit_keep(obj, area: AreaSpec, select: Selection | None = None):
    """Return the cells the implicit land and country masks keep in a box-selected ``obj``.

    The result is None when the cells are chosen another way (a ``mask`` section or an ``nc_mask``), when ``obj`` has no
    box, when both implicit masks are off, or when every cell is kept.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The box-selected data.
    area : hcwf.config.model.AreaSpec
        The area of the run (region, mask, nc_mask).
    select : hcwf.config.model.Selection, optional
        Switches the implicit masks off; None leaves both on.

    Returns
    -------
    numpy.ndarray of bool, shape (lat, lon) or None
        True for the cells kept, or None.
    """
    if not has_box(obj, area.region) or area.mask or area.nc_mask:
        return None
    land_only, shp_only = implicit_masks(select)
    if not (land_only or shp_only):
        return None
    keep = _keep_cells(obj["lat"].values, obj["lon"].values, area.region, land_only, shp_only)
    return None if keep.all() else keep


def mask_region_loaded(obj, *, area: AreaSpec, select: Selection | None = None, dataset_name=None, dir_fig=None):
    """Limit an already loaded, box-selected Dataset or DataArray to the cells of the run's mask, others set to NaN.

    The same cells as `region_select` keeps. With the implicit land and country masks the NaNs are written into the
    array's own memory (one pass, no copy); a ``mask`` section or an ``nc_mask`` goes through `hcwf.spatial.masks` as
    in `region_select`, which builds a masked copy (and raises ``MaskError`` when both are set).

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The loaded, box-selected data.
    area : hcwf.config.model.AreaSpec
        The area of the run (region, mask, nc_mask).
    select : hcwf.config.model.Selection, optional
        Switches the implicit masks off; None leaves both on.
    dataset_name : str, optional
        Labels the mask in messages, records and its figure.
    dir_fig : str or os.PathLike, optional
        The folder of the mask's figure.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The masked data.
    """
    region = area.region
    if not has_box(obj, region):
        return obj
    if area.mask or area.nc_mask:
        return region_select_masks(obj, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
    keep = implicit_keep(obj, area, select)
    if keep is None:
        return obj
    if isinstance(obj, xr.DataArray):
        return _mask_in_place(obj, keep)
    for name, var in obj.data_vars.items():
        if "lat" in var.dims and "lon" in var.dims:
            obj[name] = _mask_in_place(var, keep)
    return obj


def _readable_by_cells(obj, block_dim) -> bool:
    """Tell whether every variable of ``obj`` can be read with a cell mask (`load_blocks` ``keep``).

    That needs it still on file, floating point, and over ``block_dim``, ``lat`` and ``lon``.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The data.
    block_dim : str
        The dim the blocks are cut along.

    Returns
    -------
    bool
        True when a cell mask can be used for the read.
    """
    variables = [obj] if isinstance(obj, xr.DataArray) else list(obj.data_vars.values())
    return all(not v.variable._in_memory and np.issubdtype(v.dtype, np.floating) and {block_dim, "lat", "lon"} <= set(v.dims)
               for v in variables)


def region_select_masks(ds_reg, *, area: AreaSpec, select: Selection | None = None, dataset_name=None, dir_fig=None):
    """Apply the cell masks of `region_select` to a box-selected dataset: ``mask``, else ``nc_mask``, land and country.

    Data without a box (no region, or admin units on a ``"unit"`` dimension) has no cells to mask.

    Parameters
    ----------
    ds_reg : xarray.Dataset or xarray.DataArray
        The box-selected data.
    area : hcwf.config.model.AreaSpec
        The area of the run (region, mask, nc_mask).
    select : hcwf.config.model.Selection, optional
        Switches the implicit masks off; None leaves both on.
    dataset_name : str, optional
        Labels the mask in messages, records and its figure (default ``"data"``).
    dir_fig : str or os.PathLike, optional
        The folder of the mask's figure.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The masked data, NaN outside the mask.

    Raises
    ------
    MaskError
        Both ``mask`` and ``nc_mask`` are set.
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
                  read_by=None, keep=None, readers=None, skip_masked=False, compact=False):
    """Subset ``ds`` to the region's lat/lon box, then limit it to the cells of the run's mask.

    With a mask in ``area`` (the config's ``mask:`` section) the mask is built on this dataset's own grid by
    `hcwf.spatial.masks` (``dataset_name`` labels it in messages, records and the PNG saved to ``dir_fig``); it replaces
    the implicit land and country masks below and cannot be combined with ``nc_mask``. Without a mask the old
    behaviour is unchanged: the optional NetCDF mask ``area.nc_mask``, then the land mask and the country-shapefile
    mask (each off when `select` says ``land_only`` / ``shp_only`` is False).

    Without ``read_by`` the result stays lazy and the masks are applied with ``where``. With ``read_by`` (a dim name) the
    box is read into memory once, block by block along that dim, and the implicit masks are then written in place
    (see `mask_region_loaded`): slice inits, leads and members before calling so only those are read. ``keep`` (a
    boolean array over the box's cells, see `load_blocks`) then reads only the storage tiles holding a kept cell; the
    other cells are NaN.

    With ``skip_masked`` the cells of the implicit masks are not read at all (the storage tiles with none of the run's
    cells are skipped, the others' masked cells set to NaN as they are read): the same array as masking it after the
    read. With ``compact`` too, a selection that has masked cells comes back as only its kept cells (the dims without
    ``lat`` and ``lon``, then ``"cell"``, see `load_blocks`), nothing allocated for the others.

    With no region, or on a ``"unit"`` dimension, there is no box and no cell mask: the data is read as it is. It
    raises ``MaskError`` when both ``mask`` and ``nc_mask`` are set, and `ValueError` when the longitude box crosses
    the longitude seam of the data (see `lon_swap`).

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data, lazy or in memory.
    area : hcwf.config.model.AreaSpec
        The area of the run (region, mask, nc_mask).
    select : hcwf.config.model.Selection, optional
        Switches the implicit masks off; None leaves both on.
    dataset_name : str, optional
        Labels the mask in messages, records and its figure.
    dir_fig : str or os.PathLike, optional
        The folder of the mask's figure.
    read_by : str, optional
        The dim to read the box along, block by block; None keeps the result lazy.
    keep : numpy.ndarray of bool, shape (lat, lon), optional
        The cells to read with ``read_by`` (see `load_blocks`).
    readers : int, optional
        The most processes the read may use (see `load_blocks`); None reads in this process.
    skip_masked : bool, default False
        Do not read the cells of the implicit masks (needs ``read_by``).
    compact : bool, default False
        With ``skip_masked``, return only the kept cells.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The data inside the box, NaN outside the mask.
    """
    ds_reg = region_box(ds, region=area.region)
    if read_by:
        if skip_masked and keep is None and _readable_by_cells(ds_reg, read_by):
            keep = implicit_keep(ds_reg, area, select)
            if keep is not None and compact and keep.any():
                return xr.Dataset({name: load_blocks(var, read_by, keep, readers=readers, compact=True)
                                   for name, var in ds_reg.data_vars.items()})
            if keep is not None:
                return read_selection(ds_reg, read_by, keep, readers)
        ds_reg = read_selection(ds_reg, read_by, keep, readers)
        return mask_region_loaded(ds_reg, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
    return region_select_masks(ds_reg, area=area, select=select, dataset_name=dataset_name, dir_fig=dir_fig)
