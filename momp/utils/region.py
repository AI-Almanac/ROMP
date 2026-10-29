"""Region helpers: longitude/latitude orientation fixes, regional subsetting with masks, and domain averaging."""
import xarray as xr
import numpy as np
from momp.params.region_def import domain
from momp.utils.land_mask import polygon_mask, mask_land, shp_mask, apply_nc_mask, land_mask_for
import logging

log = logging.getLogger(__name__)


def swap_lon_axis(
    dataset: xr.Dataset, to: tuple[float, float], sort_ascending: bool = True
) -> xr.Dataset:
    """Swaps the orientation of a dataset's longitude axis.

    This is an API adopted from the xCDAT package
    (https://xcdat.readthedocs.io/en/latest/index.html)
    (https://xcdat.readthedocs.io/en/latest/_modules/xcdat/axis.html#swap_lon_axis)

    This method also swaps the axis orientation of the longitude bounds if it
    exists. Afterwards, it sorts longitude and longitude bounds values in
    ascending order.

    Note, based on how datasets are chunked, swapping the longitude dimension
    and sorting might raise ``PerformanceWarning: Slicing is producing a
    large chunk. To accept the large chunk and silence this warning, set the
    option...``. This function uses xarray's arithmetic to swap orientations,
    so this warning seems potentially unavoidable.

    Parameters
    ----------
    dataset : xr.Dataset
         The Dataset containing a longitude axis.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation
    sort_ascending : bool
        After swapping, sort in ascending order (True), or keep existing order
        (False).

    Returns
    -------
    xr.Dataset
        The Dataset with swapped lon axes orientation.
    """
    ds = dataset.copy()
    coords = get_dim_coords(ds, "X").coords
    coord_keys = list(coords.keys())

    # Attempt to swap the orientation for longitude coordinates.
    for key in coord_keys:
        new_coord = _swap_lon_axis(ds.coords[key], to)

        if ds.coords[key].identical(new_coord):
            continue

        ds.coords[key] = new_coord

    try:
        bounds = ds.bounds.get_bounds("X")
    except KeyError:
        bounds = None

    if isinstance(bounds, xr.DataArray):
        ds = _swap_lon_bounds(ds, str(bounds.name), to)
    elif isinstance(bounds, xr.Dataset):
        for key in bounds.data_vars.keys():
            ds = _swap_lon_bounds(ds, str(key), to)

    if sort_ascending:
        ds = ds.sortby(list(coords.dims), ascending=True)

    return ds


def lon_swap(ds_tag, *, region, **kwargs):
    """Rename coords to lat/lon, then return the region bounds and dataset in a consistent longitude convention.

    Returns
    -------
    tuple
        (lats, latn, lonw, lone, ds_tag); the bounds are conformed to the dataset's longitude range.
    """
    lats, latn, lonw, lone = domain(region, **kwargs)

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
        if np.min(ds_tag.lon) < 0:
            ds_tag = swap_lon_axis(ds_tag, (0, 360)).compute()
            lonw = lonw % 360
            lone = lone % 360
        else:
            ds_tag = swap_lon_axis(ds_tag, (-180, 180)).compute()
            lonw = ((lonw + 180) % 360) - 180
            lone = ((lone + 180) % 360) - 180

        log.debug("%s %s %s %s", "swapped longitude range ", np.min(ds_tag.lon), " - ", np.max(ds_tag.lon))

    return lats, latn, lonw, lone, ds_tag


def lat_swap(ds_tag):
    """Flip the dataset along lat if latitudes are in descending order."""
    if ds_tag.lat[0] > ds_tag.lat[-1]:
        ds_tag = ds_tag.isel(lat=slice(None, None, -1))

    return ds_tag


def coords_fmt(ds_tag, *, region, **kwargs):
    """Standardize lon/lat orientation and return (lats, latn, lonw, lone, ds_tag) for the region."""
    lats, latn, lonw, lone, ds_tag = lon_swap(ds_tag, region=region, **kwargs)
    ds_tag = lat_swap(ds_tag)

    return lats, latn, lonw, lone, ds_tag



def region_box(ds, *, region):
    """Subset `ds` (Dataset or DataArray) to the region's lat/lon box, lazily: nothing is read from file here.

    Longitude and latitude are put in ROMP's orientation first (see coords_fmt). A dataset that has to cross the
    longitude seam is loaded by swap_lon_axis; slice inits, leads and members before calling for large inputs.
    """
    lats, latn, lonw, lone, ds = coords_fmt(ds, region=region)
    return ds.sel(lat=slice(lats, latn), lon=slice(lonw, lone))


_KEEP_CACHE = {}


def _keep_cells(lat, lon, region, land_only, shp_only):
    """Boolean (lat, lon) array of the cells the implicit land and country masks keep; cached by grid and region.

    The land part is momp.utils.land_mask.land_mask_for (cached per grid); the country part is shp_mask run on a
    2-D field of ones, so the cells are exactly those the masks would leave unmasked on any data with these coordinates.
    """
    key = (np.asarray(lat).tobytes(), np.asarray(lon).tobytes(), str(region), bool(land_only), bool(shp_only))
    if key not in _KEEP_CACHE:
        keep = land_mask_for(lat, lon).copy() if land_only else np.ones((len(lat), len(lon)), dtype=bool)
        if shp_only:
            field = xr.DataArray(np.ones((len(lat), len(lon)), dtype="float32"), dims=("lat", "lon"),
                                 coords={"lat": np.asarray(lat), "lon": np.asarray(lon)})
            keep &= ~np.isnan(shp_mask(field, region=region).values)
        _KEEP_CACHE[key] = keep
    return _KEEP_CACHE[key]


_BLOCK_BYTES = 64 * 1024 * 1024


def load_blocks(da, block_dim):
    """Read a (lazy) DataArray into memory once, filled block by block along block_dim.

    The result is allocated once; each block (about 64 MB, a whole number of stored chunks when the source is
    chunked) is read and copied in, so the reader's temporaries for fancy-indexed reads stay block sized instead of
    the size of the whole selection. An array already in memory is returned as it is.
    """
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


def read_selection(obj, block_dim):
    """Read a box-selected Dataset or DataArray into memory (see load_blocks); Dataset variables are read one by one."""
    if isinstance(obj, xr.DataArray):
        return load_blocks(obj, block_dim)
    return obj.assign({name: load_blocks(var, block_dim) for name, var in obj.data_vars.items()})


def _mask_in_place(da, keep):
    """Set the cells outside `keep` (a (lat, lon) boolean array) to NaN in the array's own memory; returns da."""
    if not np.issubdtype(da.dtype, np.floating):
        return da.where(xr.DataArray(keep, dims=("lat", "lon"), coords={"lat": da["lat"], "lon": da["lon"]}))
    data = da.values
    view = np.moveaxis(data, (da.dims.index("lat"), da.dims.index("lon")), (-2, -1))
    view[..., ~keep] = np.nan
    return da


def mask_region_loaded(obj, *, region, nc_mask=None, land_only=True, shp_only=True, mask=None, dataset_name=None,
                       dir_fig=None, **kwargs):
    """Limit an already loaded, box-selected Dataset or DataArray to the cells of the run's mask, others set to NaN.

    The same cells as region_select keeps. With the implicit land and country masks the NaNs are written into the
    array's own memory (one pass, no copy); a `mask` section or an nc_mask goes through momp.utils.mask or
    apply_nc_mask as in region_select, which build a masked copy.
    """
    if mask or nc_mask:
        return region_select_masks(obj, region=region, nc_mask=nc_mask, land_only=land_only, shp_only=shp_only,
                                   mask=mask, dataset_name=dataset_name, dir_fig=dir_fig)
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


def region_select_masks(ds_reg, *, region, nc_mask=None, land_only=True, shp_only=True, mask=None,
                        dataset_name=None, dir_fig=None):
    """Apply the cell masks of region_select to a box-selected dataset: `mask`, else nc_mask, land and country masks."""
    if mask:
        if nc_mask:
            from momp.utils.mask import MaskError

            raise MaskError("both `mask` and `nc_mask` are set; use one. nc_mask: <file> is the same as "
                            "mask: {file: <file>}, except that the old nc_mask also applies the land and country "
                            "masks implicitly")
        from momp.utils.mask import apply_region_mask

        return apply_region_mask(ds_reg, mask, dataset_name or "data", dir_fig)

    if nc_mask:
        ds_reg = apply_nc_mask(ds_reg, nc_mask)

    if land_only:
        ds_reg = mask_land(ds_reg)

    if shp_only:
        ds_reg = shp_mask(ds_reg, region=region)

    return ds_reg


def region_select(ds, *, region, nc_mask=None, land_only=True, shp_only=True, mask=None, dataset_name=None,
                  dir_fig=None, read_by=None, **kwargs):
    """Subset `ds` to the region's lat/lon box, then limit it to the cells of the run's mask.

    With a `mask` mapping (the config's ``mask:`` section) the mask is built on this dataset's own grid by
    momp.utils.mask (dataset_name labels it in messages, records and the PNG saved to dir_fig); it replaces
    the implicit land and country masks below and cannot be combined with nc_mask. Without `mask` the old
    behaviour is unchanged: the optional NetCDF mask nc_mask, then the land mask and the country-shapefile mask.

    Without read_by the result stays lazy and the masks are applied with where. With read_by (a dim name) the
    box is read into memory once, block by block along that dim, and the implicit masks are then written in place
    (see mask_region_loaded): slice inits, leads and members before calling so only those are read.
    """
    ds_reg = region_box(ds, region=region)
    if read_by:
        ds_reg = read_selection(ds_reg, read_by)
        return mask_region_loaded(ds_reg, region=region, nc_mask=nc_mask, land_only=land_only, shp_only=shp_only,
                                  mask=mask, dataset_name=dataset_name, dir_fig=dir_fig)
    return region_select_masks(ds_reg, region=region, nc_mask=nc_mask, land_only=land_only, shp_only=shp_only,
                               mask=mask, dataset_name=dataset_name, dir_fig=dir_fig)
