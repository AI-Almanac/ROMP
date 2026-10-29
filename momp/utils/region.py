"""Region helpers: longitude/latitude orientation fixes, regional subsetting with masks, and domain averaging."""
import xarray as xr
import numpy as np
from momp.params.region_def import domain
from momp.utils.land_mask import polygon_mask, mask_land, shp_mask, apply_nc_mask


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
        print("lon NOT in coords for model --> ")
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

        print("swapped longitude range ", np.min(ds_tag.lon), " - ", np.max(ds_tag.lon))

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



def region_select(ds, *, region, nc_mask, land_only=True, shp_only=True, **kwargs):
    """Subset `ds` to the region's lat/lon box, optionally applying a NetCDF mask, land mask and country-shapefile mask."""
    lats, latn, lonw, lone, ds = coords_fmt(ds, region=region)

    ds_reg = ds.sel(lat=slice(lats, latn), lon=slice(lonw, lone))


    if nc_mask:
        ds_reg = apply_nc_mask(ds_reg, nc_mask)


    if land_only:
        ds_reg = mask_land(ds_reg)

    if shp_only:
        ds_reg = shp_mask(ds_reg, region=region)

    return ds_reg



def domain_average(da, polygon=False, weighted=False, **kwargs):
    """Average `da` over lat/lon, optionally within the polygon and with cos(lat) weights."""
    if polygon:
        da_slice = polygon_mask(da)
    else:
        da_slice = da

    if weighted:
        weights = np.cos(np.deg2rad(da_slice.lat))
        weights.name = "weights"
        domain_avg = da_slice.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    else:
        domain_avg = da_slice.mean(dim=("lat", "lon"), skipna=True)

    return domain_avg
