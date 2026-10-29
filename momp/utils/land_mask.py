"""Masking helpers: polygon, land-sea, country-shapefile and NetCDF masks, plus polygon/outline plotting."""
import hashlib
import numpy as np
from matplotlib.path import Path
from typing import Union
import regionmask
import xarray as xr
from momp.params.region_def import polygon_boundary
from momp.utils.standard import dim_fmt
import logging

log = logging.getLogger(__name__)


# Function to find grid points inside a polygon (For core-monsoon zone analysis)
def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """
    Find grid points that are inside a polygon.

    Parameters:
    polygon_lon: array of polygon longitude vertices
    polygon_lat: array of polygon latitude vertices
    grid_lons: array of grid longitude points
    grid_lats: array of grid latitude points

    Returns:
    inside_mask: boolean array indicating which points are inside
    inside_lons: longitude coordinates of points inside polygon
    inside_lats: latitude coordinates of points inside polygon
    """

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
    """mask data based on polygon boundary"""

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



def polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=1.25):
    """ add polygon boundary to basemap """
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon as ShapelyPolygon
    from cartopy.feature import ShapelyFeature

    # Cartopy gridlines create a separate layer that can cover patches added with ax.add_patch()
    # use a ShapelyFeature and add it with ax.add_feature() with a higher zorder
    # so it appears above gridlines and map features
    poly_geom = ShapelyPolygon(list(zip(polygon1_lon, polygon1_lat)))

    # wrap as a Cartopy feature
    poly_feature = ShapelyFeature(
        [poly_geom],
        crs=ccrs.PlateCarree(),  # coordinates are lon/lat
        edgecolor='black',
        facecolor='none',
        linewidth=linewidth
    )

    # add to axes above gridlines
    ax.add_feature(poly_feature, zorder=10)

    return ax



def add_polygon(ax, da, polygon, return_polygon=False, linewidth=1.25):
    """Draw the polygon boundary for `da` on `ax` when `polygon` is truthy and a boundary is defined.

    Returns `ax`, or (ax, polygon_lat, polygon_lon, polygon_defined) if `return_polygon` is True.
    """
    polygon_defined = False

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(da)

        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    # Add CMZ polygon only if defined
    if polygon_defined:

        ax = polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=linewidth)

    if return_polygon:
        return ax, polygon1_lat, polygon1_lon, polygon_defined
    else:
        return ax



def get_india_outline(shpfile_path):
    """
    Get region outline coordinates from shapefile.
    """
    import geopandas as gpd
    # Update this path to your India shapefile
    india_gdf = gpd.read_file(shpfile_path)

    boundaries = []
    for geom in india_gdf.geometry:
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
            lon_coords = [coord[0] for coord in coords]
            lat_coords = [coord[1] for coord in coords]
            boundaries.append((lon_coords, lat_coords))
        elif hasattr(geom, 'geoms'):
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    coords = list(sub_geom.exterior.coords)
                    lon_coords = [coord[0] for coord in coords]
                    lat_coords = [coord[1] for coord in coords]
                    boundaries.append((lon_coords, lat_coords))
    return boundaries



def _grid_key(lat, lon) -> str:
    """A hash of the lat and lon values (as float64), the cache key of a grid."""
    lat = np.ascontiguousarray(np.asarray(lat, dtype="float64"))
    lon = np.ascontiguousarray(np.asarray(lon, dtype="float64"))
    return hashlib.sha256(str(lat.shape).encode() + lat.tobytes() + str(lon.shape).encode() + lon.tobytes()).hexdigest()


_RAW_MASKS: dict = {}
"""Raw land masks by grid key (insertion-ordered, at most _MAX_GRIDS)."""
_MAX_GRIDS = 16


def _raw_land_mask(key: str, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """regionmask's land mask (0 on land, NaN at sea) on the grid of lat and lon, rasterised once per grid key."""
    if key not in _RAW_MASKS:
        if len(_RAW_MASKS) >= _MAX_GRIDS:
            _RAW_MASKS.pop(next(iter(_RAW_MASKS)))
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        _RAW_MASKS[key] = land_mask.mask(lon, lat=lat)
    return _RAW_MASKS[key]


def clear_land_masks() -> None:
    """Forget every cached land mask (for tests)."""
    _RAW_MASKS.clear()
    _LAND_BOOL.clear()


_LAND_BOOL: dict = {}
"""Read-only boolean land masks by grid key."""


def land_mask_for(lat, lon) -> np.ndarray:
    """Boolean land mask of shape (nlat, nlon) for 1-D lat and lon values, True on land.

    Built once per grid in this process (cached by a hash of the coordinates) and returned read-only, so callers
    can apply it in place on numpy arrays, e.g. ``arr[..., ~land_mask_for(lat, lon)] = np.nan``. It selects the
    same cells as ``mask_land``.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    key = _grid_key(lat, lon)
    if key not in _LAND_BOOL:
        if len(_LAND_BOOL) >= _MAX_GRIDS:
            _LAND_BOOL.pop(next(iter(_LAND_BOOL)))
        raw = _raw_land_mask(key, xr.DataArray(lat, dims="lat", coords={"lat": lat}, name="lat"),
                             xr.DataArray(lon, dims="lon", coords={"lon": lon}, name="lon"))
        out = np.array(~np.isnan(raw.transpose("lat", "lon").values))
        out.setflags(write=False)
        _LAND_BOOL[key] = out
    return _LAND_BOOL[key]


def create_land_sea_mask(
    obj: Union[xr.Dataset, xr.DataArray],
    as_boolean: bool = False,
) -> xr.DataArray:
    """Generate a land-sea mask (1 for land, 0 for sea) for a given xarray Dataset or DataArray.
    stemed from pcmdi_metrics.utils.land_sea_mask

    Parameters
    ----------
    obj : Union[xr.Dataset, xr.DataArray]
        The Dataset or DataArray object.
    as_boolean : bool, optional
        Set mask value to True (land) or False (ocean), by default False, thus 1 (land) and 0 (ocean).

    Returns
    -------
    xr.DataArray
        A DataArray of land-sea mask (1 or 0 for land or sea, or True or False for land or sea).

    Examples
    --------
    >>> mask = create_land_sea_mask(ds)  #  Generate land-sea mask (land: 1, sea: 0)
    >>> mask = create_land_sea_mask(ds, as_boolean=True)  # Generate land-sea mask (land: True, sea: False)
    """

    lon = obj["lon"]
    lat = obj["lat"]

    # Mask the land-sea mask to match the dataset's coordinates (rasterised once per grid, then copied)
    land_sea_mask = _raw_land_mask(_grid_key(lat.values, lon.values), lat, lon).copy()

    if as_boolean:
        # Convert the 0 (land) & nan (ocean) land-sea mask to a boolean mask
        land_sea_mask = xr.where(land_sea_mask, False, True)
    else:
        # Convert the boolean land-sea mask to a 1/0 mask
        land_sea_mask = xr.where(land_sea_mask, 0, 1)

    return land_sea_mask


def mask_land(da, land=True):
    """
    Mask a DataArray to select either land or sea areas.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be masked.
    land : bool, optional
        If True, keep land areas (default). If False, keep sea areas.

    Returns
    -------
    xarray.DataArray
        Masked DataArray with values retained for land or sea, NaNs elsewhere.
    """

    land_sea_mask = create_land_sea_mask(da)

    if land:
        da_masked = da.where(land_sea_mask == 1)
    else:
        da_masked = da.where(land_sea_mask == 0)

    return da_masked


def get_shp(region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries'):
    """ Create country boundaries"""

    import cartopy.io.shapereader as shpreader

    # Load Ethiopia shapefile
    ethiopia_shp = shpreader.natural_earth(resolution=resolution, category=category, name=name)

    # Find Ethiopia geometry
    region_geom = None
    names = {region, _registry_title(region)}
    for country in shpreader.Reader(ethiopia_shp).records():
        if country.attributes['NAME'] in names:
            region_geom = country.geometry
            return region_geom


def _registry_title(region):
    """The registry title of a region name in any case ("ethiopia" -> "Ethiopia"), else the name itself."""
    try:
        from momp.onset import regions

        return regions.load_region(str(region)).title
    except Exception:  # noqa: BLE001 - not a registry region: the name is matched as written
        return region


def shp_mask(da, region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries',
             return_mask=False, **kwargs):
    """ Create mask based on country boundaries"""
    from shapely.vectorized import contains

    # get region boundary
    region_geom = get_shp(region=region, resolution=resolution, category=category, name=name)

    if region_geom is None:
        log.warning("specified region is not in cartopy.io.shapereader")
        return da

    # Create mask for Ethiopia
    lons, lats = np.meshgrid(da.lon, da.lat)
    points = np.column_stack((lons.ravel(), lats.ravel()))
    mask = contains(region_geom, points[:, 0], points[:, 1])
    mask = mask.reshape(lons.shape)
    mask_da = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': da.lat, 'lon': da.lon})

    # Apply mask to data
    da_masked = da.where(mask_da)

    if return_mask:
        return da_masked, mask_da
    else:
        return da_masked


def shp_outline(ax, region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries'):
    """ add shapefile country boundaries to plot"""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # get region boundary
    region_geom = get_shp(region=region, resolution=resolution, category=category, name=name)

    if region_geom is None:
        log.warning("specified region is not in cartopy.io.shapereader")
        return ax

    if hasattr(ax, "add_geometries"):
        # Cartopy GeoAxes
        ax.add_geometries(
            [region_geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=10
        )
    else:
        # Regular matplotlib Axes — handle both Polygon and MultiPolygon
        geoms = [region_geom] if region_geom.geom_type == 'Polygon' else list(region_geom.geoms)
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.5, zorder=10)

    # Optional: background
    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

    return ax



def apply_nc_mask(ds, nc_mask, mask_var=None, keep_value=1):
    """
    Mask an xarray Dataset/DataArray using a 0/1 mask stored in a NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Data to mask.
    nc_mask : str
        Path to NetCDF mask file (e.g., "land.nc").
    mask_var : str, optional
        Variable name inside mask file. If None, auto-pick if only one var exists.
    keep_value : int or float, default 1
        Which value means "keep". Commonly 1 (keep) and 0 (mask out).

    Returns
    -------
    xr.Dataset or xr.DataArray
        Masked data (values where mask is False become NaN).
    """

    mask_ds = xr.open_dataset(nc_mask)
    mask_ds = dim_fmt(mask_ds)

    mask_ds = mask_ds.sel(lat=ds.lat, lon=ds.lon, method="nearest")

    try:
        # pick mask variable
        if mask_var is None:
            if len(mask_ds.data_vars) != 1:
                raise ValueError(
                    f"Mask file has multiple variables {list(mask_ds.data_vars)}; "
                    "please specify mask_var."
                )
            mask = next(iter(mask_ds.data_vars.values()))
        else:
            mask = mask_ds[mask_var]

        # convert 0/1 (or numeric) to boolean
        if mask.dtype == bool:
            mask_bool = mask
        else:
            mask_bool = (mask == keep_value)

        # align to prevent accidental broadcasting (and catch mismatched grids)
        mask_bool, ds_aligned = xr.align(mask_bool, ds, join="exact")

        for var in ds_aligned.data_vars:
            # This uses numpy's broadcasting which is very fast
            ds_aligned[var].values = np.where(mask_bool.values, ds_aligned[var].values, np.nan)

        return ds_aligned

    finally:
        mask_ds.close()

