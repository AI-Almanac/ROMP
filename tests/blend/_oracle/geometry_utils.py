# TEST-ONLY ORACLE: copied unchanged from onset_blending@1e6d696 tests/_legacy/geometry_utils.py.
# Never import this from hcwf; it is the reference the hcwf.spatial tests compare against.
# ==============================================================================
# File: geometry_utils.py
# ==============================================================================
# Purpose
#   Spec-driven geometry / regridding helpers so the pipeline can be re-targeted
#   to a new geography by supplying only a boundary shapefile + gridded NetCDF.
#
#   Consolidates the two hand-written weight builders (utils/remap_weights.py and
#   utils/remap_weights_ngcm.py) into one configurable function that:
#     - auto-detects grid resolution from the NetCDF coordinates (no hardcoded
#       half-cell size),
#     - reads the admin-unit key column, optional parent column, and CRS from a
#       `geometry` config block (defaults reproduce the current Ethiopia setup),
#     - renames the admin key to the canonical internal name `adm3_name`.
#
# `geometry` config block (all keys optional except `shapefile`):
#   geometry:
#     shapefile:       path/to/admin.shp
#     region_id_col:   adm3_pcode     # stable ID used for joins
#     region_name_col: adm3_name      # optional display label
#     region_key_col:  adm3_name      # legacy alias for region_id_col
#     parent_key_col:  adm2_name      # optional (zone overlays / labels)
#     crs:             EPSG:4326
#     region_label:    Ethiopia       # used by the map/plot utilities
#     grid_lat_var:    latitude       # NetCDF latitude coord (lat/latitude auto)
#     grid_lon_var:    longitude
#     grid_id_decimal_digits: 2       # optional deterministic <lat>_<lon> precision
#     grid_id_format:  fixed           # optional: fixed or trimmed
#
# Function index
#   get_geometry_cfg(spec)
#   get_half_delta(arr)
#   standardize_grid_coords(ds, lat_var, lon_var)
#   load_admin_geometry(cfg)
#   build_grid_to_admin_weights(sample_nc, cfg, out_csv=None)
# ==============================================================================

import os
import glob
import numpy as np

from .spatial_id_utils import (
    normalize_id_series,
)

# Canonical internal names - the rest of the pipeline keys on these.
CANON_REGION_KEY = "adm3_name"
CANON_REGION_NAME = "region_name"
CANON_PARENT_KEY = "adm2_name"
DEFAULT_CRS = "EPSG:4326"


def get_geometry_cfg(spec):
    """
    Extract the `geometry` config block from a spec dict, applying defaults.

    `region_id_col` is the stable join key. The older `region_key_col` remains
    an alias so existing specs continue to work unchanged.
    """
    g = dict((spec or {}).get("geometry") or {})
    region_id_col = (
        g.get("region_id_col") or g.get("region_key_col") or CANON_REGION_KEY
    )
    return {
        "shapefile":      g.get("shapefile"),
        "region_id_col":  region_id_col,
        "region_name_col": g.get("region_name_col"),
        "region_key_col": region_id_col,  # legacy internal alias
        "parent_key_col": g.get("parent_key_col"),  # optional
        "crs":            g.get("crs", DEFAULT_CRS),
        "region_label":   g.get("region_label"),
        "grid_lat_var":   g.get("grid_lat_var"),   # None -> auto-detect
        "grid_lon_var":   g.get("grid_lon_var"),
        "grid_id_decimal_digits": g.get("grid_id_decimal_digits"),
        "grid_id_format": g.get("grid_id_format"),
    }


def get_half_delta(arr):
    """
    Half the spacing between the first two coordinate values (grid half-cell).
    Falls back to 0.125 for single-point axes. Ported from remap_weights_ngcm.py
    so resolution is inferred from the data instead of hardcoded.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size > 1:
        return float(np.abs(arr[1] - arr[0]) / 2.0)
    return 0.125


def standardize_grid_coords(ds, lat_var=None, lon_var=None):
    """
    Return (lats, lons) arrays from an xarray Dataset, tolerant of common
    coordinate names. Explicit lat_var/lon_var win; otherwise tries
    latitude/lat and longitude/lon.
    """
    def _pick(explicit, candidates):
        if explicit is not None:
            if explicit not in ds.coords and explicit not in ds.variables:
                raise ValueError(f"Grid coord '{explicit}' not found in dataset.")
            return explicit
        for c in candidates:
            if c in ds.coords or c in ds.variables:
                return c
        raise ValueError(
            f"Could not find a coordinate among {candidates}. "
            f"Available: {list(ds.coords)}"
        )

    lat_name = _pick(lat_var, ["latitude", "lat", "y", "Y"])
    lon_name = _pick(lon_var, ["longitude", "lon", "x", "X"])
    return np.asarray(ds[lat_name].values, dtype=float), np.asarray(ds[lon_name].values, dtype=float)


def load_admin_geometry(cfg):
    """
    Read an admin-boundary shapefile and adapt its stable region ID to the
    existing internal `adm3_name` column. An optional display name is retained
    separately as `region_name`.

    cfg : dict from get_geometry_cfg (or a raw geometry block).
    """
    import geopandas as gpd

    cfg = {**get_geometry_cfg({"geometry": cfg})} if "region_key_col" not in cfg else cfg
    shp = cfg.get("shapefile")
    if not shp:
        raise ValueError("geometry.shapefile is required to load admin boundaries.")
    if not os.path.exists(shp):
        raise FileNotFoundError(f"Shapefile not found: {shp}")

    gdf = gpd.read_file(shp)

    region_col = (
        cfg.get("region_id_col") or cfg.get("region_key_col") or CANON_REGION_KEY
    )
    if region_col not in gdf.columns:
        raise ValueError(
            f"Shapefile missing region ID column '{region_col}'. Set "
            "geometry.region_id_col. "
            f"Available: {gdf.columns.tolist()}"
        )

    keep = {region_col: CANON_REGION_KEY}
    name_col = cfg.get("region_name_col")
    if name_col and name_col not in gdf.columns:
        raise ValueError(
            f"Shapefile missing region name column '{name_col}'. "
            f"Available: {gdf.columns.tolist()}"
        )
    if name_col and name_col != region_col:
        keep[name_col] = CANON_REGION_NAME
    parent_col = cfg.get("parent_key_col")
    if parent_col:
        if parent_col not in gdf.columns:
            raise ValueError(
                f"Shapefile missing parent key column '{parent_col}'. "
                f"Available: {gdf.columns.tolist()}"
            )
        keep[parent_col] = CANON_PARENT_KEY

    cols = list(keep.keys()) + ["geometry"]
    gdf = gdf[cols].rename(columns=keep).reset_index(drop=True)
    gdf[CANON_REGION_KEY] = normalize_id_series(
        gdf[CANON_REGION_KEY], context="shapefile region IDs"
    )
    if name_col == region_col:
        gdf[CANON_REGION_NAME] = gdf[CANON_REGION_KEY]
    duplicate_ids = gdf[CANON_REGION_KEY].duplicated(keep=False)
    if duplicate_ids.any():
        sample = ", ".join(gdf.loc[duplicate_ids, CANON_REGION_KEY].unique()[:10])
        raise ValueError(f"Shapefile region IDs must be unique; duplicates: {sample}")

    crs = cfg.get("crs", DEFAULT_CRS)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif crs is not None and str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    return gdf


def normalize_regrid_weights(weights_df, context="regrid weights"):
    """Accept canonical `target_id` or legacy `adm3_name` weight tables."""
    out = weights_df.copy()
    if "target_id" not in out.columns and CANON_REGION_KEY not in out.columns:
        raise ValueError(f"{context} must contain 'target_id' or '{CANON_REGION_KEY}'.")

    if "target_id" in out.columns:
        out["target_id"] = normalize_id_series(
            out["target_id"], context=f"{context} target IDs"
        )
    if CANON_REGION_KEY in out.columns:
        out[CANON_REGION_KEY] = normalize_id_series(
            out[CANON_REGION_KEY], context=f"{context} legacy target IDs"
        )
    if {"target_id", CANON_REGION_KEY}.issubset(out.columns):
        mismatch = out["target_id"] != out[CANON_REGION_KEY]
        if mismatch.any():
            raise ValueError(
                f"{context}: 'target_id' and '{CANON_REGION_KEY}' disagree."
            )
    elif "target_id" in out.columns:
        out[CANON_REGION_KEY] = out["target_id"]
    else:
        out["target_id"] = out[CANON_REGION_KEY]

    lat_col = "latitude" if "latitude" in out.columns else "lat"
    lon_col = "longitude" if "longitude" in out.columns else "lon"
    required = [lat_col, lon_col, "weight"]
    if any(column not in out.columns for column in required):
        raise ValueError(f"{context} must contain latitude, longitude, and weight.")
    duplicate_rows = out.duplicated([lat_col, lon_col, "target_id"], keep=False)
    if duplicate_rows.any():
        raise ValueError(
            f"{context} contains duplicate latitude/longitude/target_id rows."
        )
    return out


def build_grid_to_admin_weights(sample_nc, cfg, out_csv=None, admin_gdf=None):
    """
    Build area-fraction weights mapping gridded cells -> admin units.

    Overlays each grid cell (a square polygon sized by the auto-detected
    resolution) with the admin polygons and computes
    weight = intersection_area / cell_area (in the config CRS). Reproduces the
    behavior of remap_weights*.py but fully parameterized.

    Parameters
    ----------
    sample_nc : str
        Path to a representative gridded NetCDF (or a directory; the first *.nc
        that is not an *_adm3.nc is used).
    cfg : dict
        A `geometry` block (see get_geometry_cfg).
    out_csv : str or None
        If given, write the mapping CSV (columns: latitude, longitude,
        adm3_name, weight). The legacy adm3_name column carries the stable
        target ID and is accepted interchangeably with target_id on input.

    Returns
    -------
    DataFrame with columns: latitude, longitude, adm3_name, weight.
    """
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import box

    cfg = get_geometry_cfg({"geometry": cfg}) if "region_key_col" not in cfg else cfg

    # Resolve the sample NetCDF file
    if os.path.isdir(sample_nc):
        cand = [f for f in sorted(glob.glob(os.path.join(sample_nc, "*.nc")))
                if not f.endswith("_adm3.nc")]
        if not cand:
            raise FileNotFoundError(f"No .nc files found in {sample_nc}")
        sample_nc = cand[0]
    if not os.path.exists(sample_nc):
        raise FileNotFoundError(f"Sample NetCDF not found: {sample_nc}")

    admin = admin_gdf if admin_gdf is not None else load_admin_geometry(cfg)

    with xr.open_dataset(sample_nc) as ds:
        lats, lons = standardize_grid_coords(ds, cfg.get("grid_lat_var"), cfg.get("grid_lon_var"))

    half_lat = get_half_delta(lats)
    half_lon = get_half_delta(lons)
    print(f"Building grid polygons (resolution {half_lat*2:.4f} lat x {half_lon*2:.4f} lon)...")

    cell_records = []
    for lat in lats:
        for lon in lons:
            latf, lonf = float(lat), float(lon)
            poly = box(lonf - half_lon, latf - half_lat, lonf + half_lon, latf + half_lat)
            cell_records.append({
                "latitude": latf, "longitude": lonf,
                "geometry": poly, "cell_area": poly.area,
            })
    grid_gdf = gpd.GeoDataFrame(cell_records, crs=cfg.get("crs", DEFAULT_CRS))

    print("Computing intersections...")
    overlaid = gpd.overlay(grid_gdf, admin, how="intersection")
    overlaid["intersection_area"] = overlaid.geometry.area
    overlaid["weight"] = overlaid["intersection_area"] / overlaid["cell_area"]
    overlaid = overlaid[overlaid["weight"] > 1e-5]

    mapping = overlaid[["latitude", "longitude", CANON_REGION_KEY, "weight"]].reset_index(drop=True)
    normalize_regrid_weights(mapping, context="generated grid-to-admin weights")

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        mapping.to_csv(out_csv, index=False)
        print(f"Wrote {len(mapping)} grid->admin weights to {out_csv}")
    return mapping
