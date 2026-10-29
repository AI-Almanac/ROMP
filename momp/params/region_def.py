"""Region definitions: rectangular lat/lon domains and the CMZ polygon outline."""
import numpy as np
import logging

log = logging.getLogger(__name__)


# define rectangular domain
def domain(region, **kwargs):
    """Return (lats, latn, lonw, lone) bounds for a named region.

    The four old names below keep their boxes (so runs written with them score the same cells). Any other name,
    in any case, is looked up in the region registry (momp/onset/regions, ids and aliases) and takes its
    ``domain`` lat/lon box, so ``region: ethiopia`` works as well as ``Ethiopia``.
    """
    if region not in _LEGACY_BOXES:
        return _registry_box(region)
    if region == "Ethiopia":
        lats = 3
        latn = 15
        lonw = 33
        lone = 48

    if region == "Sub_Ethiopia":
        lats = 7.5
        latn = 13.5
        lonw = 37.5
        lone = 40

    if region == "India":
        lats = 6.46
        latn = 35.51
        lonw = 68.11
        lone = 91.4

    if region == "rect_boundary":
        lats = 999
        latn = 12
        lonw = 38
        lone = 42

    return lats, latn, lonw, lone


_LEGACY_BOXES = ("Ethiopia", "Sub_Ethiopia", "India", "rect_boundary")


def _registry_box(region):
    """(lats, latn, lonw, lone) of a region of the registry; an unknown name is an error listing the known ones."""
    from momp.onset import regions

    found = regions.load_region(str(region))
    if found.lat is None or found.lon is None:
        raise regions.RegionError(f"region '{found.id}' has no domain (lat/lon box). Fix: add 'domain:' to "
                                  f"momp/onset/regions/{found.id}.yaml")
    return found.lat[0], found.lat[1], found.lon[0], found.lon[1]


def polygon_boundary(da):
    """Return (lat, lon) arrays of the CMZ polygon matching the grid spacing of da (1, 2 or 4 degrees), else (None, None)."""
    polygon1_lat, polygon1_lon = None, None

    orig_lat = da.lat.values
    orig_lon = da.lon.values

    lat_diff = abs(orig_lat[1]-orig_lat[0])

    if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
        polygon1_lon = np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83])
        polygon1_lat = np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17])
        log.debug("Using 2-degree CMZ polygon coordinates")

    elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
        polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        log.debug("Using 4-degree CMZ polygon coordinates")

    elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
        polygon1_lon = np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74])
        polygon1_lat = np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18])
        log.debug("Using 1-degree CMZ polygon coordinates")

    return polygon1_lat, polygon1_lon
