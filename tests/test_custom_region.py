"""A region with no built-in definition (AI Almanac's custom ones) uses the bounds it comes with."""
import numpy as np
import pytest
import xarray as xr

from momp.params.region_def import domain
from momp.utils.region import region_select

NIGERIA = dict(lat_min=3.75, lat_max=14.5, lon_min=2.25, lon_max=15.25)


def test_builtin_regions_unchanged():
    assert domain("Ethiopia") == (3, 15, 33, 48)
    assert domain("India", **NIGERIA) == (6.46, 35.51, 68.11, 91.4)   # (its own bounds win)


def test_custom_region_uses_its_bounds():
    assert domain("custom", **NIGERIA) == (3.75, 14.5, 2.25, 15.25)


def test_custom_region_without_bounds_says_so():
    with pytest.raises(ValueError, match="lat_min, lat_max, lon_min and lon_max"):
        domain("custom")


def test_region_select_crops_to_a_custom_region():
    ds = xr.Dataset({"tp": (("lat", "lon"), np.ones((81, 81)))},
                    coords={"lat": np.arange(-5, 15.25, 0.25), "lon": np.arange(0, 20.25, 0.25)})
    out = region_select(ds, region="custom", nc_mask=None, land_only=False, shp_only=False, **NIGERIA)
    assert float(out.lat.min()) == 3.75 and float(out.lat.max()) == 14.5
    assert float(out.lon.min()) == 2.25 and float(out.lon.max()) == 15.25
