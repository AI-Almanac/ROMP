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


def test_custom_bounds_survive_the_settings_a_job_runs_with():
    """A job's settings are rebuilt from the Case and Setting fields (case_cfg in app/*.py):
    the bounds must be among them, or domain() never sees them."""
    from dataclasses import asdict

    from momp.lib.control import init_dataclass
    from momp.lib.convention import Case, Setting
    from momp.utils.region import coords_fmt

    cfg = {"region": "custom", "nc_mask": None, "project_name": "nigeria", **NIGERIA}
    case_cfg = {**asdict(init_dataclass(Case, cfg)), **asdict(init_dataclass(Setting, cfg))}
    ds = xr.Dataset({"tp": (("lat", "lon"), np.ones((81, 81)))},
                    coords={"lat": np.arange(-5, 15.25, 0.25), "lon": np.arange(0, 20.25, 0.25)})
    lats, latn, lonw, lone, _ = coords_fmt(ds, **case_cfg)
    assert (lats, latn, lonw, lone) == (3.75, 14.5, 2.25, 15.25)
    ethiopia = {**asdict(init_dataclass(Case, {"region": "Ethiopia"})), **asdict(init_dataclass(Setting, {}))}
    assert coords_fmt(ds, **ethiopia)[:4] == (3, 15, 33, 48)        # (built-ins: unchanged)
    assert (ethiopia["land_only"], ethiopia["shp_only"]) == (True, True)   # (region_select's defaults)


def test_custom_region_masks_follow_the_config(monkeypatch):
    """AI Almanac's custom regions set land_only and shp_only False; they must reach region_select
    (dropped, it masked to a country shape named "custom", which can't be opened)."""
    from dataclasses import asdict

    import momp.utils.region as region_mod
    from momp.lib.control import init_dataclass
    from momp.lib.convention import Case, Setting

    def refuse(*a, **k):
        raise AssertionError("masked although the config says land_only/shp_only = False")
    monkeypatch.setattr(region_mod, "shp_mask", refuse)
    monkeypatch.setattr(region_mod, "mask_land", refuse)
    cfg = {"region": "custom", "nc_mask": None, "land_only": False, "shp_only": False, **NIGERIA}
    case_cfg = {**asdict(init_dataclass(Case, cfg)), **asdict(init_dataclass(Setting, cfg))}
    ds = xr.Dataset({"tp": (("lat", "lon"), np.ones((81, 81)))},
                    coords={"lat": np.arange(-5, 15.25, 0.25), "lon": np.arange(0, 20.25, 0.25)})
    out = region_select(ds, **case_cfg)
    assert float(out.lat.min()) == 3.75 and float(out.lon.max()) == 15.25


def test_whole_year_window_from_a_leap_filter_year_skips_29_february():
    """Nigeria's jobs score every day of the year, with date_filter_year 2000 (a leap year)."""
    import pandas as pd

    from momp.io.input import get_initialization_dates

    args = dict(date_filter_year=2000, init_days=(0, 1, 2, 3, 4, 5, 6), start_date=(2000, 1, 1),
                end_date=(2025, 12, 31))
    dates = get_initialization_dates(2001, **args)
    assert len(dates) == 365 and pd.Timestamp("2001-02-28") in dates and pd.Timestamp("2001-03-01") in dates
    assert len(get_initialization_dates(2004, **args)) == 366          # (a leap year keeps it)
    may_oct = dict(args, start_date=(2000, 5, 1), end_date=(2025, 10, 31), init_days=(0, 3))
    window = pd.date_range("2000-05-01", "2000-10-31")
    expected = pd.to_datetime(window[window.weekday.isin((0, 3))].strftime("2001-%m-%d"))
    assert list(get_initialization_dates(2001, **may_oct)) == list(expected)   # (Ethiopia-like: unchanged)
