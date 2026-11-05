"""A blend and the evaluation grid: a blend reads its rain on evaluation units (not through the evaluation grid), so
sources on different grids are refused with a Fix, and listed forecast files have their grids looked at."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _recipes import converted
from test_blend_training import (FORECAST_YEARS, LEAD_DAYS, OBS_YEARS, config_dict, onset_date, rain)

from hcwf.blend.training import train
from hcwf.io.grids import attach, source_grids

LATS, LONS = [10.0, 10.25], [77.0, 77.25]
FINE_LATS, FINE_LONS = [9.9375, 10.0625, 10.1875, 10.3125], [76.9375, 77.0625, 77.1875, 77.3125]


def _write_observations(folder):
    folder.mkdir(parents=True)
    for year in OBS_YEARS:
        dates = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(200)]
        data = np.stack([np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
                         for i in range(len(LATS))], axis=1)
        xr.Dataset({"precip": (("time", "lat", "lon"), data)},
                   coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS}).to_netcdf(folder / f"{year}.nc")


def _write_forecasts(folder, fine, members=3):
    """Write the forecasts on the observations' cells, or on cells half their size that repeat each one."""
    folder.mkdir(parents=True)
    lats, lons = (FINE_LATS, FINE_LONS) if fine else (LATS, LONS)
    for year in FORECAST_YEARS:
        issues = [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), members, LEAD_DAYS, len(lats), len(lons)), dtype="float32")
        for t, issue in enumerate(issues):
            leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
            for m in range(members):
                for i in range(len(lats)):
                    for j in range(len(lons)):
                        cell = (i // 2 if fine else i) * 2 + (j // 2 if fine else j)
                        data[t, m, :, i, j] = rain(leads, onset_date(year, cell, shift=3 * (m - 1)))
        xr.Dataset({"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
                   coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                           "day": np.arange(1, LEAD_DAYS + 1), "lat": lats, "lon": lons}).to_netcdf(folder / f"{year}.nc")


@pytest.mark.parametrize("region", [None, "India"])
def test_a_blend_reads_each_source_on_units_so_grids_that_differ_are_refused_with_a_fix(tmp_path, region):
    """A blend reads rain on evaluation units, not through the evaluation grid: cells of different grids have no common
    units, which says what to do (a shapefile or weights, or one grid), with and without a region."""
    _write_observations(tmp_path / "obs")
    _write_forecasts(tmp_path / "ens", fine=True)
    rc = attach(converted(config_dict(tmp_path), cache_dir=tmp_path / "cache", region=region).rc())
    assert rc.area.evaluation_grid is None            # units runs have none: each source is read on the units
    with pytest.raises(Exception, match=r"different grids.*Fix: give units a shapefile or weights"):
        train(rc)


def test_listed_forecast_files_have_their_grids_looked_at(tmp_path):
    """Sources on one grid have no evaluation grid, and the grids of listed files can be looked at (attach)."""
    _write_observations(tmp_path / "obs")
    _write_forecasts(tmp_path / "ens", fine=False)
    rc = converted(config_dict(tmp_path), cache_dir=tmp_path / "cache").rc()
    assert rc.models[0].paths and attach(rc).area.evaluation_grid is None
    grids = source_grids(rc)
    assert set(grids) == {"observations", "ens"} and grids["ens"][0].tolist() == LATS
