"""Synthetic NetCDF inputs of the onset_blending input tests: yearly observations, forecast files with members and
lead days, an admin-unit (adm3_name) file and weight tables. Imports nothing of momp."""
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

LATS, LONS = [10.0, 10.25, 10.5], [70.0, 70.25]
DAYS = 12


def rain_field(rng, shape):
    """Rain-like values (many zeros, some NaN and a few tiny negatives), float32."""
    data = np.where(rng.random(shape) < 0.4, 0.0, rng.gamma(1.5, 4.0, shape)).astype("float32")
    data[rng.random(shape) < 0.03] = np.nan
    data[rng.random(shape) < 0.02] = -0.3
    return data


def write_observations(folder, years=(2003, 2004), seed=0, gap_in=None, name=None, units="mm"):
    """{year}.nc files of (time, lat, lon) daily rain over Jan 1 .. Jan 20; gap_in drops one day of that year."""
    rng = np.random.default_rng(seed)
    folder.mkdir(exist_ok=True)
    for year in years:
        times = pd.date_range(f"{year}-01-01", periods=20, freq="D")
        data = rain_field(rng, (len(times), len(LATS), len(LONS)))
        ds = xr.Dataset({"precip": (("time", "lat", "lon"), data, {"units": units} if units else {})},
                        coords={"time": times, "lat": LATS, "lon": LONS})
        if gap_in == year:
            ds = ds.isel(time=[i for i in range(len(times)) if i != 4])
        ds.to_netcdf(folder / (name(year) if name else f"{year}.nc"))
    return folder


def write_forecast(path, init, members=3, seed=1, units="mm", lead_dim="day", member_dim="number", time_dim="time",
                   days=DAYS, chunks=None, lats=LATS, lons=LONS):
    """One forecast file with the inits in `init` (dates), dims (time, member, day, lat, lon)."""
    rng = np.random.default_rng(seed)
    shape = (len(init), members, days, len(lats), len(lons))
    data = rain_field(rng, shape) * (0.001 if units == "m" else 1.0)
    attrs = {"units": units} if units else {}
    ds = xr.Dataset({"tp": ((time_dim, member_dim, lead_dim, "lat", "lon"), data, attrs)},
                    coords={time_dim: pd.to_datetime(init), member_dim: np.arange(members),
                            lead_dim: np.arange(1, days + 1), "lat": lats, "lon": lons})
    encoding = {"tp": {"chunksizes": chunks}} if chunks else None
    ds.to_netcdf(path, encoding=encoding)
    return path


def write_adm3(path, names=("north", "south", "East ward"), seed=2, units="mm"):
    """An admin-unit observation file: (time, adm3_name) with string names, Jan 1 .. Jan 20 of 2003."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2003-01-01", periods=20, freq="D")
    data = rain_field(rng, (len(times), len(names)))
    xr.Dataset({"precip": (("time", "adm3_name"), data, {"units": units})},
               coords={"time": times, "adm3_name": list(names)}).to_netcdf(path)
    return path


def grid_weights() -> pd.DataFrame:
    """Coordinate-keyed weights of two units over the 3 x 2 grid (north: 3 cells, south: 3 cells, one shared)."""
    rows = [("north", 10.0, 70.0, 0.5), ("north", 10.0, 70.25, 0.25), ("north", 10.25, 70.0, 1.0),
            ("south", 10.25, 70.0, 0.5), ("south", 10.25, 70.25, 1.0), ("south", 10.5, 70.25, 2.0)]
    return pd.DataFrame(rows, columns=["target_id", "lat", "lon", "weight"])


def id_weights() -> pd.DataFrame:
    """The same weights keyed by "<lat>_<lon>" source ids (two decimals)."""
    table = grid_weights()
    table["source_id"] = [f"{a:.2f}_{b:.2f}" for a, b in zip(table["lat"], table["lon"])]
    return table[["source_id", "target_id", "weight"]]
