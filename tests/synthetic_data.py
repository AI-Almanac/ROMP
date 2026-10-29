"""Tiny synthetic observation and forecast files for the config and dataset-detection tests."""
import numpy as np
import pandas as pd
import xarray as xr

LAT = np.linspace(3, 15, 6)
LON = np.linspace(33, 48, 6)
DAYS = pd.date_range("2020-05-01", periods=60)


def rain_field(seed=0):
    """Daily rain in mm, (time, lat, lon): independent noise, so only the right alignment correlates."""
    rng = np.random.default_rng(seed)
    return rng.gamma(0.5, 8.0, size=(len(DAYS), len(LAT), len(LON)))


def write_obs(folder, name="2020.nc", var="precip", units="mm", extra_vars=(), boundary_attr=None, dims=("time", "lat", "lon")):
    """Write an observation file with rain in mm; returns the array."""
    rain = rain_field()
    coords = {dims[0]: DAYS, dims[1]: LAT, dims[2]: LON}
    ds = xr.Dataset({var: (dims, rain, {"units": units} if units else {})}, coords=coords)
    for extra in extra_vars:
        ds[extra] = (dims, rain * 0 + 1)
    if boundary_attr:
        ds.attrs["day_boundary"] = boundary_attr
    folder.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(folder / name)
    return rain


def write_forecast(folder, label="start", name="2020.nc", var="tp", units="m", members=0, scale=1e-3,
                   init_dim="time", step_dim="step"):
    """Write a forecast file whose lead k holds the observation of day init + k - 1 (label 'start') or init + k."""
    rain = rain_field()
    inits = DAYS[:30]
    steps = np.arange(1, 4)
    shift = 0 if label == "start" else 1
    data = np.stack([[rain[i + k - 1 + shift] for k in steps] for i in range(len(inits))])
    dims = (init_dim, step_dim, "lat", "lon")
    coords = {init_dim: inits, step_dim: steps, "lat": LAT, "lon": LON}
    if members:
        data = np.stack([data * (1 + 0.01 * m) for m in range(members)], axis=1)
        dims = (init_dim, "number", step_dim, "lat", "lon")
        coords["number"] = np.arange(members)
    ds = xr.Dataset({var: (dims, data * scale, {"units": units} if units else {})}, coords=coords)
    folder.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(folder / name)
