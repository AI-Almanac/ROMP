"""python -m momp.onset explain and schematic."""
import numpy as np
import pandas as pd
import xarray as xr

from momp.onset.__main__ import main


def write_obs(path):
    """One year of seasonal rain on a 2 x 2 grid as a NetCDF file."""
    time = pd.date_range("2001-01-01", "2001-12-31")
    rng = np.random.default_rng(1)
    chance = 0.08 + 0.5 * np.clip(np.sin((time.dayofyear.to_numpy() - 100) / 365 * 2 * np.pi), 0, None)
    values = np.where(rng.random((len(time), 2, 2)) < chance[:, None, None], rng.gamma(0.9, 8.0, (len(time), 2, 2)), 0.0)
    xr.Dataset({"pr": (("time", "lat", "lon"), values)}, coords={"time": time, "lat": [8.0, 9.0], "lon": [38.0, 39.0]}
               ).to_netcdf(path)


def test_explain_prints_the_log_and_saves_the_plot(tmp_path, capsys):
    """The log ends with the result and the PNG is written."""
    obs = tmp_path / "obs.nc"
    write_obs(obs)
    png = tmp_path / "a.png"
    code = main(["explain", "--obs", str(obs), "--obs-var", "pr", "--day-boundary", "00 UTC, labelled by start",
                 "--definition", "icpac", "--lat", "9", "--lon", "38", "--year", "2001", "--png", str(png), "--dark"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Result:" in out and "Place: lat 9, lon 38" in out
    assert png.stat().st_size > 1000


def test_explain_needs_threshold_years_for_a_threshold_definition(tmp_path, capsys):
    """A definition with thresholds and no --threshold-years is a clear error, exit 1."""
    obs = tmp_path / "obs.nc"
    write_obs(obs)
    code = main(["explain", "--obs", str(obs), "--obs-var", "pr", "--day-boundary", "00 UTC, labelled by start",
                 "--definition", "liebmann_marengo", "--lat", "9", "--lon", "38", "--year", "2001"])
    assert code == 1
    assert "--threshold-years" in capsys.readouterr().err


def test_schematic_saves_a_png(tmp_path):
    """The schematic command writes the figure."""
    png = tmp_path / "s.png"
    assert main(["schematic", "icpac", "--png", str(png)]) == 0
    assert png.stat().st_size > 1000
