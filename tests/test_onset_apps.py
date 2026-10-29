"""The onset apps (spatial map and time series) run with a definition-mode config and with the old flat keys."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.app import onset_spatial, onset_time_series
from momp.io import input as romp_input

BOUNDARY = "00 UTC, labelled by start"


@pytest.fixture
def obs(tmp_path, monkeypatch):
    """A tiny 2 x 2 observation file for 2020 (daily rain, a wet spell from 1 June) and a stand-in region selection."""
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    time = pd.date_range("2020-01-01", "2020-12-31")
    rain = np.zeros((len(time), 2, 2))
    rain[time.dayofyear >= 153] = 20.0
    xr.Dataset({"pr": (("time", "lat", "lon"), rain)},
               coords={"time": time, "lat": [8.0, 9.0], "lon": [38.0, 39.0]}).to_netcdf(obs_dir / "2020.nc")
    monkeypatch.setattr(romp_input, "region_select", lambda ds, **kwargs: ds)
    return obs_dir


def base(obs, tmp_path):
    """The config keys both modes share."""
    return dict(obs_dir=str(obs), obs_file_pattern=["{}.nc"], obs_var="pr", obs_unit_cvt=None, region="x",
                dir_out=str(tmp_path / "out"), dir_fig=str(tmp_path), verification_window_list=[(1, 10)],
                obs_day_boundary=BOUNDARY, start_date=(2020, 5, 1), end_date=(2020, 7, 31),
                start_year_clim=2020, end_year_clim=2020)


def definition_cfg(obs, tmp_path):
    """A definition-mode config: onset_definition and no flat onset keys."""
    return {**base(obs, tmp_path), "onset_legacy": False, "onset_definition": "icpac"}


def legacy_cfg(obs, tmp_path):
    """A legacy config: the old flat onset keys and no onset_definition."""
    return {**base(obs, tmp_path), "onset_legacy": True, "thresh_file": None, "thresh_var": None, "wet_threshold": 20,
            "wet_init": 1, "wet_spell": 3, "dry_spell": 7, "dry_threshold": 1, "dry_extent": 0, "fallback_date": None,
            "earliest_onset_date": None}


@pytest.fixture(params=[definition_cfg, legacy_cfg], ids=["definition", "legacy"])
def cfg(request, obs, tmp_path):
    """Each mode's config in turn."""
    return request.param(obs, tmp_path)


def test_spatial_onset_runs_in_both_modes(cfg, monkeypatch):
    """spatial_onset returns the day-of-year map (onset on 1 June, day 153) and hands it to the map plot."""
    mapped = []
    monkeypatch.setattr(onset_spatial, "spatial_metrics_map", lambda da, *args, **kwargs: mapped.append(da))
    doy = onset_spatial.spatial_onset(2020, **cfg)
    assert doy.dims == ("lat", "lon") and mapped
    assert (doy.values == 153).all()


def test_time_series_runs_in_both_modes(cfg, monkeypatch):
    """obs_onset_analysis returns the season's rainfall and the onset dates, without any flat onset key in its signature."""
    plotted = []
    monkeypatch.setattr(onset_time_series, "plot_rainfall_timeseries_with_onset_and_wetspell",
                        lambda *args, **kwargs: plotted.append(kwargs["save_path"]))
    da_sub, onset = onset_time_series.obs_onset_analysis(2020, lat_select=8.0, lon_select=38.0, incl_clim=False, **cfg)
    assert plotted and onset.dims == ("lat", "lon")
    assert (onset.values == np.datetime64("2020-06-01")).all()
    assert da_sub.sizes["time"] > 0


def test_time_series_with_climatology(cfg, monkeypatch):
    """The climatological onset (one year, so equal to 2020's) is computed in both modes too."""
    plotted = []
    monkeypatch.setattr(onset_time_series, "plot_rainfall_timeseries_with_onset_and_wetspell",
                        lambda *args, **kwargs: plotted.append(kwargs["onset_clim"]))
    onset_time_series.obs_onset_analysis(2020, lat_select=8.0, lon_select=38.0, incl_clim=True, **cfg)
    assert plotted == [pd.Timestamp("2020-06-01")]
