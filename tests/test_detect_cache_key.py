"""The detection cache: its key follows the area the data are sampled in, and a failed detection is not stored."""
import numpy as np
import pandas as pd
import xarray as xr

import pytest

from hcwf.cache import reset_run_state
from hcwf.config import detect


@pytest.fixture(autouse=True)
def fresh():
    """Start every test without the state of a previous run."""
    reset_run_state()
    yield
    reset_run_state()


def _obs_folder(folder, lat, lon):
    """Write one year of observations on the given axes."""
    folder.mkdir(parents=True)
    rain = np.random.default_rng(1).gamma(0.5, 6.0, (6, len(lat), len(lon))).astype("float32")
    xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm/day"})},
               coords={"time": pd.date_range("2020-05-01", periods=6), "lat": lat, "lon": lon}
               ).to_netcdf(folder / "2020.nc")
    return folder


def test_the_detection_cache_key_changes_with_the_area(tmp_path):
    """The sampled place depends on the region and the box: each gets its own key, equal areas share one."""
    folder = _obs_folder(tmp_path / "obs", np.arange(-34.0, -30.0, 1.0), np.arange(18.0, 24.0, 1.0))
    entry = {"path": str(folder)}
    keys = {name: detect._store().fingerprint(detect._parts(entry, "observations", False, None, region, box))
            for name, (region, box) in {"none": (None, None), "ethiopia": ("ethiopia", None),
                                        "kenya": ("kenya", None), "box": (None, (3.0, 15.0, 33.0, 48.0)),
                                        "box2": (None, (4.0, 15.0, 33.0, 48.0)),
                                        "box_same": (None, [3, 15, 33, 48]),
                                        "region_box": ("ethiopia", (3.0, 15.0, 33.0, 48.0))}.items()}
    assert len({keys[k] for k in ("none", "ethiopia", "kenya", "box", "box2", "region_box")}) == 6
    assert keys["box"] == keys["box_same"]
