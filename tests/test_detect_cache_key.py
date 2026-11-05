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


def _obs_folder(folder, lat, lon, missing=False):
    """Write one year of observations on the given axes (all missing when ``missing``)."""
    folder.mkdir(parents=True)
    rain = np.random.default_rng(1).gamma(0.5, 6.0, (6, len(lat), len(lon))).astype("float32")
    if missing:
        rain[:] = np.nan
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


def test_a_detection_with_problems_is_not_cached_but_one_with_warnings_is(tmp_path, monkeypatch):
    """A failed or partial detection is recomputed; warnings alone do not stop a result from being cached."""
    from hcwf.cache import Store

    monkeypatch.setenv("HCWF_CACHE_DIR", str(tmp_path / "cache"))
    folder = _obs_folder(tmp_path / "obs", np.arange(-34.0, -30.0, 1.0), np.arange(18.0, 24.0, 1.0))
    stored = []
    real_put = Store.put

    def spy(self, parts, value, *args, **kwargs):
        """Record what is written, then write it."""
        stored.append(value)
        return real_put(self, parts, value, *args, **kwargs)

    monkeypatch.setattr(Store, "put", spy)
    entry = {"path": str(folder), "variable": "precip", "units": "mm", "day_boundary": "00 UTC, labelled by start"}
    failing = {**entry, "units": "furlongs"}   # not a unit: a problem
    first = detect.resolve_entry(failing, "observations")
    assert first.checked and first.problems
    assert stored == []   # nothing written for the failed detection
    detect.resolve_entry(failing, "observations")
    assert stored == []   # and so it is computed again, not read back
    good = detect.resolve_entry(entry, "observations")
    assert good.checked and not good.problems and len(stored) == 1
    again = detect.resolve_entry(entry, "observations")
    assert any(line.how == "detected (cached)" for line in again.lines) and len(stored) == 1
    # a result with warnings only (an all-missing sample) is stored, and replayed with its warnings
    empty = _obs_folder(tmp_path / "empty", np.arange(-34.0, -30.0, 1.0), np.arange(18.0, 24.0, 1.0), missing=True)
    warned = detect.resolve_entry({**entry, "path": str(empty)}, "observations")
    assert warned.warnings and not warned.problems and len(stored) == 2
    replay = detect.resolve_entry({**entry, "path": str(empty)}, "observations")
    assert replay.warnings == warned.warnings and len(stored) == 2
