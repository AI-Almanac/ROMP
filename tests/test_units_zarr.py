"""Polygon units read from zarr stores (one store of all years) give what the yearly NetCDF files give, with either
observed-history setting."""
import numpy as np
import pytest
import xarray as xr

geopandas = pytest.importorskip("geopandas")

import hcwf  # noqa: E402
from test_units_benchmark import _raw, _write_data  # noqa: E402


def _stores(base):
    """Write the yearly files of ``_write_data`` as one zarr store each of the observations and of the deterministic model."""
    for name, folder in (("obs", "obs"), ("DET", "DET")):
        files = sorted((base / folder).glob("*.nc"))
        xr.concat([xr.open_dataset(f).load() for f in files], dim="time").to_zarr(base / f"{name}.zarr")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("units_zarr")
    _write_data(base)
    _stores(base)
    return base


@pytest.mark.parametrize("history", ["observed", "none"])
def test_units_from_zarr_stores_equal_units_from_files(world, history):
    def config(zarr):
        cfg = _raw(world, "DET", False)
        cfg["output"] = {"save_fig": False}
        cfg["onset"] = {**cfg["onset"], "forecast_history": history}
        if zarr:
            cfg["observations"]["path"] = str(world / "obs.zarr")
            cfg["models"][0]["path"] = str(world / "DET.zarr")
        return cfg

    files = hcwf.run(config(False), quiet=True)
    store = hcwf.run(config(True), quiet=True)
    assert set(store.scores["spatial"]) == set(files.scores["spatial"])
    for key, ds in files.scores["spatial"].items():
        other = store.scores["spatial"][key]
        assert other["mean_mae"].dims == ("unit",)
        np.testing.assert_allclose(other["mean_mae"].values, ds["mean_mae"].values, rtol=1e-9)
    assert (store.observed["onset_date"].values == files.observed["onset_date"].values).all()
