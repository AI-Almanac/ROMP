"""Detection reads one contiguous sample per dataset (also from zarr), and the land mask is built once per grid."""
import numpy as np
import pytest
import xarray as xr

from momp.lib import dataset_detect as detect
from momp.utils import land_mask



def _forecast(n_init=30):
    """A small forecast DataArray (init, step, lat, lon) of daily metres."""
    rng = np.random.default_rng(0)
    data = rng.gamma(0.5, 0.004, size=(n_init, 6, 5, 7)).astype("float32")
    return xr.DataArray(data, dims=("time", "step", "lat", "lon"), name="tp",
                        coords={"time": np.datetime64("2020-05-01") + np.arange(n_init).astype("timedelta64[D]"),
                                "step": np.arange(1, 7) * np.timedelta64(1, "D"),
                                "lat": np.linspace(5, 9, 5), "lon": np.linspace(36, 42, 7)},
                        attrs={"units": "m"})


def test_load_sample_blocks_and_cache():
    """The sample is whole consecutive blocks from start, middle and end, loaded once and cached by key."""
    detect.clear_samples()
    da = _forecast()
    roles = {"init_time": "time", "step": "step", "lat": "lat", "lon": "lon"}
    out = detect.load_sample(da, roles, key=("f", 1, "tp"))
    assert out.sizes["step"] == 6 and out.sizes["lat"] == 5
    assert out.sizes["time"] == 3 * detect._SAMPLE_BLOCK_LEN
    assert out.time.values[0] == da.time.values[0] and out.time.values[-1] == da.time.values[-1]
    assert detect.load_sample(da, roles, key=("f", 1, "tp")) is out
    assert detect.load_sample(da, roles, key=("g", 1, "tp")) is not out


def test_block_indices_small_and_budget(monkeypatch):
    """A short dim is read whole; a tight budget shrinks the blocks to one index."""
    assert len(detect._block_indices(5, 10)) == 1
    monkeypatch.setattr(detect, "_SAMPLE_BUDGET", 10)
    assert all(len(b) == 1 for b in detect._block_indices(100, 1000))


def test_open_data_zarr_and_resolve(tmp_path):
    """A .zarr store opens with xr.open_zarr and detection resolves it like NetCDF."""
    pytest.importorskip("zarr")
    detect.clear_samples()
    da = _forecast()
    folder = tmp_path / "fc"
    folder.mkdir()
    da.to_dataset().to_zarr(folder / "2020.zarr")
    res = detect.resolve_entry({"path": str(folder), "dims": {"init_time": "time", "step": "step", "lat": "lat",
                                                              "lon": "lon"}}, "forecast", detect_lead_label=False)
    assert res.values["file_pattern"] == "{}.zarr"
    assert res.values["variable"] == "tp" and res.values["units"] == "m"
    assert res.checked


def test_resolve_netcdf_reads_once(tmp_path, monkeypatch):
    """Units, magnitude and accumulation checks share a single load_sample read."""
    detect.clear_samples()
    folder = tmp_path / "fc"
    folder.mkdir()
    _forecast().to_dataset().to_netcdf(folder / "2020.nc")
    calls = []
    real = xr.DataArray.load
    monkeypatch.setattr(xr.DataArray, "load", lambda self, **k: calls.append(1) or real(self, **k))
    res = detect.resolve_entry({"path": str(folder)}, "forecast", detect_lead_label=False)
    assert res.values["units"] == "m" and not res.problems.get("units")
    assert len(calls) == 3  # three blocks, no further reads


def test_land_mask_cached_and_identical():
    """land_mask_for is built once per grid, read-only, and agrees with mask_land / create_land_sea_mask."""
    land_mask.clear_land_masks()
    lat, lon = np.linspace(-10, 40, 26), np.linspace(-20, 60, 41)
    a = land_mask.land_mask_for(lat, lon)
    assert land_mask.land_mask_for(lat.copy(), lon.copy()) is a
    assert a.dtype == bool and a.shape == (26, 41) and not a.flags.writeable
    da = xr.DataArray(np.ones((26, 41)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    assert (land_mask.create_land_sea_mask(da).values == a).all()
    assert (land_mask.create_land_sea_mask(da, as_boolean=True).values == a).all()
    kept = land_mask.mask_land(da)
    assert (np.isfinite(kept.values) == a).all()
    assert (np.isfinite(land_mask.mask_land(da, land=False).values) == ~a).all()
    assert land_mask.land_mask_for(lat[::-1], lon) is not a
    assert len(land_mask._RAW_MASKS) == 2


def test_disk_cache_hit_reads_no_data(tmp_path, monkeypatch):
    """A second resolution of unchanged files comes from the cache: same values, 'detected (cached)', no reads."""
    monkeypatch.setenv("MOMP_CACHE_DIR", str(tmp_path / "cache"))
    detect.clear_samples()
    folder = tmp_path / "fc"
    folder.mkdir()
    _forecast().to_dataset().to_netcdf(folder / "2020.nc")
    first = detect.resolve_entry({"path": str(folder)}, "forecast", detect_lead_label=False, lead_label_default=True)
    assert first.values["units"] == "m" and all(ln.how != "detected (cached)" for ln in first.lines)

    def boom(*a, **k):
        raise AssertionError("data was read")
    monkeypatch.setattr(detect, "load_sample", boom)
    monkeypatch.setattr(detect, "open_data", boom)
    second = detect.resolve_entry({"path": str(folder)}, "forecast", detect_lead_label=False, lead_label_default=True)
    assert second.values == first.values
    assert [ln.text().replace(" (cached)", "") for ln in second.lines] == [ln.text() for ln in first.lines]
    assert any(ln.how == "detected (cached)" for ln in second.lines)
    # a changed file or key misses
    (folder / "2020.nc").touch()
    monkeypatch.undo()
    monkeypatch.setenv("MOMP_CACHE_DIR", str(tmp_path / "cache"))
    third = detect.resolve_entry({"path": str(folder)}, "forecast", detect_lead_label=False, lead_label_default=True)
    assert all(ln.how != "detected (cached)" for ln in third.lines)
