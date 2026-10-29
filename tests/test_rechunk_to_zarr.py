"""Synthetic NetCDF -> zarr round trips for scripts/rechunk_to_zarr.py."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

_spec = importlib.util.spec_from_file_location(
    "rechunk_to_zarr", Path(__file__).resolve().parents[1] / "scripts" / "rechunk_to_zarr.py"
)
rz = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rz
_spec.loader.exec_module(rz)

LAT = np.linspace(20, 10, 7)  # descending, as in the real forecasts
LON = np.linspace(70, 80, 9)


def _write_src(root: Path):
    rng = np.random.default_rng(0)
    for year in (2000, 2001, 2002):
        t = pd.date_range(f"{year}-01-01", periods=10 + year % 5, freq="D")
        v = rng.gamma(0.5, 3.0, (len(t), 7, 9))
        v[rng.random(v.shape) < 0.3] = np.nan
        v = np.round(v, 1).astype("float32").astype("float64")  # float32-exact, like IMD
        d = root / "india" / "obs"
        d.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), v, {"units": "mm"})},
                   {"TIME": t, "LATITUDE": LAT, "LONGITUDE": LON}, attrs={"history": "x"}
                   ).to_netcdf(d / f"data_{year}.nc", encoding={"RAINFALL": {"_FillValue": -999.0}})
        ti = pd.date_range(f"{year}-05-01", periods=9, freq="4D")
        f = rng.random((9, 3, 5, 7, 9)).astype("float32")
        f[0, 0, 0] = np.nan
        d = root / "india" / "models" / "ENS"
        d.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"tp": (("time", "number", "lead", "lat", "lon"), f, {"units": "m"})},
                   {"time": ti, "number": np.arange(3), "lead": pd.to_timedelta(np.arange(1, 6), "D"),
                    "lat": LAT, "lon": LON}).to_netcdf(d / f"{year}.nc")


def _run(src, dst, *extra):
    return rz.main(["--src", str(src), "--dst", str(dst), "--workers", "1", "--target-mb", "0.002", "--block-mb", "0.004", *extra])


@pytest.fixture()
def tree(tmp_path):
    _write_src(tmp_path / "src")
    return tmp_path / "src", tmp_path / "dst"


def test_roundtrip_exact(tree):
    src, dst = tree
    assert _run(src, dst, "--verify", "--verify-sample", "1") == 0
    obs = xr.open_dataset(dst / "india/obs.zarr", engine="zarr", consolidated=True)
    ref = xr.concat([xr.open_dataset(f) for f in sorted((src / "india/obs").glob("*.nc"))], "TIME")
    assert obs.RAINFALL.dtype == np.float32
    assert obs.RAINFALL.attrs["units"] == "mm" and obs.attrs["history"] == "x"
    np.testing.assert_array_equal(obs.RAINFALL.values, ref.RAINFALL.values)  # NaNs equal too
    np.testing.assert_array_equal(obs.LATITUDE.values, LAT)  # not flipped
    assert obs.RAINFALL.encoding["chunks"][0] == obs.sizes["TIME"]  # full time axis per chunk
    fc = xr.open_dataset(dst / "india/models/ENS.zarr", engine="zarr", consolidated=True)
    ref = xr.concat([xr.open_dataset(f) for f in sorted((src / "india/models/ENS").glob("*.nc"))], "time")
    np.testing.assert_array_equal(fc.tp.values, ref.tp.values)
    np.testing.assert_array_equal(fc.lead.values, ref.lead.values)
    c = fc.tp.encoding["chunks"]
    assert c[1] == 3 and c[2] == 5  # all members and the full lead axis in every chunk
    meta = fc.attrs[rz.ATTR_KEY]
    assert meta["complete"] and len(meta["source_files"]) == 3 and meta["chunks"] == list(c)


def test_resume_and_skip(tree):
    src, dst = tree
    assert _run(src, dst, "--only", "india/models/ENS") == 0
    store = dst / "india/models/ENS.zarr"
    ref = xr.open_dataset(store, engine="zarr", consolidated=True).tp.values
    zj = store / "zarr.json"
    mtime = zj.stat().st_mtime_ns
    assert _run(src, dst, "--only", "india/models/ENS") == 0  # finished store is skipped
    assert zj.stat().st_mtime_ns == mtime
    # simulate a crash: mark incomplete, drop one task marker and its chunks
    g = zarr.open_group(str(store), mode="r+", use_consolidated=False)
    meta = dict(g.attrs[rz.ATTR_KEY])
    meta["complete"] = False
    g.attrs[rz.ATTR_KEY] = meta
    markers = sorted((Path(str(store) + ".progress")).glob("*.json"))
    assert len(markers) > 1
    markers[0].unlink()
    for ch in (store / "tp" / "c").iterdir():
        if ch.name == "0":
            for p in ch.rglob("*"):
                if p.is_file():
                    p.unlink()
    assert _run(src, dst, "--only", "india/models/ENS", "--verify") == 0
    np.testing.assert_array_equal(xr.open_dataset(store, engine="zarr", consolidated=True).tp.values, ref)
    assert json.loads((dst / rz.STATS_FILE).read_text())["india/models/ENS"]["verify"]["ok"]


def test_changed_plan_needs_overwrite(tree):
    src, dst = tree
    assert _run(src, dst, "--only", "india/obs") == 0
    with pytest.raises(SystemExit):
        rz.main(["--src", str(src), "--dst", str(dst), "--workers", "1", "--only", "india/obs", "--target-mb", "0.003"])
    assert rz.main(["--src", str(src), "--dst", str(dst), "--workers", "1", "--only", "india/obs",
                    "--target-mb", "0.003", "--overwrite"]) == 0


def test_verify_detects_corruption(tree):
    src, dst = tree
    assert _run(src, dst, "--only", "india/obs") == 0
    z = zarr.open_group(str(dst / "india/obs.zarr"), mode="r+", use_consolidated=False)["RAINFALL"]
    z[3, 2, 2] = 12345.0
    assert _run(src, dst, "--only", "india/obs", "--verify") == 1


def test_chunk_choice():
    assert rz.choose_chunks((1000, 129, 135), ("t", "y", "x"), 4, "t", 64 * 2**20)[0] == 1000
    c = rz.choose_chunks((1118, 26, 50, 129, 135), tuple("tmlyx"), 4, "t", 64 * 2**20)
    assert c[:3] == (1, 26, 50) and np.prod(c) * 4 <= 64 * 2**20
    c = rz.choose_chunks((1118, 50, 129, 135), tuple("tlyx"), 4, "t", 64 * 2**20)
    assert c == (8, 50, 129, 135)
