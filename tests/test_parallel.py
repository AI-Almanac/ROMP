"""Parallel years give the serial answer; the parallel.workers config key; threaded reads of NetCDF stay safe."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.lib.config_schema import ConfigSchemaError
from momp.lib.config_yaml import build_config
from momp.stats import parallel
from momp.utils.workers import cores, ordered_prefetch, worker_count
from tests.test_config_schema import _raw, data  # noqa: F401 - the synthetic-data fixture

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data" / "phase7" / "baseline"


def square_years(*, years, offset=0, **kwargs):
    """A picklable stand-in for a multi-year function: one table of the squares of the years."""
    return pd.DataFrame({"year": list(years), "square": [y * y + offset for y in years]})


def test_worker_count_rules():
    """parallel false is serial; otherwise workers are the cores, the configured number, never above the tasks."""
    assert worker_count({"parallel": False, "parallel_workers": 8}, 10) == 1
    assert worker_count({"parallel": True}, 1) == 1
    assert worker_count({"parallel": True}, 3) == min(3, cores())
    assert worker_count({"parallel": True, "parallel_workers": 2}, 10) == 2
    assert worker_count({"parallel": True, "parallel_workers": 50}, 4) == 4
    assert worker_count({}, 4) == min(4, cores())   # on by default


def test_config_key_workers(data):  # noqa: F811
    """`parallel: {workers: N}`, `parallel_workers` and `parallel: false` all resolve; bad values are errors."""
    assert build_config(_raw(data))["parallel"] is True
    assert build_config(_raw(data)).get("parallel_workers") is None
    flat = build_config(_raw(data, parallel={"workers": 3}))
    assert flat["parallel"] is True and flat["parallel_workers"] == 3
    assert build_config(_raw(data, parallel=False))["parallel"] is False
    assert build_config(_raw(data, parallel_workers=2))["parallel_workers"] == 2
    for bad in ({"workers": 0}, {"workers": "x"}, {"wokers": 2}):
        with pytest.raises(ConfigSchemaError):
            build_config(_raw(data, parallel=bad))


def test_ordered_prefetch_keeps_order_and_bounds_the_lead():
    """Results come in item order whatever the thread timing; the lead over the consumer is bounded."""
    import time

    started = []

    def slow(n):
        """Later items finish sooner."""
        started.append(n)
        time.sleep(0.02 * (10 - n))
        return n * 2

    got = []
    for value in ordered_prefetch(slow, range(10), 4, ahead=3):
        got.append(value)
        assert len(started) <= len(got) + 3
    assert got == [n * 2 for n in range(10)]
    assert list(ordered_prefetch(slow, [], 4)) == []


def test_map_years_parallel_equals_serial():
    """Worker processes return the same tables in year order as the serial path."""
    years = [2001, 2002, 2003, 2004]
    serial = parallel.concat_years(square_years, {"years": years, "parallel": False, "offset": 1})
    par = parallel.concat_years(square_years, {"years": years, "parallel": True, "parallel_workers": 3, "offset": 1})
    pd.testing.assert_frame_equal(serial, par)
    assert list(par["year"]) == years


def test_threaded_obs_reads_of_netcdf_are_serialised(tmp_path):
    """obs_years over many NetCDF files (parallel on) neither crashes nor reorders; a missing year is an error value."""
    from momp.io.input import obs_years

    folder = tmp_path / "obs"
    folder.mkdir()
    lat, lon = np.linspace(10, 22, 4), np.linspace(72, 84, 4)
    for year in range(2001, 2009):
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        rain = np.full((len(time), 4, 4), float(year))
        xr.Dataset({"RAINFALL": (("time", "lat", "lon"), rain)}, coords={"time": time, "lat": lat, "lon": lon}
                   ).to_netcdf(folder / f"{year}.nc")
    kwargs = {"obs_dir": str(folder), "obs_file_pattern": ("{}.nc",), "obs_var": "RAINFALL", "obs_unit_cvt": None,
              "region": "India", "parallel": True, "years": (), "land_only": False, "shp_only": False}
    out = list(obs_years(list(range(2001, 2009)), kwargs, hold=False))
    assert [float(rain["RAINFALL"].values.flat[0]) if hasattr(rain, "data_vars") else float(rain.values.flat[0])
            for rain, error in out] == [float(y) for y in range(2001, 2009)]
    missing = list(obs_years([2008, 2030], kwargs, hold=False))
    assert missing[0][1] is None and isinstance(missing[1][1], FileNotFoundError)


@pytest.mark.skipif(not (BASELINE / "demo_mwset_new.yaml").exists(), reason="baseline demo data not present")
def test_run_parallel_equals_serial(tmp_path):
    """A whole run gives byte-identical files with parallel on and off."""
    outs = {}
    for flag in ("false", "true"):
        text = (BASELINE / "demo_mwset_new.yaml").read_text()
        out = tmp_path / flag
        text = text.replace("parallel: false", f"parallel: {flag}")
        text = "\n".join(f"  data_dir: {out}" if l.startswith("  data_dir:") else
                         f"  figure_dir: {out}/fig" if l.startswith("  figure_dir:") else l
                         for l in text.splitlines())
        cfg = tmp_path / f"{flag}.yaml"
        cfg.write_text(text)
        subprocess.run([str(ROOT / ".venv/bin/momp-run"), "-p", str(cfg), "--quiet", "--overwrite"], check=True,
                       cwd=BASELINE / "cwd", env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"})
        outs[flag] = out
    files = sorted(p.name for p in outs["false"].glob("*.nc")) + sorted(p.name for p in outs["false"].glob("*.csv"))
    assert files
    for name in files:
        assert (outs["false"] / name).read_bytes() == (outs["true"] / name).read_bytes(), name
