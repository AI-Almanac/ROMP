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


def noisy_years(*, years, **kwargs):
    """A picklable stand-in that warns and logs in the worker, one table row per year."""
    import logging
    import warnings

    for year in years:
        warnings.warn(f"odd data {year}")
        logging.getLogger("momp.test_worker").info("hello from %s", year)
    return pd.DataFrame({"year": list(years)})


@pytest.mark.parametrize("quiet", [False, True])
def test_worker_warnings_and_logs_reach_run_log_and_console(tmp_path, capsys, quiet):
    """A worker's warning and info record each appear once in run.log (with the year) and on the console as asked."""
    from momp.utils import log as logs

    logfile = tmp_path / "run.log"
    with logs.session(logs.QUIET if quiet else logs.NORMAL, logfile):
        parallel.concat_years(noisy_years, {"years": [2001, 2002, 2003], "parallel": True, "parallel_workers": 3})
        parallel.shutdown()   # the workers outlive the call (the run's pool); stopping them flushes their records
    text = logfile.read_text()
    out = capsys.readouterr().out
    for year in (2001, 2002, 2003):   # every year runs in a worker and warns through the forwarded handlers
        assert text.count(f"[year {year}] odd data {year}") == 1 and "[WARNING] momp.warnings" in text
        assert text.count(f"[year {year}] hello from {year}") == 1 and "momp.test_worker" in text
        assert out.count(f"warning: [year {year}] odd data {year}") == 1
        assert (f"[year {year}] hello from {year}" in out) == (not quiet)


def test_legacy_threshold_reloaded_each_window_detects_once(monkeypatch):
    """Equal thresholds loaded afresh per window share onset tables; a different threshold does not."""
    from momp.stats import benchmark

    calls = []
    monkeypatch.setattr(benchmark, "map_years", lambda *a, **k: calls.append(1) or [{}])
    fresh = {"value": 20.0}
    monkeypatch.setattr(benchmark, "load_thresholds",
                        lambda **kw: xr.DataArray(np.full((2, 2), fresh["value"]), dims=("lat", "lon"),
                                                  coords={"lat": [1.0, 2.0], "lon": [3.0, 4.0]}))
    benchmark.reset_onset_tables()
    common = {"years": [2001, 2002], "model": "m", "probabilistic": False}
    benchmark.onset_tables(verification_window=5, **common)
    benchmark.onset_tables(verification_window=10, **common)
    assert len(calls) == 1
    fresh["value"] = 25.0
    benchmark.onset_tables(verification_window=5, **common)
    assert len(calls) == 2
    benchmark.reset_onset_tables()


def pid_years(*, years, **kwargs):
    """A picklable stand-in that reports which process computed each year."""
    import os

    return pd.DataFrame({"year": list(years), "pid": os.getpid()})


def test_every_year_runs_in_a_worker_and_the_pool_is_reused():
    """No year runs in this process; a second stage gets the same booted workers; shutdown ends the pool."""
    import os

    years = [2001, 2002, 2003]
    config = {"years": years, "parallel": True, "parallel_workers": 3}
    try:
        first = parallel.concat_years(pid_years, config)
        pool = parallel._pool["executor"]
        second = parallel.concat_years(pid_years, config)
        assert parallel._pool["executor"] is pool
        assert os.getpid() not in set(first["pid"]) | set(second["pid"])
        assert set(second["pid"]) <= set(pool._processes)   # the workers booted for the first stage
    finally:
        parallel.shutdown()
    assert parallel._pool["executor"] is None


def test_prewarm_runs_here_once_before_the_years_and_start_returns_before_they_finish():
    """prewarm(kwargs) is called once, in this process, with all the years; start_years hands back a Pending."""
    import os

    calls = []
    config = {"parallel": True, "parallel_workers": 2, "offset": 5}
    try:
        pending = parallel.start_years(square_years, [2001, 2002, 3003], config,
                                       prewarm=lambda kw: calls.append((os.getpid(), kw["years"], kw["offset"])))
        assert calls == [(os.getpid(), [2001, 2002, 3003], 5)]
        tables = pending.result()
        assert pending.result() is tables
        assert [list(t["year"]) for t in tables] == [[2001], [2002], [3003]]
        assert [int(t["square"][0]) for t in tables] == [2001 ** 2 + 5, 2002 ** 2 + 5, 3003 ** 2 + 5]
    finally:
        parallel.shutdown()


def test_serial_start_runs_at_once_and_prewarm_is_skipped():
    """With parallel off the work is done before start_years returns (as before), in order, and nothing is pre-warmed."""
    order = []

    def func(*, years, **kwargs):
        """Note the year, return it."""
        order.append(years[0])
        return years[0]

    pending = parallel.start_years(func, [3, 1, 2], {"parallel": False}, prewarm=lambda kw: order.append("warm"))
    assert order == [3, 1, 2]
    assert pending.result() == [3, 1, 2]


def failing_year(*, years, **kwargs):
    """A picklable stand-in that fails for one year."""
    if years[0] == 2002:
        raise KeyError("boom 2002")
    return pd.DataFrame({"year": list(years)})


def test_worker_error_is_raised_by_result_and_the_pool_stays_usable():
    """An error in a worker surfaces from result(); the pool then still serves the next stage."""
    config = {"years": [2001, 2002, 2003], "parallel": True, "parallel_workers": 3}
    try:
        pending = parallel.start_concat(failing_year, config)
        with pytest.raises(KeyError, match="boom 2002"):
            pending.result()
        again = parallel.concat_years(square_years, config)
        assert list(again["year"]) == [2001, 2002, 2003]
    finally:
        parallel.shutdown()
