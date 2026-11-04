"""Shared test setup: safe under ``pytest -n auto`` (pytest-xdist).

* Thread pools (numba, BLAS, OpenMP) are sized per worker, so N workers do not each start one thread per core
  (that oversubscription made the parallel run slower than the serial one). Set before numpy/numba are imported.
* Every test gets its own HCWF_CACHE_DIR, so no test reads or writes ~/.cache/hcwf or another worker's entries.
* Tests that touch the GPU (anything with "gpu" in its id) share one xdist group, so they run on one worker
  with ``--dist loadgroup`` and do not contend for the single device.
* Modules with an expensive module-scoped fixture are put in one xdist group each (``GROUPED_MODULES``), so the
  fixture is built once, on one worker, instead of once per worker that happens to get one of the module's tests.
* Natural Earth shapefiles that the code reads are fetched once, by the session controller under a file lock, before
  any worker starts (cartopy downloads on first use, and parallel workers raced on a half-written file).
* Tests marked ``slow`` run only with ``--run-slow`` (the full run is ``pytest -n 10 --run-slow``). Tests marked
  ``bench`` (timing benchmarks: they check speed, not results, and fail under load) run only with ``--run-bench``,
  best alone on an idle machine.
"""
import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    _threads = str(max(1, (os.cpu_count() or 2) // max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))))
else:
    _threads = None
if _threads:
    for _var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, _threads)

import pytest


@pytest.fixture(autouse=True)
def _isolated_hcwf_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("HCWF_CACHE_DIR", str(tmp_path_factory.mktemp("hcwf_cache")))


# Test modules (path relative to tests/) whose module- or session-scoped fixtures take over ~1 s to build.
GROUPED_MODULES = {
    "blend/test_blend_forecast.py",
    "blend/test_blend_training.py",
    "blend/test_blend_training_differential.py",
    "test_config_entry.py",
    "test_export_onsets.py",
    "test_metrics_typed.py",
    "test_observed_fingerprint.py",
    "test_onset_hcwf_explain_cells.py",
    "test_runconfig.py",
    "test_run_config_yaml.py",
    "test_units_benchmark.py",
}

# (resolution, category, name) of every Natural Earth file the code reads: cartopy's COASTLINE/BORDERS features
# (110m default, 50m in onset maps), the country and land masks (hcwf.spatial.masks).
NATURAL_EARTH_FILES = [
    ("110m", "physical", "coastline"), ("110m", "cultural", "admin_0_boundary_lines_land"),
    ("50m", "physical", "coastline"), ("50m", "cultural", "admin_0_boundary_lines_land"),
    ("10m", "cultural", "admin_0_countries"), ("50m", "cultural", "admin_0_countries"),
    ("110m", "cultural", "admin_0_countries"), ("50m", "physical", "land"), ("110m", "physical", "land"),
]


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="also run the tests marked slow (full differential runs against onset_blending, full grids)")
    parser.addoption("--run-bench", action="store_true", default=False,
                     help="also run the timing benchmarks (marked bench); run them on an idle machine")


def _ensure_natural_earth():
    """Download the missing Natural Earth files once, serialised by a file lock; offline is left to the tests."""
    try:
        import fcntl

        import cartopy
        from cartopy.io import shapereader
    except ImportError:
        return
    data_dir = os.path.join(cartopy.config["data_dir"], "shapefiles", "natural_earth")
    try:
        os.makedirs(data_dir, exist_ok=True)
        lock = open(os.path.join(data_dir, ".hcwf_tests.lock"), "w")
    except OSError:
        return
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for resolution, category, name in NATURAL_EARTH_FILES:
            try:
                shapereader.natural_earth(resolution=resolution, category=category, name=name)
            except Exception:  # offline or read-only: the tests that need the file report it themselves
                pass


def pytest_configure(config):
    # the xdist controller (and a plain serial run) only: workers have PYTEST_XDIST_WORKER set
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        _ensure_natural_earth()


@pytest.hookimpl(tryfirst=True)  # before xdist reads the xdist_group marks into the node ids
def pytest_collection_modifyitems(config, items):
    skip_slow = None if config.getoption("--run-slow") else pytest.mark.skip(reason="slow: run with --run-slow")
    skip_bench = None if config.getoption("--run-bench") else pytest.mark.skip(reason="bench: run with --run-bench")
    for item in items:
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.xdist_group("gpu"))
        else:
            module = item.nodeid.split("::")[0]
            if module.removeprefix("tests/") in GROUPED_MODULES:
                item.add_marker(pytest.mark.xdist_group(module))
        if skip_slow is not None and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if skip_bench is not None and "bench" in item.keywords:
            item.add_marker(skip_bench)
