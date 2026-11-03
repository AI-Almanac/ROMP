"""Shared test setup: safe under ``pytest -n auto`` (pytest-xdist).

* Thread pools (numba, BLAS, OpenMP) are sized per worker, so N workers do not each start one thread per core
  (that oversubscription made the parallel run slower than the serial one). Set before numpy/numba are imported.
* Every test gets its own MOMP_CACHE_DIR, so no test reads or writes ~/.cache/momp or another worker's entries.
* Tests that touch the GPU (anything with "gpu" in its id) share one xdist group, so they run on one worker
  with ``--dist loadgroup`` and do not contend for the single device.
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
def _isolated_momp_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MOMP_CACHE_DIR", str(tmp_path_factory.mktemp("momp_cache")))


def pytest_collection_modifyitems(items):
    for item in items:
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.xdist_group("gpu"))
