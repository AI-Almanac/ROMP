"""Memory-budgeted worker counts, shared arrays, and start_items on the run's one pool."""
import multiprocessing
import os
import time

import numpy as np
import pytest

from momp.stats import parallel
from momp.utils import workers
from momp.utils.workers import SharedArrays, attach, cores, default_resources, detach, worker_count, workers_within


def test_workers_within_the_budget():
    assert workers_within([100, 300], 8, 1000) == 3
    assert workers_within([100], 8, None) == 8
    assert workers_within([], 8, 10) == 8
    assert workers_within([5000], 8, 1000) == 1   # at least one
    assert workers_within([0], 4, 10) == 4


def test_default_resources_use_the_affinity_cores():
    count, budget = default_resources()
    assert count == cores()
    assert budget is None or budget > 0


def test_worker_count_with_a_memory_budget():
    config = {"parallel": True, "parallel_workers": 6}
    assert worker_count(config, 10) == 6
    assert worker_count(config, 10, item_bytes=[400], budget_bytes=1000) == 2
    assert worker_count({**config, "memory_budget": 600}, 10, item_bytes=[300]) == 2
    assert worker_count({"parallel": False}, 10, item_bytes=[1], budget_bytes=10**9) == 1
    assert worker_count(config, 10, item_bytes=[1], budget_bytes=10**9) == 6


def test_limit_blas_threads():
    pytest.importorskip("threadpoolctl")
    workers.limit_blas_threads(1)
    from threadpoolctl import threadpool_info

    assert all(info["num_threads"] == 1 for info in threadpool_info())


def _total(specs):
    views = attach(specs)
    try:
        assert not views["a"].flags.writeable
        return float(views["a"].sum()), views["b"].tolist()
    finally:
        del views
        detach(specs)


def test_shared_arrays_reach_a_spawned_worker():
    ctx = multiprocessing.get_context("spawn")
    with SharedArrays(a=np.arange(12.0).reshape(3, 4), b=np.array([1, 2], dtype=np.int32)) as shared:
        filled = shared.empty("c", (2, 2), float, order="F")
        filled[...] = 3
        assert shared.specs["c"][3] == "F"
        with ctx.Pool(1) as pool:
            assert pool.apply(_total, (shared.specs,)) == (66.0, [1, 2])
    with pytest.raises(FileNotFoundError):
        attach(shared.specs)


def _scale(item, factor, offset=0):
    return item * factor + offset


def _fail_on_two(item):
    if item == 2:
        raise ValueError("boom")
    time.sleep(0.05)
    return item


def _pid(item):
    return os.getpid()


@pytest.fixture(autouse=True)
def _close_pool():
    yield
    parallel.shutdown()


def test_start_items_serial_runs_here_in_order():
    pending = parallel.start_items(_scale, [3, 1, 2], (2,), {"offset": 1}, config={"parallel": False})
    assert pending.result() == [7, 3, 5]
    assert parallel.map_items(_scale, [1], (1,), config={"parallel": False}) == [1]


def test_start_items_parallel_equals_serial_with_one_persistent_pool():
    items = list(range(7, 0, -1))
    serial = parallel.map_items(_scale, items, (3,), {"offset": 2}, workers=1)
    pending = parallel.start_items(_scale, items, (3,), {"offset": 2}, workers=2, label=str)
    assert pending.result() == serial
    pool = parallel._pool["executor"]
    assert pool is not None
    parallel.map_items(_pid, [1, 2], workers=2)
    assert parallel._pool["executor"] is pool   # the run's one pool is reused
    assert len(set(parallel.map_items(_pid, range(8), workers=2))) <= 2
    assert os.getpid() not in parallel.map_items(_pid, [1, 2], workers=2)


def test_start_items_failure_is_raised_and_queued_items_cancelled():
    pending = parallel.start_items(_fail_on_two, list(range(1, 40)), workers=2)
    with pytest.raises(ValueError, match="boom"):
        pending.result()
    assert any(job.cancelled() for job in pending._jobs)


def test_start_items_prewarm_runs_here_once_and_default_workers_follow_config():
    calls = []
    out = parallel.start_items(_scale, [1, 2, 3], (1,), config={"parallel": True, "parallel_workers": 2},
                               prewarm=lambda: calls.append(os.getpid())).result()
    assert out == [1, 2, 3] and calls == [os.getpid()]
