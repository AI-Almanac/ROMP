"""How many workers a run uses, ordered, bounded read-ahead of independent items, and what spawned workers share.

The config says ``parallel: false`` (one process, no threads) or leaves parallel on, optionally with
``parallel: {workers: N}``. With no number the workers are the cores available to this process,
never more than there are tasks and, when the size of a task is given, never more than fit the memory budget
(by default half of the memory available now). `limit_blas_threads` caps the BLAS threads of a worker, and
`SharedArrays`/`attach`/`detach` give spawned workers read-only views of big arrays in POSIX shared memory.
Nothing here imports the rest of momp, so any module may use it.
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


def cores() -> int:
    """The cores this process may use (its CPU affinity, else the machine's count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def memory_budget() -> int | None:
    """Half of the memory available now, in bytes (other work on the machine may hold much of the rest); None if unknown."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 // 2
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2
    except (ValueError, OSError):
        return None


def default_resources() -> tuple[int, int | None]:
    """(workers, memory budget): every core this process may use, sharing half the memory available now."""
    return cores(), memory_budget()


def workers_within(item_bytes: Sequence[int], workers: int, budget_bytes: int | None) -> int:
    """How many items to run at once so their combined peak fits the budget, sized by the largest (at least 1)."""
    if budget_bytes is None or not item_bytes:
        return max(1, int(workers))
    return max(1, min(int(workers), budget_bytes // max(max(item_bytes), 1)))


def worker_count(config: Mapping, tasks: int, item_bytes: Sequence[int] | None = None,
                 budget_bytes: int | None = None) -> int:
    """The workers to use for `tasks` independent tasks: 1 (serial) when config["parallel"] is false.

    Otherwise config["parallel_workers"] when set, else the cores available, never more than tasks. With
    `item_bytes` (the peak memory of each task) the count is also cut to what fits `budget_bytes`
    (default config["memory_budget"], else half of the memory available now).
    """
    if not config.get("parallel", True) or tasks <= 1:
        return 1
    limit = config.get("parallel_workers") or cores()
    workers = max(1, min(int(limit), int(tasks)))
    if item_bytes:
        budget = budget_bytes if budget_bytes is not None else config.get("memory_budget") or memory_budget()
        workers = workers_within(item_bytes, workers, budget)
    return workers


_BLAS_LIMIT = None


def limit_blas_threads(threads: int) -> None:
    """Cap this process's BLAS/OpenMP thread pools (a worker initializer; the BLAS side of NUMBA_NUM_THREADS).

    Needs threadpoolctl.
    """
    global _BLAS_LIMIT
    from threadpoolctl import threadpool_limits

    _BLAS_LIMIT = threadpool_limits(limits=int(threads))


def ordered_prefetch(func: Callable, items: Iterable, workers: int, ahead: int | None = None) -> Iterator:
    """Yield func(item) for each item in order, computing up to `ahead` items in worker threads in advance.

    With workers <= 1 this is a plain generator (func runs when the result is asked for). Otherwise the first
    item runs in the calling thread (so one-time setup such as masks is not raced) and the rest in threads;
    results come back in item order, and only `ahead` (default: workers) are held at a time, so memory stays
    bounded however many items there are. An exception of func is raised when its item's turn comes.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        for item in items:
            yield func(item)
        return
    ahead = max(1, ahead or workers)
    with ThreadPoolExecutor(max_workers=min(workers, len(items) - 1)) as pool:
        pending: deque = deque()
        queue = iter(items[1:])
        first = func(items[0])
        for item in queue:
            pending.append(pool.submit(func, item))
            if len(pending) >= ahead:
                break
        yield first
        del first
        while pending:
            result = pending.popleft().result()
            for item in queue:
                pending.append(pool.submit(func, item))
                break
            yield result
            del result


class SharedArrays:
    """Arrays in shared memory for spawned workers to read without copies.

        with SharedArrays(p=probabilities, years=years) as shared:
            pool.submit(task, shared.specs)          # task calls attach(specs)

    `empty` allocates one to fill in place (in either memory order).
    """

    def __init__(self, **arrays):
        """Copy the arrays into new shared blocks."""
        self._blocks, self.specs = [], {}
        try:
            for name, array in arrays.items():
                array = np.ascontiguousarray(array)
                self.empty(name, array.shape, array.dtype)[...] = array
        except BaseException:
            self.close()
            raise

    def empty(self, name: str, shape, dtype=float, order: str = "C") -> np.ndarray:
        """A new shared array, writable here and read-only in workers."""
        from multiprocessing import shared_memory

        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize
        block = shared_memory.SharedMemory(create=True, size=max(size, 1))
        self._blocks.append(block)
        self.specs[name] = (block.name, tuple(shape), dtype.str, order)
        return np.ndarray(shape, dtype, buffer=block.buf, order=order)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        """Release and remove the blocks."""
        for block in self._blocks:
            block.close()
            block.unlink()
        self._blocks = []


_ATTACHED: dict = {}


def attach(specs: dict) -> dict:
    """Read-only views of SharedArrays, in a worker (attached once per block)."""
    from multiprocessing import shared_memory

    out = {}
    for name, (block_name, shape, dtype, *order) in specs.items():
        if block_name not in _ATTACHED:
            _ATTACHED[block_name] = shared_memory.SharedMemory(name=block_name)
        view = np.ndarray(shape, dtype, buffer=_ATTACHED[block_name].buf, order=(order or ["C"])[0])
        view.flags.writeable = False
        out[name] = view
    return out


def detach(specs: dict) -> None:
    """Close a worker's views of these SharedArrays (once nothing uses them)."""
    for block_name, *_ in specs.values():
        block = _ATTACHED.pop(block_name, None)
        if block is not None:
            try:
                block.close()
            except BufferError:  # a view is still alive (an exception's frame): it goes with the view
                pass
