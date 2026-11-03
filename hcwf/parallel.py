"""Run independent items (years, files, folds, climatologies) in worker processes; results come back in item order.

This module holds the whole of the run's parallelism: how many workers (`worker_count`, `cores`, the memory budget), the
process pool (`start_items`), bounded read-ahead of independent items (`ordered_prefetch`) and what spawned workers share
(`SharedArrays`, `attach`, `detach`, `limit_blas_threads`).

`start_items` is the general function; `start_year_args`/`map_year_args`/`concat_year_args` are its year-keyed callers.

``parallel: false`` in the config keeps everything in this process. Otherwise the years of a task (pair tables,
onset tables, climatology pairs) are spread over `worker_count` processes (``parallel: {workers: N}``; default the
cores available, never more than the years):

* Every year goes to a worker. While the workers boot (they are started first, with the "spawn" method: a fork
  after numba's threads ran is unsafe; each gets cores // workers numba and BLAS threads, THREAD_LIMITS) this process pre-warms what all the
  years share (thresholds, masks; see the `prewarm` argument), so each is read, built and logged once, here, and
  then travels to the workers as one small snapshot. The pool is kept for the run (see `shutdown`), so the next
  stage (the climatology pairs after the model pairs) reuses the booted workers, and this process is free to do
  work of its own (the climatological onsets) while the workers run: `start_items` returns at once with a
  `Pending` whose `results()` waits.
* Workers receive file paths and settings, never data cubes: each reads its own year. What a worker detects or
  builds is sent back and absorbed here in year order, so the observed-onset cache on disk is written once, by this
  process.
* Log records and warnings raised in a worker are forwarded to this process's handlers (run.log, console) with
  the year in the message; see hcwf.log.forwarding.
* The results are collected in year order whatever order the workers finish in, so a parallel run gives exactly
  the numbers of a serial one.
"""
from __future__ import annotations

import atexit
import contextlib
import itertools
import logging
import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from hcwf import cache
from hcwf import log as logs

log = logging.getLogger(__name__)

NO_DATA = "No data was successfully processed"
"""Start of the ValueError the multi-year functions raise when no year yielded data (a skipped year in a chunk)."""
_tokens = itertools.count(1)
_installed = {"token": None}


def snapshot() -> dict:
    """What the workers need that this process has already read or built: the shared memos (loaded thresholds, built masks
    and their records, observed-onset tables; small arrays only, never data cubes) and the observed-onset cache's settings."""
    from hcwf.onset import observed_cache

    return {"memos": cache.shared_state(), "observed": observed_cache.export_state()}


def install(state: dict, token: int) -> None:
    """Make a snapshot this (worker) process's read-once inputs; does nothing if it already holds that snapshot."""
    if _installed["token"] == token:
        return
    from hcwf.onset import observed_cache

    cache.install_shared_state(state["memos"])
    observed_cache.install_state(state["observed"])
    _installed["token"] = token


def collect() -> dict:
    """What this (worker) process found since the last call: new observed onsets and masks, to send back."""
    from hcwf.onset import observed_cache
    from hcwf.spatial import masks

    return {"observed": observed_cache.drain_new(), "records": masks.records()}


def absorb(found: dict) -> None:
    """Take what a worker found into this process (observed onsets into the cache, new mask records)."""
    from hcwf.onset import observed_cache
    from hcwf.spatial import masks

    observed_cache.absorb(found["observed"])
    masks.absorb_records(found["records"])


def _warm() -> int:
    """Import everything a task uses (so a worker boots while this process pre-warms the shared inputs)."""
    import hcwf.scoring.onset_tables  # noqa: F401
    import hcwf.scoring.pairs  # noqa: F401
    import hcwf.metrics.bins  # noqa: F401
    import hcwf.references.conditional  # noqa: F401

    return os.getpid()


def _run_item(func, item, args, kwargs, state, token, label):
    """Worker entry: install the snapshot, run func(item, *args, **kwargs); returns (result, what was found)."""
    install(state, token)
    logs.set_worker_year(label)
    return func(item, *args, **kwargs), collect()


def _years_call(year, func, args, skip):
    """func([year], *args); None where it raises the "no data" ValueError of a skipped year."""
    try:
        return func([year], *args)
    except ValueError as exc:
        if not skip or not str(exc).startswith(skip):
            raise
        return None


_pool: dict = {"executor": None, "key": None, "stack": None}
"""The run's worker pool: the executor, what it was made for (size, numba threads, log handlers) and what to close."""


def shutdown() -> None:
    """Stop the run's worker pool (if any) and the log forwarding that served it; safe to call at any time."""
    executor, stack = _pool["executor"], _pool["stack"]
    _pool.update(executor=None, key=None, stack=None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    if stack is not None:
        stack.close()


atexit.register(shutdown)
cache.on_run_end(shutdown)


THREAD_LIMITS = ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
"""The environment variables that cap a worker's thread pools (numba, OpenBLAS, OpenMP, MKL). They are set while the
workers spawn, so that every library reads them when it loads: threadpoolctl in an initializer would miss the ones
imported later (scipy's own OpenBLAS), and N workers on N BLAS threads each is slower than N workers on one."""


def _pool_for(size: int, threads: int) -> ProcessPoolExecutor:
    """The run's pool of `size` booted workers with `threads` numba threads each, started now if there is none like it.

    A pool is reused while the size, thread count and log handlers are the same; otherwise the old one is stopped.
    New workers begin importing the task modules at once.
    """
    key = (size, threads, tuple(id(h) for h in logging.getLogger(logs.ROOT).handlers))
    if _pool["executor"] is not None and _pool["key"] == key:
        return _pool["executor"]
    shutdown()
    saved = {name: os.environ.get(name) for name in THREAD_LIMITS}
    os.environ.update({name: str(threads) for name in THREAD_LIMITS})
    stack = contextlib.ExitStack()
    try:
        queue = stack.enter_context(logs.forwarding())
        executor = ProcessPoolExecutor(
            max_workers=size, mp_context=multiprocessing.get_context("spawn"),
            initializer=None if queue is None else logs.worker_logging,
            initargs=() if queue is None else (queue,))
        for _ in range(size):
            executor.submit(_warm)   # starts a worker process each, while the environment above is in force
    except BaseException:
        stack.close()
        raise
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    _pool.update(executor=executor, key=key, stack=stack)
    return executor


class Pending:
    """Work started by start_items / start_year_args / start_concat_args: `result()` waits for it (once) and gives the value."""

    def __init__(self, collect: Callable[[], Any], jobs=()):
        """`collect` produces the value when called; `jobs` are the futures to cancel if it fails."""
        self._collect, self._jobs, self._done, self._value = collect, list(jobs), False, None

    @classmethod
    def finished(cls, value) -> "Pending":
        """A Pending whose value is already known (work done in this process)."""
        pending = cls(lambda: value)
        pending._done, pending._value = True, value
        return pending

    def result(self):
        """Wait for the work and return its value; a failure cancels the jobs still queued and is raised."""
        if not self._done:
            try:
                self._value = self._collect()
            except BrokenProcessPool as exc:
                shutdown()
                raise RuntimeError(f"a worker process died ({exc}); run with parallel: false to see the error in "
                                   "this process") from exc
            except BaseException:
                for job in self._jobs:
                    job.cancel()
                raise
            self._done = True
        return self._value


def start_items(func, items, args: tuple = (), kwargs: dict | None = None, *, config: dict | None = None,
                workers: int | None = None, label: Callable[[Any], Any] | None = None,
                prewarm: Callable[[], None] | None = None) -> Pending:
    """Start func(item, *args, **kwargs) for each item; the Pending gives the results in item order.

    Runs in worker processes of the run's pool when there are `workers` > 1 (default: worker_count of `config`, which
    is `kwargs` when not given, for the number of items), all items in workers (see the module text); a failure
    cancels the items still queued. `prewarm()`, if given, is called here once before the jobs are sent, to read or
    build what every item needs. `label(item)` names the item in the log lines a worker forwards. With one worker the
    items run here, one after the other, before this returns. func must be a module-level function (workers are spawned).
    """
    items, kwargs = list(items), kwargs or {}
    workers = worker_count(config if config is not None else kwargs, len(items)) if workers is None else workers
    if workers <= 1:
        return Pending.finished([func(item, *args, **kwargs) for item in items])
    executor = _pool_for(workers, max(1, cores() // workers))
    if prewarm is not None:
        prewarm()
    state, token = snapshot(), next(_tokens)
    jobs = [executor.submit(_run_item, func, item, args, kwargs, state, token, None if label is None else label(item))
            for item in items]

    def collect_results():
        """Wait for the jobs in item order, absorbing what each found."""
        results = []
        for job in jobs:
            result, found = job.result()
            absorb(found)
            results.append(result)
        return results

    return Pending(collect_results, jobs)


def map_items(func, items, args: tuple = (), kwargs: dict | None = None, **options) -> list:
    """func(item, *args, **kwargs) for each item, results in item order (start_items, waited for at once)."""
    return start_items(func, items, args, kwargs, **options).result()


# ----------------------------------------------------------------------------------------------------------------------
# the year-keyed forms: `args` and a ParallelSpec; func(year_list, *args)
# ----------------------------------------------------------------------------------------------------------------------

def spec_workers(parallel, tasks: int) -> int:
    """The workers for `tasks` items under a ParallelSpec (hcwf.config.model): `worker_count` of its two settings."""
    return worker_count({"parallel": parallel.enabled, "parallel_workers": parallel.workers}, tasks)


def start_year_args(func, years, args: tuple, parallel, skip: str | None = NO_DATA,
                    prewarm: Callable[[], None] | None = None) -> Pending:
    """Start func([year], *args) for each year; the Pending gives the results in year order.

    start_items over the years in workers when `parallel` (a ParallelSpec) allows and there is more than one year;
    `prewarm()`, if given, is called here once before the jobs are sent. `skip` is the start of the ValueError message
    that means "this year had no data" (a missing file): that year's result is None. `args` travel to
    each worker: pass the run and the case, not data.
    """
    years = list(years)
    return start_items(_years_call, years, (func, args, skip), workers=spec_workers(parallel, len(years)), label=int,
                       prewarm=prewarm)


def map_year_args(func, years, args: tuple, parallel, skip: str | None = NO_DATA,
                  prewarm: Callable[[], None] | None = None) -> list:
    """func([year], *args) for each year, results in year order (None for a skipped year); `start_year_args`, waited for."""
    return start_year_args(func, years, args, parallel, skip, prewarm).result()


def start_concat_args(func, years, args: tuple, parallel, prewarm: Callable[[], None] | None = None) -> Pending:
    """Start the pair tables of func(years, *args); the Pending gives them concatenated.

    With one worker func runs once on all the years before this returns. Otherwise each year is computed by
    `start_year_args` and the tables joined in year order; a year without data is left out, and ValueError("No data ...")
    is raised when none has any.
    """
    years = list(years)
    if spec_workers(parallel, len(years)) <= 1:
        return Pending.finished(func(years, *args))
    pending = start_year_args(func, years, args, parallel, prewarm=prewarm)

    def join():
        """The year tables, concatenated."""
        parts = [part for part in pending.result() if part is not None]
        if not parts:
            raise ValueError(f"{NO_DATA} for any year")
        import pandas as pd

        return pd.concat(parts, ignore_index=True)

    return Pending(join)


def concat_year_args(func, years, args: tuple, parallel, prewarm: Callable[[], None] | None = None):
    """The pair tables of func(years, *args), concatenated in year order (`start_concat_args`, waited for)."""
    return start_concat_args(func, years, args, parallel, prewarm).result()


# ---- how many workers, ordered read-ahead, shared memory (was parallel.py) ----


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
