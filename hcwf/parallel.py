"""Run independent items (years, files, folds, climatologies) in worker processes; results come back in item order.

This module holds the whole of the run's parallelism: how many workers (`worker_count`, `cores`, the memory budget), the
process pool (`start_items`), bounded read-ahead of independent items (`ordered_prefetch`) and what spawned workers share
(`SharedArrays`, `attach`, `detach`, `limit_blas_threads`).

`start_items` is the general function; `start_year_args`, `map_year_args` and `concat_year_args` are its year-keyed
callers.

``parallel: false`` in the config keeps everything in this process. Otherwise the years of a task (pair tables,
onset tables, climatology pairs) are spread over `worker_count` processes (``parallel: {workers: N}``; default the
cores available, never more than the years):

* Every year goes to a worker. While the workers boot (they are started first, with the "spawn" method: a fork
  after numba's threads ran is unsafe; each gets ``cores // workers`` numba and BLAS threads, `THREAD_LIMITS`) this process pre-warms what all the
  years share (thresholds, masks; see the `prewarm` argument), so each is read, built and logged once, here, and
  then travels to the workers as one small snapshot. The pool is kept for the run (see `shutdown`), so the next
  stage (the climatology pairs after the model pairs) reuses the booted workers, and this process is free to do
  work of its own (the climatological onsets) while the workers run: `start_items` returns at once with a
  `Pending` whose `results()` waits.
* Workers receive file paths and settings, never data cubes: each reads its own year. What a worker detects or
  builds is sent back and absorbed here in year order, so the observed-onset cache on disk is written once, by this
  process.
* Log records and warnings raised in a worker are forwarded to this process's handlers (run.log, console) with
  the year in the message; see `hcwf.log.forwarding`.
* The results are collected in year order whatever order the workers finish in, so a parallel run gives exactly
  the numbers of a serial one.

Apart from the years' pool, a big NetCDF forecast read in the run's own process is decoded by reader processes
(`reader_pool`, `reader_count`; see `hcwf.spatial.select`), because HDF5 decodes one chunk at a time per process; they
are stopped when the years' pool starts and at the end of the run, and a worker never starts any.

How many year workers there are is also limited by memory: `spec_workers` takes the peak memory of a year (an estimate
of its forecast, `hcwf.io.forecasts.forecast_year_bytes`) and keeps the workers within half the memory available.
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
"""Start of the ``ValueError`` message the multi-year functions raise when no year yielded data (a skipped year in a chunk)."""
_tokens = itertools.count(1)
_installed = {"token": None}


def snapshot() -> dict:
    """Return what the workers need that this process has already read or built.

    That is the shared memos (loaded thresholds, built masks and their records, observed-onset tables; small arrays only,
    never data cubes) and the observed-onset cache's settings.

    Returns
    -------
    dict
        ``{"memos": ..., "observed": ...}``, to be passed to `install` in a worker.
    """
    from hcwf.onset import observed_cache

    return {"memos": cache.shared_state(), "observed": observed_cache.export_state()}


def install(state: dict, token: int) -> None:
    """Make a snapshot this (worker) process's read-once inputs; do nothing if it already holds that snapshot.

    Parameters
    ----------
    state : dict
        A snapshot made by `snapshot`.
    token : int
        The identity of the snapshot; a worker that already installed this token skips the work.
    """
    if _installed["token"] == token:
        return
    from hcwf.onset import observed_cache

    cache.install_shared_state(state["memos"])
    observed_cache.install_state(state["observed"])
    _installed["token"] = token


def collect() -> dict:
    """Return what this (worker) process found since the last call, to send back to the run's process.

    Returns
    -------
    dict
        ``{"observed": ..., "records": ...}``: new observed onsets and mask records, for `absorb`.
    """
    from hcwf.onset import observed_cache
    from hcwf.spatial import masks

    return {"observed": observed_cache.drain_new(), "records": masks.records()}


def absorb(found: dict) -> None:
    """Take what a worker found into this process (observed onsets into the cache, new mask records).

    Parameters
    ----------
    found : dict
        What `collect` returned in the worker.
    """
    from hcwf.onset import observed_cache
    from hcwf.spatial import masks

    observed_cache.absorb(found["observed"])
    masks.absorb_records(found["records"])


def _warm() -> int:
    """Import everything a task uses, so a worker boots while this process pre-warms the shared inputs.

    Returns
    -------
    int
        The process id of the worker.
    """
    import hcwf.scoring.onset_tables  # noqa: F401
    import hcwf.scoring.pairs  # noqa: F401
    import hcwf.metrics.bins  # noqa: F401
    import hcwf.references.conditional  # noqa: F401

    return os.getpid()


def _run_item(func, item, args, kwargs, state, token, label):
    """Run one item in a worker: install the snapshot, then call ``func(item, *args, **kwargs)``.

    Parameters
    ----------
    func : callable
        A module-level function (workers are spawned).
    item : object
        The item to process.
    args : tuple
        Extra positional arguments of `func`.
    kwargs : dict
        Keyword arguments of `func`.
    state : dict
        The snapshot to install, from `snapshot`.
    token : int
        The identity of the snapshot.
    label : object
        Names the item in the log lines the worker forwards (see `hcwf.log.set_worker_year`).

    Returns
    -------
    tuple
        ``(result, found)``: the result of `func` and what `collect` found, for `absorb`.
    """
    install(state, token)
    logs.set_worker_year(label)
    return func(item, *args, **kwargs), collect()


def _years_call(year, func, args, skip):
    """Call ``func([year], *args)``, giving None where it raises the "no data" ``ValueError`` of a skipped year.

    Parameters
    ----------
    year : int
        The year to compute.
    func : callable
        Takes a list of years and `args`.
    args : tuple
        Extra positional arguments of `func`.
    skip : str or None
        Start of the ``ValueError`` message that means the year has no data; None re-raises every error.

    Returns
    -------
    object or None
        The result of `func`, or None for a skipped year.

    Raises
    ------
    ValueError
        Any ``ValueError`` of `func` that does not start with `skip`.
    """
    try:
        return func([year], *args)
    except ValueError as exc:
        if not skip or not str(exc).startswith(skip):
            raise
        return None


_pool: dict = {"executor": None, "key": None, "stack": None}
"""The run's worker pool: the executor, what it was made for (size, numba threads, log handlers) and what to close."""


def shutdown() -> None:
    """Stop the run's worker pool (if any), the log forwarding that served it, and the readers; safe to call at any time."""
    shutdown_readers()
    executor, stack = _pool["executor"], _pool["stack"]
    _pool.update(executor=None, key=None, stack=None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    if stack is not None:
        stack.close()


def in_worker() -> bool:
    """Tell whether this process is a spawned worker (of the run's pool or of the readers), not the run's own process.

    Returns
    -------
    bool
        True in a worker.
    """
    return multiprocessing.parent_process() is not None


def reader_pool(size: int) -> ProcessPoolExecutor:
    """Return processes for reading one file's chunks in parallel (a NetCDF/HDF5 file is decoded by one thread per process).

    The workers get one numba/BLAS thread each; the pool is kept for the run (see `shutdown`) and made larger when a
    bigger read asks, apart from the run's year pool (which may have jobs in flight). Only the run's own process asks:
    a worker that is itself one of the years' workers reads on its own.

    Parameters
    ----------
    size : int
        The number of reader processes wanted.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        The reader pool, with at least `size` workers.
    """
    executor, held = _pool.get("readers"), _pool.get("readers_size", 0)
    if executor is not None and held >= size:
        return executor
    shutdown_readers()
    saved = {name: os.environ.get(name) for name in THREAD_LIMITS}
    os.environ.update({name: "1" for name in THREAD_LIMITS})
    try:
        executor = ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("spawn"))
        for _ in range(size):
            executor.submit(_ready)   # starts a worker process each, while the environment above is in force
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    _pool.update(readers=executor, readers_size=size)
    return executor


def _ready() -> int:
    """Start a reader process: it imports what a read needs while the first blocks are being asked for.

    Returns
    -------
    int
        The process id of the reader.
    """
    import hcwf.spatial.select  # noqa: F401

    return os.getpid()


def shutdown_readers() -> None:
    """Stop the reader processes, if any."""
    executor = _pool.pop("readers", None)
    _pool.pop("readers_size", None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown)
cache.on_run_end(shutdown)


THREAD_LIMITS = ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
"""The environment variables that cap a worker's thread pools (numba, OpenBLAS, OpenMP, MKL). They are set while the
workers spawn, so that every library reads them when it loads: threadpoolctl in an initializer would miss the ones
imported later (scipy's own OpenBLAS), and N workers on N BLAS threads each is slower than N workers on one."""


def _pool_for(size: int, threads: int) -> ProcessPoolExecutor:
    """Return the run's pool of `size` booted workers with `threads` numba threads each, started now if there is none like it.

    A pool is reused while the size, thread count and log handlers are the same; otherwise the old one is stopped.
    New workers begin importing the task modules at once.

    Parameters
    ----------
    size : int
        The number of worker processes.
    threads : int
        The numba and BLAS threads of each worker.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        The pool.

    Raises
    ------
    BaseException
        Whatever stopped the pool from starting (including ``KeyboardInterrupt``), after its log forwarding is closed.
    """
    key = (size, threads, tuple(id(h) for h in logging.getLogger(logs.ROOT).handlers))
    if _pool["executor"] is not None and _pool["key"] == key:
        return _pool["executor"]
    shutdown()   # (the readers too: their memory goes to the years)
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
    """Work started by `start_items`, `start_year_args` or `start_concat_args`; `result` waits for it (once) and gives the value.

    Parameters
    ----------
    collect : callable
        Produces the value when called with no arguments.
    jobs : iterable of concurrent.futures.Future, optional
        The futures to cancel if `collect` fails.
    """

    def __init__(self, collect: Callable[[], Any], jobs=()):
        self._collect, self._jobs, self._done, self._value = collect, list(jobs), False, None

    @classmethod
    def finished(cls, value) -> "Pending":
        """Make a Pending whose value is already known (work done in this process).

        Parameters
        ----------
        value : object
            The value `result` returns.

        Returns
        -------
        Pending
            A finished Pending.
        """
        pending = cls(lambda: value)
        pending._done, pending._value = True, value
        return pending

    def result(self):
        """Wait for the work and return its value; a failure cancels the jobs still queued and is raised.

        Returns
        -------
        object
            The value of the work.

        Raises
        ------
        RuntimeError
            A worker process died (the pool is shut down); run with ``parallel: false`` to see the error in this process.
        BaseException
            Any other failure of the work, after the jobs still queued are cancelled.
        """
        if not self._done:
            try:
                self._value = self._collect()
            except BrokenProcessPool as exc:
                shutdown()
                raise RuntimeError(f"a worker process died ({exc}); run with parallel: false to see the error in "
                                   "this process") from exc
            except BaseException:
                for job in self._jobs:
                    if job is not None:
                        job.cancel()
                raise
            self._done = True
        return self._value


def start_items(func, items, args: tuple = (), kwargs: dict | None = None, *, config: dict | None = None,
                workers: int | None = None, label: Callable[[Any], Any] | None = None,
                prewarm: Callable[[], None] | None = None, sink: Callable[[int, Any], None] | None = None) -> Pending:
    """Start ``func(item, *args, **kwargs)`` for each item; the Pending gives the results in item order.

    Runs in worker processes of the run's pool when there is more than one worker, all items in workers (see the module
    text); a failure cancels the items still queued. With one worker the items run here, one after the other, before this
    returns. With `sink`, each result goes to ``sink(position, result)`` in item order as soon as it and the ones before it
    are done, and is not kept (the Pending gives None): results too big to hold together are placed one at a time.

    Parameters
    ----------
    func : callable
        A module-level function (workers are spawned).
    items : iterable
        The items to process; each is passed as the first argument of `func`.
    args : tuple, default ()
        Extra positional arguments of `func`.
    kwargs : dict, optional
        Keyword arguments of `func`; None means none.
    config : dict, optional
        Gives the worker count through `worker_count`; default `kwargs`.
    workers : int, optional
        The number of workers; None takes `worker_count` of `config` for the number of items.
    label : callable, optional
        ``label(item)`` names the item in the log lines a worker forwards.
    prewarm : callable, optional
        Called here once before the jobs are sent, to read or build what every item needs.
    sink : callable, optional
        ``sink(position, result)``, called in item order instead of keeping the results.

    Returns
    -------
    Pending
        Its `Pending.result` gives the list of results in item order (None with `sink`).
    """
    items, kwargs = list(items), kwargs or {}
    workers = worker_count(config if config is not None else kwargs, len(items)) if workers is None else workers
    if workers <= 1:
        if sink is None:
            return Pending.finished([func(item, *args, **kwargs) for item in items])
        for position, item in enumerate(items):
            sink(position, func(item, *args, **kwargs))
        return Pending.finished(None)
    executor = _pool_for(workers, max(1, cores() // workers))
    if prewarm is not None:
        prewarm()
    state, token = snapshot(), next(_tokens)
    jobs = [executor.submit(_run_item, func, item, args, kwargs, state, token, None if label is None else label(item))
            for item in items]

    def collect_results():
        """Wait for the jobs in item order, absorbing what each found."""
        results = []
        for position, job in enumerate(jobs):
            result, found = job.result()
            absorb(found)
            if sink is None:
                results.append(result)
            else:
                sink(position, result)
                jobs[position] = pending._jobs[position] = None   # the finished job holds its result: let it go
            del result, job
        return None if sink is not None else results

    pending = Pending(collect_results, jobs)
    return pending


def map_items(func, items, args: tuple = (), kwargs: dict | None = None, **options) -> list:
    """Return ``func(item, *args, **kwargs)`` for each item, in item order (`start_items`, waited for at once).

    Parameters
    ----------
    func : callable
        A module-level function.
    items : iterable
        The items to process.
    args : tuple, default ()
        Extra positional arguments of `func`.
    kwargs : dict, optional
        Keyword arguments of `func`.
    **options : dict
        Passed on to `start_items` (``config``, ``workers``, ``label``, ``prewarm``, ``sink``).

    Returns
    -------
    list
        The results in item order.
    """
    return start_items(func, items, args, kwargs, **options).result()


# ----------------------------------------------------------------------------------------------------------------------
# the year-keyed forms: `args` and a ParallelSpec; func(year_list, *args)
# ----------------------------------------------------------------------------------------------------------------------

def spec_workers(parallel, tasks: int, item_bytes: int | None = None) -> int:
    """Return the workers for `tasks` items under a ParallelSpec (`hcwf.config.model`).

    That is `worker_count` of its two settings, cut to what fits the memory budget when `item_bytes` is given.

    Parameters
    ----------
    parallel : hcwf.config.model.ParallelSpec
        The ``parallel:`` settings (``enabled`` and ``workers``).
    tasks : int
        The number of items to process.
    item_bytes : int, optional
        The peak memory of one item, in bytes.

    Returns
    -------
    int
        The number of workers, at least 1.
    """
    return worker_count({"parallel": parallel.enabled, "parallel_workers": parallel.workers}, tasks,
                        None if item_bytes is None else [item_bytes])


def start_year_args(func, years, args: tuple, parallel, skip: str | None = NO_DATA,
                    prewarm: Callable[[], None] | None = None, item_bytes: int | None = None) -> Pending:
    """Start ``func([year], *args)`` for each year; the Pending gives the results in year order.

    This is `start_items` over the years in workers when `parallel` allows and there is more than one year. `args` travel
    to each worker: pass the run and the case, not data.

    Parameters
    ----------
    func : callable
        A module-level function taking a list of years and `args`.
    years : iterable of int
        The years to compute.
    args : tuple
        Extra positional arguments of `func`.
    parallel : hcwf.config.model.ParallelSpec
        The ``parallel:`` settings.
    skip : str or None, default `NO_DATA`
        Start of the ``ValueError`` message that means "this year had no data" (a missing file): that year's result is
        None.
    prewarm : callable, optional
        Called here once before the jobs are sent.
    item_bytes : int, optional
        The peak memory of one year; cuts the workers to what fits the memory budget.

    Returns
    -------
    Pending
        Its `Pending.result` gives the list of results in year order, None for a skipped year.
    """
    years = list(years)
    return start_items(_years_call, years, (func, args, skip), workers=spec_workers(parallel, len(years), item_bytes),
                       label=int, prewarm=prewarm)


def map_year_args(func, years, args: tuple, parallel, skip: str | None = NO_DATA,
                  prewarm: Callable[[], None] | None = None, item_bytes: int | None = None) -> list:
    """Return ``func([year], *args)`` for each year, in year order (`start_year_args`, waited for).

    Parameters
    ----------
    func : callable
        A module-level function taking a list of years and `args`.
    years : iterable of int
        The years to compute.
    args : tuple
        Extra positional arguments of `func`.
    parallel : hcwf.config.model.ParallelSpec
        The ``parallel:`` settings.
    skip : str or None, default `NO_DATA`
        Start of the ``ValueError`` message that marks a year without data.
    prewarm : callable, optional
        Called once before the jobs are sent.
    item_bytes : int, optional
        The peak memory of one year.

    Returns
    -------
    list
        The results in year order; None for a skipped year.
    """
    return start_year_args(func, years, args, parallel, skip, prewarm, item_bytes).result()


def start_concat_args(func, years, args: tuple, parallel, prewarm: Callable[[], None] | None = None,
                      combine: Callable[[list], Any] | None = None) -> Pending:
    """Start the pair tables of ``func(years, *args)``; the Pending gives them concatenated.

    With one worker `func` runs once on all the years before this returns. Otherwise each year is computed by
    `start_year_args` and the tables joined in year order; a year without data is left out, and a ``ValueError`` is raised
    (when the result is asked for) if none has any.

    Parameters
    ----------
    func : callable
        A module-level function taking a list of years and `args`; it returns a table.
    years : iterable of int
        The years to compute.
    args : tuple
        Extra positional arguments of `func`.
    parallel : hcwf.config.model.ParallelSpec
        The ``parallel:`` settings.
    prewarm : callable, optional
        Called once before the jobs are sent.
    combine : callable, optional
        Joins the list of year tables into one; default ``pandas.concat`` of DataFrames with a fresh index.

    Returns
    -------
    Pending
        Its `Pending.result` gives the concatenated table.
    """
    years = list(years)
    if spec_workers(parallel, len(years)) <= 1:
        return Pending.finished(func(years, *args))
    pending = start_year_args(func, years, args, parallel, prewarm=prewarm)

    def join():
        """Concatenate the year tables."""
        parts = [part for part in pending.result() if part is not None]
        if not parts:
            raise ValueError(f"{NO_DATA} for any year")
        if combine is not None:
            return combine(parts)
        import pandas as pd

        return pd.concat(parts, ignore_index=True)

    return Pending(join)


def concat_year_args(func, years, args: tuple, parallel, prewarm: Callable[[], None] | None = None,
                     combine: Callable[[list], Any] | None = None):
    """Return the pair tables of ``func(years, *args)``, concatenated in year order (`start_concat_args`, waited for).

    Parameters
    ----------
    func : callable
        A module-level function taking a list of years and `args`; it returns a table.
    years : iterable of int
        The years to compute.
    args : tuple
        Extra positional arguments of `func`.
    parallel : hcwf.config.model.ParallelSpec
        The ``parallel:`` settings.
    prewarm : callable, optional
        Called once before the jobs are sent.
    combine : callable, optional
        Joins the list of year tables into one; default ``pandas.concat``.

    Returns
    -------
    pandas.DataFrame
        The concatenated table (whatever `combine` returns).
    """
    return start_concat_args(func, years, args, parallel, prewarm, combine).result()


# ---- how many workers, ordered read-ahead, shared memory (was parallel.py) ----


def cores() -> int:
    """Return the cores this process may use (its CPU affinity, else the machine's count).

    Returns
    -------
    int
        The number of cores, at least 1.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def memory_budget() -> int | None:
    """Return half of the memory available now, in bytes (other work on the machine may hold much of the rest).

    Returns
    -------
    int or None
        The budget in bytes; None if the memory cannot be read.
    """
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
    """Return the default workers and memory budget: every core this process may use, sharing half the memory available now.

    Returns
    -------
    tuple of (int, int or None)
        The workers, and the memory budget in bytes (None if unknown).
    """
    return cores(), memory_budget()


def workers_within(item_bytes: Sequence[int], workers: int, budget_bytes: int | None) -> int:
    """Return how many items to run at once so their combined peak fits the budget, sized by the largest.

    Parameters
    ----------
    item_bytes : sequence of int
        The peak memory of each item, in bytes.
    workers : int
        The most that may run at once.
    budget_bytes : int or None
        The memory budget in bytes; None means no memory limit.

    Returns
    -------
    int
        The number of items to run at once, at least 1.
    """
    if budget_bytes is None or not item_bytes:
        return max(1, int(workers))
    return max(1, min(int(workers), budget_bytes // max(max(item_bytes), 1)))


def worker_count(config: Mapping, tasks: int, item_bytes: Sequence[int] | None = None,
                 budget_bytes: int | None = None) -> int:
    """Return the workers to use for `tasks` independent tasks: 1 (serial) when ``config["parallel"]`` is false.

    Otherwise ``config["parallel_workers"]`` when set, else the cores available, never more than `tasks`. With
    `item_bytes` the count is also cut to what fits `budget_bytes`.

    Parameters
    ----------
    config : collections.abc.Mapping
        The flat config; ``parallel``, ``parallel_workers`` and ``memory_budget`` are read.
    tasks : int
        The number of independent tasks.
    item_bytes : sequence of int, optional
        The peak memory of each task, in bytes.
    budget_bytes : int, optional
        The memory budget in bytes; default ``config["memory_budget"]``, else half of the memory available now.

    Returns
    -------
    int
        The number of workers, at least 1.
    """
    if not config.get("parallel", True) or tasks <= 1:
        return 1
    limit = config.get("parallel_workers") or cores()
    workers = max(1, min(int(limit), int(tasks)))
    if item_bytes:
        budget = budget_bytes if budget_bytes is not None else config.get("memory_budget") or memory_budget()
        workers = workers_within(item_bytes, workers, budget)
    return workers


def reader_count(parallel) -> int:
    """Return the processes one file's chunks may be read by, for a run with the ParallelSpec `parallel`.

    That is 1 (read in this process) when parallel is off or this process is itself a worker, else its ``workers``
    setting or the cores available.

    Parameters
    ----------
    parallel : hcwf.config.model.ParallelSpec or None
        The ``parallel:`` settings; None means the defaults.

    Returns
    -------
    int
        The number of reader processes, at least 1.
    """
    if in_worker() or (parallel is not None and not parallel.enabled):
        return 1
    return max(1, int((parallel.workers if parallel is not None else None) or cores()))


_BLAS_LIMIT = None


def limit_blas_threads(threads: int) -> None:
    """Cap this process's BLAS/OpenMP thread pools (a worker initializer; the BLAS side of ``NUMBA_NUM_THREADS``).

    Needs threadpoolctl.

    Parameters
    ----------
    threads : int
        The most threads each pool may use.
    """
    global _BLAS_LIMIT
    from threadpoolctl import threadpool_limits

    _BLAS_LIMIT = threadpool_limits(limits=int(threads))


def ordered_prefetch(func: Callable, items: Iterable, workers: int, ahead: int | None = None) -> Iterator:
    """Yield ``func(item)`` for each item in order, computing up to `ahead` items in worker threads in advance.

    With one worker or fewer this is a plain generator (`func` runs when the result is asked for). Otherwise the first
    item runs in the calling thread (so one-time setup such as masks is not raced) and the rest in threads;
    results come back in item order, and only `ahead` are held at a time, so memory stays bounded however many items
    there are. An exception of `func` is raised when its item's turn comes.

    Parameters
    ----------
    func : callable
        Takes one item and returns its result.
    items : iterable
        The items to process.
    workers : int
        The threads to use; 1 or less runs everything lazily in the calling thread.
    ahead : int, optional
        How many results may be computed ahead of the consumer; default `workers`.

    Yields
    ------
    object
        The result of `func` for each item, in item order.
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

    Used as a context manager, which releases the blocks on exit::

        with SharedArrays(p=probabilities, years=years) as shared:
            pool.submit(task, shared.specs)          # task calls attach(specs)

    `empty` allocates one to fill in place (in either memory order).

    Parameters
    ----------
    **arrays : numpy.ndarray
        The arrays to copy into new shared blocks, by name.

    Attributes
    ----------
    specs : dict
        ``{name: (block name, shape, dtype string, memory order)}``, what `attach` takes in a worker.

    Raises
    ------
    BaseException
        Whatever stopped a copy (including ``KeyboardInterrupt``), after the blocks made so far are released.
    """

    def __init__(self, **arrays):
        self._blocks, self.specs = [], {}
        try:
            for name, array in arrays.items():
                array = np.ascontiguousarray(array)
                self.empty(name, array.shape, array.dtype)[...] = array
        except BaseException:
            self.close()
            raise

    def empty(self, name: str, shape, dtype=float, order: str = "C") -> np.ndarray:
        """Allocate a new shared array, writable here and read-only in workers.

        Parameters
        ----------
        name : str
            The name under which it appears in ``specs``.
        shape : tuple of int
            The shape of the array.
        dtype : numpy.dtype or type, default float
            The element type.
        order : {"C", "F"}, default "C"
            The memory order.

        Returns
        -------
        numpy.ndarray
            The array, backed by the shared block and uninitialised.
        """
        from multiprocessing import shared_memory

        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize
        block = shared_memory.SharedMemory(create=True, size=max(size, 1))
        self._blocks.append(block)
        self.specs[name] = (block.name, tuple(shape), dtype.str, order)
        return np.ndarray(shape, dtype, buffer=block.buf, order=order)

    def __enter__(self):
        """Enter the ``with`` block.

        Returns
        -------
        SharedArrays
            This object.
        """
        return self

    def __exit__(self, *exc):
        """Release the blocks when the ``with`` block ends.

        Parameters
        ----------
        *exc : tuple
            The exception type, value and traceback, if any; they are not suppressed.
        """
        self.close()

    def close(self) -> None:
        """Release and remove the blocks."""
        for block in self._blocks:
            block.close()
            block.unlink()
        self._blocks = []


_ATTACHED: dict = {}


def attach(specs: dict) -> dict:
    """Return read-only views of `SharedArrays`, in a worker (attached once per block).

    Parameters
    ----------
    specs : dict
        The ``specs`` of a `SharedArrays`.

    Returns
    -------
    dict of str to numpy.ndarray
        A read-only view of each array, by name.
    """
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
    """Close a worker's views of these `SharedArrays` (once nothing uses them).

    Parameters
    ----------
    specs : dict
        The ``specs`` of a `SharedArrays`.
    """
    for block_name, *_ in specs.values():
        block = _ATTACHED.pop(block_name, None)
        if block is not None:
            try:
                block.close()
            except BufferError:  # a view is still alive (an exception's frame): it goes with the view
                pass
