"""Run independent items (years, files, folds, climatologies) in worker processes; results come back in item order.

`start_items` is the general function; `start_years`/`map_years`/`concat_years` are its year-keyed callers.

``parallel: false`` in the config keeps everything in this process. Otherwise the years of a task (pair tables,
onset tables, climatology pairs) are spread over `worker_count` processes (``parallel: {workers: N}``; default the
cores available, never more than the years):

* Every year goes to a worker. While the workers boot (they are started first, with the "spawn" method: a fork
  after numba's threads ran is unsafe; each gets cores // workers numba threads) this process pre-warms what all the
  years share (thresholds, masks; see the `prewarm` argument), so each is read, built and logged once, here, and
  then travels to the workers as one small snapshot. The pool is kept for the run (see `shutdown`), so the next
  stage (the climatology pairs after the model pairs) reuses the booted workers, and this process is free to do
  work of its own (the climatological onsets) while the workers run: `start_items` returns at once with a
  `Pending` whose `results()` waits.
* Workers receive file paths and settings, never data cubes: each reads its own year. What a worker detects or
  builds is sent back and absorbed here in year order, so the observed-onset cache on disk is written once, by this
  process.
* Log records and warnings raised in a worker are forwarded to this process's handlers (run.log, console) with
  the year in the message; see hcwf.utils.log.forwarding.
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
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from hcwf.utils import log as logs
from hcwf.utils.workers import cores, worker_count

log = logging.getLogger(__name__)

NO_DATA = "No data was successfully processed"
"""Start of the ValueError the multi-year functions raise when no year yielded data (a skipped year in a chunk)."""
_tokens = itertools.count(1)
_installed = {"token": None}


def _mask_modules():
    """The mask-building modules whose per-grid results are shared with the workers."""
    from hcwf.utils import land_mask, mask, region

    return mask, land_mask, region


def snapshot() -> dict:
    """What the workers need that this process has already read or built (small arrays only, never data cubes)."""
    from hcwf.io import input as io_input
    from hcwf.stats import observed_cache

    mask, land_mask, region = _mask_modules()
    return {"thresholds": dict(io_input._loaded_thresholds), "observed": observed_cache.export_state(),
            "masks": (dict(mask._BUILT), dict(mask._RECORDS)), "land": dict(land_mask._LAND_BOOL),
            "keep": dict(region._KEEP_CACHE)}


def install(state: dict, token: int) -> None:
    """Make a snapshot this (worker) process's read-once inputs; does nothing if it already holds that snapshot."""
    if _installed["token"] == token:
        return
    from hcwf.io import input as io_input
    from hcwf.stats import observed_cache

    mask, land_mask, region = _mask_modules()
    io_input._loaded_thresholds.update(state["thresholds"])
    observed_cache.install_state(state["observed"])
    mask._BUILT.update(state["masks"][0])
    mask._RECORDS.update(state["masks"][1])
    land_mask._LAND_BOOL.update(state["land"])
    region._KEEP_CACHE.update(state["keep"])
    _installed["token"] = token


def collect() -> dict:
    """What this (worker) process found since the last call: new observed onsets and masks, to send back."""
    from hcwf.stats import observed_cache

    mask, _, _ = _mask_modules()
    return {"observed": observed_cache.drain_new(), "records": dict(mask._RECORDS)}


def absorb(found: dict) -> None:
    """Take what a worker found into this process (observed onsets into the cache, new mask records)."""
    from hcwf.stats import observed_cache

    mask, _, _ = _mask_modules()
    observed_cache.absorb(found["observed"])
    for name, record in found["records"].items():
        mask._RECORDS.setdefault(name, record)


def _warm() -> int:
    """Import everything a task uses (so a worker boots while this process pre-warms the shared inputs)."""
    import hcwf.stats.benchmark  # noqa: F401
    import hcwf.stats.bins  # noqa: F401
    import hcwf.stats.conditional_climatology  # noqa: F401

    return os.getpid()


def _run_item(func, item, args, kwargs, state, token, label):
    """Worker entry: install the snapshot, run func(item, *args, **kwargs); returns (result, what was found)."""
    install(state, token)
    logs.set_worker_year(label)
    return func(item, *args, **kwargs), collect()


def _year_call(year, func, args, kwargs, skip):
    """func(*args, **kwargs with years=[year]); None where it raises the "no data" ValueError of a skipped year."""
    try:
        return func(*args, **{**kwargs, "years": [year]})
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


def _pool_for(size: int, threads: int) -> ProcessPoolExecutor:
    """The run's pool of `size` booted workers with `threads` numba threads each, started now if there is none like it.

    A pool is reused while the size, thread count and log handlers are the same; otherwise the old one is stopped.
    New workers begin importing the task modules at once.
    """
    key = (size, threads, tuple(id(h) for h in logging.getLogger(logs.ROOT).handlers))
    if _pool["executor"] is not None and _pool["key"] == key:
        return _pool["executor"]
    shutdown()
    saved = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
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
        if saved is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = saved
    _pool.update(executor=executor, key=key, stack=stack)
    return executor


class Pending:
    """Work started by start_items / start_years / start_concat: `result()` waits for it (once) and gives the value."""

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


def start_years(func, years, kwargs: dict, args: tuple = (), skip: str | None = NO_DATA,
                prewarm: Callable[[dict], None] | None = None) -> Pending:
    """Start func(*args, **kwargs with years=[year]) for each year; the Pending gives the results in year order.

    start_items over the years (all in workers when kwargs["parallel"] is on and there is more than one year);
    `prewarm(kwargs)`, if given, is called here once with all the years before the jobs are sent. `skip` is the start
    of the ValueError message that means "this year had no data" (a missing file): that year's result is None
    rather than an error, as the multi-year functions skip such years.
    """
    years = list(years)
    kwargs = {k: v for k, v in kwargs.items() if k != "years"}
    return start_items(_year_call, years, (func, args, kwargs, skip), config=kwargs, label=int,
                       prewarm=None if prewarm is None else lambda: prewarm({**kwargs, "years": years}))


def map_years(func, years, kwargs: dict, args: tuple = (), skip: str | None = NO_DATA,
              prewarm: Callable[[dict], None] | None = None) -> list:
    """func(*args, **kwargs with years=[year]) for each year, results in year order (None for a skipped year).

    start_years, waited for at once; see there for the arguments.
    """
    return start_years(func, years, kwargs, args, skip, prewarm).result()


def map_items(func, items, args: tuple = (), kwargs: dict | None = None, **options) -> list:
    """func(item, *args, **kwargs) for each item, results in item order (start_items, waited for at once)."""
    return start_items(func, items, args, kwargs, **options).result()


def start_concat(func, kwargs: dict, args: tuple = (), prewarm: Callable[[dict], None] | None = None) -> Pending:
    """Start the pair tables of func(*args, **kwargs) over kwargs["years"]; the Pending gives them concatenated.

    With one worker func runs once on all the years, as it always did, before this returns. Otherwise each year is
    computed by start_years and the tables joined in year order; a year without data is left out, and
    ValueError("No data ...") is raised when none has any.
    """
    years = kwargs["years"]
    if worker_count(kwargs, len(years)) <= 1:
        return Pending.finished(func(*args, **kwargs))
    pending = start_years(func, years, kwargs, args, prewarm=prewarm)

    def join():
        """The year tables, concatenated."""
        parts = [part for part in pending.result() if part is not None]
        if not parts:
            raise ValueError(f"{NO_DATA} for any year")
        return pd.concat(parts, ignore_index=True)

    return Pending(join)


def concat_years(func, kwargs: dict, args: tuple = (), prewarm: Callable[[dict], None] | None = None):
    """The pair tables of func(*args, **kwargs) over kwargs["years"], concatenated in year order (start_concat, waited for)."""
    return start_concat(func, kwargs, args, prewarm).result()
