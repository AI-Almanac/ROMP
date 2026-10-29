"""Run independent years in worker processes and give back the results in year order.

``parallel: false`` in the config keeps everything in this process. Otherwise the years of a task (pair tables,
onset tables, climatology pairs) are spread over `worker_count` processes (``parallel: {workers: N}``; default the
cores available, never more than the years):

* The first year runs in this process while the workers start up, so everything read once (thresholds, masks, the
  observed-onset cache) is warm and logged here before the workers get a copy of it. Workers are started with the
  "spawn" method (a fork after numba's threads ran is unsafe) and each gets cores // workers numba threads.
* Workers receive file paths and settings, never data cubes: each reads its own year. The read-once inputs
  (loaded thresholds, built masks, observed onsets already detected) travel as one small snapshot with each task;
  what a worker detects or builds is sent back and absorbed here in year order, so the observed-onset cache on
  disk is written once, by this process.
* The results are collected in year order whatever order the workers finish in, so a parallel run gives exactly
  the numbers of a serial one.
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from momp.utils.workers import cores, worker_count

log = logging.getLogger(__name__)

NO_DATA = "No data was successfully processed"
"""Start of the ValueError the multi-year functions raise when no year yielded data (a skipped year in a chunk)."""
_tokens = itertools.count(1)
_installed = {"token": None}


def _mask_modules():
    """The mask-building modules whose per-grid results are shared with the workers."""
    from momp.utils import land_mask, mask, region

    return mask, land_mask, region


def snapshot() -> dict:
    """What the workers need that this process has already read or built (small arrays only, never data cubes)."""
    from momp.io import input as io_input
    from momp.stats import observed_cache

    mask, land_mask, region = _mask_modules()
    return {"thresholds": dict(io_input._loaded_thresholds), "observed": observed_cache.export_state(),
            "masks": (dict(mask._BUILT), dict(mask._RECORDS)), "land": dict(land_mask._LAND_BOOL),
            "keep": dict(region._KEEP_CACHE)}


def install(state: dict, token: int) -> None:
    """Make a snapshot this (worker) process's read-once inputs; does nothing if it already holds that snapshot."""
    if _installed["token"] == token:
        return
    from momp.io import input as io_input
    from momp.stats import observed_cache

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
    from momp.stats import observed_cache

    mask, _, _ = _mask_modules()
    return {"observed": observed_cache.drain_new(), "records": dict(mask._RECORDS)}


def absorb(found: dict) -> None:
    """Take what a worker found into this process (observed onsets into the cache, new mask records)."""
    from momp.stats import observed_cache

    mask, _, _ = _mask_modules()
    observed_cache.absorb(found["observed"])
    for name, record in found["records"].items():
        mask._RECORDS.setdefault(name, record)


def _warm() -> int:
    """Import everything a task uses (so a worker boots while this process does the first year)."""
    import momp.stats.benchmark  # noqa: F401
    import momp.stats.bins  # noqa: F401
    import momp.stats.conditional_climatology  # noqa: F401

    return os.getpid()


def _run_chunk(func, args, kwargs, years, state, token, skip):
    """Worker entry: install the snapshot, run func on the years; returns (result or None, what was found)."""
    install(state, token)
    try:
        result = func(*args, **{**kwargs, "years": list(years)})
    except ValueError as exc:
        if not skip or not str(exc).startswith(skip):
            raise
        result = None
    return result, collect()


def _run_here(func, args, kwargs, years, skip):
    """Run func on the years in this process; None where it raises the "no data" ValueError of a skipped year."""
    try:
        return func(*args, **{**kwargs, "years": list(years)})
    except ValueError as exc:
        if not skip or not str(exc).startswith(skip):
            raise
        return None


def map_years(func, years, kwargs: dict, args: tuple = (), skip: str | None = NO_DATA) -> list:
    """func(*args, **kwargs with years=[year]) for each year, results in year order (None for a skipped year).

    Runs in worker processes when kwargs["parallel"] is on and there is more than one year; see the module text.
    `skip` is the start of the ValueError message that means "this year had no data" (a missing file): that year's
    result is None rather than an error, as the multi-year functions skip such years.
    """
    years = list(years)
    workers = worker_count(kwargs, len(years))
    if workers <= 1:
        return [_run_here(func, args, kwargs, [year], skip) for year in years]
    kwargs = {k: v for k, v in kwargs.items() if k != "years"}
    pool_size = workers - 1
    saved = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, cores() // workers))
    try:
        with ProcessPoolExecutor(max_workers=pool_size, mp_context=multiprocessing.get_context("spawn")) as pool:
            booting = [pool.submit(_warm) for _ in range(pool_size)]
            first = _run_here(func, args, kwargs, years[:1], skip)
            for job in booting:
                job.result()
            state, token = snapshot(), next(_tokens)
            jobs = [pool.submit(_run_chunk, func, args, kwargs, [year], state, token, skip) for year in years[1:]]
            results = [first]
            for job in jobs:
                result, found = job.result()
                absorb(found)
                results.append(result)
    except BrokenProcessPool as exc:
        raise RuntimeError(f"a worker process died ({exc}); run with parallel: false to see the error in this "
                           "process") from exc
    finally:
        if saved is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = saved
    return results


def concat_years(func, kwargs: dict, args: tuple = ()):
    """The pair tables of func(*args, **kwargs) over kwargs["years"], concatenated in year order.

    With one worker func runs once on all the years, as it always did. Otherwise each year is computed by map_years
    and the tables joined; a year without data is left out, and ValueError("No data ...") is raised when none has any.
    """
    years = kwargs["years"]
    if worker_count(kwargs, len(years)) <= 1:
        return func(*args, **kwargs)
    parts = [part for part in map_years(func, years, kwargs, args) if part is not None]
    if not parts:
        raise ValueError(f"{NO_DATA} for any year")
    return pd.concat(parts, ignore_index=True)
