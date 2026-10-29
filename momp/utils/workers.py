"""How many workers a run uses, and ordered, bounded read-ahead of independent items.

The config says ``parallel: false`` (one process, no threads) or leaves parallel on, optionally with
``parallel: {workers: N}``. With no number the workers are the cores available to this process,
never more than there are tasks. Nothing here imports the rest of momp, so any module may use it.
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Mapping


def cores() -> int:
    """The cores this process may use (its CPU affinity, else the machine's count)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def worker_count(config: Mapping, tasks: int) -> int:
    """The workers to use for `tasks` independent tasks: 1 (serial) when config["parallel"] is false.

    Otherwise config["parallel_workers"] when set, else the cores available, never more than tasks.
    """
    if not config.get("parallel", True) or tasks <= 1:
        return 1
    limit = config.get("parallel_workers") or cores()
    return max(1, min(int(limit), int(tasks)))


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
