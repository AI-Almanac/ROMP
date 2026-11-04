"""Per-window amounts for every backend (``docs/onset_spec.md`` section 13), done by splitting jobs, not kernels.

An amount may have shape ``(S,)`` or ``(S, W)`` with ``amount_windows`` giving the ``W`` window start leads. A
candidate uses column ``w`` of the window its lead falls in (``lead = day - split + 1``, ``split`` 0 outside stitched
mode; leads before the first start use the first window). Each job is cut into one segment per window, every segment
is run by the ordinary backend with that window's ``(S,)`` amounts, and the job takes the first segment that is not
``NOT_FOUND``. That equals the first candidate that is not False, because ``NOT_FOUND`` means every candidate was
False. Segments keep the job's own ``origin`` so "season to date" spans still start at the job's first candidate.

The module also holds the small helpers the other backends share (`compare`, `span`, `first_column`). It is used by
`hcwf.onset.backends.reference`, `hcwf.onset.backends.numba_cpu` and `hcwf.onset.backends.gpu`.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from hcwf.onset import plan as P
from hcwf.onset.model import Detection, Status

__all__ = []


_NOT_FOUND = int(Status.NOT_FOUND)
_FAR = 1 << 40


def prepare(amounts: dict[str, np.ndarray], windows) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Validate amounts and windows and collapse one-column amounts.

    One-column ``(S, 1)`` amounts collapse to ``(S,)``. Amounts with ``W > 1`` columns need ``windows`` to hold ``W``
    strictly increasing start leads.

    Parameters
    ----------
    amounts : dict of str to numpy.ndarray
        The amounts by key, each of shape ``(S,)`` or ``(S, W)``.
    windows : array_like of int, optional
        The ``W`` window start leads (days, counted from 1), or ``None`` when no amount is windowed.

    Returns
    -------
    amounts : dict of str to numpy.ndarray
        The amounts, with ``(S, 1)`` arrays reduced to ``(S,)``.
    starts : numpy.ndarray of int64, shape (W,) or None
        The window start leads, or ``None`` when nothing is windowed (no two-dimensional amount, or only one column).

    Raises
    ------
    ValueError
        The amounts have different window counts, ``windows`` is missing or its length differs from the number of
        columns, or ``windows`` is not strictly increasing.
    """
    widths = {np.shape(v)[1] for v in amounts.values() if np.ndim(v) == 2}
    if not widths:
        return amounts, None
    if len(widths) > 1:
        raise ValueError(f"amounts have different window counts {sorted(widths)}")
    width = widths.pop()
    starts = None if windows is None else np.asarray(windows, dtype=np.int64).reshape(-1)
    if starts is None or len(starts) != width:
        raise ValueError(f"amounts have {width} window columns, so amount_windows needs {width} start leads")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("amount_windows must be strictly increasing")
    if width == 1:
        return {k: (v[:, 0] if np.ndim(v) == 2 else v) for k, v in amounts.items()}, None
    return amounts, starts


def _column(amounts: dict[str, np.ndarray], w: int) -> dict[str, np.ndarray]:
    """Return the ``(S,)`` amounts of window ``w``.

    Parameters
    ----------
    amounts : dict of str to numpy.ndarray
        The amounts by key, each of shape ``(S,)`` or ``(S, W)``.
    w : int
        The window index (column).

    Returns
    -------
    dict of str to numpy.ndarray
        The amounts with two-dimensional arrays reduced to column ``w``; one-dimensional arrays are passed through.
    """
    return {k: (v[:, w] if np.ndim(v) == 2 else v) for k, v in amounts.items()}


def run(plan: P.Plan, amounts: dict[str, np.ndarray], starts: np.ndarray, jobs, call) -> Detection:
    """Run jobs window by window with a backend call and merge the results.

    Curve jobs use the window of their first candidate. For first passage a job is decided by its earliest segment
    that finds something other than ``NOT_FOUND``.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    amounts : dict of str to numpy.ndarray
        The amounts by key, each of shape ``(S,)`` or ``(S, W)``.
    starts : numpy.ndarray of int64, shape (W,)
        The window start leads, strictly increasing.
    jobs : hcwf.onset.model.Jobs or hcwf.onset.model.StitchedJobs
        The jobs to run; a ``split`` field (stitched jobs) shifts the lead of a day.
    call : callable
        Called as ``call(sub_jobs, sub_amounts)`` with the jobs of one window segment and that window's ``(S,)``
        amounts; returns a `hcwf.onset.model.Detection`.

    Returns
    -------
    hcwf.onset.model.Detection
        The merged ``day`` (``-1`` where none) and ``status`` per job; ``trace`` is ``None``.
    """
    n = len(jobs.series)
    start = np.asarray(jobs.start, dtype=np.int64)
    end = np.asarray(jobs.end, dtype=np.int64)
    origin = start if jobs.origin is None else np.asarray(jobs.origin, dtype=np.int64)
    split = np.asarray(jobs.split, dtype=np.int64) if hasattr(jobs, "split") else np.zeros(n, dtype=np.int64)
    day = np.full(n, -1, dtype=np.int64)
    status = np.full(n, _NOT_FOUND, dtype=np.int8)
    edges = np.concatenate([[-_FAR], starts[1:], [_FAR]])
    pending = np.ones(n, dtype=bool)
    for w in range(len(starts)):
        if plan.family == P.CURVE:
            lead = start - split + 1
            mask = (lead >= edges[w]) & (lead < edges[w + 1])
            seg_start, seg_end = start, end
        else:
            seg_start = np.maximum(start, split + edges[w] - 1)
            seg_end = np.minimum(end, split + edges[w + 1] - 2)
            mask = pending & (seg_end >= seg_start)
        sel = np.nonzero(mask)[0]
        if len(sel) == 0:
            continue
        fields = {f.name: np.asarray(getattr(jobs, f.name))[sel] for f in dataclasses.fields(jobs)
                  if f.name != "origin"}
        fields.update(start=seg_start[sel], end=seg_end[sel], origin=origin[sel])
        out = call(type(jobs)(**fields), _column(amounts, w))
        got = out.status != _NOT_FOUND
        day[sel[got]] = out.day[got]
        status[sel[got]] = out.status[got]
        pending[sel[got]] = False
    return Detection(day, status, None)


def compare(op: int, value, amount):
    """Compare ``value`` with ``amount`` elementwise using an operator code.

    Parameters
    ----------
    op : int
        The operator code: ``plan.GE``, ``plan.GT`` or ``plan.LE``; any other code means less-than.
    value : numpy.ndarray
        The left-hand side.
    amount : numpy.ndarray or int or float
        The right-hand side, broadcastable against ``value``.

    Returns
    -------
    numpy.ndarray of bool
        ``value <op> amount``, with the shape of the broadcast.
    """
    if op == P.GE:
        return value >= amount
    if op == P.GT:
        return value > amount
    if op == P.LE:
        return value <= amount
    return value < amount


def span(job, first: int, last: int, since_start: bool):
    """Return the absolute ``[a, b)`` of a span for every candidate of a block.

    Parameters
    ----------
    job : object
        A block with ``origin`` and candidate days ``d``, both of shape ``(jobs, candidates)`` (or broadcastable).
    first : int
        The span's first day relative to the candidate day.
    last : int
        The span's end (exclusive) relative to the candidate day.
    since_start : bool
        Start the span at the job's ``origin`` ("season to date") instead of at ``d + first``.

    Returns
    -------
    a : numpy.ndarray of int64
        The first day of the span, shape of ``job.d``.
    b : numpy.ndarray of int64
        The end of the span, exclusive, shape of ``job.d``.
    """
    a = job.origin + 0 * job.d if since_start else job.d + first
    return a, job.d + last


def first_column(plan: P.Plan, start: np.ndarray, origin: np.ndarray) -> int:
    """Return the earliest history day any candidate's spans can read (never negative).

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition; its conditions, vetoes and report span set how far back spans reach.
    start : numpy.ndarray of int64, shape (jobs,)
        The first candidate day of each job.
    origin : numpy.ndarray of int64, shape (jobs,)
        The day "season to date" spans start from, per job.

    Returns
    -------
    int
        The smallest day index that is read, at least 0.
    """
    offsets = [0]
    for cond in plan.conditions:
        offsets.append(0 if cond.since_start else cond.first)
    offsets.extend(v.first - (v.placement == P.BEGINS) for v in plan.vetoes)
    if plan.report_first_wet and plan.report_span is not None:
        offsets.append(0 if plan.report_span[2] else plan.report_span[0])
    return max(0, min(int(start.min()) + min(offsets), int(origin.min())))
