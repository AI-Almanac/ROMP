"""Per-window amounts for every backend (docs/onset_spec.md section 13), done by splitting jobs rather than kernels.

An amount may have shape (S,) or (S, W) with `amount_windows` giving the W
window start leads. A candidate uses column w of the window its lead falls in
(lead = day - split + 1, split 0 outside stitched mode; leads before the first
start use the first window). Each job is cut into one segment per window, every
segment is run by the ordinary backend with that window's (S,) amounts, and the
job takes the first segment that is not NOT_FOUND. That equals the first
candidate that is not False, because NOT_FOUND means every candidate was False.
Segments keep the job's own `origin` so "season to date" spans still start at
the job's first candidate.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from momp.onset import plan as P
from momp.onset.model import Detection, Status

_NOT_FOUND = int(Status.NOT_FOUND)
_FAR = 1 << 40


def prepare(amounts: dict[str, np.ndarray], windows) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Validate amounts and windows; return (amounts, starts) where starts is None when nothing is windowed.

    One-column (S, 1) amounts collapse to (S,). Amounts with W > 1 columns need
    `windows` to hold W strictly increasing start leads.
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
    """The (S,) amounts of window w."""
    return {k: (v[:, w] if np.ndim(v) == 2 else v) for k, v in amounts.items()}


def run(plan: P.Plan, amounts: dict[str, np.ndarray], starts: np.ndarray, jobs, call) -> Detection:
    """Run Jobs or StitchedJobs window by window with `call(sub_jobs, sub_amounts) -> Detection` and merge.

    Curve jobs use the window of their first candidate. For first passage a job
    is decided by its earliest segment that finds something other than NOT_FOUND.
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
