"""CuPy backend: the reference algorithm (prefix tables, full candidate masks, first non-False via argmax) on the GPU.

Everything is integer arithmetic, so results are bit-identical to the
reference backend. Prefix tables for all series live on the device; jobs are
processed in blocks of (jobs x candidates) cells sized from free device memory.
"""
from __future__ import annotations

import numpy as np

from momp.onset import plan as P
from momp.onset.model import Detection, Jobs, Status

try:
    import cupy as cp
except ImportError:
    cp = None

TRUE, FALSE, UNKNOWN = 1, 0, -1
MISSING_FLAG, HORIZON_FLAG = 1, 2

BLOCK_CELLS: int | None = None
"""Job x candidate cells per block; None sizes blocks from free device memory."""

BYTES_PER_CELL = 400
"""Conservative device bytes of temporaries per evaluated job x candidate cell."""

MEMORY_FRACTION = 0.4
"""Share of the free device memory (after the tables are built) one block may use."""

MAX_AUTO_CELLS = 100_000_000
"""Cap on auto-sized blocks, past which extra size no longer buys speed."""


def available() -> bool:
    """True when CuPy imports and a CUDA device can run a kernel; never raises."""
    if cp is None:
        return False
    try:
        return int(cp.arange(3).sum()) == 3
    except Exception:
        return False


def _block_cells() -> int:
    """Cells per block: the BLOCK_CELLS override, else a fraction of free device memory."""
    if BLOCK_CELLS is not None:
        return max(1, int(BLOCK_CELLS))
    cp.get_default_memory_pool().free_all_blocks()
    free, _ = cp.cuda.Device().mem_info
    return int(min(MAX_AUTO_CELLS, max(1, free * MEMORY_FRACTION / BYTES_PER_CELL)))


def detect(plan: P.Plan, tenths: np.ndarray, missing: np.ndarray, amounts: dict[str, np.ndarray],
           jobs: Jobs, *, trace: bool = False) -> Detection:
    """Run every job on the GPU and return the onset day index (-1 if none) and Status per job (NumPy arrays)."""
    if trace:
        raise NotImplementedError("use the reference backend for traces")
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.full(n_jobs, int(Status.NOT_FOUND), dtype=np.int8)
    if n_jobs == 0:
        return Detection(day, status, None)
    series = cp.asarray(jobs.series, dtype=cp.int64)
    start = cp.asarray(jobs.start, dtype=cp.int64)
    end = cp.asarray(jobs.end, dtype=cp.int64)
    data_end = cp.asarray(jobs.data_end, dtype=cp.int64)
    n_cand = cp.maximum(end - start + 1, 0)
    width = int(n_cand.max())
    if width == 0:
        return Detection(day, status, None)
    tables = _Tables(plan, tenths, missing, amounts)
    block = max(1, _block_cells() // width)
    day_d = cp.full(n_jobs, -1, dtype=cp.int64)
    status_d = cp.full(n_jobs, int(Status.NOT_FOUND), dtype=cp.int8)
    for lo in range(0, n_jobs, block):
        sl = slice(lo, min(lo + block, n_jobs))
        w = int(n_cand[sl].max())
        if w == 0:
            continue
        job = _Block(series[sl], start[sl], end[sl], data_end[sl], w)
        if plan.family == P.CURVE:
            day_d[sl], status_d[sl] = _curve_block(plan, tables, job)
        else:
            day_d[sl], status_d[sl] = _first_passage_block(plan, tables, job)
    return Detection(cp.asnumpy(day_d), cp.asnumpy(status_d), None)


class _Block:
    """The jobs of one block plus their candidate grid (days d, validity mask)."""

    def __init__(self, series, start, end, data_end, width):
        """Build the (jobs, width) candidate day grid."""
        self.series = series[:, None]
        self.start = start[:, None]
        self.end = end[:, None]
        self.data_end = data_end[:, None]
        self.width = width
        self.d = start[:, None] + cp.arange(width, dtype=cp.int64)[None, :]
        self.valid = self.d <= end[:, None]


class _Tables:
    """Prefix tables over the series, built once per detect call on the device."""

    def __init__(self, plan, tenths, missing, amounts):
        """Upload the data and build prefix sums of rain, missing, wet and dry days, plus per-veto spell tables."""
        self.amounts = {k: cp.asarray(v, dtype=cp.int64) for k, v in amounts.items()}
        self.n_time = tenths.shape[1]
        missing = cp.asarray(missing, dtype=cp.bool_)
        x = cp.where(missing, 0, cp.asarray(tenths)).astype(cp.int64)
        wet_amount = self.amounts[plan.wet.key][:, None]
        wet = ~missing & _compare(plan.wet.op, x, wet_amount)
        dry = ~missing & ~wet
        self.x_shape = x.shape
        self.px = _prefix(x, cp.int64)
        del x
        self.pm = _prefix(missing, cp.int32)
        self.pw = _prefix(wet, cp.int32)
        self.pd = _prefix(dry, cp.int32)
        self.veto_prefix = [self._spell_prefix(v) for v in plan.vetoes]
        self.wet_order = None
        if plan.report_first_wet:
            # Wet days first, in time order, so the k-th wet day of a series is wet_order[s, k].
            self.wet_order = cp.argsort(~wet, axis=1, kind="stable").astype(cp.int32)
        del wet, dry, missing
        self.pd = None

    def _spell_prefix(self, veto: P.VetoPlan):
        """Prefix sum over start days u of "a spell of this veto starts at u and lies fully in the data"."""
        n_time, length = self.n_time, veto.length
        holds = cp.zeros((self.x_shape[0], n_time + 1), dtype=cp.bool_)
        if length <= n_time:
            span = n_time - length + 1
            if veto.kind == P.DRY_RUN:
                holds[:, :span] = (self.pd[:, length:] - self.pd[:, :span]) == length
            else:
                total = self.px[:, length:] - self.px[:, :span]
                amount = self.amounts[veto.cmp.key][:, None]
                complete = (self.pm[:, length:] - self.pm[:, :span]) == 0
                holds[:, :span] = complete & _compare(veto.cmp.op, total, amount)
        return _prefix(holds, cp.int32)

    def next_wet(self, series, lo):
        """First wet day at or after lo in each series, n_time if there is none."""
        rank = self.pw[series, lo]
        total = self.pw[series, self.n_time]
        found = self.wet_order[series, cp.minimum(rank, self.n_time - 1)].astype(cp.int64)
        return cp.where(rank < total, found, self.n_time)


def _prefix(values, dtype):
    """Prefix sums along time with a leading zero column, so table[:, t] sums days [0, t)."""
    out = cp.zeros((values.shape[0], values.shape[1] + 1), dtype=dtype)
    cp.cumsum(values, axis=1, dtype=dtype, out=out[:, 1:])
    return out


def _compare(op: int, value, amount):
    """Elementwise comparison selected by an operator code."""
    if op == P.GE:
        return value >= amount
    if op == P.GT:
        return value > amount
    if op == P.LE:
        return value <= amount
    return value < amount


def _gather(table, series, index):
    """table[series, index] with the index clipped into the table."""
    return table[series, cp.clip(index, 0, table.shape[1] - 1)]


def _observe(tables: _Tables, job: _Block, a, b):
    """Observability of [a, b): (observable, flags) where flags carry the missing and horizon bits."""
    e = job.data_end
    lo = cp.clip(a, 0, e)
    hi = cp.maximum(cp.minimum(b, e), lo)
    n_missing = tables.pm[job.series, hi] - tables.pm[job.series, lo]
    flags = (((a < 0) | (n_missing > 0)) * MISSING_FLAG + (b > e) * HORIZON_FLAG).astype(cp.int8)
    return flags == 0, flags


def _span(job: _Block, first: int, last: int, since_start: bool):
    """Absolute [a, b) of a span for every candidate of the block."""
    a = job.start + 0 * job.d if since_start else job.d + first
    return a, job.d + last


def _condition(plan_cond: P.CondPlan, tables: _Tables, job: _Block):
    """Three-valued code and flags of a Total or WetDays requirement."""
    a, b = _span(job, plan_cond.first, plan_cond.last, plan_cond.since_start)
    ok, flags = _observe(tables, job, a, b)
    if plan_cond.kind == P.TOTAL:
        value = _gather(tables.px, job.series, b) - _gather(tables.px, job.series, a)
        amount = tables.amounts[plan_cond.cmp.key][job.series]
        result = _compare(plan_cond.cmp.op, value, amount)
    else:
        value = _gather(tables.pw, job.series, b) - _gather(tables.pw, job.series, a)
        result = _compare(plan_cond.op, value, plan_cond.count)
    code = cp.where(ok, cp.where(result, TRUE, FALSE), UNKNOWN).astype(cp.int8)
    return code, cp.where(ok, 0, flags).astype(cp.int8)


def _veto(veto: P.VetoPlan, spell_prefix, tables: _Tables, job: _Block):
    """Code and flags of a veto: TRUE means a spell was confirmed, so the candidate is rejected."""
    a, b = _span(job, veto.first, veto.last, False)
    length = veto.length
    if veto.placement == P.INSIDE:
        if veto.last - veto.first < length:
            return cp.zeros(a.shape, dtype=cp.int8), cp.zeros(a.shape, dtype=cp.int8)
        u_last = b - length
        ok, flags = _observe(tables, job, a, b)
    else:
        u_last = b - 1
        ok, flags = _observe(tables, job, a, b + length - 1)
    u_last = cp.minimum(u_last, job.data_end - length)
    n_table = spell_prefix.shape[1]
    lo = cp.clip(a, 0, n_table - 1)
    hi = cp.maximum(cp.clip(u_last + 1, 0, n_table - 1), lo)
    confirmed = (spell_prefix[job.series, hi] - spell_prefix[job.series, lo]) > 0
    code = cp.where(confirmed, TRUE, cp.where(ok, FALSE, UNKNOWN)).astype(cp.int8)
    return code, cp.where(code == UNKNOWN, flags, 0).astype(cp.int8)


def _status_from_flags(flags):
    """CENSORED when the flags are exactly {horizon}, otherwise INSUFFICIENT_DATA."""
    return cp.where(flags == HORIZON_FLAG, int(Status.CENSORED), int(Status.INSUFFICIENT_DATA)).astype(cp.int8)


def _first_passage_block(plan: P.Plan, tables: _Tables, job: _Block):
    """Evaluate all candidates of a block and pick the first one that is not False; returns (day, status)."""
    shape = job.d.shape
    unknown_flags = cp.zeros(shape, dtype=cp.int8)
    any_false = cp.zeros(shape, dtype=cp.bool_)
    any_unknown = cp.zeros(shape, dtype=cp.bool_)
    for cond in plan.conditions:
        code, flags = _condition(cond, tables, job)
        any_false |= code == FALSE
        any_unknown |= code == UNKNOWN
        unknown_flags |= flags
    for veto, spell_prefix in zip(plan.vetoes, tables.veto_prefix):
        code, flags = _veto(veto, spell_prefix, tables, job)
        any_false |= code == TRUE
        any_unknown |= code == UNKNOWN
        unknown_flags |= flags
    state = cp.where(any_false | ~job.valid, FALSE, cp.where(any_unknown, UNKNOWN, TRUE)).astype(cp.int8)
    flags = cp.where(state == UNKNOWN, unknown_flags, 0).astype(cp.int8)

    decided = state != FALSE
    found = decided.any(axis=1)
    pick = cp.argmax(decided, axis=1)
    rows = cp.arange(len(pick))
    pick_state = state[rows, pick]
    pick_day = job.d[rows, pick]
    pick_flags = flags[rows, pick]

    status = cp.full(len(pick), int(Status.NOT_FOUND), dtype=cp.int8)
    day = cp.full(len(pick), -1, dtype=cp.int64)
    onset = found & (pick_state == TRUE)
    status = cp.where(onset, int(Status.ONSET), status).astype(cp.int8)
    day = cp.where(onset, pick_day, day)
    unknown = found & (pick_state == UNKNOWN)
    status = cp.where(unknown, _status_from_flags(pick_flags), status).astype(cp.int8)
    if plan.report_first_wet:
        first, last, since = plan.report_span
        series = job.series[:, 0]
        a = job.start[:, 0] if since else pick_day + first
        b = pick_day + last
        lo_raw = cp.maximum(a, 0)
        lo = cp.minimum(lo_raw, tables.n_time - 1)
        wet_day = tables.next_wet(series, lo)
        in_window = (lo_raw < tables.n_time) & (wet_day < b) & (wet_day < job.data_end[:, 0])
        day = cp.where(onset & in_window, wet_day, day)
    return day, status


def _curve_block(plan: P.Plan, tables: _Tables, job: _Block):
    """Evaluate the cumulative-anomaly rule for every job of a block; returns (day, status)."""
    start, end, series = job.start[:, 0], job.end[:, 0], job.series[:, 0]
    ok, flags = _observe(tables, job, job.start, job.end + 1)
    ok, flags = ok[:, 0], flags[:, 0]
    baseline = tables.amounts[plan.curve_key][series][:, None]
    steps = job.d - job.start + 1
    rain = _gather(tables.px, job.series, job.d + 1) - _gather(tables.px, job.series, job.start)
    cumulative = 100 * rain - baseline * steps
    masked = cp.where(job.valid, cumulative, np.iinfo(np.int64).max)
    minimum = cp.argmin(masked, axis=1)
    min_day = start + minimum
    any_valid = job.valid.any(axis=1)
    not_found = (min_day == end) | ~any_valid
    status = cp.where(not_found, int(Status.NOT_FOUND), int(Status.ONSET)).astype(cp.int8)
    day = cp.where(status == int(Status.ONSET), min_day + (1 if plan.curve_after_min else 0), -1)
    unobservable = ~ok & any_valid
    status = cp.where(unobservable, _status_from_flags(flags), status).astype(cp.int8)
    day = cp.where(unobservable, -1, day)
    return day.astype(cp.int64), status
