"""CuPy backend: the reference algorithm (prefix tables, full candidate masks, first non-False via argmax) on the GPU.

Everything is integer arithmetic, so results are bit-identical to the
reference backend. Prefix tables for all series live on the device; jobs are
processed in blocks of (jobs x candidates) cells sized from free device memory.

detect_stitched does the same over observed history followed by a forecast, joining
separate history and forecast tables at each job's split instead of copying history.
"""
from __future__ import annotations

import numpy as np

from momp.onset import plan as P
from momp.onset.backends import windowed
from momp.onset.model import Detection, Jobs, Status, StitchedJobs

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
           jobs: Jobs, *, trace: bool = False, amount_windows=None) -> Detection:
    """Run every job on the GPU and return the onset day index (-1 if none) and Status per job (NumPy arrays).

    Amounts may be (S, W) with amount_windows the W window start leads (backends/windowed.py).
    """
    if trace:
        raise NotImplementedError("use the reference backend for traces")
    amounts, starts = windowed.prepare(amounts, amount_windows)
    if starts is not None:
        return windowed.run(plan, amounts, starts, jobs, lambda j, a: detect(plan, tenths, missing, a, j))
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.full(n_jobs, int(Status.NOT_FOUND), dtype=np.int8)
    if n_jobs == 0:
        return Detection(day, status, None)
    series = cp.asarray(jobs.series, dtype=cp.int64)
    start = cp.asarray(jobs.start, dtype=cp.int64)
    end = cp.asarray(jobs.end, dtype=cp.int64)
    data_end = cp.asarray(jobs.data_end, dtype=cp.int64)
    origin = start if jobs.origin is None else cp.asarray(jobs.origin, dtype=cp.int64)
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
        job = _Block(series[sl], start[sl], end[sl], data_end[sl], w, origin[sl])
        if plan.family == P.CURVE:
            day_d[sl], status_d[sl] = _curve_block(plan, tables, job)
        else:
            day_d[sl], status_d[sl] = _first_passage_block(plan, tables, job)
    return Detection(cp.asnumpy(day_d), cp.asnumpy(status_d), None)


class _Block:
    """The jobs of one block plus their candidate grid (days d, validity mask)."""

    def __init__(self, series, start, end, data_end, width, origin):
        """Build the (jobs, width) candidate day grid."""
        self.series = series[:, None]
        self.start = start[:, None]
        self.origin = origin[:, None]
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
    a = job.origin + 0 * job.d if since_start else job.d + first
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
        a = job.origin[:, 0] if since else pick_day + first
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


def _variants(cell: np.ndarray, amount: np.ndarray):
    """Unique (cell, amount) pairs as (cells, amounts) and each job's index into them."""
    pairs = np.stack([cell, amount], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique[:, 0].copy(), unique[:, 1].copy(), inverse.reshape(-1).astype(np.int64)


def _first_column(plan: P.Plan, start: np.ndarray, origin: np.ndarray) -> int:
    """Earliest history day any candidate's spans can read (never negative)."""
    offsets = [0]
    for cond in plan.conditions:
        offsets.append(0 if cond.since_start else cond.first)
    offsets.extend(v.first for v in plan.vetoes)
    if plan.report_first_wet and plan.report_span is not None:
        offsets.append(0 if plan.report_span[2] else plan.report_span[0])
    return max(0, min(int(start.min()) + min(offsets), int(origin.min())))


def detect_stitched(plan: P.Plan, hist_tenths: np.ndarray, hist_missing: np.ndarray, fc_tenths: np.ndarray,
                    fc_missing: np.ndarray, amounts: dict[str, np.ndarray], jobs: StitchedJobs, *,
                    trace: bool = False, amount_windows=None) -> Detection:
    """Run stitched jobs (history before each split, then a forecast row) on the GPU without copying history into forecasts.

    History prefix tables are built once over the cells and columns the jobs can
    reach, with one wet-day (and dry-window) variant per distinct (cell, amount)
    pair since amounts are per forecast series. Forecast tables are built over
    the forecast rows. Any interval is read as P_hist(min(t, split)) plus
    P_fc(t - split) for t past the split, and spells that start in history and end in
    the forecast come from a small per-job seam table. The result equals
    detect on the explicit concatenation and never reads history at or after a split.
    Per-window amounts work as in detect().
    """
    if trace:
        raise NotImplementedError("use the reference backend for traces")
    amounts, starts = windowed.prepare(amounts, amount_windows)
    if starts is not None:
        return windowed.run(plan, amounts, starts, jobs, lambda j, a: detect_stitched(
            plan, hist_tenths, hist_missing, fc_tenths, fc_missing, a, j))
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.full(n_jobs, int(Status.NOT_FOUND), dtype=np.int8)
    if n_jobs == 0:
        return Detection(day, status, None)
    start_h = np.asarray(jobs.start, dtype=np.int64)
    end_h = np.asarray(jobs.end, dtype=np.int64)
    active = end_h >= start_h
    if not active.any():
        return Detection(day, status, None)
    tables = _StitchedTables(plan, hist_tenths, hist_missing, fc_tenths, fc_missing, amounts, jobs, active)
    n_cand = cp.maximum(cp.asarray(end_h - start_h + 1), 0)
    width = int(n_cand.max())
    max_length = max((v.length for v in plan.vetoes), default=1)
    block = max(1, _block_cells() // max(width, max_length))
    day_d = cp.full(n_jobs, -1, dtype=cp.int64)
    status_d = cp.full(n_jobs, int(Status.NOT_FOUND), dtype=cp.int8)
    for lo in range(0, n_jobs, block):
        sl = slice(lo, min(lo + block, n_jobs))
        w = int(n_cand[sl].max())
        if w == 0:
            continue
        job = _StitchedBlock(tables, sl, w)
        if plan.family == P.CURVE:
            day_d[sl], status_d[sl] = _curve_block_s(plan, tables, job)
        else:
            day_d[sl], status_d[sl] = _first_passage_block_s(plan, tables, job)
    return Detection(cp.asnumpy(day_d), cp.asnumpy(status_d), None)


class _StitchedTables:
    """History tables over the reachable cells and columns, forecast tables, and the per-job row indices into them."""

    def __init__(self, plan, hist_tenths, hist_missing, fc_tenths, fc_missing, amounts, jobs, active):
        """Build forecast tables with _Tables and history tables per cell and per (cell, amount) variant."""
        self.fc = _Tables(plan, fc_tenths, fc_missing, amounts)
        self.n_fc = self.fc.n_time
        series_h = np.asarray(jobs.series, dtype=np.int64)
        cell_h = np.asarray(jobs.cell, dtype=np.int64)
        split_h = np.asarray(jobs.split, dtype=np.int64)
        if split_h[active].min() < 0 or split_h[active].max() > hist_tenths.shape[1]:
            raise ValueError("a job's split lies outside the history")
        c1 = int(split_h[active].max())
        start_h = np.asarray(jobs.start, dtype=np.int64)
        origin_h = start_h if jobs.origin is None else np.asarray(jobs.origin, dtype=np.int64)
        self.c0 = min(_first_column(plan, start_h[active], origin_h[active]), c1)
        self.width = c1 - self.c0
        cells, cell_idx = np.unique(cell_h, return_inverse=True)
        cell_idx = cell_idx.reshape(-1).astype(np.int64)
        miss = cp.asarray(hist_missing[cells, self.c0:c1], dtype=cp.bool_)
        x = cp.where(miss, 0, cp.asarray(hist_tenths[cells, self.c0:c1])).astype(cp.int64)
        self.px = _prefix(x, cp.int64)
        self.pm = _prefix(miss, cp.int32)

        wet_amount_h = np.asarray(amounts[plan.wet.key], dtype=np.int64)[series_h]
        wcell, wamt, wv = _variants(cell_idx, wet_amount_h)
        wcell_d, wamt_d = cp.asarray(wcell), cp.asarray(wamt)
        miss_v = miss[wcell_d]
        wet = ~miss_v & _compare(plan.wet.op, x[wcell_d], wamt_d[:, None])
        self.pw = _prefix(wet, cp.int32)
        dry_prefix = _prefix(~miss_v & ~wet, cp.int32)
        self.wet_order = None
        if plan.report_first_wet:
            self.wet_order = cp.argsort(~wet, axis=1, kind="stable").astype(cp.int32)
        del wet, miss_v

        self.cell = cp.asarray(cell_idx)
        self.series = cp.asarray(series_h)
        self.split = cp.asarray(split_h)
        self.start = cp.asarray(start_h)
        self.origin = cp.asarray(origin_h)
        self.end = cp.asarray(np.asarray(jobs.end, dtype=np.int64))
        self.data_end = cp.minimum(cp.asarray(np.asarray(jobs.data_end, dtype=np.int64)), self.split + self.n_fc)
        self.wet_variant = cp.asarray(wv)
        self.spell_prefix, self.spell_variant = [], []
        for veto in plan.vetoes:
            span = self.width - veto.length + 1
            holds = cp.zeros((0, self.width + 1), dtype=cp.bool_)
            if veto.kind == P.DRY_RUN:
                variant = wv
                holds = cp.zeros((len(wcell), self.width + 1), dtype=cp.bool_)
                if span > 0:
                    holds[:, :span] = (dry_prefix[:, veto.length:] - dry_prefix[:, :span]) == veto.length
            else:
                amount_h = np.asarray(amounts[veto.cmp.key], dtype=np.int64)[series_h]
                vcell, vamt, variant = _variants(cell_idx, amount_h)
                vcell_d, vamt_d = cp.asarray(vcell), cp.asarray(vamt)
                holds = cp.zeros((len(vcell), self.width + 1), dtype=cp.bool_)
                if span > 0:
                    px_v, pm_v = self.px[vcell_d], self.pm[vcell_d]
                    total = px_v[:, veto.length:] - px_v[:, :span]
                    complete = (pm_v[:, veto.length:] - pm_v[:, :span]) == 0
                    holds[:, :span] = complete & _compare(veto.cmp.op, total, vamt_d[:, None])
            self.spell_prefix.append(_prefix(holds, cp.int32))
            self.spell_variant.append(cp.asarray(variant))
        del dry_prefix
        self.amounts = self.fc.amounts


class _StitchedBlock:
    """The jobs of one block: row indices, split, window, and the candidate grid (days d, validity mask)."""

    def __init__(self, tables: _StitchedTables, sl: slice, width: int):
        """Slice the per-job arrays and build the (jobs, width) candidate day grid."""
        self.series = tables.series[sl][:, None]
        self.cell = tables.cell[sl][:, None]
        self.wet_variant = tables.wet_variant[sl][:, None]
        self.split = tables.split[sl][:, None]
        self.start = tables.start[sl][:, None]
        self.origin = tables.origin[sl][:, None]
        self.end = tables.end[sl][:, None]
        self.data_end = tables.data_end[sl][:, None]
        self.n_time = self.split + tables.n_fc
        self.n_jobs = self.series.shape[0]
        self.variants = [v[sl][:, None] for v in tables.spell_variant]
        self.width = width
        self.d = self.start + cp.arange(width, dtype=cp.int64)[None, :]
        self.valid = self.d <= self.end


def _g(tables: _StitchedTables, hist_table, fc_table, hist_row, job: _StitchedBlock, t):
    """Prefix value at stitched index t: P_hist(min(t, split)) plus P_fc(t - split) once past the split (that table is 0 at 0)."""
    h = hist_table[hist_row, cp.clip(cp.minimum(t, job.split) - tables.c0, 0, tables.width)]
    f = fc_table[job.series, cp.clip(t - job.split, 0, tables.n_fc)]
    return h + f


def _observe_s(tables: _StitchedTables, job: _StitchedBlock, a, b):
    """Observability of the stitched interval [a, b): (observable, flags) with the missing and horizon bits."""
    e = job.data_end
    lo = cp.clip(a, 0, e)
    hi = cp.maximum(cp.minimum(b, e), lo)
    n_missing = _g(tables, tables.pm, tables.fc.pm, job.cell, job, hi) - _g(tables, tables.pm, tables.fc.pm, job.cell, job, lo)
    flags = (((a < 0) | (n_missing > 0)) * MISSING_FLAG + (b > e) * HORIZON_FLAG).astype(cp.int8)
    return flags == 0, flags


def _condition_s(plan_cond: P.CondPlan, tables: _StitchedTables, job: _StitchedBlock):
    """Three-valued code and flags of a Total or WetDays requirement on the stitched series."""
    a, b = _span(job, plan_cond.first, plan_cond.last, plan_cond.since_start)
    ok, flags = _observe_s(tables, job, a, b)
    if plan_cond.kind == P.TOTAL:
        value = (_g(tables, tables.px, tables.fc.px, job.cell, job, b) - _g(tables, tables.px, tables.fc.px, job.cell, job, a))
        amount = tables.amounts[plan_cond.cmp.key][job.series]
        result = _compare(plan_cond.cmp.op, value, amount)
    else:
        value = (_g(tables, tables.pw, tables.fc.pw, job.wet_variant, job, b)
                 - _g(tables, tables.pw, tables.fc.pw, job.wet_variant, job, a))
        result = _compare(plan_cond.op, value, plan_cond.count)
    code = cp.where(ok, cp.where(result, TRUE, FALSE), UNKNOWN).astype(cp.int8)
    return code, cp.where(ok, 0, flags).astype(cp.int8)


def _seam(veto: P.VetoPlan, tables: _StitchedTables, job: _StitchedBlock):
    """Per-job cumulative indicator (jobs, length) of spells starting in [split - length + 1, split - 1] that hold."""
    length = veto.length
    seam = cp.zeros((job.n_jobs, length), dtype=cp.int32)
    if length == 1:
        return seam
    i = cp.arange(length - 1, dtype=cp.int64)[None, :]
    u = job.split - length + 1 + i
    fc_len = i + 1
    inside = (u >= tables.c0) & (u + length <= job.n_time)
    split_col = cp.clip(job.split - tables.c0, 0, tables.width)
    u_col = cp.clip(u - tables.c0, 0, tables.width)
    fc_col = cp.clip(fc_len, 0, tables.n_fc)

    def hist_diff(table, row):
        """Sum of the history part [u, split) of a prefix table."""
        return table[row, split_col] - table[row, u_col]

    missing = hist_diff(tables.pm, job.cell) + tables.fc.pm[job.series, fc_col]
    if veto.kind == P.DRY_RUN:
        wet = hist_diff(tables.pw, job.wet_variant) + tables.fc.pw[job.series, fc_col]
        holds = inside & (missing == 0) & (wet == 0)
    else:
        total = hist_diff(tables.px, job.cell) + tables.fc.px[job.series, fc_col]
        amount = tables.amounts[veto.cmp.key][job.series]
        holds = inside & (missing == 0) & _compare(veto.cmp.op, total, amount)
    seam[:, 1:] = cp.cumsum(holds, axis=1, dtype=cp.int32)
    return seam


def _veto_s(veto: P.VetoPlan, spell_prefix, variant, fc_prefix, tables: _StitchedTables, job: _StitchedBlock):
    """Code and flags of a veto on the stitched series: TRUE means a spell was confirmed, so the candidate is rejected."""
    a, b = _span(job, veto.first, veto.last, False)
    length = veto.length
    if veto.placement == P.INSIDE:
        if veto.last - veto.first < length:
            return cp.zeros(a.shape, dtype=cp.int8), cp.zeros(a.shape, dtype=cp.int8)
        u_last = b - length
        ok, flags = _observe_s(tables, job, a, b)
    else:
        u_last = b - 1
        ok, flags = _observe_s(tables, job, a, b + length - 1)
    u_last = cp.minimum(u_last, job.data_end - length)
    lo = cp.clip(a, 0, job.n_time)
    hi = cp.maximum(cp.clip(u_last + 1, 0, job.n_time), lo)
    base = job.split - length + 1
    seam = _seam(veto, tables, job)
    rows = cp.arange(job.n_jobs)[:, None]
    x = cp.minimum(hi, base)
    count = cp.where(x > lo, spell_prefix[variant, cp.clip(x - tables.c0, 0, tables.width)]
                     - spell_prefix[variant, cp.clip(lo - tables.c0, 0, tables.width)], 0)
    x0, x1 = cp.maximum(lo, base), cp.minimum(hi, job.split)
    count = count + cp.where(x1 > x0, seam[rows, cp.clip(x1 - base, 0, length - 1)]
                             - seam[rows, cp.clip(x0 - base, 0, length - 1)], 0)
    y0 = cp.maximum(lo, job.split)
    count = count + cp.where(hi > y0, fc_prefix[job.series, cp.clip(hi - job.split, 0, tables.n_fc)]
                             - fc_prefix[job.series, cp.clip(y0 - job.split, 0, tables.n_fc)], 0)
    confirmed = count > 0
    code = cp.where(confirmed, TRUE, cp.where(ok, FALSE, UNKNOWN)).astype(cp.int8)
    return code, cp.where(code == UNKNOWN, flags, 0).astype(cp.int8)


def _first_wet_day_s(tables: _StitchedTables, job: _StitchedBlock, onset, pick_day, day, report_span):
    """Replace onset days by the first wet day in the report window (history part first, then the forecast)."""
    first, last, since = report_span
    a = job.origin[:, 0] if since else pick_day + first
    b = pick_day + last
    e = job.data_end[:, 0]
    split, series = job.split[:, 0], job.series[:, 0]
    lo = cp.maximum(a, 0)
    found = cp.zeros(lo.shape, dtype=cp.int64)
    hist_has = cp.zeros(lo.shape, dtype=cp.bool_)
    if tables.width > 0:
        row = job.wet_variant[:, 0]
        rank = tables.pw[row, cp.clip(lo - tables.c0, 0, tables.width)]
        total = tables.pw[row, cp.clip(split - tables.c0, 0, tables.width)]
        hist_has = (lo < split) & (rank < total)
        found = tables.c0 + tables.wet_order[row, cp.minimum(rank, tables.width - 1)].astype(cp.int64)
    hist_ok = hist_has & (found < b) & (found < e)
    if tables.n_fc > 0:
        f_lo = cp.minimum(cp.maximum(lo - split, 0), tables.n_fc - 1)
        wet_f = split + tables.fc.next_wet(series, f_lo)
        fc_ok = ~hist_has & (lo - split < tables.n_fc) & (wet_f < b) & (wet_f < e)
        day = cp.where(onset & fc_ok, wet_f, day)
    return cp.where(onset & hist_ok, found, day)


def _first_passage_block_s(plan: P.Plan, tables: _StitchedTables, job: _StitchedBlock):
    """Evaluate all candidates of a block of stitched jobs and pick the first one that is not False; returns (day, status)."""
    shape = job.d.shape
    unknown_flags = cp.zeros(shape, dtype=cp.int8)
    any_false = cp.zeros(shape, dtype=cp.bool_)
    any_unknown = cp.zeros(shape, dtype=cp.bool_)
    for cond in plan.conditions:
        code, flags = _condition_s(cond, tables, job)
        any_false |= code == FALSE
        any_unknown |= code == UNKNOWN
        unknown_flags |= flags
    for veto, spell_prefix, variant, fc_prefix in zip(plan.vetoes, tables.spell_prefix, job.variants,
                                                      tables.fc.veto_prefix):
        code, flags = _veto_s(veto, spell_prefix, variant, fc_prefix, tables, job)
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
        day = _first_wet_day_s(tables, job, onset, pick_day, day, plan.report_span)
    return day, status


def _curve_block_s(plan: P.Plan, tables: _StitchedTables, job: _StitchedBlock):
    """Evaluate the cumulative-anomaly rule for every stitched job of a block; returns (day, status)."""
    start, end, series = job.start[:, 0], job.end[:, 0], job.series[:, 0]
    ok, flags = _observe_s(tables, job, job.start, job.end + 1)
    ok, flags = ok[:, 0], flags[:, 0]
    baseline = tables.amounts[plan.curve_key][series][:, None]
    steps = job.d - job.start + 1
    rain = (_g(tables, tables.px, tables.fc.px, job.cell, job, job.d + 1)
            - _g(tables, tables.px, tables.fc.px, job.cell, job, job.start))
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
