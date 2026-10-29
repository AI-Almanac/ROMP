"""NumPy reference backend: the oracle that implements docs/onset_spec.md section 4 with full candidate masks.

Prefix tables are built once per call. Jobs are processed in blocks; for each
block every candidate of every job is evaluated at once as a (jobs x
candidates) array of three-valued codes, and the first candidate that is not
False decides the job.
"""
from __future__ import annotations

import numpy as np

from momp.onset import plan as P
from momp.onset.model import Detection, Jobs, Status

TRUE, FALSE, UNKNOWN = 1, 0, -1
MISSING_FLAG, HORIZON_FLAG = 1, 2

BLOCK_CELLS = 2_000_000
"""Upper bound on job x candidate cells evaluated at once."""


def detect(plan: P.Plan, tenths: np.ndarray, missing: np.ndarray, amounts: dict[str, np.ndarray],
           jobs: Jobs, *, trace: bool = False) -> Detection:
    """Run every job and return the onset day index (-1 if none) and Status per job.

    With trace=True, Detection.trace holds per-job, per-candidate arrays (padded
    to the longest window): "day" (candidate day index, -1 for padding), "valid",
    "requirements" and "vetoes" (codes 1 True, 0 False, -1 Unknown, each shaped
    (terms, jobs, candidates)), "state" (combined code) and "flags" (bit 1
    missing, bit 2 horizon). Curve plans trace "day", "valid", "cumulative".
    """
    series = np.asarray(jobs.series, dtype=np.int64)
    start = np.asarray(jobs.start, dtype=np.int64)
    end = np.asarray(jobs.end, dtype=np.int64)
    data_end = np.asarray(jobs.data_end, dtype=np.int64)
    n_jobs = len(series)
    tables = _Tables(plan, tenths, missing, amounts)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.full(n_jobs, Status.NOT_FOUND, dtype=np.int8)
    n_cand = np.maximum(end - start + 1, 0)
    width = int(n_cand.max()) if n_jobs else 0
    tr = _empty_trace(plan, n_jobs, width) if trace else None
    block = max(1, BLOCK_CELLS // max(width, 1))
    for lo in range(0, n_jobs, block):
        sl = slice(lo, min(lo + block, n_jobs))
        w = int(n_cand[sl].max()) if sl.stop > sl.start else 0
        if w == 0:
            continue
        job = _Block(series[sl], start[sl], end[sl], data_end[sl], w)
        if plan.family == P.CURVE:
            out = _curve_block(plan, tables, job)
        else:
            out = _first_passage_block(plan, tables, job)
        day[sl], status[sl] = out["day"], out["status"]
        if tr is not None:
            _store_trace(tr, sl, w, out["trace"])
    return Detection(day, status, tr)


class _Block:
    """The jobs of one block plus their candidate grid (days d, validity mask)."""

    def __init__(self, series, start, end, data_end, width):
        """Build the (jobs, width) candidate day grid."""
        self.series = series[:, None]
        self.start = start[:, None]
        self.end = end[:, None]
        self.data_end = data_end[:, None]
        self.width = width
        self.d = start[:, None] + np.arange(width, dtype=np.int64)[None, :]
        self.valid = self.d <= end[:, None]


class _Tables:
    """Prefix tables over the series, built once per detect call."""

    def __init__(self, plan, tenths, missing, amounts):
        """Build prefix sums of rain, missing days, wet days and dry days, and the per-veto spell tables."""
        self.amounts = amounts
        self.n_time = tenths.shape[1]
        missing = np.asarray(missing, dtype=bool)
        x = np.where(missing, 0, tenths).astype(np.int64)
        self.x = x
        wet_amount = amounts[plan.wet.key][:, None]
        self.wet = ~missing & _compare(plan.wet.op, x, wet_amount)
        self.dry = ~missing & ~self.wet
        self.px = _prefix(x, np.int64)
        self.pm = _prefix(missing, np.int32)
        self.pw = _prefix(self.wet, np.int32)
        self.pd = _prefix(self.dry, np.int32)
        self.veto_prefix = [self._spell_prefix(v) for v in plan.vetoes]
        self.next_wet = None
        if plan.report_first_wet:
            idx = np.where(self.wet, np.arange(self.n_time, dtype=np.int32)[None, :], self.n_time)
            self.next_wet = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]

    def _spell_prefix(self, veto: P.VetoPlan) -> np.ndarray:
        """Prefix sum over start days u of "a spell of this veto starts at u and lies fully in the data"."""
        n_time, length = self.n_time, veto.length
        holds = np.zeros((self.x.shape[0], n_time + 1), dtype=bool)
        if length <= n_time:
            span = n_time - length + 1
            if veto.kind == P.DRY_RUN:
                holds[:, :span] = (self.pd[:, length:] - self.pd[:, :span]) == length
            else:
                total = self.px[:, length:] - self.px[:, :span]
                amount = self.amounts[veto.cmp.key][:, None]
                complete = (self.pm[:, length:] - self.pm[:, :span]) == 0
                holds[:, :span] = complete & _compare(veto.cmp.op, total, amount)
        return _prefix(holds, np.int32)


def _prefix(values: np.ndarray, dtype) -> np.ndarray:
    """Prefix sums along time with a leading zero column, so table[:, t] sums days [0, t)."""
    out = np.zeros((values.shape[0], values.shape[1] + 1), dtype=dtype)
    np.cumsum(values, axis=1, dtype=dtype, out=out[:, 1:])
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


def _gather(table: np.ndarray, series: np.ndarray, index: np.ndarray) -> np.ndarray:
    """table[series, index] with the index clipped into the table."""
    return table[series, np.clip(index, 0, table.shape[1] - 1)]


def _observe(tables: _Tables, job: _Block, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observability of [a, b): (observable, flags) where flags carry the missing and horizon bits."""
    e = job.data_end
    lo = np.clip(a, 0, e)
    hi = np.maximum(np.minimum(b, e), lo)
    n_missing = tables.pm[job.series, hi] - tables.pm[job.series, lo]
    flags = (((a < 0) | (n_missing > 0)) * MISSING_FLAG + (b > e) * HORIZON_FLAG).astype(np.int8)
    return flags == 0, flags


def _span(job: _Block, first: int, last: int, since_start: bool) -> tuple[np.ndarray, np.ndarray]:
    """Absolute [a, b) of a span for every candidate of the block."""
    a = job.start + 0 * job.d if since_start else job.d + first
    return a, job.d + last


def _condition(plan_cond: P.CondPlan, tables: _Tables, job: _Block) -> tuple[np.ndarray, np.ndarray]:
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
    code = np.where(ok, np.where(result, TRUE, FALSE), UNKNOWN).astype(np.int8)
    return code, np.where(ok, 0, flags).astype(np.int8)


def _veto(veto: P.VetoPlan, spell_prefix: np.ndarray, tables: _Tables, job: _Block
          ) -> tuple[np.ndarray, np.ndarray]:
    """Code and flags of a veto: TRUE means a spell was confirmed, so the candidate is rejected."""
    a, b = _span(job, veto.first, veto.last, False)
    length = veto.length
    if veto.placement == P.INSIDE:
        if veto.last - veto.first < length:
            return np.zeros(a.shape, dtype=np.int8), np.zeros(a.shape, dtype=np.int8)
        u_last = b - length
        ok, flags = _observe(tables, job, a, b)
    else:
        u_last = b - 1
        ok, flags = _observe(tables, job, a, b + length - 1)
    u_last = np.minimum(u_last, job.data_end - length)
    n_table = spell_prefix.shape[1]
    lo = np.clip(a, 0, n_table - 1)
    hi = np.maximum(np.clip(u_last + 1, 0, n_table - 1), lo)
    confirmed = (spell_prefix[job.series, hi] - spell_prefix[job.series, lo]) > 0
    code = np.where(confirmed, TRUE, np.where(ok, FALSE, UNKNOWN)).astype(np.int8)
    return code, np.where(code == UNKNOWN, flags, 0).astype(np.int8)


def _status_from_flags(flags: np.ndarray) -> np.ndarray:
    """CENSORED when the flags are exactly {horizon}, otherwise INSUFFICIENT_DATA."""
    return np.where(flags == HORIZON_FLAG, Status.CENSORED, Status.INSUFFICIENT_DATA).astype(np.int8)


def _first_passage_block(plan: P.Plan, tables: _Tables, job: _Block) -> dict:
    """Evaluate all candidates of a block and pick the first one that is not False."""
    shape = job.d.shape
    req_codes, veto_codes = [], []
    unknown_flags = np.zeros(shape, dtype=np.int8)
    any_false = np.zeros(shape, dtype=bool)
    any_unknown = np.zeros(shape, dtype=bool)
    for cond in plan.conditions:
        code, flags = _condition(cond, tables, job)
        req_codes.append(code)
        any_false |= code == FALSE
        any_unknown |= code == UNKNOWN
        unknown_flags |= flags
    for veto, spell_prefix in zip(plan.vetoes, tables.veto_prefix):
        code, flags = _veto(veto, spell_prefix, tables, job)
        veto_codes.append(code)
        any_false |= code == TRUE
        any_unknown |= code == UNKNOWN
        unknown_flags |= flags
    state = np.where(any_false | ~job.valid, FALSE, np.where(any_unknown, UNKNOWN, TRUE)).astype(np.int8)
    flags = np.where(state == UNKNOWN, unknown_flags, 0).astype(np.int8)

    decided = state != FALSE
    found = decided.any(axis=1)
    pick = np.argmax(decided, axis=1)
    rows = np.arange(len(pick))
    pick_state = state[rows, pick]
    pick_day = job.d[rows, pick]
    pick_flags = flags[rows, pick]

    status = np.full(len(pick), Status.NOT_FOUND, dtype=np.int8)
    day = np.full(len(pick), -1, dtype=np.int64)
    onset = found & (pick_state == TRUE)
    status[onset] = Status.ONSET
    day[onset] = pick_day[onset]
    unknown = found & (pick_state == UNKNOWN)
    status[unknown] = _status_from_flags(pick_flags)[unknown]
    if plan.report_first_wet and onset.any():
        first, last, since = plan.report_span
        series = job.series[:, 0]
        a = job.start[:, 0] if since else pick_day + first
        b = pick_day + last
        lo_raw = np.maximum(a, 0)
        lo = np.minimum(lo_raw, tables.n_time - 1)
        wet_day = tables.next_wet[series, lo]
        in_window = (lo_raw < tables.n_time) & (wet_day < b) & (wet_day < job.data_end[:, 0])
        day = np.where(onset & in_window, wet_day, day)
    trace_out = {"day": np.where(job.valid, job.d, -1), "valid": job.valid, "state": state, "flags": flags,
                 "requirements": [np.where(job.valid, c, 0).astype(np.int8) for c in req_codes],
                 "vetoes": [np.where(job.valid, c, 0).astype(np.int8) for c in veto_codes]}
    return {"day": day, "status": status, "trace": trace_out}


def _curve_block(plan: P.Plan, tables: _Tables, job: _Block) -> dict:
    """Evaluate the cumulative-anomaly rule for every job of a block."""
    start, end, series = job.start[:, 0], job.end[:, 0], job.series[:, 0]
    ok, flags = _observe(tables, job, job.start, job.end + 1)
    ok, flags = ok[:, 0], flags[:, 0]
    baseline = tables.amounts[plan.curve_key][series][:, None]
    steps = job.d - job.start + 1
    rain = _gather(tables.px, job.series, job.d + 1) - _gather(tables.px, job.series, job.start)
    cumulative = 100 * rain - baseline * steps
    masked = np.where(job.valid, cumulative, np.iinfo(np.int64).max)
    minimum = np.argmin(masked, axis=1)
    min_day = start + minimum
    status = np.full(len(series), Status.ONSET, dtype=np.int8)
    status[(min_day == end) | ~job.valid.any(axis=1)] = Status.NOT_FOUND
    day = np.where(status == Status.ONSET, min_day + (1 if plan.curve_after_min else 0), -1)
    unobservable = ~ok & job.valid.any(axis=1)
    status[unobservable] = _status_from_flags(flags)[unobservable]
    day[unobservable] = -1
    return {"day": day.astype(np.int64), "status": status,
            "trace": {"day": np.where(job.valid, job.d, -1), "valid": job.valid,
                      "cumulative": np.where(job.valid, cumulative, 0)}}


def _empty_trace(plan: P.Plan, n_jobs: int, width: int) -> dict:
    """Allocate trace arrays for all jobs."""
    base = {"day": np.full((n_jobs, width), -1, dtype=np.int64), "valid": np.zeros((n_jobs, width), dtype=bool)}
    if plan.family == P.CURVE:
        base["cumulative"] = np.zeros((n_jobs, width), dtype=np.int64)
        return base
    base["requirements"] = np.zeros((len(plan.conditions), n_jobs, width), dtype=np.int8)
    base["vetoes"] = np.zeros((len(plan.vetoes), n_jobs, width), dtype=np.int8)
    base["state"] = np.zeros((n_jobs, width), dtype=np.int8)
    base["flags"] = np.zeros((n_jobs, width), dtype=np.int8)
    return base


def _store_trace(trace: dict, sl: slice, width: int, part: dict) -> None:
    """Copy one block's trace arrays into the full-size trace."""
    for name in ("day", "valid", "state", "flags", "cumulative"):
        if name in trace:
            trace[name][sl, :width] = part[name]
    for name in ("requirements", "vetoes"):
        if name in trace:
            for i, codes in enumerate(part[name]):
                trace[name][i, sl, :width] = codes
