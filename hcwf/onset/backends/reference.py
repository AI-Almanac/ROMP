"""NumPy reference backend: the oracle that implements ``docs/onset_spec.md`` section 4 with full candidate masks.

Prefix tables are built once per call (`Tables`). Jobs are processed in blocks; for each block every candidate of
every job is evaluated at once as a ``(jobs, candidates)`` array of three-valued codes, and the first candidate that
is not False decides the job. `detect` and `detect_stitched` are the entry points; `evaluate` exposes the facts
behind a decision for `hcwf.onset.explain`. The faster backends (`hcwf.onset.backends.numba_cpu`,
`hcwf.onset.backends.gpu`) are checked against this one, and per-window amounts are handled by
`hcwf.onset.backends.windowed`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hcwf.onset import plan as P
from hcwf.onset.backends import windowed
from hcwf.onset.model import Detection, Jobs, Status, StitchedJobs

TRUE, FALSE, UNKNOWN = 1, 0, -1
"""The three-valued codes of a requirement, a veto or a candidate: True, False and Unknown."""
MISSING_FLAG, HORIZON_FLAG = 1, 2
"""The bits of the flags that say why a code is Unknown: missing data (1) and the data horizon (2)."""

BLOCK_CELLS = 2_000_000
"""Upper bound on job x candidate cells evaluated at once."""


def detect(plan: P.Plan, tenths: np.ndarray, missing: np.ndarray, amounts: dict[str, np.ndarray],
           jobs: Jobs, *, trace: bool = False, amount_windows=None) -> Detection:
    """Run every job and return the onset day index (-1 if none) and `Status` per job.

    Amounts may be ``(S, W)`` with `amount_windows` the ``W`` window start leads; see
    `hcwf.onset.backends.windowed` (traces are not available then).

    With `trace` true, ``Detection.trace`` holds per-job, per-candidate arrays (padded to the longest window):
    ``"day"`` (candidate day index, -1 for padding), ``"valid"``, ``"requirements"`` and ``"vetoes"`` (codes 1 True,
    0 False, -1 Unknown, each shaped ``(terms, jobs, candidates)``), ``"state"`` (combined code) and ``"flags"``
    (bit 1 missing, bit 2 horizon). Curve plans trace ``"day"``, ``"valid"``, ``"cumulative"``.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    tenths : numpy.ndarray of int64 or float64, shape (series, time)
        Daily rain in tenths of mm (int), or in mm (float, for exact resolution).
    missing : numpy.ndarray of bool, shape (series, time)
        True where the day is missing.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key, shape ``(series,)`` or ``(series, W)``, in tenths of mm (or counts).
    jobs : hcwf.onset.model.Jobs
        The jobs: ``series``, ``start``, ``end``, ``data_end`` and optional ``origin``, each of shape ``(jobs,)``.
    trace : bool, default False
        Also return the per-candidate trace arrays.
    amount_windows : array_like of int, optional
        The ``W`` window start leads when `amounts` have ``W`` columns.

    Returns
    -------
    hcwf.onset.model.Detection
        ``day`` (int64, shape ``(jobs,)``, day index or ``-1``), ``status`` (int8) and ``trace`` (dict or ``None``).

    Raises
    ------
    NotImplementedError
        `trace` is true while the amounts are per-window.
    """
    amounts, starts = windowed.prepare(amounts, amount_windows)
    if starts is not None:
        if trace:
            raise NotImplementedError("traces are not available with per-window amounts")
        return windowed.run(plan, amounts, starts, jobs, lambda j, a: detect(plan, tenths, missing, a, j))
    series = np.asarray(jobs.series, dtype=np.int64)
    start = np.asarray(jobs.start, dtype=np.int64)
    end = np.asarray(jobs.end, dtype=np.int64)
    data_end = np.asarray(jobs.data_end, dtype=np.int64)
    origin = start if jobs.origin is None else np.asarray(jobs.origin, dtype=np.int64)
    n_jobs = len(series)
    tables = Tables(plan, tenths, missing, amounts)
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
        job = _Block(series[sl], start[sl], end[sl], data_end[sl], w, origin[sl])
        if plan.family == P.CURVE:
            out = _curve_block(plan, tables, job)
        else:
            out = _first_passage_block(plan, tables, job)
        day[sl], status[sl] = out["day"], out["status"]
        if tr is not None:
            _store_trace(tr, sl, w, out["trace"])
    return Detection(day, status, tr)


def _lookback(plan: P.Plan) -> int:
    """Return how many days before a candidate any span of the plan can start.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.

    Returns
    -------
    int
        The look-back in days, 0 when no span reaches back.
    """
    firsts = [0]
    firsts += [c.first for c in plan.conditions if not c.since_start]
    firsts += [v.first - (v.placement == P.BEGINS) for v in plan.vetoes]
    if plan.report_span is not None and not plan.report_span[2]:
        firsts.append(plan.report_span[0])
    return -min(firsts)


def detect_stitched(plan: P.Plan, hist_tenths: np.ndarray, hist_missing: np.ndarray, fc_tenths: np.ndarray,
                    fc_missing: np.ndarray, amounts: dict[str, np.ndarray], jobs: StitchedJobs, *,
                    trace: bool = False, amount_windows=None) -> Detection:
    """Run stitched jobs (observed history, then forecast) and return day indices in stitched days.

    Equivalent to `detect` on the explicit concatenation. Each job gets its own row holding only the stitched columns
    it can read, from its first candidate minus the plan's look-back (clipped at day 0) to its data end; history at or
    after the job's split is never copied. Trace arrays are as in `detect`, with candidate days in stitched days.
    Per-window amounts work as in `detect`.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    hist_tenths : numpy.ndarray of int64, shape (cell, time)
        Observed daily rain in tenths of mm.
    hist_missing : numpy.ndarray of bool, shape (cell, time)
        True where the observed day is missing.
    fc_tenths : numpy.ndarray of int64, shape (series, lead)
        Forecast daily rain in tenths of mm; column ``t`` is lead ``t + 1``.
    fc_missing : numpy.ndarray of bool, shape (series, lead)
        True where the forecast day is missing.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key, shape ``(series,)`` or ``(series, W)``.
    jobs : hcwf.onset.model.StitchedJobs
        The jobs: ``cell``, ``series``, ``split``, ``start``, ``end``, ``data_end`` and optional ``origin``, each of
        shape ``(jobs,)``, in stitched days.
    trace : bool, default False
        Also return the per-candidate trace arrays.
    amount_windows : array_like of int, optional
        The ``W`` window start leads when `amounts` have ``W`` columns.

    Returns
    -------
    hcwf.onset.model.Detection
        ``day`` (int64, shape ``(jobs,)``, stitched day index or ``-1``), ``status`` (int8) and ``trace``.

    Raises
    ------
    NotImplementedError
        `trace` is true while the amounts are per-window.
    ValueError
        A job's split lies beyond the history or its data end beyond the forecast.
    """
    P.reject_exact_stitched(hist_tenths, fc_tenths)
    amounts, starts = windowed.prepare(amounts, amount_windows)
    if starts is not None:
        if trace:
            raise NotImplementedError("traces are not available with per-window amounts")
        return windowed.run(plan, amounts, starts, jobs, lambda j, a: detect_stitched(
            plan, hist_tenths, hist_missing, fc_tenths, fc_missing, a, j))
    cell = np.asarray(jobs.cell, dtype=np.int64)
    series = np.asarray(jobs.series, dtype=np.int64)
    split = np.asarray(jobs.split, dtype=np.int64)
    start = np.asarray(jobs.start, dtype=np.int64)
    end = np.asarray(jobs.end, dtype=np.int64)
    data_end = np.asarray(jobs.data_end, dtype=np.int64)
    origin = start if jobs.origin is None else np.asarray(jobs.origin, dtype=np.int64)
    n_jobs = len(series)
    n_hist, n_fc = hist_tenths.shape[1], fc_tenths.shape[1]
    if n_jobs and (split.max() > n_hist or (data_end - split).max() > n_fc):
        raise ValueError("a stitched job's split lies beyond the history or its data end beyond the forecast")
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.full(n_jobs, Status.NOT_FOUND, dtype=np.int8)
    n_cand = np.maximum(end - start + 1, 0)
    width = int(n_cand.max()) if n_jobs else 0
    tr = _empty_trace(plan, n_jobs, width) if trace else None
    active = np.nonzero(n_cand > 0)[0]
    if len(active) == 0:
        return Detection(day, status, tr)
    low = np.maximum(np.minimum(start - _lookback(plan), origin), 0)
    row_width = int(np.maximum(data_end[active] - low[active], 1).max())
    per_block = max(1, 4 * BLOCK_CELLS // row_width)
    for lo in range(0, len(active), per_block):
        idx = active[lo:lo + per_block]
        cols = low[idx, None] + np.arange(row_width, dtype=np.int64)[None, :]
        in_hist = cols < split[idx, None]
        in_data = cols < data_end[idx, None]
        h = np.clip(cols, 0, n_hist - 1)
        f = np.clip(cols - split[idx, None], 0, n_fc - 1)
        x = np.where(in_hist, hist_tenths[cell[idx, None], h], fc_tenths[series[idx, None], f])
        miss = np.where(in_hist, hist_missing[cell[idx, None], h], fc_missing[series[idx, None], f])
        miss = miss | ~in_data
        x = np.where(miss, 0, x).astype(np.int32)
        rows = np.arange(len(idx), dtype=np.int64)
        sub_amounts = {key: value[series[idx]] for key, value in amounts.items()}
        sub = Jobs(rows, start[idx] - low[idx], end[idx] - low[idx], data_end[idx] - low[idx], origin[idx] - low[idx])
        out = detect(plan, x, miss, sub_amounts, sub, trace=trace)
        found = out.day >= 0
        day[idx] = np.where(found, out.day + low[idx], -1)
        status[idx] = out.status
        if tr is not None:
            w = out.trace["day"].shape[1]
            for name in ("day", "valid", "state", "flags", "cumulative"):
                if name in tr:
                    part = out.trace[name]
                    if name == "day":
                        part = np.where(part >= 0, part + low[idx, None], -1)
                    tr[name][idx, :w] = part
            for name in ("requirements", "vetoes"):
                if name in tr:
                    for i, codes in enumerate(out.trace[name]):
                        tr[name][i, idx, :w] = codes
    return Detection(day, status, tr)


class _Block:
    """The jobs of one block plus their candidate grid (days ``d``, validity mask).

    Every per-job array is stored as a column of shape ``(jobs, 1)`` so that it broadcasts against the candidates.

    Parameters
    ----------
    series : numpy.ndarray of int64, shape (jobs,)
        The row of the tables each job reads.
    start : numpy.ndarray of int64, shape (jobs,)
        The first candidate day of each job.
    end : numpy.ndarray of int64, shape (jobs,)
        The last candidate day of each job, inclusive.
    data_end : numpy.ndarray of int64, shape (jobs,)
        The end of usable data of each job, exclusive.
    width : int
        The number of candidate columns (the longest job in the block).
    origin : numpy.ndarray of int64, shape (jobs,)
        The day "season to date" spans start from.

    Attributes
    ----------
    d : numpy.ndarray of int64, shape (jobs, width)
        The candidate day of each cell.
    valid : numpy.ndarray of bool, shape (jobs, width)
        True where the candidate day is not past the job's ``end``.
    """

    def __init__(self, series, start, end, data_end, width, origin):
        self.series = series[:, None]
        self.start = start[:, None]
        self.origin = origin[:, None]
        self.end = end[:, None]
        self.data_end = data_end[:, None]
        self.width = width
        self.d = start[:, None] + np.arange(width, dtype=np.int64)[None, :]
        self.valid = self.d <= end[:, None]


class Tables:
    """Prefix tables over the series, built once per `detect` call.

    Builds prefix sums of rain, missing days, wet days and dry days, and the per-veto spell tables. Rain is int64
    tenths, or float64 mm (summed in time order from day 0) for exact resolution; ``per_mm`` is the number of rain
    units in a mm (10, or 1 when exact).

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition (wet and dry comparisons, vetoes, horizon and missing policies).
    tenths : numpy.ndarray of int64 or float64, shape (series, time)
        Daily rain in tenths of mm (int), or in mm (float, exact resolution).
    missing : numpy.ndarray of bool, shape (series, time)
        True where the day is missing; those days count as zero rain.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key, shape ``(series,)``.

    Attributes
    ----------
    amounts : dict of str to numpy.ndarray
        The amounts as given.
    n_time : int
        The number of days.
    horizon : str
        The plan's horizon policy.
    keep : bool
        The plan's ``keep_missing`` flag.
    missing : numpy.ndarray of bool, shape (series, time)
        The missing mask.
    x : numpy.ndarray of int64 or float64, shape (series, time)
        Rain with missing days set to 0.
    per_mm : int
        Rain units per mm: 10 for tenths, 1 for exact mm.
    wet, dry : numpy.ndarray of bool, shape (series, time)
        Wet and dry days (never on a missing day).
    px, pm, pw, pd : numpy.ndarray, shape (series, time + 1)
        Prefix sums of rain, missing days, wet days and dry days; ``p[:, t]`` sums days ``[0, t)``.
    veto_prefix : list of numpy.ndarray
        One int32 prefix table of spell start days per veto of the plan, shape ``(series, time + 2)``.
    next_wet : numpy.ndarray of int32, shape (series, time) or None
        The first wet day at or after each day (``time`` if none); ``None`` unless the plan reports the first wet day.
    """

    def __init__(self, plan, tenths, missing, amounts):
        self.amounts = amounts
        self.n_time = tenths.shape[1]
        self.horizon, self.keep = plan.horizon, plan.keep_missing
        missing = self.missing = np.asarray(missing, dtype=bool)
        x = np.where(missing, 0, tenths).astype(np.float64 if np.asarray(tenths).dtype.kind == "f" else np.int64)
        self.x = x
        self.per_mm = 1 if x.dtype.kind == "f" else 10
        wet_amount = amounts[plan.wet.key][:, None]
        self.wet = ~missing & windowed.compare(plan.wet.op, x, wet_amount)
        if plan.dry is None:
            self.dry = ~missing & ~self.wet
        else:
            self.dry = ~missing & windowed.compare(plan.dry.op, x, amounts[plan.dry.key][:, None])
        self.px = _prefix(x, x.dtype)
        self.pm = _prefix(missing, np.int32)
        self.pw = _prefix(self.wet, np.int32)
        self.pd = _prefix(self.dry, np.int32)
        self.veto_prefix = [self._spell_prefix(v) for v in plan.vetoes]
        self.next_wet = None
        if plan.report_first_wet:
            idx = np.where(self.wet, np.arange(self.n_time, dtype=np.int32)[None, :], self.n_time)
            self.next_wet = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]

    def _spell_prefix(self, veto: P.VetoPlan) -> np.ndarray:
        """Return the prefix sum over start days of "a spell of this veto starts here and lies fully in the data".

        Parameters
        ----------
        veto : hcwf.onset.plan.VetoPlan
            The veto: its spell ``length``, ``kind`` (dry run or rain total), ``placement`` and comparison.

        Returns
        -------
        numpy.ndarray of int32, shape (series, time + 2)
            The leading-zero prefix sum of the boolean table of spell start days (length ``time + 1``).
        """
        n_time, length = self.n_time, veto.length
        holds = np.zeros((self.x.shape[0], n_time + 1), dtype=bool)
        if length <= n_time:
            span = n_time - length + 1
            if veto.kind == P.DRY_RUN:
                holds[:, :span] = (self.pd[:, length:] - self.pd[:, :span]) == length
                if veto.placement == P.BEGINS:
                    holds[:, 1:span] &= ~self.dry[:, :span - 1]
            else:
                total = self.px[:, length:] - self.px[:, :span]
                amount = self.amounts[veto.cmp.key][:, None]
                complete = (self.pm[:, length:] - self.pm[:, :span]) == 0
                holds[:, :span] = complete & windowed.compare(veto.cmp.op, total, amount)
        return _prefix(holds, np.int32)


def _prefix(values: np.ndarray, dtype) -> np.ndarray:
    """Return prefix sums along time with a leading zero column, so ``table[:, t]`` sums days ``[0, t)``.

    Parameters
    ----------
    values : numpy.ndarray, shape (series, time)
        The values to sum (bool, int or float).
    dtype : numpy.dtype or type
        The dtype of the sums.

    Returns
    -------
    numpy.ndarray of `dtype`, shape (series, time + 1)
        The prefix sums.
    """
    out = np.zeros((values.shape[0], values.shape[1] + 1), dtype=dtype)
    np.cumsum(values, axis=1, dtype=dtype, out=out[:, 1:])
    return out


def _gather(table: np.ndarray, series: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return ``table[series, index]`` with the index clipped into the table.

    Parameters
    ----------
    table : numpy.ndarray, shape (series, n)
        A prefix table.
    series : numpy.ndarray of int64, shape (jobs, 1)
        The row of each job.
    index : numpy.ndarray of int64, shape (jobs, candidates)
        The column to read; clipped to ``[0, n - 1]``.

    Returns
    -------
    numpy.ndarray, shape (jobs, candidates)
        The gathered values.
    """
    return table[series, np.clip(index, 0, table.shape[1] - 1)]


def _observe(tables: Tables, job: _Block, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return whether the span ``[a, b)`` can be observed, with the reason when it cannot.

    A span is not observable when it starts before day 0, contains a missing day (unless missing days are kept), or
    reaches past the job's data end.

    Parameters
    ----------
    tables : Tables
        The prefix tables.
    job : _Block
        The block of jobs.
    a : numpy.ndarray of int64, shape (jobs, candidates)
        The first day of the span.
    b : numpy.ndarray of int64, shape (jobs, candidates)
        The end of the span, exclusive.

    Returns
    -------
    observable : numpy.ndarray of bool, shape (jobs, candidates)
        True where the whole span is observed.
    flags : numpy.ndarray of int8, shape (jobs, candidates)
        Bit 1 (`MISSING_FLAG`) for a missing or pre-history day, bit 2 (`HORIZON_FLAG`) for a span past the data end.
    """
    e = job.data_end
    lo = np.clip(a, 0, e)
    hi = np.maximum(np.minimum(b, e), lo)
    n_missing = tables.pm[job.series, hi] - tables.pm[job.series, lo]
    flags = (((a < 0) | ((n_missing > 0) & (not tables.keep))) * MISSING_FLAG + (b > e) * HORIZON_FLAG).astype(np.int8)
    return flags == 0, flags


@dataclass
class CondFacts:
    """A requirement over every candidate of a block, each field of shape ``(jobs, candidates)``."""

    code: np.ndarray
    """The three-valued code (int8: 1 True, 0 False, -1 Unknown)."""
    flags: np.ndarray
    """The flags (int8; bit 1 missing, bit 2 horizon), non-zero only where the code is Unknown."""
    first: np.ndarray
    """The first day of the span ``[first, last)``."""
    last: np.ndarray
    """The end of the span, exclusive."""
    value: np.ndarray
    """The value compared: the span's rain sum in tenths (mm when exact), or its wet-day count."""


@dataclass
class VetoFacts:
    """A veto over every candidate of a block, each field of shape ``(jobs, candidates)``."""

    code: np.ndarray
    """The three-valued code (int8): 1 means a spell was confirmed and the candidate is rejected."""
    flags: np.ndarray
    """The flags (int8; bit 1 missing, bit 2 horizon), non-zero only where the code is Unknown."""
    first: np.ndarray
    """The first day of the span ``[first, last)``."""
    last: np.ndarray
    """The end of the span, exclusive."""
    need_last: np.ndarray
    """The end (exclusive) of the days the veto needs observable."""
    lo: np.ndarray
    """The first start day of the spells it looked at (``[lo, hi)`` indexes the spell table)."""
    hi: np.ndarray
    """The end of the start days of the spells it looked at, exclusive."""


def _condition(plan_cond: P.CondPlan, tables: Tables, job: _Block) -> CondFacts:
    """Evaluate a Total or WetDays requirement for every candidate of a block.

    Parameters
    ----------
    plan_cond : hcwf.onset.plan.CondPlan
        The requirement.
    tables : Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    CondFacts
        The three-valued code and flags, with the span and value behind them.
    """
    a, b = windowed.span(job, plan_cond.first, plan_cond.last, plan_cond.since_start)
    ok, flags = _observe(tables, job, a, b)
    if plan_cond.kind == P.TOTAL:
        value = _gather(tables.px, job.series, b) - _gather(tables.px, job.series, a)
        amount = tables.amounts[plan_cond.cmp.key][job.series]
        result = windowed.compare(plan_cond.cmp.op, value, amount)
    else:
        value = _gather(tables.pw, job.series, b) - _gather(tables.pw, job.series, a)
        result = windowed.compare(plan_cond.op, value, plan_cond.count)
    code = np.where(ok, np.where(result, TRUE, FALSE), UNKNOWN).astype(np.int8)
    return CondFacts(code, np.where(ok, 0, flags).astype(np.int8), a, b, value)


def _veto(veto: P.VetoPlan, spell_prefix: np.ndarray, tables: Tables, job: _Block) -> VetoFacts:
    """Evaluate a veto for every candidate of a block.

    A code of ``TRUE`` means a spell was confirmed, so the candidate is rejected. The plan's horizon policy
    (``censor``, ``reject`` or other) decides how a span that reaches past the data end is coded.

    Parameters
    ----------
    veto : hcwf.onset.plan.VetoPlan
        The veto.
    spell_prefix : numpy.ndarray of int32, shape (series, time + 2)
        The prefix table of spell start days for this veto (`Tables.veto_prefix`).
    tables : Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    VetoFacts
        The code, flags, spans and spell start-day range.
    """
    a, b = windowed.span(job, veto.first, veto.last, False)
    length = veto.length
    if veto.placement == P.INSIDE:
        if veto.last - veto.first < length:
            zeros = np.zeros(a.shape, dtype=np.int8)
            return VetoFacts(zeros, zeros.copy(), a, b, b, a, a)
        u_last = b - length
        need_last = b
    else:
        u_last = b - 1
        need_last = b + length - 1
    ok, flags = _observe(tables, job, a, need_last)
    u_last = np.minimum(u_last, job.data_end - length)
    n_table = spell_prefix.shape[1]
    lo = np.clip(a, 0, n_table - 1)
    hi = np.maximum(np.clip(u_last + 1, 0, n_table - 1), lo)
    confirmed = (spell_prefix[job.series, hi] - spell_prefix[job.series, lo]) > 0
    code = np.where(confirmed, TRUE, np.where(ok, FALSE, UNKNOWN)).astype(np.int8)
    flags = np.where(code == UNKNOWN, flags, 0).astype(np.int8)
    if tables.horizon != "censor":
        code = np.where(flags == HORIZON_FLAG, FALSE, code).astype(np.int8)
        flags = np.where(code == UNKNOWN, flags, 0).astype(np.int8)
        if tables.horizon == "reject":
            code = np.where(b > job.data_end, TRUE, code).astype(np.int8)
            flags = np.where(code == UNKNOWN, flags, 0).astype(np.int8)
    return VetoFacts(code, flags, a, b, need_last, lo, hi)


def _status_from_flags(flags: np.ndarray) -> np.ndarray:
    """Map flags to a status: ``CENSORED`` when the flags are exactly the horizon bit, else ``INSUFFICIENT_DATA``.

    Parameters
    ----------
    flags : numpy.ndarray of int8
        The flags of the chosen candidates.

    Returns
    -------
    numpy.ndarray of int8
        The `Status` code, same shape as `flags`.
    """
    return np.where(flags == HORIZON_FLAG, Status.CENSORED, Status.INSUFFICIENT_DATA).astype(np.int8)


@dataclass
class Evaluation:
    """Every fact of a first-passage evaluation of a block of jobs, candidate by candidate.

    `conditions` and `vetoes` hold one `CondFacts` / `VetoFacts` per requirement / veto of the plan. The array fields
    are shaped ``(jobs, candidates)`` unless noted. `onset` and `status` give each job's decision as `detect`
    returns it. The helper methods read the same tables the rule was decided on.
    """

    tables: Tables
    """The prefix tables the evaluation read."""
    plan: P.Plan
    """The compiled definition."""
    block: _Block
    """The block of jobs and its candidate grid."""
    day: np.ndarray
    """The candidate day index, ``-1`` for padding."""
    valid: np.ndarray
    """True where the candidate exists (not padding)."""
    state: np.ndarray
    """The combined three-valued code (int8)."""
    flags: np.ndarray
    """The flags (int8; bit 1 missing, bit 2 horizon), only where the state is Unknown."""
    day_missing: np.ndarray
    """True where the candidate's own day is missing under the ``keep`` policy, which rejects it."""
    conditions: list
    """One `CondFacts` per requirement of the plan."""
    vetoes: list
    """One `VetoFacts` per veto of the plan."""
    onset: np.ndarray
    """The reported day index per job, shape ``(jobs,)`` (int64, ``-1`` for none)."""
    status: np.ndarray
    """The `Status` code per job, shape ``(jobs,)`` (int8)."""

    def missing_days(self, job: int, first: int, last: int) -> tuple:
        """Return the day indices of the missing days inside the observed part of ``[first, last)`` for one job.

        Parameters
        ----------
        job : int
            The row of the block.
        first : int
            The first day of the span.
        last : int
            The end of the span, exclusive.

        Returns
        -------
        tuple of int
            The missing day indices, ascending.
        """
        end = int(self.block.data_end[job, 0])
        lo = min(max(first, 0), end)
        hi = max(min(last, end), lo)
        series = int(self.block.series[job, 0])
        return tuple(int(i) for i in np.nonzero(self.tables.missing[series, lo:hi])[0] + lo)

    def first_not_wet(self, job: int, first: int, last: int) -> int | None:
        """Return the first day of ``[first, last)`` that is not wet for one job.

        Parameters
        ----------
        job : int
            The row of the block.
        first : int
            The first day of the span.
        last : int
            The end of the span, exclusive.

        Returns
        -------
        int or None
            The day index, or ``None`` when every day is wet.
        """
        gaps = np.nonzero(~self.tables.wet[int(self.block.series[job, 0]), first:last])[0]
        return int(gaps[0]) + first if len(gaps) else None

    def spell(self, veto: int, job: int, cand: int) -> tuple[int, int, float] | None:
        """Return the first spell a confirmed veto found, or ``None``.

        A dry run is extended over the dry days that follow it, up to the end of the veto's span (starts_inside and
        beginning-within spells: up to the data end).

        Parameters
        ----------
        veto : int
            The index of the veto in the plan.
        job : int
            The row of the block.
        cand : int
            The candidate column.

        Returns
        -------
        tuple of (int, int, float) or None
            The first day, the last day and the rain total in the data's unit (tenths or exact mm), or ``None``
            when the veto did not confirm a spell for this candidate.
        """
        spell_plan, tables, block = self.plan.vetoes[veto], self.tables, self.block
        facts = self.vetoes[veto]
        series, end = int(block.series[job, 0]), int(block.data_end[job, 0])
        row = tables.veto_prefix[veto][series]
        lo, hi = int(facts.lo[job, cand]), int(facts.hi[job, cand])
        if hi <= lo or row[hi] == row[lo]:
            return None
        first = int(np.searchsorted(row, row[lo], side="right")) - 1
        last = first + spell_plan.length - 1
        if spell_plan.kind == P.DRY_RUN:
            limit = min(int(facts.last[job, cand]) if spell_plan.placement == P.INSIDE else end, end) - 1
            while last + 1 <= limit and tables.dry[series, last + 1]:
                last += 1
        return first, last, tables.px[series, last + 1] - tables.px[series, first]


def evaluate(plan: P.Plan, tables: Tables, jobs: Jobs) -> Evaluation:
    """Evaluate every candidate of first-passage jobs against prebuilt tables.

    Returns the facts behind the decision of `detect`. All jobs form one block (use a few jobs, or call `detect` for
    many); `detect` decides through this same function.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled first-passage definition.
    tables : Tables
        The prefix tables of the series.
    jobs : hcwf.onset.model.Jobs
        The jobs: ``series``, ``start``, ``end``, ``data_end`` and optional ``origin``, each of shape ``(jobs,)``.

    Returns
    -------
    Evaluation
        The facts of every candidate and the decision per job.
    """
    series, start, end, data_end = (np.asarray(a, dtype=np.int64)
                                    for a in (jobs.series, jobs.start, jobs.end, jobs.data_end))
    origin = start if jobs.origin is None else np.asarray(jobs.origin, dtype=np.int64)
    return _evaluate_block(plan, tables, _Block(series, start, end, data_end, int(np.maximum(end - start + 1, 1).max()),
                                                origin))


def _evaluate_block(plan: P.Plan, tables: Tables, job: _Block) -> Evaluation:
    """Evaluate all candidates of a block and pick the first one that is not False.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled first-passage definition.
    tables : Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    Evaluation
        The facts and the decision per job (a job with no candidate that is not False is ``NOT_FOUND``).
    """
    shape = job.d.shape
    conditions, vetoes = [], []
    unknown_flags = np.zeros(shape, dtype=np.int8)
    any_false = np.zeros(shape, dtype=bool)
    any_unknown = np.zeros(shape, dtype=bool)
    for cond in plan.conditions:
        facts = _condition(cond, tables, job)
        conditions.append(facts)
        any_false |= facts.code == FALSE
        any_unknown |= facts.code == UNKNOWN
        unknown_flags |= facts.flags
    for veto, spell_prefix in zip(plan.vetoes, tables.veto_prefix):
        facts = _veto(veto, spell_prefix, tables, job)
        vetoes.append(facts)
        any_false |= facts.code == TRUE
        any_unknown |= facts.code == UNKNOWN
        unknown_flags |= facts.flags
    day_missing = np.zeros(shape, dtype=bool)
    if tables.keep:
        day_missing = tables.missing[job.series, np.clip(job.d, 0, tables.n_time - 1)] & (job.d < job.data_end)
        any_false |= day_missing
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
        a = job.origin[:, 0] if since else pick_day + first
        b = pick_day + last
        lo_raw = np.maximum(a, 0)
        lo = np.minimum(lo_raw, tables.n_time - 1)
        wet_day = tables.next_wet[series, lo]
        in_window = (lo_raw < tables.n_time) & (wet_day < b) & (wet_day < job.data_end[:, 0])
        day = np.where(onset & in_window, wet_day, day)
    return Evaluation(tables, plan, job, np.where(job.valid, job.d, -1), job.valid, state, flags, day_missing,
                      conditions, vetoes, day, status)


def _first_passage_block(plan: P.Plan, tables: Tables, job: _Block) -> dict:
    """Evaluate all candidates of a block and return the decision and the trace arrays.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled first-passage definition.
    tables : Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    dict
        ``"day"`` (int64, ``(jobs,)``), ``"status"`` (int8, ``(jobs,)``) and ``"trace"`` (the trace arrays of the
        block, see `detect`).
    """
    ev = _evaluate_block(plan, tables, job)
    trace_out = {"day": ev.day, "valid": ev.valid, "state": ev.state, "flags": ev.flags,
                 "requirements": [np.where(job.valid, c.code, 0).astype(np.int8) for c in ev.conditions],
                 "vetoes": [np.where(job.valid, c.code, 0).astype(np.int8) for c in ev.vetoes]}
    return {"day": ev.onset, "status": ev.status, "trace": trace_out}


def _curve_block(plan: P.Plan, tables: Tables, job: _Block) -> dict:
    """Evaluate the cumulative-anomaly rule for every job of a block.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled curve definition.
    tables : Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    dict
        ``"day"`` (int64, ``(jobs,)``), ``"status"`` (int8, ``(jobs,)``) and ``"trace"`` with ``"day"``,
        ``"valid"`` and ``"cumulative"``, each of shape ``(jobs, candidates)``.
    """
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
    """Allocate trace arrays for all jobs.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition; a curve plan traces fewer arrays.
    n_jobs : int
        The number of jobs.
    width : int
        The number of candidate columns (the longest job).

    Returns
    -------
    dict
        The trace arrays, see `detect`: ``-1`` days, False validity and zero codes.
    """
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
    """Copy one block's trace arrays into the full-size trace.

    Parameters
    ----------
    trace : dict
        The full-size trace from `_empty_trace`, updated in place.
    sl : slice
        The jobs of the block.
    width : int
        The number of candidate columns of the block.
    part : dict
        The block's trace arrays.
    """
    for name in ("day", "valid", "state", "flags", "cumulative"):
        if name in trace:
            trace[name][sl, :width] = part[name]
    for name in ("requirements", "vetoes"):
        if name in trace:
            for i, codes in enumerate(part[name]):
                trace[name][i, sl, :width] = codes
