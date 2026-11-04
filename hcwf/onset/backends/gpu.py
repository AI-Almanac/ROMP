"""CuPy backend: the reference algorithm (prefix tables, full candidate masks, first non-False via argmax) on the GPU.

Integer rain (tenths) is exact integer arithmetic; float rain (``rain_resolution: exact``) is summed in time order by
a one-thread-per-row kernel, because a parallel scan would round differently. Either way results are bit-identical to
the reference backend. Prefix tables for all series live on the device; jobs are processed in blocks of
``(jobs, candidates)`` cells sized from free device memory.

`detect_stitched` does the same over observed history followed by a forecast, joining separate history and forecast
tables at each job's split instead of copying history.

The module mirrors `hcwf.onset.backends.reference` (the oracle) function by function, with ``cupy`` arrays in place
of NumPy ones; per-window amounts are handled by `hcwf.onset.backends.windowed`. `available` says whether CuPy and
a CUDA device are usable.
"""
from __future__ import annotations

import numpy as np

from hcwf.onset import plan as P
from hcwf.onset.backends import windowed
from hcwf.onset.model import Detection, Jobs, Status, StitchedJobs

try:
    import cupy as cp
except ImportError:
    cp = None

TRUE, FALSE, UNKNOWN = 1, 0, -1
"""The three-valued codes of a requirement, a veto or a candidate: True, False and Unknown."""
MISSING_FLAG, HORIZON_FLAG = 1, 2
"""The bits of the flags that say why a code is Unknown: missing data (1) and the data horizon (2)."""

BLOCK_CELLS: int | None = None
"""Job x candidate cells per block; None sizes blocks from free device memory."""

BYTES_PER_CELL = 400
"""Conservative device bytes of temporaries per evaluated job x candidate cell."""

MEMORY_FRACTION = 0.4
"""Share of the free device memory (after the tables are built) one block may use."""

MAX_AUTO_CELLS = 100_000_000
"""Cap on auto-sized blocks, past which extra size no longer buys speed."""


def available() -> bool:
    """Return whether CuPy imports and a CUDA device can run a kernel; never raises.

    Returns
    -------
    bool
        True when a trivial computation on the device succeeds.
    """
    if cp is None:
        return False
    try:
        return int(cp.arange(3).sum()) == 3
    except Exception:
        return False


def _block_cells() -> int:
    """Return the number of job x candidate cells per block.

    Uses the `BLOCK_CELLS` override, else a fraction (`MEMORY_FRACTION`) of the free device memory at
    `BYTES_PER_CELL` per cell, capped at `MAX_AUTO_CELLS`.

    Returns
    -------
    int
        The cells per block, at least 1.
    """
    if BLOCK_CELLS is not None:
        return max(1, int(BLOCK_CELLS))
    cp.get_default_memory_pool().free_all_blocks()
    free, _ = cp.cuda.Device().mem_info
    return int(min(MAX_AUTO_CELLS, max(1, free * MEMORY_FRACTION / BYTES_PER_CELL)))


def detect(plan: P.Plan, tenths: np.ndarray, missing: np.ndarray, amounts: dict[str, np.ndarray],
           jobs: Jobs, *, trace: bool = False, amount_windows=None) -> Detection:
    """Run every job on the GPU and return the onset day index (-1 if none) and `~hcwf.onset.model.Status` per job (NumPy arrays).

    Amounts may be ``(S, W)`` with ``amount_windows`` the ``W`` window start leads (`hcwf.onset.backends.windowed`).

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    tenths : numpy.ndarray of int64 or float64, shape (series, time)
        Daily rain in tenths of mm (int), or in mm (float, exact resolution).
    missing : numpy.ndarray of bool, shape (series, time)
        True where the day is missing.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key, shape ``(series,)`` or ``(series, W)``.
    jobs : hcwf.onset.model.Jobs
        The jobs: ``series``, ``start``, ``end``, ``data_end`` and optional ``origin``, each of shape ``(jobs,)``.
    trace : bool, default False
        Must be false: traces come from the reference backend.
    amount_windows : array_like of int, optional
        The ``W`` window start leads when ``amounts`` have ``W`` columns.

    Returns
    -------
    hcwf.onset.model.Detection
        ``day`` (int64, shape ``(jobs,)``, ``-1`` where none), ``status`` (int8) and ``trace`` (``None``), as NumPy
        arrays on the host.

    Raises
    ------
    NotImplementedError
        `trace` is true, or a curve rule is run with exact rain resolution.
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
    if plan.family == P.CURVE and np.asarray(tenths).dtype.kind == "f":
        raise NotImplementedError("curve rules need rain_resolution 'tenths'")
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
    """The jobs of one block plus their candidate grid (days ``d``, validity mask), on the device.

    Every per-job array is stored as a column of shape ``(jobs, 1)`` so that it broadcasts against the candidates.

    Parameters
    ----------
    series : cupy.ndarray of int64, shape (jobs,)
        The row of the tables each job reads.
    start : cupy.ndarray of int64, shape (jobs,)
        The first candidate day of each job.
    end : cupy.ndarray of int64, shape (jobs,)
        The last candidate day of each job, inclusive.
    data_end : cupy.ndarray of int64, shape (jobs,)
        The end of usable data of each job, exclusive.
    width : int
        The number of candidate columns (the longest job in the block).
    origin : cupy.ndarray of int64, shape (jobs,)
        The day "season to date" spans start from.

    Attributes
    ----------
    d : cupy.ndarray of int64, shape (jobs, width)
        The candidate day of each cell.
    valid : cupy.ndarray of bool, shape (jobs, width)
        True where the candidate day is not past the job's ``end``.
    """

    def __init__(self, series, start, end, data_end, width, origin):
        self.series = series[:, None]
        self.start = start[:, None]
        self.origin = origin[:, None]
        self.end = end[:, None]
        self.data_end = data_end[:, None]
        self.width = width
        self.d = start[:, None] + cp.arange(width, dtype=cp.int64)[None, :]
        self.valid = self.d <= end[:, None]


class _Tables:
    """Prefix tables over the series, built once per `detect` call on the device.

    Uploads the data and builds prefix sums of rain, missing, wet and dry days, plus per-veto spell tables. Rain is
    int64 tenths, or float64 mm for exact resolution. ``pd`` is kept (for the stitched seam) only when the definition
    has its own ``dry_day``.

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
    horizon : str
        The plan's horizon policy.
    keep : bool
        The plan's ``keep_missing`` flag.
    amounts : dict of str to cupy.ndarray
        The amounts on the device (float64 for exact rain, else int64).
    n_time : int
        The number of days.
    px, pm, pw : cupy.ndarray, shape (series, time + 1)
        Prefix sums of rain, missing days and wet days; ``p[:, t]`` sums days ``[0, t)``.
    pd : cupy.ndarray of int32, shape (series, time + 1) or None
        Prefix counts of dry days; ``None`` unless the definition has its own ``dry_day``.
    veto_prefix : list of cupy.ndarray
        One int32 prefix table of spell start days per veto of the plan.
    wet_order : cupy.ndarray of int32, shape (series, time) or None
        The wet days of each series in time order first (stable argsort); ``None`` unless the plan reports the first
        wet day.
    """

    def __init__(self, plan, tenths, missing, amounts):
        exact = np.asarray(tenths).dtype.kind == "f"
        self.horizon, self.keep = plan.horizon, plan.keep_missing
        self.amounts = {k: cp.asarray(v, dtype=cp.float64 if exact else cp.int64) for k, v in amounts.items()}
        self.n_time = tenths.shape[1]
        missing = cp.asarray(missing, dtype=cp.bool_)
        x = cp.where(missing, 0, cp.asarray(tenths)).astype(cp.float64 if exact else cp.int64)
        wet_amount = self.amounts[plan.wet.key][:, None]
        wet = ~missing & windowed.compare(plan.wet.op, x, wet_amount)
        if plan.dry is None:
            dry = ~missing & ~wet
        else:
            dry = ~missing & windowed.compare(plan.dry.op, x, self.amounts[plan.dry.key][:, None])
        self.x_shape = x.shape
        self.px = _prefix(x, x.dtype)
        del x
        self.pm = _prefix(missing, cp.int32)
        self.pw = _prefix(wet, cp.int32)
        self.pd = _prefix(dry, cp.int32)
        self._dry = dry
        self.veto_prefix = [self._spell_prefix(v) for v in plan.vetoes]
        self.wet_order = None
        if plan.report_first_wet:
            # Wet days first, in time order, so the k-th wet day of a series is wet_order[s, k].
            self.wet_order = cp.argsort(~wet, axis=1, kind="stable").astype(cp.int32)
        del wet, dry, missing
        self._dry = None
        if plan.dry is None:
            self.pd = None

    def _spell_prefix(self, veto: P.VetoPlan):
        """Return the prefix sum over start days of "a spell of this veto starts here and lies fully in the data".

        Parameters
        ----------
        veto : hcwf.onset.plan.VetoPlan
            The veto: its spell ``length``, ``kind`` (dry run or rain total), ``placement`` and comparison.

        Returns
        -------
        cupy.ndarray of int32, shape (series, time + 2)
            The leading-zero prefix sum of the boolean table of spell start days (length ``time + 1``).
        """
        n_time, length = self.n_time, veto.length
        holds = cp.zeros((self.x_shape[0], n_time + 1), dtype=cp.bool_)
        if length <= n_time:
            span = n_time - length + 1
            if veto.kind == P.DRY_RUN:
                holds[:, :span] = (self.pd[:, length:] - self.pd[:, :span]) == length
                if veto.placement == P.BEGINS:
                    holds[:, 1:span] &= ~self._dry[:, :span - 1]
            else:
                total = self.px[:, length:] - self.px[:, :span]
                amount = self.amounts[veto.cmp.key][:, None]
                complete = (self.pm[:, length:] - self.pm[:, :span]) == 0
                holds[:, :span] = complete & windowed.compare(veto.cmp.op, total, amount)
        return _prefix(holds, cp.int32)

    def next_wet(self, series, lo):
        """Return the first wet day at or after ``lo`` in each series, ``n_time`` if there is none.

        Parameters
        ----------
        series : cupy.ndarray of int64, shape (n,)
            The table row of each query.
        lo : cupy.ndarray of int64, shape (n,)
            The first day to look at, within ``[0, n_time - 1]``.

        Returns
        -------
        cupy.ndarray of int64, shape (n,)
            The day index of the next wet day, or ``n_time``.
        """
        rank = self.pw[series, lo]
        total = self.pw[series, self.n_time]
        found = self.wet_order[series, cp.minimum(rank, self.n_time - 1)].astype(cp.int64)
        return cp.where(rank < total, found, self.n_time)


_ROW_CUMSUM = None
"""The compiled `_row_cumsum` CUDA kernel, created on first use."""


def _row_cumsum(values):
    """Return float64 prefix sums with a leading zero column, each row added in time order (as ``np.cumsum`` does).

    One CUDA thread sums one row without fused multiply-add, so the rounding equals NumPy's.

    Parameters
    ----------
    values : cupy.ndarray, shape (rows, cols)
        The values to sum; converted to float64.

    Returns
    -------
    cupy.ndarray of float64, shape (rows, cols + 1)
        The prefix sums.
    """
    global _ROW_CUMSUM
    if _ROW_CUMSUM is None:
        _ROW_CUMSUM = cp.RawKernel(r'''
        extern "C" __global__ void row_cumsum(const double* x, double* out, long long rows, long long cols) {
            long long r = (long long)blockDim.x * blockIdx.x + threadIdx.x;
            if (r >= rows) return;
            double acc = 0.0;
            out[r * (cols + 1)] = 0.0;
            for (long long t = 0; t < cols; ++t) {
                acc += x[r * cols + t];
                out[r * (cols + 1) + t + 1] = acc;
            }
        }''', "row_cumsum", options=("--fmad=false",))
    values = cp.ascontiguousarray(values, dtype=cp.float64)
    rows, cols = values.shape
    out = cp.empty((rows, cols + 1), dtype=cp.float64)
    if rows:
        _ROW_CUMSUM(((rows + 255) // 256,), (256,), (values, out, np.int64(rows), np.int64(cols)))
    return out


def _prefix(values, dtype):
    """Return prefix sums along time with a leading zero column, so ``table[:, t]`` sums days ``[0, t)``.

    Parameters
    ----------
    values : cupy.ndarray, shape (series, time)
        The values to sum (bool, int or float).
    dtype : numpy.dtype or type
        The dtype of the sums; a float dtype uses `_row_cumsum` for time-ordered rounding.

    Returns
    -------
    cupy.ndarray of ``dtype``, shape (series, time + 1)
        The prefix sums.
    """
    if np.dtype(dtype).kind == "f":
        return _row_cumsum(values)
    out = cp.zeros((values.shape[0], values.shape[1] + 1), dtype=dtype)
    cp.cumsum(values, axis=1, dtype=dtype, out=out[:, 1:])
    return out


def _gather(table, series, index):
    """Return ``table[series, index]`` with the index clipped into the table.

    Parameters
    ----------
    table : cupy.ndarray, shape (series, n)
        A prefix table.
    series : cupy.ndarray of int64, shape (jobs, 1)
        The row of each job.
    index : cupy.ndarray of int64, shape (jobs, candidates)
        The column to read; clipped to ``[0, n - 1]``.

    Returns
    -------
    cupy.ndarray, shape (jobs, candidates)
        The gathered values.
    """
    return table[series, cp.clip(index, 0, table.shape[1] - 1)]


def _observe(tables: _Tables, job: _Block, a, b):
    """Return whether the span ``[a, b)`` can be observed, with the reason when it cannot.

    A span is not observable when it starts before day 0, contains a missing day (unless missing days are kept), or
    reaches past the job's data end.

    Parameters
    ----------
    tables : _Tables
        The prefix tables.
    job : _Block
        The block of jobs.
    a : cupy.ndarray of int64, shape (jobs, candidates)
        The first day of the span.
    b : cupy.ndarray of int64, shape (jobs, candidates)
        The end of the span, exclusive.

    Returns
    -------
    observable : cupy.ndarray of bool, shape (jobs, candidates)
        True where the whole span is observed.
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        Bit 1 (`MISSING_FLAG`) for a missing or pre-history day, bit 2 (`HORIZON_FLAG`) for a span past the data end.
    """
    e = job.data_end
    lo = cp.clip(a, 0, e)
    hi = cp.maximum(cp.minimum(b, e), lo)
    n_missing = tables.pm[job.series, hi] - tables.pm[job.series, lo]
    flags = (((a < 0) | ((n_missing > 0) & (not tables.keep))) * MISSING_FLAG + (b > e) * HORIZON_FLAG).astype(cp.int8)
    return flags == 0, flags


def _condition(plan_cond: P.CondPlan, tables: _Tables, job: _Block):
    """Evaluate a Total or WetDays requirement for every candidate of a block.

    Parameters
    ----------
    plan_cond : hcwf.onset.plan.CondPlan
        The requirement.
    tables : _Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    code : cupy.ndarray of int8, shape (jobs, candidates)
        The three-valued code (1 True, 0 False, -1 Unknown).
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        The flags, non-zero only where the code is Unknown.
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
    code = cp.where(ok, cp.where(result, TRUE, FALSE), UNKNOWN).astype(cp.int8)
    return code, cp.where(ok, 0, flags).astype(cp.int8)


def _veto(veto: P.VetoPlan, spell_prefix, tables: _Tables, job: _Block):
    """Evaluate a veto for every candidate of a block.

    A code of ``TRUE`` means a spell was confirmed, so the candidate is rejected.

    Parameters
    ----------
    veto : hcwf.onset.plan.VetoPlan
        The veto.
    spell_prefix : cupy.ndarray of int32, shape (series, time + 2)
        The prefix table of spell start days for this veto (`_Tables.veto_prefix`).
    tables : _Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    code : cupy.ndarray of int8, shape (jobs, candidates)
        The three-valued code.
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        The flags, non-zero only where the code is Unknown.
    """
    a, b = windowed.span(job, veto.first, veto.last, False)
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
    return _veto_code(tables, job, b, confirmed, ok, flags)


def _veto_code(tables, job, b, confirmed, ok, flags):
    """Return the code and flags of a veto from its confirmation and observability, applying the horizon policy.

    Parameters
    ----------
    tables : _Tables or _StitchedTables
        The tables; their ``horizon`` policy (``censor``, ``reject`` or other) decides how a span past the data end
        is coded.
    job : _Block or _StitchedBlock
        The block of jobs (its ``data_end`` is read).
    b : cupy.ndarray of int64, shape (jobs, candidates)
        The end of the veto's span, exclusive.
    confirmed : cupy.ndarray of bool, shape (jobs, candidates)
        True where a spell was found.
    ok : cupy.ndarray of bool, shape (jobs, candidates)
        True where the days the veto needs are observable.
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        The flags of the observability check.

    Returns
    -------
    code : cupy.ndarray of int8, shape (jobs, candidates)
        The three-valued code.
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        The flags, non-zero only where the code is Unknown.
    """
    code = cp.where(confirmed, TRUE, cp.where(ok, FALSE, UNKNOWN)).astype(cp.int8)
    flags = cp.where(code == UNKNOWN, flags, 0).astype(cp.int8)
    if tables.horizon != "censor":
        code = cp.where(flags == HORIZON_FLAG, FALSE, code).astype(cp.int8)
        flags = cp.where(code == UNKNOWN, flags, 0).astype(cp.int8)
        if tables.horizon == "reject":
            code = cp.where(b > job.data_end, TRUE, code).astype(cp.int8)
            flags = cp.where(code == UNKNOWN, flags, 0).astype(cp.int8)
    return code, flags


def _status_from_flags(flags):
    """Map flags to a status: ``CENSORED`` when the flags are exactly the horizon bit, else ``INSUFFICIENT_DATA``.

    Parameters
    ----------
    flags : cupy.ndarray of int8
        The flags of the chosen candidates.

    Returns
    -------
    cupy.ndarray of int8
        The `~hcwf.onset.model.Status` code, same shape as ``flags``.
    """
    return cp.where(flags == HORIZON_FLAG, int(Status.CENSORED), int(Status.INSUFFICIENT_DATA)).astype(cp.int8)


def _first_passage_block(plan: P.Plan, tables: _Tables, job: _Block):
    """Evaluate all candidates of a block and pick the first one that is not False.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled first-passage definition.
    tables : _Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    day : cupy.ndarray of int64, shape (jobs,)
        The reported day index, ``-1`` for none.
    status : cupy.ndarray of int8, shape (jobs,)
        The `~hcwf.onset.model.Status` code.
    """
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
    if tables.keep:
        any_false |= ((tables.pm[job.series, cp.clip(job.d + 1, 0, tables.n_time)]
                       - tables.pm[job.series, cp.clip(job.d, 0, tables.n_time)]) > 0) & (job.d < job.data_end)
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
    """Evaluate the cumulative-anomaly rule for every job of a block.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled curve definition.
    tables : _Tables
        The prefix tables.
    job : _Block
        The block of jobs.

    Returns
    -------
    day : cupy.ndarray of int64, shape (jobs,)
        The reported day index, ``-1`` for none.
    status : cupy.ndarray of int8, shape (jobs,)
        The `~hcwf.onset.model.Status` code.
    """
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
    """Return the unique (cell, amount) pairs and each job's index into them.

    Parameters
    ----------
    cell : numpy.ndarray of int64, shape (jobs,)
        The history row of each job.
    amount : numpy.ndarray of int64, shape (jobs,)
        The amount of each job.

    Returns
    -------
    cells : numpy.ndarray of int64, shape (variants,)
        The cell of each unique pair, sorted by cell then amount.
    amounts : numpy.ndarray of int64, shape (variants,)
        The amount of each unique pair.
    inverse : numpy.ndarray of int64, shape (jobs,)
        The index of each job's pair in ``cells`` and ``amounts``.
    """
    pairs = np.stack([cell, amount], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique[:, 0].copy(), unique[:, 1].copy(), inverse.reshape(-1).astype(np.int64)


def detect_stitched(plan: P.Plan, hist_tenths: np.ndarray, hist_missing: np.ndarray, fc_tenths: np.ndarray,
                    fc_missing: np.ndarray, amounts: dict[str, np.ndarray], jobs: StitchedJobs, *,
                    trace: bool = False, amount_windows=None) -> Detection:
    """Run stitched jobs (history, then a forecast row) on the GPU without copying history into forecasts.

    History prefix tables are built once over the cells and columns the jobs can reach, with one wet-day (and
    dry-window) variant per distinct (cell, amount) pair since amounts are per forecast series. Forecast tables are
    built over the forecast rows. Any interval is read as ``P_hist(min(t, split))`` plus ``P_fc(t - split)`` for ``t``
    past the split, and spells that start in history and end in the forecast come from a small per-job seam table. The
    result equals `detect` on the explicit concatenation and never reads history at or after a split. Per-window
    amounts work as in `detect`.

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
        Must be false: traces come from the reference backend.
    amount_windows : array_like of int, optional
        The ``W`` window start leads when ``amounts`` have ``W`` columns.

    Returns
    -------
    hcwf.onset.model.Detection
        ``day`` (int64, shape ``(jobs,)``, stitched day index or ``-1``), ``status`` (int8) and ``trace`` (``None``),
        as NumPy arrays on the host.

    Raises
    ------
    NotImplementedError
        `trace` is true.
    """
    if trace:
        raise NotImplementedError("use the reference backend for traces")
    P.reject_exact_stitched(hist_tenths, fc_tenths)
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
    """History tables over the reachable cells and columns, forecast tables, and the per-job row indices into them.

    Builds the forecast tables with `_Tables` and the history tables per cell and per (cell, amount) variant.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    hist_tenths : numpy.ndarray of int64, shape (cell, time)
        Observed daily rain in tenths of mm.
    hist_missing : numpy.ndarray of bool, shape (cell, time)
        True where the observed day is missing.
    fc_tenths : numpy.ndarray of int64, shape (series, lead)
        Forecast daily rain in tenths of mm.
    fc_missing : numpy.ndarray of bool, shape (series, lead)
        True where the forecast day is missing.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key, shape ``(series,)``.
    jobs : hcwf.onset.model.StitchedJobs
        The jobs, in stitched days.
    active : numpy.ndarray of bool, shape (jobs,)
        The jobs that have at least one candidate; only they set the history range.

    Attributes
    ----------
    fc : _Tables
        The forecast tables.
    n_fc : int
        The number of forecast days.
    c0 : int
        The history day of the first history column.
    width : int
        The number of history columns kept.
    px, pm : cupy.ndarray, shape (cells, width + 1)
        History prefix sums of rain and missing days over the reachable cells.
    pw : cupy.ndarray of int32, shape (variants, width + 1)
        History prefix counts of wet days per (cell, wet amount) variant.
    hd : cupy.ndarray of int32
        History prefix counts of dry days (or ``pw`` when the definition has no own ``dry_day``).
    fd : cupy.ndarray of int32
        Forecast prefix counts of dry days (or wet days without an own ``dry_day``).
    use_pd : bool
        Whether the definition has its own ``dry_day``.
    wet_order : cupy.ndarray of int32 or None
        The wet history days of each variant first, in time order; ``None`` unless the plan reports the first wet day.
    cell, series, split, start, end, origin, data_end : cupy.ndarray of int64, shape (jobs,)
        The per-job arrays (``cell`` indexes the reachable cells; ``data_end`` is clipped to the forecast end).
    wet_variant, dry_variant : cupy.ndarray of int64, shape (jobs,)
        The row of ``pw`` and of ``hd`` of each job.
    spell_prefix : list of cupy.ndarray of int32
        One history spell-start prefix table per veto.
    spell_variant : list of cupy.ndarray of int64
        The row of ``spell_prefix`` of each job, per veto.
    amounts : dict of str to cupy.ndarray
        The amounts on the device.
    horizon : str
        The plan's horizon policy.
    keep : bool
        The plan's ``keep_missing`` flag.

    Raises
    ------
    ValueError
        A job's split lies outside the history.
    """

    def __init__(self, plan, hist_tenths, hist_missing, fc_tenths, fc_missing, amounts, jobs, active):
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
        self.c0 = min(windowed.first_column(plan, start_h[active], origin_h[active]), c1)
        self.width = c1 - self.c0
        cells, cell_idx = np.unique(cell_h, return_inverse=True)
        cell_idx = cell_idx.reshape(-1).astype(np.int64)
        miss = cp.asarray(hist_missing[cells, self.c0:c1], dtype=cp.bool_)
        x = cp.where(miss, 0, cp.asarray(hist_tenths[cells, self.c0:c1])).astype(cp.int64)
        self.px = _prefix(x, cp.int64)
        self.pm = _prefix(miss, cp.int32)

        self.horizon, self.keep = plan.horizon, plan.keep_missing
        wet_amount_h = np.asarray(amounts[plan.wet.key], dtype=np.int64)[series_h]
        wcell, wamt, wv = _variants(cell_idx, wet_amount_h)
        wcell_d, wamt_d = cp.asarray(wcell), cp.asarray(wamt)
        miss_v = miss[wcell_d]
        wet = ~miss_v & windowed.compare(plan.wet.op, x[wcell_d], wamt_d[:, None])
        self.pw = _prefix(wet, cp.int32)
        self.wet_order = None
        if plan.report_first_wet:
            self.wet_order = cp.argsort(~wet, axis=1, kind="stable").astype(cp.int32)
        # Dry days: the wet variants' complement, or one variant per (cell, dry amount) with its own dry_day.
        self.use_pd = plan.dry is not None
        if self.use_pd:
            dry_amount_h = np.asarray(amounts[plan.dry.key], dtype=np.int64)[series_h]
            dcell, damt, dv = _variants(cell_idx, dry_amount_h)
            dcell_d, damt_d = cp.asarray(dcell), cp.asarray(damt)
            dry = ~miss[dcell_d] & windowed.compare(plan.dry.op, x[dcell_d], damt_d[:, None])
        else:
            dcell, dv = wcell, wv
            dry = ~miss_v & ~wet
        dry_prefix = _prefix(dry, cp.int32)
        self.hd = dry_prefix if self.use_pd else self.pw
        self.fd = self.fc.pd if self.use_pd else self.fc.pw
        self.dry_variant = cp.asarray(dv)
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
                variant = dv
                holds = cp.zeros((len(dcell), self.width + 1), dtype=cp.bool_)
                if span > 0:
                    holds[:, :span] = (dry_prefix[:, veto.length:] - dry_prefix[:, :span]) == veto.length
                    if veto.placement == P.BEGINS:
                        holds[:, 1:span] &= ~dry[:, :span - 1]
            else:
                amount_h = np.asarray(amounts[veto.cmp.key], dtype=np.int64)[series_h]
                vcell, vamt, variant = _variants(cell_idx, amount_h)
                vcell_d, vamt_d = cp.asarray(vcell), cp.asarray(vamt)
                holds = cp.zeros((len(vcell), self.width + 1), dtype=cp.bool_)
                if span > 0:
                    px_v, pm_v = self.px[vcell_d], self.pm[vcell_d]
                    total = px_v[:, veto.length:] - px_v[:, :span]
                    complete = (pm_v[:, veto.length:] - pm_v[:, :span]) == 0
                    holds[:, :span] = complete & windowed.compare(veto.cmp.op, total, vamt_d[:, None])
            self.spell_prefix.append(_prefix(holds, cp.int32))
            self.spell_variant.append(cp.asarray(variant))
        del dry, dry_prefix
        self.amounts = self.fc.amounts


class _StitchedBlock:
    """The jobs of one block: row indices, split, window, and the candidate grid (days ``d``, validity mask).

    Slices the per-job arrays of the tables into columns of shape ``(jobs, 1)`` and builds the candidate day grid.

    Parameters
    ----------
    tables : _StitchedTables
        The tables holding the per-job arrays.
    sl : slice
        The jobs of the block.
    width : int
        The number of candidate columns (the longest job in the block).

    Attributes
    ----------
    n_time : cupy.ndarray of int64, shape (jobs, 1)
        The stitched series length of each job (``split`` plus the forecast length).
    n_jobs : int
        The number of jobs in the block.
    variants : list of cupy.ndarray
        The spell-table row of each job, per veto.
    d : cupy.ndarray of int64, shape (jobs, width)
        The candidate day (stitched) of each cell.
    valid : cupy.ndarray of bool, shape (jobs, width)
        True where the candidate day is not past the job's ``end``.
    """

    def __init__(self, tables: _StitchedTables, sl: slice, width: int):
        self.series = tables.series[sl][:, None]
        self.cell = tables.cell[sl][:, None]
        self.wet_variant = tables.wet_variant[sl][:, None]
        self.dry_variant = tables.dry_variant[sl][:, None]
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
    """Return the prefix value at stitched index ``t``.

    The value is ``P_hist(min(t, split))`` plus ``P_fc(t - split)`` once past the split (the forecast table is 0 at
    0). Only differences of two values are meaningful.

    Parameters
    ----------
    tables : _StitchedTables
        The tables (history offset ``c0``, widths).
    hist_table : cupy.ndarray, shape (history rows, width + 1)
        A history prefix table.
    fc_table : cupy.ndarray, shape (series, n_fc + 1)
        The matching forecast prefix table.
    hist_row : cupy.ndarray of int64, shape (jobs, 1)
        The row of ``hist_table`` of each job.
    job : _StitchedBlock
        The block of jobs.
    t : cupy.ndarray of int64, shape (jobs, candidates)
        The stitched day index.

    Returns
    -------
    cupy.ndarray, shape (jobs, candidates)
        The prefix values.
    """
    h = hist_table[hist_row, cp.clip(cp.minimum(t, job.split) - tables.c0, 0, tables.width)]
    f = fc_table[job.series, cp.clip(t - job.split, 0, tables.n_fc)]
    return h + f


def _observe_s(tables: _StitchedTables, job: _StitchedBlock, a, b):
    """Return whether the stitched interval ``[a, b)`` can be observed, with the reason when it cannot.

    Parameters
    ----------
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.
    a : cupy.ndarray of int64, shape (jobs, candidates)
        The first day of the interval.
    b : cupy.ndarray of int64, shape (jobs, candidates)
        The end of the interval, exclusive.

    Returns
    -------
    observable : cupy.ndarray of bool, shape (jobs, candidates)
        True where the whole interval is observed.
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        Bit 1 (`MISSING_FLAG`) for missing or pre-history days, bit 2 (`HORIZON_FLAG`) for a span past the data end.
    """
    e = job.data_end
    lo = cp.clip(a, 0, e)
    hi = cp.maximum(cp.minimum(b, e), lo)
    n_missing = _g(tables, tables.pm, tables.fc.pm, job.cell, job, hi) - _g(tables, tables.pm, tables.fc.pm, job.cell, job, lo)
    flags = (((a < 0) | ((n_missing > 0) & (not tables.keep))) * MISSING_FLAG + (b > e) * HORIZON_FLAG).astype(cp.int8)
    return flags == 0, flags


def _condition_s(plan_cond: P.CondPlan, tables: _StitchedTables, job: _StitchedBlock):
    """Evaluate a Total or WetDays requirement on the stitched series for every candidate of a block.

    Parameters
    ----------
    plan_cond : hcwf.onset.plan.CondPlan
        The requirement.
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.

    Returns
    -------
    code : cupy.ndarray of int8, shape (jobs, candidates)
        The three-valued code (1 True, 0 False, -1 Unknown).
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        The flags, non-zero only where the code is Unknown.
    """
    a, b = windowed.span(job, plan_cond.first, plan_cond.last, plan_cond.since_start)
    ok, flags = _observe_s(tables, job, a, b)
    if plan_cond.kind == P.TOTAL:
        value = (_g(tables, tables.px, tables.fc.px, job.cell, job, b) - _g(tables, tables.px, tables.fc.px, job.cell, job, a))
        amount = tables.amounts[plan_cond.cmp.key][job.series]
        result = windowed.compare(plan_cond.cmp.op, value, amount)
    else:
        value = (_g(tables, tables.pw, tables.fc.pw, job.wet_variant, job, b)
                 - _g(tables, tables.pw, tables.fc.pw, job.wet_variant, job, a))
        result = windowed.compare(plan_cond.op, value, plan_cond.count)
    code = cp.where(ok, cp.where(result, TRUE, FALSE), UNKNOWN).astype(cp.int8)
    return code, cp.where(ok, 0, flags).astype(cp.int8)


def _seam(veto: P.VetoPlan, tables: _StitchedTables, job: _StitchedBlock):
    """Return the per-job cumulative indicator of the seam spells that hold.

    The seam spells are those starting in ``[split - length + 1, split - 1]``, so they straddle the split. A begins
    run also starts at the split itself (``n = length``), and needs the history day before it not to be dry.

    Parameters
    ----------
    veto : hcwf.onset.plan.VetoPlan
        The veto.
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.

    Returns
    -------
    cupy.ndarray of int32, shape (jobs, n + 1)
        The cumulative count of holding starts, ``n = length - 1`` (``length`` for a begins veto); column ``i + 1``
        counts the starts up to ``split - length + 1 + i``. Shape ``(jobs, 1)`` of zeros when ``n`` is 0.
    """
    length, begin = veto.length, int(veto.placement == P.BEGINS)
    n_seam = length - 1 + begin
    seam = cp.zeros((job.n_jobs, n_seam + 1), dtype=cp.int32)
    if n_seam == 0:
        return seam
    i = cp.arange(n_seam, dtype=cp.int64)[None, :]
    u = job.split - length + 1 + i
    fc_len = i + 1
    inside = (u >= tables.c0) & (u + length <= job.n_time)
    split_col = cp.clip(job.split - tables.c0, 0, tables.width)
    u_col = cp.clip(u - tables.c0, 0, tables.width)
    fc_col = cp.clip(fc_len, 0, tables.n_fc)

    def hist_diff(table, row):
        """Return the sum of the history part ``[u, split)`` of a prefix table."""
        return table[row, split_col] - table[row, u_col]

    missing = hist_diff(tables.pm, job.cell) + tables.fc.pm[job.series, fc_col]
    if veto.kind == P.DRY_RUN:
        count = hist_diff(tables.hd, job.dry_variant) + tables.fd[job.series, fc_col]
        holds = inside & (missing == 0) & (count == (length if tables.use_pd else 0))
        if begin:
            prev = cp.clip(u - 1 - tables.c0, 0, tables.width)
            gap = tables.hd[job.dry_variant, u_col] - tables.hd[job.dry_variant, prev]
            if tables.use_pd:
                prev_dry = gap == 1
            else:
                prev_dry = (gap == 0) & (tables.pm[job.cell, u_col] - tables.pm[job.cell, prev] == 0)
            holds &= ~((u - 1 >= tables.c0) & prev_dry)
    else:
        total = hist_diff(tables.px, job.cell) + tables.fc.px[job.series, fc_col]
        amount = tables.amounts[veto.cmp.key][job.series]
        holds = inside & (missing == 0) & windowed.compare(veto.cmp.op, total, amount)
    seam[:, 1:] = cp.cumsum(holds, axis=1, dtype=cp.int32)
    return seam


def _veto_s(veto: P.VetoPlan, spell_prefix, variant, fc_prefix, tables: _StitchedTables, job: _StitchedBlock):
    """Evaluate a veto on the stitched series for every candidate of a block.

    A code of ``TRUE`` means a spell was confirmed, so the candidate is rejected. Spells are counted in history only,
    in the seam and in the forecast only.

    Parameters
    ----------
    veto : hcwf.onset.plan.VetoPlan
        The veto.
    spell_prefix : cupy.ndarray of int32, shape (variants, width + 1)
        The history spell-start prefix table of this veto.
    variant : cupy.ndarray of int64, shape (jobs, 1)
        The row of ``spell_prefix`` of each job.
    fc_prefix : cupy.ndarray of int32, shape (series, n_fc + 2)
        The forecast spell-start prefix table of this veto.
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.

    Returns
    -------
    code : cupy.ndarray of int8, shape (jobs, candidates)
        The three-valued code.
    flags : cupy.ndarray of int8, shape (jobs, candidates)
        The flags, non-zero only where the code is Unknown.
    """
    a, b = windowed.span(job, veto.first, veto.last, False)
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
    begin = int(veto.placement == P.BEGINS)
    seam = _seam(veto, tables, job)
    rows = cp.arange(job.n_jobs)[:, None]
    x = cp.minimum(hi, base)
    count = cp.where(x > lo, spell_prefix[variant, cp.clip(x - tables.c0, 0, tables.width)]
                     - spell_prefix[variant, cp.clip(lo - tables.c0, 0, tables.width)], 0)
    x0, x1 = cp.maximum(lo, base), cp.minimum(hi, job.split + begin)
    count = count + cp.where(x1 > x0, seam[rows, cp.clip(x1 - base, 0, length - 1 + begin)]
                             - seam[rows, cp.clip(x0 - base, 0, length - 1 + begin)], 0)
    y0 = cp.maximum(lo, job.split + begin)
    count = count + cp.where(hi > y0, fc_prefix[job.series, cp.clip(hi - job.split, 0, tables.n_fc)]
                             - fc_prefix[job.series, cp.clip(y0 - job.split, 0, tables.n_fc)], 0)
    return _veto_code(tables, job, b, count > 0, ok, flags)


def _first_wet_day_s(tables: _StitchedTables, job: _StitchedBlock, onset, pick_day, day, report_span):
    """Replace onset days by the first wet day in the report window (history part first, then the forecast).

    Parameters
    ----------
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.
    onset : cupy.ndarray of bool, shape (jobs,)
        True for the jobs that found an onset.
    pick_day : cupy.ndarray of int64, shape (jobs,)
        The onset candidate day of each job.
    day : cupy.ndarray of int64, shape (jobs,)
        The reported days so far.
    report_span : tuple of (int, int, bool)
        The report span: first and last offset from the candidate day, and whether it starts at the job's origin.

    Returns
    -------
    cupy.ndarray of int64, shape (jobs,)
        ``day`` with the onset days moved to the first wet day of the span where there is one.
    """
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
    """Evaluate all candidates of a block of stitched jobs and pick the first one that is not False.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled first-passage definition.
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.

    Returns
    -------
    day : cupy.ndarray of int64, shape (jobs,)
        The reported stitched day index, ``-1`` for none.
    status : cupy.ndarray of int8, shape (jobs,)
        The `~hcwf.onset.model.Status` code.
    """
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
    if tables.keep:
        any_false |= ((_g(tables, tables.pm, tables.fc.pm, job.cell, job, job.d + 1)
                       - _g(tables, tables.pm, tables.fc.pm, job.cell, job, job.d)) > 0) & (job.d < job.data_end)
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
    """Evaluate the cumulative-anomaly rule for every stitched job of a block.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled curve definition.
    tables : _StitchedTables
        The tables.
    job : _StitchedBlock
        The block of jobs.

    Returns
    -------
    day : cupy.ndarray of int64, shape (jobs,)
        The reported stitched day index, ``-1`` for none.
    status : cupy.ndarray of int8, shape (jobs,)
        The `~hcwf.onset.model.Status` code.
    """
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
