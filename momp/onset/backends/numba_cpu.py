"""Numba CPU backend: the same semantics as the reference backend, walking candidates per job with an early exit.

Prefix tables (rain, missing days, wet days and per-veto spell starts) are built
once per call in parallel over series. A parallel kernel then runs one job per
loop iteration, scanning candidates in order and stopping at the first one that
is not False, so no job x candidate masks are ever stored.
"""
from __future__ import annotations

import numpy as np

from momp.onset import plan as P
from momp.onset.model import Detection, Jobs, Status

try:
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None

MISSING_FLAG, HORIZON_FLAG = 1, 2
_ONSET, _NOT_FOUND, _CENSORED, _INSUFFICIENT = (int(s) for s in (
    Status.ONSET, Status.NOT_FOUND, Status.CENSORED, Status.INSUFFICIENT_DATA))


def available() -> bool:
    """Whether numba can be imported."""
    return numba is not None


if numba is not None:

    @njit(inline="always")
    def _cmp(op, value, amount):
        """Comparison selected by an operator code (GE=0, GT=1, LE=2, LT=3)."""
        if op == 0:
            return value >= amount
        if op == 1:
            return value > amount
        if op == 2:
            return value <= amount
        return value < amount

    @njit(inline="always")
    def _flags(pm_row, a, b, data_end):
        """Missing/horizon flag bits of [a, b); zero means the interval is observable."""
        lo = min(max(a, 0), data_end)
        hi = max(min(b, data_end), lo)
        flags = 0
        if a < 0 or pm_row[hi] - pm_row[lo] > 0:
            flags |= 1
        if b > data_end:
            flags |= 2
        return flags

    @njit(parallel=True, cache=True)
    def _build_tables(tenths, missing, wet_op, wet_amount, px, pm, pw):
        """Fill the prefix tables of rain (int64), missing days and wet days (int32), each with a leading zero."""
        n_series, n_time = tenths.shape
        for s in prange(n_series):
            ax = np.int64(0)
            am = 0
            aw = 0
            px[s, 0] = 0
            pm[s, 0] = 0
            pw[s, 0] = 0
            for t in range(n_time):
                miss = missing[s, t]
                x = np.int64(0) if miss else np.int64(tenths[s, t])
                ax += x
                if miss:
                    am += 1
                elif _cmp(wet_op, x, wet_amount[s]):
                    aw += 1
                px[s, t + 1] = ax
                pm[s, t + 1] = am
                pw[s, t + 1] = aw

    @njit(inline="always")
    def _dry_count(pm, pw, s, u, length):
        """Number of dry days (neither missing nor wet) in [u, u + length)."""
        return length - (pm[s, u + length] - pm[s, u]) - (pw[s, u + length] - pw[s, u])

    @njit(parallel=True, cache=True)
    def _build_spells(px, pm, pw, kind, length, op, amount, sp):
        """Fill sp[s, u + 1] cumulatively with the count of spell starts <= u whose spell lies in the data and holds."""
        n_series = px.shape[0]
        n_time = px.shape[1] - 1
        for s in prange(n_series):
            sp[s, 0] = 0
            acc = 0
            for u in range(n_time):
                holds = False
                if u + length <= n_time:
                    if kind == 0:
                        holds = _dry_count(pm, pw, s, u, length) == length
                    else:
                        holds = (pm[s, u + length] - pm[s, u] == 0) and _cmp(op, px[s, u + length] - px[s, u], amount[s])
                if holds:
                    acc += 1
                sp[s, u + 1] = acc

    @njit(parallel=True, cache=True)
    def _first_passage(px, pm, pw, sp, amounts,
                       c_kind, c_first, c_last, c_since, c_op, c_key, c_count,
                       v_kind, v_length, v_place, v_first, v_last,
                       report, r_first, r_last, r_since,
                       series, start, end, data_end, day, status):
        """Find the first candidate that is not False for every job (three-valued logic, early exit)."""
        n_time = pm.shape[1] - 1
        for j in prange(len(series)):
            s = series[j]
            st = start[j]
            e = min(data_end[j], n_time)
            pm_row = pm[s]
            day[j] = -1
            status[j] = _NOT_FOUND
            for d in range(st, end[j] + 1):
                dead = False
                unknown = False
                uflags = 0
                for c in range(len(c_kind)):
                    a = st if c_since[c] else d + c_first[c]
                    b = d + c_last[c]
                    fl = _flags(pm_row, a, b, e)
                    if fl != 0:
                        unknown = True
                        uflags |= fl
                        continue
                    if c_kind[c] == 0:
                        ok = _cmp(c_op[c], px[s, b] - px[s, a], amounts[c_key[c], s])
                    else:
                        ok = _cmp(c_op[c], pw[s, b] - pw[s, a], c_count[c])
                    if not ok:
                        dead = True
                        break
                if dead:
                    continue
                for k in range(len(v_kind)):
                    a = d + v_first[k]
                    b = d + v_last[k]
                    length = v_length[k]
                    if v_place[k] == 0:
                        if v_last[k] - v_first[k] < length:
                            continue
                        u_last = b - length
                        fl = _flags(pm_row, a, b, e)
                    else:
                        if b <= a:
                            continue
                        u_last = b - 1
                        fl = _flags(pm_row, a, b + length - 1, e)
                    u_last = min(u_last, e - length)
                    lo = min(max(a, 0), n_time)
                    hi = max(min(max(u_last + 1, 0), n_time), lo)
                    if sp[k, s, hi] - sp[k, s, lo] > 0:
                        dead = True
                        break
                    if fl != 0:
                        unknown = True
                        uflags |= fl
                if dead:
                    continue
                if unknown:
                    status[j] = _CENSORED if uflags == 2 else _INSUFFICIENT
                    break
                status[j] = _ONSET
                day[j] = d
                if report:
                    a = st if r_since else d + r_first
                    b = d + r_last
                    for t in range(max(a, 0), min(b, e)):
                        if pw[s, t + 1] - pw[s, t] > 0:
                            day[j] = t
                            break
                break

    @njit(parallel=True, cache=True)
    def _curve(px, pm, amounts, key, after, series, start, end, data_end, day, status):
        """Cumulative-anomaly minimum for every job."""
        n_time = pm.shape[1] - 1
        for j in prange(len(series)):
            s = series[j]
            st = start[j]
            e = min(data_end[j], n_time)
            day[j] = -1
            status[j] = _NOT_FOUND
            if end[j] < st:
                continue
            fl = _flags(pm[s], st, end[j] + 1, e)
            if fl != 0:
                status[j] = _CENSORED if fl == 2 else _INSUFFICIENT
                continue
            baseline = amounts[key, s]
            best = np.int64(0)
            arg = st
            for t in range(st, end[j] + 1):
                value = 100 * (px[s, t + 1] - px[s, st]) - baseline * (t - st + 1)
                if t == st or value < best:
                    best = value
                    arg = t
            if arg == end[j]:
                continue
            status[j] = _ONSET
            day[j] = arg + 1 if after else arg


def detect(plan: P.Plan, tenths: np.ndarray, missing: np.ndarray, amounts: dict[str, np.ndarray],
           jobs: Jobs, *, trace: bool = False) -> Detection:
    """Run every job and return the onset day index (-1 if none) and Status per job, bit-identical to the reference backend."""
    if trace:
        raise NotImplementedError("use the reference backend for traces")
    if numba is None:
        raise ImportError("the numba backend needs numba")
    series = np.ascontiguousarray(jobs.series, dtype=np.int64)
    start = np.ascontiguousarray(jobs.start, dtype=np.int64)
    end = np.ascontiguousarray(jobs.end, dtype=np.int64)
    data_end = np.ascontiguousarray(jobs.data_end, dtype=np.int64)
    day = np.full(len(series), -1, dtype=np.int64)
    status = np.full(len(series), _NOT_FOUND, dtype=np.int8)
    if len(series) == 0:
        return Detection(day, status, None)
    tenths = np.ascontiguousarray(tenths, dtype=np.int32)
    missing = np.ascontiguousarray(missing, dtype=np.bool_)
    n_series, n_time = tenths.shape

    keys = [plan.wet.key]
    for item in (*plan.conditions, *plan.vetoes):
        if item.cmp is not None:
            keys.append(item.cmp.key)
    if plan.curve_key is not None:
        keys.append(plan.curve_key)
    keys = list(dict.fromkeys(keys))
    stack = np.stack([np.asarray(amounts[k], dtype=np.int64) for k in keys])
    index = {k: i for i, k in enumerate(keys)}

    px = np.empty((n_series, n_time + 1), dtype=np.int64)
    pm = np.empty((n_series, n_time + 1), dtype=np.int32)
    pw = np.empty((n_series, n_time + 1), dtype=np.int32)
    _build_tables(tenths, missing, plan.wet.op, stack[index[plan.wet.key]], px, pm, pw)

    if plan.family == P.CURVE:
        _curve(px, pm, stack, index[plan.curve_key], plan.curve_after_min, series, start, end, data_end,
               day, status)
        return Detection(day, status, None)

    sp = np.zeros((len(plan.vetoes), n_series, n_time + 1), dtype=np.int32)
    for k, veto in enumerate(plan.vetoes):
        op, amount = (veto.cmp.op, stack[index[veto.cmp.key]]) if veto.cmp is not None else (0, stack[0])
        _build_spells(px, pm, pw, veto.kind, veto.length, op, amount, sp[k])

    def ints(values):
        """A tuple of ints as a small int64 array."""
        return np.array(values, dtype=np.int64)

    conds = plan.conditions
    c_op = [c.cmp.op if c.kind == P.TOTAL else c.op for c in conds]
    c_key = [index[c.cmp.key] if c.kind == P.TOTAL else 0 for c in conds]
    report = plan.report_first_wet and plan.report_span is not None
    r_first, r_last, r_since = plan.report_span if report else (0, 0, False)
    _first_passage(px, pm, pw, sp, stack,
                   ints([c.kind for c in conds]), ints([c.first for c in conds]), ints([c.last for c in conds]),
                   np.array([c.since_start for c in conds], dtype=np.bool_), ints(c_op), ints(c_key),
                   ints([c.count for c in conds]),
                   ints([v.kind for v in plan.vetoes]), ints([v.length for v in plan.vetoes]),
                   ints([v.placement for v in plan.vetoes]), ints([v.first for v in plan.vetoes]),
                   ints([v.last for v in plan.vetoes]),
                   report, r_first, r_last, r_since, series, start, end, data_end, day, status)
    return Detection(day, status, None)
