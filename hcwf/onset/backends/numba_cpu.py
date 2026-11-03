"""Numba CPU backend: the same semantics as the reference backend, walking candidates per job with an early exit.

Prefix tables (rain, missing days, wet days, dry days when the definition has its own
dry_day, and per-veto spell starts) are built once per call in parallel over series.
Rain is int32 tenths, or float64 mm for rain_resolution "exact" (then every table
sums in time order from day 0, as np.cumsum does). A parallel kernel then runs one job per
loop iteration, scanning candidates in order and stopping at the first one that
is not False, so no job x candidate masks are ever stored.

detect_stitched does the same over observed history followed by a forecast, joining
separate history and forecast tables at each job's split instead of copying history.
"""
from __future__ import annotations

import numpy as np

from hcwf.onset import plan as P
from hcwf.onset.backends import windowed
from hcwf.onset.model import Detection, Jobs, Status, StitchedJobs

try:
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None

MISSING_FLAG, HORIZON_FLAG = 1, 2
HORIZONS = {"censor": 0, "truncate": 1, "reject": 2}
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
    def _flags(pm_row, a, b, data_end, keep):
        """Missing/horizon flag bits of [a, b); zero means the interval is observable (keep ignores missing days)."""
        lo = min(max(a, 0), data_end)
        hi = max(min(b, data_end), lo)
        flags = 0
        if a < 0 or (not keep and pm_row[hi] - pm_row[lo] > 0):
            flags |= 1
        if b > data_end:
            flags |= 2
        return flags

    @njit(parallel=True, cache=True)
    def _build_tables(tenths, missing, wet_op, wet_amount, use_pd, dry_op, dry_amount, px, pm, pw, pd):
        """Fill the prefix tables of rain (px: int64, or float64 for float rain), missing days, wet days and, with
        use_pd, dry days (int32), each with a leading zero. Without use_pd, pd is never touched."""
        n_series, n_time = tenths.shape
        for s in prange(n_series):
            px[s, 0] = 0
            ax = px[s, 0]
            am = 0
            aw = 0
            ad = 0
            pm[s, 0] = 0
            pw[s, 0] = 0
            if use_pd:
                pd[s, 0] = 0
            for t in range(n_time):
                miss = missing[s, t]
                x = 0 * tenths[s, t] if miss else tenths[s, t]
                ax += x
                if miss:
                    am += 1
                else:
                    if _cmp(wet_op, x, wet_amount[s]):
                        aw += 1
                    if use_pd and _cmp(dry_op, x, dry_amount[s]):
                        ad += 1
                px[s, t + 1] = ax
                pm[s, t + 1] = am
                pw[s, t + 1] = aw
                if use_pd:
                    pd[s, t + 1] = ad

    @njit(inline="always")
    def _dry_count(pm, pw, pd, use_pd, s, u, length):
        """Number of dry days in [u, u + length): the dry table's, else those neither missing nor wet."""
        if use_pd:
            return pd[s, u + length] - pd[s, u]
        return length - (pm[s, u + length] - pm[s, u]) - (pw[s, u + length] - pw[s, u])

    @njit(parallel=True, cache=True)
    def _build_spells(px, pm, pw, pd, use_pd, kind, length, op, amount, begin, sp):
        """Fill sp[s, u + 1] cumulatively with the count of spell starts <= u whose spell lies in the data and holds.

        With begin, a dry run only counts when the day before it is not dry."""
        n_series = px.shape[0]
        n_time = px.shape[1] - 1
        for s in prange(n_series):
            sp[s, 0] = 0
            acc = 0
            for u in range(n_time):
                holds = False
                if u + length <= n_time:
                    if kind == 0:
                        holds = _dry_count(pm, pw, pd, use_pd, s, u, length) == length
                        if holds and begin and u > 0 and _dry_count(pm, pw, pd, use_pd, s, u - 1, 1) == 1:
                            holds = False
                    else:
                        holds = (pm[s, u + length] - pm[s, u] == 0) and _cmp(op, px[s, u + length] - px[s, u], amount[s])
                if holds:
                    acc += 1
                sp[s, u + 1] = acc

    @njit(parallel=True, cache=True)
    def _first_passage(px, pm, pw, sp, amounts,
                       c_kind, c_first, c_last, c_since, c_op, c_key, c_count,
                       v_kind, v_length, v_place, v_first, v_last,
                       report, r_first, r_last, r_since, horizon, keep,
                       series, start, end, data_end, origin, day, status):
        """Find the first candidate that is not False for every job (three-valued logic, early exit).

        horizon is 0 censor, 1 truncate, 2 reject; keep ignores missing days (a missing candidate is rejected)."""
        n_time = pm.shape[1] - 1
        for j in prange(len(series)):
            s = series[j]
            st = start[j]
            org = origin[j]
            e = min(data_end[j], n_time)
            pm_row = pm[s]
            day[j] = -1
            status[j] = _NOT_FOUND
            for d in range(st, end[j] + 1):
                if keep and d < e:
                    dc = min(max(d, 0), n_time - 1)
                    if pm_row[dc + 1] - pm_row[dc] > 0:
                        continue
                dead = False
                unknown = False
                uflags = 0
                for c in range(len(c_kind)):
                    a = org if c_since[c] else d + c_first[c]
                    b = d + c_last[c]
                    fl = _flags(pm_row, a, b, e, keep)
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
                        fl = _flags(pm_row, a, b, e, keep)
                    else:
                        if b <= a:
                            continue
                        u_last = b - 1
                        fl = _flags(pm_row, a, b + length - 1, e, keep)
                    u_last = min(u_last, e - length)
                    lo = min(max(a, 0), n_time)
                    hi = max(min(max(u_last + 1, 0), n_time), lo)
                    if sp[k, s, hi] - sp[k, s, lo] > 0 or (horizon == 2 and b > e):
                        dead = True
                        break
                    if fl != 0 and not (horizon > 0 and fl == 2):
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
                    a = org if r_since else d + r_first
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
            fl = _flags(pm[s], st, end[j] + 1, e, False)
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

    @njit(inline="always")
    def _g(th, tf, hr, fr, c0, split, t):
        """Prefix value at stitched index t: history table up to the split, then P_hist(split) + P_fc(t - split).

        th columns start at history day c0, so th[:, i] sums days [c0, c0 + i). Only differences of two values are
        meaningful, which is all callers use.
        """
        if t <= split:
            return th[hr, max(t - c0, 0)]
        return th[hr, max(split - c0, 0)] + tf[fr, t - split]

    @njit(inline="always")
    def _flags_s(hpm, fpm, hc, fr, c0, split, a, b, data_end, keep):
        """Missing/horizon flag bits of the stitched interval [a, b); zero means it is observable."""
        lo = min(max(a, 0), data_end)
        hi = max(min(b, data_end), lo)
        flags = 0
        if a < 0 or (not keep and _g(hpm, fpm, hc, fr, c0, split, hi) - _g(hpm, fpm, hc, fr, c0, split, lo) > 0):
            flags |= 1
        if b > data_end:
            flags |= 2
        return flags

    @njit(parallel=True, cache=True)
    def _build_hist_tables(x, missing, px, pm):
        """Prefix tables (leading zero) of rain (int64) and missing days (int32) over the history columns."""
        for r in prange(x.shape[0]):
            ax = np.int64(0)
            am = 0
            px[r, 0] = 0
            pm[r, 0] = 0
            for t in range(x.shape[1]):
                if missing[r, t]:
                    am += 1
                else:
                    ax += x[r, t]
                px[r, t + 1] = ax
                pm[r, t + 1] = am

    @njit(parallel=True, cache=True)
    def _build_hist_wet(x, missing, wet_op, vcell, vamt, pw):
        """Prefix table of wet days for each (cell, wet amount) variant row."""
        for v in prange(len(vcell)):
            c = vcell[v]
            aw = 0
            pw[v, 0] = 0
            for t in range(x.shape[1]):
                if not missing[c, t] and _cmp(wet_op, np.int64(x[c, t]), vamt[v]):
                    aw += 1
                pw[v, t + 1] = aw

    @njit(inline="always")
    def _hist_dry_day(hpm, hdry, use_pd, c, v, t):
        """Whether history column t is a dry day: from the dry table, else neither missing nor wet (hdry counts wet days)."""
        if use_pd:
            return hdry[v, t + 1] - hdry[v, t] == 1
        return hpm[c, t + 1] - hpm[c, t] == 0 and hdry[v, t + 1] - hdry[v, t] == 0

    @njit(parallel=True, cache=True)
    def _build_hist_spells(kind, length, op, vcell, vamt, hpx, hpm, hdry, use_pd, begin, sp):
        """Cumulative count of history-only spell starts for each variant row.

        Dry runs read hdry: the dry-day table of (cell, dry amount) rows when use_pd, else the wet-day table
        of the wet variants. With begin, a run only counts when the column before it is not dry."""
        n_cols = hpx.shape[1] - 1
        for v in prange(len(vcell)):
            c = vcell[v]
            sp[v, 0] = 0
            acc = 0
            for u in range(n_cols):
                holds = False
                if u + length <= n_cols:
                    if hpm[c, u + length] - hpm[c, u] == 0:
                        if kind == 0:
                            holds = hdry[v, u + length] - hdry[v, u] == (length if use_pd else 0)
                            if holds and begin and u > 0 and _hist_dry_day(hpm, hdry, use_pd, c, v, u - 1):
                                holds = False
                        else:
                            holds = _cmp(op, hpx[c, u + length] - hpx[c, u], vamt[v])
                if holds:
                    acc += 1
                sp[v, u + 1] = acc

    @njit(inline="always")
    def _fill_seam(seam, kind, length, op, amount, begin, use_pd, hpx, hpm, hdry, fpx, fpm, fdry, hc, hd, fr, c0, split,
                   n_fc):
        """Cumulative indicator of the spells starting in [split - length + 1, split - 1] (to split with begin) that hold.

        Counts come from the stitched history and forecast tables; hdry/fdry count wet days, or dry days with use_pd
        (hd is the job's row of hdry). A begin run also needs the history day before it not to be dry."""
        base = split - length + 1
        seam[0] = 0
        for i in range(length - 1 + begin):
            u = base + i
            holds = False
            if u >= c0 and u + length <= split + n_fc:
                k = u + length - split
                if (hpm[hc, max(split - c0, 0)] - hpm[hc, u - c0]) + fpm[fr, k] == 0:
                    if kind == 0:
                        holds = (hdry[hd, max(split - c0, 0)] - hdry[hd, u - c0]) + fdry[fr, k] == (length if use_pd else 0)
                        if holds and begin and u - 1 >= c0 and _hist_dry_day(hpm, hdry, use_pd, hc, hd, u - 1 - c0):
                            holds = False
                    else:
                        total = (hpx[hc, max(split - c0, 0)] - hpx[hc, u - c0]) + fpx[fr, k]
                        holds = _cmp(op, total, amount)
            seam[i + 1] = seam[i] + (1 if holds else 0)

    @njit(inline="always")
    def _spell_count(hsp, vrow, fsp, fr, c0, split, length, seam, begin, lo, hi):
        """Spells holding at starts u in [lo, hi): history-only starts, the seam, and forecast-only starts added.

        The seam covers starts in [split - length + 1, split) (through split with begin)."""
        base = split - length + 1
        total = 0
        x = min(hi, base)
        if x > lo:
            total += hsp[vrow, max(x - c0, 0)] - hsp[vrow, max(lo - c0, 0)]
        x0 = max(lo, base)
        x1 = min(hi, split + begin)
        if x1 > x0:
            total += seam[x1 - base] - seam[x0 - base]
        y0 = max(lo, split + begin)
        if hi > y0:
            total += fsp[fr, hi - split] - fsp[fr, y0 - split]
        return total

    @njit(inline="always")
    def _is_wet(x, missing, wet_op, wet_amount):
        """Whether a day has data and passes the wet-day comparison."""
        return (not missing) and _cmp(wet_op, np.int64(x), wet_amount)

    @njit(parallel=True, cache=True)
    def _first_passage_s(hx, hmiss, hpx, hpm, hpw, hpd, hsp, fx, fmiss, fpx, fpm, fpw, fpd, fsp, amounts, c0, wet_op,
                         wet_key, use_pd,
                         c_kind, c_first, c_last, c_since, c_op, c_key, c_count,
                         v_kind, v_length, v_place, v_first, v_last, v_op, v_key, v_off,
                         report, r_first, r_last, r_since, horizon, keep,
                         series, cell, split, wv, dv, vv, start, end, data_end, origin, day, status):
        """Stitched first passage: the same scan as _first_passage, with intervals split at each job's split.

        hpd/fpd are the dry-day tables (the wet-day ones when use_pd is false), dv the job's row of hpd."""
        n_fc = fpm.shape[1] - 1
        lmax = 1
        for k in range(len(v_length)):
            lmax = max(lmax, v_length[k])
        for j in prange(len(series)):
            s = series[j]
            hc = cell[j]
            hw = wv[j]
            hd = dv[j]
            sp = split[j]
            st = start[j]
            org = origin[j]
            n_time = sp + n_fc
            e = min(data_end[j], n_time)
            day[j] = -1
            status[j] = _NOT_FOUND
            seam = np.zeros((len(v_kind), lmax + 1), dtype=np.int32)
            seam_ready = np.zeros(len(v_kind), dtype=np.bool_)
            for d in range(st, end[j] + 1):
                if keep and d < e and _g(hpm, fpm, hc, s, c0, sp, d + 1) - _g(hpm, fpm, hc, s, c0, sp, d) > 0:
                    continue
                dead = False
                unknown = False
                uflags = 0
                for c in range(len(c_kind)):
                    a = org if c_since[c] else d + c_first[c]
                    b = d + c_last[c]
                    fl = _flags_s(hpm, fpm, hc, s, c0, sp, a, b, e, keep)
                    if fl != 0:
                        unknown = True
                        uflags |= fl
                        continue
                    if c_kind[c] == 0:
                        total = (_g(hpx, fpx, hc, s, c0, sp, b) - _g(hpx, fpx, hc, s, c0, sp, a))
                        ok = _cmp(c_op[c], total, amounts[c_key[c], s])
                    else:
                        count = (_g(hpw, fpw, hw, s, c0, sp, b) - _g(hpw, fpw, hw, s, c0, sp, a))
                        ok = _cmp(c_op[c], count, c_count[c])
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
                        fl = _flags_s(hpm, fpm, hc, s, c0, sp, a, b, e, keep)
                    else:
                        if b <= a:
                            continue
                        u_last = b - 1
                        fl = _flags_s(hpm, fpm, hc, s, c0, sp, a, b + length - 1, e, keep)
                    u_last = min(u_last, e - length)
                    lo = min(max(a, 0), n_time)
                    hi = max(min(max(u_last + 1, 0), n_time), lo)
                    begin = 1 if v_place[k] == 2 else 0
                    if not seam_ready[k] and min(hi, sp + begin) > max(lo, sp - length + 1):
                        _fill_seam(seam[k], v_kind[k], length, v_op[k], amounts[v_key[k], s], begin, use_pd, hpx, hpm,
                                   hpd, fpx, fpm, fpd, hc, hd, s, c0, sp, n_fc)
                        seam_ready[k] = True
                    if _spell_count(hsp, v_off[k] + vv[k, j], fsp[k], s, c0, sp, length, seam[k], begin, lo, hi) > 0 \
                            or (horizon == 2 and b > e):
                        dead = True
                        break
                    if fl != 0 and not (horizon > 0 and fl == 2):
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
                    a = org if r_since else d + r_first
                    b = d + r_last
                    wet_amount = amounts[wet_key, s]
                    for t in range(max(a, 0), min(b, e)):
                        if t < sp:
                            wet = _is_wet(hx[hc, t - c0], hmiss[hc, t - c0], wet_op, wet_amount)
                        else:
                            wet = _is_wet(fx[s, t - sp], fmiss[s, t - sp], wet_op, wet_amount)
                        if wet:
                            day[j] = t
                            break
                break

    @njit(parallel=True, cache=True)
    def _curve_s(hpx, hpm, fpx, fpm, amounts, key, after, c0, series, cell, split, start, end, data_end, day, status):
        """Stitched cumulative-anomaly minimum for every job."""
        n_fc = fpm.shape[1] - 1
        for j in prange(len(series)):
            s = series[j]
            hc = cell[j]
            sp = split[j]
            st = start[j]
            e = min(data_end[j], sp + n_fc)
            day[j] = -1
            status[j] = _NOT_FOUND
            if end[j] < st:
                continue
            fl = _flags_s(hpm, fpm, hc, s, c0, sp, st, end[j] + 1, e, False)
            if fl != 0:
                status[j] = _CENSORED if fl == 2 else _INSUFFICIENT
                continue
            baseline = amounts[key, s]
            origin = _g(hpx, fpx, hc, s, c0, sp, st)
            best = np.int64(0)
            arg = st
            for t in range(st, end[j] + 1):
                value = 100 * (_g(hpx, fpx, hc, s, c0, sp, t + 1) - origin) - baseline * (t - st + 1)
                if t == st or value < best:
                    best = value
                    arg = t
            if arg == end[j]:
                continue
            status[j] = _ONSET
            day[j] = arg + 1 if after else arg


def _stack_amounts(plan: P.Plan, amounts: dict[str, np.ndarray], exact: bool) -> tuple[np.ndarray, dict[str, int]]:
    """The amounts the plan reads stacked into one (keys, series) array (int64, or float64 mm when exact) and each key's row."""
    keys = [plan.wet.key]
    if plan.dry is not None:
        keys.append(plan.dry.key)
    for item in (*plan.conditions, *plan.vetoes):
        if item.cmp is not None:
            keys.append(item.cmp.key)
    if plan.curve_key is not None:
        keys.append(plan.curve_key)
    keys = list(dict.fromkeys(keys))
    stack = np.stack([np.asarray(amounts[k], dtype=np.float64 if exact else np.int64) for k in keys])
    return stack, {k: i for i, k in enumerate(keys)}


def _dry_args(plan: P.Plan, stack: np.ndarray, index: dict[str, int]) -> tuple[bool, int, np.ndarray]:
    """(use_pd, dry op, dry amounts) for the kernels; without a dry_day the table is not built and the op is unused."""
    if plan.dry is None:
        return False, 0, stack[0]
    return True, plan.dry.op, stack[index[plan.dry.key]]


def detect(plan: P.Plan, tenths: np.ndarray, missing: np.ndarray, amounts: dict[str, np.ndarray],
           jobs: Jobs, *, trace: bool = False, amount_windows=None) -> Detection:
    """Run every job and return the onset day index (-1 if none) and Status per job, bit-identical to the reference backend.

    Amounts may be (S, W) with amount_windows the W window start leads (backends/windowed.py).
    """
    if trace:
        raise NotImplementedError("use the reference backend for traces")
    if numba is None:
        raise ImportError("the numba backend needs numba")
    amounts, starts = windowed.prepare(amounts, amount_windows)
    if starts is not None:
        return windowed.run(plan, amounts, starts, jobs, lambda j, a: detect(plan, tenths, missing, a, j))
    series = np.ascontiguousarray(jobs.series, dtype=np.int64)
    start = np.ascontiguousarray(jobs.start, dtype=np.int64)
    end = np.ascontiguousarray(jobs.end, dtype=np.int64)
    data_end = np.ascontiguousarray(jobs.data_end, dtype=np.int64)
    origin = start if jobs.origin is None else np.ascontiguousarray(jobs.origin, dtype=np.int64)
    day = np.full(len(series), -1, dtype=np.int64)
    status = np.full(len(series), _NOT_FOUND, dtype=np.int8)
    if len(series) == 0:
        return Detection(day, status, None)
    exact = np.asarray(tenths).dtype.kind == "f"
    if exact and plan.family == P.CURVE:
        raise NotImplementedError("curve rules need rain_resolution 'tenths'")
    tenths = np.ascontiguousarray(tenths, dtype=np.float64 if exact else np.int32)
    missing = np.ascontiguousarray(missing, dtype=np.bool_)
    n_series, n_time = tenths.shape

    stack, index = _stack_amounts(plan, amounts, exact)

    px = np.empty((n_series, n_time + 1), dtype=tenths.dtype if exact else np.int64)
    pm = np.empty((n_series, n_time + 1), dtype=np.int32)
    pw = np.empty((n_series, n_time + 1), dtype=np.int32)
    use_pd, dry_op, dry_amount = _dry_args(plan, stack, index)
    pd = np.empty((n_series, n_time + 1), dtype=np.int32) if use_pd else pw
    _build_tables(tenths, missing, plan.wet.op, stack[index[plan.wet.key]], use_pd, dry_op, dry_amount, px, pm, pw, pd)

    if plan.family == P.CURVE:
        _curve(px, pm, stack, index[plan.curve_key], plan.curve_after_min, series, start, end, data_end,
               day, status)
        return Detection(day, status, None)

    sp = np.zeros((len(plan.vetoes), n_series, n_time + 1), dtype=np.int32)
    for k, veto in enumerate(plan.vetoes):
        op, amount = (veto.cmp.op, stack[index[veto.cmp.key]]) if veto.cmp is not None else (0, stack[0])
        _build_spells(px, pm, pw, pd, use_pd, veto.kind, veto.length, op, amount, veto.placement == P.BEGINS, sp[k])

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
                   report, r_first, r_last, r_since, HORIZONS[plan.horizon], plan.keep_missing,
                   series, start, end, data_end, origin, day, status)
    return Detection(day, status, None)


def _variants(cell: np.ndarray, amount: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique (cell, amount) pairs as (cells, amounts), sorted by cell then amount, and each job's index into them.

    Integer pairs are packed into one int64 key (cell and amount offset to start at 0, amount in the low digits),
    which sorts exactly as the pairs do and is far cheaper to unique than np.unique(axis=0); pairs whose key would
    not fit in an int64 take that slower route. The result is the same either way.
    """
    if cell.size and cell.dtype == np.int64 and amount.dtype == np.int64:
        c0, a0 = int(cell.min()), int(amount.min())
        span = int(amount.max()) - a0 + 1
        if (int(cell.max()) - c0 + 1) * span < (1 << 62):
            key = (cell - c0) * span + (amount - a0)
            unique, inverse = np.unique(key, return_inverse=True)
            return (np.ascontiguousarray(unique // span + c0), np.ascontiguousarray(unique % span + a0),
                    np.ascontiguousarray(inverse.reshape(-1), dtype=np.int64))
    pairs = np.stack([cell, amount], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return (np.ascontiguousarray(unique[:, 0]), np.ascontiguousarray(unique[:, 1]),
            np.ascontiguousarray(inverse.reshape(-1), dtype=np.int64))


def _first_column(plan: P.Plan, start: np.ndarray, origin: np.ndarray) -> int:
    """Earliest history day any candidate's spans can read (never negative)."""
    offsets = [0]
    for cond in plan.conditions:
        offsets.append(0 if cond.since_start else cond.first)
    offsets.extend(v.first - (v.placement == P.BEGINS) for v in plan.vetoes)
    if plan.report_first_wet and plan.report_span is not None:
        offsets.append(0 if plan.report_span[2] else plan.report_span[0])
    return max(0, min(int(start.min()) + min(offsets), int(origin.min())))


def detect_stitched(plan: P.Plan, hist_tenths: np.ndarray, hist_missing: np.ndarray, fc_tenths: np.ndarray,
                    fc_missing: np.ndarray, amounts: dict[str, np.ndarray], jobs: StitchedJobs, *,
                    trace: bool = False, amount_windows=None) -> Detection:
    """Run stitched jobs (history before each split, then a forecast row) without copying history into forecasts.

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
    if numba is None:
        raise ImportError("the numba backend needs numba")
    P.reject_exact_stitched(hist_tenths, fc_tenths)
    amounts, starts = windowed.prepare(amounts, amount_windows)
    if starts is not None:
        return windowed.run(plan, amounts, starts, jobs, lambda j, a: detect_stitched(
            plan, hist_tenths, hist_missing, fc_tenths, fc_missing, a, j))
    n_jobs = len(jobs.series)
    day = np.full(n_jobs, -1, dtype=np.int64)
    status = np.full(n_jobs, _NOT_FOUND, dtype=np.int8)
    series = np.ascontiguousarray(jobs.series, dtype=np.int64)
    cell = np.ascontiguousarray(jobs.cell, dtype=np.int64)
    split = np.ascontiguousarray(jobs.split, dtype=np.int64)
    start = np.ascontiguousarray(jobs.start, dtype=np.int64)
    end = np.ascontiguousarray(jobs.end, dtype=np.int64)
    data_end = np.ascontiguousarray(jobs.data_end, dtype=np.int64)
    origin = start if jobs.origin is None else np.ascontiguousarray(jobs.origin, dtype=np.int64)
    active = end >= start
    if not active.any():
        return Detection(day, status, None)
    if split[active].min() < 0 or split[active].max() > hist_tenths.shape[1]:
        raise ValueError("a job's split lies outside the history")
    fc_tenths = np.ascontiguousarray(fc_tenths, dtype=np.int32)
    fc_missing = np.ascontiguousarray(fc_missing, dtype=np.bool_)
    n_fc_series, n_fc = fc_tenths.shape

    stack, index = _stack_amounts(plan, amounts, False)

    c1 = int(split[active].max())
    c0 = min(_first_column(plan, start[active], origin[active]), c1)
    cells, cell_idx = np.unique(cell, return_inverse=True)
    cell_idx = np.ascontiguousarray(cell_idx.reshape(-1), dtype=np.int64)
    hx = np.ascontiguousarray(hist_tenths[cells, c0:c1], dtype=np.int32)
    hmiss = np.ascontiguousarray(hist_missing[cells, c0:c1], dtype=np.bool_)
    width = c1 - c0
    hpx = np.empty((len(cells), width + 1), dtype=np.int64)
    hpm = np.empty((len(cells), width + 1), dtype=np.int32)
    _build_hist_tables(hx, hmiss, hpx, hpm)
    wet_amount = stack[index[plan.wet.key]]
    wcell, wamt, wv = _variants(cell_idx, wet_amount[series])
    hpw = np.empty((len(wcell), width + 1), dtype=np.int32)
    _build_hist_wet(hx, hmiss, plan.wet.op, wcell, wamt, hpw)
    use_pd, dry_op, dry_amount = _dry_args(plan, stack, index)
    dcell, damt, dv, hpd = wcell, wamt, wv, hpw
    if use_pd:
        dcell, damt, dv = _variants(cell_idx, dry_amount[series])
        hpd = np.empty((len(dcell), width + 1), dtype=np.int32)
        _build_hist_wet(hx, hmiss, dry_op, dcell, damt, hpd)

    fpx = np.empty((n_fc_series, n_fc + 1), dtype=np.int64)
    fpm = np.empty((n_fc_series, n_fc + 1), dtype=np.int32)
    fpw = np.empty((n_fc_series, n_fc + 1), dtype=np.int32)
    fpd = np.empty((n_fc_series, n_fc + 1), dtype=np.int32) if use_pd else fpw
    _build_tables(fc_tenths, fc_missing, plan.wet.op, wet_amount, use_pd, dry_op, dry_amount, fpx, fpm, fpw, fpd)

    if plan.family == P.CURVE:
        _curve_s(hpx, hpm, fpx, fpm, stack, index[plan.curve_key], plan.curve_after_min, c0, series, cell_idx, split,
                 start, end, data_end, day, status)
        return Detection(day, status, None)

    n_veto = len(plan.vetoes)
    hsp_parts, offsets, vv = [], [], np.zeros((n_veto, n_jobs), dtype=np.int64)
    fsp = np.zeros((n_veto, n_fc_series, n_fc + 1), dtype=np.int32)
    v_op, v_key = [], []
    total_rows = 0
    for k, veto in enumerate(plan.vetoes):
        if veto.cmp is not None:
            op, key = veto.cmp.op, index[veto.cmp.key]
            vcell, vamt, vv[k] = _variants(cell_idx, stack[key][series])
        else:
            op, key = 0, 0
            vcell, vamt, vv[k] = dcell, damt, dv
        v_op.append(op)
        v_key.append(key)
        sp_k = np.empty((len(vcell), width + 1), dtype=np.int32)
        begin = veto.placement == P.BEGINS
        _build_hist_spells(veto.kind, veto.length, op, vcell, vamt, hpx, hpm, hpd, use_pd, begin, sp_k)
        hsp_parts.append(sp_k)
        offsets.append(total_rows)
        total_rows += len(vcell)
        _build_spells(fpx, fpm, fpw, fpd, use_pd, veto.kind, veto.length, op, stack[key], begin, fsp[k])
    hsp = np.concatenate(hsp_parts) if hsp_parts else np.zeros((0, width + 1), dtype=np.int32)

    def ints(values):
        """A tuple of ints as a small int64 array."""
        return np.array(values, dtype=np.int64)

    conds = plan.conditions
    c_op = [c.cmp.op if c.kind == P.TOTAL else c.op for c in conds]
    c_key = [index[c.cmp.key] if c.kind == P.TOTAL else 0 for c in conds]
    report = plan.report_first_wet and plan.report_span is not None
    r_first, r_last, r_since = plan.report_span if report else (0, 0, False)
    _first_passage_s(hx, hmiss, hpx, hpm, hpw, hpd, hsp, fc_tenths, fc_missing, fpx, fpm, fpw, fpd, fsp, stack, c0,
                     plan.wet.op, index[plan.wet.key], use_pd,
                     ints([c.kind for c in conds]), ints([c.first for c in conds]), ints([c.last for c in conds]),
                     np.array([c.since_start for c in conds], dtype=np.bool_), ints(c_op), ints(c_key),
                     ints([c.count for c in conds]),
                     ints([v.kind for v in plan.vetoes]), ints([v.length for v in plan.vetoes]),
                     ints([v.placement for v in plan.vetoes]), ints([v.first for v in plan.vetoes]),
                     ints([v.last for v in plan.vetoes]), ints(v_op), ints(v_key), ints(offsets),
                     report, r_first, r_last, r_since, HORIZONS[plan.horizon], plan.keep_missing,
                     series, cell_idx, split, wv, dv, vv, start, end, data_end, origin, day, status)
    return Detection(day, status, None)
