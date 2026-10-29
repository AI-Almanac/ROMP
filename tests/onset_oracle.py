"""Brute-force onset evaluator written straight from docs/onset_spec.md section 4.

It works on a model.Definition directly (not on a compiled Plan), loops over
candidates one at a time and compares with float thresholds, so it shares no
code or shortcuts with the backends it checks. Slow by design.
"""
from __future__ import annotations

import numpy as np

from momp.onset import model as m

MISSING, HORIZON = 1, 2
TRUE, FALSE, UNKNOWN = 1, 0, -1


def _q(threshold_mm: float) -> float:
    """Ten times the threshold, rounded to 6 decimals to absorb float noise, per the spec."""
    return round(10 * threshold_mm, 6)


def _compare(value, op: str, rhs) -> bool:
    """Apply a comparator code to value and rhs."""
    return {"ge": value >= rhs, "gt": value > rhs, "le": value <= rhs, "lt": value < rhs}[op]


class Series:
    """One rainfall series with its thresholds, evaluated per the spec."""

    def __init__(self, defn: m.Definition, tenths, missing, thresholds: dict[str, float]):
        """Store the series and the wet-day rule."""
        self.defn = defn
        self.x = [int(v) for v in tenths]
        self.v = [not bool(b) for b in missing]
        self.thresholds = thresholds
        wet = defn.wet_day
        self.wet = [self.v[t] and _compare(self.x[t], wet.op, wet.amount.tenths) for t in range(len(self.x))]

    def amount_rhs(self, amount: m.Amount) -> float:
        """Right-hand side, in tenths, of a comparison with this amount."""
        if amount.tenths is not None:
            return float(amount.tenths)
        return _q(self.thresholds[amount.threshold])

    def flags(self, a: int, b: int, data_end: int) -> int:
        """Why the half-open interval [a, b) is not observable (0 if it is)."""
        f = 0
        if b > data_end:
            f |= HORIZON
        if a < 0 or any(not self.v[t] for t in range(max(a, 0), min(b, data_end))):
            f |= MISSING
        return f

    def interval(self, span: m.Span, d: int, start: int) -> tuple[int, int]:
        """Absolute [a, b) of a span at candidate d."""
        if span.since_search_start:
            return start, d + span.last
        return d + span.first - 1, d + span.last

    def condition(self, cond, d: int, start: int, data_end: int) -> tuple[int, int]:
        """Evaluate one requirement at candidate d: (value, flags)."""
        a, b = self.interval(cond.span, d, start)
        f = self.flags(a, b, data_end)
        if f:
            return UNKNOWN, f
        if isinstance(cond, m.Total):
            total = sum(self.x[a:b])
            ok = _compare(total, cond.comparison.op, self.amount_rhs(cond.comparison.amount))
        else:
            ok = _compare(sum(self.wet[a:b]), cond.op, cond.count)
        return (TRUE if ok else FALSE), 0

    def spell_holds(self, spell, u: int) -> bool:
        """Whether the spell starting at u holds (its days are known to be observable)."""
        if isinstance(spell, m.DryRun):
            return all(self.v[t] and not self.wet[t] for t in range(u, u + spell.length))
        total = sum(self.x[u:u + spell.days])
        return _compare(total, spell.comparison.op, self.amount_rhs(spell.comparison.amount))

    def veto(self, veto: m.Veto, d: int, start: int, data_end: int) -> tuple[int, int]:
        """Evaluate one veto at candidate d: (value, flags)."""
        a, b = self.interval(veto.span, d, start)
        length = veto.spell.length if isinstance(veto.spell, m.DryRun) else veto.spell.days
        if veto.placement == "inside":
            if b - a < length:
                return FALSE, 0
            starts, judged = range(a, b - length + 1), (a, b)
        else:
            if b - a < 1:
                return FALSE, 0
            starts, judged = range(a, b), (a, b + length - 1)
        for u in starts:
            if not self.flags(u, u + length, data_end) and self.spell_holds(veto.spell, u):
                return TRUE, 0
        f = self.flags(*judged, data_end)
        return (UNKNOWN, f) if f else (FALSE, 0)

    def candidate(self, d: int, start: int, data_end: int) -> tuple[int, int]:
        """Kleene AND of requirements AND NOT (OR of vetoes) at candidate d."""
        rule = self.defn.rule
        req = [self.condition(c, d, start, data_end) for c in rule.require]
        vet = [self.veto(v, d, start, data_end) for v in rule.reject_if]
        if any(val == TRUE for val, _ in vet):
            vetoed = (TRUE, 0)
        elif any(val == UNKNOWN for val, _ in vet):
            vetoed = (UNKNOWN, _union(f for val, f in vet if val == UNKNOWN))
        else:
            vetoed = (FALSE, 0)
        not_vetoed = {TRUE: (FALSE, 0), FALSE: (TRUE, 0), UNKNOWN: vetoed}[vetoed[0]]
        parts = req + [not_vetoed]
        if any(val == FALSE for val, _ in parts):
            return FALSE, 0
        if any(val == UNKNOWN for val, _ in parts):
            return UNKNOWN, _union(f for val, f in parts if val == UNKNOWN)
        return TRUE, 0

    def report(self, d: int, data_end: int) -> int:
        """The reported onset day for an accepted candidate d."""
        rule = self.defn.rule
        if rule.report == "candidate":
            return d
        a, b = rule.report_span.first + d - 1, rule.report_span.last + d
        for t in range(max(a, 0), min(b, data_end)):
            if self.wet[t]:
                return t
        return d

    def first_passage(self, start: int, end: int, data_end: int) -> tuple[int, m.Status]:
        """Scan candidates in order and classify the first one that is not False."""
        for d in range(start, end + 1):
            val, f = self.candidate(d, start, data_end)
            if val == TRUE:
                return self.report(d, data_end), m.Status.ONSET
            if val == UNKNOWN:
                return -1, (m.Status.CENSORED if f == HORIZON else m.Status.INSUFFICIENT_DATA)
        return -1, m.Status.NOT_FOUND

    def curve(self, start: int, end: int, data_end: int) -> tuple[int, m.Status]:
        """Cumulative-anomaly onset over [start, end]."""
        f = self.flags(start, end + 1, data_end)
        if f:
            return -1, (m.Status.CENSORED if f == HORIZON else m.Status.INSUFFICIENT_DATA)
        baseline = int(np.rint(self.thresholds[self.defn.rule.baseline] * 1000))
        running, best, argbest = 0, None, None
        for t in range(start, end + 1):
            running += 100 * self.x[t] - baseline
            if best is None or running < best:
                best, argbest = running, t
        if argbest == end:
            return -1, m.Status.NOT_FOUND
        day = argbest + 1 if self.defn.rule.report == "day_after_minimum" else argbest
        return day, m.Status.ONSET


def _union(flag_iter) -> int:
    """Bitwise OR of flags."""
    out = 0
    for f in flag_iter:
        out |= f
    return out


def used_thresholds(defn: m.Definition) -> set[str]:
    """Names of every threshold the rule reads."""
    rule = defn.rule
    if isinstance(rule, m.Curve):
        return {rule.baseline}
    names = set()
    for c in rule.require:
        if isinstance(c, m.Total) and c.comparison.amount.threshold:
            names.add(c.comparison.amount.threshold)
    for v in rule.reject_if:
        if isinstance(v.spell, m.DryWindow) and v.spell.comparison.amount.threshold:
            names.add(v.spell.comparison.amount.threshold)
    return names


def oracle_detect(defn: m.Definition, tenths, missing, thresholds: dict[str, float],
                  start: int, end: int, data_end: int) -> tuple[int, m.Status]:
    """Onset day index (-1 if none) and Status for one job, straight from the spec."""
    if any(not np.isfinite(thresholds[name]) for name in used_thresholds(defn)):
        return -1, m.Status.INSUFFICIENT_DATA
    series = Series(defn, tenths, missing, thresholds)
    if start > end:
        return -1, m.Status.NOT_FOUND
    if isinstance(defn.rule, m.Curve):
        return series.curve(start, end, data_end)
    return series.first_passage(start, end, data_end)
