"""Reference onset backend: per-atom semantics, a brute-force cross-check, and property tests on random data."""
from fractions import Fraction

import numpy as np
import pytest

from momp.onset import model as m
from momp.onset import plan as plan_mod
from momp.onset.backends import reference

ONSET, NOT_FOUND, CENSORED, INSUFFICIENT = (int(s) for s in (
    m.Status.ONSET, m.Status.NOT_FOUND, m.Status.CENSORED, m.Status.INSUFFICIENT_DATA))
WET_1MM = m.Comparison("ge", m.Amount(tenths=10))
SEASON = (m.Season("s", m.DayRef(month=1, day=1), m.DayRef(month=12, day=31)),)


def fixed(op, tenths):
    """A comparison against a fixed amount in tenths of a mm."""
    return m.Comparison(op, m.Amount(tenths=tenths))


def named(op, name):
    """A comparison against a named threshold."""
    return m.Comparison(op, m.Amount(threshold=name))


def total(first, last, cmp, since=False):
    """A Total requirement over days first..last."""
    return m.Total(m.Span(first, last, since), cmp)


def definition(rule, wet=WET_1MM):
    """A minimal Definition around a rule."""
    return m.Definition("t", 1, "t", "t", "t", "draft", "", wet, SEASON, rule)


def run(rule, x, *, miss=None, start=0, end=None, data_end=None, wet=WET_1MM, thresholds=None, trace=False):
    """Run one series (rain in tenths) through the reference backend; return (day, status) or the Detection."""
    x = np.asarray(x, dtype=np.int32)[None, :]
    miss = np.zeros(x.shape, dtype=bool) if miss is None else np.asarray(miss, dtype=bool)[None, :]
    n = x.shape[1]
    end = n - 1 if end is None else end
    data_end = n if data_end is None else data_end
    plan = plan_mod.compile(definition(rule, wet))
    thr = {k: np.array([v], dtype=float) for k, v in (thresholds or {}).items()}
    amounts, invalid = plan_mod.resolve_amounts(plan, thr, n_series=1)
    jobs = m.Jobs(np.array([0]), np.array([start]), np.array([end]), np.array([data_end]))
    det = reference.detect(plan, x, miss, amounts, jobs, trace=trace)
    return det if trace else (int(det.day[0]), int(det.status[0]))


def passage(require=(), reject_if=(), **kw):
    """A FirstPassage rule."""
    return m.FirstPassage(tuple(require), tuple(reject_if), **kw)


# ---------------------------------------------------------------- plan and amounts

def test_resolve_amounts_rounds_by_comparator():
    """Thresholds convert with q = round(10 T, 6), then ceil for ge/lt and floor for gt/le."""
    rule = passage([total(1, 1, named(op, "t")) for op in ("ge", "gt", "lt", "le")])
    plan = plan_mod.compile(definition(rule))
    amounts, invalid = plan_mod.resolve_amounts(plan, {"t": np.array([2.04, 2.0, 0.1 + 0.2])})
    assert amounts["t:ge"].tolist() == [21, 20, 3]
    assert amounts["t:gt"].tolist() == [20, 20, 3]
    assert amounts["t:lt"].tolist() == [21, 20, 3]
    assert amounts["t:le"].tolist() == [20, 20, 3]
    assert amounts["fixed:10"].tolist() == [10, 10, 10]
    assert not invalid.any()


def test_resolve_amounts_nan_and_baseline():
    """NaN thresholds are flagged; baselines are thousandths of a mm rounded half to even."""
    plan = plan_mod.compile(definition(m.Curve("base")))
    amounts, invalid = plan_mod.resolve_amounts(plan, {"base": np.array([0.0125, 0.0135, np.nan])})
    assert amounts["base:baseline"].tolist() == [12, 14, 0]
    assert invalid.tolist() == [False, False, True]


def test_compile_offsets():
    """Spans become (first - 1, last, since_start) offsets and every amount has a key."""
    rule = passage([total(2, 4, fixed("ge", 50)), total(1, 3, fixed("lt", 90), since=True),
                    m.WetDays(m.Span(1, 3), "ge", 2)],
                   [m.Veto(m.DryRun(5), m.Span(2, 22), "inside"),
                    m.Veto(m.DryWindow(7, fixed("lt", 50)), m.Span(4, 33), "starts_inside")],
                   report="first_wet_day", report_span=m.Span(1, 3))
    plan = plan_mod.compile(definition(rule))
    assert [(c.first, c.last, c.since_start) for c in plan.conditions] == [(1, 4, False), (0, 3, True), (0, 3, False)]
    assert [(v.first, v.last, v.placement) for v in plan.vetoes] == [(1, 22, plan_mod.INSIDE), (3, 33, plan_mod.STARTS_INSIDE)]
    assert plan.report_span == (0, 3, False)
    assert {s.key for s in plan.amount_specs} == {"fixed:10", "fixed:50", "fixed:90"}


# ---------------------------------------------------------------- conditions

@pytest.mark.parametrize("op,expected", [("ge", ONSET), ("gt", NOT_FOUND), ("le", ONSET), ("lt", NOT_FOUND)])
def test_total_comparators(op, expected):
    """A 30-tenth total against 30 tenths: only the non-strict comparators hold."""
    day, status = run(passage([total(1, 3, fixed(op, 30))]), [10, 10, 10, 0, 0], end=0)
    assert status == expected
    assert day == (0 if expected == ONSET else -1)


@pytest.mark.parametrize("op,count,expected", [("ge", 2, ONSET), ("gt", 2, NOT_FOUND), ("le", 2, ONSET),
                                               ("lt", 2, NOT_FOUND), ("lt", 3, ONSET), ("ge", 3, NOT_FOUND)])
def test_wet_days_comparators(op, count, expected):
    """WetDays counts days passing the wet-day rule."""
    assert run(passage([m.WetDays(m.Span(1, 4), op, count)]), [10, 0, 10, 5, 0], end=0)[1] == expected


def test_total_named_threshold_uses_comparator_rounding():
    """T = 2.04 mm with 'more than' means more than 20 tenths; with 'at least' it means at least 21."""
    x = [20, 0, 0]
    assert run(passage([total(1, 1, named("gt", "t"))]), x, end=0, thresholds={"t": 2.04})[1] == NOT_FOUND
    assert run(passage([total(1, 1, named("lt", "t"))]), x, end=0, thresholds={"t": 2.04})[1] == ONSET
    assert run(passage([total(1, 1, named("ge", "t"))]), [21, 0], end=0, thresholds={"t": 2.04})[1] == ONSET


def test_wet_day_threshold_from_named_amount():
    """The wet-day rule may use a per-cell amount; 'more than 3 mm' excludes exactly 3 mm."""
    rule = passage([m.WetDays(m.Span(1, 1), "ge", 1)])
    assert run(rule, [30, 40], end=1, wet=named("gt", "w"), thresholds={"w": 3.0}) == (1, ONSET)
    assert run(rule, [30, 30], end=1, wet=named("gt", "w"), thresholds={"w": 3.0}) == (-1, NOT_FOUND)


def test_season_to_date_span():
    """'season to date' runs from the search start through day 1."""
    rule = passage([total(1, 1, fixed("ge", 50), since=True)])
    assert run(rule, [10, 10, 10, 10, 10, 10], start=1) == (5, ONSET)


def test_first_candidate_wins_and_start_offsets_days():
    """The onset is the first candidate meeting everything, counted on the time axis."""
    rule = passage([m.WetDays(m.Span(1, 1), "ge", 1)])
    assert run(rule, [0, 0, 0, 20, 0, 20], start=1) == (3, ONSET)
    assert run(rule, [0, 0, 0, 20, 0, 20], start=4) == (5, ONSET)


def test_empty_window_is_not_found():
    """end before start has no candidates."""
    assert run(passage([]), [1, 2, 3], start=2, end=1) == (-1, NOT_FOUND)


def test_no_requirements_first_candidate():
    """A rule with nothing to meet fires on the first candidate."""
    assert run(passage([]), [0, 0, 0], start=1) == (1, ONSET)


# ---------------------------------------------------------------- vetoes

def test_dry_run_inside_needs_whole_run_in_span():
    """The run must lie wholly inside the span; one that crosses the span end does not count."""
    rule = passage([], [m.Veto(m.DryRun(3), m.Span(2, 7), "inside")])
    dry_in = [10, 10, 0, 0, 0, 10, 10, 10, 10, 10]
    assert run(rule, dry_in, end=0)[1] == NOT_FOUND
    crossing = [10, 10, 10, 10, 10, 10, 0, 0, 0, 10]
    assert run(rule, crossing, end=0)[1] == ONSET


def test_dry_run_starts_inside_allows_run_past_span_end():
    """A run beginning in the span but ending after it vetoes when placement is starts_inside."""
    x = [10, 10, 10, 10, 10, 10, 0, 0, 0, 10]
    rule = passage([], [m.Veto(m.DryRun(3), m.Span(2, 7), "starts_inside")])
    assert run(rule, x, end=0)[1] == NOT_FOUND
    late = [10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 10]
    assert run(rule, late, end=0)[1] == ONSET


def test_dry_run_counts_dry_days_not_amount():
    """Rain below the wet-day rule is still dry; a wet day breaks the run."""
    rule = passage([], [m.Veto(m.DryRun(3), m.Span(1, 4), "inside")])
    assert run(rule, [5, 9, 0, 0, 10, 10], end=0)[1] == NOT_FOUND
    assert run(rule, [5, 9, 10, 0, 10, 10], end=0)[1] == ONSET


def test_dry_window_is_amount_based():
    """A dry window can include wet days when the total is small enough."""
    rule = passage([], [m.Veto(m.DryWindow(4, fixed("lt", 25)), m.Span(1, 6), "inside")])
    assert run(rule, [10, 0, 0, 10, 50, 50, 50], end=0)[1] == NOT_FOUND
    assert run(rule, [10, 10, 0, 10, 50, 50, 50], end=0)[1] == ONSET


def test_dry_window_starting_within():
    """A window starting in the span may end after it."""
    x = [50, 50, 50, 0, 0, 0, 0, 0, 0]
    inside = passage([], [m.Veto(m.DryWindow(4, fixed("lt", 10)), m.Span(1, 4), "inside")])
    starts = passage([], [m.Veto(m.DryWindow(4, fixed("lt", 10)), m.Span(1, 4), "starts_inside")])
    assert run(inside, x, end=0)[1] == ONSET
    assert run(starts, x, end=0)[1] == NOT_FOUND


def test_veto_longer_than_inside_span_never_fires():
    """A spell longer than the span cannot fit, so the veto is False (spec: F if b - a < L)."""
    rule = passage([], [m.Veto(m.DryRun(5), m.Span(1, 3), "inside")])
    assert run(rule, [0] * 8, end=0) == (0, ONSET)


# ---------------------------------------------------------------- three-valued logic

def test_missing_in_requirement_is_insufficient_data():
    """A missing day in a Total span leaves the candidate undecided."""
    rule = passage([total(1, 3, fixed("ge", 10))])
    miss = [0, 1, 0, 0]
    assert run(rule, [10, 0, 10, 10], miss=miss, end=0) == (-1, INSUFFICIENT)


def test_horizon_is_censored_and_mixed_flags_are_insufficient():
    """Days past the data end give CENSORED; together with missing days they give INSUFFICIENT_DATA."""
    rule = passage([total(1, 5, fixed("ge", 10))])
    assert run(rule, [0, 0, 0], end=0) == (-1, CENSORED)
    assert run(rule, [0, 0, 0], miss=[0, 1, 0], end=0) == (-1, INSUFFICIENT)
    assert run(rule, [0, 0, 0, 0, 0, 0], miss=[0, 0, 0, 0, 0, 0], end=0, data_end=3) == (-1, CENSORED)


def test_data_end_hides_days_beyond_it():
    """Days at or after data_end are not used even if present in the array."""
    rule = passage([total(1, 3, fixed("ge", 30))])
    assert run(rule, [10, 10, 10], end=0, data_end=2) == (-1, CENSORED)


def test_false_dominates_unknown():
    """A False requirement settles the candidate even if another one is Unknown, in either order."""
    unknown = total(1, 3, fixed("ge", 10))
    false = total(1, 1, fixed("ge", 10))
    x, miss = [0, 0, 0, 0], [0, 0, 1, 0]
    assert run(passage([unknown]), x, miss=miss, end=0)[1] == INSUFFICIENT
    assert run(passage([unknown, false]), x, miss=miss, end=0)[1] == NOT_FOUND
    assert run(passage([false, unknown]), x, miss=miss, end=0)[1] == NOT_FOUND


def test_unknown_candidate_before_a_true_one_blocks_the_result():
    """An undecided candidate earlier than a True one hides the later onset."""
    rule = passage([total(1, 2, fixed("ge", 10))])
    x = [0, 0, 20, 20, 20]
    miss = [0, 1, 0, 0, 0]
    assert run(rule, x, miss=miss, end=3) == (-1, INSUFFICIENT)
    assert run(rule, x, miss=miss, start=2, end=3) == (2, ONSET)


def test_true_veto_dominates_unknown():
    """A confirmed spell vetoes the candidate even when its span runs past the data or the requirements are Unknown."""
    veto = m.Veto(m.DryRun(2), m.Span(1, 6), "inside")
    assert run(passage([], [veto]), [0, 0, 10], end=0)[1] == NOT_FOUND
    assert run(passage([], [veto]), [10, 0, 10], end=0)[1] == CENSORED
    assert run(passage([total(1, 5, fixed("ge", 10))], [veto]), [0, 0, 10], end=0)[1] == NOT_FOUND


def test_unconfirmed_veto_over_missing_is_unknown():
    """With no confirmed spell, a veto whose span has a missing day is Unknown."""
    rule = passage([], [m.Veto(m.DryRun(2), m.Span(1, 5), "inside")])
    assert run(rule, [10, 10, 0, 10, 10], miss=[0, 0, 1, 0, 0], end=0) == (-1, INSUFFICIENT)
    assert run(rule, [10, 10, 10, 10, 10], end=0) == (0, ONSET)


def test_starts_inside_observability_extends_past_span():
    """starts_inside judges observability on [a, b + L - 1), so the tail matters."""
    rule = passage([], [m.Veto(m.DryRun(3), m.Span(1, 2), "starts_inside")])
    assert run(rule, [10, 10, 10], end=0) == (-1, CENSORED)
    assert run(rule, [10, 10, 10, 10], end=0) == (0, ONSET)


def test_start_before_data_marks_missing():
    """A span that begins before day 0 cannot be observed (missing flag)."""
    rule = passage([total(0, 2, fixed("ge", 0))])
    assert run(rule, [1, 1, 1], end=0) == (-1, INSUFFICIENT)


# ---------------------------------------------------------------- report

def test_report_first_wet_day():
    """report first wet day moves the onset to the first wet day of the trigger window."""
    rule = passage([total(1, 3, fixed("ge", 100))], report="first_wet_day", report_span=m.Span(1, 3))
    assert run(rule, [0, 0, 100, 0, 0], end=2) == (2, ONSET)
    assert run(passage([total(1, 3, fixed("ge", 100))]), [0, 0, 100, 0, 0], end=2) == (0, ONSET)


def test_report_first_wet_day_falls_back_to_candidate():
    """With no wet day in the window the candidate itself is reported."""
    rule = passage([total(1, 3, fixed("ge", 0))], report="first_wet_day", report_span=m.Span(1, 3))
    assert run(rule, [0, 5, 0, 0], end=0) == (0, ONSET)


# ---------------------------------------------------------------- curve

def test_curve_day_after_and_of_minimum():
    """The cumulative anomaly is minimal at day 2 here; the report picks the day or the day after."""
    x = [0, 0, 0, 100, 100, 100]
    base = {"b": 5.0}
    assert run(m.Curve("b"), x, end=5, thresholds=base) == (3, ONSET)
    assert run(m.Curve("b", "day_of_minimum"), x, end=5, thresholds=base) == (2, ONSET)


def test_curve_first_argmin_wins_ties():
    """With a flat curve the first minimum is used."""
    assert run(m.Curve("b"), [0, 0, 0], end=2, thresholds={"b": 0.0}) == (1, ONSET)
    assert run(m.Curve("b", "day_of_minimum"), [0, 0, 0, 0], end=2, thresholds={"b": 0.0}) == (0, ONSET)


def test_curve_minimum_at_end_is_not_found():
    """If the cumulative anomaly is lowest at the last day there is no onset."""
    assert run(m.Curve("b"), [0, 0, 0, 0], end=3, thresholds={"b": 1.0}) == (-1, NOT_FOUND)


def test_curve_missing_and_horizon():
    """Missing days in the window give INSUFFICIENT_DATA; running past the data gives CENSORED."""
    x = [0, 0, 100, 100]
    assert run(m.Curve("b"), x, miss=[0, 1, 0, 0], end=3, thresholds={"b": 5.0}) == (-1, INSUFFICIENT)
    assert run(m.Curve("b"), x, end=5, thresholds={"b": 5.0}) == (-1, CENSORED)
    assert run(m.Curve("b"), x, miss=[0, 1, 0, 0], end=5, thresholds={"b": 5.0}) == (-1, INSUFFICIENT)


def test_curve_start_offset():
    """The accumulation starts at the job's first candidate."""
    assert run(m.Curve("b"), [0, 0, 0, 0, 100, 100], start=2, end=5, thresholds={"b": 5.0}) == (4, ONSET)


def test_curve_empty_window():
    """An empty curve window is NOT_FOUND."""
    assert run(m.Curve("b"), [1, 2, 3], start=2, end=1, thresholds={"b": 1.0}) == (-1, NOT_FOUND)


# ---------------------------------------------------------------- trace

def test_trace_codes_flags_and_state():
    """The trace exposes per-candidate codes (1 True, 0 False, -1 Unknown), flags and the combined state."""
    rule = passage([total(1, 2, fixed("ge", 20))], [m.Veto(m.DryRun(2), m.Span(1, 3), "inside")])
    x = [10, 10, 0, 0, 10, 10]
    det = run(rule, x, miss=[0, 0, 0, 0, 0, 0], start=0, end=5, trace=True)
    tr = det.trace
    assert tr["day"][0].tolist() == [0, 1, 2, 3, 4, 5]
    assert tr["requirements"][0, 0].tolist() == [1, 0, 0, 0, 1, -1]
    assert tr["vetoes"][0, 0].tolist() == [0, 1, 1, 0, -1, -1]
    assert tr["state"][0].tolist() == [1, 0, 0, 0, -1, -1]
    assert tr["flags"][0].tolist() == [0, 0, 0, 0, 2, 2]
    assert (int(det.day[0]), int(det.status[0])) == (0, ONSET)


def test_trace_pads_shorter_windows():
    """Jobs with fewer candidates are padded with day -1 and False."""
    plan = plan_mod.compile(definition(passage([])))
    jobs = m.Jobs(np.array([0, 0]), np.array([0, 2]), np.array([3, 3]), np.array([5, 5]))
    amounts, _ = plan_mod.resolve_amounts(plan, {}, n_series=1)
    det = reference.detect(plan, np.zeros((1, 5), np.int32), np.zeros((1, 5), bool), amounts, jobs, trace=True)
    assert det.trace["day"].tolist() == [[0, 1, 2, 3], [2, 3, -1, -1]]
    assert det.trace["valid"].tolist() == [[True] * 4, [True, True, False, False]]


# ---------------------------------------------------------------- brute-force cross-check

def brute_cmp(op, value, amount):
    """Exact comparison."""
    return {"ge": value >= amount, "gt": value > amount, "le": value <= amount, "lt": value < amount}[op]


def brute(defn, x, miss, thr, start, end, E):
    """Scalar evaluation of spec section 4 with exact rational thresholds; returns (day, status)."""

    def amount(a, op=None):
        """Amount in tenths as an exact fraction."""
        return Fraction(a.tenths) if a.threshold is None else Fraction(round(thr[a.threshold] * 20), 20) * 10

    def wet(t):
        return (not miss[t]) and brute_cmp(defn.wet_day.op, Fraction(x[t]), amount(defn.wet_day.amount))

    def dry(t):
        return (not miss[t]) and not wet(t)

    def flags(a, b):
        f = 0
        if b > E:
            f |= 2
        if a < 0 or any(miss[t] for t in range(max(a, 0), min(b, E))):
            f |= 1
        return f

    def status_of(f):
        return CENSORED if f == 2 else INSUFFICIENT

    rule = defn.rule
    if isinstance(rule, m.Curve):
        if end < start:
            return -1, NOT_FOUND
        f = flags(start, end + 1)
        if f:
            return -1, status_of(f)
        base = round(thr[rule.baseline] * 1000)
        c, best, arg = 0, None, None
        for t in range(start, end + 1):
            c += 100 * x[t] - base
            if best is None or c < best:
                best, arg = c, t
        if arg == end:
            return -1, NOT_FOUND
        return (arg + 1 if rule.report == "day_after_minimum" else arg), ONSET

    def span(sp, d):
        return (start if sp.since_search_start else d + sp.first - 1), d + sp.last

    def holds(veto, u):
        L = veto.spell.length if isinstance(veto.spell, m.DryRun) else veto.spell.days
        if flags(u, u + L):
            return False
        if isinstance(veto.spell, m.DryRun):
            return all(dry(t) for t in range(u, u + L))
        return brute_cmp(veto.spell.comparison.op, Fraction(sum(x[u:u + L])), amount(veto.spell.comparison.amount))

    for d in range(start, end + 1):
        codes, fl = [], 0
        for cond in rule.require:
            a, b = span(cond.span, d)
            f = flags(a, b)
            if f:
                codes.append(None)
                fl |= f
            elif isinstance(cond, m.Total):
                codes.append(brute_cmp(cond.comparison.op, Fraction(sum(x[a:b])), amount(cond.comparison.amount)))
            else:
                codes.append(brute_cmp(cond.op, sum(1 for t in range(a, b) if wet(t)), cond.count))
        for veto in rule.reject_if:
            a, b = span(veto.span, d)
            L = veto.spell.length if isinstance(veto.spell, m.DryRun) else veto.spell.days
            if veto.placement == "inside":
                if b - a < L:
                    codes.append(True)
                    continue
                us, judged = range(a, b - L + 1), (a, b)
            else:
                us, judged = range(a, b), (a, b + L - 1)
            if any(holds(veto, u) for u in us):
                codes.append(False)
            elif flags(*judged):
                codes.append(None)
                fl |= flags(*judged)
            else:
                codes.append(True)
        if any(c is False for c in codes):
            continue
        if any(c is None for c in codes):
            return -1, status_of(fl)
        if rule.report == "first_wet_day":
            a, b = span(rule.report_span, d)
            first = next((t for t in range(max(a, 0), min(b, E)) if wet(t)), d)
            return first, ONSET
        return d, ONSET
    return -1, NOT_FOUND


THRESHOLD_NAMES = ("ta", "tb")
OPS = ("ge", "gt", "le", "lt")


def random_comparison(rng):
    """A random comparison against a fixed amount or a named threshold."""
    if rng.random() < 0.3:
        return m.Comparison(str(rng.choice(OPS)), m.Amount(threshold=str(rng.choice(THRESHOLD_NAMES))))
    return m.Comparison(str(rng.choice(OPS)), m.Amount(tenths=int(rng.integers(0, 12)) * 10))


def random_span(rng, since_ok=True, first_max=4, width_max=6):
    """A random span, sometimes 'season to date'."""
    first = int(rng.integers(1, first_max + 1))
    if since_ok and rng.random() < 0.15:
        return m.Span(first, int(rng.integers(1, 4)), True)
    return m.Span(first, first + int(rng.integers(0, width_max)))


def random_veto(rng):
    """A random dry run or dry window veto."""
    placement = "inside" if rng.random() < 0.5 else "starts_inside"
    if rng.random() < 0.5:
        spell = m.DryRun(int(rng.integers(1, 5)))
    else:
        spell = m.DryWindow(int(rng.integers(1, 6)), random_comparison(rng))
    length = spell.length if isinstance(spell, m.DryRun) else spell.days
    first = int(rng.integers(1, 4))
    last = first + length - 1 + int(rng.integers(-1, 6)) if placement == "inside" else first + int(rng.integers(0, 6))
    return m.Veto(spell, m.Span(first, max(first, last)), placement)


def random_condition(rng, since_ok=True):
    """A random Total or WetDays requirement."""
    span = random_span(rng, since_ok)
    if rng.random() < 0.6:
        return m.Total(span, random_comparison(rng))
    return m.WetDays(span, str(rng.choice(OPS)), int(rng.integers(0, 5)))


def random_rule(rng, since_ok=True, report_ok=True):
    """A random first-passage rule."""
    report = {}
    if report_ok and rng.random() < 0.3:
        report = {"report": "first_wet_day", "report_span": m.Span(1, int(rng.integers(1, 5)))}
    return m.FirstPassage(tuple(random_condition(rng, since_ok) for _ in range(rng.integers(0, 4))),
                          tuple(random_veto(rng) for _ in range(rng.integers(0, 3))), **report)


def random_data(rng, n_series, n_time, p_missing=0.08):
    """Random rain in tenths (mostly dry, bursts), a missing mask, and per-series thresholds in mm."""
    x = np.where(rng.random((n_series, n_time)) < 0.4, rng.integers(0, 60, (n_series, n_time)), 0).astype(np.int32)
    miss = rng.random((n_series, n_time)) < p_missing
    x[miss] = 0
    thr = {name: rng.integers(0, 120, n_series) / 20 for name in THRESHOLD_NAMES + ("tc",)}
    return x, miss, thr


def random_jobs(rng, n_series, n_time, n_jobs):
    """Random jobs with starts inside the data, ends possibly beyond it, and data_end <= n_time."""
    series = rng.integers(0, n_series, n_jobs)
    start = rng.integers(0, n_time, n_jobs)
    end = start + rng.integers(-1, n_time, n_jobs)
    data_end = np.where(rng.random(n_jobs) < 0.3, rng.integers(1, n_time + 1, n_jobs), n_time)
    return m.Jobs(series, start, np.minimum(end, n_time + 3), data_end)


def run_many(defn, x, miss, thr, jobs, trace=False):
    """Run jobs through the reference backend."""
    plan = plan_mod.compile(defn)
    amounts, _ = plan_mod.resolve_amounts(plan, thr, n_series=x.shape[0])
    return reference.detect(plan, x, miss, amounts, jobs, trace=trace)


@pytest.mark.parametrize("seed", range(40))
def test_reference_matches_brute_force_first_passage(seed):
    """Random definitions, data and jobs agree with a scalar implementation of the spec."""
    rng = np.random.default_rng(seed)
    defn = definition(random_rule(rng), wet=random_comparison(rng))
    x, miss, thr = random_data(rng, 6, 30)
    jobs = random_jobs(rng, 6, 30, 25)
    det = run_many(defn, x, miss, thr, jobs)
    for j in range(len(jobs.series)):
        s = jobs.series[j]
        one = {k: float(v[s]) for k, v in thr.items()}
        expect = brute(defn, x[s].tolist(), miss[s].tolist(), one, int(jobs.start[j]), int(jobs.end[j]),
                       int(jobs.data_end[j]))
        assert (int(det.day[j]), int(det.status[j])) == expect, (seed, j)


@pytest.mark.parametrize("seed", range(15))
def test_reference_matches_brute_force_curve(seed):
    """Curve results agree with a scalar implementation."""
    rng = np.random.default_rng(1000 + seed)
    report = "day_after_minimum" if seed % 2 else "day_of_minimum"
    defn = definition(m.Curve("tc", report))
    x, miss, thr = random_data(rng, 5, 25, p_missing=0.01)
    jobs = random_jobs(rng, 5, 25, 20)
    det = run_many(defn, x, miss, thr, jobs)
    for j in range(len(jobs.series)):
        s = jobs.series[j]
        one = {k: float(v[s]) for k, v in thr.items()}
        expect = brute(defn, x[s].tolist(), miss[s].tolist(), one, int(jobs.start[j]), int(jobs.end[j]),
                       int(jobs.data_end[j]))
        assert (int(det.day[j]), int(det.status[j])) == expect, (seed, j)


# ---------------------------------------------------------------- properties

def first_decided(defn, x, miss, thr, jobs):
    """Index of the first candidate that is not False for each job (a large number if none)."""
    tr = run_many(defn, x, miss, thr, jobs, trace=True).trace
    decided = tr["state"] != 0
    return np.where(decided.any(axis=1), np.argmax(decided, axis=1) + jobs.start, 10**6)


@pytest.mark.parametrize("seed", range(40))
def test_stricter_rule_never_decides_earlier(seed):
    """Adding a requirement or a veto never makes the first undecided/true candidate earlier."""
    rng = np.random.default_rng(2000 + seed)
    base = random_rule(rng, report_ok=False)
    x, miss, thr = random_data(rng, 6, 30)
    jobs = random_jobs(rng, 6, 30, 30)
    wider = m.FirstPassage(base.require + (random_condition(rng),), base.reject_if)
    vetoed = m.FirstPassage(base.require, base.reject_if + (random_veto(rng),))
    ref = first_decided(definition(base), x, miss, thr, jobs)
    assert (first_decided(definition(wider), x, miss, thr, jobs) >= ref).all()
    assert (first_decided(definition(vetoed), x, miss, thr, jobs) >= ref).all()


@pytest.mark.parametrize("seed", range(30))
def test_stricter_rule_never_gives_earlier_onset_on_complete_data(seed):
    """With complete data, onset days (when found) are never earlier under a stricter rule."""
    rng = np.random.default_rng(3000 + seed)
    base = random_rule(rng, report_ok=False)
    x, miss, thr = random_data(rng, 6, 30, p_missing=0.0)
    jobs = random_jobs(rng, 6, 30, 30)
    jobs.data_end[:] = 30
    jobs.end[:] = np.minimum(jobs.end, 29)
    strict = m.FirstPassage(base.require + (random_condition(rng),), base.reject_if + (random_veto(rng),))
    a = run_many(definition(base), x, miss, thr, jobs)
    b = run_many(definition(strict), x, miss, thr, jobs)
    both = (a.status == ONSET) & (b.status == ONSET)
    assert (b.day[both] >= a.day[both]).all()
    assert not ((a.status == NOT_FOUND) & (b.status == ONSET)).any()


@pytest.mark.parametrize("seed", range(30))
def test_stricter_amount_never_earlier(seed):
    """Raising an 'at least' total or a 'less than' dry-window limit only delays or removes the onset."""
    rng = np.random.default_rng(4000 + seed)
    x, miss, thr = random_data(rng, 6, 30, p_missing=0.0)
    jobs = random_jobs(rng, 6, 30, 30)
    jobs.data_end[:] = 30
    low, high = int(rng.integers(0, 10)) * 10, int(rng.integers(10, 25)) * 10
    veto_limit = int(rng.integers(1, 10)) * 10
    base = m.FirstPassage((total(1, 3, fixed("ge", low)),),
                          (m.Veto(m.DryWindow(4, fixed("lt", veto_limit)), m.Span(2, 12), "starts_inside"),))
    strict = m.FirstPassage((total(1, 3, fixed("ge", high)),),
                            (m.Veto(m.DryWindow(4, fixed("lt", veto_limit + 30)), m.Span(2, 12), "starts_inside"),))
    a = run_many(definition(base), x, miss, thr, jobs)
    b = run_many(definition(strict), x, miss, thr, jobs)
    both = (a.status == ONSET) & (b.status == ONSET)
    assert (b.day[both] >= a.day[both]).all()
    assert not ((a.status == NOT_FOUND) & (b.status == ONSET)).any()


@pytest.mark.parametrize("seed", range(30))
def test_earlier_start_never_gives_later_onset(seed):
    """Extending the search window backwards can only move the onset earlier (rules without season-to-date)."""
    rng = np.random.default_rng(5000 + seed)
    defn = definition(random_rule(rng, since_ok=False, report_ok=False))
    x, miss, thr = random_data(rng, 6, 30)
    jobs = random_jobs(rng, 6, 30, 30)
    earlier = m.Jobs(jobs.series, np.maximum(jobs.start - rng.integers(0, 6, len(jobs.start)), 0), jobs.end,
                     jobs.data_end)
    ref = first_decided(defn, x, miss, thr, jobs)
    assert (first_decided(defn, x, miss, thr, earlier) <= ref).all()
    a = run_many(defn, x, miss, thr, jobs)
    b = run_many(defn, x, miss, thr, earlier)
    both = (a.status == ONSET) & (b.status == ONSET)
    assert (b.day[both] <= a.day[both]).all()


@pytest.mark.parametrize("block_cells", [1, 7, 40, 10**9])
@pytest.mark.parametrize("family", ["first_passage", "curve"])
def test_block_size_independence(monkeypatch, block_cells, family):
    """Results and traces do not depend on how jobs are blocked."""
    rng = np.random.default_rng(77)
    x, miss, thr = random_data(rng, 6, 30)
    jobs = random_jobs(rng, 6, 30, 40)
    defn = definition(random_rule(rng) if family == "first_passage" else m.Curve("ta"))
    expected = run_many(defn, x, miss, thr, jobs, trace=True)
    monkeypatch.setattr(reference, "BLOCK_CELLS", block_cells)
    got = run_many(defn, x, miss, thr, jobs, trace=True)
    assert (got.day == expected.day).all() and (got.status == expected.status).all()
    for key, value in expected.trace.items():
        assert np.array_equal(got.trace[key], value), key
