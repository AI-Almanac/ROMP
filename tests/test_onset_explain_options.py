"""hcwf.onset.explain reports the backend's own evaluation under every option: dry_day, beginning within, missing
policies, exact rain and horizon policies. Its verdicts equal the engine's and the backend's per candidate."""
import dataclasses

import numpy as np
import pytest

from hcwf.onset import engine, explain
from hcwf.onset import model as m
from hcwf.onset import plan as P
from hcwf.onset.backends import reference
from hcwf.onset.model import Jobs, Status
from test_onset_engine import make_rain
from test_onset_oracle_agreement import random_data, random_definition

SEASON = (m.Season("late", m.DayRef(11, 15), m.DayRef(12, 31)),)
N_CASES = 60


def with_dry_day(defn, rng):
    """The definition with a dry_day comparison of its own."""
    cmp = m.Comparison(str(rng.choice(["lt", "le"])), m.Amount(tenths=int(rng.choice([2, 5, 8]))))
    return dataclasses.replace(defn, dry_day=cmp)


def with_begins(defn, rng):
    """The definition with every veto placed 'beginning within' its span."""
    if isinstance(defn.rule, m.Curve):
        return defn
    vetoes = tuple(dataclasses.replace(v, placement="begins") for v in defn.rule.reject_if)
    return dataclasses.replace(defn, rule=dataclasses.replace(defn.rule, reject_if=vetoes))


def make_case(seed, variant, exact=False):
    """A one-cell year of random rain with a random definition (late season), ready for detect_observed."""
    rng = np.random.default_rng(seed)
    defn = dataclasses.replace(random_definition(rng), seasons=SEASON)
    if variant:
        defn = variant(defn, rng)
    tenths, missing, thresholds = random_data(rng, 1, 366)
    rain = make_rain(tenths.reshape(1, 1, 366), missing=missing.reshape(1, 1, 366))
    if exact:
        mm = np.where(missing, 0.0, np.round(tenths / 10.0 + rng.random(tenths.shape) * 0.07, 3))
        rain = dataclasses.replace(rain, tenths=mm.reshape(1, 1, 366))
    thr = {k: v.reshape(1, 1) for k, v in thresholds.items()}
    return defn, rain, thr


def check_case(defn, rain, thr, **options):
    """The Anatomy's result equals detect_observed's and every candidate's verdict equals the backend's on its own."""
    ds = engine.detect_observed(defn, rain, thresholds=thr, years=[2000], backend="reference", **options)
    a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000, thresholds=thr, **options)
    assert a.result.status == Status(int(ds["status"].values.ravel()[0]))
    assert (a.result.day if a.result.day is not None else -1) == int(ds["onset_index"].values.ravel()[0])
    policy = engine._missing_policy(rain, options.get("missing"))
    plan = P.compile(defn, horizon=options.get("horizon", "censor"), missing=policy)
    if not a.candidates:
        return a
    per_cell = {name: values.reshape(-1) for name, values in thr.items()}
    amounts, _ = P.resolve_amounts(plan, per_cell, n_series=1, exact=rain.tenths.dtype.kind == "f")
    tenths = rain.tenths.reshape(1, -1)
    missing = engine._flat_missing(rain, policy, 1)
    for i, c in enumerate(a.candidates):
        single = Jobs(np.zeros(1, np.int64), np.array([c.day]), np.array([c.day]), np.array([a.data_end]),
                      np.array([a.first_candidate]))
        det = reference.detect(plan, tenths, missing, amounts, single)
        verdict = {"T": Status.ONSET, "F": Status.NOT_FOUND}.get(c.state)
        if verdict is not None:
            assert Status(int(det.status[0])) == verdict
        else:
            assert Status(int(det.status[0])) in (Status.CENSORED, Status.INSUFFICIENT_DATA)
        assert (c.state == "F") == (i < len(a.candidates) - 1 or Status(int(det.status[0])) == Status.NOT_FOUND)
        outcomes = [ch.outcome for ch in c.checks]
        assert c.state == ("F" if "fail" in outcomes else "U" if "unknown" in outcomes else "T")
    explain.explain(a)
    return a


def run_cases(variant=None, exact=False, **options):
    """check_case over random cases; the statuses seen."""
    seen = set()
    for seed in range(N_CASES):
        defn, rain, thr = make_case(seed, variant, exact)
        if exact and isinstance(defn.rule, m.Curve):
            continue
        seen.add(check_case(defn, rain, thr, **options).result.status)
    return seen


def test_default_options_still_agree():
    """With no option the per-candidate verdicts equal the backend's too."""
    assert Status.ONSET in run_cases()


def test_dry_day_is_explained():
    """A definition with its own dry_day traces, and the log names the dry-day comparison."""
    assert run_cases(with_dry_day)
    defn, rain, thr = make_case(1, with_dry_day)
    text = explain.explain(explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000, thresholds=thr))
    assert "Dry day: " in text


def test_beginning_within_is_explained():
    """Vetoes placed 'beginning within' trace and the spell shown is the run that begins inside the span."""
    assert run_cases(with_begins)
    for seed in range(N_CASES):
        defn, rain, thr = make_case(seed, with_begins)
        a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000, thresholds=thr)
        for c in a.candidates:
            for ch in c.checks:
                if ch.role == "veto" and ch.spell is not None:
                    assert ch.placement == "begins" and ch.first <= ch.spell.first <= ch.last


def test_dry_day_and_beginning_within_together():
    """Both options at once."""
    assert run_cases(lambda d, r: with_begins(with_dry_day(d, r), r))


@pytest.mark.parametrize("policy", ["flag", "zero", "keep", "drop"])
def test_missing_policies_are_explained(policy):
    """Each missing policy gives the engine's verdicts (keep rejects a missing candidate day, drop gives INSUFFICIENT_DATA)."""
    seen = run_cases(missing=policy)
    if policy == "drop":
        assert Status.INSUFFICIENT_DATA in seen


def test_keep_reports_a_missing_candidate_day():
    """Under keep a candidate whose own day is missing fails with a day_missing check, readable in the log."""
    from test_onset_engine import definition
    rule = m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("le", m.Amount(tenths=100))),))
    values = np.zeros((1, 1, 366), dtype=np.int32)
    gaps = np.zeros((1, 1, 366), dtype=bool)
    gaps[..., 319:321] = True
    rain = make_rain(values, missing=gaps)
    defn = definition(rule, SEASON)
    a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000, missing="keep")
    first, second, third = a.candidates[:3]
    assert first.state == "F" and first.checks[first.decisive].kind == "day_missing"
    assert second.state == "F" and third.state == "T"
    assert a.result.day == third.day
    text = explain.explain(a)
    assert "the candidate day is missing" in text
    assert "2 days: the candidate day is missing" in text
    check_case(defn, rain, {}, missing="keep")


@pytest.mark.parametrize("horizon", ["censor", "truncate", "reject"])
def test_horizon_policies_are_explained(horizon):
    """Each horizon policy gives the engine's verdicts; censoring leaves some candidates undecided."""
    seen = run_cases(horizon=horizon)
    if horizon == "censor":
        assert Status.CENSORED in seen


def test_exact_rain_is_explained():
    """Float rain traces with mm limits and values, and the threshold lines say they were compared exactly."""
    assert run_cases(exact=True)
    for seed in range(N_CASES):
        defn, rain, thr = make_case(seed, None, exact=True)
        if isinstance(defn.rule, m.Curve):
            continue
        a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000, thresholds=thr)
        for c in a.candidates:
            for ch in c.checks:
                if ch.kind == "total" and ch.value is not None:
                    assert ch.value == pytest.approx(rain.tenths[0, 0, ch.first:ch.last + 1].sum())
        if a.thresholds:
            assert "compared exactly" in explain.explain(a)
            return
    pytest.fail("no case had a named threshold")


def test_exact_rain_with_every_option():
    """Exact rain with dry_day, beginning within, keep and truncate together."""
    assert run_cases(lambda d, r: with_begins(with_dry_day(d, r), r), exact=True, missing="keep", horizon="truncate")


def test_forecast_options_are_explained():
    """trace_forecast takes the same options (alone and stitched) and equals detect_forecast."""
    rng = np.random.default_rng(7)
    from test_onset_forecast_layer import JJAS, case, hist_with
    from test_onset_engine import definition
    vetoed = m.FirstPassage((m.WetDays(m.Span(1, 2), "ge", 2),),
                            (m.Veto(m.DryRun(3), m.Span(3, 8), "begins"),))
    defn = dataclasses.replace(definition(vetoed, JJAS), dry_day=m.Comparison("lt", m.Amount(tenths=5)))
    n_leads = 20
    fc_values = np.where(rng.random((1, 1, 1, 1, n_leads)) < 0.5, rng.integers(0, 60, (1, 1, 1, 1, n_leads)), 0)
    hist, fc = case(hist_with([]), fc_values.astype(np.int32), ["2000-06-10"], members=True)
    for horizon in ("censor", "truncate", "reject"):
        for history in (None, hist):
            ds = engine.detect_forecast(defn, fc, history=history, lead_window=(1, n_leads), backend="reference",
                                        horizon=horizon, missing="keep")
            a = explain.trace_forecast(defn, fc, init="2000-06-10", lat=0, lon=0, member=0, history=history,
                                       lead_window=(1, n_leads), horizon=horizon, missing="keep")
            assert a.result.status == Status(int(ds["status"].values.ravel()[0]))
            explain.explain(a)


def test_reject_horizon_rejection_is_worded():
    """A candidate rejected only because its veto span runs past the data says so (it has no spell)."""
    from test_onset_engine import definition
    rule = m.FirstPassage((m.WetDays(m.Span(1, 1), "ge", 1),), (m.Veto(m.DryRun(2), m.Span(1, 40), "starts_inside"),))
    values = np.zeros((1, 1, 366), dtype=np.int32)
    values[..., 350:] = 20
    rain = make_rain(values)
    defn = definition(rule, (m.Season("late", m.DayRef(12, 10), m.DayRef(12, 31)),))
    a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000, horizon="reject")
    assert any(ch.role == "veto" and ch.outcome == "fail" and ch.spell is None for c in a.candidates for ch in c.checks)
    assert "'reject' horizon rejects it" in explain.explain(a)
    check_case(defn, rain, {}, horizon="reject")
