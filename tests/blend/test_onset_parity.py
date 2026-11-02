"""momp's onset backends reproduce onset_blending's onset search exactly under the parity options.

The oracles (tests/blend/_oracle) are copies of onset_blending@1e6d696: its `find_onsets` and the legacy
`find_onsets_batch` it was checked against. Random float rain (NaN, exact ties, short series, truncated
forecast-style horizons) goes through the reference, numba and GPU backends with rain_resolution "exact",
and every onset day must equal the oracle's.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from _oracle import legacy_onset_utils as legacy
from _oracle.onset_rule import OnsetRule, find_onsets
from momp.lib.convert_specs import rule_body
from momp.onset import definition as D
from momp.onset import plan as P
from momp.onset.backends import numba_cpu, reference
from momp.onset.model import Jobs, Status

HEAD = """id: parity
version: 1
title: t
citation: t
wording: t
status: draft
notes: t
season: {from: 01-01, to: 12-31}
"""


def definition_yaml(r, fixed_total: float | None = None, threshold: tuple | None = None) -> str:
    """The momp definition text of a blending rule (what `momp-run convert` writes, with this header)."""
    return HEAD + rule_body(r, fixed_total, threshold)


def _backends():
    out = [pytest.param(reference, id="reference")]
    out.append(pytest.param(numba_cpu, id="numba", marks=pytest.mark.skipif(
        not numba_cpu.available(), reason="needs numba")))
    try:
        from momp.onset.backends import gpu
        out.append(pytest.param(gpu, id="gpu", marks=pytest.mark.skipif(not gpu.available(), reason="needs a GPU")))
    except ImportError:  # pragma: no cover
        pass
    return out


BACKENDS = _backends()


def rule(**kw) -> OnsetRule:
    base = dict(window=5, wet_day_min_mm=1.0, trigger_rule="all_days_wet", min_wet_days=None,
                wet_day_comparison="gte", threshold_comparison="gt", follow_days=21,
                followup_anchor="after_trigger", mode="consecutive_dry", min_dry_days=7,
                dry_day_min_mm=1.0, dry_run_contained=False, sum_window=10, sum_min_mm=5.0)
    base.update(kw)
    return OnsetRule(**base)


def run_momp(r: OnsetRule, series, thresholds, starts, backend, *, horizon, missing="keep", fixed_total=None):
    """Onset days (1-based, NaN for none) of every row for each start vector, from momp with exact rain."""
    series = np.asarray(series, dtype=float)
    n, n_days = series.shape
    defn = D.loads(definition_yaml(r, fixed_total), fine_amounts=True)
    plan = P.compile(defn, horizon=horizon, missing=missing)
    miss = np.isnan(series)
    rain = np.where(miss, 0.0, series)
    amounts, invalid = P.resolve_amounts(plan, {"thr": np.asarray(thresholds, dtype=float)}, n_series=n, exact=True)
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    out = np.full(starts.shape, np.nan)
    for v, row in enumerate(starts):
        first = np.maximum(1, np.ceil(row)).astype(np.int64) - 1
        jobs = Jobs(np.arange(n), first, np.full(n, n_days - r.window, dtype=np.int64),
                    np.full(n, n_days, dtype=np.int64))
        det = backend.detect(plan, rain, miss, amounts, jobs)
        ok = (det.status == Status.ONSET) & ~invalid
        out[v] = np.where(ok, det.day + 1, np.nan)
    return out


def random_rain(rng, n, n_days, nan=0.03, ties=0.05):
    """Gamma rain with exact ties on the wet-day and dry-day limits and a share of NaN days."""
    series = np.where(rng.random((n, n_days)) < rng.uniform(0.2, 0.8, (n, 1)),
                      rng.gamma(0.7, 7.0, (n, n_days)), rng.random((n, n_days)) * 1.2)
    series = np.round(series, 1)
    tie = rng.random(series.shape) < ties
    series[tie] = rng.choice([0.0, 1.0, 2.5], size=int(tie.sum()))
    series[rng.random(series.shape) < nan] = np.nan
    return series


def thresholds_for(rng, n, nan_rows=3):
    thr = np.round(rng.uniform(5, 40, n), 1)
    thr[:nan_rows] = np.nan
    return thr


def assert_same(a, b):
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def rule_grid():
    """Every option combination blending supports (same-shaped configs, small windows to keep it quick)."""
    for trigger, wet_cmp, tot_cmp, anchor in itertools.product(
            ("all_days_wet", "first_day_wet"), ("gte", "gt"), ("gt", "gte"), ("after_trigger", "onset_day")):
        wets = (None, 2) if trigger == "first_day_wet" else (None,)
        for min_wet, follow in itertools.product(wets, (0, 6, 14)):
            if follow == 0:
                yield rule(window=3, trigger_rule=trigger, wet_day_comparison=wet_cmp, threshold_comparison=tot_cmp,
                           followup_anchor=anchor, follow_days=0, min_wet_days=min_wet)
                continue
            for contained in (False, True):
                yield rule(window=3, trigger_rule=trigger, wet_day_comparison=wet_cmp, threshold_comparison=tot_cmp,
                           followup_anchor=anchor, follow_days=follow, min_wet_days=min_wet, min_dry_days=4,
                           dry_run_contained=contained, dry_day_min_mm=1.5 if contained else 1.0)
            yield rule(window=3, trigger_rule=trigger, wet_day_comparison=wet_cmp, threshold_comparison=tot_cmp,
                       followup_anchor=anchor, follow_days=follow, min_wet_days=min_wet, mode="window_sum",
                       sum_window=5, sum_min_mm=4.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("horizon,short", [("reject", True), ("truncate", False)])
def test_every_option_combination_matches_find_onsets(backend, horizon, short):
    rng = np.random.default_rng(11)
    for i, r in enumerate(rule_grid()):
        series = random_rain(rng, 120, 50)
        thr = thresholds_for(rng, 120)
        starts = np.vstack([np.ones(120), rng.integers(1, 30, 120), np.full(120, 9999.0)])
        expected = find_onsets(series, thr, starts, r, reject_if_short_followup=short)
        got = run_momp(r, series, thr, starts, backend, horizon=horizon)
        try:
            assert_same(got, expected)
        except AssertionError:
            raise AssertionError(f"rule {i}: {r}") from None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("horizon,short", [("reject", True), ("truncate", False)])
@pytest.mark.parametrize("n_days", [1, 2, 3, 4, 7, 10, 13])
def test_short_series_match_find_onsets(backend, horizon, short, n_days):
    rng = np.random.default_rng(n_days)
    for r in itertools.islice(rule_grid(), 0, None, 7):
        series = random_rain(rng, 40, n_days, nan=0.1)
        thr = thresholds_for(rng, 40, nan_rows=1)
        starts = np.vstack([np.ones(40), rng.integers(1, 6, 40)])
        expected = find_onsets(series, thr, starts, r, reject_if_short_followup=short)
        assert_same(run_momp(r, series, thr, starts, backend, horizon=horizon), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("short", [True, False])
def test_the_production_rule_with_45_forecast_leads(backend, short):
    """The shipped rule (3 wet days over a gamma threshold, 21 days, runs of 5 that begin within) on a 45-lead forecast."""
    rng = np.random.default_rng(2)
    r = rule(window=3, follow_days=21, min_dry_days=5)
    series = random_rain(rng, 400, 45, nan=0.0, ties=0.0)
    thr = np.round(rng.uniform(3, 20, 400), 1)
    starts = np.vstack([np.ones(400), rng.integers(1, 40, 400)])
    expected = find_onsets(series, thr, starts, r, reject_if_short_followup=short)
    got = run_momp(r, series, thr, starts, backend, horizon="reject" if short else "truncate")
    assert_same(got, expected)
    assert 0.2 < np.isfinite(expected[0]).mean() < 0.95, "the test data must have onsets and non-onsets"


@pytest.mark.parametrize("backend", BACKENDS)
def test_float_ties_follow_blendings_prefix_sum_differences(backend):
    """Exact decimal sums and float prefix-sum differences disagree on ties; exact resolution reproduces blending's."""
    rng = np.random.default_rng(5)
    r = rule(window=3, follow_days=0, wet_day_min_mm=0.1)
    series = np.round(rng.uniform(0.1, 9.9, (4000, 40)), 1)
    thr = np.round(rng.uniform(5, 25, 4000), 1)
    expected = find_onsets(series, thr, np.ones(4000), r, reject_if_short_followup=True)
    assert_same(run_momp(r, series, thr, np.ones(4000), backend, horizon="reject"), expected)
    tenths = np.rint(series * 10).astype(np.int64)
    decimal = np.full(4000, np.nan)
    for i in range(4000):
        sums = np.convolve(tenths[i], np.ones(3, dtype=np.int64), "valid")
        hit = np.flatnonzero(sums > int(round(thr[i] * 10)))
        decimal[i] = hit[0] + 1 if len(hit) else np.nan
    assert (np.nan_to_num(decimal, nan=-1) != np.nan_to_num(expected[0], nan=-1)).any(), "no float tie was exercised"


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_fixed_amount_finer_than_a_tenth(backend):
    rng = np.random.default_rng(8)
    r = rule(window=4, follow_days=0, threshold_comparison="gte")
    series = np.round(rng.gamma(0.8, 4.0, (500, 40)), 2)
    thr = np.full(500, 20.05)
    expected = find_onsets(series, thr, np.ones(500), r, reject_if_short_followup=True)
    got = run_momp(r, series, thr, np.ones(500), backend, horizon="reject", fixed_total=20.05)
    assert_same(got, expected)
    assert 0.1 < np.isfinite(expected[0]).mean() < 0.9


# The legacy oracle covers its own subset: a total that exceeds the threshold, runs that begin within the follow-up.
LEGACY_PARAMS = [dict(trigger_rule=t, wet_day_comparison=w, followup_anchor=a, window=win, follow_days=f)
                 for t, w, a, win, f in itertools.product(("all_days_wet", "first_day_wet"), ("gte", "gt"),
                                                          ("after_trigger", "onset_day"), (3, 5), (0, 9, 21))]


def _legacy(r: OnsetRule):
    return legacy.OnsetParams(win=r.window, wet_day_min_mm=r.wet_day_min_mm, trigger_rule=r.trigger_rule,
                              wet_day_comparison=r.wet_day_comparison, follow_days=r.follow_days,
                              followup_anchor=r.followup_anchor, mode=r.mode, min_dry_days=r.min_dry_days,
                              dry_day_min_mm=r.dry_day_min_mm, sum_window=r.sum_window, sum_min_mm=r.sum_min_mm)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["consecutive_dry", "window_sum"])
@pytest.mark.parametrize("short", [True, False])
def test_legacy_find_onsets_batch_oracle(backend, mode, short):
    rng = np.random.default_rng(21)
    for kw in LEGACY_PARAMS:
        r = rule(mode=mode, min_dry_days=4, sum_window=5, sum_min_mm=4.0, **kw)
        if r.follow_days == 0:
            continue  # legacy checks the follow-up even when it is empty; blending (and momp) skip it
        series = random_rain(rng, 80, 50)
        thr = thresholds_for(rng, 80)
        starts = np.vstack([np.ones(80), rng.integers(1, 30, 80)])
        old = legacy.find_onsets_batch(series, thr, starts, _legacy(r), reject_if_short_followup=short)
        got = run_momp(r, series, thr, starts, backend, horizon="reject" if short else "truncate")
        assert_same(got, old)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("short", [True, False])
def test_legacy_scalar_find_onset_oracle(backend, short):
    rng = np.random.default_rng(22)
    r = rule(window=3, follow_days=14, min_dry_days=4)
    series = random_rain(rng, 150, 45)
    thr = thresholds_for(rng, 150, nan_rows=0)
    starts = rng.integers(1, 20, 150)
    got = run_momp(r, series, thr, starts, backend, horizon="reject" if short else "truncate")[0]
    for i in range(150):
        old = legacy.find_onset(series[i], thresh=thr[i], reject_if_short_followup=short, start_day=starts[i],
                                params=_legacy(r))
        assert (np.isnan(got[i]) and old is None) or got[i] == old, f"row {i}"
