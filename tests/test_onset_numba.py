"""Numba CPU backend: bit-identical to the reference backend on random cases and per-atom cases, and faster on forecast-shaped data."""
import time

import numpy as np
import pytest

import test_onset_reference as ref_tests
from momp.onset import model as m
from momp.onset import plan as plan_mod
from momp.onset.backends import numba_cpu, reference
from test_onset_oracle_agreement import random_data, random_definition, random_jobs

pytestmark = pytest.mark.skipif(not numba_cpu.available(), reason="numba not available")

# The per-atom, brute-force and property tests of the reference suite, re-run through numba.
# Tests that need traces (or the reference's block size) stay reference-only.
_REFERENCE_ONLY = ("trace", "block_size", "never_decides_earlier", "earlier_start")
for _name in dir(ref_tests):
    if _name.startswith("test_") and not any(word in _name for word in _REFERENCE_ONLY):
        globals()[_name] = getattr(ref_tests, _name)


@pytest.fixture(autouse=True)
def _route_reference_to_numba(monkeypatch):
    """Make the reference suite's helpers call the numba backend."""
    monkeypatch.setattr(reference, "detect", numba_cpu.detect)


def _both(defn, tenths, missing, thresholds, jobs):
    """Run reference and numba on the same inputs; return both Detections."""
    plan = plan_mod.compile(defn)
    amounts, _ = plan_mod.resolve_amounts(plan, thresholds)
    want = _REFERENCE_DETECT(plan, tenths, missing, amounts, jobs)
    got = numba_cpu.detect(plan, tenths, missing, amounts, jobs)
    return want, got


_REFERENCE_DETECT = reference.detect


@pytest.mark.parametrize("seed", range(300))
def test_numba_matches_reference_random(seed):
    """Random definitions, data, gaps and horizons give identical day and status arrays."""
    rng = np.random.default_rng(1000 + seed)
    defn = random_definition(rng)
    tenths, missing, thresholds = random_data(rng, 8, 90)
    jobs = random_jobs(rng, 8, 90, 120)
    want, got = _both(defn, tenths, missing, thresholds, jobs)
    assert np.array_equal(want.day, got.day)
    assert np.array_equal(want.status, got.status)


@pytest.mark.parametrize("seed", range(8))
def test_numba_matches_reference_long_series_many_jobs(seed):
    """T ~ 400 and thousands of jobs, with longer windows."""
    rng = np.random.default_rng(5000 + seed)
    defn = random_definition(rng)
    tenths, missing, thresholds = random_data(rng, 30, 400)
    jobs = random_jobs(rng, 30, 400, 4000)
    jobs.end = jobs.start + rng.integers(-1, 120, len(jobs.start))
    want, got = _both(defn, tenths, missing, thresholds, jobs)
    assert np.array_equal(want.day, got.day)
    assert np.array_equal(want.status, got.status)


def test_trace_is_not_supported():
    """Traces are a reference-only feature."""
    plan = plan_mod.compile(ref_tests.definition(ref_tests.passage()))
    jobs = m.Jobs(np.array([0]), np.array([0]), np.array([0]), np.array([1]))
    amounts, _ = plan_mod.resolve_amounts(plan, {}, n_series=1)
    with pytest.raises(NotImplementedError):
        numba_cpu.detect(plan, np.zeros((1, 1), np.int32), np.zeros((1, 1), bool), amounts, jobs, trace=True)


def test_numba_faster_than_reference_on_forecast_shape():
    """43 inits x 26 members x 400 cells with an ICPAC-like rule: numba beats the reference after warm-up."""
    rng = np.random.default_rng(0)
    n_series, n_time = 43 * 26 * 400, 50
    wet = rng.random((n_series, n_time)) < 0.4
    tenths = np.where(wet, rng.gamma(1.0, 60.0, (n_series, n_time)), 0).astype(np.int32)
    missing = np.zeros(tenths.shape, dtype=bool)
    rule = ref_tests.passage(
        [ref_tests.total(1, 3, ref_tests.fixed("ge", 200)), m.WetDays(m.Span(1, 3), "ge", 2)],
        [m.Veto(m.DryRun(21), m.Span(1, 30), "starts_inside")])
    defn = ref_tests.definition(rule)
    plan = plan_mod.compile(defn)
    amounts, _ = plan_mod.resolve_amounts(plan, {}, n_series=n_series)
    series = np.arange(n_series, dtype=np.int64)
    ones = np.ones(n_series, dtype=np.int64)
    jobs = m.Jobs(series, ones * 0, ones * 29, ones * n_time)

    t0 = time.perf_counter()
    numba_cpu.detect(plan, tenths[:4], missing[:4], {k: v[:4] for k, v in amounts.items()},
                     m.Jobs(series[:4], ones[:4] * 0, ones[:4] * 29, ones[:4] * n_time))
    print(f"\nnumba compile/warm-up: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    got = numba_cpu.detect(plan, tenths, missing, amounts, jobs)
    t_numba = time.perf_counter() - t0
    t0 = time.perf_counter()
    want = _REFERENCE_DETECT(plan, tenths, missing, amounts, jobs)
    t_ref = time.perf_counter() - t0
    print(f"reference {t_ref:.2f}s, numba {t_numba:.2f}s ({t_ref / t_numba:.1f}x)")
    assert np.array_equal(want.day, got.day) and np.array_equal(want.status, got.status)
    assert t_numba < t_ref
