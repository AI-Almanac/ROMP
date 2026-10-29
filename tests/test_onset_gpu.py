"""The GPU backend is bit-identical to the reference backend, whatever the block size, and a timing report."""
from __future__ import annotations

import time

import numpy as np
import pytest

from momp.onset import model as m
from momp.onset import plan as plan_mod
from momp.onset.backends import gpu, reference
from test_onset_oracle_agreement import random_data, random_definition, random_jobs

if not gpu.available():
    pytest.skip("CuPy with a usable CUDA device is required", allow_module_level=True)


def _compiled(defn, thresholds, n_series):
    """Compile a definition and resolve its amounts."""
    plan = plan_mod.compile(defn)
    amounts, _ = plan_mod.resolve_amounts(plan, thresholds, n_series)
    return plan, amounts


def _assert_same(got: m.Detection, want: m.Detection) -> None:
    """Day and status arrays are equal."""
    np.testing.assert_array_equal(got.day, want.day)
    np.testing.assert_array_equal(got.status, want.status)


@pytest.mark.parametrize("seed", range(300))
def test_matches_reference_random(seed):
    """Random definitions, data, gaps and horizons give the reference's days and statuses."""
    rng = np.random.default_rng(1000 + seed)
    defn = random_definition(rng)
    n_series, n_time = int(rng.integers(3, 40)), int(rng.integers(40, 120))
    tenths, missing, thresholds = random_data(rng, n_series, n_time)
    jobs = random_jobs(rng, n_series, n_time, 400)
    plan, amounts = _compiled(defn, thresholds, n_series)
    _assert_same(gpu.detect(plan, tenths, missing, amounts, jobs),
                 reference.detect(plan, tenths, missing, amounts, jobs))


@pytest.mark.parametrize("seed", range(6))
def test_matches_reference_long_series_many_jobs(seed):
    """T of about 400 and thousands of jobs, with long candidate windows."""
    rng = np.random.default_rng(5000 + seed)
    defn = random_definition(rng)
    n_series, n_time = 50, 400
    tenths, missing, thresholds = random_data(rng, n_series, n_time)
    jobs = random_jobs(rng, n_series, n_time, 5000)
    jobs.end[:] = jobs.start + rng.integers(-1, 150, len(jobs.start))
    plan, amounts = _compiled(defn, thresholds, n_series)
    _assert_same(gpu.detect(plan, tenths, missing, amounts, jobs),
                 reference.detect(plan, tenths, missing, amounts, jobs))


@pytest.mark.parametrize("seed", range(8))
def test_block_size_independent(seed, monkeypatch):
    """Tiny and huge blocks give the same answer."""
    rng = np.random.default_rng(9000 + seed)
    defn = random_definition(rng)
    n_series, n_time = 20, 100
    tenths, missing, thresholds = random_data(rng, n_series, n_time)
    jobs = random_jobs(rng, n_series, n_time, 700)
    plan, amounts = _compiled(defn, thresholds, n_series)
    want = reference.detect(plan, tenths, missing, amounts, jobs)
    for cells in (1, 50, 1000, 10**9):
        monkeypatch.setattr(gpu, "BLOCK_CELLS", cells)
        _assert_same(gpu.detect(plan, tenths, missing, amounts, jobs), want)


def test_trace_not_supported():
    """Traces are a reference-backend feature."""
    rng = np.random.default_rng(0)
    defn = random_definition(rng)
    tenths, missing, thresholds = random_data(rng, 3, 50)
    plan, amounts = _compiled(defn, thresholds, 3)
    with pytest.raises(NotImplementedError):
        gpu.detect(plan, tenths, missing, amounts, random_jobs(rng, 3, 50, 5), trace=True)


def test_timing_forecast_shaped(capsys):
    """Report reference and GPU wall times on a forecast-shaped case (no speed assertion)."""
    rng = np.random.default_rng(42)
    n_series, n_time, n_lead = 43 * 26 * 400, 80, 50
    wet = rng.random((n_series, n_time), dtype=np.float32) < 0.35
    tenths = np.where(wet, rng.gamma(0.9, 60.0, (n_series, n_time)), 0).astype(np.int32)
    missing = np.zeros((n_series, n_time), dtype=bool)
    wet_day = m.Comparison("ge", m.Amount(tenths=10))
    rule = m.FirstPassage(
        (m.Total(m.Span(1, 3), m.Comparison("ge", m.Amount(tenths=200))),),
        (m.Veto(m.DryRun(7), m.Span(1, 21), "starts_inside"),), "first_wet_day", m.Span(1, 3))
    defn = m.Definition(id="icpac_like", version=1, title="", citation="", wording="", status="draft", notes="",
                        wet_day=wet_day, seasons=(m.Season("main", m.DayRef(1, 1), m.DayRef(12, 31)),), rule=rule)
    plan, amounts = _compiled(defn, {}, n_series)
    jobs = m.Jobs(series=np.arange(n_series, dtype=np.int64), start=np.zeros(n_series, dtype=np.int64),
                  end=np.full(n_series, 29, dtype=np.int64), data_end=np.full(n_series, n_lead, dtype=np.int64))
    small = m.Jobs(*(a[:2000] for a in (jobs.series, jobs.start, jobs.end, jobs.data_end)))
    gpu.detect(plan, tenths[:2000], missing[:2000], {k: v[:2000] for k, v in amounts.items()}, small)

    t0 = time.perf_counter()
    got = gpu.detect(plan, tenths, missing, amounts, jobs)
    t_gpu = time.perf_counter() - t0
    t0 = time.perf_counter()
    want = reference.detect(plan, tenths, missing, amounts, jobs)
    t_ref = time.perf_counter() - t0
    _assert_same(got, want)
    with capsys.disabled():
        print(f"\n{n_series} series, {n_lead} leads: reference {t_ref:.2f} s, gpu {t_gpu:.2f} s "
              f"({t_ref / t_gpu:.1f}x)")
