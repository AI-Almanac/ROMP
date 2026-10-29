"""Numba stitched detection is bit-identical to the reference run on explicit concatenations, and avoids building them."""
from __future__ import annotations

import time

import numpy as np
import pytest

from momp.onset import model as m
from momp.onset import plan as plan_mod
from momp.onset.backends import numba_cpu, reference
from test_onset_oracle_agreement import random_definition
from test_onset_stitched_agreement import poison_after_split, random_stitched_case

pytestmark = pytest.mark.skipif(not numba_cpu.available(), reason="numba not available")


def reference_on_concatenation(plan, hist, hist_missing, fc, fc_missing, amounts, jobs) -> m.Detection:
    """reference.detect on one explicit row per job: history cut at the split, then the forecast, padded with missing days."""
    n = len(jobs.cell)
    length = int(jobs.split.max()) + fc.shape[1]
    x = np.zeros((n, length), dtype=np.int32)
    missing = np.ones((n, length), dtype=bool)
    for j in range(n):
        c, s, k = int(jobs.cell[j]), int(jobs.series[j]), int(jobs.split[j])
        x[j, :k], missing[j, :k] = hist[c, :k], hist_missing[c, :k]
        x[j, k:k + fc.shape[1]], missing[j, k:k + fc.shape[1]] = fc[s], fc_missing[s]
    per_job = {key: np.asarray(v)[jobs.series] for key, v in amounts.items()}
    plain = m.Jobs(np.arange(n, dtype=np.int64), jobs.start, jobs.end, jobs.data_end)
    return reference.detect(plan, x, missing, per_job, plain)


def assert_stitched_matches_reference(plan, hist, hist_missing, fc, fc_missing, amounts, jobs, detect=None):
    """The backend's stitched result equals the reference's on the concatenation, day and status."""
    got = (detect or numba_cpu.detect_stitched)(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    want = reference_on_concatenation(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    np.testing.assert_array_equal(got.day, want.day)
    np.testing.assert_array_equal(got.status, want.status)
    return got


def _case(rng, **kwargs):
    """A random definition with its stitched case, compiled plan and resolved amounts."""
    defn = random_definition(rng)
    hist, hist_missing, fc, fc_missing, jobs, thresholds = random_stitched_case(rng, **kwargs)
    plan = plan_mod.compile(defn)
    amounts, _ = plan_mod.resolve_amounts(plan, thresholds, n_series=fc.shape[0])
    return plan, hist, hist_missing, fc, fc_missing, amounts, jobs


@pytest.mark.parametrize("seed", range(300))
def test_numba_stitched_matches_reference_on_concatenation(seed):
    """Random definitions, gaps, splits, windows and data ends give the reference's day and status."""
    rng = np.random.default_rng(20_000 + seed)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(rng, unique_cells=seed % 3 == 0)
    assert_stitched_matches_reference(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    poisoned = poison_after_split(hist, hist_missing, jobs)
    got = numba_cpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    again = numba_cpu.detect_stitched(plan, *poisoned, fc, fc_missing, amounts, jobs)
    np.testing.assert_array_equal(got.day, again.day)
    np.testing.assert_array_equal(got.status, again.status)


@pytest.mark.parametrize("seed", range(8))
def test_numba_stitched_long_series_many_jobs(seed):
    """History of 400 days, 50 leads, thousands of jobs and long windows."""
    rng = np.random.default_rng(30_000 + seed)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(
        rng, n_cells=25, n_hist=400, n_members=4, n_leads=50, n_jobs=4000)
    jobs.split = rng.integers(60, 395, len(jobs.split)).astype(np.int64)
    jobs.start = np.maximum(jobs.split - rng.integers(0, 150, len(jobs.split)), 0).astype(np.int64)
    jobs.end = (jobs.split + rng.integers(-5, 50, len(jobs.split))).astype(np.int64)
    jobs.data_end = (jobs.split + rng.integers(25, 51, len(jobs.split))).astype(np.int64)
    assert_stitched_matches_reference(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)


@pytest.mark.parametrize("seed", range(20))
def test_numba_stitched_many_jobs_share_one_history_row(seed):
    """One history row, many splits and members: each job still sees only its own history."""
    rng = np.random.default_rng(40_000 + seed)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(
        rng, n_cells=1, n_hist=120, n_members=5, n_leads=40, n_jobs=400)
    assert len(np.unique(jobs.cell)) == 1 and len(np.unique(jobs.split)) > 20
    assert_stitched_matches_reference(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)


@pytest.mark.parametrize("seed", range(20))
def test_numba_stitched_windows_anywhere_around_the_split(seed):
    """Candidates wholly in the history, wholly in the forecast, or past the data, with data ends before the split."""
    rng = np.random.default_rng(50_000 + seed)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(rng, n_jobs=300)
    n = len(jobs.split)
    jobs.start = np.maximum(jobs.split + rng.integers(-60, 35, n), 0).astype(np.int64)
    jobs.end = (jobs.start + rng.integers(-1, 30, n)).astype(np.int64)
    jobs.data_end = np.maximum(jobs.split + rng.integers(-4, fc.shape[1] + 1, n), 0).astype(np.int64)
    assert_stitched_matches_reference(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)


def test_numba_stitched_empty_jobs():
    """No jobs, or only empty candidate windows, give NOT_FOUND without touching the tables."""
    rng = np.random.default_rng(1)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(rng, n_jobs=10)
    jobs.end = jobs.start - 1
    got = numba_cpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    assert (got.day == -1).all() and (got.status == m.Status.NOT_FOUND).all()
    empty = m.StitchedJobs(*(np.zeros(0, dtype=np.int64) for _ in range(6)))
    assert len(numba_cpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, empty).day) == 0


def test_numba_stitched_trace_is_not_supported():
    """Traces are a reference-only feature."""
    rng = np.random.default_rng(2)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(rng, n_jobs=5)
    with pytest.raises(NotImplementedError):
        numba_cpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs, trace=True)


def india_ensemble_case(n_cells=2000, n_init=43, n_member=26, n_hist=150, n_lead=50, seed=0):
    """43 inits x 26 members x cells, history of 150 days, an ICPAC-like rule with a 21-day dry-run veto.

    Candidates run from 60 days before the split to lead 29, the data end is the last lead.
    """
    rng = np.random.default_rng(seed)
    hist = np.where(rng.random((n_cells, n_hist)) < 0.35, rng.gamma(0.9, 60.0, (n_cells, n_hist)), 0).astype(np.int32)
    hist_missing = np.zeros(hist.shape, dtype=bool)
    n_series = n_init * n_member * n_cells
    fc = np.where(rng.random((n_series, n_lead), dtype=np.float32) < 0.35,
                  rng.gamma(0.9, 60.0, (n_series, n_lead)), 0).astype(np.int32)
    fc_missing = np.zeros(fc.shape, dtype=bool)
    split = np.repeat(np.arange(n_hist - n_init, n_hist, dtype=np.int64), n_member * n_cells)
    series = np.arange(n_series, dtype=np.int64)
    jobs = m.StitchedJobs(cell=series % n_cells, series=series, split=split, start=split - 60,
                          end=split + 28, data_end=split + n_lead)
    rule = m.FirstPassage(
        (m.Total(m.Span(1, 3), m.Comparison("ge", m.Amount(tenths=200))), m.WetDays(m.Span(1, 3), "ge", 2)),
        (m.Veto(m.DryRun(21), m.Span(1, 30), "starts_inside"),))
    defn = m.Definition(id="icpac_like", version=1, title="", citation="", wording="", status="draft", notes="",
                        wet_day=m.Comparison("ge", m.Amount(tenths=10)),
                        seasons=(m.Season("main", m.DayRef(1, 1), m.DayRef(12, 31)),), rule=rule)
    plan = plan_mod.compile(defn)
    amounts, _ = plan_mod.resolve_amounts(plan, {}, n_series=n_series)
    return plan, hist, hist_missing, fc, fc_missing, amounts, jobs


def materialize(hist, hist_missing, fc, fc_missing, jobs, n_member, n_cells):
    """The explicit concatenation, one (history cut at the split + forecast) row per job, padded to a common length."""
    length = int(jobs.split.max()) + fc.shape[1]
    x = np.zeros((len(jobs.cell), length), dtype=np.int32)
    missing = np.ones(x.shape, dtype=bool)
    block = n_member * n_cells
    for i in range(len(jobs.cell) // block):
        rows, k = slice(i * block, (i + 1) * block), int(jobs.split[i * block])
        x[rows, :k] = np.tile(hist[:, :k], (n_member, 1))
        missing[rows, :k] = np.tile(hist_missing[:, :k], (n_member, 1))
        x[rows, k:k + fc.shape[1]], missing[rows, k:k + fc.shape[1]] = fc[rows], fc_missing[rows]
    return x, missing


def test_numba_stitched_timing_india_ensemble(capsys):
    """Time stitched against detect on the materialized concatenation (both after warm-up); no speed assertion."""
    n_cells, n_member = 2000, 26
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = india_ensemble_case(n_cells=n_cells)
    n_jobs = len(jobs.cell)
    small = m.StitchedJobs(*(a[:200] for a in (jobs.cell, jobs.series, jobs.split, jobs.start, jobs.end, jobs.data_end)))
    numba_cpu.detect_stitched(plan, hist, hist_missing, fc[:200], fc_missing[:200],
                              {k: v[:200] for k, v in amounts.items()}, small)

    t0 = time.perf_counter()
    got = numba_cpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    t_stitched = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, missing = materialize(hist, hist_missing, fc, fc_missing, jobs, n_member, n_cells)
    t_build = time.perf_counter() - t0
    plain = m.Jobs(jobs.series, jobs.start, jobs.end, jobs.data_end)
    numba_cpu.detect(plan, x[:200], missing[:200], {k: v[:200] for k, v in amounts.items()},
                     m.Jobs(plain.series[:200], plain.start[:200], plain.end[:200], plain.data_end[:200]))
    t0 = time.perf_counter()
    want = numba_cpu.detect(plan, x, missing, amounts, plain)
    t_concat = time.perf_counter() - t0
    avoided = x.nbytes + missing.nbytes
    tables = x.shape[0] * (x.shape[1] + 1) * (8 + 4 + 4 + 4)
    np.testing.assert_array_equal(got.day, want.day)
    np.testing.assert_array_equal(got.status, want.status)
    with capsys.disabled():
        print(f"\n{n_jobs} jobs (43 inits x 26 members x {n_cells} cells), history 150 d, 50 leads:\n"
              f"  numba stitched          {t_stitched:7.2f} s\n"
              f"  numba detect on concat  {t_concat:7.2f} s  (+ {t_build:.2f} s to build the concatenation)\n"
              f"  concatenation avoided   {avoided / 2**30:.2f} GiB of rain and flags, "
              f"{tables / 2**30:.2f} GiB of its prefix tables")
