"""GPU stitched detection is bit-identical to the reference run on explicit concatenations, whatever the block size."""
from __future__ import annotations

import time

import numpy as np
import pytest

from momp.onset import model as m
from momp.onset import plan as plan_mod
from momp.onset.backends import gpu, numba_cpu
from test_onset_numba_stitched import (_case, assert_stitched_matches_reference, india_ensemble_case,
                                       reference_on_concatenation)
from test_onset_stitched_agreement import poison_after_split

if not gpu.available():
    pytest.skip("CuPy with a usable CUDA device is required", allow_module_level=True)


def check(*case):
    """The GPU's stitched result equals the reference's on the concatenation."""
    return assert_stitched_matches_reference(*case, detect=gpu.detect_stitched)


@pytest.mark.parametrize("seed", range(300))
def test_gpu_stitched_matches_reference_on_concatenation(seed):
    """Random definitions, gaps, splits, windows and data ends give the reference's day and status."""
    rng = np.random.default_rng(20_000 + seed)
    case = _case(rng, unique_cells=seed % 3 == 0)
    got = check(*case)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = case
    again = gpu.detect_stitched(plan, *poison_after_split(hist, hist_missing, jobs), fc, fc_missing, amounts, jobs)
    np.testing.assert_array_equal(got.day, again.day)
    np.testing.assert_array_equal(got.status, again.status)


@pytest.mark.parametrize("seed", range(6))
def test_gpu_stitched_long_series_many_jobs(seed):
    """History of 400 days, 50 leads, thousands of jobs and long windows."""
    rng = np.random.default_rng(30_000 + seed)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(
        rng, n_cells=25, n_hist=400, n_members=4, n_leads=50, n_jobs=4000)
    n = len(jobs.split)
    jobs.split = rng.integers(60, 395, n).astype(np.int64)
    jobs.start = np.maximum(jobs.split - rng.integers(0, 150, n), 0).astype(np.int64)
    jobs.end = (jobs.split + rng.integers(-5, 50, n)).astype(np.int64)
    jobs.data_end = (jobs.split + rng.integers(25, 51, n)).astype(np.int64)
    check(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)


@pytest.mark.parametrize("seed", range(20))
def test_gpu_stitched_many_jobs_share_one_history_row(seed):
    """One history row, many splits and members: each job still sees only its own history."""
    rng = np.random.default_rng(40_000 + seed)
    case = _case(rng, n_cells=1, n_hist=120, n_members=5, n_leads=40, n_jobs=400)
    assert len(np.unique(case[-1].cell)) == 1 and len(np.unique(case[-1].split)) > 20
    check(*case)


@pytest.mark.parametrize("seed", range(20))
def test_gpu_stitched_windows_anywhere_around_the_split(seed):
    """Candidates wholly in the history, wholly in the forecast, or past the data, with data ends before the split."""
    rng = np.random.default_rng(50_000 + seed)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(rng, n_jobs=300)
    n = len(jobs.split)
    jobs.start = np.maximum(jobs.split + rng.integers(-60, 35, n), 0).astype(np.int64)
    jobs.end = (jobs.start + rng.integers(-1, 30, n)).astype(np.int64)
    jobs.data_end = np.maximum(jobs.split + rng.integers(-4, fc.shape[1] + 1, n), 0).astype(np.int64)
    check(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)


@pytest.mark.parametrize("seed", range(8))
def test_gpu_stitched_block_size_independent(seed, monkeypatch):
    """Tiny and huge blocks give the same answer."""
    rng = np.random.default_rng(60_000 + seed)
    case = _case(rng, n_jobs=300)
    want = reference_on_concatenation(*case)
    for cells in (1, 50, 1000, 10**9):
        monkeypatch.setattr(gpu, "BLOCK_CELLS", cells)
        got = gpu.detect_stitched(*case)
        np.testing.assert_array_equal(got.day, want.day)
        np.testing.assert_array_equal(got.status, want.status)


def test_gpu_stitched_empty_jobs():
    """No jobs, or only empty candidate windows, give NOT_FOUND."""
    rng = np.random.default_rng(1)
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = _case(rng, n_jobs=10)
    jobs.end = jobs.start - 1
    got = gpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    assert (got.day == -1).all() and (got.status == m.Status.NOT_FOUND).all()
    empty = m.StitchedJobs(*(np.zeros(0, dtype=np.int64) for _ in range(6)))
    assert len(gpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, empty).day) == 0


def test_gpu_stitched_trace_is_not_supported():
    """Traces are a reference-only feature."""
    rng = np.random.default_rng(2)
    case = _case(rng, n_jobs=5)
    with pytest.raises(NotImplementedError):
        gpu.detect_stitched(*case, trace=True)


def test_gpu_stitched_timing_india_ensemble(capsys):
    """Time GPU stitched detection on the India-ensemble-shaped case; no speed assertion."""
    plan, hist, hist_missing, fc, fc_missing, amounts, jobs = india_ensemble_case()
    small = m.StitchedJobs(*(a[:2000] for a in (jobs.cell, jobs.series, jobs.split, jobs.start, jobs.end, jobs.data_end)))
    gpu.detect_stitched(plan, hist, hist_missing, fc[:2000], fc_missing[:2000],
                        {k: v[:2000] for k, v in amounts.items()}, small)
    t0 = time.perf_counter()
    got = gpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    t_gpu = time.perf_counter() - t0
    if numba_cpu.available():
        want = numba_cpu.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
        np.testing.assert_array_equal(got.day, want.day)
        np.testing.assert_array_equal(got.status, want.status)
    with capsys.disabled():
        print(f"\n{len(jobs.cell)} stitched jobs (43 inits x 26 members x 2000 cells): gpu stitched {t_gpu:.2f} s")
