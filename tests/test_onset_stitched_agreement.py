"""Stitched detection (observed history, then a forecast) agrees with the oracle run on the concatenated series."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from momp.onset import model as m
from momp.onset import plan as plan_mod
from onset_oracle import oracle_detect
from test_onset_oracle_agreement import _available, random_definition

BACKENDS = ["reference", "numba_cpu", "gpu"]


def random_stitched_case(rng, n_cells=4, n_hist=80, n_members=3, n_leads=30, n_jobs=80, unique_cells=False):
    """History rows per cell, forecast rows per (cell, member), and jobs that split each forecast into a history.

    History days at or after each job's split hold heavy rain, so a backend that
    reads them (peeking past the init) gets a different answer.
    """
    def rain(shape):
        """Gamma rain in tenths with many dry days and a few missing days."""
        wet = rng.random(shape) < 0.45
        x = np.where(wet, rng.gamma(0.9, 60.0, shape), rng.integers(0, 12, shape))
        x = np.rint(x).astype(np.int32)
        missing = rng.random(shape) < rng.choice([0.0, 0.0, 0.02])
        x[missing] = 0
        return x, missing

    if unique_cells:
        n_cells, n_members = n_jobs, 1
    series = rng.permutation(n_jobs) if unique_cells else rng.integers(0, n_cells * n_members, n_jobs)
    cell = series // n_members
    hist, hist_missing = rain((n_cells, n_hist))
    fc, fc_missing = rain((n_cells * n_members, n_leads))
    split = rng.integers(5, n_hist - 5, n_jobs)
    start = split - rng.integers(0, 25, n_jobs)
    start = np.maximum(start, 0)
    end = split + rng.integers(-3, n_leads, n_jobs)
    data_end = split + rng.integers(n_leads // 2, n_leads + 1, n_jobs)
    jobs = m.StitchedJobs(cell=cell.astype(np.int64), series=series.astype(np.int64), split=split.astype(np.int64),
                          start=start.astype(np.int64), end=end.astype(np.int64), data_end=data_end.astype(np.int64))
    thresholds = {}
    for name in ("T1", "T2", "B"):
        values = rng.uniform(0, 30, n_cells * n_members).round(2)
        thresholds[name] = np.where(rng.random(len(values)) < 0.05, np.nan, values)
    return hist, hist_missing, fc, fc_missing, jobs, thresholds


def poison_after_split(hist, hist_missing, jobs):
    """Copies of the history with each cell's days from its latest job split onward set to heavy rain and no gaps.

    No job of that cell may read those days; with one job per cell (unique_cells)
    this checks every job against its own split.
    """
    hist, hist_missing = hist.copy(), hist_missing.copy()
    for c in np.unique(jobs.cell):
        latest = int(jobs.split[jobs.cell == c].max())
        hist[c, latest:] = 999
        hist_missing[c, latest:] = False
    return hist, hist_missing


def concatenated(hist, hist_missing, fc, fc_missing, jobs, j):
    """The explicit stitched series of job j: history cut at split, then the forecast row."""
    c, s, k = int(jobs.cell[j]), int(jobs.series[j]), int(jobs.split[j])
    return (np.concatenate([hist[c, :k], fc[s]]), np.concatenate([hist_missing[c, :k], fc_missing[s]]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("unique_cells", [False, True], ids=["shared_history", "one_job_per_cell"])
@pytest.mark.parametrize("seed", range(40))
def test_stitched_detection_matches_the_oracle_on_the_concatenation(backend, unique_cells, seed):
    """detect_stitched gives the oracle's day and status for the concatenated series, and never peeks past the split."""
    if not _available(backend):
        pytest.skip(f"backend {backend} not available")
    module = importlib.import_module(f"momp.onset.backends.{backend}")
    if not hasattr(module, "detect_stitched"):
        pytest.skip(f"backend {backend} has no detect_stitched yet")
    rng = np.random.default_rng(seed)
    defn = random_definition(rng)
    hist, hist_missing, fc, fc_missing, jobs, thresholds = random_stitched_case(rng, unique_cells=unique_cells)
    plan = plan_mod.compile(defn)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=fc.shape[0])

    got = module.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs)
    poisoned = module.detect_stitched(plan, *poison_after_split(hist, hist_missing, jobs), fc, fc_missing, amounts, jobs)

    np.testing.assert_array_equal(poisoned.day, got.day)
    np.testing.assert_array_equal(poisoned.status, got.status)
    for j in range(len(jobs.cell)):
        s = int(jobs.series[j])
        if invalid[s]:
            continue
        x, miss = concatenated(hist, hist_missing, fc, fc_missing, jobs, j)
        want = oracle_detect(defn, x, miss, {k: v[s] for k, v in thresholds.items()},
                             int(jobs.start[j]), int(jobs.end[j]), int(jobs.data_end[j]))
        assert (int(got.day[j]), int(got.status[j])) == (want[0], int(want[1])), (
            f"job {j}: cell {jobs.cell[j]} series {s} split {jobs.split[j]} window {jobs.start[j]}..{jobs.end[j]} "
            f"data_end {jobs.data_end[j]}, rule {defn.rule}")
