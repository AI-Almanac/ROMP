"""Every backend agrees exactly with the brute-force oracle on random definitions, data, gaps and horizons."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from momp.onset import model as m
from momp.onset import plan as plan_mod
from onset_oracle import oracle_detect

OPS = ("ge", "gt", "le", "lt")


def _span(rng, lo=-2, hi=8, max_len=8) -> m.Span:
    """A random span relative to the candidate."""
    first = int(rng.integers(lo, hi))
    return m.Span(first, first + int(rng.integers(0, max_len)))


def _amount(rng, names) -> m.Amount:
    """A fixed amount or one of the named thresholds."""
    if names and rng.random() < 0.4:
        return m.Amount(threshold=str(rng.choice(names)))
    return m.Amount(tenths=int(rng.integers(0, 400)))


def random_definition(rng) -> m.Definition:
    """A random but valid definition exercising every construct the backends must handle."""
    names = ["T1", "T2"]
    if rng.random() < 0.2:
        rule = m.Curve(baseline="B", report=str(rng.choice(["day_after_minimum", "day_of_minimum"])))
    else:
        require = []
        for _ in range(int(rng.integers(1, 4))):
            kind = rng.random()
            if kind < 0.45:
                span = _span(rng) if rng.random() < 0.85 else m.Span(0, int(rng.integers(1, 4)), since_search_start=True)
                require.append(m.Total(span, m.Comparison(str(rng.choice(OPS)), _amount(rng, names))))
            else:
                span = _span(rng)
                n = span.last - span.first + 1
                require.append(m.WetDays(span, str(rng.choice(OPS)), int(rng.integers(0, n + 1))))
        vetoes = []
        for _ in range(int(rng.integers(0, 3))):
            if rng.random() < 0.5:
                spell = m.DryRun(int(rng.integers(1, 6)))
            else:
                spell = m.DryWindow(int(rng.integers(1, 6)), m.Comparison(str(rng.choice(OPS)), _amount(rng, names)))
            first = int(rng.integers(1, 6))
            span = m.Span(first, first + int(rng.integers(0, 12)))
            vetoes.append(m.Veto(spell, span, str(rng.choice(["inside", "starts_inside"]))))
        window = int(rng.integers(1, 6))
        report = str(rng.choice(["candidate", "first_wet_day"]))
        rule = m.FirstPassage(tuple(require), tuple(vetoes), report,
                              m.Span(1, window) if report == "first_wet_day" else None)
    wet = m.Comparison(str(rng.choice(["ge", "gt"])), m.Amount(tenths=int(rng.choice([1, 5, 10]))))
    return m.Definition(id="random", version=1, title="random", citation="", wording="", status="draft",
                        notes="", wet_day=wet, seasons=(m.Season("main", m.DayRef(1, 1), m.DayRef(12, 31)),),
                        rule=rule)


def random_data(rng, n_series: int, n_time: int):
    """Rain in tenths with many dry days, a few missing days, and random thresholds (some NaN)."""
    wet = rng.random((n_series, n_time)) < 0.45
    tenths = np.where(wet, rng.gamma(0.9, 60.0, (n_series, n_time)), rng.integers(0, 12, (n_series, n_time)))
    tenths = np.rint(tenths).astype(np.int32)
    missing = rng.random((n_series, n_time)) < rng.choice([0.0, 0.02, 0.08])
    tenths[missing] = 0
    thresholds = {}
    for name in ("T1", "T2", "B"):
        values = rng.choice([rng.uniform(0, 30, n_series).round(2), rng.integers(0, 30, n_series) / 10.0])
        values = np.where(rng.random(n_series) < 0.05, np.nan, values)
        thresholds[name] = values.astype(float)
    return tenths, missing, thresholds


def random_jobs(rng, n_series: int, n_time: int, n_jobs: int) -> m.Jobs:
    """Jobs with random windows, including empty ones and windows running past the data end."""
    series = rng.integers(0, n_series, n_jobs)
    start = rng.integers(0, n_time - 5, n_jobs)
    end = start + rng.integers(-1, 25, n_jobs)
    data_end = rng.integers(n_time // 2, n_time + 1, n_jobs)
    return m.Jobs(series=series.astype(np.int64), start=start.astype(np.int64),
                  end=end.astype(np.int64), data_end=data_end.astype(np.int64))


def run_backend(name: str, defn, tenths, missing, thresholds, jobs) -> m.Detection:
    """Compile the definition and run one backend through the public plan/backend API."""
    module = importlib.import_module(f"momp.onset.backends.{name}")
    plan = plan_mod.compile(defn)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=tenths.shape[0])
    det = module.detect(plan, tenths, missing, amounts, jobs)
    status = det.status.copy()
    day = det.day.copy()
    bad = invalid[jobs.series]
    status[bad] = m.Status.INSUFFICIENT_DATA
    day[bad] = -1
    return m.Detection(day=day, status=status)


def _available(name: str) -> bool:
    """Whether a backend module exists and can run here."""
    try:
        module = importlib.import_module(f"momp.onset.backends.{name}")
    except ImportError:
        return False
    return getattr(module, "available", lambda: True)()


@pytest.mark.parametrize("backend", ["reference", "numba_cpu", "gpu"])
@pytest.mark.parametrize("seed", range(40))
def test_backend_matches_oracle(backend, seed):
    """A backend gives the oracle's day and status for every job of a random definition."""
    if not _available(backend):
        pytest.skip(f"backend {backend} not available")
    rng = np.random.default_rng(seed)
    defn = random_definition(rng)
    n_series, n_time = 6, 70
    tenths, missing, thresholds = random_data(rng, n_series, n_time)
    jobs = random_jobs(rng, n_series, n_time, 60)

    got = run_backend(backend, defn, tenths, missing, thresholds, jobs)

    for j in range(len(jobs.series)):
        s = int(jobs.series[j])
        want = oracle_detect(defn, tenths[s], missing[s], {k: v[s] for k, v in thresholds.items()},
                             int(jobs.start[j]), int(jobs.end[j]), int(jobs.data_end[j]))
        assert (int(got.day[j]), int(got.status[j])) == (want[0], int(want[1])), (
            f"job {j}: series {s}, window {jobs.start[j]}..{jobs.end[j]}, data_end {jobs.data_end[j]}, "
            f"rule {defn.rule}, wet {defn.wet_day}"
        )


@pytest.mark.parametrize("backend", ["reference", "numba_cpu", "gpu"])
def test_first_wet_day_report_past_the_data_end_falls_back_to_the_candidate(backend):
    """A candidate whose report window lies beyond the data reports itself, not the last data day."""
    if not _available(backend):
        pytest.skip(f"backend {backend} not available")
    defn = m.Definition(
        id="edge", version=1, title="edge", citation="", wording="", status="draft", notes="",
        wet_day=m.Comparison("ge", m.Amount(tenths=10)),
        seasons=(m.Season("main", m.DayRef(1, 1), m.DayRef(12, 31)),),
        rule=m.FirstPassage(require=(m.Total(m.Span(-2, 0), m.Comparison("ge", m.Amount(tenths=0))),),
                            report="first_wet_day", report_span=m.Span(1, 3)),
    )
    n_time = 20
    tenths = np.full((1, n_time), 50, np.int32)
    missing = np.zeros((1, n_time), bool)
    jobs = m.Jobs(series=np.array([0, 0]), start=np.array([n_time - 1, n_time]),
                  end=np.array([n_time - 1, n_time]), data_end=np.array([n_time, n_time]))

    got = run_backend(backend, defn, tenths, missing, {}, jobs)

    assert got.day.tolist() == [n_time - 1, n_time]
    assert got.status.tolist() == [m.Status.ONSET, m.Status.ONSET]
