"""Per-verification-window thresholds (spec section 13): backends, climatology, engine and the ROMP model path."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.io import input as romp_input
from momp.onset import climatology, engine
from momp.onset import model as m
from momp.onset import plan as plan_mod
from onset_oracle import HORIZON, FALSE, TRUE, UNKNOWN, Series, oracle_detect, used_thresholds
from test_onset_engine import ANY_WET, JJAS, definition, make_forecast, make_rain
from test_onset_forecast_layer import _brute_force_means, _climatology_forecasts, _thresholds_definition
from test_onset_oracle_agreement import _available, random_data, random_definition, random_jobs
from test_onset_romp_history import _cfg
from test_onset_stitched_agreement import concatenated, random_stitched_case

BACKENDS = ["reference", "numba_cpu", "gpu"]


def window_column(starts, lead) -> int:
    """Index of the window a lead falls in (leads before the first start use the first window)."""
    return max(int(np.searchsorted(starts, lead, side="right")) - 1, 0)


def windowed_oracle(defn, x, miss, thresholds, starts, start, end, data_end, split):
    """Brute-force expectation: every candidate is judged by the oracle with the thresholds of its own window.

    thresholds maps name -> (W,) values of one series. Mirrors the oracle's own
    first-passage loop, but picks the Series object of the candidate's window.
    """
    if any(not np.isfinite(thresholds[name]).all() for name in used_thresholds(defn)):
        return -1, m.Status.INSUFFICIENT_DATA
    per_window = [Series(defn, x, miss, {k: v[w] for k, v in thresholds.items()}) for w in range(len(starts))]
    if start > end:
        return -1, m.Status.NOT_FOUND
    if isinstance(defn.rule, m.Curve):
        return oracle_detect(defn, x, miss, {k: v[window_column(starts, start - split + 1)]
                                             for k, v in thresholds.items()}, start, end, data_end)
    for d in range(start, end + 1):
        series = per_window[window_column(starts, d - split + 1)]
        value, flags = series.candidate(d, start, data_end)
        if value == TRUE:
            return series.report(d, data_end), m.Status.ONSET
        if value == UNKNOWN:
            return -1, (m.Status.CENSORED if flags == HORIZON else m.Status.INSUFFICIENT_DATA)
    return -1, m.Status.NOT_FOUND


def random_windows(rng, n_series, max_lead):
    """Random window starts (two or three, first one arbitrary) and (S, W) thresholds for T1, T2 and B."""
    n_win = int(rng.integers(2, 4))
    starts = np.sort(rng.choice(np.arange(-3, max_lead), size=n_win, replace=False)).astype(np.int64)
    thresholds = {}
    for name in ("T1", "T2", "B"):
        values = rng.uniform(0, 30, (n_series, n_win)).round(1)
        thresholds[name] = np.where(rng.random(values.shape) < 0.03, np.nan, values)
    return starts, thresholds


def _module(backend):
    """Import a backend, skipping when it cannot run here."""
    if not _available(backend):
        pytest.skip(f"backend {backend} not available")
    return importlib.import_module(f"momp.onset.backends.{backend}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", range(30))
def test_forecast_only_windows_match_the_windowed_oracle(backend, seed):
    """detect with (S, W) amounts gives the oracle's answer when each candidate uses its lead's window (lead = day + 1)."""
    module = _module(backend)
    rng = np.random.default_rng(seed)
    defn = random_definition(rng)
    n_series, n_time = 6, 70
    tenths, missing, _ = random_data(rng, n_series, n_time)
    jobs = random_jobs(rng, n_series, n_time, 60)
    starts, thresholds = random_windows(rng, n_series, 40)
    plan = plan_mod.compile(defn)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=n_series)

    got = module.detect(plan, tenths, missing, amounts, jobs, amount_windows=starts)

    for j in range(len(jobs.series)):
        s = int(jobs.series[j])
        want = windowed_oracle(defn, tenths[s], missing[s], {k: v[s] for k, v in thresholds.items()}, starts,
                               int(jobs.start[j]), int(jobs.end[j]), int(jobs.data_end[j]), 0)
        if invalid[s]:
            continue
        assert (int(got.day[j]), int(got.status[j])) == (want[0], int(want[1])), (
            f"job {j}: series {s} window {jobs.start[j]}..{jobs.end[j]} starts {starts} rule {defn.rule}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", range(30))
def test_stitched_windows_match_the_windowed_oracle(backend, seed):
    """detect_stitched uses lead = day - split + 1, so candidates in the history use the first window."""
    module = _module(backend)
    rng = np.random.default_rng(1000 + seed)
    defn = random_definition(rng)
    hist, hist_missing, fc, fc_missing, jobs, _ = random_stitched_case(rng)
    starts, thresholds = random_windows(rng, fc.shape[0], 20)
    plan = plan_mod.compile(defn)
    amounts, invalid = plan_mod.resolve_amounts(plan, thresholds, n_series=fc.shape[0])

    got = module.detect_stitched(plan, hist, hist_missing, fc, fc_missing, amounts, jobs, amount_windows=starts)

    for j in range(len(jobs.cell)):
        s = int(jobs.series[j])
        if invalid[s]:
            continue
        x, miss = concatenated(hist, hist_missing, fc, fc_missing, jobs, j)
        want = windowed_oracle(defn, x, miss, {k: v[s] for k, v in thresholds.items()}, starts,
                               int(jobs.start[j]), int(jobs.end[j]), int(jobs.data_end[j]), int(jobs.split[j]))
        assert (int(got.day[j]), int(got.status[j])) == (want[0], int(want[1])), (
            f"job {j}: series {s} split {jobs.split[j]} window {jobs.start[j]}..{jobs.end[j]} starts {starts} "
            f"rule {defn.rule}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", range(10))
def test_identical_window_columns_reproduce_the_unwindowed_result(backend, seed):
    """Repeating one threshold in every window column changes nothing, in either mode; (S, 1) collapses to (S,)."""
    module = _module(backend)
    rng = np.random.default_rng(2000 + seed)
    defn = random_definition(rng)
    plan = plan_mod.compile(defn)
    tenths, missing, thresholds = random_data(rng, 6, 70)
    jobs = random_jobs(rng, 6, 70, 40)
    flat, _ = plan_mod.resolve_amounts(plan, thresholds, n_series=6)
    repeated, _ = plan_mod.resolve_amounts(plan, {k: np.stack([v, v, v], axis=1) for k, v in thresholds.items()},
                                           n_series=6)
    one, _ = plan_mod.resolve_amounts(plan, {k: v[:, None] for k, v in thresholds.items()}, n_series=6)
    plain = module.detect(plan, tenths, missing, flat, jobs)
    for amounts, starts in ((repeated, [1, 9, 25]), (one, [1])):
        got = module.detect(plan, tenths, missing, amounts, jobs, amount_windows=starts)
        np.testing.assert_array_equal(got.day, plain.day)
        np.testing.assert_array_equal(got.status, plain.status)

    hist, hist_missing, fc, fc_missing, sjobs, thr = random_stitched_case(rng)
    flat, _ = plan_mod.resolve_amounts(plan, thr, n_series=fc.shape[0])
    repeated, _ = plan_mod.resolve_amounts(plan, {k: np.stack([v, v], axis=1) for k, v in thr.items()},
                                           n_series=fc.shape[0])
    plain = module.detect_stitched(plan, hist, hist_missing, fc, fc_missing, flat, sjobs)
    got = module.detect_stitched(plan, hist, hist_missing, fc, fc_missing, repeated, sjobs, amount_windows=[1, 12])
    np.testing.assert_array_equal(got.day, plain.day)
    np.testing.assert_array_equal(got.status, plain.status)


def test_windowed_amounts_need_matching_window_starts():
    """(S, W) amounts without W strictly increasing start leads are an error, and a trace is refused."""
    reference = importlib.import_module("momp.onset.backends.reference")
    plan = plan_mod.compile(definition(m.FirstPassage((m.Total(m.Span(1, 2), m.Comparison("ge", m.Amount(threshold="T"))),)),
                                       JJAS))
    amounts, _ = plan_mod.resolve_amounts(plan, {"T": np.ones((1, 2))}, n_series=1)
    jobs = m.Jobs(np.array([0]), np.array([0]), np.array([3]), np.array([10]))
    x, miss = np.zeros((1, 10), np.int32), np.zeros((1, 10), bool)
    with pytest.raises(ValueError, match="amount_windows"):
        reference.detect(plan, x, miss, amounts, jobs)
    with pytest.raises(ValueError, match="increasing"):
        reference.detect(plan, x, miss, amounts, jobs, amount_windows=[5, 5])
    with pytest.raises(NotImplementedError):
        reference.detect(plan, x, miss, amounts, jobs, amount_windows=[1, 3], trace=True)


# ---------------------------------------------------------------- climatology and engine

def test_climatology_per_window_values_match_hand_computation():
    """Each window pools only the leads inside it; the window dim carries the first and last leads."""
    forecasts = _climatology_forecasts()
    windows = [(1, 2), (3, 6)]
    out = climatology.compute_thresholds_from_forecasts(_thresholds_definition(), forecasts, (2000, 2001), (1, 6),
                                                        windows=windows)
    daily = out["daily"]
    assert daily.dims == ("window", "lat", "lon")
    assert daily["window_first"].values.tolist() == [1, 3] and daily["window_last"].values.tolist() == [2, 6]
    for w, lead_range in enumerate(windows):
        total, count, expected = _brute_force_means(forecasts, (2000, 2001), ((6, 1), (6, 30)), lead_range, 1)
        mean = np.where((expected - count) / expected > 0.1, np.nan, total / 10 / count)
        np.testing.assert_allclose(daily.values[w].reshape(2), mean)
        np.testing.assert_allclose(out["wet5"].values[w].reshape(2), 5 * mean)
    plain = climatology.compute_thresholds_from_forecasts(_thresholds_definition(), forecasts, (2000, 2001), (1, 6))
    assert plain["daily"].dims == ("lat", "lon")


@pytest.mark.parametrize("backend", ["reference", "numba", "gpu"])
@pytest.mark.parametrize("history", [False, True])
def test_engine_applies_each_candidate_the_threshold_of_its_lead_window(backend, history):
    """3 mm every day, with a 4 mm threshold on leads 1-5 and 2 mm on leads 6-10: the first onset is lead 6."""
    if not _available({"reference": "reference", "numba": "numba_cpu", "gpu": "gpu"}[backend]):
        pytest.skip(f"backend {backend} not available")
    rule = m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("ge", m.Amount(threshold="T"))),))
    spec = (("T", m.ThresholdSpec("mean_daily")),)
    defn = definition(rule, JJAS, spec)
    fc = make_forecast(np.full((1, 1, 1, 10), 30), ["2020-06-01"])
    hist = None
    if history:
        hist = make_rain(np.zeros((1, 1, 400), dtype=np.int32), start="2019-06-01")
        hist.coords.update({d: fc.coords[d] for d in ("lat", "lon")})

    def run(values, **extra):
        """Detect with a threshold DataArray over (window, lat, lon) or (lat, lon)."""
        dims = ("window", "lat", "lon") if np.ndim(values) == 3 else ("lat", "lon")
        coords = {"window_first": ("window", [1, 6])} if dims[0] == "window" else {}
        return engine.detect_forecast(defn, fc, thresholds={"T": xr.DataArray(values, dims=dims, coords=coords)},
                                      lead_window=(1, 10), backend=backend, history=hist, **extra)

    split = run(np.array([4.0, 2.0]).reshape(2, 1, 1))
    assert split["onset_lead"].values.ravel().tolist() == [6.0]
    assert run(np.full((1, 1), 4.0))["status"].values.ravel().tolist() == [int(m.Status.NOT_FOUND)]
    assert run(np.full((1, 1), 2.0))["onset_lead"].values.ravel().tolist() == [1.0]
    explicit = engine.detect_forecast(defn, fc, thresholds={"T": xr.DataArray(np.array([4.0, 2.0]).reshape(2, 1, 1),
                                      dims=("window", "lat", "lon"))}, lead_window=(1, 10), backend=backend,
                                      history=hist, threshold_windows=[1, 6])
    assert explicit["onset_lead"].values.ravel().tolist() == [6.0]
    with pytest.raises(ValueError, match="threshold_windows"):
        engine.detect_forecast(defn, fc, thresholds={"T": xr.DataArray(np.ones((2, 1, 1)), dims=("window", "lat", "lon"))},
                               lead_window=(1, 10), backend=backend, history=hist)


# ---------------------------------------------------------------- ROMP model path

def test_romp_model_thresholds_are_per_verification_window_and_cached_by_windows(tmp_path, monkeypatch):
    """A model giving 2 mm/day on leads 1-5 and 6 mm/day on 6-10 yields 10 and 30 mm for the windows (1,5), (6,10)."""
    archive = tmp_path / "model"
    archive.mkdir()
    rain = np.where(np.arange(1, 11) <= 5, 2.0, 6.0)[None, :, None, None] * np.ones((61, 10, 2, 2))
    for year in (2015, 2016):
        times = pd.date_range(f"{year}-05-01", f"{year}-06-30")
        ds = xr.Dataset({"tp": (("time", "day", "lat", "lon"), rain)},
                        coords={"time": times, "day": np.arange(1, 11), "lat": [0.0, 1.0], "lon": [0.0, 1.0]})
        ds.to_netcdf(archive / f"{year}.nc")
    monkeypatch.setattr(romp_input, "region_select", lambda ds, **kwargs: ds)
    kwargs = _cfg(onset_definition="moron_robertson", threshold_source="model", threshold_years=(2015, 2016),
                  model="TOY", model_dir=str(archive), model_var="tp", file_pattern="{}.nc", unit_cvt=None,
                  date_filter_year=2024, init_days=tuple(range(7)), start_date=(2015, 5, 1), end_date=(2015, 6, 30),
                  members=None, probabilistic=False, max_forecast_day=10, dir_out=str(tmp_path / "out"), region="x",
                  verification_window_list=((1, 5), (6, 10)))
    romp_input._loaded_thresholds.clear()
    amount = romp_input.load_thresholds(**kwargs)["wet_spell_amount"]
    assert amount.dims == ("window", "lat", "lon")
    assert amount.values[0].tolist() == [[10.0, 10.0]] * 2 and amount.values[1].tolist() == [[30.0, 30.0]] * 2
    assert amount["window_first"].values.tolist() == [1, 6]
    cache = tmp_path / "out" / ".onset_cache"
    assert len(list(cache.glob("model_wet_spell_amount_*.nc"))) == 1

    romp_input._loaded_thresholds.clear()
    monkeypatch.setattr(romp_input, "get_forecast_deterministic_twice_weekly",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache not used")))
    again = romp_input.load_thresholds(**kwargs)["wet_spell_amount"]
    assert again.values[1].tolist() == [[30.0, 30.0]] * 2

    romp_input._loaded_thresholds.clear()
    with pytest.raises(AssertionError, match="cache not used"):
        romp_input.load_thresholds(**{**kwargs, "verification_window_list": ((1, 10),)})
