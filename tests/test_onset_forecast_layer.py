"""Forecast layer: observed history plus forecast (stitched) detection, PRIOR, members, and the model climatology."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.onset import climatology, engine
from momp.onset import model as m
from momp.onset import plan as plan_mod
from momp.onset.backends import reference
from test_onset_engine import (ANY_WET, JJAS, TWO_WET, day_index, definition, make_forecast, make_rain, season)
from test_onset_oracle_agreement import _available, random_data, random_definition

ONSET, NOT_FOUND, CENSORED, PRIOR, INSUFFICIENT = (int(s) for s in m.Status)
BACKENDS = ["reference", "numba", "gpu"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Each backend that can run stitched detection here."""
    name = request.param
    module_name = {"reference": "reference", "numba": "numba_cpu", "gpu": "gpu"}[name]
    if not _available(module_name) or not hasattr(importlib.import_module(f"momp.onset.backends.{module_name}"),
                                                  "detect_stitched"):
        pytest.skip(f"backend {name} cannot run detect_stitched here")
    return name


def case(hist_values, fc_values, inits, lead_label="start", members=False, hist_start="2000-01-01"):
    """A history and a forecast on the same one-cell grid; hist_values is (1, 1, T), fc_values (init, [member], 1, 1, L)."""
    fc = make_forecast(fc_values, inits, lead_label, members=members)
    hist = make_rain(hist_values, start=hist_start)
    hist.coords.update({d: fc.coords[d] for d in ("lat", "lon")})
    return hist, fc


def zeros(n_time, n=1):
    """Zero rain shaped (n, 1, n_time)."""
    return np.zeros((n, 1, n_time), dtype=np.int32)


def hist_with(dates, n_time=366, amount=100, start="2000-01-01"):
    """Zero history of n_time days with `amount` tenths on the given ISO dates."""
    template = make_rain(zeros(n_time), start=start)
    values = zeros(n_time)
    for date in dates:
        values[..., day_index(template, date)] = amount
    return values


def at(ds, name, *index):
    """Value of a variable at a position given for (init_time, [member], season, lat, lon)."""
    return ds[name].values[index]


# ---------------------------------------------------------------- statuses

def test_prior_when_the_history_already_decides(backend):
    """An onset before lead 1 is PRIOR for every member and reported with its own day; later inits search the forecast."""
    hist, fc = case(hist_with(["2000-06-05"]), np.zeros((2, 1, 1, 1, 12), dtype=np.int32),
                    ["2000-06-10", "2000-06-01"], members=True)
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, backend=backend)
    assert ds["status"].dims == ("init_time", "member", "season", "lat", "lon")
    assert ds["status"].values[0, :, 0, 0, 0].tolist() == [PRIOR]
    assert ds["onset_date"].values[0, 0, 0, 0, 0] == np.datetime64("2000-06-05")
    assert ds["onset_lead"].values[0, 0, 0, 0, 0] == -4.0
    assert ds["status"].values[1, 0, 0, 0, 0] == NOT_FOUND
    assert ds.attrs["forecast_history"] == "observed"


def test_back_dated_onset_completed_by_the_forecast(backend):
    """A wet spell begun before lead 1 and finished by the forecast is an ONSET dated before lead 1."""
    hist = hist_with(["2000-06-09"])
    fc_values = np.zeros((2, 1, 1, 12), dtype=np.int32)
    fc_values[0, ..., 0] = 100
    hist, fc = case(hist, fc_values, ["2000-06-10", "2000-06-10"])
    ds = engine.detect_forecast(definition(TWO_WET, JJAS), fc, history=hist, backend=backend)
    assert ds["status"].values[:, 0, 0, 0].tolist() == [ONSET, CENSORED]
    assert ds["onset_date"].values[0, 0, 0, 0] == np.datetime64("2000-06-09")
    assert ds["onset_lead"].values[0, 0, 0, 0] == 0.0
    assert np.isnan(ds["onset_lead"].values[1, 0, 0, 0])


def test_dry_spell_check_that_needs_forecast_days(backend):
    """A veto about the days after the candidate is decided by the forecast, and the onset may still lie before lead 1."""
    rule = m.FirstPassage((m.WetDays(m.Span(1, 1), "ge", 1),),
                          (m.Veto(m.DryRun(3), m.Span(2, 4), "inside"),))
    fc_values = np.zeros((3, 1, 1, 12), dtype=np.int32)
    fc_values[0, ..., 2] = 100
    fc_values[1, ..., 0] = 100
    hist, fc = case(hist_with(["2000-06-09"]), fc_values, ["2000-06-10"] * 3)
    ds = engine.detect_forecast(definition(rule, JJAS), fc, history=hist, backend=backend)
    assert ds["status"].values[:, 0, 0, 0].tolist() == [ONSET, ONSET, NOT_FOUND]
    np.testing.assert_array_equal(ds["onset_lead"].values[:, 0, 0, 0], [0.0, 0.0, np.nan])


@pytest.mark.parametrize("lead_label,status,date,lead", [
    ("start", NOT_FOUND, None, None), ("end", PRIOR, "2000-06-10", 0.0)])
def test_lead_label_sets_the_split(backend, lead_label, status, date, lead):
    """Rain on the init date is forecast data for "start" labels (never read from history) and history for "end" labels."""
    hist, fc = case(hist_with(["2000-06-10"]), np.zeros((1, 1, 1, 8), dtype=np.int32), ["2000-06-10"], lead_label)
    jobs, _ = engine.build_stitched_jobs(definition(ANY_WET, JJAS), hist, fc)
    assert jobs.split.tolist() == [day_index(hist, "2000-06-10") + (lead_label == "end")]
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, backend=backend)
    assert ds["status"].values.ravel().tolist() == [status]
    if date:
        assert ds["onset_date"].values.ravel()[0] == np.datetime64(date)
        assert ds["onset_lead"].values.ravel()[0] == lead


def test_season_crossing_the_year_end(backend):
    """A Nov-Feb season starts in the history for inits on either side of New Year, and inits are matched to the right year."""
    defn = definition(ANY_WET, (season("nd", (11, 1), (2, 28)),))
    inits = ["2000-12-20", "2001-01-05"]
    fc_values = np.zeros((2, 1, 1, 10), dtype=np.int32)
    hist, fc = case(hist_with(["2000-12-05"], 450), fc_values, inits)
    jobs, _ = engine.build_stitched_jobs(defn, hist, fc)
    assert jobs.start.tolist() == [day_index(hist, "2000-11-01")] * 2
    ds = engine.detect_forecast(defn, fc, history=hist, backend=backend)
    assert ds["status"].values[:, 0, 0, 0].tolist() == [PRIOR, PRIOR]
    assert set(ds["onset_date"].values[:, 0, 0, 0]) == {np.datetime64("2000-12-05")}

    fc_values[:, ..., 1] = 100
    hist, fc = case(hist_with([], 450), fc_values, inits)
    ds = engine.detect_forecast(defn, fc, history=hist, backend=backend)
    assert ds["status"].values[:, 0, 0, 0].tolist() == [ONSET, ONSET]
    np.testing.assert_array_equal(ds["onset_date"].values[:, 0, 0, 0],
                                  np.array(["2000-12-21", "2001-01-06"], dtype="datetime64[ns]"))
    assert ds["onset_lead"].values[:, 0, 0, 0].tolist() == [2.0, 2.0]


def test_season_starting_inside_the_forecast_or_before_the_history(backend):
    """A season opening after lead 1 starts at its first lead; one opening before the history is INSUFFICIENT_DATA."""
    fc_values = np.zeros((1, 1, 1, 30), dtype=np.int32)
    fc_values[..., 0] = 100
    hist, fc = case(hist_with([], 200), fc_values, ["2000-05-25"])
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, backend=backend)
    assert ds["status"].values.ravel().tolist() == [NOT_FOUND]
    jobs, _ = engine.build_stitched_jobs(definition(ANY_WET, JJAS), hist, fc)
    assert jobs.start.tolist() == [jobs.split[0] + 7]
    hist, fc = case(hist_with([], 200, start="2000-06-05"), fc_values, ["2000-06-10"], hist_start="2000-06-05")
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, backend=backend)
    assert ds["status"].values.ravel().tolist() == [INSUFFICIENT]


# ---------------------------------------------------------------- no peeking, errors

def test_history_after_the_init_is_never_read(backend):
    """Changing history at or after lead 1 cannot change any result, whether it holds heavy rain, gaps or nothing."""
    rng = np.random.default_rng(3)
    fc_values = rng.integers(0, 40, (3, 2, 1, 1, 20)).astype(np.int32)
    inits = ["2000-06-10", "2000-07-02", "2000-08-15"]
    base = rng.integers(0, 40, (1, 1, 300)).astype(np.int32)
    hist, fc = case(base, fc_values, inits, members=True)
    defn = definition(TWO_WET, JJAS)
    want = engine.detect_forecast(defn, fc, history=hist, backend=backend)
    last = day_index(hist, "2000-08-15")
    for fill in (999, 0):
        values = base.copy()
        values[..., last:] = fill
        poisoned, _ = case(values, fc_values, inits, members=True)
        poisoned.missing[..., last:] = fill == 0
        got = engine.detect_forecast(defn, fc, history=poisoned, backend=backend)
        xr.testing.assert_identical(got, want)
        # an earlier init only sees history before its own lead 1
        early = base.copy()
        early[..., day_index(hist, "2000-06-10"):] = fill
        mid, _ = case(early, fc_values, inits, members=True)
        got = engine.detect_forecast(defn, fc, history=mid, backend=backend)
        assert got["status"].values[0].tolist() == want["status"].values[0].tolist()


def test_history_errors():
    """Mismatched grids, a history that stops short, and a lead window not starting at 1 are rejected with clear messages."""
    fc_values = np.zeros((1, 1, 1, 10), dtype=np.int32)
    hist, fc = case(hist_with([]), fc_values, ["2000-06-10"])
    shifted = make_rain(hist_with([]))
    shifted.coords.update({"lat": np.array([2.0]), "lon": fc.coords["lon"]})
    with pytest.raises(ValueError, match="differ along lat"):
        engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=shifted, backend="reference")
    short, _ = case(hist_with([], 100), fc_values, ["2000-06-10"])
    with pytest.raises(ValueError, match="does not reach"):
        engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=short, backend="reference")
    with pytest.raises(ValueError, match="start at 1"):
        engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, lead_window=(2, 9), backend="reference")


def test_lead_window_end_caps_the_candidates(backend):
    """Candidates stop at the lead window's end, so later rain gives NOT_FOUND."""
    fc_values = np.zeros((1, 1, 1, 12), dtype=np.int32)
    fc_values[..., 8] = 100
    hist, fc = case(hist_with([]), fc_values, ["2000-06-10"])
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, lead_window=(1, 5), backend=backend)
    assert ds["status"].values.ravel().tolist() == [NOT_FOUND]
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, lead_window=(1, 12), backend=backend)
    assert ds["onset_lead"].values.ravel().tolist() == [9.0]


# ---------------------------------------------------------------- members and summary

def test_members_and_ensemble_summary(backend):
    """Hand-built case: counts per status and the share of members with onset by each lead, PRIOR at lead 0."""
    fc_values = np.zeros((2, 4, 1, 1, 15), dtype=np.int32)
    fc_values[:, 0, ..., 2] = 100
    fc_values[:, 1, ..., 4] = 100
    fc, hist = make_forecast(fc_values, ["2000-06-10", "2000-06-20"], members=True), None
    fc.missing[:, 3] = True
    hist = make_rain(hist_with(["2000-06-12"]))
    hist.coords.update({d: fc.coords[d] for d in ("lat", "lon")})
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, history=hist, lead_window=(1, 10), backend=backend)
    counts = ds["members_by_status"]
    assert counts.dims == ("status_code", "init_time", "season", "lat", "lon")
    assert counts.sel(status_code=ONSET).values[:, 0, 0, 0].tolist() == [2, 0]
    assert counts.sel(status_code=NOT_FOUND).values[:, 0, 0, 0].tolist() == [1, 0]
    assert counts.sel(status_code=INSUFFICIENT).values[:, 0, 0, 0].tolist() == [1, 0]
    assert counts.sel(status_code=PRIOR).values[:, 0, 0, 0].tolist() == [0, 4]
    prob = ds["onset_probability"]
    assert prob.dims == ("init_time", "season", "lead", "lat", "lon")
    assert prob["lead"].values.tolist() == list(range(11))
    third = 1 / 3
    np.testing.assert_allclose(prob.values[0, 0, :, 0, 0], [0, 0, 0, third, third, 2 * third, 2 * third, 2 * third,
                                                           2 * third, 2 * third, 2 * third])
    np.testing.assert_array_equal(prob.values[1, 0, :, 0, 0], np.ones(11))
    assert "INSUFFICIENT_DATA" in prob.attrs["comment"]


def test_no_summary_without_members_and_history_free_mode_is_unchanged():
    """Without members there is no summary; without history onset_lead is float with NaN, leads start at 1, status as before."""
    fc_values = np.zeros((2, 1, 1, 40), dtype=np.int32)
    fc_values[:, ..., 9] = 100
    fc = make_forecast(fc_values, ["2020-05-25", "2020-12-01"])
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc, backend="reference")
    assert ds["onset_lead"].dtype == np.float64
    np.testing.assert_array_equal(ds["onset_lead"].values[:, 0, 0, 0], [10.0, np.nan])
    assert ds["status"].values[:, 0, 0, 0].tolist() == [ONSET, NOT_FOUND]
    assert ds.attrs["forecast_history"] == "none"
    assert "onset_probability" not in ds and "members_by_status" not in ds
    members = make_forecast(np.repeat(fc_values[:, None], 2, axis=1), ["2020-05-25", "2020-12-01"], members=True)
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), members, backend="reference")
    assert ds["members_by_status"].sel(status_code=ONSET).values[:, 0, 0, 0].tolist() == [2, 0]
    np.testing.assert_array_equal(ds["onset_probability"].values[0, 0, :, 0, 0],
                                  [0] * 10 + [1.0] * 21)


def test_stitched_with_everything_after_the_season_equals_history_free_search():
    """With a dry history and the season opening at lead 1, stitched and history-free searches find the same onsets."""
    rng = np.random.default_rng(5)
    fc_values = (rng.random((3, 2, 1, 1, 25)) * 30).astype(np.int32)
    inits = ["2000-06-01", "2000-06-01", "2000-06-01"]
    hist, fc = case(zeros(200), fc_values, inits, members=True)
    defn = definition(TWO_WET, JJAS)
    with_hist = engine.detect_forecast(defn, fc, history=hist, backend="reference")
    without = engine.detect_forecast(defn, fc, backend="reference")
    np.testing.assert_array_equal(with_hist["status"].values, without["status"].values)
    np.testing.assert_array_equal(with_hist["onset_lead"].values, without["onset_lead"].values)


# ---------------------------------------------------------------- PRIOR versus stitched

@pytest.mark.parametrize("seed", range(40))
def test_prior_never_disagrees_with_the_stitched_search(seed):
    """Wherever the history alone finds an onset, the full stitched search finds the same onset day."""
    rng = np.random.default_rng(seed)
    defn = random_definition(rng)
    if isinstance(defn.rule, m.Curve):
        pytest.skip("curve rules never use the PRIOR search")
    n_cell, n_member, n_leads = 3, 2, 30
    inits = pd.date_range("2000-03-01", periods=6, freq="17D").strftime("%Y-%m-%d").tolist()
    hist_t, hist_m, thresholds = random_data(rng, n_cell, 366)
    fc_t, fc_m, _ = random_data(rng, len(inits) * n_member * n_cell, n_leads)
    fc = make_forecast(fc_t.reshape(len(inits), n_member, n_cell, 1, n_leads), inits, members=True,
                       spatial=("lat", "lon"))
    fc.missing[...] = fc_m.reshape(fc.missing.shape)
    hist = make_rain(hist_t.reshape(n_cell, 1, 366))
    hist.missing[...] = hist_m.reshape(hist.missing.shape)
    hist.coords.update({d: fc.coords[d] for d in ("lat", "lon")})
    cell_thr = {k: v[:n_cell].reshape(n_cell, 1) for k, v in thresholds.items()}
    plan = plan_mod.compile(defn)
    jobs, layout = engine.build_stitched_jobs(defn, hist, fc, (1, n_leads))
    per_cell = engine._series_thresholds(plan, cell_thr, ("lat", "lon"), (n_cell, 1), (n_cell, 1))
    per_series = engine._series_thresholds(plan, cell_thr, ("lat", "lon"), (n_cell, 1), (len(inits), n_member, n_cell, 1))
    amounts, invalid = plan_mod.resolve_amounts(plan, per_series, n_series=fc.tenths.reshape(-1, n_leads).shape[0])
    stitched = reference.detect_stitched(plan, hist.tenths.reshape(n_cell, -1), hist.missing.reshape(n_cell, -1),
                                         fc.tenths.reshape(-1, n_leads), fc.missing.reshape(-1, n_leads),
                                         amounts, jobs)
    prior_day, prior_hit = engine._prior_search(plan, hist, jobs, layout, per_cell, "reference")
    k = len(layout.season_names)
    shape = (k, len(inits), n_member, n_cell)
    hit = np.broadcast_to(prior_hit[:, :, None, :], shape).ravel()
    day = np.broadcast_to(prior_day[:, :, None, :], shape).ravel()
    ok = hit & ~invalid[jobs.series]
    assert (stitched.status[ok] == ONSET).all()
    np.testing.assert_array_equal(stitched.day[ok], day[ok])

    ds = engine.detect_forecast(defn, fc, thresholds=cell_thr, history=hist, lead_window=(1, n_leads),
                                backend="reference")
    bad = invalid[jobs.series]
    expect = np.where(hit & ~bad, PRIOR, np.where(bad, INSUFFICIENT, stitched.status)).reshape(shape)
    np.testing.assert_array_equal(ds["status"].values[:, :, :, :, 0], np.moveaxis(expect, 0, 2))


# ---------------------------------------------------------------- model climatology

def _brute_force_means(forecasts, years, window, lead_range, label_offset):
    """Per-cell (sum, valid count, expected count) of forecast days whose label is in the window, by explicit loops."""
    total = np.zeros(2)
    count = np.zeros(2)
    expected = 0
    for fc in forecasts:
        n_init, n_member, _, n_leads = fc.tenths.shape[0], fc.tenths.shape[1], 2, fc.tenths.shape[-1]
        for i in range(n_init):
            init = pd.Timestamp(int(fc.init_year[i]), int(fc.init_month[i]), int(fc.init_day[i]))
            for lead in range(lead_range[0], min(lead_range[1], n_leads) + 1):
                label = init + pd.Timedelta(days=lead - 1 + label_offset)
                if not (years[0] <= label.year <= years[1] and window[0] <= (label.month, label.day) <= window[1]):
                    continue
                expected += n_member
                for member in range(n_member):
                    valid = ~fc.missing[i, member, :, 0, lead - 1]
                    total += np.where(valid, fc.tenths[i, member, :, 0, lead - 1], 0)
                    count += valid
    return total, count, expected


def _climatology_forecasts():
    """Two synthetic forecast sets (2 members, 2 cells) with inits inside, before and outside the reference years."""
    rng = np.random.default_rng(11)
    out = []
    for inits in (["2000-05-30", "2000-06-10", "2005-06-10"], ["2001-06-28", "2001-12-01"]):
        values = rng.integers(0, 120, (len(inits), 2, 2, 1, 6)).astype(np.int32)
        fc = make_forecast(values, inits, lead_label="end", members=True)
        fc.missing[...] = rng.random(fc.missing.shape) < 0.03
        fc.source = "toy"
        out.append(fc)
    return out


def _thresholds_definition():
    """A definition naming a 5-day mean total and a daily mean over June."""
    june = (m.DayRef(month=6, day=1), m.DayRef(month=6, day=30))
    specs = (("wet5", m.ThresholdSpec("mean_total", days=5, during=june)),
             ("daily", m.ThresholdSpec("mean_daily", during=june)))
    return definition(ANY_WET, JJAS, specs)


def test_model_climatology_matches_hand_computed_means():
    """Means pool members, inits of the reference years and the leads of the window whose labels fall in `during`."""
    forecasts = _climatology_forecasts()
    out = climatology.compute_thresholds_from_forecasts(_thresholds_definition(), forecasts, (2000, 2001), (2, 5))
    total, count, expected = _brute_force_means(forecasts, (2000, 2001), ((6, 1), (6, 30)), (2, 5), 1)
    assert expected > 0 and (expected - count).max() / expected < 0.1
    np.testing.assert_allclose(out["daily"].values.reshape(2), total / 10 / count)
    np.testing.assert_allclose(out["wet5"].values.reshape(2), 5 * total / 10 / count)
    assert out["daily"].dims == ("lat", "lon")
    assert out["daily"].attrs["source"] == "model:toy"
    assert out["daily"].attrs["years"] == [2000, 2001]
    assert out["daily"].attrs["lead_window"] == [2, 5]
    assert out["daily"].attrs["units"] == "mm"


def test_model_climatology_streams_and_flags_missing_cells():
    """An iterator of forecasts works, and a cell with more than 10 % of its days missing is NaN."""
    forecasts = _climatology_forecasts()
    forecasts[0].missing[:, :, 1] = True
    out = climatology.compute_thresholds_from_forecasts(_thresholds_definition(), iter(forecasts), (2000, 2001))
    values = out["daily"].values.reshape(2)
    assert np.isfinite(values[0]) and np.isnan(values[1])


def test_model_climatology_rejects_quantiles_and_empty_input():
    """quantile_total has no model climatology, and no forecasts is an error."""
    spec = (("q", m.ThresholdSpec("quantile_total", days=3, quantile=0.9)),)
    with pytest.raises(NotImplementedError, match="quantile_total"):
        climatology.compute_thresholds_from_forecasts(definition(ANY_WET, JJAS, spec), _climatology_forecasts(), (2000, 2001))
    with pytest.raises(ValueError, match="no forecasts"):
        climatology.compute_thresholds_from_forecasts(_thresholds_definition(), [], (2000, 2001))

