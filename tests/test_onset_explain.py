"""momp.onset.explain: the Anatomy of one cell equals the engine's result, and the log says why in words."""
import time

import numpy as np
import pytest

from momp.onset import definition as dfn
from momp.onset import engine, explain
from momp.onset import model as m
from momp.onset.model import Status
from test_onset_engine import ANY_WET, TWO_WET, day_index, definition, make_forecast, make_rain, season
from test_onset_forecast_layer import JJAS, case, hist_with
from test_onset_oracle_agreement import random_data, random_definition


def assert_internally_consistent(a):
    """Every candidate's state follows from its checks, and the checks' numbers agree with their outcomes."""
    days = [c.day for c in a.candidates]
    assert days == list(range(days[0], days[0] + len(days))) if days else True
    for i, c in enumerate(a.candidates):
        outcomes = [ch.outcome for ch in c.checks]
        expected = "F" if "fail" in outcomes else "U" if "unknown" in outcomes else "T"
        assert c.state == expected
        assert (c.state == "F") == (c.decisive is not None and c.checks[c.decisive].outcome == "fail")
        assert (i == len(a.candidates) - 1) or c.state == "F"
        if c.state == "U":
            assert set(c.flags) == {f for ch in c.checks if ch.outcome == "unknown" for f in ch.flags}
        for ch in c.checks:
            if ch.role == "require" and ch.outcome != "unknown":
                value, limit = round(ch.value * 10), round(ch.limit * 10)
                if ch.kind == "wet_days":
                    value, limit = round(ch.value), round(ch.limit)
                holds = {"ge": value >= limit, "gt": value > limit, "le": value <= limit, "lt": value < limit}[ch.op]
                assert holds == (ch.outcome == "pass")
            if ch.role == "require" and ch.outcome == "unknown":
                assert ch.value is None and ch.flags
            if ch.role == "veto":
                assert (ch.spell is not None) == (ch.outcome == "fail")
                if ch.spell is not None:
                    assert ch.spell.length >= ch.spell_days and ch.first <= ch.spell.first <= ch.last


def assert_result_matches(a, ds, index):
    """The Anatomy's status, onset day and (forecasts) lead equal the engine's for the cell."""
    assert a.result.status == Status(int(ds["status"].values[index]))
    if "onset_index" in ds:
        want = int(ds["onset_index"].values[index])
        assert (a.result.day if a.result.day is not None else -1) == want
    else:
        lead = float(ds["onset_lead"].values[index])
        assert (a.result.onset_lead is None) == np.isnan(lead)
        if a.result.onset_lead is not None:
            assert a.result.onset_lead == lead
        want = ds["onset_date"].values[index]
        if a.result.date is None:
            assert np.isnat(want)
        else:
            assert np.datetime64(a.result.date, "ns") == want


@pytest.mark.parametrize("seed", range(40))
def test_observed_anatomy_equals_the_engine_on_random_cells(seed):
    """Across random definitions and cells (missing days, NaN thresholds) the traced result is detect_observed's."""
    rng = np.random.default_rng(seed)
    defn = random_definition(rng)
    n_lat, n_lon, n_time = 3, 2, 366
    tenths, miss, thresholds = random_data(rng, n_lat * n_lon, n_time)
    rain = make_rain(tenths.reshape(n_lat, n_lon, n_time), missing=miss.reshape(n_lat, n_lon, n_time))
    thr = {k: v.reshape(n_lat, n_lon) for k, v in thresholds.items()}
    ds = engine.detect_observed(defn, rain, thresholds=thr, years=[2000], backend="reference")
    for i in range(n_lat):
        for j in range(n_lon):
            a = explain.trace_observed(defn, rain, lat=rain.coords["lat"][i], lon=rain.coords["lon"][j], year=2000,
                                       thresholds=thr)
            assert_result_matches(a, ds, (0, 0, i, j))
            assert a.location == {"lat": float(rain.coords["lat"][i]), "lon": float(rain.coords["lon"][j])}
            assert_internally_consistent(a)
            explain.explain(a)
            explain.explain(a, compress=False)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("stitched", [False, True])
def test_forecast_anatomy_equals_the_engine_on_random_cells(seed, stitched):
    """Alone or stitched onto history (PRIOR included), the traced result is detect_forecast's for every init and member."""
    rng = np.random.default_rng(1000 + seed)
    defn = random_definition(rng)
    n_cell, n_member, n_leads = 3, 2, 30
    inits = ["2000-03-01", "2000-03-18", "2000-04-04", "2000-04-21"]
    hist_t, hist_m, thresholds = random_data(rng, n_cell, 366)
    fc_t, fc_m, _ = random_data(rng, len(inits) * n_member * n_cell, n_leads)
    fc = make_forecast(fc_t.reshape(len(inits), n_member, n_cell, 1, n_leads), inits, members=True)
    fc.missing[...] = fc_m.reshape(fc.missing.shape)
    hist = make_rain(hist_t.reshape(n_cell, 1, 366))
    hist.missing[...] = hist_m.reshape(hist.missing.shape)
    hist.coords.update({d: fc.coords[d] for d in ("lat", "lon")})
    thr = {k: v[:n_cell].reshape(n_cell, 1) for k, v in thresholds.items()}
    use = hist if stitched else None
    ds = engine.detect_forecast(defn, fc, thresholds=thr, history=use, lead_window=(1, n_leads), backend="reference")
    for ii, init in enumerate(inits):
        for mi in range(n_member):
            for ci in range(n_cell):
                a = explain.trace_forecast(defn, fc, init=init, lat=fc.coords["lat"][ci], lon=0.0, member=mi,
                                           history=use, thresholds=thr, lead_window=(1, n_leads))
                assert_result_matches(a, ds, (ii, mi, 0, ci, 0))
                assert_internally_consistent(a)
                explain.explain(a)


def test_observed_values_match_the_raw_series():
    """The totals and counts shown for each candidate are the raw series' own sums."""
    rng = np.random.default_rng(3)
    defn = dfn.load("icpac")
    tenths, miss, _ = random_data(rng, 1, 366)
    rain = make_rain(tenths.reshape(1, 1, 366), missing=np.zeros((1, 1, 366), dtype=bool))
    a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000)
    for c in a.candidates:
        total = next(ch for ch in c.checks if ch.kind == "total")
        if total.value is not None:
            assert total.value == tenths[0, total.first:total.last + 1].sum() / 10


def test_cell_selection_and_errors():
    """The nearest cell is used, a point farther than half a grid step is an error, and bad inits and members are named."""
    rain = make_rain(np.zeros((2, 2, 366)))
    defn = definition(ANY_WET, (season("s", (3, 1), (5, 31)),))
    a = explain.trace_observed(defn, rain, lat=-4.9, lon=0.1, year=2000)
    assert a.location == {"lat": 0.0, "lon": 0.0}
    with pytest.raises(ValueError, match="half a grid step"):
        explain.trace_observed(defn, rain, lat=30.0, lon=0.0, year=2000)
    hist, fc = case(hist_with([]), np.zeros((1, 2, 1, 1, 8), dtype=np.int32), ["2000-06-10"], members=True)
    with pytest.raises(ValueError, match="no init"):
        explain.trace_forecast(definition(ANY_WET, JJAS), fc, init="2000-06-11", lat=0, lon=0, member=0)
    with pytest.raises(ValueError, match="member"):
        explain.trace_forecast(definition(ANY_WET, JJAS), fc, init="2000-06-10", lat=0, lon=0)


def test_trace_of_a_realistic_cell_takes_under_a_second():
    """One cell of a 0.25 degree year (ICPAC, with a threshold-free rule) traces and explains in well under 1 s."""
    rng = np.random.default_rng(5)
    shape = (60, 60, 366)
    values = np.where(rng.random(shape) < 0.3, rng.gamma(0.8, 70.0, shape), 0.0)
    rain = make_rain(np.rint(values).astype(np.int32))
    defn = dfn.load("icpac")
    start = time.perf_counter()
    a = explain.trace_observed(defn, rain, lat=rain.coords["lat"][30], lon=rain.coords["lon"][30], year=2000)
    text = explain.explain(a)
    assert time.perf_counter() - start < 1.0
    assert "Result:" in text


# ---------------------------------------------------------------- the log in words

def icpac_scenario():
    """A year of ICPAC rain with a total miss, an early dry-spell veto, a missing day and a late onset."""
    template = make_rain(np.zeros((1, 1, 366)))
    values = np.zeros((1, 1, 366), dtype=np.int32)

    def put(date, mm):
        """Set the rain of one ISO date in mm."""
        values[..., day_index(template, date)] = int(mm * 10)

    for date, mm in [("05-03", 6), ("05-04", 9), ("05-05", 12), ("05-18", 5), ("05-19", 5), ("05-20", 5.2),
                     ("06-02", 8), ("06-03", 9), ("06-04", 9), ("06-20", 8), ("06-21", 12), ("06-22", 6)]:
        put(f"2000-{date}", mm)
    for day in range(5, 12):
        put(f"2000-06-{day:02d}", 3)
    for day in (24, 27, 30):
        put(f"2000-06-{day}", 12)
    for day in range(1, 31, 3):
        put(f"2000-07-{day:02d}", 4)
    rain = make_rain(values)
    rain.missing[..., day_index(template, "2000-06-12")] = True
    return rain


def test_explain_reasons_in_words_and_compression():
    """Every reason kind appears with its numbers; compression merges runs failing the same way and only those."""
    a = explain.trace_observed(dfn.load("icpac"), icpac_scenario(), lat=-10, lon=-10, year=2000)
    full = explain.explain(a, compress=False)
    short = explain.explain(a)
    assert "ICPAC" in short.splitlines()[0] and "icpac v1" in short.splitlines()[0]
    assert "Season: main 2000" in short and "lat -10" in short
    assert "Onset is the first day from 1 May to 30 September" in short
    assert "18 May  ✗ 3-day total 15.2 mm, needs more than 20 mm" in full
    assert "3 May   ✗ followed by a dry run of 12 days (6-17 May) within 4-24 May" in full
    assert "1 May   ✗ day 1 not wet (0.0 mm, wet needs at least 1 mm)" in full
    assert "20 Jun  ✓ wet day; 3-day total 26.0 mm (more than 20 mm); no dry run of at least 7 days within 21 Jun - 11 Jul" \
        in full
    assert "6-17 May        ✗ 12 days: day 1 not wet" in short
    assert "18-20 May       ✗ 3 days: 3-day total 5.2-15.2 mm, needs more than 20 mm" in short
    assert "21 May - 1 Jun  ✗ 12 days: day 1 not wet" in short
    assert "10-12 Jun       ✗ 3 days: followed by a dry run of 7 days (13-19 Jun)" in short
    assert len(short.splitlines()) < len(full.splitlines()) / 2
    assert short.splitlines()[-1] == "Result: ONSET on 20 Jun 2000"
    assert a.result.date == np.datetime64("2000-06-20") and a.result.status == Status.ONSET


def test_explain_missing_day_and_not_found():
    """A missing day inside a needed span is named; a season without onset ends in NOT_FOUND."""
    rain = icpac_scenario()
    rain.missing[..., day_index(rain, "2000-06-25")] = True
    a = explain.trace_observed(dfn.load("icpac"), rain, lat=-10, lon=-10, year=2000)
    text = explain.explain(a, compress=False)
    assert "20 Jun  ✗ followed by a dry run" not in text
    unknown = [c for c in a.candidates if c.state == "U"]
    assert unknown and a.result.status == Status.INSUFFICIENT_DATA
    assert "needs " in text and "missing day (25 Jun)" in text
    assert text.splitlines()[-1].startswith("Result: INSUFFICIENT_DATA")

    dry = make_rain(np.zeros((1, 1, 366)))
    a = explain.trace_observed(dfn.load("icpac"), dry, lat=-10, lon=-10, year=2000)
    assert explain.explain(a).splitlines()[-1].startswith("Result: NOT_FOUND")
    assert "✗ 153 days: day 1 not wet" in explain.explain(a)


def test_explain_horizon_and_forecast_header():
    """A forecast that ends before a candidate's span says what it needs and where the forecast ends."""
    values = np.zeros((1, 1, 1, 21), dtype=np.int32)
    values[..., 0:3] = 100
    values[..., [5, 8, 11, 14, 17, 20]] = 20
    fc = make_forecast(values, ["2020-06-01"])
    a = explain.trace_forecast(dfn.load("icpac"), fc, init="2020-06-01", lat=0, lon=0)
    text = explain.explain(a)
    assert "1 Jun  ? needs 2-22 Jun, past the forecast (last lead 21 Jun)" in text
    assert "Forecast: init 1 Jun 2020, lead label start, forecast searched alone" in text
    assert text.splitlines()[-1].startswith("Result: CENSORED")
    assert a.result.status == Status.CENSORED and a.result.onset_lead is None
    assert a.data_end == 21 and a.split == 0


def test_explain_stitched_prior_and_back_dated():
    """History-decided onsets say so and carry a lead of 0 or less; back-dated onsets are named as such."""
    hist, fc = case(hist_with(["2000-06-05"]), np.zeros((1, 1, 1, 12), dtype=np.int32), ["2000-06-10"])
    a = explain.trace_forecast(definition(ANY_WET, JJAS), fc, init="2000-06-10", lat=0, lon=0, history=hist,
                               lead_window=(1, 12))
    text = explain.explain(a)
    assert a.result.status == Status.PRIOR and a.result.onset_lead == -4 and a.result.decided_by_history
    assert "Result: PRIOR - the onset was 5 Jun 2000, before the forecast starts" in text
    assert "onset lead -4 (before lead 1, so the observed history decided it)" in text
    assert "observed history stitched in before lead 1" in text
    assert a.split == day_index(hist, "2000-06-10") and a.source[0] == "observed" \
        and a.source[a.split - a.days[0]] == "forecast"

    values = np.zeros((1, 1, 1, 12), dtype=np.int32)
    values[..., 0] = 100
    hist, fc = case(hist_with(["2000-06-09"]), values, ["2000-06-10"])
    a = explain.trace_forecast(definition(TWO_WET, JJAS), fc, init="2000-06-10", lat=0, lon=0, history=hist,
                               lead_window=(1, 12))
    text = explain.explain(a)
    assert a.result.status == Status.ONSET and a.result.onset_lead == 0
    assert "onset lead 0 (back-dated into the observed history)" in text
    assert "9 Jun" in text and "✓" in text


def test_explain_curve_and_first_wet_day_report():
    """Curve rules explain the accumulation and minimum; first-wet-day reports name both days."""
    values = np.zeros((1, 1, 366), dtype=np.int32)
    values[..., 100:200] = 60
    values[..., 0:100:7] = 30
    rain = make_rain(values)
    curve = dfn.load("liebmann_marengo")
    a = explain.trace_observed(curve, rain, lat=-10, lon=-10, year=2000, thresholds={"mean_daily_rain": 2.0})
    text = explain.explain(a)
    assert a.curve is not None and not a.candidates
    assert "mean_daily_rain = 2 mm/day (compared as 2000 thousandths of a mm)" in text
    assert "Lowest value" in text and "The onset is the day after the minimum." in text
    assert a.result.day == a.curve.min_day + 1 and a.curve.min_mm < 0
    ds = engine.detect_observed(curve, rain, thresholds={"mean_daily_rain": 2.0}, backend="reference")
    assert_result_matches(a, ds, (0, 0, 0, 0))

    rule = m.FirstPassage((m.WetDays(m.Span(1, 3), "ge", 2),), report="first_wet_day", report_span=m.Span(1, 3))
    d3 = definition(rule, (season("s", (3, 1), (5, 31)),))
    v = np.zeros((1, 1, 366), dtype=np.int32)
    r3 = make_rain(v)
    v[..., day_index(r3, "2000-04-02")] = 100
    v[..., day_index(r3, "2000-04-03")] = 100
    a = explain.trace_observed(d3, make_rain(v), lat=-10, lon=-10, year=2000)
    text = explain.explain(a)
    assert "decided at candidate 1 Apr; reported on the first wet day 2 Apr" in text


def test_thresholds_report_the_integers_compared():
    """Named thresholds list their mm value and the tenths each comparator really used."""
    rule = m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("gt", m.Amount(threshold="amt"))),
                           m.Total(m.Span(1, 1), m.Comparison("le", m.Amount(threshold="amt")))))
    defn = definition(rule, (season("s", (3, 1), (5, 31)),), (("amt", m.ThresholdSpec("mean_daily")),))
    a = explain.trace_observed(defn, make_rain(np.zeros((1, 1, 366))), lat=-10, lon=-10, year=2000,
                               thresholds={"amt": 3.27})
    (t,) = a.thresholds
    assert t.name == "amt" and t.mm == 3.27 and t.compared == {"gt": 32, "le": 32}
    assert "amt = 3.27 mm (compared as more than 3.2 mm (32 tenths), at most 3.2 mm (32 tenths))" in explain.explain(a)

    ge = m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("ge", m.Amount(threshold="amt"))),))
    a = explain.trace_observed(definition(ge, (season("s", (3, 1), (5, 31)),), (("amt", m.ThresholdSpec("mean_daily")),)),
                               make_rain(np.zeros((1, 1, 366))), lat=-10, lon=-10, year=2000, thresholds={"amt": 3.27})
    assert a.thresholds[0].compared == {"ge": 33}
