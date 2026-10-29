"""Engine: jobs from seasons and leads, observed and forecast detection, regional averaging, climatology, save."""
import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.onset import engine
from momp.onset import model as m

ONSET, NOT_FOUND, CENSORED, PRIOR, INSUFFICIENT = (int(s) for s in m.Status)
WET_1MM = m.Comparison("ge", m.Amount(tenths=10))
ANY_WET = m.FirstPassage((m.WetDays(m.Span(1, 1), "ge", 1),))
TWO_WET = m.FirstPassage((m.WetDays(m.Span(1, 2), "ge", 2),))


def season(name, start, end):
    """A fixed-bounds season from (month, day) pairs."""
    return m.Season(name, m.DayRef(month=start[0], day=start[1]), m.DayRef(month=end[0], day=end[1]))


def definition(rule, seasons, thresholds=()):
    """A Definition around a rule and seasons."""
    return m.Definition("t", 1, "t", "t", "t", "draft", "", WET_1MM, seasons, rule, thresholds)


def labels(start, n, calendar="standard"):
    """Year, month, day arrays and the time coordinate for n daily labels from a start date."""
    if calendar == "standard":
        index = pd.date_range(start, periods=n, freq="D")
    else:
        index = xr.date_range(start, periods=n, freq="D", calendar=calendar, use_cftime=True)
    return (np.array(index.year), np.array(index.month), np.array(index.day)), np.asarray(index)


def make_rain(values, start="2000-01-01", calendar="standard", spatial=("lat", "lon"), missing=None):
    """A DailyRain from tenths shaped (*spatial, time)."""
    values = np.asarray(values, dtype=np.int32)
    (year, month, day), time = labels(start, values.shape[-1], calendar)
    miss = np.zeros(values.shape, dtype=bool) if missing is None else np.asarray(missing)
    coords = {"time": time}
    for i, dim in enumerate(spatial):
        coords[dim] = np.arange(values.shape[i], dtype=float) * 10 - 10
    return m.DailyRain(values, miss, (*spatial, "time"), coords, year, month, day, calendar, "00 UTC, labelled by start")


def day_index(rain, date):
    """Index of an ISO date in the rain's time axis."""
    y, mo, d = (int(p) for p in date.split("-"))
    return int(np.nonzero((rain.year == y) & (rain.month == mo) & (rain.day == d))[0][0])


def rain_on(rain_shape, rain_template, dates, amount=100):
    """Zeros shaped (*spatial, time) with `amount` tenths on the given ISO dates for every cell."""
    values = np.zeros(rain_shape, dtype=np.int32)
    for date in dates:
        values[..., day_index(rain_template, date)] = amount
    return values


# ---------------------------------------------------------------- observed

def test_season_crossing_year_end_and_horizon():
    """A Nov-Feb season belongs to the year of its start; running out of data gives CENSORED."""
    template = make_rain(np.zeros((1, 1, 4 * 366)))
    values = rain_on(template.tenths.shape, template, ["2000-12-15", "2001-01-10", "2002-01-20"])
    rain = make_rain(values)
    defn = definition(ANY_WET, (season("nd", (11, 1), (2, 28)),))
    ds = engine.detect_observed(defn, rain, years=[2000, 2001, 2002, 2003])
    assert ds["onset_date"].dims == ("year", "season", "lat", "lon")
    flat = ds["onset_date"].values[:, 0, 0, 0]
    assert flat[0] == np.datetime64("2000-12-15")
    assert flat[1] == np.datetime64("2002-01-20")
    assert ds["status"].values[:, 0, 0, 0].tolist() == [ONSET, ONSET, NOT_FOUND, NOT_FOUND]
    assert ds["onset_index"].values[0, 0, 0, 0] == day_index(rain, "2000-12-15")
    assert ds["onset_index"].values[2, 0, 0, 0] == -1
    assert np.isnat(ds["onset_date"].values[2, 0, 0, 0])

    censored = engine.detect_observed(definition(TWO_WET, (season("nd", (11, 1), (2, 28)),)), rain, years=[2003])
    assert censored["status"].values.ravel().tolist() == [CENSORED]


def test_status_attrs_and_extra_attrs():
    """status carries CF flags; caller attrs are merged."""
    rain = make_rain(np.zeros((1, 1, 366)))
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), rain, years=[2000],
                                attrs={"input_source": "unit test"})
    assert ds["status"].dtype == np.int8
    assert ds["status"].attrs["flag_values"].tolist() == [0, 1, 2, 3, 4]
    assert ds["status"].attrs["flag_meanings"] == "ONSET NOT_FOUND CENSORED PRIOR INSUFFICIENT_DATA"
    assert ds.attrs["input_source"] == "unit test" and ds.attrs["definition_id"] == "t"


def test_multi_season_and_spatial_cells():
    """Each (year, season, cell) is searched on its own window and cell."""
    template = make_rain(np.zeros((1, 1, 366)))
    values = np.zeros((2, 3, 366), dtype=np.int32)
    values[0, 0, day_index(template, "2000-03-05")] = 100
    values[1, 2, day_index(template, "2000-10-02")] = 100
    values[1, 2, day_index(template, "2000-03-07")] = 100
    rain = make_rain(values)
    defn = definition(ANY_WET, (season("long", (2, 1), (5, 31)), season("short", (9, 1), (12, 31))))
    ds = engine.detect_observed(defn, rain, years=[2000])
    assert ds["status"].dims == ("year", "season", "lat", "lon")
    assert ds["season"].values.tolist() == ["long", "short"]
    onset = ds["onset_date"].values[0]
    assert onset[0, 0, 0] == np.datetime64("2000-03-05")
    assert onset[0, 1, 2] == np.datetime64("2000-03-07")
    assert onset[1, 1, 2] == np.datetime64("2000-10-02")
    status = ds["status"].values[0]
    assert status[0, 0, 1] == NOT_FOUND and status[1, 0, 0] == NOT_FOUND
    assert (ds["lat"].values == rain.coords["lat"]).all()


def test_season_start_outside_data_is_insufficient():
    """A year whose season starts outside the data is not searched."""
    rain = make_rain(np.zeros((1, 1, 366)))
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), rain, years=[1999, 2000])
    assert ds["status"].values.ravel().tolist() == [INSUFFICIENT, NOT_FOUND]
    jobs, layout = engine.build_observed_jobs(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), rain, [1999, 2000])
    assert layout.preset.tolist() == [INSUFFICIENT, engine.NO_PRESET]
    assert jobs.data_end.tolist() == [366, 366]


def test_missing_rain_in_window_is_insufficient_data():
    """A missing day inside a requirement span stops the search."""
    template = make_rain(np.zeros((1, 1, 366)))
    values = rain_on(template.tenths.shape, template, ["2000-04-10"])
    missing = np.zeros(values.shape, dtype=bool)
    missing[..., day_index(template, "2000-04-01")] = True
    rain = make_rain(values, missing=missing)
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), rain, years=[2000])
    assert ds["status"].values.ravel().tolist() == [INSUFFICIENT]


def test_thresholds_per_cell_and_nan_cell():
    """Named thresholds are per cell; a NaN threshold gives INSUFFICIENT_DATA without searching."""
    rule = m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("ge", m.Amount(threshold="amt"))),))
    defn = definition(rule, (season("s", (3, 1), (5, 31)),), (("amt", m.ThresholdSpec("mean_daily")),))
    template = make_rain(np.zeros((1, 3, 366)))
    values = rain_on(template.tenths.shape, template, ["2000-04-10"], amount=50)
    rain = make_rain(values)
    thr = xr.DataArray([[3.0, 8.0, np.nan]], dims=("lat", "lon"))
    ds = engine.detect_observed(defn, rain, years=[2000], thresholds={"amt": thr})
    assert ds["status"].values[0, 0, 0].tolist() == [ONSET, NOT_FOUND, INSUFFICIENT]
    ds2 = engine.detect_observed(defn, rain, years=[2000], thresholds={"amt": thr.values})
    assert (ds2["status"].values == ds["status"].values).all()
    with pytest.raises(KeyError):
        engine.detect_observed(defn, rain, years=[2000])


def test_curve_observed():
    """The curve rule runs on every cell of a season."""
    template = make_rain(np.zeros((1, 1, 366)))
    values = np.zeros((1, 1, 366), dtype=np.int32)
    values[..., day_index(template, "2000-06-15"):day_index(template, "2000-09-30")] = 100
    rain = make_rain(values)
    defn = definition(m.Curve("base"), (season("s", (6, 1), (9, 30)),), (("base", m.ThresholdSpec("mean_daily")),))
    ds = engine.detect_observed(defn, rain, years=[2000], thresholds={"base": np.array([[1.0]])})
    assert ds["onset_date"].values.ravel()[0] == np.datetime64("2000-06-15")


def test_derived_seasons_from_climatology():
    """Derived seasons take their bounds per cell from the climatology; absent seasons are NOT_FOUND."""
    template = make_rain(np.zeros((1, 2, 3 * 365)))
    values = np.zeros((1, 2, 3 * 365), dtype=np.int32)
    values[0, :, day_index(template, "2000-04-20")] = 100
    values[0, 0, day_index(template, "2000-10-12")] = 100
    values[0, :, day_index(template, "2001-04-25")] = 100
    rain = make_rain(values)
    defn = definition(
        ANY_WET,
        m.DerivedSeasons("harmonic", m.DayRef(climatological="onset", offset_days=-10),
                         m.DayRef(climatological="cessation")))
    clim = xr.Dataset({
        "n_seasons": (("lat", "lon"), [[2, 1]]),
        "onset_doy": (("season", "lat", "lon"), [[[100.0, 100.0]], [[270.0, np.nan]]]),
        "cessation_doy": (("season", "lat", "lon"), [[[160.0, 160.0]], [[350.0, np.nan]]])})
    with pytest.raises(ValueError):
        engine.detect_observed(defn, rain, years=[2000])
    ds = engine.detect_observed(defn, rain, years=[2000, 2001], season_climatology=clim)
    assert ds["status"].dims == ("year", "season", "lat", "lon")
    status = ds["status"].values
    assert status[0, 0].ravel().tolist() == [ONSET, ONSET]
    assert status[0, 1].ravel().tolist() == [ONSET, NOT_FOUND]
    assert ds["onset_date"].values[0, 0, 0, 0] == np.datetime64("2000-04-20")
    assert ds["onset_date"].values[0, 1, 0, 0] == np.datetime64("2000-10-12")
    assert ds["onset_date"].values[1, 0, 0, 1] == np.datetime64("2001-04-25")
    assert status[1, 1].ravel().tolist() == [NOT_FOUND, NOT_FOUND]


def test_noleap_calendar_dates_are_cftime():
    """Non-standard calendars give cftime onset dates."""
    template = make_rain(np.zeros((1, 1, 730)), "2001-01-01", "noleap")
    values = np.zeros((1, 1, 730), dtype=np.int32)
    values[..., day_index(template, "2001-03-01")] = 100
    rain = make_rain(values, "2001-01-01", "noleap")
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (2, 25), (5, 31)),)), rain, years=[2001])
    date = ds["onset_date"].values.ravel()[0]
    assert isinstance(date, cftime.datetime) and (date.year, date.month, date.day) == (2001, 3, 1)
    assert date.calendar == "noleap"


def test_series_chunking_matches_unchunked(monkeypatch):
    """Splitting the series into chunks of prefix tables does not change results."""
    rng = np.random.default_rng(3)
    values = np.where(rng.random((4, 5, 800)) < 0.2, 80, 0)
    rain = make_rain(values)
    defn = definition(m.FirstPassage((m.Total(m.Span(1, 3), m.Comparison("ge", m.Amount(tenths=150))),),
                                     (m.Veto(m.DryRun(4), m.Span(2, 10), "inside"),)),
                      (season("s", (3, 1), (6, 30)),))
    base = engine.detect_observed(defn, rain, years=[2000, 2001])
    monkeypatch.setattr(engine, "MAX_TABLE_CELLS", 800 * 3)
    chunked = engine.detect_observed(defn, rain, years=[2000, 2001])
    assert (base["onset_index"].values == chunked["onset_index"].values).all()
    assert (base["status"].values == chunked["status"].values).all()


# ---------------------------------------------------------------- forecast

def make_forecast(values, inits, lead_label="start", calendar="standard", members=False, spatial=("lat", "lon")):
    """A ForecastRain from tenths shaped (init, [member], *spatial, lead)."""
    values = np.asarray(values, dtype=np.int32)
    if calendar == "standard":
        index = pd.DatetimeIndex(inits)
    else:
        index = xr.CFTimeIndex([cftime.datetime(*map(int, s.split("-")), calendar=calendar) for s in inits])
    dims = ("init_time", *(("member",) if members else ()), *spatial, "lead")
    coords = {"init_time": np.asarray(index), "lead": np.arange(1, values.shape[-1] + 1)}
    for dim in spatial:
        coords[dim] = np.arange(values.shape[dims.index(dim)], dtype=float)
    if members:
        coords["member"] = np.arange(values.shape[1])
    return m.ForecastRain(values, np.zeros(values.shape, dtype=bool), dims, coords, np.array(index.year),
                          np.array(index.month), np.array(index.day), lead_label, calendar, "00 UTC, labelled by start")


JJAS = (season("jjas", (6, 1), (9, 30)),)


@pytest.mark.parametrize("lead_label,date", [("start", "2020-06-03"), ("end", "2020-06-04")])
def test_forecast_lead_labels_and_season_window(lead_label, date):
    """Only leads whose date label is in the season are candidates; the onset date follows the lead label."""
    values = np.zeros((2, 1, 1, 40), dtype=np.int32)
    values[:, ..., 2] = 100
    values[:, ..., 9] = 100
    fc = make_forecast(values, ["2020-05-25", "2020-12-01"], lead_label)
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc)
    assert ds["status"].dims == ("init_time", "season", "lat", "lon")
    assert ds["status"].values[:, 0, 0, 0].tolist() == [ONSET, NOT_FOUND]
    assert ds["onset_lead"].values[:, 0, 0, 0].tolist() == [10, -1]
    assert ds["onset_date"].values[0, 0, 0, 0] == np.datetime64(date)
    assert np.isnat(ds["onset_date"].values[1, 0, 0, 0])


@pytest.mark.parametrize("lead_label,first_lead", [("start", 8), ("end", 7)])
def test_forecast_first_candidate_lead_depends_on_label(lead_label, first_lead):
    """With rain only before the season, the first candidate (and the CENSORED horizon check) starts at the first in-season lead."""
    values = np.zeros((1, 1, 1, 40), dtype=np.int32)
    fc = make_forecast(values, ["2020-05-25"], lead_label)
    jobs, layout = engine.build_forecast_jobs(definition(ANY_WET, JJAS), fc, (1, 30))
    assert jobs.start.tolist() == [first_lead - 1] and jobs.end.tolist() == [29] and jobs.data_end.tolist() == [40]
    values[..., first_lead - 2] = 100
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), make_forecast(values, ["2020-05-25"], lead_label))
    assert ds["status"].values.ravel().tolist() == [NOT_FOUND]
    values[..., first_lead - 1] = 100
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), make_forecast(values, ["2020-05-25"], lead_label))
    assert ds["onset_lead"].values.ravel().tolist() == [first_lead]


def test_forecast_horizon_is_censored():
    """A window running to the last lead leaves the final candidates undecided."""
    fc = make_forecast(np.zeros((1, 1, 1, 40), dtype=np.int32), ["2020-06-01"])
    ds = engine.detect_forecast(definition(TWO_WET, JJAS), fc, lead_window=(1, 40))
    assert ds["status"].values.ravel().tolist() == [CENSORED]
    ds = engine.detect_forecast(definition(TWO_WET, JJAS), fc, lead_window=(1, 30))
    assert ds["status"].values.ravel().tolist() == [NOT_FOUND]


def test_forecast_members_and_unrestricted_window():
    """Members keep their own dim and onset; restrict_to_season=False searches the whole lead window."""
    values = np.zeros((1, 2, 1, 2, 40), dtype=np.int32)
    values[0, 0, :, :, 11] = 100
    values[0, 1, :, :, 13] = 100
    values[0, 1, :, 1, 4] = 100
    fc = make_forecast(values, ["2020-05-25"], members=True)
    ds = engine.detect_forecast(definition(ANY_WET, JJAS), fc)
    assert ds["status"].dims == ("init_time", "member", "season", "lat", "lon")
    assert ds["onset_lead"].values[0, :, 0, 0, :].tolist() == [[12, 12], [14, 14]]
    free = engine.detect_forecast(definition(ANY_WET, JJAS), fc, restrict_to_season=False)
    assert free["season"].values.tolist() == ["window"]
    assert free["onset_lead"].values[0, :, 0, 0, :].tolist() == [[12, 12], [14, 5]]


def test_forecast_noleap_labels():
    """Lead labels follow the forecast's own calendar (no Feb 29 in noleap)."""
    values = np.zeros((1, 1, 1, 40), dtype=np.int32)
    values[..., 2] = 100
    fc = make_forecast(values, ["2020-02-27"], calendar="noleap")
    ds = engine.detect_forecast(definition(ANY_WET, (season("mam", (3, 1), (5, 31)),)), fc)
    assert ds["onset_lead"].values.ravel().tolist() == [3]
    date = ds["onset_date"].values.ravel()[0]
    assert isinstance(date, cftime.datetime) and (date.month, date.day) == (3, 1)
    std = make_forecast(values, ["2020-02-27"], calendar="standard")
    ds_std = engine.detect_forecast(definition(ANY_WET, (season("mam", (3, 1), (5, 31)),)), std)
    assert ds_std["status"].values.ravel().tolist() == [NOT_FOUND]
    assert ds_std["status"].attrs["flag_values"].tolist() == [0, 1, 2, 3, 4]


def test_forecast_season_wrapping_year_end():
    """An init in January can be inside the previous year's Nov-Feb season."""
    values = np.zeros((1, 1, 1, 45), dtype=np.int32)
    values[..., 5] = 100
    fc = make_forecast(values, ["2021-01-20"])
    ds = engine.detect_forecast(definition(ANY_WET, (season("nd", (11, 1), (2, 28)),)), fc)
    assert ds["onset_lead"].values.ravel().tolist() == [6]
    values[..., 5] = 0
    values[..., 40] = 100
    ds = engine.detect_forecast(definition(ANY_WET, (season("nd", (11, 1), (2, 28)),)), make_forecast(values, ["2021-01-20"]),
                                lead_window=(1, 45))
    assert ds["status"].values.ravel().tolist() == [NOT_FOUND]


def test_forecast_thresholds_broadcast_over_inits_and_members():
    """Per-cell thresholds apply to every init and member."""
    rule = m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("ge", m.Amount(threshold="amt"))),))
    defn = definition(rule, JJAS, (("amt", m.ThresholdSpec("mean_daily")),))
    values = np.zeros((2, 2, 1, 2, 40), dtype=np.int32)
    values[..., 15] = 50
    fc = make_forecast(values, ["2020-05-25", "2020-05-26"], members=True)
    ds = engine.detect_forecast(defn, fc, thresholds={"amt": np.array([[4.0, 6.0]])})
    assert (ds["status"].values[..., 0, 0, 0] == ONSET).all()
    assert (ds["status"].values[..., 0, 0, 1] == NOT_FOUND).all()


def test_forecast_derived_seasons_need_unrestricted_window():
    """Derived seasons are only supported with restrict_to_season=False."""
    defn = definition(ANY_WET, m.DerivedSeasons("harmonic", m.DayRef(climatological="onset"),
                                                m.DayRef(climatological="cessation")))
    fc = make_forecast(np.zeros((1, 1, 1, 10), dtype=np.int32), ["2020-05-25"])
    with pytest.raises(NotImplementedError):
        engine.detect_forecast(defn, fc)
    assert engine.detect_forecast(defn, fc, restrict_to_season=False)["status"].shape == (1, 1, 1, 1)


def test_unknown_backend_names():
    """Unknown backends are rejected; missing optional ones raise ImportError."""
    rain = make_rain(np.zeros((1, 1, 366)))
    defn = definition(ANY_WET, JJAS)
    with pytest.raises(ValueError):
        engine.detect_observed(defn, rain, years=[2000], backend="nope")


# ---------------------------------------------------------------- regional

def test_regional_cos_lat_and_equal_weights():
    """Weighted means are rounded half to even in tenths."""
    values = np.zeros((2, 1, 3), dtype=np.int32)
    values[0, 0] = [10, 20, 0]
    values[1, 0] = [20, 30, 0]
    rain = make_rain(values)
    rain.coords["lat"] = np.array([0.0, 60.0])
    mask = np.array([[True], [True]])
    equal = engine.regional(rain, mask, weights="equal")
    assert equal.dims == ("time",) and equal.tenths.tolist() == [15, 25, 0]
    assert equal.tenths.dtype == np.int32
    cos = engine.regional(rain, mask, weights="cos_lat")
    assert cos.tenths.tolist() == [round((10 + 0.5 * 20) / 1.5), round((20 + 0.5 * 30) / 1.5), 0]
    assert not equal.missing.any()


def test_regional_rounding_is_half_even():
    """A mean of exactly .5 tenths rounds to the even tenth."""
    rain = make_rain(np.array([[[1, 2, 3]], [[2, 3, 3]]]))
    out = engine.regional(rain, np.ones((2, 1), dtype=bool), weights="equal")
    assert out.tenths.tolist() == [2, 2, 3]


def test_regional_missing_rules_and_mask():
    """A day is missing only if every masked cell is missing; partial missing averages the rest."""
    values = np.array([[[10, 10, 10]], [[30, 30, 30]], [[90, 90, 90]]], dtype=np.int32)
    missing = np.array([[[False, True, True]], [[False, False, True]], [[True, True, True]]])
    rain = make_rain(values, missing=missing)
    rain.coords["lat"] = np.zeros(3)
    out = engine.regional(rain, np.array([[True], [True], [False]]), weights="equal")
    assert out.tenths.tolist() == [20, 30, 0]
    assert out.missing.tolist() == [False, False, True]
    with pytest.raises(ValueError):
        engine.regional(rain, np.zeros((3, 1), dtype=bool))
    with pytest.raises(ValueError):
        engine.regional(rain, np.ones((3, 1), dtype=bool), weights="bogus")


def test_regional_feeds_detection():
    """A regional series runs through detect_observed as a single cell."""
    template = make_rain(np.zeros((2, 2, 366)))
    values = rain_on(template.tenths.shape, template, ["2000-04-02"])
    rain = make_rain(values)
    series = engine.regional(rain, np.ones((2, 2), dtype=bool))
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), series, years=[2000])
    assert ds["status"].dims == ("year", "season")
    assert ds["onset_date"].values[0, 0] == np.datetime64("2000-04-02")


# ---------------------------------------------------------------- climatology and save

def test_climatological_onset_median_and_counts():
    """Median day of year over onset years only; NaN when fewer than half the years have an onset."""
    n_years = 6
    template = make_rain(np.zeros((1, 2, n_years * 366)), "2000-01-01")
    values = np.zeros((1, 2, n_years * 366), dtype=np.int32)
    onset_days = {2000: "04-10", 2001: "04-20", 2002: "05-10", 2003: "04-30"}
    for year, md in onset_days.items():
        values[0, :, day_index(template, f"{year}-{md}")] = 100
    values[0, 1, :] = 0
    for year in (2000, 2001):
        values[0, 1, day_index(template, f"{year}-04-15")] = 100
    rain = make_rain(values)
    defn = definition(ANY_WET, (season("s", (4, 1), (5, 31)),))
    out = engine.climatological_onset(defn, rain, years=range(2000, 2000 + n_years))
    assert out["onset_doy"].dims == ("season", "lat", "lon")
    assert out["n_onset_years"].values[0, 0].tolist() == [4, 2]
    doys = [pd.Timestamp(f"{year}-{md}").dayofyear for year, md in onset_days.items()]
    assert out["onset_doy"].values[0, 0, 0] == np.median(doys)
    assert np.isnan(out["onset_doy"].values[0, 0, 1])


def test_climatological_onset_season_crossing_year_end():
    """Onsets in the following calendar year count past the year length so medians stay meaningful."""
    template = make_rain(np.zeros((1, 1, 3 * 366)))
    values = np.zeros((1, 1, 3 * 366), dtype=np.int32)
    values[..., day_index(template, "2000-12-20")] = 100
    values[..., day_index(template, "2002-01-05")] = 100
    rain = make_rain(values)
    defn = definition(ANY_WET, (season("s", (11, 1), (2, 28)),))
    out = engine.climatological_onset(defn, rain, years=[2000, 2001])
    assert out["onset_doy"].values.ravel()[0] == np.mean([355, 365 + 5])
    assert out["n_onset_years"].values.ravel()[0] == 2


def test_save_round_trip(tmp_path):
    """save writes zarr that opens back with the same values and flags."""
    template = make_rain(np.zeros((1, 2, 366)))
    values = rain_on(template.tenths.shape, template, ["2000-04-02"])
    rain = make_rain(values)
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), rain, years=[2000],
                                attrs={"input_source": "x"})
    engine.save(ds, tmp_path / "out.zarr")
    back = xr.open_zarr(tmp_path / "out.zarr").load()
    assert (back["onset_date"].values == ds["onset_date"].values).all()
    assert (back["status"].values == ds["status"].values).all()
    assert back["status"].attrs["flag_meanings"] == ds["status"].attrs["flag_meanings"]
    assert back.attrs["input_source"] == "x"


def test_save_round_trip_noleap_with_missing_onset(tmp_path):
    """cftime onset dates are stored as ISO strings, empty where there is no onset."""
    template = make_rain(np.zeros((1, 2, 365)), "2001-01-01", "noleap")
    values = np.zeros((1, 2, 365), dtype=np.int32)
    values[0, 0, day_index(template, "2001-04-02")] = 100
    rain = make_rain(values, "2001-01-01", "noleap")
    ds = engine.detect_observed(definition(ANY_WET, (season("s", (3, 1), (5, 31)),)), rain, years=[2001])
    engine.save(ds, tmp_path / "noleap.zarr")
    back = xr.open_zarr(tmp_path / "noleap.zarr").load()
    assert (back["onset_index"].values == ds["onset_index"].values).all()
    assert back["onset_date"].values[0, 0, :, :].tolist() == [["2001-04-02", ""]]
