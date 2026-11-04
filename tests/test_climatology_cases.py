"""The climatology reference scores the same cases and counts the same leads as the model's pairs."""

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.metrics.bins import create_climatological_forecast_obs_pairs, forecast_pairs, season_relative_days
from hcwf.onset.climatology import _doy_index

BINS = ((1, 5), (6, 10), (11, 40))


def _onsets(per_year, years):
    """Build onset dates ``(year, lat, lon)`` of one cell from a date (or None) per year."""
    dates = np.array([np.datetime64(d, "ns") if d else np.datetime64("NaT") for d in per_year], dtype="datetime64[ns]")
    return xr.DataArray(dates.reshape(len(years), 1, 1), coords=[("year", list(years)), ("lat", [10.0]), ("lon", [70.0])])


def test_a_season_across_the_year_boundary_is_a_case_and_members_follow_their_season():
    """Init 2020-11-02 with observed onset 2021-01-10 is a case; members are placed in the season, not the calendar year."""
    years = (2018, 2019, 2020)
    # onsets in January of the year after the label year: 2019-01-05, 2020-01-08, 2021-01-10 (the observation)
    clim = _onsets(["2019-01-05", "2020-01-08", "2021-01-10"], years)
    pairs = create_climatological_forecast_obs_pairs(clim, 2020, [pd.Timestamp("2020-11-02")],
                                                     day_bins=((1, 5), (6, 10), (60, 70)))
    assert pairs.n_cases == 1                                   # the day-of-year rule dropped it (doy 307 > 10)
    assert pairs.init[0] == np.datetime64("2020-11-02")
    assert pairs.observed[0].tolist() == [0, 0, 0, 1, 0]        # observed lead 69 (2020-11-02 -> 2021-01-10)
    assert pairs.members_with_onset[0] == 3
    assert pairs.prob[0].tolist() == [0, 0, 0, 1, 0]            # leads 64, 67 and 69, all in 60-70


def test_leap_year_lead_is_the_real_day_count_of_the_placed_onset():
    """Target 2020 (leap), init 05-01, member onset 2019-06-01: lead 31, not the day-of-year 30."""
    clim = _onsets(["2019-06-01", "2020-06-01"], (2019, 2020))
    pairs = create_climatological_forecast_obs_pairs(clim, 2020, [pd.Timestamp("2020-05-01")],
                                                     day_bins=((1, 30), (31, 31), (32, 40)), leave_one_out=True)
    assert pairs.n_cases == 1 and pairs.members_with_onset[0] == 1
    assert pairs.prob[0].tolist() == [0, 0, 1, 0, 0]
    member = season_relative_days(np.array(["2019-06-01"], "datetime64[ns]"), 2019)
    init = season_relative_days(np.array(["2020-05-01"], "datetime64[ns]"), 2020)
    assert member[0] - init[0] == 31


def test_the_climatology_reference_and_the_model_have_the_same_cases():
    """Over a synthetic run (boundary-crossing season, leap year) both pair builders score the same (init, cell) set."""
    rng = np.random.default_rng(0)
    lats, lons = np.array([10.0, 10.25]), np.array([70.0, 70.25, 70.5])
    years = list(range(2010, 2021))
    shape = (len(years), len(lats), len(lons))
    base = (np.array([f"{y}-11-20" for y in years], dtype="datetime64[D]")[:, None, None]
            + rng.integers(0, 100, size=shape).astype("timedelta64[D]")).astype("datetime64[ns]")
    clim = xr.DataArray(base, coords=[("year", years), ("lat", lats), ("lon", lons)])
    obs = clim.sel(year=2020).copy()
    obs[0, 1] = np.datetime64("NaT", "ns")   # a cell without an observed onset
    inits = pd.DatetimeIndex(["2020-10-25", "2020-11-02", "2020-12-20", "2021-01-05", "2021-02-10"])
    members = pd.DataFrame([(t, la, lo, m, float(rng.integers(-3, 20)))
                            for t in inits for la in lats for lo in lons for m in range(3)],
                           columns=["init_time", "lat", "lon", "member", "onset_day"])
    model = forecast_pairs(members, obs, BINS, back_dated=False)
    ref = create_climatological_forecast_obs_pairs(clim, 2020, inits, obs, day_bins=BINS)

    def cases(p):
        return set(zip(p.init.astype(str), p.place["lat"], p.place["lon"]))

    assert model.n_cases > 0
    assert any(i == "2020-11-02" for i, *_ in cases(model))       # an init a year before its onset
    assert cases(ref) == cases(model)                             # every cell has member onsets in this run
    assert ref.n_cases == model.n_cases


def test_feb_29_is_dropped_and_31_december_kept():
    """The 365-day index drops 29 February, so 1 March is day index 59 in every year, and 31 December is 364."""
    month = np.array([2, 3, 3, 12])
    day = np.array([29, 1, 1, 31])
    assert _doy_index(month, day).tolist() == [-1, 59, 59, 364]


# --------------------------------------------------------------------------------------------
# climatological_onset: the mean onset day on the season-relative, leap-free day axis
# --------------------------------------------------------------------------------------------

def _climatological_onset(monkeypatch, onsets):
    """Run `climatological_onset` in grid-point mode over made-up years; ``onsets`` maps a year to its onset date."""
    from types import SimpleNamespace

    from hcwf.references import climatology

    years = list(onsets)

    def rain(year):
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        values = (time.month.values * 100 + time.day.values).astype(float)
        return xr.DataArray(values, dims="time", coords={"time": time})

    def detect(reader, thresh, year, rc, select):
        return xr.DataArray(np.array([[np.datetime64(onsets[year], "ns")]]), dims=("lat", "lon"),
                            coords={"lat": [10.0], "lon": [70.0]})

    monkeypatch.setattr(climatology, "_year_readers",
                        lambda years, rc, select, shared=True: [(lambda y=y: rain(y)) for y in years])
    monkeypatch.setattr(climatology, "thresholds_of", lambda *a: 1.0)
    monkeypatch.setattr(climatology, "detect_observed", detect)
    monkeypatch.setattr(climatology, "release_reference", lambda: None)
    rc = SimpleNamespace(evaluation=SimpleNamespace(climatology_years=years))
    return climatology.climatological_onset(rc, SimpleNamespace(model=None), SimpleNamespace(grid_point=True))


def test_the_climatological_onset_of_a_leap_year_is_the_day_of_the_365_day_year(monkeypatch):
    """1 June is day 152 in a leap year too (the raw day of year said 153), and the mean over mixed years is 152."""
    doy, _ = _climatological_onset(monkeypatch, {2020: "2020-06-01"})
    assert float(doy.squeeze()) == 152.0
    doy, _ = _climatological_onset(monkeypatch, {2019: "2019-06-01", 2020: "2020-06-01"})
    assert float(doy.squeeze()) == 152.0


def test_the_climatological_onset_of_a_season_across_the_year_boundary_is_beyond_365(monkeypatch):
    """A January onset counts past 365, so it is averaged with the December ones, not with day 10."""
    doy, _ = _climatological_onset(monkeypatch, {2019: "2019-12-20", 2020: "2021-01-10"})
    assert float(doy.squeeze()) == 364.0   # (354 + 375) / 2 = 364.5, rounded half to even; the raw day of year said 182


def test_the_climatological_rainfall_has_31_december_and_one_march_per_year(monkeypatch):
    """The rainfall by day of year drops 29 February: day 60 is 1 March in leap and other years, day 365 is 31 December."""
    _, rain = _climatological_onset(monkeypatch, {2019: "2019-06-01", 2020: "2020-06-01"})
    assert rain.sizes["doy"] == 365
    assert float(rain.sel(doy=60)) == 301.0   # 1 March in both years, not averaged with 2 March (302)
    assert float(rain.sel(doy=365)) == 1231.0


def test_a_mean_beyond_365_is_placed_in_the_year_after_the_init():
    """With init 2020-11-02 a climatological onset of day 375 is 2021-01-10 (lead 69), also in a leap init year."""
    from hcwf.references.climatology import compute_climatology_as_forecast

    like = xr.DataArray(np.array([[np.datetime64("2021-02-01", "ns")]]), dims=("lat", "lon"),
                        coords={"lat": [10.0], "lon": [70.0]})
    clim = xr.DataArray([[375.0]], dims=("lat", "lon"), coords={"lat": [10.0], "lon": [70.0]})
    table = compute_climatology_as_forecast(clim, pd.DatetimeIndex(["2020-11-02"]), like, max_forecast_day=70)
    assert len(table) == 1
    assert table["onset_day"].iloc[0] == 69
