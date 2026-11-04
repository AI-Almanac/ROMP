"""Days are counted from the season start of the season's year, so a season crossing New Year keeps its January onsets."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.metrics.bins import season_relative_days
from hcwf.references import kde
from hcwf.references.conditional import _conditional_by_case, season_day


def _old_season_days(dates, season_start=(1, 1), leap_shift=False):
    """Count the days as the counter did before: from the season start of each date's own calendar year."""
    index = pd.DatetimeIndex(np.asarray(dates).ravel())
    valid = ~index.isna()
    year = np.where(valid, index.year, 1970).astype(np.int64)
    month, day = season_start
    start = ((year - 1970).astype("datetime64[Y]").astype("datetime64[M]") + np.timedelta64(month - 1, "M")
             ).astype("datetime64[D]") + np.timedelta64(day - 1, "D")
    days = (index.values.astype("datetime64[D]") - start).astype(np.int64).astype(float)
    if leap_shift and month < 3:
        days = days - (index.is_leap_year & (index.month >= 3))
    return np.where(valid, days, np.nan)


@pytest.mark.parametrize("season_start", [(1, 1), (5, 1), (3, 15), (10, 1)])
@pytest.mark.parametrize("leap_shift", [False, True])
def test_an_in_year_season_counts_as_before(season_start, leap_shift):
    rng = np.random.default_rng(3)
    dates = np.datetime64("1996-01-01") + rng.integers(0, 365 * 30, 500).astype("timedelta64[D]")
    dates = dates.astype("datetime64[ns]")
    dates[::50] = np.datetime64("NaT")
    years = np.where(np.isnat(dates), 1970, pd.DatetimeIndex(dates).year).astype(np.int64)
    expected = _old_season_days(dates, season_start, leap_shift)
    np.testing.assert_array_equal(kde.season_days(dates, season_start, leap_shift), expected)
    np.testing.assert_array_equal(kde.season_days(dates, season_start, leap_shift, label_years=years),
                                  expected)


def test_a_january_date_counts_after_the_autumn_of_its_season():
    init, onset = np.datetime64("2020-11-02"), np.datetime64("2021-01-10")
    dates = np.array([init, onset])
    days = kde.season_days(dates, (10, 1), label_years=2020)
    assert days.tolist() == [32.0, 101.0]
    assert kde.season_days(dates, (10, 1)).tolist() == [32.0, -264.0]   # the date's own year, as before
    cond = season_day(dates, 2020)
    assert cond[0] == 305 and cond[1] == 305 + 69   # 2020 is a leap year: 29 February shares 1 March
    assert season_relative_days(dates, 2020).tolist() == cond.tolist()


def test_the_leap_day_shares_the_day_of_1_march_across_the_year_end():
    dates = np.array(["2020-03-01", "2021-03-01", "2021-01-01"], dtype="datetime64[ns]")
    assert season_day(dates, 2020).tolist() == [59, 365 + 59, 365]


def test_an_issue_grid_crosses_new_year():
    grid = kde.issue_grid((2019, 2020), "10-01", "01-31", 30, None)
    one = grid[grid["year"] == 2019]
    assert one["time"].iloc[0] == pd.Timestamp("2019-10-01").date()
    assert one["time"].iloc[-1] == pd.Timestamp("2020-01-31").date()
    assert one["d0"].tolist() == list(range(123))
    assert grid[grid["year"] == 2020]["d0"].tolist() == list(range(123))
    horizons = [{"start_md": "10-01", "end_md": "12-31", "forecast_window": 20},
                {"start_md": "01-01", "end_md": "01-31", "forecast_window": 10}]
    grid = kde.issue_grid((2019, 2019), "10-01", "01-31", None, horizons)
    assert grid["window"].tolist() == [20] * 92 + [10] * 31


def test_the_kde_of_a_crossing_season_puts_mass_on_january_onsets():
    rng = np.random.default_rng(1)
    rows = [("u", year, float(np.round(rng.normal(105, 6)))) for year in range(2000, 2019)]   # onsets 10 Jan or so
    onsets = pd.DataFrame(rows, columns=["id", "year", "onset_day"])
    grid, by_year, max_h = kde.plan_issues((2019, 2019), "10-01", "01-31", 40, None, False)
    result = kde.build_climatology(onsets, grid, by_year, max_h, train=(2000, 2018), test=(2019, 2019),
                                   min_onset_years=10, cv_by_year=False)
    cond = result.data["conditional"].sel(unit="u", season_day=75).values   # 15 December: onsets 20 to 45 days ahead
    assert cond[1:].sum() == pytest.approx(1.0, abs=0.05)
    assert cond[15:50].sum() > 0.9


def test_the_conditional_reference_of_a_crossing_season_counts_by_season_year():
    years = np.arange(2000, 2019)
    rng = np.random.default_rng(2)
    onset = np.array([np.datetime64(f"{y + 1}-01-01") + int(rng.integers(0, 20)) for y in years],
                     dtype="datetime64[ns]")
    clim = xr.DataArray(onset[:, None], dims=("year", "unit"), coords={"year": years, "unit": ["a"]})
    inits = pd.DatetimeIndex(["2019-11-02", "2019-12-20"])
    values, position, back = _conditional_by_case(clim, 2019, (np.array([0]),), inits, 90, False)
    prob = values[position][0, back]
    assert prob[0].sum() == pytest.approx(1.0, abs=0.05)     # 2 Nov: January onsets are 60 to 80 days ahead
    assert prob[0][55:90].sum() > 0.95
    assert prob[1][5:35].sum() > 0.95                          # 20 December: 12 to 31 days ahead
