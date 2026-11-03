"""The KDE climatology's caller logic is blending's: the issue grid, eligibility, horizons and years, and the
probabilities of a whole season (oracle: tests/blend/_oracle/climatology_utils.py, the legacy climatology)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _oracle import climatology_utils as legacy
from hcwf.references import kde
from hcwf.references.conditional import season_day

HORIZONS = [
    {"start_md": "05-01", "end_md": "05-31", "forecast_window": 30},
    {"start_md": "06-01", "end_md": "07-15", "forecast_window": 45},
]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_probabilities_are_the_legacy_kdes(seed):
    rng = np.random.default_rng(seed)
    onset_days = rng.normal(30, 12, size=25).round()
    d0 = np.repeat(np.arange(0, 90, 3), 2)
    windows = np.where(np.arange(len(d0)) % 7 == 0, 30, 45)
    windows[5] = -1  # outside every horizon

    new = kde.onset_probabilities(kde.fit_kde(onset_days), d0, 45, windows)
    old = legacy._predict_pair_for_rows(legacy.fit_kde(onset_days), d0, windows, 45)
    legacy._kde_cdf.cache_clear()

    for a, b in zip(new, old):
        np.testing.assert_array_equal(a, b)


def test_too_few_onsets_give_no_kde():
    assert kde.fit_kde(np.arange(9.0)) is None
    conditional, unconditional = kde.onset_probabilities(None, np.arange(3), 10, np.full(3, 10))
    assert np.isnan(conditional).all() and np.isnan(unconditional).all()


@pytest.mark.parametrize("horizons", [None, HORIZONS])
def test_the_issue_grid_is_the_legacy_plans(horizons):
    window = None if horizons else 45
    new = kde.issue_grid((2019, 2021), "05-01", "07-31", window, horizons)

    grid = legacy.build_issue_grid(2019, 2021, "05-01", "07-31")
    plan = legacy._build_issue_plan(grid, "05-01", window, horizons)
    old = pd.DataFrame({"time": plan["time"], "year": plan["year"], "d0": plan["d0"], "window": plan["window"]})
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def onsets_table(n_units=12, seed=0):
    """Observed onsets (id, year, onset_day since the season start), some units short of onsets."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for year in range(2000, 2019):
            if rng.random() < (0.8 if u % 4 == 3 else 0.05):
                continue
            rows.append((f"u{u:02d}", year, float(np.round(rng.normal(45, 12)))))
    return pd.DataFrame(rows, columns=["id", "year", "onset_day"])


@pytest.mark.parametrize("horizons", [None, HORIZONS])
@pytest.mark.parametrize("cv", [False, True])
def test_a_season_of_probabilities_is_the_legacy_pipelines(cv, horizons):
    onsets = onsets_table()
    window = None if horizons else 40
    train, test = kde.climatology_years(range(2000, 2019), [2019, 2020])
    assert (train, test) == ((2000, 2018), (2019, 2020))
    grid, by_year, max_h = kde.plan_issues(test, "05-01", "07-10", window, horizons, cv)
    result = kde.build_climatology(onsets, grid, by_year, max_h, train=train, test=test, min_onset_years=10,
                                   cv_by_year=cv)

    gt_train, eligible = kde.eligible_onsets(onsets, train, 10)
    assert result.eligible_units == eligible < onsets["id"].nunique()
    old_grid = legacy.build_issue_grid(*test, "05-01", "07-10")
    expected = {}
    for conditional in (True, False):
        out = legacy.compute_all_forecasts(gt_train, old_grid, "05-01", window, horizons, conditional=conditional,
                                           cv_by_year=cv)["forecasts"]
        expected[conditional] = out
    legacy._kde_cdf.cache_clear()
    new_conditional, new_unconditional = result.legacy_tables()
    for new, old in ((new_conditional, expected[True]), (new_unconditional, expected[False])):
        assert list(new.columns) == list(old.columns)
        pd.testing.assert_frame_equal(
            new.assign(time=new["time"].astype(str)).sort_values(["id", "time"], ignore_index=True),
            old.assign(time=old["time"].astype(str)).sort_values(["id", "time"], ignore_index=True),
            check_exact=True, check_dtype=False)


def test_extra_year_windows_have_their_own_eligible_units():
    onsets = onsets_table(seed=3)
    grid, by_year, max_h = kde.plan_issues((2019, 2019), "05-01", "05-20", 30, None, False)
    result = kde.build_climatology(onsets, grid, by_year, max_h, train=(2000, 2018), test=(2019, 2019),
                                   min_onset_years=6, cv_by_year=False, windows=[(2000, 2008), (2009, 2018)])
    assert set(result.windows) == {(2000, 2008), (2009, 2018)}
    early, _ = kde.eligible_onsets(onsets, (2000, 2008), 6)
    assert list(result.windows[(2000, 2008)]["unit"].values) == list(early["id"].unique())
    with pytest.raises(ValueError, match="No units have enough"):
        kde.eligible_onsets(onsets, (2000, 2002), 6)


def test_training_and_test_year_defaults():
    assert kde.climatology_years([2001, 2002, 2003], [2002, 2003]) == ((2001, 2001), (2002, 2003))
    assert kde.climatology_years([2001, 2002, 2003], [2001]) == ((2001, 2003), (2001, 2001))
    assert kde.climatology_years([2001, 2002], [2002], train_years=(2001, 2001), test_years=(2002, 2002)) == (
        (2001, 2001), (2002, 2002))
    with pytest.raises(ValueError, match="observed"):
        kde.climatology_years([], [2002])
    with pytest.raises(ValueError, match="forecast"):
        kde.climatology_years([2001], [])


def test_season_days_count_from_the_season_start_and_hcwfs_axis_from_1_january():
    dates = np.array(["2019-05-03", "2020-05-03", "2020-02-29", "2020-03-01", "NaT"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(kde.season_days(dates, (5, 1)), [2, 2, -62, -61, np.nan])
    plain = kde.season_days(dates)
    assert plain[0] == 122 and plain[1] == 123 and plain[3] == 60
    shifted = season_day(dates)
    assert shifted[0] == 122 and shifted[1] == 122 and shifted[2] == 59 and shifted[3] == 59 and np.isnan(shifted[4])
    jan1 = kde.season_days(pd.to_datetime(["2021-01-01"]))
    assert jan1[0] == 0.0


def test_tasks_of_units_give_the_same_numbers_as_one_task(monkeypatch):
    onsets = onsets_table(seed=5)
    grid, by_year, max_h = kde.plan_issues((2019, 2020), "05-01", "05-12", 20, None, True)
    train, _ = kde.eligible_onsets(onsets, (2000, 2018), 10)
    one = kde.probabilities_for_issues(train, grid, by_year, max_h, True)
    monkeypatch.setattr(kde, "UNITS_PER_TASK", 2)
    many = kde.probabilities_for_issues(train, grid, by_year, max_h, True)
    np.testing.assert_array_equal(one["conditional"].values, many["conditional"].values)
    np.testing.assert_array_equal(one["unconditional"].values, many["unconditional"].values)
