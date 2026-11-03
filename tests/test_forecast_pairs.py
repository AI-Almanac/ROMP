import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.stats.bins import create_forecast_observation_pairs_with_bins

DAY_BINS = ((1, 5), (6, 10))
INIT = pd.Timestamp("2022-06-01")


def observed_onsets(*dates):
    """Observed onset per latitude (one longitude), None for no onset."""
    values = [np.datetime64("NaT") if d is None else np.datetime64(d) for d in dates]
    return xr.DataArray(
        np.array(values, dtype="datetime64[ns]").reshape(-1, 1),
        coords={"lat": [10.0, 10.25, 10.5][: len(dates)], "lon": [70.0]},
        dims=("lat", "lon"),
    )


def member_onsets(onset_days_by_lat, init=INIT):
    """Member onset days per latitude, None where a member has no onset in the forecast."""
    rows = [
        {"init_time": init, "lat": lat, "lon": 70.0, "member": member, "onset_day": day}
        for lat, days in onset_days_by_lat.items()
        for member, day in enumerate(days)
    ]
    return pd.DataFrame(rows)


def pairs(members, onset_da):
    return create_forecast_observation_pairs_with_bins(
        members, onset_da, day_bins=DAY_BINS, max_forecast_day=10
    )


def test_bin_probability_is_the_share_of_members_with_onset_in_that_bin():
    members = member_onsets({10.0: [0, 3, 4, 8, 12, None, None, 7]})

    df = pairs(members, observed_onsets("2022-06-20"))

    assert df.set_index("bin_label")["predicted_prob"].to_dict() == pytest.approx({
        "Before day 1": 1 / 8, "Days 1-5": 2 / 8, "Days 6-10": 2 / 8, "After day 10": 3 / 8,
    })


def test_observed_onset_marks_the_bin_by_calendar_day():
    members = member_onsets({10.0: [3, 8], 10.25: [3, 8]})
    onset_da = observed_onsets("2022-06-06T18:00", "2022-06-15")

    df = pairs(members, onset_da)

    observed = df[df["observed_onset"] == 1].set_index("lat")["bin_label"]
    assert observed.to_dict() == {10.0: "Days 1-5", 10.25: "After day 10"}


def test_cases_without_an_observed_onset_after_initialization_are_skipped():
    members = member_onsets({10.0: [3, 8], 10.25: [3, 8], 10.5: [3, 8]})
    onset_da = observed_onsets("2022-06-01", None, "2022-06-10")

    df = pairs(members, onset_da)

    assert set(df["lat"]) == {10.5}


def test_rows_are_ordered_by_case_then_bin_whatever_the_input_order():
    members = pd.concat([
        member_onsets({10.25: [3, 8], 10.0: [2, None]}, init=INIT + pd.Timedelta(days=3)),
        member_onsets({10.25: [4], 10.0: [9]}),
    ]).sample(frac=1, random_state=0)

    df = pairs(members, observed_onsets("2022-07-01", "2022-07-01"))

    keys = list(zip(df["init_time"], df["lat"], df["bin_index"]))
    assert keys == sorted(keys)
    assert len(df) == 2 * 2 * (len(DAY_BINS) + 2)
