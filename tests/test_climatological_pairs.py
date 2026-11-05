import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.stats.bins import create_climatological_forecast_obs_pairs

DAY_BINS = ((1, 5), (6, 10))
INIT = pd.Timestamp("2022-06-01")  # day of year 152


def onset_grid(onset_doys):
    """Onset dates from {year: [[doy per lon] per lat]}, None for no onset."""
    years = sorted(onset_doys)
    dates = [
        [
            [
                np.datetime64("NaT") if doy is None
                else np.datetime64(f"{year}-01-01") + np.timedelta64(doy - 1, "D")
                for doy in row
            ]
            for row in onset_doys[year]
        ]
        for year in years
    ]
    return xr.DataArray(
        np.array(dates, dtype="datetime64[ns]"),
        coords={"year": years, "lat": [10.0, 10.25], "lon": [70.0]},
        dims=("year", "lat", "lon"),
    )


def pairs(clim_onset, init_dates=(INIT,)):
    return create_climatological_forecast_obs_pairs(
        clim_onset, 2022, list(init_dates), day_bins=DAY_BINS, max_forecast_day=10
    )


def probabilities_at(df, lat):
    cell = df[df["lat"] == lat].set_index("bin_label")
    return cell["predicted_prob"].to_dict()


def test_bin_probability_is_the_share_of_member_years_with_onset_in_that_bin():
    clim = onset_grid({
        2020: [[150], [None]],  # before day 1
        2021: [[155], [160]],   # day 3, day 8
        2022: [[156], [170]],   # day 4 (obs), day 18 (obs)
        2023: [[175], [None]],  # after day 10
    })

    df = pairs(clim)

    assert probabilities_at(df, 10.0) == pytest.approx({
        "Before day 1": 0.25, "Days 1-5": 0.5, "Days 6-10": 0.0, "After day 10": 0.25,
    })
    assert probabilities_at(df, 10.25) == pytest.approx({
        "Before day 1": 0.0, "Days 1-5": 0.0, "Days 6-10": 0.5, "After day 10": 0.5,
    })


def test_observed_onset_marks_the_bin_holding_the_target_years_onset():
    clim = onset_grid({2021: [[155], [160]], 2022: [[156], [170]]})

    df = pairs(clim)

    observed = df[df["observed_onset"] == 1].set_index("lat")["bin_label"]
    assert observed.to_dict() == {10.0: "Days 1-5", 10.25: "After day 10"}


def test_cells_without_an_observed_onset_after_initialization_are_skipped():
    clim = onset_grid({2021: [[155], [160]], 2022: [[150], [None]]})

    assert pairs(clim).empty


def test_rows_are_ordered_by_init_date_then_location_then_bin():
    clim = onset_grid({2021: [[155], [160]], 2022: [[180], [180]]})
    inits = (INIT, INIT + pd.Timedelta(days=3))

    df = pairs(clim, inits)

    keys = list(zip(df["init_time"], df["lat"], df["bin_index"]))
    assert keys == sorted(keys)
    assert len(df) == len(inits) * 2 * (len(DAY_BINS) + 2)
