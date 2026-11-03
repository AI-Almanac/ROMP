import numpy as np
import pandas as pd
import xarray as xr

from momp.stats.detect import compute_onset_for_all_members

MAX_FORECAST_DAY = 10
WET_SPELL = 3
SETTINGS = dict(
    wet_init=1, wet_spell=WET_SPELL, dry_spell=7, dry_threshold=1.0, dry_extent=0,
    members=None, onset_percentage_threshold=0.5, max_forecast_day=MAX_FORECAST_DAY,
    mok=False, end_date=(2022, 8, 31),
)


def forecast(*member_rain):
    """One init, one grid point; daily rain per member from step 1."""
    n_steps = MAX_FORECAST_DAY + WET_SPELL - 1
    rain = np.zeros((1, len(member_rain), n_steps, 1, 1))
    for member, days in enumerate(member_rain):
        rain[0, member, : len(days), 0, 0] = days
    return xr.DataArray(rain, dims=("init_time", "member", "step", "lat", "lon"), coords={
        "init_time": [pd.Timestamp("2022-06-01")], "member": np.arange(len(member_rain)),
        "step": np.arange(1, n_steps + 1), "lat": [10.0], "lon": [70.0],
    })


def member_onset_days(p_model, thresh=20):
    onset_da = xr.DataArray(
        np.array([["2022-07-01"]], dtype="datetime64[ns]"),
        coords={"lat": [10.0], "lon": [70.0]}, dims=("lat", "lon"),
    )
    members, _ = compute_onset_for_all_members(p_model, thresh, onset_da, **SETTINGS)
    onset_days = members.set_index("member")["onset_day"]
    return {member: None if pd.isna(day) else day for member, day in onset_days.items()}


def test_onset_is_the_first_step_starting_a_wet_spell_whose_total_exceeds_the_threshold():
    p_model = forecast(
        [0, 9, 9, 5, 5, 15],   # steps 2-4 total 23 -> day 2
        [0, 0, 0, 5, 5, 15],   # steps 4-6 total 25 -> day 4
        [5, 5, 10, 5, 5, 10],  # every spell totals exactly 20 -> none
        [30, 0.5, 30],         # one dry-ish day breaks the spell -> none
    )

    assert member_onset_days(p_model) == {0: 2, 1: 4, 2: None, 3: None}


def test_a_spell_starting_on_the_last_forecast_day_counts_and_a_missing_day_breaks_one():
    last_day_spell = [0] * (MAX_FORECAST_DAY - 1) + [10, 10, 10]
    missing_day = [10, np.nan, 10, 0, 10, 10, 10]

    assert member_onset_days(forecast(last_day_spell, missing_day)) == {
        0: MAX_FORECAST_DAY, 1: 5,
    }
