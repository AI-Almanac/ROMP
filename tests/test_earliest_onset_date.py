"""earliest_onset_date: onsets before the date are not counted, and configs using the old mok key still load."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.lib.assertion import ROMPConfigError
from momp.lib.loader import _rename_deprecated_keys
from momp.stats.detect import (
    compute_onset_for_all_members,
    compute_onset_for_deterministic_model,
    detect_observed_onset,
)

_OBS_ONSET = xr.DataArray(
    [[np.datetime64("2020-06-20", "ns")]], coords=[("lat", [0.0]), ("lon", [0.0])]
)
_ONSET_KW = dict(wet_init=1, wet_spell=5, dry_spell=7, dry_threshold=1, max_forecast_day=20, end_date=(2020, 9, 30))


def _forecast(member=True):
    """A forecast from 2020-05-25 with 10 mm/day wet spells starting on lead day 3 (May 28) and lead day 11 (Jun 5)."""
    steps = np.arange(1, 41)
    rain = np.zeros(len(steps))
    rain[2:7] = 10.0
    rain[10:15] = 10.0
    coords = {"init_time": pd.to_datetime(["2020-05-25"]), "lat": [0.0], "lon": [0.0], "step": steps}
    dims = ("init_time", "lat", "lon", "step")
    if member:
        coords["member"] = [0, 1]
        dims = ("init_time", "member", "lat", "lon", "step")
    values = np.broadcast_to(rain, [len(coords[dim]) for dim in dims]).copy()
    return xr.DataArray(values, dims=dims, coords=coords)


@pytest.mark.parametrize(("earliest", "expected"), [(None, "2020-04-11"), ((6, 2), "2020-06-05")])
def test_observed_onset_search_starts_at_the_date(earliest, expected):
    """Observed onset skips a valid wet spell in April when the earliest onset date is June 2."""
    time = pd.date_range("2020-04-01", "2020-09-30")
    rain = np.zeros(len(time))
    for start in ("2020-04-11", "2020-06-05"):
        i = time.get_loc(pd.Timestamp(start))
        rain[i : i + 5] = 10.0
    rain_da = xr.DataArray(rain[:, None, None], coords=[("time", time), ("lat", [0.0]), ("lon", [0.0])])

    onset = detect_observed_onset(
        rain_da, 20, 2020, wet_init=1, wet_spell=5, dry_spell=10, dry_threshold=5, dry_extent=0,
        start_date=(2020, 4, 1), end_date=(2020, 9, 30), fallback_date=None, earliest_onset_date=earliest,
    )

    assert pd.Timestamp(onset.values[0, 0]) == pd.Timestamp(expected)


@pytest.mark.parametrize("dry_extent", [0, 10], ids=["vectorized", "member_loop"])
@pytest.mark.parametrize(("earliest", "expected_day"), [(None, 3), ((6, 2), 11)])
def test_ensemble_forecast_onsets_before_the_date_are_skipped(dry_extent, earliest, expected_day):
    """Both ensemble detection paths ignore a forecast wet spell that starts before the earliest onset date."""
    onset_df, onset_mean_df = compute_onset_for_all_members(
        _forecast(), 20, _OBS_ONSET, dry_extent=dry_extent, earliest_onset_date=earliest,
        members=None, onset_percentage_threshold=0.5, **_ONSET_KW,
    )

    assert set(onset_df["onset_day"]) == {expected_day}
    assert list(onset_mean_df["onset_day"]) == [expected_day]


@pytest.mark.parametrize(("earliest", "expected_day"), [(None, 3), ((6, 2), 11)])
def test_deterministic_forecast_onsets_before_the_date_are_skipped(earliest, expected_day):
    """The deterministic path ignores a forecast wet spell that starts before the earliest onset date."""
    onset_df = compute_onset_for_deterministic_model(
        _forecast(member=False), 20, _OBS_ONSET, dry_extent=0, earliest_onset_date=earliest, **_ONSET_KW
    )

    assert list(onset_df["onset_day"]) == [expected_day]


def test_old_mok_key_still_sets_the_filter():
    """A config that still says mok gets the same filter under the new name, with a deprecation warning."""
    with pytest.warns(FutureWarning, match="earliest_onset_date"):
        dic = _rename_deprecated_keys({"mok": (6, 2)})

    assert dic == {"earliest_onset_date": (6, 2)}


def test_setting_both_keys_is_an_error():
    """A config may not set both mok and earliest_onset_date."""
    with pytest.raises(ROMPConfigError):
        _rename_deprecated_keys({"mok": (6, 2), "earliest_onset_date": (6, 2)})


def test_no_key_means_no_filter():
    """A config without either key gets earliest_onset_date = None."""
    assert _rename_deprecated_keys({})["earliest_onset_date"] is None
