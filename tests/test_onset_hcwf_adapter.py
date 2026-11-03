"""hcwf.onset.detect keeps the old output schemas; expected values are worked out by hand from the old rules, which
`hcwf-run convert` writes as a definition (hcwf.config.convert.flat_onset_to_rule)."""

import typed_io
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hcwf.onset.detect import (
    check_alignment,
    detect_deterministic,
    detect_ensemble,
    detect_observed,
)
from hcwf.onset import detect

from old_flat_rule import old_rule as _old_rule

RUN = dict(max_forecast_day=20, members=None, onset_percentage_threshold=0.5)


def old_rule(**changes):
    """The old flat rule as a definition (see old_flat_rule) with the run settings of these tests."""
    return {**_old_rule(**changes), **RUN}


FLAT = old_rule()


def typed(kwargs):
    """(rc, case, select): the typed inputs of the flat kwargs of a test."""
    return typed_io.from_case_kwargs(kwargs)


def obs_args(kwargs):
    """(rc, select), the typed inputs of an observed detection."""
    rc, _, select = typed(kwargs)
    return rc, select
OBS_ONSET = xr.DataArray([[np.datetime64("2020-06-20", "ns")]], coords=[("lat", [0.0]), ("lon", [0.0])])


def _obs(spells):
    """One-cell daily rain from 1 April to 30 September: 10 mm on the listed (start, days) spells, else 0."""
    time = pd.date_range("2020-04-01", "2020-09-30")
    rain = np.zeros(len(time))
    for start, days in spells:
        i = time.get_loc(pd.Timestamp(start))
        rain[i:i + days] = 10.0
    return xr.DataArray(rain[:, None, None], coords=[("time", time), ("lat", [0.0]), ("lon", [0.0])])


def _forecast(rain_by_member, init="2020-05-25", steps=45):
    """A forecast of one cell; rain_by_member is a list of {lead: mm} (a list of one gives no member dim)."""
    values = np.zeros((len(rain_by_member), steps))
    for m, spec in enumerate(rain_by_member):
        for lead, mm in spec.items():
            values[m, lead - 1] = mm
    coords = {"init_time": pd.to_datetime([init]), "lat": [0.0], "lon": [0.0], "step": np.arange(1, steps + 1)}
    data = values[None, :, None, None, :]
    dims = ("init_time", "member", "lat", "lon", "step")
    coords["member"] = np.arange(len(rain_by_member))
    return xr.DataArray(data, dims=dims, coords=coords)


def test_observed_onset_schema_and_value():
    """Observed onset is a (lat, lon) datetime64[ns] DataArray named onset_date, NaT where there is none."""
    onset = detect_observed(_obs([("2020-05-10", 3)]), {}, 2020, *obs_args(FLAT))
    assert onset.name == "onset_date" and onset.dims == ("lat", "lon") and onset.dtype == np.dtype("datetime64[ns]")
    assert pd.Timestamp(onset.values[0, 0]) == pd.Timestamp("2020-05-10")
    none = detect_observed(_obs([("2020-05-10", 2)]), {}, 2020, *obs_args(FLAT))
    assert pd.isna(none.values[0, 0])


def test_observed_search_runs_47_days_past_end_date():
    """A spell on 10 September, 41 days after the end date, is found; one on 20 September is not."""
    assert pd.Timestamp(detect_observed(_obs([("2020-09-10", 3)]), {}, 2020, *obs_args(FLAT)).values[0, 0]) \
        == pd.Timestamp("2020-09-10")
    assert pd.isna(detect_observed(_obs([("2020-09-20", 3)]), {}, 2020, *obs_args(FLAT)).values[0, 0])


def test_dry_extent_vetoes_an_onset_followed_by_a_dry_run():
    """With dry_extent 10, a spell followed by 7 dry days is skipped for the next one."""
    obs = _obs([("2020-05-10", 3), ("2020-06-20", 3)] + [(f"2020-06-{d}", 1) for d in (24, 27, 30)])
    onset = detect_observed(obs, {}, 2020, *obs_args(old_rule(dry_extent=10)))
    assert pd.Timestamp(onset.values[0, 0]) == pd.Timestamp("2020-06-20")


def test_the_threshold_may_be_a_per_cell_array():
    """A 2-D threshold is compared per cell: the same spell passes 25 mm and fails 35 mm."""
    low = {"legacy_threshold": xr.DataArray([[25.0]], coords=[("lat", [0.0]), ("lon", [0.0])])}
    high = {"legacy_threshold": low["legacy_threshold"] + 10}
    cfg = old_rule(thresh_file="t.nc", thresh_var="mwset")
    assert not pd.isna(detect_observed(_obs([("2020-05-10", 3)]), low, 2020, *obs_args(cfg)).values[0, 0])
    assert pd.isna(detect_observed(_obs([("2020-05-10", 3)]), high, 2020, *obs_args(cfg)).values[0, 0])


def test_deterministic_frame_schema():
    """Deterministic rows: one per (init, cell) before the observed onset, with the old columns and dtypes."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}]).isel(member=0, drop=True)
    df = detect_deterministic(forecast, {}, OBS_ONSET, *typed(FLAT))
    assert list(df.columns) == ["init_time", "lat", "lon", "onset_day", "onset_date", "obs_onset_date", "onset_status"]
    assert df["onset_day"].dtype == np.int64 and df["init_time"].dtype == forecast["init_time"].dtype
    assert df.iloc[0].to_dict() == {"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "onset_day": 3,
                                    "onset_date": pd.Timestamp("2020-05-28"), "obs_onset_date": pd.Timestamp("2020-06-20"),
                                    "onset_status": "ONSET"}


def test_no_forecast_onset_is_none():
    """A forecast without onset keeps its row with onset_day None and onset_date NaT."""
    forecast = _forecast([{}]).isel(member=0, drop=True)
    df = detect_deterministic(forecast, {}, OBS_ONSET, *typed(FLAT))
    assert len(df) == 1 and df["onset_day"].iloc[0] is None and pd.isna(df["onset_date"].iloc[0])


def test_inits_on_or_after_the_observed_onset_and_cells_without_one_have_no_rows():
    """Forecast initialised on the observed onset day, or for a cell without observed onset, gives no rows."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}]).isel(member=0, drop=True)
    late = xr.DataArray([[np.datetime64("2020-05-25", "ns")]], coords=OBS_ONSET.coords)
    never = xr.DataArray([[np.datetime64("NaT", "ns")]], coords=OBS_ONSET.coords)
    assert detect_deterministic(forecast, {}, late, *typed(FLAT)).empty
    assert detect_deterministic(forecast, {}, never, *typed(FLAT)).empty


def test_ensemble_frames_round_half_to_even_and_keep_the_columns():
    """Members with onsets on lead 3 and 4 give an ensemble onset on lead round(3.5) = 4."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}, {4: 10, 5: 10, 6: 10}])
    members, ensemble = detect_ensemble(forecast, {}, OBS_ONSET, *typed(FLAT))
    assert list(members.columns) == ["init_time", "lat", "lon", "member", "onset_day", "obs_onset_date", "onset_status"]
    assert members["onset_day"].tolist() == [3.0, 4.0] and members["member"].tolist() == [0, 1]
    assert list(ensemble.columns) == ["init_time", "lat", "lon", "onset_day", "onset_date", "member_onset_count",
                                      "total_members", "onset_percentage", "obs_onset_date", "onset_status"]
    row = ensemble.iloc[0]
    assert (row["onset_day"], row["onset_date"], row["member_onset_count"], row["total_members"],
            row["onset_percentage"]) == (4.0, pd.Timestamp("2020-05-29"), 2, 2, 1.0)


@pytest.mark.parametrize(("threshold", "expected"), [(0.5, 3.0), (0.75, np.nan)])
def test_ensemble_needs_the_onset_percentage(threshold, expected):
    """One of two members is enough at 0.5 but not at 0.75."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}, {}])
    _, ensemble = detect_ensemble(forecast, {}, OBS_ONSET, *typed({**FLAT, "onset_percentage_threshold": threshold}))
    assert np.allclose(ensemble["onset_day"].tolist(), [expected], equal_nan=True)
    assert ensemble["onset_percentage"].iloc[0] == 0.5


def test_ensemble_with_a_long_dry_extent_vetoes_in_the_same_rows():
    """With dry_extent above wet_spell the rule has a dry-window veto; the rows are the same."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10, 6: 3, 10: 3}] * 2)
    members, ensemble = detect_ensemble(forecast, {}, OBS_ONSET, *typed(old_rule(dry_extent=10)))
    assert members["onset_day"].tolist() == [3, 3] and ensemble["onset_day"].tolist() == [3]


DEFINITION = dict(onset_definition="icpac", obs_day_boundary="00 UTC, labelled by start",
                  model_lead_label="start", onset_backend="reference", max_forecast_day=20,
                  onset_percentage_threshold=0.5, members=None)


def test_definition_mode_adds_status_and_dates_leads_by_their_start():
    """With lead_label start, lead 3 is init + 2 days, onset_day counts days after init, and onset_status is added."""
    rain = {3: 10, 4: 10, 5: 10, **{k: 5 for k in range(7, 46, 4)}}
    forecast = _forecast([rain]).isel(member=0, drop=True)
    df = detect_deterministic(forecast, {}, OBS_ONSET, *typed(DEFINITION))
    row = df.iloc[0]
    assert (row["onset_day"], row["onset_date"], row["onset_status"]) == (2, pd.Timestamp("2020-05-27"), "ONSET")
    members, ensemble = detect_ensemble(_forecast([rain]), {}, OBS_ONSET, *typed(DEFINITION))
    assert members["onset_status"].tolist() == ["ONSET"] and ensemble["onset_status"].tolist() == ["ONSET"]


def test_definition_mode_reports_censored_as_no_onset():
    """A candidate whose veto window runs past the forecast is CENSORED and reported without an onset day."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10, **{k: 5 for k in range(7, 23, 4)}}], steps=22).isel(member=0, drop=True)
    row = detect_deterministic(forecast, {}, OBS_ONSET, *typed(DEFINITION)).iloc[0]
    assert row["onset_status"] == "CENSORED" and row["onset_day"] is None


def _alignment_data(shift):
    """Daily rain on a 6 x 6 grid and forecasts whose lead k holds the observed day init + k - 1 + shift."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2020-04-01", periods=120)
    lat, lon = np.arange(6.0), np.arange(6.0)
    rain = rng.gamma(0.4, 8.0, (len(time), 6, 6))
    obs = xr.DataArray(rain, coords=[("time", time), ("lat", lat), ("lon", lon)])
    inits = time[2:60:3]
    values = np.stack([[rain[i + k - 1 + shift] for k in range(1, 6)] for i in range(2, 60, 3)])
    forecast = xr.DataArray(values, dims=("init_time", "step", "lat", "lon"),
                            coords={"init_time": inits, "step": np.arange(1, 6), "lat": lat, "lon": lon})
    return obs, forecast


def test_alignment_check_warns_when_forecasts_look_shifted(caplog, monkeypatch):
    """Lead k holding the observed day init + k labelled "end" (one day late) is reported as shifted, once per model."""
    detect._alignment_done.clear()
    obs, forecast = _alignment_data(0)
    kwargs = {**DEFINITION, "model": "toy", "obs_dir": "x", "model_lead_label": "end", "check_alignment": True}
    caplog.set_level(logging.INFO)
    with pytest.warns(UserWarning, match="look shifted"):
        check_alignment(obs, forecast, *typed(kwargs)[:2])
    assert "likely shifted" in caplog.text
    caplog.clear()
    check_alignment(obs, forecast, *typed(kwargs)[:2])
    assert caplog.text == ""


def test_alignment_check_is_quiet_when_aligned(caplog, monkeypatch):
    """Correctly labelled forecasts raise no warning; one that is shifted does (see the test above)."""
    detect._alignment_done.clear()
    obs, forecast = _alignment_data(0)
    caplog.set_level(logging.INFO)
    check_alignment(obs, forecast, *typed({**DEFINITION, "model": "toy", "obs_dir": "x", "check_alignment": True})[:2])
    assert "aligned" in caplog.text
    caplog.clear()
