"""momp.stats.detect keeps the old output schemas; expected values are worked out by hand from the old rules."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.stats.detect import (
    check_alignment_once,
    compute_onset_for_all_members,
    compute_onset_for_deterministic_model,
    detect_observed_onset,
)
from momp.stats import detect

FLAT = dict(wet_init=1, wet_spell=3, dry_spell=7, dry_threshold=1, dry_extent=0, wet_threshold=20,
            start_date=(2020, 5, 1), end_date=(2020, 7, 31), fallback_date=None, earliest_onset_date=None,
            max_forecast_day=20, members=None, onset_percentage_threshold=0.5)
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
    onset = detect_observed_onset(_obs([("2020-05-10", 3)]), 20, 2020, **FLAT)
    assert onset.name == "onset_date" and onset.dims == ("lat", "lon") and onset.dtype == np.dtype("datetime64[ns]")
    assert pd.Timestamp(onset.values[0, 0]) == pd.Timestamp("2020-05-10")
    none = detect_observed_onset(_obs([("2020-05-10", 2)]), 20, 2020, **FLAT)
    assert pd.isna(none.values[0, 0])


def test_observed_search_runs_47_days_past_end_date():
    """A spell on 10 September, 41 days after the end date, is found; one on 20 September is not."""
    assert pd.Timestamp(detect_observed_onset(_obs([("2020-09-10", 3)]), 20, 2020, **FLAT).values[0, 0]) \
        == pd.Timestamp("2020-09-10")
    assert pd.isna(detect_observed_onset(_obs([("2020-09-20", 3)]), 20, 2020, **FLAT).values[0, 0])


def test_dry_extent_vetoes_an_onset_followed_by_a_dry_run():
    """With dry_extent 10, a spell followed by 7 dry days is skipped for the next one."""
    obs = _obs([("2020-05-10", 3), ("2020-06-20", 3)] + [(f"2020-06-{d}", 1) for d in (24, 27, 30)])
    onset = detect_observed_onset(obs, 20, 2020, **{**FLAT, "dry_extent": 10})
    assert pd.Timestamp(onset.values[0, 0]) == pd.Timestamp("2020-06-20")


def test_the_threshold_may_be_a_per_cell_array():
    """A 2-D threshold is compared per cell: the same spell passes 25 mm and fails 35 mm."""
    low = xr.DataArray([[25.0]], coords=[("lat", [0.0]), ("lon", [0.0])])
    high = low + 10
    assert not pd.isna(detect_observed_onset(_obs([("2020-05-10", 3)]), low, 2020, **FLAT).values[0, 0])
    assert pd.isna(detect_observed_onset(_obs([("2020-05-10", 3)]), high, 2020, **FLAT).values[0, 0])


def test_deterministic_frame_schema():
    """Deterministic rows: one per (init, cell) before the observed onset, with the old columns and dtypes."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}]).isel(member=0, drop=True)
    df = compute_onset_for_deterministic_model(forecast, 20, OBS_ONSET, **FLAT)
    assert list(df.columns) == ["init_time", "lat", "lon", "onset_day", "onset_date", "obs_onset_date"]
    assert df["onset_day"].dtype == np.int64 and df["init_time"].dtype == forecast["init_time"].dtype
    assert df.iloc[0].to_dict() == {"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "onset_day": 3,
                                    "onset_date": "2020-05-28", "obs_onset_date": "2020-06-20"}


def test_no_forecast_onset_is_none():
    """A forecast without onset keeps its row with onset_day and onset_date None."""
    forecast = _forecast([{}]).isel(member=0, drop=True)
    df = compute_onset_for_deterministic_model(forecast, 20, OBS_ONSET, **FLAT)
    assert len(df) == 1 and df["onset_day"].iloc[0] is None and df["onset_date"].iloc[0] is None


def test_inits_on_or_after_the_observed_onset_and_cells_without_one_have_no_rows():
    """Forecast initialised on the observed onset day, or for a cell without observed onset, gives no rows."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}]).isel(member=0, drop=True)
    late = xr.DataArray([[np.datetime64("2020-05-25", "ns")]], coords=OBS_ONSET.coords)
    never = xr.DataArray([[np.datetime64("NaT", "ns")]], coords=OBS_ONSET.coords)
    assert compute_onset_for_deterministic_model(forecast, 20, late, **FLAT).empty
    assert compute_onset_for_deterministic_model(forecast, 20, never, **FLAT).empty


def test_ensemble_frames_round_half_to_even_and_keep_the_columns():
    """Members with onsets on lead 3 and 4 give an ensemble onset on lead round(3.5) = 4."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}, {4: 10, 5: 10, 6: 10}])
    members, ensemble = compute_onset_for_all_members(forecast, 20, OBS_ONSET, **FLAT)
    assert list(members.columns) == ["init_time", "lat", "lon", "member", "onset_day", "obs_onset_date"]
    assert members["onset_day"].tolist() == [3.0, 4.0] and members["member"].tolist() == [0, 1]
    assert list(ensemble.columns) == ["init_time", "lat", "lon", "onset_day", "onset_date", "member_onset_count",
                                      "total_members", "onset_percentage", "obs_onset_date"]
    row = ensemble.iloc[0]
    assert (row["onset_day"], row["onset_date"], row["member_onset_count"], row["total_members"],
            row["onset_percentage"]) == (4.0, "2020-05-29", 2, 2, 1.0)


@pytest.mark.parametrize(("threshold", "expected"), [(0.5, 3.0), (0.75, np.nan)])
def test_ensemble_needs_the_onset_percentage(threshold, expected):
    """One of two members is enough at 0.5 but not at 0.75."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10}, {}])
    _, ensemble = compute_onset_for_all_members(forecast, 20, OBS_ONSET, **{**FLAT, "onset_percentage_threshold": threshold})
    assert np.allclose(ensemble["onset_day"].tolist(), [expected], equal_nan=True)
    assert ensemble["onset_percentage"].iloc[0] == 0.5


def test_ensemble_with_a_long_dry_extent_uses_the_loop_row_order():
    """With dry_extent above wet_spell the old loop path ran: its columns are the same and onset_day is int or None."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10, 6: 3, 10: 3}] * 2)
    members, ensemble = compute_onset_for_all_members(forecast, 20, OBS_ONSET, **{**FLAT, "dry_extent": 10})
    assert members["onset_day"].tolist() == [3, 3] and ensemble["onset_day"].tolist() == [3]


def test_too_few_forecast_steps_is_an_error_for_flat_keys():
    """Fewer lead days than max_forecast_day + wet_spell - 1 raises the old ValueError."""
    forecast = _forecast([{}], steps=21)
    with pytest.raises(ValueError, match="Not enough forecast time steps"):
        compute_onset_for_all_members(forecast, 20, OBS_ONSET, **FLAT)


DEFINITION = dict(onset_definition="icpac", onset_legacy=False, obs_day_boundary="00 UTC, labelled by start",
                  model_lead_label="start", onset_backend="reference", max_forecast_day=20,
                  onset_percentage_threshold=0.5, members=None)


def test_definition_mode_adds_status_and_dates_leads_by_their_start():
    """With lead_label start, lead 3 is init + 2 days, onset_day counts days after init, and onset_status is added."""
    rain = {3: 10, 4: 10, 5: 10, **{k: 5 for k in range(7, 46, 4)}}
    forecast = _forecast([rain]).isel(member=0, drop=True)
    df = compute_onset_for_deterministic_model(forecast, {}, OBS_ONSET, **DEFINITION)
    row = df.iloc[0]
    assert (row["onset_day"], row["onset_date"], row["onset_status"]) == (2, "2020-05-27", "ONSET")
    members, ensemble = compute_onset_for_all_members(_forecast([rain]), {}, OBS_ONSET, **DEFINITION)
    assert members["onset_status"].tolist() == ["ONSET"] and ensemble["onset_status"].tolist() == ["ONSET"]


def test_definition_mode_reports_censored_as_no_onset():
    """A candidate whose veto window runs past the forecast is CENSORED and reported without an onset day."""
    forecast = _forecast([{3: 10, 4: 10, 5: 10, **{k: 5 for k in range(7, 23, 4)}}], steps=22).isel(member=0, drop=True)
    row = compute_onset_for_deterministic_model(forecast, {}, OBS_ONSET, **DEFINITION).iloc[0]
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


def test_alignment_check_warns_when_forecasts_look_shifted(capsys, monkeypatch):
    """Lead k holding the observed day init + k labelled "end" (one day late) is reported as shifted, once per model."""
    monkeypatch.setattr(detect, "_alignment_done", set())
    obs, forecast = _alignment_data(0)
    kwargs = {**DEFINITION, "model": "toy", "obs_dir": "x", "model_lead_label": "end"}
    with pytest.warns(UserWarning, match="look shifted"):
        check_alignment_once(obs, forecast, **kwargs)
    assert "likely shifted" in capsys.readouterr().out
    check_alignment_once(obs, forecast, **kwargs)
    assert capsys.readouterr().out == ""


def test_alignment_check_is_quiet_when_aligned_and_skipped_without_a_boundary(capsys, monkeypatch):
    """Correctly labelled forecasts raise no warning; a legacy config with unknown boundary prints a skip note."""
    monkeypatch.setattr(detect, "_alignment_done", set())
    obs, forecast = _alignment_data(0)
    check_alignment_once(obs, forecast, **{**DEFINITION, "model": "toy", "obs_dir": "x"})
    assert "aligned" in capsys.readouterr().out
    flat = dict(FLAT, model="toy2", obs_dir="y", region="atlantis")
    check_alignment_once(obs, forecast, **flat)
    assert "skipped" in capsys.readouterr().out
