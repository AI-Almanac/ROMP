"""Forecast history (stitching observed rain before each forecast) and the model's own threshold climatology in ROMP."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.io import input as romp_input
from momp.lib.onset_config import OnsetConfigError, history_needs_previous_year, onset_setup
from momp.stats.benchmark import compute_onset_metrics_with_windows
from momp.stats.bins import create_forecast_observation_pairs_with_bins
from momp.stats.detect import compute_onset_for_deterministic_model

BOUNDARY = "00 UTC, labelled by start"


def _cfg(**values):
    """A definition-mode config with the loader's flags set."""
    return {"onset_legacy": False, "onset_definition": "icpac", "obs_day_boundary": BOUNDARY, **values}


# --- configuration ---------------------------------------------------------------------------------

def test_defaults_are_observed_history_and_observed_thresholds():
    """Definition mode defaults to stitching and observation thresholds, and prints both."""
    setup = onset_setup(_cfg())
    assert (setup.forecast_history, setup.threshold_source) == ("observed", "observations")
    lines = setup.lines()
    assert "Forecast history : observed  [default]" in lines
    assert "Threshold source : observations  [default]" in lines


def test_explicit_values_are_used():
    """Explicit settings win and name their source."""
    setup = onset_setup(_cfg(forecast_history="none", onset_definition="moron_robertson", threshold_source="model",
                             threshold_years=(2000, 2010)))
    assert (setup.forecast_history, setup.threshold_source) == ("none", "model")
    assert "Forecast history : none  [config forecast_history]" in setup.lines()


@pytest.mark.parametrize(("key", "value"), [("forecast_history", "stitched"), ("threshold_source", "forecast")])
def test_unknown_values_are_errors(key, value):
    """Anything but the documented choices is an error naming the key."""
    with pytest.raises(OnsetConfigError, match=key):
        onset_setup(_cfg(**{key: value}))


def test_model_thresholds_need_mean_based_thresholds(tmp_path):
    """A quantile threshold cannot come from the model; the error names it."""
    from momp.onset.definition import REGISTRY_DIR

    text = (REGISTRY_DIR / "moron_robertson.yaml").read_text()
    head, _, rest = text.partition("thresholds:")
    _, _, tail = rest.partition("onset:")
    quantile = (head + "thresholds:\n  wet_spell_amount:\n    quantile of totals over: 5 days\n    quantile: 0.8\n"
                "    during: 06-01 to 09-30\n\nonset:" + tail)
    file = tmp_path / "q.yaml"
    file.write_text(quantile)
    with pytest.raises(OnsetConfigError, match="wet_spell_amount.*quantile_total"):
        onset_setup(_cfg(onset_definition=str(file), threshold_source="model", threshold_years=(2000, 2010)))


@pytest.mark.parametrize(("key", "value"), [("forecast_history", "observed"), ("threshold_source", "model")])
def test_legacy_mode_rejects_the_new_choices(key, value):
    """Old flat-key configs only work with history "none" and observation thresholds."""
    flat = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_spell=7, dry_extent=0, end_date=(2019, 7, 31))
    with pytest.raises(OnsetConfigError, match=key):
        onset_setup({**flat, key: value})
    setup = onset_setup({**flat, "forecast_history": "none"})
    assert setup.forecast_history == "none"


def test_previous_year_is_needed_for_wrapping_seasons_only(tmp_path):
    """A season ending before it starts, or starting within the look-back of 1 January, needs last year's file."""
    assert not history_needs_previous_year(onset_setup(_cfg()))
    assert not history_needs_previous_year(onset_setup(_cfg(forecast_history="none")))
    file = tmp_path / "wrap.yaml"
    file.write_text("id: wrap\nversion: 1\ntitle: t\nstatus: draft\nextends: icpac\nseason: {from: 11-01, to: 02-28}\n")
    assert history_needs_previous_year(onset_setup(_cfg(onset_definition=str(file))))
    file.write_text("id: early\nversion: 1\ntitle: t\nstatus: draft\nextends: icpac\nseason: {from: 01-02, to: 05-30}\n"
                    "onset: {require: [{first day: wet}, {total: more than 20 mm, over: days 0 to 3}]}\n")
    assert history_needs_previous_year(onset_setup(_cfg(onset_definition=str(file))))


def test_load_history_adds_the_previous_year_only_when_needed(tmp_path, monkeypatch):
    """The history is the year's observations, plus last year's file for a wrapping season."""
    loaded = []

    def fake(year, **kwargs):
        """One year of dry observations."""
        loaded.append(year)
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        return xr.DataArray(np.zeros((len(time), 1, 1)), coords=[("time", time), ("lat", [0.0]), ("lon", [0.0])])

    monkeypatch.setattr(romp_input, "load_imd_rainfall", fake)
    current = fake(2020)
    loaded.clear()
    assert romp_input.load_history(2020, current, **_cfg()) is current
    assert romp_input.load_history(2020, current, **_cfg(forecast_history="none")) is None
    assert loaded == []
    file = tmp_path / "wrap.yaml"
    file.write_text("id: wrap\nversion: 1\ntitle: t\nstatus: draft\nextends: icpac\nseason: {from: 11-01, to: 02-28}\n")
    history = romp_input.load_history(2020, current, **_cfg(onset_definition=str(file)))
    assert loaded == [2019] and history.sizes["time"] == 365 + 366
    flat = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_spell=7, dry_extent=0, end_date=(2019, 7, 31))
    assert romp_input.load_history(2020, current, **flat) is None


# --- adapter in stitched mode -----------------------------------------------------------------------

DEFINITION = dict(onset_definition="icpac", onset_legacy=False, obs_day_boundary=BOUNDARY, model_lead_label="start",
                  onset_backend="reference", max_forecast_day=20, onset_percentage_threshold=0.5, members=None,
                  dry_spell_check=False, forecast_history="observed")
OBS_ONSET = xr.DataArray([[np.datetime64("2020-06-20", "ns")]], coords=[("lat", [0.0]), ("lon", [0.0])])


def _history(spells, start="2020-04-01", end="2020-09-30"):
    """One cell of daily observed rain, 10 mm on each (first day, number of days) spell."""
    time = pd.date_range(start, end)
    rain = np.zeros(len(time))
    for first, days in spells:
        i = time.get_loc(pd.Timestamp(first))
        rain[i:i + days] = 10.0
    return xr.DataArray(rain[:, None, None], coords=[("time", time), ("lat", [0.0]), ("lon", [0.0])])


def _forecast(init, leads, steps=30):
    """A one-cell deterministic forecast with 10 mm on the given leads."""
    values = np.zeros(steps)
    for lead in leads:
        values[lead - 1] = 10.0
    return xr.DataArray(values[None, None, None, :], dims=("init_time", "lat", "lon", "step"),
                        coords={"init_time": pd.to_datetime([init]), "lat": [0.0], "lon": [0.0],
                                "step": np.arange(1, steps + 1)})


def test_a_spell_started_in_the_observations_is_back_dated():
    """Wet days on 23 and 24 May (observed) and 25 May (lead 1) make an onset on 23 May: onset_day -2, ONSET."""
    history = _history([("2020-05-23", 2)])
    forecast = _forecast("2020-05-25", [1])
    stitched = compute_onset_for_deterministic_model(forecast, {}, OBS_ONSET, history=history, **DEFINITION)
    row = stitched.iloc[0]
    assert (row["onset_day"], row["onset_date"], row["onset_status"]) == (-2, "2020-05-23", "ONSET")
    alone = compute_onset_for_deterministic_model(forecast, {}, OBS_ONSET, **DEFINITION)
    assert alone["onset_day"].iloc[0] is None and alone["onset_status"].iloc[0] == "NOT_FOUND"


def test_nothing_observed_after_the_init_is_used():
    """Observed wet days after the init do not create an onset in the forecast."""
    history = _history([("2020-05-25", 5)])
    row = compute_onset_for_deterministic_model(_forecast("2020-05-25", []), {}, OBS_ONSET, history=history,
                                                **DEFINITION).iloc[0]
    assert row["onset_status"] == "NOT_FOUND"


def test_inits_after_the_observed_onset_give_no_rows_and_prior_is_never_scored():
    """With the onset confirmed by the observations, the init is on or after the observed onset: no row."""
    history = _history([("2020-05-10", 3)])
    obs_onset = xr.DataArray([[np.datetime64("2020-05-10", "ns")]], coords=OBS_ONSET.coords)
    forecast = _forecast("2020-05-25", [])
    assert compute_onset_for_deterministic_model(forecast, {}, obs_onset, history=history, **DEFINITION).empty


def test_a_prior_result_in_a_scored_row_warns_and_counts_as_no_onset():
    """If the observed onset date disagrees with the history, the PRIOR row is reported without onset, with a warning."""
    history = _history([("2020-05-10", 3)])
    with pytest.warns(UserWarning, match="PRIOR"):
        df = compute_onset_for_deterministic_model(_forecast("2020-05-25", []), {}, OBS_ONSET, history=history,
                                                   **DEFINITION)
    assert df["onset_status"].tolist() == ["PRIOR"] and df["onset_day"].iloc[0] is None


def test_history_from_the_previous_year_serves_a_wrapping_season(tmp_path):
    """A November-February season, wet 31 December and 1 January, init 2 January: back-dated by two days."""
    file = tmp_path / "wrap.yaml"
    file.write_text("id: wrap\nversion: 1\ntitle: t\nstatus: draft\nextends: icpac\nseason: {from: 11-01, to: 02-28}\n"
                    "onset: {reject if: []}\n")
    history = _history([("2019-12-31", 2)], start="2019-11-01", end="2020-03-31")
    obs_onset = xr.DataArray([[np.datetime64("2020-02-10", "ns")]], coords=OBS_ONSET.coords)
    row = compute_onset_for_deterministic_model(
        _forecast("2020-01-02", [1]), {}, obs_onset, history=history,
        **{**DEFINITION, "onset_definition": str(file)}).iloc[0]
    assert (row["onset_day"], row["onset_date"], row["onset_status"]) == (-2, "2019-12-31", "ONSET")


# --- scoring of back-dated onsets -------------------------------------------------------------------

def _frame(model_onset, obs_onset, init="2020-05-25"):
    """A one-row deterministic onset table."""
    return pd.DataFrame([{"init_time": pd.Timestamp(init), "lat": 0.0, "lon": 0.0, "onset_day": None,
                          "onset_date": model_onset, "obs_onset_date": obs_onset}])


def test_back_dated_onsets_are_scored_in_the_first_window_only():
    """An onset on or before the init counts as a first-lead onset in a window starting at lead 1, and is ignored later."""
    first, _ = compute_onset_metrics_with_windows(_frame("2020-05-23", "2020-05-30"), tolerance_days=3,
                                                  verification_window=(1, 15))
    assert (first["false_positive"].iloc[0], first["true_positive"].iloc[0], first["false_negative"].iloc[0]) == (1, 0, 0)
    near, _ = compute_onset_metrics_with_windows(_frame("2020-05-24", "2020-05-26"), tolerance_days=3,
                                                 verification_window=(1, 15))
    assert (near["true_positive"].iloc[0], near["false_positive"].iloc[0]) == (1, 0)
    later, _ = compute_onset_metrics_with_windows(_frame("2020-05-23", "2020-06-20"), tolerance_days=5,
                                                  verification_window=(16, 30))
    counted = later[["true_positive", "false_positive", "false_negative", "true_negative"]].iloc[0].sum()
    assert counted == 0


def test_back_dated_members_fall_in_the_before_day_one_bin():
    """Ensemble members with onset_day <= 0 put their probability in "Before day 1", where nothing was observed."""
    members = pd.DataFrame({"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "member": [0, 1],
                            "onset_day": [-2.0, 3.0], "obs_onset_date": "2020-06-20"})
    pairs = create_forecast_observation_pairs_with_bins(members, OBS_ONSET, day_bins=((1, 5), (6, 10)),
                                                        max_forecast_day=10)
    before = pairs[pairs["bin_label"] == "Before day 1"].iloc[0]
    assert (before["predicted_prob"], before["observed_onset"]) == (0.5, 0)
    assert pairs[pairs["bin_label"] == "Days 1-5"]["predicted_prob"].iloc[0] == 0.5


# --- model thresholds -------------------------------------------------------------------------------

def test_model_thresholds_are_computed_from_the_forecast_archive_and_cached(tmp_path, monkeypatch):
    """A constant 2 mm/day archive gives the 5-day amount 10 mm in every cell; the second call reads the cache."""
    archive = tmp_path / "model"
    archive.mkdir()
    inits = pd.date_range("2015-05-01", "2015-06-30")
    for year in (2015, 2016):
        times = pd.date_range(f"{year}-05-01", f"{year}-06-30")
        ds = xr.Dataset({"tp": (("time", "day", "lat", "lon"), np.full((len(times), 10, 2, 2), 2.0))},
                        coords={"time": times, "day": np.arange(1, 11), "lat": [0.0, 1.0], "lon": [0.0, 1.0]})
        ds.to_netcdf(archive / f"{year}.nc")
    assert len(inits) == 61
    monkeypatch.setattr(romp_input, "region_select", lambda ds, **kwargs: ds)
    kwargs = _cfg(onset_definition="moron_robertson", threshold_source="model", threshold_years=(2015, 2016),
                  model="TOY", model_dir=str(archive), model_var="tp", file_pattern="{}.nc", unit_cvt=None,
                  date_filter_year=2024, init_days=tuple(range(7)), start_date=(2015, 5, 1), end_date=(2015, 6, 30),
                  members=None, probabilistic=False, max_forecast_day=10, dir_out=str(tmp_path / "out"), region="x")
    thresholds = romp_input.load_thresholds(**kwargs)
    amount = thresholds["wet_spell_amount"]
    assert np.allclose(amount.values, 10.0) and amount.attrs["source"].startswith("model:")
    assert list((tmp_path / "out" / ".onset_cache").glob("model_wet_spell_amount_*.nc"))
    romp_input._loaded_thresholds.clear()
    monkeypatch.setattr(romp_input, "get_forecast_deterministic_twice_weekly",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache not used")))
    again = romp_input.load_thresholds(**kwargs)
    assert np.allclose(again["wet_spell_amount"].values, 10.0)
