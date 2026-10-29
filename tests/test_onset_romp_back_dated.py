"""Counts of back-dated forecast onsets in the score outputs (definition mode, forecast_history "observed")."""

import numpy as np
import pandas as pd
import xarray as xr

from momp.io.output import add_back_dated_map, save_metrics_to_netcdf, save_score_results
from momp.stats.benchmark import compute_onset_metrics_with_windows
from momp.stats.bins import create_forecast_observation_pairs_with_bins

OBSERVED = {"onset_legacy": False, "onset_definition": "icpac", "obs_day_boundary": "00 UTC, labelled by start"}
LEGACY = {"onset_legacy": True}
OBS_ONSET = xr.DataArray([[np.datetime64("2020-06-20", "ns")]], coords=[("lat", [0.0]), ("lon", [0.0])])


def _frame():
    """Two cases at one point: one back-dated onset (init 25 May) and one ordinary onset."""
    rows = [{"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "onset_date": "2020-05-23",
             "obs_onset_date": "2020-06-20"},
            {"init_time": pd.Timestamp("2020-05-28"), "lat": 0.0, "lon": 0.0, "onset_date": "2020-06-01",
             "obs_onset_date": "2020-06-20"}]
    return pd.DataFrame(rows)


def test_metrics_count_back_dated_only_in_the_first_window():
    """The column counts back-dated cases for a window starting at lead 1 and is 0 for later windows."""
    first, _ = compute_onset_metrics_with_windows(_frame(), tolerance_days=3, verification_window=(1, 15), **OBSERVED)
    later, _ = compute_onset_metrics_with_windows(_frame(), tolerance_days=3, verification_window=(16, 30), **OBSERVED)
    assert first["n_back_dated"].tolist() == [1]
    assert later["n_back_dated"].tolist() == [0]


def test_legacy_and_none_history_have_no_column():
    """Legacy configs and forecast_history "none" give exactly the old table."""
    for cfg in (LEGACY, {**OBSERVED, "forecast_history": "none"}, {}):
        table, _ = compute_onset_metrics_with_windows(_frame(), tolerance_days=3, verification_window=(1, 15), **cfg)
        assert "n_back_dated" not in table.columns


def test_empty_table_still_carries_the_column():
    """No forecast onsets in a year: the empty table has the column in observed mode."""
    table, _ = compute_onset_metrics_with_windows(pd.DataFrame(), tolerance_days=3, verification_window=(1, 15),
                                                  **OBSERVED)
    assert "n_back_dated" in table.columns and table.empty


def test_spatial_map_and_global_attribute(tmp_path):
    """The map sums the counts per cell and the file gets the total as a global attribute; legacy adds nothing."""
    table, _ = compute_onset_metrics_with_windows(_frame(), tolerance_days=3, verification_window=(1, 15), **OBSERVED)
    spatial = {"mean_mae": xr.DataArray([[1.0]], coords=OBS_ONSET.coords)}
    add_back_dated_map(spatial, {2020: table, 2021: table}, {2020: OBS_ONSET, 2021: OBS_ONSET})
    assert spatial["n_back_dated"].values.tolist() == [[2.0]]
    save_metrics_to_netcdf(spatial, {"dir_out": str(tmp_path), "case_name": "c"})
    ds = xr.open_dataset(tmp_path / "spatial_metrics_c.nc")
    assert ds.attrs["n_back_dated_total"] == 2 and "n_back_dated" in ds

    legacy_table, _ = compute_onset_metrics_with_windows(_frame(), tolerance_days=3, verification_window=(1, 15),
                                                         **LEGACY)
    plain = {"mean_mae": xr.DataArray([[1.0]], coords=OBS_ONSET.coords)}
    add_back_dated_map(plain, {2020: legacy_table}, {2020: OBS_ONSET})
    assert list(plain) == ["mean_mae"]


def _members():
    """One case with members at onset day -2, 3 and none."""
    return pd.DataFrame({"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "member": [0, 1, 2],
                         "onset_day": [-2.0, 3.0, np.nan], "obs_onset_date": "2020-06-20"})


def _scores():
    """Minimal score results around a pair table, enough for save_score_results."""
    return {"BS": {"fair_brier_score": 0.1, "bin_fair_brier_scores": {}},
            "BS_ref": {"bin_fair_brier_scores": {}},
            "RPS": {"fair_rps": 0.2}, "AUC": {"auc": 0.6, "bin_auc_scores": {}},
            "AUC_ref": {"auc": 0.5, "bin_auc_scores": {}},
            "skill_results": {"fair_brier_skill_score": 0.3, "fair_rps_skill_score": 0.4,
                              "bin_fair_brier_skill_scores": {}}}


def test_member_share_column_in_the_overall_csv(tmp_path):
    """One of the two member onsets is back-dated: share 0.5 in observed mode, no column otherwise."""
    kwargs = dict(day_bins=((1, 5), (6, 10)), max_forecast_day=10)
    pairs = create_forecast_observation_pairs_with_bins(_members(), OBS_ONSET, **kwargs, **OBSERVED)
    assert pairs["back_dated_members"].iloc[0] == 1 and pairs["onset_members"].iloc[0] == 2
    save_score_results({**_scores(), "forecast_obs_df": pairs}, model="M", verification_window=(1, 10),
                       dir_out=str(tmp_path))
    assert pd.read_csv(tmp_path / "overall_skill_scores_M_1-10.csv")["back_dated_member_share"].tolist() == [0.5]

    old = create_forecast_observation_pairs_with_bins(_members(), OBS_ONSET, **kwargs, **LEGACY)
    assert "back_dated_members" not in old.columns
    save_score_results({**_scores(), "forecast_obs_df": old}, model="L", verification_window=(1, 10),
                       dir_out=str(tmp_path))
    assert "back_dated_member_share" not in pd.read_csv(tmp_path / "overall_skill_scores_L_1-10.csv").columns
