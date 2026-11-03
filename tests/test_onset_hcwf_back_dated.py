"""Counts of back-dated forecast onsets in the score outputs (definition mode, forecast_history "observed")."""

import numpy as np
import pandas as pd
import xarray as xr

from hcwf.io.output import add_back_dated_map, save_metrics_to_netcdf, save_score_results
from hcwf.config.model import Window
from hcwf.spatial import axes
from hcwf.stats.benchmark import onset_metrics
from hcwf.stats.bins import forecast_pairs

OBSERVED = {"back_dated": True}   # forecast_history "observed" (the default): back-dated onsets are counted
NONE = {"back_dated": False}      # forecast_history "none": exactly the old tables
OBS_ONSET = xr.DataArray([[np.datetime64("2020-06-20", "ns")]], coords=[("lat", [0.0]), ("lon", [0.0])])


def _metrics(frame, window, *, back_dated):
    """onset_metrics of a table at (grid) points over the window (first lead, last lead) with 3 days tolerance."""
    return onset_metrics(frame, Window(*window, 3), back_dated=back_dated, default_dims=axes.GRID)


def _frame():
    """Two cases at one point: one back-dated onset (init 25 May) and one ordinary onset."""
    rows = [{"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "onset_date": "2020-05-23",
             "obs_onset_date": "2020-06-20"},
            {"init_time": pd.Timestamp("2020-05-28"), "lat": 0.0, "lon": 0.0, "onset_date": "2020-06-01",
             "obs_onset_date": "2020-06-20"}]
    return pd.DataFrame(rows)


def test_metrics_count_back_dated_only_in_the_first_window():
    """The column counts back-dated cases for a window starting at lead 1 and is 0 for later windows."""
    first, _ = _metrics(_frame(), (1, 15), **OBSERVED)
    later, _ = _metrics(_frame(), (16, 30), **OBSERVED)
    assert first["n_back_dated"].tolist() == [1]
    assert later["n_back_dated"].tolist() == [0]


def test_none_history_has_no_column():
    """forecast_history "none" gives exactly the old table."""
    table, _ = _metrics(_frame(), (1, 15), **NONE)
    assert "n_back_dated" not in table.columns


def test_empty_table_still_carries_the_column():
    """No forecast onsets in a year: the empty table has the column in observed mode."""
    table, _ = _metrics(pd.DataFrame(), (1, 15), **OBSERVED)
    assert "n_back_dated" in table.columns and table.empty


def test_spatial_map_and_global_attribute(tmp_path):
    """The map sums the counts per cell and the file gets the total as a global attribute; history none adds nothing."""
    table, _ = _metrics(_frame(), (1, 15), **OBSERVED)
    spatial = {"mean_mae": xr.DataArray([[1.0]], coords=OBS_ONSET.coords)}
    add_back_dated_map(spatial, {2020: table, 2021: table}, {2020: OBS_ONSET, 2021: OBS_ONSET})
    assert spatial["n_back_dated"].values.tolist() == [[2.0]]
    save_metrics_to_netcdf(spatial, _case("c"), None, _inputs(tmp_path).output, _inputs(tmp_path).ev, None)
    ds = xr.open_dataset(tmp_path / "spatial_metrics_c_1-10.nc")
    assert ds.attrs["n_back_dated_total"] == 2 and "n_back_dated" in ds

    plain_table, _ = _metrics(_frame(), (1, 15), **NONE)
    plain = {"mean_mae": xr.DataArray([[1.0]], coords=OBS_ONSET.coords)}
    add_back_dated_map(plain, {2020: plain_table}, {2020: OBS_ONSET})
    assert list(plain) == ["mean_mae"]


def _members():
    """One case with members at onset day -2, 3 and none."""
    return pd.DataFrame({"init_time": pd.Timestamp("2020-05-25"), "lat": 0.0, "lon": 0.0, "member": [0, 1, 2],
                         "onset_day": [-2.0, 3.0, np.nan], "obs_onset_date": "2020-06-20"})


def _case(model):
    """A typed case of the model over the window 1-10 (the file names and attributes read the model and window)."""
    from hcwf.config.model import Case, ModelSpec, Window

    spec = ModelSpec(name=model, dir="", var=None, unit_cvt=None, file_pattern=None, accumulation=None,
                     accumulation_detected=None, daily_boundary=None, dims=None, paths=None, units=None)
    return Case(spec, Window(1, 10, 0))


def _inputs(tmp_path):
    """The typed output and evaluation specs of a flat config writing to tmp_path."""
    from typed_io import reader_inputs

    return reader_inputs({"dir_out": str(tmp_path), "max_forecast_day": 10, "years": (2020,)})


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
    day_bins = ((1, 5), (6, 10))
    pairs = forecast_pairs(_members(), OBS_ONSET, day_bins, **OBSERVED)
    assert pairs["back_dated_members"].iloc[0] == 1 and pairs["onset_members"].iloc[0] == 2
    save_score_results({**_scores(), "forecast_obs_df": pairs}, _case("M"), None, _inputs(tmp_path).output)
    assert pd.read_csv(tmp_path / "overall_skill_scores_M_1-10.csv")["back_dated_member_share"].tolist() == [0.5]

    old = forecast_pairs(_members(), OBS_ONSET, day_bins, **NONE)
    assert "back_dated_members" not in old.columns
    save_score_results({**_scores(), "forecast_obs_df": old}, _case("L"), None, _inputs(tmp_path).output)
    assert "back_dated_member_share" not in pd.read_csv(tmp_path / "overall_skill_scores_L_1-10.csv").columns
