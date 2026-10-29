"""NetCDF export of gridded onset dates: file layouts, encodings, provenance, and equality with what the scoring consumes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cftime
import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from momp import export
from momp.io.input import (get_forecast_deterministic_twice_weekly, get_forecast_probabilistic_twice_weekly,
                           get_initialization_dates, load_history, load_imd_rainfall, load_thresholds)
from momp.lib.config_yaml import build_config
from momp.lib.onset_config import onset_setup
from momp.onset import definition as definition_mod
from momp.onset.model import Status
from momp.stats.climatology import compute_climatological_onset, compute_climatological_onset_dataset
from momp.stats.climatology import compute_climatology_as_forecast
from momp.stats.conditional_climatology import (compute_conditional_climatology_as_forecast,
                                                create_conditional_climatological_forecast_obs_pairs)
from momp.stats.detect import (compute_onset_for_all_members, compute_onset_for_deterministic_model,
                               detect_observed_onset)

ROOT = Path(__file__).resolve().parents[1]
CLIM_YEARS = range(2006, 2021)
LAT, LON = [6.0, 8.0, 10.0], [38.0, 40.0, 42.0]
STEPS, MAX_LEAD, MEMBERS = 14, 10, 3
BOUNDARY = "00 UTC, labelled by start"


def _obs_rain(year):
    """Daily rain (mm) of May-August, (time, lat, lon): dry until a per-cell onset day, then wet."""
    rng = np.random.default_rng(year)
    time = pd.date_range(f"{year}-05-01", f"{year}-08-31")
    onset = rng.integers(8, 40, size=(1, len(LAT), len(LON)))
    day = np.arange(len(time))[:, None, None]
    rain = rng.gamma(0.6, 6.0, size=(len(time), len(LAT), len(LON))) * np.where(day < onset, 0.1, 2.0)
    return time, rain


def _write_obs(folder):
    """One observation file per climatology year."""
    folder.mkdir(parents=True)
    for year in CLIM_YEARS:
        time, rain = _obs_rain(year)
        xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm"})},
                   coords={"time": time, "lat": LAT, "lon": LON}).to_netcdf(folder / f"{year}.nc")


def _write_forecast(folder, members):
    """Daily-init forecasts for May 2019 and 2020 (lead k holds the observed day init + k - 1, with noise)."""
    folder.mkdir(parents=True)
    for year in (2019, 2020):
        time, rain = _obs_rain(year)
        inits = pd.date_range(f"{year}-05-01", f"{year}-05-31")
        rng = np.random.default_rng(year + 7)
        data = np.stack([rain[i:i + STEPS] for i in range(len(inits))])
        dims, coords = ("time", "step", "lat", "lon"), {"time": inits, "step": np.arange(1, STEPS + 1), "lat": LAT,
                                                       "lon": LON}
        if members:
            data = np.stack([data * rng.uniform(0.5, 1.6, size=data.shape) for _ in range(members)], axis=1)
            dims, coords = ("time", "number", "step", "lat", "lon"), {**coords, "number": np.arange(members)}
        else:
            data = data * rng.uniform(0.5, 1.6, size=data.shape)
        xr.Dataset({"tp": (dims, data, {"units": "mm"})}, coords=coords).to_netcdf(folder / f"{year}.nc")


def _raw(base, model, ensemble, **onset):
    """A nested run config over the synthetic data."""
    return {"region": "Ethiopia",
            "observations": {"path": str(base / "obs"), "units": "mm", "day_boundary": BOUNDARY},
            "models": [{"name": model, "path": str(base / model), "units": "mm", "lead_label": "start"}],
            "onset": {"dry_spell_check": False, "check_alignment": False, **onset},
            "evaluation": {"start_date": [2019, 5, 1], "end_date": [2020, 5, 31], "init_days": [0, 1, 2, 3, 4, 5, 6],
                           "start_year_clim": 2006, "end_year_clim": 2020, "verification_windows": [[1, MAX_LEAD]],
                           "tolerances": [3], "max_forecast_day": MAX_LEAD, "day_bins": [[1, 5], [6, 10]],
                           "probabilistic": ensemble},
            "output": {"data_dir": str(base / ("out_" + model)), "figure_dir": str(base / "fig")}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic data, one deterministic and one ensemble config (flat dicts), and the exports of both."""
    base = tmp_path_factory.mktemp("export")
    _write_obs(base / "obs")
    _write_forecast(base / "DET", 0)
    _write_forecast(base / "ENS", MEMBERS)
    raws = {"DET": _raw(base, "DET", False), "ENS": _raw(base, "ENS", True)}
    cfgs = {name: export._complete(build_config(raw)) for name, raw in raws.items()}
    paths = {"DET": export.export_onsets(cfgs["DET"], sources=["observed", "DET", "climatology",
                                                                "conditional_climatology"]),
             "ENS": export.export_onsets(cfgs["ENS"], sources=["ENS"])}
    return SimpleNs(base=base, raws=raws, cfgs=cfgs, paths=paths)


class SimpleNs:
    """A bag of attributes."""

    def __init__(self, **kwargs):
        """Store the keyword arguments as attributes."""
        self.__dict__.update(kwargs)


def _open(path):
    """Open an export file with times decoded and the `lead` coordinate left as plain numbers."""
    return export.open_onsets(path)


# --- layouts, dtypes, attributes -----------------------------------------------------------------------------

def test_returns_one_path_per_source_with_the_documented_names(world):
    """The result maps each source to onsets_<source>.nc, and the files exist."""
    paths = world.paths["DET"]
    assert list(paths) == ["observed", "DET", "climatology", "conditional_climatology"]
    assert [p.name for p in paths.values()] == ["onsets_observed.nc", "onsets_DET.nc", "onsets_climatology.nc",
                                                "onsets_conditional_climatology.nc"]
    assert all(p.exists() and p.parent.name == "onsets" for p in paths.values())
    assert world.paths["ENS"]["ENS"].name == "onsets_ENS.nc"


def test_observed_layout_dtypes_and_flags(world):
    """Observed onsets over (year, season, lat, lon) with CF dates, float day numbers and flagged int8 status."""
    ds = _open(world.paths["DET"]["observed"])
    assert ds["onset_date"].dims == ds["onset_doy"].dims == ds["status"].dims == ("year", "season", "lat", "lon")
    assert list(ds["year"].values) == [2019, 2020] and ds["season"].size == 1
    assert ds["onset_date"].dtype == np.dtype("datetime64[ns]")
    assert ds["onset_doy"].dtype == np.float32 and ds["status"].dtype == np.int8
    assert ds["status"].attrs["flag_values"].tolist() == [0, 1, 2, 3, 4]
    assert ds["status"].attrs["flag_meanings"] == "ONSET NOT_FOUND CENSORED PRIOR INSUFFICIENT_DATA"
    assert ds["lat"].attrs["units"] == "degrees_north" and ds["lat"].attrs["standard_name"] == "latitude"
    assert ds["lon"].attrs["units"] == "degrees_east" and ds["lon"].attrs["standard_name"] == "longitude"
    assert set(ds.data_vars) == {"onset_date", "onset_doy", "status"}


def test_global_attributes_hold_the_provenance(world):
    """Every file carries the CF convention, definition, thresholds, mask, history mode, lead window, paths and config."""
    cfg = world.cfgs["DET"]
    defn = onset_setup(export.case_kwargs(cfg, "DET")).definition
    for source, path in world.paths["DET"].items():
        attrs = _open(path).attrs
        assert attrs["Conventions"] == "CF-1.8"
        assert (attrs["definition_id"], attrs["definition_version"]) == (defn.id, defn.version)
        assert attrs["definition_hash"] == definition_mod.definition_hash(defn)
        assert attrs["definition_yaml"] == defn.source_yaml and "wet_day" in attrs["definition_yaml"]
        assert attrs["definition_sentence"] and attrs["definition_overrides"] == "reject if (dry_spell_check = false)"
        assert attrs["dry_spell_check"] == "off"
        assert attrs["threshold_source"] == "observations" and attrs["threshold_values_hash"] == "none"
        assert json.loads(attrs["mask_summary"])["n_cells"] == 9
        assert attrs["forecast_history"] == "observed" and attrs["lead_label"] == "start"
        assert attrs["lead_window"] == f"1-{MAX_LEAD}" and attrs["obs_day_boundary"] == BOUNDARY
        assert attrs["forecast_day_boundary"] == "00 UTC, labelled by start"
        assert json.loads(attrs["dataset_paths"])["observations"]["dir"] == str(world.base / "obs")
        assert attrs["romp_version"] and attrs["created"].endswith("Z") and attrs["years"] == "2019,2020"
        assert "region: ethiopia" in attrs["resolved_config_yaml"] or "ethiopia" in attrs["resolved_config_yaml"]
        assert attrs["source_name"] == source


def test_overrides_and_threshold_hash_are_recorded(world):
    """An onset override is listed, changes the hash, and a threshold definition records a values hash."""
    base = world.base
    raw = _raw(base, "DET", False, override={"wet_day": "at least 2 mm"})
    raw["output"]["data_dir"] = str(base / "out_override")
    cfg = export._complete(build_config(raw))
    path = export.export_onsets(cfg, sources=["observed"])["observed"]
    attrs = _open(path).attrs
    assert attrs["definition_overrides"] == "wet_day,reject if (dry_spell_check = false)"
    assert attrs["definition_hash"] != _open(world.paths["DET"]["observed"]).attrs["definition_hash"]
    assert "at least 2 mm" in attrs["definition_yaml"]
    hashed = export._threshold_hash({"amount": xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("lat", "lon"))})
    assert len(hashed) == 64 and hashed != "none"


def test_deterministic_model_layout(world):
    """A deterministic model has (init_time, season, lat, lon) variables with lead and status."""
    ds = _open(world.paths["DET"]["DET"])
    dims = ("init_time", "season", "lat", "lon")
    for name in ("onset_date", "onset_doy", "onset_lead", "onset_day", "status"):
        assert ds[name].dims == dims, name
    assert ds["scored"].dims == ("init_time", "lat", "lon")
    assert ds["init_time"].size == 62 and ds["init_time"].attrs["standard_name"] == "forecast_reference_time"
    assert ds["onset_lead"].dtype == ds["onset_doy"].dtype == np.float32 and ds["status"].dtype == np.int8
    assert ds["status"].attrs["flag_values"].tolist() == [0, 1, 2, 3, 4]
    assert "members_by_status" not in ds


def test_ensemble_layout(world):
    """An ensemble has a member dim, the engine's summary variables and the ensemble onset ROMP scores."""
    ds = _open(world.paths["ENS"]["ENS"])
    dims = ("init_time", "member", "season", "lat", "lon")
    for name in ("onset_date", "onset_doy", "onset_lead", "onset_day", "status"):
        assert ds[name].dims == dims, name
    assert ds["members_by_status"].dims == ("status_code", "init_time", "season", "lat", "lon")
    assert ds["onset_probability"].dims == ("init_time", "season", "lead", "lat", "lon")
    assert ds["ensemble_onset_date"].dims == ("init_time", "season", "lat", "lon")
    assert ds["lead"].attrs["units"] == "days" and "lead_label 'start'" in ds["lead"].attrs["comment"]
    assert list(ds["lead"].values) == list(range(MAX_LEAD + 1)) and ds["member"].size == MEMBERS
    assert (ds["members_by_status"].sum("status_code") == MEMBERS).all()
    assert ds["status_code"].attrs["flag_meanings"].startswith("ONSET")
    assert ds["ensemble_status"].dtype == np.int8 and ds["ensemble_onset_date"].attrs["comment"].count("mean")
    probability = ds["onset_probability"].values
    assert np.nanmin(probability) >= 0 and np.nanmax(probability) <= 1
    assert (np.diff(probability, axis=2)[~np.isnan(np.diff(probability, axis=2))] >= -1e-6).all()


def test_climatology_layouts(world):
    """The climatologies hold the per-cell onset and the per-init forecast, the conditional one also its probability."""
    clim = _open(world.paths["DET"]["climatology"])
    assert clim["climatological_onset_doy"].dims == clim["climatological_onset_date"].dims == ("season", "lat", "lon")
    for name in ("onset_date", "onset_doy", "onset_lead", "status"):
        assert clim[name].dims == ("init_time", "season", "lat", "lon"), name
    assert clim["climatological_onset_date"].dtype == np.dtype("datetime64[ns]")
    cond = _open(world.paths["DET"]["conditional_climatology"])
    assert cond["onset_probability"].dims == ("init_time", "season", "lead", "lat", "lon")
    assert cond.attrs["climatology_cv"] == "leave_one_year_out" and "climatology_cv_configured" in cond.attrs
    assert cond["onset_lead"].dims == ("init_time", "season", "lat", "lon") and cond["status"].dtype == np.int8


def test_nat_and_nan_round_trip(world):
    """No onset is NaT/NaN after decoding, a fill value on disk, and always agrees with the status."""
    path = world.paths["DET"]["DET"]
    ds = _open(path)
    none = ds["status"].values != Status.ONSET
    assert none.any() and (~none).any()
    assert np.isnat(ds["onset_date"].values[none]).all() and not np.isnat(ds["onset_date"].values[~none]).any()
    assert np.isnan(ds["onset_doy"].values[none]).all() and np.isnan(ds["onset_lead"].values[none]).all()
    assert not np.isnan(ds["onset_doy"].values[~none]).any()
    with netCDF4.Dataset(path) as raw:
        var = raw["onset_date"]
        var.set_auto_maskandscale(False)
        stored = var[:]
        assert var.dtype == np.int32 and var._FillValue == -2147483648 and var.units == "days since 1970-01-01"
        assert (stored[none] == -2147483648).all() and (stored[~none] != -2147483648).all()
        assert var.filters()["zlib"] and var.chunking() != "contiguous"
        assert raw["status"].dtype == np.int8 and raw["onset_doy"].dtype == np.float32


def test_doy_matches_the_date(world):
    """onset_doy is the date's day of the calendar year."""
    ds = _open(world.paths["DET"]["observed"])
    found = ds["status"].values == Status.ONSET
    dates = pd.DatetimeIndex(ds["onset_date"].values[found])
    assert (ds["onset_doy"].values[found] == dates.dayofyear.to_numpy()).all()


def test_non_standard_calendar_dates_round_trip(tmp_path):
    """cftime dates (noleap) are written with their calendar, none stays masked, day of year follows the calendar."""
    dates = np.array([[cftime.datetime(2020, 3, 1, calendar="noleap"), None],
                      [cftime.datetime(2020, 12, 31, calendar="noleap"), cftime.datetime(2021, 1, 2, calendar="noleap")]],
                     dtype=object)
    days = export._days(dates, "noleap")
    assert np.isnan(days[0, 1]) and days[0, 0] == cftime.date2num(dates[0, 0], export.DATE_UNITS, "noleap")
    assert export._doy(days, "noleap").tolist()[1][0] == 365.0 and np.isnan(export._doy(days, "noleap")[0, 1])
    ds = xr.Dataset({"onset_date": export._date_var(days, ("a", "b"), "noleap", "onset"),
                     "status": export.status_var([[0, 1], [0, 0]], ("a", "b"))}, coords={"a": [0, 1], "b": [0, 1]})
    path = export.write_netcdf(ds, tmp_path / "noleap.nc")
    with netCDF4.Dataset(path) as raw:
        assert raw["onset_date"].calendar == "noleap" and raw["onset_date"].units == "days since 1970-01-01"
        assert bool(raw["onset_date"][:].mask[0, 1])
    got = export.open_onsets(path)["onset_date"].values
    assert got[0, 0] == dates[0, 0] and got[1, 1] == dates[1, 1] and got[1, 0] == dates[1, 0]
    assert got[0, 1] is None


def test_non_standard_calendar_observed_export(tmp_path):
    """Observed onsets on a noleap time axis export through the real path with the calendar in the file."""
    base = tmp_path
    (base / "obs").mkdir()
    for year in CLIM_YEARS:
        time, rain = _obs_rain(year)
        time = xr.date_range(f"{year}-05-01", f"{year}-08-31", calendar="noleap", use_cftime=True)
        xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm"})},
                   coords={"time": time, "lat": LAT, "lon": LON}).to_netcdf(base / "obs" / f"{year}.nc")
    _write_forecast(base / "DET", 0)
    cfg = export._complete(build_config(_raw(base, "DET", False)))
    path = export.export_onsets(cfg, sources=["observed"], out=base / "nl")["observed"]
    ds = export.open_onsets(path)
    assert ds.attrs["calendar"] == "noleap"
    found = ds["status"].values == Status.ONSET
    assert found.any() and all(d.calendar == "noleap" for d in ds["onset_date"].values[found])


# --- equality with what the scoring consumes --------------------------------------------------------------

def _rows(ds, frame, variables):
    """The exported variables at the (init_time, lat, lon) rows of a scored table, as arrays."""
    pick = dict(init_time=xr.DataArray(pd.to_datetime(frame["init_time"]).values, dims="row"),
                lat=xr.DataArray(frame["lat"].values, dims="row"), lon=xr.DataArray(frame["lon"].values, dims="row"))
    if "member" in frame:
        pick["member"] = xr.DataArray(frame["member"].values, dims="row")
    sel = ds[variables].sel(**pick).isel(season=0)
    return {name: sel[name].values for name in variables}


def _same_dates(exported, strings):
    """Exported datetime64 values equal a table's 'YYYY-MM-DD' (or None) column."""
    want = pd.to_datetime(pd.Series(strings, dtype=object)).to_numpy()
    assert (np.isnat(exported) == np.isnat(want)).all()
    assert (exported[~np.isnat(want)] == want[~np.isnat(want)]).all()


def _scoring_inputs(cfg, model):
    """What the scoring does per year: kwargs, thresholds and per year (year, rain, observed onsets, history)."""
    kw = export.case_kwargs(cfg, model)
    thresholds = load_thresholds(**kw)
    out = []
    for year in kw["years"]:
        rain = load_imd_rainfall(year, **kw)
        out.append((year, rain, detect_observed_onset(rain, thresholds, year, **kw), load_history(year, rain, **kw)))
    return kw, thresholds, out


def test_observed_equals_the_onsets_the_scoring_detects(world):
    """Exported observed dates are the (lat, lon) onset arrays detect_observed_onset gives for each year."""
    ds = _open(world.paths["DET"]["observed"])
    _, _, inputs = _scoring_inputs(world.cfgs["DET"], "DET")
    assert len(inputs) == 2
    for year, _, onset_da, _ in inputs:
        got = ds["onset_date"].sel(year=year).isel(season=0).transpose("lat", "lon")
        assert got.shape == onset_da.shape
        assert (got.values.astype("datetime64[ns]") == onset_da.values)[~np.isnat(onset_da.values)].all()
        assert (np.isnat(got.values) == np.isnat(onset_da.values)).all()


def test_deterministic_model_equals_the_scored_table(world):
    """Every row of the table the deterministic scoring consumes is in the file with the same date, day and status."""
    ds = _open(world.paths["DET"]["DET"])
    kw, thresholds, inputs = _scoring_inputs(world.cfgs["DET"], "DET")
    total = 0
    for year, rain, onset_da, history in inputs:
        p_model = get_forecast_deterministic_twice_weekly(year, **kw)
        frame = compute_onset_for_deterministic_model(p_model, thresholds, onset_da, history=history, **kw)
        got = _rows(ds, frame, ["onset_date", "onset_day", "onset_doy", "status", "scored"])
        _same_dates(got["onset_date"], frame["onset_date"])
        want_day = frame["onset_day"].astype(float).to_numpy()
        assert np.array_equal(got["onset_day"], want_day, equal_nan=True)
        assert [Status(c).name for c in got["status"]] == frame["onset_status"].tolist()
        assert (got["scored"] == 1).all()
        total += len(frame)
    assert total > 100 and int(ds["scored"].sum()) == total
    assert (ds["status"].values == Status.ONSET).any() and np.isfinite(ds["onset_lead"].values).any()


def test_ensemble_model_equals_the_scored_tables(world):
    """Members and the ensemble onset match the member and ensemble tables the probabilistic scoring consumes."""
    ds = _open(world.paths["ENS"]["ENS"])
    kw, thresholds, inputs = _scoring_inputs(world.cfgs["ENS"], "ENS")
    total = 0
    for year, rain, onset_da, history in inputs:
        p_model = get_forecast_probabilistic_twice_weekly(year, **kw)
        members, ensemble = compute_onset_for_all_members(p_model, thresholds, onset_da, history=history, **kw)
        got = _rows(ds, members, ["onset_day", "status"])
        assert np.array_equal(got["onset_day"], members["onset_day"].astype(float).to_numpy(), equal_nan=True)
        assert [Status(c).name for c in got["status"]] == members["onset_status"].tolist()
        got = _rows(ds, ensemble, ["ensemble_onset_date", "ensemble_onset_day", "ensemble_status"])
        _same_dates(got["ensemble_onset_date"], ensemble["onset_date"])
        assert np.array_equal(got["ensemble_onset_day"], ensemble["onset_day"].astype(float).to_numpy(),
                              equal_nan=True)
        assert [Status(c).name for c in got["ensemble_status"]] == ensemble["onset_status"].tolist()
        total += len(ensemble)
    assert total > 100 and int(ds["scored"].sum()) == total
    # the engine's ensemble summary: probability at the last lead equals the share of members with an onset by then
    onset = ds["onset_lead"].values
    reached = (onset <= MAX_LEAD).sum(axis=1) / (ds["status"].values != Status.INSUFFICIENT_DATA).sum(axis=1)
    usable = (ds["status"].values != Status.INSUFFICIENT_DATA).sum(axis=1) > 0
    usable &= ds["scored"].values[:, None] == 1  # PRIOR members (unscored inits) count at lead 0 in the engine only
    assert usable.sum() > 100
    assert np.allclose(ds["onset_probability"].values[:, :, -1][usable], reached[usable], atol=1e-6)


def test_climatology_equals_the_scored_table(world):
    """The climatology per-cell onset and per-init forecast match compute_climatology_as_forecast."""
    ds = _open(world.paths["DET"]["climatology"])
    kw = export.case_kwargs(world.cfgs["DET"], "climatology", reference=True)
    clim_doy = compute_climatological_onset(**kw)
    assert np.array_equal(ds["climatological_onset_doy"].isel(season=0).values, clim_doy.values.astype(np.float32),
                          equal_nan=True)
    doy = ds["climatological_onset_date"].isel(season=0)
    assert (doy.dt.dayofyear.values[~np.isnan(clim_doy.values)] == clim_doy.values[~np.isnan(clim_doy.values)]).all()
    total, found = 0, 0
    _, _, inputs = _scoring_inputs(world.cfgs["DET"], "DET")
    for year, _, onset_da, _ in inputs:
        frame = compute_climatology_as_forecast(clim_doy, year, get_initialization_dates(year, **kw), onset_da, **kw)
        got = _rows(ds, frame, ["onset_date", "onset_day", "onset_lead", "onset_doy", "status", "scored"])
        _same_dates(got["onset_date"], frame["onset_date"])
        assert np.array_equal(got["onset_day"], frame["onset_day"].astype(float).to_numpy(), equal_nan=True)
        assert np.array_equal(got["onset_lead"], got["onset_day"], equal_nan=True) and (got["scored"] == 1).all()
        assert ((got["status"] == Status.ONSET) == ~np.isnan(got["onset_day"])).all()
        total, found = total + len(frame), found + int(frame["onset_date"].notna().sum())
    assert total > 100 and found > 0 and int(ds["scored"].sum()) == total


def test_conditional_climatology_equals_the_scored_table_and_pairs(world):
    """The conditional median onset matches the scored table, and the probability reproduces the scored bin probabilities."""
    ds = _open(world.paths["DET"]["conditional_climatology"])
    kw = export.case_kwargs(world.cfgs["DET"], "conditional_climatology", reference=True)
    clim_onset = compute_climatological_onset_dataset(**kw)
    _, _, inputs = _scoring_inputs(world.cfgs["DET"], "DET")
    total, found = 0, 0
    for year, _, onset_da, _ in inputs:
        init_dates = get_initialization_dates(year, **kw)
        frame = compute_conditional_climatology_as_forecast(clim_onset, year, init_dates, onset_da, **kw)
        got = _rows(ds, frame, ["onset_date", "onset_day", "onset_lead", "status"])
        _same_dates(got["onset_date"], frame["onset_date"])
        assert np.array_equal(got["onset_day"], frame["onset_day"].astype(float).to_numpy(), equal_nan=True)
        total, found = total + len(frame), found + int(frame["onset_date"].notna().sum())
        pairs = create_conditional_climatological_forecast_obs_pairs(clim_onset, year, init_dates, onset_da,
                                                                     day_bins=((1, 5), (6, 10)))
        pairs = pairs[pairs["bin_label"].isin(["Days 1-5", "Days 6-10"])].copy()
        pairs["init_time"] = pd.to_datetime(pairs["init_time"])
        sel = ds["onset_probability"].isel(season=0).sel(
            init_time=xr.DataArray(pairs["init_time"].values, dims="row"), lat=xr.DataArray(pairs["lat"].values, dims="row"),
            lon=xr.DataArray(pairs["lon"].values, dims="row"))
        upper = np.where(pairs["bin_label"] == "Days 1-5", 5, 10)
        lower = upper - 5
        mass = sel.values[np.arange(len(pairs)), upper] - sel.values[np.arange(len(pairs)), lower]
        assert np.allclose(mass, pairs["predicted_prob"].to_numpy(), atol=1e-5)
    assert total > 100 and found > 0 and int(ds["scored"].sum()) == total
    cdf = ds["onset_probability"].values
    assert np.nanmax(cdf) <= 1 + 1e-6 and (cdf[:, :, 0][~np.isnan(cdf[:, :, 0])] == 0).all()


def test_lead_window_matches_the_scored_window(world):
    """The recorded lead window is the shortened one the run scored (no CENSORED results appear)."""
    cfg = world.cfgs["DET"]
    assert cfg["lead_window"] == (1, MAX_LEAD) and cfg["max_forecast_day"] == MAX_LEAD
    ds = _open(world.paths["DET"]["DET"])
    assert not (ds["status"].values == Status.CENSORED).any()
    assert ds.attrs["max_forecast_day"] == MAX_LEAD


def test_unknown_source_is_an_error(world, tmp_path):
    """A source the config cannot export is rejected with the allowed names."""
    with pytest.raises(ValueError, match="unknown source.*Nope.*observed"):
        export.export_onsets(world.cfgs["DET"], sources="observed,Nope", out=tmp_path)


def test_save_onsets_is_a_config_switch(world):
    """output.save_onsets exists, defaults to false and flattens to the flat key."""
    assert world.cfgs["DET"]["save_onsets"] is False
    raw = _raw(world.base, "DET", False)
    raw["output"]["save_onsets"] = True
    assert build_config(raw)["save_onsets"] is True


# --- command line and run switch ---------------------------------------------------------------------------

def _cli(cwd, *args):
    """Run `python -m momp.cli` in cwd."""
    return subprocess.run([sys.executable, "-m", "momp.cli", *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})


def test_cli_export_writes_the_requested_files(world, tmp_path):
    """`momp-run export -p run.yaml --sources ... --out DIR` writes those files and nothing else."""
    raw = _raw(world.base, "DET", False)
    raw["output"]["data_dir"] = str(tmp_path / "out")
    raw["output"]["figure_dir"] = str(tmp_path / "fig")
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(raw))
    done = _cli(tmp_path, "export", "-p", "run.yaml", "--sources", "observed,DET", "--out", "files")
    assert done.returncode == 0, done.stderr[-2000:] + done.stdout[-1000:]
    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == ["onsets_DET.nc", "onsets_observed.nc"]
    assert "onsets_DET.nc" in done.stdout
    same = _open(tmp_path / "files" / "onsets_DET.nc")["onset_date"].values
    ours = _open(world.paths["DET"]["DET"])["onset_date"].values
    assert ((same == ours) | (np.isnat(same) & np.isnat(ours))).all()
    default = _cli(tmp_path, "export", "-p", "run.yaml", "--sources", "observed")
    assert default.returncode == 0 and (tmp_path / "out" / "onsets" / "onsets_observed.nc").exists()
    bad = _cli(tmp_path, "export", "-p", "run.yaml", "--sources", "Nope")
    assert bad.returncode == 1 and "unknown source" in bad.stderr
    assert "export" in _cli(tmp_path, "-h").stdout


def test_save_onsets_writes_files_during_a_normal_run(world, tmp_path):
    """A normal run with output.save_onsets: true leaves the files in dir_out/onsets/ next to its other outputs."""
    raw = _raw(world.base, "DET", False)
    raw["output"].update(data_dir=str(tmp_path / "out"), figure_dir=str(tmp_path / "fig"), save_onsets=True,
                         save_fig=False, plot_spatial_far_mr_mae=False, plot_climatology_onset=False,
                         plot_panel_heatmap_error=False, save_nc_climatology=False)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(raw))
    done = _cli(tmp_path, "-p", "run.yaml")
    assert done.returncode == 0, done.stderr[-2000:] + done.stdout[-1500:]
    files = sorted(p.name for p in (tmp_path / "out" / "onsets").iterdir())
    assert files == ["onsets_DET.nc", "onsets_climatology.nc", "onsets_observed.nc"]
    assert (tmp_path / "out" / "resolved_config.yaml").exists()
