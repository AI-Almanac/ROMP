"""The typed readers of hcwf.io (R2 S6): the typed specs built by hand give what the specs of a flat config give, for grid,
units, listed-files and zarr data; `Selection` names every per-call key the readers take that is not config."""
import typed_io
import flat_values
import ast
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from _inputs import write_adm3, write_forecast
from hcwf.config.model import (AreaSpec, EvaluationSpec, ModelSpec, ObsSpec, OnsetOptions, OutputSpec, Selection,
                               Window)
from hcwf.io import forecasts as forecasts_io
from hcwf.io import observations as observations_io
from hcwf.io import reference as reference_io
from hcwf.io import thresholds as thresholds_io
from hcwf.spatial import onto_units
from hcwf.config import schema
from hcwf.config.onset import onset_setup
from hcwf.spatial.units import UnitsSpec

BOUNDARY = "00 UTC, labelled by start"
ONSET = {"onset_definition": "icpac", "obs_day_boundary": BOUNDARY, "model_lead_label": "start"}
LAT = np.arange(2.0, 17.0, 1.0)
LON = np.arange(32.0, 50.0, 1.0)


@pytest.fixture(autouse=True)
def fresh():
    """No observation years held and no thresholds loaded before or after a test."""
    reference_io.clear_obs_cache()
    thresholds_io._loaded_thresholds.clear()
    yield
    reference_io.clear_obs_cache()
    thresholds_io._loaded_thresholds.clear()


# -------------------------------------------------------------------------------------------- typed specs by hand

def obs_spec(**values) -> ObsSpec:
    """An ObsSpec with the fields not given left unset."""
    fields = dict(name="obs", dir=None, file_pattern=None, var="rain", unit_cvt=None, paths=None, dims=None,
                  day_boundary=None, units=None)
    return ObsSpec(**{**fields, **values})


def model_spec(**values) -> ModelSpec:
    """A ModelSpec with the fields not given left unset."""
    fields = dict(name="M", dir=None, var="tp", unit_cvt=None, file_pattern="{}.nc", accumulation=None,
                  accumulation_detected=None, daily_boundary=None, dims=None, paths=None, units=None)
    return ModelSpec(**{**fields, **values})


def area_spec(region=None, **values) -> AreaSpec:
    """An AreaSpec without a mask section."""
    return AreaSpec(region=region, mask=values.get("mask"), nc_mask=values.get("nc_mask"), shpfile_dir=None,
                    polygon=False)


def options_spec(**values) -> OnsetOptions:
    """OnsetOptions with every option unset except those given."""
    fields = {f.name: None for f in dataclasses.fields(OnsetOptions)}
    return OnsetOptions(**{**fields, "percentage_threshold": 0.5, **values})


def evaluation_spec(**values) -> EvaluationSpec:
    """An EvaluationSpec of the 2020 season."""
    fields = dict(start_date=(2020, 5, 1), end_date=(2020, 7, 31), years=(2020,), climatology_years=(2020,),
                  init_days=(0, 3), date_filter_year=2024, windows=(Window(1, 10, 0),), max_forecast_day=10,
                  day_bins=((1, 10),), probabilistic=False, members=None, lead_window=None)
    return EvaluationSpec(**{**fields, **values})


def output_spec(tmp_path) -> OutputSpec:
    """An OutputSpec writing under tmp_path."""
    return OutputSpec(**{f.name: None for f in dataclasses.fields(OutputSpec)} | {
        "data_dir": str(tmp_path / "out"), "figure_dir": None})


def write_obs_year(path, year, seed):
    """A year of daily observations on (time, lat, lon), float32."""
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    data = np.random.default_rng(seed).gamma(0.5, 4.0, (len(time), len(LAT), len(LON))).astype("float32")
    xr.Dataset({"rain": (("time", "lat", "lon"), data)}, coords={"time": time, "lat": LAT, "lon": LON}).to_netcdf(path)


# ------------------------------------------------------------------------------------------------ observations

def test_grid_observations_read_the_same_from_flat_keys(tmp_path):
    write_obs_year(tmp_path / "2020.nc", 2020, 1)
    flat = dict(obs_dir=str(tmp_path), obs_file_pattern=["{}.nc"], obs_var="rain", obs_unit_cvt=2.0,
                region="Ethiopia", shp_only=False, years=[2020])
    old = typed_io.observations(2020, flat)
    reference_io.clear_obs_cache()
    direct = observations_io.load_observations(
        2020, obs_spec(dir=str(tmp_path), file_pattern=("{}.nc",), unit_cvt=2.0), area_spec("Ethiopia"),
        options_spec(), Selection(shp_only=False), run_years=(2020,))
    xr.testing.assert_identical(old, direct)
    assert np.isnan(direct.values).any() and not np.isnan(direct.values).all()   # the land mask kept some cells


def test_zarr_observations_read_the_same_from_flat_keys(tmp_path):
    for year, seed in ((2019, 2), (2020, 3)):
        write_obs_year(tmp_path / f"{year}.nc", year, seed)
    store = tmp_path / "obs.zarr"
    xr.concat([xr.open_dataset(tmp_path / f"{y}.nc") for y in (2019, 2020)], dim="time").to_zarr(store)
    flat = dict(obs_dir=str(store), obs_file_pattern=["{}.nc"], obs_var="rain", obs_unit_cvt=None, region="Ethiopia",
                shp_only=False, years=[2020])
    old = typed_io.observations(2020, flat)
    reference_io.clear_obs_cache()
    obs, area, options = obs_spec(dir=str(store), file_pattern=("{}.nc",)), area_spec("Ethiopia"), options_spec()
    direct = observations_io.load_observations(2020, obs, area, options, Selection(shp_only=False), run_years=(2020,))
    xr.testing.assert_identical(old, direct)
    from_files = observations_io.load_observations(2020, obs_spec(dir=str(tmp_path), file_pattern=("{}.nc",)), area, options,
                                            Selection(shp_only=False))
    np.testing.assert_array_equal(direct.values, from_files.values)
    both = [rain for rain, error in observations_io.read_obs_years([2019, 2020], obs, area, options,
                                                            Selection(shp_only=False), run_years=(2019, 2020))]
    flat_reads = [rain for rain, error in typed_io.obs_years([2019, 2020], {**flat, "years": [2019, 2020]})]
    for typed, flat_read in zip(both, flat_reads):
        xr.testing.assert_identical(typed, flat_read)


def test_listed_files_and_a_grid_point_read_the_same_from_flat_keys(tmp_path):
    write_obs_year(tmp_path / "obs_2020_v2.nc", 2020, 4)
    flat = dict(obs_dir=str(tmp_path), obs_paths=["obs_*.nc"], obs_var="rain", obs_unit_cvt=None, region=None,
                grid_point=True, lat_select=5.0, lon_select=40.0, obs_files_years=[2020])
    old = typed_io.observations(2020, flat)
    direct = observations_io.load_observations(
        2020, obs_spec(dir=str(tmp_path), paths=("obs_*.nc",)), area_spec(), options_spec(),
        Selection(grid_point=True, lat=5.0, lon=40.0, obs_files_years=[2020]))
    xr.testing.assert_identical(old, direct)
    assert direct.sizes["lat"] == 1 and direct.sizes["lon"] == 1
    with pytest.raises(FileNotFoundError):
        observations_io.load_observations(2019, obs_spec(dir=str(tmp_path), paths=("obs_*.nc",)), area_spec(),
                                   options_spec(), Selection())


def test_unit_observations_read_the_same_from_flat_keys(tmp_path):
    write_adm3(tmp_path / "adm3_2003.nc")
    flat = {**ONSET, "obs_dir": str(tmp_path), "obs_file_pattern": ["adm3_{}.nc"], "obs_var": "precip",
            "obs_unit_cvt": None, "region": None, "units_mode": "cells", "years": [2003]}
    old = typed_io.observations(2003, flat)
    reference_io.clear_obs_cache()
    obs = obs_spec(dir=str(tmp_path), file_pattern=("adm3_{}.nc",), var="precip", units=UnitsSpec(mode="cells"))
    direct = observations_io.load_observations(2003, obs, area_spec(), options_spec(), Selection(), onset=onset_setup(flat),
                                        run_years=(2003,))
    xr.testing.assert_identical(old, direct)
    assert "unit" in direct.dims and bool(direct["present"].all())
    assert onto_units.point_dims(obs) == onto_units.point_dims(typed_io.inputs(flat).obs) == ("unit",)
    with pytest.raises(ValueError, match="onset setup"):
        observations_io.load_observations(2003, obs, area_spec(), options_spec(), Selection())


# --------------------------------------------------------------------------------------------------- forecasts

def write_model_year(path, year, ensemble):
    """A year of daily forecasts on (time, [number], day, lat, lon), inits May-July, 14 days."""
    init = pd.date_range(f"{year}-05-01", f"{year}-07-31")
    write_forecast(path, init, members=3 if ensemble else 1, days=14, lats=LAT, lons=LON)


@pytest.mark.parametrize("ensemble", [False, True])
def test_forecasts_read_the_same_from_flat_keys(tmp_path, ensemble):
    write_model_year(tmp_path / "2020.nc", 2020, True)
    store = tmp_path / "m.zarr"
    xr.open_dataset(tmp_path / "2020.nc").to_zarr(store)
    ev = evaluation_spec(members=(0, 2) if ensemble else None, probabilistic=ensemble)
    onset = onset_setup({**ONSET, "max_forecast_day": 10})
    dims = {"member": "number", "step": "day"}
    for where in (str(tmp_path), str(store)):
        flat = dict(model_dir=where, model="M", model_var="tp", unit_cvt=1.0, file_pattern="{}.nc", region="Ethiopia",
                    members=ev.members, date_filter_year=2024, init_days=[0, 3], start_date=(2020, 5, 1),
                    end_date=(2020, 7, 31), max_forecast_day=10, model_dims=dims, shp_only=False, **ONSET)
        old = typed_io.forecast(2020, flat, ensemble)
        model = model_spec(dir=where, unit_cvt=1.0, dims=tuple(dims.items()))
        direct = forecasts_io.load_forecast(2020, model, ev, area_spec("Ethiopia"), options_spec(), onset,
                                        ensemble=ensemble, select=Selection(shp_only=False))
        xr.testing.assert_identical(old, direct)
        assert direct.sizes["step"] == min(14, forecasts_io.steps_needed(10, onset))   # the file holds 14 days
        assert list(pd.to_datetime(direct.init_time.values)) == list(forecasts_io.initialization_dates(2020, ev))
        assert list(forecasts_io.initialization_dates(2020, ev)) == list(typed_io.initialization_dates(2020, flat))


def test_listed_forecast_files_read_the_same_from_flat_keys(tmp_path):
    days = [pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-04")]
    for day in days:
        write_forecast(tmp_path / f"live_{day:%Y%m%d}.nc", [day], members=2, days=14, lats=LAT, lons=LON)
    flat = dict(model_dir=str(tmp_path), model="M", model_var="tp", unit_cvt=None, file_pattern="{}.nc", region=None,
                members=None, date_filter_year=2024, init_days=list(range(7)), start_date=(2020, 5, 1),
                end_date=(2020, 7, 31), model_paths=["live_*.nc"], model_files_years=[2020],
                model_dims={"member": "number", "step": "day"})
    old = typed_io.forecast(2020, flat, True)
    ev = evaluation_spec(init_days=tuple(range(7)), max_forecast_day=None)
    direct = forecasts_io.load_forecast(
        2020, model_spec(dir=str(tmp_path), dims=(("member", "number"), ("step", "day"))), ev, area_spec(),
        options_spec(), None, ensemble=True, select=Selection(model_paths=["live_*.nc"], model_files_years=[2020]))
    xr.testing.assert_identical(old, direct)
    assert direct.sizes["init_time"] == 2


def test_a_unit_forecast_reads_the_same_from_a_model_spec_and_from_flat_keys(tmp_path):
    path = write_forecast(tmp_path / "f.nc", [pd.Timestamp("2020-06-01")], members=3, days=6, lats=LAT[:3], lons=LON[:3])
    units = UnitsSpec(mode="cells")
    flat = {"model_var": "tp", "model_dims": {"member": "number", "step": "day"}, "unit_cvt": None, "first_lead": 2,
            "last_lead": 5, "max_member": 1}
    got = typed_io.inputs(flat)
    with forecasts_io.UnitForecast(path, got.model, units, select=got.select) as old:
        old_runs = [rain.values for _, _, rain in old.runs()]
        old_days, old_members = old.days.tolist(), old.members.tolist()
    model = model_spec(dims=(("member", "number"), ("step", "day")))
    with forecasts_io.UnitForecast(path, model, units, select=Selection(first_lead=2, last_lead=5, max_member=1)) as new:
        runs = [rain.values for _, _, rain in new.runs()]
        assert new.days.tolist() == old_days == [2, 3, 4, 5] and new.members.tolist() == old_members == [0, 1]
    np.testing.assert_array_equal(np.concatenate(runs), np.concatenate(old_runs))


# --------------------------------------------------------------------------------------------------- thresholds

def test_thresholds_read_the_same_from_flat_keys(tmp_path):
    from test_phase7_perf import _obs_files, _reference_config

    _obs_files(tmp_path / "obs", range(2000, 2007))
    flat = {**_reference_config(tmp_path), "region": None, "cache_dir": str(tmp_path / "c1")}
    old = typed_io.thresholds(flat)
    thresholds_io._loaded_thresholds.clear()
    reference_io.clear_obs_cache()
    obs = obs_spec(dir=str(tmp_path / "obs"), file_pattern=("obs_{}.nc",), day_boundary=flat["obs_day_boundary"])
    ev = evaluation_spec(start_date=(2000, 5, 1), end_date=(2000, 8, 31), years=(2006,), max_forecast_day=10,
                         climatology_years=tuple(range(2000, 2007)), windows=())
    direct = thresholds_io.load_thresholds(onset_setup(flat), obs, area_spec(), options_spec(), ev,
                                      output_spec(tmp_path), model=model_spec(),
                                      cache_dir=str(tmp_path / "c2"))
    assert set(old) == set(direct) and old
    for name in old:
        xr.testing.assert_equal(old[name], direct[name])
    shutil.rmtree(tmp_path / "c1", ignore_errors=True)


# ------------------------------------------------------------------------------------- options, units and keys

def test_onset_options_give_the_ingest_and_engine_keywords():
    assert options_spec().ingest(False, None) == {} and options_spec().ingest(True, None) == {}
    options = options_spec(rain_resolution="exact", missing="keep", negatives="keep", gaps="refuse", horizon="truncate")
    assert options.ingest(False, None) == {"rain_resolution": "exact", "missing": "keep", "negatives": "keep",
                                           "gaps": "refuse"}
    assert options.ingest(True, None, check_units=False) == {"rain_resolution": "exact", "missing": "keep",
                                                             "negatives": "keep", "check_units": False}
    assert options.engine(None) == {"horizon": "truncate", "missing": "keep"}
    assert options_spec().engine(None) == {"horizon": "censor", "missing": None}
    transform = UnitsSpec(mode="cell_transform", weights="w.csv")   # the policy is applied to the source cells instead
    assert options_spec(missing="zero").ingest(False, transform)["missing"] == "keep"
    assert options_spec(missing="zero").ingest(True, transform)["missing"] == "zero"
    flat = {"rain_resolution": "exact", "missing": "keep", "gaps": "refuse", "check_units": False}
    assert typed_io.ingest_options(flat) == {"rain_resolution": "exact", "missing": "keep", "gaps": "refuse"}
    assert typed_io.ingest_options(flat, forecast=True) == {"rain_resolution": "exact", "missing": "keep",
                                                            "check_units": False}


def test_units_are_resolved_once_per_source_and_read_from_the_spec():
    flat = {"units_mode": {"observations": "cell_transform", "ens": "regrid"},
            "units_weights": {"observations": "o.csv", "ens": "m.csv"}, "units_shapefile": "s.shp",
            "units_unit_id_col": "id"}
    obs, ens = UnitsSpec.from_flat(flat, "observations"), UnitsSpec.from_flat(flat, "ens")
    assert (obs.how, obs.weights, obs.shapefile) == ("cell_transform", "o.csv", None) and ens.weights == "m.csv"
    assert onto_units.units_spec(obs_spec(units=obs)) is obs and onto_units.units_spec(model_spec(units=ens)) is ens
    assert onto_units.units_spec(obs_spec()) is None and onto_units.units_files(obs_spec()) == []
    assert set(UnitsSpec.from_flat.__globals__["UNITS_KEYS"]) == {f.key for f in schema.FIELDS if f.section == "units"}


# the modules whose flat-key reads moved to the typed specs and `Selection` (design S6)
MODULES = ("io/stores.py", "io/observations.py", "io/reference.py", "io/forecasts.py", "io/thresholds.py", "io/warmup.py",
           "spatial/onto_units.py", "spatial/units.py", "spatial/select.py", "spatial/masks.py", "io/paths.py")
CONFIG_NAMES = {"kwargs", "kw", "cfg", "config", "values", "dic"}


def _keys_read(path):
    """Every literal key read from a flat config in a module: name.get("key"), name["key"]."""
    tree = ast.parse(path.read_text())
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in CONFIG_NAMES \
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys.add(node.args[0].value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in CONFIG_NAMES \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
    return keys


def test_selection_keys_cover_every_non_schema_key_the_readers_read():
    root = Path(__file__).resolve().parents[2] / "hcwf"
    named = set(flat_values.SELECTION_KWARGS.values())
    assert set(flat_values.SELECTION_KWARGS) == {f.name for f in dataclasses.fields(Selection)}
    read = {key for module in MODULES for key in _keys_read(root / module)}
    outside = sorted(read - set(schema.BY_KEY) - named - set(flat_values._RUNTIME))
    assert not outside, f"read from a flat config but neither a schema key nor a Selection key: {outside}"


def test_the_selection_names_every_selection_key_the_old_readers_took():
    """The keys the old readers took from kwargs beyond the schema: each is a Selection field, so a call that sets it
    passes it through the Selection."""
    old_extras = {"keep_unit_ids", "member_ids", "max_member", "issue_dates", "first_lead", "last_lead",
                  "model_files_years", "obs_files_years", "model_paths", "grid_point", "lat_select", "lon_select",
                  "land_only", "shp_only", "reference_onset", "rain_horizon", "check_units"}
    assert old_extras == set(flat_values.SELECTION_KWARGS.values())
    got = typed_io.reader_inputs({"keep_unit_ids": ["a"], "lat_select": 1.0, "land_only": False})
    assert got.select == Selection(keep_unit_ids=["a"], lat=1.0, land_only=False)
