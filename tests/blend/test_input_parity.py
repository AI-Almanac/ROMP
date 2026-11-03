"""hcwf's single input path reads what onset_blending's legacy table readers read (tests/_legacy/nc_utils.py, copied
into _oracle/nc_utils_io.py): observations, adm3 files, forecast files, sparse tile reads, and the aggregation of source
cells onto units in either order of scaling."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import typed_io
import xarray as xr

from _inputs import DAYS, LATS, LONS, grid_weights, id_weights, write_adm3, write_forecast, write_observations
from _oracle import nc_utils_filters as filters
from _oracle import nc_utils_io as legacy
from _oracle.spatial_id_utils import GridIdConvention
from hcwf.spatial import onto_units
from hcwf.onset.ingest import ingest
from hcwf.spatial import EvaluationUnits
from hcwf.spatial.select import load_blocks
from hcwf.io.dims import dim_fmt, dim_fmt_model_ensemble

BOUNDARY = "00 UTC, labelled by start"
CONVENTION = GridIdConvention(2, "fixed", "explicit")
FORECAST_KWARGS = dict(model_var="tp", region=None, date_filter_year=2014, init_days=list(range(7)),
                       start_date=(0, 5, 1), end_date=(0, 7, 31), members=None)
INITS = [dt.date(2014, 6, 2), dt.date(2014, 6, 9), dt.date(2014, 6, 16)]


def legacy_ids(df):
    return filters.ensure_id_col(df, convention=CONVENTION)


def legacy_spec(tmp_path, weights):
    path = tmp_path / "weights.csv"
    weights.to_csv(path, index=False)
    return {"options": {"cell_transform_enabled": True, "cell_transform_file": str(path)},
            "geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}


def hcwf_obs(path, year, **kwargs):
    return typed_io.observations(year, dict(obs_var="precip", obs_unit_cvt=None, region=None, obs_paths=[str(path)],
                                            **kwargs))


def units_of(weights, id_keyed=False):
    return EvaluationUnits.from_weights(id_weights() if id_keyed else weights)


# -------------------------------------------------------------------------------------------- observations

def test_observations_read_as_the_legacy_table_reads_them(tmp_path):
    write_observations(tmp_path, years=(2003,))
    old = legacy.nc_read_groundtruth_long(str(tmp_path / "2003.nc"), "precip", {})
    new = hcwf_obs(tmp_path / "2003.nc", 2003)
    assert new.dims == ("time", "lat", "lon") and new.dtype == np.float32
    np.testing.assert_array_equal(new.values.reshape(-1), old["precip"].to_numpy())
    assert (new.values < 0).any()  # negative values are in the data, and stay
    assert pd.DatetimeIndex(new["time"].values).date.tolist() == sorted(set(old["time"]))


def test_no_region_box_means_the_files_own_order(tmp_path):
    lats = [10.5, 10.25, 10.0]
    ds = xr.Dataset({"precip": (("time", "lat", "lon"), np.arange(24, dtype="float32").reshape(4, 3, 2),
                                {"units": "mm"})},
                    coords={"time": pd.date_range("2003-01-01", periods=4), "lat": lats, "lon": [70.0, 70.25]})
    ds.to_netcdf(tmp_path / "2003.nc")
    new = hcwf_obs(tmp_path / "2003.nc", 2003)
    assert new["lat"].values.tolist() == lats and new.values.tolist() == ds["precip"].values.tolist()
    boxed = typed_io.observations(2003, dict(obs_var="precip", obs_unit_cvt=None, region="india", land_only=False,
                                             shp_only=False, obs_paths=[str(tmp_path / "2003.nc")]))
    assert boxed["lat"].values.tolist() == sorted(lats)  # HCWF's region read puts latitude ascending


def test_negative_values_are_kept_or_flagged_as_asked(tmp_path):
    write_observations(tmp_path, years=(2003,))
    da = hcwf_obs(tmp_path / "2003.nc", 2003)
    default = ingest(da, day_boundary=BOUNDARY, units="mm")
    kept = ingest(da, day_boundary=BOUNDARY, units="mm", rain_resolution="exact", negatives="keep", missing="keep")
    values = da.transpose("lat", "lon", "time").values.astype(np.float64)
    finite = np.isfinite(values)
    np.testing.assert_array_equal(kept.missing, ~finite)
    np.testing.assert_array_equal(kept.tenths[finite], values[finite])
    assert (values[finite] < -0.05).any()
    assert default.missing[finite & (values < -0.05)].all()  # HCWF's default: bad data
    forecast = da.rename(time="init_time").expand_dims(lead=[1]).transpose("init_time", "lat", "lon", "lead")
    f_default = ing_forecast(forecast, "default")
    f_keep = ing_forecast(forecast, "keep")
    assert (f_default.tenths >= 0).all() and (f_keep.tenths[~f_keep.missing] < 0).any()
    with pytest.raises(ValueError, match="negatives"):
        ingest(da, day_boundary=BOUNDARY, units="mm", negatives="clip")


def ing_forecast(da, negatives):
    from hcwf.onset.ingest import ingest_forecast

    return ingest_forecast(da, day_boundary=BOUNDARY, lead_label="start", units="mm", accumulation="daily",
                           check_units=False, rain_resolution="exact", negatives=negatives)


def test_a_gap_inside_a_year_is_filled_or_refused_as_the_legacy_reader_refuses_it(tmp_path):
    write_observations(tmp_path, years=(2004,), gap_in=2004)
    path = tmp_path / "2004.nc"
    with pytest.raises(ValueError, match="first gap is 2004-01-04 to 2004-01-06") as old:
        legacy.nc_read_groundtruth_long(str(path), "precip", {})
    da = hcwf_obs(path, 2004)
    with pytest.raises(ValueError, match="first gap is 2004-01-04 to 2004-01-06"):
        ingest(da, day_boundary=BOUNDARY, units="mm", gaps="refuse")
    with pytest.warns(UserWarning, match="Filled 1 missing days"):
        filled = ingest(da, day_boundary=BOUNDARY, units="mm")
    assert filled.missing[..., 4].all() and filled.tenths.shape[-1] == 20
    assert old.value is not None
    # a gap between two calendar years is not inside one
    two = xr.concat([xr.DataArray(np.ones((2, 1, 1), "float32"), dims=("time", "lat", "lon"),
                                  coords={"time": pd.to_datetime(["2003-12-30", "2003-12-31"]), "lat": [1.0],
                                          "lon": [1.0]}),
                     xr.DataArray(np.ones((2, 1, 1), "float32"), dims=("time", "lat", "lon"),
                                  coords={"time": pd.to_datetime(["2004-01-02", "2004-01-03"]), "lat": [1.0],
                                          "lon": [1.0]})], dim="time")
    with pytest.warns(UserWarning, match="Filled 1"):
        assert ingest(two, day_boundary=BOUNDARY, units="mm", gaps="refuse").tenths.shape[-1] == 5


def test_ingest_options_follow_the_config_and_default_to_hcwfs(tmp_path):
    def options(config, forecast=False):
        """OnsetOptions.ingest of a flat config."""
        got = typed_io.inputs(config)
        return got.options.ingest(forecast, got.obs.units, check_units=got.select.check_units)

    assert options({}) == {} and options({}, forecast=True) == {}
    config = {"rain_resolution": "exact", "missing": "keep", "negatives": "keep", "gaps": "refuse", "check_units": False}
    assert options(config) == {"rain_resolution": "exact", "missing": "keep", "negatives": "keep", "gaps": "refuse"}
    assert options(config, forecast=True) == {"rain_resolution": "exact", "missing": "keep", "negatives": "keep",
                                              "check_units": False}


# ----------------------------------------------------------------------------------------------- admin units

def test_an_adm3_file_reads_as_the_legacy_table_reads_it(tmp_path):
    write_adm3(tmp_path / "adm3_2003.nc", names=("north", "south", "East ward"))
    old = legacy.nc_read_groundtruth_long(str(tmp_path / "adm3_2003.nc"), "precip", {})
    new = hcwf_obs(tmp_path / "adm3_2003.nc", 2003)
    assert new.dims == ("time", "unit") and new["unit"].values.tolist() == ["north", "south", "East ward"]
    np.testing.assert_array_equal(new.values.reshape(-1), old["precip"].to_numpy())
    assert old["id"].tolist()[:3] == ["north", "south", "East ward"]
    unit = onto_units.read_unit_dim(xr.open_dataset(tmp_path / "adm3_2003.nc").pipe(dim_fmt)["precip"],
                                  block_dim="time", ids=["East ward", "north"])
    np.testing.assert_array_equal(unit.values, new.sel(unit=["East ward", "north"]).values)
    series = ingest(unit, day_boundary=BOUNDARY, units="mm", rain_resolution="exact", negatives="keep", missing="keep")
    assert series.dims == ("unit", "time") and series.coords["unit"].tolist() == ["East ward", "north"]


# ---------------------------------------------------------------------------- cells onto units (a transform)

@pytest.mark.parametrize("policy", ["keep", "zero"])
@pytest.mark.parametrize("keyed", ["id", "coordinate"])
def test_observed_cells_onto_units_are_the_legacy_cell_transform(tmp_path, policy, keyed):
    write_observations(tmp_path, years=(2003,), units="mm")
    spec = legacy_spec(tmp_path, id_weights())
    old = legacy.nc_read_groundtruth_long(str(tmp_path / "2003.nc"), "precip", {}, missing_rain_policy=policy)
    old = legacy.transform_groundtruth_long(legacy_ids(old), filters.read_cell_transform(spec), "precip", spec=spec)
    expect = old.pivot(index="time", columns="id", values="precip")
    da = dim_fmt(xr.open_dataset(tmp_path / "2003.nc"))["precip"]
    units = units_of(grid_weights(), id_keyed=keyed == "id")
    got = onto_units.read_on_units(da, units, block_dim="time", missing=policy, convention=CONVENTION if keyed == "id" else None)
    assert got.dims == ("time", "unit") and got["unit"].values.tolist() == ["north", "south"]
    np.testing.assert_array_equal(got.values, expect[["north", "south"]].to_numpy())
    assert got.dtype == np.float64


def test_a_dropped_source_leaves_the_means_of_its_year(tmp_path):
    rng = np.random.default_rng(5)
    data = rng.gamma(1.5, 4.0, (20, 3, 2)).astype("float32")
    data[3:5, 0, 1] = np.nan  # the cell (10.0, 70.25) is incomplete
    xr.Dataset({"precip": (("time", "lat", "lon"), data, {"units": "mm"})},
               coords={"time": pd.date_range("2004-01-01", periods=20), "lat": LATS, "lon": LONS}) \
        .to_netcdf(tmp_path / "2004.nc")
    weights = id_weights()
    spec = legacy_spec(tmp_path, weights[weights["source_id"] != "10.00_70.25"])
    old = legacy.nc_read_groundtruth_long(str(tmp_path / "2004.nc"), "precip", {}, missing_rain_policy="drop")
    old = legacy.transform_groundtruth_long(legacy_ids(old), filters.read_cell_transform(spec), "precip", spec=spec)
    expect = old.pivot(index="time", columns="id", values="precip")
    da = dim_fmt(xr.open_dataset(tmp_path / "2004.nc"))["precip"]
    got, support, years = onto_units.read_on_units(da, units_of(grid_weights()), block_dim="time", missing="drop",
                                                 support=True)
    np.testing.assert_array_equal(got.values, expect[["north", "south"]].to_numpy())
    assert years.tolist() == [2004] and support.tolist() == [[True], [True]]
    # a unit all of whose sources are dropped has no series that year
    data[0, 0, 0] = np.nan
    only_north = EvaluationUnits.from_weights(grid_weights()[lambda t: t["target_id"] == "north"])
    cells = da.copy(deep=True)
    cells.values[0, 0, 0] = np.nan
    cells.values[0, 1, 0] = np.nan
    empty, support, _ = onto_units.unit_series(cells, only_north.transform(cells["lat"], cells["lon"]),
                                             missing="drop", support=True)
    assert np.isnan(empty.values).all() and support.tolist() == [[False]]


def test_the_weights_of_a_grid_swapped_to_another_longitude_convention_match_after_wrapping():
    swapped = {70.0: 280.0, 70.25: 280.25}  # the weights say 0..360, the grid was swapped to -180..180
    units = EvaluationUnits.from_weights(grid_weights().assign(lon=lambda t: t["lon"].map(swapped)))
    grid_lon = np.array([-80.0, -79.75])
    with pytest.raises(ValueError, match="outside the compiled axes"):
        units.transform(LATS, grid_lon)
    wrapped = units.transform(LATS, grid_lon, wrap_longitude=True)
    plain = EvaluationUnits.from_weights(grid_weights()).transform(LATS, LONS)
    assert (wrapped.weighted_matrix != plain.weighted_matrix).nnz == 0
    # a shifted longitude with float noise needs a tolerance
    noisy = EvaluationUnits.from_weights(grid_weights().assign(lon=lambda t: t["lon"].map({70.0: 280.1, 70.25: 280.35})))
    with pytest.raises(ValueError, match="outside the compiled axes"):
        noisy.transform(LATS, np.array([-79.9, -79.65]), wrap_longitude=True)
    assert noisy.transform(LATS, np.array([-79.9, -79.65]), wrap_longitude=True, tolerance=1e-6).shape == (2, 6)


def test_a_region_box_cuts_without_masks_and_the_weights_match_the_flipped_grid(tmp_path):
    write_observations(tmp_path, years=(2003,))
    ds = xr.open_dataset(tmp_path / "2003.nc")
    flipped = dim_fmt(ds.isel(lat=slice(None, None, -1)))["precip"]
    units = units_of(grid_weights())
    own = onto_units.read_on_units(flipped, units, block_dim="time")
    boxed = onto_units.read_on_units(flipped, units, block_dim="time", region="india")
    np.testing.assert_array_equal(boxed.values, own.values)
    assert boxed.dtype == np.float64


def test_a_region_unit_matches_its_own_grid_even_with_repeated_coordinates():
    keep = np.array([[True], [True], [False]])
    units = EvaluationUnits.from_region(keep, np.zeros(3), np.array([5.0]), weights="equal")
    values = np.array([[10.0], [30.0], [90.0]])
    assert units.transform(np.zeros(3), np.array([5.0])).observed_mean(values)[0, 0] == 20.0


# -------------------------------------------------------------------------------------------------- forecasts

def _write_inits(tmp_path, **kwargs):
    path = tmp_path / "f2014.nc"
    write_forecast(path, INITS, **kwargs)
    return path


def _hcwf_forecast(tmp_path, unit_cvt=None, **kwargs):
    return typed_io.forecast(2014, dict(unit_cvt=unit_cvt, model_dir=str(tmp_path), file_pattern="f{}.nc",
                                        **{**FORECAST_KWARGS, **kwargs}), True)


def test_a_forecast_reads_as_the_legacy_table_reads_it(tmp_path):
    path = _write_inits(tmp_path)
    old = legacy.nc_read_forecast_wide(str(path), "tp", {}, {"options": {}}, day_dim="day", prefix="rain")
    new = _hcwf_forecast(tmp_path).transpose("init_time", "member", "lat", "lon", "step")
    cols = [f"rain_day_{d}" for d in range(1, DAYS + 1)]
    np.testing.assert_array_equal(new.values.reshape(-1, DAYS), old[cols].to_numpy())
    assert new.dtype == np.float32 and new["step"].values.tolist() == list(range(1, DAYS + 1))


def test_a_declared_units_file_is_converted_by_the_caller_and_blends_scale_matches(tmp_path):
    path = _write_inits(tmp_path, units=None)
    old = legacy.nc_read_forecast_wide(str(path), "tp", {}, {"options": {}}, day_dim="day", prefix="rain")
    cols = [f"rain_day_{d}" for d in range(1, DAYS + 1)]
    scaled = _hcwf_forecast(tmp_path, unit_cvt=1000.0).transpose("init_time", "member", "lat", "lon", "step")
    np.testing.assert_array_equal(scaled.values.reshape(-1, DAYS), old[cols].to_numpy() * 1000.0)


def test_forecast_cells_onto_units_are_the_legacy_cell_transform_with_the_scale_applied_first(tmp_path):
    path = _write_inits(tmp_path, units=None)
    scale = 1000.0
    spec = legacy_spec(tmp_path, id_weights())
    old = legacy.nc_read_forecast_wide(str(path), "tp", {}, {"options": {}}, day_dim="day", prefix="rain")
    cols = [f"rain_day_{d}" for d in range(1, DAYS + 1)]
    old[cols] = old[cols].to_numpy() * scale
    old = legacy.transform_forecast_wide(legacy_ids(old), filters.read_cell_transform(spec), spec=spec)
    expect = np.stack([old[(old["id"] == unit)].sort_values(["time", "number"])[cols].to_numpy()
                       for unit in ("north", "south")], axis=1)  # (init * member, unit, day)
    ds = dim_fmt_model_ensemble(xr.open_dataset(path))
    da = ds["tp"].sel(init_time=pd.to_datetime(INITS))
    got = onto_units.read_on_units(da, units_of(None, id_keyed=True), block_dim="init_time", scale=scale,
                                 convention=CONVENTION)
    assert got.dims == ("init_time", "member", "step", "unit") or set(got.dims) == {"init_time", "member", "step", "unit"}
    got = got.transpose("init_time", "member", "unit", "step")
    np.testing.assert_array_equal(got.values.reshape(-1, 2, DAYS), expect)
    assert got.dtype == np.float64


def test_a_regrid_scales_after_the_averaging(tmp_path):
    path = _write_inits(tmp_path, units=None)
    scale = 0.1
    ds = dim_fmt_model_ensemble(xr.open_dataset(path))
    units = EvaluationUnits.from_weights(grid_weights())
    after = onto_units.read_on_units(ds["tp"], units, block_dim="init_time", post_scale=scale)
    first = onto_units.read_on_units(ds["tp"], units, block_dim="init_time", scale=scale)
    da = dim_fmt_model_ensemble(xr.open_dataset(path))["tp"].load()
    transform = units.transform(da["lat"].values, da["lon"].values)
    cells = da.transpose("lat", "lon", "init_time", "member", "step").values.reshape(len(LATS) * len(LONS), -1)
    expect_after = transform.observed_mean(cells.astype(float)).T * scale
    expect_first = transform.observed_mean((cells * np.float32(scale)).astype(float)).T
    order = ("init_time", "member", "step", "unit")
    np.testing.assert_array_equal(after.transpose(*order).values.reshape(-1, 2), expect_after)
    np.testing.assert_array_equal(first.transpose(*order).values.reshape(-1, 2), expect_first)
    assert not np.array_equal(expect_after, expect_first)  # the two orders differ in the last digits
    # an array already in memory is not scaled in place
    loaded = ds["tp"].load()
    before = loaded.values.copy()
    onto_units.read_on_units(loaded, units, block_dim="init_time", scale=scale)
    np.testing.assert_array_equal(loaded.values, before)


# ----------------------------------------------------------------------------------------------- sparse reads

@pytest.fixture
def chunked(tmp_path):
    lats, lons = np.arange(10.0, 10.0 + 8 * 0.25, 0.25), np.arange(70.0, 70.0 + 8 * 0.25, 0.25)
    path = tmp_path / "f2014.nc"
    write_forecast(path, INITS, members=2, lats=lats, lons=lons, chunks=(1, 2, DAYS, 4, 4))
    return path, lats, lons


def test_load_blocks_skips_the_tiles_without_a_kept_cell(chunked, monkeypatch):
    path, lats, lons = chunked
    ds = dim_fmt_model_ensemble(xr.open_dataset(path))
    da = ds["tp"].transpose("init_time", "member", "step", "lat", "lon")
    assert da.encoding["chunksizes"] is not None or da.encoding.get("chunks")
    full = load_blocks(da, "init_time").values
    keep = np.zeros((8, 8), dtype=bool)
    keep[1, 6] = keep[2, 5] = True  # both in the tile (rows 0-3, columns 4-7)
    seen = []
    from xarray.backends import netCDF4_

    original = netCDF4_.NetCDF4ArrayWrapper._getitem

    def spy(self, key):
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(netCDF4_.NetCDF4ArrayWrapper, "_getitem", spy)
    got = load_blocks(da, "init_time", keep=keep).values
    monkeypatch.undo()
    tiles = {(k[3].start, k[4].start) for k in seen}
    assert tiles == {(0, 4)}, tiles
    np.testing.assert_array_equal(got[..., 1, 6], full[..., 1, 6])
    np.testing.assert_array_equal(got[..., 2, 5], full[..., 2, 5])
    assert np.isnan(got[..., ~keep]).all() and got.shape == full.shape
    two = np.zeros((8, 8), dtype=bool)
    two[0, 0] = two[7, 7] = True
    assert np.isfinite(load_blocks(da, "init_time", keep=two).values[..., 0, 0]).any()


def test_units_read_only_the_tiles_that_feed_them(chunked, monkeypatch):
    path, lats, lons = chunked
    weights = pd.DataFrame({"target_id": ["u", "u"], "lat": [lats[1], lats[2]], "lon": [lons[6], lons[5]],
                            "weight": [1.0, 3.0]})
    units = EvaluationUnits.from_weights(weights)
    da = dim_fmt_model_ensemble(xr.open_dataset(path))["tp"].transpose("init_time", "member", "step", "lat", "lon")
    from xarray.backends import netCDF4_

    seen = []
    original = netCDF4_.NetCDF4ArrayWrapper._getitem
    monkeypatch.setattr(netCDF4_.NetCDF4ArrayWrapper, "_getitem",
                        lambda self, key: seen.append(key) or original(self, key))
    got = onto_units.read_on_units(da, units, block_dim="init_time")
    monkeypatch.undo()
    assert {(k[3].start, k[4].start) for k in seen} == {(0, 4)}
    full = da.load()
    expect = (full.values[..., 1, 6] * 1.0 + full.values[..., 2, 5] * 3.0) / 4.0
    np.testing.assert_allclose(got.values[..., 0], expect, rtol=1e-6)


def test_load_blocks_without_keep_is_what_it_was(chunked):
    path, _, _ = chunked
    with pytest.raises(ValueError, match="keep has shape"):
        load_blocks(dim_fmt_model_ensemble(xr.open_dataset(path))["tp"], "init_time", keep=np.ones((2, 2), dtype=bool))
    da = dim_fmt_model_ensemble(xr.open_dataset(path))["tp"]
    np.testing.assert_array_equal(load_blocks(da, "init_time").values, da.load().values)
