"""Reading, figures, init, conversion and detection for a run whose area is a box or a mask (no region).

The synthetic global file has 1 degree cells in 20 x 20 degree storage tiles, so the chunks a read decodes show which part
of the file it read.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

from _decode_count import count_decodes  # noqa: E402
from hcwf.cache import reset_run_state  # noqa: E402
from hcwf.config.init import write_init  # noqa: E402
from hcwf.config.model import NO_SELECTION, AreaSpec  # noqa: E402
from hcwf.spatial.masks import MaskSpec  # noqa: E402
from hcwf.spatial.select import region_select  # noqa: E402

BOX = {"lat_min": 3.0, "lat_max": 15.0, "lon_min": 33.0, "lon_max": 48.0}


@pytest.fixture(autouse=True)
def fresh():
    """Start every test without the state of a previous run."""
    reset_run_state()
    yield
    reset_run_state()


def _tiled_global(path):
    """Write a global 1 degree rain file (0..360, descending latitude) in 20 x 20 degree tiles, one time step per chunk."""
    lat, lon = np.arange(89.5, -90.0, -1.0), np.arange(0.5, 360.0, 1.0)
    rain = np.random.default_rng(3).gamma(0.5, 4.0, (4, lat.size, lon.size)).astype("float32")
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), rain)},
                    coords={"time": pd.date_range("2020-01-01", periods=4), "lat": lat, "lon": lon})
    ds.to_netcdf(path, encoding={"rain": {"chunksizes": (1, 20, 20), "zlib": True}})
    return path


def test_a_mask_without_a_box_reads_only_the_chunks_of_the_masks_bounds(tmp_path):
    """Region null with a country mask: the area is the country's bounds, so a global file is read only there."""
    path = _tiled_global(tmp_path / "global.nc")
    area = AreaSpec(region=None, mask=MaskSpec.from_mapping({"country": "Ethiopia"}), nc_mask=None, shpfile_dir=None,
                    polygon=False)
    assert area.bounds().origin == "mask bounds"
    with xr.open_dataset(path) as ds, count_decodes() as decodes:
        out = region_select(ds[["rain"]], area=area, select=NO_SELECTION, read_by="time", dataset_name="obs",
                            dir_fig=None)
    seen = decodes.only("rain")
    # Ethiopia lies within 3 to 15 N and 33 to 48 E, padded by a cell: 2 tile rows by 2 tile columns, 4 time steps
    assert 0 < seen.chunks <= 4 * 2 * 2 and seen.worst() == 1
    assert out.sizes["lat"] < 20 and out.sizes["lon"] < 25
    inside = out["rain"].isel(time=0).sel(lat=9.5, lon=40.5)
    outside = out["rain"].isel(time=0).sel(lat=14.5, lon=34.5)   # inside the bounds, in Sudan
    assert np.isfinite(float(inside)) and np.isnan(float(outside))


def test_a_box_without_a_region_applies_no_implicit_masks(tmp_path):
    """A box without a region keeps ocean and neighbouring land: only a mask section masks."""
    path = _tiled_global(tmp_path / "global.nc")
    area = AreaSpec(region=None, mask=None, nc_mask=None, shpfile_dir=None, polygon=False, box=BOX)
    with xr.open_dataset(path) as ds, count_decodes() as decodes:
        out = region_select(ds[["rain"]], area=area, select=NO_SELECTION, read_by="time", dataset_name="obs", dir_fig=None)
    assert out.sizes["lat"] == 12 and out.sizes["lon"] == 15 and bool(out["rain"].notnull().all())
    assert decodes.only("rain").chunks <= 4 * 2 * 2


# --- figures without a region ---------------------------------------------------------------------------------

def _spec(area):
    """Return the figure spec of a run without a region."""
    from hcwf.config.model import FigureSpec

    return FigureSpec(dir_fig=None, region=None, shpfile_dir=None, polygon=False, save_fig=False, show_plot=False,
                      show_panel=False, date_filter_year=None, short_names=False, models=(), max_forecast_day=10,
                      units=None, area=area)


def _field():
    """Return a small onset field over Ethiopia's neighbourhood."""
    lat, lon = np.arange(3.0, 15.0, 1.0), np.arange(33.0, 48.0, 1.0)
    return xr.DataArray(np.full((lat.size, lon.size), 160.0), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


@pytest.mark.parametrize("area", [
    AreaSpec(region=None, mask=MaskSpec.from_mapping({"country": "Ethiopia"}), nc_mask=None, shpfile_dir=None, polygon=False),
    AreaSpec(region=None, mask=None, nc_mask=None, shpfile_dir=None, polygon=False, box=BOX),
    AreaSpec(region=None, mask=None, nc_mask=None, shpfile_dir=None, polygon=False, box="all")],
    ids=["mask", "box", "all"])
def test_maps_without_a_region_draw_the_areas_outline_and_do_not_raise(area, caplog):
    """The climatology map of a run without a region takes the outline and extent from its area."""
    import cartopy.crs as ccrs

    from hcwf.figures.onset_map import plot_spatial_climatology_onset
    from hcwf.figures.visual import set_basemap

    caplog.set_level("WARNING")
    fig, ax, path = plot_spatial_climatology_onset({"clim": _field()}, _spec(area), (2006, 2020), domain_mask=True)
    assert path is None
    plt.close(fig)
    assert not [r for r in caplog.records if "shapereader" in r.getMessage() or "unknown region" in r.getMessage()]
    fig = plt.figure()
    axes = fig.add_subplot(111, projection=ccrs.PlateCarree())
    set_basemap(axes, _spec(area), _field())
    west, east, south, north = axes.get_extent(ccrs.PlateCarree())
    if area.box == "all":   # the whole domain of the files: the extent of the field
        assert (south, north) == pytest.approx((2.5, 14.5)) and (west, east) == pytest.approx((32.5, 47.5))
    elif area.box is not None:
        assert (south, north, west, east) == pytest.approx((3, 15, 33, 48))
    else:   # the mask's bounds: Ethiopia
        assert 3 < south < 4 and 14 < north < 15.1 and 32 < west < 34 and 47 < east < 49
    plt.close(fig)


def test_the_spatial_metric_maps_of_a_run_without_a_region_do_not_raise():
    """The three-panel metric figure draws the mask's shapes."""
    from hcwf.figures.maps import plot_spatial_metrics

    field = _field()
    metrics = xr.Dataset({"mean_mae": field * 0 + 3.0, "false_alarm_rate": field * 0 + 0.2, "miss_rate": field * 0 + 0.1})
    area = AreaSpec(region=None, mask=MaskSpec.from_mapping({"country": "Ethiopia"}), nc_mask=None, shpfile_dir=None,
                    polygon=False)
    try:
        fig, axes = plot_spatial_metrics(metrics, None, _spec(area))
    except TypeError:   # the metrics carry a year dimension in a run
        metrics = metrics.expand_dims(year=[2019])
        fig, axes = plot_spatial_metrics(metrics, None, _spec(area))
    plt.close(fig)


def test_shp_mask_and_get_shp_take_the_mask_without_a_region():
    """`shp_mask` applies the mask and `get_shp` returns its shape when there is no region."""
    from hcwf.spatial.masks import get_shp, shp_mask

    mask = MaskSpec.from_mapping({"country": "Ethiopia"})
    shape = get_shp(None, mask=mask)
    assert shape is not None and shape.geom_type in ("Polygon", "MultiPolygon")
    assert get_shp(None) is None
    field = _field()
    masked = shp_mask(field, None, mask)
    assert masked.notnull().sum() < field.notnull().sum() and masked.notnull().sum() > 0
    assert shp_mask(field, None) is field


# --- init -----------------------------------------------------------------------------------------------------

def _obs_folder(folder, lat, lon):
    """Write one year of observations on the given axes."""
    folder.mkdir(parents=True)
    rain = np.random.default_rng(1).gamma(0.5, 6.0, (6, len(lat), len(lon))).astype("float32")
    xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm/day"})},
               coords={"time": pd.date_range("2020-05-01", periods=6), "lat": lat, "lon": lon}
               ).to_netcdf(folder / "2020.nc")
    return folder


def test_init_on_files_no_region_matches_writes_a_box_and_todo_lines(tmp_path):
    """A small extent that fits no registered region: region null, the files' extent as the box, TODOs for the rest."""
    obs = _obs_folder(tmp_path / "obs", np.arange(-34.0, -30.0, 1.0), np.arange(18.0, 24.0, 1.0))
    target = write_init(tmp_path / "run.yaml", obs=str(obs))
    text = target.read_text()
    config = yaml.safe_load(text)
    assert config["region"] is None
    assert config["box"] == {"lat_min": -34.0, "lat_max": -31.0, "lon_min": 18.0, "lon_max": 23.0}
    assert "TODO: no region, so no default definition" in text and "season" in text and "start_year_clim" in text
    assert "The region supplies" not in text


def test_init_on_global_files_writes_a_box_placeholder_not_the_global_extent(tmp_path):
    """Global files: the box is a placeholder to fill in."""
    obs = _obs_folder(tmp_path / "obs", np.arange(-60.0, 61.0, 20.0), np.arange(-180.0, 180.0, 1.0))
    text = write_init(tmp_path / "run.yaml", obs=str(obs)).read_text()
    config = yaml.safe_load(text)
    assert config["region"] is None and config["box"]["lat_min"] == "<south>"
    assert "TODO: the area to read; the files are global" in text and "box: all" in text


def test_init_on_files_a_region_matches_keeps_the_region(tmp_path):
    """Files over Ethiopia's domain still get the detected region and no box."""
    obs = _obs_folder(tmp_path / "obs", np.arange(3.0, 15.5, 1.0), np.arange(33.0, 48.5, 1.0))
    config = yaml.safe_load(write_init(tmp_path / "run.yaml", obs=str(obs)).read_text())
    assert str(config["region"]).lower().startswith("ethiopia") and "box" not in config
