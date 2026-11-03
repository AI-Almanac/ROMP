"""ROMP's benchmark on political boundaries: every product of a run over polygon units (a shapefile in a temp folder),
for deterministic and ensemble models, the climatology references, the onset export and the maps."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

geopandas = pytest.importorskip("geopandas")
from shapely.geometry import MultiPolygon, box  # noqa: E402

import momp  # noqa: E402
from momp import export  # noqa: E402
from momp.lib.config_yaml import build_config  # noqa: E402

LAT, LON = [6.0, 7.0, 8.0, 9.0], [38.0, 39.0, 40.0, 41.0]
STEPS, MAX_LEAD, MEMBERS = 14, 10, 3
BOUNDARY = "00 UTC, labelled by start"
IDS = ["U1", "U2", "U3", "U4"]


def _obs_rain(year):
    """Daily rain (mm) of May-August, (time, lat, lon): dry until a per-cell onset day, then wet."""
    rng = np.random.default_rng(year)
    time = pd.date_range(f"{year}-05-01", f"{year}-08-31")
    onset = rng.integers(8, 40, size=(1, len(LAT), len(LON)))
    day = np.arange(len(time))[:, None, None]
    return time, rng.gamma(0.6, 6.0, size=(len(time), len(LAT), len(LON))) * np.where(day < onset, 0.1, 2.0)


def _write_data(base):
    """Observations of 2006-2020, a deterministic and an ensemble forecast of 2019-2020 and four polygons."""
    (base / "obs").mkdir(parents=True)
    for year in range(2006, 2021):
        time, rain = _obs_rain(year)
        xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm"})},
                   coords={"time": time, "lat": LAT, "lon": LON}).to_netcdf(base / "obs" / f"{year}.nc")
    for name, members in (("DET", 0), ("ENS", MEMBERS)):
        (base / name).mkdir()
        for year in (2019, 2020):
            _, rain = _obs_rain(year)
            inits = pd.date_range(f"{year}-05-01", f"{year}-05-31")
            rng = np.random.default_rng(year + 7)
            data = np.stack([rain[i:i + STEPS] for i in range(len(inits))])
            dims, coords = ("time", "step", "lat", "lon"), {"time": inits, "step": np.arange(1, STEPS + 1),
                                                           "lat": LAT, "lon": LON}
            if members:
                data = np.stack([data * rng.uniform(0.5, 1.6, size=data.shape) for _ in range(members)], axis=1)
                dims, coords = ("time", "number", "step", "lat", "lon"), {**coords, "number": np.arange(members)}
            else:
                data = data * rng.uniform(0.5, 1.6, size=data.shape)
            xr.Dataset({"tp": (dims, data, {"units": "mm"})}, coords=coords).to_netcdf(base / name / f"{year}.nc")
    polygons = [box(37.5, 5.5, 39.5, 7.5), box(39.5, 5.5, 41.5, 7.5), box(37.5, 7.5, 39.5, 9.5),
                MultiPolygon([box(39.5, 7.5, 40.5, 9.5), box(40.5, 8.5, 41.5, 9.5)])]
    geopandas.GeoDataFrame({"pcode": IDS, "name": ["Alpha", "Beta", "Gamma", "Delta"]}, geometry=polygons,
                           crs="EPSG:4326").to_file(base / "units.shp")


def _raw(base, model, ensemble, **output):
    """A nested run config on the polygons, with the files and figures of every product switched on."""
    return {"region": "Ethiopia", "parallel": False,
            "observations": {"path": str(base / "obs"), "units": "mm", "day_boundary": BOUNDARY},
            "models": [{"name": model, "path": str(base / model), "units": "mm", "lead_label": "start"}],
            "onset": {"dry_spell_check": False, "check_alignment": False},
            "units": {"shapefile": str(base / "units.shp"), "unit_id_col": "pcode", "unit_name_col": "name"},
            "evaluation": {"start_date": [2019, 5, 1], "end_date": [2020, 5, 31], "init_days": [0, 1, 2, 3, 4, 5, 6],
                           "start_year_clim": 2006, "end_year_clim": 2020, "verification_windows": [[1, MAX_LEAD]],
                           "tolerances": [3], "max_forecast_day": MAX_LEAD, "day_bins": [[1, 5], [6, 10]],
                           "probabilistic": ensemble},
            "output": {"data_dir": str(base / f"out_{model}"), "figure_dir": str(base / f"fig_{model}"),
                       "save_nc_spatial_far_mr_mae": True, "save_csv_score": True, "save_nc_climatology": True,
                       "plot_spatial_far_mr_mae": True, "plot_climatology_onset": True,
                       "plot_heatmap_bss_auc": ensemble, "plot_reliability": ensemble,
                       "plot_panel_heatmap_error": True, "show_plot": False, "show_panel": False, **output}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The data and one written run of each kind over the polygons."""
    base = tmp_path_factory.mktemp("units_benchmark")
    _write_data(base)
    runs = {name: momp.run(_raw(base, name, name == "ENS"), write=True, quiet=True) for name in ("DET", "ENS")}
    return base, runs


def test_deterministic_scores_are_per_unit(world):
    _, runs = world
    spatial = runs["DET"].scores["spatial"]
    assert set(spatial) == {("DET", "1-10"), ("climatology", "1-10")}
    for ds in spatial.values():
        assert ds["mean_mae"].dims == ("unit",) and list(ds["unit"].values) == IDS
        assert {"false_alarm_rate", "miss_rate", "mae_2019", "mae_2020"} <= set(ds)
        assert np.isfinite(ds["mean_mae"].values).all()
    assert runs["DET"].observed["onset_date"].dims == ("year", "unit")
    assert list(runs["DET"].observed["unit_name"].values) == ["Alpha", "Beta", "Gamma", "Delta"]


def test_spatial_metric_files_are_keyed_by_unit_id(world):
    base, runs = world
    out = Path(runs["DET"].folder)
    table = pd.read_csv(out / "spatial_metrics_DET_1-10.csv")
    assert list(table["unit_id"]) == IDS and list(table["name"]) == ["Alpha", "Beta", "Gamma", "Delta"]
    assert {"lat", "lon", "mean_mae", "false_alarm_rate", "miss_rate"} <= set(table)
    ds = xr.open_dataset(out / "spatial_metrics_DET_1-10.nc")
    assert ds["mean_mae"].dims == ("unit",) and ds["lon"].dims == ("unit",)
    np.testing.assert_allclose(ds["mean_mae"].values, table["mean_mae"].values)
    frame = geopandas.read_file(out / "spatial_metrics_DET_1-10.geojson")
    assert list(frame["unit_id"]) == IDS and frame.geometry.geom_type.tolist() == ["Polygon"] * 3 + ["MultiPolygon"]
    np.testing.assert_allclose(frame["mean_mae"].values, table["mean_mae"].values)
    clim = xr.open_dataset(out / "climatology_onset_doy_2006-2020.nc")
    assert "unit" in clim.dims and clim["unit_name"].size == 4


def test_deterministic_figures_are_drawn(world):
    _, runs = world
    names = {p.name for p in Path(runs["DET"].config["dir_fig"]).glob("*.png")}
    assert {"spatial_metrics_DET_1-10.png", "spatial_metrics_climatology_1-10.png", "climatology_onset_2006-2020.png"} <= names


def test_ensemble_scores_and_figures_pool_the_units(world):
    _, runs = world
    overall = runs["ENS"].scores["overall"]
    assert len(overall) == 1 and np.isfinite(overall[["Fair_Brier_Score", "Fair_RPS", "AUC"]].to_numpy()).all()
    assert len(runs["ENS"].scores["binned"]) == 2
    figs = {p.name for p in Path(runs["ENS"].config["dir_fig"]).glob("*.png")}
    assert any(n.startswith("reliability") for n in figs) and any(n.startswith("skill_scores_heatmap") for n in figs)


def test_pair_tables_are_per_unit_cases():
    """Member onsets of units become (init, unit) cases with bins, and score like grid cases."""
    from momp.stats.bins import create_forecast_observation_pairs_with_bins
    from momp.stats.score import calculate_brier_score, calculate_rps

    inits = pd.to_datetime(["2020-05-01", "2020-05-08"])
    rows = [(i, u, m, d) for i in inits for u, d in (("a", 3.0), ("b", np.nan)) for m in range(3)]
    members = pd.DataFrame(rows, columns=["init_time", "unit", "member", "onset_day"])
    obs = xr.DataArray(np.array(["2020-05-04", "2020-05-20"], dtype="datetime64[ns]"), dims="unit",
                       coords={"unit": ["a", "b"]})
    pairs = create_forecast_observation_pairs_with_bins(members, obs, day_bins=[(1, 5), (6, 10)], max_forecast_day=10)
    assert "unit" in pairs and "lat" not in pairs
    assert pairs.groupby(["init_time", "unit"]).ngroups == 3   # unit a is scored on the first init only
    assert np.isfinite(calculate_rps(pairs)["rps"]) and np.isfinite(calculate_brier_score(pairs)["brier_score"])


def test_onset_export_per_unit(world):
    base, _ = world
    cfg = export._complete(build_config(_raw(base, "DET", False)))
    paths = export.export_onsets(cfg, sources=["observed", "DET", "climatology", "conditional_climatology"],
                                 out=base / "onsets_det")
    observed = export.open_onsets(paths["observed"])
    assert observed["onset_date"].dims == ("year", "season", "unit") and observed["unit"].size == 4
    assert list(observed["unit_name"].values)[0] == "Alpha"
    model = export.open_onsets(paths["DET"])
    assert model["onset_date"].dims[-1] == "unit" and model["scored"].dims == ("init_time", "unit")
    climatology = export.open_onsets(paths["climatology"])
    assert climatology["climatological_onset_doy"].dims == ("season", "unit")
    conditional = export.open_onsets(paths["conditional_climatology"])
    assert conditional["onset_probability"].dims == ("init_time", "season", "lead", "unit")
    ens = export.export_onsets(export._complete(build_config(_raw(base, "ENS", True))), sources=["ENS"],
                               out=base / "onsets_ens")
    members = export.open_onsets(ens["ENS"])
    assert members["ensemble_onset_date"].dims[-1] == "unit" and members["members_by_status"].dims[-1] == "unit"


def test_a_single_field_draws_cells_or_polygons():
    import matplotlib.pyplot as plt

    from momp.graphics.field import draw_field, extent
    from momp.spatial import EvaluationUnits

    table = pd.DataFrame({"id": ["a", "b"], "lat": [0.5, 0.5], "lon": [0.5, 1.5]})
    units = EvaluationUnits(table, "polygon", geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    field = xr.DataArray([1.0, np.nan], dims="unit", coords={"unit": ["a", "b"]})
    fig, ax = plt.subplots()
    assert draw_field(ax, field, units, cmap="Reds", vmin=0, vmax=2).get_array().tolist() == [1.0]
    assert [a.tolist() for a in extent(field, units)] == [[0.0, 1.0], [0.0, 2.0]]
    grid = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]})
    draw_field(ax, grid, shading="flat", cmap="Reds", vmin=0, vmax=2)
    plt.close(fig)


def test_figures_need_no_data_files_a_run_with_every_save_switch_off_writes_none(world):
    """Plots take the scores from memory: with the save_* switches off the run still draws its figures and writes no
    table or NetCDF (only run.yaml, the log and the cross-run onset cache)."""
    base, _ = world
    off = {"save_nc_spatial_far_mr_mae": False, "save_csv_score": False, "save_nc_climatology": False,
           "save_onsets": False}
    for model, ensemble in (("DET", False), ("ENS", True)):
        plots = {"plot_panel_heatmap_skill": ensemble, "plot_bar_bss_rpss_auc": ensemble}
        raw = _raw(base, model, ensemble, **off, **plots)
        raw["output"]["data_dir"] = str(base / f"quiet_out_{model}")
        raw["output"]["figure_dir"] = str(base / f"quiet_fig_{model}")
        run = momp.run(raw, write=True, quiet=True)
        folder = Path(run.folder)
        data = [p.name for p in folder.rglob("*") if p.is_file() and not any(part.startswith(".") for part in
                p.relative_to(folder).parts) and p.suffix in {".csv", ".nc", ".geojson", ".pkl", ".zarr", ".parquet"}]
        assert data == []
        figures = {p.name for p in Path(run.config["dir_fig"]).glob("*.png")}
        if ensemble:
            assert {"panel_portrait_BSS_AUC_1-10.png", "panel_bar_BSS_RPSS_AUC_1-10.png"} <= figures
        else:
            assert {"spatial_metrics_DET_1-10.png", "climatology_onset_2006-2020.png"} <= figures
