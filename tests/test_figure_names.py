"""File names come from run_folder.file_name; the figure and writer functions take typed specs, not the flat config."""
import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from hcwf.config.model import FigureSpec
from hcwf.runs.folder import RunPaths, file_name

FIGURES = {"heatmap": (("M", "1-10"), "skill_scores_heatmap_M_1-10.png"),
           "reliability": (("M", "1-10"), "reliability_M_1-10.png"),
           "spatial_map": (("M_1-10",), "spatial_metrics_M_1-10.png"),
           "climatology_map": (("2006-2020",), "climatology_onset_2006-2020.png"),
           "field_map": (("obs", "onset", "1-10"), "map_obs_onset_1-10.png"),
           "panel_error": (("M", 15), "panel_portrait_mae_far_mr_M_15day.png"),
           "panel_skill": (("1-10",), "panel_portrait_BSS_AUC_1-10.png"),
           "panel_bar": (("1-10",), "panel_bar_BSS_RPSS_AUC_1-10.png")}
DATA = {"overall": (("M", "1-10"), "overall_skill_scores_M_1-10.csv", "overall_M_1-10.csv"),
        "binned": (("M", "1-10"), "binned_skill_scores_M_1-10.csv", "binned_M_1-10.csv"),
        "spatial": (("M_1-10",), "spatial_metrics_M_1-10.nc", "spatial_M_1-10.nc"),
        "climatology": (("2006-2020",), "climatology_onset_doy_2006-2020.nc", "climatology_onset.nc"),
        "onset": (("ERA5",), "onsets_ERA5.nc", "ERA5.nc")}


def _paths(short):
    root = Path("x")
    return RunPaths(False, root, "n", "f", root, root, root, None, None, short)


def test_figure_names_do_not_depend_on_the_layout():
    for kind, (parts, name) in FIGURES.items():
        assert file_name(None, kind, *parts) == file_name(_paths(True), kind, *parts) == \
            file_name(_paths(False), kind, *parts) == name


def test_data_names_are_short_in_run_folders_and_historical_otherwise():
    for kind, (parts, long, short) in DATA.items():
        assert file_name(None, kind, *parts) == file_name(_paths(False), kind, *parts) == long
        assert file_name(_paths(True), kind, *parts) == short
    with pytest.raises(ValueError):
        file_name(None, "nope")


def test_figure_and_writer_functions_take_specs_not_kwargs():
    from hcwf.figures import func_map, heatmap, maps, onset_map, panel_bar_skill, panel_portrait_error, \
        panel_portrait_skill, reliability
    from hcwf.io import output
    from hcwf.figures import visual

    for func in (func_map.spatial_metrics_map, heatmap.create_heatmap, maps.plot_spatial_metrics,
                 onset_map.plot_spatial_climatology_onset, panel_bar_skill.panel_bar_bss_rpss_auc,
                 panel_portrait_error.panel_portrait_mae_far_mr, panel_portrait_skill.panel_portrait_bss_auc,
                 reliability.plot_reliability_diagram, output.save_score_results, output.save_metrics_to_netcdf,
                 output.save_climatology_onset, visual.set_basemap):
        assert not any(p.kind is p.VAR_KEYWORD for p in inspect.signature(func).parameters.values()), func.__name__


def test_the_figure_spec_carries_what_the_figures_read(tmp_path):
    names = {f.name for f in fields(FigureSpec)}
    assert {"dir_fig", "region", "shpfile_dir", "polygon", "save_fig", "show_plot", "show_panel", "models",
            "max_forecast_day", "units"} <= names
    from typed_io import figure_spec

    spec = figure_spec({"dir_fig": str(tmp_path), "region": "x", "model_list": ["A", "B"], "max_forecast_day": 7})
    assert spec.models == ("A", "B") and spec.max_forecast_day == 7 and spec.units is None and spec.show_plot
