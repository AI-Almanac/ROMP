"""Edge cases of the figure helpers: missing outputs, missing metrics, unsupported portrait styles."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from hcwf.figures.panel_portrait_error import nested_dict_to_array
from hcwf.figures.visual import add_polygon, annotate_heatmap, heatmap, portrait_plot


def test_nested_dict_to_array_gives_nan_for_missing_entries():
    leaf = xr.DataArray([1.0, 3.0])
    results = {"a": {"w1": {"m": leaf}, "w2": {}}, "b": {"w1": {"other": leaf}, "w2": {"m": leaf}}}
    arr, rows, cols = nested_dict_to_array(results, "m")
    assert rows == ["a", "b"] and cols == ["w1", "w2"]
    assert arr[0, 0] == 2.0 and arr[1, 1] == 2.0
    assert np.isnan(arr[0, 1]) and np.isnan(arr[1, 0])


def test_add_polygon_returns_empty_polygon_when_not_drawn():
    fig, ax = plt.subplots()
    out = add_polygon(ax, xr.DataArray(np.zeros((2, 2))), False, return_polygon=True)
    assert out[0] is ax and len(out[1]) == 0 and len(out[2]) == 0 and out[3] is False
    plt.close(fig)


def test_portrait_plot_rejects_stacked_arrays_and_bad_shapes():
    stacked = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="stacked"):
        portrait_plot(stacked, list("abcd"), list("xyz"))
    with pytest.raises(ValueError):
        portrait_plot(np.zeros((3, 4)), list("abc"), list("xyz"))
    plt.close("all")


def test_portrait_plot_passes_the_text_colour_threshold_on():
    data = np.array([[0.0, 5.0], [10.0, 20.0]])
    fig, ax, im = portrait_plot(data, ["a", "b"], ["x", "y"], annotate=True, colorbar_off=True,
                                annotate_textcolors=("red", "blue"), annotate_textcolors_threshold=(4.0, 12.0))
    colours = {t.get_text(): t.get_color() for t in ax.texts}
    assert colours == {"0.00": "blue", "5.00": "red", "10.00": "red", "20.00": "blue"}
    plt.close(fig)


def test_annotate_heatmap_needs_no_fontsize():
    fig, ax = plt.subplots()
    data = np.arange(4.0).reshape(2, 2)
    ax, im = heatmap(data, ["a", "b"], ["x", "y"], ax=ax, cmap="viridis")
    annotate_heatmap(im, ax, data=data, cmap="viridis")
    assert len(ax.texts) == 4 and ax.texts[0].get_fontsize() == pytest.approx(plt.rcParams["font.size"])
    plt.close(fig)


def test_climatology_map_without_a_figure_folder_returns_no_path():
    from types import SimpleNamespace

    from hcwf.figures.onset_map import plot_spatial_climatology_onset

    lat, lon = np.arange(8.0, 12.0), np.arange(75.0, 79.0)
    onset = xr.DataArray(np.full((4, 4), 160.0), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    spec = SimpleNamespace(polygon=False, dir_fig=None, region="India", shpfile_dir=None, show_plot=False,
                           units=None)
    fig, ax, path = plot_spatial_climatology_onset({"clim": onset}, spec, (2006, 2020))
    assert path is None
    plt.close(fig)
