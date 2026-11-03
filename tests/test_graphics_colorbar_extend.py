"""The colour bars of the maps draw an arrow at an end where values lie beyond the colour limits."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from hcwf.graphics.field import colorbar_extend, draw_field  # noqa: E402


def _field(values):
    values = np.asarray(values, dtype=float).reshape(2, 2)
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]})


LEVELS = np.arange(135, 245, 3)


@pytest.mark.parametrize("values,expected", [([140, 150, 160, 170], "max"), ([100, 150, 160, 170], "both"),
                                              ([140, 150, 300, np.nan], "max")])
def test_a_boundary_norm_that_extends_up_gets_an_arrow_down_where_values_fall_below(values, expected):
    """The climatology onset scale (extend max): early onsets below the first level are drawn in the lowest colour."""
    norm = mcolors.BoundaryNorm(LEVELS, plt.cm.Spectral.N, extend="max")
    fig, ax = plt.subplots()
    im = draw_field(ax, _field(values), cmap=plt.cm.Spectral, norm=norm)
    assert colorbar_extend(im) == expected
    bar = fig.colorbar(im, ax=ax, extend=colorbar_extend(im))
    assert bar.extend == expected
    plt.close(fig)


@pytest.mark.parametrize("values,expected", [([0, 5, 10, 15], "neither"), ([-1, 5, 10, 15], "min"),
                                              ([0, 5, 10, 20], "max"), ([-1, 5, 10, 20], "both")])
def test_a_plain_scale_gets_an_arrow_at_each_end_that_values_pass(values, expected):
    fig, ax = plt.subplots()
    im = draw_field(ax, _field(values), cmap="OrRd", vmin=0, vmax=15, shading="flat")
    assert colorbar_extend(im) == expected
    plt.close(fig)
