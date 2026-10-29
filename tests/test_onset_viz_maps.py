"""momp.onset.viz.maps: the drill-down figure builds in both modes and marks the explained cell."""
import matplotlib
import pytest

matplotlib.use("Agg")

from momp.onset import definition as dfn
from momp.onset import engine, explain
from momp.onset.viz.maps import drill_down
from test_onset_anatomy import forecast_anatomy, seasonal_rain


def setup():
    """A detect_observed result over the grid and the anatomy of one cell in 2001."""
    rain = seasonal_rain()
    defn = dfn.load("icpac")
    result = engine.detect_observed(defn, rain, years=[2001])
    return result, explain.trace_observed(defn, rain, lat=rain.coords["lat"][1], lon=rain.coords["lon"][0], year=2001)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_builds_in_both_modes(mode):
    """The map, colourbar and anatomy panels are in one figure and it renders."""
    result, anatomy = setup()
    fig = drill_down(result, anatomy, mode=mode)
    fig.canvas.draw()
    assert len(fig.axes) >= 4
    assert fig.axes[0].get_xlabel().startswith("longitude")


def test_marks_the_cell():
    """The map has a marker at the anatomy's cell."""
    result, anatomy = setup()
    ax = drill_down(result, anatomy).axes[0]
    marks = {(l.get_xdata()[0], l.get_ydata()[0]) for l in ax.lines}
    assert (anatomy.location["lon"], anatomy.location["lat"]) in marks


def test_forecast_anatomy_is_refused():
    """Only observed anatomies can be drilled into."""
    result, _ = setup()
    with pytest.raises(ValueError, match="observed"):
        drill_down(result, forecast_anatomy())
