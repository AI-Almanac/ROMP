"""momp.onset.viz.forecast: forecast_vs_truth builds in both modes and plots exactly the engine's onset dates."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from momp.onset import engine
from momp.onset import model as m
from momp.onset.viz.forecast import forecast_vs_truth
from test_onset_engine import JJAS, TWO_WET, definition
from test_onset_forecast_layer import case

INITS = ["2000-06-01", "2000-06-05", "2000-06-09", "2000-06-13", "2000-06-17"]


def synthetic(seed=2):
    """A one-cell history (2000, wet from late May) and a 4-member forecast for five June inits."""
    rng = np.random.default_rng(seed)
    hist = np.where(rng.random((1, 1, 366)) < 0.12, 120, 0)
    fc = np.where(rng.random((len(INITS), 4, 1, 1, 25)) < 0.45, 80, 0)
    return case(hist, fc, INITS, members=True)


def dots(fig):
    """(x, y) arrays of the member dots: the translucent round markers."""
    line = next(l for l in fig.axes[0].lines if l.get_marker() == "o" and l.get_alpha() is not None)
    return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_builds_in_both_modes(mode):
    """The figure builds and renders, with one y axis and a legend of several series."""
    hist, fc = synthetic()
    fig = forecast_vs_truth(definition(TWO_WET, JJAS), fc, history=hist, lat=0, lon=0, lead_window=(1, 25), mode=mode)
    fig.canvas.draw()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_legend().get_texts()) >= 3


def test_dates_equal_the_engines():
    """Every plotted member dot is an ONSET member's onset date at its init, as detect_forecast gives it."""
    hist, fc = synthetic()
    defn = definition(TWO_WET, JJAS)
    fig = forecast_vs_truth(defn, fc, history=hist, lat=0, lon=0, lead_window=(1, 25))
    ds = engine.detect_forecast(defn, fc, history=hist, lead_window=(1, 25)).isel(season=0, lat=0, lon=0)
    status, date = ds["status"].values, ds["onset_date"].values.astype("datetime64[D]").astype(np.int64)
    inits = np.array(INITS, dtype="datetime64[D]").astype(np.int64)
    expected = sorted((int(inits[i]), int(date[i, j])) for i in range(len(INITS)) for j in range(4)
                      if status[i, j] == m.Status.ONSET)
    x, y = dots(fig)
    assert expected
    assert sorted(zip(x.astype(int), y.astype(int))) == expected


def test_errors():
    """Derived seasons and non-standard calendars are refused."""
    hist, fc = synthetic()
    fc.calendar = "noleap"
    with pytest.raises(ValueError, match="standard calendar"):
        forecast_vs_truth(definition(TWO_WET, JJAS), fc, history=hist, lat=0, lon=0)
