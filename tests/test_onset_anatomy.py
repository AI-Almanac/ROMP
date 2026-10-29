"""momp.onset.viz.anatomy: the anatomy plot builds for every registry definition and follows the figure rules."""
import numpy as np
import pytest
from matplotlib.colors import to_hex

from momp.onset import climatology, explain
from momp.onset import definition as dfn
from momp.onset import model as m
from momp.onset.model import Status
from momp.onset.viz import style
from momp.onset.viz.anatomy import anatomy_plot
from test_onset_engine import ANY_WET, definition, make_forecast, make_rain, season
from test_onset_forecast_layer import JJAS, case, hist_with


def seasonal_rain(seed=0, n_years=3, shape=(2, 2)):
    """Three years of seasonal synthetic rain (wet from the middle of the year) on a small grid."""
    rng = np.random.default_rng(seed)
    n_time = 365 * n_years + 1
    doy = np.arange(n_time) % 365
    chance = 0.08 + 0.5 * np.clip(np.sin((doy - 100) / 365 * 2 * np.pi), 0, None)
    wet = rng.random((*shape, n_time)) < chance
    values = np.where(wet, rng.gamma(0.9, 80.0, (*shape, n_time)), 0.0)
    return make_rain(np.rint(values).astype(np.int32))


def trace_registry(ident, rain):
    """Trace one registry definition at the first cell of 2001, with whatever thresholds and seasons it needs."""
    defn = dfn.load(ident)
    thresholds = climatology.compute_thresholds(defn, rain, (2000, 2002)) if defn.thresholds else None
    clim = None
    if isinstance(defn.seasons, m.DerivedSeasons):
        clim = climatology.harmonic_seasons(rain, (2000, 2002))
    return explain.trace_observed(defn, rain, lat=rain.coords["lat"][0], lon=rain.coords["lon"][0], year=2001,
                                  thresholds=thresholds, season_climatology=clim)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("ident", dfn.registry_ids())
def test_every_registry_definition_plots_in_both_modes(ident, mode):
    """The plot builds and renders for every registry definition on synthetic data."""
    a = trace_registry(ident, seasonal_rain())
    fig = anatomy_plot(a, mode=mode)
    fig.canvas.draw()
    assert fig.get_facecolor()[:3] == pytest.approx(
        tuple(int(style.palette(mode).surface[i:i + 2], 16) / 255 for i in (1, 3, 5)), abs=1e-3)
    assert a.definition.title in fig.texts[0].get_text()


def forecast_anatomy(stitched=True):
    """A stitched (or forecast-only) anatomy with observed history, a forecast and a missing day."""
    values = np.zeros((1, 1, 1, 20), dtype=np.int32)
    values[..., 3:5] = 100
    hist, fc = case(hist_with(["2000-06-02", "2000-06-03"]), values, ["2000-06-10"])
    hist.missing[..., 160] = True
    return explain.trace_forecast(definition(m.FirstPassage((m.WetDays(m.Span(1, 2), "ge", 2),)), JJAS), fc,
                                  init="2000-06-10", lat=0, lon=0, history=hist if stitched else None,
                                  lead_window=(1, 20))


@pytest.mark.parametrize("stitched", [True, False])
def test_forecast_plot_marks_lead_one_and_labels_both_series(stitched):
    """Forecast days take the forecast colour, a legend names both series, and lead 1 is marked."""
    a = forecast_anatomy(stitched)
    fig = anatomy_plot(a)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("forecast from 10 Jun") for t in texts)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert ("Forecast rain" in legend) and (("Observed rain" in legend) == stitched)
    colours = {to_hex(bar.get_facecolor()) for container in ax.containers for bar in container}
    assert style.palette("light").forecast in colours


def test_figure_rules_single_axes_thin_bars_text_ink_and_markers():
    """One y-axis per panel, bars at most 0.8 day wide, markers at least 8 pt, lines 2 pt, and text only in ink tokens."""
    a = trace_registry("icpac", seasonal_rain(seed=2))
    for mode in ("light", "dark"):
        p = style.palette(mode)
        fig = anatomy_plot(a, mode=mode)
        assert len(fig.axes) == 3
        xs = [tuple(ax.get_shared_x_axes().get_siblings(ax)) for ax in fig.axes]
        assert all(len(s) == 3 for s in xs)
        for ax in fig.axes:
            for container in ax.containers:
                assert all(bar.get_width() <= style.BAR_MAX_WIDTH_DAYS + 1e-9 for bar in container)
            for line in ax.lines:
                if line.get_marker() not in (None, "None", ""):
                    assert line.get_markersize() >= style.MARKER_SIZE
                elif line.get_linestyle() == "-" and line.get_linewidth() > 1.0:
                    assert line.get_linewidth() == style.LINE_WIDTH
        allowed = {p.ink, p.ink_secondary, p.ink_muted} | {c for c, _, _ in style.STATUS_STYLE.values()}
        texts = [t for ax in fig.axes for t in ax.texts] + list(fig.texts)
        texts += [t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()]
        for ax in fig.axes:
            if ax.get_legend():
                texts += ax.get_legend().get_texts()
        assert all(to_hex(t.get_color()) in allowed for t in texts)
        series = {p.observed, p.dry_spell, p.forecast}
        assert not any(to_hex(t.get_color()) in series for t in texts)


def test_no_label_on_every_bar():
    """Labels are selective: far fewer text objects than days."""
    a = trace_registry("icpac", seasonal_rain(seed=2))
    fig = anatomy_plot(a)
    assert sum(len(ax.texts) for ax in fig.axes) < 12 < len(a.days)


def test_unknown_missing_and_not_found_cases_plot():
    """Undecided, missing-data and no-onset anatomies draw without error and name their status."""
    defn = dfn.load("icpac")
    dry = make_rain(np.zeros((1, 1, 366)))
    values = np.zeros((1, 1, 366), dtype=np.int32)
    values[..., 130:133] = 100
    values[..., 133::5] = 30
    cut = make_rain(values[..., :150])
    cut.missing[..., 140] = True
    for rain, status in ((dry, Status.NOT_FOUND), (make_rain(values[..., :150]), Status.CENSORED),
                         (cut, Status.INSUFFICIENT_DATA)):
        a = explain.trace_observed(defn, rain, lat=-10, lon=-10, year=2000)
        assert a.result.status == status
        fig = anatomy_plot(a, mode="dark")
        fig.canvas.draw()
        assert style.status_style(status)[1] in [t.get_text() for t in fig.texts]
