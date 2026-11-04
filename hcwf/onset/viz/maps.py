"""Drill-down figure: a map of observed onset day of year with the anatomy of one marked cell beside it.

`drill_down` is the entry point. It draws the map here and the anatomy panels with `hcwf.onset.viz.anatomy`, using the
colours of `hcwf.onset.viz.style`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hcwf.onset.explain import Anatomy
from hcwf.onset.model import Status
from hcwf.onset.viz import style
from hcwf.onset.viz.anatomy import PANEL_RATIOS, _subtitle, _title, anatomy_panels, draw_anatomy
from hcwf.onset.units import MONTH_NAMES as _MONTH_NAMES

_MAP_IN = 4.4
_LEFT_IN = 0.8
_GAP_IN = 2.0
_RIGHT_IN = 1.0


def _ramp(pal: style.Palette) -> LinearSegmentedColormap:
    """Return a single-hue sequential ramp in the observed hue.

    The ramp runs light to dark on the light surface and dark to light on the dark one.

    Parameters
    ----------
    pal : hcwf.onset.viz.style.Palette
        The palette of the figure.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A three-colour colormap named ``"onset_doy"``.
    """
    hue = np.array(to_rgb(pal.observed))
    tint = hue * 0.3 + 0.7
    shade = hue * 0.45
    colours = [shade, hue, tint] if pal.surface == style.DARK.surface else [tint, hue, shade]
    return LinearSegmentedColormap.from_list("onset_doy", colours)


def _month_ticks(year: int, lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Return the day-of-year values of the month starts (and labels) between two days, thinned to about six.

    Parameters
    ----------
    year : int
        The year whose calendar gives the day-of-year values.
    lo : float
        First day of year of the range.
    hi : float
        Last day of year of the range.

    Returns
    -------
    ticks : list of float
        Day of year of each tick.
    labels : list of str
        The tick labels, such as ``"1 May"``.
    """
    starts = [(pd.Timestamp(year, m, 1).dayofyear, f"1 {_MONTH_NAMES[m - 1]}") for m in range(1, 13)]
    inside = [(d, t) for d, t in starts if lo <= d <= hi]
    step = max(1, int(np.ceil(len(inside) / 6)))
    inside = inside[::step]
    return [d for d, _ in inside], [t for _, t in inside]


def _draw_map(fig: Figure, ax, cax, pal: style.Palette, result: xr.Dataset, a: Anatomy, map_h: float) -> None:
    """Draw the onset day-of-year map, with the colourbar and status legend.

    Cells without an onset are grey and the cell of the anatomy is marked.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that owns the axes.
    ax : matplotlib.axes.Axes
        The map axes.
    cax : matplotlib.axes.Axes
        The axes of the horizontal colourbar.
    pal : hcwf.onset.viz.style.Palette
        The palette of the figure.
    result : xarray.Dataset
        The observed result with ``status`` and ``onset_date`` over ``(year, season, lat, lon)``.
    a : hcwf.onset.explain.Anatomy
        The anatomy that gives the year, season and the cell to mark.
    map_h : float
        The height of the map axes in inches, used to place the legend below the colourbar.
    """
    sel = result.sel(year=a.year, season=a.season)
    status = sel["status"]
    onset = pd.DatetimeIndex(sel["onset_date"].values.reshape(-1)).dayofyear.to_numpy().astype(float)
    onset = onset.reshape(sel["onset_date"].shape)
    has = (status.values == int(Status.ONSET)) & np.isfinite(onset)
    lat, lon = sel["lat"].values, sel["lon"].values
    grey = style.status_style(Status.NOT_FOUND, pal.mode)[0]
    ax.set_facecolor(pal.surface)
    ax.grid(False)
    ax.pcolormesh(lon, lat, np.where(has, np.nan, 1.0), cmap=LinearSegmentedColormap.from_list("grey", [grey, grey]),
                  shading="nearest", alpha=0.45, rasterized=True)
    lo, hi = (float(np.nanmin(onset[has])), float(np.nanmax(onset[has]))) if has.any() else (1.0, 365.0)
    mesh = ax.pcolormesh(lon, lat, np.where(has, onset, np.nan), cmap=_ramp(pal), vmin=lo, vmax=max(hi, lo + 1),
                         shading="nearest", rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    cell_lat, cell_lon = a.location["lat"], a.location["lon"]
    ax.plot([cell_lon], [cell_lat], marker="s", ms=11, mfc="none", mec=pal.surface, mew=3.2, ls="none", zorder=5)
    ax.plot([cell_lon], [cell_lat], marker="s", ms=11, mfc="none", mec=pal.ink, mew=1.6, ls="none", zorder=6)
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    ticks, labels = _month_ticks(int(a.year), lo, hi)
    bar.set_ticks(ticks, labels=labels)
    bar.outline.set_visible(False)
    cax.tick_params(length=2, labelsize=8)
    cax.set_xlabel(f"onset date, {a.year} {a.season}")
    handles = [Line2D([], [], marker="s", ms=8, mfc="none", mec=pal.ink, mew=1.6, ls="none", label="explained cell")]
    for s in (Status.NOT_FOUND, Status.CENSORED, Status.INSUFFICIENT_DATA, Status.PRIOR):
        n = int((status == int(s)).sum())
        if n:
            colour, icon, label = style.status_style(s, pal.mode)
            handles.append(Patch(facecolor=grey, alpha=0.45, label=f"{icon} {label}: {n} cells"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -2.35 / map_h), ncol=1, handletextpad=0.5,
              labelcolor=pal.ink_secondary, borderaxespad=0)


def drill_down(result: xr.Dataset, anatomy: Anatomy, *, mode: str = "light") -> Figure:
    """Draw an onset map (day of year) with the anatomy of one marked cell beside it.

    ``result`` is a ``detect_observed`` dataset (dims ``year, season, lat, lon``) and ``anatomy`` a
    `hcwf.onset.explain.trace_observed` anatomy of one of its cells and years; the map shows the anatomy's year and
    season, cells without an onset in neutral grey (the legend gives each status with its icon and count), and the
    anatomy's cell is outlined. The anatomy panels share the figure on the right.

    Parameters
    ----------
    result : xarray.Dataset
        The observed result, with variables ``status`` and ``onset_date`` over ``(year, season, lat, lon)``.
    anatomy : hcwf.onset.explain.Anatomy
        The anatomy of one cell and year of ``result``.
    mode : {"light", "dark"}, default "light"
        The colour mode.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not shown or saved.

    Raises
    ------
    ValueError
        ``anatomy`` is of a forecast or has no year, or ``result`` lacks ``lat`` or ``lon`` dims.
    """
    if anatomy.is_forecast or anatomy.year is None:
        raise ValueError("drill_down needs an observed Anatomy (trace_observed)")
    if "lat" not in result.dims or "lon" not in result.dims:
        raise ValueError(f"drill_down needs a result with lat and lon dims, got {tuple(result.dims)}")
    panels = anatomy_panels(anatomy)
    right_in = 7.2
    width = _LEFT_IN + _MAP_IN + _GAP_IN + right_in + _RIGHT_IN
    height = 2.2 + 0.85 * sum(PANEL_RATIOS[k] for k in panels)
    with style.figure_style(mode) as pal:
        fig = Figure(figsize=(width, height), dpi=100)
        FigureCanvasAgg(fig)
        fig.set_facecolor(pal.surface)
        top_in, bottom_in = 1.3, 0.65
        map_h = min(_MAP_IN, height - top_in - 2.4)
        ax = fig.add_axes([_LEFT_IN / width, 1 - (top_in + map_h) / height, _MAP_IN / width, map_h / height])
        cax = fig.add_axes([_LEFT_IN / width, 1 - (top_in + map_h + 0.85) / height, _MAP_IN / width, 0.12 / height])
        _draw_map(fig, ax, cax, pal, result, anatomy, map_h)
        left = (_LEFT_IN + _MAP_IN + _GAP_IN) / width
        right = (_LEFT_IN + _MAP_IN + _GAP_IN + right_in) / width
        grid = fig.subplots(len(panels), 1, sharex=True,
                            gridspec_kw={"height_ratios": [PANEL_RATIOS[k] for k in panels], "hspace": 0.38, "left": left,
                                         "right": right, "top": 1 - top_in / height, "bottom": bottom_in / height})
        draw_anatomy(dict(zip(panels, np.atleast_1d(grid))), anatomy, pal)
        fig.text(_LEFT_IN / width, 1 - 0.28 / height, _title(anatomy), size=11, weight="semibold", color=pal.ink, va="center")
        _subtitle(fig, anatomy, _LEFT_IN / width, 1 - 0.58 / height, pal)
    return fig
