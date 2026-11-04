"""Compare several onset definitions on one cell and year: the rain, and where each definition puts the onset.

`compare_definitions` is the entry point: it runs ``engine.detect_observed`` for each definition on the one cell and
draws the daily rain above a dot plot of the onset dates, in the colours of `hcwf.onset.viz.style`.
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from hcwf.onset import climatology, engine
from hcwf.onset.model import DailyRain, Definition, DerivedSeasons, Status
from hcwf.onset.viz import style

__all__ = []

_STANDARD = {"standard", "gregorian", "proleptic_gregorian"}
ROW_IN = 0.42
"""Height in inches of one definition's row in the dot plot."""
PAD_DAYS = 12
"""Days of room kept right of the last onset date, for its label."""


def _nearest(rain: DailyRain, lat: float, lon: float) -> tuple[tuple[slice, ...], float, float]:
    """Return the one-cell slices and coordinates of the grid cell nearest ``(lat, lon)``.

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The gridded rain; it needs ``lat`` and ``lon`` dims.
    lat : float
        Wanted latitude, in degrees.
    lon : float
        Wanted longitude, in degrees.

    Returns
    -------
    slices : tuple of slice
        One slice per spatial dim of ``rain``, of length 1 along ``lat`` and ``lon``.
    lat : float
        Latitude of the cell, in degrees.
    lon : float
        Longitude of the cell, in degrees.

    Raises
    ------
    ValueError
        ``rain`` lacks a ``lat`` or ``lon`` dim, or the point is farther than one grid step from the nearest cell.
    """
    index = []
    found = {}
    for dim, target in (("lat", lat), ("lon", lon)):
        if dim not in rain.dims[:-1]:
            raise ValueError(f"rain has dims {rain.dims}; compare_definitions needs lat and lon dims")
        values = np.asarray(rain.coords[dim], dtype=float)
        i = int(np.argmin(np.abs(values - target)))
        step = np.median(np.abs(np.diff(values))) if values.size > 1 else np.inf
        if abs(values[i] - target) > step:
            raise ValueError(f"{dim}={target} is outside the rain grid ({values.min():g} to {values.max():g}). "
                             f"Fix: pick a {dim} inside the grid")
        found[dim] = float(values[i])
        index.append(i)
    slices = [slice(None)] * (len(rain.dims) - 1)
    for dim, i in zip(("lat", "lon"), index):
        slices[rain.dims.index(dim)] = slice(i, i + 1)
    return tuple(slices), found["lat"], found["lon"]


def _cell_rain(rain: DailyRain, slices: tuple[slice, ...]) -> DailyRain:
    """Return the rain restricted to one grid cell (dims kept, length 1).

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The gridded rain.
    slices : tuple of slice
        One slice per spatial dim, as returned by `_nearest`.

    Returns
    -------
    hcwf.onset.model.DailyRain
        The rain of the cell.
    """
    coords = dict(rain.coords)
    for dim, sl in zip(rain.dims[:-1], slices):
        coords[dim] = np.asarray(rain.coords[dim])[sl]
    return replace(rain, tenths=rain.tenths[(*slices, slice(None))], missing=rain.missing[(*slices, slice(None))],
                   coords=coords)


def _cell_thresholds(values: dict, dims: tuple[str, ...], slices: tuple[slice, ...]) -> dict:
    """Restrict the given thresholds to the chosen cell.

    Parameters
    ----------
    values : dict
        The thresholds by name, each an `xarray.DataArray` (cut by dim name) or an array-like over the spatial dims.
    dims : tuple of str
        The spatial dims of the rain.
    slices : tuple of slice
        One slice per spatial dim, as returned by `_nearest`.

    Returns
    -------
    dict
        The thresholds of the cell, under the same names.
    """
    out = {}
    for name, v in values.items():
        if hasattr(v, "isel"):
            out[name] = v.isel({d: sl for d, sl in zip(dims, slices) if d in v.dims})
        else:
            out[name] = np.asarray(v)[slices]
    return out


def _labels(definitions: list[Definition]) -> list[str]:
    """Return the row labels: definition ids, told apart by the title's last parenthesis (or a number) when they repeat.

    Parameters
    ----------
    definitions : list of hcwf.onset.model.Definition
        The definitions, in row order.

    Returns
    -------
    list of str
        One label per definition.
    """
    ids = [d.id for d in definitions]
    labels = []
    seen: dict[str, int] = {}
    for d in definitions:
        seen[d.id] = seen.get(d.id, 0) + 1
        if ids.count(d.id) == 1 or seen[d.id] == 1:
            labels.append(d.id)
        elif d.title.endswith(")") and "(" in d.title:
            labels.append(f"{d.id} ({d.title.rsplit('(', 1)[1]}")
        else:
            labels.append(f"{d.id} ({seen[d.id]})")
    return labels


def _detect(definition: Definition, cell: DailyRain, year: int, thresholds, threshold_years, season):
    """Run ``detect_observed`` for one definition on the cell.

    Parameters
    ----------
    definition : hcwf.onset.model.Definition
        The definition to run.
    cell : hcwf.onset.model.DailyRain
        The rain of the one cell.
    year : int
        The year to detect.
    thresholds : dict, optional
        The thresholds by definition id, each a mapping of name to array; used when it has this definition.
    threshold_years : tuple of (int, int), optional
        The first and last year from which missing thresholds (and derived seasons) are computed.
    season : str, optional
        Keep this season only; None keeps every season of the definition.

    Returns
    -------
    list of tuple of (str, hcwf.onset.model.Status, pandas.Timestamp or None)
        The season name, status and onset date (None without an onset), one per season.

    Raises
    ------
    ValueError
        The definition needs thresholds or derived seasons and ``threshold_years`` is None, or ``season`` is not one of
        its seasons.
    """
    needs = bool(definition.thresholds)
    values = None
    if needs:
        given = None if thresholds is None else thresholds.get(definition.id)
        if given is not None:
            values = given
        elif threshold_years is None:
            raise ValueError(f"definition '{definition.id}' needs thresholds ({', '.join(n for n, _ in definition.thresholds)}). "
                             "Fix: pass thresholds={id: {name: DataArray}} or threshold_years=(first, last)")
        else:
            values = climatology.compute_thresholds(definition, cell, threshold_years)
    seasons_clim = None
    if isinstance(definition.seasons, DerivedSeasons):
        if threshold_years is None:
            raise ValueError(f"definition '{definition.id}' derives its seasons from climatology. "
                             "Fix: pass threshold_years=(first, last)")
        seasons_clim = climatology.harmonic_seasons(cell, threshold_years, definition.seasons.max_seasons)
    ds = engine.detect_observed(definition, cell, thresholds=values, years=[year], season_climatology=seasons_clim)
    names = [str(s) for s in ds["season"].values]
    if season is not None:
        if season not in names:
            raise ValueError(f"definition '{definition.id}' has no season '{season}' (it has: {', '.join(names)}). "
                             "Fix: pass season=None or one of those names")
        names = [season]
    out = []
    for name in names:
        sel = ds.sel(year=year, season=name).squeeze()
        status = Status(int(sel["status"]))
        date = None
        if status == Status.ONSET:
            date = pd.Timestamp(sel["onset_date"].values.item())
        out.append((name, status, date))
    return out


def _bars(cell: DailyRain, lo: pd.Timestamp, hi: pd.Timestamp):
    """Return the dates and rain in mm of the cell's days within ``[lo, hi]``, skipping missing days.

    Parameters
    ----------
    cell : hcwf.onset.model.DailyRain
        The rain of the one cell.
    lo : pandas.Timestamp
        First date, inclusive.
    hi : pandas.Timestamp
        Last date, inclusive.

    Returns
    -------
    dates : numpy.ndarray of datetime64
        The dates kept.
    mm : numpy.ndarray of float64
        The daily rain in mm on those dates.
    """
    dates = pd.to_datetime({"year": cell.year, "month": cell.month, "day": cell.day})
    mm = cell.tenths.reshape(-1, cell.tenths.shape[-1])[0] / 10.0
    keep = (dates >= lo) & (dates <= hi) & ~cell.missing.reshape(-1, cell.tenths.shape[-1])[0]
    return dates[keep].to_numpy(), mm[keep]


def compare_definitions(definitions: list[Definition], rain: DailyRain, *, lat: float, lon: float, year: int,
                        thresholds: dict | None = None, threshold_years: tuple[int, int] | None = None,
                        season: str | None = None, mode: str = "light") -> Figure:
    """Draw the daily rain for one cell and year above a dot plot of each definition's onset date.

    One row per definition (labelled with its id); each onset is a dot with its date. A definition with no onset shows
    its status icon and label at the row end. Dates come from ``engine.detect_observed``. Needs a standard calendar.

    Parameters
    ----------
    definitions : list of hcwf.onset.model.Definition
        The definitions to compare, one row each.
    rain : hcwf.onset.model.DailyRain
        The gridded observed rain, with ``lat`` and ``lon`` dims.
    lat : float
        Latitude of the cell, in degrees.
    lon : float
        Longitude of the cell, in degrees.
    year : int
        The year to show.
    thresholds : dict, optional
        Maps a definition id to ``{name: xarray.DataArray}``; definitions that need thresholds and are not in it get
        them computed from ``rain`` over ``threshold_years``.
    threshold_years : tuple of (int, int), optional
        The first and last year of the climatology for computed thresholds (also needed for derived seasons).
    season : str, optional
        Show one season by name; None shows every season of each definition.
    mode : {"light", "dark"}, default "light"
        The colour mode.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not shown or saved.

    Raises
    ------
    ValueError
        ``definitions`` is empty, the calendar is not standard, or the cell, thresholds or season cannot be resolved
        (see `_nearest` and `_detect`).
    """
    definitions = list(definitions)
    if not definitions:
        raise ValueError("compare_definitions needs at least one definition")
    if rain.calendar not in _STANDARD:
        raise ValueError(f"compare_definitions needs a standard calendar, got '{rain.calendar}'")
    slices, cell_lat, cell_lon = _nearest(rain, lat, lon)
    cell = _cell_rain(rain, slices)
    results = []
    for d in definitions:
        given = None if thresholds is None else thresholds.get(d.id)
        given = None if given is None else _cell_thresholds(given, rain.dims[:-1], slices)
        results.append(_detect(d, cell, year, {d.id: given} if given is not None else None, threshold_years, season))
    onset_dates = [date for r in results for _, _, date in r if date is not None]
    lo = pd.Timestamp(year, 1, 1)
    hi = max([pd.Timestamp(year, 12, 31), *[t + pd.Timedelta(days=PAD_DAYS) for t in onset_dates]])
    with style.figure_style(mode) as pal:
        n = len(definitions)
        fig = Figure(figsize=(9.0, 2.6 + ROW_IN * n))
        FigureCanvasAgg(fig)
        top, dots = fig.subplots(2, 1, sharex=True, height_ratios=[2.6, ROW_IN * n + 0.5])
        labels = _labels(definitions)
        renderer = fig.canvas.get_renderer()
        label_in = max(renderer.get_text_width_height_descent(t, FontProperties(size=9.0), False)[0]
                       for t in labels) / fig.dpi
        fig.subplots_adjust(left=min(0.5, (label_in + 0.5) / 9.0), right=0.97, top=0.9, bottom=0.12, hspace=0.08)
        dates, mm = _bars(cell, lo, hi)
        top.bar(dates, mm, width=style.BAR_MAX_WIDTH_DAYS, color=pal.observed, linewidth=0)
        top.grid(False)
        top.set_ylabel("daily rain (mm)")
        top.set_title(f"{cell_lat:g}\N{DEGREE SIGN}N {cell_lon:g}\N{DEGREE SIGN}E, {year}: daily rain and onset by definition")
        _draw_dots(fig, dots, pal, results, labels, lo, hi)
        fig.patch.set_facecolor(pal.surface)
    return fig


def _draw_dots(fig: Figure, ax, pal: style.Palette, results, labels: list[str], lo, hi) -> None:
    """Draw one row per definition: a dot and its date per onset, status text at the row end otherwise.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that owns the axes.
    ax : matplotlib.axes.Axes
        The dot-plot axes.
    pal : hcwf.onset.viz.style.Palette
        The palette of the figure.
    results : list of list of tuple
        For each definition the ``(season name, status, date or None)`` entries of `_detect`.
    labels : list of str
        The row labels, one per definition.
    lo : pandas.Timestamp
        First date of the x axis.
    hi : pandas.Timestamp
        Last date of the x axis.
    """
    n = len(results)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=pal.ink)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(mdates.date2num(lo) - 1, mdates.date2num(hi) + 1)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    renderer = fig.canvas.get_renderer()
    for i, per_season in enumerate(results):
        several = len(per_season) > 1
        missing = []
        for name, status, date in per_season:
            if date is None:
                colour, icon, label = style.status_style(status, pal.mode)
                missing.append((colour, f"{icon} {name + ': ' if several else ''}{label}"))
                continue
            ax.plot([date], [i], "o", color=pal.ink, markersize=style.MARKER_SIZE, zorder=3)
            text = f"{date.day} {date:%b}" + (f" ({name})" if several else "")
            _label_dot(ax, renderer, pal, date, i, text)
        right = 1.0
        for colour, text in reversed(missing):
            t = ax.text(right, i, text, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                        color=colour, clip_on=False)
            right = ax.transAxes.inverted().transform(t.get_window_extent(renderer))[0][0] - 0.02


def _label_dot(ax, renderer, pal: style.Palette, date, row: int, text: str) -> None:
    """Write the date right of its dot, or left of it when the right side is out of the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The dot-plot axes.
    renderer : matplotlib.backend_bases.RendererBase
        The renderer used to measure the text.
    pal : hcwf.onset.viz.style.Palette
        The palette of the figure.
    date : pandas.Timestamp
        The date of the dot.
    row : int
        The row of the dot.
    text : str
        The label text.
    """
    x = mdates.date2num(date)
    label = ax.text(x, row, "  " + text, ha="left", va="center", fontsize=8.5, color=pal.ink, zorder=4)
    right_edge = ax.get_window_extent(renderer).x1
    if label.get_window_extent(renderer).x1 > right_edge:
        label.set_text(text + "  ")
        label.set_ha("right")
