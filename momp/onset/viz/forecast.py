"""Forecast onset dates against the observed onset: one dot per member and init, with the ensemble median.

x is the init date and y the onset date of each ensemble member (engine.detect_forecast with the observed
history stitched in before every forecast). Onsets below the "init date" diagonal are back-dated into the
history; inits whose onset was already decided by the history alone (PRIOR) get the status marker; the
members without an onset are counted per init above the axes.
"""
from __future__ import annotations

import textwrap
import warnings
from dataclasses import replace

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from momp.onset import engine
from momp.onset.explain import _cell_of, _season_index, _slice_coords, format_date
from momp.onset.model import DailyRain, Definition, DerivedSeasons, ForecastRain, Status
from momp.onset.viz import style

_STANDARD = {"standard", "gregorian", "proleptic_gregorian"}
_NO_ONSET = (Status.NOT_FOUND, Status.CENSORED, Status.INSUFFICIENT_DATA)
_ROW_IN = 0.2
_FIG_WIDTH = 9.6


def _coord_text(lat: float, lon: float) -> str:
    """Latitude and longitude as "9.0°N 38.75°E"."""
    return f"{abs(lat):g}°{'N' if lat >= 0 else 'S'} {abs(lon):g}°{'E' if lon >= 0 else 'W'}"


def _one_cell(forecast: ForecastRain, history: DailyRain, lat: float, lon: float, thresholds):
    """The forecast, history and thresholds cut to the cell nearest (lat, lon), with every dim kept (length 1)."""
    n_lead = 1 + ("member" in forecast.dims)
    f_dims = forecast.dims[n_lead:-1]
    f_cell = _cell_of(f_dims, forecast.coords, lat, lon)
    f_sl = {dim: slice(i, i + 1) for dim, i in zip(f_dims, f_cell)}
    f_index = tuple(f_sl.get(dim, slice(None)) for dim in forecast.dims)
    fc1 = replace(forecast, tenths=forecast.tenths[f_index], missing=forecast.missing[f_index],
                  coords=_slice_coords(forecast.coords, f_sl))
    h_dims = history.dims[:-1]
    h_cell = _cell_of(h_dims, history.coords, lat, lon)
    h_sl = {dim: slice(i, i + 1) for dim, i in zip(h_dims, h_cell)}
    h_index = tuple(h_sl.get(dim, slice(None)) for dim in history.dims)
    hist1 = replace(history, tenths=history.tenths[h_index], missing=history.missing[h_index],
                    coords=_slice_coords(history.coords, h_sl))
    out = None
    if thresholds:
        out = {}
        for name, value in thresholds.items():
            if isinstance(value, xr.DataArray) and value.ndim:
                value = value.sel({d: [float(fc1.coords[d][0])] for d in f_dims if d in value.dims}, method="nearest")
            out[name] = value
    place = {dim: float(fc1.coords[dim][0]) for dim in f_dims if dim in fc1.coords}
    return fc1, hist1, out, place


def _inits(forecast: ForecastRain) -> np.ndarray:
    """The init dates as datetime64[D]."""
    return pd.to_datetime({"year": forecast.init_year, "month": forecast.init_month,
                           "day": forecast.init_day}).to_numpy().astype("datetime64[D]")


def _days(dates: np.ndarray) -> np.ndarray:
    """Dates (NaT allowed) as float days since 1970-01-01, NaN for NaT."""
    values = dates.astype("datetime64[D]")
    return np.where(np.isnat(values), np.nan, values.astype("int64").astype(float))


def _median_days(days: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Per-init median of the days of the kept members (NaN where none are kept)."""
    masked = np.where(keep, days, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(masked, axis=1)


def _climatology_cell(climatology: xr.Dataset, hist1: DailyRain, season: int) -> tuple[xr.Dataset, int]:
    """The season climatology cut to the history's single cell (every dim kept, length 1) and that cell's season count.

    Raises ValueError when the cell has fewer than `season` seasons.
    """
    dims = hist1.dims[:-1]
    cell = climatology.sel({d: [float(hist1.coords[d][0])] for d in dims}, method="nearest")
    n = int(cell["n_seasons"].values.reshape(-1)[0]) if "n_seasons" in cell else int(
        np.isfinite(cell["onset_doy"].values.reshape(cell.sizes["season"], -1)[:, 0]).sum())
    if season > n:
        raise ValueError(f"season {season} requested but the cell has {n} season{'s' if n != 1 else ''}")
    return cell, n


def _observed_onsets(definition, hist1, thresholds, years, season_name, season_climatology=None):
    """{year: (status, date or None)} of the observed onset of each year at the cell."""
    obs = engine.detect_observed(definition, hist1, thresholds=thresholds, years=years,
                                 season_climatology=season_climatology)
    out = {}
    for year in years:
        sel = obs.sel(year=year, season=season_name).isel({d: 0 for d in obs["status"].dims if d not in ("year", "season")})
        status = Status(int(sel["status"]))
        date = None if status != Status.ONSET else pd.Timestamp(sel["onset_date"].values.item())
        out[int(year)] = (status, date)
    return out


def _thin(positions: list[float], priority: list[float], gap: float) -> list[int]:
    """Indices to label: higher priority first, skipping any closer than `gap` to one already kept."""
    kept: list[int] = []
    for i in sorted(range(len(positions)), key=lambda j: -priority[j]):
        if all(abs(positions[i] - positions[j]) >= gap for j in kept):
            kept.append(i)
    return kept


def _count_rows(fig, ax, init_x, status, mode: str = "light") -> None:
    """Small icon-and-count labels per init above the axes, one row per status present, thinned to stay legible."""
    present = [s for s in _NO_ONSET if (status == s).any()]
    renderer = fig.canvas.get_renderer()
    x_to_px = ax.transData.transform([(x, 0) for x in init_x])[:, 0]
    for row, s in enumerate(present):
        colour, icon, label = style.status_style(s, mode)
        counts = (status == s).sum(axis=1)
        idx = np.nonzero(counts)[0]
        y = 1.0 + (_ROW_IN * (len(present) - 1 - row) + 0.06) / (ax.get_position().height * fig.get_figheight())
        probe = ax.text(0, y, f"{icon} {int(counts.max())}", transform=ax.transAxes, size=7)
        gap = probe.get_window_extent(renderer).width + 3
        probe.remove()
        for i in (idx[j] for j in _thin(list(x_to_px[idx]), list(counts[idx]), gap)):
            ax.text(init_x[i], y, f"{icon} {int(counts[i])}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    size=7, color=colour, clip_on=False)
        ax.text(1.012, y, f"{icon} {label}", transform=ax.transAxes, ha="left", va="bottom", size=8, color=colour,
                clip_on=False)


def forecast_vs_truth(definition: Definition, forecast: ForecastRain, *, history: DailyRain, lat: float, lon: float,
                      season: str | int | None = None, thresholds=None, lead_window: tuple[int, int] = (1, 30),
                      season_climatology: xr.Dataset | None = None, mode: str = "light") -> Figure:
    """Onset date of every ensemble member against the init date, with the observed onset as a line.

    The dates come from engine.detect_forecast on the cell nearest (lat, lon) with `history` stitched in
    before lead 1, and the observed onset from engine.detect_observed on the history. Each member with an
    onset is a dot; the median of the members that have one is the ink marker joined by a thin line; inits
    whose onset the history alone had decided (PRIOR) carry the status marker. Dots below the diagonal are
    back-dated into the history. Members without an onset are counted per init above the axes. season picks
    one season by name (default: the first); thresholds are as for detect_forecast. Needs a standard
    calendar.

    Derived (per-cell) seasons need season_climatology (n_seasons, onset_doy, cessation_doy, as from
    climatology.harmonic_seasons) and take `season` as 1 or 2 (default 1; ValueError when the cell has fewer).
    The forecast and the observed onset both use that cell's derived season (engine.detect_forecast with
    season_climatology), and the subtitle names the season.
    """
    if forecast.calendar not in _STANDARD:
        raise ValueError(f"forecast_vs_truth needs a standard calendar, got '{forecast.calendar}'")
    derived = isinstance(definition.seasons, DerivedSeasons)
    fc1, hist1, thr1, place = _one_cell(forecast, history, lat, lon, thresholds)
    clim1 = None
    if derived:
        if season_climatology is None:
            raise ValueError("derived seasons need season_climatology (n_seasons, onset_doy, cessation_doy)")
        number = 1 if season is None else int(season)
        if number < 1:
            raise ValueError(f"season must be 1 or 2 for derived seasons, got {season!r}")
        clim1, n_seasons = _climatology_cell(season_climatology, hist1, number)
        season_name = str(number)
        det = engine.detect_forecast(definition, fc1, thresholds=thr1, lead_window=lead_window, history=hist1,
                                     season_climatology=clim1)
        det = det.sel(season=season_name)
    else:
        det = engine.detect_forecast(definition, fc1, thresholds=thr1, lead_window=lead_window, history=hist1)
        names = [str(s) for s in det["season"].values]
        season_name = names[_season_index(names, season)]
        det = det.sel(season=season_name)
    spatial = {d: 0 for d in det["status"].dims if d not in ("init_time", "member", "season")}
    det = det.isel(spatial)
    status = det["status"].values.astype(int)
    onset = det["onset_date"].values
    if status.ndim == 1:
        status, onset = status[:, None], onset[:, None]
    inits = _inits(fc1)
    days = _days(onset)
    n_init, n_member = status.shape
    years = sorted({int(y) for y in fc1.init_year})
    observed = _observed_onsets(definition, hist1, thr1, years, season_name, clim1)

    keep_onset = status == Status.ONSET
    prior_rows = (status == Status.PRIOR).any(axis=1)
    median = _median_days(days, (status == Status.ONSET) | (status == Status.PRIOR))
    init_num = inits.astype("int64").astype(float)

    present = [s for s in _NO_ONSET if (status == s).any()]
    cell = _coord_text(place.get("lat", lat), place.get("lon", lon))
    with style.figure_style(mode) as pal:
        title = f"{definition.title} - {cell.replace(chr(32), chr(160))} - forecast onset by init"
        title = textwrap.fill(title, 78)
        n_title = title.count("\n")
        extra = (0.2 if derived else 0.0) + 0.2 * n_title
        height = 5.4 + extra + _ROW_IN * len(present)
        fig = Figure(figsize=(_FIG_WIDTH, height), dpi=100)
        FigureCanvasAgg(fig)
        fig.set_facecolor(pal.surface)
        top_in = 1.25 + extra + _ROW_IN * len(present)
        ax = fig.add_axes([0.085, 0.72 / height, 1 - 0.085 - 1.45 / _FIG_WIDTH, 1 - (top_in + 0.72) / height])

        all_days = np.concatenate([days[np.isfinite(days)], [d.toordinal() - 719163 for _, d in observed.values() if d is not None]])
        lo_y = (all_days.min() if len(all_days) else init_num.min()) - 3
        hi_y = (all_days.max() if len(all_days) else init_num.max()) + 3
        lo_x, hi_x = init_num.min() - 1.5, init_num.max() + 1.5
        lo, hi = min(lo_x, lo_y), max(hi_x, hi_y)
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(lo_y, hi_y)
        ax.grid(True, axis="x", color=pal.grid, lw=style.HAIRLINE)

        ax.plot([lo, hi], [lo, hi], color=pal.ink_muted, lw=style.HAIRLINE, ls=(0, (4, 3)), zorder=1.5)
        xs, ys = np.nonzero(keep_onset)
        ax.plot(init_num[xs], days[xs, ys], "o", ms=3.2, color=pal.observed, alpha=0.35 if mode == "light" else 0.55, mew=0, zorder=2)
        for year, (st, date) in observed.items():
            in_year = [i for i, y in enumerate(fc1.init_year) if int(y) == year]
            x0, x1 = init_num[in_year].min() - 1.5, init_num[in_year].max() + 1.5
            if date is None:
                continue
            y = date.toordinal() - 719163
            ax.plot([x0, x1], [y, y], color=pal.ink_muted, lw=style.LINE_WIDTH, zorder=2.5, solid_capstyle="butt")
            ax.text(x1 - 0.5, y + 0.6, f"observed onset {format_date(date.to_datetime64(), True)}", ha="right", va="bottom",
                    size=8, color=pal.ink, zorder=5,
                    bbox={"facecolor": pal.surface, "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
        ok = np.isfinite(median)
        ax.plot(init_num[ok], median[ok], "-", color=pal.ink, lw=1.0, alpha=0.6, zorder=3)
        ax.plot(init_num[ok & ~prior_rows], median[ok & ~prior_rows], "o", ms=5, color=pal.ink, mec=pal.surface, mew=0.8,
                zorder=4)
        p_colour, p_icon, p_label = style.status_style(Status.PRIOR, pal.mode)
        if (ok & prior_rows).any():
            ax.plot(init_num[ok & prior_rows], median[ok & prior_rows], marker="<", ms=4.5, color=p_colour, mec=pal.surface,
                    mew=0.4, ls="none", zorder=4.5)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.yaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        fmt = mdates.DateFormatter("%-d %b")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("init date")
        ax.set_ylabel("onset date")
        _diagonal_label(ax, pal, lo, hi, bool(((days < init_num[:, None]) & keep_onset).any() or prior_rows.any()))

        handles = [Line2D([], [], marker="o", ms=5, color=pal.observed, alpha=0.6, ls="none",
                          label=f"member onset ({n_member} members)" if n_member > 1 else "onset"),
                   Line2D([], [], marker="o", ms=6, color=pal.ink, ls="-", lw=1.0, mec=pal.surface, label="ensemble median"),
                   Line2D([], [], color=pal.ink_muted, lw=style.LINE_WIDTH, label="observed onset")]
        if prior_rows.any():
            handles.insert(2, Line2D([], [], marker="<", ms=5, color=p_colour, ls="none", label=f"{p_icon} {p_label}"))
        legend_y = 1.0 + (_ROW_IN * len(present) + 0.2) / (ax.get_position().height * height)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, legend_y), ncol=len(handles), handletextpad=0.4,
                  columnspacing=1.6, labelcolor=pal.ink_secondary, borderaxespad=0)
        fig.canvas.draw()
        _count_rows(fig, ax, init_num, status, pal.mode)
        fig.text(0.085, 1 - 0.28 / height, title, size=11, weight="semibold", color=pal.ink, va="center")
        sub = f"{fc1.source or 'forecast'} with {hist1.source or 'observed'} history stitched in before lead 1; lead window {lead_window[0]}-{lead_window[1]}"
        if derived:
            sub += f"\nseason {season_name} of {n_seasons} at this cell"
        fig.text(0.085, 1 - (0.46 + 0.2 * n_title if derived else 0.56 + 0.2 * n_title) / height, sub, size=9,
                 color=pal.ink_secondary, va="top" if derived else "center", linespacing=1.4)
    return fig


def _diagonal_label(ax, pal, lo: float, hi: float, back_dated: bool) -> None:
    """Label the init-date diagonal along the line, saying that onsets below it are back-dated."""
    (x0, y0), (x1, y1) = ax.transData.transform([(lo, lo), (hi, hi)])
    angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    (xl, xr), (yl, yr) = ax.get_xlim(), ax.get_ylim()
    first, last = max(xl, yl), min(xr, yr)
    t = first + 0.62 * (last - first)
    text = "init date: onsets below this line are back-dated" if back_dated else "init date"
    ax.annotate(text, (t, t), xytext=(0, 4), textcoords="offset points", rotation=angle, rotation_mode="anchor",
                ha="center", va="bottom", size=8, color=pal.ink_muted)
