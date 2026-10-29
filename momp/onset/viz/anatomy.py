"""The anatomy plot: daily rain, the candidates the search evaluated and the rolling total (or anomaly) behind its decision.

One figure per Anatomy (momp.onset.explain). Panels share the date axis and each has a single y-axis:
(a) daily rain with the search window, trigger window, dry spells and the onset day; (b) one marker per
evaluated candidate by outcome; (c) the window total with its threshold, or the cumulative anomaly with
its minimum. Colours come from momp.onset.viz.style roles, never from hard-coded hues.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from momp.onset.explain import _OP_WORDS, Anatomy, _total_label, date_ymd, format_date
from momp.onset.model import Status
from momp.onset.viz import style

_ROW_LABELS = ("requirement not met", "dry spell followed", "undecided / onset")
_OP_SIGNS = {"ge": "≥", "gt": ">", "le": "≤", "lt": "<"}
_MONTH_STEPS = (1, 2, 3, 5, 7, 10, 14, 21)


def _coord_text(location: dict) -> str:
    """Latitude and longitude as "9.0°N 38.75°E"."""
    parts = []
    if "lat" in location:
        parts.append(f"{abs(location['lat']):g}°{'N' if location['lat'] >= 0 else 'S'}")
    if "lon" in location:
        parts.append(f"{abs(location['lon']):g}°{'E' if location['lon'] >= 0 else 'W'}")
    return " ".join(parts)


def _title(a: Anatomy) -> str:
    """Definition and place (plus the year, or the init and member)."""
    place = _coord_text(a.location)
    if a.is_forecast:
        member = f", member {a.member}" if a.member is not None else ""
        when = f"init {format_date(a.init, True)}{member}"
    else:
        when = f"{a.season} {a.year}"
    return " - ".join(part for part in (a.definition.title, place, when) if part)


def _result_text(a: Anatomy) -> str:
    """The result in words, without the status icon."""
    r = a.result
    noun = "forecast" if a.is_forecast else "data"
    lead = f", lead {r.onset_lead}" if r.onset_lead is not None else ""
    if r.status == Status.ONSET:
        return f"onset {format_date(r.date, True)}{lead}"
    if r.status == Status.PRIOR:
        return f"onset {format_date(r.date, True)}, before the forecast{lead}"
    if r.status == Status.NOT_FOUND:
        return "no onset in the search window"
    if r.status == Status.CENSORED:
        return f"undecided at {format_date(a.date_of(r.decision_day))}: needs days past the {noun}"
    return "missing data"


def _tick_positions(a: Anatomy, n: int) -> tuple[list[int], list[str]]:
    """Tick offsets and labels: month starts for long ranges, else a regular day step."""
    first = int(a.days[0])
    if n > 70:
        positions = [i for i in range(n) if date_ymd(a.date_of(first + i))[2] == 1]
    else:
        step = next((s for s in _MONTH_STEPS if n / s <= 9), 28)
        positions = list(range(0, n, step))
    return positions, [format_date(a.date_of(first + i)) for i in positions]


def _first_total(a: Anatomy):
    """Index of the first total requirement of the rule, or None when it has none."""
    for c in a.candidates:
        for check in c.checks:
            if check.role == "require" and check.kind == "total":
                return check.index
    return None


def _status_label(status: Status, is_forecast: bool) -> tuple[str, str, str]:
    """(colour, icon, label) of a status, with the label saying "data" instead of "forecast" for observations."""
    colour, icon, label = style.status_style(status)
    return colour, icon, label if is_forecast else label.replace("forecast", "data")


def _washes(ax, a: Anatomy, p: style.Palette, x0: int, handles: dict) -> None:
    """Search-window, trigger-window and veto-span washes behind the bars."""
    ax.axvspan(a.first_candidate - x0 - 0.5, a.last_candidate - x0 + 0.5, color=p.wash, lw=0, zorder=0)
    handles["Search window"] = Patch(facecolor=p.wash, edgecolor=p.axis, lw=style.HAIRLINE)
    onset = next((c for c in a.candidates if c.state == "T"), None)
    if onset is None:
        return
    for role, colour, label in (("require", p.observed, "Trigger window"), ("veto", p.dry_spell, "Dry-spell check window")):
        spans = [(c.first, c.last) for c in onset.checks if c.role == role and c.n_days > 0]
        if not spans:
            continue
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        ax.axvspan(lo - x0 - 0.5, hi - x0 + 0.5, color=colour, alpha=style.WASH_ALPHA, lw=0, zorder=0.2)
        handles[label] = Patch(facecolor=colour, alpha=style.WASH_ALPHA * 2.5, edgecolor=colour, lw=style.HAIRLINE)


def _vetoing_spells(a: Anatomy) -> list[tuple[int, int, int]]:
    """Distinct dry spells (first, last, length) that ruled out candidates, in date order."""
    seen = {}
    for c in a.candidates:
        if c.state == "F" and c.checks[c.decisive].role == "veto":
            spell = c.checks[c.decisive].spell
            seen.setdefault((spell.first, spell.last), spell.length)
    return sorted((f, l, n) for (f, l), n in seen.items())


def _lane_label(ax, text: str, bars: list, y: float, days_per_inch: float, n: int, ink: dict,
                only: int | None = None, avoid: tuple = ()) -> None:
    """Label a baseline lane beside its first bar (or the bar `only`), on whichever side has room."""
    lo, hi = bars[0] if only is None else bars[only]
    width = len(text) * 8 * 0.56 / 72 * days_per_inch
    others = [b for b in bars if b != (lo, hi)] + list(avoid)
    right_clear = hi + 0.5 + width < n and not any(b[0] < hi + 0.5 + width and b[1] > hi for b in others)
    if right_clear:
        ax.text(hi + 0.5, y, text, ha="left", va="center", **ink)
    else:
        ax.text(lo - 0.5, y, text, ha="right", va="center", **ink)


def _rain_panel(ax, a: Anatomy, p: style.Palette, x0: int, n: int, handles: dict) -> None:
    """Daily rain bars with the lanes, hairlines and direct labels of panel (a)."""
    x = a.days - x0
    _washes(ax, a, p, x0, handles)
    forecast = a.source == "forecast"
    colours = np.where(forecast, p.forecast, p.observed)
    ax.bar(x, a.rain_mm, width=style.BAR_MAX_WIDTH_DAYS, color=colours, lw=0, zorder=2)
    if (~forecast).any():
        handles["Observed rain"] = Patch(facecolor=p.observed, lw=0)
    if forecast.any():
        handles["Forecast rain"] = Patch(facecolor=p.forecast, lw=0)
    top = max(float(a.rain_mm.max()) * 1.3, 5.0)
    lane = 0.05 * top
    spells = [s for s in _vetoing_spells(a) if s[1] >= a.days[0] and s[0] <= a.days[-1]]
    missing = np.nonzero(a.missing)[0]
    bottom = -0.5 * lane
    ink = {"size": 8, "color": p.ink_muted}
    onset = a.result.day
    avoid = () if onset is None else ((onset - x0 - 0.5, onset - x0 + 0.5),)
    days_per_inch = n / (ax.figure.get_figwidth() * (ax.get_position().width))
    if len(missing):
        bottom -= 1.6 * lane
        for i in missing:
            ax.add_patch(Rectangle((x[i] - 0.5, bottom + 0.3 * lane), 1, 0.9 * lane, color=p.ink_muted, lw=0, zorder=2))
        _lane_label(ax, "missing days", [(x[i] - 0.5, x[i] + 0.5) for i in missing], bottom + 0.75 * lane,
                    days_per_inch, n, ink, avoid=avoid)
    if spells:
        bottom -= 1.6 * lane
        bars = [(f - x0 - 0.5, l - x0 + 0.5) for f, l, _ in spells]
        for lo, hi in bars:
            ax.add_patch(Rectangle((lo, bottom + 0.3 * lane), hi - lo, 0.9 * lane, color=p.dry_spell, lw=0, zorder=2))
        _lane_label(ax, f"dry spell, {spells[0][2]} days", bars, bottom + 0.75 * lane, days_per_inch, n, ink, only=0, avoid=avoid)
    ax.set_ylim(bottom, top)
    ax.axhline(a.wet_mm, color=p.ink_muted, lw=style.HAIRLINE, zorder=1.5)
    ax.text(1.01, a.wet_mm, f"wet day\n{_OP_SIGNS[a.wet_op]} {a.wet_mm:g} mm", transform=ax.get_yaxis_transform(),
            ha="left", va="center", **ink)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= top])
    ax.set_ylabel("rain (mm/day)")
    if a.is_forecast:
        _forecast_marker(ax, a, p, x0, n, top)
    onset = a.result.day
    if onset is not None and a.days[0] <= onset <= a.days[-1]:
        ox = onset - x0
        ax.axvline(ox, color=p.ink, lw=style.LINE_WIDTH, zorder=3)
        right = ox > 0.75 * n
        ax.text(ox + (-0.5 if right else 0.5), 0.93 * top, f"Onset {format_date(a.date_of(onset))}", ha="right" if right else "left",
                va="center", size=9, weight="semibold", color=p.ink, zorder=4)


def _forecast_marker(ax, a: Anatomy, p: style.Palette, x0: int, n: int, top: float) -> None:
    """A hairline where the forecast begins (lead 1), labelled with its date."""
    if a.split < int(a.days[0]):
        return
    fx = a.split - x0 - 0.5
    ax.axvline(fx, color=p.ink_muted, lw=style.HAIRLINE, zorder=1.8)
    left = fx < 0.5 * n
    ax.text(fx + (0.5 if left else -0.5), 0.80 * top, f"forecast from {format_date(a.date_of(a.split))}",
            ha="left" if left else "right", va="center", size=8, color=p.ink_secondary)


def _candidate_panel(ax, a: Anatomy, p: style.Palette, x0: int, handles: dict) -> None:
    """One marker per evaluated candidate, on a row by how it was decided."""
    rows = {"require": 0, "veto": 1}
    for c in a.candidates:
        x = c.day - x0
        if c.state == "F":
            role = c.checks[c.decisive].role
            colour = p.dry_spell if role == "veto" else p.ink_muted
            ax.plot([x], [rows[role]], marker="|", ms=style.MARKER_SIZE, mew=1.5, color=colour, ls="none", zorder=3)
        elif c.state == "T":
            colour, icon, label = _status_label(Status.ONSET, a.is_forecast)
            ax.plot([x], [2], marker="o", ms=style.MARKER_SIZE + 1, color=colour, ls="none", zorder=4)
            handles[f"{icon} {label}"] = Line2D([], [], marker="o", ms=style.MARKER_SIZE, color=colour, ls="none")
        else:
            status = Status.CENSORED if c.flags == ("horizon",) else Status.INSUFFICIENT_DATA
            colour, icon, label = _status_label(status, a.is_forecast)
            ax.plot([x], [2], marker="D", ms=style.MARKER_SIZE, color=colour, ls="none", zorder=4)
            handles[f"{icon} {label}"] = Line2D([], [], marker="D", ms=style.MARKER_SIZE, color=colour, ls="none")
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([0, 1, 2], _ROW_LABELS)
    ax.tick_params(axis="y", length=0)


def _total_panel(ax, a: Anatomy, p: style.Palette, x0: int, index: int) -> None:
    """The rolling window total of the first total requirement for every candidate, against its threshold."""
    days, values, check0 = [], [], None
    for c in a.candidates:
        check = next(ch for ch in c.checks if ch.role == "require" and ch.index == index)
        check0 = check0 or check
        days.append(c.day - x0)
        values.append(np.nan if check.value is None else check.value)
    values = np.array(values)
    ax.plot(days, values, color=p.observed, lw=style.LINE_WIDTH, zorder=3)
    if len(days) == 1 or np.isfinite(values).sum() == 1:
        ax.plot(days, values, marker="o", ms=style.MARKER_SIZE, color=p.observed, ls="none")
    ax.axhline(check0.limit, color=p.ink_muted, lw=style.HAIRLINE, zorder=2)
    ax.text(1.01, check0.limit, f"{_OP_WORDS[check0.op]}\n{check0.limit:g} mm", transform=ax.get_yaxis_transform(),
            ha="left", va="center", size=8, color=p.ink_muted)
    onset = next((c for c in a.candidates if c.state == "T"), None)
    if onset is not None:
        ax.plot([onset.day - x0], [next(ch.value for ch in onset.checks if ch.role == "require" and ch.index == index)],
                marker="o", ms=style.MARKER_SIZE + 1, color=p.ink, ls="none", zorder=4)
    ax.set_ylim(min(0.0, np.nanmin(values) if np.isfinite(values).any() else 0.0), None)
    ax.set_ylabel(f"{_total_label(check0)}\nfrom each candidate (mm)" if not check0.since_start else "season total (mm)")


def _curve_panel(ax, a: Anatomy, p: style.Palette, x0: int) -> None:
    """The cumulative anomaly with its minimum marked and labelled."""
    c = a.curve
    ax.plot(c.days - x0, c.cumulative_mm, color=p.observed, lw=style.LINE_WIDTH, zorder=3)
    ax.axhline(0, color=p.ink_muted, lw=style.HAIRLINE, zorder=2)
    mx = c.min_day - x0
    ax.plot([mx], [c.min_mm], marker="o", ms=style.MARKER_SIZE + 1, color=p.ink, ls="none", zorder=4)
    right = mx > 0.7 * (a.days[-1] - x0)
    ax.annotate(f"minimum {c.min_mm:.0f} mm, {format_date(a.date_of(c.min_day))}", (mx, c.min_mm),
                xytext=(-8 if right else 8, 8), textcoords="offset points", ha="right" if right else "left",
                va="bottom", size=8, color=p.ink)
    ax.set_ylabel(f"cumulative anomaly (mm)\nrain minus {c.baseline_mm:.3g} mm/day")


def _subtitle(fig: Figure, a: Anatomy, left: float, y: float, p: style.Palette) -> None:
    """The result line: the status icon in its colour, then its label and detail in ink."""
    colour, icon, label = _status_label(a.result.status, a.is_forecast)
    icon_text = fig.text(left, y, icon, size=10, weight="bold", color=colour, va="center")
    fig.canvas.draw()
    box = icon_text.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(box.x1 + 0.006, y, f"{label}: {_result_text(a)}" if a.result.status != Status.ONSET else _result_text(a),
             size=9, color=p.ink_secondary, va="center")


def anatomy_plot(a: Anatomy, *, mode: str = "light", figsize: tuple[float, float] | None = None) -> Figure:
    """Draw the anatomy of one onset search as stacked panels sharing the date axis.

    mode is "light" or "dark". The figure has the daily rain, the candidate strip (first-passage rules)
    and the rolling total or cumulative anomaly (when the rule has one).
    """
    total_index = _first_total(a)
    panels = ["rain"]
    if a.candidates:
        panels.append("candidates")
    if a.curve is not None or total_index is not None:
        panels.append("curve" if a.curve is not None else "total")
    ratios = {"rain": 3.6, "candidates": 1.1, "curve": 2.0, "total": 2.0}
    size = figsize or (9.2, 2.2 + 0.85 * sum(ratios[k] for k in panels))
    with style.figure_style(mode) as p:
        fig = Figure(figsize=size, dpi=100)
        fig.set_facecolor(p.surface)
        left, right = 0.17, 0.89
        top, bottom = 1 - 1.3 / size[1], 0.65 / size[1]
        axes = fig.subplots(len(panels), 1, sharex=True,
                            gridspec_kw={"height_ratios": [ratios[k] for k in panels], "hspace": 0.38,
                                         "left": left, "right": right, "top": top, "bottom": bottom})
        axes = np.atleast_1d(axes)
        x0 = int(a.days[0])
        n = len(a.days)
        extra = max((ch.need_last for c in a.candidates if c.state == "U" for ch in c.checks
                     if ch.outcome == "unknown"), default=a.days[-1]) - a.days[-1]
        n_axis = n + max(int(extra), 0)
        handles: dict = {}
        by_name = dict(zip(panels, axes))
        _rain_panel(by_name["rain"], a, p, x0, n_axis, handles)
        if "candidates" in by_name:
            cand_handles: dict = {}
            _candidate_panel(by_name["candidates"], a, p, x0, cand_handles)
            if cand_handles:
                by_name["candidates"].legend(cand_handles.values(), cand_handles.keys(), loc="lower right",
                                             bbox_to_anchor=(1.0, 1.0), ncol=len(cand_handles), handletextpad=0.3,
                                             columnspacing=1.2, labelcolor=p.ink_secondary)
        if "total" in by_name:
            _total_panel(by_name["total"], a, p, x0, total_index)
        if "curve" in by_name:
            _curve_panel(by_name["curve"], a, p, x0)
        by_name["rain"].legend(handles.values(), handles.keys(), loc="lower left", bbox_to_anchor=(0.0, 1.0),
                               ncol=len(handles), handletextpad=0.5, columnspacing=1.4, labelcolor=p.ink_secondary)
        axes[-1].set_xlim(-0.5, n_axis - 0.5)
        ticks, labels = _tick_positions(a, n_axis)
        axes[-1].set_xticks(ticks, labels)
        years = sorted({date_ymd(d)[0] for d in a.dates})
        axes[-1].set_xlabel(" - ".join(str(y) for y in (years[0], years[-1])) if len(years) > 1 else str(years[0]))
        for ax in axes:
            ax.tick_params(axis="x", length=3)
        fig.text(left, 1 - 0.28 / size[1], _title(a), size=11, weight="semibold", color=p.ink, va="center")
        _subtitle(fig, a, left, 1 - 0.58 / size[1], p)
    return fig
