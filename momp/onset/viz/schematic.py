"""Rule schematics: a picture of what an onset definition asks for, drawn from the Definition alone.

First-passage rules become one labelled band per requirement and one bracket per
veto on a shared axis of days relative to the candidate day (day 1); curve rules
become a stylised cumulative-anomaly curve. All text is placed from measured
extents, one row per item, so labels never overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from momp.onset.describe import describe
from momp.onset.model import (
    Amount, Comparison, Curve, DayRef, Definition, DerivedSeasons, DryRun, FirstPassage, Span, Status,
    ThresholdSpec, Total, Veto, WetDays,
)
from momp.onset.viz import style

WIDTH_IN = 9.0
MARGIN_IN = 0.35
LABEL_PT = 8.5
SMALL_PT = 8.0
BAND_IN = 0.24
BRACKET_IN = 0.12
SPELL_IN = 0.2
ROW_GAP_IN = 0.16
SEARCH_START_DAYS = 5
MEASURE_SLACK = 1.04

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_OP_WORDS = {"ge": "at least", "gt": "more than", "le": "at most", "lt": "less than"}


class _Measure:
    """Text widths and line heights in inches, from the figure's own renderer."""

    def __init__(self, fig: Figure):
        """Bind to the renderer of `fig`."""
        self.fig = fig
        self.renderer = fig.canvas.get_renderer()

    def width(self, text: str, size: float, weight: str = "normal") -> float:
        """Width of one line of text, with a little slack for font differences between renderers."""
        fp = FontProperties(size=size, weight=weight)
        w, _, _ = self.renderer.get_text_width_height_descent(text, fp, False)
        return w / self.fig.dpi * MEASURE_SLACK

    def line(self, size: float) -> float:
        """Height of one line of text including leading."""
        fp = FontProperties(size=size)
        _, h, _ = self.renderer.get_text_width_height_descent("Hgy", fp, False)
        return h / self.fig.dpi * 1.35

    def wrap(self, text: str, size: float, max_width: float, weight: str = "normal") -> list[str]:
        """Greedy word wrap so that every line is at most max_width inches."""
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if current and self.width(trial, size, weight) > max_width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        return lines or [""]


def _ref_text(ref: DayRef) -> str:
    """A season or threshold bound in words: '1 May' or 'climatological onset - 50 days'."""
    if ref.climatological is None:
        return f"{ref.day} {_MONTHS[ref.month - 1]}"
    if ref.offset_days == 0:
        return f"climatological {ref.climatological}"
    sign = "+" if ref.offset_days > 0 else "-"
    return f"climatological {ref.climatological} {sign} {abs(ref.offset_days)} days"


def _range_text(first: DayRef, last: DayRef) -> str:
    """Two bounds as '1 May-30 Sep', or 'a to b' when either is climatological."""
    if first.climatological is None and last.climatological is None:
        return f"{_ref_text(first)}-{_ref_text(last)}"
    return f"{_ref_text(first)} to {_ref_text(last)}"


def _threshold_words(spec: ThresholdSpec) -> str:
    """A threshold definition in words: 'mean 5-day total, 1 May-30 Sep'."""
    during = "whole year" if spec.during is None else _range_text(*spec.during)
    if spec.kind == "mean_total":
        what = f"mean {spec.days}-day total"
    elif spec.kind == "mean_daily":
        what = "mean daily rain"
    else:
        what = f"{spec.quantile:g} quantile of {spec.days}-day totals"
    return f"{what}, {during}"


def _wraps_year(season) -> bool:
    """Whether a fixed season ends in the year after it starts."""
    if season.start.month is None or season.end.month is None:
        return False
    return (season.end.month, season.end.day) < (season.start.month, season.start.day)


def _season_words(defn: Definition) -> str:
    """The search window(s) in words, for the header."""
    seasons = defn.seasons
    if isinstance(seasons, DerivedSeasons):
        return (f"Search window: derived per cell by harmonic analysis of the mean annual cycle "
                f"(one or two seasons a year), from {_ref_text(seasons.start)} to {_ref_text(seasons.end)}")
    parts = [f"{_range_text(s.start, s.end)}{' (ends next year)' if _wraps_year(s) else ''}" for s in seasons]
    if len(seasons) == 1:
        return f"Search window: {parts[0]}"
    return "Search windows: " + "; ".join(f"{s.name} {p}" for s, p in zip(seasons, parts))


def _amount_words(amount: Amount) -> str:
    """'20 mm' or the threshold's name."""
    return f"{amount.tenths / 10:g} mm" if amount.tenths is not None else str(amount.threshold)


def _cmp_words(c: Comparison) -> str:
    """'more than 20 mm'."""
    return f"{_OP_WORDS[c.op]} {_amount_words(c.amount)}"


def _days_words(first: int, last: int) -> str:
    """'day 3' or 'days 2 to 22'."""
    return f"day {first}" if first == last else f"days {first} to {last}"


def _requirement_label(cond: Total | WetDays, wet: Comparison) -> str:
    """A requirement as a short label for its band."""
    span = cond.span
    if isinstance(cond, Total):
        text = _cmp_words(cond.comparison)
        if span.since_search_start:
            return f"season-to-date total {text}: from the search start through day {span.last}"
        if span.first == span.last:
            return f"day {span.first} total {text}"
        if span.first == 1:
            return f"{span.last}-day total {text}"
        return f"total of {_days_words(span.first, span.last)} {text}"
    length = span.last - span.first + 1
    wet_text = _cmp_words(wet)
    if cond.op == "ge" and cond.count == length:
        if length == 1:
            return f"day {span.first} wet: {wet_text}"
        if span.first != 1:
            return f"every day of {_days_words(span.first, span.last)} wet ({wet_text} each)"
        every = "both days" if length == 2 else f"all {length} days"
        return f"{every} wet ({wet_text} each)"
    return f"{_OP_WORDS[cond.op]} {cond.count} wet days in {_days_words(span.first, span.last)} (wet: {wet_text})"


def _spell_length(spell) -> int:
    """Length in days of a dry run or dry window."""
    return spell.length if isinstance(spell, DryRun) else spell.days


def _veto_label(veto: Veto) -> str:
    """A veto as a label: 'no dry run of at least 7 days within days 2 to 22'."""
    spell = veto.spell
    if isinstance(spell, DryRun):
        what = (f"dry run of more than {spell.length - 1} days" if spell.phrase == "more than"
                else f"dry run of at least {spell.length} days")
    else:
        what = f"dry window of {spell.days} days totalling {_cmp_words(spell.comparison)}"
    where = _days_words(veto.span.first, veto.span.last)
    if veto.placement == "inside":
        return f"no {what} within {where}"
    return f"no {what} starting within {where} (it may end later)"


def _spell_example_text(veto: Veto) -> str:
    """The text drawn on the example spell."""
    spell = veto.spell
    if isinstance(spell, DryRun):
        return f"{spell.length} dry days"
    return f"{spell.days} days, {_cmp_words(spell.comparison)} in total"


@dataclass
class _Row:
    """One drawn item: its kind, label, lines, and the vertical space it needs."""

    kind: str
    label: str
    span: Span
    veto: Veto | None = None
    lines: list[str] | None = None
    label_x: float = 0.0
    height: float = 0.0


def _first_passage_rows(defn: Definition) -> list[_Row]:
    """The rows of a first-passage schematic, in the order requirements, vetoes, report."""
    rule = defn.rule
    rows = [_Row("require", _requirement_label(c, defn.wet_day), c.span) for c in rule.require]
    rows += [_Row("veto", _veto_label(v), v.span, v) for v in rule.reject_if]
    if rule.report == "first_wet_day":
        s = rule.report_span
        rows.append(_Row("report", f"reported onset: the first wet day within {_days_words(s.first, s.last)}", s))
    return rows


def _x_range(rows: list[_Row]) -> tuple[float, float]:
    """Day-axis limits (cell edges) that show every span, the candidate day, and every example spell."""
    lo, hi = 1, 3
    for row in rows:
        s = row.span
        if not s.since_search_start:
            lo = min(lo, s.first)
        hi = max(hi, s.last)
        if row.veto is not None:
            start = s.last if row.veto.placement == "starts_inside" else s.first
            hi = max(hi, start + _spell_length(row.veto.spell) - 1)
    left = lo - 0.5 - (SEARCH_START_DAYS if any(r.span.since_search_start for r in rows) else 0)
    return left, hi + 0.5


class _Canvas:
    """The one axis with helpers that work in inches for text and in days for x."""

    def __init__(self, ax, meas: _Measure, pal: style.Palette, xlim: tuple[float, float], width_in: float):
        """Wrap the axis; `width_in` is its drawn width."""
        self.ax, self.meas, self.pal = ax, meas, pal
        self.xlim, self.width_in = xlim, width_in
        self.per_day = width_in / (xlim[1] - xlim[0])

    def inch(self, x: float) -> float:
        """Inches from the left of the axis to day-axis position x."""
        return (x - self.xlim[0]) * self.per_day

    def x(self, inches: float) -> float:
        """Day-axis position of an offset in inches from the left of the axis."""
        return self.xlim[0] + inches / self.per_day

    def text(self, x: float, y: float, s: str, *, ha="left", va="bottom", size=LABEL_PT, color=None, weight="normal"):
        """Draw unclipped text at day x and inch y."""
        return self.ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color or self.pal.ink,
                            fontweight=weight, clip_on=False)

    def label_block(self, row: _Row, y_top: float, anchor: float) -> float:
        """Place a row's wrapped label so it stays inside the axis; return the y below it."""
        line_h = self.meas.line(LABEL_PT)
        for i, line in enumerate(row.lines):
            w = self.meas.width(line, LABEL_PT)
            left = min(self.inch(anchor), self.width_in - w)
            self.text(self.x(max(left, 0.0)), y_top - (i + 1) * line_h, line)
        return y_top - len(row.lines) * line_h

    def placed_inside_or_beside(self, s: str, x0: float, x1: float, y: float, height: float, size=SMALL_PT) -> None:
        """Write s centred in [x0, x1] when it fits, else just right of x1, else just left of x0."""
        w = self.meas.width(s, size)
        span_in = (x1 - x0) * self.per_day
        yc = y + height / 2
        if w + 0.12 <= span_in:
            self.text((x0 + x1) / 2, yc, s, ha="center", va="center", size=size)
        elif self.inch(x1) + 0.08 + w <= self.width_in:
            self.text(self.x(self.inch(x1) + 0.08), yc, s, va="center", size=size, color=self.pal.ink_secondary)
        else:
            self.text(self.x(self.inch(x0) - 0.08), yc, s, ha="right", va="center", size=size,
                      color=self.pal.ink_secondary)


def _draw_cells(cv: _Canvas, first: float, last: float, y: float, h: float, *, face, edge, alpha=style.WASH_ALPHA,
                dashed=False):
    """Draw a band covering days first..last with a hairline between days."""
    ax = cv.ax
    ax.add_patch(Rectangle((first - 0.5, y), last - first + 1, h, facecolor=face, alpha=alpha, edgecolor="none",
                           zorder=2))
    ax.add_patch(Rectangle((first - 0.5, y), last - first + 1, h, facecolor="none", edgecolor=edge,
                           linewidth=style.HAIRLINE, linestyle="--" if dashed else "-", zorder=3))
    if last - first + 1 <= 40:
        for d in np.arange(first + 0.5, last + 0.5, 1.0):
            ax.plot([d, d], [y, y + h], color=edge, linewidth=0.5, alpha=0.45, zorder=3)


def _draw_row(cv: _Canvas, row: _Row, y_top: float, anchor: float) -> None:
    """Draw one requirement, veto or report row with its top at y_top."""
    pal, ax = cv.pal, cv.ax
    y = cv.label_block(row, y_top, anchor) - 0.05
    s = row.span
    if row.kind == "require" and s.since_search_start:
        ymid = y - BAND_IN / 2
        tail = cv.xlim[0] + 0.15
        ax.annotate("", xy=(s.last + 0.5, ymid), xytext=(tail, ymid), zorder=3,
                    arrowprops={"arrowstyle": "-|>", "color": pal.observed, "linewidth": style.LINE_WIDTH,
                                "shrinkA": 0, "shrinkB": 0})
        ax.plot([tail, tail], [ymid - 0.07, ymid + 0.07], color=pal.observed, linewidth=style.LINE_WIDTH, zorder=3)
        cv.text(tail, ymid - 0.1, "search start", va="top", size=SMALL_PT, color=pal.ink_secondary)
    elif row.kind == "require":
        _draw_cells(cv, s.first, s.last, y - BAND_IN, BAND_IN, face=pal.observed, edge=pal.observed)
    elif row.kind == "report":
        _draw_cells(cv, s.first, s.last, y - BAND_IN, BAND_IN, face=pal.ink_muted, edge=pal.ink_secondary, dashed=True)
    else:
        _draw_veto(cv, row, y)


def _draw_veto(cv: _Canvas, row: _Row, y: float) -> None:
    """Draw a veto's bracket and an example spell below its label (y is the top of the bracket zone)."""
    pal, ax = cv.pal, cv.ax
    s, veto = row.span, row.veto
    x0, x1 = s.first - 0.5 + 0.04, s.last + 0.5 - 0.04
    ax.plot([x0, x0, x1, x1], [y - BRACKET_IN, y, y, y - BRACKET_IN], color=pal.dry_spell,
            linewidth=style.LINE_WIDTH, solid_joinstyle="miter", zorder=3)
    length = _spell_length(veto.spell)
    start = s.last if veto.placement == "starts_inside" else s.first
    y_spell = y - BRACKET_IN - 0.05 - SPELL_IN
    kept = min(start + length - 1, s.last)
    ax.add_patch(Rectangle((start - 0.5, y_spell), kept - start + 1, SPELL_IN, facecolor=pal.dry_spell, alpha=0.2,
                           edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((start - 0.5, y_spell), length, SPELL_IN, facecolor="none", edgecolor=pal.dry_spell,
                           linewidth=style.HAIRLINE, linestyle="-" if kept == start + length - 1 else "--", zorder=3))
    cv.placed_inside_or_beside(_spell_example_text(veto), start - 0.5, start + length - 0.5, y_spell, SPELL_IN)


def _row_height(cv: _Canvas, row: _Row) -> float:
    """Vertical space of a row: label lines, then its band, bracket or arrow."""
    height = len(row.lines) * cv.meas.line(LABEL_PT) + 0.05 + ROW_GAP_IN
    if row.kind == "veto":
        return height + BRACKET_IN + 0.05 + SPELL_IN
    if row.span.since_search_start:
        return height + BAND_IN + 0.14
    return height + BAND_IN


def _prepare_rows(cv: _Canvas, rows: list[_Row]) -> None:
    """Wrap each label to the axis width and set the row heights."""
    for row in rows:
        row.lines = cv.meas.wrap(row.label, LABEL_PT, cv.width_in)
        row.height = _row_height(cv, row)


def _header_lines(defn: Definition, meas: _Measure, width: float) -> list[tuple[str, float, str, str, float]]:
    """Header text as (line, size, weight, colour role, space above in inches)."""
    lines: list[tuple[str, float, str, str, float]] = []

    def add(text: str, size: float, weight: str, role: str, gap: float) -> None:
        """Wrap text and append its lines."""
        for i, line in enumerate(meas.wrap(text, size, width, weight)):
            lines.append((line, size, weight, role, gap if i == 0 else 0.0))

    add(defn.title, 11.0, "bold", "ink", 0.0)
    add(f"{defn.id}  v{defn.version}", SMALL_PT, "normal", "ink_muted", 0.02)
    add(describe(defn), LABEL_PT, "normal", "ink_secondary", 0.08)
    add(_season_words(defn), LABEL_PT, "normal", "ink", 0.08)
    wet = f"Wet day: {_cmp_words(defn.wet_day)}"
    if isinstance(defn.rule, FirstPassage) and any(isinstance(v.spell, DryRun) for v in defn.rule.reject_if):
        wet += " (a dry day is any other day with data)"
    add(wet, LABEL_PT, "normal", "ink", 0.02)
    if isinstance(defn.rule, Curve):
        add(_curve_header_extra(defn), LABEL_PT, "normal", "ink", 0.02)
        return lines
    for name, spec in defn.thresholds:
        add(f"{name} = {_threshold_words(spec)}", LABEL_PT, "normal", "ink", 0.02)
    return lines


def _first_passage_body(pal, meas, defn, body_width: float):
    """Layout of a first-passage schematic: its height and a function that draws it into an axis."""
    rows = _first_passage_rows(defn)
    xlim = _x_range(rows)
    sizer = _Canvas(None, meas, pal, xlim, body_width)
    _prepare_rows(sizer, rows)
    footer = meas.line(SMALL_PT) + 0.1
    height = sum(r.height for r in rows) + footer + 0.05

    def draw(ax) -> None:
        """Draw every row, the candidate column and the footer into ax."""
        cv = _Canvas(ax, meas, pal, xlim, body_width)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, height)
        ax.add_patch(Rectangle((0.5, 0), 1, height, facecolor=pal.wash, edgecolor="none", zorder=0))
        y = height
        for row in rows:
            anchor = row.span.first - 0.5 if not row.span.since_search_start else xlim[0]
            _draw_row(cv, row, y, anchor)
            y -= row.height
        note = "shaded: the candidate day (day 1)"
        if defn.rule.report == "candidate":
            note += ", which is the onset date reported"
        left = min(cv.inch(0.5), body_width - meas.width(note, SMALL_PT))
        cv.text(cv.x(max(left, 0.0)), 0.04, note, size=SMALL_PT, color=pal.ink_secondary)
        to_date = any(r.span.since_search_start for r in rows)
        first_day = int(np.ceil(xlim[0])) + (SEARCH_START_DAYS if to_date else 0)
        ticks = list(range(first_day, int(np.floor(xlim[1])) + 1))
        if cv.per_day < 0.2:
            ticks = [d for d in ticks if d == 1 or d % 5 == 0]
        ax.set_xticks(ticks)
        has_back = xlim[0] < 0.5 and not to_date
        ax.set_xlabel("days relative to the candidate day (day 1 is the candidate, day 0 the day before)"
                      if has_back else "days after the candidate day (day 1 is the candidate)")
        ax.set_yticks([])

    return height, draw


def _curve_body(pal, meas, defn, body_width: float):
    """Layout of a curve schematic: its height and a drawing function."""
    rule: Curve = defn.rule
    height = 2.6

    def draw(ax) -> None:
        """Draw the stylised cumulative anomaly with its minimum and the reported onset."""
        t = np.linspace(0, 1, 61)
        curve = 10 * (t - 0.42) ** 2 - 1.76 + 0.09 * np.sin(24 * t)
        curve -= curve[0]
        k = int(np.argmin(curve))
        onset = k + 1 if rule.report == "day_after_minimum" else k
        ax.plot(t, curve, color=pal.observed, linewidth=style.LINE_WIDTH, zorder=3)
        ax.axhline(0, color=pal.axis, linewidth=style.HAIRLINE, zorder=1)
        lo, hi = curve.min(), curve.max()
        pad = (hi - lo) * 0.55
        ax.set_ylim(lo - pad, hi + (hi - lo) * 0.1)
        ax.set_xlim(-0.01, 1.01)
        ax.plot([t[k]], [curve[k]], "o", color=pal.ink, markersize=style.MARKER_SIZE * 0.7, zorder=4)
        colour, icon, _ = style.status_style(Status.ONSET, pal.mode)
        ax.plot([t[onset]], [curve[onset]], "o", color=colour, markersize=style.MARKER_SIZE * 0.7, zorder=5)
        ytext = curve[k] - (hi - lo) * 0.3
        arrow = {"arrowstyle": "-", "color": pal.ink_muted, "linewidth": style.HAIRLINE, "shrinkA": 2, "shrinkB": 4}
        ax.annotate("minimum of the cumulative anomaly", xy=(t[k], curve[k]), xytext=(t[k] - 0.02, ytext),
                    ha="right", va="center", fontsize=LABEL_PT, color=pal.ink, arrowprops=arrow, annotation_clip=False)
        where = "the day after the minimum" if rule.report == "day_after_minimum" else "the day of the minimum"
        ax.annotate(f"{icon} onset = {where}", xy=(t[onset], curve[onset]), xytext=(t[onset] + 0.02, ytext),
                    ha="left", va="center", fontsize=LABEL_PT, color=pal.ink, arrowprops=arrow,
                    annotation_clip=False)
        ax.set_xticks([0])
        ax.set_xticklabels(["search start"])
        ax.set_yticks([0])
        ax.set_yticklabels(["0"])
        ax.set_xlabel("days since the start of the search window")
        ax.set_ylabel("cumulative anomaly")

    return height, draw


def _curve_header_extra(defn: Definition) -> str:
    """The sentence that says what is accumulated, for curve rules."""
    rule = defn.rule
    spec = defn.threshold_specs.get(rule.baseline)
    base = _threshold_words(spec) if spec else rule.baseline
    return (f"Baseline: {rule.baseline} = {base}. Each day adds (that day's rain - {rule.baseline}) to a running "
            "sum; the curve below is illustrative.")


def schematic(definition: Definition, *, mode: str = "light", figsize: tuple[float, float] | None = None) -> Figure:
    """A figure that explains the rule of `definition` using nothing but the definition.

    First-passage rules show a labelled band per requirement and a bracket plus example
    spell per veto over days relative to the candidate (day 1); curve rules show a
    stylised cumulative-anomaly curve. A header gives the generated sentence, the
    search window and the threshold definitions. figsize sets the width; the height
    grows as needed to fit the content.
    """
    with style.figure_style(mode) as pal:
        width = figsize[0] if figsize else WIDTH_IN
        fig = Figure(figsize=(width, 4.0))
        FigureCanvasAgg(fig)
        meas = _Measure(fig)
        inner = width - 2 * MARGIN_IN
        header = _header_lines(definition, meas, inner)
        if isinstance(definition.rule, Curve):
            body_h, draw = _curve_body(pal, meas, definition, inner)
            bottom, left = 0.7, MARGIN_IN + 0.35
        else:
            body_h, draw = _first_passage_body(pal, meas, definition, inner)
            bottom, left = 0.75, MARGIN_IN
        head_h = 0.18 + sum(meas.line(size) + gap for _, size, _, _, gap in header) + 0.12
        total = head_h + body_h + bottom
        if figsize and figsize[1] > total:
            body_h += figsize[1] - total
            total = figsize[1]
        fig.set_size_inches(width, total)
        y = total - 0.18
        for line, size, weight, role, gap in header:
            y -= gap + meas.line(size)
            fig.text(MARGIN_IN / width, y / total + 0.25 * meas.line(size) / total, line, fontsize=size,
                     fontweight=weight, color=getattr(pal, role), ha="left", va="baseline")
        ax = fig.add_axes([left / width, bottom / total, (width - MARGIN_IN - left) / width, body_h / total])
        ax.grid(False)
        ax.spines["left"].set_visible(isinstance(definition.rule, Curve))
        draw(ax)
        fig.patch.set_facecolor(pal.surface)
    return fig
