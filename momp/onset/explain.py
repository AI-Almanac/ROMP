"""Explain why one cell got the onset date it did: a structured Anatomy of the search and a readable log.

trace_observed and trace_forecast build the engine's own jobs for a single cell, run the reference backend
with trace=True on that one job, and read the decision from the engine's own rules (including the
history-only PRIOR search of stitched forecasts), so the Anatomy's result is the engine's result. The
descriptive values shown next to each outcome (window totals, wet-day counts, the first confirmed dry spell)
are recomputed from the series for display only. explain() turns an Anatomy into text; the plot lives in
momp.onset.viz.anatomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from momp.onset import engine
from momp.onset import plan as P
from momp.onset.backends import reference
from momp.onset.describe import describe
from momp.onset.model import (
    Curve, DailyRain, Definition, ForecastRain, Jobs, Status,
)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_OP_WORDS = {"ge": "at least", "gt": "more than", "le": "at most", "lt": "less than"}
_OP_NAMES = {code: name for name, code in P.OP_CODES.items()}
_OUTCOME = {1: "pass", 0: "fail", -1: "unknown"}
_VETO_OUTCOME = {1: "fail", 0: "pass", -1: "unknown"}
_STATE = {1: "T", 0: "F", -1: "U"}


@dataclass(frozen=True)
class ThresholdValue:
    """A named per-cell threshold: its value in mm and the integers the backend actually compared.

    compared maps a comparator ("ge", "gt", "le", "lt") to the amount in tenths of a mm (the threshold
    rounded for that comparator); for a curve baseline the single key "baseline" holds thousandths of a mm.
    """

    name: str
    mm: float
    compared: dict


@dataclass(frozen=True)
class Spell:
    """A confirmed dry spell: first and last day index, length in days and the rain total in mm."""

    first: int
    last: int
    length: int
    total_mm: float


@dataclass(frozen=True)
class Check:
    """One requirement or veto evaluated for one candidate.

    role is "require" or "veto" and kind "total", "wet_days", "dry_run" or "dry_window". outcome is
    "pass", "fail" or "unknown" (for a veto, "fail" means a spell was confirmed and the candidate rejected).
    first..last (day indices, inclusive) is the span the rule looks at and need_first..need_last the days
    it needs to be observable. value is the window total in mm or the wet-day count (None when not
    observable); op/limit the comparison (limit in mm, a day count or a wet-day count); limit_name the
    threshold's name; spell the confirmed spell of a failing veto; first_not_wet the first non-wet day of a
    wet-day requirement; flags the reasons an unknown outcome is unknown ("missing", "horizon");
    missing_days the missing day indices inside the observed part of the need.
    """

    role: str
    index: int
    kind: str
    outcome: str
    first: int
    last: int
    need_first: int
    need_last: int
    n_days: int
    op: str
    limit: float
    limit_name: str | None = None
    value: float | None = None
    placement: str | None = None
    spell_days: int = 0
    spell: Spell | None = None
    first_not_wet: int | None = None
    flags: tuple = ()
    missing_days: tuple = ()
    since_start: bool = False
    offset_first: int = 0


@dataclass(frozen=True)
class CandidateTrace:
    """One candidate day: its three-valued state ("T", "F", "U"), flags and every check behind them.

    decisive is the position in checks of the first failing check (state F) or first unknown check
    (state U), None for a candidate that passes everything.
    """

    day: int
    date: object
    state: str
    flags: tuple
    checks: tuple
    decisive: int | None


@dataclass(frozen=True)
class CurveTrace:
    """The cumulative anomaly of a curve rule over the search window, in mm, and where it is lowest."""

    days: np.ndarray
    cumulative_mm: np.ndarray
    min_day: int
    min_mm: float
    baseline_mm: float
    after_minimum: bool


@dataclass(frozen=True)
class Result:
    """The outcome: the reported onset day index and date (None without one), status, onset lead and decision day.

    onset_lead is the 1-based lead of the onset (0 or less when back-dated or PRIOR) for forecasts, else None.
    decision_day is the candidate that decided the search (it differs from day when the rule reports the
    first wet day).
    """

    day: int | None
    date: object
    status: Status
    onset_lead: int | None
    decision_day: int | None
    decided_by_history: bool


@dataclass
class Anatomy:
    """Everything needed to see why one cell and year (or forecast init) got its onset date.

    days are day indices of the series searched (observed time index, or for forecasts the stitched index
    with lead 1 at `split`); dates label them (datetime64[D] for standard calendars, else (y, m, d)).
    rain_mm, missing, wet and source (a "observed"/"forecast" string per day) cover the plotted day range.
    first_candidate/last_candidate bound the search window and data_end is the first day without data.
    thresholds are the named thresholds with the integers compared; candidates run in date order up to the
    decision; curve is set instead of candidates for curve rules.
    """

    definition: Definition
    sentence: str
    location: dict
    season: str
    year: int | None
    init: object
    member: int | None
    split: int
    lead_label: str | None
    stitched: bool
    days: np.ndarray
    dates: np.ndarray
    rain_mm: np.ndarray
    missing: np.ndarray
    wet: np.ndarray
    source: np.ndarray
    wet_mm: float
    wet_op: str
    first_candidate: int
    last_candidate: int
    data_end: int
    thresholds: tuple
    candidates: tuple
    curve: CurveTrace | None
    result: Result
    labels: "_Labels" = field(repr=False, default=None)

    @property
    def is_forecast(self) -> bool:
        """Whether the series is a forecast (alone or stitched onto history)."""
        return self.lead_label is not None

    def rain_mm_at(self, day: int) -> float:
        """Rain in mm on a day index inside the plotted range."""
        return float(self.rain_mm[int(day) - int(self.days[0])])

    def date_of(self, day: int):
        """The date label of any day index, including days just outside the series."""
        return self.labels.date(day)


class _Labels:
    """Calendar labels of every day of a series, extended past either end by calendar arithmetic."""

    def __init__(self, calendar: str, year, month, day):
        """Remember the labels of the series days."""
        self.calendar = calendar
        self.y, self.m, self.d = (np.asarray(a, dtype=np.int64) for a in (year, month, day))
        self.n = len(self.y)
        self._extra: dict = {}

    @property
    def standard(self) -> bool:
        """Whether dates are real datetimes (standard calendars) rather than (y, m, d) tuples."""
        return self.calendar in engine._STANDARD_CALENDARS

    def ymd(self, i: int) -> tuple[int, int, int]:
        """(year, month, day) of day index i."""
        i = int(i)
        if 0 <= i < self.n:
            return int(self.y[i]), int(self.m[i]), int(self.d[i])
        if i not in self._extra:
            base = 0 if i < 0 else self.n - 1
            y, m, d = engine._add_days(self.calendar, self.y[base:base + 1], self.m[base:base + 1],
                                       self.d[base:base + 1], np.array([i - base]))
            self._extra[i] = (int(y[0, 0]), int(m[0, 0]), int(d[0, 0]))
        return self._extra[i]

    def date(self, i: int):
        """datetime64[D] of day index i for standard calendars, else a (y, m, d) tuple."""
        y, m, d = self.ymd(i)
        if self.standard:
            return (np.datetime64(f"{y:04d}-{m:02d}", "M").astype("datetime64[D]") + np.timedelta64(d - 1, "D"))
        return (y, m, d)

    def dates(self, idx: np.ndarray) -> np.ndarray:
        """Dates of an array of day indices inside the series."""
        idx = np.asarray(idx, dtype=np.int64)
        if self.standard:
            months = ((self.y[idx] - 1970) * 12 + self.m[idx] - 1).astype("datetime64[M]").astype("datetime64[D]")
            return months + (self.d[idx] - 1).astype("timedelta64[D]")
        out = np.empty(len(idx), dtype=object)
        for k, i in enumerate(idx):
            out[k] = (int(self.y[i]), int(self.m[i]), int(self.d[i]))
        return out


def date_ymd(date) -> tuple[int, int, int]:
    """(year, month, day) of an Anatomy date (datetime64[D] or tuple)."""
    if isinstance(date, tuple):
        return date
    day = np.datetime64(date, "D")
    month = day.astype("datetime64[M]")
    return (int(month.astype(np.int64) // 12 + 1970), int(month.astype(np.int64) % 12 + 1),
            int((day - month.astype("datetime64[D]")).astype(np.int64) + 1))


def format_date(date, with_year: bool = False) -> str:
    """A date as "18 May" (or "18 May 2020")."""
    y, m, d = date_ymd(date)
    return f"{d} {_MONTHS[m - 1]}" + (f" {y}" if with_year else "")


# ---------------------------------------------------------------------------------- locating the cell

def _nearest(coords: np.ndarray, value: float, name: str) -> int:
    """Index of the grid coordinate nearest to value; an error when it is farther than half a grid step."""
    coords = np.asarray(coords, dtype=np.float64)
    i = int(np.argmin(np.abs(coords - value)))
    if len(coords) > 1:
        step = float(np.median(np.abs(np.diff(coords))))
        if abs(coords[i] - value) > 0.5 * step + 1e-9:
            raise ValueError(f"{name} {value} is {abs(coords[i] - value):g} from the nearest grid cell "
                             f"({coords[i]:g}), more than half a grid step ({step / 2:g})")
    return i


def _cell_of(spatial_dims: tuple[str, ...], coords: dict, lat, lon) -> tuple[int, ...]:
    """Index of the nearest cell along each spatial dim (empty for a series without spatial dims)."""
    wanted = {"lat": lat, "lon": lon}
    cell = []
    for dim in spatial_dims:
        if dim not in wanted or dim not in coords:
            raise ValueError(f"cannot locate a cell along dim {dim!r}; need dims named lat and lon with coordinates")
        if wanted[dim] is None:
            raise ValueError(f"{dim} is required")
        cell.append(_nearest(coords[dim], float(wanted[dim]), dim))
    return tuple(cell)


def _location(spatial_dims: tuple[str, ...], coords: dict, cell: tuple[int, ...]) -> dict:
    """The coordinates of the chosen cell, by dim name."""
    return {dim: float(np.asarray(coords[dim])[i]) for dim, i in zip(spatial_dims, cell) if dim in coords}


def _slice_coords(coords: dict, sliced: dict) -> dict:
    """Coordinates with the given dims cut to one-element slices."""
    return {k: (np.asarray(v)[sliced[k]] if k in sliced else v) for k, v in coords.items()}


def _single_rain(rain: DailyRain, cell: tuple[int, ...]) -> DailyRain:
    """The one-cell DailyRain (spatial dims kept with length 1)."""
    sl = tuple(slice(i, i + 1) for i in cell)
    sliced = dict(zip(rain.dims[:-1], sl))
    return DailyRain(rain.tenths[sl], rain.missing[sl], rain.dims, _slice_coords(rain.coords, sliced), rain.year,
                     rain.month, rain.day, rain.calendar, rain.day_boundary, rain.source)


def _single_forecast(fc: ForecastRain, init: int, member: int | None, cell: tuple[int, ...]) -> ForecastRain:
    """The one-init, one-member, one-cell ForecastRain (dims kept with length 1)."""
    sliced = {"init_time": slice(init, init + 1)}
    if "member" in fc.dims:
        sliced["member"] = slice(member, member + 1)
    for dim, i in zip(fc.dims[1 + ("member" in fc.dims):-1], cell):
        sliced[dim] = slice(i, i + 1)
    sl = tuple(sliced.get(dim, slice(None)) for dim in fc.dims)
    sl_init = sliced["init_time"]
    return ForecastRain(fc.tenths[sl], fc.missing[sl], fc.dims,
                        _slice_coords(fc.coords, {k: v for k, v in sliced.items() if k in fc.coords}),
                        fc.init_year[sl_init], fc.init_month[sl_init], fc.init_day[sl_init], fc.lead_label,
                        fc.calendar, fc.day_boundary, fc.source)


def _cell_thresholds(plan: P.Plan, thresholds, spatial_dims, spatial_shape, cell) -> dict[str, np.ndarray]:
    """One-element float arrays of the thresholds the plan needs at this cell, via the engine's own broadcasting."""
    per_cell = engine._series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape)
    flat = int(np.ravel_multi_index(cell, spatial_shape)) if spatial_shape else 0
    return {name: values[flat:flat + 1].copy() for name, values in per_cell.items()}


def _threshold_values(plan: P.Plan, per_cell: dict, amounts: dict) -> tuple[ThresholdValue, ...]:
    """The named thresholds with the integer amounts compared for each comparator."""
    found: dict[str, dict] = {}
    for spec in plan.amount_specs:
        if spec.kind == "fixed":
            continue
        entry = found.setdefault(spec.name, {})
        entry["baseline" if spec.kind == "baseline" else _OP_NAMES[spec.op]] = int(amounts[spec.key][0])
    return tuple(ThresholdValue(name, float(per_cell[name][0]), compared) for name, compared in found.items())


# ---------------------------------------------------------------------------------- the displayed series

def _prefix(values: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading zero."""
    return np.concatenate([[0], np.cumsum(values, dtype=np.int64)])


def _compare(op: int, value, amount):
    """Elementwise comparison selected by an operator code."""
    return reference._compare(op, value, amount)


class _Series:
    """Prefix tables over the one series being explained (rain, missing, wet and dry days)."""

    def __init__(self, tenths: np.ndarray, missing: np.ndarray, wet_op: int, wet_amount: int):
        """Build the tables; missing days count as no rain and are neither wet nor dry."""
        self.miss = np.asarray(missing, dtype=bool)
        self.x = np.where(self.miss, 0, tenths).astype(np.int64)
        self.n = len(self.x)
        self.wet = ~self.miss & _compare(wet_op, self.x, wet_amount)
        self.dry = ~self.miss & ~self.wet
        self.px, self.pm, self.pw, self.pd = (_prefix(a) for a in (self.x, self.miss, self.wet, self.dry))

    def observe(self, a: int, b: int, end: int) -> tuple[tuple, tuple]:
        """(flags, missing day indices) of the span [a, b) against data ending at `end` (the backend's rule)."""
        lo = min(max(a, 0), end)
        hi = max(min(b, end), lo)
        n_missing = int(self.pm[hi] - self.pm[lo])
        flags = []
        if a < 0 or n_missing > 0:
            flags.append("missing")
        if b > end:
            flags.append("horizon")
        days = tuple(int(i) for i in np.nonzero(self.miss[lo:hi])[0] + lo) if n_missing else ()
        return tuple(flags), days


def _check_requirement(cp: P.CondPlan, i: int, code: int, d: int, s: _Series, amounts: dict, start: int,
                       end: int) -> Check:
    """Describe requirement i of candidate d, with the values behind its outcome."""
    a = start if cp.since_start else d + cp.first
    b = d + cp.last
    flags, missing_days = s.observe(a, b, end)
    observable = not flags
    if cp.kind == P.TOTAL:
        value = (s.px[b] - s.px[a]) / 10 if observable else None
        key = cp.cmp.key
        name = None if key.startswith("fixed:") else key.rsplit(":", 1)[0]
        return Check("require", i, "total", _OUTCOME[code], a, b - 1, a, b - 1, b - a, _OP_NAMES[cp.cmp.op],
                     float(amounts[key][0]) / 10, name, value, None, 0, None, None,
                     () if observable else flags, missing_days, cp.since_start, cp.first)
    value = float(s.pw[b] - s.pw[a]) if observable else None
    first_not_wet = None
    if observable and code == reference.FALSE:
        gaps = np.nonzero(~s.wet[a:b])[0]
        first_not_wet = int(gaps[0]) + a if len(gaps) else None
    return Check("require", i, "wet_days", _OUTCOME[code], a, b - 1, a, b - 1, b - a, _OP_NAMES[cp.op],
                 float(cp.count), None, value, None, 0, None, first_not_wet, () if observable else flags,
                 missing_days, cp.since_start, cp.first)


def _find_spell(vp: P.VetoPlan, a: int, b: int, s: _Series, amounts: dict, end: int) -> Spell | None:
    """The first dry spell the veto confirms for the span [a, b), or None (mirrors the backend's rule)."""
    length = vp.length
    if vp.placement == P.INSIDE:
        if vp.last - vp.first < length:
            return None
        last_start = b - length
    else:
        last_start = b - 1
    last_start = min(last_start, end - length)
    first_start = max(a, 0)
    if last_start < first_start:
        return None
    starts = np.arange(first_start, last_start + 1)
    if vp.kind == P.DRY_RUN:
        holds = (s.pd[starts + length] - s.pd[starts]) == length
    else:
        total = s.px[starts + length] - s.px[starts]
        complete = (s.pm[starts + length] - s.pm[starts]) == 0
        holds = complete & _compare(vp.cmp.op, total, amounts[vp.cmp.key])
    if not holds.any():
        return None
    u = int(starts[int(np.argmax(holds))])
    last = u + length - 1
    if vp.kind == P.DRY_RUN:
        limit = min(b if vp.placement == P.INSIDE else end, end) - 1
        while last + 1 <= limit and s.dry[last + 1]:
            last += 1
    return Spell(u, last, last - u + 1, float(s.px[last + 1] - s.px[u]) / 10)


def _check_veto(vp: P.VetoPlan, i: int, code: int, d: int, s: _Series, amounts: dict, end: int) -> Check:
    """Describe veto i of candidate d, with the confirmed dry spell when it rejects the candidate."""
    a, b = d + vp.first, d + vp.last
    inside = vp.placement == P.INSIDE
    need_last = b - 1 if inside else b + vp.length - 2
    if inside and vp.last - vp.first < vp.length:
        flags, missing_days = (), ()
    else:
        flags, missing_days = s.observe(a, need_last + 1, end)
    spell = _find_spell(vp, a, b, s, amounts, end) if code == reference.TRUE else None
    kind = "dry_run" if vp.kind == P.DRY_RUN else "dry_window"
    op, limit, name = "ge", float(vp.length), None
    if vp.kind == P.DRY_WINDOW:
        op, limit = _OP_NAMES[vp.cmp.op], float(amounts[vp.cmp.key][0]) / 10
        name = None if vp.cmp.key.startswith("fixed:") else vp.cmp.key.rsplit(":", 1)[0]
    return Check("veto", i, kind, _VETO_OUTCOME[code], a, b - 1, a, need_last, b - a, op, limit, name, None,
                 "inside" if inside else "starts_inside", vp.length, spell,
                 None, flags if code == reference.UNKNOWN else (), missing_days if code == reference.UNKNOWN else (),
                 False, vp.first)


# ---------------------------------------------------------------------------------- building the Anatomy

def _flag_names(bits: int) -> tuple:
    """Names of the backend's flag bits."""
    return tuple(name for bit, name in ((reference.MISSING_FLAG, "missing"), (reference.HORIZON_FLAG, "horizon"))
                 if bits & bit)


def _season_index(names: list[str], season: str | None) -> int:
    """Position of the requested season (the first when none is given)."""
    if season is None:
        return 0
    if season not in names:
        raise ValueError(f"unknown season {season!r}; this definition has {names}")
    return names.index(season)


def _first_passage_candidates(plan, defn_trace, amounts, s, labels, start, end, data_end) -> tuple:
    """CandidateTraces up to and including the first candidate that is not False."""
    tr = defn_trace
    out = []
    for c in range(tr["day"].shape[1]):
        if not tr["valid"][0, c]:
            continue
        d = int(tr["day"][0, c])
        state = int(tr["state"][0, c])
        checks = [_check_requirement(cp, i, int(tr["requirements"][i][0, c]), d, s, amounts, start, data_end)
                  for i, cp in enumerate(plan.conditions)]
        checks += [_check_veto(vp, i, int(tr["vetoes"][i][0, c]), d, s, amounts, data_end)
                   for i, vp in enumerate(plan.vetoes)]
        decisive = None
        for k, check in enumerate(checks):
            if check.outcome == ("fail" if state == reference.FALSE else "unknown") and decisive is None:
                decisive = k
        out.append(CandidateTrace(d, labels.date(d), _STATE[state], _flag_names(int(tr["flags"][0, c])),
                                  tuple(checks), decisive))
        if state != reference.FALSE:
            break
    return tuple(out)


def _curve_trace(plan, trace, amounts) -> CurveTrace | None:
    """The cumulative anomaly of a curve job in mm, with its minimum."""
    valid = trace["valid"][0]
    if not valid.any():
        return None
    days = trace["day"][0][valid]
    cumulative = trace["cumulative"][0][valid] / 1000.0
    k = int(np.argmin(cumulative))
    return CurveTrace(days, cumulative, int(days[k]), float(cumulative[k]), amounts[plan.curve_key][0] / 1000.0,
                      plan.curve_after_min)


def _plan_reach(plan: P.Plan) -> int:
    """How many days past a candidate its spans can reach."""
    reach = max([c.last for c in plan.conditions] + [v.last + (v.length - 1 if v.placement == P.STARTS_INSIDE else 0)
                                                    for v in plan.vetoes] + [1])
    if plan.report_span is not None:
        reach = max(reach, plan.report_span[1])
    return reach


@dataclass
class _Case:
    """Everything the shared builder needs about one search: the series and where the job sits in it."""

    tenths: np.ndarray
    missing: np.ndarray
    labels: _Labels
    source: np.ndarray
    split: int
    start: int
    end: int
    data_end: int
    preset: int
    run: object


def _build(defn: Definition, plan: P.Plan, case: _Case, per_cell: dict, amounts: dict, invalid: bool, *,
           location: dict, season: str, year, init, member, lead_label, stitched: bool) -> Anatomy:
    """Run the traced search for one prepared case and assemble the Anatomy."""
    wet_key = plan.wet.key
    wet_amount = int(amounts[wet_key][0])
    s = _Series(case.tenths, case.missing, plan.wet.op, wet_amount)
    status, day, trace, data_end = Status.NOT_FOUND, -1, None, case.data_end
    if case.preset >= 0:
        status = Status(case.preset)
    elif invalid:
        status = Status.INSUFFICIENT_DATA
    else:
        status, day, trace, data_end = case.run(s)
    candidates, curve = (), None
    if trace is not None:
        if plan.family == P.CURVE:
            curve = _curve_trace(plan, trace, amounts)
        else:
            candidates = _first_passage_candidates(plan, trace, amounts, s, case.labels, case.start, case.end, data_end)
    decision = next((c.day for c in candidates if c.state != "F"), None)
    if curve is not None and status in (Status.ONSET,):
        decision = curve.min_day
    lo = max(case.start, 0) if case.end >= case.start else 0
    if candidates:
        hi = candidates[-1].day + _plan_reach(plan) + 2
    elif curve is not None:
        hi = int(curve.days[-1]) + 3
    else:
        hi = lo + 30
    if day >= 0:
        hi = max(hi, day + 3)
    if lead_label is not None:
        hi = max(hi, case.split + 3)
    hi = min(max(hi, lo + 1), s.n)
    lo = min(lo, hi - 1)
    days = np.arange(lo, hi)
    found = day >= 0
    onset_lead = None
    if lead_label is not None and found:
        onset_lead = int(day - case.split + 1)
    result = Result(int(day) if found else None, case.labels.date(day) if found else None, status, onset_lead,
                    decision, status == Status.PRIOR)
    return Anatomy(
        definition=defn, sentence=describe(defn), location=location, season=season, year=year, init=init,
        member=member, split=case.split, lead_label=lead_label, stitched=stitched, days=days,
        dates=case.labels.dates(days), rain_mm=s.x[days] / 10.0, missing=s.miss[days], wet=s.wet[days],
        source=case.source[days], wet_mm=wet_amount / 10.0, wet_op=_OP_NAMES[plan.wet.op],
        first_candidate=int(case.start), last_candidate=int(case.end), data_end=int(data_end),
        thresholds=_threshold_values(plan, per_cell, amounts), candidates=candidates, curve=curve, result=result,
        labels=case.labels)


def _detect_single(plan: P.Plan, amounts: dict, start: int, end: int, data_end: int):
    """A run callback: the reference backend, traced, on one job of the displayed series."""
    def run(s: _Series):
        """Return (status, day, trace, data_end) of the job."""
        det = reference.detect(plan, s.x[None, :], s.miss[None, :], amounts,
                               Jobs(np.zeros(1, np.int64), np.array([start]), np.array([end]), np.array([data_end])),
                               trace=True)
        return Status(int(det.status[0])), int(det.day[0]), det.trace, data_end
    return run


def trace_observed(defn: Definition, rain: DailyRain, *, lat: float | None, lon: float | None, year: int,
                   season: str | None = None, thresholds=None,
                   season_climatology: xr.Dataset | None = None) -> Anatomy:
    """Trace the onset search of the cell nearest (lat, lon) in the season of `year`.

    The cell must lie within half a grid step of (lat, lon). The jobs are the engine's own
    (build_observed_jobs on the one-cell series) and the Anatomy's result equals detect_observed's.
    thresholds are as for detect_observed; season picks a season by name (the first by default).
    """
    plan = P.compile(defn)
    spatial_dims, spatial_shape = rain.dims[:-1], rain.tenths.shape[:-1]
    cell = _cell_of(spatial_dims, rain.coords, lat, lon)
    rain1 = _single_rain(rain, cell)
    clim1 = None
    if season_climatology is not None:
        clim1 = season_climatology.isel({d: slice(i, i + 1) for d, i in zip(spatial_dims, cell)})
    jobs, layout = engine.build_observed_jobs(defn, rain1, [year], clim1)
    k = _season_index(layout.season_names, season)
    per_cell = _cell_thresholds(plan, thresholds, spatial_dims, spatial_shape, cell)
    amounts, invalid = P.resolve_amounts(plan, per_cell, n_series=1)
    start, end, data_end = int(jobs.start[k]), int(jobs.end[k]), int(jobs.data_end[k])
    row = (*cell, slice(None))
    labels = _Labels(rain.calendar, rain.year, rain.month, rain.day)
    case = _Case(rain.tenths[row], rain.missing[row], labels, np.full(len(labels.y), "observed"), 0, start, end,
                 data_end, int(layout.preset[k]), _detect_single(plan, amounts, start, end, data_end))
    return _build(defn, plan, case, per_cell, amounts, bool(invalid[0]),
                  location=_location(spatial_dims, rain.coords, cell), season=layout.season_names[k], year=year,
                  init=None, member=None, lead_label=None, stitched=False)


def _find_init(forecast: ForecastRain, init) -> int:
    """Index of the init whose date label equals `init` (a date, datetime64, string or (y, m, d))."""
    if isinstance(init, tuple):
        ymd = tuple(int(v) for v in init)
    elif hasattr(init, "year") and hasattr(init, "month") and not isinstance(init, (np.datetime64, str)):
        ymd = (int(init.year), int(init.month), int(init.day))
    else:
        ymd = date_ymd(np.datetime64(init, "D"))
    hits = np.nonzero((forecast.init_year == ymd[0]) & (forecast.init_month == ymd[1]) & (forecast.init_day == ymd[2]))[0]
    if len(hits) == 0:
        raise ValueError(f"the forecast has no init on {ymd[0]:04d}-{ymd[1]:02d}-{ymd[2]:02d}")
    return int(hits[0])


def _find_member(forecast: ForecastRain, member) -> int | None:
    """Index of the requested member (by coordinate value, else by position)."""
    if "member" not in forecast.dims:
        if member is not None:
            raise ValueError("the forecast has no member dim")
        return None
    if member is None:
        raise ValueError("the forecast has members; pass member=")
    values = np.asarray(forecast.coords["member"]) if "member" in forecast.coords else np.arange(
        forecast.tenths.shape[1])
    hits = np.nonzero(values == member)[0]
    if len(hits):
        return int(hits[0])
    if isinstance(member, (int, np.integer)) and 0 <= member < len(values):
        return int(member)
    raise ValueError(f"no member {member!r}")


def trace_forecast(defn: Definition, forecast: ForecastRain, *, init, lat: float | None, lon: float | None,
                   member=None, history: DailyRain | None = None, thresholds=None,
                   lead_window: tuple[int, int] = (1, 30), season: str | None = None,
                   restrict_to_season: bool = True) -> Anatomy:
    """Trace the onset search of one init (and member) at the cell nearest (lat, lon).

    With `history` the search is stitched (observed history before lead 1, then the forecast, with the
    engine's history-only PRIOR decision first); without it the forecast is searched alone. The jobs are
    the engine's own (build_forecast_jobs / build_stitched_jobs on the one-cell series), so the result
    equals detect_forecast's. Days are indexed in the stitched series (lead 1 is day `split`) or, without
    history, by lead - 1.
    """
    plan = P.compile(defn)
    series_dims = forecast.dims[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims = series_dims[n_lead_dims:]
    spatial_shape = forecast.tenths.shape[n_lead_dims:-1]
    cell = _cell_of(spatial_dims, forecast.coords, lat, lon)
    ii = _find_init(forecast, init)
    mi = _find_member(forecast, member)
    fc1 = _single_forecast(forecast, ii, mi, cell)
    per_cell = _cell_thresholds(plan, thresholds, spatial_dims, spatial_shape, cell)
    amounts, invalid = P.resolve_amounts(plan, per_cell, n_series=1)
    n_leads = forecast.tenths.shape[-1]
    fc_row = (ii, *(() if mi is None else (mi,)), *cell, slice(None))
    fc_tenths, fc_missing = forecast.tenths[fc_row], forecast.missing[fc_row]
    fy, fm, fd = (a[0] for a in engine._label_table(fc1, n_leads))
    init_date = _Labels(forecast.calendar, [forecast.init_year[ii]], [forecast.init_month[ii]],
                        [forecast.init_day[ii]]).date(0)
    common = dict(location=_location(spatial_dims, forecast.coords, cell), year=None, init=init_date, member=member,
                  lead_label=forecast.lead_label)
    if history is None:
        jobs, layout = engine.build_forecast_jobs(defn, fc1, lead_window, restrict_to_season)
        k = _season_index(layout.season_names, season)
        start, end, data_end = int(jobs.start[k]), int(jobs.end[k]), int(jobs.data_end[k])
        labels = _Labels(forecast.calendar, fy, fm, fd)
        case = _Case(fc_tenths, fc_missing, labels, np.full(n_leads, "forecast"), 0, start, end, data_end,
                     int(layout.preset[k]), _detect_single(plan, amounts, start, end, data_end))
        return _build(defn, plan, case, per_cell, amounts, bool(invalid[0]), season=layout.season_names[k],
                      stitched=False, **common)
    hist1 = _single_rain(history, cell)
    jobs, layout = engine.build_stitched_jobs(defn, hist1, fc1, lead_window, restrict_to_season)
    k = _season_index(layout.season_names, season)
    split, start, end, data_end = (int(a[k]) for a in (jobs.split, jobs.start, jobs.end, jobs.data_end))
    hist_row = (*cell, slice(None))
    labels = _Labels(forecast.calendar, np.concatenate([history.year[:split], fy]),
                     np.concatenate([history.month[:split], fm]), np.concatenate([history.day[:split], fd]))
    tenths = np.concatenate([history.tenths[hist_row][:split], fc_tenths])
    missing = np.concatenate([history.missing[hist_row][:split], fc_missing])
    source = np.array(["observed"] * split + ["forecast"] * n_leads)
    preset = int(layout.preset[k])
    run = _stitched_run(plan, amounts, start, end, data_end, split, preset, bool(invalid[0]))
    case = _Case(tenths, missing, labels, source, split, start, end, data_end, preset, run)
    return _build(defn, plan, case, per_cell, amounts, bool(invalid[0]), season=layout.season_names[k],
                  stitched=True, **common)


def _stitched_run(plan: P.Plan, amounts: dict, start: int, end: int, data_end: int, split: int, preset: int,
                  invalid: bool):
    """A run callback for a stitched job: the history-only PRIOR search first (first-passage rules), then the full search."""
    def run(s: _Series):
        """Return (status, day, trace, data_end) as the engine decides it."""
        if plan.family == P.FIRST_PASSAGE:
            prior_end = min(end, split - 1)
            if prior_end >= start:
                det = reference.detect(plan, s.x[None, :split], s.miss[None, :split], amounts,
                                       Jobs(np.zeros(1, np.int64), np.array([start]), np.array([prior_end]),
                                            np.array([split])), trace=True)
                if int(det.status[0]) == Status.ONSET:
                    return Status.PRIOR, int(det.day[0]), det.trace, split
        det = reference.detect(plan, s.x[None, :], s.miss[None, :], amounts,
                               Jobs(np.zeros(1, np.int64), np.array([start]), np.array([end]), np.array([data_end])),
                               trace=True)
        return Status(int(det.status[0])), int(det.day[0]), det.trace, data_end
    return run


# ---------------------------------------------------------------------------------- the readable log

def _span_text(a: Anatomy, first: int, last: int, with_year: bool = False) -> str:
    """Day indices first..last as "2-22 Jul", "24 May - 13 Jun" or "5 Jul"."""
    f, l = date_ymd(a.date_of(first)), date_ymd(a.date_of(last))
    if first == last:
        return format_date(a.date_of(first), with_year)
    if f[:2] == l[:2]:
        return f"{f[2]}-{l[2]} {_MONTHS[f[1] - 1]}" + (f" {f[0]}" if with_year else "")
    return f"{format_date(a.date_of(first), with_year and f[0] != l[0])} - {format_date(a.date_of(last), with_year)}"


def _amount_text(check: Check) -> str:
    """The limit of a check as "20 mm" (with the threshold's name when it has one)."""
    text = f"{check.limit:g} mm"
    return f"{text} ({check.limit_name})" if check.limit_name else text


def _total_label(check: Check) -> str:
    """What a total requirement measures, e.g. "3-day total"."""
    if check.since_start:
        return "total since the search start"
    if check.offset_first == 0:
        return f"{check.n_days}-day total"
    return f"{check.n_days}-day total over days {check.offset_first + 1}-{check.offset_first + check.n_days}"


def _all_wet(check: Check) -> bool:
    """Whether a wet-days requirement asks for every day of its span to be wet."""
    return check.kind == "wet_days" and check.op == "ge" and check.limit == check.n_days


def _unknown_text(a: Anatomy, check: Check) -> str:
    """Why a check cannot be decided: what it needs and which part is missing or past the data."""
    need = _span_text(a, check.need_first, check.need_last)
    parts = []
    if "horizon" in check.flags:
        noun = "last lead" if a.is_forecast else "last day"
        parts.append(f"past the {'forecast' if a.is_forecast else 'data'} ({noun} {format_date(a.date_of(a.data_end - 1))})")
    if "missing" in check.flags:
        if check.need_first < 0:
            parts.append("before the first day of data")
        if check.missing_days:
            shown = ", ".join(format_date(a.date_of(i)) for i in check.missing_days[:3])
            more = f" and {len(check.missing_days) - 3} more" if len(check.missing_days) > 3 else ""
            parts.append(f"{len(check.missing_days)} missing day{'s' if len(check.missing_days) > 1 else ''} "
                         f"({shown}{more})")
    return f"needs {need}, " + ", ".join(parts)


def _veto_phrase(check: Check) -> str:
    """The spell a veto looks for, e.g. "a dry run of at least 7 days"."""
    if check.kind == "dry_run":
        return f"dry run of at least {check.spell_days} days"
    return f"{check.spell_days}-day window totalling {_OP_WORDS[check.op]} {_amount_text(check)}"


def _reason(a: Anatomy, check: Check, wet_text: str) -> str:
    """The reason one check gave its outcome, in words with the numbers."""
    if check.outcome == "unknown":
        return _unknown_text(a, check)
    if check.role == "require":
        if check.kind == "total":
            if check.outcome == "fail":
                return f"{_total_label(check)} {check.value:.1f} mm, needs {_OP_WORDS[check.op]} {_amount_text(check)}"
            return f"{_total_label(check)} {check.value:.1f} mm ({_OP_WORDS[check.op]} {_amount_text(check)})"
        if _all_wet(check):
            if check.outcome == "fail":
                day = check.first_not_wet - check.first + 1
                mm = a.rain_mm_at(check.first_not_wet)
                return f"day {day} not wet ({mm:.1f} mm, wet needs {wet_text})"
            return "wet day" if check.n_days == 1 else f"{check.n_days} consecutive wet days"
        return f"{check.value:.0f} wet days of {check.n_days}, needs {_OP_WORDS[check.op]} {check.limit:.0f}"
    where = _span_text(a, check.first, check.last)
    if check.outcome == "fail":
        spell = check.spell
        what = (f"a dry run of {spell.length} days" if check.kind == "dry_run"
                else f"a {check.spell_days}-day window totalling {spell.total_mm:.1f} mm")
        starts = "" if check.placement == "inside" else " that starts"
        return f"followed by {what} ({_span_text(a, spell.first, spell.last)}){starts} within {where}"
    return f"no {_veto_phrase(check)} within {where}"


def _group_key(c: CandidateTrace):
    """What makes two failing candidates fail 'for the same reason' (so they can collapse into one line)."""
    check = c.checks[c.decisive]
    if check.role == "veto":
        spell = check.spell
        return ("veto", check.index, spell.first, spell.last)
    return ("require", check.index, check.first_not_wet - check.first if _all_wet(check) and check.first_not_wet is not None
            else None)


def _group_reason(a: Anatomy, group: list[CandidateTrace], wet_text: str) -> str:
    """The shared reason of consecutive candidates failing the same check, with value ranges."""
    checks = [c.checks[c.decisive] for c in group]
    first = checks[0]
    if first.role == "veto":
        spell = first.spell
        what = (f"a dry run of {spell.length} days" if first.kind == "dry_run"
                else f"a {first.spell_days}-day window totalling {spell.total_mm:.1f} mm")
        starts = "" if first.placement == "inside" else " that starts within the following days"
        return f"followed by {what} ({_span_text(a, spell.first, spell.last)}){starts}"
    if first.kind == "total":
        values = [c.value for c in checks]
        lo, hi = min(values), max(values)
        shown = f"{lo:.1f} mm" if lo == hi else f"{lo:.1f}-{hi:.1f} mm"
        return f"{_total_label(first)} {shown}, needs {_OP_WORDS[first.op]} {_amount_text(first)}"
    if _all_wet(first):
        return f"day {first.first_not_wet - first.first + 1} not wet"
    values = [c.value for c in checks]
    lo, hi = min(values), max(values)
    shown = f"{lo:.0f}" if lo == hi else f"{lo:.0f}-{hi:.0f}"
    return f"{shown} wet days of {first.n_days}, needs {_OP_WORDS[first.op]} {first.limit:.0f}"


def _wet_text(a: Anatomy) -> str:
    """The wet-day comparison as "at least 1 mm"."""
    return f"{_OP_WORDS[a.wet_op]} {a.wet_mm:g} mm"


def _candidate_lines(a: Anatomy, compress: bool) -> list[tuple[str, str]]:
    """(date column, text) per line of the candidate list, in date order."""
    wet_text = _wet_text(a)
    years = {date_ymd(c.date)[0] for c in a.candidates}
    with_year = len(years) > 1
    rows: list[tuple[str, str]] = []
    i, cands = 0, a.candidates
    while i < len(cands):
        c = cands[i]
        if c.state == "F":
            j = i
            if compress:
                key = _group_key(c)
                while j + 1 < len(cands) and cands[j + 1].state == "F" and cands[j + 1].day == cands[j].day + 1 \
                        and _group_key(cands[j + 1]) == key:
                    j += 1
            if j > i:
                group = list(cands[i:j + 1])
                head = _span_text(a, c.day, cands[j].day, with_year)
                rows.append((head, f"✗ {len(group)} days: {_group_reason(a, group, wet_text)}"))
            else:
                rows.append((format_date(c.date, with_year), f"✗ {_reason(a, c.checks[c.decisive], wet_text)}"))
            i = j + 1
            continue
        if c.state == "U":
            reasons = [_reason(a, ch, wet_text) for ch in c.checks if ch.outcome == "unknown"]
            rows.append((format_date(c.date, with_year), f"? {'; '.join(reasons)}"))
        else:
            passed = "; ".join(_reason(a, ch, wet_text) for ch in c.checks)
            rows.append((format_date(c.date, with_year), f"✓ {passed}"))
        i += 1
    return rows


def _threshold_lines(a: Anatomy) -> list[str]:
    """One line per named threshold with the integers actually compared."""
    lines = []
    for t in a.thresholds:
        if "baseline" in t.compared:
            lines.append(f"  {t.name} = {t.mm:.4g} mm/day (compared as {t.compared['baseline']} thousandths of a mm)")
            continue
        used = ", ".join(f"{_OP_WORDS[op]} {v / 10:g} mm ({v} tenths)" for op, v in t.compared.items())
        lines.append(f"  {t.name} = {t.mm:.4g} mm (compared as {used})")
    return lines


def _result_lines(a: Anatomy) -> list[str]:
    """The closing lines: the status in words, the onset date and (forecasts) the lead and what decided it."""
    r = a.result
    when = format_date(r.date, True) if r.date is not None else None
    lines = []
    if r.status == Status.ONSET:
        lines.append(f"Result: ONSET on {when}")
    elif r.status == Status.PRIOR:
        lines.append(f"Result: PRIOR - the onset was {when}, before the forecast starts")
    elif r.status == Status.NOT_FOUND:
        lines.append("Result: NOT_FOUND - no candidate in the search window met every requirement")
    elif r.status == Status.CENSORED:
        noun = "forecast" if a.is_forecast else "data"
        lines.append(f"Result: CENSORED - the first undecided candidate needs days past the end of the {noun}")
    else:
        lines.append("Result: INSUFFICIENT_DATA - " + _insufficient_reason(a))
    if a.curve is None and r.decision_day is not None and r.day is not None and r.decision_day != r.day:
        lines.append(f"  decided at candidate {format_date(a.date_of(r.decision_day))}; reported on the first wet day "
                     f"{format_date(r.date)}")
    if a.is_forecast and r.onset_lead is not None:
        lead = f"  onset lead {r.onset_lead}"
        if r.onset_lead <= 0:
            lead += " (before lead 1, so the observed history decided it)" if r.decided_by_history else \
                " (back-dated into the observed history)"
        lines.append(lead)
    if a.is_forecast and a.stitched and r.status != Status.PRIOR:
        lines.append("  observed history up to the init was stitched in before the forecast")
    return lines


def _insufficient_reason(a: Anatomy) -> str:
    """Why a search ended INSUFFICIENT_DATA: a missing threshold, a missing season start, or missing days."""
    if any(not np.isfinite(t.mm) for t in a.thresholds):
        return "a threshold is not available at this cell"
    if not a.candidates and a.curve is None:
        return "the season starts outside the data"
    return "days the first undecided candidate needs are missing"


def _curve_lines(a: Anatomy) -> list[str]:
    """The curve rule's explanation: accumulation, minimum and where the onset falls."""
    c = a.curve
    if c is None:
        return ["No search ran for this cell."]
    first, last = format_date(a.date_of(int(c.days[0]))), format_date(a.date_of(int(c.days[-1])))
    lines = [f"Cumulative anomaly (rain minus {c.baseline_mm:.3g} mm/day) accumulated {first} - {last}.",
             f"Lowest value {c.min_mm:.1f} mm on {format_date(a.date_of(c.min_day))}."]
    if c.min_day == int(c.days[-1]):
        lines.append("The minimum is the last day of the window, so no onset is found.")
    else:
        word = "the day after" if c.after_minimum else "the day of"
        lines.append(f"The onset is {word} the minimum.")
    return lines


def explain(a: Anatomy, *, compress: bool = True) -> str:
    """A readable log of the search: header, every candidate with its reason, and the result.

    With compress=True, consecutive candidates that fail for the same reason become one line.
    """
    d = a.definition
    place = ", ".join(f"{k} {v:g}" for k, v in a.location.items())
    lines = [f"{d.title} ({d.id} v{d.version})", f"Place: {place}" if place else "Place: (no spatial dims)"]
    if a.is_forecast:
        extra = f", member {a.member}" if a.member is not None else ""
        history = "observed history stitched in before lead 1" if a.stitched else "forecast searched alone"
        lines.append(f"Forecast: init {format_date(a.init, True)}{extra}, lead label {a.lead_label}, {history}")
    else:
        lines.append(f"Season: {a.season} {a.year}")
    if a.is_forecast and a.season:
        lines[-1] += f", season {a.season}"
    lines.append(f"Rule: {a.sentence}")
    lines.append(f"Wet day: {_wet_text(a)}")
    if a.thresholds:
        lines.append("Thresholds:")
        lines += _threshold_lines(a)
    lines.append("")
    if a.curve is not None or isinstance(d.rule, Curve):
        lines += _curve_lines(a)
    elif a.candidates:
        window = _span_text(a, a.first_candidate, a.last_candidate, True)
        lines.append(f"Candidates (search window {window}; {len(a.candidates)} evaluated before the decision):")
        rows = _candidate_lines(a, compress)
        width = max(len(r[0]) for r in rows)
        lines += [f"  {head:<{width}}  {text}" for head, text in rows]
    else:
        lines.append("No candidates were evaluated.")
    lines.append("")
    lines += _result_lines(a)
    return "\n".join(lines)
