"""Explain why one cell got the onset date it did: a structured `~hcwf.onset.explain.Anatomy` of the search and a readable log.

`trace_observed` and `trace_forecast` build the engine's own jobs for a single cell and evaluate them with the
reference backend (`hcwf.onset.backends.reference.evaluate`), so the `Anatomy`'s result is the engine's result
(including the history-only ``PRIOR`` search of stitched forecasts) and every fact shown next to an outcome (window
totals, wet-day counts, spans, missing days, the confirmed dry spell) is the backend's own; nothing of the rule is
re-evaluated here. The run options (exact rain, missing and horizon policies, ``dry_day``, beginning within) are the
backend's. `explain` turns an `Anatomy` into text; the plot lives in `hcwf.onset.viz.anatomy`, and
`hcwf.onset.explain_cells` writes both for the cells of a run configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from hcwf.onset import engine
from hcwf.onset import plan as P
from hcwf.onset.backends import reference
from hcwf.onset.describe import describe
from hcwf.onset.model import (
    DEFAULT_HORIZON, Curve, DailyRain, Definition, DerivedSeasons, ForecastRain, Jobs, Status, resolution_of,
)
from hcwf.onset.units import MONTH_NAMES as _MONTHS

_OP_WORDS = {"ge": "at least", "gt": "more than", "le": "at most", "lt": "less than"}
_OP_NAMES = {code: name for name, code in P.OP_CODES.items()}
_PLACEMENTS = {P.INSIDE: "inside", P.STARTS_INSIDE: "starts_inside", P.BEGINS: "begins"}
_PLACEMENT_VERBS = {"starts_inside": "starts", "begins": "begins"}
_OUTCOME = {1: "pass", 0: "fail", -1: "unknown"}
_VETO_OUTCOME = {1: "fail", 0: "pass", -1: "unknown"}
_STATE = {1: "T", 0: "F", -1: "U"}


@dataclass(frozen=True)
class ThresholdValue:
    """A named per-cell threshold: its value in mm and the integers the backend actually compared."""

    name: str
    """The threshold's name."""
    mm: float
    """The threshold's value at the cell, in mm."""
    compared: dict
    """Maps a comparator (``"ge"``, ``"gt"``, ``"le"``, ``"lt"``) to the amount in tenths of a mm (the threshold
    rounded for that comparator); for a curve baseline the single key ``"baseline"`` holds thousandths of a mm. With
    exact rain it holds the unrounded mm and `exact` is true."""
    exact: bool = False
    """Whether the rain is exact (so `compared` holds mm, not integers)."""


@dataclass(frozen=True)
class Spell:
    """A confirmed dry spell: first and last day index, length in days and the rain total in mm."""

    first: int
    """Day index of the first day of the spell."""
    last: int
    """Day index of the last day of the spell (inclusive)."""
    length: int
    """Length in days."""
    total_mm: float
    """Rain total over the spell, in mm."""


@dataclass(frozen=True)
class Check:
    """One requirement or veto evaluated for one candidate.

    ``first..last`` (day indices, inclusive) is the span the rule looks at and ``need_first..need_last`` the days it
    needs to be observable.
    """

    role: str
    """``"require"`` or ``"veto"``."""
    index: int
    """Position of the requirement or veto in the plan (-1 for ``"day_missing"``)."""
    kind: str
    """``"total"``, ``"wet_days"``, ``"dry_run"``, ``"dry_window"`` or ``"day_missing"`` (the candidate day itself is
    missing under the ``"keep"`` policy, so it is rejected)."""
    outcome: str
    """``"pass"``, ``"fail"`` or ``"unknown"``; for a veto, ``"fail"`` means a spell was confirmed and the candidate
    rejected."""
    first: int
    """First day index of the span the rule looks at."""
    last: int
    """Last day index of that span (inclusive)."""
    need_first: int
    """First day index that must be observable."""
    need_last: int
    """Last day index that must be observable (inclusive)."""
    n_days: int
    """Length of the span in days."""
    op: str
    """The comparator of the comparison (``"ge"``, ``"gt"``, ``"le"``, ``"lt"``)."""
    limit: float
    """The limit compared against: mm, a day count or a wet-day count."""
    limit_name: str | None = None
    """The threshold's name, or None for a fixed limit."""
    value: float | None = None
    """The window total in mm or the wet-day count; None when not observable."""
    placement: str | None = None
    """For a veto, where the spell may sit: ``"inside"``, ``"starts_inside"`` or ``"begins"``."""
    spell_days: int = 0
    """For a veto, the length in days of the spell it looks for."""
    spell: Spell | None = None
    """The confirmed spell of a failing veto; None when the horizon ``"reject"`` policy rejected the candidate because
    its span runs past the data."""
    first_not_wet: int | None = None
    """The day index of the first non-wet day of a wet-day requirement."""
    flags: tuple = ()
    """The reasons an unknown outcome is unknown (``"missing"``, ``"horizon"``)."""
    missing_days: tuple = ()
    """The missing day indices inside the observed part of the need."""
    since_start: bool = False
    """Whether a total accumulates from the search start rather than over a fixed span."""
    offset_first: int = 0
    """The offset in days of the span's first day from the candidate."""


@dataclass(frozen=True)
class CandidateTrace:
    """One candidate day: its three-valued state, flags and every check behind them."""

    day: int
    """Day index of the candidate."""
    date: object
    """The date label of the candidate (``datetime64[D]`` or a ``(y, m, d)`` tuple)."""
    state: str
    """``"T"``, ``"F"`` or ``"U"`` (true, false, unknown)."""
    flags: tuple
    """The backend's flag names (``"missing"``, ``"horizon"``)."""
    checks: tuple
    """The `Check` objects of the candidate, requirements first, then vetoes."""
    decisive: int | None
    """The position in `checks` of the first failing check (state F) or first unknown check (state U); None for a
    candidate that passes everything."""


@dataclass(frozen=True)
class CurveTrace:
    """The cumulative anomaly of a curve rule over the search window, in mm, and where it is lowest."""

    days: np.ndarray
    """Day indices of the search window, ``int64``, shape (day,)."""
    cumulative_mm: np.ndarray
    """The cumulative anomaly in mm, ``float64``, shape (day,)."""
    min_day: int
    """Day index of the lowest value."""
    min_mm: float
    """The lowest cumulative anomaly, in mm."""
    baseline_mm: float
    """The baseline subtracted from the daily rain, in mm per day."""
    after_minimum: bool
    """Whether the onset is the day after the minimum (else the day of it)."""


@dataclass(frozen=True)
class Result:
    """The outcome of the search: the reported onset, its status and what decided it."""

    day: int | None
    """The reported onset day index, or None without one."""
    date: object
    """The reported onset date label, or None without one."""
    status: Status
    """The search status."""
    onset_lead: int | None
    """The 1-based lead of the onset (0 or less when back-dated or ``PRIOR``) for forecasts, else None."""
    decision_day: int | None
    """The candidate that decided the search (it differs from `day` when the rule reports the first wet day)."""
    decided_by_history: bool
    """Whether the observed history decided the search (status ``PRIOR``)."""


@dataclass
class Anatomy:
    """Everything needed to see why one cell and year (or forecast init) got its onset date.

    Day indices are those of the series searched: the observed time index, or for forecasts the stitched index with
    lead 1 at `split`.
    """

    definition: Definition
    """The onset definition."""
    sentence: str
    """The definition described in words."""
    location: dict
    """The traced cell's coordinates by dim name (``lat``, ``lon``), in degrees."""
    season: str
    """The name of the season searched."""
    year: int | None
    """The season's year for observed series; None for forecasts."""
    init: object
    """The forecast init date label; None for observed series."""
    member: int | None
    """The ensemble member of a forecast, as asked for; None otherwise."""
    split: int
    """Day index of lead 1 in the series (0 when the forecast is searched alone)."""
    lead_label: str | None
    """The forecast's lead label; None for observed series."""
    stitched: bool
    """Whether observed history was stitched in before the forecast."""
    days: np.ndarray
    """The plotted day indices, ``int64``, shape (day,)."""
    dates: np.ndarray
    """Labels of `days`: ``datetime64[D]`` for standard calendars, else an object array of ``(y, m, d)``."""
    rain_mm: np.ndarray
    """Daily rain in mm over `days`, ``float64``, shape (day,)."""
    missing: np.ndarray
    """Whether each of `days` is missing, ``bool``, shape (day,)."""
    wet: np.ndarray
    """Whether each of `days` is a wet day, ``bool``, shape (day,)."""
    source: np.ndarray
    """``"observed"`` or ``"forecast"`` for each of `days`, shape (day,)."""
    wet_mm: float
    """The wet-day amount in mm."""
    wet_op: str
    """The wet-day comparator (``"ge"``, ``"gt"``, ...)."""
    first_candidate: int
    """First day index of the search window."""
    last_candidate: int
    """Last day index of the search window."""
    data_end: int
    """The first day index without data."""
    thresholds: tuple
    """The named thresholds with the integers compared, as `ThresholdValue` objects."""
    candidates: tuple
    """The `CandidateTrace` objects in date order up to the decision."""
    curve: CurveTrace | None
    """Set instead of `candidates` for curve rules."""
    result: Result
    """The outcome of the search."""
    labels: "_Labels" = field(repr=False, default=None)
    """The calendar labels of the series, used by `date_of`."""
    dry_mm: float | None = None
    """The dry-day amount in mm, or None when the definition has no dry day."""
    dry_op: str | None = None
    """The dry-day comparator, or None when the definition has no dry day."""

    @property
    def is_forecast(self) -> bool:
        """Whether the series is a forecast (alone or stitched onto history)."""
        return self.lead_label is not None

    def rain_mm_at(self, day: int) -> float:
        """Return the rain in mm on a day index inside the plotted range.

        Parameters
        ----------
        day : int
            A day index within `days`.

        Returns
        -------
        float
            The rain on that day, in mm.
        """
        return float(self.rain_mm[int(day) - int(self.days[0])])

    def date_of(self, day: int):
        """Return the date label of any day index, including days just outside the series.

        Parameters
        ----------
        day : int
            A day index.

        Returns
        -------
        numpy.datetime64 or tuple of (int, int, int)
            ``datetime64[D]`` for standard calendars, else a ``(year, month, day)`` tuple.
        """
        return self.labels.date(day)


class _Labels:
    """Calendar labels of every day of a series, extended past either end by calendar arithmetic.

    Parameters
    ----------
    calendar : str
        The calendar name.
    year : array_like of int
        Year of each day of the series.
    month : array_like of int
        Month of each day.
    day : array_like of int
        Day of month of each day.

    Attributes
    ----------
    calendar : str
        The calendar name.
    y, m, d : numpy.ndarray of int64, shape (day,)
        Year, month and day of month of each day of the series.
    n : int
        The number of days in the series.
    """

    def __init__(self, calendar: str, year, month, day):
        self.calendar = calendar
        self.y, self.m, self.d = (np.asarray(a, dtype=np.int64) for a in (year, month, day))
        self.n = len(self.y)
        self._extra: dict = {}

    @property
    def standard(self) -> bool:
        """Whether dates are real datetimes (standard calendars) rather than ``(y, m, d)`` tuples."""
        return self.calendar in engine._STANDARD_CALENDARS

    def ymd(self, i: int) -> tuple[int, int, int]:
        """Return the ``(year, month, day)`` of a day index, extending the calendar past either end of the series.

        Parameters
        ----------
        i : int
            A day index, possibly outside ``0 .. n - 1``.

        Returns
        -------
        tuple of (int, int, int)
            The year, month and day of month.
        """
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
        """Return the date of a day index: ``datetime64[D]`` for standard calendars, else a ``(y, m, d)`` tuple.

        Parameters
        ----------
        i : int
            A day index, possibly outside the series.

        Returns
        -------
        numpy.datetime64 or tuple of (int, int, int)
            The date label.
        """
        y, m, d = self.ymd(i)
        if self.standard:
            return (np.datetime64(f"{y:04d}-{m:02d}", "M").astype("datetime64[D]") + np.timedelta64(d - 1, "D"))
        return (y, m, d)

    def dates(self, idx: np.ndarray) -> np.ndarray:
        """Return the dates of an array of day indices inside the series.

        Parameters
        ----------
        idx : numpy.ndarray of int64, shape (day,)
            Day indices within ``0 .. n - 1``.

        Returns
        -------
        numpy.ndarray, shape (day,)
            ``datetime64[D]`` for standard calendars, else an object array of ``(y, m, d)`` tuples.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if self.standard:
            months = ((self.y[idx] - 1970) * 12 + self.m[idx] - 1).astype("datetime64[M]").astype("datetime64[D]")
            return months + (self.d[idx] - 1).astype("timedelta64[D]")
        out = np.empty(len(idx), dtype=object)
        for k, i in enumerate(idx):
            out[k] = (int(self.y[i]), int(self.m[i]), int(self.d[i]))
        return out


def date_ymd(date) -> tuple[int, int, int]:
    """Return the ``(year, month, day)`` of an `Anatomy` date.

    Parameters
    ----------
    date : numpy.datetime64 or tuple of (int, int, int)
        A date label as found in an `Anatomy`.

    Returns
    -------
    tuple of (int, int, int)
        The year, month and day of month.
    """
    if isinstance(date, tuple):
        return date
    day = np.datetime64(date, "D")
    month = day.astype("datetime64[M]")
    return (int(month.astype(np.int64) // 12 + 1970), int(month.astype(np.int64) % 12 + 1),
            int((day - month.astype("datetime64[D]")).astype(np.int64) + 1))


def format_date(date, with_year: bool = False) -> str:
    """Format a date as ``"18 May"`` (or ``"18 May 2020"``).

    Parameters
    ----------
    date : numpy.datetime64 or tuple of (int, int, int)
        A date label as found in an `Anatomy`.
    with_year : bool, default False
        Append the year.

    Returns
    -------
    str
        The formatted date.
    """
    y, m, d = date_ymd(date)
    return f"{d} {_MONTHS[m - 1]}" + (f" {y}" if with_year else "")


# ---------------------------------------------------------------------------------- locating the cell

def _nearest(coords: np.ndarray, value: float, name: str) -> int:
    """Return the index of the grid coordinate nearest to a value.

    Parameters
    ----------
    coords : numpy.ndarray, shape (n,)
        The grid coordinates, in degrees.
    value : float
        The wanted coordinate, in degrees.
    name : str
        The dim name, for the error message.

    Returns
    -------
    int
        The index of the nearest coordinate.

    Raises
    ------
    ValueError
        The nearest coordinate is farther than half a grid step (the median spacing) from ``value``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    i = int(np.argmin(np.abs(coords - value)))
    if len(coords) > 1:
        step = float(np.median(np.abs(np.diff(coords))))
        if abs(coords[i] - value) > 0.5 * step + 1e-9:
            raise ValueError(f"{name} {value} is {abs(coords[i] - value):g} from the nearest grid cell "
                             f"({coords[i]:g}), more than half a grid step ({step / 2:g})")
    return i


def _cell_of(spatial_dims: tuple[str, ...], coords: dict, lat, lon) -> tuple[int, ...]:
    """Return the index of the nearest cell along each spatial dim.

    Parameters
    ----------
    spatial_dims : tuple of str
        The spatial dims of the series (empty for a series without spatial dims).
    coords : dict
        The coordinate values by dim name.
    lat : float, optional
        The wanted latitude, in degrees.
    lon : float, optional
        The wanted longitude, in degrees.

    Returns
    -------
    tuple of int
        The index along each of ``spatial_dims``; empty without spatial dims.

    Raises
    ------
    ValueError
        A spatial dim is not ``lat``/``lon`` or has no coordinates, or its wanted value is None.
    """
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
    """Return the coordinates of the chosen cell, by dim name.

    Parameters
    ----------
    spatial_dims : tuple of str
        The spatial dims of the series.
    coords : dict
        The coordinate values by dim name.
    cell : tuple of int
        The index along each of ``spatial_dims``.

    Returns
    -------
    dict of str to float
        The coordinate value of the cell, for each dim that has coordinates.
    """
    return {dim: float(np.asarray(coords[dim])[i]) for dim, i in zip(spatial_dims, cell) if dim in coords}


def _slice_coords(coords: dict, sliced: dict) -> dict:
    """Return the coordinates with the given dims cut to slices.

    Parameters
    ----------
    coords : dict
        The coordinate values by dim name.
    sliced : dict of str to slice
        The slice to apply to each dim named; dims not named are kept whole.

    Returns
    -------
    dict
        The sliced coordinates, under the same names.
    """
    return {k: (np.asarray(v)[sliced[k]] if k in sliced else v) for k, v in coords.items()}


def _single_rain(rain: DailyRain, cell: tuple[int, ...]) -> DailyRain:
    """Return the one-cell `~hcwf.onset.model.DailyRain` (spatial dims kept with length 1).

    Parameters
    ----------
    rain : hcwf.onset.model.DailyRain
        The gridded observed rain.
    cell : tuple of int
        The index along each spatial dim.

    Returns
    -------
    hcwf.onset.model.DailyRain
        The rain of that cell, with the calendar labels and metadata of ``rain``.
    """
    sl = tuple(slice(i, i + 1) for i in cell)
    sliced = dict(zip(rain.dims[:-1], sl))
    return DailyRain(rain.tenths[sl], rain.missing[sl], rain.dims, _slice_coords(rain.coords, sliced), rain.year,
                     rain.month, rain.day, rain.calendar, rain.day_boundary, rain.source)


def _single_forecast(fc: ForecastRain, init: int, member: int | None, cell: tuple[int, ...]) -> ForecastRain:
    """Return the one-init, one-member, one-cell `~hcwf.onset.model.ForecastRain` (dims kept with length 1).

    Parameters
    ----------
    fc : hcwf.onset.model.ForecastRain
        The gridded forecast.
    init : int
        Index of the init time.
    member : int, optional
        Index of the member; None when the forecast has no ``member`` dim.
    cell : tuple of int
        The index along each spatial dim.

    Returns
    -------
    hcwf.onset.model.ForecastRain
        The forecast of that init, member and cell.
    """
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
    """Return the thresholds the plan needs at one cell, via the engine's own broadcasting.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    thresholds : mapping of str to xarray.DataArray, optional
        The named thresholds as given to the engine.
    spatial_dims : tuple of str
        The spatial dims of the series.
    spatial_shape : tuple of int
        The lengths of the spatial dims.
    cell : tuple of int
        The index along each spatial dim.

    Returns
    -------
    dict of str to numpy.ndarray of float64, shape (1,)
        The threshold value at the cell, in mm, by threshold name.
    """
    per_cell = engine._series_thresholds(plan, thresholds, spatial_dims, spatial_shape, spatial_shape)
    flat = int(np.ravel_multi_index(cell, spatial_shape)) if spatial_shape else 0
    return {name: values[flat:flat + 1].copy() for name, values in per_cell.items()}


def _threshold_values(plan: P.Plan, per_cell: dict, amounts: dict, exact: bool) -> tuple[ThresholdValue, ...]:
    """Return the named thresholds with the amounts compared for each comparator.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition (``amount_specs`` lists what is compared).
    per_cell : dict of str to numpy.ndarray
        The threshold value at the cell in mm, by name (see `_cell_thresholds`).
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key, in tenths of a mm (thousandths for a baseline), or mm when ``exact``.
    exact : bool
        Whether the rain is exact (amounts are then floats in mm).

    Returns
    -------
    tuple of ThresholdValue
        One entry per named threshold used by the plan; fixed amounts are skipped.
    """
    found: dict[str, dict] = {}
    for spec in plan.amount_specs:
        if spec.kind == "fixed":
            continue
        entry = found.setdefault(spec.name, {})
        value = amounts[spec.key][0]
        entry["baseline" if spec.kind == "baseline" else _OP_NAMES[spec.op]] = float(value) if exact else int(value)
    return tuple(ThresholdValue(name, float(per_cell[name][0]), compared, exact) for name, compared in found.items())


# ---------------------------------------------------------------------------------- the backend's facts

def _threshold_name(key: str) -> str | None:
    """Return the threshold name an amount key refers to.

    Parameters
    ----------
    key : str
        An amount key of the plan; ``"fixed:..."`` for a fixed amount, else ``"<name>:<suffix>"``.

    Returns
    -------
    str or None
        The threshold's name, None for a fixed amount.
    """
    return None if key.startswith("fixed:") else key.rsplit(":", 1)[0]


def _requirement_check(ev: reference.Evaluation, i: int, c: int, amounts: dict) -> Check:
    """Return requirement ``i`` of candidate ``c`` as the backend evaluated it.

    Parameters
    ----------
    ev : hcwf.onset.backends.reference.Evaluation
        The backend's evaluation of the job.
    i : int
        Index of the requirement in the plan.
    c : int
        Index of the candidate.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key.

    Returns
    -------
    Check
        The requirement's check, with totals and limits converted to mm.
    """
    cp, f, per_mm = ev.plan.conditions[i], ev.conditions[i], ev.tables.per_mm
    code, a, b = int(f.code[0, c]), int(f.first[0, c]), int(f.last[0, c])
    unknown = code == reference.UNKNOWN
    flags = _flag_names(int(f.flags[0, c])) if unknown else ()
    missing_days = ev.missing_days(0, a, b) if unknown else ()
    value = None if unknown else float(f.value[0, c])
    if cp.kind == P.TOTAL:
        key = cp.cmp.key
        return Check("require", i, "total", _OUTCOME[code], a, b - 1, a, b - 1, b - a, _OP_NAMES[cp.cmp.op],
                     float(amounts[key][0]) / per_mm, _threshold_name(key), None if unknown else value / per_mm,
                     None, 0, None, None, flags, missing_days, cp.since_start, cp.first)
    first_not_wet = ev.first_not_wet(0, a, b) if code == reference.FALSE else None
    return Check("require", i, "wet_days", _OUTCOME[code], a, b - 1, a, b - 1, b - a, _OP_NAMES[cp.op],
                 float(cp.count), None, value, None, 0, None, first_not_wet, flags, missing_days, cp.since_start,
                 cp.first)


def _veto_check(ev: reference.Evaluation, i: int, c: int, amounts: dict) -> Check:
    """Return veto ``i`` of candidate ``c`` as the backend evaluated it.

    The confirmed dry spell is attached when the veto rejects the candidate.

    Parameters
    ----------
    ev : hcwf.onset.backends.reference.Evaluation
        The backend's evaluation of the job.
    i : int
        Index of the veto in the plan.
    c : int
        Index of the candidate.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key.

    Returns
    -------
    Check
        The veto's check; its outcome ``"fail"`` means a spell was confirmed.
    """
    vp, f, per_mm = ev.plan.vetoes[i], ev.vetoes[i], ev.tables.per_mm
    code, a, b = int(f.code[0, c]), int(f.first[0, c]), int(f.last[0, c])
    need_last = int(f.need_last[0, c]) - 1
    unknown = code == reference.UNKNOWN
    spell = None
    found = ev.spell(i, 0, c) if code == reference.TRUE else None
    if found is not None:
        spell = Spell(found[0], found[1], found[1] - found[0] + 1, float(found[2]) / per_mm)
    kind = "dry_run" if vp.kind == P.DRY_RUN else "dry_window"
    op, limit, name = "ge", float(vp.length), None
    if vp.kind == P.DRY_WINDOW:
        op, limit, name = _OP_NAMES[vp.cmp.op], float(amounts[vp.cmp.key][0]) / per_mm, _threshold_name(vp.cmp.key)
    return Check("veto", i, kind, _VETO_OUTCOME[code], a, b - 1, a, need_last, b - a, op, limit, name, None,
                 _PLACEMENTS[vp.placement], vp.length, spell, None,
                 _flag_names(int(f.flags[0, c])) if unknown else (),
                 ev.missing_days(0, a, need_last + 1) if unknown else (), False, vp.first)


def _day_missing_check(day: int) -> Check:
    """Return the ``"keep"`` policy's rejection of a candidate whose own day is missing.

    Parameters
    ----------
    day : int
        Day index of the candidate.

    Returns
    -------
    Check
        A failing ``"day_missing"`` requirement.
    """
    return Check("require", -1, "day_missing", "fail", day, day, day, day, 0, "ge", 0.0)


# ---------------------------------------------------------------------------------- building the Anatomy

def _flag_names(bits: int) -> tuple:
    """Return the names of the backend's flag bits that are set.

    Parameters
    ----------
    bits : int
        The flag bit field of the backend.

    Returns
    -------
    tuple of str
        ``"missing"`` and/or ``"horizon"``.
    """
    return tuple(name for bit, name in ((reference.MISSING_FLAG, "missing"), (reference.HORIZON_FLAG, "horizon"))
                 if bits & bit)


def _season_index(names: list[str], season: str | None) -> int:
    """Return the position of the requested season (the first when none is given).

    Parameters
    ----------
    names : list of str
        The season names of the job layout.
    season : str, optional
        The season asked for.

    Returns
    -------
    int
        Its index in ``names``; 0 when ``season`` is None.

    Raises
    ------
    ValueError
        ``season`` is not one of ``names``.
    """
    if season is None:
        return 0
    if season not in names:
        raise ValueError(f"unknown season {season!r}; this definition has {names}")
    return names.index(season)


def _first_passage_candidates(ev: reference.Evaluation, amounts: dict, labels) -> tuple:
    """Return the `CandidateTrace` objects up to and including the first candidate that is not false.

    Parameters
    ----------
    ev : hcwf.onset.backends.reference.Evaluation
        The backend's evaluation of the single job.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key.
    labels : _Labels
        The calendar labels of the series.

    Returns
    -------
    tuple of CandidateTrace
        The valid candidates in date order; the last is the decisive one unless every candidate failed.
    """
    out = []
    for c in range(ev.day.shape[1]):
        if not ev.valid[0, c]:
            continue
        d = int(ev.day[0, c])
        state = int(ev.state[0, c])
        checks = [_requirement_check(ev, i, c, amounts) for i in range(len(ev.conditions))]
        checks += [_veto_check(ev, i, c, amounts) for i in range(len(ev.vetoes))]
        if ev.day_missing[0, c]:
            checks.append(_day_missing_check(d))
        decisive = None
        for k, check in enumerate(checks):
            if check.outcome == ("fail" if state == reference.FALSE else "unknown") and decisive is None:
                decisive = k
        out.append(CandidateTrace(d, labels.date(d), _STATE[state], _flag_names(int(ev.flags[0, c])),
                                  tuple(checks), decisive))
        if state != reference.FALSE:
            break
    return tuple(out)


def _curve_trace(plan, trace, amounts) -> CurveTrace | None:
    """Return the cumulative anomaly of a curve job in mm, with its minimum.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled curve definition.
    trace : dict of str to numpy.ndarray
        The backend's curve trace arrays (``valid``, ``day``, ``cumulative`` in thousandths of a mm), one job.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key (the baseline is in thousandths of a mm).

    Returns
    -------
    CurveTrace or None
        The trace over the valid days; None when no day is valid.
    """
    valid = trace["valid"][0]
    if not valid.any():
        return None
    days = trace["day"][0][valid]
    cumulative = trace["cumulative"][0][valid] / 1000.0
    k = int(np.argmin(cumulative))
    return CurveTrace(days, cumulative, int(days[k]), float(cumulative[k]), amounts[plan.curve_key][0] / 1000.0,
                      plan.curve_after_min)


def _plan_reach(plan: P.Plan) -> int:
    """Return how many days past a candidate its spans can reach.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.

    Returns
    -------
    int
        The largest ``last`` offset of the requirements, vetoes (with their spell length when not inside) and the
        report span.
    """
    reach = max([c.last for c in plan.conditions] + [v.last + (v.length - 1 if v.placement != P.INSIDE else 0)
                                                    for v in plan.vetoes] + [1])
    if plan.report_span is not None:
        reach = max(reach, plan.report_span[1])
    return reach


@dataclass
class _Case:
    """Everything the shared builder needs about one search: the series and where the job sits in it."""

    tenths: np.ndarray
    """Daily rain in tenths of a mm of the searched series, shape (day,)."""
    missing: np.ndarray
    """Whether each day is missing, ``bool``, shape (day,)."""
    labels: _Labels
    """The calendar labels of the series."""
    source: np.ndarray
    """``"observed"`` or ``"forecast"`` for each day, shape (day,)."""
    split: int
    """Day index of lead 1 (0 when the forecast is searched alone or the series is observed)."""
    start: int
    """First day index of the search window."""
    end: int
    """Last day index of the search window."""
    data_end: int
    """The first day index without data."""
    preset: int
    """A `~hcwf.onset.model.Status` value fixed by the job layout, or a negative number when the search must run."""
    run: object
    """The callback that decides the job; it takes the `~hcwf.onset.backends.reference.Tables`."""


def _build(defn: Definition, plan: P.Plan, case: _Case, per_cell: dict, amounts: dict, invalid: bool, *,
           location: dict, season: str, year, init, member, lead_label, stitched: bool) -> Anatomy:
    """Run the traced search for one prepared case and assemble the `Anatomy`.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The onset definition.
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    case : _Case
        The series and the job window.
    per_cell : dict of str to numpy.ndarray
        The threshold values at the cell, in mm.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key.
    invalid : bool
        Whether a threshold is unavailable at the cell (status ``INSUFFICIENT_DATA``).
    location : dict
        The cell's coordinates by dim name.
    season : str
        The season name.
    year : int, optional
        The season's year (observed series).
    init : object
        The forecast init date label, or None.
    member : int, optional
        The ensemble member, or None.
    lead_label : str, optional
        The forecast's lead label, or None for observed series.
    stitched : bool
        Whether the series is a forecast stitched onto history.

    Returns
    -------
    Anatomy
        The structured explanation of the search.
    """
    tables = reference.Tables(plan, case.tenths[None, :], case.missing[None, :], amounts)
    per_mm = tables.per_mm
    x, miss, wet = tables.x[0], tables.missing[0], tables.wet[0]
    status, day, facts, data_end = Status.NOT_FOUND, -1, None, case.data_end
    if case.preset >= 0:
        status = Status(case.preset)
    elif invalid:
        status = Status.INSUFFICIENT_DATA
    else:
        status, day, facts, data_end = case.run(tables)
    candidates, curve = (), None
    if facts is not None:
        if plan.family == P.CURVE:
            curve = _curve_trace(plan, facts, amounts)
        else:
            candidates = _first_passage_candidates(facts, amounts, case.labels)
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
    hi = min(max(hi, lo + 1), tables.n_time)
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
        dates=case.labels.dates(days), rain_mm=x[days] / per_mm, missing=miss[days], wet=wet[days],
        source=case.source[days], wet_mm=float(amounts[plan.wet.key][0]) / per_mm, wet_op=_OP_NAMES[plan.wet.op],
        first_candidate=int(case.start), last_candidate=int(case.end), data_end=int(data_end),
        thresholds=_threshold_values(plan, per_cell, amounts, per_mm == 1), candidates=candidates, curve=curve,
        result=result, labels=case.labels,
        dry_mm=None if plan.dry is None else float(amounts[plan.dry.key][0]) / per_mm,
        dry_op=None if plan.dry is None else _OP_NAMES[plan.dry.op])


def _decide(plan: P.Plan, tables: reference.Tables, start: int, end: int, data_end: int):
    """Return the backend's decision on one job of the tables.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    tables : hcwf.onset.backends.reference.Tables
        The prepared series tables.
    start : int
        First day index of the search window.
    end : int
        Last day index of the search window.
    data_end : int
        The first day index without data.

    Returns
    -------
    tuple of (Status, int, object, int)
        The status, the onset day index (negative without one), the facts and ``data_end``. The facts are the
        first-passage ``Evaluation``, or the curve rule's trace arrays.
    """
    jobs = Jobs(np.zeros(1, np.int64), np.array([start]), np.array([end]), np.array([data_end]))
    if plan.family == P.CURVE:
        det = reference.detect(plan, tables.x, tables.missing, tables.amounts, jobs, trace=True)
        return Status(int(det.status[0])), int(det.day[0]), det.trace, data_end
    ev = reference.evaluate(plan, tables, jobs)
    return Status(int(ev.status[0])), int(ev.onset[0]), ev, data_end


def _detect_single(plan: P.Plan, start: int, end: int, data_end: int):
    """Return a run callback: the reference backend, with its facts, on one job of the displayed series.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    start : int
        First day index of the search window.
    end : int
        Last day index of the search window.
    data_end : int
        The first day index without data.

    Returns
    -------
    callable
        A function of the `~hcwf.onset.backends.reference.Tables` returning ``(status, day, facts, data_end)`` as `_decide` does.
    """
    def run(tables: reference.Tables):
        """Return (status, day, facts, data_end) of the job."""
        return _decide(plan, tables, start, end, data_end)
    return run


def trace_observed(defn: Definition, rain: DailyRain, *, lat: float | None, lon: float | None, year: int,
                   season: str | None = None, thresholds=None, season_climatology: xr.Dataset | None = None,
                   horizon: str = DEFAULT_HORIZON, missing: str | None = None) -> Anatomy:
    """Trace the onset search of the cell nearest ``(lat, lon)`` in the season of ``year``.

    The cell must lie within half a grid step of ``(lat, lon)``. The jobs are the engine's own
    (``build_observed_jobs`` on the one-cell series) and the `Anatomy`'s result equals ``detect_observed``'s.
    ``thresholds``, ``horizon`` and ``missing`` are as for ``detect_observed`` (exact rain, every missing policy, ``dry_day``
    and beginning-within vetoes all work). A cell farther than half a grid step from ``(lat, lon)``, or an unknown
    ``season``, raises `ValueError`.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The onset definition.
    rain : hcwf.onset.model.DailyRain
        The observed daily rain.
    lat : float, optional
        Latitude of the cell, in degrees; required when ``rain`` has a ``lat`` dim.
    lon : float, optional
        Longitude of the cell, in degrees; required when ``rain`` has a ``lon`` dim.
    year : int
        The year of the season to trace.
    season : str, optional
        The season to trace, by name; the first by default.
    thresholds : mapping of str to xarray.DataArray, optional
        The named thresholds the definition refers to, in mm.
    season_climatology : xarray.Dataset, optional
        The season climatology for derived seasons, on the grid of ``rain``.
    horizon : {"censor", "reject"}, default "censor"
        The policy for spans that run past the data.
    missing : str, optional
        The missing-day policy; None takes the engine's default for ``rain``.

    Returns
    -------
    Anatomy
        The explanation of the search at the cell.
    """
    policy = engine._missing_policy(rain, missing)
    plan = P.compile(defn, horizon=horizon, missing=policy)
    spatial_dims, spatial_shape = rain.dims[:-1], rain.tenths.shape[:-1]
    cell = _cell_of(spatial_dims, rain.coords, lat, lon)
    rain1 = _single_rain(rain, cell)
    clim1 = None
    if season_climatology is not None:
        clim1 = season_climatology.isel({d: slice(i, i + 1) for d, i in zip(spatial_dims, cell)})
    jobs, layout = engine.build_observed_jobs(defn, rain1, [year], clim1)
    k = _season_index(layout.season_names, season)
    per_cell = _cell_thresholds(plan, thresholds, spatial_dims, spatial_shape, cell)
    amounts, invalid = P.resolve_amounts(plan, per_cell, n_series=1, exact=resolution_of(rain.tenths) == "exact")
    if policy == "drop":
        engine._drop_incomplete_years(rain1, np.asarray([year]), layout)
    start, end, data_end = int(jobs.start[k]), int(jobs.end[k]), int(jobs.data_end[k])
    row = (*cell, slice(None))
    labels = _Labels(rain.calendar, rain.year, rain.month, rain.day)
    case = _Case(rain.tenths[row], engine._flat_missing(rain1, policy, 1)[0], labels,
                 np.full(len(labels.y), "observed"), 0, start, end, data_end, int(layout.preset[k]),
                 _detect_single(plan, start, end, data_end))
    return _build(defn, plan, case, per_cell, amounts, bool(invalid[0]),
                  location=_location(spatial_dims, rain.coords, cell), season=layout.season_names[k], year=year,
                  init=None, member=None, lead_label=None, stitched=False)


def _find_init(forecast: ForecastRain, init) -> int:
    """Return the index of the init whose date label equals ``init``.

    Parameters
    ----------
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    init : datetime.date or numpy.datetime64 or str or tuple of (int, int, int)
        The init date.

    Returns
    -------
    int
        The index of the first matching init.

    Raises
    ------
    ValueError
        The forecast has no init on that date.
    """
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
    """Return the index of the requested member (by coordinate value, else by position).

    Parameters
    ----------
    forecast : hcwf.onset.model.ForecastRain
        The forecast.
    member : int, optional
        The member's coordinate value, or its position when no coordinate matches.

    Returns
    -------
    int or None
        The member index; None for a forecast without a ``member`` dim.

    Raises
    ------
    ValueError
        A member is given for a forecast without members, none is given for one with members, or it does not exist.
    """
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
                   restrict_to_season: bool = True, season_climatology: xr.Dataset | None = None,
                   horizon: str = DEFAULT_HORIZON, missing: str | None = None) -> Anatomy:
    """Trace the onset search of one init (and member) at the cell nearest ``(lat, lon)``.

    With ``history`` the search is stitched (observed history before lead 1, then the forecast, with the engine's
    history-only ``PRIOR`` decision first); without it the forecast is searched alone. The jobs are the engine's own
    (``build_forecast_jobs`` / ``build_stitched_jobs`` on the one-cell series), so the result equals
    ``detect_forecast``'s. Days are indexed in the stitched series (lead 1 is day ``split``) or, without history, by
    lead - 1.

    `~hcwf.onset.model.DerivedSeasons` (with ``restrict_to_season`` true) need ``season_climatology``, as for ``detect_forecast`` (dims
    ``(season, *spatial)`` on the forecast's grid); the traced cell's own season is used and ``season`` picks among the
    derived seasons by name (``"1"``, ``"2"``, ...; the first by default). ``horizon`` and ``missing`` are as for
    ``detect_forecast``.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The onset definition.
    forecast : hcwf.onset.model.ForecastRain
        The forecast rain.
    init : datetime.date or numpy.datetime64 or str or tuple of (int, int, int)
        The init date to trace.
    lat : float, optional
        Latitude of the cell, in degrees.
    lon : float, optional
        Longitude of the cell, in degrees.
    member : int, optional
        The ensemble member (required when the forecast has members).
    history : hcwf.onset.model.DailyRain, optional
        Observed history to stitch before lead 1; None searches the forecast alone.
    thresholds : mapping of str to xarray.DataArray, optional
        The named thresholds the definition refers to, in mm.
    lead_window : tuple of (int, int), default (1, 30)
        The first and last lead day of the search window.
    season : str, optional
        The derived season to trace, by name; the first by default.
    restrict_to_season : bool, default True
        Restrict the search to the season.
    season_climatology : xarray.Dataset, optional
        The season climatology for derived seasons, on the forecast's grid.
    horizon : {"censor", "reject"}, default "censor"
        The policy for spans that run past the forecast.
    missing : str, optional
        The missing-day policy; None takes the engine's default.

    Returns
    -------
    Anatomy
        The explanation of the search at the cell.

    Raises
    ------
    ValueError
        The definition has derived seasons but no ``season_climatology`` is given, the missing policy is ``"drop"``,
        or the cell, init, member or season cannot be found.
    """
    if isinstance(defn.seasons, DerivedSeasons) and restrict_to_season and season_climatology is None:
        raise ValueError("this definition has derived seasons: pass season_climatology "
                         "(n_seasons, onset_doy, cessation_doy) to trace_forecast")
    policy = engine._missing_policy(forecast, missing)
    if policy == "drop":
        raise ValueError("missing 'drop' removes whole cell-years of observed rain; use 'zero', 'keep' or 'flag' "
                         "for forecasts")
    plan = P.compile(defn, horizon=horizon, missing=policy)
    series_dims = forecast.dims[:-1]
    n_lead_dims = 2 if "member" in series_dims else 1
    spatial_dims = series_dims[n_lead_dims:]
    spatial_shape = forecast.tenths.shape[n_lead_dims:-1]
    cell = _cell_of(spatial_dims, forecast.coords, lat, lon)
    ii = _find_init(forecast, init)
    mi = _find_member(forecast, member)
    clim1 = None
    if season_climatology is not None:
        clim1 = season_climatology.isel({d: slice(i, i + 1) for d, i in zip(spatial_dims, cell)})
    fc1 = _single_forecast(forecast, ii, mi, cell)
    per_cell = _cell_thresholds(plan, thresholds, spatial_dims, spatial_shape, cell)
    amounts, invalid = P.resolve_amounts(plan, per_cell, n_series=1, exact=resolution_of(forecast.tenths) == "exact")
    n_leads = forecast.tenths.shape[-1]
    fc_row = (ii, *(() if mi is None else (mi,)), *cell, slice(None))
    fc_tenths, fc_missing = forecast.tenths[fc_row], engine._flat_missing(fc1, policy, 1)[0]
    fy, fm, fd = (a[0] for a in engine._label_table(fc1, n_leads))
    init_date = _Labels(forecast.calendar, [forecast.init_year[ii]], [forecast.init_month[ii]],
                        [forecast.init_day[ii]]).date(0)
    common = dict(location=_location(spatial_dims, forecast.coords, cell), year=None, init=init_date, member=member,
                  lead_label=forecast.lead_label)
    if history is None:
        jobs, layout = engine.build_forecast_jobs(defn, fc1, lead_window, restrict_to_season, clim1)
        k = _season_index(layout.season_names, season)
        start, end, data_end = int(jobs.start[k]), int(jobs.end[k]), int(jobs.data_end[k])
        labels = _Labels(forecast.calendar, fy, fm, fd)
        case = _Case(fc_tenths, fc_missing, labels, np.full(n_leads, "forecast"), 0,
                     start, end, data_end, int(layout.preset[k]), _detect_single(plan, start, end, data_end))
        return _build(defn, plan, case, per_cell, amounts, bool(invalid[0]), season=layout.season_names[k],
                      stitched=False, **common)
    P.reject_exact_stitched(history.tenths, forecast.tenths)
    hist1 = _single_rain(history, cell)
    jobs, layout = engine.build_stitched_jobs(defn, hist1, fc1, lead_window, restrict_to_season, clim1)
    k = _season_index(layout.season_names, season)
    split, start, end, data_end = (int(a[k]) for a in (jobs.split, jobs.start, jobs.end, jobs.data_end))
    hist_row = (*cell, slice(None))
    labels = _Labels(forecast.calendar, np.concatenate([history.year[:split], fy]),
                     np.concatenate([history.month[:split], fm]), np.concatenate([history.day[:split], fd]))
    tenths = np.concatenate([history.tenths[hist_row][:split], fc_tenths])
    missing = np.concatenate([engine._flat_missing(hist1, policy, 1)[0][:split], fc_missing])
    source = np.array(["observed"] * split + ["forecast"] * n_leads)
    preset = int(layout.preset[k])
    run = _stitched_run(plan, amounts, start, end, data_end, split)
    case = _Case(tenths, missing, labels, source, split, start, end, data_end, preset, run)
    return _build(defn, plan, case, per_cell, amounts, bool(invalid[0]), season=layout.season_names[k],
                  stitched=True, **common)


def _stitched_run(plan: P.Plan, amounts: dict, start: int, end: int, data_end: int, split: int):
    """Return a run callback for a stitched job.

    For first-passage rules the history-only ``PRIOR`` search runs first, then the full search.

    Parameters
    ----------
    plan : hcwf.onset.plan.Plan
        The compiled definition.
    amounts : dict of str to numpy.ndarray
        The resolved amounts by key.
    start : int
        First day index of the search window.
    end : int
        Last day index of the search window.
    data_end : int
        The first day index without data.
    split : int
        Day index of lead 1 in the stitched series.

    Returns
    -------
    callable
        A function of the `~hcwf.onset.backends.reference.Tables` returning ``(status, day, facts, data_end)`` as the engine decides it.
    """
    def run(tables: reference.Tables):
        """Return (status, day, facts, data_end) as the engine decides it."""
        if plan.family == P.FIRST_PASSAGE:
            prior_end = min(end, split - 1)
            if prior_end >= start:
                history = reference.Tables(plan, tables.x[:, :split], tables.missing[:, :split], amounts)
                status, day, facts, _ = _decide(plan, history, start, prior_end, split)
                if status == Status.ONSET:
                    return Status.PRIOR, day, facts, split
        return _decide(plan, tables, start, end, data_end)
    return run


# ---------------------------------------------------------------------------------- the readable log

def _span_text(a: Anatomy, first: int, last: int, with_year: bool = False) -> str:
    """Format the day indices ``first..last`` as ``"2-22 Jul"``, ``"24 May - 13 Jun"`` or ``"5 Jul"``.

    Parameters
    ----------
    a : Anatomy
        The anatomy that labels the days.
    first : int
        First day index.
    last : int
        Last day index.
    with_year : bool, default False
        Include the year (the first one only when it differs from the last).

    Returns
    -------
    str
        The span in words.
    """
    start, end = date_ymd(a.date_of(first)), date_ymd(a.date_of(last))
    if first == last:
        return format_date(a.date_of(first), with_year)
    if start[:2] == end[:2]:
        return f"{start[2]}-{end[2]} {_MONTHS[start[1] - 1]}" + (f" {start[0]}" if with_year else "")
    return f"{format_date(a.date_of(first), with_year and start[0] != end[0])} - {format_date(a.date_of(last), with_year)}"


def _amount_text(check: Check) -> str:
    """Format the limit of a check as ``"20 mm"`` (with the threshold's name when it has one).

    Parameters
    ----------
    check : Check
        The check.

    Returns
    -------
    str
        The limit in mm.
    """
    text = f"{check.limit:g} mm"
    return f"{text} ({check.limit_name})" if check.limit_name else text


def _total_label(check: Check) -> str:
    """Describe what a total requirement measures, for example ``"3-day total"``.

    Parameters
    ----------
    check : Check
        A ``"total"`` check.

    Returns
    -------
    str
        The description.
    """
    if check.since_start:
        return "total since the search start"
    if check.offset_first == 0:
        return f"{check.n_days}-day total"
    return f"{check.n_days}-day total over days {check.offset_first + 1}-{check.offset_first + check.n_days}"


def _all_wet(check: Check) -> bool:
    """Return whether a wet-days requirement asks for every day of its span to be wet.

    Parameters
    ----------
    check : Check
        The check.

    Returns
    -------
    bool
        True for a ``"wet_days"`` check with comparator ``"ge"`` and a limit of the span length.
    """
    return check.kind == "wet_days" and check.op == "ge" and check.limit == check.n_days


def _unknown_text(a: Anatomy, check: Check) -> str:
    """Explain why a check cannot be decided: what it needs and which part is missing or past the data.

    Parameters
    ----------
    a : Anatomy
        The anatomy that labels the days.
    check : Check
        A check with outcome ``"unknown"``.

    Returns
    -------
    str
        The reason in words.
    """
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
    """Describe the spell a veto looks for, for example ``"dry run of at least 7 days"``.

    Parameters
    ----------
    check : Check
        A veto check.

    Returns
    -------
    str
        The description.
    """
    if check.kind == "dry_run":
        return f"dry run of at least {check.spell_days} days"
    return f"{check.spell_days}-day window totalling {_OP_WORDS[check.op]} {_amount_text(check)}"


def _horizon_reject_text(a: Anatomy, check: Check) -> str:
    """Explain why the horizon ``"reject"`` policy rejected a veto's candidate: its span runs past the data.

    Parameters
    ----------
    a : Anatomy
        The anatomy that labels the days.
    check : Check
        The veto check.

    Returns
    -------
    str
        The reason in words.
    """
    noun = "forecast" if a.is_forecast else "data"
    return f"needs {_span_text(a, check.need_first, check.need_last)}, past the {noun} (the 'reject' horizon rejects it)"


def _reason(a: Anatomy, check: Check, wet_text: str) -> str:
    """Return the reason one check gave its outcome, in words with the numbers.

    Parameters
    ----------
    a : Anatomy
        The anatomy that labels the days.
    check : Check
        The check.
    wet_text : str
        The wet-day comparison in words (see `_wet_text`).

    Returns
    -------
    str
        The reason.
    """
    if check.outcome == "unknown":
        return _unknown_text(a, check)
    if check.kind == "day_missing":
        return "the candidate day is missing"
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
        if spell is None:
            return _horizon_reject_text(a, check)
        what = (f"a dry run of {spell.length} days" if check.kind == "dry_run"
                else f"a {check.spell_days}-day window totalling {spell.total_mm:.1f} mm")
        starts = "" if check.placement == "inside" else f" that {_PLACEMENT_VERBS[check.placement]}"
        return f"followed by {what} ({_span_text(a, spell.first, spell.last)}){starts} within {where}"
    return f"no {_veto_phrase(check)} within {where}"


def _group_key(c: CandidateTrace):
    """Return what makes two failing candidates fail "for the same reason" (so they can collapse into one line).

    Parameters
    ----------
    c : CandidateTrace
        A failing candidate.

    Returns
    -------
    tuple
        A hashable key of the decisive check: its role and index, and the spell span or first non-wet offset.
    """
    check = c.checks[c.decisive]
    if check.role == "veto":
        spell = check.spell
        return ("veto", check.index) + ((spell.first, spell.last) if spell is not None else (None, None))
    return ("require", check.index, check.first_not_wet - check.first if _all_wet(check) and check.first_not_wet is not None
            else None)


def _group_reason(a: Anatomy, group: list[CandidateTrace]) -> str:
    """Return the shared reason of consecutive candidates failing the same check, with value ranges.

    Parameters
    ----------
    a : Anatomy
        The anatomy that labels the days.
    group : list of CandidateTrace
        Consecutive failing candidates with equal `_group_key`.

    Returns
    -------
    str
        The reason, with the range of the values over the group.
    """
    checks = [c.checks[c.decisive] for c in group]
    first = checks[0]
    if first.kind == "day_missing":
        return "the candidate day is missing"
    if first.role == "veto" and first.spell is None:
        return _horizon_reject_text(a, first)
    if first.role == "veto":
        spell = first.spell
        what = (f"a dry run of {spell.length} days" if first.kind == "dry_run"
                else f"a {first.spell_days}-day window totalling {spell.total_mm:.1f} mm")
        starts = "" if first.placement == "inside" else f" that {_PLACEMENT_VERBS[first.placement]} within the following days"
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
    """Format the wet-day comparison, for example ``"at least 1 mm"``.

    Parameters
    ----------
    a : Anatomy
        The anatomy.

    Returns
    -------
    str
        The comparison in words.
    """
    return f"{_OP_WORDS[a.wet_op]} {a.wet_mm:g} mm"


def _candidate_lines(a: Anatomy, compress: bool) -> list[tuple[str, str]]:
    """Return the lines of the candidate list, in date order.

    Parameters
    ----------
    a : Anatomy
        The anatomy.
    compress : bool
        Collapse consecutive candidates that fail for the same reason into one line.

    Returns
    -------
    list of tuple of (str, str)
        The date column and the text of each line.
    """
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
                rows.append((head, f"✗ {len(group)} days: {_group_reason(a, group)}"))
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
    """Return one line per named threshold with the integers actually compared.

    Parameters
    ----------
    a : Anatomy
        The anatomy.

    Returns
    -------
    list of str
        The lines.
    """
    lines = []
    for t in a.thresholds:
        if "baseline" in t.compared:
            lines.append(f"  {t.name} = {t.mm:.4g} mm/day (compared as {t.compared['baseline']} thousandths of a mm)")
            continue
        if t.exact:
            used = ", ".join(f"{_OP_WORDS[op]} {v:g} mm" for op, v in t.compared.items())
            lines.append(f"  {t.name} = {t.mm:.4g} mm (compared exactly, as {used})")
            continue
        used = ", ".join(f"{_OP_WORDS[op]} {v / 10:g} mm ({v} tenths)" for op, v in t.compared.items())
        lines.append(f"  {t.name} = {t.mm:.4g} mm (compared as {used})")
    return lines


def _result_lines(a: Anatomy) -> list[str]:
    """Return the closing lines: the status in words, the onset date and (forecasts) the lead and what decided it.

    Parameters
    ----------
    a : Anatomy
        The anatomy.

    Returns
    -------
    list of str
        The lines.
    """
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
    """Explain why a search ended ``INSUFFICIENT_DATA``: a missing threshold, a missing season start, or missing days.

    Parameters
    ----------
    a : Anatomy
        The anatomy.

    Returns
    -------
    str
        The reason in words.
    """
    if any(not np.isfinite(t.mm) for t in a.thresholds):
        return "a threshold is not available at this cell"
    if not a.candidates and a.curve is None:
        return "the season starts outside the data"
    return "days the first undecided candidate needs are missing"


def _curve_lines(a: Anatomy) -> list[str]:
    """Return the curve rule's explanation: accumulation, minimum and where the onset falls.

    Parameters
    ----------
    a : Anatomy
        The anatomy.

    Returns
    -------
    list of str
        The lines.
    """
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
    """Return a readable log of the search: header, every candidate with its reason, and the result.

    Parameters
    ----------
    a : Anatomy
        The anatomy of the search, from `trace_observed` or `trace_forecast`.
    compress : bool, default True
        Collapse consecutive candidates that fail for the same reason into one line.

    Returns
    -------
    str
        The log, one line per row, without a trailing newline.
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
    if a.dry_mm is not None:
        lines.append(f"Dry day: {_OP_WORDS[a.dry_op]} {a.dry_mm:g} mm")
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
