"""Validated, unit-free onset definitions and the containers shared by every part of `hcwf.onset`.

Everything here is already normalized: rainfall amounts are integer tenths of a millimetre, durations are whole days,
and positions are 1-based days relative to the candidate day (day 1 is the candidate itself, day 0 the day before).
``docs/onset_spec.md`` defines what each field means; parsing YAML into these types lives in `hcwf.onset.definition`
(with the text grammar in `hcwf.onset.units`), and `hcwf.onset.ingest` builds `DailyRain` and `ForecastRain`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Union

import numpy as np


class Status(IntEnum):
    """The outcome of one onset search (stored as ``int8``)."""

    ONSET = 0
    """Onset was found, and every earlier candidate in the window was ruled out."""
    NOT_FOUND = 1
    """Every candidate in the window was ruled out (an empty window gives this too)."""
    CENSORED = 2
    """An earlier candidate could not be decided because it needs days past the data end; no onset day is reported."""
    PRIOR = 3
    """Onset had already happened before the search window opened; only the forecast layer sets this."""
    INSUFFICIENT_DATA = 4
    """An earlier candidate needs days with missing rainfall, or the cell has no threshold; no onset day is reported."""


Op = Literal["ge", "gt", "le", "lt"]
"""A comparison operator: greater or equal, greater, less or equal, less."""


@dataclass(frozen=True)
class Amount:
    """A rainfall amount: fixed tenths of a mm, a fixed finer amount in mm, or the name of a per-cell threshold.

    Exactly one of the three fields is set. `mm` is only set for amounts that are not a whole number of tenths; they
    need ``rain_resolution: exact``.
    """

    tenths: int | None = None
    """A fixed amount in tenths of a mm."""
    threshold: str | None = None
    """The name of a per-cell threshold (see `ThresholdSpec`)."""
    mm: float | None = None
    """A fixed amount in mm that is not a whole number of tenths."""

    def __post_init__(self):
        """Require exactly one of tenths, mm or a threshold name.

        Raises
        ------
        ValueError
            Not exactly one of the three fields is set.
        """
        if sum(v is not None for v in (self.tenths, self.threshold, self.mm)) != 1:
            raise ValueError("Amount needs exactly one of tenths, mm or threshold")

    @property
    def fixed_mm(self) -> float | None:
        """The fixed amount in mm, or None for a threshold."""
        return self.tenths / 10 if self.tenths is not None else self.mm


@dataclass(frozen=True)
class Comparison:
    """A comparator and the amount on its right-hand side, for example more than 20 mm."""

    op: Op
    """The comparison operator."""
    amount: Amount
    """The right-hand side."""


@dataclass(frozen=True)
class Span:
    """Days `first` to `last` (inclusive) relative to the candidate day.

    With `since_search_start` the span instead runs from the first day of the search window to `last`, and `first` is
    ignored.
    """

    first: int
    """The first day, 1-based relative to the candidate (day 1 is the candidate)."""
    last: int
    """The last day, inclusive."""
    since_search_start: bool = False
    """Run from the first day of the search window instead of from `first`."""

    def __post_init__(self):
        """Reject empty spans.

        Raises
        ------
        ValueError
            The span ends before it starts (and does not run from the search start).
        """
        if not self.since_search_start and self.last < self.first:
            raise ValueError(f"Span ends (day {self.last}) before it starts (day {self.first})")


@dataclass(frozen=True)
class Total:
    """A rainfall total over a span compared with an amount."""

    span: Span
    """The days summed."""
    comparison: Comparison
    """The test the total must pass."""


@dataclass(frozen=True)
class WetDays:
    """The number of wet days in a span compared with a count (all days wet is ``ge`` the span length)."""

    span: Span
    """The days counted."""
    op: Op
    """The comparison operator."""
    count: int
    """The number of wet days compared against."""


Condition = Union[Total, WetDays]
"""A requirement of a first-passage rule: a rainfall total or a wet-day count."""


@dataclass(frozen=True)
class DryRun:
    """A run of at least `length` consecutive dry days (count based)."""

    length: int
    """The least number of consecutive dry days."""
    phrase: Literal["at least", "more than"] = field(default="at least", compare=False)
    """How the definition wrote it (``more than 7 days`` is length 8), so generated sentences keep its wording."""


@dataclass(frozen=True)
class DryWindow:
    """A block of `days` consecutive days whose total satisfies a comparison (amount based)."""

    days: int
    """The block length in days."""
    comparison: Comparison
    """The test the block total must pass."""


@dataclass(frozen=True)
class Veto:
    """A rule that rejects a candidate when a dry spell lies wholly inside, starts inside, or begins inside a span.

    ``begins`` counts only a dry run whose previous day is not dry (the start of a maximal run).
    """

    spell: Union[DryRun, DryWindow]
    """The dry spell looked for."""
    span: Span
    """Where it is looked for."""
    placement: Literal["inside", "starts_inside", "begins"]
    """How the spell must lie in the span."""


@dataclass(frozen=True)
class FirstPassage:
    """A rule whose onset is the first candidate that meets every requirement and no veto.

    ``report="first_wet_day"`` reports the first wet day inside `report_span` rather than the candidate itself.
    """

    require: tuple[Condition, ...]
    """The conditions a candidate must all meet."""
    reject_if: tuple[Veto, ...] = ()
    """The vetoes, none of which may apply."""
    report: Literal["candidate", "first_wet_day"] = "candidate"
    """Which day is reported as the onset."""
    report_span: Span | None = None
    """Where to look for the first wet day, for ``report="first_wet_day"``."""


@dataclass(frozen=True)
class Curve:
    """A rule on the cumulative anomaly of daily rain minus a per-cell daily baseline, from the search start."""

    baseline: str
    """The name of the threshold that holds the daily baseline."""
    report: Literal["day_after_minimum", "day_of_minimum"] = "day_after_minimum"
    """Which day around the curve minimum is reported."""


@dataclass(frozen=True)
class DayRef:
    """A search-window bound: a fixed month-day, or a climatological onset or cessation date plus an offset."""

    month: int | None = None
    """The month (1-12) of a fixed bound."""
    day: int | None = None
    """The day of the month of a fixed bound."""
    climatological: Literal["onset", "cessation"] | None = None
    """Which climatological date the bound is, for a bound that is not fixed."""
    offset_days: int = 0
    """The signed offset in days, added to the date."""

    def __post_init__(self):
        """Require either a month-day or a climatological reference, not both.

        Raises
        ------
        ValueError
            Both or neither of a month-day and a climatological reference are given.
        """
        fixed = self.month is not None and self.day is not None
        if fixed == (self.climatological is not None):
            raise ValueError("DayRef needs either month and day, or climatological")


@dataclass(frozen=True)
class Season:
    """A named search window; an end month-day before the start falls in the following year."""

    name: str
    """The season's name."""
    start: DayRef
    """The first day of the window."""
    end: DayRef
    """The last day of the window."""


@dataclass(frozen=True)
class DerivedSeasons:
    """Search windows taken per cell from a climatological season analysis (one or two seasons a year)."""

    method: Literal["harmonic"]
    """The method of the season analysis."""
    start: DayRef
    """The earliest start of a search window."""
    end: DayRef
    """The latest end of a search window."""
    max_seasons: int = 2
    """The most seasons a year a cell may have."""


@dataclass(frozen=True)
class ThresholdSpec:
    """A per-cell amount computed from a reference climatology, in mm.

    * ``kind="mean_total"``: `days` x the mean daily rain within `during`.
    * ``kind="mean_daily"``: the mean daily rain within `during`.
    * ``kind="quantile_total"``: the `quantile` of all `days`-day running totals within `during`.

    ``during=None`` means the whole year.
    """

    kind: Literal["mean_total", "mean_daily", "quantile_total"]
    """How the threshold is computed."""
    days: int = 1
    """The number of days of the total (``mean_total`` and ``quantile_total``)."""
    during: tuple[DayRef, DayRef] | None = None
    """The period of the year the climatology is taken over, or None for the whole year."""
    quantile: float | None = None
    """The quantile (0 to 1) of ``quantile_total``."""


@dataclass(frozen=True)
class Definition:
    """A validated onset definition: provenance plus the rule every backend runs."""

    id: str
    """The definition's identifier."""
    version: int
    """The version of the definition."""
    title: str
    """A short title."""
    citation: str
    """The reference the definition comes from."""
    wording: str
    """The definition in words."""
    status: Literal["draft", "verified"]
    """Whether the definition has been checked against its source."""
    notes: str
    """Free-text notes."""
    wet_day: Comparison
    """The test that makes a day wet."""
    seasons: Union[tuple[Season, ...], DerivedSeasons]
    """The search windows."""
    rule: Union[FirstPassage, Curve]
    """The rule that finds the onset in a window."""
    thresholds: tuple[tuple[str, ThresholdSpec], ...] = ()
    """The named per-cell thresholds, as ``(name, spec)`` pairs."""
    regions: tuple[str, ...] = field(default=(), compare=False)
    """The ids of the regions the definition is meant for (see `hcwf.onset.regions`)."""
    source_yaml: str = field(default="", compare=False)
    """The YAML text the definition was parsed from."""
    dry_day: Comparison | None = None
    """The test that makes a day dry (``less than`` or ``at most`` a fixed amount), or None when a dry day is a valid
    day that is not wet."""

    @property
    def threshold_specs(self) -> dict[str, ThresholdSpec]:
        """The named thresholds as a dict."""
        return dict(self.thresholds)

    def windows(self) -> dict[str, int | None]:
        """Return the window lengths of a first-passage rule, from its spans.

        ``trigger`` is the longest requirement span starting at day 1, ``follow`` the longest veto span, and
        ``dry_run`` and ``dry_window`` the length of the first dry run and the first dry window veto.

        Returns
        -------
        dict of str to int or None
            The four lengths in days, None where absent; all None for a rule that is not a first-passage rule.
        """
        out: dict[str, int | None] = {"trigger": None, "follow": None, "dry_run": None, "dry_window": None}
        if not isinstance(self.rule, FirstPassage):
            return out
        firsts = [c.span.last for c in self.rule.require if c.span.first == 1 and not c.span.since_search_start]
        out["trigger"] = max(firsts, default=None)
        spans = [v.span.last - v.span.first + 1 for v in self.rule.reject_if]
        out["follow"] = max(spans, default=None)
        for veto in self.rule.reject_if:
            key, length = ("dry_run", veto.spell.length) if isinstance(veto.spell, DryRun) else (
                "dry_window", veto.spell.days)
            out[key] = out[key] or length
        return out


RESOLUTIONS = ("tenths", "exact")
"""The values of ``rain_resolution``."""
MISSING_POLICIES = ("flag", "zero", "drop", "keep")
"""The values of the missing-data policy."""
HORIZON_POLICIES = ("censor", "truncate", "reject")
"""The values of the horizon policy."""


def _check_dtype(values: np.ndarray) -> None:
    """Check that rain is int32 tenths of a mm, or float64 mm for ``rain_resolution: exact``.

    Parameters
    ----------
    values : numpy.ndarray
        The rain array.

    Raises
    ------
    TypeError
        The dtype is neither ``int32`` nor ``float64``.
    """
    if values.dtype not in (np.int32, np.float64):
        raise TypeError(f"tenths must be int32 (or float64 for exact resolution), got {values.dtype}")


def resolution_of(values: np.ndarray) -> str:
    """Return the rain resolution of an array.

    Parameters
    ----------
    values : numpy.ndarray
        The rain array, ``float64`` mm or ``int32`` tenths.

    Returns
    -------
    str
        ``"exact"`` for float rain, ``"tenths"`` for ``int32``.
    """
    return "exact" if values.dtype.kind == "f" else "tenths"


@dataclass(frozen=True, eq=False)
class Cells:
    """The cells of a (lat, lon) grid that a `DailyRain` holds when it holds only some: those of the dims (cell, time).

    A cell the rain lacks is one the region's masks drop: what ingestion gives it (`fill_tenths` mm or tenths,
    `fill_missing`, the same every day) is what ``on_grid`` and `spread` put there, so that results on the whole grid
    are the ones the whole grid gives.
    """

    flat: np.ndarray
    """The cells' positions in the grid, latitude-major (``row * number of longitudes + column``)."""
    lat: np.ndarray
    """The grid's latitudes."""
    lon: np.ndarray
    """The grid's longitudes."""
    dims: tuple[str, ...]
    """The grid's two spatial dims, in the order the whole grid has them."""
    fill_tenths: float
    """The rain value ingestion gives a cell the masks drop (tenths, or mm for exact rain)."""
    fill_missing: bool
    """The missing flag ingestion gives such a cell."""

    def _where(self) -> tuple[np.ndarray, ...]:
        """Return the index of each cell along the grid's dims.

        Returns
        -------
        tuple of numpy.ndarray of int64, shape (cell,)
            One index array per dim, in `dims` order.
        """
        by = {"lat": self.flat // len(self.lon), "lon": self.flat % len(self.lon)}
        return tuple(by[d] for d in self.dims)

    def shape(self) -> tuple[int, ...]:
        """Return the grid's shape in `dims` order.

        Returns
        -------
        tuple of int
            The length of each spatial dim.
        """
        return tuple(len({"lat": self.lat, "lon": self.lon}[d]) for d in self.dims)

    def coords(self) -> dict[str, np.ndarray]:
        """Return the grid's axes by dim name.

        Returns
        -------
        dict of str to numpy.ndarray
            The ``"lat"`` and ``"lon"`` coordinates.
        """
        return {"lat": self.lat, "lon": self.lon}

    def spread(self, kept: np.ndarray, fill: np.ndarray) -> np.ndarray:
        """Place per-cell values on the whole grid, giving every other cell a fill value.

        Parameters
        ----------
        kept : numpy.ndarray, shape (..., cell)
            The values of the cells held: leading dims, then the cells.
        fill : numpy.ndarray, shape (..., 1)
            What every other cell has: the leading dims and one.

        Returns
        -------
        numpy.ndarray
            ``kept`` on the whole grid: the leading dims, then the grid's dims in `dims` order; the dtype of ``kept``.
        """
        out = np.empty(kept.shape[:-1] + self.shape(), dtype=kept.dtype)
        out[...] = fill.reshape(fill.shape[:-1] + (1,) * len(self.dims))
        out[(...,) + self._where()] = kept
        return out

    def masked(self, rain: "DailyRain") -> "DailyRain":
        """Return one cell of the rain's days that the masks dropped, as ingestion gives it.

        Parameters
        ----------
        rain : DailyRain
            The rain whose calendar and labels the cell takes.

        Returns
        -------
        DailyRain
            A rain of dims ``(cell, time)`` with one cell, holding `fill_tenths` and `fill_missing` every day.
        """
        n = rain.tenths.shape[-1]
        return DailyRain(tenths=np.full((1, n), self.fill_tenths, dtype=rain.tenths.dtype),
                         missing=np.full((1, n), self.fill_missing), dims=("cell", "time"), coords={},
                         year=rain.year, month=rain.month, day=rain.day, calendar=rain.calendar,
                         day_boundary=rain.day_boundary, source=rain.source, missing_policy=rain.missing_policy)


@dataclass
class DailyRain:
    """Daily rainfall that has passed the ingestion contract, ready for detection.

    `tenths` and `missing` have dims `dims`, with ``"time"`` last. Missing days hold 0 in `tenths`. `year`, `month`
    and `day` label each day in the dataset's own calendar, so search windows resolve without datetime arithmetic.
    With resolution ``"exact"`` `tenths` holds ``float64`` millimetres instead of ``int32`` tenths (see
    `hcwf.onset.ingest`, ``rain_resolution``). With `cells` the rain holds only some cells of a grid, as the dims
    (cell, time) (see `Cells`); `on_grid` gives the rain of the whole grid.
    """

    tenths: np.ndarray
    """The rain: ``int32`` tenths of a mm, or ``float64`` mm for exact resolution; the shape follows `dims`."""
    missing: np.ndarray
    """``bool`` flags, True where the day is missing; the shape of `tenths`."""
    dims: tuple[str, ...]
    """The dims of `tenths`, ``"time"`` last."""
    coords: dict[str, np.ndarray]
    """The coordinates by dim name."""
    year: np.ndarray
    """The year of each time step, shape (time,)."""
    month: np.ndarray
    """The month of each time step, shape (time,)."""
    day: np.ndarray
    """The day of the month of each time step, shape (time,)."""
    calendar: str
    """The name of the dataset's calendar (``standard``, ``noleap``, ...)."""
    day_boundary: str
    """The day boundary of the data, for example ``"08:30 IST, labelled by end"`` (see `hcwf.onset.ingest`)."""
    source: str = ""
    """A label of where the data came from."""
    missing_policy: str = "flag"
    """The missing-data policy applied at ingestion (one of `MISSING_POLICIES`)."""
    cells: Cells | None = None
    """The cells held, when the rain holds only some cells of a grid."""

    def __post_init__(self):
        """Check the arrays agree with each other and with the contract.

        Raises
        ------
        ValueError
            ``time`` is not the last dim, `missing` is not a bool array shaped like `tenths`, or the labels do not
            cover every time step.
        """
        if self.dims[-1] != "time":
            raise ValueError(f"time must be the last dim, got {self.dims}")
        _check_dtype(self.tenths)
        if self.missing.shape != self.tenths.shape or self.missing.dtype != bool:
            raise ValueError("missing must be a bool array shaped like tenths")
        n_time = self.tenths.shape[-1]
        if not (len(self.year) == len(self.month) == len(self.day) == n_time):
            raise ValueError("year/month/day must label every time step")

    @property
    def resolution(self) -> str:
        """The resolution of the rain: ``"exact"`` (float mm) or ``"tenths"`` (``int32``)."""
        return resolution_of(self.tenths)

    def on_grid(self) -> "DailyRain":
        """Return the rain of the whole grid, with the cells it lacks as ingestion gives them.

        Returns
        -------
        DailyRain
            This rain itself when it holds no `cells`, else a new one with the spatial dims of the grid.
        """
        cells = self.cells
        if cells is None:
            return self
        shape = cells.shape() + (self.tenths.shape[-1],)
        tenths = np.full(shape, cells.fill_tenths, dtype=self.tenths.dtype)
        missing = np.full(shape, cells.fill_missing)
        where = cells._where()
        tenths[where], missing[where] = self.tenths, self.missing
        coords = {**cells.coords(), "time": self.coords["time"]}
        return DailyRain(tenths=tenths, missing=missing, dims=cells.dims + ("time",), coords=coords, year=self.year,
                         month=self.month, day=self.day, calendar=self.calendar, day_boundary=self.day_boundary,
                         source=self.source, missing_policy=self.missing_policy)


@dataclass
class ForecastRain:
    """Daily forecast rainfall that has passed the ingestion contract.

    `tenths` and `missing` have dims `dims`: ``"init_time"``, optionally ``"member"``, the spatial dims, then
    ``"lead"`` last. Lead k (1-based) is the k-th 24 h after the init; its date label is init + k - 1 days for
    `lead_label` ``"start"`` and init + k for ``"end"``. `init_year`, `init_month` and `init_day` label each init in
    `calendar`.
    """

    tenths: np.ndarray
    """The rain: ``int32`` tenths of a mm, or ``float64`` mm for exact resolution; the shape follows `dims`."""
    missing: np.ndarray
    """``bool`` flags, True where the value is missing; the shape of `tenths`."""
    dims: tuple[str, ...]
    """The dims of `tenths`, ``"init_time"`` first and ``"lead"`` last."""
    coords: dict[str, np.ndarray]
    """The coordinates by dim name."""
    init_year: np.ndarray
    """The year of each init, shape (init_time,)."""
    init_month: np.ndarray
    """The month of each init, shape (init_time,)."""
    init_day: np.ndarray
    """The day of the month of each init, shape (init_time,)."""
    lead_label: Literal["start", "end"]
    """Whether a lead is dated by the start or the end of its 24 h."""
    calendar: str
    """The name of the calendar of the init labels."""
    day_boundary: str
    """The day boundary of the data, for example ``"08:30 IST, labelled by end"`` (see `hcwf.onset.ingest`)."""
    source: str = ""
    """A label of where the data came from."""
    missing_policy: str = "flag"
    """The missing-data policy applied at ingestion (one of `MISSING_POLICIES`)."""

    def __post_init__(self):
        """Check the arrays agree with each other and with the contract.

        Raises
        ------
        ValueError
            `dims` does not start with ``init_time`` and end with ``lead``, `missing` is not a bool array shaped like
            `tenths`, the init labels do not cover every init, or `lead_label` is not ``"start"`` or ``"end"``.
        """
        if self.dims[0] != "init_time" or self.dims[-1] != "lead":
            raise ValueError(f"dims must start with init_time and end with lead, got {self.dims}")
        _check_dtype(self.tenths)
        if self.missing.shape != self.tenths.shape or self.missing.dtype != bool:
            raise ValueError("missing must be a bool array shaped like tenths")
        n_init = self.tenths.shape[0]
        if not (len(self.init_year) == len(self.init_month) == len(self.init_day) == n_init):
            raise ValueError("init_year/month/day must label every init")
        if self.lead_label not in ("start", "end"):
            raise ValueError(f"lead_label must be 'start' or 'end', got {self.lead_label!r}")


@dataclass
class Jobs:
    """One onset search per row: which series, the candidate days, and where the data ends.

    `start` and `end` are inclusive day indices of the first and last candidate; `data_end` is exclusive (the series
    has no usable data at or after it).
    """

    series: np.ndarray
    """The row of the rain array each job searches, shape (job,)."""
    start: np.ndarray
    """The first candidate day index, inclusive, shape (job,)."""
    end: np.ndarray
    """The last candidate day index, inclusive, shape (job,)."""
    data_end: np.ndarray
    """The day index at which usable data ends, exclusive, shape (job,)."""
    origin: np.ndarray | None = None
    """The day index ``season to date`` spans start from, shape (job,); None means `start`.

    Only the per-window threshold split sets it (spec section 13).
    """


@dataclass
class StitchedJobs:
    """One onset search per row over observed history followed by a forecast (spec section 13).

    Stitched day t is history day t of row `cell` for t < `split`, and forecast column t - `split` of row `series`
    (lead t - `split` + 1) for t >= `split`. `start` and `end` are inclusive candidate days and `data_end` the
    exclusive end of usable data, all in stitched days. History days at or after `split` are never read, so nothing
    observed after the init leaks into the forecast.
    """

    cell: np.ndarray
    """The history row of each job, shape (job,)."""
    series: np.ndarray
    """The forecast row of each job, shape (job,)."""
    split: np.ndarray
    """The stitched day at which the forecast begins, shape (job,)."""
    start: np.ndarray
    """The first candidate day, inclusive, in stitched days, shape (job,)."""
    end: np.ndarray
    """The last candidate day, inclusive, in stitched days, shape (job,)."""
    data_end: np.ndarray
    """The stitched day at which usable data ends, exclusive, shape (job,)."""
    origin: np.ndarray | None = None
    """The stitched day ``season to date`` spans start from, shape (job,); None means `start`."""


@dataclass
class Detection:
    """The backend output, one entry per job: the onset day index (-1 if none) and its `Status`."""

    day: np.ndarray
    """The onset day index, -1 where there is none, shape (job,)."""
    status: np.ndarray
    """The `Status` of each job, ``int8``, shape (job,)."""
    trace: dict | None = None
    """Per-job diagnostics, recorded only by a backend that is asked to trace."""
