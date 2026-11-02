"""Validated, unit-free onset definitions and the containers shared by every part of momp.onset.

Everything here is already normalized: rainfall amounts are integer tenths of a
millimetre, durations are whole days, and positions are 1-based days relative
to the candidate day (day 1 is the candidate itself, day 0 the day before).
docs/onset_spec.md defines what each field means; parsing YAML into these
types lives in momp.onset.definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Union

import numpy as np


class Status(IntEnum):
    """Outcome of one onset search (stored as int8)."""

    ONSET = 0
    NOT_FOUND = 1
    CENSORED = 2
    PRIOR = 3
    INSUFFICIENT_DATA = 4


Op = Literal["ge", "gt", "le", "lt"]


@dataclass(frozen=True)
class Amount:
    """A rainfall amount: fixed tenths of a mm, a fixed finer amount in mm, or the name of a per-cell threshold.

    mm is only set for amounts that are not a whole number of tenths; they need rain_resolution "exact".
    """

    tenths: int | None = None
    threshold: str | None = None
    mm: float | None = None

    def __post_init__(self):
        """Require exactly one of tenths, mm or a threshold name."""
        if sum(v is not None for v in (self.tenths, self.threshold, self.mm)) != 1:
            raise ValueError("Amount needs exactly one of tenths, mm or threshold")

    @property
    def fixed_mm(self) -> float | None:
        """The fixed amount in mm, or None for a threshold."""
        return self.tenths / 10 if self.tenths is not None else self.mm


@dataclass(frozen=True)
class Comparison:
    """A comparator and the amount on its right-hand side, e.g. more than 20 mm."""

    op: Op
    amount: Amount


@dataclass(frozen=True)
class Span:
    """Days first..last (inclusive) relative to the candidate day.

    With since_search_start the span instead runs from the first day of the
    search window to `last`, and `first` is ignored.
    """

    first: int
    last: int
    since_search_start: bool = False

    def __post_init__(self):
        """Reject empty spans."""
        if not self.since_search_start and self.last < self.first:
            raise ValueError(f"Span ends (day {self.last}) before it starts (day {self.first})")


@dataclass(frozen=True)
class Total:
    """Rainfall total over a span compared with an amount."""

    span: Span
    comparison: Comparison


@dataclass(frozen=True)
class WetDays:
    """Number of wet days in a span compared with a count ("all days wet" is ge span length)."""

    span: Span
    op: Op
    count: int


Condition = Union[Total, WetDays]


@dataclass(frozen=True)
class DryRun:
    """A run of at least `length` consecutive dry days (count based).

    phrase only records how the definition wrote it ("more than 7 days" is
    length 8), so generated sentences can keep the source's wording.
    """

    length: int
    phrase: Literal["at least", "more than"] = field(default="at least", compare=False)


@dataclass(frozen=True)
class DryWindow:
    """A block of `days` consecutive days whose total satisfies a comparison (amount based)."""

    days: int
    comparison: Comparison


@dataclass(frozen=True)
class Veto:
    """Reject a candidate when a dry spell lies wholly inside, starts inside, or begins inside a span.

    "begins" counts only a dry run whose previous day is not dry (the start of a maximal run).
    """

    spell: Union[DryRun, DryWindow]
    span: Span
    placement: Literal["inside", "starts_inside", "begins"]


@dataclass(frozen=True)
class FirstPassage:
    """Onset is the first candidate that meets every requirement and no veto.

    report="first_wet_day" reports the first wet day inside report_span rather
    than the candidate itself.
    """

    require: tuple[Condition, ...]
    reject_if: tuple[Veto, ...] = ()
    report: Literal["candidate", "first_wet_day"] = "candidate"
    report_span: Span | None = None


@dataclass(frozen=True)
class Curve:
    """Cumulative anomaly of daily rain minus a per-cell daily baseline, accumulated from the search start."""

    baseline: str
    report: Literal["day_after_minimum", "day_of_minimum"] = "day_after_minimum"


@dataclass(frozen=True)
class DayRef:
    """A search-window bound: a fixed month-day, or a climatological onset/cessation date plus an offset."""

    month: int | None = None
    day: int | None = None
    climatological: Literal["onset", "cessation"] | None = None
    offset_days: int = 0

    def __post_init__(self):
        """Require either a month-day or a climatological reference, not both."""
        fixed = self.month is not None and self.day is not None
        if fixed == (self.climatological is not None):
            raise ValueError("DayRef needs either month and day, or climatological")


@dataclass(frozen=True)
class Season:
    """A named search window; an end month-day before the start falls in the following year."""

    name: str
    start: DayRef
    end: DayRef


@dataclass(frozen=True)
class DerivedSeasons:
    """Search windows taken per cell from a climatological season analysis (one or two seasons a year)."""

    method: Literal["harmonic"]
    start: DayRef
    end: DayRef
    max_seasons: int = 2


@dataclass(frozen=True)
class ThresholdSpec:
    """A per-cell amount computed from a reference climatology, in mm.

    kind="mean_total": `days` x the mean daily rain within `during`.
    kind="mean_daily": the mean daily rain within `during`.
    kind="quantile_total": the `quantile` of all `days`-day running totals within `during`.
    during=None means the whole year.
    """

    kind: Literal["mean_total", "mean_daily", "quantile_total"]
    days: int = 1
    during: tuple[DayRef, DayRef] | None = None
    quantile: float | None = None


@dataclass(frozen=True)
class Definition:
    """A validated onset definition: provenance plus the rule every backend runs."""

    id: str
    version: int
    title: str
    citation: str
    wording: str
    status: Literal["draft", "verified"]
    notes: str
    wet_day: Comparison
    seasons: Union[tuple[Season, ...], DerivedSeasons]
    rule: Union[FirstPassage, Curve]
    thresholds: tuple[tuple[str, ThresholdSpec], ...] = ()
    regions: tuple[str, ...] = field(default=(), compare=False)
    source_yaml: str = field(default="", compare=False)
    dry_day: Comparison | None = None

    @property
    def threshold_specs(self) -> dict[str, ThresholdSpec]:
        """The named thresholds as a dict."""
        return dict(self.thresholds)

    def windows(self) -> dict[str, int | None]:
        """Window lengths of a first-passage rule, from its spans (None where absent).

        trigger: the longest requirement span starting at day 1; follow: the longest veto span;
        dry_run and dry_window: the length of the first dry run and the first dry window veto.
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
MISSING_POLICIES = ("flag", "zero", "drop", "keep")
HORIZON_POLICIES = ("censor", "truncate", "reject")


def _check_dtype(values: np.ndarray) -> None:
    """Rain is int32 tenths of a mm, or float64 mm for rain_resolution "exact"."""
    if values.dtype not in (np.int32, np.float64):
        raise TypeError(f"tenths must be int32 (or float64 for exact resolution), got {values.dtype}")


def resolution_of(values: np.ndarray) -> str:
    """"exact" for float rain, "tenths" for int32."""
    return "exact" if values.dtype.kind == "f" else "tenths"


@dataclass
class DailyRain:
    """Daily rainfall that has passed the ingestion contract, ready for detection.

    tenths and missing have dims `dims`, with "time" last. Missing days hold 0 in
    tenths. year/month/day label each day in the dataset's own calendar, so
    search windows resolve without datetime arithmetic. With resolution "exact"
    tenths holds float64 millimetres instead of int32 tenths (see momp.onset.ingest, rain_resolution).
    """

    tenths: np.ndarray
    missing: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    calendar: str
    day_boundary: str
    source: str = ""
    missing_policy: str = "flag"

    def __post_init__(self):
        """Check the arrays agree with each other and with the contract."""
        if self.dims[-1] != "time":
            raise ValueError(f"time must be the last dim, got {self.dims}")
        _check_dtype(self.tenths)
        if self.missing.shape != self.tenths.shape or self.missing.dtype != bool:
            raise ValueError("missing must be a bool array shaped like tenths")
        n_time = self.tenths.shape[-1]
        if not (len(self.year) == len(self.month) == len(self.day) == n_time):
            raise ValueError("year/month/day must label every time step")


@dataclass
class ForecastRain:
    """Daily forecast rainfall that has passed the ingestion contract.

    tenths and missing have dims `dims`: "init_time", optionally "member", the
    spatial dims, then "lead" last. Lead k (1-based) is the k-th 24 h after the
    init; its date label is init + k - 1 days for lead_label "start" and
    init + k for "end". init_year/month/day label each init in `calendar`.
    """

    tenths: np.ndarray
    missing: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    init_year: np.ndarray
    init_month: np.ndarray
    init_day: np.ndarray
    lead_label: Literal["start", "end"]
    calendar: str
    day_boundary: str
    source: str = ""
    missing_policy: str = "flag"

    def __post_init__(self):
        """Check the arrays agree with each other and with the contract."""
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

    start and end are inclusive day indices of the first and last candidate;
    data_end is exclusive (the series has no usable data at or after it).
    origin, when given, is the day "season to date" spans start from; it
    defaults to start. Only the per-window threshold split sets it (spec
    section 13).
    """

    series: np.ndarray
    start: np.ndarray
    end: np.ndarray
    data_end: np.ndarray
    origin: np.ndarray | None = None


@dataclass
class StitchedJobs:
    """One onset search per row over observed history followed by a forecast (spec section 13).

    Stitched day t is history day t of row `cell` for t < split, and forecast
    column t - split of row `series` (lead t - split + 1) for t >= split.
    start and end are inclusive candidate days and data_end the exclusive end of
    usable data, all in stitched days. History days at or after split are never
    read, so nothing observed after the init leaks into the forecast.
    """

    cell: np.ndarray
    series: np.ndarray
    split: np.ndarray
    start: np.ndarray
    end: np.ndarray
    data_end: np.ndarray
    origin: np.ndarray | None = None


@dataclass
class Detection:
    """Backend output, one entry per job: the onset day index (-1 if none) and its Status."""

    day: np.ndarray
    status: np.ndarray
    trace: dict | None = None
