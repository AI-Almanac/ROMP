"""Plain-English sentence for an onset definition, recorded with every result."""
from __future__ import annotations

from momp.onset.model import (
    Amount, Comparison, Curve, DayRef, Definition, DerivedSeasons, DryRun, FirstPassage, Span, ThresholdSpec,
    Total, Veto, WetDays,
)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
_OP_WORDS = {"ge": "at least", "gt": "more than", "le": "at most", "lt": "less than"}
_NEGATED = {"ge": "lt", "gt": "le", "le": "gt", "lt": "ge"}


def describe(defn: Definition) -> str:
    """Return one or two plain-English sentences that say what the definition detects."""
    names = defn.threshold_specs
    if isinstance(defn.rule, Curve):
        return _describe_curve(defn, names)
    return _describe_first_passage(defn, defn.rule, names)


def _date(ref: DayRef) -> str:
    """Return a bound as '1 March' or 'the climatological onset' with its offset."""
    if ref.climatological is None:
        return f"{ref.day} {_MONTHS[ref.month - 1]}"
    base = f"the climatological {ref.climatological}"
    if ref.offset_days == 0:
        return base
    unit = "day" if abs(ref.offset_days) == 1 else "days"
    return f"{abs(ref.offset_days)} {unit} {'after' if ref.offset_days > 0 else 'before'} {base}"


def _season_phrase(defn: Definition) -> str:
    """Return where the search runs, e.g. 'from 1 March to 31 May'."""
    seasons = defn.seasons
    if isinstance(seasons, DerivedSeasons):
        return ("in each rainy season found at every location by harmonic analysis of the mean annual "
                f"cycle, from {_date(seasons.start)} to {_date(seasons.end)}")
    parts = []
    for season in seasons:
        wraps = (season.end.month, season.end.day) < (season.start.month, season.start.day)
        parts.append(f"from {_date(season.start)} to {_date(season.end)}{' of the following year' if wraps else ''}")
    if len(parts) == 1:
        return parts[0]
    named = [f"{s.name} ({p})" for s, p in zip(seasons, parts)]
    return f"in each of the seasons {', '.join(named[:-1])} and {named[-1]}"


def _during(spec: ThresholdSpec) -> str:
    """Return the reference period of a threshold."""
    if spec.during is None:
        return "over the whole year"
    return f"from {_date(spec.during[0])} to {_date(spec.during[1])}"


def _threshold_text(name: str, spec: ThresholdSpec | None) -> str:
    """Return what a named threshold is, e.g. 'the mean 5-day total from 1 April to 31 October'."""
    if spec is None:
        return name
    if spec.kind == "mean_total":
        return f"the mean {spec.days}-day total {_during(spec)}"
    if spec.kind == "mean_daily":
        return f"the mean daily rain {_during(spec)}"
    return f"the {spec.quantile:g} quantile of {spec.days}-day totals {_during(spec)}"


def _amount(amount: Amount, names: dict[str, ThresholdSpec]) -> str:
    """Return an amount as '20 mm' or 'name (what it is)'."""
    if amount.tenths is not None:
        return f"{amount.tenths / 10:g} mm"
    return f"{amount.threshold} ({_threshold_text(amount.threshold, names.get(amount.threshold))})"


def _comparison(comparison: Comparison, names: dict[str, ThresholdSpec]) -> str:
    """Return a comparison as 'more than 20 mm'."""
    return f"{_OP_WORDS[comparison.op]} {_amount(comparison.amount, names)}"


def _days_text(span: Span) -> str:
    """Return the days of a span counted from the candidate (day 1)."""
    if span.first == span.last:
        return f"day {span.first}"
    return f"days {span.first} to {span.last}"


def _window_of(spans: list[Span], first: int) -> int | None:
    """Return W when a span starting at day first follows a trigger window of W days (days 1 to W)."""
    return first - 1 if first > 2 and any(s.first == 1 and s.last == first - 1 for s in spans) else None


def _rule_spans(rule: FirstPassage) -> list[Span]:
    """Return every span used by the requirements."""
    return [c.span for c in rule.require]


def _veto_span(span: Span, window: int | None) -> str:
    """Return where a spell must lie, e.g. 'the next 21 days'."""
    length = span.last - span.first + 1
    if span.first == 2:
        return f"the next {length} days"
    if window is not None:
        return f"the {length} days after the {window}-day window"
    return f"{_days_text(span)} (counting the onset day as day 1)"


class _WetNotes:
    """Defines 'wet day' and 'dry day' on their first mention only."""

    def __init__(self, wet_day: Comparison):
        """Remember the wet-day comparison."""
        self.wet_day = wet_day
        self.wet_defined = False
        self.dry_defined = False

    def wet(self) -> str:
        """Return the wet-day definition the first time it is needed, else nothing."""
        if self.wet_defined:
            return ""
        self.wet_defined = True
        return f" ({_comparison(self.wet_day, {})})"

    def dry(self) -> str:
        """Return the dry-day definition the first time it is needed, else nothing."""
        if self.dry_defined or self.wet_defined:
            return ""
        self.dry_defined = True
        negated = Comparison(_NEGATED[self.wet_day.op], self.wet_day.amount)
        return f" (a dry day has {_comparison(negated, {})})"


def _condition(cond: Total | WetDays, notes: _WetNotes, names: dict[str, ThresholdSpec]) -> str:
    """Describe one requirement as a clause."""
    span = cond.span
    if isinstance(cond, Total):
        text = _comparison(cond.comparison, names)
        if span.since_search_start:
            return f"for which the total rain from the start of the search window to that day is {text}"
        if span.first == 1:
            return f"on which the {span.last}-day total is {text}"
        return f"for which the total over {_days_text(span)} (counting that day as day 1) is {text}"
    length = span.last - span.first + 1
    if span.first == span.last == 1 and cond.op == "ge" and cond.count == 1:
        return f"that is a wet day{notes.wet()}"
    if cond.op == "ge" and cond.count == length:
        return f"that begins {length} consecutive wet days{notes.wet()}" if span.first == 1 else (
            f"for which every day over {_days_text(span)} is wet{notes.wet()}")
    return (f"that has {_OP_WORDS[cond.op]} {cond.count} wet days{notes.wet()} over {_days_text(span)} "
            "(counting that day as day 1)")


def _dry_run_text(run: DryRun) -> str:
    """Return a dry run in the words the definition used."""
    if run.phrase == "more than":
        return f"a dry run of more than {run.length - 1} days"
    return f"a dry run of at least {run.length} days"


def _veto(veto: Veto, rule_spans: list[Span], notes: _WetNotes, names: dict[str, ThresholdSpec]) -> str:
    """Describe one veto as a clause."""
    if isinstance(veto.spell, DryRun):
        spell = _dry_run_text(veto.spell) + notes.dry()
    else:
        spell = f"a dry window of {veto.spell.days} days totalling {_comparison(veto.spell.comparison, names)}"
    where = _veto_span(veto.span, _window_of(rule_spans, veto.span.first))
    if veto.placement == "inside":
        return f"that is not followed by {spell} within {where}"
    return f"that is not followed by {spell} that starts within {where}"


def _join(clauses: list[str]) -> str:
    """Join clauses as 'a, b, and c'."""
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return f"{', '.join(clauses[:-1])}, and {clauses[-1]}"


def _describe_first_passage(defn: Definition, rule: FirstPassage, names: dict[str, ThresholdSpec]) -> str:
    """Describe a first-passage rule."""
    notes = _WetNotes(defn.wet_day)
    spans = _rule_spans(rule)
    clauses = [_condition(c, notes, names) for c in rule.require]
    clauses += [_veto(v, spans, notes, names) for v in rule.reject_if]
    sentence = f"Onset is the first day {_season_phrase(defn)} {_join(clauses)}."
    if rule.report == "first_wet_day":
        span = rule.report_span
        sentence += f" The onset date reported is the first wet day within {_days_text(span)}."
    return sentence


def _describe_curve(defn: Definition, names: dict[str, ThresholdSpec]) -> str:
    """Describe a curve rule."""
    rule = defn.rule
    which = "day after" if rule.report == "day_after_minimum" else "day of"
    baseline = f"{_threshold_text(rule.baseline, names.get(rule.baseline))} ({rule.baseline})"
    return (f"Onset is the {which} the minimum of the cumulative rainfall anomaly (daily rain minus "
            f"{baseline}), accumulated from the start of each search window ({_season_phrase(defn)}).")
