"""A plain-English sentence for an onset definition, recorded with every result.

`describe` is the entry point; the other functions build the clauses of the sentence from the types of
`hcwf.onset.model`.
"""
from __future__ import annotations

from hcwf.onset.model import (
    Amount, Comparison, Curve, DayRef, Definition, DerivedSeasons, DryRun, FirstPassage, Span, ThresholdSpec,
    Total, Veto, WetDays,
)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
_OP_WORDS = {"ge": "at least", "gt": "more than", "le": "at most", "lt": "less than"}
_NEGATED = {"ge": "lt", "gt": "le", "le": "gt", "lt": "ge"}


def describe(defn: Definition) -> str:
    """Return one or two plain-English sentences that say what the definition detects.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition to describe.

    Returns
    -------
    str
        The sentence for a first-passage rule or a curve rule.
    """
    names = defn.threshold_specs
    if isinstance(defn.rule, Curve):
        return _describe_curve(defn, names)
    return _describe_first_passage(defn, defn.rule, names)


def _date(ref: DayRef) -> str:
    """Return a bound as ``1 March`` or ``the climatological onset`` with its offset.

    Parameters
    ----------
    ref : hcwf.onset.model.DayRef
        The bound.

    Returns
    -------
    str
        The date in words, for example ``"3 days before the climatological onset"``.
    """
    if ref.climatological is None:
        return f"{ref.day} {_MONTHS[ref.month - 1]}"
    base = f"the climatological {ref.climatological}"
    if ref.offset_days == 0:
        return base
    unit = "day" if abs(ref.offset_days) == 1 else "days"
    return f"{abs(ref.offset_days)} {unit} {'after' if ref.offset_days > 0 else 'before'} {base}"


def _season_phrase(defn: Definition) -> str:
    """Return where the search runs, for example ``from 1 March to 31 May``.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition whose seasons are described.

    Returns
    -------
    str
        The phrase for one season, several named seasons or derived seasons.
    """
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
    """Return the reference period of a threshold.

    Parameters
    ----------
    spec : hcwf.onset.model.ThresholdSpec
        The threshold.

    Returns
    -------
    str
        ``"over the whole year"`` or ``"from <date> to <date>"``.
    """
    if spec.during is None:
        return "over the whole year"
    return f"from {_date(spec.during[0])} to {_date(spec.during[1])}"


def _threshold_text(name: str, spec: ThresholdSpec | None) -> str:
    """Return what a named threshold is, for example ``the mean 5-day total from 1 April to 31 October``.

    Parameters
    ----------
    name : str
        The threshold's name.
    spec : hcwf.onset.model.ThresholdSpec or None
        Its specification, or None when the definition has none.

    Returns
    -------
    str
        The description, or `name` itself when `spec` is None.
    """
    if spec is None:
        return name
    if spec.kind == "mean_total":
        return f"the mean {spec.days}-day total {_during(spec)}"
    if spec.kind == "mean_daily":
        return f"the mean daily rain {_during(spec)}"
    return f"the {spec.quantile:g} quantile of {spec.days}-day totals {_during(spec)}"


def _amount(amount: Amount, names: dict[str, ThresholdSpec]) -> str:
    """Return an amount as ``20 mm`` or ``name (what it is)``.

    Parameters
    ----------
    amount : hcwf.onset.model.Amount
        The amount.
    names : dict of str to hcwf.onset.model.ThresholdSpec
        The thresholds of the definition by name.

    Returns
    -------
    str
        The fixed amount in mm, or the threshold's name with its description.
    """
    if amount.threshold is None:
        return f"{amount.fixed_mm:g} mm"
    return f"{amount.threshold} ({_threshold_text(amount.threshold, names.get(amount.threshold))})"


def _comparison(comparison: Comparison, names: dict[str, ThresholdSpec]) -> str:
    """Return a comparison as ``more than 20 mm``.

    Parameters
    ----------
    comparison : hcwf.onset.model.Comparison
        The comparison.
    names : dict of str to hcwf.onset.model.ThresholdSpec
        The thresholds of the definition by name.

    Returns
    -------
    str
        The operator in words followed by the amount.
    """
    return f"{_OP_WORDS[comparison.op]} {_amount(comparison.amount, names)}"


def _days_text(span: Span) -> str:
    """Return the days of a span counted from the candidate (day 1).

    Parameters
    ----------
    span : hcwf.onset.model.Span
        The span.

    Returns
    -------
    str
        ``"day N"`` or ``"days A to B"``.
    """
    if span.first == span.last:
        return f"day {span.first}"
    return f"days {span.first} to {span.last}"


def _window_of(defn: Definition, first: int) -> int | None:
    """Return W when a span starting at day `first` follows the trigger window of W days (days 1 to W).

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition, whose trigger window is read.
    first : int
        The first day of the span.

    Returns
    -------
    int or None
        The trigger window length W in days, or None if the span does not start right after it.
    """
    window = defn.windows()["trigger"]
    return window if window is not None and first > 2 and first == window + 1 else None


def _veto_span(span: Span, window: int | None) -> str:
    """Return where a spell must lie, for example ``the next 21 days``.

    Parameters
    ----------
    span : hcwf.onset.model.Span
        The span of the veto.
    window : int or None
        The trigger window length W, or None when the span does not follow it.

    Returns
    -------
    str
        The span in words.
    """
    length = span.last - span.first + 1
    if span.first == 2:
        return f"the next {length} days"
    if window is not None:
        return f"the {length} days after the {window}-day window"
    return f"{_days_text(span)} (counting the onset day as day 1)"


class _WetNotes:
    """A helper that defines ``wet day`` and ``dry day`` on their first mention only.

    Parameters
    ----------
    wet_day : hcwf.onset.model.Comparison
        The definition's wet-day test.

    Attributes
    ----------
    wet_day : hcwf.onset.model.Comparison
        The wet-day test.
    wet_defined : bool
        Whether the wet day has been defined in the sentence yet.
    dry_defined : bool
        Whether the dry day has been defined in the sentence yet.
    """

    def __init__(self, wet_day: Comparison):
        self.wet_day = wet_day
        self.wet_defined = False
        self.dry_defined = False

    def wet(self) -> str:
        """Return the wet-day definition the first time it is needed, else nothing.

        Returns
        -------
        str
            ``" (<wet-day test>)"`` on the first call, ``""`` after.
        """
        if self.wet_defined:
            return ""
        self.wet_defined = True
        return f" ({_comparison(self.wet_day, {})})"

    def dry(self) -> str:
        """Return the dry-day definition the first time it is needed, else nothing.

        Returns
        -------
        str
            ``" (a dry day has <negated wet-day test>)"`` on the first call unless the wet day has been defined,
            ``""`` otherwise.
        """
        if self.dry_defined or self.wet_defined:
            return ""
        self.dry_defined = True
        negated = Comparison(_NEGATED[self.wet_day.op], self.wet_day.amount)
        return f" (a dry day has {_comparison(negated, {})})"


def _condition(cond: Total | WetDays, notes: _WetNotes, names: dict[str, ThresholdSpec]) -> str:
    """Describe one requirement as a clause.

    Parameters
    ----------
    cond : hcwf.onset.model.Total or hcwf.onset.model.WetDays
        The requirement.
    notes : _WetNotes
        Defines the wet day on first use.
    names : dict of str to hcwf.onset.model.ThresholdSpec
        The thresholds of the definition by name.

    Returns
    -------
    str
        The clause, to follow ``"the first day ... "``.
    """
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
    """Return a dry run in the words the definition used.

    Parameters
    ----------
    run : hcwf.onset.model.DryRun
        The dry run.

    Returns
    -------
    str
        ``"a dry run of at least N days"`` or ``"a dry run of more than N days"``.
    """
    if run.phrase == "more than":
        return f"a dry run of more than {run.length - 1} days"
    return f"a dry run of at least {run.length} days"


def _veto(veto: Veto, defn: Definition, notes: _WetNotes, names: dict[str, ThresholdSpec]) -> str:
    """Describe one veto as a clause.

    Parameters
    ----------
    veto : hcwf.onset.model.Veto
        The veto.
    defn : hcwf.onset.model.Definition
        The definition, whose trigger window locates the veto span.
    notes : _WetNotes
        Defines the dry day on first use.
    names : dict of str to hcwf.onset.model.ThresholdSpec
        The thresholds of the definition by name.

    Returns
    -------
    str
        The clause, starting ``"that is not followed by"``.
    """
    if isinstance(veto.spell, DryRun):
        spell = _dry_run_text(veto.spell) + notes.dry()
    else:
        spell = f"a dry window of {veto.spell.days} days totalling {_comparison(veto.spell.comparison, names)}"
    where = _veto_span(veto.span, _window_of(defn, veto.span.first))
    if veto.placement == "inside":
        return f"that is not followed by {spell} within {where}"
    if veto.placement == "begins":
        return f"that is not followed by {spell} that begins within {where} (a run already under way does not count)"
    return f"that is not followed by {spell} that starts within {where}"


def _join(clauses: list[str]) -> str:
    """Join clauses as ``a, b, and c``.

    Parameters
    ----------
    clauses : list of str
        The clauses, at least one.

    Returns
    -------
    str
        The clauses joined with commas and ``and``.
    """
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return f"{', '.join(clauses[:-1])}, and {clauses[-1]}"


def _describe_first_passage(defn: Definition, rule: FirstPassage, names: dict[str, ThresholdSpec]) -> str:
    """Describe a first-passage rule.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition.
    rule : hcwf.onset.model.FirstPassage
        Its rule.
    names : dict of str to hcwf.onset.model.ThresholdSpec
        The thresholds of the definition by name.

    Returns
    -------
    str
        The sentence.
    """
    notes = _WetNotes(defn.wet_day)
    clauses = [_condition(c, notes, names) for c in rule.require]
    clauses += [_veto(v, defn, notes, names) for v in rule.reject_if]
    sentence = f"Onset is the first day {_season_phrase(defn)} {_join(clauses)}."
    if rule.report == "first_wet_day":
        span = rule.report_span
        sentence += f" The onset date reported is the first wet day within {_days_text(span)}."
    return sentence


def _describe_curve(defn: Definition, names: dict[str, ThresholdSpec]) -> str:
    """Describe a curve rule.

    Parameters
    ----------
    defn : hcwf.onset.model.Definition
        The definition, whose rule is a `hcwf.onset.model.Curve`.
    names : dict of str to hcwf.onset.model.ThresholdSpec
        The thresholds of the definition by name.

    Returns
    -------
    str
        The sentence.
    """
    rule = defn.rule
    which = "day after" if rule.report == "day_after_minimum" else "day of"
    baseline = f"{_threshold_text(rule.baseline, names.get(rule.baseline))} ({rule.baseline})"
    return (f"Onset is the {which} the minimum of the cumulative rainfall anomaly (daily rain minus "
            f"{baseline}), accumulated from the start of each search window ({_season_phrase(defn)}).")
