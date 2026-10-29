"""Parsers for the small text grammar of onset definitions (docs/onset_spec.md, section 3.7).

Each parser turns one YAML string into a model type: amounts and comparisons
(fixed amounts become integer tenths of a mm), counts, durations, spans, day
references and dry spells. Failures raise UnitsError; its message says how to
fix the text, and momp.onset.definition prefixes it with the YAML path.
"""
from __future__ import annotations

import re
from decimal import Decimal

from momp.onset.model import Amount, Comparison, DayRef, DryRun, DryWindow, Op, Span


class UnitsError(ValueError):
    """Text that does not follow the onset grammar; the message ends with how to fix it."""


_COMPARATORS: dict[str, Op] = {
    "at least": "ge",
    "more than": "gt",
    "less than": "lt",
    "at most": "le",
    ">=": "ge",
    ">": "gt",
    "<": "lt",
    "<=": "le",
}
_COMPARATOR_RE = re.compile(r"^(at least|more than|less than|at most|>=|<=|>|<)\s*(.*)$", re.IGNORECASE)
_COMPARATOR_HINT = "at least, more than, less than, at most (or >=, >, <, <=)"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _normalize(text: object, example: str) -> str:
    """Return the text with collapsed whitespace, or raise if it is not a string."""
    if not isinstance(text, str):
        raise UnitsError(f"expected text but found {text!r}. Fix: write it as text, e.g. '{example}'")
    return " ".join(text.split())


def split_comparator(text: object, example: str = "at least 20 mm") -> tuple[Op, str]:
    """Split a leading comparator off the text, returning (op, rest)."""
    text = _normalize(text, example)
    match = _COMPARATOR_RE.match(text)
    if match is None:
        raise UnitsError(
            f"'{text}' has no comparator. Fix: start with one of {_COMPARATOR_HINT}, e.g. '{example}'")
    rest = match.group(2).strip()
    if not rest:
        raise UnitsError(f"'{text}' has nothing after the comparator. Fix: write e.g. '{example}'")
    return _COMPARATORS[match.group(1).lower()], rest


def parse_comparison(text: object) -> Comparison:
    """Parse '<comparator> <number> mm' or '<comparator> <threshold name>' into a Comparison."""
    op, rest = split_comparator(text)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*mm", rest, re.IGNORECASE)
    if match is not None:
        tenths = Decimal(match.group(1)) * 10
        if tenths != tenths.to_integral_value():
            raise UnitsError(
                f"'{rest}' is finer than 0.1 mm. Fix: round the amount to a whole tenth of a mm, e.g. 20.1 mm")
        return Comparison(op, Amount(tenths=int(tenths)))
    if re.fullmatch(r"\d+(?:\.\d+)?", rest):
        raise UnitsError(f"'{rest}' has no unit. Fix: write '{rest} mm'")
    if _NAME_RE.match(rest):
        return Comparison(op, Amount(threshold=rest))
    raise UnitsError(
        f"cannot read the amount '{rest}'. Fix: write a number with its unit, e.g. '20 mm', "
        "or the name of a threshold, e.g. 'wet_spell_amount'")


def parse_count(text: object) -> tuple[Op, int]:
    """Parse '<comparator> <integer>' (e.g. 'at least 3') into (op, count)."""
    op, rest = split_comparator(text, "at least 3")
    if not re.fullmatch(r"\d+", rest):
        raise UnitsError(f"'{rest}' is not a whole number. Fix: write a count such as 'at least 3'")
    return op, int(rest)


def parse_duration(text: object) -> int:
    """Parse '<integer> day(s)' into a number of days (at least 1)."""
    text = _normalize(text, "5 days")
    match = re.fullmatch(r"(\d+)\s+days?", text, re.IGNORECASE)
    if match is None:
        raise UnitsError(f"'{text}' is not a duration. Fix: write a whole number of days, e.g. '5 days'")
    days = int(match.group(1))
    if days < 1:
        raise UnitsError(f"'{text}' is not positive. Fix: use at least '1 day'")
    return days


def parse_span(text: object, window: int | None = None) -> Span:
    """Parse a span: 'days A to B', 'day A', 'N days after onset|window' or 'season to date'.

    window is the trigger window length W, needed for 'N days after window'.
    """
    text = _normalize(text, "days 11 to 20")
    lowered = text.lower()
    if lowered == "season to date":
        return Span(0, 1, since_search_start=True)
    match = re.fullmatch(r"days (-?\d+) to (-?\d+)", lowered)
    if match is not None:
        return _make_span(int(match.group(1)), int(match.group(2)), text)
    match = re.fullmatch(r"day (-?\d+)", lowered)
    if match is not None:
        return Span(int(match.group(1)), int(match.group(1)))
    match = re.fullmatch(r"(\d+) days? after (onset|window)", lowered)
    if match is not None:
        days = int(match.group(1))
        if days < 1:
            raise UnitsError(f"'{text}' is not positive. Fix: use at least '1 day after {match.group(2)}'")
        if match.group(2) == "onset":
            return Span(2, days + 1)
        if window is None:
            raise UnitsError(
                f"'{text}' needs the trigger window length. Fix: add 'window: N days' under onset, "
                "or use 'N days after onset' or 'days A to B'")
        return Span(window + 1, window + days)
    if re.fullmatch(r"\d+ days?", lowered):
        raise UnitsError(
            f"'{text}' does not say what it counts from. Fix: write '{text} after onset' or "
            f"'{text} after window'")
    raise UnitsError(
        f"cannot read the span '{text}'. Fix: use 'days A to B', 'day A', 'N days after onset', "
        "'N days after window' or 'season to date'")


def _make_span(first: int, last: int, text: str) -> Span:
    """Build a Span, turning an empty range into a UnitsError."""
    if last < first:
        raise UnitsError(f"'{text}' ends before it starts. Fix: write the earlier day first, e.g. 'days {last} to {first}'")
    return Span(first, last)


def parse_dayref(text: object) -> DayRef:
    """Parse 'MM-DD' or 'climatological onset|cessation [+|- N days]' into a DayRef."""
    text = _normalize(text, "06-02")
    match = re.fullmatch(r"(\d{2})-(\d{2})", text)
    if match is not None:
        month, day = int(match.group(1)), int(match.group(2))
        if (month, day) == (2, 29):
            raise UnitsError("'02-29' is not allowed because not every year has it. Fix: use 02-28 or 03-01")
        if not 1 <= month <= 12 or not 1 <= day <= _MONTH_DAYS[month - 1]:
            raise UnitsError(f"'{text}' is not a valid month-day. Fix: write MM-DD, e.g. '06-02'")
        return DayRef(month=month, day=day)
    match = re.fullmatch(r"climatological (onset|cessation)(?: ([+-]) ?(\d+) days?)?", text.lower())
    if match is not None:
        offset = int(match.group(3)) if match.group(3) else 0
        return DayRef(climatological=match.group(1), offset_days=-offset if match.group(2) == "-" else offset)
    raise UnitsError(
        f"cannot read the day '{text}'. Fix: write MM-DD (e.g. '06-02') or "
        "'climatological onset - 50 days' (also 'climatological cessation', '+ N days')")


def parse_during(text: object) -> tuple[DayRef, DayRef] | None:
    """Parse 'MM-DD to MM-DD' into two DayRefs, or 'whole year' into None."""
    text = _normalize(text, "04-01 to 10-31")
    if text.lower() == "whole year":
        return None
    match = re.fullmatch(r"(\d{2}-\d{2}) to (\d{2}-\d{2})", text)
    if match is None:
        raise UnitsError(f"cannot read the period '{text}'. Fix: write 'MM-DD to MM-DD', e.g. '04-01 to 10-31', or 'whole year'")
    return parse_dayref(match.group(1)), parse_dayref(match.group(2))


def parse_dry_run(text: object) -> DryRun:
    """Parse 'at least N days' or 'more than N days' into a DryRun of at least N (or N + 1) days."""
    text = _normalize(text, "at least 7 days")
    match = re.fullmatch(r"(at least|more than) (\d+) days?", text.lower())
    if match is None:
        raise UnitsError(f"cannot read the dry run '{text}'. Fix: write 'at least 7 days' or 'more than 7 days'")
    length = int(match.group(2)) + (match.group(1) == "more than")
    if length < 1:
        raise UnitsError(f"'{text}' is not a run of days. Fix: use at least '1 day'")
    return DryRun(length, phrase=match.group(1))


def parse_dry_window(text: object) -> DryWindow:
    """Parse 'N days with <comparison>' (e.g. '10 days with less than 5 mm') into a DryWindow."""
    text = _normalize(text, "10 days with less than 5 mm")
    match = re.fullmatch(r"(\d+) days? with (.+)", text, re.IGNORECASE)
    if match is None or int(match.group(1)) < 1:
        raise UnitsError(
            f"cannot read the dry window '{text}'. Fix: write e.g. '10 days with less than 5 mm'")
    return DryWindow(int(match.group(1)), parse_comparison(match.group(2)))
