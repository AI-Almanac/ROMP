"""Parsers for the small text grammar of onset definitions (``docs/onset_spec.md``, section 3.7).

Each parser turns one YAML string into a type of `hcwf.onset.model`: amounts and comparisons (fixed amounts become
integer tenths of a mm), counts, durations, spans, day references and dry spells. Failures raise
`hcwf.errors.UnitsError`; its message says how to fix the text, and `hcwf.onset.definition` prefixes it with the YAML
path.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from typing import Iterator

from hcwf.errors import UnitsError
from hcwf.onset.model import Amount, Comparison, DayRef, DryRun, DryWindow, Op, Span


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
_FINE_AMOUNTS: ContextVar[bool] = ContextVar("fine_amounts", default=False)


@contextmanager
def fine_amounts(allowed: bool = True) -> Iterator[None]:
    """Allow fixed amounts finer than 0.1 mm to parse, within the block.

    Inside the block `parse_comparison` returns ``Amount(mm=...)`` for such an amount instead of raising. These
    amounts only run at ``rain_resolution: exact``; `~hcwf.onset.plan.resolve_amounts` rejects them otherwise.

    Parameters
    ----------
    allowed : bool, default True
        Whether fine amounts are allowed inside the block.

    Yields
    ------
    None
        Nothing; the setting is restored when the block ends, also on an exception.
    """
    token = _FINE_AMOUNTS.set(allowed)
    try:
        yield
    finally:
        _FINE_AMOUNTS.reset(token)


def _normalize(text: object, example: str) -> str:
    """Return the text with runs of whitespace collapsed to single spaces.

    Parameters
    ----------
    text : object
        The value read from YAML; it must be a string.
    example : str
        A valid text, quoted in the error message.

    Returns
    -------
    str
        ``text`` stripped, with every run of whitespace replaced by one space.

    Raises
    ------
    UnitsError
        ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise UnitsError(f"expected text but found {text!r}. Fix: write it as text, e.g. '{example}'")
    return " ".join(text.split())


def split_comparator(text: object, example: str = "at least 20 mm") -> tuple[Op, str]:
    """Split a leading comparator off the text.

    Parameters
    ----------
    text : object
        The YAML value, for example ``"at least 20 mm"``.
    example : str, default "at least 20 mm"
        A valid text, quoted in error messages.

    Returns
    -------
    tuple of (str, str)
        The operator, one of ``"ge"``, ``"gt"``, ``"lt"``, ``"le"``, and the text after the comparator.

    Raises
    ------
    UnitsError
        ``text`` is not a string, has no leading comparator, or nothing after it.
    """
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
    """Parse ``<comparator> <number> mm`` or ``<comparator> <threshold name>`` into a comparison.

    A fixed amount becomes integer tenths of a mm. One finer than 0.1 mm becomes ``Amount(mm=...)`` inside a
    `fine_amounts` block and is an error outside it.

    Parameters
    ----------
    text : object
        The YAML value, for example ``"at least 20 mm"`` or ``"more than wet_spell_amount"``.

    Returns
    -------
    hcwf.onset.model.Comparison
        The operator and the amount (tenths, a threshold name, or fine mm).

    Raises
    ------
    UnitsError
        ``text`` is not text, has no comparator or no unit, is finer than 0.1 mm outside `fine_amounts`, or
        the amount is neither a number with ``mm`` nor a threshold name.
    """
    op, rest = split_comparator(text)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*mm", rest, re.IGNORECASE)
    if match is not None:
        tenths = Decimal(match.group(1)) * 10
        if tenths != tenths.to_integral_value():
            if _FINE_AMOUNTS.get():
                return Comparison(op, Amount(mm=float(Decimal(match.group(1)))))
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
    """Parse ``<comparator> <integer>`` (for example ``"at least 3"``) into an operator and a count.

    Parameters
    ----------
    text : object
        The YAML value.

    Returns
    -------
    tuple of (str, int)
        The operator (``"ge"``, ``"gt"``, ``"lt"`` or ``"le"``) and the count.

    Raises
    ------
    UnitsError
        ``text`` has no comparator or what follows it is not a whole number.
    """
    op, rest = split_comparator(text, "at least 3")
    if not re.fullmatch(r"\d+", rest):
        raise UnitsError(f"'{rest}' is not a whole number. Fix: write a count such as 'at least 3'")
    return op, int(rest)


def parse_duration(text: object) -> int:
    """Parse ``<integer> day(s)`` into a number of days.

    Parameters
    ----------
    text : object
        The YAML value, for example ``"5 days"``.

    Returns
    -------
    int
        The number of days, at least 1.

    Raises
    ------
    UnitsError
        ``text`` is not a whole number of days, or is below 1.
    """
    text = _normalize(text, "5 days")
    match = re.fullmatch(r"(\d+)\s+days?", text, re.IGNORECASE)
    if match is None:
        raise UnitsError(f"'{text}' is not a duration. Fix: write a whole number of days, e.g. '5 days'")
    days = int(match.group(1))
    if days < 1:
        raise UnitsError(f"'{text}' is not positive. Fix: use at least '1 day'")
    return days


def parse_span(text: object, window: int | None = None) -> Span:
    """Parse a span: ``days A to B``, ``day A``, ``N days after onset|window`` or ``season to date``.

    Parameters
    ----------
    text : object
        The YAML value.
    window : int, optional
        The trigger window length W in days; needed for ``N days after window``.

    Returns
    -------
    hcwf.onset.model.Span
        The first and last day of the span, relative to the day being tested (``season to date`` sets
        ``since_search_start``).

    Raises
    ------
    UnitsError
        ``text`` cannot be read, ends before it starts, is not positive, says nothing about what it counts from, or
        needs ``window`` and none is given.
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
    """Build a span, turning an empty range into an error.

    Parameters
    ----------
    first : int
        The first day.
    last : int
        The last day.
    text : str
        The original text, quoted in the error message.

    Returns
    -------
    hcwf.onset.model.Span
        The span from ``first`` to ``last``.

    Raises
    ------
    UnitsError
        ``last`` is before ``first``.
    """
    if last < first:
        raise UnitsError(f"'{text}' ends before it starts. Fix: write the earlier day first, e.g. 'days {last} to {first}'")
    return Span(first, last)


def parse_dayref(text: object) -> DayRef:
    """Parse ``MM-DD`` or ``climatological onset|cessation [+|- N days]`` into a day reference.

    Parameters
    ----------
    text : object
        The YAML value, for example ``"06-02"`` or ``"climatological onset - 50 days"``.

    Returns
    -------
    hcwf.onset.model.DayRef
        A fixed month and day, or the climatological onset or cessation with a signed offset in days.

    Raises
    ------
    UnitsError
        ``text`` is ``02-29``, not a valid month-day, or cannot be read.
    """
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
    """Parse ``MM-DD to MM-DD`` into two day references, or ``whole year`` into None.

    Parameters
    ----------
    text : object
        The YAML value.

    Returns
    -------
    tuple of (hcwf.onset.model.DayRef, hcwf.onset.model.DayRef) or None
        The start and end of the period, or None for ``whole year``.

    Raises
    ------
    UnitsError
        ``text`` cannot be read, or a month-day in it is invalid.
    """
    text = _normalize(text, "04-01 to 10-31")
    if text.lower() == "whole year":
        return None
    match = re.fullmatch(r"(\d{2}-\d{2}) to (\d{2}-\d{2})", text)
    if match is None:
        raise UnitsError(f"cannot read the period '{text}'. Fix: write 'MM-DD to MM-DD', e.g. '04-01 to 10-31', or 'whole year'")
    return parse_dayref(match.group(1)), parse_dayref(match.group(2))


def parse_dry_run(text: object) -> DryRun:
    """Parse ``at least N days`` or ``more than N days`` into a dry run of at least N (or N + 1) days.

    Parameters
    ----------
    text : object
        The YAML value.

    Returns
    -------
    hcwf.onset.model.DryRun
        The run length in days (at least 1) and the phrase (``"at least"`` or ``"more than"``) it was written with.

    Raises
    ------
    UnitsError
        ``text`` cannot be read or the run is shorter than one day.
    """
    text = _normalize(text, "at least 7 days")
    match = re.fullmatch(r"(at least|more than) (\d+) days?", text.lower())
    if match is None:
        raise UnitsError(f"cannot read the dry run '{text}'. Fix: write 'at least 7 days' or 'more than 7 days'")
    length = int(match.group(2)) + (match.group(1) == "more than")
    if length < 1:
        raise UnitsError(f"'{text}' is not a run of days. Fix: use at least '1 day'")
    return DryRun(length, phrase=match.group(1))


def parse_dry_window(text: object) -> DryWindow:
    """Parse ``N days with <comparison>`` (for example ``"10 days with less than 5 mm"``) into a dry window.

    Parameters
    ----------
    text : object
        The YAML value.

    Returns
    -------
    hcwf.onset.model.DryWindow
        The window length in days and the comparison of the window total.

    Raises
    ------
    UnitsError
        ``text`` cannot be read, the length is below 1, or the comparison is invalid (see `parse_comparison`).
    """
    text = _normalize(text, "10 days with less than 5 mm")
    match = re.fullmatch(r"(\d+) days? with (.+)", text, re.IGNORECASE)
    if match is None or int(match.group(1)) < 1:
        raise UnitsError(
            f"cannot read the dry window '{text}'. Fix: write e.g. '10 days with less than 5 mm'")
    return DryWindow(int(match.group(1)), parse_comparison(match.group(2)))
