"""Tests for the onset text grammar in momp.onset.units."""
import pytest

from momp.onset import units
from momp.onset.model import Amount, Comparison, DayRef, DryRun, DryWindow, Span


@pytest.mark.parametrize("text, op", [
    ("at least 20 mm", "ge"), ("more than 20 mm", "gt"), ("less than 20 mm", "lt"), ("at most 20 mm", "le"),
    (">= 20 mm", "ge"), ("> 20 mm", "gt"), ("< 20 mm", "lt"), ("<= 20 mm", "le"), (">=20mm", "ge"),
])
def test_comparators(text, op):
    """Every comparator, in words or symbols, maps to its operator."""
    assert units.parse_comparison(text) == Comparison(op, Amount(tenths=200))


def test_fixed_amounts_are_tenths():
    """Fixed amounts become integer tenths of a mm."""
    assert units.parse_comparison("at least 0.5 mm").amount == Amount(tenths=5)
    assert units.parse_comparison("more than 1 mm").amount == Amount(tenths=10)
    assert units.parse_comparison("at least 20.10 mm").amount == Amount(tenths=201)


def test_threshold_amount():
    """A bare name is a threshold reference."""
    assert units.parse_comparison("more than wet_spell_amount") == Comparison("gt", Amount(threshold="wet_spell_amount"))


@pytest.mark.parametrize("text, hint", [
    ("20 mm", "no comparator"),
    ("at least 0.25 mm", "finer than 0.1 mm"),
    ("at least 20", "no unit"),
    ("at least", "nothing after"),
    ("at least 20 inches", "cannot read the amount"),
    (20, "expected text"),
])
def test_amount_errors(text, hint):
    """Bad amounts say what is wrong and how to fix it."""
    with pytest.raises(units.UnitsError, match=hint) as exc:
        units.parse_comparison(text)
    assert "Fix:" in str(exc.value)


def test_count():
    """Counts need a comparator and a whole number."""
    assert units.parse_count("at least 3") == ("ge", 3)
    assert units.parse_count("<= 2") == ("le", 2)
    for bad in ("3", "at least 3.5", "at least three"):
        with pytest.raises(units.UnitsError):
            units.parse_count(bad)


def test_duration():
    """Durations are whole days."""
    assert units.parse_duration("5 days") == 5
    assert units.parse_duration("1 day") == 1
    for bad in ("5", "0 days", "5 weeks", 5):
        with pytest.raises(units.UnitsError, match="Fix:"):
            units.parse_duration(bad)


def test_spans():
    """Every span form normalizes to day offsets from the candidate."""
    assert units.parse_span("days 11 to 20") == Span(11, 20)
    assert units.parse_span("day 1") == Span(1, 1)
    assert units.parse_span("21 days after onset") == Span(2, 22)
    assert units.parse_span("21 days after window", window=3) == Span(4, 24)
    assert units.parse_span("season to date") == Span(0, 1, since_search_start=True)
    assert units.parse_span("days 0 to 3") == Span(0, 3)


@pytest.mark.parametrize("text, window, hint", [
    ("21 days", None, "after onset"),
    ("21 days after window", None, "window: N days"),
    ("days 5 to 2", None, "ends before it starts"),
    ("next week", None, "cannot read the span"),
])
def test_span_errors(text, window, hint):
    """Ambiguous or empty spans are errors with a fix."""
    with pytest.raises(units.UnitsError, match=hint):
        units.parse_span(text, window)


def test_dayrefs():
    """Fixed and climatological day references."""
    assert units.parse_dayref("06-02") == DayRef(month=6, day=2)
    assert units.parse_dayref("climatological onset - 50 days") == DayRef(climatological="onset", offset_days=-50)
    assert units.parse_dayref("climatological cessation") == DayRef(climatological="cessation")
    assert units.parse_dayref("climatological cessation + 10 days") == DayRef(climatological="cessation", offset_days=10)
    for bad, hint in (("02-29", "not allowed"), ("13-01", "valid month-day"), ("04-31", "valid month-day"),
                      ("June 2", "cannot read")):
        with pytest.raises(units.UnitsError, match=hint):
            units.parse_dayref(bad)


def test_during():
    """Reference periods are two fixed days or the whole year."""
    assert units.parse_during("04-01 to 10-31") == (DayRef(month=4, day=1), DayRef(month=10, day=31))
    assert units.parse_during("whole year") is None
    with pytest.raises(units.UnitsError, match="Fix:"):
        units.parse_during("April to October")


def test_dry_spells():
    """Dry runs count days; more than N means N + 1."""
    assert units.parse_dry_run("at least 7 days") == DryRun(7)
    assert units.parse_dry_run("more than 7 days") == DryRun(8, phrase="more than")
    assert units.parse_dry_window("10 days with less than 5 mm") == DryWindow(10, Comparison("lt", Amount(tenths=50)))
    for bad in ("less than 7 days", "7"):
        with pytest.raises(units.UnitsError):
            units.parse_dry_run(bad)
    with pytest.raises(units.UnitsError, match="Fix:"):
        units.parse_dry_window("10 days")
