"""The old flat ROMP onset keys translate into the Definition that reproduces the old detection."""

import warnings

import pytest

from momp.onset import legacy
from momp.onset.legacy import LegacyConfigError, from_flat_config
from momp.onset.model import Amount, Comparison, DayRef, DryRun, FirstPassage, Season, Span, Total, Veto, WetDays

BASE = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_threshold=1, dry_spell=7, dry_extent=0,
            thresh_file=None, thresh_var=None, earliest_onset_date=None,
            start_date=(2019, 5, 1), end_date=(2019, 7, 31))


def _rule(window=3, amount=Amount(tenths=200), veto=()):
    """The expected FirstPassage: every window day wet, total above the amount, optional vetoes."""
    return FirstPassage(
        require=(WetDays(Span(1, window), "ge", window), Total(Span(1, window), Comparison("gt", amount))),
        reject_if=veto)


def _main(start=(5, 1), end=(7, 31), wet_spell=3):
    """The expected single season: the start day to the last day whose wet spell fits by end_date + 47 days."""
    return (Season("main", DayRef(month=start[0], day=start[1]),
                   DayRef(month=end[0], day=end[1], offset_days=47 - (wet_spell - 1))),)


def test_plain_config():
    """No dry extent: wet days and a total above wet_threshold, searched from start_date to end_date + 45 days."""
    defn = from_flat_config(BASE)
    assert defn.wet_day == Comparison("ge", Amount(tenths=10))
    assert defn.seasons == _main()
    assert defn.rule == _rule()


def test_dry_extent_beyond_the_window_adds_the_observed_veto_window():
    """dry_extent above wet_spell and at least dry_spell vetoes a dry run in days 2 to dry_extent + 1."""
    defn = from_flat_config({**BASE, "dry_extent": 10})
    assert defn.rule == _rule(veto=(Veto(DryRun(7), Span(2, 11), "inside"),))


@pytest.mark.parametrize(("extent", "spell"), [(3, 7), (0, 7), (5, 7)])
def test_no_veto_when_the_extent_is_short(extent, spell):
    """A dry extent within the wet window, or shorter than the dry spell, gives no veto."""
    defn = from_flat_config({**BASE, "dry_extent": extent, "dry_spell": spell})
    assert defn.rule == _rule()


def test_earliest_onset_date_and_threshold_file():
    """earliest_onset_date replaces the season start; a threshold file makes the total compare with a named threshold."""
    defn = from_flat_config({**BASE, "earliest_onset_date": (6, 2), "thresh_file": "t.nc", "thresh_var": "mwset",
                             "wet_spell": 5, "wet_init": 0.5})
    assert defn.wet_day == Comparison("ge", Amount(tenths=5))
    assert defn.seasons == _main(start=(6, 2), wet_spell=5)
    assert defn.rule == _rule(window=5, amount=Amount(threshold="legacy_threshold"))


def test_dry_threshold_is_ignored():
    """dry_threshold never mattered to the old code, so it does not change the Definition."""
    assert from_flat_config({**BASE, "dry_threshold": 5}) == from_flat_config({**BASE, "dry_threshold": 1})


@pytest.mark.parametrize("key", ["wet_init", "wet_threshold"])
def test_amounts_must_be_whole_tenths(key):
    """Amounts finer than 0.1 mm are an error naming the key."""
    with pytest.raises(LegacyConfigError, match=key):
        from_flat_config({**BASE, key: 0.25})


def test_missing_keys_are_named():
    """A flat config without wet_spell says so."""
    with pytest.raises(LegacyConfigError, match="wet_spell"):
        from_flat_config({k: v for k, v in BASE.items() if k != "wet_spell"})


def test_deprecation_warning_comes_once_per_process(monkeypatch):
    """The FutureWarning pointing to onset_definition is emitted on the first translation only."""
    monkeypatch.setattr(legacy, "_warned", False)
    with pytest.warns(FutureWarning, match="onset_definition"):
        from_flat_config(BASE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_flat_config(BASE)
