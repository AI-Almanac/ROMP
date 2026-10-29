"""Tests for loading, normalizing and validating onset definitions."""
import dataclasses
import textwrap

import pytest
import yaml

from momp.onset import definition as dfn
from momp.onset.model import (
    Amount, Comparison, Curve, DayRef, DerivedSeasons, DryRun, DryWindow, FirstPassage, Season, Span,
    ThresholdSpec, Total, Veto, WetDays,
)

HEADER = """\
id: t
version: 1
title: T
citation: TODO
wording: TODO
status: draft
notes: n
wet_day: at least 1 mm
season: {from: 03-01, to: 05-31}
"""

ICPAC = HEADER + """\
onset:
  window: 3 days
  require:
    - first day: wet
    - total: more than 20 mm
  reject if:
    - dry run: at least 7 days
      within: 21 days after onset
"""


def make(extra: str = "", base: str = HEADER) -> str:
    """Return the header plus extra YAML."""
    return base + textwrap.dedent(extra)


def onset(body: str) -> str:
    """Return a definition whose onset block is the indented body."""
    return HEADER + "onset:\n" + textwrap.indent(textwrap.dedent(body), "  ")


def rule(text: str) -> FirstPassage:
    """Load a definition and return its first-passage rule."""
    return dfn.loads(text).rule


def fails(text: str, *needles: str):
    """Assert loading raises a DefinitionError whose message has every needle."""
    with pytest.raises(dfn.DefinitionError) as exc:
        dfn.loads(text)
    for needle in needles:
        assert needle in str(exc.value), str(exc.value)


def test_icpac_normalization():
    """The spec's example normalizes to the exact model objects."""
    d = dfn.loads(ICPAC)
    assert d.wet_day == Comparison("ge", Amount(tenths=10))
    assert d.seasons == (Season("main", DayRef(month=3, day=1), DayRef(month=5, day=31)),)
    assert d.rule == FirstPassage(
        require=(WetDays(Span(1, 1), "ge", 1), Total(Span(1, 3), Comparison("gt", Amount(tenths=200)))),
        reject_if=(Veto(DryRun(7), Span(2, 22), "inside"),))
    assert d.thresholds == ()
    assert yaml.safe_load(d.source_yaml)["id"] == "t"


def test_require_forms():
    """Each requirement form normalizes with the window or its own span."""
    r = rule(onset("""\
        window: 5 days
        require:
          - first day: wet
          - all days: wet
          - wet days: at least 3
          - total: at least 20 mm
          - total: at least 20 mm
            over: days 11 to 20
          - total: at least 51 mm
            over: season to date
          - wet days: at most 2
            over: day 3
        """))
    at20 = Comparison("ge", Amount(tenths=200))
    assert r.require == (
        WetDays(Span(1, 1), "ge", 1),
        WetDays(Span(1, 5), "ge", 5),
        WetDays(Span(1, 5), "ge", 3),
        Total(Span(1, 5), at20),
        Total(Span(11, 20), at20),
        Total(Span(0, 1, since_search_start=True), Comparison("ge", Amount(tenths=510))),
        WetDays(Span(3, 3), "le", 2),
    )


def test_all_days_over_own_span():
    """'all days' over a span needs every day of that span."""
    r = rule(onset("""\
        require:
          - all days: wet
            over: days 2 to 4
        """))
    assert r.require == (WetDays(Span(2, 4), "ge", 3),)


def test_vetoes_and_offset_spans():
    """Dry runs, dry windows and both placements normalize exactly."""
    r = rule(onset("""\
        window: 4 days
        require:
          - total: at least 1 mm
        reject if:
          - dry run: more than 7 days
            within: 21 days after window
          - dry window: 10 days with less than 5 mm
            starting within: 30 days after onset
          - dry run: at least 7 days
            within: days 5 to 20
        """))
    assert r.reject_if == (
        Veto(DryRun(8), Span(5, 25), "inside"),
        Veto(DryWindow(10, Comparison("lt", Amount(tenths=50))), Span(2, 31), "starts_inside"),
        Veto(DryRun(7), Span(5, 20), "inside"),
    )


def test_report_first_wet_day():
    """'first wet day' keeps the trigger window as report span; the default is the candidate."""
    r = rule(onset("""\
        window: 3 days
        require:
          - total: at least 1 mm
        report: first wet day
        """))
    assert (r.report, r.report_span) == ("first_wet_day", Span(1, 3))
    r = rule(onset("""\
        window: 3 days
        require:
          - total: at least 1 mm
        report: first day
        """))
    assert (r.report, r.report_span) == ("candidate", None)


def test_wet_day_more_than():
    """'more than' is a valid wet-day rule."""
    d = dfn.loads(ICPAC.replace("at least 1 mm", "more than 1 mm"))
    assert d.wet_day == Comparison("gt", Amount(tenths=10))


def test_seasons_forms():
    """season, named seasons in file order, and derived seasons."""
    named = ICPAC.replace("season: {from: 03-01, to: 05-31}", """\
seasons:
  short_rains: {from: 09-01, to: 12-31}
  long_rains: {from: 02-01, to: 05-31}""")
    d = dfn.loads(named)
    assert [s.name for s in d.seasons] == ["short_rains", "long_rains"]
    assert d.seasons[0].end == DayRef(month=12, day=31)
    derived = dfn.loads(ICPAC.replace("season: {from: 03-01, to: 05-31}", """\
seasons:
  derive: harmonic regime
  from: climatological onset - 50 days
  to: climatological cessation"""))
    assert derived.seasons == DerivedSeasons(
        "harmonic", DayRef(climatological="onset", offset_days=-50), DayRef(climatological="cessation"))


def test_thresholds():
    """The three threshold kinds, with and without a period."""
    d = dfn.loads(make("""\
        thresholds:
          a:
            mean total over: 5 days
            during: 04-01 to 10-31
          b:
            mean daily rain: true
            during: whole year
          c:
            quantile of totals over: 3 days
            quantile: 0.8
        curve:
          accumulate: rain minus b
          onset: day of minimum
        """))
    assert d.thresholds == (
        ("a", ThresholdSpec("mean_total", days=5, during=(DayRef(month=4, day=1), DayRef(month=10, day=31)))),
        ("b", ThresholdSpec("mean_daily")),
        ("c", ThresholdSpec("quantile_total", days=3, quantile=0.8)),
    )
    assert d.rule == Curve(baseline="b", report="day_of_minimum")
    assert d.threshold_specs["b"].kind == "mean_daily"


def test_curve_day_after_minimum():
    """The curve default report maps 'day after minimum'."""
    d = dfn.loads(make("""\
        thresholds:
          m: {mean daily rain: true}
        curve: {accumulate: rain minus m, onset: day after minimum}
        """))
    assert d.rule == Curve("m", "day_after_minimum")


def test_threshold_used_in_require_and_dry_window():
    """Threshold names are accepted in requirements and dry windows when defined."""
    d = dfn.loads(make("""\
        thresholds:
          amt: {mean total over: 5 days}
        onset:
          window: 5 days
          require:
            - total: at least amt
          reject if:
            - dry window: 5 days with less than amt
              within: 10 days after window
        """))
    assert d.rule.require[0].comparison.amount == Amount(threshold="amt")
    assert d.rule.reject_if[0].spell.comparison.amount == Amount(threshold="amt")


@pytest.mark.parametrize("text, path, fix", [
    (ICPAC.replace("wet_day: at least 1 mm", "wet_day: at least 1 mm\nbogus: 1"), "bogus", "unknown key"),
    (ICPAC.replace("  window: 3 days", "  window: 3 days\n  windw: 3"), "onset.windw", "unknown key"),
    (ICPAC.replace("wet_day: at least 1 mm\n", ""), "wet_day", "missing required key"),
    (ICPAC.replace("wet_day: at least 1 mm", "wet_day: 1 mm"), "wet_day", "no comparator"),
    (ICPAC.replace("wet_day: at least 1 mm", "wet_day: less than 1 mm"), "wet_day", "at least X mm"),
    (ICPAC.replace("wet_day: at least 1 mm", "wet_day: at least wet_thr"), "wet_day", "Fix:"),
    (ICPAC.replace("more than 20 mm", "20 mm"), "onset.require[1].total", "no comparator"),
    (ICPAC.replace("more than 20 mm", "more than 20.05 mm"), "onset.require[1].total", "0.1 mm"),
    (ICPAC.replace("more than 20 mm", "more than nowhere"), "onset.require[1].total", "not defined"),
    (ICPAC.replace("21 days after onset", "21 days"), "onset.reject if[0].within", "after onset"),
    (ICPAC.replace("within: 21 days after onset", "other: 1"), "onset.reject if[0]", "unknown key"),
    (ICPAC.replace("      within: 21 days after onset\n", ""), "onset.reject if[0]", "within"),
    (ICPAC.replace("within: 21", "within: 21 days after onset\n      starting within: 21"), "onset.reject if[0]", "exactly one"),
    (ICPAC.replace("at least 7 days", "at least 30 days"), "onset.reject if[0].within", "cannot fit"),
    (ICPAC.replace("dry run: at least 7 days", "dry run: at least 7 days\n      dry window: 5 days with less than 1 mm"),
     "onset.reject if[0]", "exactly one"),
    (ICPAC.replace("  window: 3 days\n", "").replace("more than 20 mm", "more than 20 mm\n      over: days 1 to 3").replace("onset:\n", "onset:\n  report: first wet day\n"), "onset.report", "window"),
    (ICPAC.replace("  window: 3 days\n", ""), "onset.require[1].total", "window"),
    (ICPAC.replace("window: 3 days", "window: 3"), "onset.window", "5 days"),
    (ICPAC.replace("  window: 3 days\n", "").replace("more than 20 mm", "more than 20 mm\n      over: days 1 to 3").replace("21 days after onset", "21 days after window"),
     "onset.reject if[0].within", "window: N days"),
    (ICPAC.replace("first day: wet", "first day: dry"), "onset.require[0].first day", "first day: wet"),
    (ICPAC.replace("- total: more than 20 mm", "- total: more than 20 mm\n      wet days: at least 1"),
     "onset.require[1]", "exactly one"),
    (ICPAC.replace("onset:\n", "onset:\n  report: first wet days\n"), "onset.report", "first wet day"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "season: {from: 02-29, to: 05-31}"), "season.from", "02-29"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "season: {from: 03-01}"), "season", "missing required key 'to'"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "season: {from: 03-01, to: 05-31}\nseasons:\n  a: {from: 01-01, to: 02-01}"),
     "season", "exactly one"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}\n", ""), "season", "exactly one"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "season: {from: climatological onset, to: 05-31}"),
     "season.from", "derive"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "seasons: {derive: harmonic regime, from: 03-01, to: climatological cessation}"),
     "seasons.from", "climatological"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "seasons: {derive: harmonic regime, from: climatological onset, to: climatological cessation, x: 1}"),
     "seasons.x", "unknown key"),
    (ICPAC.replace("season: {from: 03-01, to: 05-31}", "seasons: {derive: monsoon, from: climatological onset, to: climatological cessation}"),
     "seasons.derive", "harmonic regime"),
    (ICPAC.replace("status: draft", "status: final"), "status", "draft"),
    (ICPAC.replace("version: 1", "version: one"), "version", "whole number"),
    (ICPAC.replace("id: t", "id: Bad Id"), "id", "lowercase"),
    (ICPAC.replace("status: draft", "status: verified"), "citation", "TODO"),
    (ICPAC + "curve: {accumulate: rain minus m, onset: day after minimum}\n", "onset", "exactly one"),
    (HEADER, "onset", "exactly one"),
    (HEADER + "curve: {accumulate: rain minus m, onset: day after minimum}\n", "curve.accumulate", "not defined"),
    (HEADER + "thresholds: {m: {mean daily rain: true}}\ncurve: {accumulate: m, onset: day after minimum}\n",
     "curve.accumulate", "rain minus"),
    (HEADER + "thresholds: {m: {mean daily rain: true}}\ncurve: {accumulate: rain minus m, onset: first}\n",
     "curve.onset", "day after minimum"),
    (HEADER + "thresholds: {m: {mean daily rain: true, mean total over: 5 days}}\ncurve: {accumulate: rain minus m, onset: day of minimum}\n",
     "thresholds.m", "exactly one"),
    (HEADER + "thresholds: {m: {during: whole year}}\ncurve: {accumulate: rain minus m, onset: day of minimum}\n",
     "thresholds.m", "exactly one"),
    (HEADER + "thresholds: {m: {mean total over: 5 days, quantile: 0.5}}\ncurve: {accumulate: rain minus m, onset: day of minimum}\n",
     "thresholds.m.quantile", "only applies"),
    (HEADER + "thresholds: {m: {quantile of totals over: 5 days}}\ncurve: {accumulate: rain minus m, onset: day of minimum}\n",
     "thresholds.m.quantile", "between 0 and 1"),
    (HEADER + "thresholds: {m: {mean daily rain: yes please}}\ncurve: {accumulate: rain minus m, onset: day of minimum}\n",
     "thresholds.m.mean daily rain", "true"),
    (HEADER + "thresholds: {m: {mean total over: 5 days, during: April}}\ncurve: {accumulate: rain minus m, onset: day of minimum}\n",
     "thresholds.m.during", "MM-DD to MM-DD"),
    (ICPAC.replace("wet_day:", "id: u\nwet_day:"), "duplicate key", "Fix"),
    ("- a\n- b\n", "definition", "mapping"),
    ("a: [1\n", "definition", "invalid YAML"),
])
def test_validation_errors(text, path, fix):
    """Each strictness rule names the YAML path and says how to fix it."""
    with pytest.raises(dfn.DefinitionError) as exc:
        dfn.loads(text)
    message = str(exc.value)
    assert path in message and fix in message, message


def test_total_without_window_or_over():
    """A total with no window and no over needs one of them."""
    fails(onset("""\
        require:
          - total: at least 1 mm
        """), "onset.require[0].total", "window: N days", "over:")


def test_all_days_season_to_date_rejected():
    """'season to date' only applies to totals."""
    fails(onset("""\
        window: 3 days
        require:
          - wet days: at least 1
            over: season to date
        """), "onset.require[0].over", "not allowed here")


def test_extends_merge_and_override(tmp_path):
    """Mappings merge key by key; lists and scalars replace."""
    (tmp_path / "parent.yaml").write_text(ICPAC)
    child = dfn.loads("""\
extends: parent.yaml
id: child
version: 2
title: Child
status: draft
season: {to: 06-30}
onset:
  reject if: []
""", base_dir=tmp_path)
    assert child.id == "child" and child.version == 2
    assert child.seasons[0].start == DayRef(month=3, day=1) and child.seasons[0].end == DayRef(month=6, day=30)
    assert child.rule.reject_if == ()
    assert child.rule.require == dfn.loads(ICPAC).rule.require
    assert "extends" not in yaml.safe_load(child.source_yaml)
    assert yaml.safe_load(child.source_yaml)["wet_day"] == "at least 1 mm"


def test_extends_list_replaces(tmp_path):
    """A child list replaces the parent's."""
    (tmp_path / "p.yaml").write_text(ICPAC)
    child = dfn.loads("""\
extends: p.yaml
id: c
version: 1
title: C
status: draft
onset:
  require:
    - total: at least 5 mm
""", base_dir=tmp_path)
    assert child.rule.require == (Total(Span(1, 3), Comparison("ge", Amount(tenths=50))),)
    assert len(child.rule.reject_if) == 1


def test_extends_registry_and_switch_family():
    """Extending by registry id works, and a child 'seasons' replaces the parent's 'season'."""
    d = dfn.load("dunning")
    assert isinstance(d.seasons, DerivedSeasons)
    assert d.rule == dfn.load("liebmann_marengo").rule
    assert d.threshold_specs["mean_daily_rain"].kind == "mean_daily"


def test_extends_must_restate_identity():
    """An extending definition restates id, version, title and status."""
    fails("extends: icpac\nid: x\n", "extends", "'version' must be restated")


def test_extends_unknown_and_cycle(tmp_path):
    """Unknown parents and cycles are errors."""
    fails("extends: nope\nid: x\nversion: 1\ntitle: T\nstatus: draft\n", "extends", "registry id")
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nid: a\nversion: 1\ntitle: A\nstatus: draft\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nid: b\nversion: 1\ntitle: B\nstatus: draft\n")
    with pytest.raises(dfn.DefinitionError, match="cycle"):
        dfn.load(tmp_path / "a.yaml")
    (tmp_path / "self.yaml").write_text("extends: self.yaml\nid: s\nversion: 1\ntitle: S\nstatus: draft\n")
    with pytest.raises(dfn.DefinitionError, match="cycle"):
        dfn.load(tmp_path / "self.yaml")


def test_load_path_and_errors(tmp_path):
    """load takes a file path; anything else is an error listing the registry."""
    file = tmp_path / "x.yaml"
    file.write_text(ICPAC)
    assert dfn.load(file).id == "t"
    assert dfn.load(str(file)).id == "t"
    with pytest.raises(dfn.DefinitionError, match="registry id"):
        dfn.load("missing")


def test_hash_ignores_provenance_and_tracks_rule():
    """Provenance edits keep the hash; semantic edits change it."""
    base = dfn.definition_hash(dfn.loads(ICPAC))
    assert base == dfn.definition_hash(dfn.loads(ICPAC))
    edited = ICPAC.replace("title: T", "title: Other").replace("notes: n", "notes: m").replace(
        "version: 1", "version: 3").replace("id: t", "id: z")
    assert dfn.definition_hash(dfn.loads(edited)) == base
    for change in (("more than 20 mm", "more than 21 mm"), ("at least 7 days", "at least 8 days"),
                   ("at least 1 mm", "more than 1 mm"), ("05-31", "06-01"), ("window: 3", "window: 4")):
        assert dfn.definition_hash(dfn.loads(ICPAC.replace(*change))) != base, change
    assert len(base) == 64


def test_hash_tracks_thresholds():
    """Threshold specs are part of the hash."""
    text = make("""\
        thresholds:
          m: {mean daily rain: true}
        curve: {accumulate: rain minus m, onset: day after minimum}
        """)
    assert dfn.definition_hash(dfn.loads(text)) != dfn.definition_hash(
        dfn.loads(text.replace("onset: day after", "onset: day of")))
    assert dfn.definition_hash(dfn.loads(text)) != dfn.definition_hash(
        dfn.loads(text.replace("m: {mean daily rain: true}", "m: {mean daily rain: true, during: 04-01 to 10-31}")))


def test_registry_entries_load():
    """Every registry entry loads, is a draft, and has the id of its file."""
    ids = dfn.registry_ids()
    assert ids == sorted(["moron_robertson", "moron_robertson_mok", "icpac", "sivakumar", "marteau",
                          "omotosho", "liebmann_marengo", "dunning"])
    for ident in ids:
        d = dfn.load(ident)
        assert d.id == ident and d.status == "draft"
        assert d.version == (2 if ident.startswith("moron_robertson") else 1)
        assert d.wording == "TODO paste the original definition text"
        assert d.source_yaml and len(dfn.definition_hash(d)) == 64
        assert dfn.loads(d.source_yaml).rule == d.rule


def test_registry_rules():
    """Spot-check the encoded rules against the spec."""
    mr = dfn.load("moron_robertson")
    assert mr.threshold_specs["wet_spell_amount"] == ThresholdSpec(
        "mean_total", days=5, during=(DayRef(month=5, day=1), DayRef(month=9, day=30)))
    assert mr.rule.reject_if == (
        Veto(DryWindow(10, Comparison("lt", Amount(tenths=50))), Span(6, 35), "inside"),)
    mok = dfn.load("moron_robertson_mok")
    assert mok.rule.reject_if == () and mok.rule.require == mr.rule.require
    assert mok.seasons[0].start == DayRef(month=6, day=2)
    assert dfn.load("icpac").seasons[0].start == DayRef(month=5, day=1)
    siv = dfn.load("sivakumar").rule
    assert siv.reject_if[0].spell == DryRun(8) and siv.reject_if[0].span == Span(2, 31)
    assert siv.require == (Total(Span(1, 3), Comparison("ge", Amount(tenths=200))),)
    mar = dfn.load("marteau").rule
    assert mar.require[0] == WetDays(Span(1, 2), "ge", 2)
    assert mar.reject_if[0].span == Span(3, 22)
    omo = dfn.load("omotosho")
    assert [c.span for c in omo.rule.require] == [Span(1, 10), Span(11, 20)]
    assert "RULE UNVERIFIED" in omo.notes
    assert dataclasses.is_dataclass(dfn.load("liebmann_marengo").rule)


def test_dry_run_phrase_not_hashed():
    """'more than 7 days' and 'at least 8 days' differ in wording only, not in meaning or hash."""
    more = dfn.loads(ICPAC.replace("at least 7 days", "more than 7 days"))
    least = dfn.loads(ICPAC.replace("at least 7 days", "at least 8 days"))
    assert more.rule.reject_if[0].spell.phrase == "more than"
    assert more.rule == least.rule
    assert dfn.definition_hash(more) == dfn.definition_hash(least)


def test_regions_default_empty():
    """A definition without regions carries an empty tuple."""
    assert dfn.loads(ICPAC).regions == ()
