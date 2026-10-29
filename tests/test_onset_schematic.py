"""Rule schematics: every registry definition and every rule construct draws without overlapping text."""
import itertools

import matplotlib
import pytest

matplotlib.use("Agg")

from momp.onset import definition as defn_mod
from momp.onset import model as m
from momp.onset.viz.schematic import schematic

WET = m.Comparison("ge", m.Amount(tenths=10))


def span_total(first, last, tenths=200, op="gt"):
    """A total requirement over days first..last."""
    return m.Total(m.Span(first, last), m.Comparison(op, m.Amount(tenths=tenths)))


def make(rule, seasons=None, thresholds=(), id="synthetic"):
    """A Definition around a rule with one default season."""
    seasons = seasons or (m.Season("s", m.DayRef(month=3, day=1), m.DayRef(month=5, day=31)),)
    return m.Definition(id, 1, "Synthetic definition", "", "", "draft", "", WET, seasons, rule, thresholds)


MEAN5 = ("wet_spell_amount", m.ThresholdSpec("mean_total", days=5, during=(m.DayRef(month=5, day=1), m.DayRef(month=9, day=30))))
SYNTHETIC = {
    "look_back": make(m.FirstPassage((span_total(-4, 1), m.WetDays(m.Span(0, 1), "ge", 2)))),
    "offset_spans": make(m.FirstPassage((span_total(1, 10), span_total(11, 20, 200, "ge"), span_total(3, 3)))),
    "season_to_date": make(m.FirstPassage((m.Total(m.Span(1, 1, since_search_start=True),
                                                    m.Comparison("ge", m.Amount(tenths=510))), span_total(1, 3)))),
    "both_placements": make(m.FirstPassage(
        (m.WetDays(m.Span(1, 1), "ge", 1), span_total(1, 5)),
        (m.Veto(m.DryRun(7), m.Span(2, 22), "inside"), m.Veto(m.DryRun(10, "more than"), m.Span(6, 35), "starts_inside")))),
    "dry_windows": make(m.FirstPassage(
        (m.WetDays(m.Span(1, 2), "ge", 2),),
        (m.Veto(m.DryWindow(10, m.Comparison("lt", m.Amount(tenths=50))), m.Span(3, 32), "inside"),
         m.Veto(m.DryWindow(5, m.Comparison("le", m.Amount(threshold="wet_spell_amount"))), m.Span(3, 12),
                "starts_inside"))), thresholds=(MEAN5,)),
    "wet_counts": make(m.FirstPassage((m.WetDays(m.Span(1, 5), "ge", 3), m.WetDays(m.Span(1, 4), "ge", 4),
                                       m.WetDays(m.Span(3, 6), "ge", 4)))),
    "report_first_wet_day": make(m.FirstPassage((span_total(1, 5),), report="first_wet_day", report_span=m.Span(1, 5))),
    "derived_curve": make(m.Curve("mean_daily_rain", "day_of_minimum"),
                          seasons=m.DerivedSeasons("harmonic", m.DayRef(climatological="onset", offset_days=-50),
                                                   m.DayRef(climatological="cessation")),
                          thresholds=(("mean_daily_rain", m.ThresholdSpec("mean_daily")),)),
    "two_seasons": make(m.FirstPassage((span_total(1, 3),)), seasons=(
        m.Season("long_rains", m.DayRef(month=2, day=1), m.DayRef(month=5, day=31)),
        m.Season("short_rains", m.DayRef(month=9, day=1), m.DayRef(month=1, day=15)))),
    "long_axis": make(m.FirstPassage((span_total(1, 3),), (m.Veto(m.DryRun(20), m.Span(2, 120), "inside"),))),
}


def text_boxes(fig):
    """Window extents of every non-empty text in the figure, with a name for failures."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    texts = list(fig.texts) + list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]
    return [(t.get_text(), t.get_window_extent(renderer)) for t in texts if t.get_text().strip() and t.get_visible()]


def assert_clean(fig):
    """No two texts overlap and every text lies inside the figure."""
    boxes = text_boxes(fig)
    width, height = fig.canvas.get_width_height()
    for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
        assert not ba.overlaps(bb), f"{a!r} overlaps {b!r}"
    for text, box in boxes:
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1, f"{text!r} is clipped"


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("name", defn_mod.registry_ids())
def test_registry_definitions_draw_cleanly(name, mode):
    """Every bundled definition gets a schematic in both modes with no overlapping text."""
    assert_clean(schematic(defn_mod.load(name), mode=mode))


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("name", SYNTHETIC)
def test_every_construct_draws_cleanly(name, mode):
    """Look-back, offset, season-to-date, both veto placements, dry windows, reporting, derived seasons and curves."""
    assert_clean(schematic(SYNTHETIC[name], mode=mode))


def test_variants_of_registry_definitions_draw():
    """Definitions changed by override or without their vetoes still draw."""
    base = defn_mod.load("icpac")
    assert_clean(schematic(defn_mod.without_dry_spell_check(base)))


def test_figsize_sets_width_and_grows_height():
    """The width follows figsize and a too-small height is raised to fit the content."""
    fig = schematic(defn_mod.load("icpac"), figsize=(11, 1))
    assert fig.get_size_inches()[0] == pytest.approx(11)
    assert fig.get_size_inches()[1] > 3
    assert_clean(fig)


def test_text_reads_from_the_definition_only():
    """Labels carry the span, the amounts, the dry run and the threshold in words."""
    fig = schematic(SYNTHETIC["both_placements"])
    words = " ".join(t for t, _ in text_boxes(fig))
    assert "no dry run of at least 7 days within days 2 to 22" in words
    assert "no dry run of more than 9 days starting within days 6 to 35 (it may end later)" in words
    assert "day 1 wet: at least 1 mm" in words
    fig = schematic(SYNTHETIC["dry_windows"])
    words = " ".join(t for t, _ in text_boxes(fig))
    assert "wet_spell_amount = mean 5-day total, 1 May-30 Sep" in words
    assert "both days wet" in words


def test_season_to_date_label_and_search_start():
    """Season-to-date requirements are labelled and anchored at the search start."""
    words = " ".join(t for t, _ in text_boxes(schematic(SYNTHETIC["season_to_date"])))
    assert "season-to-date total at least 51 mm" in words
    assert "search start" in words


def test_curve_marks_minimum_and_onset():
    """Curve rules mark the minimum and say where the onset falls."""
    words = " ".join(t for t, _ in text_boxes(schematic(defn_mod.load("liebmann_marengo"))))
    assert "minimum" in words and "the day after the minimum" in words
    words = " ".join(t for t, _ in text_boxes(schematic(SYNTHETIC["derived_curve"])))
    assert "the day of the minimum" in words


def test_unknown_mode_is_an_error():
    """A mode other than light or dark is rejected."""
    with pytest.raises(ValueError, match="mode"):
        schematic(defn_mod.load("icpac"), mode="sepia")
