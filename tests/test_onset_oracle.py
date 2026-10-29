"""Hand-worked cases for the brute-force oracle, so the oracle itself is known to follow the spec."""
import numpy as np

from momp.onset import model as m
from onset_oracle import oracle_detect


def _icpac_like():
    """Wet day >= 1 mm; day 1 wet and 3-day total > 20 mm; no dry run of 7 days in the 21 days after onset."""
    return m.Definition(
        id="t", version=1, title="t", citation="", wording="", status="draft", notes="",
        wet_day=m.Comparison("ge", m.Amount(tenths=10)),
        seasons=(m.Season("main", m.DayRef(1, 1), m.DayRef(12, 31)),),
        rule=m.FirstPassage(
            require=(m.WetDays(m.Span(1, 1), "ge", 1), m.Total(m.Span(1, 3), m.Comparison("gt", m.Amount(tenths=200)))),
            reject_if=(m.Veto(m.DryRun(7), m.Span(2, 22), "inside"),),
        ),
    )


def _series():
    """A false start on day 2 (followed by 7 dry days), then a real onset on day 40."""
    x = np.full(80, 15, dtype=np.int32)
    x[0:2] = 0
    x[2:5] = 80
    x[5:7] = 20
    x[7:14] = 0
    x[40:43] = 90
    return x


def test_false_start_is_vetoed_and_the_later_spell_is_onset():
    """The day-2 spell is followed by a dry run, so onset is day 40."""
    x = _series()
    assert oracle_detect(_icpac_like(), x, np.zeros(80, bool), {}, 0, 60, 80) == (40, m.Status.ONSET)


def test_veto_window_past_the_horizon_is_censored():
    """With data ending on day 50, the day-40 candidate's veto span runs past the horizon."""
    x = _series()
    assert oracle_detect(_icpac_like(), x, np.zeros(80, bool), {}, 0, 60, 50) == (-1, m.Status.CENSORED)


def test_missing_day_in_the_veto_span_is_insufficient_data():
    """A missing day inside the day-40 candidate's veto span leaves it undecided."""
    x = _series()
    missing = np.zeros(80, bool)
    missing[45] = True
    x[45] = 0
    assert oracle_detect(_icpac_like(), x, missing, {}, 0, 60, 80) == (-1, m.Status.INSUFFICIENT_DATA)


def test_a_false_requirement_outweighs_a_missing_day():
    """A candidate whose first day is dry is False even when its window has a missing day."""
    x = _series()
    missing = np.zeros(80, bool)
    missing[1] = True
    assert oracle_detect(_icpac_like(), x, missing, {}, 0, 60, 80) == (40, m.Status.ONSET)


def test_curve_onset_is_the_day_after_the_minimum():
    """Rain above the 1 mm/day baseline starts on day 3, so the anomaly bottoms out on day 2."""
    defn = m.Definition(
        id="c", version=1, title="c", citation="", wording="", status="draft", notes="",
        wet_day=m.Comparison("ge", m.Amount(tenths=10)),
        seasons=(m.Season("main", m.DayRef(1, 1), m.DayRef(12, 31)),),
        rule=m.Curve(baseline="B"),
    )
    x = np.array([0, 0, 0, 50, 50, 50, 0, 0], dtype=np.int32)
    assert oracle_detect(defn, x, np.zeros(8, bool), {"B": 1.0}, 0, 7, 8) == (3, m.Status.ONSET)
