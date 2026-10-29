"""Forecast search windows honour day offsets on season bounds (the legacy translation ends 47 days after end_date)."""
import numpy as np
import pytest

from momp.onset import model as m
from momp.onset.engine import build_forecast_jobs


def _definition(end_offset: int) -> m.Definition:
    """A one-day rule searched from 1 June to 30 September plus end_offset days."""
    return m.Definition(
        id="o", version=1, title="o", citation="", wording="", status="draft", notes="",
        wet_day=m.Comparison("ge", m.Amount(tenths=10)),
        seasons=(m.Season("main", m.DayRef(6, 1), m.DayRef(9, 30, offset_days=end_offset)),),
        rule=m.FirstPassage(require=(m.WetDays(m.Span(1, 1), "ge", 1),)),
    )


def _forecast(lead_label: str) -> m.ForecastRain:
    """One init on 2020-09-20 with 40 leads on a single cell."""
    n_leads = 40
    return m.ForecastRain(
        tenths=np.zeros((1, 1, 1, n_leads), np.int32), missing=np.zeros((1, 1, 1, n_leads), bool),
        dims=("init_time", "lat", "lon", "lead"),
        coords={"init_time": np.array(["2020-09-20"], "datetime64[ns]"), "lat": np.array([0.0]),
                "lon": np.array([0.0]), "lead": np.arange(1, n_leads + 1)},
        init_year=np.array([2020]), init_month=np.array([9]), init_day=np.array([20]),
        lead_label=lead_label, calendar="standard", day_boundary="00 UTC, labelled by start",
    )


@pytest.mark.parametrize(("lead_label", "end_offset", "last_lead"), [
    ("end", 0, 10),      # lead k is dated 20 Sep + k: 30 Sep is lead 10
    ("end", 15, 25),     # 15 Oct is lead 25
    ("start", 0, 11),    # lead k is dated 20 Sep + k - 1
    ("start", 47, 40),   # 16 Nov is past the last lead, so every lead qualifies
])
def test_season_end_offset_extends_the_candidate_leads(lead_label, end_offset, last_lead):
    """The last candidate lead moves with the end bound's day offset."""
    jobs, _ = build_forecast_jobs(_definition(end_offset), _forecast(lead_label), lead_window=(1, 40))

    assert int(jobs.start[0]) == 0
    assert int(jobs.end[0]) == last_lead - 1
