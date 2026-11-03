"""A year that cannot be processed stops the climatological onset, once, naming the year (no silent skip)."""
import typed_io
import logging
import sys
from pathlib import Path

import pytest
from hcwf.references import climatology

sys.path.insert(0, str(Path(__file__).parent))
from old_flat_rule import old_rule  # noqa: E402

RUN = {**old_rule(), "max_forecast_day": 20, "years_clim": [2001, 2002]}


@pytest.fixture
def failing_reads(monkeypatch):
    """Every climatology year fails to read."""
    monkeypatch.setattr(climatology, "thresholds_of", lambda rc, model, select=None: 1.0)
    monkeypatch.setattr(climatology, "_read_years",
                        lambda years, rc, select, shared=True: iter([(None, OSError("no file")) for _ in years]))


def test_a_failing_year_stops_the_climatology_and_is_named_once(failing_reads, caplog):
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="no file") as caught:
        climatology.climatological_onset(*typed_io.from_case_kwargs(RUN))
    assert any("year 2001" in note for note in caught.value.__notes__)
    assert not [r for r in caplog.records if "2001" in r.getMessage()]


def test_a_failing_year_stops_the_climatological_dataset_too(failing_reads, caplog):
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="no file") as caught:
        climatology.climatological_onset_dataset(*typed_io.from_case_kwargs(RUN))
    assert any("year 2001" in note for note in caught.value.__notes__)
    assert not [r for r in caplog.records if "2001" in r.getMessage()]
