"""A year that cannot be processed stops the climatological onset, once, naming the year (no silent skip)."""
import logging

import pytest

from hcwf.stats import climatology


def test_a_failing_year_stops_the_climatology_and_is_named_once(monkeypatch, caplog):
    monkeypatch.setattr(climatology, "load_thresholds", lambda **kw: 1.0)
    monkeypatch.setattr(climatology, "_read_years",
                        lambda years, kwargs, shared=True: iter([(None, OSError("no file")) for _ in years]))
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="no file") as caught:
        climatology.compute_climatological_onset(years_clim=[2001, 2002])
    assert any("year 2001" in note for note in caught.value.__notes__)
    assert not [r for r in caplog.records if "2001" in r.getMessage()]


def test_a_failing_year_stops_the_climatological_dataset_too(monkeypatch, caplog):
    monkeypatch.setattr(climatology, "load_thresholds", lambda **kw: 1.0)
    monkeypatch.setattr(climatology, "_read_years",
                        lambda years, kwargs, shared=True: iter([(None, OSError("no file")) for _ in years]))
    names = ("obs_dir", "obs_file_pattern", "obs_var", "start_date")
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="no file") as caught:
        climatology.compute_climatological_onset_dataset(years_clim=[2001, 2002], **dict.fromkeys(names))
    assert any("year 2001" in note for note in caught.value.__notes__)
    assert not [r for r in caplog.records if "2001" in r.getMessage()]
