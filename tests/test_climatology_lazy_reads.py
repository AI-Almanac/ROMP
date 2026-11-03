"""The climatology years are read only when their onsets are not cached (detect_observed calls the year's reader)."""
from types import SimpleNamespace

import pytest

from hcwf.references import climatology


@pytest.fixture
def reads(monkeypatch):
    """Records the years observation_years_of is asked for; it yields the year as the rain of each."""
    asked = []

    def fake(years, rc, select):
        asked.append(list(years))
        for year in years:
            yield f"rain{year}", None

    monkeypatch.setattr(climatology, "observation_years_of", fake)
    monkeypatch.setattr(climatology, "reference_held", lambda: False)
    return asked


RC = SimpleNamespace(onset=SimpleNamespace(obs_day_boundary=None), observations=None, area=None)


def test_nothing_is_read_until_a_reader_is_called(reads):
    readers = climatology._year_readers([2000, 2001, 2002], RC, None)
    assert reads == []
    assert [r() for r in readers] == ["rain2000", "rain2001", "rain2002"]
    assert reads == [[2000, 2001, 2002]]   # one pass over the years, in order


def test_a_year_after_cached_ones_restarts_the_read_there(reads):
    readers = climatology._year_readers([2000, 2001, 2002, 2003], RC, None)
    assert readers[2]() == "rain2002"   # 2000 and 2001 were cached: never read
    assert readers[3]() == "rain2003"
    assert reads == [[2002, 2003]]


def test_held_reference_years_are_not_read(reads, monkeypatch):
    monkeypatch.setattr(climatology, "reference_held", lambda: True)
    monkeypatch.setattr(climatology, "reference_rain_year", lambda year, *a: f"held{year}" if year < 2002 else None)
    readers = climatology._year_readers([2000, 2001, 2002, 2003], RC, None)
    assert [r() for r in readers] == ["held2000", "held2001", "rain2002", "rain2003"]
    assert reads == [[2002, 2003]]


def test_a_read_error_is_raised_by_the_reader(monkeypatch):
    monkeypatch.setattr(climatology, "reference_held", lambda: False)
    monkeypatch.setattr(climatology, "observation_years_of", lambda years, rc, select: iter([(None, OSError("gone"))]))
    with pytest.raises(OSError, match="gone"):
        climatology._year_readers([2000], RC, None)[0]()
