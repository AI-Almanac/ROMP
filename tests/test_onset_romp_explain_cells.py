"""ROMP explain_cells: the files are written per cell and year, legacy mode is skipped, and bad values are rejected."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.lib import explain_cells
from momp.lib.assertion import ROMPConfigError, ROMPValidator


def rain_year(year):
    """A year of seasonal rain (time, lat, lon) in mm."""
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    rng = np.random.default_rng(year)
    chance = 0.08 + 0.5 * np.clip(np.sin((time.dayofyear.to_numpy() - 100) / 365 * 2 * np.pi), 0, None)
    values = np.where(rng.random((len(time), 2, 2)) < chance[:, None, None], rng.gamma(0.9, 8.0, (len(time), 2, 2)), 0.0)
    return xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time, "lat": [8.0, 9.0], "lon": [38.0, 39.0]})


def config(tmp_path, cells):
    """A definition-mode config namespace for explain_cells over 2001-2002."""
    return SimpleNamespace(explain_cells=cells, onset_legacy=False, onset_definition="icpac",
                           obs_day_boundary="00 UTC, labelled by start", dir_fig=str(tmp_path), start_date=(2001, 5, 1),
                           end_date=(2002, 7, 31))


@pytest.fixture
def data(monkeypatch):
    """Serve synthetic observations and no thresholds instead of reading files."""
    monkeypatch.setattr(explain_cells, "load_imd_rainfall", lambda year, **kw: rain_year(year))
    monkeypatch.setattr(explain_cells, "load_thresholds", lambda **kw: {})


def test_writes_a_png_and_log_per_cell_and_year(tmp_path, data):
    """Each cell and year gets an anatomy PNG and the explain() log under dir_fig/explain."""
    written = explain_cells.write_explain_cells(config(tmp_path, [(9.0, 38.0), (8.0, 39.0)]))
    names = sorted(p.name for p in written)
    assert names == ["icpac_8N_39E_2001.png", "icpac_8N_39E_2002.png", "icpac_9N_38E_2001.png", "icpac_9N_38E_2002.png"]
    for png in written:
        assert png.parent == tmp_path / "explain" and png.stat().st_size > 1000
        assert "Result:" in png.with_suffix(".txt").read_text()


def test_unset_writes_nothing(tmp_path, data):
    """explain_cells = None is a no-op."""
    assert explain_cells.write_explain_cells(config(tmp_path, None)) == []
    assert not (tmp_path / "explain").exists()


def test_legacy_mode_is_skipped_with_a_note(tmp_path, data, monkeypatch, capsys):
    """With the old flat keys it prints a note and writes nothing."""
    monkeypatch.setattr(explain_cells, "onset_setup", lambda kwargs: SimpleNamespace(legacy=True))
    assert explain_cells.write_explain_cells(config(tmp_path, [(9.0, 38.0)])) == []
    assert "skipped" in capsys.readouterr().out


@pytest.mark.parametrize("value", [(9.0, 38.0), [(9.0,)], [(9.0, "a")], [(9.0, 38.0, 1.0)], "x"])
def test_validation_rejects_malformed_cells(value):
    """Anything but None or a list of numeric pairs is a config error."""
    validator = ROMPValidator({"explain_cells": value})
    validator._check_explain_cells()
    assert validator.errors


@pytest.mark.parametrize("value", [None, [], [(9.0, 38.75), [8, 39]]])
def test_validation_accepts_good_cells(value):
    """None, an empty list and numeric pairs pass."""
    validator = ROMPValidator({"explain_cells": value})
    validator._check_explain_cells()
    assert not validator.errors
