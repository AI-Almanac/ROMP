"""Definition comparison figure: dates match the engine, status rows, thresholds and error paths."""
import itertools

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from momp.onset import climatology, engine
from momp.onset import definition as defn_mod
from momp.onset import model as m
from momp.onset.viz.compare import compare_definitions

WET = m.Comparison("ge", m.Amount(tenths=10))


def make_rain(n_years=4, seed=3):
    """Four years of synthetic daily rain on a 2 x 2 grid (lat 8, 9; lon 38, 39) with a wet May to September."""
    index = pd.date_range("2000-01-01", periods=int(n_years * 365.25) + 1, freq="D")
    index = index[index.year < 2000 + n_years]
    rng = np.random.default_rng(seed)
    doy = index.dayofyear.to_numpy()
    wet_season = (doy > 120) & (doy < 270)
    values = np.where(rng.random((2, 2, len(index))) < np.where(wet_season, 0.6, 0.08),
                      rng.gamma(2.0, 6.0, (2, 2, len(index))), 0.0)
    tenths = np.rint(values * 10).astype(np.int32)
    return m.DailyRain(tenths, np.zeros(tenths.shape, bool), ("lat", "lon", "time"),
                       {"lat": np.array([8.0, 9.0]), "lon": np.array([38.0, 39.0]), "time": np.asarray(index)},
                       index.year.to_numpy(), index.month.to_numpy(), index.day.to_numpy(), "standard",
                       "00 UTC, labelled by start")


def engine_dates(definition, rain, year, iy, ix, thresholds=None):
    """(season, status, date) per season from detect_observed on the full grid."""
    clim = None
    if isinstance(definition.seasons, m.DerivedSeasons):
        clim = climatology.harmonic_seasons(rain, (2000, 2003))
    ds = engine.detect_observed(definition, rain, thresholds=thresholds, years=[year], season_climatology=clim)
    out = []
    for name in ds["season"].values:
        sel = ds.isel(lat=iy, lon=ix).sel(year=year, season=name)
        out.append((str(name), int(sel["status"]), None if int(sel["status"]) else pd.Timestamp(sel["onset_date"].values.item())))
    return out


def plotted_onsets(fig):
    """Dates of the dots in the dot panel as sorted Timestamps."""
    dots = fig.axes[1]
    return sorted(pd.Timestamp(line.get_xdata()[0]) for line in dots.lines if line.get_marker() == "o")


def test_dates_equal_the_engine_for_every_registry_definition():
    """The plotted onset dates (and statuses) are exactly detect_observed's, for each definition."""
    rain = make_rain()
    for name in ("icpac", "sivakumar", "marteau", "moron_robertson", "liebmann_marengo"):
        d = defn_mod.load(name)
        fig = compare_definitions([d], rain, lat=9.0, lon=38.0, year=2001, threshold_years=(2000, 2003))
        thresholds = climatology.compute_thresholds(d, rain, (2000, 2003)) if d.thresholds else None
        expected = engine_dates(d, rain, 2001, 1, 0, thresholds)
        want = sorted(date.normalize() for _, status, date in expected if date is not None)
        assert [t.normalize() for t in plotted_onsets(fig)] == want, name


def test_derived_seasons_match_the_engine():
    """Derived-season curve definitions use the harmonic climatology of the threshold years."""
    rain = make_rain()
    d = defn_mod.load("dunning")
    fig = compare_definitions([d], rain, lat=9.0, lon=39.0, year=2001, threshold_years=(2000, 2003))
    thresholds = climatology.compute_thresholds(d, rain, (2000, 2003))
    expected = engine_dates(d, rain, 2001, 1, 1, thresholds)
    want = sorted(date.normalize() for _, _, date in expected if date is not None)
    assert [t.normalize() for t in plotted_onsets(fig)] == want


def test_rows_labelled_with_ids_and_duplicates_told_apart():
    """Row labels are definition ids, with the no-dry-spell variant told apart from its parent."""
    rain = make_rain()
    base = defn_mod.load("icpac")
    fig = compare_definitions([base, defn_mod.without_dry_spell_check(base), defn_mod.load("sivakumar")], rain,
                              lat=9.0, lon=39.0, year=2002)
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert labels == ["icpac", "icpac (no dry-spell check)", "sivakumar"]
    expected = engine_dates(defn_mod.without_dry_spell_check(base), rain, 2002, 1, 1)
    assert expected[0][2] is not None


def test_row_without_onset_shows_status_icon_and_label():
    """A definition that finds no onset gets its status icon and label at the row end and no dot."""
    rain = make_rain()
    never = m.Definition("never", 1, "t", "", "", "draft", "", WET,
                         (m.Season("s", m.DayRef(month=1, day=1), m.DayRef(month=3, day=1)),),
                         m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("gt", m.Amount(tenths=9000))),)))
    fig = compare_definitions([never, defn_mod.load("icpac")], rain, lat=8.0, lon=38.0, year=2001)
    texts = [t.get_text() for t in fig.axes[1].texts]
    assert "– no onset" in texts
    assert len(plotted_onsets(fig)) == 1


def test_given_thresholds_are_used_and_slicing_matches_the_full_grid():
    """Thresholds passed in (full-grid DataArrays) give the same dates as computing them."""
    rain = make_rain()
    d = defn_mod.load("moron_robertson")
    given = {d.id: climatology.compute_thresholds(d, rain, (2000, 2003))}
    a = compare_definitions([d], rain, lat=9.0, lon=39.0, year=2001, thresholds=given)
    b = compare_definitions([d], rain, lat=9.0, lon=39.0, year=2001, threshold_years=(2000, 2003))
    assert plotted_onsets(a) == plotted_onsets(b)
    assert len(plotted_onsets(a)) == 1


def test_nearest_cell_and_title():
    """The cell is the nearest grid point and the title names it."""
    rain = make_rain()
    fig = compare_definitions([defn_mod.load("icpac")], rain, lat=8.9, lon=38.2, year=2001)
    assert fig.axes[0].get_title("left").startswith("9\N{DEGREE SIGN}N 38\N{DEGREE SIGN}E")


def test_dark_mode_and_no_overlapping_labels():
    """Dark mode builds, and no two text labels overlap."""
    rain = make_rain()
    defs = [defn_mod.load("icpac"), defn_mod.without_dry_spell_check(defn_mod.load("icpac")), defn_mod.load("marteau")]
    for mode in ("light", "dark"):
        fig = compare_definitions(defs, rain, lat=9.0, lon=38.0, year=2001, mode=mode)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[1]
        boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in ax.texts + ax.get_yticklabels() if t.get_text()]
        for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
            assert not ba.overlaps(bb), f"{a!r} overlaps {b!r}"


def test_season_selection():
    """Choosing a season by name keeps only that season; an unknown name is an error."""
    rain = make_rain()
    two = m.Definition("two", 1, "t", "", "", "draft", "", WET, (
        m.Season("early", m.DayRef(month=3, day=1), m.DayRef(month=6, day=30)),
        m.Season("late", m.DayRef(month=7, day=1), m.DayRef(month=10, day=31))),
        m.FirstPassage((m.Total(m.Span(1, 3), m.Comparison("ge", m.Amount(tenths=100))),)))
    both = compare_definitions([two], rain, lat=9.0, lon=38.0, year=2001)
    only = compare_definitions([two], rain, lat=9.0, lon=38.0, year=2001, season="late")
    assert len(plotted_onsets(both)) == 2 and len(plotted_onsets(only)) == 1
    with pytest.raises(ValueError, match="no season 'autumn'"):
        compare_definitions([two], rain, lat=9.0, lon=38.0, year=2001, season="autumn")


def test_error_paths():
    """Missing thresholds, bad inputs and a non-standard calendar are clear errors."""
    rain = make_rain()
    with pytest.raises(ValueError, match="definition 'moron_robertson' needs thresholds"):
        compare_definitions([defn_mod.load("icpac"), defn_mod.load("moron_robertson")], rain, lat=9.0, lon=38.0, year=2001)
    derived = m.Definition("derived", 1, "t", "", "", "draft", "", WET,
                           m.DerivedSeasons("harmonic", m.DayRef(climatological="onset"), m.DayRef(climatological="cessation")),
                           m.FirstPassage((m.Total(m.Span(1, 1), m.Comparison("ge", m.Amount(tenths=100))),)))
    with pytest.raises(ValueError, match="derives its seasons"):
        compare_definitions([derived], rain, lat=9.0, lon=38.0, year=2001)
    with pytest.raises(ValueError, match="at least one definition"):
        compare_definitions([], rain, lat=9.0, lon=38.0, year=2001)
    with pytest.raises(ValueError, match="outside the rain grid"):
        compare_definitions([defn_mod.load("icpac")], rain, lat=20.0, lon=38.0, year=2001)
    with pytest.raises(ValueError, match="standard calendar"):
        compare_definitions([defn_mod.load("icpac")], make_rain().__class__(**{**rain.__dict__, "calendar": "noleap"}),
                            lat=9.0, lon=38.0, year=2001)
    with pytest.raises(ValueError, match="mode"):
        compare_definitions([defn_mod.load("icpac")], rain, lat=9.0, lon=38.0, year=2001, mode="sepia")
