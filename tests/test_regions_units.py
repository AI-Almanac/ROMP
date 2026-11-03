"""Region boxes, per-region evaluation units and year-crossing seasons."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from hcwf.onset import calendar as cal
from hcwf.onset import definition as dfn
from hcwf.onset import regions as rg
from hcwf.onset.model import DayRef, Season
from hcwf.spatial.region_def import domain

NE_COUNTRY = {"ethiopia": "Ethiopia", "india": "India", "kenya": "Kenya"}


@pytest.fixture
def region_dir(tmp_path, monkeypatch):
    """Point the region registry at a temporary directory."""
    monkeypatch.setattr(rg, "REGIONS_DIR", tmp_path)
    rg._index.cache_clear()
    yield tmp_path
    rg._index.cache_clear()


def test_every_region_has_a_box():
    """Every bundled region declares a lat/lon box with the south below the north."""
    for ident in rg.region_ids():
        lats, latn, lonw, lone = rg.box(ident)
        assert -90 <= lats < latn <= 90 and -180 <= lonw < lone <= 360, ident


@pytest.mark.parametrize("ident,country", NE_COUNTRY.items())
def test_country_box_holds_its_geometry(ident, country):
    """A country's box contains its Natural Earth shape's centroid and, within a margin, its whole outline."""
    from hcwf.spatial.masks import country_geometry

    geometry = country_geometry([country], "110m")[0]
    lats, latn, lonw, lone = rg.box(ident)
    centre = geometry.centroid
    assert lats <= centre.y <= latn and lonw <= centre.x <= lone
    west, south, east, north = geometry.bounds
    assert lats <= south + 0.6 and north - 0.6 <= latn and lonw <= west + 0.6 and east - 0.6 <= lone


def test_region_boxes_nest_around_their_countries():
    """A region's box holds the box of the countries that are part of it (the Sahel strip aside)."""
    for ident in rg.region_ids():
        child = rg.load_region(ident)
        for parent in rg.ancestors(child):
            if ident in ("sahel", "sub_ethiopia") or parent.id == "sahel":
                continue
            (a, b), (c, d) = parent.lat, parent.lon
            assert a <= child.lat[0] and child.lat[1] <= b and c <= child.lon[0] and child.lon[1] <= d, (ident, parent.id)


def test_lookups_ignore_case_and_use_aliases():
    """Any spelling of a region gives the same entry and, bar the legacy name, the same box."""
    assert rg.box("EAST AFRICA") == rg.box("east_africa") == rg.box("East Africa")
    assert rg.box("Amazon Basin") == rg.box("amazon")
    assert rg.box("sub_ethiopia") == rg.box("Sub_Ethiopia") == (7.5, 13.5, 37.5, 40)
    assert rg.box("ETHIOPIA") == domain("Ethiopia") == (3, 15, 33, 48)
    with pytest.raises(rg.RegionError, match="unknown region"):
        rg.box("atlantis")


def test_india_has_one_box_whatever_the_spelling():
    """`India`, `india` and `INDIA` are the same region with the registry's box."""
    assert domain("India") == rg.box("India") == domain("india") == domain("INDIA") == (6.5, 38.5, 66.5, 100)
    assert rg.load_region("India") is rg.load_region("india")


def test_box_without_domain_is_an_error(region_dir):
    """A region with no box says how to add one."""
    (region_dir / "flat.yaml").write_text("id: flat\ntitle: Flat\nkind: region\n")
    with pytest.raises(rg.RegionError, match="no domain"):
        rg.box("flat")


def test_legacy_domains_are_no_longer_a_region_key(region_dir):
    """The case-sensitive legacy box was retired: a region file naming it is an unknown-key error."""
    (region_dir / "flat.yaml").write_text("id: flat\ntitle: Flat\nkind: region\nlegacy_domains:\n  Flat: {lat: [5, 1], lon: [0, 1]}\n")
    with pytest.raises(rg.RegionError, match="legacy_domains"):
        rg.load_region("flat")


# --- units ---------------------------------------------------------------------------------------------------

def _shapes(folder):
    """Two square polygons written as a GeoJSON file."""
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame({"code": ["a1", "b2"], "label": ["Alpha", "Beta"]},
                             geometry=[box(35, 5, 36, 6), box(37, 7, 38, 8)], crs="EPSG:4326")
    path = folder / "units.geojson"
    frame.to_file(path, driver="GeoJSON")
    return path


def test_polygon_units_from_a_region_entry(region_dir, monkeypatch):
    """A declared shapefile (path with ${ENV} and relative parts) and a dissemination list give polygon units."""
    data = region_dir / "data"
    data.mkdir()
    _shapes(data)
    pd.DataFrame({"id": ["b2"]}).to_csv(data / "dissem.csv", index=False)
    monkeypatch.setenv("HCWF_TEST_UNITS", str(data))
    (region_dir / "land.yaml").write_text(
        "id: land\ntitle: Land\nkind: region\ndomain: {lat: [0, 10], lon: [30, 40]}\nunits:\n"
        "  shapefile: ${HCWF_TEST_UNITS}/units.geojson\n  unit_id_col: code\n  unit_name_col: label\n"
        "  dissemination: data/dissem.csv\n")
    (region_dir / "town.yaml").write_text("id: town\ntitle: Town\nkind: region\npart_of: [land]\n")
    for name in ("land", "town"):  # town inherits the units of land
        units = rg.units_for(name)
        assert units.kind == "polygon"
        assert list(units.ids) == ["a1", "b2"]
        assert list(units.column("name")) == ["Alpha", "Beta"]
        assert list(units.subsets["dissemination"]) == [False, True]


def test_weights_units_from_a_region_entry(region_dir):
    """A weights CSV with a centroids CSV, relative to the YAML, gives units named by the weights' targets."""
    pd.DataFrame({"source_id": ["s1", "s2"], "target_id": ["u1", "u2"], "weight": [1.0, 1.0]}).to_csv(
        region_dir / "w.csv", index=False)
    pd.DataFrame({"target_id": ["u1", "u2"], "lat": [1.0, 2.0], "lon": [3.0, 4.0]}).to_csv(region_dir / "c.csv", index=False)
    (region_dir / "land.yaml").write_text("id: land\ntitle: Land\nkind: region\nunits:\n  weights: w.csv\n  centroids: c.csv\n")
    units = rg.units_for("land")
    assert list(units.ids) == ["u1", "u2"] and list(units.lat) == [1.0, 2.0]


def test_grid_units_from_the_region_box(region_dir):
    """Without a units block the units are the grid cells inside the box and the mask; a grid is required."""
    (region_dir / "box.yaml").write_text("id: box\ntitle: Box\nkind: region\ndomain: {lat: [0, 2], lon: [10, 12]}\n")
    lat, lon = np.arange(-2.0, 5.0), np.arange(8.0, 15.0)
    units = rg.units_for("box", lat, lon)
    assert units.kind == "grid" and len(units) == 9
    assert units.lat.min() == 0 and units.lat.max() == 2 and units.lon.min() == 10 and units.lon.max() == 12
    with pytest.raises(rg.RegionError, match="need lat and lon"):
        rg.units_for("box")


@pytest.mark.parametrize("block,message", [
    ("units: {unit_id_col: id}", "exactly one"),
    ("units: {shapefile: a.shp, weights: w.csv}", "exactly one"),
    ("units: {shapefile: a.shp}", "unit_id_col"),
    ("units: {weights: w.csv, unit_id_col: id}", "does not go"),
    ("units: {shapefile: a.shp, unit_id_col: id, colour: red}", "unknown key"),
])
def test_bad_units_blocks(region_dir, block, message):
    """A malformed units block is a RegionError naming the key."""
    (region_dir / "bad.yaml").write_text(f"id: bad\ntitle: Bad\nkind: region\n{block}\n")
    with pytest.raises(rg.RegionError, match=message):
        rg.load_region("bad")


def test_unset_environment_variable_in_a_units_path(region_dir, monkeypatch):
    """An unset ${VAR} is reported, not read as a folder name."""
    monkeypatch.delenv("HCWF_TEST_MISSING", raising=False)
    (region_dir / "bad.yaml").write_text(
        "id: bad\ntitle: Bad\nkind: region\nunits: {shapefile: '${HCWF_TEST_MISSING}/u.shp', unit_id_col: id}\n")
    with pytest.raises(rg.RegionError, match="not set"):
        rg.units_for("bad")


# --- seasons crossing 1 January ------------------------------------------------------------------------------

def test_every_region_default_season_is_representable():
    """Each region's default and also definitions give fixed windows that end after they begin, in any year."""
    keys = cal.label_keys(*np.array([(d.year, d.month, d.day) for d in pd.date_range("2000-01-01", "2004-12-31")]).T)
    for ident in rg.region_ids():
        default, also = rg.definitions_for(ident)
        for name in ([default] if default else []) + also:
            seasons = dfn.load(name).seasons
            if not isinstance(seasons, tuple):
                continue
            for season in seasons:
                first, last = cal.fixed_bounds(keys, [2001, 2002], season.start, season.end)
                assert (first >= 0).all() and (last >= first).all(), (ident, name, season.name)


def test_southern_hemisphere_season_crosses_the_new_year():
    """An October-March window (Amazon, southern Africa) ends in the following year."""
    keys = cal.label_keys(*np.array([(d.year, d.month, d.day) for d in pd.date_range("2000-01-01", "2002-12-31")]).T)
    season = Season("wet", DayRef(month=10, day=1), DayRef(month=3, day=31))
    first, last = cal.fixed_bounds(keys, [2000, 2001], season.start, season.end)
    assert list(keys[first]) == [20001001, 20011001] and list(keys[last]) == [20010331, 20020331]
    assert rg.box("amazon")[0] < 0 < rg.box("amazon")[1]  # the Amazon box straddles the equator, longitudes negative
    assert (last - first + 1).tolist() == [182, 182]


def test_region_dataclass_stays_hashable():
    """Region entries with units stay usable as dict keys and frozen."""
    spec = rg.UnitsSpec(base=rg.REGIONS_DIR, shapefile="a.shp", unit_id_col="id")
    region = dataclasses.replace(rg.load_region("kenya"), units=spec)
    assert hash(region) is not None
