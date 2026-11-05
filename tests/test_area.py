"""The area of a run: ``box``, a run without a region, and what a run without a region must state itself.

A run may have ``region: null`` only with an area: a ``box`` (or ``box: all``) or a ``mask`` with a country or a file.
Without a region nothing of a region's defaults applies: the onset definition, the season (of a definition with more
than one), the climatology years and the threshold years (of a definition that computes thresholds) must be set.
"""
import numpy as np
import pytest
import xarray as xr

from hcwf.cache import reset_run_state
from hcwf.config import boxspec
from hcwf.config.build import resolve
from hcwf.errors import ConfigSchemaError, HCWFConfigError
from hcwf.spatial import area as area_mod
from hcwf.spatial.select import area_box
from test_run_experience import raw_config, write_config, write_forecast, write_obs


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Observations and one deterministic forecast on the 2 degree synthetic grid."""
    base = tmp_path_factory.mktemp("area_world")
    write_obs(base / "obs")
    write_forecast(base / "DET")
    return base


@pytest.fixture(autouse=True)
def fresh():
    """Start every test without the state of a previous run."""
    reset_run_state()
    yield
    reset_run_state()


def no_region(world, **top):
    """Return a nested config without a region, with the keys of ``top`` added and the definition set."""
    raw = raw_config(world)
    raw["region"] = None
    raw["onset"]["definition"] = "icpac"
    raw.update(top)
    return raw


# --- the box key ---------------------------------------------------------------------------------------------

def test_a_box_needs_the_four_bounds_known_keys_and_ordered_latitudes():
    """Check that a box needs the four bounds known keys and ordered latitudes."""
    ok = {"lat_min": 3, "lat_max": 15, "lon_min": 33, "lon_max": 48}
    assert boxspec.parse_box(ok, allow_all=True) == {k: float(v) for k, v in ok.items()}
    assert boxspec.parse_box("all", allow_all=True) == "all"
    for bad, text in (({"lat_min": 3}, "lat_max is missing"), ({**ok, "south": 1}, "unknown key 'south'"),
                      ({**ok, "lat_min": 20}, "lat_min 20 is above lat_max 15"), ({**ok, "lat_max": 95}, "outside -90 to 90"),
                      ({**ok, "lon_min": "west"}, "must be a number"), ([1, 2, 3, 4], "expected a mapping"),
                      ("everything", "expected a mapping")):
        with pytest.raises(ValueError, match=text):
            boxspec.parse_box(bad, allow_all=True)
    with pytest.raises(ValueError):   # 'all' is only for the run's box
        boxspec.parse_box("all")


def test_the_units_box_uses_the_same_parser_with_open_bounds():
    """Check that the units box uses the same parser with open bounds."""
    assert boxspec.parse_units_box({"lat_min": 1}) == {"lat_min": 1.0}
    assert boxspec.parse_units_box("kenya") == "kenya"
    with pytest.raises(ValueError, match="unknown key"):
        boxspec.parse_units_box({"lat_low": 1})


def test_an_invalid_box_is_a_config_error_naming_the_key(world, tmp_path):
    """Check that an invalid box is a config error naming the key."""
    raw = no_region(world, box={"lat_min": 9, "lat_max": 3, "lon_min": 33, "lon_max": 48})
    with pytest.raises(ConfigSchemaError, match=r"'box': .*lat_min 9 is above lat_max 3"):
        resolve(write_config(tmp_path, raw), write=False)


def test_a_box_with_a_region_replaces_the_regions_box(world, tmp_path):
    """Check that a box with a region replaces the regions box."""
    raw = raw_config(world)
    raw["box"] = {"lat_min": 7, "lat_max": 11, "lon_min": 37, "lon_max": 41}
    rc = resolve(write_config(tmp_path, raw), write=False)
    bounds = rc.area.bounds()
    assert bounds.tuple == (7.0, 11.0, 37.0, 41.0) and bounds.origin == "box" and rc.area.region == "ethiopia"
    plain = resolve(write_config(tmp_path, raw_config(world), "plain.yaml"), write=False).area.bounds()
    assert plain.origin == "region box" and plain.tuple != bounds.tuple
    lat, lon = np.arange(0.0, 20.0, 1.0), np.arange(30.0, 50.0, 1.0)
    ds = xr.Dataset({"p": (("lat", "lon"), np.zeros((20, 20)))}, coords={"lat": lat, "lon": lon})
    cut = area_box(ds, rc.area)
    assert (cut.lat.min(), cut.lat.max(), cut.lon.min(), cut.lon.max()) == (7, 11, 37, 41)


@pytest.mark.parametrize("lons, box_lon", [(np.arange(0.0, 360.0, 1.0), (33, 48)), (np.arange(-180.0, 180.0, 1.0), (33, 48)),
                                           (np.arange(0.0, 360.0, 1.0), (-20, -10)), (np.arange(-180.0, 180.0, 1.0), (340, 350))])
def test_a_box_in_either_longitude_convention_cuts_either_kind_of_file(lons, box_lon):
    """Check that a box in either longitude convention cuts either kind of file."""
    spec = area_mod.Bounds(3, 15, box_lon[0], box_lon[1], wrap=True, pad=False, origin="box")
    area = _FakeArea(spec)
    ds = xr.Dataset({"p": (("lat", "lon"), np.zeros((181, lons.size)))},
                    coords={"lat": np.arange(90.0, -91.0, -1.0), "lon": lons})
    cut = area_box(ds, area)
    assert cut.lat.values[0] < cut.lat.values[-1] and cut.lat.min() == 3 and cut.lat.max() == 15
    wrapped = np.mod(cut.lon.values - box_lon[0], 360)
    assert cut.sizes["lon"] == box_lon[1] - box_lon[0] + 1 and wrapped.max() == box_lon[1] - box_lon[0]


def test_a_box_across_the_seam_of_a_file_is_refused_clearly():
    """Check that a box across the seam of a file is refused clearly."""
    area = _FakeArea(area_mod.Bounds(3, 15, 350, 10, wrap=True, pad=False, origin="box"))
    ds = xr.Dataset({"p": (("lat", "lon"), np.zeros((181, 360)))},
                    coords={"lat": np.arange(-90.0, 91.0), "lon": np.arange(0.0, 360.0)})
    with pytest.raises(ValueError, match="crosses the longitude seam"):
        area_box(ds, area)


class _FakeArea:
    """The part of an `AreaSpec` that `area_box` reads: its bounds."""

    region, box, mask = None, None, None

    def __init__(self, bounds):
        """Hold the bounds."""
        self._bounds = bounds

    def bounds(self):
        """Return the bounds."""
        return self._bounds


# --- region null needs an area -------------------------------------------------------------------------------

def test_region_null_without_an_area_is_an_error_with_a_fix(world, tmp_path):
    """Check that region null without an area is an error with a fix."""
    raw = no_region(world)
    with pytest.raises(ConfigSchemaError) as exc:
        resolve(write_config(tmp_path, raw), write=False)
    text = str(exc.value)
    assert "'region' is null and the run has no area" in text
    assert "Fix: set region (see `hcwf-run onset regions`), or mask (a country or a file), or box" in text
    assert "box: all" in text
    land = no_region(world, mask={"land": True})   # land alone has no extent: still no area
    with pytest.raises(ConfigSchemaError, match="the run has no area"):
        resolve(write_config(tmp_path, land, "land.yaml"), write=False)


def test_region_null_with_box_all_or_a_mask_resolves(world, tmp_path):
    """Check that region null with box all or a mask resolves."""
    whole = resolve(write_config(tmp_path, no_region(world, box="all")), write=False)
    assert whole.area.bounds() is None and "whole domain" in whole.area.describe()
    boxed = resolve(write_config(tmp_path, no_region(world, box={"lat_min": 5, "lat_max": 11, "lon_min": 37,
                                                                  "lon_max": 43}), "b.yaml"), write=False)
    assert boxed.area.bounds().tuple == (5.0, 11.0, 37.0, 43.0)
    masked = resolve(write_config(tmp_path, no_region(world, mask={"country": "Ethiopia"}), "m.yaml"), write=False)
    south, north, west, east = masked.area.bounds().tuple
    assert masked.area.bounds().origin == "mask bounds" and 3 < south < 4 and 14 < north < 15.1 and 32 < west < 34 and 47 < east < 49


# --- no region means no region defaults ----------------------------------------------------------------------

def _resolved_error(world, tmp_path, mutate):
    """Return the text of the error a no-region config gives after ``mutate`` edits it."""
    raw = no_region(world, box="all")
    mutate(raw)
    with pytest.raises((ConfigSchemaError, HCWFConfigError)) as exc:
        resolve(write_config(tmp_path, raw), write=False)
    return str(exc.value)


def test_a_run_without_a_region_must_name_its_onset_definition(world, tmp_path):
    """Check that a run without a region must name its onset definition."""
    text = _resolved_error(world, tmp_path, lambda raw: raw["onset"].pop("definition"))
    assert "onset_definition is not set" in text and "Fix: set onset_definition" in text


def test_a_run_without_a_region_must_name_the_season_of_a_definition_with_several(world, tmp_path):
    """Check that a run without a region must name the season of a definition with several."""
    two = tmp_path / "two.yaml"
    two.write_text("id: two\nversion: 1\ntitle: t\nstatus: draft\nextends: icpac\nseasons:\n"
                   "  long_rains: {from: 02-01, to: 05-31}\n  short_rains: {from: 09-01, to: 12-31}\n")
    text = _resolved_error(world, tmp_path, lambda raw: raw["onset"].update(definition=str(two)))
    assert "several seasons (long_rains, short_rains)" in text and "Fix: set onset_season" in text


def test_a_run_without_a_region_must_give_the_climatology_years(world, tmp_path):
    """Check that a run without a region must give the climatology years."""
    def drop(raw):
        del raw["evaluation"]["start_year_clim"], raw["evaluation"]["end_year_clim"]

    text = _resolved_error(world, tmp_path, drop)
    assert text.count("missing required key") == 2
    assert "Fix: set evaluation.start_year_clim" in text and "Fix: set evaluation.end_year_clim" in text


def test_a_run_without_a_region_must_give_the_threshold_years_a_definition_computes(world, tmp_path):
    """Check that a run without a region must give the threshold years a definition computes."""
    text = _resolved_error(world, tmp_path, lambda raw: raw["onset"].update(definition="moron_robertson_blend",
                                                                          season="main"))
    assert "threshold_years" in text and "Fix: set threshold_years" in text
