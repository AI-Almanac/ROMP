"""Validation in one place (step S2 of docs/review/runconfig_design.md): hcwf.config.validate gives the message of every
kind of invalid config (the texts below are the ones users have seen), and `--check` prints the text in
tests/data/check/*.txt."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent / "blend"))

import differential_onsets as synthetic
from hcwf.config import validate
from hcwf.config.build import resolve
from typed_io import flat_config
from hcwf.errors import HCWFConfigError
from hcwf.runs.check import build_report
from hcwf.config.convert_specs import convert_blend
from hcwf import log as logs
from config_world import build_world, fixture_configs

GOLDEN = Path(__file__).parent / "data" / "check"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = build_world(tmp_path_factory.mktemp("validate_world"))
    return base, fixture_configs(base, base / "cfgs")


@pytest.fixture(scope="module")
def flat(world):
    return flat_config(world[1]["det_climatology"])


@pytest.fixture(scope="module")
def blend_flat(world):
    base = world[0]
    data = base / "bdata"
    data.mkdir()
    synthetic.write_data(data)
    (base / "b.yaml").write_text(convert_blend(synthetic.configs(data)["plain"]).to_yaml())
    return flat_config(base / "b.yaml")


def new_message(values) -> str | None:
    found = validate.problems(values)
    assert not (found.units or found.blend)
    return found.message() if found else None


FATAL = {   # name: (the values that make the config invalid, the message)
    'start after end': ({'start_date': (2019, 8, 1), 'end_date': (2019, 5, 1)},
        'HCWF Configuration is invalid:\n  • start_date (2019-08-01) is after end_date (2019-05-01)\n  • start_year_clim (2006) is after end_year_clim (2020)'),
    'start equals end': ({'start_date': (2019, 5, 1), 'end_date': (2019, 5, 1)},
        'HCWF Configuration is invalid:\n  • start_date (2019-05-01) is after end_date (2019-05-01)'),
    'bad date': ({'start_date': (2019, 13, 1)},
        'HCWF Configuration is invalid:\n  • Date formatting error: month must be in 1..12'),
    'years outside': ({'years': (2010, 2019)},
        'HCWF Configuration is invalid:\n  • evaluation.years [2010] lie outside 2019-2020, the range of start_date and end_date. Fix: list only years in that range, or widen the range'),
    'climatology years outside': ({'years_clim': (1990, 2000)},
        'HCWF Configuration is invalid:\n  • evaluation.climatology_years [1990, 2000] lie outside 2006-2020, the range of start_year_clim and end_year_clim. Fix: list only years in that range, or widen the range'),
    'model list too long': ({'model_paths_list': ('a', 'b', 'c')},
        "HCWF Configuration is invalid:\n  • Length Mismatch: 'model_paths_list' has 3 items,                                         expected 1"),
    'tolerances do not match windows': ({'tolerance_days_list': (3, 3, 3)},
        "HCWF Configuration is invalid:\n  • Length Mismatch: 'tolerance_days_list' has 3 items,                                         expected 1"),
    'space in model name': ({'model_list': ('A B',)},
        "HCWF Configuration is invalid:\n  • Invalid model name: 'A B' (Spaces are not allowed)."),
    'window not ordered': ({'verification_window_list': ((5, 5),)},
        'HCWF Configuration is invalid:\n  • Invalid range in verification_window_list: (5, 5).                             First number must be smaller than the second.'),
    'bins not ordered': ({'day_bins': ((6, 2),)},
        'HCWF Configuration is invalid:\n  • Invalid range in day_bins: (6, 2).                             First number must be smaller than the second.'),
    'window beyond max forecast day': ({'max_forecast_day': 3},
        'HCWF Configuration is invalid:\n  • Verification window limit (10)                     exceeds max_forecast_day (3).'),
    'bins beyond windows': ({'probabilistic': True, 'day_bins': ((1, 5), (6, 40))},
        'HCWF Configuration is invalid:\n  • Window mismatch: Largest verification day                         (10) is smaller than largest day_bin (40).'),
    'malformed explain cells': ({'explain_cells': [(9.0,)]},
        'HCWF Configuration is invalid:\n  • explain_cells must be None or a list of (lat, lon) pairs of numbers, e.g. [(9.0, 38.75), (8.5, 39.0)]'),
    'removed onset key': ({'wet_init': 3},
        "HCWF Configuration is invalid:\n  • unknown key 'wet_init'. The old flat onset keys (wet_init, wet_threshold, wet_spell, dry_threshold, dry_spell, dry_extent, thresh_file, thresh_var, earliest_onset_date, mok) are no longer read. Fix: write the rule as an onset definition; `hcwf-run convert old.yaml -o new.yaml` translates a config that still has them"),
    'removed key': ({'layout': 1},
        "HCWF Configuration is invalid:\n  • unknown key 'layout'. This key was removed: it had no effect. Fix: delete it"),
    'unknown onset definition': ({'onset_definition': 'no_such_definition'},
        "HCWF Configuration is invalid:\n  • onset_definition = 'no_such_definition': 'no_such_definition' is neither a registry id (dunning, icpac, icpac_blend, liebmann_marengo, marteau, moron_robertson, moron_robertson_blend, moron_robertson_mok, omotosho, sivakumar) nor a file. Fix: use a registry id or the path of a .yaml file"),
    'several at once': ({'model_list': ('A B',), 'day_bins': ((6, 2),), 'max_forecast_day': 3, 'layout': 1, 'explain_cells': 'x'},
        "HCWF Configuration is invalid:\n  • Invalid model name: 'A B' (Spaces are not allowed).\n  • Invalid range in day_bins: (6, 2).                             First number must be smaller than the second.\n  • Verification window limit (10)                     exceeds max_forecast_day (3).\n  • unknown key 'layout'. This key was removed: it had no effect. Fix: delete it\n  • explain_cells must be None or a list of (lat, lon) pairs of numbers, e.g. [(9.0, 38.75), (8.5, 39.0)]"),
}


@pytest.mark.parametrize("name", sorted(FATAL))
def test_every_validator_message_is_the_expected_one(flat, name):
    changes, expected = FATAL[name]
    assert new_message({**flat, **changes}) == expected


def test_a_valid_config_has_no_problem(flat):
    assert not validate.problems(flat)


def test_a_missing_key_fails_with_an_attribute_error(flat):
    values = {k: v for k, v in flat.items() if k != "model_list"}
    with pytest.raises(AttributeError):
        validate.problems(values)


def test_a_bad_mask_is_a_problem(flat):
    found = validate.problems({**flat, "mask": {"overlap_fraction": "wide"}})
    assert found.fatal and found.fatal[0].startswith("mask: ")


UNITS = {   # name: (the values that make the units section invalid, the message)
    'mode': ({'units_mode': 'bogus'},
        "units: units mode must be one of ('cells', 'regrid', 'cell_transform'), got 'bogus'"),
    'boundary': ({'units_mode': 'regrid', 'units_boundary': 'bogus'},
        "units: units boundary must be one of ('contains', 'intersects'), got 'bogus'"),
    'scope': ({'units_mode': 'regrid', 'units_dissemination_scope': 'bogus'},
        "units: units dissemination_scope must be one of ('inputs', 'training'), got 'bogus'"),
    'no table': ({'units_mode': 'regrid'},
        "units: units mode 'regrid' needs a shapefile or a weights table"),
    'cells with a file': ({'units_mode': 'cells', 'units_weights': 'w.csv'},
        "units: units mode 'cells' reads the file's own cells; it takes no shapefile or weights"),
}


@pytest.mark.parametrize("name", sorted(UNITS))
def test_every_units_message_is_the_expected_one(flat, name):
    changes, expected = UNITS[name]
    found = validate.problems({**flat, **changes})
    assert found.units == (expected,) and not found.fatal and not found.blend


BLEND = {   # name: (the values that make the blend section invalid, the messages)
    'transform': ({'blend_rain_transform': 'cube'},
        ["blend: features.rain_transform: 'cube' is not identity, sqrt, fourth_root, log1p or power:<p>"]),
    'external prediction': ({'blend_external_predictions': [{'name': 'x', 'fle': 'p.nc'}]},
        ['blend: blend.external_predictions[0].fle: unknown key. Did you mean "file"? Keys: name, file, method', 'blend: blend.external_predictions[0]: name and file are required']),
    'mme weights': ({'blend_mme': {'weights': [0.5, 0.5]}},
        ['blend: blend.mme.weights: expected a mapping of source name to weight or the path of a weights file, got [0.5, 0.5]']),
}


@pytest.mark.parametrize("name", sorted(BLEND))
def test_every_blend_message_is_the_expected_one(blend_flat, name):
    changes, expected = BLEND[name]
    found = validate.problems({**blend_flat, **changes})
    assert list(found.blend) == expected and not found.fatal and not found.units and found.recipe is None


def test_a_valid_blend_keeps_its_recipe(blend_flat):
    found = validate.problems(blend_flat)
    assert not found and found.recipe is not None


def test_a_units_error_fails_at_resolve_before_a_run_folder(world, flat):
    with pytest.raises(HCWFConfigError, match="units mode must be one of"):
        flat_config({**flat, "units_mode": "bogus"})
    dic, found = flat_config({**flat, "units_mode": "bogus"}, check=True)
    assert dic is not None and found.units
    dic, found = flat_config({**flat, "day_bins": ((6, 2),)}, check=True)
    assert dic is None and found.fatal


def _text(values) -> str:
    rc, found = resolve(values, check=True)
    with logs.session(logs.QUIET):
        return build_report(rc, found).text()


def _golden(name: str, text: str, base: Path) -> None:
    import tempfile

    def plain(value: str) -> str:   # the run folder's fingerprint depends on the paths; (cached) on the order of reading
        return re.sub(r"obs-[0-9a-f]{12}", "obs-<fp>", re.sub(r"_[0-9a-f]{4}(?=[/;\s]|$)", "_<fp>", value.replace(" (cached)", "")))

    text = text.replace(str(base), "<base>").replace(str(Path(tempfile.gettempdir())), "<tmp>") + "\n"
    assert plain(text) == plain((GOLDEN / f"{name}.txt").read_text())


def test_check_text_is_the_golden_text(world, flat, blend_flat):
    """--check for a valid config, a bad units section, a valid blend and a bad blend prints the golden text."""
    base = world[0]
    _golden("valid", _text(world[1]["det_climatology"]), base)
    _golden("bad_units", _text({**flat, "units_mode": "bogus"}), base)
    _golden("blend_valid", _text(blend_flat), base)
    _golden("bad_blend", _text({**blend_flat, "blend_rain_transform": "cube"}), base)
