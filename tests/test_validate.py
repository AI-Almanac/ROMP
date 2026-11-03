"""Validation in one place (step S2 of docs/review/runconfig_design.md): hcwf.config.validate gives every message of the
frozen validators (tests/_oracle_config.py: HCWFValidator, check_config, and the units and blend problems `--check`
reported), and `--check` prints the text it printed before (tests/data/check/*.txt, made by the code before S2)."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent / "blend"))

import _oracle_config as old  # noqa: E402
import differential_onsets as synthetic
from hcwf.config import validate
from hcwf.config.build import resolve
from typed_io import flat_config
from hcwf.errors import HCWFConfigError
from hcwf.runs.check import build_report
from hcwf.config.convert_specs import convert_blend
from hcwf import log as logs
from test_oracle_config import build_world, fixture_configs

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


def old_message(values) -> str | None:
    """The message the frozen HCWFValidator raised for values (None when valid)."""
    try:
        old.HCWFValidator(values).validate()
    except HCWFConfigError as exc:
        return str(exc)
    return None


def new_message(values) -> str | None:
    found = validate.problems(values)
    assert not (found.units or found.blend)
    return found.message() if found else None


FATAL = {
    "start after end": {"start_date": (2019, 8, 1), "end_date": (2019, 5, 1)},
    "start equals end": {"start_date": (2019, 5, 1), "end_date": (2019, 5, 1)},
    "bad date": {"start_date": (2019, 13, 1)},
    "years outside": {"years": (2010, 2019)},
    "climatology years outside": {"years_clim": (1990, 2000)},
    "model list too long": {"model_paths_list": ("a", "b", "c")},
    "tolerances do not match windows": {"tolerance_days_list": (3, 3, 3)},
    "space in model name": {"model_list": ("A B",)},
    "window not ordered": {"verification_window_list": ((5, 5),)},
    "bins not ordered": {"day_bins": ((6, 2),)},
    "window beyond max forecast day": {"max_forecast_day": 3},
    "bins beyond windows": {"probabilistic": True, "day_bins": ((1, 5), (6, 40))},
    "malformed explain cells": {"explain_cells": [(9.0,)]},
    "removed onset key": {"wet_init": 3},
    "removed key": {"layout": 1},
    "unknown onset definition": {"onset_definition": "no_such_definition"},
    "several at once": {"model_list": ("A B",), "day_bins": ((6, 2),), "max_forecast_day": 3, "layout": 1,
                        "explain_cells": "x"},
}


@pytest.mark.parametrize("name", sorted(FATAL))
def test_every_validator_message_is_the_old_one(flat, name):
    values = {**flat, **FATAL[name]}
    expected = old_message(values)
    assert expected is not None, "the oracle accepts this config; the case tests nothing"
    assert new_message(values) == expected


def test_a_valid_config_has_no_problem(flat):
    assert old_message(flat) is None and not validate.problems(flat)


def test_a_missing_key_fails_like_before(flat):
    values = {k: v for k, v in flat.items() if k != "model_list"}
    with pytest.raises(AttributeError):
        old.HCWFValidator(values).validate()
    with pytest.raises(AttributeError):
        validate.problems(values)


def test_a_bad_mask_is_a_problem(flat):
    found = validate.problems({**flat, "mask": {"overlap_fraction": "wide"}})
    assert found.fatal and found.fatal[0].startswith("mask: ")


UNITS = {"mode": {"units_mode": "bogus"}, "boundary": {"units_mode": "regrid", "units_boundary": "bogus"},
         "scope": {"units_mode": "regrid", "units_dissemination_scope": "bogus"},
         "no table": {"units_mode": "regrid"},
         "cells with a file": {"units_mode": "cells", "units_weights": "w.csv"}}


@pytest.mark.parametrize("name", sorted(UNITS))
def test_every_units_message_is_the_old_one(flat, name):
    values = {**flat, **UNITS[name]}
    expected = old.check_units_problem(values)
    assert expected is not None
    found = validate.problems(values)
    assert found.units == (expected,) and not found.fatal and not found.blend


BLEND = {"transform": {"blend_rain_transform": "cube"},
         "external prediction": {"blend_external_predictions": [{"name": "x", "fle": "p.nc"}]},
         "mme weights": {"blend_mme": {"weights": [0.5, 0.5]}}}


@pytest.mark.parametrize("name", sorted(BLEND))
def test_every_blend_message_is_the_old_one(blend_flat, name):
    values = {**blend_flat, **BLEND[name]}
    expected = old.check_blend_problems(values)
    assert expected
    found = validate.problems(values)
    assert list(found.blend) == expected and not found.fatal and not found.units and found.recipe is None


def test_a_valid_blend_keeps_its_recipe(blend_flat):
    found = validate.problems(blend_flat)
    assert not found and found.recipe is not None and old.check_blend_problems(blend_flat) == []


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


def test_check_text_is_the_old_text(world, flat, blend_flat):
    """--check for a valid config, a bad units section, a valid blend and a bad blend prints what it printed before S2."""
    base = world[0]
    _golden("valid", _text(world[1]["det_climatology"]), base)
    _golden("bad_units", _text({**flat, "units_mode": "bogus"}), base)
    _golden("blend_valid", _text(blend_flat), base)
    _golden("bad_blend", _text({**blend_flat, "blend_rain_transform": "cube"}), base)
