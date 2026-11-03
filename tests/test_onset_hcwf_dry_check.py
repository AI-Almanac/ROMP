"""Switching off a definition's dry-spell condition: override, without_dry_spell_check, config and loading."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hcwf.io.forecasts import steps_needed
from hcwf.config.onset import onset_setup
from hcwf.onset import definition as definition_mod
from hcwf.onset.describe import describe
from hcwf.errors import DefinitionError, OnsetConfigError

BOUNDARY = "00 UTC, labelled by start"


def test_override_merges_replaces_and_keeps_provenance():
    """Mappings merge, scalars replace, the hash follows the rule and the YAML records the change."""
    icpac = definition_mod.load("icpac")
    changed = definition_mod.override(icpac, {"wet_day": "at least 2 mm", "onset": {"window": "4 days"}})
    assert changed.wet_day.amount.tenths == 20
    assert changed.rule.require[1].span.last == 4 and icpac.rule.require[1].span.last == 3
    assert changed.rule.reject_if == icpac.rule.reject_if
    assert changed.citation == icpac.citation and changed.regions == icpac.regions
    assert "at least 2 mm" in changed.source_yaml
    assert definition_mod.definition_hash(changed) != definition_mod.definition_hash(icpac)


def test_override_validates_again():
    """An invalid change is a DefinitionError naming the path."""
    with pytest.raises(DefinitionError, match="wet_day"):
        definition_mod.override(definition_mod.load("icpac"), {"wet_day": "20 mm"})


@pytest.mark.parametrize("ident", ["icpac", "moron_robertson"])
def test_without_dry_spell_check_drops_the_vetoes(ident):
    """No vetoes, no dry run or window in the sentence, a different hash, and the change recorded."""
    base = definition_mod.load(ident)
    plain = definition_mod.without_dry_spell_check(base)
    assert plain.rule.reject_if == () and base.rule.reject_if
    sentence = describe(plain)
    assert "dry run" not in sentence and "dry window" not in sentence and "dry spell" not in sentence
    assert definition_mod.definition_hash(plain) != definition_mod.definition_hash(base)
    assert plain.title.endswith("(no dry-spell check)")
    assert plain.notes.endswith("Dry-spell check switched off by the run configuration.")
    assert "reject if: []" in plain.source_yaml


@pytest.mark.parametrize("ident", ["moron_robertson_mok", "liebmann_marengo"])
def test_without_dry_spell_check_is_a_no_op_without_vetoes(ident):
    """Rules without vetoes, and curve rules, come back unchanged."""
    base = definition_mod.load(ident)
    assert definition_mod.without_dry_spell_check(base) is base


def _cfg(**values):
    """A config with a definition."""
    return {"onset_definition": "icpac", "obs_day_boundary": BOUNDARY, **values}


def test_config_switches_the_check_off_and_prints_it():
    """dry_spell_check False gives the variant, and the printed line says off, on or none."""
    off = onset_setup(_cfg(dry_spell_check=False))
    assert off.definition.rule.reject_if == () and off.dry_check == "off"
    assert "Dry-spell check  : off  [config dry_spell_check]" in off.lines()
    assert onset_setup(_cfg()).dry_check == "on"
    assert "Dry-spell check  : on  [default]" in onset_setup(_cfg()).lines()
    none = onset_setup(_cfg(onset_definition="moron_robertson_mok", threshold_years=(2000, 2010), dry_spell_check=False))
    assert none.dry_check == "none in this definition" and none.definition.rule.reject_if == ()


def test_dry_spell_check_must_be_a_bool():
    """A non-bool value is an error naming the key."""
    with pytest.raises(OnsetConfigError, match="dry_spell_check"):
        onset_setup(_cfg(dry_spell_check="no"))


def test_fewer_leads_are_loaded_without_the_check():
    """ICPAC's 21-day dry run needs 21 more leads than the variant without it."""
    kw = dict(max_forecast_day=30)
    assert steps_needed(30, onset_setup(_cfg(**kw))) == 51
    assert steps_needed(30, onset_setup(_cfg(dry_spell_check=False, **kw))) == 32


def test_generate_config_env_var(tmp_path):
    """HCWF_DRY_SPELL_CHECK is a bool valid with and without a definition."""
    script = Path(__file__).parents[1] / "scripts" / "generate_config.py"
    base = {k: v for k, v in os.environ.items() if not k.startswith("HCWF_")}
    base.update(HCWF_OBS_DIR="o", HCWF_MODEL_DIR="m", HCWF_MODEL_NAME="x", HCWF_DIR_OUT="d", HCWF_DIR_FIG="f",
                HCWF_CONFIG_PATH=str(tmp_path / "c.yaml"))
    for extra, expected in (({}, True), ({"HCWF_DRY_SPELL_CHECK": "False"}, False),
                            ({"HCWF_DRY_SPELL_CHECK": "False", "HCWF_ONSET_DEFINITION": "icpac"}, False)):
        assert subprocess.run([sys.executable, str(script)], env={**base, **extra}).returncode == 0
        import yaml

        ns = yaml.safe_load((tmp_path / "c.yaml").read_text())
        assert ns["onset"]["dry_spell_check"] is expected
    bad = subprocess.run([sys.executable, str(script)], env={**base, "HCWF_DRY_SPELL_CHECK": "maybe"},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "HCWF_DRY_SPELL_CHECK" in bad.stderr
