"""Switching off a definition's dry-spell condition: override, without_dry_spell_check, config and loading."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from momp.io.input import onset_steps_needed
from momp.lib.onset_config import OnsetConfigError, onset_setup
from momp.onset import definition as definition_mod
from momp.onset.describe import describe
from momp.onset.legacy import from_flat_config

BOUNDARY = "00 UTC, labelled by start"
FLAT = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_spell=7, dry_extent=10, end_date=(2019, 7, 31))


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
    with pytest.raises(definition_mod.DefinitionError, match="wet_day"):
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
    """A definition-mode config."""
    return {"onset_legacy": False, "onset_definition": "icpac", "obs_day_boundary": BOUNDARY, **values}


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


def test_legacy_mode_drops_the_translated_veto():
    """In legacy mode False has the effect of dry_extent = 0."""
    cfg = {**FLAT, "region": "atlantis"}
    assert len(onset_setup(cfg).definition.rule.reject_if) == 1
    off = onset_setup({**cfg, "dry_spell_check": False})
    assert off.dry_check == "off"
    assert off.definition.rule == from_flat_config({**FLAT, "dry_extent": 0}).rule


def test_fewer_leads_are_loaded_without_the_check():
    """ICPAC's 21-day dry run needs 21 more leads than the variant without it."""
    kw = dict(max_forecast_day=30, wet_spell=3, dry_extent=0)
    assert onset_steps_needed(**_cfg(**kw)) == 51
    assert onset_steps_needed(**_cfg(dry_spell_check=False, **kw)) == 32


def test_generate_config_env_var(tmp_path):
    """ROMP_DRY_SPELL_CHECK is a bool valid with and without a definition."""
    script = Path(__file__).parents[1] / "scripts" / "generate_config.py"
    base = {k: v for k, v in os.environ.items() if not k.startswith("ROMP_")}
    base.update(ROMP_OBS_DIR="o", ROMP_MODEL_DIR="m", ROMP_MODEL_NAME="x", ROMP_DIR_OUT="d", ROMP_DIR_FIG="f",
                ROMP_CONFIG_PATH=str(tmp_path / "c.yaml"))
    for extra, expected in (({}, True), ({"ROMP_DRY_SPELL_CHECK": "False"}, False),
                            ({"ROMP_DRY_SPELL_CHECK": "False", "ROMP_ONSET_DEFINITION": "icpac"}, False)):
        assert subprocess.run([sys.executable, str(script)], env={**base, **extra}).returncode == 0
        import yaml

        ns = yaml.safe_load((tmp_path / "c.yaml").read_text())
        assert ns["onset"]["dry_spell_check"] is expected
    bad = subprocess.run([sys.executable, str(script)], env={**base, "ROMP_DRY_SPELL_CHECK": "maybe"},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "ROMP_DRY_SPELL_CHECK" in bad.stderr
