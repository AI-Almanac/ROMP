"""The onset definition lives in the run file: registry id, override, or a complete inline rule."""

import pytest
import yaml

from momp.lib.onset_config import OnsetConfigError, onset_setup
from momp.onset import definition as definition_mod


def _cfg(**values):
    """A definition-mode config for Ethiopia."""
    return {"onset_legacy": False, "region": "ethiopia", "obs_day_boundary": None, **values}


def test_override_changes_the_rule_and_is_listed():
    """An override is merged with the same rules as extends; the changed keys are listed and the hash changes."""
    base = onset_setup(_cfg(onset_definition="icpac"))
    over = onset_setup(_cfg(onset_definition="icpac", onset_override={"wet_day": "at least 2 mm"}))
    assert over.overrides == ("wet_day",)
    assert definition_mod.definition_hash(over.definition) != definition_mod.definition_hash(base.definition)
    assert "overridden: wet_day" in "\n".join(over.lines())


def test_override_errors_are_strict():
    """A bad override raises the definition's own error, naming the path."""
    with pytest.raises(OnsetConfigError, match="nonsense"):
        onset_setup(_cfg(onset_definition="icpac", onset_override={"nonsense": 1}))


def test_a_complete_inline_rule_is_a_definition():
    """onset.rule holds a whole definition; it cannot be combined with onset.definition."""
    rule = yaml.safe_load(definition_mod.load("icpac").source_yaml)
    rule["id"] = "my_icpac"
    setup = onset_setup(_cfg(onset_rule=rule))
    assert setup.definition.id == "my_icpac"
    assert "inline" in setup.sources["definition"]
    with pytest.raises(OnsetConfigError, match="cannot be combined"):
        onset_setup(_cfg(onset_rule=rule, onset_definition="icpac"))


def test_dry_spell_check_is_an_override():
    """dry_spell_check = false appears in the override list."""
    setup = onset_setup(_cfg(onset_definition="icpac", dry_spell_check=False))
    assert any("dry_spell_check" in item for item in setup.overrides)
