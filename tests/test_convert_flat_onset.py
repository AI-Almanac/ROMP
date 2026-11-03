"""`hcwf-run convert` turns a run YAML's old flat onset keys (wet_init, wet_spell, ...) into an onset definition that
reproduces the old detection; a run config that still has them is refused with a pointer to the converter."""

import pytest
import yaml

from hcwf.cli import main as cli_main
from hcwf.lib.config_schema import ConfigSchemaError
from hcwf.lib.convert import convert_flat_onset, flat_onset_to_rule
from hcwf.onset import definition as definition_mod
from hcwf.onset.model import Amount, Comparison, DayRef, DryRun, FirstPassage, Season, Span, Total, Veto, WetDays

BASE = dict(wet_init=1, wet_threshold=20, wet_spell=3, dry_threshold=1, dry_spell=7, dry_extent=0,
            thresh_file=None, thresh_var=None, earliest_onset_date=None,
            start_date=(2019, 5, 1), end_date=(2019, 7, 31))


def _defn(flat):
    """The Definition the converter writes for these flat keys."""
    rule, _ = flat_onset_to_rule(flat)
    return definition_mod.loads(yaml.safe_dump(rule, sort_keys=False))


def _rule(window=3, amount=Amount(tenths=200), veto=()):
    """The expected FirstPassage: every window day wet, total above the amount, optional vetoes."""
    return FirstPassage(
        require=(WetDays(Span(1, window), "ge", window), Total(Span(1, window), Comparison("gt", amount))),
        reject_if=veto)


def _main(start=(5, 1), end=(9, 14), wet_spell=3):
    """The expected single season: the start day to the last day whose wet spell fit by end_date + 47 days."""
    assert wet_spell == 3 and end == (9, 14)    # 07-31 + 45 days
    return (Season("main", DayRef(month=start[0], day=start[1]), DayRef(month=end[0], day=end[1])),)


def test_plain_config():
    """No dry extent: wet days and a total above wet_threshold, searched from start_date to end_date + 45 days."""
    defn = _defn(BASE)
    assert defn.wet_day == Comparison("ge", Amount(tenths=10))
    assert defn.seasons == _main()
    assert defn.rule == _rule()


def test_dry_extent_beyond_the_window_adds_the_observed_veto_window():
    """dry_extent above wet_spell and at least dry_spell vetoes a dry run in days 2 to dry_extent + 1."""
    defn = _defn({**BASE, "dry_extent": 10})
    assert defn.rule == _rule(veto=(Veto(DryRun(7), Span(2, 11), "inside"),))


@pytest.mark.parametrize(("extent", "spell"), [(3, 7), (0, 7), (5, 7)])
def test_no_veto_when_the_extent_is_short(extent, spell):
    """A dry extent within the wet window, or shorter than the dry spell, gives no veto."""
    assert _defn({**BASE, "dry_extent": extent, "dry_spell": spell}).rule == _rule()


def test_earliest_onset_date_and_threshold_file():
    """earliest_onset_date replaces the season start; a threshold file makes the total compare with a named threshold."""
    rule, thresholds = flat_onset_to_rule({**BASE, "earliest_onset_date": (6, 2), "thresh_file": "t.nc",
                                           "thresh_var": "mwset", "wet_spell": 5, "wet_init": 0.5})
    defn = definition_mod.loads(yaml.safe_dump(rule, sort_keys=False))
    assert thresholds == {"legacy_threshold": "t.nc:mwset"}
    assert defn.wet_day == Comparison("ge", Amount(tenths=5))
    assert defn.seasons[0].start == DayRef(month=6, day=2)
    assert defn.rule == _rule(window=5, amount=Amount(threshold="legacy_threshold"))


def test_mok_is_the_old_name_of_earliest_onset_date():
    """The India-specific key mok means earliest_onset_date; both together are an error."""
    assert _defn({**BASE, "mok": (6, 2)}).seasons[0].start == DayRef(month=6, day=2)
    with pytest.raises(ConfigSchemaError, match="mok"):
        flat_onset_to_rule({**BASE, "mok": (6, 2), "earliest_onset_date": (6, 2)})


def test_dry_threshold_is_ignored():
    """dry_threshold never mattered to the old code, so it does not change the definition."""
    assert flat_onset_to_rule({**BASE, "dry_threshold": 5}) == flat_onset_to_rule({**BASE, "dry_threshold": 1})


@pytest.mark.parametrize("key", ["wet_init", "wet_threshold"])
def test_amounts_must_be_whole_tenths(key):
    """Amounts finer than 0.1 mm are an error naming the key."""
    with pytest.raises(ConfigSchemaError, match=key):
        flat_onset_to_rule({**BASE, key: 0.25})


def test_missing_keys_are_named():
    """A flat config without wet_spell says so."""
    with pytest.raises(ConfigSchemaError, match="wet_spell"):
        flat_onset_to_rule({k: v for k, v in BASE.items() if k != "wet_spell"})


def _old_run_yaml(tmp_path, **extra):
    """A run YAML as earlier versions read it: the flat onset keys at the top level."""
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({
        "region": "ethiopia", "wet_init": 1, "wet_threshold": 20, "wet_spell": 3, "dry_threshold": 1, "dry_spell": 7,
        "dry_extent": 10, "thresh_file": None, "earliest_onset_date": [6, 2],
        "evaluation": {"start_date": [2019, 5, 1], "end_date": [2019, 7, 31]}, **extra}))
    return path


def test_convert_writes_an_onset_rule_in_place_of_the_flat_keys(tmp_path):
    """The converted YAML has no flat keys; onset.rule holds the definition and everything else is kept."""
    out = convert_flat_onset(_old_run_yaml(tmp_path), tmp_path / "new.yaml")
    new = yaml.safe_load(out.read_text())
    assert not set(new) & {"wet_init", "wet_threshold", "wet_spell", "dry_spell", "dry_extent", "earliest_onset_date"}
    assert new["region"] == "ethiopia" and new["evaluation"]["end_date"] == [2019, 7, 31]
    defn = definition_mod.loads(yaml.safe_dump(new["onset"]["rule"], sort_keys=False))
    assert defn.seasons[0].start == DayRef(month=6, day=2)
    assert defn.rule == _rule(veto=(Veto(DryRun(7), Span(2, 11), "inside"),))
    with pytest.raises(FileExistsError):
        convert_flat_onset(_old_run_yaml(tmp_path), out)


def test_convert_refuses_a_definition_beside_the_flat_keys(tmp_path):
    """Flat keys together with onset.definition cannot be translated."""
    with pytest.raises(ConfigSchemaError, match="onset.definition"):
        convert_flat_onset(_old_run_yaml(tmp_path, onset={"definition": "icpac"}), tmp_path / "new.yaml")


def test_the_command_converts_and_a_run_config_with_the_keys_is_refused(tmp_path, capsys):
    """`hcwf-run convert old.yaml` writes the file; loading the old YAML as a run config names the converter."""
    source = _old_run_yaml(tmp_path)
    assert cli_main(["convert", str(source), "-o", str(tmp_path / "new.yaml")]) in (0, None)
    assert "onset" in yaml.safe_load((tmp_path / "new.yaml").read_text())
    from hcwf.lib.config_yaml import build_config

    with pytest.raises(ConfigSchemaError, match="hcwf-run convert"):
        build_config(yaml.safe_load(source.read_text()))


@pytest.mark.parametrize("key", ["layout", "work_dir", "pkg_dir", "debug", "years", "years_clim", "fallback_date"])
def test_removed_keys_are_named_as_removed(key):
    """A key that had no effect and was removed gets a message saying so, not a spelling suggestion."""
    from hcwf.lib.config_yaml import build_config

    with pytest.raises(ConfigSchemaError, match=f"unknown key '{key}'. This key was removed"):
        build_config({key: None})


def test_keys_that_were_removed_are_dropped_from_the_converted_config(tmp_path):
    source = tmp_path / "old.yaml"
    source.write_text(yaml.safe_dump({
        "region": "india", "work_dir": "/x", "layout": ["model"], "debug": False, "years": None,
        "wet_init": 1, "wet_threshold": 20, "wet_spell": 5, "dry_threshold": 1, "dry_spell": 7, "dry_extent": 0,
        "evaluation": {"start_date": [2019, 5, 1], "end_date": [2020, 10, 31]}}))
    target = convert_flat_onset(source, tmp_path / "new.yaml")
    text = target.read_text()
    converted = yaml.safe_load(text)
    assert not {"work_dir", "layout", "debug", "years"} & set(converted)
    assert "Dropped keys that have no effect any more: " in text and "onset" in converted
