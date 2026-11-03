"""`hcwf-run convert` turns a run YAML's old flat onset keys (wet_init, wet_spell, ...) into an onset definition that
reproduces the old detection; a run config that still has them is refused with a pointer to the converter."""

import datetime

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


@pytest.mark.parametrize("key", ["layout", "work_dir", "pkg_dir", "debug", "fallback_date"])
def test_removed_keys_are_named_as_removed(key):
    """A key that had no effect and was removed gets a message saying so, not a spelling suggestion."""
    from hcwf.lib.config_yaml import build_config

    with pytest.raises(ConfigSchemaError, match=f"unknown key '{key}'. This key was removed"):
        build_config({key: None})


def test_keys_that_were_removed_are_dropped_from_the_converted_config(tmp_path):
    source = tmp_path / "old.yaml"
    source.write_text(yaml.safe_dump({
        "region": "india", "work_dir": "/x", "layout": ["model"], "debug": False,
        "wet_init": 1, "wet_threshold": 20, "wet_spell": 5, "dry_threshold": 1, "dry_spell": 7, "dry_extent": 0,
        "evaluation": {"start_date": [2019, 5, 1], "end_date": [2020, 7, 31]}}))
    target = convert_flat_onset(source, tmp_path / "new.yaml")
    text = target.read_text()
    converted = yaml.safe_load(text)
    assert not {"work_dir", "layout", "debug"} & set(converted)
    assert "Dropped keys that have no effect any more: " in text and "onset" in converted


def test_convert_writes_what_the_old_run_always_used(tmp_path):
    """B-1: the old flat-key run took forecasts as given and labelled leads by end; the converted config says so."""
    new = yaml.safe_load(convert_flat_onset(_old_run_yaml(tmp_path), tmp_path / "new.yaml").read_text())
    assert new["onset"]["forecast_history"] == "none" and new["onset"]["lead_label"] == "end"


def test_convert_keeps_the_history_and_label_the_config_sets(tmp_path):
    """A config that already chose its history and label keeps them (nested, or under the flat key)."""
    source = _old_run_yaml(tmp_path, onset={"forecast_history": "observed", "lead_label": "start"})
    new = yaml.safe_load(convert_flat_onset(source, tmp_path / "a.yaml").read_text())
    assert new["onset"]["forecast_history"] == "observed" and new["onset"]["lead_label"] == "start"
    other = tmp_path / "other"
    other.mkdir()
    new = yaml.safe_load(convert_flat_onset(_old_run_yaml(other, model_lead_label="start"), other / "b.yaml").read_text())
    assert "lead_label" not in new["onset"] and new["model_lead_label"] == "start"


@pytest.mark.parametrize("end_date", ["2019-07-31", [2019, 7, 31], datetime.date(2019, 7, 31)])
@pytest.mark.parametrize("earliest", ["06-02", [6, 2]])
def test_every_date_form_the_old_schema_accepted_converts(end_date, earliest):
    """B-2: 'YYYY-MM-DD', YAML dates and 'MM-DD' text go through the schema's kinds, as the old loader did."""
    defn = _defn({**BASE, "end_date": end_date, "start_date": "2019-05-01", "earliest_onset_date": earliest})
    assert defn.seasons == _main(start=(6, 2))


def test_a_date_that_is_not_one_is_a_config_error():
    with pytest.raises(ConfigSchemaError, match="end_date"):
        flat_onset_to_rule({**BASE, "end_date": "next week"})


def test_a_threshold_file_needs_its_variable():
    """R-5c: without thresh_var the path would be read as a table of units, not a grid."""
    with pytest.raises(ConfigSchemaError, match="thresh_var"):
        flat_onset_to_rule({**BASE, "thresh_file": "t.nc", "thresh_var": None})


@pytest.mark.parametrize("end_date", [(2019, 1, 15), (2019, 2, 1), (2019, 2, 28)])
def test_an_end_that_passes_29_february_is_refused(end_date):
    """R-5a: the old end was counted in days, a fixed month-day is one day off in leap years."""
    with pytest.raises(ConfigSchemaError, match="29 February"):
        flat_onset_to_rule({**BASE, "start_date": (2019, 1, 1), "end_date": end_date})


def test_an_end_before_15_january_and_after_29_february_is_kept():
    """No 29 February between the end date and the end of the search: the month-day is exact in every year."""
    assert _defn({**BASE, "start_date": (2019, 1, 1), "end_date": (2019, 1, 14)}).seasons[0].end == DayRef(month=2, day=28)
    assert _defn({**BASE, "start_date": (2019, 3, 1), "end_date": (2019, 3, 31)}).seasons[0].end == DayRef(month=5, day=15)


def test_a_season_that_ends_in_the_next_year_wraps():
    """R-5b: an end in the second half of November ends in January or February; that is a wrapping season."""
    defn = _defn({**BASE, "start_date": (2019, 10, 1), "end_date": (2019, 11, 20)})
    assert defn.seasons[0].start == DayRef(month=10, day=1) and defn.seasons[0].end == DayRef(month=1, day=4)


def test_a_season_longer_than_a_year_is_refused_with_the_reason():
    with pytest.raises(ConfigSchemaError, match="longer than a year"):
        flat_onset_to_rule({**BASE, "start_date": (2019, 1, 1), "end_date": (2019, 12, 1)})


def test_years_move_to_evaluation_and_all_is_dropped(tmp_path):
    """B-4: old years / years_clim become evaluation.years / climatology_years; 'All' and null were the default."""
    new = yaml.safe_load(convert_flat_onset(_old_run_yaml(tmp_path, years=[2019], years_clim=[2016, 2018]),
                                            tmp_path / "new.yaml").read_text())
    assert new["evaluation"]["years"] == [2019] and new["evaluation"]["climatology_years"] == [2016, 2018]
    assert "years" not in new and "years_clim" not in new
    other = tmp_path / "other"
    other.mkdir()
    new = yaml.safe_load(convert_flat_onset(_old_run_yaml(other, years="All", years_clim=None),
                                            other / "new.yaml").read_text())
    assert "years" not in new["evaluation"] and "climatology_years" not in new["evaluation"]


def test_the_header_names_the_path_given_and_region_is_refused_for_run_yaml(tmp_path, capsys):
    source = _old_run_yaml(tmp_path)
    out = tmp_path / "new.yaml"
    convert_flat_onset(source, out)
    assert f"hcwf-run -p {out} --check" in out.read_text()
    with pytest.raises(SystemExit):
        cli_main(["convert", str(source), "-o", str(tmp_path / "x.yaml"), "--region", "kenya"])
