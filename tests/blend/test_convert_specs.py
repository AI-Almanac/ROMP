"""`hcwf-run convert`: onset_blending configs and legacy spec sets become HCWF run YAML (hcwf.config.convert_specs).

The conversions are checked against what blending itself does with the same settings (the file lists, rain-window
numbers) and against onset_blending's shipped configs and archived spec sets."""
import fnmatch
import glob
import re
from pathlib import Path

import pytest
import yaml

from _recipes import ORACLE_SRC, blending_dict, converted, partial, with_stub_data
from hcwf.config import convert_specs as C
from hcwf.config.datasets import file_year, resolve_paths
from hcwf.errors import ConvertError

CONFIGS = sorted(glob.glob(str(ORACLE_SRC / "configs" / "*.yml")))
SPECS = ORACLE_SRC / "archive" / "specs"
needs_oracle = pytest.mark.skipif(not CONFIGS, reason="onset_blending's configs are not available")

NAMES = ["rain_2001.nc", "rain_20x1.nc", "xrain_2001.nc", "rain_2001.nc4", "2001_adm3.nc", "2026_a_adm3.nc",
         "2026a_adm3.nc", "2026_adm3.nc", "data_1999.nc", "data_199.nc", "abc.nc", "a_c.nc", "abcd.nc", "adm3.txt", "x.nc"]


@pytest.mark.parametrize("pattern", [r"\.nc$", r"^rain_(19|20)\d{2}\.nc$", r"^[0-9]{4}_adm3\.nc$", r"^2026.*_adm3\.nc$",
                                     r"^2026_.*_adm3\.nc$", r"adm3", r"^data_\d{4}\.nc$", r"^a.c\.nc$", r"rain_"])
def test_a_glob_matches_the_names_its_regular_expression_matches(pattern):
    glob_text = C.regex_to_glob(pattern)
    assert glob_text is not None
    for name in NAMES:
        assert fnmatch.fnmatchcase(name, glob_text) == bool(re.search(pattern, name)), (pattern, glob_text, name)


@pytest.mark.parametrize("pattern", [r"^(a|b)\.nc$", r"^\d+\.nc$", r"^x(yz)?\.nc$", r"^[ab]\.nc$"])
def test_a_regular_expression_a_glob_cannot_say_is_not_translated(pattern):
    assert C.regex_to_glob(pattern) is None


def test_the_files_of_a_folder_are_those_blending_lists_in_its_order(tmp_path):
    for name in ("2001.nc", "2003.nc", "x_2002.nc", "2002.txt", "noyear.nc", "b.nc2", "2004.nc"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.nc").mkdir()

    def blending_list(pattern, years=()):   # onset_blending.core.io.list_files
        found = [p for p in tmp_path.iterdir() if p.is_file() and re.search(pattern, p.name)]
        if years:
            found = [p for p in found if file_year(p) in set(years)]
        return sorted(found, key=lambda p: (file_year(p) or 0, str(p)))

    for pattern, years in ((r"\.nc$", ()), (r"\.nc$", [2001, 2003]), (r"^[0-9]{4}\.nc$", ()), (r"^(2001|2004)\.nc$", ())):
        data = {"files": {"folder": str(tmp_path), "pattern": pattern, **({"years": years} if years else {})}}
        conversion = C._Convert({}, None, None).files(data["files"], "observations")
        got = resolve_paths(conversion["paths"], None, conversion.get("years"))
        assert got == blending_list(pattern, years), pattern


def test_listed_files_keep_blendings_years_filter(tmp_path):
    for year in (2001, 2002, 2003):
        (tmp_path / f"f{year}.nc").write_bytes(b"")
    entry = C._Convert({}, tmp_path, None).files({"paths": ["f2001.nc", "f2002.nc", "f2003.nc"], "years": "2002:2003"}, "m")
    assert [p.name for p in resolve_paths(entry["paths"], None, entry["years"])] == ["f2002.nc", "f2003.nc"]


def minimal(tmp_path, **changes):
    """A small BlendConfig mapping on files under tmp_path (not read by the converter)."""
    data = {"name": "mini", "data_root": str(tmp_path),
            "observations": {"files": {"folder": "obs"}, "variable": "rain"},
            "models": [{"name": "m", "files": {"folder": "m"}, "unit_scale": 1000.0,
                        "rain_predictors": [{"agg": "diff"}, {"agg": "max", "window": "follow"}]}],
            "onset": {"threshold": {"mm": 20}, "definition": {"follow_days": 30, "followup_anchor": "onset_day"}},
            "evaluation": {"years": {"training": "2001:2003", "cv_holdout": 2003}, "formulas": {"b": "outcome ~ prob_clim_qx"}}}
    for key, value in changes.items():
        data[key] = value
    return data


def test_the_parity_options_and_sources_are_written_out(tmp_path):
    run = C.convert_blend(minimal(tmp_path)).mapping
    onset = run["onset"]
    assert {k: onset[k] for k in ("rain_resolution", "missing", "negatives", "gaps", "horizon", "lead_label",
                                  "forecast_history")} == dict(
        rain_resolution="exact", missing="keep", negatives="keep", gaps="refuse", horizon="truncate", lead_label="end",
        forecast_history="none")
    assert onset["thresholds"] == {"thr": 20.0} and "threshold_max_missing" not in onset
    assert run["region"] is None and run["box"] == "all" and run["observations"]["unit_cvt"] == 1.0
    assert "box" not in C.convert_blend(minimal(tmp_path), region="India").mapping   # a region brings its own box
    assert run["observations"]["day_boundary"] == "00 UTC, labelled by end"
    [model] = run["models"]
    assert (model["unit_cvt"], model["accumulation"]) == (1000.0, "daily") and model["paths"][0].endswith("m/*.nc")
    assert "years" not in run and run["blend"]["years"] == {"training": "2001:2003", "cv_holdout": 2003}


def test_a_computed_threshold_counts_partial_years_and_keeps_every_unit(tmp_path):
    data = minimal(tmp_path)
    data["onset"] = {"threshold": {"climatology": {"years": "2001:2003", "start": "05-01", "end": "08-31"}},
                     "definition": {"window": 3}}
    onset = C.convert_blend(data).mapping["onset"]
    assert (onset["threshold_max_missing"], onset["threshold_partial_years"], onset["threshold_years"]) == (
        1.0, True, [2001, 2003])
    assert onset["rule"]["thresholds"] == {"thr": {"mean total over": "3 days", "during": "05-01 to 08-31"}}


def test_a_table_threshold_names_its_columns_and_its_scale(tmp_path):
    data = minimal(tmp_path)
    data["onset"] = {"threshold": {"file": "thr.csv", "id_col": "uid", "scale": {"factor": 2.0, "max": 40}}}
    onset = C.convert_blend(data).mapping["onset"]
    assert onset["thresholds"] == {"thr": str(tmp_path / "thr.csv")}
    assert onset["threshold_table"]["id_col"] == "uid" and onset["threshold_table"]["value_col"] == "onset_thresh"
    assert onset["threshold_scale"] == {"factor": 2.0, "offset": 0.0, "max": 40}


def test_blendings_follow_token_keeps_its_number(tmp_path):
    """`follow` is follow_days (30) in blending; the definition's follow span from the onset day is 31: the number stays."""
    loaded = converted_minimal(tmp_path)
    [model] = loaded.recipe.models
    assert [(p.agg, p.window) for p in model.rain_predictors] == [("diff", "trigger"), ("max", 30)]
    # a token the definition reads to the same number of days stays a token
    after = minimal(tmp_path)
    after["onset"]["definition"] = {"follow_days": 21}
    run = C.convert_blend(after).mapping
    assert run["blend"]["models"]["m"]["rain_predictors"][1] == {"agg": "max", "window": "follow"}


def converted_minimal(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    for folder, dims in (("obs", ("time", "lat", "lon")), ("m", ("time", "day", "lat", "lon"))):
        (tmp_path / folder).mkdir(exist_ok=True)
        times = pd.date_range("2001-05-01", periods=4)
        shape = (4, 2, 2) if folder == "obs" else (4, 12, 2, 2)
        xr.Dataset({"rain": (dims, np.ones(shape, "float32"))},
                   coords={"time": times, "lat": [10.0, 10.25], "lon": [70.0, 70.25],
                           **({"day": np.arange(1, 13)} if folder == "m" else {})}).to_netcdf(tmp_path / folder / "2001.nc")
    data = minimal(tmp_path)
    data["models"][0].update(variable="rain", lead_days={"max": 40})
    return converted(data)


def test_a_converted_run_has_no_benchmark_settings_and_still_loads(tmp_path):
    """A blend-only run file writes no `evaluation:` block: the dates and climatology years a run folder and `--check`
    name come from blend.years (and the onset's threshold years)."""
    loaded = converted_minimal(tmp_path)
    assert "evaluation" not in loaded.conversion.mapping
    assert "evaluation" not in yaml.safe_load(loaded.conversion.to_yaml())
    cfg = loaded.cfg
    assert (cfg["start_date"], cfg["end_date"]) == ((2001, 1, 1), (2003, 12, 31))
    assert (cfg["start_year_clim"], cfg["end_year_clim"]) == (2001, 2003)
    assert cfg["config_sources"]["start_date"] == "blend.years"


def test_benchmark_keys_a_blend_run_does_state_are_kept(tmp_path):
    """Given dates are not replaced by the ones derived from blend.years; a run without a blend still needs them."""
    from typed_io import flat_config
    from hcwf.errors import ConfigSchemaError

    mapping = with_stub_data(C.convert_blend(minimal(tmp_path)).mapping, tmp_path / "stub")
    stated = flat_config({**mapping, "evaluation": {"start_date": [2002, 5, 1], "end_date": [2003, 6, 1],
                                                     "start_year_clim": 2001, "end_year_clim": 2002}}, write=False, tmp=tmp_path)
    assert stated["start_date"] == (2002, 5, 1) and stated["config_sources"]["start_date"] == "run config"
    benchmark = {k: v for k, v in mapping.items() if k != "blend"}
    with pytest.raises(ConfigSchemaError, match="start_date"):
        flat_config(benchmark, write=False, tmp=tmp_path)


def test_unknown_keys_and_missing_parts_are_reported_together_with_suggestions(tmp_path):
    data = minimal(tmp_path)
    data["onset"]["definiton"] = {}
    data["observations"]["varible"] = "x"
    del data["name"]
    with pytest.raises(ConvertError) as caught:
        C.convert_blend(data)
    text = str(caught.value)
    assert 'onset.definiton: unknown key. Did you mean "definition"?' in text
    assert 'observations.varible: unknown key. Did you mean "variable"?' in text
    assert "name: required" in text


def test_exactly_one_threshold_and_one_file_source_are_required(tmp_path):
    data = minimal(tmp_path)
    data["onset"]["threshold"] = {"mm": 5, "file": "t.csv"}
    data["observations"]["files"] = {"folder": "obs", "paths": ["a.nc"]}
    with pytest.raises(ConvertError) as caught:
        C.convert_blend(data)
    assert any("exactly one of 'mm', 'file' or 'climatology'" in p for p in caught.value.problems)
    assert any("either 'folder'" in p for p in caught.value.problems)


def test_units_domain_and_weights_per_source(tmp_path):
    data = minimal(tmp_path)
    data["domain"] = {"grid_ids": {"decimal_digits": 2, "format": "fixed"}, "regrid": {"weights": "w.csv"},
                      "area_of_interest": {"bbox": {"lat_min": 9, "lat_max": 12}},
                      "dissemination": {"file": "d.csv", "id_col": "i"}, "dissemination_scope": "training"}
    data["models"][0]["cell_transform"] = "t.csv"
    units = C.convert_blend(data).mapping["units"]
    assert units["mode"] == {"observations": "regrid", "m": "cell_transform"}
    assert units["weights"] == {"observations": str(tmp_path / "w.csv"), "m": str(tmp_path / "t.csv")}
    assert units["box"] == {"lat_min": 9.0, "lat_max": 12.0} and units["id_digits"] == 2
    assert (units["dissemination_scope"], units["dissemination_id_col"]) == ("training", "i")


def test_a_shapefile_converts_with_the_planar_areas_onset_blending_weighs_by(tmp_path):
    data = minimal(tmp_path)
    data["domain"] = {"regrid": {"shapefile": "s.shp", "unit_id_col": "code"}}
    units = C.convert_blend(data).mapping["units"]
    assert units["shapefile"] == str(tmp_path / "s.shp") and units["cell_area"] == "planar"


def test_what_is_not_carried_over_is_listed(tmp_path):
    data = minimal(tmp_path)
    data["domain"] = {"regrid": {"weights": "w.csv", "crs": "EPSG:3857", "grid_lat_var": "la"}}
    conversion = C.convert_blend(data)
    text = conversion.to_yaml()
    assert "# Not carried over:" in text
    assert any("domain.regrid.crs" in n for n in conversion.notes) and any("grid_lat_var" in n for n in conversion.notes)


def test_the_two_registry_ids_are_blendings_readings():
    """moron_robertson_blend and icpac_blend are the rules `convert` writes for those definitions (icpac_blend with its
    20 mm as a fixed total)."""
    from hcwf.onset import definition as D

    template = {"trigger_rule": "first_day_wet", "wet_day_comparison": "gt", "follow_days": 30,
                "followup_anchor": "onset_day", "dry_spell": {"mode": "window_sum"}}
    rule = C.resolve_rule(template)
    made = D.loads(yaml.safe_dump(C._rule_mapping(rule, "x", "05-01", (5, "04-01", "10-31"))), fine_amounts=True)
    listed = D.load("moron_robertson_blend")
    assert (listed.rule, listed.thresholds, listed.windows()) == (made.rule, made.thresholds, made.windows())

    icpac = C.resolve_rule({"window": 5, "trigger_rule": "first_day_wet", "min_wet_days": 3,
                            "threshold_comparison": "gte", "follow_days": 0})
    text = C._rule_mapping(icpac, "x", "05-01", None)
    text["thresholds"] = None
    text["onset"]["require"][-1] = {"total": "at least 20.0 mm"}
    del text["thresholds"]
    assert D.load("icpac_blend").rule == D.loads(yaml.safe_dump(text), fine_amounts=True).rule


# -- onset_blending's configs and spec sets --------------------------------------------------------------------------


@needs_oracle
@pytest.mark.parametrize("path", CONFIGS)
def test_every_shipped_config_converts_loads_and_checks(path, tmp_path):
    """The converted YAML is read back unchanged, its blend section builds the Recipe, and with small stand-in files for the
    NetCDF sources the whole run config loads and `--check` finds no problem (a units file the config names that is not on
    this machine is the one error left)."""
    import hcwf

    conversion = C.convert_blend_file(path)
    assert yaml.safe_load(conversion.to_yaml()) == conversion.mapping
    loaded = partial(conversion)
    assert loaded.recipe.name == blending_dict(path)["name"]
    report = hcwf.check(with_stub_data(conversion.mapping, tmp_path))
    unexpected = [e for e in report.errors if "the units'" not in e or "file is missing" not in e]
    assert not unexpected, report.text()


def spec_paths(name):
    sets = {
        "country_template": dict(ground_truth="raw_data/ground_truth_template.yml",
                                 forecasts={"forecast_model": "raw_data/forecast_template.yml"},
                                 combine="combine/combine_country_template.yml",
                                 connect="2025_blend/connect_country_template.yml",
                                 cv_models="2025_blend/cv_models_country_template.yml", regrid="regrid/country_template.yml"),
        "fixed_cutoff_2026": dict(ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
                                  forecasts={"aifs_ens": "raw_data/aifs_ens_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
                                  combine="combine/combine_template_fixed_cutoff_2026.yml",
                                  connect="2025_blend/connect_fixed_cutoff_2026.yml",
                                  cv_models="2025_blend/cv_models_fixed_cutoff_2026.yml"),
        "fixed_cutoff_2026_gencast": dict(ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
                                          forecasts={"gencast": "raw_data/gencast_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
                                          combine="combine/combine_template_fixed_cutoff_2026_gencast.yml",
                                          connect="2025_blend/connect_fixed_cutoff_2026_gencast.yml",
                                          cv_models="2025_blend/cv_models_fixed_cutoff_2026_gencast.yml"),
        "fixed_cutoff_2026_ngcm": dict(ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
                                       forecasts={"ngcm": "raw_data/ngcm_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
                                       combine="combine/combine_template_fixed_cutoff_2026_ngcm.yml",
                                       connect="2025_blend/connect_fixed_cutoff_2026_ngcm.yml",
                                       cv_models="2025_blend/cv_models_fixed_cutoff_2026_ngcm.yml"),
        "fixed_cutoff_one_2026": dict(ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
                                      forecasts={"aifs_ens": "raw_data/aifs_ens_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
                                      combine="combine/combine_template_fixed_cutoff_2026.yml",
                                      connect="2025_blend/connect_fixed_cutoff_2026.yml",
                                      cv_models="2025_blend/cv_models_fixed_cutoff_one_2026.yml"),
    }
    paths = dict(sets[name])
    paths["forecasts"] = {m: SPECS / p for m, p in paths["forecasts"].items()}
    return {k: (SPECS / v if isinstance(v, str) else v) for k, v in paths.items()}


SPEC_SETS = ["country_template", "fixed_cutoff_2026", "fixed_cutoff_2026_gencast", "fixed_cutoff_2026_ngcm",
             "fixed_cutoff_one_2026"]


@needs_oracle
@pytest.mark.skipif(not SPECS.exists(), reason="onset_blending's archived specs are not available")
@pytest.mark.parametrize("name", SPEC_SETS)
def test_every_spec_set_converts_to_what_its_config_converts_to(name, tmp_path):
    """A spec set and the config onset_blending checked it against give the same run (the same Recipe, the same
    sections), and the run loads and checks."""
    import hcwf

    from_specs = C.convert_spec_files(name=name, data_root=str((ORACLE_SRC / "archive").resolve()), **spec_paths(name))
    from_config = C.convert_blend_file(ORACLE_SRC / "configs" / f"{name}.yml")
    assert partial(from_specs).recipe == partial(from_config).recipe
    assert {k: v for k, v in from_specs.mapping.items() if k != "blend"} == {
        k: v for k, v in from_config.mapping.items() if k != "blend"}
    assert any("run.MR" in n for n in from_specs.notes) or from_specs.notes
    report = hcwf.check(with_stub_data(from_specs.mapping, tmp_path))
    assert not [e for e in report.errors if "the units'" not in e or "file is missing" not in e], report.text()


def test_the_spec_converter_reports_disagreements_together(tmp_path):
    if not SPECS.exists():
        pytest.skip("onset_blending's archived specs are not available")
    load = lambda p: yaml.safe_load(Path(p).read_text())
    paths = spec_paths("country_template")
    specs = dict(ground_truth=load(paths["ground_truth"]), forecasts={m: load(p) for m, p in paths["forecasts"].items()},
                 combine=load(paths["combine"]), connect=load(paths["connect"]), cv_models=load(paths["cv_models"]),
                 regrid=load(paths["regrid"]))
    specs["forecasts"]["forecast_model"]["options"]["window"] = 3
    specs["cv_models"]["n_bins"] = 5
    with pytest.raises(ConvertError) as caught:
        C.convert_specs(name="t", **specs)
    assert any("the onset definition" in p for p in caught.value.problems)
    assert any("the number of week bins" in p for p in caught.value.problems)


@needs_oracle
@pytest.mark.skipif(not SPECS.exists(), reason="onset_blending's archived specs are not available")
def test_hcwf_run_convert_specs_writes_the_run_of_a_spec_set(tmp_path, capsys):
    from hcwf.cli import main

    paths = spec_paths("fixed_cutoff_2026")
    target = tmp_path / "run.yaml"
    main(["convert", "--specs", "--name", "fixed_cutoff_2026", "--ground-truth", str(paths["ground_truth"]),
          *(a for m, f in paths["forecasts"].items() for a in ("--forecast", f"{m}={f}")),
          "--combine", str(paths["combine"]), "--connect", str(paths["connect"]), "--cv-models", str(paths["cv_models"]),
          "--data-root", str(ORACLE_SRC / "archive"), "-o", str(target)])
    text = target.read_text()
    assert text.startswith("# Converted from the stage specs") and "# Not carried over:" in text and "#   combine: " in text
    raw = yaml.safe_load(text)
    assert raw["project_name"] == "fixed_cutoff_2026" and set(raw["blend"]["models"]) <= {"aifs_ens", "aifs"}
    with pytest.raises(SystemExit):
        main(["convert", "--specs", "--name", "x", "-o", str(tmp_path / "x.yaml")])
    assert "--specs needs" in capsys.readouterr().err


def test_saved_mme_weights_are_an_explicit_input_after_conversion(tmp_path):
    data = minimal(tmp_path)
    data["evaluation"] = {**data["evaluation"], "mme": {"enabled": True, "optimize": "never", "weights_year_tag": "_2000_2024"}}
    conversion = C.convert_blend(data)
    assert "weights_year_tag" not in conversion.mapping["blend"]["mme"]
    assert any("blend.mme.weights" in note and "_2000_2024" in note for note in conversion.notes)


def test_a_blend_gets_the_blend_lead_window_through_every_entry(tmp_path):
    """Path, Config, mapping, flat dict and command line all give a blend (1, max_forecast_day), not the benchmark window."""
    from hcwf.config import Config
    from typed_io import flat_config, flat_command_line

    mapping = with_stub_data(C.convert_blend(minimal(tmp_path)).mapping, tmp_path / "stub")
    path = tmp_path / "blend_run.yaml"
    path.write_text(yaml.safe_dump(mapping))
    flat = flat_config(mapping, write=False, tmp=tmp_path)
    for dic in (flat, flat_config(path), flat_config(Config.from_mapping(mapping)), flat_config(dict(flat)),
                flat_command_line(["-p", str(path)])):
        assert dic["lead_window"] == (1, dic["max_forecast_day"]) and dic["onset_lead_lines"] == ()
