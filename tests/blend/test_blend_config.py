"""The `blend:` section of a run YAML and the dataset keys a blend needs (`paths`, `years`, `unit_cvt`, dims, per-source
units): schema, did-you-mean, the Recipe built from them, and the resolved config that is written for a run."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hcwf.blend import recipe as R
from hcwf.config import Blend, Config, Dataset, Units
from hcwf.lib import config_schema as schema
from hcwf.lib.config_schema import ConfigSchemaError
from hcwf.lib.config_yaml import build_config

BASE = {"model_list": ("m",), "blend_years": {"training": [2000, 2001], "cv_holdout": [2001]},
        "blend_formulas": {"f": "outcome ~ prob_clim_qx"}}


def test_blend_keys_are_a_schema_section_with_help_and_suggestions():
    assert {f.nested_name for f in schema.FIELDS if f.section == "blend"} >= {"years", "formulas", "models", "mme"}
    assert all(f.help for f in schema.FIELDS if f.section == "blend") and set(schema.BLEND_KEYS) <= set(schema.BY_KEY)
    errors = []
    schema.flatten_sections({"blend": {"yeras": {}}}, errors)
    assert errors == [schema.unknown_key_message("yeras", "blend")] and 'Did you mean "years"?' in errors[0]
    assert schema.has_blend({"blend_years": {}}) and not schema.has_blend({"blend_years": None, "region": "x"})


def test_a_recipe_takes_blendings_defaults():
    recipe = R.recipe_from_config(BASE)
    assert recipe.features == R.Features() and recipe.climatology == R.ClimatologySpec()
    assert recipe.models == (R.Model("m"),) and recipe.evaluation.mme == R.Mme() and recipe.outputs == R.Outputs()
    assert recipe.evaluation.baselines == R.Baselines() and recipe.onset == R.OnsetExtras()
    assert recipe.evaluation.years.training == (2000, 2001) and recipe.final_formula() == "f"


def test_the_section_is_read_into_the_recipe():
    cfg = {**BASE, "project_name": "p", "blend_years": {"training": "2000:2002,2005", "cv_holdout": 2001, "true_holdout": 2005},
           "blend_cutoff_mode": "no_ref_filter", "blend_cv_methods": ("global", "local"),
           "blend_bins": {"count": 2, "days": 14}, "blend_probability_days": 28, "blend_rain_transform": "sqrt",
           "blend_fixed_cutoff": "06-10", "blend_reference_onset": {"month_day": "06-15"},
           "blend_climatology": {"issue_end": "08-31", "train_years": "2000:2010", "windows": [[2000, 2005], "2006:2010"],
                                 "workers": 2},
           "blend_baselines": {"windows": [[2000, 2005]]}, "blend_forecasts": [{"model": "m", "variant": "base"}],
           "blend_mme": {"enabled": True, "sources": [{"name": "clim_raw"}, {"name": "m", "source": "forecast"}]},
           "blend_random_forest": {"trees": 10, "max_features": "all"},
           "blend_models": {"m": {"lead_days": {"min": 1, "max": 40}, "max_member": 3,
                                  "rain_horizon": {"max_day": 40, "policy": "truncate"},
                                  "rain_predictors": [{"agg": "diff"}, {"agg": "min", "window": 10}]}},
           "blend_outputs": {"intermediates": ["features"], "pickles": True}}
    recipe = R.recipe_from_config(cfg)
    assert recipe.name == "p" and recipe.evaluation.years.training == (2000, 2001, 2002, 2005)
    assert recipe.evaluation.years.true_holdout == (2005,) and recipe.evaluation.cutoff_mode == "no_ref_filter"
    assert recipe.features.count == 2 and recipe.features.days == 14 and recipe.onset.fixed_cutoff == "06-10"
    assert recipe.onset.reference.month_day == "06-15" and recipe.climatology.train_years == (2000, 2010)
    assert recipe.climatology.windows == ((2000, 2005), (2006, 2010)) and recipe.climatology.workers == 2
    assert recipe.evaluation.random_forest.max_features is None and recipe.evaluation.random_forest.days_per_week == 14
    model = recipe.model("m")
    assert (model.lead_days, model.max_member, model.rain_horizon.policy) == ((1, 40), 3, "truncate")
    assert model.rain_predictors == (R.RainPredictor("diff", "trigger"), R.RainPredictor("min", 10))
    assert recipe.outputs == R.Outputs(None, ("features",), True) and recipe.evaluation.mme.enabled
    assert recipe.evaluation.forecasts == (R.ForecastScoring("m", "base"),)


def test_every_problem_of_the_section_is_reported_with_suggestions():
    cfg = {**BASE, "blend_years": {"trainin": [2000], "cv_holdout": "20x0"}, "blend_cv_methods": ("globl",),
           "blend_models": {"mm": {"lead_days": [1]}, "m": {"rain_predictors": [{"agg": "avg"}]}},
           "blend_mme": {"enabled": True, "optimize": "sometimes"}, "blend_rain_transform": "cube"}
    with pytest.raises(R.RecipeError) as caught:
        R.recipe_from_config(cfg)
    text = "\n".join(caught.value.problems)
    for expected in ('blend.years.trainin: unknown key. Did you mean "training"?', "blend.years.cv_holdout",
                     "blend.cv_methods: 'globl' is not one of", 'blend.models.mm: not one of the models. Did you mean "m"?',
                     "blend.models.m.rain_predictors[0].agg: 'avg' is not one of", "blend.mme.optimize"):
        assert expected in text, expected
    with pytest.raises(R.RecipeError, match="blend.years: required"):
        R.recipe_from_config({"model_list": ("m",), "blend_formulas": {"f": "outcome ~ x"}})
    with pytest.raises(R.RecipeError, match="rain_transform"):
        R.recipe_from_config({**BASE, "blend_rain_transform": "cube"})


def write_files(folder, years=(2001, 2002), name=lambda y: f"{y}.nc", forecast=False):
    folder.mkdir(parents=True, exist_ok=True)
    for year in years:
        times = pd.date_range(f"{year}-05-01", periods=10)
        if forecast:
            ds = xr.Dataset({"tp": (("time", "day", "lat", "lon"), np.ones((10, 5, 2, 2), "float32"))},
                            coords={"time": times, "day": np.arange(1, 6), "lat": [10.0, 10.25], "lon": [70.0, 70.25]})
        else:
            ds = xr.Dataset({"rain": (("time", "lat", "lon"), np.ones((10, 2, 2), "float32"))},
                            coords={"time": times, "lat": [10.0, 10.25], "lon": [70.0, 70.25]})
        ds.to_netcdf(folder / name(year))


def run_mapping(tmp_path, **sections):
    write_files(tmp_path / "obs", (2001, 2002, 2003))
    write_files(tmp_path / "m", (2001, 2002, 2003), forecast=True)
    raw = {"region": None, "observations": {"path": str(tmp_path / "obs"), "variable": "rain", "unit_cvt": 1.0,
                                            "day_boundary": "00 UTC, labelled by end"},
           "models": [{"name": "m", "path": str(tmp_path / "m"), "variable": "tp", "unit_cvt": 1000.0,
                       "lead_label": "end", "accumulation": "daily"}],
           "onset": {"definition": "icpac_blend", "thresholds": {"thr": 20.0}} if False else {"definition": "icpac_blend"},
           "evaluation": {"start_date": [2001, 5, 1], "end_date": [2003, 12, 31], "start_year_clim": 2001,
                          "end_year_clim": 2003, "max_forecast_day": 5, "day_bins": [[1, 5]],
                          "verification_windows": [[1, 5]], "tolerances": [3]}}
    for key, value in sections.items():
        raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
    return raw


def test_listed_files_years_unit_cvt_and_a_null_region_resolve_to_the_flat_keys(tmp_path):
    raw = run_mapping(tmp_path, observations={"paths": [str(tmp_path / "obs" / "*.nc")], "years": "2002:2003", "path": None})
    raw["observations"].pop("path")
    raw["models"][0].update(paths=["*.nc"], years=[2001, 2003], path=str(tmp_path / "m"))
    flat = build_config(raw, source="t")
    assert [p.rsplit("/", 1)[1] for p in flat["obs_paths"]] == ["2002.nc", "2003.nc"] and flat["obs_dir"] == ""
    assert [p.rsplit("/", 1)[1] for p in flat["model_paths_list"][0]] == ["2001.nc", "2003.nc"]
    assert flat["obs_unit_cvt"] == 1.0 and flat["unit_cvt_list"] == (1000.0,) and flat["region"] is None
    assert flat["model_dims_list"][0]["step"] == "day"


def test_a_blend_reads_every_yearly_file_of_a_folder_and_pattern(tmp_path):
    raw = run_mapping(tmp_path, blend={"years": {"training": [2001], "cv_holdout": [2002]}, "formulas": {"f": "outcome ~ prob_clim_qx"}})
    flat = build_config(raw, source="t")
    assert [p.rsplit("/", 1)[1] for p in flat["obs_paths"]] == ["2001.nc", "2002.nc", "2003.nc"]
    assert len(flat["model_paths_list"][0]) == 3 and flat["blend_years"]["training"] == [2001]
    assert "obs_paths" not in build_config(run_mapping(tmp_path), source="t")   # without a blend nothing is listed


def test_only_dims_the_user_gave_rename_observations(tmp_path):
    raw = run_mapping(tmp_path)
    assert "obs_dims" not in build_config(raw, source="t")
    raw["observations"]["dims"] = {"time": "time", "lat": "lat", "lon": "lon"}
    assert build_config(raw, source="t")["obs_dims"]["lat"] == "lat"


def test_blend_problems_are_reported_with_the_config_errors(tmp_path):
    raw = run_mapping(tmp_path, blend={"years": {"training": [2001]}, "formulas": {}, "models": {"mm": {}}})
    with pytest.raises(ConfigSchemaError) as caught:
        build_config(raw, source="t")
    assert "blend: blend.models.mm: not one of the models" in str(caught.value)


def test_units_mode_is_one_mode_or_a_mapping_by_source():
    errors = []
    field = schema.BY_KEY["units_mode"]
    assert schema.normalise_value(field, {"observations": "cells", "m": "regrid"}, "units.mode", errors)
    assert schema.normalise_value(field, "cells", "units.mode", errors) == "cells" and not errors
    schema.normalise_value(field, {"m": "bogus"}, "units.mode", errors)
    assert len(errors) == 1 and "mapping of source name" in errors[0]


def test_config_objects_carry_units_and_blend_sections_and_listed_files():
    cfg = Config(region=None, observations=Dataset(paths=["a*.nc"], years="2000:2001", unit_cvt=1.0),
                 units=Units(mode={"observations": "cells"}), blend=Blend(years={"training": [2000]}))
    mapping = cfg.to_mapping()
    assert mapping["units"] == {"mode": {"observations": "cells"}} and mapping["blend"] == {"years": {"training": [2000]}}
    assert mapping["observations"] == {"paths": ["a*.nc"], "years": "2000:2001", "unit_cvt": 1.0}
    assert Config.from_mapping(mapping).to_mapping() == mapping   # the YAML round trip keeps both sections


def test_check_says_which_box_a_region_name_selects():
    from hcwf.lib.check import _box_text

    plain = _box_text({"region": "india"})
    assert _box_text({"region": "India"}) == plain   # the spelling does not matter
    assert "the domain of region india" in plain and "lat 6.5 to 38.5" in plain
    assert "no region" in _box_text({"region": None}) and "not in the registry" in _box_text({"region": "nowhere"})


def test_an_external_prediction_needs_a_name_and_a_file():
    with pytest.raises(R.RecipeError, match=r"blend.external_predictions\[0\]: name and file are required"):
        R.recipe_from_config({**BASE, "blend_external_predictions": [{"name": "x"}]})
    with pytest.raises(R.RecipeError, match="blend.external_predictions\\[0\\].fle: unknown key. Did you mean \"file\""):
        R.recipe_from_config({**BASE, "blend_external_predictions": [{"name": "x", "fle": "p.nc"}]})
    recipe = R.recipe_from_config({**BASE, "blend_external_predictions": [{"name": "x", "file": "p.nc", "method": "local"}]})
    assert recipe.evaluation.external_predictions == (R.ExternalPrediction("x", "p.nc", "local"),)


def test_mme_weights_are_a_mapping_or_a_file_and_must_fit_the_sources(tmp_path):
    sources = [{"name": "clim_raw"}, {"name": "m", "source": "forecast", "cal_variant": "fixed_cutoff"}]
    cfg = {**BASE, "blend_mme": {"enabled": True, "sources": sources, "weights": {"clim_raw": 0.25, "m": 0.75}}}
    assert R.recipe_from_config(cfg).evaluation.mme.weights == {"clim_raw": 0.25, "m": 0.75}
    cfg["blend_mme"]["weights"] = "weights.csv"
    assert R.recipe_from_config(cfg).evaluation.mme.weights == "weights.csv"
    for bad, message in (({"clim_raw": 1.0}, "one weight for each source"), ({"clim_raw": 0.5, "m": 0.9}, "sum to 1")):
        with pytest.raises(R.RecipeError, match=message):
            R.recipe_from_config({**cfg, "blend_mme": {**cfg["blend_mme"], "weights": bad}})
    with pytest.raises(R.RecipeError, match="cannot be 'always'"):
        R.recipe_from_config({**cfg, "blend_mme": {**cfg["blend_mme"], "optimize": "always"}})
    with pytest.raises(R.RecipeError, match="blend.mme.weights"):
        R.recipe_from_config({**cfg, "blend_mme": {"weights": [0.5, 0.5]}})
    with pytest.raises(R.RecipeError, match="blend.mme.weights.m"):
        R.recipe_from_config({**cfg, "blend_mme": {"weights": {"m": "heavy"}}})
    # the loader anchors a relative weights file at the config file
    from hcwf.config.build import _anchor_mme_weights

    dic = {"blend_mme": {"weights": "w/weights.csv"}, "config_file": str(tmp_path / "run.yaml")}
    _anchor_mme_weights(dic)
    assert dic["blend_mme"]["weights"] == str(tmp_path / "w" / "weights.csv")
    inline = {"blend_mme": {"weights": {"a": 1.0}}, "config_file": str(tmp_path / "run.yaml")}
    _anchor_mme_weights(inline)
    assert inline["blend_mme"]["weights"] == {"a": 1.0}
