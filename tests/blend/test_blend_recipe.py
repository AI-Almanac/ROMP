"""The blend Recipe: the shipped blending configs translate, their HCWF-config half resolves to an OnsetSetup, and
the recipe's rules refuse what blending's BlendConfig.validate refused."""
import dataclasses
import glob

import pytest

from _recipes import ORACLE_SRC, blending_dict, partial
from hcwf.blend import recipe as R
from hcwf.config.convert_specs import convert_blend
from hcwf.metrics.bins import LeadBins
from hcwf.errors import RecipeError

CONFIGS = sorted(glob.glob(str(ORACLE_SRC / "configs" / "*.yml")))


@pytest.mark.skipif(not CONFIGS, reason="onset_blending's configs are not available")
@pytest.mark.parametrize("path", CONFIGS)
def test_the_shipped_configs_translate(path):
    data = blending_dict(path)
    t = partial(convert_blend(data))
    window = (data["onset"].get("definition") or {}).get("window", 5)
    count = ((data.get("features") or {}).get("bins") or {}).get("count", 4)
    assert t.recipe.name == data["name"] and t.setup.definition.windows()["trigger"] == window
    assert t.cfg["rain_resolution"] == "exact"
    assert t.recipe.outcome_labels() == [*(f"week{w}" for w in range(1, count + 1)), "later"]


@pytest.fixture(scope="module")
def recipe():
    return partial(convert_blend(blending_dict(CONFIGS[1])))


def test_rain_windows_are_read_from_the_definition(recipe):
    predictor = R.RainPredictor("max", "follow")
    assert recipe.recipe.window_days(predictor, recipe.setup.definition) == 21
    with pytest.raises(RecipeError, match="dry window"):
        recipe.recipe.window_days(R.RainPredictor("min", "dry_spell"), recipe.setup.definition)


def test_formulas_may_only_name_features_the_recipe_builds(recipe):
    bad = dataclasses.replace(recipe.recipe.evaluation, formulas=(("x", R.Formula(text="outcome ~ prob_clim_qx + nope_qx")),),
                              final_formula="x", window_variants=())
    with pytest.raises(RecipeError, match="nope_qx"):
        dataclasses.replace(recipe.recipe, evaluation=bad).check(recipe.setup.definition)


def test_bins_must_fit_the_probability_days(recipe):
    features = dataclasses.replace(recipe.recipe.features, bins=LeadBins.uniform(5, 7))
    with pytest.raises(RecipeError, match="probability days"):
        dataclasses.replace(recipe.recipe, features=features)


def test_derived_names(recipe):
    r = recipe.recipe
    assert R.probability_feature_base("m", "ref_onset") == "m_p_onset_ref"
    assert R.window_tag((2000, 2010)) == "2000_2010"
    assert r.output_tag().startswith("_fixed_cutoff_") or r.output_tag().startswith("_")


def test_a_climatology_horizon_needs_its_forecast_window():
    problems = []
    R._climatology({"horizons": [{"start": "05-01", "end": "05-31"}]}, problems)
    assert len(problems) == 1 and "horizons[0].forecast_window: required" in problems[0]
    problems = []
    spec = R._climatology({"horizons": [{"start": "05-01", "end": "05-31", "forecast_window": 30}]}, problems)
    assert not problems and spec.horizon_dicts()[0]["forecast_window"] == 30


@pytest.mark.parametrize("value", ["02-30", "04-31", "02-29", "5-1"])
def test_recipe_rejects_month_days_that_not_every_year_has(value):
    from hcwf.blend.recipe import _is_month_day
    assert not _is_month_day(value)
    assert _is_month_day("05-01") and _is_month_day("12-31")


def test_schema_help_restates_the_blend_defaults():
    from hcwf.blend import random_forest
    from hcwf.config import schema
    help_of = {f.key: f.help for f in schema.FIELDS}
    assert f"'{R.DEFAULT_FIXED_CUTOFF}'" in help_of["blend_fixed_cutoff"]
    assert f"count: {R.DEFAULT_BINS[0]}, days: {R.DEFAULT_BINS[1]}" in help_of["blend_bins"]
    assert f"lead_days: {list(R.DEFAULT_LEAD_DAYS)}" in help_of["blend_models"]
    assert random_forest.DEFAULT_MISSING_ONSET in help_of["blend_random_forest"]
