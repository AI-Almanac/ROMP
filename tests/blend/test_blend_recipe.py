"""The blend Recipe: the shipped blending configs translate, their ROMP-config half resolves to an OnsetSetup, and
the recipe's rules refuse what blending's BlendConfig.validate refused."""
import dataclasses
import glob

import pytest

from _recipes import ORACLE_SRC, load_blending_config, translate
from momp.blend import recipe as R
from momp.stats.bins import LeadBins

CONFIGS = sorted(glob.glob(str(ORACLE_SRC / "configs" / "*.yml")))


@pytest.mark.skipif(not CONFIGS, reason="onset_blending's configs are not available")
@pytest.mark.parametrize("path", CONFIGS)
def test_the_shipped_configs_translate(path):
    config = load_blending_config(path)
    t = translate(config, strict=False)
    assert t.recipe.name == config.name and t.setup.definition.windows()["trigger"] == config.onset.definition.window
    assert t.cfg["onset_legacy"] is False and t.cfg["rain_resolution"] == "exact"
    assert t.recipe.outcome_labels() == [*(f"week{w}" for w in range(1, config.features.bins.count + 1)), "later"]


@pytest.fixture(scope="module")
def recipe():
    config = load_blending_config(CONFIGS[1])
    return translate(config, strict=False)


def test_rain_windows_are_read_from_the_definition(recipe):
    predictor = R.RainPredictor("max", "follow")
    assert recipe.recipe.window_days(predictor, recipe.setup.definition) == 21
    with pytest.raises(R.RecipeError, match="dry window"):
        recipe.recipe.window_days(R.RainPredictor("min", "dry_spell"), recipe.setup.definition)


def test_formulas_may_only_name_features_the_recipe_builds(recipe):
    bad = dataclasses.replace(recipe.recipe.evaluation, formulas=(("x", R.Formula(text="outcome ~ prob_clim_qx + nope_qx")),),
                              final_formula="x", window_variants=())
    with pytest.raises(R.RecipeError, match="nope_qx"):
        dataclasses.replace(recipe.recipe, evaluation=bad).check(recipe.setup.definition)


def test_bins_must_fit_the_probability_days(recipe):
    features = dataclasses.replace(recipe.recipe.features, bins=LeadBins.uniform(5, 7))
    with pytest.raises(R.RecipeError, match="probability days"):
        dataclasses.replace(recipe.recipe, features=features)


def test_derived_names(recipe):
    r = recipe.recipe
    assert R.probability_feature_base("m", "ref_onset") == "m_p_onset_ref"
    assert R.window_tag((2000, 2010)) == "2000_2010"
    assert r.output_tag().startswith("_fixed_cutoff_") or r.output_tag().startswith("_")
