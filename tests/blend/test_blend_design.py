"""momp.blend.design: design matrices from arrays are Patsy's to the bit, and formulas expand and resolve as the
legacy build_formulas_from_spec and resolve_formula_str did (oracle: onset_blending@1e6d696)."""

import numpy as np
import pandas as pd
import patsy
import pytest

from momp.blend import design
from tests.blend._oracle import blend_evaluation_utils as legacy
from tests.blend._samples import FrameSamples, weekly_table


def test_a_plan_builds_patsys_matrix_to_the_bit():
    frame = weekly_table()
    formula = "outcome ~ prob_clim_week1 * diff_m_week1 + prob_clim_week2 + diff_m_week1:prob_clim_week2"
    samples = FrameSamples(frame)
    rows = np.arange(10, 90, 3)

    plan = design.plan(formula, samples.column_dtypes())
    built = plan.build(samples.take, rows)

    expected = patsy.dmatrix(design.right_hand_side(formula), frame.iloc[rows], return_type="dataframe",
                             NA_action=patsy.NAAction(NA_types=[])).drop(columns="Intercept")
    assert list(plan.columns) == list(expected.columns)
    np.testing.assert_array_equal(built, expected.to_numpy())
    assert set(plan.features) == {"prob_clim_week1", "diff_m_week1", "prob_clim_week2"}


@pytest.mark.parametrize("formula", ["outcome ~ np.log(diff_m_week1 ** 2 + 1)", "outcome ~ C(year) + diff_m_week1",
                                     "outcome ~ id", "outcome ~ 1"])
def test_formulas_without_a_numeric_product_have_no_plan(formula):
    assert design.plan(formula, FrameSamples(weekly_table()).column_dtypes()) is None


def test_expansion_is_the_legacy_one():
    for text in ("outcome ~ prob_clim_qx + diff_fc_qx", "outcome ~ prob_clim_qx * diff_fc_qx", "y ~ a * b * c + d_week2"):
        for n_bins in (2, 4):
            assert design.expand(text, n_bins) == legacy.expand_formula_str(text, n_bins)


def test_resolution_drops_the_terms_a_rain_horizon_cannot_build():
    text = "outcome ~ prob_clim_qx + diff_fc_qx + diff_fc_week1"
    unavailable = {"diff_fc_week2": {"rain_day_max": 7}}
    data = pd.DataFrame({"a": [1]})
    data.attrs["unavailable_rain_predictors"] = unavailable

    resolved, dropped = design.resolve(text, unavailable, 3)

    assert (resolved, dropped) == legacy.resolve_formula_str(text, data, 3)
    assert dropped == ["diff_fc_week2"]
    with pytest.raises(ValueError, match="no constructible predictor terms"):
        design.resolve("outcome ~ diff_fc_qx", {f"diff_fc_week{i}": {} for i in (1, 2, 3)}, 3)


def test_formulas_are_build_formulas_from_spec_with_window_variants():
    entries = {"blend": design.Formula(text="outcome ~ prob_clim_qx * diff_fc_qx"),
               "clim_only": design.Formula(text="outcome ~ prob_clim_qx"),
               "off": design.Formula(text="outcome ~ x", enabled=False)}
    variants = [design.WindowVariant(base="blend", windows=((2018, 2024),))]
    spec = {"models": {"formulas": {"blend": {"text": entries["blend"].text}, "clim_only": {"text": entries["clim_only"].text}}}}

    formulas, resolution = design.formulas(entries, variants, {}, 2, "fixed_cutoff")

    legacy_formulas, legacy_resolution = legacy.build_formulas_from_spec(spec, "fixed_cutoff", n_bins=2, return_resolution=True)
    assert {k: formulas[k] for k in legacy_formulas if k in ("blend", "clim_only")} == {
        k: legacy_formulas[k] for k in ("blend", "clim_only")}
    assert resolution["blend"] == legacy_resolution["blend"]
    assert list(formulas) == ["blend", "clim_only", "blend_2018_2024"]
    assert formulas["blend_2018_2024"] == formulas["blend"].replace("prob_clim_week", "prob_clim_2018_2024_week")
    assert resolution["blend_2018_2024"]["original_formula"] == "outcome ~ prob_clim_2018_2024_qx * diff_fc_qx"


def test_window_variants_can_be_limited_to_a_cutoff_mode_and_must_name_a_base():
    entries = {"blend": design.Formula(text="outcome ~ prob_clim_qx")}

    only = design.WindowVariant(base="blend", windows=((2018, 2024),), only_if_cutoff_mode="ref_onset")
    assert list(design.formulas(entries, [only], {}, 2, "fixed_cutoff")[0]) == ["blend"]
    with pytest.raises(ValueError, match="must match a name"):
        design.formulas(entries, [design.WindowVariant(base="other", windows=((2018, 2024),))], {}, 2, "x")
    with pytest.raises(ValueError, match="models.formulas"):
        design.formulas({}, [], {}, 2, "x")


def test_sample_support_is_the_legacy_one():
    frame = weekly_table()
    formulas = {"a": "outcome ~ prob_clim_week1 + diff_m_week1", "b": "outcome ~ prob_clim_week2"}

    keep, diagnostics = design.sample_support(FrameSamples(frame), formulas)

    expected, expected_diagnostics = legacy.formula_sample_support(frame, formulas)
    np.testing.assert_array_equal(keep, np.flatnonzero(expected))
    assert diagnostics == expected_diagnostics
    assert 5 not in keep and len(keep) == len(frame) - 1

    with pytest.raises(ValueError, match="missing predictor columns: nope"):
        design.sample_support(FrameSamples(frame), {"a": "outcome ~ nope"})
    with pytest.raises(ValueError, match="no finite values"):
        design.sample_support(FrameSamples(frame.assign(diff_m_week1=np.nan)), formulas)
