"""momp.blend.evaluation: the final fit (`fit_final`, and `evaluate(final=True)` alongside the CV folds) is
3_fit_final_model.py's bundle. The deferred part of onset_blending's tests/test_model.py (the model's own tests are
tests/blend/test_blend_logit.py); oracle onset_blending@1e6d696, tests/blend/_oracle."""

import pickle
import warnings

import numpy as np
import pandas as pd
import patsy
import pytest
from _blend_configs import blend_config, dissemination_ids, legacy_spec, recipe_of
from sklearn.preprocessing import StandardScaler

from momp.blend.evaluation import evaluate, fit_final
from tests.blend._oracle.blend_evaluation_utils import (
    _make_multinom_clf,
    apply_formula_sample_support,
    build_formulas_from_spec,
    resolve_climatology_weeks,
    restrict_to_allowed,
)
from tests.blend._oracle.nc_utils_filters import ensure_id_col
from tests.blend._oracle.spatial_id_utils import resolve_grid_id_convention

BINS = ["week1", "week2", "later"]
FORMULA = "outcome ~ prob_clim_week1 + prob_clim_week2 + diff_m_week1 + prob_clim_week1:diff_m_week1"


def _legacy_fit_final_model(formula_str, train_df):
    """3_fit_final_model.py's fit_final_model, as it was (prints removed)."""
    rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    X_df = patsy.dmatrix(
        rhs, train_df, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[])
    )
    X_df = X_df.drop(columns=["Intercept"], errors="ignore")
    feature_cols = list(X_df.columns)
    y = train_df["outcome"].values
    valid = np.array([isinstance(v, str) for v in y])
    X_vals = X_df.values.astype(float)
    mask = valid & np.isfinite(X_vals).all(axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals[mask])
    clf = _make_multinom_clf()
    clf.fit(X_scaled, y[mask])
    rows = []
    for i, cls in enumerate(clf.classes_):
        for j, feat in enumerate(feature_cols):
            rows.append(
                {
                    "class": cls,
                    "feature": feat,
                    "coefficient": float(clf.coef_[i, j]),
                    "intercept": float(clf.intercept_[i]),
                }
            )
    return {
        "coefs": pd.DataFrame(rows),
        "scaler": scaler,
        "features": feature_cols,
        "formula": formula_str,
    }


def _legacy_final_training_rows(wide_df, spec, dissemination_input, model):
    """3_fit_final_model.py's main(): the formula and rows it fitted."""
    training_years = list(map(int, spec["run"]["training_years"]))
    true_holdout_years = set(map(int, spec["run"].get("true_holdout_years") or []))
    fit_years = [year for year in training_years if year not in true_holdout_years]
    wide_df = ensure_id_col(wide_df, spec=spec, context="blend input IDs")
    id_convention = resolve_grid_id_convention(
        spec=spec, authoritative_ids=wide_df["id"], context="blend input/dissemination ID handoff"
    )
    dissemination_cells = ensure_id_col(
        dissemination_input, spec=spec, convention=id_convention, context="blend dissemination IDs"
    ).drop_duplicates("id")
    _, clim_weeks = resolve_climatology_weeks(spec, wide_df)
    formulas = build_formulas_from_spec(
        spec, spec["run"]["cutoff_mode"], data=wide_df, n_bins=max(clim_weeks)
    )
    wide_df, _ = apply_formula_sample_support(wide_df, formulas)
    wide_df_allowed = restrict_to_allowed(wide_df, dissemination_cells)
    return formulas[model], wide_df_allowed[wide_df_allowed["year"].isin(fit_years)].copy()


def weekly_table(seed=0, years=range(2010, 2021), n_cells=4, issues=8):
    rng = np.random.default_rng(seed)
    rows = [
        {"id": f"unit{c}", "year": year, "issue": k}
        for c in range(n_cells)
        for year in years
        for k in range(issues)
    ]
    frame = pd.DataFrame(rows)
    n = len(frame)
    frame["prob_clim_week1"] = rng.normal(-1.0, 0.5, n)
    frame["prob_clim_week2"] = rng.normal(-1.0, 0.5, n)
    frame["diff_m_week1"] = rng.normal(size=n)
    # Outcomes that depend on the features, so the coefficients mean something.
    logits = np.column_stack([frame["prob_clim_week1"] + frame["diff_m_week1"], frame["prob_clim_week2"], np.zeros(n)])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    frame["outcome"] = [BINS[rng.choice(3, p=row)] for row in p]
    frame.loc[rng.random(n) < 0.05, "outcome"] = np.nan
    frame.loc[5, "diff_m_week1"] = np.nan  # outside the formula sample support
    return frame


CONFIG = blend_config(
    {
        "cutoff_mode": "fixed_cutoff",
        "years": {"training": "2010:2019", "true_holdout": [2019], "cv_holdout": [2018]},
        "formulas": {"blend": "outcome ~ prob_clim_qx + diff_m_week1 + prob_clim_week1:diff_m_week1"},
    },
    models=("m",),
)


SPEC = legacy_spec(CONFIG)
RECIPE = recipe_of(CONFIG)


def dissemination(weekly):
    return pd.DataFrame({"adm3_name": sorted(weekly["id"].unique())[:-1]})


def final_model(weekly):
    return fit_final(weekly, RECIPE, dissemination_ids(dissemination(weekly)))


def test_final_fit_is_the_scripts_bundle():
    weekly = weekly_table()
    model = final_model(weekly)
    formula, train = _legacy_final_training_rows(weekly, SPEC, dissemination(weekly), "blend")
    legacy = _legacy_fit_final_model(formula, train)

    assert model.formula == formula == FORMULA
    assert pickle.dumps(model.to_legacy_bundle()) == pickle.dumps(legacy)
    assert model.classes == ("later", "week1", "week2")
    assert model.n_samples == int(legacy["scaler"].n_samples_seen_)


def test_final_fit_beside_the_cv_folds_is_the_scripts_bundle():
    weekly = weekly_table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluation = evaluate(weekly, RECIPE, dissemination_ids(dissemination(weekly)), workers=2, final=True)
    formula, train = _legacy_final_training_rows(weekly, SPEC, dissemination(weekly), "blend")
    bundle, legacy = evaluation.final_model.to_legacy_bundle(), _legacy_fit_final_model(formula, train)

    # The same content (a worker's strings are its own objects, so the pickle's bytes differ).
    assert sorted(bundle) == sorted(legacy)
    pd.testing.assert_frame_equal(bundle["coefs"], legacy["coefs"], check_exact=True)
    assert (bundle["features"], bundle["formula"]) == (legacy["features"], legacy["formula"])
    assert list(vars(bundle["scaler"])) == list(vars(legacy["scaler"]))
    for name, value in vars(legacy["scaler"]).items():
        np.testing.assert_array_equal(vars(bundle["scaler"])[name], value, strict=True)


def test_final_fit_requires_every_outcome_class():
    weekly = weekly_table()
    weekly.loc[weekly["outcome"] == "week2", "outcome"] = "later"

    with pytest.raises(ValueError, match="missing outcome classes: week2"):
        final_model(weekly)
