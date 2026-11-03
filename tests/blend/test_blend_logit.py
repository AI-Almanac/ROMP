"""hcwf.blend.logit: FittedModel fits, converts and predicts as 3_fit_final_model.py and apply_blend_model.py did, and
the cross-validations give the legacy ones' numbers to the last bit (oracle: onset_blending@1e6d696, tests/blend/_oracle)."""

import dataclasses
import io
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import patsy
import pytest
from sklearn.preprocessing import StandardScaler

from hcwf.blend import design, logit
from hcwf.blend.logit import FittedModel
from tests.blend._oracle import blend_evaluation_utils as legacy
from tests.blend._samples import BINS, FrameSamples, weekly_table

FORMULA = "outcome ~ prob_clim_week1 + prob_clim_week2 + diff_m_week1 + prob_clim_week1:diff_m_week1"


def _legacy_fit_final_model(formula_str, train_df):
    """3_fit_final_model.py's fit_final_model, as it was (prints removed)."""
    rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    X_df = patsy.dmatrix(rhs, train_df, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[]))
    X_df = X_df.drop(columns=["Intercept"], errors="ignore")
    feature_cols = list(X_df.columns)
    y = train_df["outcome"].values
    valid = np.array([isinstance(v, str) for v in y])
    X_vals = X_df.values.astype(float)
    mask = valid & np.isfinite(X_vals).all(axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals[mask])
    clf = legacy._make_multinom_clf()
    clf.fit(X_scaled, y[mask])
    rows = []
    for i, cls in enumerate(clf.classes_):
        for j, feat in enumerate(feature_cols):
            rows.append({"class": cls, "feature": feat, "coefficient": float(clf.coef_[i, j]),
                         "intercept": float(clf.intercept_[i])})
    return {"coefs": pd.DataFrame(rows), "scaler": scaler, "features": feature_cols, "formula": formula_str}


def _legacy_apply(coef_bundle, test_df):
    """apply_blend_model.py's prediction path, as it was."""
    coef_df, scaler, feature_cols = coef_bundle["coefs"], coef_bundle["scaler"], coef_bundle["features"]
    formula_str = coef_bundle["formula"]
    rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    X_df = (
        patsy.dmatrix(rhs, test_df, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[]))
        .drop(columns=["Intercept"], errors="ignore")
        .loc[:, feature_cols]
    )
    if scaler is not None:
        X_df = pd.DataFrame(scaler.transform(X_df.values.astype(float)), columns=feature_cols, index=X_df.index)
    classes = sorted(coef_df["class"].unique())
    X = X_df[feature_cols].values.astype(float)
    W = np.zeros((len(classes), len(feature_cols)))
    b = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        cls_rows = coef_df[coef_df["class"] == cls]
        b[i] = cls_rows["intercept"].iloc[0]
        for j, feat in enumerate(feature_cols):
            feat_row = cls_rows[cls_rows["feature"] == feat]
            if not feat_row.empty:
                W[i, j] = float(feat_row["coefficient"].iloc[0])
    scores = X @ W.T + b
    bad = ~np.isfinite(X).all(axis=1)
    probs = np.full((len(X), len(classes)), np.nan)
    if (~bad).any():
        s = scores[~bad]
        s -= s.max(axis=1, keepdims=True)
        exp_s = np.exp(s)
        probs[~bad] = exp_s / exp_s.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, columns=[f"cv_{c}" for c in classes], index=X_df.index)


def final_rows(weekly):
    """The rows 3_fit_final_model.py fitted: dissemination units, training years without the true holdout."""
    allowed = sorted(weekly["id"].unique())[:-1]
    return np.flatnonzero(weekly["id"].isin(allowed) & weekly["year"].between(2010, 2018) & weekly["diff_m_week1"].notna())


def fitted(formula=FORMULA):
    weekly = weekly_table()
    return weekly, logit.fit_samples(formula, FrameSamples(weekly), final_rows(weekly))


def test_final_fit_is_the_scripts_bundle():
    weekly, model = fitted()
    legacy_bundle = _legacy_fit_final_model(FORMULA, weekly.iloc[final_rows(weekly)])

    assert model.formula == FORMULA
    assert pickle.dumps(model.to_legacy_bundle()) == pickle.dumps(legacy_bundle)
    assert model.classes == ("later", "week1", "week2")
    assert model.n_samples == int(legacy_bundle["scaler"].n_samples_seen_)


def test_final_fit_without_a_design_plan_is_the_scripts_bundle(monkeypatch):
    weekly = weekly_table()
    monkeypatch.setattr(design, "plan", lambda formula, dtypes: None)
    model = logit.fit_samples(FORMULA, FrameSamples(weekly), final_rows(weekly))

    assert pickle.dumps(model.to_legacy_bundle()) == pickle.dumps(
        _legacy_fit_final_model(FORMULA, weekly.iloc[final_rows(weekly)]))


def test_legacy_bundle_round_trip():
    _, model = fitted()
    bundle = model.to_legacy_bundle()

    assert pickle.dumps(FittedModel.from_legacy_bundle(bundle).to_legacy_bundle()) == pickle.dumps(bundle)


def test_arrays_round_trip_without_pickles():
    _, model = fitted()
    meta, arrays = model.to_arrays()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    buffer.seek(0)

    with np.load(buffer, allow_pickle=False) as loaded:
        again = FittedModel.from_arrays(json.loads(json.dumps(meta)), dict(loaded))

    assert pickle.dumps(again.to_legacy_bundle()) == pickle.dumps(model.to_legacy_bundle())


def test_predictions_are_apply_blend_models():
    _, model = fitted()
    test = weekly_table(seed=1, years=[2021]).set_index(pd.RangeIndex(100, 100 + 4 * 8))
    test.loc[105, "diff_m_week1"] = 0.5
    test.loc[103, "prob_clim_week2"] = np.nan

    predicted = model.predict(test)

    pd.testing.assert_frame_equal(predicted, _legacy_apply(model.to_legacy_bundle(), test), check_exact=True)
    assert list(predicted.columns) == ["cv_later", "cv_week1", "cv_week2"]
    assert predicted.loc[103].isna().all() and predicted.drop(index=103).notna().all().all()
    np.testing.assert_allclose(predicted.drop(index=103).sum(axis=1), 1.0)


def test_cv_fold_bundles_predict_like_apply_blend_model():
    weekly = weekly_table().dropna(subset=["diff_m_week1"])
    _, coefs = legacy.compute_cv_global(FORMULA, weekly, [2018, 2019], training_years=range(2010, 2020),
                                        save_coefs=True, required_classes=BINS)
    test = weekly_table(seed=2, years=[2021])

    for bundle in coefs.values():
        pd.testing.assert_frame_equal(FittedModel.from_legacy_bundle(bundle).predict(test),
                                      _legacy_apply(bundle, test), check_exact=True)


def test_bundles_without_scaler_or_formula():
    _, model = fitted()
    coefs = model.to_legacy_bundle()["coefs"]
    bundle = {"coefs": coefs, "scaler": None, "features": list(coefs["feature"].unique())}
    test = weekly_table(seed=3, years=[2021])

    with pytest.raises(ValueError, match="no saved formula"):
        FittedModel.from_legacy_bundle(bundle)
    unscaled = FittedModel.from_legacy_bundle(bundle, formula=FORMULA)

    assert unscaled.scaler_mean is None and unscaled.to_arrays()[0]["n_samples"] is None
    pd.testing.assert_frame_equal(unscaled.predict(test), _legacy_apply({**bundle, "formula": FORMULA}, test),
                                  check_exact=True)


def test_missing_coefficients_count_as_zero_with_a_warning():
    _, model = fitted()
    bundle = model.to_legacy_bundle()
    coefs = bundle["coefs"]
    bundle["coefs"] = coefs[~((coefs["class"] == "week1") & (coefs["feature"] == "diff_m_week1"))]

    with pytest.warns(UserWarning, match="Feature 'diff_m_week1' not found in coefs for class 'week1'"):
        partial = FittedModel.from_legacy_bundle(bundle)

    assert partial.coef[partial.classes.index("week1"), partial.features.index("diff_m_week1")] == 0.0


def test_application_data_must_have_the_saved_features():
    _, model = fitted()
    test = weekly_table(seed=4, years=[2021])

    other_terms = dataclasses.replace(
        model, formula="outcome ~ prob_clim_week1 + prob_clim_week2 + diff_m_week1 + prob_clim_week2:diff_m_week1")
    with pytest.raises(ValueError, match="does not match the saved training schema"):
        other_terms.predict(test)
    with pytest.raises(ValueError, match="no finite values"):
        model.predict(test.assign(diff_m_week1=np.nan))


def test_models_are_immutable():
    _, model = fitted()

    with pytest.raises(ValueError):
        model.coef[0, 0] = 1.0
    with pytest.raises(AttributeError):
        model.formula = "outcome ~ x"


def test_fold_models_round_trip_through_one_dataset():
    weekly = weekly_table().dropna(subset=["diff_m_week1"])
    plan = design.plan(FORMULA, FrameSamples(weekly).column_dtypes())
    folds, _ = logit.global_cv(plan, FrameSamples(weekly).take, np.arange(len(weekly)), np.ones(len(weekly), bool),
                               weekly["year"].to_numpy(), weekly["outcome"].to_numpy(dtype=object),
                               holdout_years=[2017, 2018], training_years=range(2010, 2019))
    models = {year: model for year, _, _, model in folds}

    again = logit.fold_models(logit.fold_coefficients(models))

    assert set(again) == {2017, 2018}
    for year, model in models.items():
        assert pickle.dumps(again[year].to_legacy_bundle()) == pickle.dumps(model.to_legacy_bundle())


# --- cross-validation against the legacy functions --------------------------------------------------------------

CV_FORMULA = "outcome ~ x1 + x2 + x1:x2"


def cv_table(seed=0, n_cells=8, years=range(2001, 2009), per=6):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([{"id": f"c{c}", "year": y, "issue": k, "cluster": c % 3}
                          for c in range(n_cells) for y in years for k in range(per)])
    frame["x1"] = rng.normal(size=len(frame))
    frame["x2"] = rng.normal(size=len(frame))
    score = np.column_stack([frame["x1"], frame["x2"], rng.normal(size=len(frame))])
    frame["outcome"] = [BINS[k] for k in (score + rng.gumbel(size=score.shape)).argmax(axis=1)]
    frame.loc[rng.random(len(frame)) < 0.05, "outcome"] = np.nan
    return frame


def probabilities_frame(folds, n):
    """The folds' probabilities as the legacy cv_ columns, by sample position."""
    out = pd.DataFrame(np.nan, index=range(n), columns=[f"cv_{b}" for b in sorted(BINS)])
    for _, test, probabilities, model in folds:
        out.loc[test, [f"cv_{c}" for c in model.classes]] = probabilities
    return out


@pytest.mark.parametrize("workers", [1, 3])
def test_global_cv_is_compute_cv_global(workers):
    frame = cv_table(seed=1)
    holdout, training, true_holdout = [2002, 2005, 2006], list(range(2001, 2008)), (2008,)
    samples = FrameSamples(frame)
    plan = design.plan(CV_FORMULA, samples.column_dtypes())
    trainable = np.isin(frame["id"], ["c0", "c1", "c2", "c4", "c6"])

    folds, final = logit.global_cv(
        plan, samples.take, np.arange(len(frame)), trainable, frame["year"].to_numpy(),
        frame["outcome"].to_numpy(dtype=object), holdout_years=holdout, training_years=training,
        true_holdout_years=true_holdout, required_classes=BINS, final_years=training, workers=workers)
    expected, coefs = legacy.compute_cv_global(
        CV_FORMULA, frame[trainable], holdout, training, true_holdout, data_pred=frame, save_coefs=True,
        required_classes=BINS)

    positions = np.concatenate([test for _, test, _, _ in folds])
    mine = probabilities_frame(folds, len(frame)).loc[positions].reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, expected[list(mine.columns)], check_exact=True)
    for year, _, _, model in folds:
        pd.testing.assert_frame_equal(model.to_legacy_bundle()["coefs"], coefs[year]["coefs"].drop(columns="test_year"),
                                      check_exact=True)
    legacy_final = _legacy_fit_final_model(CV_FORMULA, frame[trainable & frame["year"].isin(training)])
    clf, scaler, used, rows = final
    bundle = FittedModel.from_estimators(CV_FORMULA, plan.columns, clf, scaler).to_legacy_bundle()
    pd.testing.assert_frame_equal(bundle["coefs"], legacy_final["coefs"], check_exact=True)


def test_global_cv_without_a_design_plan_is_compute_cv_global():
    frame = cv_table(seed=2)
    formula = "outcome ~ np.log(x1 ** 2 + 1) + x2"
    samples = FrameSamples(frame)
    assert design.plan(formula, samples.column_dtypes()) is None
    holdout, training = [2003, 2004], list(range(2001, 2009))

    folds = logit.global_cv_frames(formula, samples, np.arange(len(frame)), np.ones(len(frame), bool),
                                   holdout_years=holdout, training_years=training, required_classes=BINS)
    expected, _ = legacy.compute_cv_global(formula, frame, holdout, training, required_classes=BINS)

    positions = np.concatenate([test for _, test, _, _ in folds])
    mine = probabilities_frame(folds, len(frame)).loc[positions].reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, expected[list(mine.columns)], check_exact=True)


def sequence_frame(entries, n):
    parts = []
    for test, probabilities, classes in entries:
        part = pd.DataFrame(probabilities, columns=[f"cv_{c}" for c in classes], index=test)
        parts.append(part.reindex(columns=[f"cv_{b}" for b in sorted(BINS)]))
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("plan", [True, False])
def test_local_cv_is_compute_cv_local(plan, monkeypatch):
    if not plan:
        monkeypatch.setattr(design, "plan", lambda formula, dtypes: None)
    frame = cv_table(seed=3, n_cells=4, per=10)
    holdout = [2003, 2006, 2020]

    entries = logit.local_cv(CV_FORMULA, FrameSamples(frame), np.arange(len(frame)), holdout)
    expected = legacy.compute_cv_local(CV_FORMULA, frame, holdout)

    mine = sequence_frame(entries, len(frame))
    pd.testing.assert_frame_equal(mine, expected[list(mine.columns)], check_exact=True)


@pytest.mark.parametrize("plan", [True, False])
def test_cluster_cv_is_compute_cv_clusters(plan, monkeypatch):
    if not plan:
        monkeypatch.setattr(design, "plan", lambda formula, dtypes: None)
    frame = cv_table(seed=4)
    holdout = [2002, 2005]

    entries = logit.cluster_cv(CV_FORMULA, FrameSamples(frame), np.arange(len(frame)), holdout, "cluster")
    expected = legacy.compute_cv_clusters(CV_FORMULA, frame, holdout, "cluster")

    mine = sequence_frame(entries, len(frame))
    pd.testing.assert_frame_equal(mine, expected[list(mine.columns)], check_exact=True)


def test_global_cv_checks_its_folds():
    frame = cv_table(seed=5)
    samples = FrameSamples(frame)
    plan = design.plan(CV_FORMULA, samples.column_dtypes())
    args = (plan, samples.take, np.arange(len(frame)), np.ones(len(frame), bool), frame["year"].to_numpy(),
            frame["outcome"].to_numpy(dtype=object))

    with pytest.raises(ValueError, match="has no test rows"):
        logit.global_cv(*args, holdout_years=[1990], training_years=range(2001, 2009))
    with pytest.raises(ValueError, match="missing outcome classes: nope"):
        logit.global_cv(*args, holdout_years=[2003], training_years=range(2001, 2009), required_classes=["nope"])
