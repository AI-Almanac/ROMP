"""The faster evaluation paths give exactly what the code they replaced gave.

The parts of onset_blending's tests/test_blend_evaluation_equivalence.py that concern the evaluation: Platt calibration
on arrays against the original that filtered and copied the table per fold, and global CV in worker processes against
the serial folds (here through `evaluate`). The legacy `_fit_columns` is a table notion with no counterpart (a formula
without a design plan builds its Patsy design per fit: tests/blend/test_blend_logit.py).
"""
import numpy as np
import pandas as pd
import pytest
from _blend_configs import blend_config, recipe_of
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from momp.blend import calibration
from momp.blend.evaluation import evaluate

BINS = ["week1", "week2", "later"]


def _original_platt_cv_multibin(df, prob_cols, holdout_years, true_holdout_years=(),
                                outcome_col="outcome", year_col="year", cv_prefix="cv_", allowed_cells=None):
    """platt_cv_multibin as it was, filtering and copying the table per fold."""
    bins = prob_cols
    out_list = []
    for test_year in holdout_years:
        train = df[(df[year_col] != test_year) & (~df[year_col].isin(true_holdout_years))].copy()
        test = df[df[year_col] == test_year].copy()
        if test.empty:
            continue
        if allowed_cells is not None:
            allowed_ids = set(allowed_cells["id"])
            train = train[train["id"].isin(allowed_ids)]
        cal_mat = np.full((len(test), len(bins)), np.nan)
        for j, b in enumerate(bins):
            p_tr = np.clip(train[b].values.astype(float), 1e-6, 1 - 1e-6)
            outcome = train[outcome_col]
            y_tr = ((outcome == b).astype(float).where(outcome.notna(), np.nan).to_numpy())
            ok = np.isfinite(p_tr) & ~np.isnan(y_tr)
            if ok.sum() < 10 or len(np.unique(y_tr[ok])) < 2:
                cal_mat[:, j] = test[b].values.astype(float)
                continue
            X = p_tr[ok].reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
            clf.fit(X, y_tr[ok])
            test_p = np.clip(test[b].values.astype(float), 1e-6, 1 - 1e-6).reshape(-1, 1)
            cal_mat[:, j] = expit(clf.intercept_[0] + clf.coef_[0][0] * test_p.flatten())
        rs = cal_mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        cal_mat /= rs
        test_out = test.copy()
        for j, b in enumerate(bins):
            test_out[f"{cv_prefix}{b}"] = cal_mat[:, j]
        out_list.append(test_out)
    if out_list:
        return pd.concat(out_list, ignore_index=True)
    return pd.DataFrame()


def _table(seed=0, n_cells=8, years=range(2001, 2009), rows_per_cell_year=6):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([{"id": f"c{cell}", "year": year, "issue": k}
                          for cell in range(n_cells) for year in years for k in range(rows_per_cell_year)])
    probs = rng.dirichlet(np.ones(len(BINS)), size=len(frame))
    for j, b in enumerate(BINS):
        frame[b] = probs[:, j]
    frame["outcome"] = rng.choice(BINS, size=len(frame))
    frame.loc[rng.random(len(frame)) < 0.05, "outcome"] = np.nan
    return frame


@pytest.mark.parametrize("allowed", [None, ["c0", "c2", "c3", "c5"]])
@pytest.mark.parametrize("workers", [1, 2])
def test_platt_cv_matches_filtering_the_table_per_fold(allowed, workers):
    frame = _table()
    holdout, true_holdout = [2003, 2004, 2007], (2008,)
    codes = frame["outcome"].map({b: k for k, b in enumerate(BINS)}).fillna(-1).astype(int).to_numpy()
    year = frame["year"].to_numpy()
    trainable = ~np.isin(year, true_holdout)
    if allowed is not None:
        trainable &= frame["id"].isin(allowed).to_numpy()

    folds = calibration.cross_validated(frame[BINS].to_numpy(), codes, year, trainable, holdout, workers=workers)
    expected = _original_platt_cv_multibin(frame, BINS, holdout, true_holdout,
                                           allowed_cells=None if allowed is None else pd.DataFrame({"id": allowed}))

    np.testing.assert_array_equal(np.concatenate([p for _, _, p in folds]),
                                  expected[[f"cv_{b}" for b in BINS]].to_numpy())


def weekly_table(seed=1, years=range(2016, 2024), n_cells=6, issues=5):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([{"id": f"u{c}", "time": f"{y}-06-{1 + 7 * k:02d}", "year": y}
                          for c in range(n_cells) for y in years for k in range(issues)])
    n = len(frame)
    frame["outcome"] = rng.choice(["week1", "week2", "later"], size=n)
    for prefix in ("prob_clim", "prob_clim_unc"):
        for w in (1, 2):
            frame[f"{prefix}_week{w}"] = rng.normal(-1.0, 0.5, n)
    frame["prob_clim_unc_earlier"] = rng.normal(-2.0, 0.5, n)
    for w in (1, 2):
        frame[f"fc_p_onset_fixed_cutoff_week{w}"] = rng.uniform(0.0, 0.5, n)
        frame[f"diff_fc_week{w}"] = rng.normal(size=n)
    return frame


def test_global_cv_in_worker_processes_matches_the_serial_folds():
    frame = weekly_table()
    recipe = recipe_of(blend_config({
        "cutoff_mode": "fixed_cutoff",
        "years": {"training": list(range(2016, 2023)), "cv_holdout": [2018, 2019, 2021], "true_holdout": [2023]},
        "formulas": {"b": "outcome ~ prob_clim_qx * diff_fc_qx"}, "final_formula": "b",
        "forecasts": [{"model": "fc", "variant": "fixed_cutoff", "platt_weights": True}]}))
    ids = np.asarray(sorted(frame["id"].unique()), dtype=object)

    serial = evaluate(frame, recipe, ids, workers=1, final=True)
    pooled = evaluate(frame, recipe, ids, workers=3, final=True)

    for key, preds in serial.cv_preds.items():
        np.testing.assert_array_equal(pooled.cv_preds[key].probabilities, preds.probabilities)
    for year, fold in serial.coefficients["b", "global"].items():
        pd.testing.assert_frame_equal(pooled.coefficients["b", "global"][year].to_legacy_bundle()["coefs"],
                                      fold.to_legacy_bundle()["coefs"], check_exact=True)
    pd.testing.assert_frame_equal(pooled.cell_metrics, serial.cell_metrics, check_exact=True)
    pd.testing.assert_frame_equal(pooled.final_model.to_legacy_bundle()["coefs"],
                                  serial.final_model.to_legacy_bundle()["coefs"], check_exact=True)
    for stem, weights in serial.platt_weights.items():
        pd.testing.assert_frame_equal(pooled.platt_weights[stem], weights, check_exact=True)
