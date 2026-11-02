"""momp.blend.evaluation: external CV predictions load and score like an internal model's.

Ported from onset_blending's tests/test_external_predictions.py: the legacy scoring (oracle onset_blending@1e6d696,
tests/blend/_oracle) of the same tables is the reference for the arrays path. An entry without a file is a recipe
(config) error, checked with the recipe; the relative-file rule is `evaluate`'s.
"""
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from _blend_configs import blend_config, dissemination_ids, recipe_of

from momp.blend import evaluation
from momp.blend.predictions import Predictions
from momp.metrics import probabilistic as metrics
from tests.blend._oracle.blend_evaluation_utils import score_model_predictions
from tests.blend._oracle.misc import interval_bins

N_BINS = 4
LABELS = interval_bins(N_BINS)
CV_COLS = [f"cv_{b}" for b in LABELS]


def make_cv_preds(seed=0, n_cells=6, years=(2001, 2002, 2003), issues_per_year=4):
    rng = np.random.default_rng(seed)
    rows = [{"id": f"{10 + c}.0_{70 + c}.0", "year": year, "issue": k}
            for c in range(n_cells) for year in years for k in range(issues_per_year)]
    frame = pd.DataFrame(rows)
    frame["outcome"] = rng.choice(LABELS, size=len(frame))
    probs = rng.dirichlet(np.ones(len(CV_COLS)), size=len(frame))
    for j, col in enumerate(CV_COLS):
        frame[col] = probs[:, j]
    frame["p_day_1"] = rng.random(len(frame))
    return frame


def write_pickle(path, obj):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def allowed(frame):
    return pd.DataFrame({"id": sorted(frame["id"].unique())[:-1]})


def score(preds: Predictions, model, method, allowed_ids=None):
    """What evaluate's scoring does with predictions: (per-unit metrics, yearly rows)."""
    keep = None if allowed_ids is None else np.isin(preds.ids, list(allowed_ids))
    cells = metrics.cell_metrics(preds.probabilities, preds.outcome, preds.ids, preds.labels, allowed=keep,
                                 earlier=preds.earlier)
    cells["model"], cells["cv_method"] = model, method
    return cells, metrics.yearly_metrics(preds.probabilities, preds.outcome, preds.year, model, method,
                                         auc_method="sklearn")


def test_external_predictions_score_like_an_internal_model(tmp_path):
    cv_preds = make_cv_preds()
    write_pickle(tmp_path / "cv_preds_forest.pkl", cv_preds)

    loaded = evaluation.load_external_predictions("forest", str(tmp_path / "cv_preds_forest.pkl"), LABELS)
    external_cells, external_yearly = score(loaded, "forest", "global", allowed(cv_preds)["id"])
    internal_cells, internal_yearly = score(Predictions.from_frame(cv_preds, LABELS), "forest", "global",
                                            allowed(cv_preds)["id"])
    legacy_cells, legacy_yearly = score_model_predictions(cv_preds, "forest", "global", allowed(cv_preds), N_BINS)

    pd.testing.assert_frame_equal(external_cells, internal_cells)
    assert external_yearly == internal_yearly
    pd.testing.assert_frame_equal(external_cells[legacy_cells.columns], legacy_cells, check_exact=True)
    assert external_yearly == legacy_yearly
    assert set(external_cells["model"]) == {"forest"}
    assert {row["year"] for row in external_yearly} == {2001, 2002, 2003}


def test_perfect_predictions_score_zero_brier_and_rps():
    cv_preds = make_cv_preds()
    for col in CV_COLS:
        cv_preds[col] = (cv_preds["outcome"] == col[len("cv_"):]).astype(float)

    cells, yearly = score(Predictions.from_frame(cv_preds, LABELS), "oracle", "global")

    pooled = cells[cells["id"] == "ALL"].iloc[0]
    assert pooled["brier"] == 0.0 and pooled["rps"] == 0.0
    assert all(row["brier"] == 0.0 and row["rps"] == 0.0 for row in yearly)


def test_cells_outside_the_allowed_set_are_not_scored():
    cv_preds = make_cv_preds()
    allowed_cells = allowed(cv_preds)

    cells, _ = score(Predictions.from_frame(cv_preds, LABELS), "forest", "global", allowed_cells["id"])

    assert set(cells["id"]) - {"ALL"} == set(allowed_cells["id"])


def test_missing_cv_column_is_reported_by_name(tmp_path):
    write_pickle(tmp_path / "preds.pkl", make_cv_preds().drop(columns=["cv_later"]))

    with pytest.raises(ValueError, match=r"forest.*missing columns \['cv_later'\]"):
        evaluation.load_external_predictions("forest", str(tmp_path / "preds.pkl"), LABELS)


def test_week_bins_beyond_the_configured_count_are_rejected(tmp_path):
    write_pickle(tmp_path / "preds.pkl", make_cv_preds().assign(cv_week5=0.0))

    with pytest.raises(ValueError, match="cv_week5"):
        evaluation.load_external_predictions("forest", str(tmp_path / "preds.pkl"), LABELS)


def test_non_dataframe_pickle_is_rejected(tmp_path):
    write_pickle(tmp_path / "preds.pkl", {"cv_week1": [0.1]})

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        evaluation.load_external_predictions("forest", str(tmp_path / "preds.pkl"), LABELS)


def test_netcdf_predictions_must_have_the_outcome_bins(tmp_path):
    preds = Predictions.from_frame(make_cv_preds(), LABELS)
    preds.to_netcdf(tmp_path / "preds.nc")

    again = evaluation.load_external_predictions("forest", str(tmp_path / "preds.nc"), LABELS)
    assert again.labels == LABELS
    with pytest.raises(ValueError, match="bins"):
        evaluation.load_external_predictions("forest", str(tmp_path / "preds.nc"), LABELS[:-1])


def weekly(n_cells=3):
    rng = np.random.default_rng(0)
    rows = [{"id": f"u{c}", "time": f"{y}-06-01", "year": y, "outcome": rng.choice(["week1", "week2", "later"]),
             "prob_clim_unc_earlier": -2.0, "prob_clim_unc_week1": -1.0, "prob_clim_unc_week2": -1.0,
             "prob_clim_week1": -1.0, "prob_clim_week2": -1.0, "diff_fc_week1": 0.0, "diff_fc_week2": 0.0}
            for c in range(n_cells) for y in range(2016, 2024)]
    return pd.DataFrame(rows)


def external_config(file, method="global"):
    return blend_config({
        "years": {"training": list(range(2016, 2024)), "cv_holdout": [2022], "true_holdout": [2023]},
        "formulas": {"b": "outcome ~ prob_clim_qx"}, "cv_methods": [],
        "external_predictions": [{"name": "forest", "file": file, "method": method}]})


def test_a_relative_file_resolves_against_results_dir_and_an_absolute_one_is_used_as_is(tmp_path):
    frame = weekly()
    external = frame[["id", "year", "outcome"]].copy()
    for j, col in enumerate(["cv_week1", "cv_week2", "cv_later"]):
        external[col] = [1 / 3] * len(external)
    (tmp_path / "sub").mkdir()
    write_pickle(tmp_path / "sub" / "preds.pkl", external)
    ids = dissemination_ids(pd.DataFrame({"adm3_name": sorted(frame["id"].unique())}))

    relative = evaluation.evaluate(frame, recipe_of(external_config("sub/preds.pkl")), ids, results_dir=tmp_path)
    absolute = evaluation.evaluate(frame, recipe_of(external_config(str(tmp_path / "sub" / "preds.pkl"), "local")),
                                   ids)

    assert set(relative.cell_metrics.loc[relative.cell_metrics["model"] == "forest", "cv_method"]) == {"global"}
    assert set(absolute.cell_metrics.loc[absolute.cell_metrics["model"] == "forest", "cv_method"]) == {"local"}
    assert os.path.isabs(str(tmp_path / "sub" / "preds.pkl"))


def test_no_external_predictions_configured():
    assert evaluation._load_external_predictions((), "/results", LABELS) == []
