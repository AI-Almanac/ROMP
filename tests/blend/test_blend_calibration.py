"""momp.blend.calibration: Platt calibration as platt_cv_multibin and fit_platt_weights_export (oracle:
onset_blending@1e6d696) computed it, serially and in worker processes."""

import numpy as np
import pandas as pd
import pytest

from momp.blend import calibration
from tests.blend._oracle import blend_evaluation_utils as legacy

BINS = ["week1", "week2", "later"]


def table(seed=0, n_cells=8, years=range(2001, 2009), per=6):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([{"id": f"c{c}", "year": y, "issue": k} for c in range(n_cells) for y in years for k in range(per)])
    probs = rng.dirichlet(np.ones(len(BINS)), size=len(frame))
    for j, b in enumerate(BINS):
        frame[b] = probs[:, j]
    frame["outcome"] = rng.choice(BINS, size=len(frame))
    frame.loc[rng.random(len(frame)) < 0.05, "outcome"] = np.nan
    return frame


def arrays(frame):
    codes = frame["outcome"].map({b: k for k, b in enumerate(BINS)}).fillna(-1).astype(int).to_numpy()
    return frame[BINS].to_numpy(), codes


@pytest.mark.parametrize("allowed", [None, ["c0", "c2", "c3", "c5"]])
def test_cross_validated_is_platt_cv_multibin(allowed):
    frame = table()
    holdout, true_holdout = [2003, 2004, 2007, 1999], (2008,)
    P, outcome = arrays(frame)
    trainable = ~frame["year"].isin(true_holdout).to_numpy()
    if allowed is not None:
        trainable &= frame["id"].isin(allowed).to_numpy()

    folds = calibration.cross_validated(P, outcome, frame["year"].to_numpy(), trainable, holdout)
    expected = legacy.platt_cv_multibin(frame, BINS, holdout, true_holdout,
                                        allowed_cells=None if allowed is None else pd.DataFrame({"id": allowed}))

    assert [year for year, _, _ in folds] == [2003, 2004, 2007]
    mine = np.concatenate([p for _, _, p in folds])
    np.testing.assert_array_equal(mine, expected[[f"cv_{b}" for b in BINS]].to_numpy())
    np.testing.assert_array_equal(np.concatenate([t for _, t, _ in folds]),
                                  np.concatenate([np.flatnonzero(frame["year"] == y) for y in [2003, 2004, 2007]]))


def test_worker_processes_calibrate_the_same():
    frame = table(seed=1)
    holdout = [2002, 2005, 2006]
    P, outcome = arrays(frame)
    trainable = np.ones(len(frame), dtype=bool)
    args = (P, outcome, frame["year"].to_numpy(), trainable, holdout)

    serial = calibration.cross_validated(*args)
    pooled = calibration.cross_validated(*args, workers=2)

    for (y1, t1, p1), (y2, t2, p2) in zip(serial, pooled):
        assert y1 == y2
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(p1, p2)


def test_export_weights_are_fit_platt_weights_export():
    frame = table(seed=2)
    training = list(range(2001, 2007))
    train = frame[frame["year"].isin(training)]
    P, outcome = arrays(train)

    weights = calibration.export_weights(P, outcome, BINS)

    expected = legacy.fit_platt_weights_export(frame, BINS, training)["weights_df"]
    pd.testing.assert_frame_equal(weights, expected, check_exact=True)


def test_a_bin_with_too_few_samples_is_not_calibrated():
    frame = table(seed=3).iloc[:8]
    P, outcome = arrays(frame)

    weights = calibration.export_weights(P, outcome, BINS)

    assert weights["intercept"].tolist() == [0.0] * 3 and weights["slope"].tolist() == [1.0] * 3
