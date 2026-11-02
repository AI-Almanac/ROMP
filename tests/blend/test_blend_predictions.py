"""momp.blend.predictions: probabilities with their samples' keys, to and from the legacy cv table and NetCDF."""

import numpy as np
import pandas as pd
import pytest

from momp.blend.predictions import Predictions, outcome_codes, outcome_series
from tests.blend._samples import FrameSamples

LABELS = ["week1", "week2", "later"]


def cv_frame(n=30, seed=0, earlier=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "id": [f"u{i % 3}" for i in range(n)],
        "time": pd.date_range("2001-06-01", periods=n, freq="7D"),
        "year": 2001 + np.arange(n) % 4,
        "lat": np.repeat([10.0, 11.0, 12.0], n // 3), "lon": 70.0,
        "outcome": pd.Series(rng.choice([*LABELS, ""], size=n)).replace("", np.nan),
    })
    P = rng.dirichlet(np.ones(len(LABELS)), size=n)
    for j, label in enumerate(LABELS):
        frame[f"cv_{label}"] = P[:, j]
    if earlier:
        frame["cv_earlier"] = rng.random(n)
    return frame


def test_outcome_codes_and_series_are_inverse_for_known_labels():
    values = np.array(["week2", None, "later", np.nan, "other"], dtype=object)

    codes = outcome_codes(values, LABELS)

    assert codes.tolist() == [1, -1, 2, -1, 3]
    series = outcome_series(codes, LABELS)
    assert series[[0, 2]].tolist() == ["week2", "later"] and series[[1, 3, 4]].isna().all()


@pytest.mark.parametrize("earlier", [False, True])
def test_the_legacy_table_round_trips(earlier):
    frame = cv_frame(earlier=earlier)

    again = Predictions.from_frame(frame, LABELS).frame()

    assert list(again.columns) == ["id", "time", "year", "lat", "lon", "outcome", *[f"cv_{b}" for b in LABELS],
                                   *(["cv_earlier"] if earlier else [])]
    np.testing.assert_array_equal(again[[f"cv_{b}" for b in LABELS]].to_numpy(), frame[[f"cv_{b}" for b in LABELS]].to_numpy())
    pd.testing.assert_series_equal(again["outcome"], frame["outcome"].astype(object), check_names=False, check_dtype=False)
    assert (pd.to_datetime(again["time"]).to_numpy() == frame["time"].to_numpy()).all()


def test_samples_of_a_weekly_table_become_predictions():
    frame = cv_frame().assign(outcome=lambda f: f["outcome"])
    samples = FrameSamples(frame)
    rows = np.array([4, 1, 7])
    P = np.random.default_rng(1).dirichlet(np.ones(3), size=3)
    daily = np.random.default_rng(2).dirichlet(np.ones(29), size=3)

    predictions = Predictions.of_samples(samples, rows, P, LABELS, daily=daily)

    np.testing.assert_array_equal(predictions.probabilities, P)
    assert predictions.ids.tolist() == ["u1", "u1", "u1"]
    np.testing.assert_array_equal(predictions.outcome, outcome_codes(frame["outcome"].to_numpy(dtype=object)[rows], LABELS))
    table = predictions.frame()
    assert {"p_day_1", "p_day_28", "p_later"} <= set(table.columns)


def test_netcdf_round_trip(tmp_path):
    predictions = Predictions.from_frame(cv_frame(earlier=True), LABELS)

    predictions.to_netcdf(tmp_path / "p.nc")
    again = Predictions.open(tmp_path / "p.nc")

    assert again.labels == LABELS
    np.testing.assert_array_equal(again.probabilities, predictions.probabilities)
    np.testing.assert_array_equal(again.earlier, predictions.earlier)
    np.testing.assert_array_equal(again.outcome, predictions.outcome)
    pd.testing.assert_frame_equal(again.frame(), predictions.frame())


def test_take_concat_and_with_probabilities():
    predictions = Predictions.from_frame(cv_frame(), LABELS)
    first, second = predictions.take(np.arange(10)), predictions.take(np.arange(10, 30))

    joined = Predictions.concat([first, second])
    other = joined.with_probabilities(np.full((30, 3), 1 / 3))

    pd.testing.assert_frame_equal(joined.frame(), predictions.frame())
    assert len(other) == 30 and other.labels == LABELS and (other.probabilities == 1 / 3).all()
    np.testing.assert_array_equal(other.outcome, predictions.outcome)
