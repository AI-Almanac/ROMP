"""hcwf.blend.mme: the multi-model ensemble's weights are optimize_mme_weights' (oracle: onset_blending@1e6d696)."""

import numpy as np
import pandas as pd
import pytest

from hcwf.blend import mme
from hcwf.blend.predictions import Predictions
from tests.blend._oracle import blend_evaluation_utils as legacy

LABELS = ["week1", "week2", "later"]


@pytest.mark.parametrize("n_sources", [1, 2, 3])
def test_weights_are_the_legacy_optimisers(n_sources):
    rng = np.random.default_rng(5)
    sources = [rng.dirichlet(np.ones(3), size=300) for _ in range(n_sources)]
    sources[-1][:5] = np.nan
    Y = np.eye(3)[rng.integers(0, 3, size=300)]
    cleaned = [legacy.clean_probs5(P) for P in sources]

    weights, rps = mme.optimise_weights(cleaned, Y)

    expected = legacy.optimize_mme_weights(cleaned, Y, [f"w_{i}" for i in range(n_sources)])
    np.testing.assert_array_equal(weights, expected["weights"])
    assert rps == expected["rps"]


def test_the_simplex_grid_is_the_legacy_one():
    for n in (1, 2, 4):
        np.testing.assert_array_equal(mme._simplex_grid(n), legacy.simplex_grid(n))


def predictions(ids, times, probabilities):
    frame = pd.DataFrame({"id": ids, "time": pd.to_datetime(times), "year": 2001, "outcome": "week1"})
    for j, label in enumerate(LABELS):
        frame[f"cv_{label}"] = probabilities[:, j]
    return Predictions.from_frame(frame, LABELS)


def test_sources_align_on_their_samples_keys():
    rng = np.random.default_rng(1)
    P = [rng.dirichlet(np.ones(3), size=4) for _ in range(2)]
    first = predictions(["a", "a", "b", "b"], ["2001-06-01", "2001-06-08"] * 2, P[0])
    second = predictions(["b", "a", "c"], ["2001-06-08", "2001-06-08", "2001-06-08"], P[1][:3])

    aligned, positions = mme.align([first, second])

    assert positions.tolist() == [1, 3]
    np.testing.assert_array_equal(aligned[0], P[0][[1, 3]])
    np.testing.assert_array_equal(aligned[1], P[1][[1, 0]])


def test_a_mixture_is_cleaned_and_sums_to_one():
    rng = np.random.default_rng(2)
    P = [rng.dirichlet(np.ones(3), size=10) for _ in range(2)]
    P[0][3] = np.nan

    mixed = mme.mix(np.array([0.3, 0.7]), P)

    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)
    np.testing.assert_array_equal(mixed, legacy.clean_probs5(0.3 * legacy.clean_probs5(P[0]) + 0.7 * legacy.clean_probs5(P[1])))
