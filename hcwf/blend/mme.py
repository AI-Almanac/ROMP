"""The multi-model ensemble: a weighted mixture of other predictions, with weights chosen to minimise the RPS.

The sources are the climatology baselines and calibrated forecasts (`hcwf.blend.predictions.Predictions`); the
weights minimise the pooled RPS in the dissemination units.

The weights start from the best point of a coarse simplex grid (step 0.1) and are refined by BFGS in softmax space,
as the legacy ``optimize_mme_weights`` (and R's Section 4) did. Sources are aligned on their samples' keys (id and
issue date), so any number of them mix.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import minimize

from hcwf.metrics import probabilistic as metrics

from .predictions import Predictions

__all__ = []


def align(sources: list[Predictions]) -> tuple[list[np.ndarray], np.ndarray]:
    """Return each source's probabilities on the samples every source has, in the first source's order.

    Samples are matched on ``(id, issue date)``; the issue date is the ``time`` coordinate truncated to days, or 0
    for a source without one.

    Parameters
    ----------
    sources : list of hcwf.blend.predictions.Predictions
        The sources to align; the first one sets the order.

    Returns
    -------
    aligned : list of numpy.ndarray of float64, shape (n, bin)
        The probabilities of each source on the ``n`` common samples.
    positions : numpy.ndarray of int64, shape (n,)
        The positions of the common samples in the first source.
    """
    def keys(p: Predictions):
        """Return the ``(id, day)`` key of each sample of `p`."""
        times = p.data["time"].values if "time" in p.data.coords else np.zeros(len(p))
        return list(zip(p.ids.tolist(), np.asarray(times).astype("datetime64[D]").tolist()))

    first = keys(sources[0])
    positions = []
    for source in sources[1:]:
        index = {key: i for i, key in enumerate(keys(source))}
        positions.append(index)
    common = [i for i, key in enumerate(first) if all(key in index for index in positions)]
    common = np.asarray(common, dtype=np.int64)
    aligned = [sources[0].probabilities[common]]
    for source, index in zip(sources[1:], positions):
        aligned.append(source.probabilities[[index[first[i]] for i in common]])
    return aligned, common


def mix(weights: np.ndarray, probabilities: list[np.ndarray]) -> np.ndarray:
    """Return the weighted mixture of the sources' probabilities.

    Parameters
    ----------
    weights : numpy.ndarray of float64, shape (source,)
        The weight of each source.
    probabilities : list of numpy.ndarray of float64, shape (sample, bin)
        The probabilities of each source; each is cleaned (`hcwf.metrics.probabilistic.clean`) before mixing.

    Returns
    -------
    numpy.ndarray of float64, shape (sample, bin)
        The mixture, cleaned again.
    """
    return metrics.clean(sum(w * metrics.clean(P) for w, P in zip(weights, probabilities)))


def mixed_rps(weights: np.ndarray, cleaned: list[np.ndarray], Y: np.ndarray) -> float:
    """Return the pooled RPS of the weighted mixture of already cleaned probabilities.

    Parameters
    ----------
    weights : numpy.ndarray of float64, shape (source,)
        The weight of each source.
    cleaned : list of numpy.ndarray of float64, shape (sample, bin)
        The cleaned probabilities of each source.
    Y : numpy.ndarray of float64, shape (sample, bin)
        The observed outcome, one-hot over the bins.

    Returns
    -------
    float
        The RPS of the cleaned mixture (`hcwf.metrics.probabilistic.rps`).
    """
    return metrics.rps(metrics.clean(sum(w * P for w, P in zip(weights, cleaned))), Y)


def optimise_weights(cleaned: list[np.ndarray], Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the weights that minimise the pooled RPS of the mixture.

    The best point of a simplex grid of step 0.1 (the first grid point if none has a finite RPS) is the start of a
    BFGS minimisation (at most 200 iterations) in softmax space.

    Parameters
    ----------
    cleaned : list of numpy.ndarray of float64, shape (sample, bin)
        The cleaned probabilities of each source.
    Y : numpy.ndarray of float64, shape (sample, bin)
        The observed outcome, one-hot over the bins.

    Returns
    -------
    weights : numpy.ndarray of float64, shape (source,)
        The weights, non-negative and summing to 1.
    rps : float
        The pooled RPS of the mixture with these weights.
    """
    best_rps, best = np.inf, None
    for w in _simplex_grid(len(cleaned), step=0.1):
        value = mixed_rps(w, cleaned, Y)
        if np.isfinite(value) and value < best_rps:
            best_rps, best = value, w
    if best is None:
        best = _simplex_grid(len(cleaned), step=0.1)[0]
    w0 = np.maximum(best, 1e-6)
    w0 = w0 / w0.sum()
    result = minimize(
        fun=lambda theta: mixed_rps(_softmax(theta), cleaned, Y), x0=np.log(w0), method="BFGS",
        options={"maxiter": 200},
    )
    weights = _softmax(result.x)
    return weights, mixed_rps(weights, cleaned, Y)


def _simplex_grid(n: int, step: float = 0.1) -> np.ndarray:
    """Return the points of the probability simplex on a regular grid.

    Parameters
    ----------
    n : int
        The number of weights.
    step : float, default 0.1
        The grid step of the first ``n - 1`` weights; the last takes the rest.

    Returns
    -------
    numpy.ndarray of float64, shape (point, n)
        Every combination of weights that are multiples of `step` and sum to 1.
    """
    if n == 1:
        return np.array([[1.0]])
    values = np.arange(0, 1 + step / 2, step)
    grid = []
    for combo in product(*([values] * (n - 1))):
        total = sum(combo)
        if total <= 1.0 + 1e-9:
            grid.append(list(combo) + [1.0 - total])
    return np.array(grid)


def _softmax(theta: np.ndarray) -> np.ndarray:
    """Return the softmax of `theta`.

    Parameters
    ----------
    theta : numpy.ndarray of float64, shape (n,)
        The unconstrained values.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        Positive weights that sum to 1.
    """
    e = np.exp(theta - np.max(theta))
    return e / e.sum()
