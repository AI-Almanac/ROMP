"""Probabilistic skill on arrays: Brier score, RPS, AUC and Pietra index of lead-bin probabilities.

One implementation for HCWF's pooled ensemble scores and the blending package's per-unit, pooled and per-year
scores. Everything is array-based: ``P`` holds the probabilities and ``Y`` the 0/1 outcomes of (samples, bins) with
the bins in lead order; ``members`` (the ensemble size behind each probability, a scalar, one per sample or one per
entry) switches on the fair scores, which take out the sampling noise of the probability, ``p(1-p)/(members-1)``.

Conventions are options, never second implementations:

* scale of the Brier score: ``"element"`` averages the squared error over all (sample, bin) entries (HCWF);
  ``"sample"`` sums it over the bins and averages over the samples (blending: m times ``"element"``); ``"pooled"`` is
  ``"sample"`` written as one sum over everything divided by the samples (blending's pooled row; equal to
  ``"sample"`` up to rounding).
* mean of a bin's entries: ``"numpy"`` is ``numpy.mean`` of the column (blending); ``"compensated"`` adds the entries
  one by one with Kahan compensation, as a pandas groupby mean does (HCWF's per-bin scores, to the last bit).
* summation of the RPS: ``"cumulative"`` is ``numpy.cumsum`` over the bins (blending); ``"prefix"`` sums the first
  ``k`` bins for each ``k`` and adds the terms one after the other (HCWF). They differ only in the last bits, and for
  8 bins or more.
* which bins are scored is the columns passed in: HCWF scores its inner day bins only, blending its week bins and
  ``later`` (plus the ``earlier`` column in the RPS, see `rps`).
* AUC counts ties as 1/2, as the Mann-Whitney U statistic and sklearn's ``roc_auc_score`` do.

`hcwf.metrics.score` wraps these functions in HCWF's result layout; the blend package calls them directly.
"""

from __future__ import annotations

import warnings

import numba
import numpy as np
import pandas as pd

BRIER_SCALES = ("element", "sample", "pooled")
"""The accepted ``scale`` values of `brier_scores`."""
BIN_MEANS = ("numpy", "compensated")
"""The accepted ``bin_mean`` values of `brier_scores`."""
RPS_SUMMATIONS = ("cumulative", "prefix")
"""The accepted ``summation`` values of `rps_cases`."""
AUC_METHODS = ("rank", "sklearn")
"""The accepted ``auc_method`` values of `yearly_metrics`."""


def one_hot(outcome: np.ndarray, n_classes: int) -> np.ndarray:
    """Return the indicators of each sample's outcome code.

    Parameters
    ----------
    outcome : numpy.ndarray of int, shape (samples,)
        The outcome code of each sample (-1: none).
    n_classes : int
        The number of classes (columns).

    Returns
    -------
    numpy.ndarray of float64, shape (samples, classes)
        1.0 in the column of a sample's code, 0.0 elsewhere (an all-zero row for -1).
    """
    return np.column_stack([(outcome == k).astype(float) for k in range(n_classes)])


def clean(P: np.ndarray) -> np.ndarray:
    """Return the probabilities with non-finite or negative entries at 0 and rows summing to 1.

    A row with nothing left (sum 0) becomes uniform.

    Parameters
    ----------
    P : numpy.ndarray, shape (samples, bins)
        The probabilities; not changed.

    Returns
    -------
    numpy.ndarray of float64, shape (samples, bins)
        The cleaned copy.
    """
    P = np.array(P, dtype=float)
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= 0).flatten()
    if zero.any():
        P[zero] = 1.0 / P.shape[1]
        rs = P.sum(axis=1, keepdims=True)
    return P / rs


@numba.njit(cache=True)
def _compensated_column_means(values):
    """Compute the mean of each column over its non-NaN entries, added in order with Kahan compensation.

    This is the way pandas' groupby mean adds. Compiled with numba (``njit``): ``values`` must be a C-contiguous
    ``float64`` array.

    Parameters
    ----------
    values : numpy.ndarray of float64, shape (samples, bins)
        The entries.

    Returns
    -------
    numpy.ndarray of float64, shape (bins,)
        The column means, NaN for a column without a non-NaN entry.
    """
    n, m = values.shape
    means = np.empty(m)
    for j in range(m):
        total = 0.0
        compensation = 0.0
        count = 0
        for i in range(n):
            if np.isnan(values[i, j]):
                continue
            count += 1
            y = values[i, j] - compensation
            t = total + y
            compensation = t - total - y
            total = t
        means[j] = total / count if count else np.nan
    return means


def _bin_means(values, how, skipna=False):
    """Return the mean of each column of a (samples, bins) array (see `BIN_MEANS`); ``"compensated"`` always skips NaN.

    Parameters
    ----------
    values : numpy.ndarray of float64, shape (samples, bins)
        The entries.
    how : {"numpy", "compensated"}
        How to take the mean.
    skipna : bool, default False
        Leave NaN out of the ``"numpy"`` mean (``numpy.nanmean``).

    Returns
    -------
    numpy.ndarray of float64, shape (bins,)
        The column means.

    Raises
    ------
    ValueError
        ``how`` is not in `BIN_MEANS`.
    """
    if how not in BIN_MEANS:
        raise ValueError(f"bin_mean must be one of {BIN_MEANS}, not {how!r}")
    if how == "compensated":
        return _compensated_column_means(np.ascontiguousarray(values, dtype=np.float64))
    mean = np.nanmean if skipna else np.mean
    return np.array([mean(values[:, j]) for j in range(values.shape[1])])


def _fair_terms(squares, P, members):
    """Return the squared errors less ``p(1-p)/(members-1)`` entry by entry.

    Parameters
    ----------
    squares : numpy.ndarray of float64, shape (samples, bins)
        The squared errors.
    P : numpy.ndarray of float64, shape (samples, bins)
        The probabilities.
    members : float or numpy.ndarray
        The ensemble size: a scalar, one per sample (shape (samples,)) or one per entry (shape (samples, bins)).

    Returns
    -------
    numpy.ndarray of float64, shape (samples, bins)
        The fair terms; ``inf`` members take nothing off, and a single member gives NaN (or infinity) as the division does.
    """
    members = np.asarray(members, dtype=float)
    if members.ndim == 1 and P.ndim == 2:
        members = members[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):   # one member: no fair term, as the NaN says
        return squares - (P * (1 - P)) / (members - 1)


def brier_scores(P: np.ndarray, Y: np.ndarray, members=None, *, scale: str = "sample", by_bin: bool = False,
                 bin_mean: str = "numpy", skipna: bool = False) -> dict:
    """Compute the Brier score, and with ``members`` the fair Brier score, of probabilities and outcomes.

    The fair score subtracts ``p(1-p)/(members-1)`` from each squared error, so ``members`` must exceed 1 (an infinite
    member count, as of a climatology, takes nothing off). With the ``"sample"`` and ``"pooled"`` scales and one member
    count per sample or for all, the correction is summed over the bins before it is taken off, as blending does.

    Parameters
    ----------
    P : numpy.ndarray of float64, shape (samples, bins)
        The predicted probabilities.
    Y : numpy.ndarray, shape (samples, bins)
        The outcomes (0 or 1).
    members : float or numpy.ndarray, optional
        The ensemble size behind the probabilities (a scalar, one per sample or one per entry); None for no fair score.
    scale : {"element", "sample", "pooled"}, default "sample"
        How the squared errors are averaged (see the module description).
    by_bin : bool, default False
        Also return the score of each bin.
    bin_mean : {"numpy", "compensated"}, default "numpy"
        How the mean of a bin's entries is taken (see the module description).
    skipna : bool, default False
        Leave NaN entries out of the means (HCWF: a case with one member has 0/0 in the fair term when ``p`` is 0 or 1).

    Returns
    -------
    dict
        ``brier`` (float); ``fair_brier`` (only with ``members``); with ``by_bin`` also ``bin_brier`` and ``bin_fair_brier``,
        the mean over the samples of each bin's entries (``float64`` arrays, one value per column).

    Raises
    ------
    ValueError
        ``scale`` is not in `BRIER_SCALES`, or ``bin_mean`` is not in `BIN_MEANS` (with ``by_bin``).
    """
    if scale not in BRIER_SCALES:
        raise ValueError(f"scale must be one of {BRIER_SCALES}, not {scale!r}")
    P, Y = np.asarray(P, dtype=float), np.asarray(Y)
    squares = (P - Y) ** 2
    fair = None if members is None else _fair_terms(squares, P, members)
    if scale == "element":
        mean = np.nanmean if skipna else np.mean
        out = {"brier": mean(squares)}
        if fair is not None:
            out["fair_brier"] = mean(fair)
    else:
        out = {"brier": float(np.mean(np.sum(squares, axis=1)) if scale == "sample" else np.sum(squares) / len(P))}
        if fair is not None:
            if np.ndim(members) < 2:
                rows = np.sum(squares, axis=1) - np.sum(P * (1 - P), axis=1) / (np.asarray(members, dtype=float) - 1)
            else:
                rows = np.sum(fair, axis=1)
            out["fair_brier"] = float(np.mean(rows))
    if by_bin:
        out["bin_brier"] = _bin_means(squares, bin_mean, skipna)
        if fair is not None:
            out["bin_fair_brier"] = _bin_means(fair, bin_mean, skipna)
    return out


def brier(P: np.ndarray, Y: np.ndarray, *, scale: str = "sample") -> float:
    """Compute the Brier score of (samples, bins) probabilities and outcomes.

    Parameters
    ----------
    P : numpy.ndarray of float64, shape (samples, bins)
        The predicted probabilities.
    Y : numpy.ndarray, shape (samples, bins)
        The outcomes (0 or 1).
    scale : {"element", "sample", "pooled"}, default "sample"
        How the squared errors are averaged (see `brier_scores`).

    Returns
    -------
    float
        The Brier score.
    """
    return brier_scores(P, Y, scale=scale)["brier"]


def rps_cases(P: np.ndarray, Y: np.ndarray, members=None, *, summation: str = "cumulative",
              valid: np.ndarray | None = None):
    """Compute the RPS of every sample, and with ``members`` the fair RPS.

    The RPS is the sum over the bins ``k`` of ``(cumulative p - cumulative y)^2``; the fair RPS takes
    ``cum_p(1-cum_p)/(members-1)`` off each term.

    Parameters
    ----------
    P : numpy.ndarray of float64, shape (samples, bins)
        The predicted probabilities.
    Y : numpy.ndarray, shape (samples, bins)
        The outcomes (0 or 1).
    members : float or numpy.ndarray, optional
        The ensemble size behind the probabilities (a scalar, or shape (samples,)); None for no fair RPS.
    summation : {"cumulative", "prefix"}, default "cumulative"
        How the cumulative sums are made (see the module description).
    valid : numpy.ndarray of bool, shape (samples, bins), optional
        False where a sample has no such bin (padding of samples with fewer bins, which must be the last columns); only
        the ``"prefix"`` summation takes it.

    Returns
    -------
    rps : numpy.ndarray of float64, shape (samples,)
        The RPS of each sample.
    fair_rps : numpy.ndarray of float64, shape (samples,) or None
        The fair RPS of each sample; None without ``members``.

    Raises
    ------
    ValueError
        ``summation`` is not in `RPS_SUMMATIONS`, or ``valid`` is given with the ``"cumulative"`` summation.
    """
    if summation not in RPS_SUMMATIONS:
        raise ValueError(f"summation must be one of {RPS_SUMMATIONS}, not {summation!r}")
    P, Y = np.asarray(P, dtype=float), np.asarray(Y)
    denominator = None if members is None else np.asarray(members, dtype=float) - 1
    if summation == "cumulative":
        if valid is not None:
            raise ValueError("valid needs the 'prefix' summation")
        CP, CY = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
        squares = (CP - CY) ** 2
        fair = None if members is None else np.sum(_fair_terms(squares, CP, members), axis=1)
        return np.sum(squares, axis=1), fair
    rps, fair = 0, 0
    for k in range(1, P.shape[1] + 1):
        cum_p = np.sum(P[:, :k], axis=1)
        cum_y = np.sum(Y[:, :k], axis=1)
        term = (cum_y - cum_p) ** 2
        fair_term = None if members is None else term - (cum_p * (1 - cum_p)) / denominator
        if valid is not None:
            term = np.where(valid[:, k - 1], term, 0.0)
            fair_term = None if fair_term is None else np.where(valid[:, k - 1], fair_term, 0.0)
        rps = rps + term
        fair = None if members is None else fair + fair_term
    return np.asarray(rps), None if members is None else np.asarray(fair)


def rps(P: np.ndarray, Y: np.ndarray, *, summation: str = "cumulative") -> float:
    """Compute the mean over the samples of the RPS of probabilities and one-hot outcomes.

    Parameters
    ----------
    P : numpy.ndarray of float64, shape (samples, bins)
        The predicted probabilities.
    Y : numpy.ndarray, shape (samples, bins)
        The one-hot outcomes.
    summation : {"cumulative", "prefix"}, default "cumulative"
        How the cumulative sums are made (see `rps_cases`).

    Returns
    -------
    float
        The mean RPS.
    """
    return float(np.mean(rps_cases(P, Y, summation=summation)[0]))


def auc(y01: np.ndarray, score: np.ndarray) -> float:
    """Compute the AUC by the Wilcoxon rank sum over the finite scores.

    Parameters
    ----------
    y01 : numpy.ndarray, shape (n,)
        The outcomes (0 or 1; NaN entries are left out).
    score : numpy.ndarray of float64, shape (n,)
        The predicted scores; non-finite entries are left out.

    Returns
    -------
    float
        The AUC, NaN when there are no positives or no negatives.
    """
    ok = np.isfinite(score) & ~np.isnan(y01)
    y = y01[ok].astype(int)
    s = score[ok]
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata   # scipy.stats is slow to import and only these two functions need it

    ranks = rankdata(s, method="average")
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)


def grouped_auc(y01: np.ndarray, score: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Compute the `auc` of each group of finite scores and 0/1 outcomes, from one sort of all the scores.

    The ranks within a group, ties averaged, and so the rank sums are those of `auc` on that group's values alone.

    Parameters
    ----------
    y01 : numpy.ndarray, shape (n,)
        The outcomes (0 or 1).
    score : numpy.ndarray of float64, shape (n,)
        The finite predicted scores.
    group : numpy.ndarray of int, shape (n,)
        The group code of each entry, in ``0 .. n_groups - 1``.
    n_groups : int
        The number of groups.

    Returns
    -------
    numpy.ndarray of float64, shape (n_groups,)
        The AUC of each group, NaN for a group without positives or without negatives.
    """
    y01 = np.asarray(y01)
    group = np.asarray(group)
    n = len(score)
    if n == 0:
        return np.full(n_groups, np.nan)
    order = np.argsort(score)    # ties land in any order: their ranks are averaged
    if n_groups > 1:
        by_group = group[order].astype(np.uint16 if n_groups <= 2**16 else np.int64)
        order = order[np.argsort(by_group, kind="stable")]    # radix sort for 16-bit codes
    s, g = score[order], group[order]
    new = np.ones(n, dtype=bool)
    new[1:] = (s[1:] != s[:-1]) | (g[1:] != g[:-1])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    size = np.bincount(group, minlength=n_groups)
    first = np.cumsum(size) - size                       # where each group begins in the sorted order
    average = (starts + ends - 2 * first[g[starts]] + 1) / 2.0    # of the 1-based ranks a run covers
    ranks = np.repeat(average, ends - starts)
    positive = y01[order] == 1
    rank_sum = np.bincount(g[positive], weights=ranks[positive], minlength=n_groups)
    n1 = np.bincount(g[positive], minlength=n_groups).astype(float)
    n0 = size - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def pietra(P: np.ndarray, Y: np.ndarray) -> float:
    """Compute the Pietra index.

    This is the Kolmogorov-Smirnov statistic between the probabilities of the bins that happened and of those that
    did not.

    Parameters
    ----------
    P : numpy.ndarray
        The predicted probabilities, any shape.
    Y : numpy.ndarray
        The outcomes (0 or 1), the shape of ``P``.

    Returns
    -------
    float
        The statistic, NaN when no bin happened or none failed to.
    """
    p = P.flatten()
    y = Y.flatten().astype(int)
    positive, negative = p[y == 1], p[y == 0]
    from scipy.stats import ks_2samp

    if len(positive) and len(negative):
        return float(ks_2samp(positive, negative).statistic)
    return np.nan


def skill_score(score, reference, what: str = "score"):
    """Compute ``1 - score/reference``: positive when the score beats the reference (both scores lower-is-better).

    A zero reference gives NaN, with one warning for the call.

    Parameters
    ----------
    score : float or numpy.ndarray or pandas.Series
        The score being judged.
    reference : float or numpy.ndarray or pandas.Series
        The reference score.
    what : str, default "score"
        What the reference is, named in the warning (for example ``"fair Brier score"``).

    Returns
    -------
    float or numpy.ndarray or pandas.Series
        The skill score, NaN where ``reference`` is 0.
    """
    zero = np.asarray(reference == 0)
    if zero.any():
        warnings.warn(f"the reference's {what} is zero: skill set to NaN")
        if isinstance(reference, (pd.Series, pd.DataFrame)):
            reference = reference.where(~zero)
        elif np.ndim(reference):
            reference = np.where(zero, np.nan, reference)
        else:
            reference = np.nan
    return 1 - (score / reference)


def cell_metrics(P: np.ndarray, outcome: np.ndarray, ids: np.ndarray, labels, *,
                 allowed: np.ndarray | None = None, earlier: np.ndarray | None = None) -> pd.DataFrame:
    """Compute per-unit Brier, RPS, AUC and n, and a pooled ``ALL`` row.

    The pooled row adds the Pietra index and each bin's Brier score and AUC. Samples without an outcome or with a
    missing probability are left out. Brier is the ``"sample"`` scale (``"pooled"`` for the pooled row), RPS the
    ``"cumulative"`` summation.

    Parameters
    ----------
    P : numpy.ndarray of float64, shape (samples, bins)
        The probabilities of the interval bins (``labels``).
    outcome : numpy.ndarray of int, shape (samples,)
        The code of the bin that happened (-1 without one).
    ids : numpy.ndarray, shape (samples,)
        The unit of each sample.
    labels : sequence of str
        The label of each column of ``P``.
    allowed : numpy.ndarray of bool, shape (samples,), optional
        The samples scored (the dissemination units); all when None.
    earlier : numpy.ndarray of float64, shape (samples,), optional
        The probability of an onset before the issue, prepended for the RPS (no outcome is "earlier").

    Returns
    -------
    pandas.DataFrame
        One row per unit (``id``, ``brier``, ``rps``, ``auc``, ``n``) and the ``ALL`` row (with ``lat`` and ``lon`` NaN,
        ``pietra`` and the columns ``brier_<label>`` and ``auc_<label>``). Empty when no sample is kept.
    """
    P = np.asarray(P, dtype=float)
    keep = (outcome >= 0) & ~np.isnan(P).any(axis=1)
    if allowed is not None:
        keep &= allowed
    if not keep.any():
        return pd.DataFrame()
    P, outcome, ids = P[keep], outcome[keep], np.asarray(ids)[keep]
    n_bins = P.shape[1]
    Y = one_hot(outcome, n_bins)
    if earlier is not None:
        early = np.asarray(earlier, dtype=float)[keep]
        P_rps = np.column_stack([np.where(np.isfinite(early), early, 0.0),
                                 np.where(np.isfinite(P), P, 0.0)])
        Y_rps = np.column_stack([np.zeros(len(P)), Y])
    else:
        P_rps, Y_rps = P.copy(), Y.copy()

    unit, units = pd.factorize(ids.astype(str), sort=True)
    order = np.argsort(unit, kind="stable")
    bounds = np.searchsorted(unit[order], np.arange(len(units) + 1))
    if np.isfinite(P).all():
        unit_auc = grouped_auc(Y.ravel(), P.ravel(), np.repeat(unit, n_bins), len(units))   # one ranking of all scores
    else:
        unit_auc = None
    rows = []
    for k, name in enumerate(units):
        pos = order[bounds[k]:bounds[k + 1]]
        g_P, g_Y = P[pos], Y[pos]
        rows.append({
            "id": name,
            "brier": brier(g_P, g_Y),
            "rps": rps(P_rps[pos], Y_rps[pos]),
            "auc": auc(g_Y.flatten().astype(int), g_P.flatten()) if unit_auc is None else float(unit_auc[k]),
            "n": len(pos),
        })
    cells = pd.DataFrame(rows)

    pooled_scores = brier_scores(P, Y, scale="pooled", by_bin=True)
    pooled = {
        "id": "ALL",
        "lat": np.nan,
        "lon": np.nan,
        "brier": pooled_scores["brier"],
        "rps": rps(P_rps, Y_rps),
        "n": len(P),
        "auc": auc(Y.flatten().astype(int), P.flatten()),
        "pietra": pietra(P, Y),
    }
    for j, label in enumerate(labels):
        pooled[f"brier_{label}"] = float(pooled_scores["bin_brier"][j])
    for j, label in enumerate(labels):
        pooled[f"auc_{label}"] = auc(Y[:, j].astype(int), P[:, j])
    return pd.concat([cells, pd.DataFrame([pooled])], ignore_index=True)


def yearly_metrics(P: np.ndarray, outcome: np.ndarray, year: np.ndarray, model: str, method: str, *,
                   auc_method: str = "rank") -> list[dict]:
    """Compute the per-year pooled Brier, RPS and AUC over the interval bins, every unit.

    Samples without an outcome or with a missing probability are left out.

    Parameters
    ----------
    P : numpy.ndarray of float64, shape (samples, bins)
        The probabilities of the interval bins.
    outcome : numpy.ndarray of int, shape (samples,)
        The code of the bin that happened (-1 without one).
    year : numpy.ndarray, shape (samples,)
        The year of each sample.
    model : str
        The model name written in each row.
    method : str
        The cross-validation method written in each row (``cv_method``).
    auc_method : {"rank", "sklearn"}, default "rank"
        ``"rank"`` is the rank-sum AUC (`auc`); ``"sklearn"`` is sklearn's ``roc_auc_score`` (NaN where a year lacks
        positives or negatives), which blending's yearly scores use and which differs from the rank-sum one in the last
        bits.

    Returns
    -------
    list of dict
        One dict per year: ``year``, ``model``, ``cv_method``, ``brier``, ``rps`` and ``auc``.

    Raises
    ------
    ValueError
        ``auc_method`` is not in `AUC_METHODS`.
    """
    if auc_method not in AUC_METHODS:
        raise ValueError(f"auc_method must be one of {AUC_METHODS}, not {auc_method!r}")
    P = np.asarray(P, dtype=float)
    keep = (outcome >= 0) & ~np.isnan(P).any(axis=1)
    P, outcome, year = P[keep], outcome[keep], np.asarray(year)[keep]
    Y = one_hot(outcome, P.shape[1])
    rows = []
    for y in np.unique(year):
        pos = np.flatnonzero(year == y)
        g_P, g_Y = P[pos], Y[pos]
        rows.append({
            "year": int(y), "model": model, "cv_method": method,
            "brier": brier(g_P, g_Y), "rps": rps(g_P, g_Y),
            "auc": auc(g_Y.flatten(), g_P.flatten()) if auc_method == "rank" else _sklearn_auc(g_Y, g_P),
        })
    return rows


def _sklearn_auc(Y: np.ndarray, P: np.ndarray) -> float:
    """Compute sklearn's ``roc_auc_score`` of flattened outcomes and probabilities, NaN when it cannot be computed.

    Parameters
    ----------
    Y : numpy.ndarray
        The outcomes (0 or 1), any shape.
    P : numpy.ndarray
        The probabilities, the shape of ``Y``.

    Returns
    -------
    float
        The score; NaN when the outcomes are all of one class or sklearn raises.
    """
    from sklearn.metrics import roc_auc_score

    y_long, p_long = Y.flatten(), P.flatten()
    try:
        return roc_auc_score(y_long, p_long) if len(np.unique(y_long)) == 2 else np.nan
    except Exception:
        return np.nan


def pooled_skill(cells: pd.DataFrame, baseline: str = "unc_clim_raw") -> pd.DataFrame:
    """Return the pooled rows of every model, with skill against the baseline's.

    Parameters
    ----------
    cells : pandas.DataFrame
        The per-unit and pooled rows of every model: ``id`` (``ALL`` for the pooled row), ``model``, ``brier``, ``rps``
        and ``auc`` (see `cell_metrics`).
    baseline : str, default "unc_clim_raw"
        The model whose pooled row is the reference.

    Returns
    -------
    pandas.DataFrame
        The pooled rows with ``brier_skill`` (BSS), ``rps_skill`` (RPSS) and ``AUC diff`` (the AUC difference times 100)
        against the baseline.

    Raises
    ------
    ValueError
        No pooled row of ``baseline`` is in ``cells``.
    """
    pooled = cells[cells["id"] == "ALL"].copy()
    reference = pooled[pooled["model"] == baseline]
    if reference.empty:
        raise ValueError(f"Baseline model '{baseline}' not found in all_cells.")
    reference = reference.iloc[0]
    pooled["brier_skill"] = skill_score(pooled["brier"], reference["brier"], "pooled Brier score")
    pooled["rps_skill"] = skill_score(pooled["rps"], reference["rps"], "pooled RPS")
    pooled["AUC diff"] = (pooled["auc"] - reference["auc"]) * 100.0
    return pooled
