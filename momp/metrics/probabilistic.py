"""Probabilistic skill on arrays: Brier score, RPS, AUC and Pietra index of lead-bin probabilities.

One implementation for ROMP's pooled ensemble scores and the blending package's per-unit, pooled and per-year
scores. Everything is array-based: P holds the probabilities and Y the 0/1 outcomes of (samples, bins) with the
bins in lead order; `members` (the ensemble size behind each probability, a scalar, one per sample or one per
entry) switches on the fair scores, which take out the sampling noise of the probability, p(1-p)/(members-1).

Conventions are options, never second implementations:

- scale of the Brier score: "element" averages the squared error over all (sample, bin) entries (ROMP); "sample"
  sums it over the bins and averages over the samples (blending: m times "element"); "pooled" is "sample" written
  as one sum over everything divided by the samples (blending's pooled row; equal to "sample" up to rounding).
- mean of a bin's entries: "numpy" is np.mean of the column (blending); "compensated" adds the entries one by one
  with Kahan compensation, as a pandas groupby mean does (ROMP's per-bin scores, to the last bit).
- summation of the RPS: "cumulative" is np.cumsum over the bins (blending); "prefix" sums the first k bins for each
  k and adds the terms one after the other (ROMP). They differ only in the last bits, and for 8 bins or more.
- which bins are scored is the columns passed in: ROMP scores its inner day bins only, blending its week bins and
  `later` (plus the `earlier` column in the RPS, see `rps`).
- AUC counts ties as 1/2, as the Mann-Whitney U statistic and sklearn's roc_auc_score do.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata

BRIER_SCALES = ("element", "sample", "pooled")
BIN_MEANS = ("numpy", "compensated")
RPS_SUMMATIONS = ("cumulative", "prefix")


def one_hot(outcome: np.ndarray, n_classes: int) -> np.ndarray:
    """(samples, classes) indicators of each sample's outcome code (-1: none)."""
    return np.column_stack([(outcome == k).astype(float) for k in range(n_classes)])


def clean(P: np.ndarray) -> np.ndarray:
    """Probabilities with non-finite or negative entries at 0, rows summing to 1 (uniform where a row has nothing)."""
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
    """Mean of each column over its non-NaN entries, added in order with Kahan compensation (pandas' groupby mean)."""
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
    """The mean of each column of an (samples, bins) array (see BIN_MEANS); "compensated" always skips NaN."""
    if how not in BIN_MEANS:
        raise ValueError(f"bin_mean must be one of {BIN_MEANS}, not {how!r}")
    if how == "compensated":
        return _compensated_column_means(np.ascontiguousarray(values, dtype=np.float64))
    mean = np.nanmean if skipna else np.mean
    return np.array([mean(values[:, j]) for j in range(values.shape[1])])


def _fair_terms(squares, P, members):
    """The squared errors less p(1-p)/(members-1) entry by entry; members is a scalar, one per sample or one per
    entry."""
    members = np.asarray(members, dtype=float)
    if members.ndim == 1 and P.ndim == 2:
        members = members[:, None]
    return squares - (P * (1 - P)) / (members - 1)


def brier_scores(P: np.ndarray, Y: np.ndarray, members=None, *, scale: str = "sample", by_bin: bool = False,
                 bin_mean: str = "numpy", skipna: bool = False) -> dict:
    """Brier score, and with `members` the fair Brier score, of (samples, bins) probabilities and outcomes.

    Returns {"brier": float, "fair_brier": float (only with members)}; with by_bin also "bin_brier" and
    "bin_fair_brier", the mean over the samples of each bin's entries (arrays, one value per column; bin_mean says
    how the mean is taken, see BIN_MEANS). skipna leaves NaN entries out of the means (ROMP: a case with one
    member has 0/0 in the fair term when p is 0 or 1).
    The fair score subtracts p(1-p)/(members-1) from each squared error, so members must exceed 1 (an infinite
    member count, as of a climatology, takes nothing off). With the "sample" and "pooled" scales and one member
    count per sample or for all, the correction is summed over the bins before it is taken off, as blending does.
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
    """Brier score of (samples, bins) probabilities and outcomes (see `brier_scores` for the scales)."""
    return brier_scores(P, Y, scale=scale)["brier"]


def rps_cases(P: np.ndarray, Y: np.ndarray, members=None, *, summation: str = "cumulative",
              valid: np.ndarray | None = None):
    """The RPS of every sample, sum over the bins k of (cumulative p - cumulative y)^2, and with `members` the
    fair RPS (each term less cum_p(1-cum_p)/(members-1)); returns (rps, fair_rps) arrays, fair_rps None
    without members.

    valid: (samples, bins) False where a sample has no such bin (padding of samples with fewer bins, which must
    be the last columns); only the "prefix" summation takes it.
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
    """Mean over the samples of the RPS of (samples, bins) probabilities and one-hot outcomes."""
    return float(np.mean(rps_cases(P, Y, summation=summation)[0]))


def auc(y01: np.ndarray, score: np.ndarray) -> float:
    """AUC by the Wilcoxon rank-sum over the finite scores; NaN when there are no positives or no negatives."""
    ok = np.isfinite(score) & ~np.isnan(y01)
    y = y01[ok].astype(int)
    s = score[ok]
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s, method="average")
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)


def pietra(P: np.ndarray, Y: np.ndarray) -> float:
    """Pietra index: the Kolmogorov-Smirnov statistic between the probabilities of the bins that happened and of
    those that did not."""
    p = P.flatten()
    y = Y.flatten().astype(int)
    positive, negative = p[y == 1], p[y == 0]
    if len(positive) and len(negative):
        return float(ks_2samp(positive, negative).statistic)
    return np.nan


def skill_score(score, reference):
    """1 - score/reference: positive when the score beats the reference (both scores lower-is-better)."""
    return 1 - (score / reference)


def cell_metrics(P: np.ndarray, outcome: np.ndarray, ids: np.ndarray, labels, *,
                 allowed: np.ndarray | None = None, earlier: np.ndarray | None = None) -> pd.DataFrame:
    """Per-unit Brier, RPS, AUC and n, and a pooled 'ALL' row that adds the Pietra index and each bin's Brier
    and AUC.

    P: (samples, bins) over the interval bins (`labels`); outcome: their codes (-1 without one); allowed: the
    samples scored (the dissemination units); earlier: the probability of an onset before the issue, prepended
    for the RPS (no outcome is 'earlier'). Samples without an outcome or with a missing probability are left
    out. Brier is the "sample" scale, RPS the "cumulative" summation.
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

    units, unit = np.unique(ids.astype(str), return_inverse=True)
    order = np.argsort(unit, kind="stable")
    bounds = np.searchsorted(unit[order], np.arange(len(units) + 1))
    rows = []
    for k, name in enumerate(units):
        pos = order[bounds[k]:bounds[k + 1]]
        g_P, g_Y = P[pos], Y[pos]
        rows.append({
            "id": name,
            "brier": brier(g_P, g_Y),
            "rps": rps(P_rps[pos], Y_rps[pos]),
            "auc": auc(g_Y.flatten().astype(int), g_P.flatten()),
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


def yearly_metrics(P: np.ndarray, outcome: np.ndarray, year: np.ndarray, model: str, method: str) -> list[dict]:
    """Per-year pooled Brier, RPS and AUC over the interval bins, every unit.

    The AUC is the rank-sum one, which equals sklearn's roc_auc_score up to rounding.
    """
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
            "brier": brier(g_P, g_Y), "rps": rps(g_P, g_Y), "auc": auc(g_Y.flatten(), g_P.flatten()),
        })
    return rows


def pooled_skill(cells: pd.DataFrame, baseline: str = "unc_clim_raw") -> pd.DataFrame:
    """The pooled rows of every model, with skill against the baseline's (BSS, RPSS, AUC difference x100)."""
    pooled = cells[cells["id"] == "ALL"].copy()
    reference = pooled[pooled["model"] == baseline]
    if reference.empty:
        raise ValueError(f"Baseline model '{baseline}' not found in all_cells.")
    reference = reference.iloc[0]
    pooled["brier_skill"] = 1.0 - (pooled["brier"] / reference["brier"])
    pooled["rps_skill"] = 1.0 - (pooled["rps"] / reference["rps"])
    pooled["AUC diff"] = (pooled["auc"] - reference["auc"]) * 100.0
    return pooled
