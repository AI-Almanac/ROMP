"""Brier score, RPS, AUC and skill-score calculations for binned onset forecast-observation pairs.

The pairs (`hcwf.metrics.bins.PairArrays`: (case, bin) matrices) are scored by `hcwf.metrics.probabilistic`; this
module only keeps HCWF's result layout (dicts keyed by bin label) and its guards. `hcwf.metrics.skill` calls the
functions of this module for the forecast and its reference, and `hcwf.metrics.tables` reads their result dicts.
"""

import numpy as np
import pandas as pd
import warnings
from hcwf.metrics import probabilistic
from hcwf.metrics.bins import LeadBins
import logging

log = logging.getLogger(__name__)


def get_target_bins(brier_forecast, brier_climatology):
    """Return the bin labels both results score, restricted to the ``Days a-b`` bins and sorted by first day.

    Parameters
    ----------
    brier_forecast : dict
        The forecast's Brier results (`calculate_brier_score`); its ``bin_fair_brier_scores`` keys are used.
    brier_climatology : dict
        The reference's Brier results; its ``bin_fair_brier_scores`` keys are used.

    Returns
    -------
    list of str
        The labels in both, starting with ``Days``, in day order.
    """
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    common_bins = all_forecast_bins.intersection(all_clim_bins)

    target_bins = []
    for bin_label in common_bins:
        if (bin_label.startswith('Days ') and
            not bin_label.startswith('After') and
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)

    return sorted(target_bins, key=LeadBins.first_day_of)


def _brier_results(pairs, members):
    """Return the Brier results of pairs in HCWF's layout: overall and per-bin Brier and fair Brier scores.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        The pairs to score.
    members : numpy.ndarray, shape (case,)
        The ensemble size behind each case, for the fair correction.

    Returns
    -------
    dict
        ``brier_score`` and ``fair_brier_score`` (float), ``bin_brier_scores`` and ``bin_fair_brier_scores`` (dicts
        of bin label to float, in label order).
    """
    scores = probabilistic.brier_scores(pairs.prob, pairs.observed, members, scale="element", by_bin=True,
                                        bin_mean="compensated", skipna=True)
    labels = pairs.labels
    by_label = sorted(range(len(labels)), key=lambda j: labels[j])
    return {
        'brier_score': scores['brier'],
        'fair_brier_score': scores['fair_brier'],
        'bin_brier_scores': {labels[j]: float(scores['bin_brier'][j]) for j in by_label},
        'bin_fair_brier_scores': {labels[j]: float(scores['bin_fair_brier'][j]) for j in by_label},
    }


def _require_members(pairs, what):
    """Check that the first case has more than one ensemble member (the fair scores divide by members - 1).

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        The pairs; without cases the member count is taken as 1.
    what : str
        The name of the score, for the message.

    Raises
    ------
    ValueError
        The first case has one member or fewer.
    """
    n_members = pairs.members[0] if pairs.n_cases > 0 else 1
    if n_members <= 1:
        raise ValueError(
            f"Fair {what} requires >1 ensemble member, but found total_members={n_members}. "
            "Check that your model data contains ensemble members and that the 'members' / "
            "'probabilistic' parameters are configured correctly."
        )


# Function to calculate Brier Score and Fair Brier Score for the model forecasts (both overall and bin-wise)
def calculate_brier_score(pairs):
    """Calculate the Brier score and the fair Brier score of probabilistic forecasts.

    The scores are::

        Brier Score = (1/n*m) * sum(Y_ij - p_ij)^2
        Fair Brier Score = (1/n*m) * sum[(Y_ij - p_ij)^2 - p_ij(1-p_ij)/(ens-1)]

    where ``n`` is the number of forecasts, ``m`` the number of bins per forecast, ``Y_ij`` is 1 if the onset occurred
    in bin ``j`` for forecast ``i`` (0 otherwise), ``p_ij`` the predicted probability for bin ``j`` in forecast ``i``
    and ``ens`` the number of ensemble members.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        Output of `hcwf.metrics.bins.forecast_pairs`, restricted to the scored bins (`PairArrays.select`).

    Returns
    -------
    dict
        ``brier_score``, ``fair_brier_score`` (float), ``bin_brier_scores`` and ``bin_fair_brier_scores`` (dicts of
        bin label to float). A `ValueError` is raised (by `_require_members`) when the first case has one ensemble
        member or fewer.

    """
    _require_members(pairs, "Brier Score")

    brier_results = _brier_results(pairs, pairs.members)

    log.debug(f"Brier Score: {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

    return brier_results


def _auc_results(pairs):
    """Return the AUC results of pairs in HCWF's layout: the overall AUC and the AUC of each bin.

    The AUC of a bin is NaN where it has only positives or only negatives.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        The pairs to score.

    Returns
    -------
    dict
        ``auc`` (float) and ``bin_auc_scores`` (bin label to float). When the overall AUC is NaN (all cases positive or
        all negative) ``bin_auc_scores`` is empty and ``n_positive`` and ``n_negative`` are added.
    """
    P, Y = pairs.prob, pairs.observed
    p_ij, y_ij = P.ravel(), Y.ravel()
    auc = probabilistic.auc(y_ij, p_ij)
    if np.isnan(auc):
        log.warning("Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': np.sum(y_ij),
            'n_negative': np.sum(1 - y_ij),
            'bin_auc_scores': {},
        }
    return {
        'auc': auc,
        'bin_auc_scores': {label: probabilistic.auc(Y[:, j], P[:, j]) for j, label in enumerate(pairs.labels)},
    }


# Function to calculate Area Under the Curve (AUC) for the model forecasts (both overall and bin-wise)
def calculate_auc(pairs):
    """Calculate the area under the curve (AUC) of probabilistic forecasts.

    The AUC is::

        AUC = sum_{i,j,i',j'} Y_ij (1 - Y_i'j') 1[p_ij > p_i'j'] / [(sum_ij Y_ij) (sum_ij (1 - Y_ij))]

    where ``Y_ij`` is 1 if the onset occurred in bin ``j`` for forecast ``i`` (0 otherwise), ``p_ij`` the predicted
    probability for bin ``j`` in forecast ``i`` and ``1[...]`` the indicator function. Ties count as 1/2.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        Output of `hcwf.metrics.bins.forecast_pairs`, restricted to the scored bins (`PairArrays.select`).

    Returns
    -------
    dict
        The results of `_auc_results`: ``auc`` and ``bin_auc_scores``.
    """
    return _auc_results(pairs)


# Function to calculate Ranked Probability Score (RPS) and Fair RPS for the model forecasts for (either 15 day or 30 day forecast)
def calculate_rps(pairs):
    """Calculate the ranked probability score (RPS) and the fair RPS of probabilistic forecasts.

    The scores are::

        RPS = (1/n*m) * sum_i sum_k (sum_{j<=k} (Y_ij - p_ij))^2
        Fair RPS = (1/n*m) * sum_i sum_k [(sum_{j<=k} (Y_ij - p_ij))^2
                                          - (sum_{j<=k} p_ij)(1 - sum_{j<=k} p_ij)/(ens-1)]

    where ``n`` is the number of forecasts, ``m`` the number of bins per forecast, ``Y_ij`` is 1 if the onset occurred
    in bin ``j`` for forecast ``i`` (0 otherwise), ``p_ij`` the predicted probability for bin ``j`` in forecast ``i``,
    ``k`` the cumulative index (1 to ``m``) and ``ens`` the number of ensemble members. The cases are taken in order of
    init and place, so that the mean over them is reproducible.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        Output of `hcwf.metrics.bins.forecast_pairs`, restricted to the scored bins (`PairArrays.select`).

    Returns
    -------
    dict
        ``rps`` and ``fair_rps`` (float) and ``n_forecasts`` (int, the number of cases). A `ValueError` is raised (by `_require_members`) when the first case
        has one ensemble member or fewer.

    """
    _require_members(pairs, "RPS")

    # The cases in order of init and place; the mean over them is taken in that order
    order = pairs.sorted_order()
    if (order == np.arange(len(order))).all():
        P, Y, members = pairs.prob, pairs.observed, pairs.members
    else:
        P, Y, members = pairs.prob[order], pairs.observed[order], pairs.members[order]
    n_forecasts = pairs.n_cases
    rps_values, fair_rps_values = probabilistic.rps_cases(np.ascontiguousarray(P), Y, members, summation="prefix")

    # Calculate overall RPS (average over all forecasts)
    rps = np.mean(rps_values)
    fair_rps = np.mean(fair_rps_values)

    rps_results = {
        'rps': rps,
        'fair_rps': fair_rps,
        'n_forecasts': int(n_forecasts),

    }

    log.debug(f"RPS: {rps_results['rps']:.4f}")
    log.debug(f"Fair RPS: {rps_results['fair_rps']:.4f}")
    log.debug(f"Number of forecasts: {rps_results['n_forecasts']}")

    return rps_results


def calculate_brier_score_climatology(pairs):
    """Calculate the Brier score and the fair Brier score of a climatology (or other pooled-member) reference.

    The scores of `calculate_brier_score`, over the bins that are not "Before" bins (e.g. ``Before day N``), with the
    fair correction taken from each case's ``members_with_onset`` instead of ``members``.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        Output of `hcwf.metrics.bins.create_climatological_forecast_obs_pairs`, restricted to the scored bins; needs
        `PairArrays.members_with_onset`.

    Returns
    -------
    dict
        The keys of `calculate_brier_score` plus ``n_samples`` (int, entries scored) and ``filtered_bins`` (sorted
        labels), and ``excluded_bins`` (``['Before initialization']``). With no bin left, the scores are NaN, the
        bin dicts empty, ``n_samples`` 0 and ``filtered_bins`` empty.
    """
    # Filter out "Before initialization" bin
    filtered = pairs.without_before()

    if filtered.prob.size == 0:
        log.warning("No data remaining after filtering out 'Before initialization' bin")
        return {
            'brier_score': np.nan,
            'fair_brier_score': np.nan,
            'bin_brier_scores': {},
            'bin_fair_brier_scores': {},
            'n_samples': 0,
            'filtered_bins': []
        }

    log.debug("Calculating Brier Score excluding 'Before initialization' bin")
    log.debug(f"Original samples: {pairs.prob.size}, After filtering: {filtered.prob.size}")

    brier_results = _brier_results(filtered, filtered.members_with_onset)
    brier_results.update({
        'n_samples': filtered.prob.size,
        'filtered_bins': sorted(filtered.labels),
        'excluded_bins': ['Before initialization']
    })

    log.debug(f"Brier Score (excluding 'Before initialization'): {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score (excluding 'Before initialization'): {brier_results['fair_brier_score']:.4f}")
    log.debug(f"Bins included in calculation: {brier_results['filtered_bins']}")

    return brier_results


def calculate_auc_climatology(pairs):
    """Calculate the AUC of a climatology reference: `calculate_auc` over the bins that are not "Before" bins.

    Parameters
    ----------
    pairs : hcwf.metrics.bins.PairArrays
        Output of `hcwf.metrics.bins.create_climatological_forecast_obs_pairs`, restricted to the scored bins.

    Returns
    -------
    dict
        The results of `_auc_results`: ``auc`` and ``bin_auc_scores``.
    """
    return _auc_results(pairs.without_before())


def calculate_skill_scores(brier_forecast, rps_forecast,
                          brier_climatology, rps_climatology):
    """Calculate the skill scores of the forecast model relative to its reference (climatology or another model).

    The skill score is ``1 - (forecast_score / climatology_score)``, from the fair scores: overall for the Brier score
    and the RPS, and per ``Days a-b`` bin for the Brier score. The tables of the scores are written to the debug log.

    Parameters
    ----------
    brier_forecast : dict
        Brier results of the forecast model (`calculate_brier_score`).
    rps_forecast : dict
        RPS results of the forecast model (`calculate_rps`).
    brier_climatology : dict
        Brier results of the reference (`calculate_brier_score_climatology` or `calculate_brier_score`).
    rps_climatology : dict
        RPS results of the reference.

    Returns
    -------
    dict
        ``fair_brier_skill_score`` and ``fair_rps_skill_score`` (float) and ``bin_fair_brier_skill_scores`` (bin label
        to float, NaN where a bin is missing in one of them). ``fair_brier_skill_score`` is NaN, with a warning, when
        the reference's fair Brier score is zero.
    """
    skill_scores = {}

    log.debug("="*60)
    log.debug("SKILL SCORE CALCULATIONS")
    log.debug("="*60)

    # Fair Brier Skill Score (1-15 day overall)
    if brier_climatology['fair_brier_score'] == 0:
        fair_bss_overall = float("nan")
        warnings.warn("the reference's fair Brier score is zero: fair BSS set to NaN")
    else:
        fair_bss_overall = probabilistic.skill_score(brier_forecast['fair_brier_score'], brier_climatology['fair_brier_score'])

    skill_scores['fair_brier_skill_score'] = fair_bss_overall

    log.debug(f"Fair Brier Skill Score (1-15 day): {fair_bss_overall:.4f}")

    # Fair RPS Skill Score (1-15 day overall)
    fair_rpss_overall = probabilistic.skill_score(rps_forecast['fair_rps'], rps_climatology['fair_rps'])
    skill_scores['fair_rps_skill_score'] = fair_rpss_overall

    log.debug(f"Fair RPS Skill Score (1-15 day): {fair_rpss_overall:.4f}")

    # Automatically extract target bins from the data: the "Days X-Y" bins both have, in day order
    target_bins = get_target_bins(brier_forecast, brier_climatology)
    common_bins = set(brier_forecast['bin_fair_brier_scores']) & set(brier_climatology['bin_fair_brier_scores'])
    excluded_bins = [bin_label for bin_label in common_bins if bin_label not in target_bins]

    log.debug(f"\nAutomatically detected target bins: {target_bins}")
    log.debug(f"Excluded bins: {excluded_bins}")

    # Bin-wise Fair Brier Skill Scores
    bin_fair_bss = {}

    log.debug("\nBin-wise Fair Brier Skill Scores:")
    for bin_label in target_bins:
        if bin_label in brier_forecast['bin_fair_brier_scores'] and bin_label in brier_climatology['bin_fair_brier_scores']:
            forecast_fair_brier_bin = brier_forecast['bin_fair_brier_scores'][bin_label]
            clim_fair_brier_bin = brier_climatology['bin_fair_brier_scores'][bin_label]

            fair_bss_bin = probabilistic.skill_score(forecast_fair_brier_bin, clim_fair_brier_bin)
            bin_fair_bss[bin_label] = fair_bss_bin

            log.debug(f"  {bin_label}: Fair BSS = {fair_bss_bin:.4f}")
        else:
            bin_fair_bss[bin_label] = np.nan
            log.debug(f"  {bin_label}: Fair BSS = NaN (missing data)")

    skill_scores['bin_fair_brier_skill_scores'] = bin_fair_bss

    # Dynamic table header based on detected bins
    header = f"{'Metric':<30} {'Overall (1-15 day)':<18}"
    for bin_name in target_bins:
        # Shorten bin names for table display
        short_name = bin_name.replace('Days ', '')
        header += f" {short_name:<12}"

    # Calculate table width
    table_width = 30 + 18 + 12 * len(target_bins)

    # Summary table
    log.debug("\n" + "="*table_width)
    log.debug("SKILL SCORE SUMMARY TABLE")
    log.debug("="*table_width)
    log.debug(header)
    log.debug("-"*table_width)

    # Fair Brier Skill Score row
    fair_bss_row = f"{'Fair Brier Skill Score':<30} {fair_bss_overall:<18.4f}"
    for bin_name in target_bins:
        if bin_name in bin_fair_bss and not pd.isna(bin_fair_bss[bin_name]):
            fair_bss_row += f" {bin_fair_bss[bin_name]:<12.4f}"
        else:
            fair_bss_row += f" {'N/A':<12}"
    log.debug(fair_bss_row)

    # Fair RPS Skill Score row
    fair_rpss_row = f"{'Fair RPS Skill Score':<30} {fair_rpss_overall:<18.4f}"
    for _ in target_bins:
        fair_rpss_row += f" {'N/A':<12}"  # RPS is overall only
    log.debug(fair_rpss_row)

    log.debug("-"*table_width)

    # Add interpretation guide
    log.debug("\nInterpretation Guide:")
    log.debug("• Positive skill scores indicate forecast is better than climatology")
    log.debug("• Negative skill scores indicate forecast is worse than climatology")
    log.debug("• Skill score = 0 means forecast equals climatology")
    log.debug("• Perfect score = 1.0")

    # Additional detailed results
    log.debug("\n" + "="*60)
    log.debug("DETAILED RESULTS")
    log.debug("="*60)

    log.debug(f"\nForecast Fair Brier Score : {brier_forecast['fair_brier_score']:.4f}")
    log.debug(f"Climatology Fair Brier Score : {brier_climatology['fair_brier_score']:.4f}")
    log.debug(f"Fair Brier Skill Score: {fair_bss_overall:.4f}")

    log.debug(f"\nForecast Fair RPS : {rps_forecast['fair_rps']:.4f}")
    log.debug(f"Climatology Fair RPS : {rps_climatology['fair_rps']:.4f}")
    log.debug(f"Fair RPS Skill Score: {fair_rpss_overall:.4f}")

    log.debug("\nBin-wise Fair Brier Score Comparisons:")
    for bin_name in target_bins:
        if bin_name in brier_forecast['bin_fair_brier_scores'] and bin_name in brier_climatology['bin_fair_brier_scores']:
            forecast_val = brier_forecast['bin_fair_brier_scores'][bin_name]
            clim_val = brier_climatology['bin_fair_brier_scores'][bin_name]
            skill_val = bin_fair_bss[bin_name]
            log.debug(f"  {bin_name}:")
            log.debug(f"    Forecast: {forecast_val:.4f}")
            log.debug(f"    Climatology: {clim_val:.4f}")
            log.debug(f"    Skill Score: {skill_val:.4f}")
        else:
            log.debug(f"  {bin_name}: Missing data")

    return skill_scores
