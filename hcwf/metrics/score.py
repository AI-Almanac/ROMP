"""Brier score, RPS, AUC and skill-score calculations for binned onset forecast-observation pairs.

The pairs (PairArrays: (case, bin) matrices) are scored by hcwf.metrics.probabilistic; this module only keeps
HCWF's result layout (dicts keyed by bin label) and its guards.
"""

import numpy as np
import pandas as pd
import warnings
from hcwf.metrics import probabilistic
from hcwf.metrics.bins import LeadBins
import logging

log = logging.getLogger(__name__)


def get_target_bins(brier_forecast, brier_climatology):
    """Extract and sort target bins"""
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
    """Brier results of pairs in HCWF's layout: overall and per-bin Brier and fair Brier scores (`members`: the
    ensemble size of each case)."""
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
    """Raise unless the first case has more than one ensemble member (the fair scores divide by members - 1)."""
    n_members = pairs.members[0] if pairs.n_cases > 0 else 1
    if n_members <= 1:
        raise ValueError(
            f"Fair {what} requires >1 ensemble member, but found total_members={n_members}. "
            "Check that your model data contains ensemble members and that the 'members' / "
            "'probabilistic' parameters are configured correctly."
        )


# Function to calculate Brier Score and Fair Brier Score for the model forecasts (both overall and bin-wise)
def calculate_brier_score(pairs):
    """
    Calculate Brier Score and Fair Brier Score for probabilistic forecasts.

    Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
    Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]

    where:
    - n = number of forecasts
    - m = number of bins per forecast
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - ens = number of ensemble members

    Parameters:
    -----------
    pairs : PairArrays
        Output from forecast_pairs(), restricted to the scored bins (PairArrays.select)

    Returns:
    --------
    dict with Brier score metrics
    """
    _require_members(pairs, "Brier Score")

    brier_results = _brier_results(pairs, pairs.members)

    log.debug(f"Brier Score: {brier_results['brier_score']:.4f}")
    log.debug(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

    return brier_results


def _auc_results(pairs):
    """AUC results of pairs in HCWF's layout: the overall AUC and the AUC of each bin (NaN where a bin has
    only positives or only negatives)."""
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
    """
    Calculate Area Under the Curve (AUC) for probabilistic forecasts.

    AUC = Σ_{i,j,i',j'} Y_{ij}(1-Y_{i'j'}) · 1[p_{ij} > p_{i'j'}] /
          [(Σ_{i,j} Y_{ij})(Σ_{i,j} (1-Y_{ij}))]

    where:
    - Y_{ij} = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_{ij} = predicted probability for bin j in forecast i
    - 1[p_{ij} > p_{i'j'}] = indicator function (1 if true, 0 if false)

    Parameters:
    -----------
    pairs : PairArrays
        Output from forecast_pairs(), restricted to the scored bins (PairArrays.select)

    Returns:
    --------
    dict with AUC metrics
    """
    return _auc_results(pairs)


# Function to calculate Ranked Probability Score (RPS) and Fair RPS for the model forecasts for (either 15 day or 30 day forecast)
def calculate_rps(pairs):
    """
    Calculate Ranked Probability Score (RPS) and Fair RPS for probabilistic forecasts.

    RPS = (1/n*m) * Σ_i Σ_k (Σ_j≤k (Y_ij - p_ij))²
    Fair RPS = (1/n*m) * Σ_i Σ_k [(Σ_j≤k (Y_ij - p_ij))² - (Σ_j≤k p_ij)(1 - Σ_j≤k p_ij)/(ens-1)]

    where:
    - n = number of forecasts
    - m = number of bins per forecast
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - k = cumulative index (1 to m)
    - ens = number of ensemble members

    Parameters:
    -----------
    pairs : PairArrays
        Output from forecast_pairs(), restricted to the scored bins (PairArrays.select)

    Returns:
    --------
    dict with RPS metrics
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
    """
    Calculate Brier Score and Fair Brier Score of a climatology (or other pooled-member) reference.

    The scores of calculate_brier_score, over the bins that are not "Before" bins (e.g. "Before day N"), with the
    fair correction taken from each case's members_with_onset instead of members.

    Parameters:
    -----------
    pairs : PairArrays
        Output from create_climatological_forecast_obs_pairs(), restricted to the scored bins; needs members_with_onset

    Returns:
    --------
    dict with Brier score metrics
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

    log.debug(f"Calculating Brier Score excluding 'Before initialization' bin")
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
    """
    Calculate Area Under the Curve (AUC) of a climatology reference: calculate_auc over the bins that are not
    "Before" bins.

    Parameters:
    -----------
    pairs : PairArrays
        Output from create_climatological_forecast_obs_pairs(), restricted to the scored bins

    Returns:
    --------
    dict with AUC metrics
    """
    return _auc_results(pairs.without_before())


def calculate_skill_scores(brier_forecast, rps_forecast,
                          brier_climatology, rps_climatology):
    """
    Calculate skill scores for forecast model relative to climatology.

    Skill Score = 1 - (forecast_score / climatology_score)

    Parameters:
    -----------
    brier_forecast : dict
        Brier score results from forecast model
    rps_forecast : dict
        RPS results from forecast model
    brier_climatology : dict
        Brier score results from climatology
    rps_climatology : dict
        RPS results from climatology

    Returns:
    --------
    dict with skill scores
    """

    skill_scores = {}

    log.debug("="*60)
    log.debug("SKILL SCORE CALCULATIONS")
    log.debug("="*60)

    # Fair Brier Skill Score (1-15 day overall)
    if brier_climatology == 0:
        fair_bss_bin = float("nan")
        warnings.warn("\n brier_climatology is zero → fair BSS set to NaN \n")
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

    log.debug(f"\nBin-wise Fair Brier Skill Scores:")
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
    log.debug(f"\n" + "="*table_width)
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
    log.debug(f"\nInterpretation Guide:")
    log.debug(f"• Positive skill scores indicate forecast is better than climatology")
    log.debug(f"• Negative skill scores indicate forecast is worse than climatology")
    log.debug(f"• Skill score = 0 means forecast equals climatology")
    log.debug(f"• Perfect score = 1.0")

    # Additional detailed results
    log.debug(f"\n" + "="*60)
    log.debug("DETAILED RESULTS")
    log.debug("="*60)

    log.debug(f"\nForecast Fair Brier Score : {brier_forecast['fair_brier_score']:.4f}")
    log.debug(f"Climatology Fair Brier Score : {brier_climatology['fair_brier_score']:.4f}")
    log.debug(f"Fair Brier Skill Score: {fair_bss_overall:.4f}")

    log.debug(f"\nForecast Fair RPS : {rps_forecast['fair_rps']:.4f}")
    log.debug(f"Climatology Fair RPS : {rps_climatology['fair_rps']:.4f}")
    log.debug(f"Fair RPS Skill Score: {fair_rpss_overall:.4f}")

    log.debug(f"\nBin-wise Fair Brier Score Comparisons:")
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
