"""Compute forecast and reference-model BS, RPS, AUC and derived skill scores from forecast-observation pairs."""


from hcwf.metrics.score import calculate_brier_score, calculate_auc, calculate_rps, calculate_brier_score_climatology, calculate_auc_climatology, calculate_skill_scores
import logging

log = logging.getLogger(__name__)

def create_score_results(case, prepared, metrics, ev):
    """Compute Brier score, RPS and AUC (as flagged by metrics.BS, .RPS, .AUC) for the forecast model and its reference,
    plus skill scores if metrics.skill_score is set, from the pairs `prepared` (PreparedPairs).

    The scored day bins are ev.day_bins (a window's bins: Window.bins). The reference is climatology or conditional
    climatology (P(onset on lead k | no onset by the init)) when prepared.climatology is set, otherwise another model.
    Returns a dict with forecast_pairs, climatology_pairs (the PairArrays of the window's bins), BS, RPS, AUC, their
    *_ref counterparts, skill_score and (if skill_score) skill_results.
    """
    BS, RPS, AUC, day_bins = metrics.BS, metrics.RPS, metrics.AUC, ev.day_bins
    results = {}

    log.debug("\n1. Processing forecast model...")

    # select bins without "Day before" and "Day After"
    forecast = prepared.forecast.select(day_bins)

    results["forecast_pairs"] = forecast

    results["BS"] = calculate_brier_score(forecast) if BS else None
    results["RPS"] = calculate_rps(forecast) if RPS else None
    results["AUC"] = calculate_auc(forecast) if AUC else None


    log.debug("\n2. Processing reference model...")

    results["BS_ref"], results["RPS_ref"], results["AUC_ref"] = None, None, None
    results['skill_score'] = None

    if prepared.climatology is not None:

        climatology = prepared.climatology.select(day_bins)

        results["climatology_pairs"] = climatology

        results["BS_ref"] = calculate_brier_score_climatology(climatology) if BS else None
        results["RPS_ref"] = calculate_rps(climatology) if RPS else None
        results["AUC_ref"] = calculate_auc_climatology(climatology) if AUC else None

    else:
        results["climatology_pairs"] = None

        # the reference model's own data and settings, never the forecast model's
        reference = prepared.reference.select(day_bins)

        results["BS_ref"] = calculate_brier_score(reference) if BS else None
        results["RPS_ref"] = calculate_rps(reference) if RPS else None
        results["AUC_ref"] = calculate_auc(reference) if AUC else None


    log.debug("%s %s", "results[BS]  =  ", results["BS"])
    log.debug("%s %s", "results[BS_ref]  =  ", results["BS_ref"])
    log.debug("%s %s", "results[RPS]  =  ", results["RPS"])
    log.debug("%s %s", "results[RPS_ref]  =  ", results["RPS_ref"])
    if metrics.skill_score:

        skill_results = calculate_skill_scores(
        results["BS"], results["RPS"],
        results["BS_ref"], results["RPS_ref"]
        )

        results["skill_results"] = skill_results

    return results

