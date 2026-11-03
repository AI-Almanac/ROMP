"""Compute forecast and reference-model BS, RPS, AUC and derived skill scores from forecast-observation pairs."""


from hcwf.metrics.score import calculate_brier_score, calculate_auc, calculate_rps, calculate_brier_score_climatology, calculate_auc_climatology, calculate_skill_scores
from hcwf.metrics.bins import extract_pd_bins
import logging

log = logging.getLogger(__name__)

def create_score_results(case, prepared, metrics, ev):
    """Compute Brier score, RPS and AUC (as flagged by metrics.BS, .RPS, .AUC) for the forecast model and its reference,
    plus skill scores if metrics.skill_score is set, from the pair tables `prepared` (PreparedPairs).

    The scored day bins are ev.day_bins (a window's bins: Window.bins). The reference is climatology or conditional
    climatology (P(onset on lead k | no onset by the init)) when prepared.climatology is set, otherwise another model.
    Returns a dict with forecast_obs_df, climatology_obs_df, BS, RPS, AUC, their
    *_ref counterparts, skill_score and (if skill_score) skill_results.
    """
    BS, RPS, AUC, day_bins = metrics.BS, metrics.RPS, metrics.AUC, ev.day_bins
    results = {}

    log.debug("\n1. Processing forecast model...")

    # select bins without "Day before" and "Day After"
    forecast_obs_df = extract_pd_bins(prepared.forecast, day_bins)

    results["forecast_obs_df"] = forecast_obs_df

    results["BS"] = calculate_brier_score(forecast_obs_df) if BS else None
    results["RPS"] = calculate_rps(forecast_obs_df) if RPS else None
    results["AUC"] = calculate_auc(forecast_obs_df) if AUC else None


    log.debug("\n2. Processing reference model...")

    results["BS_ref"], results["RPS_ref"], results["AUC_ref"] = None, None, None
    results['skill_score'] = None

    if prepared.climatology is not None:

        climatology_obs_df = extract_pd_bins(prepared.climatology, day_bins)

        results["climatology_obs_df"] = climatology_obs_df

        results["BS_ref"] = calculate_brier_score_climatology(climatology_obs_df) if BS else None
        results["RPS_ref"] = calculate_rps(climatology_obs_df) if RPS else None
        results["AUC_ref"] = calculate_auc_climatology(climatology_obs_df) if AUC else None

    else:
        results["climatology_obs_df"] = None

        # the reference model's own data and settings, never the forecast model's
        ref_obs_df = extract_pd_bins(prepared.reference, day_bins)

        results["BS_ref"] = calculate_brier_score(ref_obs_df) if BS else None
        results["RPS_ref"] = calculate_rps(ref_obs_df) if RPS else None
        results["AUC_ref"] = calculate_auc(ref_obs_df) if AUC else None


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

