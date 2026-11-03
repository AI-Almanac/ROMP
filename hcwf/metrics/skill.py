"""Compute forecast and reference-model BS, RPS, AUC and derived skill scores from forecast-observation pairs."""
from typing import Any, NamedTuple

import pandas as pd

from hcwf.config.model import Case, Selection
from hcwf.stats.climatology import climatological_onset_dataset
from hcwf.stats.bins import climatological_pairs, start_pairs
from hcwf.stats.conditional_climatology import CLIMATOLOGY_REFERENCES, conditional_pairs, is_conditional_climatology
from hcwf.stats.score import calculate_brier_score, calculate_auc, calculate_rps, calculate_brier_score_climatology, calculate_auc_climatology, calculate_skill_scores
from hcwf.io.dict import extract_pd_bins
from hcwf.io import input as io_input
from hcwf.stats.parallel import concat_year_args
import logging

log = logging.getLogger(__name__)

class PreparedPairs(NamedTuple):
    """The pair tables a model's scores are made from, built once per model (`prepare_pairs`)."""

    forecast: pd.DataFrame
    """The model's forecast-observation pairs, every day bin."""
    reference: pd.DataFrame | None
    """The reference model's pairs, from the reference model's own data; None for a climatology reference."""
    clim_onset: Any
    """The climatology years' onsets (a climatology reference), else None."""
    climatology: pd.DataFrame | None
    """The climatology reference's pairs (conditional or not); None for a model reference."""


def _climatology_years(years, clim_onset, rc, case, select, day_bins):
    """The climatology reference's pair tables of `years` (the form `parallel.concat_year_args` calls)."""
    if is_conditional_climatology(rc.reference.model):
        return conditional_pairs(clim_onset, rc, case, select, years=years, day_bins=day_bins)
    return climatological_pairs(clim_onset, rc, years=years, day_bins=day_bins)


def _climatology_pairs(clim_onset, rc, case, select):
    """Pair tables of the climatology reference (conditional or not) for all years, in worker processes unless parallel is off."""
    return concat_year_args(_climatology_years, rc.evaluation.years,
                            (clim_onset, rc, case, select, rc.evaluation.day_bins), rc.parallel)


def _reference_model(rc, model):
    """The reference as a source to read when `model` is the forecast model it is scored against (the ensemble flow's rule,
    written once in RunConfig.reference_cases)."""
    return rc.reference_cases(like=model)[0].model


def prepare_pairs(rc, model, select=Selection()):
    """Build the forecast-observation pair tables (and the reference's) of `model` (ModelSpec) shared across its cases.

    Returns PreparedPairs: the model's pairs, the reference model's own pairs (None for climatology and
    conditional_climatology), the climatological onsets and the climatology pairs; create_score_results reuses them.

    With worker processes the model's years are sent off first and the climatological onsets are computed here
    while they run; the climatology pairs then go to the same workers.
    """
    case = Case(model, rc.evaluation.windows[0])
    climatology = rc.reference.model in CLIMATOLOGY_REFERENCES

    pending = start_pairs(rc, model, select)

    clim_onset = None
    climatology_obs_df_all = None
    ref_pending = None

    if not climatology:
        ref_pending = start_pairs(rc, _reference_model(rc, model), select)
        io_input.release_reference()
    else:
        clim_onset = climatological_onset_dataset(rc, case, select)
        io_input.release_reference()

    forecast_obs_df_all = pending.result()
    ref_obs_df_all = ref_pending.result() if ref_pending is not None else None

    if climatology:
        climatology_obs_df_all = _climatology_pairs(clim_onset, rc, case, select)

    return PreparedPairs(forecast_obs_df_all, ref_obs_df_all, clim_onset, climatology_obs_df_all)


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

