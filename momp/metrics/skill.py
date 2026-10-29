"""Compute forecast and reference-model BS, RPS, AUC and derived skill scores from forecast-observation pairs."""
from momp.stats.climatology import compute_climatological_onset_dataset
from momp.stats.bins import multi_year_forecast_obs_pairs, multi_year_climatological_forecast_obs_pairs
from momp.stats.conditional_climatology import CLIMATOLOGY_REFERENCES, is_conditional_climatology
from momp.stats.conditional_climatology import multi_year_conditional_climatological_forecast_obs_pairs
from momp.stats.conditional_climatology import parallel_conditional_climatological_forecast_obs_pairs
from momp.stats.score import calculate_brier_score, calculate_auc, calculate_rps, calculate_brier_score_climatology, calculate_auc_climatology, calculate_skill_scores
from momp.utils.practical import restore_args
from momp.io.dict import extract_pd_bins
from momp.stats.parallel import parallel_climatological_forecast_obs_pairs
from momp.stats.parallel import parallel_forecast_obs_pairs
from momp.stats.parallel import parallel_climatological_onset_dataset


def reference_model_kwargs(kwargs):
    """Copy kwargs with the reference model's name, directory, variable, file pattern, unit conversion and dims.

    The forecast model's own model_dims are dropped (the reference archive is
    read by name inference) unless kwargs carries ref_model_dims.
    """
    return {**kwargs,
            'model': kwargs['ref_model'],
            'model_dir': kwargs['ref_model_dir'],
            'model_var': kwargs['ref_model_var'],
            'file_pattern': kwargs['ref_model_file_pattern'],
            'unit_cvt': kwargs['ref_model_unit_cvt'],
            'model_dims': kwargs.get('ref_model_dims')}


def _climatology_pairs(clim_onset, **kwargs):
    """Pair tables of the climatology reference (conditional or not) for all years, in parallel when kwargs["parallel"] is set."""
    if is_conditional_climatology(kwargs["ref_model"]):
        return (parallel_conditional_climatological_forecast_obs_pairs(clim_onset, **kwargs) if kwargs.get("parallel")
                else multi_year_conditional_climatological_forecast_obs_pairs(clim_onset, **kwargs))
    return (parallel_climatological_forecast_obs_pairs(clim_onset, **kwargs) if kwargs.get("parallel")
            else multi_year_climatological_forecast_obs_pairs(clim_onset, **kwargs))


def _pairs(**kwargs):
    """Forecast-observation pairs for all years, in parallel when kwargs["parallel"] is set."""
    return parallel_forecast_obs_pairs(**kwargs) if kwargs.get("parallel") else multi_year_forecast_obs_pairs(**kwargs)


def prepare_score_cache(**kwargs):
    """Build the forecast-observation pair tables (and the reference's) shared across cases.

    Returns a dict with keys _forecast_obs_df_all, _ref_obs_df_all (reference
    model pairs, built from the reference model's own data; None for
    climatology and conditional_climatology), _clim_onset and _climatology_obs_df_all, which
    create_score_results reuses instead of recomputing.
    """
    parallel = kwargs.get('parallel')
    ref_model = kwargs["ref_model"]

    forecast_obs_df_all = (
        parallel_forecast_obs_pairs(**kwargs)
        if parallel
        else multi_year_forecast_obs_pairs(**kwargs)
    )

    clim_onset = None
    climatology_obs_df_all = None
    ref_obs_df_all = None

    if ref_model not in CLIMATOLOGY_REFERENCES:
        ref_obs_df_all = _pairs(**reference_model_kwargs(kwargs))

    if ref_model in CLIMATOLOGY_REFERENCES:
        clim_onset = (
            parallel_climatological_onset_dataset(**kwargs)
            if parallel
            else compute_climatological_onset_dataset(**kwargs)
        )
        climatology_obs_df_all = _climatology_pairs(clim_onset, **kwargs)

    return {
        "_forecast_obs_df_all": forecast_obs_df_all,
        "_ref_obs_df_all": ref_obs_df_all,
        "_clim_onset": clim_onset,
        "_climatology_obs_df_all": climatology_obs_df_all,
    }


def create_score_results(*, BS, RPS, AUC, skill_score,
                         ref_model, ref_model_dir, ref_model_var, ref_model_file_pattern, ref_model_unit_cvt,
                         years, years_clim, obs_dir, obs_file_pattern, obs_var,
                         thresh_file, thresh_var, wet_threshold,
                         date_filter_year, init_days, start_date, end_date,
                         model_dir, model_var, unit_cvt, file_pattern,
                         wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, fallback_date, earliest_onset_date,
                         members, onset_percentage_threshold, max_forecast_day, day_bins, **kwargs):
    """Compute Brier score, RPS and AUC (as flagged by BS, RPS, AUC) for the forecast model and its reference, plus skill scores if skill_score is set.

    Uses cached pair tables from prepare_score_cache when present in kwargs. The
    reference is climatology if ref_model is "climatology", conditional climatology
    (P(onset on lead k | no onset by the init), kwargs "climatology_cv") if it is
    "conditional_climatology", otherwise another model.
    Returns a dict with forecast_obs_df, climatology_obs_df, BS, RPS, AUC, their
    *_ref counterparts, skill_score and (if skill_score) skill_results.
    """
    kwargs = restore_args(create_score_results, kwargs, locals())

    parallel = kwargs.get('parallel')

    results = {}

    print("\n1. Processing forecast model...")

    import time
    start = time.perf_counter()

    # select bins without "Day before" and "Day After"
    forecast_obs_df_all = kwargs.get("_forecast_obs_df_all")
    if forecast_obs_df_all is None:
        if parallel:
            forecast_obs_df_all = parallel_forecast_obs_pairs(**kwargs)
        else:
            forecast_obs_df_all = multi_year_forecast_obs_pairs(**kwargs)
    forecast_obs_df = extract_pd_bins(forecast_obs_df_all, day_bins)

    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")

    results["forecast_obs_df"] = forecast_obs_df

    results["BS"] = calculate_brier_score(forecast_obs_df) if BS else None
    results["RPS"] = calculate_rps(forecast_obs_df) if RPS else None
    results["AUC"] = calculate_auc(forecast_obs_df) if AUC else None


    print("\n2. Processing reference model...")

    results["BS_ref"], results["RPS_ref"], results["AUC_ref"] = None, None, None
    results['skill_score'] = None

    if ref_model in CLIMATOLOGY_REFERENCES:

        clim_onset = kwargs.get("_clim_onset")
        if clim_onset is None:
            if parallel:
                clim_onset = parallel_climatological_onset_dataset(**kwargs)
            else:
                clim_onset = compute_climatological_onset_dataset(**kwargs)

        import time
        start = time.perf_counter()

        climatology_obs_df_all = kwargs.get("_climatology_obs_df_all")
        if climatology_obs_df_all is None:
            climatology_obs_df_all = _climatology_pairs(clim_onset, **{**kwargs, "ref_model": ref_model})
        climatology_obs_df = extract_pd_bins(climatology_obs_df_all, day_bins)

        end = time.perf_counter()
        print(f"clim_forecast_obs_paris Execution time: {end - start:.4f} seconds")

        results["climatology_obs_df"] = climatology_obs_df

        if BS:
            brier_ref = calculate_brier_score_climatology(climatology_obs_df)
            results["BS_ref"] = brier_ref
        else:
            results["BS_ref"] = None

        if RPS:
            rps_ref = calculate_rps(climatology_obs_df)
            results["RPS_ref"] = rps_ref
        else:
            results["RPS_ref"] = None

        if AUC:
            auc_ref = calculate_auc_climatology(climatology_obs_df)
            results["AUC_ref"] = auc_ref
        else:
            results["AUC_ref"] = None

    else:
        results["climatology_obs_df"] = None

        # the reference model's own data and settings, never the forecast model's
        ref_obs_df_all = kwargs.get("_ref_obs_df_all")
        if ref_obs_df_all is None:
            ref_obs_df_all = _pairs(**reference_model_kwargs(kwargs))
        ref_obs_df = extract_pd_bins(ref_obs_df_all, day_bins)

        if BS:
            brier_ref = calculate_brier_score(ref_obs_df)
            results["BS_ref"] = brier_ref
        else:
            results["BS_ref"] = None

        if RPS:
            rps_ref = calculate_rps(ref_obs_df)
            results["RPS_ref"] = rps_ref
        else:
            results["RPS_ref"] = None

        if AUC:
            auc_ref = calculate_auc(ref_obs_df)
            results["AUC_ref"] = auc_ref
        else:
            results["AUC_ref"] = None


    print("results[BS]  =  ", results["BS"])
    print("results[BS_ref]  =  ", results["BS_ref"])
    print("results[RPS]  =  ", results["RPS"])
    print("results[RPS_ref]  =  ", results["RPS_ref"])
    if skill_score:

        skill_results = calculate_skill_scores(
        results["BS"], results["RPS"],
        results["BS_ref"], results["RPS_ref"]
        )

        results["skill_results"] = skill_results

    return results
