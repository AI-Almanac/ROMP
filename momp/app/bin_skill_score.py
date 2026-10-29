"""Entry point computing binned and overall BSS, RPSS and AUC skill scores and their plots for ensemble forecasts."""
from dataclasses import asdict
from itertools import product
import copy

from momp.metrics.skill import create_score_results, prepare_score_cache
from momp.graphics.heatmap import create_heatmap
from momp.graphics.reliability import plot_reliability_diagram
from momp.graphics.panel_portrait_skill import panel_portrait_bss_auc
from momp.graphics.panel_bar_skill import panel_bar_bss_rpss_auc
from momp.io.output import save_score_results
from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
from momp.lib.loader import get_cfg, get_setting
from momp.utils.printing import tuple_to_str
from momp.io.dict import select_key_at_level
from momp.lib.control import filter_bins_in_window
from collections import defaultdict


def skill_score_in_bins(cfg=None, setting=None):
    """Compute and plot binned and overall BSS, RPSS and AUC skill scores for each case of an ensemble forecast.

    Does nothing unless the configuration is probabilistic. If cfg or setting is
    None, the global configuration is loaded.
    """

    if cfg is None:
        cfg = get_cfg()
    if setting is None:
        setting = get_setting()

    # only execute for ensemble forecasts
    if not cfg.probabilistic:
        return

    result_overall = defaultdict(dict)
    result_binned = defaultdict(dict)
    score_input_cache = {}

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window "
                f"{case.verification_window}, case: {case.case_name}")

        full_day_bins = cfg.day_bins
        day_bins_filtered = filter_bins_in_window(case.day_bins, case.verification_window)

        cache_key = (
            case.model,
            tuple(case.years) if case.years else None,
            tuple(case.years_clim) if case.years_clim else None,
            case.obs_dir if hasattr(case, "obs_dir") else setting.obs_dir,
            setting.obs_file_pattern,
            case.obs_var,
            case.thresh_file,
            case.thresh_var,
            case.wet_threshold,
            setting.date_filter_year,
            tuple(setting.init_days) if setting.init_days else None,
            tuple(setting.start_date) if setting.start_date else None,
            tuple(setting.end_date) if setting.end_date else None,
            case.model_dir,
            case.model_var,
            case.file_pattern,
            case.unit_cvt,
            tuple(case.members) if case.members else None,
            case.wet_init,
            case.wet_spell,
            case.dry_spell,
            case.dry_threshold,
            case.dry_extent,
            setting.fallback_date,
            case.earliest_onset_date,
            case.onset_percentage_threshold,
            case.max_forecast_day,
            tuple(full_day_bins),
            case.ref_model,
            setting.ref_model_dir,
            case.ref_model_var,
            setting.ref_model_file_pattern,
            setting.ref_model_unit_cvt,
            setting.parallel,
        )

        if cache_key not in score_input_cache:
            prep_case = copy.copy(case)
            prep_case.day_bins = full_day_bins
            prep_cfg = {**asdict(setting), **asdict(prep_case)}
            score_input_cache[cache_key] = prepare_score_cache(**prep_cfg)

        case.day_bins = day_bins_filtered
        case_cfg = {**asdict(setting), **asdict(case)}
        case_cfg.update(score_input_cache[cache_key])

        import time
        start = time.perf_counter()

        # Create bin skill score metrics
        score_results = create_score_results(**case_cfg)

        end = time.perf_counter()
        print(f"\n\n\n skill score Execution time: {end - start:.4f} seconds\n\n\n")

        # save score results as csv file
        if case_cfg['save_csv_score']:
            binned_data, overall_scores = save_score_results(score_results, **case_cfg)

        window_str = tuple_to_str(case.verification_window)
        result_binned[case.model][window_str] = binned_data
        result_overall[case.model][window_str] = overall_scores

        # heatmap plot
        if case_cfg['plot_heatmap_bss_auc']:
            create_heatmap(score_results, **case_cfg)

        # reliability plot
        if case_cfg['plot_reliability']:
            plot_reliability_diagram(score_results["forecast_obs_df"], **case_cfg)


    max_forecast_day = cfg.max_forecast_day

    if 2 > 3:
        import pickle
        import os
        fout = os.path.join(cfg.dir_out,f"combi_binned_skill_scores_{max_forecast_day}day.pkl")
        with open(fout, "wb") as f:
            pickle.dump(result_binned, f)

        fout = os.path.join(cfg.dir_out,f"combi_overall_skill_scores_{max_forecast_day}day.pkl")
        with open(fout, "wb") as f:
            pickle.dump(result_overall, f)

    from pprint import pprint

    # panel heatmap plot for binned BSS and AUC
    if case_cfg['plot_panel_heatmap_skill']:
        for verification_window in cfg.verification_window_list:
            window_str = tuple_to_str(verification_window)
            # extract dict for specific window only
            result_binned_window = select_key_at_level(result_binned, 2, window_str)
            print("\n\n\n result_binned_window = ", result_binned_window)
            pprint(result_binned_window)
            panel_portrait_bss_auc(result_binned_window, verification_window, **vars(cfg))

    # bar plot for BSS, RPSS, AUC in window
    if case_cfg['plot_bar_bss_rpss_auc']:
        for verification_window in cfg.verification_window_list:
            window_str = tuple_to_str(verification_window)
            result_overall_window = select_key_at_level(result_overall, 2, window_str)
            print("\n\n\n result_overall_window = ", result_overall_window)
            panel_bar_bss_rpss_auc(result_overall_window, verification_window, **vars(cfg))

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    skill_score_in_bins()
