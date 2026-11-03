"""Entry point computing binned and overall BSS, RPSS and AUC skill scores and their plots for ensemble forecasts."""
from dataclasses import replace

from hcwf.metrics.skill import create_score_results, prepare_pairs
from hcwf.graphics.heatmap import create_heatmap
from hcwf.graphics.reliability import plot_reliability_diagram
from hcwf.graphics.panel_portrait_skill import panel_portrait_bss_auc
from hcwf.graphics.panel_bar_skill import panel_bar_bss_rpss_auc
from hcwf.io.output import save_score_results, score_tables
from hcwf.utils.printing import tuple_to_str
from hcwf.io.dict import select_key_at_level
from collections import defaultdict
import logging
from hcwf.utils.log import progress

log = logging.getLogger(__name__)


def skill_score_in_bins(rc):
    """Compute and plot binned and overall BSS, RPSS and AUC skill scores for each case of an ensemble forecast.

    rc is the resolved RunConfig. Does nothing (returns None) unless the run is probabilistic. Returns
    {"overall": DataFrame, "binned": DataFrame}: the score tables of every model and verification window (columns
    model and window first), whether or not CSV files are written.
    """

    # only execute for ensemble forecasts
    if not rc.evaluation.probabilistic:
        return

    spec, output = rc.figure_spec(), rc.output
    result_overall = defaultdict(dict)
    result_binned = defaultdict(dict)
    overall_frames, binned_frames = [], []
    prepared = {}   # the pair tables of each model: everything else the tables depend on is fixed within a run

    for case in progress(list(rc.cases()), desc="cases"):
        model, window = case.model.name, case.window.span

        log.debug(f"{'='*50}")
        log.info(f"processing {model} onset evaluation for verification window "
                f"{window}, case: {case.name}")

        # the pair tables hold every day bin; each window then reads the bins inside it
        if model not in prepared:
            prepared[model] = prepare_pairs(rc, case.model)

        day_bins = case.window.bins(rc.evaluation.day_bins)

        import time
        start = time.perf_counter()

        # Create bin skill score metrics
        score_results = create_score_results(case, prepared[model], rc.metrics,
                                             replace(rc.evaluation, day_bins=day_bins))

        end = time.perf_counter()
        log.debug(f"\n\n\n skill score Execution time: {end - start:.4f} seconds\n\n\n")

        window_str = tuple_to_str(window)
        overall_df, binned_df, binned_data, overall_scores = score_tables(score_results, model, window)
        overall_frames.append(overall_df.assign(model=model, window=window_str))
        binned_frames.append(binned_df.assign(model=model, window=window_str))

        # save score results as csv file
        if output.save_csv_score:
            save_score_results(score_results, case, rc.paths, output)

        result_binned[model][window_str] = binned_data
        result_overall[model][window_str] = overall_scores

        # heatmap plot
        if output.plot_heatmap_bss_auc:
            create_heatmap(score_results, case, spec)

        # reliability plot
        if output.plot_reliability:
            plot_reliability_diagram(score_results["forecast_obs_df"], case, spec, day_bins)


    import pprint

    # panel heatmap plot for binned BSS and AUC
    if output.plot_panel_heatmap_skill:
        for verification_window in (w.span for w in rc.evaluation.windows):
            window_str = tuple_to_str(verification_window)
            # extract dict for specific window only
            result_binned_window = select_key_at_level(result_binned, 2, window_str)
            log.debug("%s %s", "\n\n\n result_binned_window = ", result_binned_window)
            log.debug("%s", pprint.pformat(result_binned_window))
            panel_portrait_bss_auc(result_binned_window, verification_window, spec)

    # bar plot for BSS, RPSS, AUC in window
    if output.plot_bar_bss_rpss_auc:
        for verification_window in (w.span for w in rc.evaluation.windows):
            window_str = tuple_to_str(verification_window)
            result_overall_window = select_key_at_level(result_overall, 2, window_str)
            log.debug("%s %s", "\n\n\n result_overall_window = ", result_overall_window)
            panel_bar_bss_rpss_auc(result_overall_window, verification_window, spec)

    import pandas as pd

    def stacked(frames):
        """The per-case tables as one DataFrame with the model and window columns first."""
        table = pd.concat(frames, ignore_index=True)
        return table[["model", "window"] + [c for c in table.columns if c not in ("model", "window")]]

    return {"overall": stacked(overall_frames), "binned": stacked(binned_frames)}
