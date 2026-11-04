"""Entry point computing binned and overall BSS, RPSS and AUC skill scores and their plots for probabilistic (ensemble) forecasts.

`skill_score_in_bins` loops over the cases of a run: it prepares the pairs of each model (`hcwf.scoring.pairs`), scores
them (`hcwf.metrics.skill`), builds the tables (`hcwf.metrics.tables`) and writes or plots them. The spatial
counterpart is `hcwf.scoring.deterministic`.
"""
from dataclasses import replace

from hcwf.metrics.skill import create_score_results
from hcwf.scoring.pairs import prepare_pairs
from hcwf.io.output import save_score_results
from hcwf.metrics.tables import score_tables
from hcwf.naming import tuple_to_str
from collections import defaultdict
import logging
from hcwf.log import progress

__all__ = ["skill_score_in_bins"]

log = logging.getLogger(__name__)


def select_key_at_level(d, level, key):
    """Select a key at a given nesting level in a nested dict and remove that level from the returned dict.

    Levels count from 1: ``level=1`` is the top level and ``level=2`` the second one. Branches that have nothing under
    ``key`` are left out of the result.

    Parameters
    ----------
    d : dict
        The nested dict.
    level : int
        The nesting level of the key to select.
    key : object
        The key to select at that level.

    Returns
    -------
    dict
        The selected subtree (``{}`` if ``key`` is missing) for ``level=1``, otherwise a dict of the top-level keys that
        have a non-empty subtree.
    """
    if level == 1:
        # drop this level: return the selected subtree
        return d.get(key, {})

    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub = select_key_at_level(v, level - 1, key)
            if sub:
                out[k] = sub
    return out


def skill_score_in_bins(rc):
    """Compute and plot binned and overall BSS, RPSS and AUC skill scores for each case of an ensemble forecast.

    Does nothing unless the run is probabilistic. The score tables are returned whether or not CSV files are written;
    what is saved and plotted follows the ``output`` flags of the run.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.

    Returns
    -------
    dict of str to pandas.DataFrame or None
        ``{"overall": ..., "binned": ...}``, the score tables of every model and verification window (columns ``model``
        and ``window`` first); None when the run is not probabilistic.
    """
    # only execute for ensemble forecasts
    if not rc.evaluation.probabilistic:
        return

    spec, output = rc.figure_spec(), rc.output
    result_overall = defaultdict(dict)
    result_binned = defaultdict(dict)
    overall_frames, binned_frames = [], []
    prepared = {}   # the pairs of each model: everything else the tables depend on is fixed within a run

    for case in progress(list(rc.cases()), desc="cases"):
        model, window = case.model.name, case.window.span

        log.debug(f"{'='*50}")
        log.info(f"processing {model} onset evaluation for verification window "
                f"{window}, case: {case.name}")

        # the pairs hold every day bin; each window then reads the bins inside it
        if model not in prepared:
            prepared[model] = prepare_pairs(rc, case.model)

        day_bins = case.window.bins(rc.evaluation.day_bins)

        import time
        start = time.perf_counter()

        # Create bin skill score metrics
        score_results = create_score_results(prepared[model], rc.metrics,
                                             replace(rc.evaluation, day_bins=day_bins))

        end = time.perf_counter()
        log.debug(f"\n\n\n skill score Execution time: {end - start:.4f} seconds\n\n\n")

        window_str = tuple_to_str(window)
        overall_df, binned_df, binned_data, overall_scores = score_tables(score_results)
        overall_frames.append(overall_df.assign(model=model, window=window_str))
        binned_frames.append(binned_df.assign(model=model, window=window_str))

        # save score results as csv file
        if output.save_csv_score:
            save_score_results(score_results, case, rc.paths, output)

        result_binned[model][window_str] = binned_data
        result_overall[model][window_str] = overall_scores

        # heatmap plot
        if output.plot_heatmap_bss_auc:
            from hcwf.figures.heatmap import create_heatmap   # figures load matplotlib only when asked for

            create_heatmap(score_results, case, spec)

        # reliability plot
        if output.plot_reliability:
            from hcwf.figures.reliability import plot_reliability_diagram

            plot_reliability_diagram(score_results["forecast_pairs"], case, spec)


    import pprint

    # panel heatmap plot for binned BSS and AUC
    if output.plot_panel_heatmap_skill:
        from hcwf.figures.panel_portrait_skill import panel_portrait_bss_auc

        for verification_window in (w.span for w in rc.evaluation.windows):
            window_str = tuple_to_str(verification_window)
            # extract dict for specific window only
            result_binned_window = select_key_at_level(result_binned, 2, window_str)
            log.debug("%s %s", "\n\n\n result_binned_window = ", result_binned_window)
            log.debug("%s", pprint.pformat(result_binned_window))
            panel_portrait_bss_auc(result_binned_window, verification_window, spec)

    # bar plot for BSS, RPSS, AUC in window
    if output.plot_bar_bss_rpss_auc:
        from hcwf.figures.panel_bar_skill import panel_bar_bss_rpss_auc

        for verification_window in (w.span for w in rc.evaluation.windows):
            window_str = tuple_to_str(verification_window)
            result_overall_window = select_key_at_level(result_overall, 2, window_str)
            log.debug("%s %s", "\n\n\n result_overall_window = ", result_overall_window)
            panel_bar_bss_rpss_auc(result_overall_window, verification_window, spec)

    return {"overall": _stacked(overall_frames), "binned": _stacked(binned_frames)}


def _stacked(frames):
    """Stack the per-case tables into one DataFrame with the ``model`` and ``window`` columns first.

    Parameters
    ----------
    frames : list of pandas.DataFrame
        The per-case tables, each with ``model`` and ``window`` columns.

    Returns
    -------
    pandas.DataFrame
        The rows of all ``frames`` with a fresh index.
    """
    import pandas as pd

    table = pd.concat(frames, ignore_index=True)
    return table[["model", "window"] + [c for c in table.columns if c not in ("model", "window")]]
