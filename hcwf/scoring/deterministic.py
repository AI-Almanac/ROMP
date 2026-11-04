"""Compute, save and plot spatial MAE, FAR and MR maps for deterministic and ensemble (probabilistic) forecast models.

The module holds the flow of the spatial scores of a run: for each case it gets the per-year metrics tables from
`hcwf.scoring.onset_tables`, makes the maps with `hcwf.metrics.error` and writes or plots them. The probabilistic
counterpart is `hcwf.scoring.probabilistic`.
"""
from hcwf.config.model import CLIMATOLOGY
from hcwf.metrics.error import create_spatial_far_mr_mae
from hcwf.scoring.onset_tables import compute_metrics
from hcwf.io.output import save_climatology_onset, save_metrics_to_netcdf
from hcwf.metrics.tables import add_back_dated_map
import logging
from hcwf.log import progress

log = logging.getLogger(__name__)


def set_nested(result_dict, keys, value):
    """Set a value in a nested dict, creating the intermediate dicts on the fly.

    Parameters
    ----------
    result_dict : dict
        The nested dict, changed in place.
    keys : sequence
        The keys from the top level down; the last one holds ``value``.
    value : object
        What to store.

    Returns
    -------
    dict
        ``result_dict``, the same object.
    """
    d = result_dict
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
    return result_dict


def spatial_far_mr_mae_map(rc):
    """Compute, save and plot spatial MAE, FAR and MR maps for each deterministic forecast case.

    Also evaluates the reference model (climatology or a user-specified model) when the run has one
    (``rc.reference.model``). For an ensemble run the forecast onset is the ensemble onset (`~hcwf.scoring.onset_tables.compute_metrics` picks it
    from the run's ``probabilistic`` setting); the climatology reference never reads a forecast. What is saved and
    plotted follows the ``output`` flags of the run.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.

    Returns
    -------
    dict
        The maps nested by case key (`hcwf.config.model.Case.key`): ``{key level: ... {"false_alarm_rate": ...}}``, each
        leaf a dict of str to `xarray.DataArray` as made by `hcwf.metrics.error.create_spatial_far_mr_mae`.
    """
    results = {}
    spec, output = rc.figure_spec(), rc.output
    ev, units = rc.evaluation, rc.observations.units

    for case in progress(list(rc.cases()), desc="cases"):
        combi = case.key

        log.debug(f"{'='*50}")
        log.info(f"processing {case.model.name} onset evaluation for verification window "
                f"{case.window.span}, case: {case.name}")

        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics(rc, case)

        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        add_back_dated_map(spatial_metrics, metrics_df_dict, onset_da_dict)

        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)


        # Save spatial metrics to NetCDF
        if output.save_nc_spatial_far_mr_mae:

            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }

            save_metrics_to_netcdf(spatial_metrics, case, rc.paths, output, ev, units, desc_dict=desc_dict)


        # make spatial metrics plot
        if output.plot_spatial_far_mr_mae:
            from hcwf.figures.maps import plot_spatial_metrics   # figures load matplotlib only when asked for

            plot_spatial_metrics(spatial_metrics, case, spec)


    # ------------------------------------------------------------------------
    # baseline metrics (climatology or user specified model)

    if not rc.reference.model:
        return results

    for case in progress(list(rc.reference_cases()), desc="cases"):
        combi = case.key
        log.debug(f"{'='*50}")
        log.info(f"processing {case.model.name} onset evaluation for verification window"
                f"{case.window.span}, case: {case.name}")

        # the reference's own data location, variable, pattern, unit and dims (the reference case holds them, even if
        # the forecast loop never ran)
        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics(rc, case)

        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        add_back_dated_map(spatial_metrics, metrics_df_dict, onset_da_dict)

        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if output.save_nc_spatial_far_mr_mae:
            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }
            save_metrics_to_netcdf(spatial_metrics, case, rc.paths, output, ev, units, desc_dict=desc_dict)


        # make spatial metrics plot
        if output.plot_spatial_far_mr_mae:
            from hcwf.figures.maps import plot_spatial_metrics   # figures load matplotlib only when asked for

            plot_spatial_metrics(spatial_metrics, case, spec)


    # save climatological onset to netcdf
    if case.model.name == CLIMATOLOGY and output.save_nc_climatology:
        save_climatology_onset(onset_da_dict, rc.paths, output, units, ev.climatology_years)

    # spatial map of climatology onset day
    if output.plot_climatology_onset:
        from hcwf.figures.onset_map import plot_spatial_climatology_onset

        plot_spatial_climatology_onset(onset_da_dict, spec, ev.climatology_years)

    # panel portrait plot of mae, far, mr
    if output.plot_panel_heatmap_error:
        from hcwf.figures.panel_portrait_error import panel_portrait_mae_far_mr

        panel_portrait_mae_far_mr(results, spec)

    return results
