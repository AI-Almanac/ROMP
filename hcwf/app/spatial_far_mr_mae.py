"""Compute, save and plot spatial MAE, FAR and MR maps for deterministic and ensemble (probabilistic) forecast models."""
from hcwf.metrics.error import create_spatial_far_mr_mae
from hcwf.stats.benchmark import compute_metrics
from hcwf.graphics.maps import plot_spatial_metrics
from hcwf.graphics.onset_map import plot_spatial_climatology_onset
from hcwf.graphics.panel_portrait_error import panel_portrait_mae_far_mr
from hcwf.io.output import add_back_dated_map, save_climatology_onset, save_metrics_to_netcdf
from hcwf.io.output import set_nested
import logging
from hcwf.utils.log import progress

log = logging.getLogger(__name__)


def spatial_far_mr_mae_map(rc):
    """Compute, save and plot spatial MAE, FAR and MR maps for each deterministic forecast case.

    Also evaluates the reference model (climatology or a user-specified model)
    when the run has one (rc.reference.model). rc is the resolved RunConfig. Returns a nested dict of spatial metrics
    keyed by case combination. For an ensemble run the forecast onset is the ensemble onset (compute_metrics
    picks it from probabilistic); the climatology reference never reads a forecast.
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
            plot_spatial_metrics(spatial_metrics, case, spec)


    # save climatological onset to netcdf
    if case.model.name == 'climatology' and output.save_nc_climatology:
        save_climatology_onset(onset_da_dict, rc.paths, output, units, ev.climatology_years)

    # spatial map of climatology onset day
    if output.plot_climatology_onset:
        plot_spatial_climatology_onset(onset_da_dict, spec, ev.climatology_years)

    # panel portrait plot of mae, far, mr
    if output.plot_panel_heatmap_error:
        panel_portrait_mae_far_mr(results, spec)

    return results
