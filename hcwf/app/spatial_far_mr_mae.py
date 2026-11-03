"""Compute, save and plot spatial MAE, FAR and MR maps for deterministic and ensemble (probabilistic) forecast models."""
from hcwf.config.flat_bridge import case_kwargs
from hcwf.metrics.error import create_spatial_far_mr_mae
from hcwf.stats.benchmark import compute_metrics_multiple_years
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
    keyed by case combination. For an ensemble run the forecast onset is the ensemble onset (compute_metrics_multiple_years
    picks it from probabilistic); the climatology reference never reads a forecast.
    """

    results = {}

    for case in progress(list(rc.cases()), desc="cases"):
        combi = case.key

        log.debug(f"{'='*50}")
        log.info(f"processing {case.model.name} onset evaluation for verification window "
                f"{case.window.span}, case: {case.name}")

        case_cfg = case_kwargs(rc, case, "deterministic")

        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg)

        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        add_back_dated_map(spatial_metrics, metrics_df_dict, onset_da_dict)

        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)


        # Save spatial metrics to NetCDF
        if case_cfg["save_nc_spatial_far_mr_mae"]:

            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }

            save_metrics_to_netcdf(spatial_metrics, case_cfg, desc_dict=desc_dict)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg)


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
        case_cfg_ref = case_kwargs(rc, case, "deterministic")

        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)

        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        add_back_dated_map(spatial_metrics, metrics_df_dict, onset_da_dict)

        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if case_cfg_ref["save_nc_spatial_far_mr_mae"]:
            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }
            save_metrics_to_netcdf(spatial_metrics, case_cfg_ref, desc_dict=desc_dict)


        # make spatial metrics plot
        if case_cfg_ref['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg_ref)


    # save climatological onset to netcdf
    if case.model.name == 'climatology' and case_cfg_ref['save_nc_climatology']:
        save_climatology_onset(onset_da_dict, case_cfg_ref, rc.evaluation.climatology_years)

    # spatial map of climatology onset day
    if case_cfg_ref['plot_climatology_onset']:
        plot_spatial_climatology_onset(onset_da_dict, **case_cfg_ref)

    # panel portrait plot of mae, far, mr
    if case_cfg_ref['plot_panel_heatmap_error']:
       panel_portrait_mae_far_mr(results, **case_cfg_ref)

    return results
