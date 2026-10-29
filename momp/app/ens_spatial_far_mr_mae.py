"""Compute, save and plot spatial MAE, FAR and MR maps for ensemble (probabilistic) forecast models."""
import os
from dataclasses import asdict
from itertools import product
import copy

from momp.metrics.error import create_spatial_far_mr_mae
from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.maps import plot_spatial_metrics
from momp.graphics.onset_map import plot_spatial_climatology_onset
from momp.graphics.panel_portrait_error import panel_portrait_mae_far_mr
from momp.io.output import add_back_dated_map, save_metrics_to_netcdf
from momp.io.output import set_nested
from momp.utils.printing import tuple_to_str_range

from momp.stats.benchmark import ens_compute_metrics_multiple_years
from momp.stats.parallel import parallel_metrics_multiple_years


def ens_spatial_far_mr_mae_map(cfg=None, setting=None):
    """Compute, save and plot spatial MAE, FAR and MR maps for each ensemble forecast case.

    Also evaluates the reference model (climatology or a user-specified model)
    when cfg.ref_model is set. Returns a nested dict of spatial metrics keyed by
    case combination, or None if the configuration is not probabilistic.
    """

    if cfg is None:
        cfg = get_cfg()
    if setting is None:
        setting = get_setting()

    if not cfg.probabilistic:
        return

    results = {}

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window \
                {case.verification_window}{case.case_name}")

        case_cfg = {**asdict(case), **asdict(setting)}

        # model-obs onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(ens=True, **case_cfg)
        else:
            metrics_df_dict, onset_da_dict = ens_compute_metrics_multiple_years(**case_cfg)

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

    if not cfg.ref_model:
        return results

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    layout_pool = iter_list(vars(cfg_ref))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        print(f"processing model onset evaluation for {case.case_name}")

        case_ref = {'model_dir': case_cfg['ref_model_dir'],
                    'model_var': case_cfg['ref_model_var'],
                    'file_pattern': case_cfg['ref_model_file_pattern'],
                    'unit_cvt': case_cfg['ref_model_unit_cvt']
                    }

        case.update(case_ref)

        case_cfg_ref = {**asdict(case), **asdict(setting)}

        from pprint import pprint
        pprint(case_cfg_ref)

        # model-obs onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(ens=True, **case_cfg_ref)
        else:
            metrics_df_dict, onset_da_dict = ens_compute_metrics_multiple_years(**case_cfg_ref)

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
            save_metrics_to_netcdf(spatial_metrics, case_cfg_ref, desc_dict=desc_dict)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg_ref)


    # save climatological onset to netcdf
    if case.model == 'climatology' and case_cfg['save_nc_climatology']:
        fout = os.path.join(case_cfg_ref['dir_out'], "climatology_onset_doy_{}.nc")
        fout = fout.format(tuple_to_str_range(case_cfg_ref['years_clim']))
        climatological_onset_doy = next(iter(onset_da_dict.values()))
        climatological_onset_doy.attrs["years"] = case.years_clim
        climatological_onset_doy.to_netcdf(fout)

    # spatial map of climatology onset day
    if case_cfg['plot_climatology_onset']:
        plot_spatial_climatology_onset(onset_da_dict, **case_cfg_ref)

    # panel portrait plot of mae, far, mr
    if case_cfg['plot_panel_heatmap_error']:
       panel_portrait_mae_far_mr(results, **case_cfg_ref)

    return results

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    import time
    start = time.perf_counter()

    results = ens_spatial_far_mr_mae_map()

    end = time.perf_counter()
    print(f"\n\n\n ens_spatial metrics Execution time: {end - start:.4f} seconds\n\n\n")
