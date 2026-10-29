"""Plot the spatial map of observed onset day of year for a single year."""
from momp.io.input import load_imd_rainfall
from momp.io.input import load_thresholds
from momp.stats.detect import detect_observed_onset
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.func_map import spatial_metrics_map


def spatial_onset(year, **kwargs):
    """Detect observed onset for a year, map it as day of year, and return the day-of-year DataArray.

    kwargs is the run configuration, in definition mode (onset_definition, ...) or legacy mode (wet_init,
    wet_spell, ... flat keys); the onset keys are resolved by detect_observed_onset, not named here.
    """
    thresh_slice = load_thresholds(**kwargs)

    print("Loading observational rainfall data...")
    rainfall_ds = load_imd_rainfall(year, **kwargs)

    print("Detecting observed onset...")
    onset_da = detect_observed_onset(rainfall_ds, thresh_slice, year, **kwargs)

    onset_doy = onset_da.dt.dayofyear.astype(float)
    onset_doy = onset_doy.where(~onset_da.isnull())


    kwargs.update({'verification_window': kwargs['verification_window_list'][0]})

    spatial_metrics_map(onset_doy, "observation year "+str(year),
                    fig=None, ax=None, figsize=(8, 6), cmap='YlOrRd', n_colors=10,
                    onset_plot=True, cbar_ssn=True, domain_mask=True, polygon_only=False,
                    show_ylabel=True, title="onset day", rect_box=True, **kwargs)

    return onset_doy


if __name__ == "__main__":
    from momp.lib.loader import get_cfg, get_setting

    cfg, setting = get_cfg(), get_setting()

    spatial_onset(2020, **vars(cfg))
