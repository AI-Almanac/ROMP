"""Plot the spatial map of observed onset day of year for a single year."""
from hcwf.io.input import observations_of, thresholds_of
from hcwf.stats.detect import detect_observed
from hcwf.graphics.func_map import spatial_metrics_map
import logging

log = logging.getLogger(__name__)


def spatial_onset(year, rc):
    """Detect observed onset for a year, map it as day of year, and return the day-of-year DataArray.

    rc is the run (a RunConfig): its observations, onset definition and thresholds.
    """
    thresh_slice = thresholds_of(rc, rc.models[0])

    log.debug("Loading observational rainfall data...")
    rainfall_ds = observations_of(year, rc)

    log.debug("Detecting observed onset...")
    onset_da = detect_observed(rainfall_ds, thresh_slice, year, rc)

    onset_doy = onset_da.dt.dayofyear.astype(float)
    onset_doy = onset_doy.where(~onset_da.isnull())


    spatial_metrics_map(onset_doy, "observation year "+str(year), rc.figure_spec(),
                    window=rc.evaluation.windows[0].span, fig=None, ax=None, figsize=(8, 6), cmap='YlOrRd', n_colors=10,
                    onset_plot=True, cbar_ssn=True, domain_mask=True, polygon_only=False,
                    show_ylabel=True, title="onset day", rect_box=True)

    return onset_doy
