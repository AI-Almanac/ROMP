"""`hcwf-run --check` reads a numeric lead axis exactly as the run does (one implementation)."""
import numpy as np
import xarray as xr

from hcwf.lib import check
from hcwf.onset import ingest


def test_check_has_no_second_lead_reader_and_hours_axis_is_read_as_hours():
    """Leads 24, 48, ... without units are hours for the run (not 24-hour days, as the old check assumed)."""
    assert not hasattr(check, "_step_hours")
    ds = xr.Dataset(coords={"step": np.array([24.0, 48.0, 72.0])})
    assert ingest.lead_hours(ds, "step").tolist() == [24.0, 48.0, 72.0]
    days = xr.Dataset(coords={"step": np.array([1.0, 2.0, 3.0])})
    assert ingest.lead_hours(days, "step").tolist() == [24.0, 48.0, 72.0]


def test_panel_portrait_error_compiles_without_escape_warnings():
    import ast
    import hcwf.graphics.panel_portrait_error as module
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with open(module.__file__) as fh:
            ast.parse(fh.read())
