"""steps_needed: how many forecast lead days the loader must keep."""

import pytest

from hcwf.io.forecasts import steps_needed
from hcwf.config.onset import onset_setup


@pytest.mark.parametrize(("definition", "needed"), [("icpac", 30 + 21), ("moron_robertson_mok", 30 + 4)])
def test_the_loader_keeps_every_lead_day_a_definition_reads(definition, needed):
    """With a definition, the last lead read is the farthest day any requirement or veto reaches from the last candidate."""
    cfg = dict(onset_definition=definition, obs_day_boundary="00 UTC, labelled by start",
               threshold_years=(2000, 2010), max_forecast_day=30)
    assert steps_needed(cfg["max_forecast_day"], onset_setup(cfg)) == needed
