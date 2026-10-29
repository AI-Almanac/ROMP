"""onset_steps_needed: how many forecast lead days the loader must keep."""

import pytest

from momp.io.input import onset_steps_needed


@pytest.mark.parametrize(
    ("max_forecast_day", "wet_spell", "dry_extent", "needed"),
    [(30, 3, 0, 32), (30, 3, 7, 37), (30, 1, 0, 30), (30, 3, None, None)],
)
def test_the_loader_keeps_every_lead_day_the_flat_keys_read(max_forecast_day, wet_spell, dry_extent, needed):
    """For the old flat keys, onset_steps_needed returns the lead days the old detection read, or None without dry_extent."""
    assert (
        onset_steps_needed(max_forecast_day=max_forecast_day, wet_spell=wet_spell, dry_extent=dry_extent)
        == needed
    )


@pytest.mark.parametrize(("definition", "needed"), [("icpac", 30 + 21), ("moron_robertson_mok", 30 + 4)])
def test_the_loader_keeps_every_lead_day_a_definition_reads(definition, needed):
    """With a definition, the last lead read is the farthest day any requirement or veto reaches from the last candidate."""
    cfg = dict(onset_definition=definition, obs_day_boundary="00 UTC, labelled by start",
               threshold_years=(2000, 2010), onset_legacy=False, max_forecast_day=30, wet_spell=3, dry_extent=0)
    assert onset_steps_needed(**cfg) == needed
