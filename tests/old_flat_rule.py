"""The rule of the old flat onset keys as the onset definition `hcwf-run convert` writes for it, for tests that used
those keys: wet days of 1 mm, a 3-day spell above 20 mm, a 7-day dry run (checked 10 days when dry_extent says so)."""
from hcwf.lib.convert import flat_onset_to_rule


def old_rule(**changes):
    """Config keys (onset_rule, thresholds, day boundary, lead label) for the old flat rule, keys in `changes` replacing
    its values (wet_init, wet_threshold, wet_spell, dry_spell, dry_extent, thresh_file, earliest_onset_date, ...)."""
    flat = dict(wet_init=1, wet_spell=3, dry_spell=7, dry_extent=0, wet_threshold=20,
                start_date=(2020, 5, 1), end_date=(2020, 7, 31))
    flat.update(changes)
    rule, files = flat_onset_to_rule(flat)
    return dict(onset_rule=rule, thresholds=files or None, obs_day_boundary="00 UTC, labelled by start",
                model_lead_label="end", start_date=flat["start_date"],
                end_date=flat["end_date"])
