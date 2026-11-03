"""Every schema key is read by a RunConfig projection (design docs/review/runconfig_design.md 2.3), so that a new key
fails here until a projection claims it, and no second list of keys exists.

`RunConfig.from_flat` is given a full flat record (every schema key, absent ones as None) in a dict that notes which
projection (`_project_*` function of hcwf.config.model) reads each key. The hand-written projections own disjoint keys;
the three that delegate to existing code (the onset setup, the units specs and the blend recipe) may also read keys
another projection owns (region, obs_day_boundary, ...), as that code does.
"""
import sys
from collections import defaultdict

import pytest

from hcwf.config.build import resolve_flat
from hcwf.config.model import RunConfig
from hcwf.lib import config_schema as schema
from test_oracle_config import configs, world  # noqa: F401  (fixtures)

RUN_KEYS = ("accumulation_detected", "config_file", "config_sources", "dataset_report", "lead_window",
            "onset_lead_lines", "run_paths", "cache_dir", "short_names")
"""The keys the resolve steps add to the schema's (design 1.1)."""
DELEGATES = {"_project_onset", "_project_units", "_project_blend"}


class Recording(dict):
    """A flat record that notes which projection reads each key (a copy with dict() does not count as a read)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.claims = defaultdict(set)

    def _note(self, key):
        frame = sys._getframe(2)
        while frame is not None:
            name = frame.f_code.co_name
            if name.startswith("_project_") and frame.f_globals["__name__"] == "hcwf.config.model":
                self.claims[key].add(name)
                return
            frame = frame.f_back
        self.claims[key].add("<elsewhere>")

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)


def _full(dic):
    """dic with every schema key present (None when absent): what a config that sets every key would give."""
    return {**{key: None for key in schema.BY_KEY}, **dic}


ONSET_SETUP_KEYS = {"onset_definition", "onset_override", "onset_rule", "onset_season", "threshold_years", "thresholds",
                    "model_lead_label", "check_alignment", "dry_spell_check", "forecast_history", "threshold_source",
                    "onset_backend"}
"""The keys only OnsetSetup holds (onset_setup reads them); the units_* and blend_* keys are only read by their delegates too."""


def test_every_schema_key_and_run_key_is_claimed_by_one_projection(configs):
    from hcwf.lib import onset_config

    dic = resolve_flat(configs["model_reference"])
    assert set(RUN_KEYS) <= set(dic)
    onset_config._cache.clear()                                   # a cached setup would read no keys
    values = Recording(_full(dic))
    RunConfig.from_flat(values)
    claims = values.claims
    outside = {k: sorted(ps) for k, ps in claims.items() if "<elsewhere>" in ps}
    assert not outside, f"read outside a _project_* function: {outside}"
    unclaimed = [k for k in [*schema.BY_KEY, *RUN_KEYS] if k not in claims]
    assert not unclaimed, f"no projection of hcwf/config/model.py reads {unclaimed}: add each to the spec that holds it"
    shared = {k: sorted(ps - DELEGATES) for k, ps in claims.items() if len(ps - DELEGATES) > 1}
    assert not shared, f"keys read by several hand-written projections: {shared}"
    only_delegated = {k for k, ps in claims.items() if ps <= DELEGATES}
    assert {k for k in only_delegated if not k.startswith(("blend_", "units_"))} <= ONSET_SETUP_KEYS, only_delegated


def test_a_key_outside_the_schema_is_not_read(configs):
    values = Recording(_full({**resolve_flat(configs["model_reference"]), "extra_key": 1}))
    RunConfig.from_flat(values)
    assert "extra_key" not in values.claims
