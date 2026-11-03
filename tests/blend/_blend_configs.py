"""Small blending configs for the evaluation and training tests, as hcwf's recipe and the legacy oracle's spec.

`blend_config` is onset_blending's tests/_configs.py (a BlendConfig whose models have a diff rain predictor and whose
files are never read); `recipe_of` is the hcwf Recipe it converts to (tests/blend/_recipes.py); `legacy_spec` the legacy
stage spec the oracle script reads. `dissemination_ids` reads a unit table as hcwf reads a dissemination list.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from _recipes import blending_config, oracle, recipe_of  # noqa: F401

from hcwf.spatial.units import read_id_list

oracle()
from tests.blend._oracle import specs  # noqa: E402  (needs the checkout on sys.path)


def blend_config(evaluation: dict, *, models=("fc",), bins: int = 2, windows=(), dissemination_id_col=None,
                 grid_ids=None):
    """A BlendConfig whose models have a diff rain predictor (`diff_<model>_qx`); its files are never read."""
    domain = {}
    if dissemination_id_col is not None:
        domain["dissemination"] = {"file": "cells.csv", "id_col": dissemination_id_col}
    if grid_ids is not None:
        domain["grid_ids"] = grid_ids
    return blending_config({
        "name": "test",
        "domain": domain,
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": m, "files": {"folder": m}, "rain_predictors": [{"agg": "diff"}]} for m in models],
        "onset": {"threshold": {"mm": 20}},
        "climatology": {"windows": [list(w) for w in windows]},
        "features": {"bins": {"count": bins}},
        "evaluation": evaluation,
    })


def legacy_spec(config) -> dict:
    return specs.cv_spec(config)


def dissemination_ids(table: pd.DataFrame, id_col: str | None = None) -> np.ndarray:
    """The ids of a unit table (as read from a CSV), the way hcwf reads a dissemination list."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cells.csv"
        table.to_csv(path, index=False)
        return np.asarray(read_id_list(path, id_col), dtype=object)
