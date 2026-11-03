"""The stitched forecast search (forecast_history observed) honours the effective missing policy, as the plain one does."""
from __future__ import annotations

import numpy as np
import pytest

from hcwf.onset import engine, explain
from hcwf.onset.model import Status
from test_onset_engine import JJAS, TWO_WET, day_index, definition
from test_onset_forecast_layer import case, hist_with

BACKENDS = ["reference", "numba"]


def _stitched(flagged: bool):
    """A history with two wet days (flagged missing in the history when `flagged`) and a forecast whose first two leads
    are wet and flagged missing when `flagged`: the amounts are in the data either way."""
    values = hist_with(["2000-05-30", "2000-05-31"])
    fc_values = np.zeros((1, 1, 1, 12), dtype=np.int32)
    fc_values[..., :2] = 100
    hist, fc = case(values, fc_values, ["2000-06-10"])
    if flagged:
        hist.missing[..., day_index(hist, "2000-05-30"):day_index(hist, "2000-05-31") + 1] = True
        fc.missing[..., :2] = True
    return hist, fc


def _detect(hist, fc, policy, backend):
    return engine.detect_forecast(definition(TWO_WET, JJAS), fc, history=hist, backend=backend, missing=policy)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_policy_ignores_the_flags_of_history_and_forecast(backend):
    """Under zero a flagged day is judged as the rain it holds, exactly as if no day were flagged."""
    flagged, clean = _stitched(True), _stitched(False)
    zero = _detect(*flagged, "zero", backend)
    reference = _detect(*clean, "zero", backend)
    assert int(reference["status"].values.ravel()[0]) == Status.ONSET
    for name in ("status", "onset_lead", "onset_date"):
        np.testing.assert_array_equal(zero[name].values, reference[name].values)
    # the flags do matter under another policy, so the test can tell the two apart
    assert int(_detect(*flagged, "flag", backend)["status"].values.ravel()[0]) != Status.ONSET


def test_explain_follows_the_engine_under_zero():
    """The traced stitched search of a flagged series equals detect_forecast's under zero, and sees no missing day."""
    hist, fc = _stitched(True)
    defn = definition(TWO_WET, JJAS)
    ds = _detect(hist, fc, "zero", "reference")
    a = explain.trace_forecast(defn, fc, init="2000-06-10", lat=0.0, lon=0.0, history=hist, missing="zero")
    assert a.result.status == Status(int(ds["status"].values.ravel()[0]))
    assert not a.missing.any()
    flag = explain.trace_forecast(defn, fc, init="2000-06-10", lat=0.0, lon=0.0, history=hist, missing="flag")
    assert flag.missing.any()
