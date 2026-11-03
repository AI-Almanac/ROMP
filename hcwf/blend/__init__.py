"""Blending methods: each turns the shared stages' features into onset probabilities over the week bins.

logit: the multinomial logistic blend of formula features.
random_forest: the CRPS random forest over each model's daily rainfall and the climatology, giving daily
probabilities too.
mme: the multi-model ensemble, a weighted mixture of other predictions.
calibration: Platt recalibration of week-bin probabilities.
design: formulas and design matrices from arrays. predictions: a model's probabilities with the samples' keys.
trained: TrainedBlend (hcwf.blend.trained), the operational file; forecast: applying it to live forecast files.
"""


def __getattr__(name):
    """`hcwf.blend.TrainedBlend`, imported on first use (the blend modules load lazily)."""
    if name == "TrainedBlend":
        from hcwf.blend.trained import TrainedBlend

        return TrainedBlend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
