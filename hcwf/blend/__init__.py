"""Blending methods: each turns the shared stages' features into onset probabilities over the week bins.

* `hcwf.blend.logit`: the multinomial logistic blend of formula features.
* `hcwf.blend.random_forest`: the CRPS random forest over each model's daily rainfall and the climatology, giving
  daily probabilities too.
* `hcwf.blend.mme`: the multi-model ensemble, a weighted mixture of other predictions.
* `hcwf.blend.calibration`: Platt recalibration of week-bin probabilities.
* `hcwf.blend.design`: formulas and design matrices from arrays.
* `hcwf.blend.predictions`: a model's probabilities with the samples' keys.
* `hcwf.blend.trained`: `hcwf.blend.trained.TrainedBlend`, the operational file (also reachable as
  ``hcwf.blend.TrainedBlend``).
* `hcwf.blend.forecast`: applying a trained blend to live forecast files.
"""


def __getattr__(name):
    """Return `hcwf.blend.trained.TrainedBlend`, imported on first use (the blend modules load lazily).

    Parameters
    ----------
    name : str
        The attribute asked for; only ``"TrainedBlend"`` exists.

    Returns
    -------
    type
        The class `hcwf.blend.trained.TrainedBlend`.

    Raises
    ------
    AttributeError
        `name` is not ``"TrainedBlend"``.
    """
    if name == "TrainedBlend":
        from hcwf.blend.trained import TrainedBlend

        return TrainedBlend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
