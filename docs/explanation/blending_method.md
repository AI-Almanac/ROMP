# The blending method

A blend turns the onset climatology and the onset dates each forecast model produces into probabilities that the onset
falls in week 1, 2, 3, 4 or later after an issue date. Training cross-validates the blend against climatology baselines and
fits the operational model; applying a trained blend to a live forecast gives the probabilities for that issue date. The
steps are in the guide {doc}`../blending`.

## Blends

* The *logit blend* is a multinomial logistic regression over standardised formula features of the weekly samples.
* The *CRPS random forest* learns, from each model's daily rainfall and the conditional climatology, the lead day of the
  onset; it gives daily probabilities, and the week bins are their sums.
* The *multi-model ensemble* is a weighted mixture of other predictions; weekly probabilities can also be recalibrated
  (Platt).

## Cross-validation

The logit is cross-validated leaving one year out, either pooling every unit (global), within each unit (local) or within
each unit's cluster. The cross-validated probabilities are scored against the climatology baselines with the same
probabilistic scores as a benchmark ({doc}`scoring_and_references`).

## What a run is made of

A blend is the `blend:` section of the run file, used together with the rest of the config (data, onset definition, units,
years). Training produces a `trained.blend` file holding everything needed to apply it: the run config, the definition, the
recipe, the models, the observed onsets, the units, thresholds and the reference. A live forecast must have at least what
the blend was trained on (lead days, members, domain) with the same units, accumulation, grid spacing and issue hour.
