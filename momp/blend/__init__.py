"""Blending methods: each turns the shared stages' features into onset probabilities over the week bins.

logit: the multinomial logistic blend of formula features.
random_forest: the CRPS random forest over each model's daily rainfall and the climatology, giving daily
probabilities too.
mme: the multi-model ensemble, a weighted mixture of other predictions.
calibration: Platt recalibration of week-bin probabilities.
design: formulas and design matrices from arrays. predictions: a model's probabilities with the samples' keys.
"""
