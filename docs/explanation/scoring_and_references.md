# Scoring and reference forecasts

## Scores

Onset is detected by the same definition on the observations and on every forecast (see {doc}`../onset_spec`). A forecast
onset and an observed onset are paired per case, and the pairs are scored:

* deterministic scores: the false alarm rate, miss rate and mean absolute error of the onset date;
* probabilistic scores, on the probability that the onset falls in each lead bin: the Brier score, the ranked probability
  score (RPS), the area under the ROC curve (AUC) and the Pietra index. The fair versions take out the sampling noise of an
  ensemble's probability when the number of members is known;
* skill scores, which compare a forecast's score with the reference forecast's.

The scores are pure functions on arrays and tables (`hcwf.metrics`); the flows that read years, detect onsets and pair them
(`hcwf.scoring`) call them. The public functions are in the {doc}`../api`.

## Reference forecasts

The reference of the skill scores is a forecast made from the observed onsets alone. It is chosen with the `reference:`
section of the run file (see the {doc}`../reference/config_keys`).

* The *climatology* is the climatological onset date of the climatology years, used as a baseline forecast.
* The *conditional climatology* is the probability that onset falls on lead `k` given that it has not happened by the
  issue date, `P(onset on d + k | none by d) = (F(d + k) - F(d + k - 1)) / (1 - F(d))`, where `F` is the CDF of a Gaussian
  kernel density estimate of a cell's observed onset days. Each probability is floored at a tiny minimum while each row keeps
  its sum, so no lead has probability exactly zero. A cell with too few onsets has no forecast.
* The truth of the reference is the run's own observed onsets (the evaluation's detection) over the climatology years, and
  with the leave-one-year-out option each test year is forecast from a climatology that does not contain its own onset.

Details of the numerics are in {doc}`../onset_spec` (section 7) and the module docstring of `hcwf.references.kde`.
