# Onset in ROMP

ROMP detects onset with the `momp.onset` component (contract: `docs/onset_spec.md`).
`momp/stats/detect.py` is a thin adapter: it runs the same definition on the
observations and on every forecast, then shapes the results for the metrics code.

## Configuring a run

Section 6 of your config file (start from `examples/config.in`) holds the onset settings. Every one of them can be left
`None`; it then takes the default of the region set by `region` (section 5), and
ROMP stops with a message naming the key if neither is known.

```
onset_definition = None   # registry id ("icpac") or path to a YAML file; None = region default
onset_season = None       # only for definitions with several seasons
threshold_years = None    # (first, last) years of observations for climatological thresholds
thresholds = None         # {name: "path.nc:variable"} to use precomputed thresholds
obs_day_boundary = None   # e.g. "00 UTC, labelled by start"
model_lead_label = None   # "start" or "end"; None = from obs_day_boundary
check_alignment = True
onset_backend = "numba"   # "numba", "reference" or "gpu"
onset_percentage_threshold = 0.5
```

Resolution order for each setting: explicit config value, then the region's
default, then an error. Region names are case-insensitive and may be aliases
(`India`, `ethiopia`); a country inherits the defaults of the regions
containing it. The choices and where each came from are printed at the start of
a run.

Useful commands:

```
python -m momp.onset list                # definitions
python -m momp.onset describe icpac      # the rule in words, hash, status, notes
python -m momp.onset regions             # regions with their defaults
python -m momp.onset check-alignment --obs obs.nc --obs-var RAINFALL \
    --obs-day-boundary "00 UTC, labelled by start" --model aifs.nc --model-var tp \
    --units m --lead-label start
```

A rule that differs a little from a registry definition goes in a YAML file with
`extends: <id>` and only the changed keys; set `onset_definition` to its path.

Thresholds (for example the local mean 5-day total) are computed from the
observations of `threshold_years` and cached under `dir_out/.onset_cache`.
Definitions with derived seasons (`dunning`) also compute a harmonic season
analysis from those years, and `onset_season` is `"1"` or `"2"`.

Setting `onset_definition` together with any old flat key (`wet_init`,
`wet_threshold`, `wet_spell`, `dry_threshold`, `dry_spell`, `dry_extent`,
`thresh_file`, `thresh_var`, `earliest_onset_date`) is an error: put the change
in a definition with `extends:`.

Results of definition runs carry an `onset_status` column (`ONSET`, `NOT_FOUND`,
`CENSORED`, `INSUFFICIENT_DATA`) and the run prints status counts. `CENSORED`
(the rule needed days beyond the forecast horizon) cannot happen: see below.

### Lead window

A candidate on lead k needs leads up to k + R - 1, where R is how many days from
the onset day the definition reads (ICPAC: 22; Moron-Robertson: 35; none of its
vetoes: the window length). At the start of a run ROMP reads the number of leads of
each model's first-year file and searches only candidates on leads up to
n_leads - R + 1. If that is below `max_forecast_day`, `max_forecast_day`,
`verification_window_list`, `tolerance_days_list` and `day_bins` are clipped (windows
and bins starting later are dropped) and a line such as `Lead window : 1-29 for AIFS
(icpac reads 22 days from the onset day; forecasts have 50 leads)  [shortened from
max_forecast_day 30]` is printed. The run uses the shortest window over its models.
With no usable lead ROMP stops and suggests `dry_spell_check = False`, which shrinks R.
The window is stored as the `lead_window` attribute of the spatial-metric NetCDFs.
Legacy mode keeps the "Not enough forecast time steps" error.

## Switching off the dry-spell condition

`dry_spell_check = False` drops the definition's `reject if` vetoes, such as
Moron-Robertson's 10-day dry window or ICPAC's 7-day dry run. This suits demos and
forecasts too short for the veto window (`CENSORED` otherwise). The run prints
`Dry-spell check  : off`; the title, notes, YAML, sentence and hash written to the
outputs are those of the variant. A definition without vetoes (or a curve rule)
is left alone and the line reads `none in this definition`. In legacy mode it has
the effect of `dry_extent = 0`. In code: `definition.without_dry_spell_check(defn)`,
and `python -m momp.onset describe <id> --no-dry-spell-check` prints the variant.
Env var for generated configs: `ROMP_DRY_SPELL_CHECK`.

## Forecast history

`forecast_history = "observed"` (the default for definitions) searches each forecast
together with the observed rain up to the day before its first lead. A wet spell
that began in the observations can be completed by the forecast, and a season that
started before the init is searched from its start. Nothing observed on or after
lead 1's date is read. `"none"` searches the forecast leads alone, as earlier runs
did; old flat-key configs always use it. ROMP loads the previous year's observations
as well when the season wraps over New Year or the rule looks back past 1 January.

Statuses: when the observations alone already confirm the onset the result is
`PRIOR`; ROMP still scores nothing for those inits, because inits on or after the
cell's observed onset produce no rows (a `PRIOR` result in a scored row is counted,
warned about and reported as no onset). A forecast may also complete an onset that
began before the init: the row then has `onset_day <= 0` (days after the init; the
date is in the observed history) and status `ONSET`.

How back-dated onsets (`onset_day <= 0`) are scored:

* Deterministic and ensemble-mean tables (`compute_onset_metrics_with_windows`): the
  onset is treated as falling on the first lead for window membership, so in a window
  starting at lead 1 it is a hit if its real date is within the tolerance of the
  observed onset and a false alarm otherwise; in windows starting later it is ignored,
  like any forecast onset before the window. MAE uses its real date. Before this
  rule such onsets were dropped from every window, which also applied to an onset on
  the init day itself (lead 1 with lead label "start"); this changes `"none"` runs
  with that label slightly.
* Ensemble probability bins: members with `onset_day < 1` fall in the "Before day 1"
  bin, where the observed onset (always after the init in scored rows) is 0, so they
  are penalised by the Brier score.

## Threshold source

`threshold_source = "observations"` computes climatological thresholds from the
reference observations; `"model"` computes them from the model's own forecasts: the
mean over the leads 1 to `max_forecast_day`, the forecast days whose date lies in the
threshold's `during` period in `threshold_years`, with ROMP's forecast loader (init
days, unit conversion). Only mean-based thresholds (`mean total over`,
`mean daily rain`) support it; others are an error naming them. Results are cached
under `dir_out/.onset_cache` by model, years, lead window and definition hash. Old
flat-key configs use observations only.

## Legacy mode

A config with no `onset_definition` key but the old flat keys runs as before
(`scripts/generate_config.py` still writes such configs unless a `ROMP_ONSET_*`
variable is set). The keys are translated by `momp.onset.legacy.from_flat_config`
and a `FutureWarning` points to `onset_definition`. Lead k of a forecast is dated
init + k (`lead_label` "end"). Day boundaries are needed only for the alignment
check, which is skipped with a note when they are unknown.

Translation: a wet day has at least `wet_init` mm; the search runs from
`earliest_onset_date` (or the month and day of `start_date`) to the month and day
of `end_date` plus 47 days minus `wet_spell - 1` (the old code cut its data off
47 days past `end_date`, so later candidates could never complete their wet
spell); a candidate needs all `wet_spell` days wet and a total
above the threshold; with `dry_extent > wet_spell` and `dry_extent >= dry_spell`
a dry run of `dry_spell` days within days 2 to `dry_extent + 1` after the
candidate cancels it.

Deliberate differences from the old code:

* The forecast dry window is one day longer. The old forecast window was days 1 to
  `dry_extent`; the observed window (days 2 to `dry_extent + 1`) is now used for both.
* Missing (NaN) days in a window now stop the search with `INSUFFICIENT_DATA`
  instead of being skipped.
* Input is rounded to 0.1 mm, and `wet_init` and `wet_threshold` must be whole
  tenths of a mm.
* Negative forecast values are set to 0 (the old code summed them as negative
  rain). Aurora has about 7 % negative values and GraphCast reaches -20 mm.
* `dry_threshold` was never used and still is not.
* A veto window may read observations beyond `end_date + 47` days. The old code
  cut the data there and treated the unseen days as having no dry spell, so a
  late candidate could pass its dry-spell check without it being made.
* A series that starts after the search start gives `INSUFFICIENT_DATA` rather
  than starting at its first day.
* `fallback_date` is no longer used.
* Forecast candidates are limited to the season, so a `max_forecast_day`
  reaching past its end finds no onset there.

## Alignment check

At the start of a run, once per model, ROMP compares the first year's forecasts
(leads 1-3) with the observations at lags -3 to +3 and prints the result. If the
forecasts look shifted by a day it warns, which usually means `model_lead_label`
or `obs_day_boundary` is wrong. The check never stops a run; set
`check_alignment = False` to skip it.

Finding on AIFS forecasts: CHIRPS v3 (00 UTC day, labelled by start) lines up with
AIFS under `lead_label` "start" (lead k is init + k - 1), while IMD (day ending at
03 UTC, labelled by end) lines up under "end". ROMP's old convention was always
"end", so for Ethiopia forecast onset dates were one day late against CHIRPS.
