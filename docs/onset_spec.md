# Onset component specification (`momp.onset`)

A precipitation-only onset component. It turns a published onset definition
into a short readable YAML file, and runs that file identically on
observations and on model forecasts. This document is the contract: the
YAML grammar, the meaning of every term, the exact detection semantics, and
what each module and backend must do. `momp/onset/model.py` holds the
normalized types these sections refer to.

Out of scope here: non-precipitation variables, predicting onset, and skill
scoring (ROMP's metrics consume this component's results).

## 1. Glossary

| Term | Meaning |
|---|---|
| **day** | One daily rainfall total in mm, covering 24 h that start at the dataset's day boundary. |
| **wet day** | A day whose total meets the definition's `wet_day` rule, e.g. at least 1 mm. |
| **dry day** | A day with data that is not a wet day. Days with missing data are neither wet nor dry. |
| **dry run** | At least N *consecutive dry days*. It is count based: rain on the days only matters through the wet-day rule. ICPAC's "dry spell of 7 or more days" is a dry run. |
| **dry window** | A block of N consecutive days whose *total* meets a comparison, e.g. 10 days with less than 5 mm. It is amount based, and some of its days may be wet. Moron-Robertson's "10-day dry spell (5 mm in total)" is a dry window. |
| **candidate day** | A day tested as the possible onset. It is *day 1*; the day after is day 2, the day before is day 0. |
| **span** | Days relative to the candidate, e.g. "days 1 to 3", "21 days after onset" (days 2 to 22). |
| **trigger window** | The `window` of the rule, days 1 to W. `total`, `all days`, `wet days` and `first day` refer to it unless they have their own `over`. |
| **requirement** | A condition every onset must meet (`require`). |
| **veto** | A dry spell that cancels a candidate (`reject if`). |
| **search window** | The candidate days considered in one season of one year (`season` / `seasons`). |
| **first passage** | Rule-based family: onset is the first candidate in the search window that meets every requirement and no veto. |
| **curve** | Curve-based family: onset comes from the minimum of the cumulative rainfall anomaly over the search window. |
| **threshold** | A named per-cell amount computed from a reference climatology, e.g. the local mean 5-day total. |
| **data end** | The first day index with no data for a series, such as the forecast horizon. |

## 2. Statuses

Every search returns an onset day (or none) and one status (`model.Status`, int8):

| Code | Status | Meaning |
|---|---|---|
| 0 | `ONSET` | Onset was found, and every earlier candidate in the window was ruled out. |
| 1 | `NOT_FOUND` | Every candidate in the window was ruled out (an empty window gives this too). |
| 2 | `CENSORED` | An earlier candidate could not be decided because it needs days past the data end, e.g. the forecast horizon. No onset day is reported. ROMP never produces it: it only searches leads whose whole rule fits in the forecast (docs/onset_in_romp.md). |
| 3 | `PRIOR` | Onset had already happened before the search window opened. Only the forecast layer sets this, for example for inits after the observed onset. |
| 4 | `INSUFFICIENT_DATA` | An earlier candidate could not be decided because it needs days with missing rainfall, or the cell has no threshold. No onset day is reported. |

## 3. Definition YAML

### 3.1 Example

```yaml
id: icpac
version: 1
title: ICPAC operational seasonal rainfall onset
citation: TODO
wording: TODO paste the original definition text
status: draft
notes: >
  "Within the next 21 days" is read as the 21 days after the onset day.

wet_day: at least 1 mm

season:
  from: 03-01
  to: 05-31

onset:
  window: 3 days
  require:
    - first day: wet
    - total: more than 20 mm
  reject if:
    - dry run: at least 7 days
      within: 21 days after onset
```

### 3.2 Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | `[a-z0-9_]+`, unique within the registry. |
| `version` | yes | Integer. Bump it whenever the rule changes. |
| `title`, `citation`, `wording`, `notes` | yes | Provenance. `TODO` is allowed while `status: draft`. `wording` is the paper's original text. |
| `status` | yes | `draft` or `verified` (wording and golden series checked). |
| `extends` | no | The id of another definition. Mappings are merged key by key; lists and scalars replace the parent's. `id`/`version`/`title`/`status` must be restated. |
| `wet_day` | yes | `at least X mm` or `more than X mm`. |
| `season` or `seasons` | one of them | See 3.5. |
| `thresholds` | if referenced | See 3.6. |
| `onset` or `curve` | exactly one | First-passage rule (3.3) or curve rule (3.4). |

Unknown keys are errors, with the YAML path in the message.

### 3.3 `onset:` (first passage)

```yaml
onset:
  window: 5 days                 # trigger window: days 1 to 5
  require:                       # every item must hold
    - first day: wet             # day 1 is a wet day
    - all days: wet              # every day of the window is wet
    - wet days: at least 3       # number of wet days in the window
    - total: at least wet_spell_amount      # window total vs a threshold
    - total: at least 20 mm
      over: days 11 to 20        # its own span instead of the window
    - total: at least 51 mm
      over: season to date       # from the first day of the search window to day 1
  reject if:                     # any one cancels the candidate
    - dry run: at least 7 days                       # or: more than 7 days
      within: 21 days after onset                    # the whole run inside the span
    - dry window: 10 days with less than 5 mm
      starting within: 30 days after window          # the spell starts inside, may end after
  report: first day              # or: first wet day   (default: first day)
```

* `report: first day` reports the candidate. `first wet day` reports the
  first wet day of the trigger window (identical when `first day: wet` is
  required).
* `within` means the whole spell lies inside the span. `starting within`
  means the spell's first day lies inside it. Exactly one of the two is
  required.

### 3.4 `curve:` (cumulative anomaly)

```yaml
curve:
  accumulate: rain minus mean_daily_rain    # a threshold giving mm/day
  onset: day after minimum                  # or: day of minimum
```

### 3.5 Seasons

```yaml
season: {from: 06-01, to: 09-30}            # one search window a year

seasons:                                    # several, named
  long_rains: {from: 02-01, to: 05-31}
  short_rains: {from: 09-01, to: 12-31}

seasons:                                    # derived per cell from climatology
  derive: harmonic regime                   # one or two seasons a year per cell
  from: climatological onset - 50 days
  to: climatological cessation
```

* Bounds are `MM-DD`, or `climatological onset|cessation [+|- N days]`.
  `02-29` is not allowed.
* `to` before `from` within the year means `to` falls in the next year. The
  season belongs to the year of `from`.

### 3.6 Thresholds

```yaml
thresholds:
  wet_spell_amount:
    mean total over: 5 days        # 5 x mean daily rain during the period
    during: 04-01 to 10-31         # default: whole year
  mean_daily_rain:
    mean daily rain: true
    during: whole year
  heavy_3day:
    quantile of totals over: 3 days
    quantile: 0.8
    during: 06-01 to 09-30
```

The definition says what to compute. The run supplies the reference
dataset and years, and the output records both (section 9).

### 3.7 Grammar

* **Amount:** `<comparator> <number> mm`, or `<comparator> <threshold name>`.
  Comparators: `at least` (>=), `more than` (>), `less than` (<), `at most`
  (<=). The symbols `>=`, `>`, `<`, `<=` are also accepted. Fixed amounts must
  be whole tenths of a mm.
* **Count:** `<comparator> <integer>`.
* **Duration:** `<integer> day` or `<integer> days`.
* **Span:** `days A to B`, `day A`, `N days after onset` (days 2 to N+1),
  `N days after window` (days W+1 to W+N), or `season to date`.

**Ambiguity is an error, not a default.** Each of these fails validation
with a message saying how to fix it:

* an amount without a comparator (`20 mm`);
* `within: 21 days` without `after onset|window`;
* a veto with neither `within` nor `starting within`;
* `first wet day` reporting without a window;
* a threshold name that isn't defined;
* an amount finer than 0.1 mm;
* a `dry run`/`dry window` whose length doesn't fit inside a `within` span.

## 4. Detection semantics

Notation, for one series:

* `x[t]` is rainfall in tenths of a mm and `v[t]` says day t has data,
  for `t` in `[0, T)`.
* `E <= T` is the job's data end.
* Day `t` is wet when `v[t]` and `x[t]` passes the wet-day comparison. It
  is dry when `v[t]` and it is not wet.
* Candidates `d` run from the job's `start` to `end` (inclusive), in order.

**Spans.** A span (first, last) at candidate d covers the half-open
interval `[d + first - 1, d + last)`. `season to date` covers
`[start, d + last)`.

**Observability.** An interval `[a, b)` is observable when `a >= 0`,
`b <= E`, and no day in it is missing. When it is not observable it carries
flags:

* `horizon` if `b > E`;
* `missing` if `a < 0` or any day in `[a, min(b, E))` is missing.

**Three-valued logic.** Every component evaluates to True (T), False (F) or
Unknown (U). An Unknown carries the flags above.

* **Total**: U if the span is unobservable, otherwise `cmp(sum x[a:b], amount)`.
* **WetDays**: U if the span is unobservable, otherwise `cmp(#wet in [a, b), count)`.
* **Veto, `within`**: a spell of length L starting at u occupies
  `[u, u+L)`, for u in `[a, b-L]`.
  * It is *confirmed* when `[u, u+L)` is observable and the spell holds
    (all days dry for a dry run; `cmp(sum, amount)` for a dry window).
  * The veto is F if `b - a < L`. Otherwise it is T if any u is confirmed,
    U if `[a, b)` is unobservable, and F otherwise.
* **Veto, `starting within`**: u runs over `[a, b-1]` and observability is
  judged on `[a, b + L - 1)`. The veto is F if the span is empty (`b <= a`);
  the spell may be longer than the span, because it only has to start inside it.
  Otherwise it is T, U or F as for `within`.
* **Combining** (Kleene):
  * AND is F if any operand is F, else U if any is U, else T.
  * OR is T if any operand is T, else U if any is U, else F.
  * NOT swaps T and F.
  * The flags of a U result are the union of the flags of its U operands.
  * candidate = AND(require) AND NOT(OR(reject if)).

**First passage.** Scan d from start to end and take the first candidate
that is not F.

* If it is T, the status is `ONSET`. The day is d, or the first wet day of
  the trigger window for `report: first wet day` (falling back to d if the
  window has no wet day).
* If it is U, no day is reported. The status is `CENSORED` if its flags are
  exactly {horizon}, otherwise `INSUFFICIENT_DATA`.
* If no candidate is T or U, the status is `NOT_FOUND`.

**Curve.**

* The baseline b is the cell's threshold in thousandths of a mm,
  `round_half_even(mm * 1000)`.
* If `[start, end + 1)` is unobservable, the result is U, with the status
  set from its flags as above.
* Otherwise let `C(t) = sum over i = start..t of (100 * x[i] - b)`, and let
  m be the first argmin of C over `[start, end]`.
  * If `m == end`, the status is `NOT_FOUND`.
  * Otherwise the day is `m + 1` (`day after minimum`) or m
    (`day of minimum`), with status `ONSET`.

**Thresholds to integers.** A per-cell threshold T in mm (float) is
converted once, on the CPU, by comparator:

* first `q = round(10 * T, 6)`;
* then `ge` → `ceil(q)`, `gt` → `floor(q)`, `lt` → `ceil(q)`, `le` → `floor(q)`.

For integer x, comparing against the converted value is then exactly the
same as comparing against T. Cells whose threshold is NaN get
`INSUFFICIENT_DATA` without being searched.

**Properties** that tests check, all for the same data:

* a stricter requirement, or an added veto, never gives an earlier onset;
* an earlier search start never gives a later onset;
* the reference, numba and GPU backends agree exactly.

## 5. Ingestion contract (`momp.onset.ingest`)

`ingest(da, *, day_boundary, units=None, source="") -> DailyRain`

* Finds the time dim (`time`, `TIME`, `valid_time`, `date`) and renames
  spatial dims to `lat`/`lon` (`latitude`, `LATITUDE`, `longitude`,
  `LONGITUDE`, ...). The order of the other dims is kept, with time moved
  last.
* Units come from the `units` argument or `da.attrs["units"]`; having
  neither is an error.
  * `mm`, `mm/day`, `mm day-1`, `mm d-1`, `kg m-2`, `kg m-2 day-1` are taken as is.
  * `m`, `m/day` are multiplied by 1000.
  * Anything else is an error.
* Values are rounded half-even to int32 tenths. NaN is missing (stored as 0
  and flagged).
  * Values in `[-0.05, 0)` mm become 0.
  * Values below -0.05 mm become missing, with a warning giving the count.
* The time axis must be daily and strictly increasing. Duplicates are an
  error. Gaps are filled with missing days, with a warning.
* The calendar comes from cftime objects or `standard`. year/month/day are
  labelled in that calendar.
* `day_boundary` is required, e.g. `"00 UTC, labelled by start"` or
  `"03 UTC, labelled by end"` (IMD's 08:30 IST day). It is parsed and
  validated, then stored.

`ingest_forecast(da, *, day_boundary, lead_label, units=None, source="",
init_dim=None, lead_dim=None, member_dim=None) -> ForecastRain`

* The dims become `init_time`, optional `member`, then spatial dims, then
  `lead` last. Lead k (1-based) is the k-th 24 h after the init.
* `lead_label` is required: `"start"` labels lead k with the date
  init + k - 1, and `"end"` labels it init + k (ROMP's historical choice).
* Accumulation (`accumulation="auto"|"daily"|"since_init"`, default auto):
  * `daily`: each step is a per-interval amount. 24 h steps are the daily
    totals; shorter steps (1, 3, 6, 12 h, any step dividing 24 h, all on a
    complete grid) are summed per day.
  * `since_init`: running total since the init. Day j total = acc(end of day j)
    - acc(end of day j-1), acc(0) = 0, using the steps at the 24 h boundaries
    (a missing one is an error naming it). Step 0 is ignored.
  * Steps are timedeltas or numbers. Numbers use the coordinate's `units`
    attribute (hours or days); without it, integers 1..n or below 24 are
    days, else hours.
  * Incomplete final days are dropped, with a note.
  * `detect_accumulation(da, lead_dim) -> (kind, evidence)` (auto): `since_init`
    if values are non-decreasing along lead (tolerance 0.1 % of the series
    maximum) in >= 99 % of the non-constant finite series and the last lead's
    mean is > 3x the first's; `daily` if clearly not; ambiguous cases (cumulative
    without growth, or mostly cumulative) are an error asking to set it.
    Constant data or a single lead count as daily.
  * Auto always emits an `IngestInfo` warning (a `UserWarning` subclass, so
    visible by default) such as `accumulation: since_init [non-decreasing along
    lead in 99.8 % of series; mean 0.4 -> 38]; steps: 6-hourly`.
    Explicit settings emit it only for sub-daily steps, a daily_boundary or drops.
  * The result carries `ingest_info` (a `ForecastIngestInfo` with accumulation,
    evidence, step_hours, resolution, offsets, notes) as a plain attribute;
    `ForecastRain` is unchanged. `ingest_forecast_info` returns `(forecast, info)`.
* `daily_boundary` (e.g. `"03 UTC"`, default None): by default days run 24 h
  from the init. With it, days are cut at that UTC hour, so day j covers
  (b + 24 (j-1), b + 24 j] where b is the first boundary after the init
  (b = init if it is already on the boundary). The part before b is dropped,
  and lead k is the k-th complete day after b. Init date labels move to the UTC
  date b falls on, so `lead_label` keeps its meaning (a 12 UTC init with 00 UTC
  days is labelled with the next date). Inits may be at different hours (each
  gets its own cut, the common number of days is kept). If the steps cannot hit
  b, the error names the nearest UTC hours that work; with
  `boundary_tolerance_hours` >= the miss it snaps to the grid with a warning
  giving the distance.
* Units are also checked against magnitudes (`check_units=False` to skip): the
  99.9th percentile of the daily totals below 0.5 contradicts `mm` (suggests
  `units='m'`), and above 1 contradicts `m` (suggests `mm`).
  `detect_units(da, accumulation_kind) -> (units, evidence)` gives the guess.
* Every negative forecast value becomes 0, with a warning giving the count and
  the lowest value. Model rainfall dips below zero numerically (Aurora has 7 % of
  values below 0, GraphCast down to -20 mm), so treating these as missing would
  discard whole forecasts.

## 6. Engine and backends

**Plan.** `plan.compile(definition) -> Plan` is a pure function of the
definition. It turns spans into offsets and lists the prefix tables and
per-cell amounts the rule needs. `plan.resolve_amounts(plan, thresholds,
n_series=None)` applies the rounding in section 4 and returns `(amounts,
invalid)`. `invalid` flags series with a NaN threshold, which the engine marks
`INSUFFICIENT_DATA` without searching. `n_series` is needed when the rule uses
only fixed amounts.

**Backends.** All backends have one signature:

```python
detect(plan, tenths: (S, T) int32, missing: (S, T) bool,
       amounts: dict[str, (S,) int64], jobs: model.Jobs, *, trace=False,
       amount_windows=None)
    -> model.Detection
```

* Each row of `tenths` is one series: a cell for observations, or an
  (init, member, cell) for forecasts.
* Job indices are into the time axis.
* The backends:
  * `reference`: NumPy, with prefix tables and full candidate masks. It is
    the oracle. With `trace=True` it returns every intermediate: per-candidate
    T/F/U codes for each requirement and veto, their flags, and the combined
    state.
  * `numba`: parallel over jobs, with an early exit at the first candidate
    that is not F.
  * `gpu`: CuPy, full masks plus argmax. It is optional and needs `cupy`.
* Prefix sums are int64 (rain) and int32 (counts). Results must be
  bit-identical across backends.

**High level.**

* `detect_observed(definition, rain: DailyRain, *, thresholds=None, years=None,
  season_climatology=None, backend="numba", attrs=None)` returns an
  `xr.Dataset` with dims (`year`, `season`, *spatial*).
* `detect_forecast(definition, forecast: ForecastRain, *, thresholds=None,
  lead_window=(1, 30), restrict_to_season=True, backend="numba", attrs=None,
  history: DailyRain | None = None, threshold_windows=None)` returns an
  `xr.Dataset` with dims (`init_time`, [`member`], `season`, *spatial*).
  Candidates are the leads in `lead_window` whose date label falls in a
  season window. The data end is the last lead. `history` stitches observed
  rain before lead 1 (section 13); `threshold_windows` gives the start leads of
  per-window thresholds.
* `regional(rain, mask, weights="cos_lat"|"equal")` area-averages the cells
  in `mask` into one series before detection.

## 7. Climatology (`momp.onset.climatology`)

* `compute_thresholds(definition, rain: DailyRain, years=(y0, y1)) ->
  dict[str, xr.DataArray]` computes per-cell thresholds in mm.
  * Days in `during` are pooled over the years.
  * A cell is NaN when more than 10 % of its reference days are missing.
* `harmonic_seasons(rain, years, ...) -> xr.Dataset` returns `n_seasons`,
  `onset_doy` and `cessation_doy` per season and cell, from harmonic
  analysis of the mean annual cycle. The rule is that the second harmonic's
  amplitude exceeding the first's means two seasons; this is a draft until
  checked against Dunning et al. 2016.
* `ThresholdCache(directory)` stores computed fields keyed by
  `sha256(source, years, spec, grid)`, so each threshold is computed once.
* `compute_thresholds_from_forecasts(definition, forecasts, years,
  lead_window=(1, 30), windows=None)` does the same from the model's own
  forecasts (section 13).
* `engine.climatological_onset(definition, rain, *, thresholds=None, years, ...)`
  gives the median onset day of year per cell and season, from the observed
  detection.

### 7.1 Conditional climatology (`momp.onset.climatology_kde`)

The reference forecast `ref_model = "conditional_climatology"` is the
probability that onset falls on lead k given no onset by the issue date,
`P(onset on d + k | none by d) = (F(d + k) - F(d + k - 1)) / (1 - F(d))`.
F is the CDF of a Gaussian KDE (Scott's bandwidth) of a cell's observed onset
days, built as R's `density()` plus `approxfun(rule = 2)` do (512 grid points,
end values held outside). Each row is floored at `MIN_PROB = 5e-7` while
keeping its sum; a cell with fewer than 10 onsets has no forecast.
`conditional_probabilities(onset_days_by_cell, issue_days, horizon, *,
cv_years=None)` returns (cell, [year,] issue, lead) for leads 1..horizon;
`unconditional_probabilities` adds lead 0 (onset already happened). With
`cv_years` (ROMP: `climatology_cv = "leave_one_year_out"`) each test year's KDE
is refitted without its own onset. ROMP trains on the observed onsets of
`start_year_clim..end_year_clim` (the evaluation's own detection), with days
counted from 1 January and no leap-day shift. Probabilistic pairs hold the
bin probabilities (Before is zero when the bins start at lead 1) with infinite
member counts, so the fair corrections vanish; the deterministic forecast is
the first lead whose conditional CDF reaches 0.5 within `max_forecast_day`.

## 8. Temporal alignment (`momp.onset.alignment`)

Datasets that use different day boundaries or lead labels are shifted
against each other, often by exactly one day.

* `check_alignment(reference, other, max_lag=3) -> AlignmentReport`:
  * Correlates `sqrt(rain)` anomalies (each cell's mean removed), pooled
    over shared cells and days, at lags -3..+3.
  * Reports the best lag and a verdict: `aligned`, `likely shifted by N
    day(s)`, or `inconclusive`.
  * The verdict is "shifted" when the best lag is not 0, beats lag 0 by more
    than 0.02, and has r > 0.1. It is "aligned" when lag 0 has r > 0.1 and
    beats every other lag by more than 0.02. Anything else, including a near
    tie between lag 0 and another lag, is "inconclusive".
  * It also reports time-axis problems (gaps, duplicates, calendars,
    overlap) and a mismatch between the declared day boundaries.
* `check_forecast_alignment(obs, forecast, leads=(1, 2, 3), max_lag=3)` does
  the same per lead, comparing each lead's date label with the observed day.
  A best lag that is consistently nonzero suggests the wrong `lead_label`.
* `lagged_correlations` / `forecast_alignment_from_samples` give the same
  verdicts from prepared sample arrays: the forecast is aligned once against a
  stacked (lag, ...) view of the observed days and every lag and lead is
  correlated in one vectorised step.
* `momp.lib.align_sample` is the fast check behind `momp-run --check
  --alignment` and the lead-label detection of zarr datasets. It reads a
  deterministic sample once (up to 40 inits spread over the evaluated period,
  up to about 3000 cells on a regular subgrid of the shared land cells, leads
  1-3, members averaged on sqrt rain, in contiguous selections per file or
  store), judges it under the declared (or both) lead labels, runs several
  models in parallel threads and caches each verdict under a fingerprint of both
  datasets, the label, the day boundaries and the sample specification, in
  `<cache>/alignment` next to the detection cache. On the real archives its
  verdicts and best lags equal the whole-year check's.

## 9. Output and reproducibility

* Results are `xr.Dataset`s with:
  * `onset_date`, `status` (int8, CF `flag_values`/`flag_meanings`), and
    `onset_index`;
  * for forecasts, also `onset_lead`.
* Attributes:
  * `definition_id`, `definition_version`, `definition_hash`;
  * `definition_yaml`, the source text;
  * `definition_sentence`, from `describe()`;
  * `threshold_reference` (JSON: source, years, specs), `input_source` and
    `backend`.
* `definition_hash` is the sha256 of the canonical JSON of the semantic
  fields (wet_day, seasons, rule, thresholds). Provenance edits don't
  change it.
* `save(ds, path)` writes zarr.

## 10. Registry

`momp/onset/registry/*.yaml`. All entries are `status: draft`: wording and
golden station series are placeholders until checked against the sources.

| id | Family | Summary |
|---|---|---|
| `moron_robertson` | first passage | From 1 Apr: first wet day of a 5-day spell totalling at least the local mean 5-day amount, with no 10-day dry window (< 5 mm) in the next 30 days |
| `moron_robertson_mok` | first passage | Same trigger with no veto, searching from 2 Jun (the median MOK date) |
| `icpac` | first passage | First wet day of 3 days totalling more than 20 mm, with no dry run of at least 7 days in the next 21 days |
| `sivakumar` | first passage | After 1 May: 3 days totalling at least 20 mm, with no dry run of more than 7 days in the next 30 days |
| `marteau` | first passage | First of 2 consecutive wet days totalling more than 20 mm, with no 7-day dry window (< 5 mm) in the next 20 days |
| `omotosho` | first passage | Rule needs checking against the paper |
| `liebmann_marengo` | curve | Day after the minimum of the cumulative anomaly about the annual mean daily rain |
| `dunning` | curve, derived seasons | Liebmann-Marengo per season, with seasons from harmonic analysis |

## 11. Module layout

```
momp/onset/
  model.py        shared types (written first; everything imports it)
  units.py        amount / duration / span / day-ref grammar
  definition.py   YAML -> model.Definition: validation, extends, hash, registry
  describe.py     the generated sentence
  registry/       the definition YAML files
  ingest.py       DailyRain / ForecastRain ingestion contract
  alignment.py    temporal misalignment checks
  climatology.py  thresholds, harmonic seasons, cache, climatological onset
  plan.py         compile + resolve_amounts
  backends/       reference.py, numba_cpu.py, gpu.py, windowed.py (per-window amounts)
  engine.py       jobs from seasons and leads, detect_observed / detect_forecast, output, save
  calendar.py     date keys and season bounds
  regions.py      region registry (section 12); regions/*.yaml
  legacy.py       old flat ROMP keys -> a Definition
  explain.py      trace_observed / trace_forecast and explain(): why a cell got its onset
  viz/            anatomy_plot, schematic, compare_definitions, forecast_vs_truth, drill_down
  __main__.py     python -m momp.onset: list, describe, regions, check-alignment, explain, schematic
```

## 12. Regions and countries

`momp/onset/regions/*.yaml`, loaded by `momp.onset.regions`. An entry says
where definitions apply and which reference data an area uses. Unknown keys
are errors.

```yaml
id: ethiopia
title: Ethiopia
kind: country            # country | region
iso: ETH                 # ISO 3166-1 alpha-3; required for countries, forbidden for regions
aliases: [Ethiopia]      # optional extra names, matched case-insensitively
part_of: [east_africa]   # optional region ids
domain: {lat: [3, 15], lon: [33, 48]}
observations:            # the reference truth dataset
  dataset: CHIRPS v3
  day_boundary: 00 UTC, labelled by start
climatology:
  years: 1998-2024       # YYYY-YYYY, start <= end
definitions:
  default: icpac
  also: [sivakumar]
notes: ...
```

* Every definition id must exist in the definition registry, and every
  `part_of` id in the region registry.
* `ancestors(region)` follows `part_of`, nearest first, and is cycle safe.
* `definitions_for(name)` returns `(default, also)`. The nearest area in the
  chain that names a default wins; `also` is the union along the chain,
  without the default. A country with no `definitions` inherits its regions'.
* A definition may list `regions: [ids]` (validated, not part of its hash).
  `check_definition_region(defn, name)` returns a warning when neither the
  area nor any area containing it is listed, and None otherwise or when the
  definition lists no regions.

## 13. Forecast layer: observed history, members and censoring

**Stitched series.** For a forecast init, the series searched is the
observed history up to the day before lead 1, followed by the forecast.
Stitched day t is history day t for t < split, and forecast lead t - split + 1
for t >= split. `split` is the history index of lead 1's date label, so it
depends on `lead_label`. History on or after `split` is never read, so
nothing observed after the init leaks in. Every member of an init shares the
same history.

**Search window.** Candidates run from the season start (usually in the
history) to the earlier of the season end and the last usable lead, the one
whose whole rule still fits in the forecast. The data end is `split` plus the
number of leads.

**Statuses**, decided in this order:

1. `PRIOR`: the history alone already gives `ONSET`. This means a search
   with `data_end = split` finds it, with every earlier candidate ruled out
   and the onset day's own spans fully observed. The result is the same for
   every member. The onset day is reported, and it lies before lead 1.
2. Otherwise the stitched search gives `ONSET`, `NOT_FOUND`, `CENSORED` or
   `INSUFFICIENT_DATA` exactly as in section 4. An `ONSET` day may lie
   *before* lead 1 (a back-dated onset). For example, a wet spell that
   started in the history and is completed by the forecast, or a dry-spell
   check that needs forecast days. `onset_lead = day - split + 1` is then
   0 or negative.

Decided values never change when more data arrives, so step 1 and step 2
agree wherever both decide. `PRIOR` is exactly the case where the forecast
adds nothing.

**Thresholds.** A named threshold comes either from the observed climatology
(section 7) or from the model's own climatology, chosen explicitly by the
caller. The model climatology pools each cell's forecast days over the leads
in the lead window, the inits of the reference years, and the valid dates
within `during`. Only the mean-based kinds (`mean_total`, `mean_daily`)
support it. With the default observed climatology one amount applies per job,
whether a window lies in the history, in the forecast, or across both.

**Per-window thresholds.** With a model climatology, each verification window
(ROMP's `verification_window_list`, after any lead-window shortening) gets its
own threshold per cell, pooled from the model's forecast days at leads inside
that window (same pooling over inits, members and `during` as above).
`compute_thresholds_from_forecasts(..., windows=[(1, 15), (16, 30)])` returns
thresholds with a leading `window` dim (coordinates `window_first`,
`window_last`).

* A candidate uses the column of the window its lead falls in. Its lead is
  `day - split + 1` (the lead index plus 1 in forecast-only mode). A candidate
  before lead 1 (stitched history) uses the first window. A span that crosses a
  window boundary is judged by the candidate's own lead, never by the lead of the
  days it covers.
* Everything the candidate evaluates uses that column: total comparisons, the
  wet-day amount (when it is a threshold), dry-window vetoes and the
  first-wet-day report. A curve rule uses the window of its first candidate.
  The history-only PRIOR search uses the first window. Observed detection, which
  has no lead, also uses the first window. A NaN in any window makes the series
  `INSUFFICIENT_DATA`.
* Backends take amounts shaped `(S,)` or `(S, W)` plus
  `amount_windows=[w_0, ..., w_{W-1}]`, the strictly increasing start leads of the
  windows (`w_0` is taken as minus infinity). All backends and both `detect` and
  `detect_stitched` give identical results; without windows nothing changes. A
  job is cut at the window boundaries into segments, each run with its window's
  `(S,)` amounts, and the first segment that is not `NOT_FOUND` decides it
  (`Jobs.origin` keeps "season to date" spans anchored at the job's first
  candidate). Traces are not available with per-window amounts.

**Backend contract.**

```python
detect_stitched(plan, hist_tenths: (C, T) int32, hist_missing: (C, T) bool,
                fc_tenths: (S, H) int32, fc_missing: (S, H) bool,
                amounts: dict[str, (S,) int64], jobs: model.StitchedJobs,
                *, trace=False) -> model.Detection
```

* The result must equal `detect` run on the explicitly concatenated series
  (history row `cell` cut at `split`, then forecast row `series`), with the
  same start, end and data end.
* `numba` and `gpu` must not copy history into each forecast row.
  * Prefix tables are joined at the split by adding: for t >= split,
    P(t) = P_hist(split) + P_fc(t - split).
  * Spells crossing the split are handled by a small per-job seam table
    holding the spell-start indicator for starts in
    [split - L + 1, split - 1].
* `reference` may build the concatenation.
* Only the history columns needed (the earliest candidate minus the longest
  look-back, up to `split`) have to be read.

**Ensemble summary.** For forecasts with members, the engine also returns,
per init and cell:

* the number of members in each status;
* `onset_probability` by lead: the share of members whose onset falls on or
  before that lead, with PRIOR members counting at lead 0.

**Decisions recorded from the first implementation** (`engine`, `reference`, `climatology`).

* `onset_lead` is float64 with NaN for no onset, in both modes, because stitched
  onsets can have lead 0 or below. `PRIOR` days are reported the same way (negative
  lead). `forecast_history` is `"observed"` or `"none"`.
* The `PRIOR` search is run for first-passage rules only. A curve onset is the
  minimum over the whole window, so a longer window can move it and the history-only
  result would not equal the stitched one.
* `members_by_status` uses the dim name `status_code` (coordinates: the code, and
  `status_name`), because a dim named `status` clashes with the `status` variable.
* `onset_probability` leaves `CENSORED` and `NOT_FOUND` members in the denominator
  (they never count as reached); it is NaN when every member is `INSUFFICIENT_DATA`.
* Season year, stitched mode: the first of the init's year minus 1, the year and the
  year after whose season window contains a lead of the lead window. If the season
  start is not in the history the job is `INSUFFICIENT_DATA`; if the season starts
  after lead 1, candidates start at the first lead inside it. With
  `restrict_to_season=False` the single season "window" runs from lead 1 to the end of the
  lead window and has no `PRIOR`.
* Model climatology: a forecast day counts when its own date label (not the init) lies in
  the `during` window of a year in `years` (a window crossing the year end belongs to the
  year of its start). The expected day count is members x such (init, lead) pairs.
