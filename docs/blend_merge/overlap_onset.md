# Onset slice: onset_blending -> ROMP (momp/onset) overlap report

Paths: `OB/` = /home/hcwf/code/onset_blending/onset_blending, `MO/` = /home/hcwf/code/ROMP/momp.
Scratch checks (run with ROMP's venv, with onset_blending added to sys.path), all in this scratchpad folder:
`cmp_rules.py` (rule-by-rule comparison of blending `find_onsets` with ROMP's reference backend, 4000 random series x 120 days;
numba was asserted identical to reference for every case), `diag.py` (sorts the mismatches by cause), `fc.py` (forecast
horizon), `kde.py` (KDE), `thr.py` (thresholds).

## 0. Summary of verdicts

| Concept | Verdict | One implementation after merge |
|---|---|---|
| Onset rule (trigger, veto, anchor, comparisons) | SAME for every shipped blending config; EXTEND for 3 options | MO/onset definition + plan + backends; OB/core/onset_rule.py deleted |
| Forecast horizon (follow-up past last lead) | CONFLICT: blending truncates, ROMP censors | add one `horizon` policy to the engine |
| Missing rain (keep/drop/zero) | CONFLICT: different NaN semantics | one policy at ingest + ROMP's three-valued logic |
| Rain quantisation | ROMP rounds to 0.1 mm integers, blending uses floats | ROMP integers (fixes blending's float ties); optional finer scale |
| Search start variants (base / fixed_cutoff / ref) | EXTEND (engine job builders only, no backend change) | `search_start` argument on detect_observed/detect_forecast |
| Forecast member counts per lead | EXTEND (a histogram in `_ensemble_summary`) | engine output `onset_count` |
| Rain predictors | NEW to ROMP; reads the definition and thresholds only | moves to ROMP's feature code; needs a `Definition.windows()` accessor |
| KDE climatology core | SAME (ROMP's file is a port, bit-identical on the same axis) | MO/onset/climatology_kde.py; OB core functions deleted |
| Climatology issue grid / horizons / year windows / CV | EXTEND (caller-side) | ROMP climatology module gains blending's grid |
| Computed thresholds | EXTEND (missing cap, unit series, scale, tables) | MO/onset/climatology.compute_threshold |
| Threshold / onset caches | 4 key schemes today -> 1 | a generic fingerprint store built from observed_cache.py |

## 1. Onset-rule options mapped to ROMP's definition language

Blending reads the rule from `OnsetDefinition`/`DrySpell` (OB/config.py:315-341) into `OnsetRule` (OB/core/onset_rule.py:20-55)
and searches in `find_onsets` (onset_rule.py:64-179). ROMP parses YAML (MO/onset/definition.py:412-501) into
`FirstPassage/Total/WetDays/Veto` (MO/onset/model.py:52-130). It compiles that to a Plan (MO/onset/plan.py:100-146) and runs the
backends (MO/onset/backends/reference.py:287-337, numba_cpu.py:108, gpu.py:244).

### 1.1 Expressible today (verified numerically, `cmp_rules.py`, observed mode with `reject_if_short_followup=True`)

Result of each case: (mismatches out of 4000 series); the causes come from `diag.py`.

| Blending option | ROMP YAML | Result |
|---|---|---|
| `trigger_rule: all_days_wet` (onset_rule.py:121-123) | `- all days: wet` | |
| `trigger_rule: first_day_wet` (:116-117) | `- first day: wet` | |
| `min_wet_days: K` (:118-120) | `- wet days: at least K` | |
| `wet_day_comparison gte/gt`, `wet_day_min_mm` (:100-105) | `wet_day: at least X mm` / `more than X mm` | see 1.2(a) for gt |
| `threshold_comparison gt/gte` (:106, :127) | `- total: more than thr` / `- total: at least thr` | |
| `follow_days: 0` | `reject if: []` | F: 1 (float tie, see 1.3) |
| consecutive_dry, `contained: true`, after_trigger (:143-152) | `- {dry run: at least L days, within: F days after window}` | B: 1 (float tie) |
| consecutive_dry, `contained: false`, after_trigger (:153-156) | `- {dry run: at least L days, starting within: F days after window}` | A: 63, all at the series end (1.2c) |
| `followup_anchor: onset_day` (:90-92): the span is days d..d+F, so it includes the onset day | `within`/`starting within: days 1 to F+1` (not `F days after onset`, which is days 2..F+1) | D: 63, series end only |
| `mode: window_sum`, `sum_window: W`, `sum_min_mm: S` (:157-167); always "wholly inside" | `- {dry window: W days with less than S mm, within: F days after window}` | E: 23 (float ties) |

The shipped production configs translate exactly. The configs are OB/../configs/india_aifs.yml and fixed_cutoff_2026*.yml:
window 3, all_days_wet, gte 1 mm, gt threshold, follow 21, after_trigger, consecutive_dry 5, not contained.

```yaml
wet_day: at least 1 mm
season: {from: 05-01, to: 12-31}        # blending searches to the end of each season's data; pick the real end
thresholds: {onset_threshold: {mean total over: 3 days, during: 04-01 to 10-31}}  # or supplied per unit (sec. 4)
onset:
  window: 3 days
  require: [{all days: wet}, {total: more than onset_threshold}]
  reject if:
    - {dry run: at least 5 days, starting within: 21 days after window}
```
For `threshold: {mm: 20}` (india_aifs.yml), write `total: more than 20 mm`. ROMP rejects fixed amounts finer than 0.1 mm
(definition.py:274-282, units.py:61-64). Blending accepts any float, so a non-tenth `mm` must be supplied as a per-cell
threshold array instead.

The country template (configs/country_template.yml; archive/readme-scripts.md:431-445 calls it the "original Moron-Robertson"):
```yaml
wet_day: more than 1 mm
onset:
  window: 5 days
  require: [{first day: wet}, {total: more than thr}]
  reject if:
    - {dry window: 10 days with less than 5 mm, within: days 1 to 31}
```
REGISTRY CONFLICT: MO/onset/registry/moron_robertson.yaml reads the same paper differently: `at least 1 mm`, `total: at least`,
and `within: 30 days after window`. Blending's ICPAC (tests/test_onset_rule.py:91-92: 5 days, >= 20 mm, >= 3 wet days, no veto)
is also not ROMP's icpac.yaml (3 days, > 20 mm, 7-day dry run within 21 days after onset). After the merge each reading needs
its own registry id. Do not silently pick one.

### 1.2 Needs an extension (numerically confirmed differences)

(a) **Separate dry-day definition.** Blending treats a day as dry when `rain < dry_day_min_mm`, which defaults to
wet_day_min_mm (onset_rule.py:51, :132). ROMP's dry day is "valid and not wet" (reference.py:179). The two differ when
`dry_day_min_mm` differs from the wet threshold, or with `wet_day_comparison: gt`, where a day of exactly 1.0 mm is neither wet
nor dry in blending but dry in ROMP. Case G (`more than 1 mm`) gave 116/4000 mismatches.
- Spec: an optional top-level `dry_day: less than X mm` (default: not wet). It compiles to `Plan.dry` (a Cmp) and is part of
  the definition_hash.
- Backends that must change:
  - reference: `_Tables.dry` (reference.py:179).
  - numba: `_dry_count` (numba_cpu.py:84) derives dry from pm/pw. It needs its own dry prefix table, also in the stitched
    path (`_build_hist_spells` :264, `_fill_seam` :284).
  - gpu: `_spell_prefix` (gpu.py:141) and the stitched tables (gpu.py:429).
- Priority: low. No shipped config hits this (india_aifs sets dry 1.0 = wet 1.0 gte; the template uses window_sum).

(b) **"Run begins within" placement.** Blending's non-contained veto counts only starts of maximal dry runs: a run start
requires the previous day to be not dry (onset_rule.py:138-141). ROMP's `starting within` also counts a run that began before
the span and carries L dry days into it (reference.py:270-277). The two differ only when the day before the follow-up can be
dry, which means first_day_wet + after_trigger + contained false. Case C gave 190/4000 mismatches, where ROMP vetoes more,
giving later or censored onsets.
- Option 1: a `dry run ... beginning within` placement, implemented as `holds[u] &= ~dry[u-1]` in the three spell-prefix
  builders above.
- Option 2 (recommended unless a user needs it): adopt ROMP's reading and document it. No shipped config hits it.

(c) **Horizon policy (the biggest risk).**
- Observed: blending refuses a candidate whose follow-up runs past the data (`reject_if_short_followup=True`,
  onsets.py:137-139). Forecasts: it uses the default False (onsets.py:661), so the follow-up is checked only over the leads
  that exist (archive/readme-scripts.md:446-450).
- ROMP marks any span past `data_end` as Unknown with the horizon flag. First passage stops at the first non-False candidate,
  so the job is CENSORED (spec §4; reference.py:229-236, :306-323). ROMP's runner then shortens `max_forecast_day` so the
  rule fits (MO/lib/lead_window.py:17, MO/stats/detect.py:137-160).
- A `starting within` veto also needs L-1 days beyond the span (reference.py:272). That is the source of the series-end
  mismatches in A and D, even in observed mode.
- Measured on the production rule with 45 leads (`fc.py`): blending finds 2119 onsets, ROMP 1274 (846 mismatches; 2726
  censored). On leads 1-28 (probability_days) it is 1598 vs 1274. Blending-only onsets begin at lead 15-21 (median 27).
- Emulation: padding the series with wet days beyond lead 45, while keeping candidates within the real leads, reproduces
  blending (4 mismatches, all float ties). That is the exact semantics to build.
- Spec: a run option `horizon: censor | truncate` (default censor). Under truncate, a veto whose only Unknown flag is
  horizon is False, and a requirement span past the data stays Unknown.
- Backends: change `_veto` code selection in reference.py:278, numba `_first_passage`/`_first_passage_s`, and gpu `_veto`
  (gpu.py:218). It is one branch each, passed as a plan or detect flag. It must not enter definition_hash, but it must be
  recorded in the output attrs.

(d) **Missing rain.** Blending policies (Observations.missing_rain, OB/config.py:241; applied at onsets.py:79-105, :129-130):
- `zero`: NaN becomes 0.
- `drop`: a unit-season with any NaN is removed.
- `keep`: NaN counts 0 in totals, the candidate day must be finite, and a NaN day is neither wet nor dry
  (onset_rule.py:113, :127, :132).

ROMP's ingest flags NaN as missing (spec §5). Any missing day in a span makes it Unknown, which stops the search as
INSUFFICIENT_DATA. With 2% NaN days, `keep` gave 1541 onsets vs ROMP's 507 (2870 insufficient).
- Spec: put a single `missing: flag | zero | drop_season` option in ingest (`flag` = ROMP today).
- Retire blending's `keep` semantics in favour of three-valued logic, and re-baseline. `zero` and `drop` map exactly. No
  backend change.

(e) **Search start per (year, unit) and the fixed cutoff.**
- Observed: blending starts at `max(season_start, reference onset)` per unit-year (onsets.py:131-136).
- Forecasts: blending runs 3 variants (onsets.py:427-429, :502-521, :657-664):
  - base: lead 1;
  - fixed_cutoff: the first lead whose date is on or after the fixed_cutoff month-day;
  - ref: the first lead on or after the reference onset (a table per year and/or unit, OB/core/reference.py:227).
- ROMP takes starts only from `season.from` (fixed month-day, or `climatological onset` from season_climatology) in
  build_observed_jobs/build_forecast_jobs (MO/onset/engine.py:206-250, :500-562).
- `Jobs` already has a start per job and allows repeated series (model.py:286-301), so the backends need no change.
- Spec: `search_start: Mapping[str, xr.DataArray | "MM-DD"]` (dims year x cell, or init x cell) on
  detect_observed/detect_forecast. Each entry adds a slot to the existing season axis, with
  `start = max(season start, entry)`.
- All variants share one backend call and one set of prefix tables, as blending's loop over variants does
  (onset_rule.py:174-178).
- Keep it out of the definition, since the variant is an evaluation choice. Record it in attrs.

(f) **Threshold sources** (constant / per-unit table / computed, plus scale) are covered in section 4. The engine already
takes per-cell float thresholds (engine.py:82-116), so polygon units work as a `unit` spatial dim.

(g) **Season end.** Blending has no explicit end: it searches each season's whole series. ROMP requires `season.to`. The
translator must set it to the last day blending would read. Using 12-31 for calendar-year files matches.

### 1.3 Numerical notes that change results either way

- **Float ties (blending bug).** Blending's prefix-sum differences in float64 misjudge exact ties. Example: a window
  summing to 34.9 with threshold 34.9 under `gt` passes in blending, row 2425 in `diag.py`. ROMP's integer tenths are exact.
  Every residual mismatch in B/C2/E/F is this tie. Expect it on 0.1 mm-resolution data such as IMD and on 0.1 mm threshold
  tables.
- **Quantisation.** ROMP rounds rain half-even to int32 tenths (MO/onset/ingest.py:158, :205). It rounds thresholds by
  comparator: ge takes ceil, gt takes floor (plan.py:186-191). On unrounded float rain (forecasts, area-weighted unit means),
  rule A mismatched in 414/4000 series, for example 0.96 mm becoming 1.0 mm and so wet.
  - Option: make the integer scale a parameter (10 per mm today). It touches `ingest`, `resolve_amounts` and
    `units.parse_comparison`, plus the curve constant `100 * x` at reference.py:348 and its numba/gpu twins. The
    first-passage backends are scale-agnostic.
  - Otherwise accept tenths and re-baseline.

## 2. Forecast onsets

Blending, per model file (OB/core/onsets.py:634-682), produces over (issue, unit, lead day):
- `count_{base,fixed_cutoff,ref}`: uint8 members whose onset falls on each lead;
- `members` per issue;
- `rain_mean`: a Kahan member mean in the file dtype (:608-631, a legacy pandas match);
- `onset_threshold` per unit.

Probabilities are count/members, with every member in the denominator (:454-456). Rain predictors (OB/core/features.py:402-450)
take the max, min or "diff" (transform(max) - transform(threshold)) of rolling W-day sums of `rain_mean`, per week of starts
(`_complete_starts` :502). W is resolved from definition tokens such as trigger, dry_spell/sum_window, follow and min_dry
(OB/config.py:277, :890-904).

ROMP (`detect_forecast`, engine.py:813-896) produces per member: `onset_lead`, `onset_date`, `status`. For ensembles,
`_ensemble_summary` (:786-810) adds `members_by_status` and the cumulative `onset_probability` (share with lead <= k).
INSUFFICIENT_DATA members are excluded from the denominator. `forecast_history: observed` stitches the observed history
(PRIOR, back-dated leads). MO/stats/detect.py:259-275 turns members into an ensemble onset (mean day if fraction >=
onset_percentage_threshold), which is a deterministic scoring layer that blending does not need.

What blending needs from the engine beyond onset dates:
1. **Per-lead onset counts per start variant.** These are derivable from per-member `onset_lead`.
   - Add `onset_count (init, season|start, lead, cell)` to `_ensemble_summary`. It is the histogram that `onset_probability`
     already cumulates.
   - OB/blend/_crps_tree.py:299 `lead_day_histogram` is the same histogram, so after the merge one function serves both.
   - Denominator: blending uses all members. Under `horizon: truncate` with ingest `missing: zero`, no member is
     INSUFFICIENT, so the counts are identical. Expose `members` as `members_by_status.sum()`.
2. **The search-start variants:** see 1.2(e).
3. **Forecast-only search.** Blending must run with `forecast_history: none`. Stitching would back-date onsets whose
   trigger began before the issue, changing counts at leads 1-3.
4. **Truncated horizon:** see 1.2(c). This is mandatory for equivalence.
5. **Rain predictors are not engine work.** They need:
   (a) the rule's window lengths. ROMP's `FirstPassage` has no `window` field; it is folded into spans (definition.py:415,
   :460). Add `Definition.windows() -> {trigger, follow, dry_run, dry_window}` from the plan spans. describe.py:87
   `_window_of` already infers it. One accessor serves describe and rain predictors.
   (b) the float per-unit threshold in mm (the `thresholds` input), not the ceil/floor-rounded amounts.
   (c) a member-mean rain series. ROMP's ForecastRain is int32 tenths (model.py:249-283). Computing `rain_mean` from tenths
   shifts it by up to 0.05 mm per member-day, which changes features and therefore the trained forest. Recommend that the
   feature code take the member mean from the float cube before `ingest_forecast` rounds it, or accept and re-baseline.
   Per-day probability by lead and window rain sums do not need backend changes. Per-member `onset_lead` (init x member x
   cell, float64) is smaller than blending's counts (init x cell x 45 uint8) only for fewer than ~6 members. Use int16
   leads or emit counts directly if memory matters.

Efficiency: blending computes trigger and veto masks once and loops variants (onset_rule.py:110-178). ROMP's tables are built
once per detect call (reference.py:47), so putting all variants in one call (1.2e) keeps that property.

## 3. Climatology

**KDE core: SAME.** MO/onset/climatology_kde.py:1-21 says it is a port, and these match operation for operation:
- `fit_kde`: OB/core/climatology.py:218 = MO:38;
- `_cdf` = `onset_cdf`: :231 = :54;
- `_floored` = `floor_probabilities`: :242 = :69;
- `_predict_pair` = `onset_probabilities`: :283 = :113.

`kde.py`: on the same day axis the max abs difference is 0 over 300 random cells. On ROMP's axis it is 2.5e-8. ROMP counts
days since 1 Jan without leap shift (MO/stats/conditional_climatology.py:44-53); blending counts days since season_start
(OB/core/onsets.py:143). Translation changes the 512-point grid in the last bits. It is below MIN_PROB (5e-7) but breaks exact
equality tests. Fix: callers pass days since the season start.

**Around the core: EXTEND (ROMP gains blending's caller logic; nothing else is duplicated).** Blending-only, numbers-relevant:
- (i) the issue grid from season_start to `issue_end` with per-issue-date `horizons` (OB/core/climatology.py:157-186).
  ROMP's `onset_probabilities` takes one horizon, so the caller loops horizons as `_predict_pair` does.
- (ii) eligibility by `min_onset_years` inside the training window (:128-139), separate from the fixed MIN_ONSETS = 10 of the
  KDE. ROMP has only the latter.
- (iii) train/test year defaults (:103-125) and extra year-window climatologies (`windows`, :395-400).
- (iv) dedupe per distinct (season_day, horizon) and parallel tasks (:189-206, :326-367). This is a performance difference
  only.
- (v) the legacy tables (:84-100).

CV: blending leaves out the test year (`unit_years != year`, :204); ROMP uses `days[years != year]` (MO:186), so they are the
same. Move (i)-(iv) into a ROMP module (e.g. MO/onset/climatology_kde.py `probabilities_for_issues(...)`), used by both the
ROMP reference forecast (conditional_climatology.py:88) and the blend.

**Duplication inside ROMP.** There are two "climatological onset" implementations: the mean day of year rounded
(MO/stats/climatology.py:53-137) and the median (MO/onset/engine.py:939). Blending's `reference_onset` (a month-day or a table,
OB/config.py:401-420) is a third source of a "reference onset date". Make one provider of reference onset dates
(table | month-day | computed mean | computed median) feeding the derived seasons, the climatology forecast and
`search_start` (1.2e).

## 4. Thresholds and caching

**Threshold sources.**

| Blending (OB/config.py:352-398, OB/core/reference.py:164-209) | ROMP today | Unified |
|---|---|---|
| `mm: 20` | fixed amount in YAML (tenths only) | YAML fixed amount; non-tenth values supplied as an array |
| `file:` CSV by unit id/lat-lon, NetCDF, `.mat`, plus `scale` (factor, offset, min, max) | `threshold_files` NetCDF on the grid only (MO/io/input.py:362-365) | ROMP loader gains per-unit CSV/.mat and `scale`, as one reader |
| `climatology: {days, start, end, years}`: days x mean daily rain over finite days, on the regridded unit series (onsets.py:378-410) | `mean total over: N days, during: A to B` + run `threshold_years` (MO/onset/climatology.py:41-98) | ROMP spec |

Differences in the computed threshold that change numbers (`thr.py`, 500 cells, 0-20% NaN):
- ROMP sets a cell to NaN when more than 10% of its reference days are missing (climatology.py:15, :74). In `thr.py`, 256
  cells became NaN, which means no onset; blending had 0 NaN cells.
- Rain read as tenths moves the result by up to 0.012 mm.
- ROMP skips years whose `during` window is incomplete (:38); blending uses partial years.
- Blending computes on the unit series after regrid or cell transform; ROMP on whatever DailyRain it is given. For polygons
  that must be the unit series, not the grid, or the threshold is a different quantity.
- Blending defaults `days` to the trigger window; ROMP requires it explicitly.

Make `max_missing_fraction` a parameter. Blending's behaviour is a fraction of 1.0, though a cap is the better default.

**Caches: four key schemes today -> one.**
- Blending `_threshold_key` (OB/core/onsets.py:327-343): the config sections + spec + window + (path, size, mtime_ns) of every
  observation and referenced file (:357-359), versioned, with an atomic write (:362-375).
- ROMP `ThresholdCache.key` (MO/onset/climatology.py:342-348): source string + years + spec repr + grid coordinate bytes,
  with no file identity, so changed data under the same `source` reads stale. Two more problems:
  - With string unit ids (object dtype) the key hashes object pointers (`tobytes()` of an object array), so it never hits
    across processes. That matters for polygon units.
  - The write is not atomic (:358-359) while years run in parallel workers (commit 5b19614).
- ROMP `_model_thresholds` (MO/io/input.py:322-348): its own JSON key (model dir name, no file identity).
- ROMP `observed_cache` (MO/stats/observed_cache.py:27, :64-69): fingerprint parts including file (path, size, mtime_ns), with
  miss diagnostics.

Recommendation: generalise observed_cache's part-hash fingerprint into one `momp/cache.py`:
- `Store(kind).get_or_compute(parts: dict, compute)` under `runs/.cache/<kind>/<fingerprint>`;
- an atomic write;
- one `file_identity()` helper, which deletes blending `_fingerprint` and ROMP `_file_id`;
- string coords hashed by value.

Kinds: thresholds (obs and model), harmonic seasons, observed onsets, regrid/cell-transform weights.

## 5. Risks and how to verify equivalence

Risks, ordered by size of effect on blending results:
1. **Horizon censoring:** about 40% fewer forecast onsets at probability_days 28 (`fc.py`). Must add `horizon: truncate`.
2. **Missing-rain `keep` semantics:** large effect wherever NaN occur, such as partially covered polygons, gaps, or ROMP
   ingest's negative values becoming missing.
3. **Tenths quantisation of float rain and of regridded unit means** (about 10% of series moved one or more days in the
   unrounded test). The threshold ceil/floor rounding itself is exact.
4. **Computed threshold differences:** the 10% missing cap, partial years, and which series it is computed on.
5. **Semantic options** (dry_day_min_mm, gt dry days, maximal-run starts): only in configs not shipped today.
6. **Float-tie fixes:** rare, and ROMP is the correct one.
7. **KDE day axis:** about 1e-8.
8. **Registry name clashes** (moron_robertson, icpac).

Verification plan:
- **Rule oracle.** Port OB/tests/test_onset_rule.py and the `tests/_legacy/onset_utils.py` oracle (`find_onsets_batch` :702,
  `find_onset` :491) into ROMP's test suite. Run the translated definition through all three backends on 0.1 mm-rounded
  inputs, with `horizon: truncate` for forecast-style and censor for observed. Require exact equality except rows flagged
  as exact ties, where the integer sum equals the integer threshold; check those against an exact-decimal oracle.
  `cmp_rules.py` here is the template. Cover the cases above plus min_wet_days, follow 0, both anchors, both dry modes, and
  short series.
- **Translator tests.** Add a single blending-config -> Definition translator next to MO/onset/legacy.py. Round-trip each
  shipped config (configs/*.yml and the legacy specs, OB/legacy/specs.py) and assert that the extended options either
  translate or raise naming the missing extension.
- **End-to-end on real data.** Run the old package (pinned commit) and ROMP on the India adm3 config for 2-3 years. Compare
  observed onset_day per (unit, year), count_* per (issue, unit, lead), and features. Expect identical results with
  `missing: zero`, truncate, and tenth-rounded inputs. Report the distributions of the differences that come only from
  quantisation, then re-baseline the CV scores once.
- **Thresholds.** Re-point OB/tests/test_thresholds.py at `compute_threshold` with `max_missing_fraction=1.0` on unit series.
  Assert agreement to 0.05 x days mm, or exact with a float scale. Keep the cache-reuse tests (:85-126) against the unified
  store.
- **Climatology.** Keep OB/tests/test_climatology.py (legacy KDE oracle, :11-36) pointed at MO/onset/climatology_kde.py
  with days since season start, which gives exact equality. Port the issue-grid test to the new ROMP grid module.
