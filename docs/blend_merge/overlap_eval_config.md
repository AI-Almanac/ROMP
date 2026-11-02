# Overlap report: evaluation, metrics, config, entry points (onset_blending -> ROMP)

Scope: blending `core/{features,design,predictions,metrics,calibration}.py`, `blend/*`, `evaluation.py`,
`training.py`, `trained.py`, `forecast.py`, `config.py`, `__main__.py`, `legacy/specs.py`,
`docs/pipeline-redesign.md`, compared with ROMP `momp/metrics`, `momp/stats/{score,bins,benchmark,
climatology,conditional_climatology}.py`, `momp/lib/*config*`, `check.py`, `run_folder.py`, `api.py`, `cli.py`,
`driver.py`, `io/output.py` and `graphics/reliability.py`.
Paths: `OB/` = /home/hcwf/code/onset_blending/onset_blending, `M/` = /home/hcwf/code/ROMP/momp.

Summary of the shape: ROMP scores **long pair tables** (one row per init x cell x bin, `M/stats/bins.py:66-198`),
pooled over the region, ensemble-aware (fair scores). Blending scores **wide (samples, bins) matrices**
(`OB/core/predictions.py`, `OB/core/metrics.py`), per unit, pooled and per year, for non-ensemble
probabilities (blends, calibrated, KDE climatology). Both compute the same quantities, but with different
normalisation and different bin sets. Neither has the other's whole feature set.

---------------------------------------------------------------------------------------------------------
## 1. Metrics

| Metric | Blending | ROMP | Verdict | Numeric differences |
|---|---|---|---|---|
| Brier (pooled) | `OB/core/metrics.py:29-30`, pooled row `:124` | `calculate_brier_score` `M/stats/score.py:14-79`; climatology variant `:277-357` | SAME concept, **numbers differ** | (a) Blending sums over bins per sample then averages over samples: `mean_i sum_j (p-y)^2`. ROMP averages over all (case, bin) rows, `mean_ij`, so ROMP = blending / m (m = number of scored bins). (b) Bin sets differ: blending scores week1..N **and `later`** (probabilities sum to 1); ROMP scores only the interior `day_bins` (`extract_pd_bins`, `M/metrics/skill.py:128`, `M/io/dict.py:66`), so Before/After are dropped and the rows do not sum to 1. (c) ROMP also reports **fair Brier** `-p(1-p)/(M-1)` (`score.py:56-60`) and refuses M<=1 (`:40-46`); blending has no fair term (`fair_brier` is a dead legacy key, `OB/legacy/specs.py:960`). |
| Brier per bin | pooled `brier_<label>` `OB/core/metrics.py:130-131` | `bin_brier_scores`, `bin_fair_brier_scores` `score.py:66-67` | SAME | Both are `mean_i (p_ij-y_ij)^2` per bin: numerically identical over the same samples. Blending also has `brier_later`, which ROMP has no equivalent for. |
| RPS | `rps` `OB/core/metrics.py:33-37`; `earlier` column prepended for `unc_clim_raw` `:95-100` | `calculate_rps` + `_rps_per_forecast` `score.py:169-274` | SAME concept, numbers differ | Both are `mean_i sum_k (CP_k - CY_k)^2` with no 1/m: the same formula. Different bin sets as in the Brier row: ROMP's cumulative sum runs over the interior bins only, while blending's runs over earlier?+weeks+later, where the last term is 0. ROMP also has **fair RPS** (`:197`). ROMP's per-case grouping is by (init_time, lat, lon) (`:254`). Blending's sample is (unit, issue). |
| AUC | Wilcoxon `rankdata(average)` `OB/core/metrics.py:40-50`; yearly uses `sklearn.roc_auc_score` `:149` | `mannwhitneyu(..., 'greater').U / (n1 n0)` `score.py:131-134`, per bin `:137-159` | SAME | Both pool every (sample, bin) pair and count ties as 1/2, so they agree to rounding. Different bin sets again: blending includes `later`, ROMP only the interior bins. `roc_auc_score` and Wilcoxon are identical. |
| AUC per bin | `auc_<label>` `:132-133` | `bin_auc_scores` | SAME | identical over the same samples |
| Pietra index | KS statistic, `OB/core/metrics.py:53-59` | none | NEW | |
| Per-unit probabilistic metrics | `cell_metrics` `OB/core/metrics.py:76-134` (id rows + `ALL`) | none. ROMP's per-cell maps are deterministic only (`M/metrics/error.py:10`) | NEW | |
| Per-year metrics | `yearly_metrics` `:137-156` | none (ROMP pools all years) | NEW | |
| Skill vs a baseline | `pooled_skill` `:159-169`: BSS, RPSS, `AUC diff`x100 vs `unc_clim_raw` | `calculate_skill_scores` `score.py:446-560`: fair BSS, fair RPSS, per-bin fair BSS vs `ref_model` | SAME (1 - S/S_ref) | ROMP uses the fair scores. AUC diff is NEW. ROMP's per-bin BSS is NEW to blending. |
| Reliability diagram | none | `M/graphics/reliability.py:12` (10 probability bins over pair rows) | EXTEND blending (free once predictions convert to pairs) | |
| Deterministic MAE/FAR/MR | none | `M/stats/benchmark.py:25`, `M/metrics/error.py:10` | ROMP-only; blending gains it if blended onsets are made deterministic (not required) | |
| Probability cleaning | `clean` `OB/core/metrics.py:62-73` (used by MME) | none | NEW (MME only) | |
| Platt calibration | `OB/core/calibration.py` (CV per bin, `glm(y~p)`; export on the logit) | none | NEW | |

**Recommendation (one implementation).** Keep **one array kernel** module, `momp/metrics/probabilistic.py`,
seeded from `OB/core/metrics.py`. It is already vectorised per unit and per year, and the matrix form fits
both callers. Extend it with:
- `members=` (an array or None) for the fair corrections of Brier and RPS. ROMP's ensemble and climatology-ensemble
  scoring keep their fair numbers (`total_members` / `total_members_with_onset`), and the conditional
  climatology keeps infinite members, so its correction is 0 (`M/stats/conditional_climatology.py:10-13`);
- `bins=` (a subset of columns scored: ROMP's interior day_bins vs blending's weeks+later);
- `brier_mean="sample"|"cell"` to reproduce ROMP's `/m` normalisation for the existing ROMP CSV columns
  (`M/io/output.py:15-64`) until a deliberate, documented switch is made.
ROMP's `calculate_brier_score`, `calculate_brier_score_climatology`, `calculate_auc`, `calculate_auc_climatology`
and `calculate_rps` become thin adapters that pivot the long pair table (`bin_index` order, `score.py:178`) into
(cases, bins) + members and call the kernel. Delete their separate loops; the two `*_climatology` copies
differ only in the 'Before' filter and the member column. `calculate_skill_scores` and `pooled_skill`
merge into one `skill(score, ref)`. Before switching, test that the existing ROMP CSVs reproduce
bit-for-bit, as both repos already require.

---------------------------------------------------------------------------------------------------------
## 2. Bins / weeks

| | Blending | ROMP |
|---|---|---|
| Definition | `features.bins.count` (4) x `bins.days` (7) `OB/config.py:464-466`; labels `week1..weekN, later` `OB/config.py:973-974` | arbitrary `day_bins` tuples, default 6x5 days `M/lib/config_schema.py:405,556`; `max_forecast_day` `:555` (overwritten by `max(day_bins)[1]`, `M/stats/bins.py:86`) |
| Lead convention | `lead = onset - issue (days)`; week w: `(w-1)*d < lead <= w*d`; lead<=0 or NaN -> no outcome (-1) `OB/core/features.py:115-130, 318` | observed in bin if `init+start <= onset <= init+end` `M/stats/bins.py:141-143`; only `init < onset` kept `:116`. The **same convention** (lead 1 = day after issue). |
| Edge bins | `later` (> N*d), scored; `earlier` only as an RPS column for `unc_clim_raw` | `Before day k` and `After day N`, built but **not scored** (`extract_pd_bins`) |
| Days used for features | `features.probability_days` (28) must be >= count*days `OB/config.py:713`; RF `max_day` must equal count*days `:800-804` | `max_forecast_day` |
| Ensemble bin probs | member onset counts per lead day / members, summed over the week (`OB/core/features.py:109-112, 386`) | `members_with_onset/total_members` per bin `M/stats/bins.py:144-172` |

**Unify: yes.** Put one `LeadBins` type in `momp/stats/bins.py` with edges (any `day_bins`, uniform weekly as
a constructor), outcome codes from lead days (the blending `_outcome_codes` logic; ROMP's `observed` masks
are the same predicate), label styles (`week1/later` for the blend output contract, `Days a-b`/`After day N`
for ROMP CSVs), and the member-count-to-bin-probability step used by both ROMP pairs and blending features.
Config: blending `features.bins.{count,days}` becomes `evaluation.day_bins` (e.g. `[[1,7],[8,14],[15,21],[22,28]]`)
and `probability_days` becomes `max_forecast_day`. The blend still needs a `later` column, so the blend scorer
passes `bins=interior+later`. A blend needs **contiguous bins starting at day 1** (its multinomial
classes are the bins), so validate that in the blend section only.

---------------------------------------------------------------------------------------------------------
## 3. Config

Two schemas exist: blending's frozen-dataclass tree with a `validate()` cross-check (`OB/config.py:606-880`,
the loader `_build/_convert` `:1004-1105`), and ROMP's flat-key `Field` table with sections
(`M/lib/config_schema.py:66-571`), catalog-aware dataset resolution (`M/lib/config_yaml.py:296`) and the
`Config` notebook objects generated from it (`M/config.py:104-127`). **Keep ROMP's schema machinery**
(one loader, one did-you-mean, one `--check`) and add a `blend:` section. Its nested structure needs a
`kind: "mapping"` Field or a sub-schema, because ROMP fields are flat today. The dataclass loader in
`OB/config.py:1004-1182` is deleted.

Proposed merged YAML:
```yaml
region: india                      # ROMP region registry (M/onset/regions/*.yaml)
mask: {...}                        # ROMP
units: {shapefile: ..., id_col: adm3_name, ...}   # NEW top-level: polygon units (blending domain.regrid)
observations: {dataset|path, variable, units, dims, day_boundary}
models: [{name, dataset|path, variable, units, dims, members, lead_label, ...}]
onset: {definition, override|rule, season, thresholds, threshold_years, threshold_source, ...}
reference: {model: conditional_climatology, climatology_cv}
evaluation: {start_date, end_date, start_year_clim, end_year_clim, init_days, day_bins, max_forecast_day, ...}
metrics: {BS, RPS, AUC, Reliability, skill_score, Pietra?}
output: {data_dir, ...}
blend: {years, cutoff_mode, cv_methods, formulas, final_formula, window_variants, baselines,
        forecasts, mme, external_predictions, random_forest, features: {rain_transform, rain_predictors...}}
```

### BlendConfig key -> ROMP home
| BlendConfig key (`OB/config.py` line) | Proposed home | Note |
|---|---|---|
| `name` :609 | `project_name` | |
| `data_root` :610 | `work_dir` (make it functional) / YAML-relative paths | ROMP `work_dir` is informational today (`config_schema.py:438`) |
| `domain.grid_ids.{decimal_digits,format}` :170-175 | `units.grid_ids` (blend-only; ids are derived) | id formatting is needed for the output contract |
| `domain.regrid.*` (shapefile, unit_id_col, unit_name_col, parent_id_col, crs, grid_lat/lon_var, clip_to_coverage, weights) :186-199 | new top-level `units:` | ROMP `polygon`/`shpfile_dir`/`mask` (`:444-447`) only mask cells; scoring on polygons is NEW. The grid_lat/lon vars come from ROMP dims detection |
| `domain.dissemination`, `dissemination_scope` :226-231 | `units.dissemination`, `units.dissemination_scope` | NEW |
| `domain.centroids` :232 | `units.centroids` | NEW |
| `domain.area_of_interest.{bbox,geometry}` :211-216 | `mask` (add `bbox`), `mask.file` for geometry | collapses into ROMP mask |
| `observations.files.{folder,pattern,paths,years}` :149-157 | `observations.path`, `file_pattern` (ROMP uses `{}` = year, blending uses regex) | ROMP detection resolves patterns. Drop `paths`/regex or add `paths:` to the ROMP dataset entry |
| `observations.variable`, `dims` :239-240 | same in `observations` | |
| `observations.missing_rain` :241 | `observations.missing_rain` (new key) | |
| `observations.cell_transform` :244 | `units.cell_transform` | |
| `observations.date_marks` :245 | `observations.day_boundary` (ROMP's "labelled by start/end") | collapses |
| `models[].name, files, variable, dims` :293-296 | `models[]` entry (`MODEL_ENTRY_KEYS` `config_schema.py:577`) | |
| `models[].unit_scale` :301 | `models[].units` (ROMP derives `unit_cvt`) | collapses |
| `models[].max_member` :306 | `models[].members` / `evaluation.members` | collapses |
| `models[].lead_days.{min,max}` :256-258 | `max_forecast_day` + ROMP lead detection | collapses |
| `models[].rain_horizon.{max_day,policy}` :262-266 | `blend.features.rain_horizon` (per model override) | blend-only |
| `models[].rain_predictors[]` :281-289 | `blend.features.rain_predictors` keyed by model | blend-only |
| `models[].rain_transform` :310 | `blend.features.rain_transform` per model | |
| `models[].cell_transform`, `regrid_weights` :311-312 | `units.weights` per model grid | |
| `onset.definition.*` (window, trigger_rule, min_wet_days, wet_day_min_mm, comparisons, follow_days, followup_anchor, dry_spell.*) :316-341 | `onset.definition` (registry id) / `onset.rule` / `onset.override` | **must collapse**: ROMP's one definition system (`M/onset/definition.py`). The converter maps each blending definition to a rule. |
| `onset.threshold.mm` / `file` / `climatology.{days,start,end,years}` / csv cols / `scale` :353-384 | `onset.thresholds`, `onset.threshold_years`, `threshold_source`, threshold in the definition | CSV-per-polygon tables + `scale` are NEW to ROMP thresholds |
| `onset.season_start` :427 | definition `season.from` (via `onset.override`) | collapses |
| `onset.fixed_cutoff` :430 | `blend.fixed_cutoff` | blend-only |
| `onset.reference_onset.*` :402-412 | `blend.reference_onset` | blend-only (`cutoff_mode: ref_onset`) |
| `climatology.issue_end` :445 | `evaluation.end_date` month-day | collapses |
| `climatology.forecast_window`, `horizons` :446-447 | `max_forecast_day` / `blend.climatology.horizons` | |
| `climatology.cv_by_year` :448 | `reference.climatology_cv` (`config_schema.py:495`) | **same concept** |
| `climatology.min_onset_years` :449 | ROMP KDE constant (>=10, `config_schema.py:491`) -> make it a key | |
| `climatology.train_years/test_years/windows` :450-456 | `start_year_clim`/`end_year_clim` + `blend.baselines.windows` | |
| `climatology.workers` :460 | `parallel.workers` | collapses |
| `features.bins.*`, `probability_days` :464-472 | `evaluation.day_bins`, `max_forecast_day` | see section 2 |
| `features.join`, `trim_after_true_onset` :473-476 | drop `join` (inner only) / ROMP already keeps only inits before onset (`bins.py:116`) | collapses |
| `features.rain_transform` :477 | `blend.features.rain_transform` | |
| `evaluation.years.{training,cv_holdout,true_holdout}` :481-484 | `blend.years` (ROMP `years` reserved key `config_schema.py:453` could host training) | NEW |
| `evaluation.cutoff_mode` :579 | `blend.cutoff_mode` | |
| `evaluation.cv_methods`, `cluster_var` :580-581 | `blend.cv.methods`, `blend.cv.cluster_var` | |
| `evaluation.formulas`, `final_formula`, `window_variants` :582-586 | `blend.formulas`, `blend.final_formula`, `blend.window_variants` | |
| `evaluation.climatology_baselines.{unconditional,conditional,windows}` :511-514 | `reference.model` (+ list of extra baselines in `blend.baselines`) | ROMP `climatology` (empirical year-members) and `conditional_climatology` (KDE) vs blending `unc_clim_raw` (KDE unconditional, with `earlier`) and `clim_raw` (KDE conditional). The conditional ones should become one implementation (other slice). |
| `evaluation.forecasts[].{model,variant,raw,calibrated,platt_weights}` :520-527 | `blend.forecasts` | raw-model scoring overlaps ROMP's probabilistic model scoring |
| `evaluation.mme.*` :557-566 | `blend.mme` | |
| `evaluation.external_predictions[]` :570-573 | `blend.external_predictions` | |
| `evaluation.random_forest.*` :538-553 | `blend.random_forest` (`max_day` derived from day_bins) | |
| `outputs.dir` :598 | `output.data_dir` / run folder | collapses |
| `outputs.intermediates`, `pickles` :599-602 | `output.save_*` toggles (`config_schema.py:406-421`) | collapses |

Derived helpers `OB/config.py:882-993` (`output_tag`, `year_tag`, `cutoff_tag`, `feature_bases`, ...) move with
the blend config. `output_tag` must be kept for the contract filenames (section 6).

**Multiple regions.** ROMP's `region` is one required id (`config_schema.py:442`) that drives defaults for
the definition, season and climatology years (`M/lib/onset_config.py:129-147`). Blending has no region; it
has one domain per config. Proposal: one run file = one region + optional polygon `units:`; several regions
= several run files (run folders are named by region, `M/lib/run_folder.py:116`). If one blend must span
regions, allow `region: [a, b]` only when they share a definition. Defer that.

---------------------------------------------------------------------------------------------------------
## 4. Entry points

Today:
- Blending: `onset-blend train CONFIG -o DIR [--intermediate observed|weekly] [--workers] [--cache-dir]`;
  `forecast BLEND --forecast MODEL=FILE... [--issue-date] -o DIR`; `convert-specs ...` (`OB/__main__.py:49-85`).
  Python: `train(BlendConfig)` -> `TrainingResult.save()` (`OB/training.py:126-184`),
  `TrainedBlend.load().forecast()` (`OB/trained.py:334,406`).
- ROMP: `momp-run -p run.yaml [--check] [--overwrite] [--quiet|--verbose] [overrides]`, `init`, `datasets`,
  `convert old.in`, `export` (`M/cli.py:1-42,160-180`). `--check` -> `M/lib/check.py` via `M/driver.py:38-57`.
  Python: `momp.run(cfg, write=...)` / `momp.check(cfg)` (`M/api.py:129-152`). Run folders + fingerprint
  (`M/lib/run_folder.py:1-17,94`).

Proposed merged surface (one CLI, one notebook API):
| Command | Behaviour |
|---|---|
| `momp-run -p run.yaml` | as today. If the config has a `blend:` section, **benchmark + train + CV-evaluate** in the same run folder (scores/, plus `trained.blend`). Replaces `onset-blend train`. `--intermediate` becomes `output.save_*` toggles; `--workers` becomes `parallel.workers`; `--cache-dir` becomes `runs/.cache` (ROMP's shared threshold/onset cache already does this, `run_folder.py:15`). |
| `momp-run -p run.yaml --check` | add blend checks: formulas resolve against the features (`OB/core/design.py:173`), years split, MME sources, dissemination units present, signatures consistent across training files (`OB/trained.py:53,143`). Runs `BlendConfig.validate` rules, ported into ROMP schema checks. |
| `momp-run forecast RUN_DIR/trained.blend --forecast MODEL=FILE ... [--issue-date] -o DIR` | new subcommand; same semantics as `OB/__main__.py:36-46`. No config overrides allowed (redesign rule, `docs/pipeline-redesign.md:217-230`). Its `--check` equivalent = `TrainedBlend.check_forecast_files` (`OB/trained.py:379`). |
| `momp-run convert` | extend: `convert old.in` (existing) **and** `convert --specs ...` (blending `legacy/specs.py` `main`, `:1139`), both writing a ROMP run.yaml. One converter command. |
| `momp-run init` | add `--blend` to write a `blend:` skeleton (from `configs/country_template.yml`). |
| Python | `momp.run(cfg)` returns `RunResult` with `.blend` (Evaluation tables + TrainedBlend) when `blend:` is set; `momp.TrainedBlend.load(path).forecast(...)`; `momp.check(cfg)`. Drop `onset_blending.train`/`TrainingResult` as separate entry points: `TrainingResult.save` atomic staging (`OB/training.py:132-158`) folds into run-folder writing. |

Run folder fingerprint: add the `blend:` keys to the fingerprint and exclude worker counts
(`_NOT_IN_FINGERPRINT` `run_folder.py:31-37`). The ROMP parser's per-key overrides (`M/lib/parser.py:7`) must
not expose blend keys on `forecast`.

---------------------------------------------------------------------------------------------------------
## 5. Purely new modules: proposed locations

| Blending module | Proposed `momp/` location | Notes |
|---|---|---|
| `OB/core/features.py` (Rows, Weekly, rain predictors, DailyInputs) | `momp/blend/features.py` | must consume ROMP onsets/climatology/ingest instead of `OB/core/onsets.py`/`climatology.py` (other slices). `_week_sums`/`_outcome_codes` go to `momp/stats/bins.py` (`LeadBins`) |
| `OB/core/design.py` (Plan, formulas, sample_support) | `momp/blend/design.py` | needs `patsy` |
| `OB/core/predictions.py` (`Predictions`, outcome codes) | `momp/blend/predictions.py`, or better `momp/stats/predictions.py` as the **one wide probability container**, with `.to_pairs()` producing ROMP's pair table for reliability/heatmaps | |
| `OB/core/metrics.py` | `momp/metrics/probabilistic.py` (merged kernel, section 1) | |
| `OB/core/calibration.py` (Platt CV + export) | `momp/blend/calibration.py` | sklearn, threadpoolctl |
| `OB/blend/logit.py` (FittedModel, global/local/cluster CV) | `momp/blend/logit.py` | CV-year logic shared with RF folds; consider `momp/blend/cv.py` for `folds` (`OB/blend/random_forest.py:185`) + holdout handling |
| `OB/blend/random_forest.py`, `_crps_tree.py` | `momp/blend/random_forest.py`, `momp/blend/_crps_tree.py` | numba (ROMP already has it), joblib. Its `_jit` cache fallback duplicates any ROMP numba-cache helper in `M/onset/backends/numba_cpu.py`; keep one |
| `OB/blend/mme.py` | `momp/blend/mme.py` | uses the merged metrics `rps`/`clean` |
| `OB/evaluation.py` | `momp/blend/evaluation.py` | scoring goes through the merged kernel; baselines via ROMP reference models |
| `OB/training.py` | `momp/blend/training.py` (called from `momp/api.execute`) | `Resources` and `core/parallel.py` collapse into `M/stats/parallel.py` |
| `OB/trained.py` (TrainedBlend zip, digests, FEATURES_VERSION, signatures) | `momp/blend/trained.py` | `forecast_signature` reading of units/dims/cell_methods overlaps ROMP's `M/lib/dataset_detect.py` (units `:221`, dims, lead_label `:268`) and `M/onset/ingest.py`; build signatures from ROMP's resolved dataset entry, so there is one detector |
| `OB/forecast.py` | `momp/blend/forecast.py` | |
| `OB/legacy/specs.py` | `momp/lib/convert_specs.py` (beside `M/lib/convert.py`, the config.in converter) | output must be a ROMP run.yaml, not a BlendConfig |
| `OB/config.py` | removed; keys go into `M/lib/config_schema.py` (`blend` section) + `M/config.py` `Blend` section class | validation rules ported |

---------------------------------------------------------------------------------------------------------
## 6. Output contract to preserve (ai-almanac; `docs/pipeline-redesign.md:91-104`)

| File | Columns / content | Written by |
|---|---|---|
| `summary_models_pooled<tag>.csv` | `model, auc, brier, rps, brier_skill, rps_skill, pietra, n, auc_week1..N, auc_later, brier_week1..N, brier_later` (+ `id=ALL, lat, lon, cv_method, AUC diff`); models incl. `unc_clim_raw`, `clim_raw`, `blended_model` | `OB/evaluation.py:204-206`, `OB/core/metrics.py:120-169` |
| `summary_models<tag>.csv` | per-unit `id, model, lat, lon, n, rps, brier` (+auc, cv_method) and an `ALL` row | same |
| `yearly_metrics_global<tag>.csv` | `year, model, cv_method, brier, rps, auc` | `OB/core/metrics.py:152-155` |
| `blended_forecast_probabilities.csv` | `id, time, onset_threshold, lat, lon, cv_week1..N, cv_later, clim_week1..N, clim_later` | `OB/forecast.py:31,221-233` |
| `daily_onset_probabilities.csv` | `id, time, lat, lon, p_day_1..28, p_later` (+ week sums) | `OB/forecast.py:32,47-50` |
| `trained.blend` | zip: manifest.json (format `onset-blending/trained-blend` v1, features_version, digests), config.yml, model.json/npz, forest.*, observed.npz, units.json, signatures.json, reference.* | `OB/trained.py:41-44,281-331` |
| other artifacts | `cv_preds_<model>_<method><tag>.nc`, `coefs_*`, `platt_weights_*`, MME weights CSV, `formula_sample_support<tag>.json`, `config.yml`, optional `weekly_features.nc`, `observed_onsets.nc` | `OB/evaluation.py:173-215`, `OB/training.py:132-158` |

Constraints this puts on the merge: the Brier stored in these CSVs is the **sum-over-bins, non-fair**
blending number, over week bins + `later`. The merged kernel must reproduce that exactly (`brier_mean="sample"`,
`bins=weeks+later`, `members=None`). The `<tag>` comes from `OB/config.py:977-993` (cutoff + year tags) and stays
in the file names even inside a ROMP run folder, so blend files must **not** take ROMP's short-name
scheme (`run_folder.py:227`). The FORMAT string in `trained.blend` should stay (or be versioned with
backward-compatible load), because ai-almanac stores these files. Bin labels `weekN/later` and the `cv_` prefix are fixed.

---------------------------------------------------------------------------------------------------------
## 7. Dependencies

| Package | blending (`pyproject.toml`) | ROMP | Action |
|---|---|---|---|
| scikit-learn>=1.3 | yes (logit, Platt, AUC) | **no** (not imported anywhere in `momp/`) | add |
| patsy>=0.5 | yes (design) | **no** | add |
| joblib>=1.3 | yes (forest trees) | **no** (only transitively via sklearn) | add |
| threadpoolctl>=3 | yes (1 BLAS thread per fit) | **no** | add |
| packaging | yes | no | add (or drop the use) |
| numba | >=0.60 | >=0.68.1 | ROMP pin wins; check that `_crps_tree` caching works on it |
| numpy, pandas>=2.0, scipy>=1.10, xarray>=2023.1, netCDF4, pyyaml | yes | yes (unpinned) | keep ROMP's; consider pandas>=2 |
| geopandas/pyproj/shapely (`regrid` extra) | optional | geopandas required (brings shapely, pyproj); regionmask | covered |
| cartopy, matplotlib, seaborn (`maps` extra) | optional | required | covered |
| python | >=3.11 | >=3.10,<3.14 | blending code uses no 3.11-only syntax found (`kw_only` dataclasses are 3.10); keep ROMP's range, run tests on 3.10 |

Console script `onset-blend` is removed (folded into `momp-run`); keep it only as a deprecation alias if
ai-almanac's local runner calls it.
