# onset_blending -> momp coverage audit (final verification wave)

Oracle: onset_blending@1e6d696 (`/home/hcwf/code/onset_blending/onset_blending/`). momp = `/home/hcwf/code/ROMP/momp/`.
Verdicts: HOME (file:symbol), RETIRED (reason), MISSING. Line numbers are as of this audit (other agents were editing
`momp/io/input.py`, `momp/blend/forecast.py`, graphics concurrently).

## 1. Coverage table

### Modules, classes, functions

| onset_blending | momp home / verdict |
|---|---|
| `__main__.main` (train, forecast, convert-specs) | `momp/cli.py` (`momp-run`, `momp-run forecast`, `momp-run convert [--specs]`); `momp/blend/forecast.py:parse_forecast_arguments` |
| `training.Resources`, `core/parallel.default_resources` | `momp/blend/training.py:resources`, `momp/utils/workers.py:default_resources` |
| `training.Inputs`, `prepare` | `momp/blend/training.py:Inputs, prepare` |
| `training.TrainingResult`, `train`, `TRAINED_BLEND_FILENAME` | `momp/blend/training.py:TrainingResult, train`; `momp/blend/trained.py:TRAINED_FILENAME` |
| `config.BlendConfig`, `load_config`, `ConfigError`, all section dataclasses | RETIRED as a class tree: run YAML (`momp/lib/config_schema.py`, `blend_*` fields) + typed `momp/blend/recipe.py:Recipe, recipe_from_config, RecipeError`; BlendConfig YAML -> run YAML via `momp/lib/convert_specs.py:convert_blend, convert_blend_file` |
| `config.FileSet, GridIds, UnitTable, Regrid, BBox, AreaOfInterest, Domain, Observations, LeadDays, ForecastModel` | run YAML `observations/models` (`paths`, `years`), `units:` (`momp/spatial/units.py:UnitsSpec`), `region`/box; `blend.models.<m>.lead_days` (`recipe.py:Model`) |
| `config.RainHorizon, RainPredictor` | `recipe.py:RainHorizon, RainPredictor` |
| `config.DrySpell, OnsetDefinition, ThresholdScale, ThresholdClimatology, Threshold, Onset` | ROMP definition (`onset: {rule: ...}` via `convert_specs._rule_mapping`), `momp/onset/thresholds.py:ThresholdScale`, `momp/onset/climatology.py:compute_threshold` |
| `config.ReferenceOnset` | `recipe.py:ReferenceSpec`; reading: `momp/onset/climatology.py:ReferenceOnset.read` |
| `config.ClimatologyHorizon, Climatology, Bins, Features, Years, Formula, WindowVariant, ClimatologyBaselines, ForecastEvaluation, MmeSource, Mme, ExternalPrediction, Evaluation, Outputs, RandomForest` | same names in `recipe.py` (`ClimatologySpec, Features, Years, design.Formula/WindowVariant, Baselines, ForecastScoring, MmeSource, Mme, ExternalPrediction, Evaluation, Outputs`); random forest `random_forest.Params` |
| `config.window_tag, rain_feature_base, probability_feature_base, cutoff_variant, probability_variants, feature_bases, formula_terms, climatology_baseline_names, outcome_labels, cutoff_tag, year_tag, output_tag, CLIMATOLOGY_FEATURE, VARIANT_SUFFIX` | `recipe.py` (module functions / `Recipe` methods; `resolve_window` -> `Recipe.window_days` + `Definition.windows()` `momp/onset/model.py:228`) |
| `evaluation.RunSettings, prepare_inputs, log_sample_support, Evaluation, evaluate, fit_final, final_rows, dissemination_table` | `momp/blend/evaluation.py` (same names; `dissemination_table` -> `dissemination_ids`, units table `momp/spatial/units.py`) |
| `forecast.run, Forecast, IncompatibleForecast, WEEKLY_FILENAME, DAILY_FILENAME` | `momp/blend/forecast.py` (same) |
| `trained.TrainedBlend, forecast_signature, issue_dates, FORMAT*, FEATURES_VERSION` | `momp/blend/trained.py:TrainedBlend` (old files still load: `OLD_FORMAT`); signature: `momp/lib/dataset_detect.py:signature_differences` (+ `TrainedBlend.signatures`); `Forecast.issue_dates` |
| `core/calibration.cross_validated, export_weights` | `momp/blend/calibration.py` |
| `core/climatology.fit_kde, build, Climatology, years, MIN_PROB, EPS` | `momp/onset/climatology_kde.py` (`fit_kde, build_climatology, Climatology`); `momp/blend/features.py:climatology` |
| `core/design.*` | `momp/blend/design.py` (same names) |
| `core/domain.SparseTransform, Regridder, regrid_weights, point_sources, grid_axes, AreaFilter, id_convention, with_ids` | `momp/spatial/regrid.py:SparseTransform, read_weights, point_sources, grid_axes`; `momp/spatial/units.py:EvaluationUnits` (AreaFilter -> `UnitsSpec` box/outline/dissemination); `spatial/ids.py:with_ids` |
| `core/features.rows, weekly, daily_inputs, rain_transform, Rows, Weekly, DailyInputs` | `momp/blend/features.py`, `momp/blend/samples.py` |
| `core/ids.*` | `momp/spatial/ids.py` (same names) |
| `core/io.*` (`list_files, file_year, FileSet reading, ForecastFile, ObservationArray, read_observations, forecast_format, rainfall_scale, MILLIMETRES_PER_UNIT, lead_days_in_order, LeadSelection, adm3_names, nc_time, time_to_dates, forecast_points`) | RETIRED, replaced by ROMP's one reader: `momp/io/input.py` (`load_imd_rainfall`, `UnitForecast`, `read_on_units`, `load_blocks`), `momp/onset/ingest.py` (`unit_scale`, `ingest`), `momp/lib/dataset_schema.py:file_year`, `resolve_paths`, `momp/utils/standard.py:ROLE_ALIASES`; xarray/cftime time decoding |
| `core/metrics.*` | `momp/metrics/probabilistic.py` (same names; `stats/score.py` adapters) |
| `core/onset_rule.OnsetRule, find_onsets` | RETIRED: `momp/onset/engine.py` options (`rain_resolution: exact, missing, horizon, ...`) |
| `core/onsets.observed, forecast, ObservedOnsets, ForecastOnsets, member_mean, climatological_thresholds, observation_context, forecast_context` | `momp/stats/detect.py` (adapter), `momp/blend/features.py:observed, forecast, member_mean`, `momp/onset/climatology.py:compute_thresholds`, `momp/io/input.py:load_thresholds` |
| `core/parallel.map_ordered, workers_within, limit_blas_threads, SharedArrays, attach, detach` | `momp/stats/parallel.py:map_items`, `momp/utils/workers.py` (same names) |
| `core/predictions.*` | `momp/blend/predictions.py` |
| `core/reference.Reference, read_thresholds, read_reference_onset, read_cell_transform` | `momp/onset/thresholds.py`, `momp/onset/climatology.py:ReferenceOnset`, `momp/spatial/regrid.py:read_weights` |
| `blend/logit.*`, `mme.*`, `random_forest.*`, `_crps_tree.*` | `momp/blend/logit.py, mme.py, random_forest.py, _crps_tree.py` (same names) |
| `legacy/specs.convert_specs, convert_spec_files, main, Conversion` | `momp/lib/convert_specs.py` (same names), CLI `momp-run convert --specs` |

### CLI options

| onset_blending | momp |
|---|---|
| `train CONFIG -o DIR` | `momp-run -p run.yaml` (run folder `blend/`; `output.data_dir`) |
| `train --intermediate observed|weekly` | `blend.outputs.intermediates` (observations, forecasts, climatology, features; only observations->`observed_onsets.nc` and features->`weekly_features.nc` exist, `momp/api.py:246`). Differs: momp writes `weekly_features.nc` by default (see Disk writes) |
| `train --workers N` | RETIRED to YAML `parallel: {workers: N}` (no CLI flag; only `--parallel True|False`). MISSING as a flag (minor) |
| `train --cache-dir` | YAML `cache_dir` (cache is on by default in `<runs>/.cache`). No CLI flag (minor) |
| `-v/--verbose` | `--verbose` / `--quiet` |
| `forecast BLEND --forecast M=F --issue-date -o` | `momp-run forecast BLEND... --forecast MODEL=PATH --issue-date -o` (adds `--fewer-members`, globs, several blends) |
| `forecast --workers` | MISSING (minor: forecast follows the run's default workers) |
| `convert-specs --name --ground-truth --forecast --combine --connect --cv-models --regrid --data-root -o` | `momp-run convert --specs` same options (+ `--region`, `--force`) |

### Config keys (BlendConfig)

All 33 section dataclasses' keys are read by `convert_blend` (`momp/lib/convert_specs.py:742 and `_Convert.convert``, `_BLEND_KEYS` rejects unknown ones). Carried
with a note instead of used, all RETIRED because ROMP finds/does it itself: `domain.regrid.crs`, `domain.regrid.clip_to_coverage`
(ROMP always clips to observed cells), `domain.regrid.grid_lat_var/grid_lon_var`; `features.join` (only `inner` ever existed; other value is a
conversion error). Everything else maps one to one: `observations.{files,variable,dims,missing_rain,cell_transform,date_marks}` ->
`observations`/`units`/`onset.missing`; `models[].{files,variable,dims,unit_scale,max_member,lead_days,rain_horizon,rain_predictors,
rain_transform,cell_transform,regrid_weights}`; `onset.{definition,threshold.*,season_start,fixed_cutoff,reference_onset}`;
`climatology.*`; `features.*`; `evaluation.*`; `outputs.{dir,intermediates,pickles}`; `name`, `data_root` (paths made absolute).

**MISSING items:** none of substance. Minor: CLI flags `--workers` (train, forecast) and `--cache-dir` exist only as YAML keys.

## 2. Duplication audit (grep across momp/)

| # | Pair | Where | Verdict |
|---|---|---|---|
| 1 | observed-onset detection | `blend/features.py:observed` calls `stats/detect.py:detect_observed_onset` | one path: OK |
| 2 | caches | `cache.py:Store` is the base of `stats/observed_cache.py`, `onset/climatology.py:ThresholdCache`, `lib/dataset_detect.py:357`, `io/input.py` | OK; **but** `lib/align_sample.py:455-485` (`fingerprint`, `_store`: own sha256 of `repr`, own pickle + tmp/replace) is a second store. DUPLICATE: move to `Store` |
| 3 | content/fingerprint schemes | `cache.py:part_hash` vs `stats/benchmark.py:154 _threshold_fingerprint` (sha1 over array bytes), `export.py:287` (sha256 over threshold values), `blend/trained.py:162 _recipe_digest` (sha256 of `repr`), `lib/align_sample.py:450` | DUPLICATE: threshold-content digest exists 3 times (benchmark, export, observed_cache via part_hash); `_recipe_digest` relies on `repr` of dataclasses (fragile) -> use `part_hash`. `definition_hash`, `run_folder` hash and mask `file_hash` serve other keys: OK |
| 4 | worker pools | `stats/parallel.py` (spawn process pool, `map_items`) is the run's pool; `blend/_crps_tree.py:344` runs its own joblib/loky pool (same as oracle); `utils/workers.py:ordered_prefetch` (thread pool, I/O) and `lib/align_sample.py:531` (another ThreadPoolExecutor) | `_crps_tree` joblib: DUPLICATE of the process pool (kept for forest parity/threads; could use `map_items`); align_sample thread pool vs `ordered_prefetch`: minor DUPLICATE |
| 5 | file-year regexes | `lib/dataset_schema.py:53 FILE_YEAR_RE/file_year`, `lib/dataset_detect.py:47 YEAR_RE`, `lib/align_sample.py:88` inline regex | DUPLICATE, three different regexes (the last two accept 18xx, need non-digit neighbours; the first is blending's) |
| 6 | dimension/coordinate name tables | `utils/standard.py:ROLE_ALIASES` is the one for dims; `utils/mask.py:53-54 _LAT_NAMES/_LON_NAMES`, `spatial/units.py:41-42` (table columns), `spatial/regrid.py:126` | minor DUPLICATE (mask.py list adds nav_lat etc.; units.py is for CSV columns) |
| 7 | polygon / land masks | `utils/mask.py` only (the former `utils/land_mask.py` is merged): the implicit `land_only` / `shp_only` / `nc_mask` masks are `default_mask(...)`, a `MaskSpec` preset, evaluated by `build_mask` (`rasterise`, `cell_shares`); `land_mask_for`, `mask_land`, `shp_mask`, `create_land_sea_mask`, `apply_nc_mask` are thin wrappers over it; graphics use `shp_mask`/`get_shp` | OK: one algorithm for the masks (kept cells identical to the old module, `tests/test_mask_unification.py`). Remaining: `points_inside_polygon` / `polygon_mask` (matplotlib `contains_points` on the monsoon-zone polygon, graphics only) differ from shapely on cells on the polygon edge, so they stay. `spatial/regrid.py:polygon_weights` reuses `mask.cell_shares`: OK |
| 8 | longitude conforming | `utils/region.py:11 swap_lon_axis` / `lon_swap`, `spatial/regrid.py:69 conform_longitudes` and `:463` | partial DUPLICATE (different objects: dataset vs weight table) |
| 9 | id normalisation | `spatial/ids.py:normalize` used everywhere | one: OK |
| 10 | KDE / probability | `onset/climatology_kde.py` only; `stats/conditional_climatology.py` and `blend/features.climatology` both call it | one: OK |
| 11 | histogram / bins | `stats/bins.py:LeadBins` used by `blend/recipe.py`, `stats/score.py`, `conditional_climatology.py` | one: OK |
| 12 | Brier / RPS / AUC | `metrics/probabilistic.py`; `stats/score.py` are adapters (`_brier_results` shared by climatology and model variants) | one: OK. `io/output.py:124 save_ref_score_results` / `:137 load_ref_score_results` have no callers (dead) |
| 13 | climatological onset | `stats/climatology.py:55 compute_climatological_onset` (used by benchmark, export, app) vs `onset/engine.py:1158 climatological_onset` (no caller in momp/) | DUPLICATE (engine copy unused; docstring `onset/climatology.py:424` points at it) |
| 14 | writers of observed onset dates | `export.py:392` (final per-source NetCDF), `stats/observed_cache.py:61` (cache NetCDF), `blend/training.py:164` (`observed_onsets.nc` via `samples.py:228`), `onset/engine.py:1195 save` (zarr, no caller) | SUSPECT: 4 writers, 4 layouts; `engine.save` dead |
| 15 | writers of one product, twin formats | forecasts: CSV and NetCDF (`blend/forecast.py:115-119`); metric tables: CSV and optional pickle (`blend/evaluation.py:198+`) | see Disk writes |
| 16 | config representations | flat cfg `blend_*` fields (`lib/config_schema.py:623+`) and `blend/recipe.py` dataclasses with their own allowed-key tuples (`recipe.py:940`) | layered by design (validate, then type); key lists exist twice (drift risk): minor |
| 17 | readers | one input path (`io/input.py`); `blend/forecast.py:_read` -> `dataset_detect.read_forecast_file`, a header reader for signatures only | OK |
| 18 | spatial metrics | `io/output.py:save_metrics_to_netcdf` and `app/*` vs `export.py` | different products: OK |

## 3. Disk writes (intermediates must not touch disk)

Rule: FINAL OUTPUT (a user/consumer reads it), CACHE (reused across runs), INTERMEDIATE (should be an in-memory object).

| Write (file:line) | What | Read back in the same run? | Class |
|---|---|---|---|
| `blend/evaluation.py:177,181,186,190,198` | `formula_sample_support*.json`, `cv_preds_*.nc`, `coefs_*.nc`, weights CSVs, summary/yearly CSVs (contract artifacts) | no (except MME weights, below) | FINAL |
| `blend/evaluation.py:190` -> read at `:747,781,790 _read_mme_weights` | MME/Platt weights CSV written by one `save`, read from `results_dir` by a later run when `mme.optimize` is false | not in the same run; a later run reads a file a previous run wrote: reads an input, but the weights could be taken from the trained blend (`weights.npz`) | FINAL output used as INPUT; replace by weights stored in the blend / a `weights:` config value |
| `blend/evaluation.py:198+ (pickles)` | pickled twin of CSV tables, `outputs.pickles` | no | INTERMEDIATE-like debug (opt-in, default off): OK |
| `blend/training.py:146,164-166` | `weekly_features.nc` (default `intermediates=("weekly",)`; `api.py:246` passes none, so it is written for every run) and optional `observed_onsets.nc` | no | INTERMEDIATE written by default (80 MB for Ethiopia, 280+ MB for India); should default to none (onset_blending's CLI default was none) |
| `blend/trained.py:451` | `trained.blend` zip (recipe, model, observed, units, reference, weights, signatures) | by forecast, a different run | FINAL |
| `blend/forecast.py:115-119` (`WEEKLY_NETCDF`, `DAILY_NETCDF` at `:32-33`) | forecast CSV plus a NetCDF twin of the same table | no | FINAL (CSV is the contract); NetCDF twin: no consumer needs it, drop or make opt-in |
| `lib/resolved.py:88,90` | `run.yaml`, `definition.yaml` in the run folder | no | FINAL (provenance) |
| `cache.py:253-255` (`Store`) | thresholds, harmonic seasons, dataset detection, issue climatology | next runs | CACHE (`<runs>/.cache/<kind>/<fingerprint>`) |
| `stats/observed_cache.py:61` | observed onsets per year | next runs (and within the run: in-memory dict first) | CACHE (`.onset_cache` / `.cache/observed_onsets`) |
| `lib/align_sample.py:480` | alignment report pickle | next `--check`/runs | CACHE (own store, see duplicate #2) |
| `io/output.py:75,79` (`save_score_results`) -> read at `graphics/panel_bar_skill.py:132`, `graphics/panel_portrait_skill.py:76` | overall and binned skill CSVs | YES: the bar and portrait panels re-read the CSVs written earlier in the same run (`lib/assertion.py:170`: plots require `save_csv_score`) | CSV is FINAL; the read-back is an INTERMEDIATE round trip: panels should take the score tables from memory |
| `io/output.py:240` (`save_metrics_to_netcdf`, `spatial_metrics_*.nc`) -> read at `graphics/panel_map_error.py:51,199`, `graphics/panel_portrait_error.py:91,111` | spatial metrics | YES (same round trip, plots require `save_nc_spatial_far_mr_mae`) | FINAL file; read-back is INTERMEDIATE |
| `io/output.py:260` (`climatology_onset_doy_*.nc`) -> read at `graphics/func_map.py:261` | climatological onset day of year | YES | FINAL file; read-back INTERMEDIATE |
| `io/output.py:133` (`save_ref_score_results`, pickle) | reference scores | `load_ref_score_results` (`:137`) has no caller | dead code: delete |
| `app/bin_skill_score.py:143-148` | combi pickles under `if 2 > 3:` | never | dead code: delete |
| `export.py:392` | `onsets_<source>.nc` (`momp-run export`) | `export.py:403` re-reads only to validate/resume an existing file | FINAL |
| `onset/engine.py:1208 save` | zarr result store | no caller | dead or API-only: confirm, else delete |
| graphics `savefig` (graphics/*.py, `utils/mask.py:637` mask PNG, `lib/explain_cells.py:74`) | figures | no | FINAL (figures); `utils/mask.py:637` PNG of the mask is written per run (debug-like but documented as a record) |
| `cli.py:195`, `lib/init_config.py:164`, `lib/convert.py:199`, `config.py:162` | `convert`/`init` output YAML | no | FINAL |

Summary of what should change: (1) `weekly_features.nc` off by default; (2) forecast NetCDF twins opt-in or removed;
(3) the three graphics read-backs (CSV, spatial NetCDF, climatology NetCDF) should consume the in-memory results;
(4) MME weights should come from the trained blend, not a file in `results_dir`; (5) delete dead writers
(`save_ref_score_results`, the `2 > 3` pickles, `engine.save` if unused).
