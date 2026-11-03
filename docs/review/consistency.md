# Review: duplication, consistency, dead code and caching

Read-only review of `hcwf/` on branch `monorepo` (HEAD `6194117`). Rules from `docs/blend_merge.md`: nothing
implemented twice, results within tolerance, no intermediates on disk. Package layout is in `docs/review/layout.md`
and profiling in `docs/review/perf_*.md`. This file does not repeat them.

Size: S < half a day, M 1-2 days, L more. "Verify" always includes: `.venv/bin/python -m pytest -n auto -q`; the
regression harness (`MOD=hcwf run.sh <repo> <out>` then `cmp.sh .../out_base <out>`, scripts in
`.../scratchpad/regress/`); and for anything under `blend/` the blend differential/parity tests (`tests/blend/test_blend_*differential*.py`, `test_input_parity.py`, `test_kde_parity.py`, `test_metrics_parity.py`).

## 0. Findings that are bugs or correctness risks (do these first)

| # | Where | Problem | Fix | Size |
|---|---|---|---|---|
| B1 | `io/input.py:360-371` `_model_thresholds` | The `model_thresholds` cache parts are model name, `file_identity(model_dir)`, years, windows, definition, pending, lead_label, init_days, members. **Missing:** `region`/`mask`/grid, `model_var`, `unit_cvt`, `accumulation`, `daily_boundary`, `model_dims`, `file_pattern`. Two configs on the same `model_dir` with a different region or unit factor get each other's thresholds: wrong grid, or values off by the unit factor. | Add those keys, plus the grid of the first cube read (as `ThresholdCache.parts` does, `onset/climatology.py:395`). Bump the parts, which just makes one cold run. | S |
| B2 | `export.py:286-298` `_threshold_hash` | `np.asarray(value, dtype=float)` raises `ValueError` on a threshold **table** (`onset/thresholds.py:79` returns a DataFrame with a text id column). Reproduced: `_threshold_hash({'onset': DataFrame({'id': ['a','b'], ...})})` gives `could not convert string to float: 'a'`. So `save_onsets` on unit runs with table thresholds fails. `api.py:325` only logs the failure. | Delete it and use the threshold hash of `stats/observed_cache.py:88` (it handles tables, see D3). | S |
| B3 | observed onsets, thresholds, harmonic seasons, model thresholds (`stats/observed_cache.py:177`, `onset/climatology.py:395`, `io/input.py:321,360`) | No code-version part. A fix to `onset/engine.py`, the backends or `compute_threshold` keeps serving the old cached onsets and thresholds from every `cache_dir` that runs share. Only detection (`lib/dataset_detect.py:346`) and alignment (`lib/align_sample.py:57`) have a `_CACHE_VERSION`, and it is bumped by hand. | One `hcwf.cache.SCHEMA` dict `{kind: version}` that `Store` adds as a `version` part, bumped in the same commit as any change to the semantics. Add a test that fails when a backend or engine file changes without a bump (a hash of their source checked into the test). | S |
| B4 | `stats/climatology.py:99-102` | `except Exception: log.warning(...compute_climagoloical_onset...); raise; continue`. The `continue` can never run, so "skip a bad year" is dead code; the message has a typo, and the error is logged and then raised again (it is reported twice). | Remove the `log.warning` and the `continue`. Re-raise with the year added to the message. | S |
| B5 | `utils/mask.py:645` `_RECORDS`, `io/input.py:67` `_stores`, `lib/onset_config.py:211` `_cache`, `stats/bins.py:294`, `utils/region.py:153`, `lib/dataset_detect.py:136` | These live for the whole process. `api.execute` clears some of them (`api.py:288-293`) and `_execute_blend` clears fewer (`api.py:251-254`). Nothing clears `_stores`, the open zarr handles: a store rewritten between two `hcwf.run` calls in one notebook is read through stale metadata. Nothing clears `onset_config._cache` either, which is keyed by `repr` of config values, not file contents. | One memo registry (C4). | M |
| B6 | `stats/benchmark.py:151-157` `_threshold_fingerprint` | Hashes any non-DataArray, non-number object by `id()`. That is safe today only because `_loaded_thresholds` keeps the object alive for the whole run. | Use the shared threshold hash (D3). | S |
| B7 | `lib/check.py:474` `_step_hours` vs `onset/ingest.py:451` `_lead_hours` | The two decide differently whether a plain numeric step counts days or hours. `check` treats numbers as days unless the units attribute says hours. `ingest` treats 1..n or values below 24 as days, and anything else as hours. So `hcwf-run --check` can accept a lead axis that the run then reads differently. | `check` calls `ingest._lead_hours` (make it public as `lead_hours`). | S |

## 1. Duplication

Status of the suspects in `docs/blend_merge/coverage.md` section 2, checked again at HEAD:

- **Resolved:** #12 (`save_ref_score_results`/`load_ref_score_results` are gone), #13 (`engine.climatological_onset`
  is gone; `stats/climatology.py:111` and `onset/climatology.py:463` share `reduce_onset_days`) and #14 in part
  (`engine.save` is gone).
- **Still OK:** #1, #9, #10, #11, #17, #18.
- **Still open, with more found:** #2-#8, #16. These and the new ones are listed below.

### D1. `align_sample` keeps its own cache, store and readers (coverage #2, #4)
- Second store: `lib/align_sample.py:418-485`. It has its own `cache_dir`, `_signature` (a third file-identity
  scheme, which reads up to 200 kB of zarr metadata at `:434`), `fingerprint` (sha256 of a `repr` of a tuple),
  `_load`/`_store` (pickle, tmp file plus `os.replace`) and `_CACHE_VERSION`. All of this repeats `cache.Store`,
  `cache.file_identity` and `cache.atomic_write` (`cache.py:38,115,193`).
- It also has its own readers. `_forecast_sources` (`align_sample.py:143`) repeats `io/input.py:803 _forecast_sources`,
  but **does not support `paths` lists**, and `_signature` only counts `*.nc` files in a folder. A dataset given by
  `paths` gets a fingerprint that never changes when its files change, and a sample read that differs from the run's.
  `_years_in` (`:83`) carries a third copy of the year regex (D2).
- What survives: `Store("alignment", root, label="Alignment")`, with parts `{version, lead_label, boundaries,
  keys: {k: kw.get(k)}, obs_files, model_files, sample}`. The two files parts come from the same function as
  `observed_cache.obs_files`, extended to the model. Readers go through `io.input` (`year_files`, `open_store`,
  `is_zarr`).
- Migration: replace `_load`/`_store`/`fingerprint` with `Store.get`/`put`, keeping `use_cache` as it is. Existing
  `~/.cache/hcwf/alignment/*.pkl` are simply never read again (C2 cleans them up).
- Verify: `tests/test_align_sample.py`, plus `hcwf-run --check` on the slice configs twice. The second run must say
  "cached". Size M.

### D2. Year-of-file parsing: five ways (coverage #5)
- `lib/dataset_schema.py:53` `FILE_YEAR_RE = (19|20)\d{2}`: matches inside longer numbers (`v20240101` gives 2024,
  and `x12020` gives 2020).
- `lib/dataset_detect.py:47` `YEAR_RE`: `(?<!\d)((?:18|19|20)\d{2})(?!\d)`. It is used side by side with `file_year`
  in **the same function**, `_lead_label` (`:267-286`).
- `lib/align_sample.py:88`: the same pattern as `YEAR_RE`, written inline.
- `lib/config_yaml.py:58`: a glob with `[0-9][0-9][0-9][0-9]`.
- `io/input.py:534`: `file_year`. `blend/features.py:77` and `lib/check.py:226` also use `file_year`.
- What survives: one `file_year(path)` in `dataset_schema` using the stricter `YEAR_RE`, and `dataset_detect` and
  `align_sample` import it. This changes behaviour only for names where a year sits inside a longer digit run; the
  blending datasets have none.
- Verify: blend parity tests, plus `grep -rn "re.compile.*\\\\d{2}" hcwf` showing a single regex. Size S.

### D3. Content hashes of thresholds: three copies (coverage #3)
- `stats/observed_cache.py:88` (uses `part_hash`, handles DataFrames, memoised by id).
- `export.py:286` (sha256, crashes on tables: B2).
- `stats/benchmark.py:151` (sha1, `id()` fallback: B6).
- What survives: `observed_cache._threshold_hash`, moved to `hcwf/cache.py` as `threshold_hash(thresh)` (public).
  `export` keeps its attribute name `threshold_values_hash`, so the value written to the NetCDF changes once. Check
  `tests/test_export_onsets.py` for pinned digests.
- `blend/trained.py:163` `_recipe_digest` = `sha256(repr(recipe))`. A digest of `repr` changes whenever a dataclass
  gains a field with a default, so old `.blend` files are refused for no reason. Use
  `part_hash(dataclasses.asdict(recipe))`, keeping `repr` as a fallback when reading old files. Size S.

### D4. Zarr detection and opening: three copies
- `io/input.py:58 is_zarr` and `lib/dataset_detect.py:140 is_zarr_store` are the same code line for line.
  `align_sample.py:94,101` wraps `io.input`.
- `dataset_detect.py:150 open_data` and `io/input.py:71 open_store` both open zarr with `chunks=None`; only the second
  is shared per path.
- `dataset_detect.py:368,458` list `".nc" in name or endswith(".zarr")` twice.
- What survives: `io.input.is_zarr` / `open_store`, or a small `hcwf/io/files.py` if layout.md moves it.
  `dataset_detect.is_zarr_store` becomes an import. Size S.

### D5. Two masking systems in the run path (coverage #7; the earlier verdict "graphics only" is wrong)
- `utils/land_mask.py` (ROMP) is used by the **pipeline**, not only by graphics: `utils/region.py:5` imports
  `polygon_mask`, `mask_land`, `shp_mask`, `apply_nc_mask` and `land_mask_for` for the flat keys `land_only`,
  `shp_only`, `polygon` and `nc_mask`. `utils/mask.py` serves the `mask:` mapping.
- Each piece has a pair:

  | `land_mask.py` | `mask.py` |
  |---|---|
  | `get_shp` `:277` (re-reads the 10m admin shapefile on **every call**, matches `NAME` only) | `country_geometry` `:282` (cached `_NE_CACHE`, matches `NAME`/`NAME_LONG`/`ADM0_A3`) |
  | `land_mask_for`/`_raw_land_mask` `:168-208` (regionmask `land_110`) | `land_geometry` `:311` and `rasterise` |
  | `apply_nc_mask` `:371` (opens the file per call, `sel(..., method="nearest")`) | `read_raster_mask` `:345` and `align_raster_mask` `:419` |
  | `points_inside_polygon` `:16` / `polygon_mask` `:57` | `rasterise(rule="centre")` |
  | `_grid_key` `:156` (sha256 of float64 copies) | the key at `mask.py:669` (sha256 of `lat.tobytes() + lon.tobytes()`) |

- What survives: `mask.py`. The flat keys become a `MaskSpec`: `land_only` becomes `{land: 110m}`, `shp_only` becomes
  `{country: region}`, `nc_mask` becomes `{file: ..., keep_value}`, and `polygon` becomes the CMZ polygon from
  `params/region_def.py:20`. `land_mask.py` keeps only the plotting helpers (`polygon_outline`, `add_polygon`,
  `shp_outline`, `get_india_outline`), which can move to `graphics/`. `region._KEEP_CACHE` (`utils/region.py:153`)
  then goes away, being `mask._BUILT` under another name.
- Risk: **cell edges**. regionmask `land_110` and `rasterise(centre)` on Natural Earth 110m should pick the same
  centres, but shapely `contains` vs `covers` on boundary points can differ by a few coastal cells.
- Verify: before switching, script a comparison of the boolean masks of both paths on the 4 slice grids and on IMD
  0.25°. A switch is acceptable only with 0 differing cells, or with the differences listed and accepted. Then run
  run.sh/cmp.sh. Size L. This is the largest real duplicate left.

### D6. Smaller duplicates
| What | Copies | Survivor | Size |
|---|---|---|---|
| Rain unit tables | `onset/ingest.py:21 MM_PER_UNIT` (the one table, by its own docstring); `lib/dataset_schema.py:55 UNIT_FACTORS = {"mm": None, "m": 1000}`; `dataset_detect.py:217` reads `MM_PER_UNIT` | Build `UNIT_FACTORS`/`unit_factor` from `MM_PER_UNIT` | S |
| Lat/lon name lists (coverage #6) | `utils/standard.py:14 ROLE_ALIASES`; `utils/mask.py:53-54` (adds `nav_lat`, `lats`, `long`); `spatial/units.py:41-42` (CSV columns `center_lat`); `spatial/regrid.py:126` (inline) | Mask files: extend `ROLE_ALIASES` and drop `mask._LAT_NAMES`. Table columns: one `TABLE_LAT/LON` in `spatial/units.py`, which `regrid` imports | S |
| `doy_to_mmm_dd` | `graphics/func_map.py:223` and `graphics/onset_map.py:167` (same text; the comment says 2018, the code uses 2024); `onset_map.py:161 doy_to_date_string` is unused | One copy in `graphics/` | S |
| `contrast_text_color` | `graphics/func_map.py:79` and `utils/visual.py:593`, nested functions with the same body | One module-level function | S |
| Thread pool | `align_sample.py:531 check_models` (its own `ThreadPoolExecutor`) vs `utils/workers.py:87 ordered_prefetch` | Keep both, because the needs differ (unordered model checks vs ordered prefetch). Have `check_models` take `worker_count(cfg, len(jobs))` instead of the fixed `workers=8` | S |
| Process pools (coverage #4) | `stats/parallel.py` (spawn pool) and `blend/_crps_tree.py:345` (joblib loky, `n_jobs=-1`, capped by `random_forest.workers` in `blend/evaluation.py:489`) | Keep loky: forest parity and memory-mapped X are worth it. Document that loky's workers stay alive after `parallel.shutdown()` (`joblib.externals.loky.get_reusable_executor().shutdown()` at run end) | S |
| Domain average | `onset/engine.py:1102 regional` (cos-lat mean of a DailyRain; called only from tests) vs `spatial/units.py:277` (cos_lat unit weights) | Delete `engine.regional` and its tests (`tests/test_onset_engine.py:498-533`) | S |
| Recipe key lists (coverage #16) | Nested allowed keys exist only as `_strict(...)` tuples (`blend/recipe.py:719-964`, 17 places). `lib/config_schema.py:623-684` types them `"any"` and lists them again in prose | Keep the tuples as the truth, as module constants (e.g. `FOREST_KEYS`), and have the schema descriptions and `describe_schema` read them. A test asserts every key named in a `blend_*` description appears in the matching tuple | S |
| Threshold keys | `io/input.py:421 _thresholds_key` + `load_thresholds` key, `io/input.py:96 _reference_key`, `stats/climatology.py:18 _MODEL_ONLY`, `stats/benchmark.py:142 _PER_CASE`, `lib/onset_config.py:213 _SETUP_KEYS`. These are five hand-written lists of "keys that matter" | Long term: derive them from the schema (a `Field(..., affects=("observed", "thresholds"))` tag). Short term: one module `hcwf/lib/keysets.py` that holds the five lists side by side, so a new key is added in one place | M |

## 2. Consistency

- **Error types.** Config problems raise three families. `HCWFConfigError(Exception)` (`lib/assertion.py:25`) is the
  base of `ConfigSchemaError`, `DatasetError`, `MaskError`, `ConvertError` and `OnsetConfigError`. Plain `ValueError`
  subclasses are `RecipeError` `blend/recipe.py:48`, `DefinitionError` `onset/definition.py:35`, `RegionError`
  `onset/regions.py:30`, `LegacyConfigError` `onset/legacy.py:28`, `UnitsSpecError`/`AreaError` `spatial/units.py:45-49`,
  `UnitsError` `onset/units.py:18` and `BlendFileError` `blend/trained.py:64`. Then there are bare `ValueError`s.
  `driver.py:80,106,112` catches only `HCWFConfigError`, so a `RecipeError` or `RegionError` reaches the generic
  `except Exception` at `driver.py:115`, which prints a traceback-style message. `cli.py:90,227,238,261` catches
  `ValueError`/`Exception` instead.
  **Proposal:** `class HCWFConfigError(ValueError)` and re-base every config error class above on it, so old
  `except ValueError` still works. Then `driver`/`cli` catch the one base. Size S. Verify with `tests/test_config_schema.py`, `tests/blend/test_blend_config.py`, `tests/test_onset_regions.py` and `tests/test_run_experience.py`
  (the CLI exit codes there).
- **Logging vs print.** Clean. Outside the CLIs, `print` remains only in `lib/assertion.py:191-193` (a ROMP
  validator: it prints the exception instead of raising it), `utils/printing.py:64-105` (`print_cfg`,
  `print_data_info`, both dead) and `lib/init_config.py:159` (CLI output, fine). Delete the dead ones and make
  `assertion` raise.
- **Private names across modules.** `stats/parallel.py:62-78` reads and writes `io_input._loaded_thresholds`,
  `mask._BUILT`, `mask._RECORDS`, `land_mask._LAND_BOOL` and `region._KEEP_CACHE`. `api.py:288-293` clears
  `detect._alignment_done`, `io_input._loaded_thresholds` and `io_input._reported_ingest`. `onset/explain.py` calls 7
  `engine._*` helpers (`:221,323,553,566,570,631-675`). `onset/viz/*` import `explain._cell_of`, `_season_index`,
  `_OP_WORDS`. `spatial/regrid.py:32` imports `mask._nearest`/`_overlap`. Others: `onset/climatology.py:175`
  (`engine._bound_key`, `_label_table`), `lib/onset_config.py:200` (`engine._load_backend`), `export.py:465`
  (`detect._detect_forecast`), `api.py:122` and `export.py:94` (`loader._rename_deprecated_keys`), `lib/dataset_schema.py:346`
  (`ingest._parse_clock`) and `lib/init_config.py` (`convert._flow`).
  **Proposal:** drop the underscore from every name used across modules. The memo dicts go behind the registry
  (C4), which also makes `parallel.snapshot/install` generic. `__all__` exists only in 9 `blend/` modules, so add it
  to the public modules (`api`, `config`, `cache`, `export`, `onset.*`) or nowhere. Size M, mechanical.
- **Vocabulary.** `blend/` says *issue* (160 uses vs 2 *init_time*). `stats/`, `export.py` and `io/` say
  *init_time* (65/34/30). `onset/` uses both (74/22). Lead appears as *lead*, *step* (71 in `onset/`, 31 in `io/`)
  and *day*. Dims are standardised (`init_time`, `step`, `lat`, `lon`). Keep `init_time`/`step` as dim names, use
  *issue date* only in blend's user-facing text, and do not rename code now: the cost is high and the gain cosmetic.
  Record the glossary in `docs/onset_in_hcwf.md`.
- **Docstrings and type hints.** Numpy-style `Parameters` blocks survive only in the ROMP modules
  `metrics/error.py`, `io/output.py`, `io/dict.py`, `utils/region.py`, `utils/land_mask.py` and `utils/visual.py`;
  the rest uses one-line prose. 26 modules have no return annotations at all: all of `app/`, `graphics/`,
  `metrics/` and `utils/visual|printing|standard`, plus `stats/score.py`, `lib/loader.py` and `lib/assertion.py`.
  Only worth fixing as those modules get touched. `graphics/panel_portrait_error.py:30,39,50` has invalid `"\D"`
  escapes that give a `SyntaxWarning` now and become a `SyntaxError` in a future Python: use raw strings (S, do now).
- **Return types.** Benchmark functions return DataFrames, or tuples of `(df, summary)`. Onset functions return
  DataArrays. Blend returns dataclasses. Dict-shaped containers remain in `io/dict.py` and `app/bin_skill_score.py:45`.
  This is coherent per layer; no action.
- **Paths.** `lib/loader.py:83` makes `work_dir`/`pkg_dir` `Path` and the rest `str` (`set_dir`).
  `lib/run_folder.py:217` stores `cache_dir` as `str`. `Store` accepts both. No bug; convert at the boundary only.
- **Constants.** `onset/climatology.py:19 _DAYS_IN_MONTH`/`_YEAR_LENGTH`, plus 366 literals in `io/input.py:598,615`
  and `stats/climatology.py:120`: move them to `onset/calendar.py`. The CMZ polygon is hard-coded at
  `params/region_def.py:28-43` while every other region lives in the registry YAML: move it to
  `onset/regions/india.yaml` as a named polygon.

## 3. Dead code

From `uvx vulture hcwf --min-confidence 60` (105 hits), with each name grepped across `hcwf/`, `tests/`, `scripts/`,
`examples/` and `docs/`.

**No caller anywhere (delete):**
`blend/_crps_tree.py:438 predict_quantiles` · `blend/trained.py:590 check_forecast_files` ·
`graphics/onset_map.py:161 doy_to_date_string` · `io/output.py:248 analyze_nested_dict` ·
`lib/assertion.py:163 _check_graphics_dependencies` · `lib/config_schema.py:57 LIST_SECTIONS` ·
`lib/config_yaml.py:41 load_yaml_config` · `lib/control.py:148 ref_cfg_layout`, `:169 ref_model_case` ·
`lib/dataset_detect.py:903 forecast_issue_dates` · `lib/parser.py:205 parse_tuple`, `:218 parse_num_to_tuple` ·
`onset/climatology_kde.py:470 legacy_window_table` · `onset/engine.py:256 _is_leap` ·
`spatial/units.py:438 clear_transforms`, `:535 with_weights` · `utils/mask.py:43 CELL_RULES` ·
`utils/practical.py:27,48,64 set_dir3/set_dir2/set_dir_safe` · `utils/printing.py:64 print_cfg`, `:82 print_data_info` ·
`export.py:62 SOURCE_FILE`.
Unused locals and parameters: `stats/bins.py:14 _first_close` (unused import) ·
`stats/benchmark.py:289 save_nc_climatology` (a parameter that is never read) · `blend/samples.py:116 has_spread`
(set but never read) · `lib/dataset_detect.py:80 Resolution.sample_file` (written, never read, and pickled into
every detection entry) · `graphics/*` (`bars1-3`, `max_val`, `cbar_*`, `overall_mae_*`) ·
`graphics/rainfall_time_series.py:9 wetspell_date` · `utils/visual.py:169-170` · `onset/ingest.py:428 step_hours` ·
`onset/model.py:211-212` · `lib/onset_config.py:309-310`.

**Called only by tests (decide; delete together with the tests, or mark as public API):**
`onset/engine.py:1102 regional` (D6) · `onset/climatology_kde.py:271 unconditional_probabilities`, `:463 legacy_tables` ·
`onset/climatology.py:460 ReferenceOnset.from_onsets` · `onset/regions.py:299 units_for` ·
`spatial/ids.py:248 with_ids` · `spatial/regrid.py:272 sources_of`, `:335 point_sources`, `:340 grid_axes`,
`:416 coverage_geometry` · `spatial/units.py:190 to_frame`, `:285 with_subset` ·
`lib/dataset_detect.py:198 clear_samples`, `:770 check_entry` · `utils/land_mask.py:178 clear_land_masks` ·
`utils/workers.py:43 default_resources`, `:76 limit_blas_threads` · `blend/logit.py:172,192 to/from_legacy_bundle`.
**Public API (documented, keep):** `export.open_onsets`, `onset.explain.trace_forecast`, `onset.viz.*`
(`compare_definitions`, `forecast_vs_truth`, `drill_down`), `lib.config_schema.describe_schema`/`describe_mask_schema`,
and `hcwf.config.Mask`/`flat_config` (the typed config API in `config.py:3`). These are referenced in
`docs/onset_spec.md`/`docs/onset_in_hcwf.md`.

**Schema keys:** all 125 `Field`s are read somewhere. The five that look unread (`blend_cluster_var`,
`blend_final_formula`, `blend_probability_days`, `blend_rain_transform`, `blend_trim_after_true_onset`) are read
through `f"blend_{key}"` at `blend/recipe.py:812,852`. That pattern defeats grep: list them explicitly.

Verify the deletions with the test suite plus `uvx vulture hcwf tests --min-confidence 60`, which must lose these
entries. Size S (one commit per area).

## 4. Caching

### 4.1 Inventory

Disk (all go through `cache.Store` except alignment):

| Store kind | Defined | Root | Parts | Payload | Lifetime / invalidation |
|---|---|---|---|---|---|
| `detect` | `lib/dataset_detect.py:356` | `$HCWF_CACHE_DIR` or `~/.cache/hcwf` (global) | version, kind, entry `repr`s, sources, file identities | pickle of `Resolution` | forever; `_CACHE_VERSION` bumped by hand; `memory=False` |
| `alignment` | `lib/align_sample.py:418-485` (**not a Store**) | `~/.cache/hcwf/alignment` | sha256 of a `repr` of 16 keys and `_signature` | pickle | forever; `_CACHE_VERSION`; misses `paths` datasets (D1) |
| `observed_onsets` | `stats/observed_cache.py:66` | `cache_dir` or `<run>/.onset_cache` | obs_files, obs_ingest, definition, thresholds, years, cells or units, options | NetCDF (year × cells) | per folder; **no code version** (B3) |
| `thresholds` | `onset/climatology.py:392` | same | name, source, years, spec, grid, files, options | NetCDF | same (B3) |
| `harmonic_seasons` | `io/input.py:325` | same | source, years, files, grid | NetCDF | same (B3) |
| `model_thresholds` | `io/input.py:370` | same | see **B1** (incomplete) | NetCDF groups | same |

In memory: `Store._memory` (`cache.py:210`) is **unbounded** per Store instance. That is harmless, because the
instances live for one run or one call (`io/input.py:325,370` build a new Store per call, so the in-memory layer
there does nothing).

| Memo | Key | Cleared by | Bound |
|---|---|---|---|
| `io/input.py:67 _stores` (open zarr) | path | **never** | none |
| `io/input.py:80 _obs_years` | file stat + settings | `clear_obs_cache` (run start/end) | none (one run's years) |
| `io/input.py:91 _reference` | `_reference_key` | `release_reference` | 1 |
| `io/input.py:421 _loaded_thresholds` | setup key and 8 strings | `api.py:292` (benchmark only) | none |
| `io/input.py:180 _reported_ingest` | — | `api.py:293` | — |
| `stats/climatology.py:17 _MEMO` | — | `reset_memo` | none |
| `stats/benchmark.py:140 _ONSET_TABLES` | `_family_key` (`repr` of all kwargs) | `reset_onset_tables` | none |
| `stats/observed_cache.py:32-34 _tables/_memo` | fingerprint / `id` | `reset` | none |
| `stats/bins.py:294 _YEAR_TEXT` | year tuple | **never** | none (small) |
| `utils/mask.py:253 _NE_CACHE` | resolution | never | tens of MB for 10m records |
| `utils/mask.py:645 _BUILT/_RECORDS` | spec, dataset, grid hash | `mask.reset` (benchmark only, not blend) | none |
| `utils/land_mask.py:163,184` | grid hash | `clear_land_masks` (tests) | 16 grids (the only bounded memo) |
| `utils/region.py:153 _KEEP_CACHE` | lat/lon **bytes**, region | never | none |
| `lib/dataset_detect.py:136 _SAMPLES` | file, mtime, var | `clear_samples` (tests) | none, a few MB each |
| `lib/onset_config.py:211 _cache` | `repr` of 15 keys | never | none |
| `onset/regions.py:202 _index` (lru_cache) | — | never | 1 (registry files; fine) |
| `stats/parallel.py:128 _pool` | size, threads, handler ids | `shutdown` | 1 |

### 4.2 Measurements (this review, 20-core aarch64, other agents running)

- Cold vs warm on the four phase-7 slices (`run.sh` then `warm.sh`, `MOD=hcwf`, fresh `HCWF_CACHE_DIR`):

  | slice | cold | warm | Δ |
  |---|---|---|---|
  | c eth_aifsens | 20.4 s | 20.0 s | -2% |
  | d eth_aifs | 2.2 s | 2.8 s | noise |
  | b india_graphcast | 9.7 s | 9.4 s | -3% |
  | a india_aifsens | 53.4 s | 52.7 s | -1% |

  Warm hits: Thresholds, Observed onsets, detection. Warm outputs match cold, and `cmp.sh` against `out_base` shows no
  difference in any of the 34 files. **On these slices the caches neither help nor cost anything measurable.** The
  time goes into reading forecasts and detecting forecast onsets (see perf_*.md). The caches pay off only when
  threshold years are long and observations large (IMD 0.25° over 30+ years), or across many runs sharing `cache_dir`.
- Cache cost in a warm slice_b run (cProfile, 9.35 s total): every `Store.get`, `fingerprint_parts`, `part_hash`
  (390 calls, 13 ms in all) and `file_identity` (241 calls, 3 ms) add up to about 0.1 s. Most of `load_thresholds`'
  0.57 s on a hit is the one-year probe read at `io/input.py:409`, made only to learn the grid.
- `part_hash` on realistic values: 1-D lat/lon 0.03 ms. Threshold grid 121×161 f32 0.04 ms. Global 721×1440 f64
  19.5 ms. 5 000 unit ids 0.5 ms, 50 000 ids 5 ms. An 850 MB rain cube takes 850 ms, against 343 ms for a raw sha256
  over the buffer: `_feed` copies the array through `tobytes()` and wraps ndarrays in a DataArray (`cache.py:82,88`).
  No cache part hashes a data cube today. **xxhash or blake3 is not worth adding.** One cheap fix: feed
  `memoryview(np.ascontiguousarray(values))` instead of `.tobytes()`, which removes the copy (S).
- Disk sizes: `.onset_cache` holds 0.7-2.7 MB per slice, and `~/.cache/hcwf/detect` 2.4 MB. The **old
  `~/.cache/momp/detect` holds 86 MB in 8 369 entries and `~/.cache/momp/alignment` 668 K**, orphaned by the rename
  and never cleaned: concrete evidence that eviction is needed.

### 4.3 Proposals

- **C1. Fix B1-B3 first** (above). Size S.
- **C2. `hcwf-run cache {info,clean}`** (`cli.py`), with no new dependency. `info` lists each root (the global one
  and a run's `cache_dir`), with kinds, entry count, bytes and oldest/newest. `clean [--kind K] [--older-than 30d]
  [--max-size 1G] [--all] [--legacy]` deletes whole entry folders, oldest mtime first. `--legacy` removes
  `~/.cache/momp`, the old `alignment/*.pkl` and any `*.tmp*` left by killed writers. Implement it as
  `Store.entries()` / `Store.prune(max_bytes, older_than)` in `cache.py`, so tests cover it once. Also call
  `prune(max_bytes=HCWF_CACHE_MAX, default 2 GB)` on the global root at most once a day: touch a stamp file, then
  prune. **Do not adopt diskcache**: it would be a second store format next to `Store`, which already gives atomic
  writes and parts for diagnosing misses; the folder-per-entry layout keeps `prune` trivial. Size M. Verify with a
  unit test on a temp root, and by running `cache clean --legacy` on this machine.
- **C3. Zarr for cached arrays: no.** The payloads are 0.1-3 MB of NetCDF, written once and read whole. Zarr would
  add a directory tree per entry, and `file_identity` would have to walk it. Revisit only if `observed_onsets` reaches
  hundreds of MB (unit runs over 50 000 units × 40 years × float64 ≈ 16 MB, still fine).
- **C4. One bounded, registered memo** (fixes B5 and the private-name access in §2). Add a `cache.Memo(name, *,
  maxsize=None, run_scoped=True, share=False)` that is a small `OrderedDict` LRU. Every module-level dict in the 4.1
  table becomes `_x = Memo("…")`. `cache.reset_run()` clears every `run_scoped` memo; `api.execute` and
  `_execute_blend` call it at the start and end, replacing the 6-9 scattered clears at `api.py:251-254,288-293,340-342`.
  `parallel.snapshot()`/`install()` iterate over the memos with `share=True` instead of naming five private dicts.
  `_stores` closes its datasets on reset. Bounds: `_NE_CACHE` 2, `_SAMPLES` 32, `_KEEP_CACHE`/`_BUILT` 16 grids (as
  `land_mask` already does), the rest unbounded but run-scoped. Size M. Verify: tests, regress, and a two-run notebook
  test (`hcwf.run(a); hcwf.run(b)` with b's zarr rewritten in between must read the new data).
- **C5. Observed-onset store writes.** `observed_cache.store` (`:254-274`) and `absorb` (`:302-312`) call `Store.put`
  **once per year**, rewriting the whole growing NetCDF each time. That is O(years²) bytes: 30 rewrites of up to
  2.6 MB on slice_a. Mark the table dirty and `put` once per fingerprint in `reset()`/at the end of the run (and in
  `absorb` after the loop). A crash mid-run then loses that run's new onsets, which is acceptable for a cache. Size S.
  Verify with warm.sh (hits must still show "30 years cached").
- **C6. `years` part of observed onsets** (`observed_cache.py:196`). Entries are tables by year, but the fingerprint
  includes the set of years (and `obs_files` covers exactly those years). Running 1991-2020 and then 1991-2021
  recomputes all 31 years. Making parts per-year needs per-year file identities: identify the files of all years
  available in the folder, not just the requested ones, and drop `years`. Size M. Benefit: only in year-sweep
  workflows. Defer until somebody runs them.
- **C7. Small correctness hardening** (S each): `Store.nearest` (`cache.py:290`) reads every `parts.json` in the kind
  folder on the first miss of each fingerprint, which is O(entries) and noticeable once the global detect folder holds
  thousands; cap it at the 200 newest. `io/input.py:96 _reference_key` lacks `obs_paths`/`obs_dims`. That is safe
  inside one run, but add them anyway. `observed_cache._threshold_hash` memoises by `id(thresh)` while holding a
  reference: that is correct and should stay.

## 5. Suggested order

1. B1, B2, B4, B7 and the `SyntaxWarning`s: small, independent, testable (S).
2. B3 (version parts) and C5 (S).
3. D2, D3, D4, D6 rows, and the §3 deletions (S each, one commit per row).
4. C4 memo registry together with the private-name clean-up (M).
5. D1 alignment onto `Store` and C2 `cache clean` (M).
6. D5 mask unification, only after the cell-by-cell mask comparison script passes (L).

Performance notes seen along the way, for perf_*.md: `land_mask.get_shp` re-reads the 10m admin shapefile on every
`shp_mask` call, and graphics call it per figure (cache it, or let D5 remove it). `apply_nc_mask` opens the mask
file once per call. `land_mask._grid_key` and `region._KEEP_CACHE` copy lat/lon into keys (cheap at these sizes).
The `load_thresholds` hit path reads a whole year only to learn the grid; the grid could come from the `thresholds`
entry's coords or from a lazy open.
