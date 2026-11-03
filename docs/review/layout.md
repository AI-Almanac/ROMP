# Review: package layout and leftovers of the original implementation

Status: read-only review (2026-11-03, branch `monorepo`, HEAD 6194117). Nothing here is applied yet.
Cross-references: duplication/consistency/dead code/caching in `docs/review/consistency.md`;
measured hot spots in `docs/review/perf_*.md`. This document covers package layout, the
original implementation's leftovers, and import cost.

## 1. Current layout

38.6k lines in 117 modules. The import graph was built with `ast` over `hcwf/` (top-level and
function-level imports, scratchpad `ig.py`). "<-" lists who imports a module; `*` marks a function-level import.

| package / module | lines | role | imported by (main) | verdict |
|---|---|---|---|---|
| `__init__.py` | 20 | lazy `run`, `check` | users | fine, but too small (see §2.3) |
| `api.py` | 374 | `run`/`check`/`forecast`, `RunResult`, `execute` (the run orchestration), `_execute_blend` | `__init__`*, `cli`*, `driver`* | does two jobs: public API and orchestration |
| `cli.py` | 307 | `hcwf-run` subcommands init/datasets/convert/forecast/export | `driver`* | split across two modules with `driver.py` |
| `driver.py` | 177 | `run_hcwf`: run/check/batch, argv handling | `cli`* | **import cycle** cli.main→driver.run_hcwf (cli.py:301)→cli.handled_without_run (driver.py:152) |
| `config.py` | 223 | `Config`/`Dataset`/`Mask` + section dataclasses generated from the schema | `api`*, `lib.loader`*, `lib.init_config`* | the right public face for config; the rest of config is in `lib/` |
| `export.py` | 748 | onset NetCDF export; `case_kwargs` (used by `lib/check`, `lib/align_sample`) | `api`*, `cli`*, `lib.check`* | mixes output writing, re-running detection, and building case kwargs |
| `cache.py` | 321 | fingerprint store | io.input, blend, onset.climatology, stats.observed_cache | fine (caching is covered by consistency.md) |
| `app/` | 460 | `bin_skill_score` and `spatial_far_mr_mae` (the two scoring flows); `onset_spatial` and `onset_time_series` (notebook plots, imported only by tests/test_onset_apps.py) | `api`* | named after the original "apps"; the flows belong with scoring |
| `blend/` | 5.9k | blending | api, cli | good concept package; `recipe.py` (979) is the blend config, see §2 |
| `graphics/` | 1.5k | benchmark figures | app | figure code is split over three places: `graphics/`, `utils/visual.py` (767), and the plot half of `utils/land_mask.py` |
| `io/input.py` | 1461 | obs/forecast/threshold readers, unit reading, lead/member selection, `UnitForecast`, process-wide stores | 20 modules | **too big, several concepts** |
| `io/output.py` | 293 | score tables (computation), CSV/NetCDF writers, nested-dict helpers | app, api, export | half of it is not I/O (`score_tables`, `back_dated_member_share`, `add_back_dated_map`, `set_nested`) |
| `io/dict.py` | 95 | nested-dict and DataFrame helpers | metrics.skill, graphics | not I/O |
| `lib/` | 9.5k, 19 modules | **grab-bag**: config (10 modules), `--check` report, run folders, dataset detection/schema, converters, sampled alignment, explain cells, lead window, `Case`/`Setting` | everyone | split by concept |
| `metrics/` | 650 | `error` (FAR/MR/MAE maps), `probabilistic` (BS/RPS/AUC kernels), `skill` (the score flow) | app, blend, stats.score | overlaps `stats/score.py` and `stats/bins.py` |
| `onset/` | 9.8k | definitions, ingest, plan, engine, backends, explain, regions registry, climatology thresholds, viz, its own CLI `__main__` (159) | everyone | good core; holds two leftovers (`legacy.py`, `regions.legacy_domains`) |
| `params/region_def.py` | 44 | `domain()` (calls `onset.regions.box`), hard-coded India CMZ polygons | graphics.maps, utils.* | one-module package, magic numbers |
| `spatial/` | 1.4k | units, regrid, ids, axes | io, stats, blend, export | good; reading data onto units is in `io/input.py:983-1250` |
| `stats/` | 3.1k | **grab-bag**: `detect` (onset detection driven by the flat kwargs), `benchmark` (contingency), `bins`, `score`, `climatology` and `conditional_climatology` (references), `observed_cache`, `parallel` | app, export, blend, metrics | not statistics; four different concepts |
| `utils/` | 3.0k | **grab-bag**: `mask` (727, run masks), `land_mask` (426, masks + plotting), `region` (358, selection + block loading), `visual` (767, plotting), `log`, `workers`, `standard` (dim names), `practical` (paths), `printing` | everyone | nearly all of it is domain code (spatial, figures, io) |

Overlapping or misplaced roles:
- **Config is spread over 12 modules**: `config.py`, `lib/config_schema.py` (890, key table),
  `lib/config_yaml.py` (424, YAML→flat incl. dataset detection), `lib/dataset_schema.py`,
  `lib/dataset_detect.py` (969), `lib/onset_config.py` (480, `OnsetSetup`), `lib/convention.py`
  (`Case`/`Setting`), `lib/control.py` (case building), `lib/loader.py` (finishing steps + global
  config), `lib/parser.py` (argparse overrides), `lib/resolved.py` (writes the resolved config),
  `lib/init_config.py`, `lib/convert.py` (flat↔nested + config.in), `lib/lead_window.py`,
  `lib/assertion.py` (`HCWFValidator`, `HCWFConfigError`), and `blend/recipe.py` (blend section).
  Validation alone happens in four places: schema normalisation, `HCWFValidator`,
  `onset_config.check_config`, `check.build_report`.
- **A flat config is finished by four entry paths**: `loader.build_cfg` + `load_run_config`
  (lib/loader.py:160,197), `api._prepare` (api.py:116, calls the loader's private
  `_rename_deprecated_keys/_set_onset_mode/_resolve_paths`), `export._complete` (export.py:88, same
  private calls), and `driver._run` (driver.py:91, `plan_run` + validate). Each applies a slightly
  different subset of the steps.
- **Detection orchestration**: `onset/engine.py` is the typed engine. `stats/detect.py` is the
  adapter from the flat kwargs to it (`engine_options(kwargs)`, `_convention(kwargs)`, prints status
  counts), and is called from benchmark, bins, climatology, export, blend.features and app.
  `export.collect_*` (export.py:425-716) loops over years and calls detection again, alongside
  `stats/benchmark._onset_tables_of_years` (overlap is in consistency.md).
- **Outputs**: score CSV/NetCDF in `io/output.py`, onset NetCDF in `export.py`, resolved config in
  `lib/resolved.py`, run folder naming in `lib/run_folder.py`, figures in `graphics/`, `onset/viz/`,
  `utils/visual.py`, and `blend/{trained,forecast,training}.save`.
- **Two command lines**: `hcwf-run` (cli.py + driver.py) and `python -m hcwf.onset`
  (onset/__main__.py: list, describe, regions, schematic, explain, check-alignment).
- **Alignment**: `onset/alignment.py` (the check) and `lib/align_sample.py` (the sampled run-level
  check) sit next to `stats/detect.check_alignment_once`.
- **Masks/regions**: `onset/regions.py` (registry, box), `params/region_def.py` (box wrapper + CMZ),
  `utils/region.py` (selection), `utils/mask.py` (MaskSpec, built masks), `utils/land_mask.py`
  (land/shape masks + outline plotting), `spatial/units.py` (units in an area).

## 2. Target layout (by concept)

```
hcwf/
  __init__.py        public API, lazy: run, check, forecast, Config, RunResult, TrainedBlend, HCWFConfigError
  api.py             run / check / forecast / RunResult only (thin)
  run.py             execute(): the benchmark + blend orchestration now in api.execute/_execute_blend; RunState (§3.5)
  errors.py          HCWFConfigError, ConfigSchemaError, OnsetConfigError, RunExistsError, BlendFileError
  log.py             utils/log.py + the banner of utils/printing.py
  cache.py           unchanged; observed onsets move here as cache/observed.py if cache becomes a package
  parallel.py        stats/parallel.py + utils/workers.py
  config/
    __init__.py      Config, Dataset, Mask, sections (today's config.py)
    schema.py        lib/config_schema.py
    build.py         lib/config_yaml.py (YAML/mapping -> RunConfig) + lib/loader.py finishing steps, ONE resolve()
    model.py         the typed RunConfig, ModelSpec, ObsSpec, Window, OutputSpec (§3.1); replaces convention/control
    datasets.py      lib/dataset_schema.py ; detect.py <- lib/dataset_detect.py
    onset.py         lib/onset_config.py (OnsetSetup)
    validate.py      lib/assertion.py HCWFValidator + onset_config.check_config
    leads.py         lib/lead_window.py
    resolved.py      lib/resolved.py + flat_to_mapping/render_yaml of lib/convert.py
    init.py          lib/init_config.py
    convert/         blend.py <- lib/convert_specs.py ; legacy_in.py <- lib/convert.read_legacy (or deleted, §3.3)
  runs/              folder.py <- lib/run_folder.py ; check.py <- lib/check.py (the --check report)
  io/
    stores.py        open_store/is_zarr/_stores/_obs_years (io/input.py:58-135)
    observations.py  obs readers (io/input.py:520-800), load_observations (was load_imd_rainfall)
    forecasts.py     forecast readers, select_leads/select_members, UnitForecast (io/input.py:800-980,1281-1461)
    thresholds.py    load_thresholds/load_onset_thresholds/_model_thresholds (io/input.py:149-440)
    dims.py          utils/standard.py
    scores.py        writers of io/output.py ; onsets.py <- export.py writer half (write_netcdf, open_onsets, attrs)
  spatial/           + masks.py <- utils/mask.py ; land.py <- utils/land_mask.py (non-plot) ;
                     select.py <- utils/region.py ; read.py <- io/input.py:983-1250 (unit_series, read_on_units...)
                     regions box/CMZ: params/region_def.py -> onset/regions.py (box) and region YAML (polygons)
  onset/             unchanged core + detect.py <- stats/detect.py (run-level detection on typed specs),
                     sample_alignment.py <- lib/align_sample.py, explain_cells.py <- lib/explain_cells.py
  references/        climatology.py <- stats/climatology.py ; conditional.py <- stats/conditional_climatology.py ;
                     kde.py <- onset/climatology_kde.py  (onset/climatology.py thresholds stay in onset/)
  metrics/           contingency.py <- stats/benchmark.py ; bins.py <- stats/bins.py ; probabilistic.py ;
                     skill.py <- metrics/skill.py + stats/score.py ; error.py ; tables.py <- io/output.py score_tables etc.
  scoring/           deterministic.py <- app/spatial_far_mr_mae.py ; ensemble.py <- app/bin_skill_score.py ;
                     onsets.py <- export.py collect_* (the onset products of a run)
  blend/             unchanged (recipe.py keeps the blend config; it consumes RunConfig, not flat keys)
  figures/           graphics/* + utils/visual.py + land_mask plotting ; notebook helpers <- app/onset_spatial,
                     app/onset_time_series ; onset/viz stays (it is the onset component's own figure set)
  cli/
    __init__.py      main(), dispatch, HELP
    run.py           run/check/batch (driver.py + lib/parser.py overrides, generated from the schema)
    commands.py      init, datasets, convert, export, forecast (cli.py)
    onset.py         `hcwf-run onset ...` = onset/__main__.py (keep `python -m hcwf.onset` as a 3-line shim)
```
Deleted packages: `app/`, `lib/`, `stats/`, `utils/`, `params/`, `graphics/`; modules `driver.py`,
`export.py`, `io/input.py`, `io/output.py`, `io/dict.py` (its two used helpers go to `metrics/tables.py`).

### 2.1 Move table (old -> new; M = merged, S = split, D = deleted)

| old | new | kind |
|---|---|---|
| driver.py, cli.py, lib/parser.py | cli/{__init__,run,commands}.py | M+S, removes the cli↔driver cycle |
| onset/__main__.py | cli/onset.py (+ shim) | moved |
| api.execute/_execute_blend/_observed_dataset/_scores | run.py | S |
| config.py | config/__init__.py | moved |
| lib/config_schema.py, lib/config_yaml.py, lib/loader.py | config/schema.py, config/build.py | M (one resolve path) |
| lib/convention.py, lib/control.py | config/model.py (typed) | replaced (§3.1, §3.2) |
| lib/dataset_schema.py, lib/dataset_detect.py | config/datasets.py, config/detect.py | moved |
| lib/onset_config.py, lib/assertion.py | config/onset.py, config/validate.py (+errors.py) | moved/M |
| lib/lead_window.py, lib/resolved.py, lib/init_config.py | config/{leads,resolved,init}.py | moved |
| lib/convert.py | config/resolved.py (flat_to_mapping, render_yaml) + config/convert/legacy_in.py or D | S |
| lib/convert_specs.py | config/convert/blend.py | moved |
| lib/run_folder.py, lib/check.py | runs/folder.py, runs/check.py | moved |
| lib/align_sample.py, lib/explain_cells.py | onset/sample_alignment.py, onset/explain_cells.py | moved |
| stats/detect.py | onset/detect.py | moved, typed args (§3.1) |
| stats/benchmark.py, stats/bins.py, stats/score.py | metrics/contingency.py, metrics/bins.py, metrics/skill.py | moved/M |
| stats/climatology.py, stats/conditional_climatology.py, onset/climatology_kde.py | references/* | moved |
| stats/observed_cache.py | cache (observed table) | moved |
| stats/parallel.py, utils/workers.py | parallel.py | M |
| app/spatial_far_mr_mae.py, app/bin_skill_score.py | scoring/deterministic.py, scoring/ensemble.py | moved |
| app/onset_spatial.py, app/onset_time_series.py | figures/notebook.py | M (or D: tests only) |
| export.py | io/onsets.py (writers) + scoring/onsets.py (collect) | S |
| io/input.py | io/{stores,observations,forecasts,thresholds}.py + spatial/read.py | S |
| io/output.py, io/dict.py | io/scores.py + metrics/tables.py | S |
| utils/mask.py, utils/region.py, utils/land_mask.py | spatial/{masks,select,land}.py + figures/outline.py | moved/S |
| utils/visual.py, graphics/* | figures/* | moved |
| utils/standard.py, utils/log.py, utils/printing.py, utils/practical.py | io/dims.py, log.py, log.py, D (only `set_dir` is used; `set_dir2/3/_safe` have 0 users) | M/D |
| params/region_def.py | onset/regions.py (`box`) + region YAML (`polygons:`) | D |

### 2.2 Rules for the moves
- Moves are pure renames done with `git mv` plus import rewrites (a mechanical script), with **no
  re-export shims** inside the package. Only the external entry points stay where they are:
  `hcwf.cli:main` (pyproject.toml script, and regress/run.sh calls it), `python -m hcwf.onset`,
  `hcwf.blend.TrainedBlend`, `hcwf.config.Config`.
- 30 test files import `hcwf.lib.*`, 23 import `hcwf.stats.*`, 20 import `hcwf.io.*`. They are
  rewritten by the same script in the same commit.
- Pickled or spawned state: functions sent to spawned workers (stats/parallel.py:161) are found by
  module path. Nothing is persisted by module path except the trained.blend format (blend/trained.py
  stores its own format id, not class paths: safe). Check `cache.py` fingerprints for module names
  before moving (`__qualname__`/`__module__` in keys would invalidate on-disk caches; that is harmless
  but costs a cold run).

### 2.3 Public API
```python
import hcwf
hcwf.run(cfg, write=False, **overrides) -> hcwf.RunResult
hcwf.check(cfg, **overrides) -> hcwf.runs.check.CheckReport
hcwf.forecast(blend, forecasts, ...) -> hcwf.blend.forecast.Forecast   # today only hcwf.api.forecast
hcwf.Config, hcwf.Dataset, hcwf.Mask       # lazy re-exports of hcwf.config
hcwf.TrainedBlend                          # lazy, as hcwf.blend.TrainedBlend
hcwf.HCWFConfigError, hcwf.RunResult
hcwf.onset.load_definition / describe / regions   # what `python -m hcwf.onset` uses
hcwf.spatial.EvaluationUnits, UnitsSpec    # already exported (spatial/__init__.py)
```
Everything stays lazy (module `__getattr__`, as `blend/__init__.py:13` already does), so
`import hcwf` stays at 45 µs. Overrides: today these are flat keys (api.py:96-103); after §3.1
also accept dotted section paths (`evaluation.max_forecast_day=15`), keeping the flat names as
aliases from `schema.BY_KEY`.

## 3. Leftovers of the original implementation

### 3.1 The flat kwargs dict is still the internal currency (L)
The YAML has sections, but `config_yaml.build_config` flattens them into 147 flat keys
(`config_schema.FIELDS`: 39 without a section, 13 internal). Everything downstream passes that dict
around as `**kwargs`. There are 187 `**kwargs/**case_cfg/**vars(cfg)` sites, and the main functions
have very long keyword lists:
`metrics/skill.py:93 create_score_results` (36 params + `**kwargs`), `stats/benchmark.py:283
compute_metrics_multiple_years` (31 + `**kwargs`), `stats/bins.py:502 multi_year_forecast_obs_pairs`
(26), `graphics/func_map.py:22 spatial_metrics_map` (24), `utils/visual.py:150 portrait_plot` (37),
`io/input.py:751 load_imd_rainfall` and `io/input.py:917/939` (which recover their own kwargs with
`utils/practical.restore_args(locals())`, io/input.py:934). Models are kept as parallel
`*_list` keys (`model_dir_list`, `model_var_list`, ...). `control.make_case` (control.py:94) then
looks every one of them up again by list index (`case_across_list`, control.py:30) and pairs
`tolerance_days_list` with the windows by hand (control.py:112, 137).
**Proposal:** a frozen, typed `RunConfig` built once by `config/build.py`:
`observations: ObsSpec`, `models: tuple[ModelSpec, ...]` (dir/paths, var, units scale, pattern, dims,
members, accumulation, daily_boundary), `reference: ReferenceSpec | None`, `onset: OnsetSetup`
(exists), `evaluation: Evaluation` (dates, `windows: tuple[Window(start, end, tolerance)]`, day_bins,
max_forecast_day, init_days, probabilistic), `units: UnitsSpec | None` (exists), `mask: MaskSpec`
(exists), `output: OutputSpec`, `blend: Recipe | None` (exists), `paths: RunPaths` (exists),
`parallel: ParallelSpec`. A case is `(ModelSpec, Window)`. Readers take an `ObsSpec`/`ModelSpec`
plus the few run values they use. Migrate **leaf-first**:
io readers → onset/detect → metrics → scoring flows → figures. During the migration, keep
`RunConfig.flat()` as the single adapter (deleted at the end). `resolved_config.yaml` and the run
fingerprint keep reading the flat view, so run folder names and fingerprints do not change (verify
with `runs/folder.fingerprint` on the regression configs).
Benefit: signatures say what a function needs; no silent typos (an unknown kwarg is swallowed today);
no 130-field `asdict` per case. Risk: large surface; fingerprints/resolved YAML must stay identical.
Depends on the §2 moves (do them first so each file is touched once). Verify: regress run.sh +
cmp.sh (MOD=hcwf), full suite, blend parity tests (tests/blend), and fingerprints unchanged.

### 3.2 `Setting`/`Case` dataclasses and `asdict` merging (M, part of 3.1)
`lib/convention.py:99` generates `Setting` (130 fields) from the schema minus `Case`'s 32 fields
(convention.py:5-58). Each case merges both with `{**asdict(setting), **asdict(case)}`:
app/bin_skill_score.py:99,103, app/spatial_far_mr_mae.py:47,104, lib/control.py:190,
graphics/onset_map.py:209, export.py:130. The merge order is inverted between the two flows (case
wins in bin_skill_score, setting wins in spatial_far_mr_mae). It only works because the field sets
are disjoint, and `asdict` deep-copies every field on every case. The reference-model case is built
four times by hand (spatial_far_mr_mae.py:94-102, control.py:169-192, onset_map.py:194-201,
export.py:126-129). The 35-entry hand-written `cache_key` (bin_skill_score.py:58-94) re-lists config
fields and must be kept in step by hand.
**Proposal:** delete `convention.py`, `control.py`, `init_dataclass`, `get_setting`. Add
`RunConfig.case(model, window)` and one `RunConfig.reference_case(window)`. The prepare cache key
becomes `(model.key, evaluation.key, onset_setup_key)`, hashes of the typed specs.

### 3.3 Old `config.in` and `convert` of it (S)
`lib/convert.py:42 read_legacy` `exec()`s a Python config.in (convert.py:52). It is reached only by
`hcwf-run convert old.in` (cli.py:192-194) and named in loader.py:57 and the help text. `_LEGACY_RENAMES
{"mok": ...}` (convert.py:26) and `loader._rename_deprecated_keys` (loader.py:62) keep the `mok` key.
**Proposal (user decision):** HCWF is a clean break (blend_merge.md Wave 6), so delete config.in
conversion, `read_legacy`, the `mok` rename, and the help lines. Users with old files convert with
the last ROMP release. If it is kept, move it to `config/convert/legacy_in.py` and parse with
`ast.literal_eval` of assignments instead of `exec`. Verify: tests/test_run_config_yaml.py
and the convert tests updated; suite.

### 3.4 Legacy onset mode (flat wet_init/wet_spell keys) (M)
`onset/legacy.py` (97), `onset_config._legacy_setup` (onset_config.py:291), `is_legacy/detect_legacy`
(:106-117), the `onset_legacy` flag (loader.py:75), 9 schema fields marked "legacy:" (config_schema.py:505-514),
6 argparse options (parser.py:35-47), `Case.wet_*` fields, and the magic `EXTEND_END_DAYS = 47`
(legacy.py:22). Regression and example configs use definitions, not flat keys.
**Proposal (user decision):** remove legacy mode; `convert` (or a one-off message) maps flat keys to a
definition with `extends:` + override, which `onset/legacy.py` already knows how to build. Keep
legacy.py's translation only inside the converter. Verify: tests/test_onset_hcwf_legacy.py becomes a
converter test; regression unchanged (it uses definitions).

### 3.5 Global mutable module state (M)
- Process-wide config: `lib/loader.py:26-27 _cfg/_setting`, `use_config` (loader.py:205), `get_cfg`
  (:217, `sys.exit(1)` on invalid config unless `debug`), `get_setting`. They are used only as fallbacks
  when `cfg is None` in app/* and by `__main__` blocks; `api.execute` installs them around the run
  (api.py:316). → Delete them; pass `RunConfig`.
- Memo tables reset by hand in `api.execute` (api.py:288-293, 343-346): `utils/mask._BUILT/_RECORDS`
  (mask.py:645), `stats/benchmark._ONSET_TABLES` (:140), `stats/climatology._MEMO` (:17),
  `stats/detect._alignment_done` (:31), `io/input._loaded_thresholds/_reported_ingest` (:421,:180).
  Plus never-reset ones: `io/input._stores/_obs_years` (:67,:80), `onset_config._cache` (:211),
  `dataset_detect._SAMPLES` (:136), `land_mask._RAW_MASKS/_LAND_BOOL`, `region._KEEP_CACHE`,
  `mask._NE_CACHE`, `bins._YEAR_TEXT`, `observed_cache._tables/_new/_memo`.
  `_execute_blend` resets a different subset (api.py:247-251) than `execute`. A notebook that calls
  `hcwf.run` twice therefore depends on which tables happen to be reset.
  → One `RunState` object (in `run.py`) owns the per-run tables (records, onset tables, reported
  sets) and is created per `execute`. Pure memoisation of immutable inputs (Natural Earth shapes,
  land masks per grid) moves to one bounded cache (`functools.lru_cache` or the mechanism
  consistency.md proposes). Verify: run twice in one process with different configs (new test) and
  get the same results as fresh processes; suite.

### 3.6 Case-sensitive legacy India box (S, needs a user decision)
`onset/regions/india.yaml:8-12 legacy_domains: {India: ...}`, `regions.box` (regions.py:276-290),
`check._box_text` (check.py:136-152), and `params/region_def.domain` (region_def.py:8). Writing
`region: India` gives a different box from `region: india`. **All regression configs use `region:
India`** (data/phase7/profile/cfg/slice_a/b*.yaml, examples/config.yaml).
**Proposal:** make the box explicit and case-insensitive: either a region id `india_imd_box` (alias
of india with the old domain), or a `region_box:`/`mask.box:` key. `region: India` then raises a
one-line `HCWFConfigError` that names the replacement (no silent change of cells). Update the
regression and example configs in the same commit. Verify: cmp.sh against out_base
must show identical outputs once the configs are updated (same cells). Without the config update the
India slices would differ, which is expected.

### 3.7 India-specific CMZ polygons and outlines (S)
`params/region_def.py:21-44 polygon_boundary` hard-codes the India Core Monsoon Zone outline for 1/2/4°
grids. `graphics/maps.py:16-35` averages over "CMZ", with `# Add India outline` at :144, :190 and :230.
`utils/land_mask.py:130 get_india_outline` ("Update this path to your India shapefile", :135).
`get_shp/shp_mask/shp_outline` default to `region='Ethiopia'` (land_mask.py:277,304,332). The
`polygon` key controls it (config_schema.py:484).
**Proposal:** move the polygons into `india.yaml` as `polygons: {cmz: ...}` (data, not code). Figures
draw the region's polygon when it has one. Remove the Ethiopia defaults (make `region` required).
Verify: figure tests (tests/test_graphics_colorbar_extend.py), regression `.png` count unchanged.

### 3.8 print vs logging (S)
Library code logs properly (`utils/log.py`); `print` remains in cli.py (15), driver.py (10),
onset/__main__.py (15) (fine for a CLI), `utils/printing.py:79,98-105` (`print_cfg`,
`print_data_info`, both with 0 users → delete), `lib/assertion.py:185-193` (`__main__`), and
`lib/init_config.py:159` (writing to stdout for `-o -`, fine). `api._execute_blend` logs with
f-strings, `log.info(f"  {line}")` (api.py:235, 300-307). Make this `%s` consistently.
`print_hcwf_banner` takes a namespace only to read `project_name` (printing.py:37; api.py:232
builds `SimpleNamespace(**dic)` for it). → `log.banner(project_name)`.
`stats/detect._print_back_dated/_print_status_counts` (detect.py:276,284) are named print but log.

### 3.9 `if __name__ == "__main__"` blocks (S)
app/bin_skill_score.py:167, app/spatial_far_mr_mae.py:146, app/onset_spatial.py:40,
app/onset_time_series.py:94, graphics/onset_map.py:174-229 (a 55-line script that rebuilds a case
by hand), lib/assertion.py:185, driver.py:176. All of them depend on the global `get_cfg`. → Delete.
Keep only cli.py:306 and onset/__main__.py (as a shim).

### 3.10 Naming (S)
- `load_imd_rainfall` (io/input.py:751; 15 call sites) reads every observation dataset →
  `load_observations(year, obs: ObsSpec)`. Variables named `imd` (stats/benchmark.py:190) → `obs`.
- `get_forecast_{deterministic,probabilistic}_twice_weekly` (io/input.py:917,939): nothing about them
  is twice-weekly (the init days come from config) → `load_forecast(year, model, ensemble=...)`.
- `spatial_far_mr_mae_map` / `skill_score_in_bins` → `scoring.deterministic.run` / `scoring.ensemble.run`.
- Config keys from the original: `save_nc_spatial_far_mr_mae`, `plot_heatmap_bss_auc`,
  `plot_bar_bss_rpss_auc`, `plot_panel_heatmap_*`, `layout` ("leave as is", config_schema.py:475; the
  case product is always model × window), `work_dir`/`pkg_dir` ("informational", :473-474), `debug`
  (:489, only consumed by the global `get_cfg`), and `years`/`years_clim`/`fallback_date` ("reserved",
  :490-492). → Delete `layout`, `work_dir`, `pkg_dir`, `debug`, and the reserved keys, or implement
  them. Renaming output keys is a user decision (it breaks run files; `convert` could map them).
- `stats/` and `utils/practical.py` ("practical") are names without meaning; `io/dict.py` shadows a builtin name.

### 3.11 Commented-out code and stale comments (S)
There is little left: lib/assertion.py:15-22 (old key notes), stats/climatology.py:229-230 (`# def ...
same as ...`), and the `import time` / `perf_counter` / `pprint` debugging in bin_skill_score.py:106-113,
141-147. Remove them with the move.

### 3.12 Magic numbers (S)
`legacy.EXTEND_END_DAYS = 47` (goes with 3.4). CMZ polygons (3.7). `obs_onset_analysis(lat_select=11,
lon_select=39)` (app/onset_time_series.py:27, Ethiopia). `init --years 2019 2024` default
(cli.py:76). The NetCDF description strings duplicated at spatial_far_mr_mae.py:62-65,124-127. Make
them named constants or region data. Numeric tolerances (`abs(lat_diff - 2.0) < 0.1`,
region_def.py:30) go with the polygons.

## 4. Proposals: benefit, risk, size, verification, waves

| id | proposal | benefit | risk | size | depends on |
|---|---|---|---|---|---|
| A | cli package: merge driver.py + cli.py + parser.py; generate overrides from the schema; `hcwf-run onset` | removes the cycle, one argv path | CLI behaviour (test_run_experience, test_onset_cli_explain) | S | — |
| B | one config resolve path (`config/build.resolve`) used by api, cli, export, check | removes 4 finishing variants | subtle step-order changes | M | — |
| C | package moves of §2.1 (lib/stats/utils/app/params/graphics → concept packages), split io/input.py, export.py, io/output.py | navigable layout | import churn, merge conflicts with perf work | L (mechanical) | A, B first (they rewrite those files) |
| D | delete config.in convert, `mok`, legacy onset mode, globals `get_cfg`/`use_config`, `__main__` blocks, dead helpers | less surface | user-visible removals | S–M | user decision |
| E | typed RunConfig + cases; delete Setting/Case/control | clear signatures, no asdict | largest change | L | C |
| F | RunState for per-run tables + bounded memo | correct repeated runs in notebooks | cache behaviour | M | C; coordinate with consistency.md caching |
| G | India box explicit + CMZ polygons as region data; Ethiopia defaults gone | no hidden case rule | regression configs must change | S | user decision |
| H | naming (load_observations, load_forecast, scoring.*) and output key renames | readability | key renames break run files | S (code), M (keys) | C (do with moves) |
| I | lazy heavy imports (§5) | faster CLI/config/worker start | none for results | S | C (fewer files after) |

**Waves (each with disjoint files, one commit per item, all verified the same way):**
1. **Wave R1 (parallel):** A (cli.py, driver.py, lib/parser.py, onset/__main__.py) ‖ D-part
   "delete" (lib/convert.py read_legacy, loader `mok`, `__main__` blocks in app/graphics/assertion,
   utils/practical/printing dead helpers) ‖ G (onset/regions*, params/region_def.py, region YAML,
   regression/example configs).
2. **Wave R2:** B (lib/loader.py, api.py `_prepare`, export.py `_complete`, lib/check.py) then
   D-part "legacy onset mode" (onset/legacy.py, lib/onset_config.py, schema fields) after the user decides.
3. **Wave R3:** C as a single scripted move commit per package (lib → config/runs/onset; stats →
   metrics/references/onset; utils → spatial/figures/io/log/parallel; app → scoring; graphics → figures),
   then the splits of io/input.py, export.py, io/output.py. No behaviour change. I (lazy imports) right after.
4. **Wave R4:** E, leaf-first in sub-commits (io readers → onset/detect → metrics → scoring → figures
   → delete Setting/Case/flat adapter), with H's function renames folded in.
5. **Wave R5:** F (needs the typed per-run object from E), coordinated with consistency.md caching.

Verification for every commit: `MOD=hcwf regress/run.sh /home/hcwf/code/ROMP <out>` then `cmp.sh`
against `out_base` (tolerance rule of blend_merge.md); `.venv/bin/python -m pytest -n auto -q`; blend
parity tests under tests/blend. For C additionally `python -c "import hcwf.cli, hcwf.config,
hcwf.blend.trained"` and a grep that no `hcwf.lib|stats|utils|app|params|graphics` import remains
(docs and notebooks included). For E/F additionally: run fingerprints of the four regression configs
are unchanged, and two `hcwf.run` calls in one process give the same results as two fresh processes.

## 5. Performance and efficiency in this area

Measured with `python -X importtime` (cumulative, cold cache from a warm disk):

| import | time | why |
|---|---|---|
| `hcwf` | 0.05 ms | lazy, good |
| `hcwf.cli` | 1 ms; `hcwf-run -h` 0.02 s wall | lazy, good |
| `hcwf.api` | 232 ms | pandas + xarray at module top, needed only in `_observed_dataset/_scores` |
| `hcwf.config` | 303 ms | config.py → lib/config_yaml → lib/dataset_detect → onset/ingest → **numba** (65 ms) + xarray + pandas |
| `hcwf.lib.init_config` / `lib.check` | 299 / 344 ms | same chain |
| `hcwf.lib.convert_specs` | 717 ms | pulls blend and scoring modules for `convert` |
| `hcwf.stats.detect` / `stats.benchmark` | 426 / 651 ms | io/input → utils/region → utils/land_mask → **regionmask + matplotlib + geopandas + pyproj** (84 ms), + scipy.stats (136 ms) |
| `hcwf.app.bin_skill_score` | 750 ms | + matplotlib.pyplot, seaborn via graphics at module top |
| `hcwf.export` | 700 ms | everything above |

Implications and proposals:
- **Spawned workers pay the import chain.** stats/parallel.py:161 uses `spawn`, so every worker
  imports the module of its target function. Today that chain includes regionmask/matplotlib/
  geopandas/pyproj (utils/land_mask.py:4-6, imported by io/input via utils/region) and numba
  (onset/ingest.py:186). Making `regionmask`/`matplotlib.path` lazy inside the land-mask functions
  and keeping figure imports out of the scoring path saves about 0.1-0.3 s per worker start. That
  multiplies by the worker count and by the separate pools (scoring, KDE, blend). Measure in perf_*.md.
- `hcwf.config` (and so `hcwf.run` with a `Config`, `hcwf-run init`, `datasets`) should not import numba:
  `config_yaml` needs `dataset_detect` only to resolve datasets. Import `onset.ingest` lazily in
  dataset_detect, or move the numba kernel to a submodule (`onset/_tenths_kernel.py`) imported on first
  use. Expected: `import hcwf.config` 303 → ~150 ms.
- `api.py`: move `pandas`/`xarray` into the two functions (the file already has `from __future__ import
  annotations`). `import hcwf.api` 232 → ~1 ms. This helps `hcwf.check` and `hcwf.api.forecast` start-up.
- graphics modules import `matplotlib.pyplot`, `seaborn`, `cartopy.crs` at top (graphics/func_map.py:5,9,
  onset_map.py:5,9, heatmap.py:3-4). The scoring flows import them even when every `plot_*` switch is
  off (`hcwf.run(write=False)` turns them all off, api.py:44). Import figure modules inside the
  `if plot_...` branches (the scoring flows already decide per switch).
- Repeated config resolution: `onset_setup(cfg)` is called from 36 sites and keys its memo by
  `repr` of 15 values on each call (onset_config.py:220-236); legacy mode is not memoised at all.
  `asdict(setting)` deep-copies 130 fields per case per flow (§3.2), and `case_kwargs` (export.py:113)
  rebuilds Setting and Case for every source and for `--check`. `HCWFValidator` wraps the dict
  in a new `SimpleNamespace` each time (assertion.py:35), and it runs 2-3 times per run (loader, api,
  driver). All of this goes away with the typed `RunConfig` (E), where the setup is resolved once and
  is an attribute. Cost today is milliseconds per call: correctness and clarity matter more than speed here.
- `hcwf.check` followed by `hcwf.run` resolves datasets twice, opening sample files both times
  (dataset_detect `_SAMPLES` memo, dataset_detect.py:136, softens it). A `RunConfig` returned by
  `check` and accepted by `run` avoids the second pass.
