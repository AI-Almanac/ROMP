# RunConfig design (wave R2: one config path, typed)

**Status: done (S0-S11, 2026-11-03).**

Status: design, nothing implemented. Scope is R2a (one resolve path, one validation module) and R2b (typed
frozen `RunConfig` replacing the flat kwargs, `Setting`/`Case` and `control.make_case`). Line numbers are
from `652d2b0` (branch `monorepo`).

Summary: the schema (`lib/config_schema.py`) stays the only place a key is declared. One function,
`hcwf/config/build.py:resolve`, turns a YAML path, `Config`, mapping or flat dict plus flat overrides into a
validated `RunConfig`. The `RunConfig` keeps the resolved flat record as is, because `run.yaml` and the fingerprint
are computed from it, so neither can change. It also carries typed, frozen projections of that record (specs).
Consumers move to the specs leaf-first, behind one bridge module that is deleted at the end.

---

## 1. Current state

### 1.1 The flat dict

* `config_schema.FIELDS` declares 133 keys (`config_schema.py:464-699`): 23 have no section, 13 are internal
  (the 9 `*_list` model keys plus `obs_unit_cvt`, `obs_paths`, `obs_dims` and `ref_model_unit_cvt`). The keys
  marked `fill=False` are absent when not given. Absent is not the same as `None` (fingerprint JSON, `flat_to_mapping`).
* The resolve steps add 9 keys that are not in the schema: `accumulation_detected`, `config_file`,
  `config_sources`, `dataset_report`, `lead_window`, `onset_lead_lines`, `run_paths` (a `RunPaths`),
  `cache_dir` and `short_names`. A resolved regression config (`slice_c_eth_aifsens`) has 83 keys and pickles to 4.0 KB.
* The case dict of the scoring is `{**asdict(Setting), **asdict(Case)}`. It has 147 keys: 124 `Setting` fields plus
  23 `Case` fields. Pickled it is 3.4 KB (`slice_a_india_aifsens`).

### 1.2 The four resolve paths

| step | `loader.build_cfg`/`load_run_config` (loader.py:134,169) | `api._flat`/`_prepare` (api.py:83,116) | `export.load_cfg`/`_complete`/`_as_dict` (export.py:73,86,97) | `driver._run` (driver.py:91) |
|---|---|---|---|---|
| sources accepted | YAML path (`-p` / `HCWF_CONFIG`) | path, `Config`, SimpleNamespace, flat dict (guessed from `model_list`/`obs_dir`, api.py:97), nested mapping | path (goes to `build_cfg`); dict, SimpleNamespace, `Config` | from the CLI (goes to `load_run_config`); a given dict or namespace |
| YAML to flat | `Config.from_yaml().to_flat()` (loader.py:49-57) | same, or `build_config` | via `build_cfg` | via loader |
| overrides | argparse (`parser.py:7-135`), applied **after** path resolution, **not** normalised and **not** recorded in `config_sources` (loader.py:146-148) | schema-normalised, source `"override"` (api.py:104-111), applied **before** path resolution | none | none |
| `_resolve_paths` | yes (loader.py:141) | **private call** (api.py:122) | **private call** (export.py:91), run again on an already resolved dict | via loader, or not at all for a given dict |
| lead window | `plan_run`; a blend gets `(1, max_forecast_day)` (loader.py:153-166) | `plan_run` | `apply_lead_window` when missing, **also for blends** (export.py:92-93) | `plan_run` if `run_paths` is missing |
| run folder plan and fingerprint | yes (`run_folder.plan/apply`) | yes, with the `write` flag; then `_NO_FILES` and tmp dirs when `write=False` (api.py:125-131) | `load_cfg` only; `_as_dict` plans nothing (`onsets_folder` falls back, export.py:128-131) | as the loader |
| validation | `HCWFValidator` (loader.py:172) | `HCWFValidator` (api.py:124) | `load_cfg` yes (export.py:82); `_as_dict` **no** | yes (driver.py:105) |

On top of these, `get_cfg` (loader.py:189) caches a config process-wide and calls `sys.exit(1)` when it is
invalid, `use_config` (loader.py:177) installs one for a `with` block (api.py:313), and `Config.to_flat`
(config.py:154) does the YAML-to-flat step only.

### 1.3 Where validation happens

1. Schema: `flatten_sections` (unknown keys with did-you-mean, config_schema.py:765), `normalise_value`,
   `normalise` and `apply_defaults` (:819-866). Then `config_yaml.build_config` (config_yaml.py:354-424): dataset
   detection problems, `mask`+`nc_mask` (:393-395), `_check_blend` (:272), the required `models`.
2. `HCWFValidator` (assertion.py:25-173): dates, list lengths, model names, window ranges, the forecast limit,
   probabilistic bins, explain_cells and the graphics conflicts. It is called at loader.py:172, api.py:124,
   export.py:82, driver.py:105, loader.py:194 and assertion.py:180.
3. Onset: `onset_config.check_config` (onset_config.py:376) runs from `HCWFValidator._check_onset`.
   `onset_setup(cfg)` can also raise later, wherever it is called.
4. `check.build_report` (check.py:582): `_units` catches `UnitsSpecError` (:345) and `_blend` catches `RecipeError`
   (:384). The rest is data checks: files, grids, leads, alignment, run folder, cache.
5. Lazy failures at run time: `recipe_from_config` (recipe.py:938, called at api.py:226), `units_spec`
   (io/input.py:1148, at read time, after the run folder exists) and `MaskSpec.from_mapping` (utils/mask.py:95).

### 1.4 Consumers of the flat dict

Counted with an AST scan of `hcwf/`. There are 237 functions that take a config as `dic`/`cfg`/`kwargs`/`config`
or as `**kwargs`. There are 143 call sites that splat a config dict (`**kwargs`, `**case_cfg`, `**vars(cfg)`,
`**prep_cfg`, ...), and 58 `**kwargs` definitions.

| area | modules: functions taking the config / `**config` call sites |
|---|---|
| io readers and spatial | io/input.py 38 / 18 (+8 `**kwargs` defs; `restore_args` at :934 etc.); utils/region.py 4 / 2; spatial/units.py 2 / 2; utils/mask.py 1 |
| detection and references | stats/detect.py 10 / 6; stats/observed_cache.py 8 (cache fingerprint parts, :179); stats/climatology.py 4 / 4; stats/conditional_climatology.py 6 / 5 |
| metrics | stats/benchmark.py 6 / 19; stats/bins.py 5 / 10; metrics/skill.py 6 / 9 |
| scoring flows | app/bin_skill_score.py 1 / 7; app/spatial_far_mr_mae.py 1 / 6; app/onset_time_series.py 2 / 6; app/onset_spatial.py 1 / 5 |
| figures and writers | graphics/* 9 functions with `**kwargs` (maps, func_map, heatmap, reliability, panel_*, onset_map); utils/visual.py 2 / 3; io/output.py 4 |
| run and config | lib/check.py 18 / 2; lib/run_folder.py 8; lib/onset_config.py 8; lib/config_yaml.py 11; lib/resolved.py 4; lib/loader.py 5; lib/lead_window.py 2; stats/parallel.py 8 / 4; export.py 15 / 16; api.py 5 / 2 |
| blend | blend/features.py 5 / 3; blend/training.py 3; blend/recipe.py 3; blend/trained.py 2 |

Biggest signatures:
* `metrics/skill.py:93 create_score_results`: 26 named parameters plus `**kwargs`, called with the 147-key dict.
* `graphics/func_map.py:23 spatial_metrics_map`: 24 plus `**kwargs`.
* `stats/benchmark.py:270 compute_metrics_multiple_years`: 21 plus `**kwargs`, and it rebuilds its kwargs with
  `restore_args`.
* `stats/bins.py:501 multi_year_forecast_obs_pairs`: 16 plus `**kwargs`.
* `utils/visual.py:149 portrait_plot` (35) is out of scope: those are plot-style parameters, not config.

There are 8 `restore_args` calls in io/input, benchmark, bins, skill and climatology. The flat dict is also used as
an extension channel: blend and export add keys that are not in the schema (`keep_unit_ids`, `member_ids`,
`max_member`, `issue_dates`, `last_lead`, `model_files_years`, `grid_point`, `lat_select`/`lon_select`,
`land_only`/`shp_only`, `reference_onset`, `rain_horizon`), and io/input reads them with `kwargs.get`.

### 1.5 Setting, Case and `control.make_case`

* `Case` (convention.py:6-49) has 23 fields. 11 of them are not schema keys: `case_name`, `model`, `model_var`,
  `model_dir`, `verification_window`, `tolerance_days`, `file_pattern`, `unit_cvt`, `accumulation`,
  `daily_boundary` and `model_dims`. `Setting` (convention.py:88) is built with `make_dataclass` and holds every
  schema key that is not a `Case` field, plus `cache_dir`, `short_names` and `lead_window`. So
  `run_paths`, `config_sources`, `dataset_report`, `config_file` and `accumulation_detected` never reach a case.
* `make_case(Case, (model, window), dic)` (control.py:108-156):
  1. `init_dataclass` copies the matching keys.
  2. For every `Case` field `x` with a `x_list` key, the value is looked up at `model_list.index(model)`
     (`case_across_list`, :33). `model_dir` is made absolute (:136-138). `tolerance_days` comes from the window's
     index in `verification_window_list` (:153).
  3. `years`/`years_clim` are resolved (:68-76), `members` is normalised (`"All"` becomes `None`, `"1-5"` a tuple,
     :79-105), and `case_name = combi_to_str(combi)`.
* Each flow loops over `product(model_list, verification_window_list)` (`iter_list`, :42) and merges the dicts.
  `bin_skill_score.py:89,93` lets the case win; `spatial_far_mr_mae.py:47,104` lets the setting win. The result is
  the same only because the field sets do not overlap.
* The reference case is built by hand: spatial_far_mr_mae.py:83-104, export.py:115-124 and the `__main__` block of
  onset_map.py:156-215. `control.ref_model_case`/`ref_cfg_layout` (:159-203) are dead code. There are also two
  rules nobody wrote down:
  * The **deterministic** flow sets `model_list=(ref,)` while the other `*_list` keys stay the forecast models'.
    The reference case therefore inherits **`models[0]`'s `accumulation` and `daily_boundary`**, and its
    `model_dims=None`, so `reference.dims` is ignored. The ensemble flow uses
    `skill.reference_model_kwargs` (skill.py:16) instead: the **current model's** accumulation and boundary, and
    `ref_model_dims`.
  * `export.case_kwargs(reference=True)` follows the deterministic rule. Parity requires keeping both rules (§4 Q2).
    **Superseded (Q2 decision, after R2):** both rules are replaced by one: the reference is read with its own
    `accumulation`, `daily_boundary` and `dims` (`reference.accumulation` / `.daily_boundary` / `.dims`, the accumulation
    detected from the reference files like a model's); `RunConfig.reference_cases(for_model=...)` only names the ensemble
    flow's cases after the forecast model.
* The 24-entry `cache_key` (bin_skill_score.py:59-84) holds only run-wide values plus per-model values. Within one
  run it is equivalent to the model name.

### 1.6 What goes to workers

* `parallel.start_years` (parallel.py:241) submits `_run_item(_year_call, year, (func, args, kwargs, skip), state,
  token, label)` for each year. Here `kwargs` is the whole 147-key case dict, plus `ens` and, for climatology
  sources, the climatological-onset DataArray or Dataset (benchmark.py:233).
* `blend/features.py:103,215` sends the flat run dict plus blend extras. logit, kde and calibration send their own
  contexts, not config.
* `state` is a snapshot of thresholds, observed-onset cache, masks and land masks (parallel.py:53-80).
* Each worker rebuilds `onset_setup(kwargs)` (setup_key memo, onset_config.py:176-186), which re-reads the
  definition files.

---

## 2. Design

### 2.1 Principles

1. **The schema declares each key once**: key, kind, default, help, section, choices. `RunConfig` declares no keys
   and has **no defaults**. Every spec value comes from the validated flat record, which got its defaults from
   `apply_defaults`.
2. **The resolved flat record is kept** inside `RunConfig` as read-only data. `run.yaml`, the fingerprint and
   `RunResult.config` are computed from it, so all three are unchanged by construction. No computation reads it
   once R2 is finished.
3. **Specs are projections built once** in `RunConfig.from_flat(values)`. They are frozen, `slots=True`, hold only
   tuples, scalars, paths and other specs (hashable, picklable), and are named after the schema: attribute =
   `Field.nested_name`.
4. R2 **renames functions** (proposal H: `load_observations`, `load_forecast`) and **changes signatures**, but does
   **not move modules**. New code goes straight into `hcwf/config/`, at the names layout.md §2 gives
   (`config/build.py`, `config/model.py`, `config/validate.py`), so R3 only moves the old `lib/` modules.

### 2.2 The type tree (`hcwf/config/model.py`)

The names avoid the user-facing section classes `hcwf.config.Evaluation`, `Output`, `Metrics`, `Reference`,
`Onset` and `Units` (config.py:122-128).

```python
@dataclass(frozen=True, slots=True)
class Window:                 # verification_window_list[i] + tolerance_days_list[i]
    start: int; end: int; tolerance: int
    span -> (start, end); label -> "1-15"; bins(day_bins) -> bins inside (control.filter_bins_in_window)

@dataclass(frozen=True, slots=True)
class ModelSpec:              # the *_list keys at one index (config_yaml.py:223-235)
    name: str; dir: Path | str; var: str; unit_cvt: float | None; file_pattern: str
    accumulation: str; accumulation_detected: bool; daily_boundary: str | None
    dims: tuple[tuple[str, str], ...] | None      # model_dims_list[i]; .dims_map() -> dict
    paths: tuple[str, ...] | None                 # model_paths_list[i] (blend)
    units: UnitsSpec | None                       # io.input.units_spec(values, name)  (pure, spatial/units.py:443)

@dataclass(frozen=True, slots=True)
class ObsSpec:                # observations section + obs_* internals
    name: str; dir: Path | str; file_pattern: tuple[str, ...]; var: str; unit_cvt: float | None
    paths: tuple[str, ...] | None; dims: tuple[tuple[str, str], ...] | None; day_boundary: str | None
    units: UnitsSpec | None

@dataclass(frozen=True, slots=True)
class ReferenceSpec:          # reference section + ref_model_unit_cvt
    model: str; kind: Literal["climatology", "conditional_climatology", "model"]; climatology_cv: str
    dir: Path | str | None; file_pattern: str | None; var: str | None; unit_cvt: float | None
    dims: tuple[tuple[str, str], ...] | None
    def as_model(self, like: ModelSpec, *, dims) -> ModelSpec   # explicit form of §1.5's inheritance rule

@dataclass(frozen=True, slots=True)
class EvaluationSpec:         # evaluation section + lead_window
    start_date: tuple[int, int, int]; end_date: tuple[int, int, int]
    years: tuple[int, ...]; climatology_years: tuple[int, ...]   # resolved (control.py:68-76)
    init_days: tuple[int, ...]; date_filter_year: int; windows: tuple[Window, ...]
    max_forecast_day: int; day_bins: tuple[tuple[int, int], ...]; probabilistic: bool
    members: tuple[int, ...] | None             # normalised (control.take_ensemble_members)
    lead_window: tuple[int, int]

@dataclass(frozen=True, slots=True)
class OnsetOptions:           # onset-section keys OnsetSetup does not hold
    rain_resolution, missing, negatives, gaps, horizon, threshold_scale, threshold_table,
    threshold_max_missing, threshold_partial_years, percentage_threshold, explain_cells
    def ingest(self, forecast: bool, units) -> dict   # replaces io/input.ingest_options(kwargs) (:160)
    def engine(self, units) -> dict                    # replaces stats/detect.engine_options(kwargs) (:33)

@dataclass(frozen=True, slots=True)
class AreaSpec:  region: str | None; mask: MaskSpec | None; nc_mask: Path | None; shpfile_dir: Path | None; polygon: bool
@dataclass(frozen=True, slots=True)
class ParallelSpec: enabled: bool; workers: int | None          # utils/workers.worker_count takes this
OutputSpec, MetricsSpec       # generated from the schema's output / metrics sections (2.3)
@dataclass(frozen=True, slots=True)
class FigureSpec:             # what figures read: dir_fig, region, shpfile_dir, polygon, save_fig, show_plot,
    ...                       # show_panel, date_filter_year, short_names, model names; built by rc.figure_spec()

@dataclass(frozen=True, slots=True)
class Case:                   # replaces convention.Case: one (source, window) pair
    model: ModelSpec; window: Window; reference: bool = False
    name -> combi_to_str((model.name, window.span)).replace(" ", "_")   # == today's case_name
    key  -> (model.name, window.span)                                  # == today's set_nested combi

@dataclass(frozen=True, eq=False, slots=True)
class RunConfig:
    project_name: str; area: AreaSpec; observations: ObsSpec; reference: ReferenceSpec
    models: tuple[ModelSpec, ...]; evaluation: EvaluationSpec
    onset: OnsetSetup            # existing (lib/onset_config.py:34), built once by onset_setup(values)
    onset_options: OnsetOptions; metrics: MetricsSpec; output: OutputSpec
    blend: Recipe | None         # existing (blend/recipe.py:309), recipe_from_config(values, onset)
    parallel: ParallelSpec
    paths: RunPaths              # existing (lib/run_folder.py:53); holds the fingerprint
    config_file: str | None; sources: tuple[tuple[str, str], ...]; report: tuple[str, ...]; lead_lines: tuple[str, ...]
    _values: dict                # the resolved flat record; exposed read-only as .values (MappingProxyType)
    def model(self, name) -> ModelSpec
    def cases(self) -> tuple[Case, ...]                # models x windows, in product order (today's loop)
    def reference_cases(self, like: ModelSpec | None = None) -> tuple[Case, ...]
    def figure_spec(self) -> FigureSpec
    def without_files(self, tmp: Path) -> RunConfig    # write=False: output off, explain_cells None, paths in tmp
```

Invariants, checked in `from_flat` and covered by tests:
* `len(models) >= 1`, and the names are unique and contain no spaces.
* `windows` are ordered, and each `end <= max_forecast_day`.
* `reference.kind == "model"` exactly when `dir`/`var` are set or defaulted.
* `blend is not None` exactly when `schema.has_blend(values)`.
* `paths.fingerprint == run_folder.fingerprint(values)`.

`MaskSpec` (utils/mask.py:65) is `AreaSpec.mask` (explicit `mask:` only). The implicit default mask stays in
`mask.default_mask`, because blend passes `land_only`/`shp_only` per call. `UnitsSpec` is resolved per source into
`ObsSpec.units` and `ModelSpec.units`; it was checked to be pure (no I/O), so this happens once at build time, and
the `units_spec(kwargs, source)` lookups disappear.

### 2.3 Keeping the key declared once

* **Generated specs.** `OutputSpec` and `MetricsSpec` are pure toggles and folders read by name. They are generated
  with `make_dataclass(frozen=True, slots=True)` from `schema.FIELDS` of their section, attribute = `nested_name`
  (the mechanism already in config.py:109-119). A new output or metrics key therefore needs only its `Field`.
* **Hand-written specs** are the ones that restructure keys: `ModelSpec` from `*_list`, `Window` from two lists,
  `ObsSpec`, `ReferenceSpec`, `EvaluationSpec`, `OnsetOptions` and `AreaSpec`. Restructuring is their purpose.
  They declare attribute names and types only; defaults, help and kinds stay in the schema.
* **Coverage test** (`tests/test_runconfig_coverage.py`). It builds `RunConfig.from_flat` from a full flat dict
  wrapped in a mapping that records every key read, and asserts that every `schema.FIELDS` key and each of the 9
  run keys is read exactly once by some projection. A new schema key fails CI until a projection claims it, and no
  second list of keys is kept anywhere.
* `ModelSpec` fields correspond to the `*_list` fields (`x_list` maps to `x`, `model_x_list` to `x`), which the test
  also checks.

### 2.4 One resolve path (`hcwf/config/build.py`)

```python
def resolve(source, overrides: Mapping[str, Any] = (), *, write=True, tmp: Path | None = None,
            check=False) -> RunConfig | tuple[RunConfig | None, Problems]:
    values, config_file = _read(source)        # path -> Config.from_yaml().to_flat(source=path); Config -> to_flat;
                                               # nested mapping -> build_config; flat dict -> copy (tests, old callers);
                                               # RunConfig -> returned as is (overrides then raise TypeError)
    _apply_overrides(values, overrides)        # api.py:104-111 verbatim: BY_KEY, not internal, normalise_value,
                                               # sources[key] = "override"; CLI flags arrive here too
    finish(values, config_file, write)         # loader._resolve_paths + anchors (loader.py:60-131), lead window /
                                               # blend window (loader.py:153-166), run_folder.plan + apply
    problems = validate.problems(values)       # 2.5
    ...raise (or return them in check mode)...
    rc = RunConfig.from_flat(values)
    return rc.without_files(tmp) if not write else rc
```

* `finish` replaces `_complete` and the private calls. Running it twice is not allowed: `resolve(RunConfig)`
  returns the object unchanged, so the "finishing an already finished dict" case (export.py:87) disappears.
* The order is `api._prepare`'s: overrides, then paths, then lead window, then plan. The CLI overridable keys
  (parser.py) are not paths, so moving their application before path resolution changes nothing.
* The CLI builds an overrides dict from the argparse flags actually given (argparse `default=None`), so it gets the
  same normalisation and `"override"` source as `hcwf.run` (§4 Q1).
* Callers, all through `resolve`:
  * `api.run`/`check`
  * `driver._run`/`_check`
  * the `export` entry points (they accept a path, `Config`, mapping or `RunConfig`)
  * `hcwf.check`
* `loader.build_cfg`, `load_run_config`, `plan_run`, `get_cfg`, `get_setting` and `use_config`, `export._complete`,
  `load_cfg` and `_as_dict`, and `api._flat` and `_prepare` are deleted.

### 2.5 Validation in one place (`hcwf/config/validate.py`)

`problems(values) -> Problems` runs every config check once:
* `HCWFValidator`'s checks, moved as functions with the same messages;
* `check_config` (removed keys and onset resolution);
* `units_spec` for every source;
* `recipe_from_config(values, setup)` when there is a `blend:` section;
* `MaskSpec.from_mapping`.

The schema-level checks stay in `config_schema` and `config_yaml`, where the values are normalised; they run inside
`_read`. `Problems` keeps today's two tiers:
* `fatal`: schema, cross-key and onset problems. `--check` reports these today as "the config cannot be resolved".
* `section`: `units` and `blend` problems. `--check` lists these in its Units and Blend sections today.

`resolve(check=False)` raises one `HCWFConfigError` holding both tiers, before any folder exists. `resolve(check=True)`
returns `(rc, problems)`: `rc` is `None` on a fatal problem, and a section problem leaves that spec `None`.
`check.build_report(rc, problems)` prints these in their present sections, so `--check` text stays identical (§4 Q3).
`HCWFValidator` and `check_config` are deleted, and `assertion.py` keeps only `HCWFConfigError` (R3 moves it to
`errors.py`).

### 2.6 Overrides

* `hcwf.run(cfg, **overrides)` and `hcwf.check(cfg, **overrides)` keep their flat names, validation and
  did-you-mean (api.py:104-111 moves into `build._apply_overrides`).
* `hcwf.run(rc)` with a `RunConfig` is accepted without overrides. A `RunConfig` cannot be re-overridden, because
  overrides apply before finishing (lead-window clipping, the plan): resolve again from the source instead.
* Run-mode changes (`write=False`) are not schema overrides. `rc.without_files(tmp)` does a `dataclasses.replace`
  of `output`, `onset_options.explain_cells` and `paths`. It also updates `_values` exactly as api.py:125-131 does,
  so `RunResult.config` is unchanged.

### 2.7 Resolved run.yaml and fingerprints

* `run_folder.fingerprint(values)` (run_folder.py:105) is called in `finish` exactly as today, and its result goes
  into `RunPaths`.
* `resolved.write_resolved(rc)` reads `rc.values`, `rc.sources` and `rc.onset`, calling the same
  `resolved_mapping`/`flat_to_mapping` code (resolved.py:31-77). The second write after the masks (api.py:330) stays.
* `_NOT_IN_FINGERPRINT` (run_folder.py:36) stays. It lists flat keys that are not results, so it is part of the
  fingerprint definition, not a key declaration.
* The observed-onset cache key (`observed_cache.fingerprint_parts`, :179) moves to typed inputs in S7, with a test
  that the parts are identical (on-disk caches stay valid).

### 2.8 Cases (replacing Case, Setting and make_case)

* `rc.cases()` is `tuple(Case(m, w) for m in models for w in evaluation.windows)`, the same order as
  `product(model_list, verification_window_list)`.
* The reference is `rc.reference_cases(like)`: `Case(reference.as_model(like, dims=...), w, reference=True)` for
  each window. The deterministic flow passes `like=models[0], dims=None` and the ensemble flow
  `like=case.model, dims=reference.dims`. This keeps today's two rules written down in one place (§4 Q2).
  Climatology references get a `ModelSpec` named `climatology`/`conditional_climatology` with the `ref_*` values, as
  today.
* The prepared-pairs cache in the ensemble flow becomes `{model.name: prepare_pairs(rc, model)}`: loop models, then
  windows. The 24-entry key goes away.
* `stats/benchmark._family_key` (:151) becomes `(ens, threshold_hash, years, case.model)`; the `repr` of 147 items
  goes away.
* Deleted:
  * convention.py, and all of control.py except four helpers that move into `config/model.py`:
    `evaluation_years`, `climatology_years`, `take_ensemble_members` and `filter_bins_in_window`;
  * `init_dataclass` and `export.case_kwargs`;
  * the `__main__` blocks that use `get_cfg` (app/*, onset_map.py:156, assertion.py:176).

### 2.9 What scoring, metrics and figures receive

Flows receive `(rc, case)`. Leaves receive the narrow specs. Plot-style keywords (cmap, figsize, ...) stay explicit.

| today | after (R2 name; R3 moves the module) |
|---|---|
| `load_imd_rainfall(year, **kw)` io/input.py:751 | `load_observations(year, obs: ObsSpec, area: AreaSpec, options: OnsetOptions, select: Selection = Selection())` |
| `get_forecast_{deterministic,probabilistic}_twice_weekly(year, **kw)` io/input.py:917,939 | `load_forecast(year, model: ModelSpec, ev: EvaluationSpec, area, options, *, ensemble: bool, select=Selection())` |
| `load_thresholds(**kw)` | `load_thresholds(onset: OnsetSetup, obs, area, options, ev)` |
| `detect_observed_onset(rain, thr, year, **kw)` stats/detect.py | `detect_observed_onset(rain, thr, year, onset, options, units)` |
| `compute_metrics_multiple_years(**147)` benchmark.py:270 | `compute_metrics(rc, case)` |
| `multi_year_forecast_obs_pairs(**kw)` bins.py:501 | `forecast_obs_pairs(rc, model, years)` |
| `prepare_score_cache(**kw)` skill.py:52 | `prepare_pairs(rc, model) -> PreparedPairs` (NamedTuple of the 4 tables) |
| `create_score_results(**147)` skill.py:93 | `create_score_results(case, prepared, metrics: MetricsSpec, ev: EvaluationSpec)` |
| `save_score_results(res, **kw)`, `save_metrics_to_netcdf(m, case_cfg)` io/output.py | `save_score_results(res, case, paths: RunPaths, output: OutputSpec)` etc. |
| `plot_spatial_metrics(m, **case_cfg)`, `create_heatmap(..., **kw)`, `panel_*(..., **vars(cfg))` | `plot_spatial_metrics(m, case, fig: FigureSpec)`, `create_heatmap(res, case, fig)`, `panel_*(results, window, fig)` |
| `spatial_metrics_map(da, name, *, years, shpfile_dir, polygon, dir_fig, region, ..., **kw)` func_map.py:23 | `spatial_metrics_map(da, name, fig: FigureSpec, years, *, <style keywords>)` |

`Selection` (frozen; `hcwf/config/model.py`) holds the per-call keys that are not config:
`keep_unit_ids`, `member_ids`, `max_member`, `issue_dates`, `last_lead`, `files_years`, `grid_point`, `lat`/`lon`,
`land_only`/`shp_only`, `reference_onset` and `rain_horizon`. It turns the extension channel of §1.4 into an
explicit argument. Blend features build one per call.

### 2.10 Picklability and workers

* Everything in `RunConfig` is a frozen dataclass, tuple, scalar or Path. `OnsetSetup` pickles to 2.9 KB today,
  and `Recipe` and `RunPaths` are dataclasses. `_values` is a plain dict, because a `MappingProxyType` cannot be
  pickled; the proxy is created in the property.
* Expected size is about 10 KB. A test asserts that `pickle.dumps(rc)` is under 32 KB and that it round-trips equal.
* Workers receive `(rc, case)`, or narrower specs where a leaf needs less. Year helpers take `args` instead of
  kwargs: `parallel.start_years(func, years, args, parallel: ParallelSpec)` calls `func(year_list, *args)`.
  `worker_count(rc.parallel, n)` replaces `worker_count(kwargs, n)`.
* Climatology arrays go as explicit arguments, not hidden in kwargs.
* Workers stop rebuilding `onset_setup` from flat keys: the resolved `OnsetSetup` travels inside `rc`.

### 2.11 Public compatibility

* Kept:
  * `hcwf.run`/`check` with a path, `Config`, nested mapping or flat dict, plus flat overrides;
  * `hcwf.config.Config/Dataset/Mask` and the section classes;
  * `RunResult.config` (still the resolved flat dict, from `rc.values`);
  * CLI flags, `--check` text, run folder names, `run.yaml`/`resolved_config.yaml`, `definition.yaml`.
* Added: `RunResult.run_config: RunConfig`.

---

## 3. Migration plan

**Transitional adapter.** There is exactly one module, `hcwf/config/flat_bridge.py`, marked `# R2-bridge`:
* `to_flat(rc) -> dict`: a copy of `rc.values`, for callees that are not yet migrated and take the run dict.
* `case_kwargs(rc, case, flow) -> dict`: byte-for-byte today's `{**asdict(Setting), **asdict(Case)}`, including both
  reference rules (`flow="deterministic"|"ensemble"`), for callees that are not yet migrated and take `**case_cfg`.
* `from_case_kwargs(kw) -> (RunConfig, Case, Selection)`: rebuilds the typed inputs from the flat names that a
  migrated function used to read. Each migrated function keeps its **old name** as a two-line `# R2-bridge` wrapper
  in its own module calling this, so callers in other files are untouched until their own step. Callers may have
  edited kwargs in place (`years=[year]`, `max_forecast_day=n_day`, the reference keys); rebuilding from the values
  keeps those edits.

S11 deletes the module and every `# R2-bridge` wrapper.

**Oracle.** `tests/_oracle_config.py` is a frozen copy of today's `Setting`/`Case`/`make_case`/`iter_list`, both
reference-case constructions, `export.case_kwargs` and the four finishing variants. The repo already does this
(`tests/_oracle_land_mask.py`, `tests/onset_oracle.py`). Parity tests compare old and new in the same process, so
temporary paths and fingerprints compare directly, and no golden files are stored.

**Checks for every step** (V):
* (a) `pytest -n auto` (`--testmon`/`--lf` while iterating);
* (b) `pytest tests/blend -n auto` (blend parity, also part of (a));
* (c) `MOD=hcwf /tmp/claude-1000/-home-hcwf-code-ROMP/f3fbeea6-493c-485b-bfb6-ad02edb09411/scratchpad/regress/run.sh
  /home/hcwf/code/ROMP <scratch>/out_rN`, then `cmp.sh <scratch>/out_base <scratch>/out_rN`: all SAME, floats
  rtol 1e-9, discrete values exact, `resolved_config.yaml` and `resolved_definition.yaml` identical (heavy runs can
  go on the second Spark);
* (d) `fp.py` (S0, scratchpad) prints `run_folder.fingerprint` and the `resolved_config_text` hash for the 4
  regression configs, `examples/config.yaml` and the test fixture configs: identical to `fp_before.txt`.

Each step is one Sonnet agent and one commit; the overseer verifies with (a)-(d) and does not re-run suites the
agent already ran.

| step | files | change | tests to add | V | risks |
|---|---|---|---|---|---|
| **S0** package and oracle | `git mv hcwf/config.py hcwf/config/__init__.py`; new `tests/_oracle_config.py`; scratchpad `fp.py` and `fp_before.txt` | none in behaviour | the oracle imports and agrees with the live code (`test_oracle_config.py`) | a, d | an import cycle from `hcwf.config` becoming a package (it imports lib.config_yaml); check `python -c "import hcwf.config"` and the CLI |
| **S1** one resolve path (R2a) | new `config/build.py`; `lib/loader.py` (keeps `_get_config_path_pre_parse` only), `api.py`, `export.py` (entry functions only), `driver.py`, `lib/parser.py` (defaults `None`, returns overrides), `lib/check.py` (entry), `hcwf/__init__.py` | §2.4; still returns the **flat dict** (`resolve_flat`). Delete `build_cfg/load_run_config/plan_run/_complete/load_cfg/_as_dict/_flat/_prepare` and update their 11+6 test uses | path, `Config`, mapping and flat dict give equal dicts; CLI flags equal `hcwf.run` overrides; `resolve` of a blend config gets the blend lead window through every entry; `export` on a dict now validates and plans | a-d | entry paths that skipped validation (`export._as_dict`) now validate, so a test fixture that was invalid fails: fix the fixture; CLI-overridden runs record `override` (Q1) |
| **S2** validation in one place | new `config/validate.py`; `lib/assertion.py`, `lib/onset_config.py` (`check_config` moves), `lib/check.py` (`_units`/`_blend` print `Problems`), `config/build.py`, `api.py:226` | §2.5 | every `HCWFValidator` message and every units/blend message is identical (oracle list of about 20 invalid configs); `--check` text for 3 fixtures (valid, bad units, bad blend) equals the oracle's | a, b, d | a units error now fails at resolve (no half-written run folder): intended, and noted in the commit |
| **S3** RunConfig types and bridge (parallel with S2) | new `config/model.py`, `config/flat_bridge.py`; tests only | §2.2-2.3, `from_flat`, `cases`, `reference_cases`, `without_files`; bridge `to_flat`/`case_kwargs`/`from_case_kwargs` | coverage (2.3); `case_kwargs(rc, case, flow) ==` oracle merge for every case and both reference rules over the fixture configs and the 4 regression configs; `from_case_kwargs(case_kwargs(...))` round-trips; pickle under 32 KB and equal; frozen; `paths.fingerprint == fingerprint(values)` | a | `from_flat` imports `io.input.units_spec` and `blend.recipe`: lazy imports keep `import hcwf.config` light |
| **S4** entry points carry RunConfig | `config/build.py` (returns `RunConfig`), `api.py` (`execute(rc)`, `_execute_blend(rc)`, `RunResult.run_config`), `driver.py`, `lib/resolved.py` (`write_resolved(rc)`), `lib/check.py` (`build_report(rc, problems)` with `to_flat` inside), `export.py` entry functions, `lib/explain_cells.py` entry, `utils/printing.print_hcwf_banner(rc.project_name)` | the top level takes `rc`; inner calls use `flat_bridge.to_flat(rc)` | `RunResult.config ==` the old dict; `write=False` leaves the same values as api.py:125-131 | a-d | `SimpleNamespace(**dic)` users (api.py:292, banner, explain_cells) |
| **S5** flows on cases; delete Setting/Case | `app/bin_skill_score.py`, `app/spatial_far_mr_mae.py`, `app/onset_spatial.py`, `app/onset_time_series.py`, `export.py` (`case_kwargs` users :410,453,558), `lib/check.py:479,565`, `graphics/onset_map.py` (`__main__`), `lib/loader.py` (globals), `lib/assertion.py` (`__main__`); delete `lib/convention.py`, `lib/control.py` | loops over `rc.cases()`/`rc.reference_cases(...)` call `flat_bridge.case_kwargs`; cache keyed by model (§2.8); test uses of `case_kwargs`/`init_dataclass` move to the bridge | per-flow: the sequence of kwargs given to `create_score_results`/`compute_metrics_multiple_years` equals the oracle's (record by monkeypatch) on 2 models x 2 windows with a model reference and a climatology reference | a-d | the two reference rules (§1.5) must come from `case_kwargs(flow=...)` |
| **S6** io leaves | `io/input.py`, `spatial/units.py`, `utils/region.py`, `utils/mask.py` (`shp_mask`), `utils/practical.py` (keep `restore_args` until S11) | `load_observations`, `load_forecast`, typed `load_thresholds`, `units_spec(obs or model spec)`, `ingest_options`/`engine_options` as `OnsetOptions` methods, `Selection`; old names stay as bridge wrappers; this module's tests move to the typed API | reader outputs equal through the wrapper and directly, for grid, units, listed-files and zarr fixtures; `Selection` keys cover every non-schema `kwargs.get` (AST test) | a-d | io/input.py is 1456 lines: if needed, split into S6a (observations, thresholds) and S6b (forecasts, `UnitForecast`), run one after the other |
| **S7** detection and references (parallel with S6) | `stats/detect.py`, `stats/observed_cache.py`, `stats/climatology.py`, `stats/conditional_climatology.py` | typed `detect_*`, `compute_climatological_onset(rc, ...)`; `fingerprint_parts(rain, thr, obs, area, onset, options, select)` | `fingerprint_parts` equals the oracle's (kwargs version) on cells, units, mask and `reference_onset` fixtures, so on-disk caches still hit | a-d | the cache key drifting would not break results but would silently cost re-detection, hence the test |
| **S8** metrics | `stats/benchmark.py`, `stats/bins.py`, `metrics/skill.py`, `stats/score.py`, `metrics/error.py`, `stats/parallel.py` (args-based year helpers next to the kwargs ones) | §2.9 rows `compute_metrics`, `forecast_obs_pairs`, `prepare_pairs`, `create_score_results`; `_family_key` typed; workers get `(rc, case)` | serial equals parallel (existing `test_parallel.py`) on the typed path; worker payload under 32 KB (measure `pickle.dumps` of the args) | a-d | the order of worker results and the `skip` semantics of `_year_call` must be kept |
| **S9** figures and writers (parallel with S8) | `graphics/*.py`, `utils/visual.py` (`set_basemap`, `heatmap` config parts only), `io/output.py`, `graphics/field.units_of` | `FigureSpec`, writers on `(case, paths, output)` | file names and figure sets equal (existing `test_units_benchmark.py` checks the names); `run_folder.file_name(paths, kind, ...)` | a, c | figure pixels are not compared: names, counts and `save_fig`/`show_*` behaviour are |
| **S10** remaining callers | `export.py` (`collect_*`), `lib/check.py`, `lib/align_sample.py`, `lib/explain_cells.py`, `blend/features.py`, `blend/training.py`, `blend/recipe.py` (`recipe_from_config` called only by `from_flat`), `app/*` (calls typed functions directly), `lib/lead_window.py` | every caller uses the typed API; `flat_bridge` is used only by tests | none new; blend parity is the check | a-d | blend features rebuild `max_forecast_day` per file (features.py:23): this becomes `dataclasses.replace(ev, max_forecast_day=n)` |
| **S11** delete the bridge | delete `config/flat_bridge.py`, every `# R2-bridge` wrapper, `utils/practical.restore_args`, the kwargs year helpers in `parallel.py`; migrate the remaining tests (`load_imd_rainfall` 18 uses, `get_forecast_*` 22, `create_score_results` 17, `detect_observed_onset` 14) | none in behaviour | guards: no `import flat_bridge`; no `**kwargs` parameter in io/, stats/, metrics/, graphics/, app/, export.py (AST test with an allowlist for plot-style and blend-internal `**options`); only `config/`, `lib/run_folder.py`, `lib/resolved.py` and `api.RunResult` read `RunConfig.values` | a-d | tests written against the old names: done here in one sweep, or step by step in S6-S10 (preferred: each step moves its own module's tests) |

Order and parallelism: S0, then S1, then S2 and S3 in parallel (disjoint files), then S4, then S5. After that,
S6 and S7 in parallel, then S8 and S9 in parallel, then S10, then S11. Every step leaves the suite and the
regression green, so the work can stop after any step. S1 and S2 complete R2a. S3 to S11 complete R2b.

---

## 4. Open questions (each with a recommendation)

**Decisions (user, 2026-11-03):** Q1 yes — command-line flags become validated, recorded overrides and
`--model_list` selects a subset; Q2 keep both reference rules in R2 for parity, then a separate small fix makes both
flows use the reference model's own settings and `reference.dims`; Q3-Q7 as recommended.

1. **CLI overrides as overrides.** Today `--max_forecast_day` and the other flags bypass normalisation and are
   recorded as "run config" in `run.yaml`. `--model_list A` without the matching `*_list` fails validation, or
   silently renames models when the lengths happen to match.
   **Recommend:** normalise the flags like `hcwf.run` overrides and record them as `override`. Make `--model_list`
   select a subset of `models:`, with an error on an unknown name. Only runs that use these flags see a different
   `run.yaml`; their results do not change.
2. **Reference model rules.** The deterministic flow's model reference inherits `models[0]`'s accumulation and
   daily boundary and ignores `reference.dims`. The ensemble flow uses the current model's accumulation and
   boundary and `reference.dims`.
   **Recommend:** keep both rules in R2 for parity, written down in `ReferenceSpec.as_model`. Afterwards, in one
   small commit with its own test, use `reference.dims` in both flows (a fix: no regression config sets it). Add
   `reference.accumulation`/`daily_boundary` schema keys later if users need them.
3. **`--check` with an invalid `units:` or `blend:` section.** Today these are listed inside the plan.
   **Recommend:** keep that output through the two validation tiers (§2.5). The simpler alternative, stopping at
   "cannot be resolved", changes the contract.
4. **`RunResult.config`.** **Recommend:** keep the flat dict (notebooks and tests read it) and add
   `RunResult.run_config`.
5. **Generated versus hand-written specs.** **Recommend:** the hybrid of §2.3: generate the toggle sections, write
   the restructuring specs by hand, and enforce one declaration with the coverage test. Generating every spec would
   lose the readable types the wave is for.
6. **Ordering against R3.** layout.md §3.1 suggested doing the module moves first; proposal.md puts R2 first.
   **Recommend:** R2 first, as approved. R2 renames functions (H) but moves no module, and puts new code directly
   in `hcwf/config/`. R3 then moves `lib/`, `io/input.py`, `stats/` and `app/` with the scripted `git mv`, and the
   new signatures move with them unchanged.
7. **Oracle lifetime.** **Recommend:** keep `tests/_oracle_config.py` until R3 is done, because the moves benefit
   from the same parity tests, then delete it.
