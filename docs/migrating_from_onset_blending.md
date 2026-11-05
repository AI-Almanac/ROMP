# Migrating from onset_blending (and from momp)

The `onset_blending` repository now lives in HCWF: training, cross-validation and live forecasts of onset blends are part of the `hcwf` package and `hcwf-run`, and use the same data readers, onset definitions, regions and run folders as benchmarking. The package was called `momp` (command `momp-run`, project ROMP) before; section 6 lists those renames.

Results agree with onset_blending within a tolerance (floats to a relative 1e-9; onset days, statuses, counts and unit ids exactly), with the same procedures (onset rule, horizon, missing-rain handling, thresholds, CV folds, formulas). The parity options that make this so (`rain_resolution: exact`, `horizon: truncate`, ...) are written for you by `hcwf-run convert`.

## 1. The short version

```bash
# before
python -m onset_blending train configs/india_aifs.yml -o out/india
python -m onset_blending forecast out/india/trained.blend --forecast aifs=live/aifs.nc --issue-date 2026-06-09 -o out/india/2026-06-09

# now
hcwf-run convert configs/india_aifs.yml -o india.yaml        # once; read the comments at the top of india.yaml
hcwf-run -p india.yaml                                       # writes runs/<name>/blend/
hcwf-run forecast runs/<name>/blend/trained.blend --forecast aifs=live/aifs.nc --issue-date 2026-06-09 -o out/india/2026-06-09
```

Existing `trained.blend` files keep working: `hcwf-run forecast old/trained.blend ...` reads files with the format ids `onset-blending/trained-blend` (onset_blending) and `momp/trained-blend` (momp) as well as `hcwf/trained-blend`. Loaded old files forecast exactly as before but are not written again; retrain to get an `hcwf/trained-blend` file. On the synthetic test data, forecasting with a file trained by onset_blending and with the file retrained from its converted config gave byte-identical `blended_forecast_probabilities.csv`.

## 2. Commands

| onset_blending | HCWF |
|---|---|
| `python -m onset_blending train CONFIG -o DIR` | `hcwf-run convert CONFIG -o run.yaml`, then `hcwf-run -p run.yaml`. Output goes to the run folder's `blend/` subfolder; set `output.data_dir: DIR` to choose the folder (the blend then goes to `DIR/blend/`). A folder that already holds a run is refused unless `--overwrite`. |
| `python -m onset_blending forecast BLEND --forecast MODEL=FILE --issue-date D -o DIR` | `hcwf-run forecast BLEND --forecast MODEL=PATH --issue-date D -o DIR` (same flags; `-o` is optional now) |
| `python -m onset_blending convert-specs ...` | `hcwf-run convert --specs --name N --ground-truth GT --forecast MODEL=SPEC --combine C --connect K --cv-models M [--regrid R] [--data-root DIR] -o run.yaml` |
| `train --workers N` | `parallel: {workers: N}` in the run file |
| `train --cache-dir DIR` | gone; HCWF's caches are `runs/.cache` beside the run file and `$HCWF_CACHE_DIR` (default `~/.cache/hcwf`); see `hcwf-run cache info` |
| `train --intermediate observed\|weekly` | `blend.outputs.intermediates: [observations, features]` (writes `observed_onsets.nc`, `weekly_features.nc`). Only those two exist; `hcwf-run convert` leaves out `forecasts` and `climatology` with a note, and a run file naming them is refused |
| `-v` | `--verbose` (`--quiet` for warnings only) |
| `momp-run -p run.yaml` | `hcwf-run -p run.yaml` |
| `momp-run forecast ...`, `momp-run convert ...` | `hcwf-run forecast ...`, `hcwf-run convert ...` |

New: `hcwf-run -p run.yaml --check` validates and prints the plan (including a `== Blend ==` block) without running; several `-p` (or a glob) run several files, one region per file; several blend files can be given to `hcwf-run forecast` to apply each to the same live files (each in its own subfolder of `-o`, named after the folder that holds the file).

`hcwf-run convert` takes a BlendConfig YAML (`configs/*.yml`; `--region ID` names an HCWF region, default none, the data's whole domain, as blending read it). Old-style flat run YAMLs with `wet_init`, `wet_spell`, ... are also converted by it (to an `onset.rule`). It writes a run YAML with `region: null`, absolute paths (a `data_root` is resolved), `observations` and `models` with `paths`, an inline `onset.rule` with the parity options, `units:` (grid cells unless the config has a `domain`; a blend run file without `units:` gets the same grid-cell default) and the `blend:` section, and lists what it did not carry over in comments at the top. Review it, then `--check`.

### Forecast contract

Unchanged from onset_blending's output contract:

* `blended_forecast_probabilities.csv`: `id, time, onset_threshold, lat, lon, cv_week1..cv_week4, cv_later, clim_week1..clim_later`.
* `daily_onset_probabilities.csv` for a random forest blend: `id, time, lat, lon, p_day_1..p_day_N, p_later` and the week bins.
* The folder must be new or empty. Default folder: `forecasts/<issue date>/` beside the blend file (`<first>_<last>` for files holding several issues).

New: `--netcdf` also writes `blended_forecast_probabilities.nc` (and `daily_onset_probabilities.nc`) with the probabilities over (`time`, `id`); `--fewer-members` lets an ensemble hold only some of the trained members; an incompatible live file is reported with every difference and exit code 1. Details: [blending.md](blending.md).

### Training outputs

The evaluation artifacts keep onset_blending's names: `summary_models_<tag>.csv`, `summary_models_pooled_<tag>.csv`, `yearly_metrics_global_<tag>.csv`, `cv_preds_<model>_<method>_<tag>.nc`, `coefs_<model>_<method>_<tag>.nc`, `formula_sample_support_<tag>.json`, Platt and MME weight CSVs, `trained.blend`. What changed is the folder around them: they sit in `blend/` of the run folder, next to `run.yaml` (the resolved config with the source of every value, replacing the copied `config.yml`), `definition.yaml` and `run.log`.

## 3. Config: BlendConfig to run YAML

| BlendConfig | run YAML |
|---|---|
| `name` | `project_name` |
| `data_root`, `observations.files`, `models[].files`, `years` | `observations.paths`, `models[].paths` (names or globs), `years`; relative paths are made absolute by convert |
| `observations.variable`, `dims`, `missing_rain`, `date_marks` | `observations.variable`, `dims`, `onset.missing`, `observations.day_boundary` |
| `models[].unit_scale` | `models[].unit_cvt` |
| `models[].lead_days`, `rain_horizon`, `rain_predictors`, `max_member` | `blend.models.<name>.*` |
| `domain` (grid ids, regrid, dissemination, centroids, area of interest) | `units:` (`id_digits`, `id_format`, `shapefile` (converted with `cell_area: planar`, onset_blending's degrees-squared areas; HCWF's own default is `sphere`), `weights`, `dissemination`, `centroids`, `box`, `outline`) and `region` |
| `onset.definition`, `onset.threshold` | `onset.rule` (inline definition), `onset.thresholds`, `threshold_table`, `threshold_scale`; or a registry id (`moron_robertson_blend`, `icpac_blend` are onset_blending's readings of the two papers) |
| `onset.reference_onset`, `onset.fixed_cutoff` | `blend.reference_onset`, `blend.fixed_cutoff` |
| `climatology` | `blend.climatology` |
| `features`, `evaluation.years`, `formulas`, `cv_methods`, `baselines`, `mme`, `random_forest`, ... | `blend.years`, `blend.formulas`, `blend.cv_methods`, `blend.baselines`, `blend.mme`, `blend.random_forest`, `blend.bins`, `blend.probability_days`, ... |
| `outputs` | `blend.outputs` |

Once converted you can switch the inline rule for a registry definition or change the region (`region: ethiopia`) to use its box, day boundary and definition; the parity options then apply as you set them in `onset:`.

## 4. Python API

onset_blending's `BlendConfig` object tree is gone: configuration is the run YAML (`hcwf.config.build.resolve("run.yaml")` returns the typed `RunConfig`; `rc.blend` is the blend settings, `hcwf.blend.recipe.Recipe`).

| onset_blending | HCWF |
|---|---|
| `BlendConfig.from_yaml(p)`, `load_config` | `hcwf.config.build.resolve(p)` (a run YAML); `hcwf.config.convert_specs.convert_blend_file(p).to_yaml()` converts a BlendConfig in Python |
| `ConfigError` | `hcwf.errors.HCWFConfigError` (a `ValueError`); `RecipeError`, `ConvertError`, `BlendFileError` in the same module |
| `train(config, resources=...)`, `TrainingResult` | `hcwf.run("run.yaml", write=True).blend`, or `hcwf.blend.training.train(rc)`; `TrainingResult.save(dir, intermediates=("observed", "weekly"))` |
| `Resources` | `parallel: {workers: N}`; `hcwf.blend.training.resources(rc)`, `hcwf.parallel.default_resources()` |
| `TrainedBlend.load(p)`, `.forecast(files, issue_date=, resources=)` | `hcwf.blend.trained.TrainedBlend.load(p)`, `.forecast(files, issue_date=, fewer_members=, workers=)`; or `hcwf.api.forecast(p, files, write=True, netcdf=True)` (`from hcwf.api import forecast`) |
| `Forecast`, `forecast.run`, `WEEKLY_FILENAME`, `DAILY_FILENAME` | `hcwf.blend.forecast` (same names); `IncompatibleForecast` is in `hcwf.errors` (and importable from `hcwf.blend.forecast`) |
| `evaluation.evaluate`, `Evaluation`, `fit_final` | `hcwf.blend.evaluation` (same names; they take a `Recipe`) |
| `blend.logit`, `blend.mme`, `blend.random_forest`, `blend._crps_tree` | `hcwf.blend.logit`, `.mme`, `.random_forest`, `._crps_tree` (`fold_models` is in `hcwf.blend.logit`) |
| `core.design`, `core.calibration`, `core.predictions` (`Predictions.open`) | `hcwf.blend.design`, `.calibration`, `.predictions` |
| `core.features` (`rows`, `weekly`, `daily_inputs`, `rain_transform`, `Weekly`, `DailyInputs`, `Rows`) | `hcwf.blend.features` (functions) and `hcwf.blend.samples` (`Weekly`, `DailyInputs`, `ObservedOnsets`, `ForecastOnsets`); `Weekly.open` is unchanged |
| `core.onsets` (`observed`, `forecast`, `member_mean`) | `hcwf.blend.features.observed/forecast/member_mean`, on HCWF's one detection adapter `hcwf.onset.detect` |
| `core.onset_rule` (`OnsetRule`, `find_onsets`) | removed: `hcwf.onset.engine` with the parity options (`onset.rain_resolution`, `missing`, `negatives`, `gaps`, `horizon`) |
| `core.climatology` (`fit_kde`, `build`, `Climatology`) | `hcwf.references.kde` (`fit_kde`, `build_climatology`, `Climatology`) |
| `core.metrics` | `hcwf.metrics.probabilistic` (same names) |
| `core.domain` (`SparseTransform`, `regrid_weights`, `point_sources`, `grid_axes`, `AreaFilter`, `Regridder`) | `hcwf.spatial.regrid` (`SparseTransform`, `read_weights`, `point_sources`, `grid_axes`); `hcwf.spatial.units` (`EvaluationUnits`, `UnitsSpec`, replacing `AreaFilter` and `Regridder`) |
| `core.ids` (`with_id_column`, `GridIdConvention`, ...) | `hcwf.spatial.ids` (same names) |
| `core.reference` | `hcwf.onset.thresholds`, `hcwf.onset.climatology` (`ReferenceOnset`), `hcwf.io.thresholds`, `hcwf.spatial.regrid.read_weights` |
| `core.io` (file listing, reading, `file_year`) | `hcwf.io` (`stores`, `observations`, `forecasts`); `hcwf.config.datasets.file_year`; datasets are described by the run YAML |
| `core.parallel` (`map_ordered`, `workers_within`, `limit_blas_threads`, `SharedArrays`) | `hcwf.parallel` (`map_items`, the other names unchanged) |
| `legacy.specs.convert_specs` | `hcwf.config.convert_specs.convert_specs`, `convert_spec_files` |

A read of a run in a notebook:

```python
import hcwf
from hcwf.blend import TrainedBlend

result = hcwf.run("india.yaml", write=True)            # trains, writes runs/<name>/blend/
result.blend.evaluation.pooled                         # summary_models_pooled as a DataFrame
blend = TrainedBlend.load("runs/<name>/blend/trained.blend")
forecast = blend.forecast({"aifs": "live/aifs_2026-06-09.nc"})
forecast.weekly_table                                  # the CSV contract as a DataFrame
```

## 5. What stayed the same

* The blend: features, formulas, CV methods, baselines, MME, random forest, Platt calibration, scores.
* The forecast contract and the artifact names above.
* The compatibility rules for live files (they may hold more, never less, than the blend was trained on).

## 6. If you came from momp (ROMP)

| momp | HCWF |
|---|---|
| package `momp`, `momp.lib.*`, `momp.utils.*`, `momp.stats.*`, `momp.app.*`, `momp.graphics.*` | package `hcwf`, by concept: `config/`, `runs/`, `cli/`, `io/`, `spatial/`, `onset/`, `references/`, `metrics/`, `scoring/`, `figures/`, `blend/`, plus `errors.py`, `log.py`, `parallel.py`, `naming.py`, `cache.py` (the full table of moves is in `docs/review/r3_moves.md`) |
| `momp-run`, `MOMP_*` variables, `~/.cache/momp` | `hcwf-run`, `HCWF_*`, `~/.cache/hcwf` (or `$HCWF_CACHE_DIR`) |
| format `momp/trained-blend` | still loads; written files are `hcwf/trained-blend` |
| flat onset keys (`wet_init`, `wet_spell`, `dry_extent`, ...) | removed; `hcwf-run convert old.yaml -o new.yaml` rewrites a run YAML that has them as an `onset.rule` (see `docs/onset_in_hcwf.md`) |
| `config.in` files | no longer read; write a YAML run file (`hcwf-run init`) |
| `region: India` with the fixed case-sensitive box | `region: india` and `region: India` are the same registry region (box 6.5-38.5 N, 66.5-100 E), so India results can differ from older runs |
| keys `layout`, `work_dir`, `pkg_dir`, `debug`, `years`, `years_clim`, `fallback_date` | removed (they had no effect); `evaluation.years` and `evaluation.climatology_years` are new keys that restrict the years |
| reliability figure at 600 dpi | 300 dpi like the other figures |
