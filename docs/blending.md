# Blending forecasts into onset probabilities

A blend turns the onset climatology and the onset dates each forecast model produces into probabilities that the onset falls in week 1, 2, 3, 4 or later after an issue date. Training cross-validates the blend against climatology baselines and fits the operational model. Applying a trained blend to a live forecast file gives the probabilities for that issue date.

A blend is part of the normal run file: a `blend:` section next to the usual `observations`, `models`, `onset`, `units` and `evaluation` sections. There is no separate training command: `hcwf-run -p run.yaml` trains when the file has a `blend:` section, and the benchmark of the other sections is not run.

## A first blend

```yaml
project_name: Blend of one model
region: null                       # no region: the whole domain of the files; or a region id such as ethiopia
observations:
  paths: [/data/obs/*.nc]          # any number of files per year; names or globs
  variable: precip
  units: mm
  day_boundary: "00 UTC, labelled by start"   # needed without a region
models:
  - name: ens
    paths: [/data/ens/*.nc]
    variable: tp
    units: mm
onset:
  definition: moron_robertson_blend   # a registry id; the region's default if there is a region
  threshold_years: [2000, 2009]
  lead_label: start
units:
  mode: cells                      # a blend runs on units; here the files' own grid cells
evaluation:
  start_date: [2010, 5, 1]         # the years of the blend's issue dates
  end_date: [2013, 9, 30]
  start_year_clim: 2000
  end_year_clim: 2013
blend:
  years: {training: '2010:2012', cv_holdout: '2010:2012', true_holdout: 2013}
  fixed_cutoff: 06-10
  reference_onset: {month_day: 06-15}
  formulas:
    blended_model: outcome ~ prob_clim_qx + diff_ens_qx
  climatology: {forecast_window: 40, min_onset_years: 5}
  models:
    ens: {lead_days: [1, 40], rain_predictors: [{agg: diff}]}
```

Observations and models are read as the benchmark reads them: `path` is a folder of yearly files (`file_pattern`, `{}` is the year), a zarr store holding every year (`path: data/obs.zarr`), or `paths` lists files. The same dataset entries therefore work for a benchmark and a blend, and a model's forecasts may be one zarr store of all years (training reads the store's issues of the `evaluation` years, `start_date` to `end_date`, as the benchmark does; yearly files are read whole):

```yaml
observations: {path: data/ethiopia/obs.zarr}
models:
  - {name: aifs, path: data/ethiopia/models/AIFS_v2.zarr}
```

```bash
hcwf-run -p blend.yaml --check     # the plan, including a "== Blend ==" block (outcomes, years, formulas, tag, artifact folder)
hcwf-run -p blend.yaml             # train, cross-validate, fit the final model, write blend/
hcwf-run init --blend --obs /path/to/obs --model /path/to/aifs -o blend.yaml   # a skeleton to edit
```

`hcwf-run init --blend` writes a skeleton with `TODO` marks; fill them in and it runs. A blend file without a `units:` section evaluates on the grid cells of every source (the default `hcwf-run convert` writes for a gridded config: `mode: cells`, trimmed grid ids, `dissemination_scope: inputs`); `--check` shows it under Units. Add a `units:` section only to change that.

## The `blend:` keys

Blending works in one fixed season of the onset definition: the only one, or the one `onset.season` names when the definition has several (the onset setup refuses several without it). A definition with derived seasons (`dunning`) is refused, as is a season starting at a climatological date, before any data is read.

`hcwf/config/schema.py` has every key with its help line. The main ones:

| Key | Meaning |
|---|---|
| `years` | `training` (fit on these), `cv_holdout` (scored by leave-one-year-out cross-validation) and optional `true_holdout` (scored, never fitted). Each a list or text such as `'2000:2024,2026'`. Required. |
| `formulas` | Name to formula (`outcome ~ prob_clim_qx + diff_<model>_qx`) or `{text, enabled}`. Features: `prob_clim_qx`, `<model>_p_onset_qx`, `diff_<model>_qx`, `max_<model>_<n>day_qx`, or a `_week<n>` variant. `diff_<model>` is the model's rain minus the onset definition's named rain threshold, so it needs a definition whose first named threshold is a rain total (`mean total over` or a quantile of totals, as in `moron_robertson`); a definition such as `icpac`, whose rule gives the amount inline, has none, and `liebmann_marengo` names a daily mean for its cumulative-anomaly curve, not a total, so `--check` reports the `diff` predictor as an error for both. Window tokens (`trigger`, `follow`, ...) need a first-passage rule and are refused for the curve rules (`liebmann_marengo`, `dunning`); numeric windows work for every definition. Use `<model>_p_onset_qx` or a `max`/`min` predictor there. |
| `final_formula` | The formula fitted on every training year for operations (default: the only enabled formula). |
| `cutoff_mode`, `fixed_cutoff`, `reference_onset` | Which issue dates are scored (`fixed_cutoff`, `ref_onset`, `no_ref_filter`); the `MM-DD` of the fixed cutoff; the reference onset (`{month_day: MM-DD}` or a table of dates). |
| `cv_methods` | `global` (default), `local` (one fit per unit), `clusters` (with `cluster_var`). |
| `models` | Per forecast model, by its name in `models:`: `lead_days`, `max_member`, `rain_horizon`, `rain_predictors`, `rain_transform`. |
| `climatology` | The KDE climatology: `issue_end`, `forecast_window` or `horizons`, `cv_by_year`, `min_onset_years`, `windows`. |
| `baselines`, `forecasts`, `mme`, `external_predictions`, `random_forest`, `window_variants` | Extra scored models: climatology baselines, each model's own probabilities (raw and Platt-calibrated), a multi-model ensemble with RPS-optimised weights, predictions made elsewhere, the CRPS random forest (which also gives daily probabilities), windowed-climatology variants. |
| `bins`, `probability_days`, `trim_after_true_onset`, `rain_transform` | The outcome bins (default week1..week4 and `later`), lead days summed into them (28), dropping issues after the observed onset, the rain feature transform. |
| `outputs` | `intermediates` (`observations`, `features`: written as `observed_onsets.nc`, `weekly_features.nc`; any other name is refused when the file is read), `pickles`, `dir` (where `external_predictions` files are read from). |

The onset itself comes from the `onset:` section: a registry definition or an inline `rule`, plus the options that make an onset search match another tool's exactly (`rain_resolution: exact`, `missing`, `negatives`, `gaps`, `horizon`), and thresholds from the observations, a table or a file (`threshold_years`, `thresholds`, `threshold_table`, `threshold_scale`).

## Where it runs: grid, polygons, political boundaries, any region

* **Region.** `region:` is any region or country of `hcwf-run onset regions` (its box, default definition, day boundary and climatology years apply), or `null` for the whole domain of the data. One region per run file.
* **Grid cells.** `units: {mode: cells}` uses the cells of the files that lie in the region's box. `mask:` limits them further (see README, "Region masks"). `units.id_digits` and `units.id_format` (`trimmed` or `fixed`) set how cell ids such as `10.00_70.25` are written.
* **Polygons and political boundaries.** Give the polygons in `units.shapefile` (with `unit_id_col`, optionally `unit_name_col` and `parent_col`) or a weights table in `units.weights` (source cell, unit id, weight, optionally one table per source). Rain is averaged onto each unit and the blend is trained and scored per unit. Relative paths in `units:` start at the run file's folder.
* **Subsets.** `units.dissemination` is a CSV list of units; `dissemination_scope: inputs` reads only those units, `training` keeps all units but marks the list as a scored subset. `units.box`, `units.outline` and `units.boundary` keep only units whose centroid lies inside a box or an outline.

```yaml
units:
  shapefile: /data/adm3.shp
  unit_id_col: ADM3_PCODE
  unit_name_col: ADM3_EN
  dissemination: /data/dissemination_units.csv
  dissemination_scope: training
```

Benchmarks (runs without `blend:`) use the same `units:` section; see `examples/units.yaml`.

## Several regions: batch

```bash
hcwf-run -p ethiopia.yaml -p india.yaml        # one after another
hcwf-run -p 'regions/*.yaml' --overwrite       # a glob names every file it matches
```

A failing file does not stop the others; the exit code is 1 if any failed. Flags apply to every file. Each file has its own run folder.

## What a training run writes

Run folder `runs/<region>_<definition>_<years>_<fingerprint>/` (next to the run file, or `output.data_dir`):

```
run.yaml  definition.yaml  run.log
blend/
  trained.blend                              the operational blend (a zip: run config, definition, recipe, models, observed onsets, units, thresholds, reference)
  summary_models_pooled_<tag>.csv            one row per model: brier, rps, auc, pietra, brier_skill, rps_skill, per-week brier_week1.. and auc_week1..
  summary_models_<tag>.csv                   per unit (id, model, cv_method, lat, lon, n, rps, brier ...)
  yearly_metrics_global_<tag>.csv            year, model, cv_method, brier, rps, auc
  cv_preds_<formula>_<method>_<tag>.nc       the cross-validated probabilities
  coefs_<formula>_<method>_<tag>.nc          the coefficients of each cross-validation fold
  formula_sample_support_<tag>.json          which features had data
  (platt and MME weights as CSV when configured)
```

`<tag>` encodes the cutoff mode and years, for example `fixed_cutoff_2010_2013`. A folder that already holds a run is refused unless `--overwrite`.

The names are those of the former `onset_blending` output contract, so consumers of those files keep working.

## Applying a trained blend

```bash
hcwf-run forecast runs/<run>/blend/trained.blend \
    --forecast aifs_ens='live/ens_2026-06-09.nc' --forecast aifs_det='live/det_2026-06-09.nc' \
    [--issue-date 2026-06-09] [--fewer-members] [--netcdf] [-o forecasts/2026-06-09]
```

* `--forecast MODEL=PATH`: one per trained model (the model's name in `models:`); PATH is a file, a glob or a zarr store of forecasts (all years in one store: pick the issue with `--issue-date`), and a model may be given several times for several files.
* The live files may hold more than the blend was trained on (lead days, members, a larger domain) but not less, and must have its units, accumulation, grid spacing and issue hour. Every difference is reported before anything is read, and the command exits with 1. `--fewer-members` lets an ensemble hold only some of the trained members.
* The issue date is the one the files hold; it must lie in the blend's issue window. Files holding one season forecast every issue in it; `--issue-date` picks one issue of files or of a store holding several, and files or a store holding several seasons need it.
* Output goes to `-o DIR`, which must be new or empty; default `forecasts/<issue date>/` beside the blend file. With several blend files (one per region) each gets a subfolder of `-o` named after the folder that holds its file, or after the run folder when that folder is `blend` (so `runs/<run>/blend/trained.blend` gives `-o/<run>/`); they are applied in turn to the same live files. If two blends would get the same name the command stops before writing anything and names them; give them separate calls with their own `-o`.

Files written:

* `blended_forecast_probabilities.csv`: one row per unit and issue date with `id, time, onset_threshold, lat, lon, cv_week1..cv_week4, cv_later` (the blend's probabilities) followed by `clim_week1..clim_later` (the climatology's).
* `daily_onset_probabilities.csv`, for a random forest blend: `id, time, lat, lon, p_day_1..p_day_N, p_later` and the week bins `cv_*`.
* `--netcdf` adds `blended_forecast_probabilities.nc` and `daily_onset_probabilities.nc`: the same probabilities over (`time`, `id`), with `lat`, `lon` and `onset_threshold` over `id`.

From Python:

```python
from hcwf.api import forecast
fc = forecast("runs/<run>/blend/trained.blend", {"aifs_ens": "live/ens_2026-06-09.nc"}, write=True, netcdf=True)
fc.weekly_table     # the CSV contract as a DataFrame
fc.weekly           # the same as an xarray.Dataset over (time, id)
fc.folder           # where it was written
```

Training from Python: `result = hcwf.run("blend.yaml", write=True)`; `result.blend` is the `hcwf.blend.training.TrainingResult` (`.evaluation` holds the score tables as DataFrames and the predictions; `.save(dir)` writes the artifacts). A saved blend loads with `hcwf.blend.trained.TrainedBlend.load(path)` (also `from hcwf.blend import TrainedBlend`) and `.forecast({...})`.

Blends trained by `onset_blending` (`onset-blending/trained-blend`) and by the earlier `momp` (`momp/trained-blend`) still load and forecast. They are not written again: retrain to get an `hcwf/trained-blend` file.
