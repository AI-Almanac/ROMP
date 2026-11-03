# HCWF: Human-Centered Weather Forecasts

HCWF detects **rainy season onset** in observations and forecasts, benchmarks the forecasts and blends them: deterministic and probabilistic metrics, skill scores, spatial maps and reliability diagrams. Runs are driven by one config file.

Onset detection is done by the `hcwf.onset` component. An onset definition is a short YAML file (for example ICPAC's), run identically on observations and on every forecast.

## Install

Python 3.10 to 3.13. From a clone of the repository, in a fresh virtual or conda environment:

```bash
git clone https://github.com/AI-Almanac/HCWF.git
cd HCWF
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .                                       # or: pip install -e ".[dev]"
```

Optional extras:

* `pip install ".[gpu]"` adds CuPy for `onset_backend = "gpu"` (CUDA 13 build).
* `.[dev]` adds pytest, jupyterlab, black, ruff and others.

Check it: `python -c "import hcwf; print(hcwf.__file__)"` should print a path in your checkout.

## Quick start

One YAML file describes a run and is the single source of truth. Start one from your own data; HCWF detects the dataset details (variable, units, dims, members, file pattern, lead label) from the files and writes them with the evidence:

```bash
hcwf-run init --obs /path/to/obs --model /path/to/aifs --years 2019 2024 -o run.yaml
```

Review `run.yaml` (look for `TODO`), then run it:

```bash
hcwf-run -p run.yaml --check  # validate, resolve and print the plan; runs nothing (exit 1 on errors)
hcwf-run -p run.yaml          # or set HCWF_CONFIG=run.yaml, e.g. in a notebook
hcwf-run datasets -p run.yaml # what each dataset resolves to, with detections, warnings, problems
hcwf-run -p run.yaml -h       # lists the command-line overrides of config values
```

`--check` prints the settings with their sources, each dataset with what was detected and the evidence, the mask, lead windows, which files exist, the model-versus-observation alignment, hints (each names the key to set: units, latitude order, grid mismatch, leads that are not whole days, accumulation, a short climatology), the run folder and the cache. `--quiet` shows only warnings, `--verbose` adds debug lines (`run.log` always has them); progress shows as a tqdm bar if tqdm is installed, else as log lines.

`hcwf-run -h` lists all commands. `examples/config.yaml` is a short commented example. `hcwf-run init` writes only what you must decide; the region supplies the onset definition, day boundary and climatology years, and schema defaults fill the rest. Defaults are resolved at run time and never copied into your file. The sections `observations`, `models`, `onset`, `evaluation`, `metrics` and `output` flatten to the flat keys the code uses; every key, type, default and help line is in `hcwf/config/schema.py`. Unknown keys are errors with "did you mean" suggestions. `evaluation.years` and `evaluation.climatology_years` (a list or text such as `'2000:2020,2022'`, the syntax of `blend.years`) restrict the evaluated and climatology years to some years of the date ranges; unset, every year of `start_date`..`end_date` and `start_year_clim`..`end_year_clim` is used.

Datasets are described inline (`path`, `variable`, `units: mm|m`, `dims`, `day_boundary` for observations, `lead_label`, `accumulation` and `daily_boundary` for forecasts; a reference model, `reference:`, takes `accumulation`, `daily_boundary` and `dims` of its own and is read with them, not with a forecast model's). Left-out keys are detected from the first file and reported; a key that cannot be detected reliably, or that the files contradict, is an error naming it. Paths may use `~` and `${ENV_VAR}`. `hcwf-run init --dataset-template obs|forecast` prints a blank entry; entries can also live in your own files listed under `dataset_files:` and be referenced with `dataset: <id>` (inline keys win).

In a notebook:

```python
import hcwf
result = hcwf.run("run.yaml", max_forecast_day=15)   # a path, a hcwf.config.Config or a mapping; flat-key overrides
result.observed    # xarray.Dataset: observed onset dates (year, lat, lon)
result.scores      # deterministic: {"spatial": {(model, window): Dataset}, "summary": DataFrame}; ensemble: {"overall", "binned"} DataFrames
result.forecasts   # {source: Dataset} gridded onset dates of each model (computed on first use)
plan = hcwf.check("run.yaml")                          # what --check prints
```

`hcwf.run` writes no files unless `write=True` (then it behaves like `hcwf-run`; `overwrite=True` re-runs an existing run folder). `hcwf.config.Config(...).to_flat()` gives the validated flat dict and `.to_yaml(path)` writes the file.

**Container.** `scripts/entrypoint.sh` writes a run config from `HCWF_*` variables (`scripts/generate_config.py` lists them) and prints which onset definition it chose and why. With no `HCWF_ONSET_DEFINITION` that is the region's default (`HCWF_REGION`, default Ethiopia: ICPAC); earlier images defaulted to the flat rule of 1 mm wet days, a 3-day spell above 20 mm and a 7-day dry spell, and the variables `HCWF_WET_*`, `HCWF_DRY_*` and `HCWF_THRESH_FILE` are refused now. To keep the old rule, convert it with `hcwf-run convert` (see `docs/onset_in_hcwf.md`) and point `HCWF_ONSET_DEFINITION` at the file.

**Blends and batches.** A run file with a `blend:` section trains an onset blend instead of benchmarking (keys: `years`, `cutoff_mode`, `cv_methods`, `formulas`, `final_formula`, `window_variants`, `baselines`, `forecasts`, `mme`, `external_predictions`, `random_forest`, `fixed_cutoff`, `reference_onset`, `bins`, `probability_days`, `trim_after_true_onset`, `rain_transform`, `climatology`, `outputs`, and `models:` for per-model `lead_days`, `max_member`, `rain_horizon`, `rain_predictors`, `rain_transform`; `hcwf-run init --blend` writes a skeleton, `--check` describes it and the evaluation `units:`). Datasets may list their files (`paths`: names or globs, any number per year; `years` keeps those of some years; `unit_cvt` is the multiplier to mm of files that declare no units), `region: null` means no region, and `units.mode` / `units.weights` may be mappings by source. The artifacts go into the run folder's `blend/` subfolder under their own names. `hcwf-run -p a.yaml -p b.yaml` (or `-p 'regions/*.yaml'`) runs several files one after another, one region per file. `hcwf-run convert blend.yml -o run.yaml` (an onset_blending config) and `hcwf-run convert --specs --name N --ground-truth GT --forecast MODEL=SPEC --combine C --connect K --cv-models M -o run.yaml` (a legacy spec set) write the equivalent run YAML with the parity options (`rain_resolution: exact`, `horizon: truncate`, ...), an inline `onset.rule`, `units:` and `blend:`, and list what is not carried over in comments.

**Run folders.** Without `output.data_dir`, a run writes to `runs/<region>_<definition>[+mod]_<years>_<fingerprint>/` next to the config file (at most 40 characters; `+mod` marks an overridden definition; the 4-character fingerprint hashes everything that sets the results, not output switches). The same config is refused unless `--overwrite`, which clears that folder first; a changed config gets its own folder. Inside: `run.yaml` (every value after defaults, region, detection and overrides, each with its source) and `definition.yaml` (with the hash), `run.log`, and `onsets/`, `scores/`, `figures/` (a subfolder only when it would hold more than one file; short file names such as `scores/spatial_AIFS_1-15.nc`). `runs/.cache/` is shared by all runs. With an explicit `data_dir` (and `figure_dir`) the files and their names are the historical ones: `resolved_config.yaml`, `resolved_definition.yaml`, `spatial_metrics_<model>_<window>.nc`, ... in that folder, overwritten on a re-run. Old `config.in` files are no longer read.

**Observed-onset cache.** Observed onsets are detected once per run and shared by all models, windows and references, and kept across runs in `runs/.cache/observed_onsets/` (`<data_dir>/.onset_cache/` with an explicit `data_dir`), keyed by a fingerprint of the observation files, ingest settings, evaluated cells and mask, definition, thresholds and season climatology, and years (not the backend, models, windows or reference). The run prints a hit or miss and, on a miss, which parts differ from the nearest cached entry.

Each run loads the config, reads observations and forecasts, detects onset, scores the forecasts against the observed onset and a reference (climatology by default), then writes NetCDF/CSV results and figures.

Export the gridded onset dates (observed, each model, the climatology references) as NetCDF with `hcwf-run export -p run.yaml` or `output.save_onsets: true`; see "Exporting onset dates" in `docs/onset_in_hcwf.md`.

## Onset definitions

List, read and draw definitions from the command line:

```bash
hcwf-run onset list                     # ids, titles, regions
hcwf-run onset describe icpac           # the rule in words, hash, status, notes
hcwf-run onset regions                  # regions and countries with their defaults
hcwf-run onset schematic icpac --png icpac.png
hcwf-run onset explain --obs obs.nc --obs-var RAINFALL \
    --day-boundary "00 UTC, labelled by start" --definition icpac \
    --lat 9.0 --lon 38.75 --year 2020 --png anatomy.png
hcwf-run onset check-alignment --help   # are obs and forecast dates shifted?
```

`explain` prints why one cell got its onset date (each candidate day, which condition failed). The `hcwf.onset.viz` package also has Python functions for comparing definitions, forecast-versus-truth plots and map drill-downs.

Choose the definition in the `onset:` section of the run file, inline:

```yaml
region: Ethiopia               # known region: its default definition and day boundary apply
onset:
  definition: icpac            # optional: registry id or path to a definition YAML
  override: {wet_day: at least 2 mm}   # optional: change part of it (same rules as `extends`)
  # rule: {...}                # or a complete definition written inline
```

`override` is merged like `extends` and validated with the same strict errors; the run prints the definition's description, "icpac v1, overridden: wet_day" and its hash, and records them in `resolved_definition.yaml`. `dry_spell_check: false` is an override too. A path to a shared definition file still works. Regions are YAML files in `hcwf/onset/regions/`. A custom region in `hcwf/spatial/region_def.py` has no default definition, so set `onset.definition` and `observations.day_boundary` yourself.

The old flat onset keys (`wet_init`, `wet_spell`, `dry_extent`, ...) are no longer read: a config needs an onset definition, and one that still has the keys is refused with a pointer to `hcwf-run convert old.yaml -o new.yaml`, which writes them as an `onset.rule`. The keys `layout`, `work_dir`, `pkg_dir`, `debug`, `years`, `years_clim` and `fallback_date` were removed too (they had no effect).

More:

* `docs/onset_in_hcwf.md`: how HCWF uses the component (config keys, lead windows, forecast history, thresholds, converting the old flat keys).
* `docs/onset_spec.md`: the specification (YAML grammar, detection semantics, backends).

## Region masks

The optional `mask:` section limits the evaluated cells. The parts you give intersect:

```yaml
mask:
  land: true                 # land only (Natural Earth land polygons from cartopy)
  country: Ethiopia          # one or more names (NAME, NAME_LONG or ADM0_A3, any case)
  file: /path/mask.nc        # a 0/1 NetCDF (or a shapefile / GeoJSON); variable and keep_value pick the field
  condition: {file: /path/dem.nc, variable: elevation, above: 1500}   # a mask from a field (above / below)
  cell_rule: centre          # centre | overlap_fraction (with overlap_fraction: 0.5)
  resolution: 50m            # Natural Earth resolution: 10m, 50m or 110m
```

The mask is built on each dataset's own grid (the observations and every model separately); rainfall is never regridded. A shape is rasterised by the cell rule. A mask file is aligned first (latitude order, 0-360 or -180-180 longitudes, axis names); on the same grid it is used as it is, otherwise the mask (not the rainfall) is regridded: a finer mask keeps a cell when kept mask cells cover at least `overlap_fraction` of it, a coarser one gives the nearest mask cell, and a mask that does not cover the data is an error naming both extents. The run prints the sources, rule, alignment actions and kept cell count per dataset, saves `mask_<dataset>.png` in the figure folder, and records the sources with content hashes in `resolved_config.yaml` and the NetCDF outputs. The old flat `nc_mask` key still works as before (it cannot be combined with `mask`). Natural Earth data downloads on first use; `scripts/prefetch_cartopy_data.py` fetches it for offline machines. See `docs/onset_in_hcwf.md`.

## Package layout

```
hcwf/
  driver.py     hcwf-run entry point; api.py is hcwf.run / hcwf.check
  onset/        onset component: definitions, registry, regions, ingestion, backends, explain, viz
  app/          benchmarking workflow and diagnostics
  stats/        HCWF's onset adapter, ensembles, climatology
  metrics/      error and skill scores
  io/           input and output
  lib/          config schema, YAML loading, dataset detection, validation, lead windows
  config.py     Config objects for notebooks
  cli.py        the hcwf-run command
  graphics/     plots and maps
  params/       region_def.py (custom regions)
  utils/        helpers
examples/config.yaml  example run config
docs/                guides and specification
demo/                demo data and notebooks
tests/
```

## Outputs

* Run folder (or `dir_out`): spatial-metric NetCDFs, skill-score CSVs, the resolved config and definition, `run.log`, onsets; thresholds and observed onsets are cached in `runs/.cache/` (or `.onset_cache/`).
* Figures: metric plots and maps; `explain/` holds anatomy plots for cells listed in `explain_cells`.

## Docs and demos

* `docs/package_configuration.ipynb`: the config, section by section.
* `docs/package_run_guidance_et.ipynb`: running HCWF and reading the results.
* `demo/et/`: Ethiopia demo notebooks and [companion guidance](https://docs.google.com/document/d/15_evi1VI4GcCeSoiI6DcrfidwjEE4g-UPiiOY5kF2Zo/edit?usp=sharing).

## Development

```bash
pip install -e ".[dev]"
pytest -n auto        # full suite in parallel (GPU tests are grouped on one worker: --dist loadgroup is set in pyproject.toml)
pytest --testmon      # iterate: re-run only tests affected by your edits (cannot be combined with -n)
pytest --lf           # re-run the last failures
pytest tests/test_cache.py   # a single file needs no extra options
```

The pytest cache (`.pytest_cache/`) and testmon data (`.testmondata*`) are ignored by Git.

Current version 0.0.1; APIs may change. MIT License.

## Citation and contact

HCWF: Human-Centered Weather Forecasts, UChicago HCWF Authors, 2026.
Contact: Bo Dong, bodong@uchicago.edu
