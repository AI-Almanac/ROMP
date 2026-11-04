# HCWF: Human-Centered Weather Forecasts

HCWF detects **rainy season onset** in observations and forecasts, benchmarks the forecasts (deterministic and probabilistic metrics, skill scores, maps, reliability diagrams) and trains and applies **onset blends** that turn forecasts and climatology into onset probabilities. Everything is driven by one YAML run file, on grid cells, polygons or political boundaries, for any region.

Onset is detected by the `hcwf.onset` component. An onset definition is a short YAML file (for example ICPAC's), run identically on observations and on every forecast.

HCWF absorbed the former `onset_blending` repository and was called `momp` / ROMP before. If you used either, read [docs/migrating_from_onset_blending.md](docs/migrating_from_onset_blending.md).

## Install

Python 3.10 to 3.13. From a clone of the repository, in a fresh virtual or conda environment:

```bash
git clone https://github.com/AI-Almanac/HCWF.git
cd HCWF
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .                                       # or: pip install -e ".[dev]"
```

Optional extras:

* `pip install ".[gpu]"` adds CuPy for `onset.backend: gpu` (CUDA 13 build).
* `.[dev]` adds pytest (with xdist and testmon), jupyterlab, black, ruff and others.
* `.[docs]` adds Sphinx for the documentation: `uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html`. Docstring conventions: [docs/contributing_docstrings.md](docs/contributing_docstrings.md).

Check it: `python -c "import hcwf; print(hcwf.__file__)"` should print a path in your checkout, and `hcwf-run -h` should list the commands.

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

`examples/config.yaml` runs on the sample data in `demo/data` (from the repository root: `hcwf-run -p examples/config.yaml`); `examples/units.yaml` is the same on polygon units.

`--check` prints the settings with their sources, each dataset with what was detected and the evidence, the mask and units, lead windows, which files exist, the model-versus-observation alignment, hints (each names the key to set: units, latitude order, grid mismatch, leads that are not whole days, accumulation, a short climatology), the blend (if there is one), the run folder and the cache. `--quiet` shows only warnings, `--verbose` adds debug lines (`run.log` always has them); progress shows as a tqdm bar if tqdm is installed, else as log lines.

## Commands

| Command | What it does |
|---|---|
| `hcwf-run -p run.yaml [--check] [--overwrite]` | Run the file: a benchmark, or a blend training when it has a `blend:` section. Repeat `-p` or give a glob to run several files in turn. |
| `hcwf-run init` | Write a starting run file (`--obs`, `--model`, `--region`, `--years`, `--definition`, `--blend`, `--dataset-template obs\|forecast`). |
| `hcwf-run datasets -p run.yaml` | Show how the datasets resolve. |
| `hcwf-run forecast BLEND --forecast MODEL=PATH` | Apply a trained blend to live forecast files. |
| `hcwf-run convert SOURCE -o run.yaml` | Convert an old run file with flat onset keys, an onset_blending config, or (`--specs`) a legacy spec set. |
| `hcwf-run export -p run.yaml` | Write the gridded onset dates of each source as NetCDF, with no scoring. |
| `hcwf-run cache info\|clean` | List or delete cached results. |
| `hcwf-run onset list\|describe\|regions\|schematic\|explain\|check-alignment` | The onset definitions and regions. |

Training a blend has no subcommand of its own: `hcwf-run -p run.yaml` trains when the file has a `blend:` section.

## The run file

Sections: `observations`, `reference`, `models` (a list), `onset`, `units`, `blend`, `evaluation`, `metrics`, `output`, plus the top-level keys `project_name`, `region`, `mask`, `parallel`, `dataset_files` and `catalog_dirs`. `hcwf/config/schema.py` lists every key with its type, default and help line; unknown keys are errors with "did you mean" suggestions. Defaults are resolved at run time and never copied into your file. The region (`hcwf-run onset regions`) supplies the onset definition, day boundary, box and climatology years when it has them; `region: null` means no region (the whole domain of the files).

```yaml
region: india
observations: {path: data/obs, variable: RAINFALL, units: mm}
models:
  - {name: AIFS, path: data/aifs, variable: tp, units: m, lead_label: end}
onset:
  threshold_years: [2013, 2020]
evaluation:
  start_date: [2015, 5, 1]            # the years evaluated; month and day bound the forecast init dates
  end_date: [2015, 6, 30]
  years: 2015                         # optional: only these years (a list or '2010:2013,2015'; may skip years)
  climatology_years: '2013:2020'      # optional: the same for the climatology
  verification_windows: [[1, 15]]
  tolerances: [3]
  max_forecast_day: 15
```

Datasets are described inline (`path` or `paths`, `variable`, `units: mm|m`, `dims`, `day_boundary` for observations, `lead_label`, `accumulation` and `daily_boundary` for forecasts; a reference model, `reference:`, takes `accumulation`, `daily_boundary` and `dims` of its own). Left-out keys are detected from the first file and reported; a key that cannot be detected reliably, or that the files contradict, is an error naming it. Paths may use `~` and `${ENV_VAR}`. `paths` lists files (names or globs, any number per year; `years` keeps those of some years; `unit_cvt` is the multiplier to mm of files that declare no units). Entries can also live in your own files listed under `dataset_files:` and be referenced with `dataset: <id>` (inline keys win).

The same run from Python:

```python
import hcwf
result = hcwf.run("run.yaml", max_forecast_day=15)   # a path, a hcwf.config.Config or a mapping; flat-key overrides
result.observed    # xarray.Dataset: observed onset dates (year, lat, lon; year, unit on evaluation units)
result.scores      # deterministic: {"spatial": {(model, window): Dataset}, "summary": DataFrame}; ensemble: {"overall", "binned"} DataFrames
result.forecasts   # {source: Dataset} gridded onset dates of each model (computed on first use)
plan = hcwf.check("run.yaml")                        # what --check prints
```

`hcwf.run` writes no files unless `write=True` (then it behaves like `hcwf-run`; `overwrite=True` re-runs an existing run folder). Overrides use the flat key names of the schema (`verification_window_list`, `max_forecast_day`, ...). `hcwf.config.Config(...).to_flat()` gives the validated flat dict and `.to_yaml(path)` writes the file. For a run file with a `blend:` section `result.blend` is the training result. `from hcwf.api import forecast` gives the function that applies a trained blend (see docs/blending.md).

## Evaluation units

By default HCWF works on the grid cells of the data inside the region's box. The `units:` section evaluates on other units instead:

```yaml
units:
  shapefile: adm3.shp            # polygons: admin units, catchments, ... (a relative path starts at the run file's folder)
  unit_id_col: ADM3_PCODE
  unit_name_col: ADM3_NAME       # optional, as is parent_col
```

Rain is read on each dataset's own grid, averaged onto the polygons with overlap-area weights (or the `weights:` table you give: source cells by id or lat and lon, unit id, weight), and onsets are detected, scored and blended per unit. Benchmark outputs are written per unit as CSV, GeoJSON and NetCDF. `units.mode` is `cells` (the files' own cells), `regrid` or `cell_transform` (a weights table applied before averaging); it may be a mapping by source (`observations`, a model's name, `'*'`). Other keys: `centroids`, `box`, `outline`, `boundary`, `dissemination` (a subset of units; `dissemination_scope`), `id_digits`, `id_format`. `examples/units.yaml` runs on the sample data.

## Blending

A run file with a `blend:` section trains an onset blend: cross-validated logit (and optionally CRPS random forest) over the climatology's and each forecast model's weekly onset features, scored against climatology baselines. The artifacts, including `trained.blend`, go into the run folder's `blend/` subfolder under their own names. `hcwf-run forecast` then applies a trained blend to live forecast files. See [docs/blending.md](docs/blending.md) for the keys, outputs and the forecast contract; `hcwf-run init --blend` writes a skeleton. Without a `units:` section a blend evaluates on the grid cells.

```bash
hcwf-run -p blend.yaml                                  # train and evaluate
hcwf-run forecast runs/<run>/blend/trained.blend --forecast aifs='live/aifs_*.nc' --netcdf
hcwf-run -p a.yaml -p b.yaml                            # several run files, one after another (or -p 'regions/*.yaml')
```

## Run folders

Without `output.data_dir`, a run writes to `runs/<region>_<definition>[+mod]_<years>_<fingerprint>/` next to the config file (at most 40 characters; `+mod` marks an overridden definition; the 4-character fingerprint hashes everything that sets the results, not output switches). The same config is refused unless `--overwrite`, which clears that folder first; a changed config gets its own folder. Inside: `run.yaml` (every value after defaults, region, detection and overrides, each with its source), `definition.yaml` (with the hash), `run.log`, and `onsets/`, `scores/`, `figures/` (a subfolder only when it would hold more than one file; short names such as `scores/spatial_AIFS_1-15.nc`) or, for a blend, `blend/`. With an explicit `output.data_dir` (and `figure_dir`) the files and their names are the historical ones: `resolved_config.yaml`, `resolved_definition.yaml`, `spatial_metrics_<model>_<window>.nc`, ... in that folder, overwritten on a re-run.

## Caches

Observed onsets are detected once per run, shared by all models, windows and references, and kept across runs in `runs/.cache/observed_onsets/` (`<data_dir>/.onset_cache/` with an explicit `data_dir`), keyed by a fingerprint of the observation files, ingest settings, evaluated cells and mask, definition, thresholds and season climatology, and years (not the backend, models, windows or reference). The run prints a hit or miss and, on a miss, which parts differ from the nearest cached entry. Dataset detections are kept in the shared cache `$HCWF_CACHE_DIR`, else `~/.cache/hcwf`.

```bash
hcwf-run cache info                                  # kinds, entries, sizes, ages of runs/.cache here and the shared cache
hcwf-run cache clean --older-than 30d --dry-run      # list what would go; drop --dry-run to delete
hcwf-run cache clean --kind detect                   # one kind (repeatable); --all deletes every entry; --dir DIR picks the folder
```

Whole entries are deleted; what a run needs again is recomputed once.

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

`override` is merged like `extends` and validated with the same strict errors; the run prints the definition's description, "icpac v1, overridden: wet_day" and its hash, and records them in the run's definition file. `dry_spell_check: false` is an override too. Regions are YAML files in `hcwf/onset/regions/`; one that has no `definitions` and is not `part_of` a region that has them has no default definition, so set `onset.definition` and `observations.day_boundary` yourself.

The flat onset keys of the first HCWF versions (`wet_init`, `wet_spell`, `dry_extent`, ...) are not read; a config that has them is refused with a pointer to `hcwf-run convert old.yaml -o new.yaml`, which writes them as an `onset.rule`.

More:

* [docs/onset_in_hcwf.md](docs/onset_in_hcwf.md): how HCWF uses the component (config keys, lead windows, forecast history, thresholds, exporting onset dates, converting the old flat keys).
* [docs/onset_spec.md](docs/onset_spec.md): the specification (YAML grammar, detection semantics, backends).

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

The mask is built on each dataset's own grid (the observations and every model separately); rainfall is never regridded. A shape is rasterised by the cell rule. A mask file is aligned first (latitude order, 0-360 or -180-180 longitudes, axis names); on the same grid it is used as it is, otherwise the mask (not the rainfall) is regridded: a finer mask keeps a cell when kept mask cells cover at least `overlap_fraction` of it, a coarser one gives the nearest mask cell, and a mask that does not cover the data is an error naming both extents. The run prints the sources, rule, alignment actions and kept cell count per dataset, saves `mask_<dataset>.png` in the figure folder, and records the sources with content hashes in the resolved config and the NetCDF outputs. The flat `nc_mask` key (a 0/1 NetCDF; it also applies the land and country masks implicitly) still works and cannot be combined with `mask`. Natural Earth data downloads on first use; `scripts/prefetch_cartopy_data.py` fetches it for offline machines. See `docs/onset_in_hcwf.md`.

## Container

`scripts/entrypoint.sh` writes a run config from `HCWF_*` variables (`scripts/generate_config.py` lists them) and runs it. It prints which onset definition it chose and why: with no `HCWF_ONSET_DEFINITION` it is the region's default (`HCWF_REGION`, default Ethiopia: ICPAC). The variables `HCWF_WET_*`, `HCWF_DRY_*` and `HCWF_THRESH_FILE` of the flat-key onset mode are refused; convert such a rule with `hcwf-run convert` (see `docs/onset_in_hcwf.md`) and point `HCWF_ONSET_DEFINITION` at the file.

## Package layout

```
hcwf/
  api.py        hcwf.run / hcwf.check / forecast
  cli/          hcwf-run entry point (driver, parser, onset subcommand)
  config/       run config: schema, YAML mapping, validation, dataset description and detection, onset setup, lead windows, convert
  runs/         run folder, resolved config, --check report, onset-date export (collection)
  io/           reading (stores, observations, forecasts, reference, thresholds, warmup, dims) and writing (output)
  spatial/      evaluation units, regrid, masks, region selection, reading cells onto units
  onset/        onset component: definitions, registry, regions (YAML and lookup), ingestion, engines, backends, explain, viz
  references/   climatology and conditional-climatology references
  metrics/      pure scores: pairs, contingency, probabilistic, error, tables
  scoring/      per-case scoring flows (pairs, onset tables, deterministic and probabilistic skill)
  figures/      plots and maps
  blend/        blend training, evaluation, trained.blend files, live forecasts
  cache.py, parallel.py, log.py, naming.py, errors.py   shared infrastructure (errors.py: every error HCWF raises on purpose)
examples/            example run configs
docs/                guides and specification
demo/                sample data and notebooks
tests/
```

## Outputs

* Run folder (or `output.data_dir`): spatial-metric NetCDFs (per-unit CSV and GeoJSON on polygon units), skill-score CSVs, the resolved config and definition, `run.log`, onsets; for a blend the `blend/` folder (see docs/blending.md).
* Figures: metric plots and maps; `explain/` holds anatomy plots for cells listed in `onset.explain_cells`.

## Docs and demos

* [docs/blending.md](docs/blending.md): training, evaluating and applying blends.
* [docs/migrating_from_onset_blending.md](docs/migrating_from_onset_blending.md): for users of `onset_blending` and `momp`.
* `docs/package_configuration.ipynb`: the config, section by section.
* `docs/package_run_guidance_et.ipynb`: running HCWF and reading the results.
* `demo/et/`: Ethiopia demo notebooks and [companion guidance](https://docs.google.com/document/d/15_evi1VI4GcCeSoiI6DcrfidwjEE4g-UPiiOY5kF2Zo/edit?usp=sharing).
* `docs/blend_merge.md`, `docs/review/`: the plan and reviews of the merge and restructuring (historical).

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
