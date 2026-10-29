# ROMP: Rainy season Onset Metrics Package

ROMP detects **rainy season onset** in observations and forecasts and benchmarks the forecasts: deterministic and probabilistic metrics, skill scores, spatial maps and reliability diagrams. Runs are driven by one config file.

Onset detection is done by the `momp.onset` component. An onset definition is a short YAML file (for example ICPAC's), run identically on observations and on every forecast.

## Install

Python 3.10 to 3.13. From a clone of the repository, in a fresh virtual or conda environment:

```bash
git clone https://github.com/AI-Almanac/ROMP.git
cd ROMP
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .                                       # or: pip install -e ".[dev]"
```

Optional extras:

* `pip install ".[gpu]"` adds CuPy for `onset_backend = "gpu"` (CUDA 13 build).
* `.[dev]` adds pytest, jupyterlab, black, ruff and others.

Check it: `python -c "import momp; print(momp.__file__)"` should print a path in your checkout.

## Quick start

One YAML file describes a run and is the single source of truth. Start one from your own data; ROMP detects the dataset details (variable, units, dims, members, file pattern, lead label) from the files and writes them with the evidence:

```bash
momp-run init --obs /path/to/obs --model /path/to/aifs --years 2019 2024 -o run.yaml
```

Review `run.yaml` (look for `TODO`), then run it:

```bash
momp-run -p run.yaml          # or set MOMP_CONFIG=run.yaml, e.g. in a notebook
momp-run datasets -p run.yaml # what each dataset resolves to, with detections, warnings, problems
momp-run -p run.yaml -h       # lists the command-line overrides of config values
```

`momp-run -h` lists all commands. `examples/config.yaml` is a short commented example. `momp-run init` writes only what you must decide; the region supplies the onset definition, day boundary and climatology years, and schema defaults fill the rest. Defaults are resolved at run time and never copied into your file. The sections `observations`, `models`, `onset`, `evaluation`, `metrics` and `output` flatten to the flat keys the code uses; every key, type, default and help line is in `momp/lib/config_schema.py`. Unknown keys are errors with "did you mean" suggestions.

Datasets are described inline (`path`, `variable`, `units: mm|m`, `dims`, `day_boundary` for observations, `lead_label` and `accumulation` for forecasts). Left-out keys are detected from the first file and reported; a key that cannot be detected reliably, or that the files contradict, is an error naming it. Paths may use `~` and `${ENV_VAR}`. `momp-run init --dataset-template obs|forecast` prints a blank entry; entries can also live in your own files listed under `dataset_files:` and be referenced with `dataset: <id>` (inline keys win).

In a notebook use Python objects instead: `from momp.config import Config, Dataset, Onset`; `Config(...).to_flat()` gives the validated flat dict and `.to_yaml(path)` writes the file.

Every run writes `resolved_config.yaml` (every value after defaults, region, detection and overrides, each with its source) and `resolved_definition.yaml` (with the hash) into `dir_out`; `momp-run -p resolved_config.yaml` reproduces the run. Old `config.in` files are no longer read: `momp-run convert old.in -o run.yaml` writes the equivalent YAML (same outputs).

Each run loads the config, reads observations and forecasts, detects onset, scores the forecasts against the observed onset and a reference (climatology by default), then writes NetCDF/CSV results to `dir_out` and figures to `dir_fig`.

Export the gridded onset dates (observed, each model, the climatology references) as NetCDF with `momp-run export -p run.yaml` or `output.save_onsets: true`; see "Exporting onset dates" in `docs/onset_in_romp.md`.

## Onset definitions

List, read and draw definitions from the command line:

```bash
python -m momp.onset list                     # ids, titles, regions
python -m momp.onset describe icpac           # the rule in words, hash, status, notes
python -m momp.onset regions                  # regions and countries with their defaults
python -m momp.onset schematic icpac --png icpac.png
python -m momp.onset explain --obs obs.nc --obs-var RAINFALL \
    --day-boundary "00 UTC, labelled by start" --definition icpac \
    --lat 9.0 --lon 38.75 --year 2020 --png anatomy.png
python -m momp.onset check-alignment --help   # are obs and forecast dates shifted?
```

`explain` prints why one cell got its onset date (each candidate day, which condition failed). The `momp.onset.viz` package also has Python functions for comparing definitions, forecast-versus-truth plots and map drill-downs.

Choose the definition in the `onset:` section of the run file, inline:

```yaml
region: Ethiopia               # known region: its default definition and day boundary apply
onset:
  definition: icpac            # optional: registry id or path to a definition YAML
  override: {wet_day: at least 2 mm}   # optional: change part of it (same rules as `extends`)
  # rule: {...}                # or a complete definition written inline
```

`override` is merged like `extends` and validated with the same strict errors; the run prints the definition's description, "icpac v1, overridden: wet_day" and its hash, and records them in `resolved_definition.yaml`. `dry_spell_check: false` is an override too. A path to a shared definition file still works. Regions are YAML files in `momp/onset/regions/`. A custom region in `momp/params/region_def.py` has no default definition, so set `onset.definition` and `observations.day_boundary` yourself.

Old flat onset keys (`wet_init`, `wet_spell`, `dry_extent`, ...) still run in a legacy mode with a warning (written at the top level of the YAML); they cannot be combined with `onset.definition`.

More:

* `docs/onset_in_romp.md`: how ROMP uses the component (config keys, lead windows, forecast history, thresholds, legacy mode).
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

The mask is built on each dataset's own grid (the observations and every model separately); rainfall is never regridded. A shape is rasterised by the cell rule. A mask file is aligned first (latitude order, 0-360 or -180-180 longitudes, axis names); on the same grid it is used as it is, otherwise the mask (not the rainfall) is regridded: a finer mask keeps a cell when kept mask cells cover at least `overlap_fraction` of it, a coarser one gives the nearest mask cell, and a mask that does not cover the data is an error naming both extents. The run prints the sources, rule, alignment actions and kept cell count per dataset, saves `mask_<dataset>.png` in the figure folder, and records the sources with content hashes in `resolved_config.yaml` and the NetCDF outputs. The old flat `nc_mask` key still works as before (it cannot be combined with `mask`). Natural Earth data downloads on first use; `scripts/prefetch_cartopy_data.py` fetches it for offline machines. See `docs/onset_in_romp.md`.

## Package layout

```
momp/
  driver.py     momp-run entry point
  onset/        onset component: definitions, registry, regions, ingestion, backends, explain, viz
  app/          benchmarking workflow and diagnostics
  stats/        ROMP's onset adapter, ensembles, climatology
  metrics/      error and skill scores
  io/           input and output
  lib/          config schema, YAML loading, dataset detection, validation, lead windows
  config.py     Config objects for notebooks
  cli.py        the momp-run command
  graphics/     plots and maps
  params/       region_def.py (custom regions)
  utils/        helpers
examples/config.yaml  example run config
docs/                guides and specification
demo/                demo data and notebooks
tests/
```

## Outputs

* `dir_out`: spatial-metric NetCDFs, skill-score CSVs, `resolved_config.yaml`, `resolved_definition.yaml` and `.onset_cache/` (thresholds).
* `dir_fig`: metric plots and maps; `explain/` holds anatomy plots for cells listed in `explain_cells`.

## Docs and demos

* `docs/package_configuration.ipynb`: the config, section by section.
* `docs/package_run_guidance_et.ipynb`: running ROMP and reading the results.
* `demo/et/`: Ethiopia demo notebooks and [companion guidance](https://docs.google.com/document/d/15_evi1VI4GcCeSoiI6DcrfidwjEE4g-UPiiOY5kF2Zo/edit?usp=sharing).

## Development

```bash
pip install -e ".[dev]"
pytest -q tests/
```

Current version 0.0.1; APIs may change. MIT License.

## Citation and contact

ROMP: Rainy season Onset Metrics Package, UChicago HCWF Authors, 2026.
Contact: Bo Dong, bodong@uchicago.edu
