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

1. Copy the template next to your project and edit it (paths, models, region, windows):

   ```bash
   cp examples/config.in my_config.in
   ```

2. Run, always naming the config:

   ```bash
   momp-run -p my_config.in
   ```

   In a notebook, set `MOMP_CONFIG=/path/to/my_config.in` instead. Without `-p` or `MOMP_CONFIG` ROMP stops with a message. Many config keys can also be overridden on the command line (`momp-run -p my_config.in -h` lists them).

Each run loads the config, reads observations and forecasts, detects onset, scores the forecasts against the observed onset and a reference (climatology by default), then writes NetCDF/CSV results to `dir_out` and figures to `dir_fig`.

`examples/config.in` is commented key by key; `docs/package_configuration.ipynb` walks through every section. Onset settings are in section 6 and need not be filled in for a known region: they default from `region`.

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

Choose the definition in section 6 of the config:

```python
region = "Ethiopia"            # known region: its default definition and obs day boundary apply
onset_definition = "icpac"     # optional: registry id or path to a YAML file
```

To change a rule slightly, write a YAML file that extends a registry definition and restates only what differs:

```yaml
# my_icpac.yaml
extends: icpac
id: my_icpac
version: 1
title: ICPAC with a 25 mm trigger
status: draft
onset:
  require:
    - first day: wet
    - total: more than 25 mm
```

then set `onset_definition = "my_icpac.yaml"`. Regions are YAML files in `momp/onset/regions/`. A custom region you define in `momp/params/region_def.py` has no default definition, so set `onset_definition` and `obs_day_boundary` yourself.

Old configs with flat keys (`wet_init`, `wet_spell`, `dry_extent`, ...) still run in a legacy mode with a warning; they cannot be combined with `onset_definition`.

More:

* `docs/onset_in_romp.md`: how ROMP uses the component (config keys, lead windows, forecast history, thresholds, legacy mode).
* `docs/onset_spec.md`: the specification (YAML grammar, detection semantics, backends).

## Package layout

```
momp/
  driver.py     momp-run entry point
  onset/        onset component: definitions, registry, regions, ingestion, backends, explain, viz
  app/          benchmarking workflow and diagnostics
  stats/        ROMP's onset adapter, ensembles, climatology
  metrics/      error and skill scores
  io/           input and output
  lib/          config loading, validation, lead windows
  graphics/     plots and maps
  params/       region_def.py (custom regions)
  utils/        helpers
examples/config.in   config template
docs/                guides and specification
demo/                demo data and notebooks
tests/
```

## Outputs

* `dir_out`: spatial-metric NetCDFs, skill-score CSVs, and `.onset_cache/` (thresholds).
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
