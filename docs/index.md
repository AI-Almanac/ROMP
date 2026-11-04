# HCWF: Human-Centered Weather Forecasts

HCWF detects **rainy season onset** in observations and forecasts, benchmarks the forecasts (deterministic and
probabilistic metrics, skill scores, maps, reliability diagrams) and trains and applies **onset blends** that turn
forecasts and climatology into onset probabilities. Everything is driven by one YAML run file, on grid cells, polygons
or political boundaries, for any region.

Onset is detected by the `hcwf.onset` component. An onset definition is a short YAML file (a registry definition or your own), run
identically on observations and on every forecast.

HCWF absorbed the former `onset_blending` repository and was called `momp` / ROMP before. If you used either, read
{doc}`migrating_from_onset_blending`.

## Quick start

```bash
pip install .
hcwf-run init --obs /path/to/obs --model /path/to/forecasts --years <first> <last> -o run.yaml
hcwf-run -p run.yaml --check     # validate, resolve and print the plan; runs nothing
hcwf-run -p run.yaml             # run it
```

The same run from Python:

```python
import hcwf
result = hcwf.run("run.yaml", max_forecast_day=15)
plan = hcwf.check("run.yaml")
```

See {doc}`howto/run_benchmark` for a first run, {py:obj}`hcwf.run` and {py:obj}`hcwf.check` for the Python interface and the
repository README for run folders and caches.

The documentation is organised by purpose:

* **Tutorials**: worked notebooks that run HCWF end to end on sample data.
* **How-to guides**: one task each: configure a run, add a dataset, run a benchmark, train and apply a blend, choose or override an onset definition.
* **Explanation**: what the onset definitions, the scores, the reference forecasts and the blending method mean.
* **Reference**: the Python interface, the `hcwf-run` command line, every run-file key, and the public names of every module.
* **Project notes**: migration notes, contributing conventions and the records of the merge.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Learn

tutorials/index
howto/index
explanation/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Look up

reference/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Project

project/index
```
