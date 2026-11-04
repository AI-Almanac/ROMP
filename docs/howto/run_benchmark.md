# Run a benchmark

A run is described by one YAML file. Start one from your data, check it, then run it.

```bash
hcwf-run init --obs /path/to/obs --model /path/to/forecasts --years <first> <last> -o run.yaml
hcwf-run -p run.yaml --check     # validate, resolve and print the plan; runs nothing (exit 1 on errors)
hcwf-run -p run.yaml             # run it (or set HCWF_CONFIG=run.yaml)
hcwf-run -p run.yaml -h          # the command-line overrides of config values
```

`--check` prints the settings with their sources, each dataset with what was detected and the evidence, the mask and
units, lead windows, which files exist, the model-versus-observation alignment, hints (each names the key to set) and the
run folder. `--quiet` shows only warnings; `--verbose` adds debug lines (`run.log` always has them).

## A run file

A minimal file (placeholders in angle brackets):

```yaml
region: <region>                       # null: no region, the whole domain of the files
observations: {path: <obs folder>, variable: <rain variable>, units: mm}
models:
  - {name: <model>, path: <forecast folder>, variable: <rain variable>, units: m}
evaluation:
  start_date: [<year>, <month>, <day>]   # the years evaluated; month and day bound the forecast issue dates
  end_date: [<year>, <month>, <day>]
  verification_windows: [[1, 15]]
  tolerances: [3]
  max_forecast_day: 15
```

Every key, with its type and default, is in the {doc}`../reference/config_keys`. Defaults are resolved at run time and
never copied into your file. Unknown keys are errors with "did you mean" suggestions.

## From Python

```python
import hcwf
result = hcwf.run("run.yaml", max_forecast_day=15)   # a path, a hcwf.config.Config or a mapping; flat-key overrides
result.observed    # observed onset dates
result.scores      # the scores
result.forecasts   # {source: Dataset} gridded onset dates of each model (computed on first use)
plan = hcwf.check("run.yaml")                        # what --check prints
```

`hcwf.run` writes no files unless `write=True` (then it behaves like `hcwf-run`; `overwrite=True` re-runs an existing run
folder). The same run can be built from objects: {py:class}`hcwf.config.Config` writes the file with `to_yaml` and the
validated flat dict with `to_flat`. See {py:func}`hcwf.run`, {py:func}`hcwf.check` and the {doc}`../reference/python_api`.

Run folders, caches and outputs are described in the repository README.
