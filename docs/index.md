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

See {py:obj}`hcwf.run`, {py:obj}`hcwf.check` and the {doc}`api` for the Python interface, and the repository README for the command
reference, the run file sections, run folders and caches.

```{toctree}
:maxdepth: 2
:caption: Guides

blending
onset_in_hcwf
onset_spec
reading_data
migrating_from_onset_blending
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

package_configuration
package_run_guidance_et
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Contributing

contributing_docstrings
```
