# HCWF demo notebooks

Four notebooks walk through the workflow and interface of `hcwf`: config, `hcwf.check`, runs, result objects, figures, and (for blending) training and applying a blend.

| Notebook | Shows |
|---|---|
| `ethiopia/ethiopia_benchmarking.ipynb` | Ethiopia (default definition `icpac`): config and check, onset definition and one explained onset, deterministic benchmark (AIFS_v2, GraphCast) and probabilistic benchmark (AIFS_ENS_v2) against climatology, result objects and figures, polygon units, caches |
| `ethiopia/ethiopia_blending.ipynb` | Ethiopia: train a blend (AIFS_v2 + GraphCast + AIFS_ENS_v2 + climatology) straight from the zarr stores, evaluation tables, `trained.blend`, a live 2025 forecast from the stores as `Forecast` (`weekly`, `weekly_table`) and a map |
| `india/india_benchmarking.ipynb` | The same for India (default definition `moron_robertson_mok`, day boundary 03 UTC labelled by end) |
| `india/india_blending.ipynb` | The same for India (blend over central and north India) |

Each runs in under a minute on a many-core machine (20-50 s measured).

## Data

The notebooks read zarr stores under one data root:

```
<data root>/ethiopia/obs.zarr          <data root>/india/obs.zarr
<data root>/ethiopia/models/{AIFS_ENS_v2,AIFS_v2,graphcast}.zarr   (same for india/)
```

The first code cell sets the data root in one variable, `DATA`: the environment variable `HCWF_DEMO_DATA` if set, else the nearest `data/` folder above the folder the notebook runs in (the repository's `data/`).

```bash
export HCWF_DEMO_DATA=/path/to/data     # then start jupyter from any folder
```

## Running

Install hcwf (`pip install ".[dev]"` from the repository root, which brings Jupyter), then open the notebooks and run all cells. Runs write into `runs/` next to the notebook (ignored by version control, overwritten on a re-run; delete it to start clean). Everything is read from the zarr stores directly: the benchmarks (also on polygon units), the blend training and the live forecast.

Kenya notebooks will use the `liebmann_marengo` onset definition once Kenya data is available.
