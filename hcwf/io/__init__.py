"""Input and output: reading model and observation data, writing score tables, the dimension names, path resolution.

The modules are `hcwf.io.dims` (the one table of dimension names and the renaming helpers), `hcwf.io.stores` (where the
files and zarr stores of a year are), `hcwf.io.observations` and `hcwf.io.forecasts` (reading the data),
`hcwf.io.thresholds` and `hcwf.io.reference` (the thresholds and reference onsets), `hcwf.io.warmup` (what a run builds
once before the years go to workers), `hcwf.io.output` (writing score tables) and `hcwf.io.paths` (path resolution).
"""

__all__ = []

