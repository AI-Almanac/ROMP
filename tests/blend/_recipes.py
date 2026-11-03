"""Test helpers: an onset_blending config (the mapping its YAML holds) goes through `hcwf-run convert` and is loaded
by HCWF's normal config path, giving (flat config, Recipe, OnsetSetup) as a blend run uses them.

    loaded = converted(data)              # a run's flat config, Recipe and OnsetSetup, from a BlendConfig mapping
    loaded = partial(conversion)          # the same without reading data files (the shipped configs' data is not here)

onset_blending's own checkout is only used to read its shipped configs (ONSET_BLENDING_SRC, default
/home/hcwf/code/onset_blending); it is never modified.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from hcwf.lib.convert_specs import Conversion, convert_blend, validate_conversion

ORACLE_SRC = Path(os.environ.get("ONSET_BLENDING_SRC", "/home/hcwf/code/onset_blending"))


@dataclass
class Loaded:
    """A blend run's flat config, Recipe and OnsetSetup, and the conversion they came from."""

    cfg: dict
    recipe: object
    setup: object
    conversion: Conversion

    def rc(self, **flat):
        """The run's RunConfig, from its flat config with the keys of `flat` replaced."""
        from hcwf.config.model import RunConfig

        return RunConfig.from_flat({**self.cfg, **flat})


def oracle():
    """The onset_blending package (from its checkout), imported once: only for tests that need its own objects (its specs
    writer, its BlendConfig loader)."""
    if str(ORACLE_SRC) not in sys.path:
        sys.path.insert(0, str(ORACLE_SRC))
    import onset_blending

    return onset_blending


def blending_config(data: dict):
    """onset_blending's own BlendConfig from a mapping (to feed its oracle code)."""
    oracle()
    from onset_blending.config import BlendConfig

    return BlendConfig.from_dict(data)


def recipe_of(config):
    """The Recipe of an onset_blending BlendConfig object, through the converter (no data is read)."""
    return partial(convert_blend(config.to_dict())).recipe


def blending_dict(path) -> dict:
    """A BlendConfig YAML file as a mapping, `data_root` resolved as onset_blending resolves it (relative to the file)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    root = path.parent.resolve() / (data.get("data_root") or ".")
    return {**data, "data_root": str(root.resolve())}


def converted(data: dict, cache_dir=None, region=None) -> Loaded:
    """Convert a BlendConfig mapping, load the run YAML through the normal path (`hcwf.config.build.resolve`: the same steps as
    `hcwf-run`) and build its Recipe and OnsetSetup. Computed thresholds are kept in cache_dir (a new temporary folder
    without one)."""
    from hcwf.config.build import resolve
    from hcwf.blend.recipe import recipe_from_config
    from hcwf.lib.onset_config import onset_setup

    conversion = convert_blend(data, region=region)
    tmp = Path(tempfile.mkdtemp(prefix="blend-run-"))
    cfg = dict(resolve(conversion.mapping, write=False, tmp=tmp).values)
    cfg["cache_dir"] = str(cache_dir or tmp / "cache")
    setup = onset_setup(cfg)
    return Loaded(cfg, recipe_from_config(cfg, setup), setup, conversion)


def partial(conversion: Conversion) -> Loaded:
    """The Recipe and OnsetSetup of a conversion without reading any data file (so the shipped configs, whose data is
    not on this machine, can be loaded): `hcwf.lib.convert_specs.validate_conversion`."""
    cfg, recipe, setup = validate_conversion(conversion)
    return Loaded(cfg, recipe, setup, conversion)


def with_stub_data(mapping: dict, folder: Path) -> dict:
    """A copy of a converted run mapping whose observation and forecast datasets are small NetCDF files in `folder`, with
    the variable and dimension names the mapping states (so a shipped config can be loaded and checked without its data)."""
    import copy

    import numpy as np
    import pandas as pd
    import xarray as xr

    out = copy.deepcopy(mapping)
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)

    def write(entry, name, forecast):
        dims = {**({"init_time": "time", "step": "day", "member": "number", "lat": "lat", "lon": "lon"} if forecast
                   else {"time": "time", "lat": "lat", "lon": "lon"}), **(entry.get("dims") or {})}
        spatial = ("unit",) if "unit" in dims else ("lat", "lon")
        sizes = {"init_time": 3, "step": 12, "member": 2, "time": 20, "lat": 3, "lon": 2, "unit": 3}
        roles = (["init_time", "member", "step"] if forecast else ["time"]) + list(spatial)
        coords = {dims.get(r, r): (pd.date_range("2001-05-01", periods=sizes[r], freq="7D" if r == "init_time" else "D")
                                   if r in ("time", "init_time") else
                                   np.arange(1, sizes[r] + 1) if r == "step" else
                                   [f"u{i}" for i in range(3)] if r == "unit" else np.arange(sizes[r]) * 0.25 + 10.0)
                  for r in roles}
        data = rng.gamma(1.0, 3.0, [sizes[r] for r in roles]).astype("float32")
        path = folder / f"{name}_2001.nc"
        xr.Dataset({entry["variable"]: ([dims.get(r, r) for r in roles], data)}, coords=coords).to_netcdf(path)
        entry.pop("years", None)
        entry["paths"] = [str(path)]

    write(out["observations"], "obs", False)
    for model in out["models"]:
        write(model, model["name"], True)
    return out
