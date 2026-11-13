# ROMP — Rainy season Onset Metrics Package

**ROMP** is a Python package for detecting and benchmarking **rainy season onset** in observational and forecast datasets. It provides tools for onset detection, ensemble forecast statistics, binned and spatial metrics, and visualization workflows commonly used in climate research.


## Key Capabilities

- rainy season onset detection with pluggable, parameterised definitions (`legacy`, `two_stage`, `moron_robertson`, or your own — see [docs/onset_rules.md](docs/onset_rules.md))
- deterministic and probabilistic benchmarking metrics
- Skill Scores (overall and binned)
- Spatial metrics (MAE, False Alarm Rate, Miss Rate)
- Reliability diagrams and spatial maps
- Customizable region definitions (rectangular boundary, shapefile, polygon outline)
- Config-driven, reproducible workflows


## Installation

ROMP is intended for **local installation from source** (GitHub or local checkout). Installation from conda-forge chanel will be available in it's future versions.  

Installing ROMP consists of **two steps**:

1. **Create and activate a Python environment**
2. **Install the ROMP source code into that environment**


### Step 1 — Set up a Python environment

#### Option A — Python virtual environment (pip-only)
This option is a lightweight setup which isolates project dependencies assuming the underlying operating system provides the necessary heavy lifting (doesn't duplicate system-level files).

For **Windows** users, follow the steps below to set up Python environment  

1. `python -m venv .venv-momp`
   Creates a new virtual environment directory named `.venv-mpop` in the current folder.

2. `.venv-momp/Scripts/activate.bat`
   Activates the virtual environment so that the terminal uses the local Python instance.


For **Linux/Mac** users, follow steps below:  
```bash
python -m venv .venv-momp
source .venv-momp/bin/activate
```

#### Option B — Set up Conda environment

This option is good if you work with **NetCDF, HDF5, or other system-level scientific libraries**  

1. `conda create -n momp "python>=3.10"`
Create a New Conda Environment

2. `conda activate momp`
Activate the environment:


### Step 2 — Install the package from source

#### Clone from GitHub and install

with python or conda env activate from step 1, clone the source code from package repository    

```bash
git clone https://github.com/bosup/ROMP.git
cd momp
```

For **Windows** users,  


`python -m pip install -U pip` Upgrades the `pip` package manager to the latest version to ensure compatibility.  

`pip install .` Installs the package with all project dependencies. Python is isolated at this point.   

For **Mac/Linux** users,  

```
pip install -U pip
pip install .
```

## Verify installation
`python -c "import momp; print(momp.__file__)"`

You should see a path pointing to your **source directory**

## Configuration
Experiment configuration is controlled via:  
`params/config.in`

This file defines:
- Input and output directories
- Dataset selection
- Ensemble definitions
- Verification windows
- Region settings

Region boundaries are defined in:  
`params/region_def.py`

## Run ROMP
With user-defined `config.in`, the main benchmarking workflow is executed via CLI::  

`momp-run`

Typical steps performed:
1. Load configuration
2. Read model and observation data
3. Detect rainy season onset
4. Evaluate model against reference data 
5. Generate benchmarking metrics
6. Save NetCDF outputs and figures
7. Make metric plots

## Python Requirements
- Python ≥ 3.10  

Runtime dependencies include:
- NumPy
- Pandas
- Xarray
- NetCDF4
- Matplotlib
- Scipy
- geopandas
- seaborn
- regionmask
- gcsfs
- zarr
- cartopy


## Package Organization (high level)
- driver.py — main package workflow entry point
- app/ — high-level benchmarking workflow
- stats/ — onset detection and statistical processing
- metrics/ — error and skill score metrics calculation
- params/ — configuration files and region definitions
- lib/ — core workflow control, parsing, conventions
- io/ — input/output handling
- graphics/ — plotting and visualization
- utils/ — shared helper utilities

## Outputs
Results are written to (default or user specified dirs):

`output/` — NetCDF and serialized metric files  
`figure/` — generated plots and maps

## Development Notes
Install in editable mode for development:
`pip install -e .`  

Code is organized to separate:
- I/O
- statistics
- metrics
- visualization

## Versioning
Semantic versioning is used (MAJOR.MINOR.PATCH)  
Current version: 0.0.1  
APIs may evolve during development

## License
MIT License

## Citation
If you use ROMP in your research, please cite:
ROMP: Rainy season Onset Metrics Package, UChicago HCWF Authors, 2026

## Contact
Author: bosup  
Email: bodong@uchicago.edu



