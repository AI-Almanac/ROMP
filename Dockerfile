# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# System dependencies required by Cartopy, GEOS, PROJ, HDF5, and NetCDF4
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libhdf5-dev \
    libnetcdf-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Tell Cartopy where to store (and find) Natural Earth data at both build and runtime
ENV CARTOPY_DATA_DIR=/usr/local/share/cartopy

WORKDIR /app

# Copy package metadata first so dependency layer is cached separately from source
COPY pyproject.toml .
COPY hcwf/ hcwf/

# Install the package and gcsfs into the system Python (no venv needed in container)
RUN uv pip install --system --no-cache ".[dev]" gcsfs

# Pre-download Natural Earth datasets used by HCWF so containers run offline
COPY scripts/prefetch_cartopy_data.py scripts/prefetch_cartopy_data.py
RUN python scripts/prefetch_cartopy_data.py

# Copy remaining runtime scripts
COPY scripts/ scripts/

# The container writes its run config from HCWF_* variables (scripts/generate_config.py) and prints which onset
# definition it chose. Without HCWF_ONSET_DEFINITION that is the region's default (HCWF_REGION, default Ethiopia:
# ICPAC). Earlier images defaulted to a flat rule (1 mm wet days, a 3-day spell above 20 mm, a 7-day dry spell) and read
# HCWF_WET_*, HCWF_DRY_*, HCWF_THRESH_FILE; those variables are now refused. Keep the old rule by converting it
# (`hcwf-run convert`, docs/onset_in_hcwf.md) and pointing HCWF_ONSET_DEFINITION at the file.
ENTRYPOINT ["scripts/entrypoint.sh"]
