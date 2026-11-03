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

ENTRYPOINT ["scripts/entrypoint.sh"]
