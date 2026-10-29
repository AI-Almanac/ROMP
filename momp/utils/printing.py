"""Helpers for formatting tuples as strings and printing banners, config values and data-file info."""
import logging
from importlib.metadata import version as pkg_version

def combi_to_str(combi, sep="_", tuple_sep="-", suffix=""):
    """
    Convert a tuple like ('a', (1,15), 'X') into a string tag:
    - Strings or numbers → as is
    - Tuples → joined with `tuple_sep`
    """
    parts = []
    for item in combi:
        if isinstance(item, tuple):
            # join tuple elements with tuple_sep
            parts.append(tuple_sep.join(map(str, item)))
        else:
            parts.append(str(item))
    return sep.join(parts) + suffix


def tuple_to_str(item):
    """Join the elements of a tuple with "-"."""
    return "-".join(map(str, item))


def tuple_to_str_range(item):
    """ ignore all middle elements and only join the first and last items of a tuple """
    if len(item) == 0:
        return ""
    elif len(item) == 1:
        return str(item[0])
    else:
        return f"{item[0]}-{item[-1]}"



def print_momp_banner(cfg):
    """Print the ROMP start-up banner with package version, project name and start time."""
    version = pkg_version("momp")
    project_name = cfg.project_name

    banner = fr"""
================================================================================
  ____    ___   __  __  ____
 |  _ \  / _ \ |  \/  ||  _ \
 | |_) || | | || |\/| || |_) |
 |  _ < | |_| || |  | ||  __/
 |_| \_\ \___/ |_|  |_||_|

 Rainy season Onset Metrics Package (ROMP)
 Version : {version}

--------------------------------------------------------------------------------
 Project    : {project_name}
 Start Time : {__import__('datetime').datetime.now().isoformat(timespec='seconds')}
--------------------------------------------------------------------------------

 Initializing analysis pipeline...
================================================================================
"""
    logging.getLogger(__name__).info(banner)


def print_cfg(cfg, key_patterns):
    """
    Print key-value pairs from config where the key matches one or more patterns.

    Parameters:
        cfg (SimpleNameSpace): SimpleNameSpace object
        key_patterns (str or list/tuple of str): Pattern(s) to match in keys.
    """
    # Ensure key_patterns is a list/tuple
    if isinstance(key_patterns, str):
        key_patterns = [key_patterns]

    for key, value in vars(cfg).items():
        # Check if any pattern matches (endswith or contained)
        if any(key.endswith(p) or p in key for p in key_patterns):
            print(f"{key}: {value}")


def print_data_info(data_dir, pattern="*.nc"):
    """Print the file name, dimensions and variables of the first file in `data_dir` matching `pattern`."""
    import xarray as xr

    # List all .nc files
    nc_files = list(data_dir.glob(pattern))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {data_dir}")

    # Pick the first file
    first_file = nc_files[0]

    # Open with xarray
    ds = xr.open_dataset(first_file)

    # Nicely print information
    print(f"\nFile: {first_file.name}")
    print("\nDimensions:")
    for dim, size in ds.sizes.items():
        print(f"  {dim}: {size}")

    print("\nVariables:")
    for var in ds.data_vars:
        print(f"  {var}")

    # Optional: close dataset if you don't need it open
    ds.close()
