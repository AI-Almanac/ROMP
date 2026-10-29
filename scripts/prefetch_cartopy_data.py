"""
Pre-download all Natural Earth datasets used by ROMP at image build time.
Run once during `docker build`; data lands in CARTOPY_DATA_DIR.
"""
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

print("Prefetching Natural Earth data for ROMP...")

# 1. 10m admin_0_countries — used by get_shp() / shp_mask() / shp_outline()
print("  -> 10m cultural/admin_0_countries")
shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')

# 2. 110m physical/coastline — used by cfeature.COASTLINE (Cartopy default resolution)
print("  -> 110m physical/coastline")
shpreader.natural_earth(resolution='110m', category='physical', name='coastline')

# 3. 110m cultural/admin_0_boundary_lines_land — used by cfeature.BORDERS
print("  -> 110m cultural/admin_0_boundary_lines_land")
shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_boundary_lines_land')

# 4. 50m datasets — used by onset_map.py climatology plots
print("  -> 50m physical/coastline")
shpreader.natural_earth(resolution='50m', category='physical', name='coastline')

print("  -> 50m cultural/admin_0_boundary_lines_land")
shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_boundary_lines_land')

# 4b. Region masks (momp.utils.mask: `mask.land`, `mask.country`) read land and admin_0_countries at the
#     resolution `mask.resolution` (default 50m), so fetch all three resolutions of both layers.
for _res in ("10m", "50m", "110m"):
    for _category, _name in (("cultural", "admin_0_countries"), ("physical", "land")):
        print(f"  -> {_res} {_category}/{_name}")
        shpreader.natural_earth(resolution=_res, category=_category, name=_name)

# 4c. Exercise cfeature.COASTLINE and cfeature.BORDERS via a throwaway figure
#    so Cartopy's feature downloader also triggers and caches the geometries.
print("  -> Exercising cfeature COASTLINE + BORDERS via throwaway plot")
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
plt.close(fig)

# 5. regionmask natural_earth_v5_0_0 land masks — used by momp land/sea masking.
#    Must be pre-cached at build time to avoid race conditions when parallel
#    workers all try to download/extract the shapefile simultaneously at runtime.
print("  -> regionmask natural_earth_v5_0_0 land_110")
import regionmask
_ = regionmask.defined_regions.natural_earth_v5_0_0.land_110
print("  -> regionmask natural_earth_v5_0_0 land_50")
_ = regionmask.defined_regions.natural_earth_v5_0_0.land_50
print("  -> regionmask natural_earth_v5_0_0 land_10")
_ = regionmask.defined_regions.natural_earth_v5_0_0.land_10

print("Prefetch complete.")
