"""Configuration for the heatwave analysis pipeline.

Pipeline role:
- Provide declarative settings used across the package.

Responsibilities:
- Define file paths and output paths.
- Define region definitions and seasons.
- Store source-specific constants.
- Store default analysis settings and optional run presets.

Out of scope:
- Data loading.
- Scientific computations.
- Runtime analysis logic.
"""

ARCO_PATH  = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_TOKEN = "anon"
ARCO_OPEN_MAX_ATTEMPTS: int = 4
ARCO_OPEN_RETRY_BASE_DELAY_SECONDS: float = 15.0
G_M_S2: float = 9.806
CP_J_KG_K: float = 1005.0
EARTH_RADIUS_M: float = 6.371e6

REGIONS: dict[str, tuple[slice, slice]] = {
    "pnw_bartusek": (slice(40, 60), slice(-130.0, -110.0)),
}

#standard grid, daily
# CANESM_LWA_ROOT: str   = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/CanESM5/historical"
ERA5_LWA_ROOT: str     = "/home/mhpereir/data-mhpereir/LWA_calculation/outputs/ERA5"
#/z500/LWA_day_ERA5_2deg.500.nc
'''
dimensions
time (daily)
lat, lon (standard grid, northern hemisphere only, 2deg resolution)

variables
LWA (time, lat, lon) - daily LWA values on the standard grid
LWA_a (time, lat, lon)
LWA_c (time, lat, lon)
'''


#standard grid, daily
# CANESM_TAS_ROOT: str   = "/home/mhpereir/data-mhpereir/standard_grid_daily/CMIP6/CanESM5/tas/historical"
ERA5_TAS_ROOT: str     = "/home/mhpereir/data-mhpereir/standard_grid_daily/REANALYSIS/ERA5/tas"
#/tas_daily_ERA_{year: 1940-2024}_2x2_bil.nc
'''
dimensions
valid_time (daily) - includes leap days, so 365 or 366 time steps per year
lat, lon (standard grid, global, 2deg resolution)
bnds - can be dropped, not needed for analysis

variables
tas (valid_time, lat, lon) - daily 2m air temperature on the standard grid
'''

#spatially aggregated on standard grid, daily
LWA_THRESH_ROOT: str   = "/home/mhpereir/data-mhpereir/LWA_thresholds/outputs"
#/{region}/{dataset: ERA5 or CanESM5}/q{quantile}/{dataset}_LWAthresh_{method}_1970_2014_q{quantile}_{region}.500.nc
#/pnw_bartusek/ERA5/q95/ERA5_LWAthresh_block_1970_2014_q95_pnw_bartusek.500.nc
'''
dimensions
dayofyear (1-366)

variables
LWA (dayofyear) - LWA threshold for each day of year, spatially aggregated over the region of interest.
LWA_a (dayofyear)
LWA_c (dayofyear)
'''

#spatially aggregated on standard grid, daily
HW_THRESH_ROOT: str    = "/home/mhpereir/HW_thresholds/outputs"
#/{region}/{dataset: ERA5 or CanESM5}/{method: block or evolving}/q{quantile}/{dataset}_HWthresh_{method}_1970_2014_tas_q{quantile}_{region}_{ensemble_member}.nc
#/pnw_bartusek/ERA5/evolving/q95/ERA5_HWthresh_evolving_1950_2024_tas_q95_pnw_bartusek.nc
#ERA5 doesn't have "ensemble_member" dimension, so that part of the filename is omitted for ERA5 files.
'''
# FOR EVOLVING METHOD,
dimensions
year (1950-2024)
dayofyear (1-366)

variables
threshold (year, dayofyear) - daily HW threshold for each day of year and year, spatially aggregated over the region of interest.
climatology (year, dayofyear) - daily climatology for each day of year and year, spatially aggregated over the region of interest.

# FOR BLOCK METHOD (IGNORE FOR NOW... WILL CHANGE NAMING CONVENTION TO MATCH EVOLVING METHOD),
dimensions
dayofyear (1-366)

variables
tas_thresh_p95_win31 (dayofyear) - daily HW threshold for each day of year, spatially aggregated
'''

#eulerian heat budget, spatially aggregated from ERA5 native grid, hourly
ERA5_HEAT_BUDGET_ROOT: str = "/home/mhpereir/eulerian_heat_budget/results/production/pnw_full_run/annual"
#heat_budget_{year}.nc
'''
dimensions
time (hourly, 4414 time steps per year May 1 - September 30)

variables
time(time) - hours since 1900-01-01 00:00:00
d_dt_T (time)
dT_dt (time)
dT_dt_2 (time)
dV_dt (time)
advective_error (time)
adiabatic_term (time)
diabatic_term (time)
T_domain_avg (time)
domain_volume (time)
T_scale int
advection_term (time)
net_mass_advection (time)
flux_contribution_west (time)
flux_contribution_east (time)
flux_contribution_north (time)
flux_contribution_south (time)
flux_contribution_top (time)
mass_flux_contribution_west (time)
mass_flux_contribution_east (time)
mass_flux_contribution_north (time)
mass_flux_contribution_south (time)
mass_flux_contribution_top (time)
abs_mass_advection_residual_fraction (time) - diagnostic quantity, not necessary
'''

# locally stored hourly ARCO/ERA5 full diagnostics
ERA5_PBL_P_ROOT: str = "/home/mhpereir/data-mhpereir/arco_era5/PBL_download/outputs"
ERA5_CLOUD_COVER_ROOT: str = "/home/mhpereir/data-mhpereir/arco_era5/CloudCover_download/outputs"
ERA5_HOURLY_SURFACE_ROOT: str = "/home/mhpereir/downloads-mhpereir/REANALYSIS/ERA5/hourly"
ERA5_NSLR_ROOT: str = f"{ERA5_HOURLY_SURFACE_ROOT}/nslr"
ERA5_NSSR_ROOT: str = f"{ERA5_HOURLY_SURFACE_ROOT}/nssr"
ERA5_SLHF_ROOT: str = f"{ERA5_HOURLY_SURFACE_ROOT}/slhf"
ERA5_SSHF_ROOT: str = f"{ERA5_HOURLY_SURFACE_ROOT}/sshf"
ERA5_SOIL_MOISTURE_ROOT: str = f"{ERA5_HOURLY_SURFACE_ROOT}/soil_moisture"
