import pandas as pd
import numpy as np
import geopandas as gpd
import pathlib as Path
from shapely import MultiPolygon


Rd = 287.058  # unit: Jkg^-1K^-1
r_env = 500.0  # km


# wind pressure relationship (WPR) and its inverse
def wpr(a, mslp_hpa, pc_hpa, b):
    """ """
    return a * (mslp_hpa - pc_hpa) ** b


def wpr_inverse_pc(a, mslp_hpa, wind_speed, b):
    # Solve for pc given wind speed
    # wind_speed = a * (mslp - pc)^b
    # (wind_speed/a)^(1/b) = mslp - pc
    # pc = mslp - (wind_speed/a)^(1/b)
    
    # Avoid division by zero and negative exponents by ensuring a and b are positive
    a_safe = np.where(a <= 0, 1e-10, a)
    b_safe = np.where(b <= 0, 1e-10, b)
    
    return mslp_hpa - (wind_speed / a_safe) ** (1 / b_safe)


# max pressure drop (MPD)
def mpd(A: pd.Series, B: pd.Series, C: pd.Series, sst_celsius: pd.Series):
    T0 = 30.0
    return A + B * np.exp(C * (sst_celsius - T0))


def delta_pc(c0, c1, c2, c3, eps_P, delta_pc_shifted, pc, Y):
    return c0 + c1 * delta_pc_shifted + c2 * np.exp(-c3 * (pc - Y)) + eps_P


# max potential intensity (MPI)
def mpi(mslp_hpa, X):
    return mslp_hpa * np.exp(-X)

# Specific humidity in the eye and at environmental conditions
def qenv(rh: pd.Series, mslp_hpa: pd.Series, sst_kelvin: pd.Series) -> pd.Series:
    """
    Calculate the environmental specific humidity (qenv) based on relative humidity (rh), mean sea level pressure (mslp), and sea surface temperature (sst).

    Specific humidity (q) is the mass of water vapor per unit mass of air (kg/kg).
    Expected range: 0 < q < 0.04

    Parameters:
    - df: xarray.Dataset containing the following variables:
        - rh: relative humidity (dimensionless, 0-1)
        - mslp: mean sea level pressure (hPa)
        - sst: sea surface temperature (K)

    Returns:
        - qenv: environmental specific humidity (kg/kg)
    """
    # Scale relative humidity to ensure it's in the range 0-1 and multiply by 1000 to convert from g/kg to kg/kg
    #scaled_rh = np.clip(rh, 0, 1)*1000
    scaled_rh = rh *1000
    # Avoid division by zero by replacing zero or negative pressure values with a small positive value
    mslp_safe = np.where(mslp_hpa <= 0, 1e-10, mslp_hpa)
    
    return (3.802 * scaled_rh / mslp_safe) * np.exp(
        (17.67 * (sst_kelvin - 273.15)) / (sst_kelvin - 29.65)
    )

def qc(pc_shifted_hpa, sst_kelvin) -> pd.Series:
    """
    Calculate the specific humidity of the cyclone eye (qc) based on the previous centrale pressure level (pc_shifted) and sea surface temperature (sst).

    Parameters:
    - pc_shifted: numpy array of shifted pressure values (previous step pc)
    - sst_celsius: numpy array of sea surface temperature values
    """
    # T Celsius = T Kelvin−273.15
    RH_star = 1.0
    
    # Avoid division by zero by replacing zero or negative pressure values with a small positive value
    pc_safe = np.where(pc_shifted_hpa <= 0, 1e-10, pc_shifted_hpa)
    
    return (
        1000.0
        * RH_star
        * (3.802 / pc_safe)
        * np.exp((17.67 * (sst_kelvin - 273.15)) / (sst_kelvin - 29.65)
                 )
    )
    

def delta_Sm(mslp_hpa, pc_shifted_hpa, qc_star, q_env, sst_kelvin):
    # Latent heat of vaporization
    Lv = 2.5e6  # J/g # From thermo_MPI_func
    
    # Avoid division by zero and log of zero/negative values
    pc_safe = np.where(pc_shifted_hpa <= 0, 1e-10, pc_shifted_hpa)
    sst_safe = np.where(sst_kelvin <= 0, 273.15, sst_kelvin)  # Use 0°C as minimum
    
    return (
        Rd * sst_safe* np.log(mslp_hpa / pc_safe)
        + (Lv)* ((qc_star - q_env) / 1000) 
    )

def eps_thermo(sst_kelvin, T_tropo_kelvin):
    # Avoid division by zero by replacing zero or negative temperature values with a small positive value
    sst_safe = np.where(sst_kelvin <= 0, 273.15, sst_kelvin)  # Use 0°C as minimum
    return (sst_safe - T_tropo_kelvin) / sst_safe # thermodynamic efficiency

def coriolis(lat_degrees):
    """
    Compute the Coriolis parameter.
    Latitude is in degrees.
    Args:
        lat: numpy array of latitudes in degrees
    Returns:
        numpy array: Coriolis parameter
    """
    omega = 7.2921 * 10 ** (-5)  # rad/s # From thermo_MPI_func
    return 2 * omega * np.sin(np.radians(lat_degrees))


def X(sst_kelvin, T_tropo_kelvin, delta_Sm, lat_deg):
    eps = eps_thermo(sst_kelvin=sst_kelvin, T_tropo_kelvin=T_tropo_kelvin)
    num = eps  * delta_Sm - (
        (coriolis(lat_deg) * (r_env * 1000)) ** 2 / 4.0
    )
    # Avoid division by zero by replacing zero or negative temperature values with a small positive value
    sst_safe = np.where(sst_kelvin <= 0, 273.15, sst_kelvin)  # Use 0°C as minimum
    denom = Rd * sst_safe
    return num / denom




def get_is_on_land_fast(
        points_gs,
        _land_prepared):
    # vectorized: returns boolean ndarray
    return np.array([_land_prepared.contains(geom) for geom in points_gs], dtype=bool)

def dist_to_coast_m(gdf_4326, 
                    cost_tree,
                    coast_gdf,
                    _wgs84_to_3857):
    # nearest coast segment index for each point
    nearest_ix = cost_tree.nearest_all(gdf_4326.geometry.values)[1]  # Shapely 2: (src_idx, tgt_idx)
    nearest_geoms = coast_gdf.geometry.values[nearest_ix]

    # project to meters in one shot
    px, py = _wgs84_to_3857(gdf_4326.geometry.x.values, gdf_4326.geometry.y.values)
    nx, ny = _wgs84_to_3857(
        np.array([geom.coords[0][0] for geom in nearest_geoms]),
        np.array([geom.coords[0][1] for geom in nearest_geoms]),
    )
    return np.hypot(px - nx, py - ny)

def get_is_on_land(geometry, land: MultiPolygon) -> pd.Series:
    """
    Download land polygons here: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_land.zip
    Store zip file in data dir.

    Args:
        geometry (_type_): _description_
        ne_10m_land_zip (Path): _description_

    Returns:
        pd.Series: _description_
    """
    return geometry.within(land)


def get_dist_to_coast(
    intersect_gdf: gpd.GeoDataFrame, ne_10m_coastline_zip: Path
) -> gpd.GeoDataFrame:
    """Compute distance to nearest coastline.

    Download costlines here: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip
    Store zip file in data dir.
    Args:
        intersect_gdf (_type_): _description_
        ne_10m_coastline_zip (_type_): _description_

    Returns:
        gpd.GeoDataFrame: _description_
    """
    coastlines = gpd.read_file(ne_10m_coastline_zip)
    coastlines = coastlines.query("featurecla == 'Coastline'")
    # TODO: sort and drop duplicates to output the same shape as input
    dist2coast = gpd.sjoin_nearest(
        intersect_gdf.to_crs("EPSG: 3857"),
        coastlines.to_crs("EPSG: 3857"),
        how="left",
        distance_col="dist2coast_meters",
    ).drop(columns=["index_right", "featurecla", "scalerank", "min_zoom"])
    return dist2coast