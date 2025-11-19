import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import Polygon


def round_to(value: float, decimals: int = 0) -> float:
    """
    Rounds 'value' to 'decimals' places.

    Parameters
    -----------
    - value (float): Number to round.
    - decimals (int): Decimal places to round to, default 0.

    Returns:
    ----------
    - float: Rounded number.
    """
    factor = 10**decimals
    return np.round(value * factor) / factor


def add_LAT_LON_ref_columns(tracks_proj) -> pd.DataFrame:
    """
    Adds reference columns 'LAT_ref' and 'LON_ref' to the tracks_proj DataFrame for alignment with SEDAC projections.
    Reference columns are calculated to the nearest multiples of 0.125 for latitude and 0.25 for longitude to facilitate
    merging with SEDAC's gridded data.

    Parameters
    -----------
    - tracks_proj : pd.DataFrame
        DataFrame containing 'LAT' and 'LON' columns with latitude and longitude values.

    Returns
    --------
    - pd.DataFrame
        The input DataFrame augmented with 'LAT_ref' and 'LON_ref' columns.

    Notes
    ----------
    The function assumes the SEDAC grid increments: latitude by 0.125 and longitude by 0.25 degrees.

    """
    # Check if all LON values are within the range -180 to 180
    if not tracks_proj["LON"].between(-180, 180).all():
        raise ValueError("Not all LON values are within the range -180 to 180.")

    # Check if all LAT values are within the range -55.875 to 83.625
    if not tracks_proj["LAT"].between(-55.875, 83.625).all():
        raise ValueError("Not all LAT values are within the range -55.875 to 83.625.")

    # Calculate LON_ref based on the nearest multiple of 0.25
    tracks_proj["LON_ref"] = ((tracks_proj["LON"] + 180) // 0.25) * 0.25 - 180

    # Calculate LAT_ref based on the nearest multiple of 0.125
    tracks_proj["LAT_ref"] = ((tracks_proj["LAT"] - 83.625) // 0.125) * 0.125 + 83.625

    return tracks_proj


def process_tracks(tracks_proj: pd.DataFrame, is_scaled=True) -> pd.DataFrame:
    """
    Preprocesses storm track data by adjusting geographic coordinates for grid alignment,
    converting time data to datetime objects, rounding years, and handling wind speed data.

    Parameters
    -----------
    - tracks_proj : DataFrame
        Storm track data with geographic and time information.
    - is_scaled : bool, default True
        Indicates if wind speed data is scaled.

    Returns
    --------
    - DataFrame
        The processed storm track data ready for analysis.
    """
    # Round latitude and longitude to the nearest 1/8 degree
    # tracks_proj["LAT_ref"] = tracks_proj["LAT"].apply(lambda x: round_to(float(x), 3))
    # tracks_proj["LON_ref"] = tracks_proj["LON"].apply(lambda x: round_to(float(x), 3))
    tracks_proj["ISO_TIME_POSIXct"] = pd.to_datetime(tracks_proj["ISO_TIME_POSIXct"])

    tracks_proj = add_LAT_LON_ref_columns(tracks_proj)
    # Convert ISO_TIME_POSIXct to datetime format and round year to the nearest 10
    tracks_proj["ISO_TIME_POSIXct"] = pd.to_datetime(
        tracks_proj["ISO_TIME_POSIXct"], unit="s"
    )
    tracks_proj["Year_ref"] = tracks_proj["ISO_TIME_POSIXct"].dt.year.apply(
        lambda x: round_to(x, -1)
    )

    # Convert wind speeds to numeric format
    if is_scaled:
        tracks_proj["wind_scaled"] = pd.to_numeric(
            tracks_proj["wind_scaled"], errors="coerce"
        )
    else:
        tracks_proj["wind"] = pd.to_numeric(tracks_proj["wind"], errors="coerce")

    return tracks_proj


def get_iiasa_region(catherina_fit_path: Path, tracks_proj) -> pd.DataFrame:
    """
    Maps ISO3 country codes in storm track data to IIASA SSP R32 regions. Supports fetching region mapping data
    from a local database or an S3 bucket.

    Parameters
    -----------
    - tracks_proj : DataFrame
        Storm track data containing 'admin' field with ISO3 country codes.
    - mode : str, default 'S3'
        Mode of data retrieval for region mapping: 'local' for local database, 'S3' for S3 bucket.

    Returns
    --------
    - DataFrame
        Storm track data with an additional column mapping each 'admin' value to its corresponding IIASA region.

    Notes
    -----
    The function modifies specific region codes for compatibility and ensures that only unique mappings are used.
    The 'admin' field is assumed to contain ISO3 country codes.
    """

    # load local catherina db from zenodo original repo
    with sqlite3.connect(catherina_fit_path) as conn:
        region_admin = pd.read_sql_query("SELECT * FROM IIASA_ISO3_region_admin", conn)

    region_admin["REGION"] = region_admin["REGION"].str.replace("R32ANUZ", "R32AUNZ")

    region_admin["REGION"] = region_admin["REGION"].str.replace("R32TWN", "R32CHN")

    region_admin = region_admin.drop_duplicates()
    tracks_proj = tracks_proj[tracks_proj["admin"].isin(region_admin["admin"])]
    tracks_proj = pd.merge(tracks_proj, region_admin, on="admin", how="left")
    return tracks_proj


# Already Coded in pyDistGeo.py
def get_coastlines(catherina_fit_path: Path, global_coastline_path: Path):
    """
    Retrieve coastline data and intersect it with basin geometries.

    This function loads basin coordinates and global coastline data, and then
    intersects these to find the coastlines limited to specified basins. It
    supports fetching data from either a local file system or from an S3 bucket,
    based on the specified mode.

    Parameters
    ----------
    mode : str, optional
        The mode of data retrieval. It can be "local" for local file system
        access or any other string (default is "S3") for fetching data from
        an S3 bucket.
    global_coastline_path: "data/input/coastlines/ne_10m_coastline.shp"

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the intersected geometries of coastlines
        and basin polygons.

    Notes
    -----
    - The function assumes the presence of specific files (like 'Catherina_fit.db'
      and 'ne_10m_coastline.shp') and their specific formats.
    - The Coordinate Reference System (CRS) used is WGS84.

    """
    # Define the CRS (Coordinate Reference System)
    wgs84 = CRS.from_string("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

    basins_coords = gpd.read_file(catherina_fit_path, layer="basins_coords")

    basins_coords["geometry"] = gpd.points_from_xy(
        basins_coords["Easting"], basins_coords["Northing"], crs=wgs84
    )

    aggregated_basins = (
        basins_coords.groupby("Basin")["geometry"]
        .apply(lambda x: Polygon(x.tolist()))
        .reset_index()
    )
    aggregated_basins_gdf = gpd.GeoDataFrame(
        aggregated_basins, geometry="geometry", crs=wgs84
    )

    # Load the global coastline data and reproject it to WGS84
    GlobalCoastline = gpd.read_file(global_coastline_path)

    GlobalCoastline = GlobalCoastline.to_crs(wgs84)

    # Perform the intersection of coastlines with basin polygons
    coastlines_lim = gpd.overlay(
        GlobalCoastline, aggregated_basins_gdf, how="intersection"
    )

    return coastlines_lim
