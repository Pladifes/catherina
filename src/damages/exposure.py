import os
import warnings

import geopandas as gpd
from loguru import logger
import pandas as pd
from pyproj import CRS
from shapely.geometry import box
from tqdm import tqdm


def exposure_along_tracks(
    tracks_proj: pd.DataFrame,
    sector: str,
    wealth_density_path: str = "data/input/densities/",
    buffer_dist=0.3,
) -> pd.DataFrame:
    """
    Computes potential exposure to cyclones by evaluating cyclone track data with LitPop asset values within a specified buffer zone.
    Optimizes memory usage by reading only necessary columns, performing spatial joins with indexed geodataframes, and
    reducing data size prior to computationally intensive operations. Supports both local and S3 data retrieval modes.

    Parameters
    -----------
    - tracks_proj : pd.DataFrame
        Cyclone track data with 'iso3', 'LAT', 'LON', 'SID', and 'seed' columns.
    - wealth_density_path : str
        Base path for LitPop data files.
    - buffer_dist : float, optional
        Buffer distance for exposure calculation, defaults to 0.3 degrees.

    Returns
    --------
    - pd.DataFrame
        Dataframe with original track data and added 'exposed.cap' column for exposure values.
    """
    seedmax, ISO3, tracks_proj_rtn, wgs84, tracks_proj_aggregated = expo_set_up(
        tracks_proj, wealth_density_path
    )
    # TODO: optimize bottleneck
    for i in tqdm(range(len(ISO3)), desc="Computing exposure within countries"):
        iso3_code = ISO3[i]

        # For faster computations we only take 4 columns needed to perform the buffering
        tracks_proj_int = tracks_proj.loc[
            tracks_proj["iso3"] == iso3_code, ["LAT", "LON", "SID", "seed"]
        ].reset_index(drop=True)

        if tracks_proj_int.shape[0] < 10:
            continue

        for col in ["LAT", "LON"]:
            tracks_proj_int[col] = pd.to_numeric(tracks_proj_int[col], errors="coerce")

        # Creating GeoDataFrame
        tracks_proj_int_gpd = gpd.GeoDataFrame(
            tracks_proj_int,
            geometry=gpd.points_from_xy(tracks_proj_int["LON"], tracks_proj_int["LAT"]),
            crs=wgs84,
        )

        csv_file_path = (
            f"{wealth_density_path}/{sector}/sectoral_density_{iso3_code}.csv"
        )
        if not os.path.exists(csv_file_path):
            logger.info(f"{csv_file_path} does not exist. Skipping...")
            continue
        expo = pd.read_csv(
            csv_file_path, usecols=["value", "latitude", "longitude"], dtype=float
        )
        # Now we can work with 'expo' DataFrame
        expo_gpd = gpd.GeoDataFrame(
            expo,
            geometry=gpd.points_from_xy(expo["longitude"], expo["latitude"]),
            crs=wgs84,
        )

        for chunk in range(1, seedmax + 1):
            tracks_proj_chunk = tracks_proj_int_gpd[
                tracks_proj_int_gpd["seed"] == chunk
            ].copy()

            if tracks_proj_chunk.empty:
                continue  # Skip the rest of the loop for empty data

            # Buffering
            # Suppress specific warning about geographic CRS
            warnings.filterwarnings(
                "ignore",
                message=".*Geometry is in a geographic CRS.*",
                category=UserWarning,
            )
            # The warning emerges from the fact that buffering operations on geometries in a geographic CRS
            # (which uses degrees) do not return exact values, as the calculation assumes a flat,
            # two-dimensional plane, rather than the Earth's curved surface. However, this may not be a
            # significant issue for this specific case, as the introduced error is relatively small.

            tracks_proj_chunk.loc[:, "geometry"] = tracks_proj_chunk.buffer(buffer_dist)

            # Reduce data size before spatial join
            bounds = tracks_proj_chunk.total_bounds
            expo_gpd_reduced = expo_gpd[expo_gpd.within(box(*bounds))]

            if expo_gpd_reduced.empty:
                continue  # No overlapping areas, move to next chunk

            # Use spatial indexing for faster spatial operations
            expo_gpd_reduced = expo_gpd[expo_gpd.within(box(*bounds))].copy()
            expo_gpd_reduced["sindex"] = expo_gpd_reduced.sindex

            # Spatial join and aggregation
            sjoin = gpd.sjoin(
                expo_gpd_reduced, tracks_proj_chunk, how="inner", predicate="within"
            )
            aggregated_values = (
                sjoin.groupby("index_right")["value"]
                .sum()
                .reset_index(name="exposed.cap")
            )

            # If this is our first chunk, we initialize our final dataframe with it. Otherwise, we concatenate.
            if "tracks_proj_aggregated" not in locals():
                tracks_proj_aggregated = (
                    tracks_proj_chunk.merge(
                        aggregated_values,
                        left_index=True,
                        right_on="index_right",
                        how="left",
                    )
                    .fillna(0)
                    .drop(columns=["index_right"])
                    .reset_index(drop=True)
                )
            else:
                more_aggregated = (
                    tracks_proj_chunk.merge(
                        aggregated_values,
                        left_index=True,
                        right_on="index_right",
                        how="left",
                    )
                    .fillna(0)
                    .drop(columns=["index_right"])
                    .reset_index(drop=True)
                )

                tracks_proj_aggregated = pd.concat(
                    [tracks_proj_aggregated, more_aggregated], ignore_index=True
                )  # Concatenating with index reset for efficiency

        # Now we want to make sure we carry over this information to the next iso3 code processing
        tracks_proj_int = tracks_proj_int.merge(
            tracks_proj_aggregated[["SID", "LAT", "LON", "seed", "exposed.cap"]],
            on=["SID", "LAT", "LON", "seed"],
            how="left",
        )

        tracks_proj_rtn = pd.concat(
            [tracks_proj_rtn, tracks_proj_int], ignore_index=True
        )

    # Add the rest of the columns from tracks_proj
    tracks_proj_rtn = tracks_proj_rtn.merge(
        tracks_proj, on=["LAT", "LON", "SID", "seed"], how="left"
    )

    return tracks_proj_rtn


# TODO: delete
def expo_set_up(
    tracks_proj: pd.DataFrame,
    wealth_density_path: str = "data/input/densities/",
):
    """
    Initializes variables for exposure computation with track projections and LitPop data.

    Parameters
    -----------
    - tracks_proj : pd.DataFrame
        Data with 'seed' and 'iso3' columns for track projections.
    - wealth_density_path : str
        File path for the LitPop data, defaults to ``"densities/"``.

    Returns
    --------
    - tuple
        Contains max seed value, unique ISO3 codes, empty DataFrame, WGS84 CRS, empty DataFrame for aggregated tracks.

    Raises
    ------
    TypeError
        If inputs are not a DataFrame for tracks_proj or a string for wealth_density_path.
    """
    seedmax = max(tracks_proj["seed"])

    ISO3 = tracks_proj["iso3"].dropna().unique()
    tracks_proj_rtn = (
        pd.DataFrame()
    )  # Initializing as an empty DataFrame for better practice
    wgs84 = CRS.from_string("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    # The tracks_proj_aggregated DataFrame is declared here before the loops start
    tracks_proj_aggregated = pd.DataFrame()
    return seedmax, ISO3, tracks_proj_rtn, wgs84, tracks_proj_aggregated
