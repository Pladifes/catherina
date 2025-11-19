import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import Point
from shapely.prepared import prep
from typing import Tuple
from scipy.interpolate import griddata
import random
from loguru import logger



def uniqueid():
    seed = random.getrandbits(32)
    while True:
        yield seed
        seed += 1

def read_ibtracs(fpath: Path, nature: str = "TS", signed_coords: bool = False, **kwargs):
    """
    This function loads ibtracs and apply the processing required to generate tracks candidates.
    It can be done once before looping on several seeds.
    :param mode: pick either local (read csv) or S3 (point to AWS bucket)
    :return: processed historical tracks in ibtracs to calibrate initialisation

    """
    # with open(dtypes_path, "rb") as f:
    #     dtypes = json.load(f)
    # TODO: read only necesasry cols usecols
    # TODO: slow read, try polars
    ibtracs = pd.read_csv(
        fpath,
        skiprows=[1],
        na_values={
            "USA_LAT": " ",
            "USA_LON": " ",
            "USA_WIND": " ",
            "NEWDELHI_WIND": " ",
            "CMA_WIND": " ",
            "WMO_WIND": " ",
            "TOKYO_WIND": " ",
            "HKO_WIND": " ",
            "REUNION_WIND": " ",
            "BOM_WIND": " ",
            "NADI_WIND": " ",
            "WELLINGTON_WIND": " ",
            "DS824_WIND": " ",
            "TD9636_WIND": " ",
            "TD9635_WIND": " ",
            "NEUMANN_WIND": " ",
            "MLC_WIND": " ",
            "STORM_DIR": " ",
            "STORM_SPEED": " ",
            "WMO_PRES": " ",
            "USA_PRES": " ",
            "TOKYO_PRES": " ",
            "CMA_PRES": " ",
            "HKO_PRES": " ",
            "NEWDELHI_PRES": " ",
            "REUNION_PRES": " ",
            "BOM_PRES": " ",
            "NADI_PRES": " ",
            "WELLINGTON_PRES": " ",
            "DS824_PRES": " ",
            "TD9636_PRES": " ",
            "TD9635_PRES": " ",
            "NEUMANN_PRES": " ",
            "MLC_PRES": " ",
        },
        dtype={
            "USA_LON": float,
            "USA_LAT": float,
            "USA_WIND": float,
            "NEWDELHI_WIND": float,
            "CMA_WIND": float,
            "WMO_WIND": float,
            "TOKYO_WIND": float,
            "HKO_WIND": float,
            "REUNION_WIND": float,
            "BOM_WIND": float,
            "NADI_WIND": float,
            "WELLINGTON_WIND": float,
            "DS824_WIND": float,
            "TD9636_WIND": float,
            "TD9635_WIND": float,
            "NEUMANN_WIND": float,
            "MLC_WIND": float,
            "STORM_DIR": float,
            "STORM_SPEED": float,
            "WMO_PRES": float,
            "USA_PRES": float,
            "TOKYO_PRES": float,
            "CMA_PRES": float,
            "HKO_PRES": float,
            "NEWDELHI_PRES": float,
            "REUNION_PRES": float,
            "BOM_PRES": float,
            "NADI_PRES": float,
            "WELLINGTON_PRES": float,
            "DS824_PRES": float,
            "TD9636_PRES": float,
            "TD9635_PRES": float,
            "NEUMANN_PRES": float,
            "MLC_PRES": float,
        },
        parse_dates=["ISO_TIME"],
        date_format="%Y-%m-%d %H:%M:%S",
        **kwargs
    )
    # Pandas parses "NA" = North America as nan
    # Convert back to NA
    ibtracs["BASIN"] = ibtracs["BASIN"].fillna("NA")

    # Make datetime col UTC timezone-aware
    ibtracs["ISO_TIME"] = ibtracs["ISO_TIME"].dt.tz_localize("UTC")

    # Some longitude values are between 180 and 360
    if signed_coords:
        ibtracs.loc[ibtracs["LON"] > 180, "LON"] -= 360.0

    # Only keep tropical cyclones
    ibtracs = ibtracs.loc[ibtracs["NATURE"] == nature]

    # Remove tracks with a single point
    ibtracs = ibtracs.loc[ibtracs.groupby("SID").transform("size") != 1]
    return ibtracs.reset_index(drop=True)

def get_basins_poly(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    polygons = {
        "SI": Polygon([(10, -60), (10, 0), (135, 0), (135, -60)]),  # SI
        "SP": MultiPolygon(
            [
                Polygon([(135, -60), (135, 0), (180, 0), (180, -60)]),
                Polygon([(-180, -60), (-180, 0), (-70, 0), (-70, -60)]),
            ]
        ),  # SP
        "SA": Polygon([(-70, -60), (-70, 0), (10, 0), (10, -60)]),  # SA
        "NI": Polygon([(30, 0), (30, 60), (100, 60), (100, 0)]),  # NI
        "WP": Polygon([(100, 0), (100, 60), (180, 60), (180, 0)]),  # WP
        "EP": Polygon([(-180, 0), (-180, 60), (-90, 60), (-90, 0)]),  # EP
        "NA": Polygon([(-90, 0), (-90, 60), (30, 60), (30, 0)]),  # EP
    }

    basins_poly = gpd.GeoDataFrame(
        list(polygons.items()), columns=["basin", "geometry"]
    )
    # Set the coordinate reference system (CRS) to WGS84 (Lat/Lon)
    basins_poly.set_crs(crs, allow_override=True, inplace=True)
    return basins_poly


def get_basins() -> pd.DataFrame:
    # Cyclone tracks inserted must be pre-processed tracks
    # Including basin def
    Eastern_pacific = ["Eastern Pacific", "EP", 14.5]
    North_atlantic = ["North Atlantic", "NA", 10.8]
    North_indian = ["North Indian", "NI", 2.0]
    South_indian = ["South Indian", "SI", 12.3]
    South_pacific = ["South Pacific", "SP", 9.3]
    Western_pacific = ["Western Pacific", "WP", 22.5]

    # SA removed because near-zero historical TC occurrences
    basins = pd.DataFrame(
        [
            Eastern_pacific,
            North_atlantic,
            North_indian,
            South_indian,
            South_pacific,
            Western_pacific,
        ],
        columns=["name", "abv", "lambda_poisson"],
    )
    return basins

def get_histo_TC_prob(ibtracs: pd.DataFrame) -> pd.DataFrame:
    ibtracs = ibtracs.sort_values(by="ISO_TIME", ascending=True)
    # Genesis point for each historical TC track: we assume the first point
    # in chronological order to be the origin of any TC
    genesis = ibtracs.drop_duplicates(subset="SID", keep="first")
    genesis = genesis.assign(month=genesis["ISO_TIME"].dt.month)
    # TODO: check that all 12 months are present including 0 counts if no event occurs
    month_basin_counts = (
        genesis.groupby(["BASIN", "month"]).size().to_frame("TC_counts")
    )
    month_basin_counts["TC_total_counts"] = month_basin_counts.groupby(
        "BASIN"
    ).transform("sum")
    month_basin_counts["TC_proba"] = (
        month_basin_counts["TC_counts"] / month_basin_counts["TC_total_counts"]
    )
    month_basin_counts = month_basin_counts.reset_index()
    # TODO: check nan
    p_hist = month_basin_counts.pivot(index="BASIN", columns="month", values="TC_proba")
    return p_hist.fillna(0.0)  # TODO: Temp fill

def get_genesis_loc(ibtracs: pd.DataFrame) -> pd.DataFrame:
    """Extracts first step of each historical cyclone track.

    Args:
        ibtracs (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: genesis locations
    """
    genesis = (
        ibtracs.loc[:, ["SID", "ISO_TIME", "LAT", "LON", "BASIN"]]
        .sort_values(by="ISO_TIME", ascending=True)
        .drop_duplicates(subset="SID", keep="first")
        .reset_index(drop=True)
    )
    genesis = genesis.assign(month=genesis["ISO_TIME"].dt.month)
    genesis = gpd.GeoDataFrame(
        genesis, geometry=gpd.points_from_xy(genesis.LON, genesis.LAT, crs="WGS 84")
    )
    return genesis


def interpolate_basins(
    genesis: pd.DataFrame, basins_poly: gpd.GeoDataFrame, resolution: int
) -> gpd.GeoDataFrame:
    # get xmin, xmax and ymin, ymax for each basin to feed to interpolate_grid
    basin_counts_dict = {}
    for basin in basins_poly["basin"].unique():
        # Interpolated grid of TC locations to sample from
        # for generating synthetic tracks
        basin_grid = get_basin_grid(
            basin=basin, resolution=resolution, basins=basins_poly
        )
        # TODO: monthly basin grid
        basin_counts = get_basin_counts(basin_grid=basin_grid, genesis=genesis)
        basin_counts["centroid"] = basin_counts.to_crs(
            "EPSG: 3857"
        ).geometry.centroid.to_crs("WGS 84")
        basin_counts["lon"] = basin_counts["centroid"].x
        basin_counts["lat"] = basin_counts["centroid"].y

        # Cubic interpolation
        basin_counts_arr, lat_grid, lon_grid = interp_grid(
            scattered_lat=basin_counts.lat.values,
            scattered_lon=basin_counts.lon.values,
            scattered_values=basin_counts["count"].values,
            resolution=resolution,
        )

        # Convert interpolated values to geodataframe
        inter_gdf = gpd.GeoDataFrame(
            data={"interp_count": np.ravel(basin_counts_arr)},
            geometry=gpd.points_from_xy(x=np.ravel(lon_grid), y=np.ravel(lat_grid)),
            crs="WGS 84",
        ).dropna(subset="interp_count")
        # Merge interpolated values to previous counts
        # TODO: check this
        toto = basin_counts.sjoin(inter_gdf, how="left", predicate="within")
        basin_counts_dict[basin] = basin_counts
    return toto, basin_counts_dict


def get_basin_grid(
    basin: str,
    resolution: float,
    basins: gpd.GeoDataFrame,
    xlim=(-180, 180),
    ylim=(-90, 90),
    crs: str = "WGS 84",
) -> gpd.GeoDataFrame:
    """Divide a basin into subpixels of a given resolution.

    Args:
        basin (str): _description_
        resolution (float): _description_
        basins (gpd.GeoDataFrame): _description_
        xlim (tuple, optional): _description_. Defaults to (-180, 180).
        ylim (tuple, optional): _description_. Defaults to (-90, 90).
        crs (str, optional): _description_. Defaults to "WGS 84".

    Returns:
        gpd.GeoDataFrame: _description_
    """
    bounds = xlim[0], ylim[0], xlim[1], ylim[1]
    # Global grid subdivided into res°xres° subgrids
    global_grid = create_grid(resolution=resolution, bounds=bounds, CRS=crs)
    basin_geometry = basins.query("basin == @basin")["geometry"].iloc[0]
    basin_grid = global_grid.loc[global_grid.geometry.within(basin_geometry)].copy()
    # Add centroid point to each pixel
    basin_grid["centroid"] = basin_grid.to_crs("EPSG: 3857").geometry.centroid.to_crs(
        "WGS 84"
    )
    basin_grid["basin"] = basin
    return basin_grid.reset_index(drop=True)


def create_grid(resolution: float, bounds: tuple, CRS: str) -> gpd.GeoDataFrame:
    """
    Generate a regular grid of polygons over a specified bounding box.

    Parameters:
    ----------
    resolution : float
        Resolution of grid cells (in degrees or meters).
    bounds : tuple
        Spatial extent of the grid (minx, miny, maxx, maxy).
    CRS : str
        Coordinate Reference System for the grid.

    Returns:
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the grid polygons.
    """
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)

    grid_polygons = []
    for x in x_coords:
        for y in y_coords:
            grid_polygons.append(
                Polygon(
                    [
                        (x, y),
                        (x + resolution, y),
                        (x + resolution, y + resolution),
                        (x, y + resolution),
                    ]
                )
            )

    grid = gpd.GeoDataFrame(geometry=grid_polygons, crs=CRS)

    return grid


def get_basin_counts(
    basin_grid: gpd.GeoDataFrame, genesis: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """_summary_

    Args:
        basin_grid (gpd.GeoDataFrame): basin divided into pixels of a given resolution
        genesis (gpd.GeoDataFrame): location of historical TC tracks origination points

    Returns:
        gpd.GeoDataFrame: _description_
    """
    joined = basin_grid.sjoin(genesis, predicate="contains", how="left")
    basin_counts = (
        joined.groupby(["geometry", "basin"])
        .agg(count=("month", lambda x: x.notna().sum()))
        .reset_index()
    )
    # Transform unnormalised weights into probability distributions for each region
    basin_counts["b_prob"] = basin_counts["count"] / basin_counts.groupby("basin")[
        "count"
    ].transform("sum")
    return gpd.GeoDataFrame(basin_counts, geometry="geometry")


def interp_grid(
    scattered_lat: np.ndarray,
    scattered_lon: np.ndarray,
    scattered_values: np.ndarray,
    resolution: int,
    xmin: float = -180,
    xmax: float = 180,
    ymin: float = -90,
    ymax: float = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        np (_type_): _description_
        np (_type_): _description_
        scattered_lat (_type_, optional): _description_. Defaults to -180, xmax: float = 180, ymin: float = -90, ymax: float = 90, )->tuple(np.ndarray.

    Returns:
        _type_: _description_
    """
    grid_lat = np.arange(-90, 90 + 1, resolution)  # New regular latitude grid
    grid_lon = np.arange(-180, 180 + 1, resolution)  # New regular longitude grid
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    # Interpolate the missing values
    interpolated_grid = griddata(
        points=np.array([scattered_lat, scattered_lon]).T,  # Non-NaN locations
        values=scattered_values,  # Values at those locations
        xi=(lat_grid, lon_grid),  # Regular grid to interpolate onto
        method="cubic",  # Interpolation method: cubic, linear, or nearest
    )
    # Interpolated values may be negative, since no positivity constraint is enforced
    # Hence, we set them to zero
    interpolated_grid[interpolated_grid < 0.0] = 0.0
    # Interpolated values are continuous, we round them to the nearest integer
    interpolated_grid = interpolated_grid.round()
    return interpolated_grid, lat_grid, lon_grid


def displace_to_ocean(point, land_union, geod, max_distance_km=50, step_km=1):
    """_summary_

    Args:
        point (_type_): _description_
        land_union (_type_): _description_
        geod:     geod = Geod(ellps="WGS84")
        max_distance_km (int, optional): _description_. Defaults to 50.
        step_km (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    prep_land_union = prep(land_union)
    # Try in a circle around the point
    azimuths = np.arange(0, 360, 15)
    for dist_km in np.arange(step_km, max_distance_km + step_km, step_km):
        lons, lats, _ = geod.fwd(
            np.full_like(azimuths, point.x),
            np.full_like(azimuths, point.y),
            azimuths,
            np.full_like(azimuths, dist_km * 1000)
        )
        for lon, lat in zip(lons, lats):
            test_point = Point(lon, lat)
            if not prep_land_union.contains(test_point):
                return test_point
    return point  # fallback (couldn't find ocean point)

def preprocess_ibtracs(ibtracs: pd.DataFrame, threshold: float):
    # TODO: clean
    ibtracs_proc = input_processing(ibtracs)
    ibtracs_av_pres = get_average_pres(ibtracs_proc)
    ibtracs_av_wind = get_average_wind(ibtracs_av_pres)
    ibtracs_av_wind_filtered = filter_wind(ibtracs_av_wind, threshold)
    # TODO: add dropna
    # Drop tracks that have at least one missing pressure value
    tracks_to_keep_mask = ibtracs_av_wind_filtered.groupby("SID")["pres"].transform(
        lambda x: x.notna().all()
    )
    ibtracs_proc = ibtracs_av_wind_filtered.loc[tracks_to_keep_mask].reset_index(drop=True)

    selected_cols = [
    "SID",
    "ISO_TIME_POSIXct",
    "NAME",
    "LAT",
    "LON",
    "BASIN",
    "DIST2LAND",
    "LANDFALL",
    "year",
    'month',
    'day',
    'wind',
    'pres'
    ]

    ibtracs_proc = ibtracs_proc.loc[:,selected_cols].rename(columns={'ISO_TIME_POSIXct':'datetime'})


    return ibtracs_proc

def input_processing(ibtracs: pd.DataFrame, year_lim_low=1989,year_lim_up=2015) -> pd.DataFrame:
    # Quick processing made in the paper
    logger.info("initiating basic input processing...")
    ibtracs = ibtracs.rename(columns={"ISO_TIME": "ISO_TIME_POSIXct"})
    ibtracs["month"] = ibtracs["ISO_TIME_POSIXct"].dt.month
    ibtracs["day"] = ibtracs["ISO_TIME_POSIXct"].dt.day
    ibtracs["year"] = ibtracs["ISO_TIME_POSIXct"].dt.year
    ibtracs["DayMonth"] = (
        ibtracs["day"].astype(str).str.zfill(2)
        + "-"
        + ibtracs["month"].astype(str).str.zfill(2)
    )
    # TODO: why replace nan with NA, which refers to North America?
    ibtracs["BASIN"] = np.where(ibtracs["BASIN"].isna(), "NA", ibtracs["BASIN"])
    ibtracs_proc = ibtracs[
        (ibtracs["year"] > year_lim_low)
        &(ibtracs["year"] < year_lim_up)
        & (ibtracs["NATURE"] != "ET")
        & (ibtracs["NATURE"] != "DS")
        & (ibtracs["BASIN"] != "SA")
    ]
    return ibtracs_proc


def get_average_pres(ibtracs):
    # function to extract and average reported pressures
    logger.info("retrieving the pressure from IBTrACS...")
    pres_cols = [col for col in ibtracs.columns if col.endswith("_PRES")]
    ibtracs[pres_cols] = ibtracs[pres_cols].apply(pd.to_numeric, errors="coerce")
    ibtracs.loc[:, "pres"] = ibtracs[pres_cols].mean(axis=1, skipna=True)
    return ibtracs


def get_average_wind(ibtracs):
    logger.info("retrieving and convert winds from IBTrACS...")
    # Function to extract and average winds
    wind_cols = [col for col in ibtracs.columns if col.endswith("_WIND")]
    # TODO: why do we multiply by .88 for these specific countries?
    ibtracs[["USA_WIND", "NEWDELHI_WIND", "CMA_WIND"]] *= 0.88
    ibtracs.loc[
        ibtracs["WMO_AGENCY"].isin(["hurdat_atl", "hurdat_epa", "newdeli"]),
        "WMO_WIND",
    ] *= 0.88
    ibtracs = ibtracs.drop(["WMO_AGENCY"], axis=1)  # TODO: why? .iloc[:-1]
    ibtracs[wind_cols] = ibtracs[wind_cols].apply(pd.to_numeric, errors="coerce")
    ibtracs.loc[:, "wind"] = ibtracs[wind_cols].mean(axis=1, skipna=True)
    ibtracs = ibtracs.dropna(subset=["LANDFALL", "wind"])
    return ibtracs


def filter_wind(ibtracs: pd.DataFrame, threshold=35):
    logger.info("filtering storm with wind above threshold...")
    # TODO: why use this factor?
    ibtracs["wind"] = ibtracs["wind"] * 0.514444 #Knots to m.s-1
    # Find maximum wind value (e.g. peak) for each track
    max_wind_df = ibtracs.groupby("SID")["wind"].max()
    # filtering out storms with wind speed inferior to threshold
    tracks_to_keep = max_wind_df[max_wind_df >= threshold].index
    filtered_df = ibtracs.loc[ibtracs["SID"].isin(tracks_to_keep)]
    return filtered_df


def skip_row_func(row_number):
    if (
        row_number == 1
    ):  # skip second row since it contain units, to read the types properly
        return True
    return False

def process_ibtracs_data(ibtracksf_folder,stop_year=2014, min_wind_speed=32.92):
    """
    Reads the ibtracs dataset, filters it, and prepares the wind data.
    Returns the processed dataframes required for frequency computation.
    """

    ibtracs = pd.read_csv(
        ibtracksf_folder,
        skiprows=skip_row_func,
        low_memory=False,
    )
    ibtracs = ibtracs.rename(columns={"SEASON": "year"})

    # Time formatting for easier extraction
    ibtracs["ISO_TIME_POSIXct"] = pd.to_datetime(ibtracs["ISO_TIME"])
    ibtracs["year"] = ibtracs["ISO_TIME_POSIXct"].dt.year
    ibtracs["DayMonth"] = ibtracs["ISO_TIME_POSIXct"].dt.strftime("%d-%m")
    ibtracs["BASIN"].fillna("NA", inplace=True)

    # Data Filtering
    ibtracs_proc = ibtracs[
        (ibtracs["year"] > 1990)
        & (ibtracs["year"] < stop_year)
        & (ibtracs["NATURE"] != "ET")
        & (ibtracs["NATURE"] != "DS")
        & (ibtracs["BASIN"] != "SA")
    ]

    # Wind Data Extraction and Conversion
    selected_cols = [
        "SID",
        "ISO_TIME_POSIXct",
        "WMO_AGENCY",
        "NAME",
        "LAT",
        "LON",
        "BASIN",
        "DIST2LAND",
        "LANDFALL",
    ]
    wind_cols = [col for col in ibtracs_proc.columns if col.endswith("_WIND")]
    selected_cols.extend(wind_cols)

    ibtracs_winds = ibtracs_proc[selected_cols].copy()

    # List of columns to convert to float and apply the conversion factor
    columns_to_convert = ["USA_WIND", "NEWDELHI_WIND", "CMA_WIND"]

    for column in columns_to_convert:
        ibtracs_winds[column] = (
            pd.to_numeric(ibtracs_winds[column], errors="coerce") * 0.88
        )

    ibtracs_winds["WMO_WIND"] = pd.to_numeric(
        ibtracs_winds["WMO_WIND"], errors="coerce"
    )

    agencies_to_convert = ["hurdat_atl", "hurdat_epa", "newdeli"]
    ibtracs_winds.loc[
        ibtracs_winds["WMO_AGENCY"].isin(agencies_to_convert), "WMO_WIND"
    ] *= 0.88

    ibtracs_winds = ibtracs_winds.iloc[:-1].drop(columns=["WMO_AGENCY"])
    # Data Reformatting
    num_cols = ibtracs_winds.columns[8:]
    ibtracs_winds[num_cols] = ibtracs_winds[num_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    ibtracs_winds["Mean"] = ibtracs_winds[num_cols].mean(axis=1, skipna=True)

    selected_cols = [
        "SID",
        "ISO_TIME_POSIXct",
        "NAME",
        "LAT",
        "LON",
        "BASIN",
        "DIST2LAND",
        "LANDFALL",
        "Mean",
    ]
    ibtracs_av_wind = ibtracs_winds[selected_cols].dropna()
    ibtracs_av_wind.rename(columns={"Mean": "wind"}, inplace=True)

    # Speed Filtration
    ibtracs_av_wind["wind_ms"] = ibtracs_av_wind["wind"] * 0.514444
    storm_filter_df = ibtracs_av_wind.groupby("SID")["wind_ms"].max().reset_index()

    ibtracs_final = storm_filter_df[storm_filter_df["wind_ms"] > min_wind_speed]

    ibtracs_unique = ibtracs_proc.drop_duplicates(subset=["SID", "BASIN"])
    ibtracs_final = ibtracs_final.merge(
        ibtracs_unique[["SID", "BASIN", "year"]], how="left", on="SID"
    )
    print("IBTrACS processed successfully.")
    return ibtracs_final