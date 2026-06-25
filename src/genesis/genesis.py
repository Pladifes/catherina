import pandas as pd
import numpy as np
from loguru import logger
import geopandas as gpd
from pyproj import Geod
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from shapely.prepared import prep


from src.genesis.genesis_utils import (
    get_basins, 
    get_basins_poly,
    get_histo_TC_prob,
    get_genesis_loc,
    interpolate_basins,
    displace_to_ocean,
    uniqueid
    )

def sample_monthly_events(
    rngs: dict[np.random.Generator], annual_events: pd.DataFrame, p_basin: pd.DataFrame
) -> pd.DataFrame:
    """_summary_

    Args:
        rngs (dict[np.random.Generator]): simulation paths
        annual_events (pd.DataFrame): sample of annual TC events
        p_basin (pd.DataFrame): monthly distributions of TC events

    Returns:
        pd.DataFrame: _description_
    """
    monthly_events = []
    for (basin, seed), basin_seed_df in annual_events.groupby(["abv", "seed"]):
        rng = np.random.default_rng(seed)
        n_values = basin_seed_df["count"].tolist()
        _p_basin = p_basin.loc[p_basin.index == basin].values.flatten()
        # Efficiently sample using np.apply_along_axis (loops internally)
        monthly_events_arr = np.vstack([rng.multinomial(n, _p_basin) for n in n_values])
        monthly_events_df = pd.concat(
            [basin_seed_df.reset_index(drop=True), pd.DataFrame(monthly_events_arr)],
            axis=1,
        )
        monthly_events.append(monthly_events_df)
    monthly_events = pd.concat(monthly_events, axis=0, ignore_index=True)
    # Reshape
    monthly_events_melt = monthly_events.drop(columns="count").melt(
        id_vars=["abv", "seed", "year"], var_name="month", value_name="count"
    )
    monthly_events_melt["month"] += 1  # 1-12
    return monthly_events_melt


def sample_annual_events(
    rngs: dict[int : np.random.Generator], basins: pd.DataFrame, years: list[int]
) -> pd.DataFrame:
    """Sample TC counts for each basin, year and simulation.

    Args:
        rngs (_type_): _description_
        basins (pd.DataFrame): _description_
        years (list[int]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    seeds = list(rngs.keys())
    index = pd.MultiIndex.from_product([basins["abv"], seeds], names=["abv", "seed"])
    df = pd.DataFrame(index=index, columns=years)
    for seed, rng in rngs.items():
        for i, (abv, lam) in basins[["abv", "lambda_poisson"]].iterrows():
            df.loc[(abv, seed)] = rng.poisson(lam=lam, size=len(years))
    melted = df.melt(value_name="count", var_name="year", ignore_index=False)
    melted.set_index("year", append=True, inplace=True)
    return melted.reset_index()

def simulate_tc_events(
    seeds: list[int],
    basins: pd.DataFrame,
    p_hist: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    rngs = {seed: np.random.default_rng(seed) for seed in seeds}
    years = list(range(start_year, end_year + 1))
    # For each basin, sample the number of annual TC events
    annual_sim = sample_annual_events(rngs=rngs, basins=basins, years=years)

    # For each basin and year, sample the month of occurrence of each TC
    # based on historical distribution of occurrences from IBTrACs

    monthly_sim = sample_monthly_events(
        rngs=rngs, annual_events=annual_sim, p_basin=p_hist
    )
    return monthly_sim

def sample_monthly_coords_old(
    monthly_events_melt: pd.DataFrame, basin_counts_dict: dict
) -> pd.DataFrame:
    # TODO: locations are currently sampled from the basin level
    # TODO: next implementation should include distribution at the basin-month level
    monthly_locs = []
    for (basin, seed, month), basin_seed_df in tqdm(
        monthly_events_melt.groupby(["abv", "seed", "month"]),
        desc="TC genesis for groups (basin, seed, month)",
        ascii=" >=",
        leave=False,
        position=1,
    ):
        rng = np.random.default_rng(seed)
        n_values = basin_seed_df["count"].tolist()
        # TODO: replace with basin-month distribution basin_counts_dict[basin][month]
        basin_counts = basin_counts_dict[basin]
        basin_counts["b_prob"] = basin_counts["count"] / basin_counts.groupby("basin")[
            "count"
        ].transform("sum")

        # Index of a given location from the historical distribution
        b_prob_array = basin_counts["b_prob"].values
        monthly_locs_arr = [
            rng.choice(a=len(basin_counts), size=n, p=b_prob_array) for n in n_values
        ]

        basin_seed_df["genesis_loc_index"] = monthly_locs_arr
        # Extract genesis locations for basin with monthly events
        # Get coords from sampled location index
        basin_seed_df = basin_seed_df.explode("genesis_loc_index").dropna(
            subset="genesis_loc_index"
        )
        loc_index = basin_seed_df["genesis_loc_index"].tolist()
        basin_seed_df["centroid"] = basin_counts.loc[
            basin_counts.index[loc_index], "centroid"
        ].tolist()
        monthly_locs.append(basin_seed_df)

    monthly_locs = (
        pd.concat(monthly_locs, axis=0, ignore_index=True)
        .drop(columns=["count", "genesis_loc_index"])
        .sort_values(["seed", "year", "month"])
        .rename(columns={"abv": "basin"})
        .reset_index(drop=True)
    )

    unique_seq = uniqueid()
    monthly_locs["SID"] = [next(unique_seq) for _ in range(len(monthly_locs))]

    monthly_locs = gpd.GeoDataFrame(
        data=monthly_locs,
        geometry="centroid",
        crs="WGS 84",
    )

    monthly_locs["lat"] = monthly_locs["centroid"].y
    monthly_locs["lon"] = monthly_locs["centroid"].x
    return monthly_locs.drop(columns="centroid")


def simulate_tc_genesis(
    ibtracs, 
    land_zip_path, 
    resolution, 
    n_seeds, 
    start_year, 
    end_year, 
    max_seed_existing,
    save_dir: Path, 
    displace: bool = False
):
    
    #Get land data
    land = gpd.read_file(land_zip_path).union_all()
    land_union = prep(land)
    #Get basin data
    basins = get_basins()
    basins_poly = get_basins_poly()
    p_hist = get_histo_TC_prob(ibtracs=ibtracs)
    # Historical genesis points
    logger.info("Extracting TC genesis locations from IBTracs...")
    genesis = get_genesis_loc(ibtracs=ibtracs)
    basin_counts, basin_counts_dict = interpolate_basins(
        genesis=genesis, basins_poly=basins_poly, resolution=resolution
    )
    logger.info("Sampling annual TC events for each basin")
    monthly_sim = simulate_tc_events(
        seeds=list(range(max_seed_existing, max_seed_existing + n_seeds)),
        basins=basins,
        p_hist=p_hist,
        start_year=start_year,
        end_year=end_year,
    )

    logger.info("Sampling synthetic TC genesis locations...")
    synth_genesis = sample_monthly_coords_old( # TODO: update with new function
        monthly_events_melt=monthly_sim, basin_counts_dict=basin_counts_dict
    )

    logger.info("Displacing land locations to ocean...")
    # Get land indices
    synth_genesis = gpd.GeoDataFrame(
        synth_genesis,
        geometry=gpd.points_from_xy(x=synth_genesis.lon, y=synth_genesis.lat),
        crs="EPSG: 4326",
    )
    # 3) Build GeoDataFrame in the correct CRS (no space)
    points = gpd.GeoDataFrame(
        synth_genesis,
        geometry=gpd.points_from_xy(synth_genesis.lon, synth_genesis.lat),
        crs="EPSG:4326",
    )

    is_land_mask = pd.DataFrame({"is_land": False}, index=points.index)
    for i in tqdm(points.index, desc="Finding land locations"):
        point = points.at[i, "geometry"]
        if land_union.contains(point):
            is_land_mask.at[i, "is_land"] = True
    land_index = points.index[is_land_mask["is_land"]]
    if displace:
        land_points = points.loc[land_index, "geometry"].tolist()
        # Displace land locations
        geod = Geod(ellps="WGS84")
        displaced_points = Parallel(n_jobs=1, prefer="processes")(
            delayed(displace_to_ocean)(point, land_union=land_union, geod=geod)
            for point in tqdm(land_points, desc="Displacing points from land to ocean")
        )
        # Replace old values
        synth_genesis.loc[land_index, "geometry"] = displaced_points
        
    else:
        on_land_fast = points.geometry.apply(land_union.covers)  # boolean Series
        synth_genesis = points.loc[~on_land_fast].reset_index(drop=True)
       
    # Geometry column not handled by pyarrow
    synth_genesis.drop(columns="geometry", inplace=True)
    
    # Convert to Arrow Table
    table = pa.Table.from_pandas(synth_genesis)

    # Write Hive-style partitioned Parquet
    logger.info(f"Writing dataset to {save_dir.resolve()}")
    pq.write_to_dataset(
        table=table,
        root_path=save_dir,  # output path
        partition_cols=["seed", "basin"],  # Hive-style columns
        existing_data_behavior="overwrite_or_ignore",  # optional: clean write
    )