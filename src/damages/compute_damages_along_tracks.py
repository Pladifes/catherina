import gc
import sqlite3
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from shapely.geometry import box
from tqdm import tqdm

from src.damages.damage_functions import emanuel_2011
from src.damages.exposure import expo_set_up
from src.damages.utils import get_iiasa_region, process_tracks


def run_damages(
    int_tracks_dir: List[str],
    catherina_fit_path: Path,
    wealth_density_dir: Path,
    country_rmsf_corr_path: Path,
    func: str,
    buffer_dist: float,
    scenario: str,
    model: str,
    group_country: bool,
    save_dir: Path,
    alphas: Tuple[Optional[float], Optional[float]] = (None, None),
):
    """
    Reads cyclone track files, computes damages using specified parameters, and saves the results as CSV files.

    Parameters
    -----------
    - files : List[str]
        Paths to intensified track files.
        example: ["outputs/ukesm1_0_ll/ssp2_4_5/AR_tracks_proc_1y_s2050seed1v1.2modukesm1_0_ll_ssp2_4_5.csv"]
    - func : climada damage function
        Function to compute damages: TDR1.0, TDR1.5 or RMSF.
    - buffer_dist : float
        Buffer distance for damage computation.
    - config :
        Experimental configuration, such as 'historical', 'ssp1_2_6', 'ssp2_8_5', etc.
    - model : str
        CMIP6 Climate model used for retrieving the climate data.
    - group_country : bool
        Whether to group results by country.
    - alphas : tuple in the format (alpha1,alpha2)
        Alpha values for damage computation.  Defaults to (None, None)

    Returns
    -------
    - pandas.DataFrame or None
        DataFrame containing computed damage data, or None if SSP1 is not available.
    """
    # Define columns to read
    columns_to_read = [
        "SID",
        "LAT",
        "LON",
        "ISO_TIME_POSIXct",
        "BASIN",
        "step",
        "seed",
        "start_year",
        "wind_wpr",
        "wind_sdr",
        "ADMIN",
        "ADM0_A3",
        "distance_to_coast",
        "step_on_land",
        "wind",
    ]

    # TODO: merge intensified track files into one parquet
    tracks = pl.read_csv(int_tracks_dir / "*.csv", columns=columns_to_read)
    tracks = tracks.to_pandas()
    # TODO: for debug
    # tracks = pd.concat(
    #     [
    #         pd.read_csv(p, usecols=columns_to_read)
    #         for p in list(int_tracks_dir.glob("*.csv"))[:2]
    #     ]
    # ).reset_index(drop=True)
    logger.info("Computing exposures...")
    tracks["BASIN"] = tracks["BASIN"].fillna("NA")
    tracks = tracks.rename(columns={"ADMIN": "admin", "ADM0_A3": "iso3"})

    # TODO: if exposure already computed skip
    tracks = exposure_along_tracks_opt(
        tracks_proj=tracks,
        buffer_dist=buffer_dist,
        wealth_density_dir=wealth_density_dir,
        save_dir=save_dir,
    )
    
    logger.info("Computing damages along tracks..")
    # TODO: review granularity of damages
    tracks = compute_damages(
        tracks=tracks,
        catherina_fit_path=catherina_fit_path,
        country_rmsf_corr_path=country_rmsf_corr_path,
        func=func,
        scenario=scenario,
        model=model,
        group_country=group_country,
        alphas=alphas,
    )

    output_fpath = save_dir / "damages.parquet"
    tracks.to_parquet(output_fpath, index=False)
    logger.info("Damage calculations ended sucessfully")
    logger.info("Impact evaluation saved as:")
    logger.info(output_fpath)


def compute_damages(
    tracks: pl.DataFrame,
    catherina_fit_path: Path,
    country_rmsf_corr_path: Path,
    func: str,
    scenario: str,
    model: str,
    group_country: bool = False,
    alphas: tuple = (None, None),
) -> pl.DataFrame:
    """
    Calculates damages from track data.

    Parameters:
    ------------
    - tracks: DataFrame with track data.
    - func: Damage function used for calculations, default 'RMSF'.
    - buffer_dist: Buffer distance, default 0.25.
    - scenario: transition path
    - model: global climate model
    - group_country: Group by country, default False: returns tracks with damages along each step.

    Returns:
    ---------
    - DataFrame with calculated damages.
    """
    with sqlite3.connect(catherina_fit_path) as conn:
        damage_funcs = pd.read_sql_query("SELECT * FROM CLIMADA_damage_funcs", conn)

    tracks = damage_along_tracks_ratio(
        tracks_proj=tracks, damage_funcs=damage_funcs, scaled=False, func=func
    )

    tracks["model"] = model

    # TODO: why do we need regions?
    # Getting region based on the tracks
    tracks = get_iiasa_region(catherina_fit_path=catherina_fit_path, tracks_proj=tracks)

    # TODO: what does this do?
    tracks = process_tracks(tracks, is_scaled=False)

    # Calculating the annualized damage
    tracks = annualized_damage(
        tracks_proj=tracks,
        catherina_fit_path=catherina_fit_path,
        country_rmsf_corr_path=country_rmsf_corr_path,
        scenario=scenario,
        per_mod=False,
        Tresh=1,
        group_country=group_country,
        alphas=alphas,
        FUNC=func,
    )
    return tracks


def damage_along_tracks_ratio(
    tracks_proj, damage_funcs: pd.DataFrame, func="RMSF", scaled=True
):
    """
    Computes the damage along tracks ratio using Climada damage functions

    Parameters:
    -----------
    - tracks_proj (pandas.DataFrame): The dataframe of storm tracks.
    - func (string): Climada damage function : TDR1.0, TDR1.5, RMSF
    - is_scaled (bool): True if wind speeds are scaled, False otherwise.

    Returns:
    ----------
    - tracks_proj (pandas.DataFrame): The same dataframe with the damage function data added.
    """
    global_vhalf = damage_funcs.loc[damage_funcs["ID"] == f"GLB{func}", "v_half"].iloc[
        0
    ]

    # TODO: multiple damage functions replcace == with .isin(dfuncs)
    damage_funcs = (
        damage_funcs[damage_funcs["Method"] == func]
        .rename(columns={"country": "iso3"})[["iso3", "v_half"]]
        .drop_duplicates()
    )
    logger.info(type(tracks_proj))
    logger.info(type(damage_funcs))

    logger.info(tracks_proj.dtypes)
    logger.info(damage_funcs.dtypes)

    # TODO: refactor typing
    tracks_proj = tracks_proj.astype("object")
    damage_funcs = damage_funcs.astype("object")

    tracks_proj = pd.merge(tracks_proj, damage_funcs, on="iso3", how="left")

    # The missing countries in Climada Damage functions get the global value
    tracks_proj["v_half"] = tracks_proj["v_half"].fillna(global_vhalf)
    logger.info(tracks_proj.columns)

    wind_col = "wind_scaled" if scaled else "wind"
    tracks_proj["applied.ratio"] = emanuel_2011(
        tracks_proj[wind_col], 25.7, tracks_proj["v_half"]
    )
    return tracks_proj


def exposure_along_tracks_opt(
    tracks_proj: pd.DataFrame,
    wealth_density_dir: Union[str, Path],
    save_dir: Path,
    buffer_dist=0.3,
    chunk_size: int = 100,  # TODO: add to config file
) -> pd.DataFrame:
    """
    Computes potential exposure to cyclones by evaluating cyclone track data against LitPop asset values within a buffer zone.

    Parameters
    -----------
    - tracks_proj
    - wealth_densiy_dir
    - buffer_dist

    Returns
    --------
    - Dataframe
    """
    seedmax, ISO3, tracks_proj_rtn, wgs84, tracks_proj_aggregated = expo_set_up(
        tracks_proj, wealth_density_dir
    )
    # TODO: add skip already computed exposures

    expo_dir = save_dir / "exposures"
    expo_dir.mkdir(parents=True, exist_ok=True)
    # TODO: casting should be done prior
    tracks_proj = tracks_proj.astype({"LAT": float, "LON": float})

    # TODO: bottleneck, loop sjoin on each country inefficient
    # TODO: replace this with a single parquet file per sector
    # Compute damages for each (sector, country) wealth density
    # National wealth densities have a single sector (e.g. the global economy)
    sector_dirs = [p for p in wealth_density_dir.iterdir() if p.is_dir()]
    sector_pbar = tqdm(
        sector_dirs, desc="Wealth densities", leave=True
    )
    results = []
    for sector_dir in sector_pbar:
        sector = sector_dir.name.split("_")[-1]
        sector_pbar.set_postfix(sector=sector)
        country_pbar = tqdm(
            list(sector_dir.glob("*.csv")), desc="For each country...", leave=False
        )
        sector_results = []
        for country_fpath in country_pbar:
            country_iso3 = country_fpath.name.strip(".csv").split("_")[-1]
            country_pbar.set_postfix(country=country_iso3)
            #
            tracks_proj_int = tracks_proj.loc[
                tracks_proj["iso3"] == country_iso3, ["LAT", "LON", "SID", "seed"]
            ].reset_index(drop=True)
            if tracks_proj_int.shape[0] < 10:
                logger.debug(f"No tracks overlapping with {country_iso3}")
                continue

            tracks_proj_int_gpd = gpd.GeoDataFrame(
                tracks_proj_int,
                geometry=gpd.points_from_xy(
                    tracks_proj_int["LON"], tracks_proj_int["LAT"]
                ),
                crs=wgs84,
            )
            # Wealth exposed to climate hazard
            expo = pd.read_csv(
                country_fpath, usecols=["value", "latitude", "longitude"], dtype=float
            )
            expo_gpd = gpd.GeoDataFrame(
                expo,
                geometry=gpd.points_from_xy(expo["longitude"], expo["latitude"]),
                crs=wgs84,
            )

            # TODO: this block should replace what follows below
            # TODO: standy for now
            # TODO: sjoin REFACTOR START
            # tracks_proj_gpd = gpd.GeoDataFrame(
            #     tracks_proj,
            #     geometry=gpd.points_from_xy(tracks_proj["LON"], tracks_proj["LAT"]),
            #     crs=wgs84,
            # )
            # expo_gpd = dgpd.from_geopandas(expo_gpd, npartitions=1)
            # tracks_proj_gpd = dgpd.from_geopandas(tracks_proj_gpd, npartitions=20)
            # sjoin = dgpd.sjoin(
            #     expo_gpd, tracks_proj_gpd, how="inner", predicate="within"
            # )
            # TODO: sjoin REFACTOR END

            # Implements an inner join between all intensified tracks
            # and exposed assets / wealth densities (i.e. expo_gpd)
            tracks_proj_aggregated = process_chunks(
                tracks_proj_int_gpd=tracks_proj_int_gpd,
                expo_gpd=expo_gpd,
                seedmax=seedmax,  # TODO: remove seedmax arg
                chunk_size=chunk_size,
                buffer_dist=buffer_dist,
            )
            tracks_proj_int = tracks_proj_int.merge(
                tracks_proj_aggregated[["SID", "LAT", "LON", "seed", "exposed.cap"]],
                on=["SID", "LAT", "LON", "seed"],
                how="left",
            )

            # TODO: debug here, the concat generates error
            # tracks_proj_rtn = pd.concat(
            #     [tracks_proj_rtn, tracks_proj_int], ignore_index=True
            # )

            tracks_proj_rtn = tracks_proj_int.merge(
                tracks_proj, on=["LAT", "LON", "SID", "seed"], how="left"
            )
            # Save damages by group
            tracks_proj_rtn["sector"] = sector
            sector_results.append(tracks_proj_rtn)
        sector_results = pd.concat(sector_results, axis=0, ignore_index=True)
        sector_results.to_parquet(expo_dir / f"{sector}.parquet")
        results.append(sector_results)
    results = pd.concat(results, axis=0, ignore_index=True)
    return results


def process_chunks(
    tracks_proj_int_gpd: gpd.GeoDataFrame,
    expo_gpd: gpd.GeoDataFrame,
    seedmax: int,
    chunk_size: int,
    buffer_dist: float,
) -> pd.DataFrame:
    for start in range(0, seedmax, chunk_size):
        end = min(start + chunk_size, seedmax)

        tracks_proj_chunk = tracks_proj_int_gpd[
            (tracks_proj_int_gpd["seed"] >= start)
            & (tracks_proj_int_gpd["seed"] <= end)
        ].copy()

        if tracks_proj_chunk.empty:
            continue

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

        bounds = tracks_proj_chunk.total_bounds
        expo_gpd_reduced = expo_gpd[expo_gpd.within(box(*bounds))]

        if expo_gpd_reduced.empty:
            continue

        sjoin = gpd.sjoin(
            expo_gpd_reduced, tracks_proj_chunk, how="inner", predicate="within"
        )
        aggregated_values = (
            sjoin.groupby("index_right")["value"].sum().reset_index(name="exposed.cap")
        )

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
            )
            gc.collect()

    return tracks_proj_aggregated


def annualized_damage(
    tracks_proj: pd.DataFrame,
    catherina_fit_path: Path,
    country_rmsf_corr_path: Path,
    scenario: str = "historical",
    Tresh: float = 1,
    per_mod: bool = False,
    damage_correction: bool = True,
    group_country: bool = True,
    alphas: tuple = (None, None),
    FUNC: str = "RMSF",
):
    """
    Aggregate storm tracks annually.

    Parameters:
    -----------
    - tracks_proj (pandas.DataFrame): The dataframe of storm tracks with computed damages.
    - config : 'historical' if tracks are historical, 'PROJ' if tracks are projections of the future (introducing SSPs)
    - Tresh (float): keep only SED values larger than Tresh
    - per_mod (bool): False if result dataframe aggregates the models using mean, otherwise the annualized damages per mod.**
    - damage_correction (bool): True to add RMSF correction
    - country_rmsf_corr_path: "data/input/fit/country_rmsf_corr.xlsx"

    Returns:
    ---------
    - processed_tracks (pandas.DataFrame): The preprocessed dataframe of storm tracks annualized in every country.
    """
    # TODO: per_mod =? per_model
    # Correction factor due to over-simplification
    tracks_proj = tracks_proj.drop(columns=["v_half"]).rename(
        columns={"applied.ratio": FUNC}
    )

    corr_factor = pd.read_excel(country_rmsf_corr_path, engine="calamine")

    # TODO: move clip bounds to config
    corr_factor["correction"] = corr_factor["correction"].clip(0.5, 2)

    corr_factor["correction"] = np.where(
        corr_factor["iso3"] == "USA", 0.3, corr_factor["correction"]
    )
    corr_factor["correction"] = np.where(
        corr_factor["iso3"] == "IND", 1, corr_factor["correction"]
    )
    tracks_proj = tracks_proj.merge(corr_factor, on="iso3", how="left")

    tracks_proj["correction"] = np.where(
        pd.isna(tracks_proj["correction"]), 1, tracks_proj["correction"]
    )

    if not damage_correction:
        tracks_proj["correction"] = 1

    aggregation_dict = {
        "SED": "sum",
        "SED_sqrt": "sum",
        "SED_third": "sum",
        "SED_pop": "sum",
        "SED_gdp": "sum",
        "SED_none": "sum",
    }

    if scenario != "historical":
        with sqlite3.connect(catherina_fit_path) as conn:
            # TODO: single query
            SSP_map = pd.read_sql_query("SELECT * FROM SEDAC_population_var", conn)
            GDP_per_cap_var = pd.read_sql_query(
                "SELECT * FROM IIASA_SSP_GDP_per_cap_variation", conn
            )

        # Filtering the scenario
        GDP_per_cap_var = GDP_per_cap_var[GDP_per_cap_var["SCENARIO"] == scenario]
        SSP_map = SSP_map[SSP_map["SCENARIO"] == scenario]

        tracks_proj["ISO_TIME_POSIXct"] = pd.to_datetime(
            tracks_proj["ISO_TIME_POSIXct"],
            format="%Y-%m-%d %H:%M:%S",
            # origin="1970-01-01 00:00:00",
            utc=True,
        )
        tracks_proj["year"] = tracks_proj["ISO_TIME_POSIXct"].dt.year
        conversions = {"LAT_ref": str, "LON_ref": str}
        tracks_proj = tracks_proj.astype(conversions)
        tracks_proj["Year_ref"] = tracks_proj["Year_ref"].astype(int).astype(str)

        SSP_map = SSP_map.astype(conversions)
        SSP_map["Year_ref"] = SSP_map["Year_ref"].astype(int).astype(str)
        SSP_map = SSP_map.rename(columns={"Delta_P": "Delta_Pop"})
        # merging tracks with socio-economic info in the future (x,y,t)
        tracks_proj = tracks_proj.merge(
            SSP_map, on=["LAT_ref", "LON_ref", "Year_ref"], how="left"
        )
        # getting macro scenario GDP per cap
        GDP_per_cap_var["Year_ref"] = GDP_per_cap_var["Year_ref"].astype(str)
        tracks_proj = tracks_proj.merge(
            GDP_per_cap_var, on=["Year_ref", "SCENARIO", "REGION"], how="left"
        )
        # getting annual average
        tracks_proj["year"] = tracks_proj["ISO_TIME_POSIXct"].dt.year

        functions = {
            "none": lambda x: (x["exposed.cap"] / x["correction"]) * x[FUNC],
            "pop": lambda x: (x["Delta_Pop"] * x["exposed.cap"] / x["correction"])
            * x[FUNC],
            "gdp": lambda x: (x["Delta_y"] * x["exposed.cap"] / x["correction"])
            * x[FUNC],
            "sed": lambda x: (
                x["Delta_y"] * x["Delta_Pop"] * x["exposed.cap"] / x["correction"]
            )
            * x[FUNC],
            "sqrt": lambda x: (
                np.sqrt(x["Delta_y"])
                * x["Delta_Pop"]
                * x["exposed.cap"]
                / x["correction"]
            )
            * x[FUNC],
            "third": lambda x: (
                (x["Delta_y"] ** (1 / 3))
                * x["Delta_Pop"]
                * x["exposed.cap"]
                / x["correction"]
            )
            * x[FUNC],
        }

        for key, value in functions.items():
            tracks_proj[f"SED_{key}"] = value(tracks_proj)
        tracks_proj = tracks_proj.rename(columns={"SED_sed": "SED"})  # TODO: clean

        if alphas[0] is not None and alphas[1] is not None:
            tracks_proj[f"SED_{alphas[0]}_{alphas[1]}"] = (
                (tracks_proj["Delta_y"] ** (alphas[0]))
                * (tracks_proj["Delta_Pop"] ** (alphas[1]))
                * tracks_proj["exposed.cap"]
                / tracks_proj["correction"]
            ) * tracks_proj[FUNC]
            aggregation_dict[f"SED_{alphas[0]}_{alphas[1]}"] = "sum"
        if group_country:
            tracks_proj = (
                tracks_proj.groupby(["iso3", "SCENARIO", "year", "model", "seed"])
                .agg(aggregation_dict)
                .reset_index()
            )
            if not per_mod:
                # grouping by year, country, scenario, model, and run
                tracks_proj = (
                    tracks_proj.groupby(["iso3", "SCENARIO", "year", "seed"])
                    .agg(
                        {
                            "SED": "sum",
                            "SED_sqrt": "sum",
                            "SED_third": "sum",
                            "SED_gdp": "sum",
                            "SED_pop": "sum",
                            "SED_none": "sum",
                        }
                    )
                    .reset_index()
                )

        tracks_proj = tracks_proj[tracks_proj["SED"] > Tresh].reset_index(drop=True)
        return tracks_proj
    else:
        # HIST
        tracks_proj["year"] = tracks_proj["ISO_TIME_POSIXct"].dt.year
        tracks_proj["PHI"] = tracks_proj["exposed.cap"] / tracks_proj["correction"]
        tracks_proj["SED"] = tracks_proj["PHI"] * tracks_proj[FUNC]
        if group_country:
            # grouping by year, country, model and seed
            tracks_proj = (
                tracks_proj.groupby(["iso3", "year", "model", "seed"])
                .agg({"SED": "sum"})
                .reset_index()
            )

        tracks_proj = tracks_proj[tracks_proj["SED"] > Tresh]
        return tracks_proj


# TODO: ratio = damage function from CLIMADA
def get_all_ratio(tracks_proj, is_scaled=True):
    """
    Apply multiple CLIMADA damage functions to track projections and compute ratios.

    This function takes a GeoDataFrame of track projections and applies different
    CLIMADA damage functions (RMSF, TDR1.5, TDR1.0) to calculate damage ratios.
    It supports scaling of the damage based on the 'is_scaled' parameter. The function
    sequentially processes each damage function, removes irrelevant columns, and renames
    the 'applied.ratio' column to the name of the respective damage function.

    Parameters
    ----------
    - tracks_proj : GeoDataFrame
        A GeoDataFrame containing the track projections data.
    - is_scaled : bool, optional
        A flag to determine if the damage calculation should be scaled.
        Default is True.

    Returns
    -------
    DataFrame
        The input DataFrame with additional columns for each damage function's ratio.

    Notes
    -----
    - The damage functions, RMSF, TDR1.5, and TDR1.0, are specific to the CLIMADA model.
    - The function expects the input GeoDataFrame to be compatible with the CLIMADA
      damage functions.

    """
    # TODO: allow for damage_along_tracks_ratio to take a list of damage functions as input
    # TODO: rename the applied.ratio column to the damage function name in damage_along_tracks_ratio
    # TODO: and then delete this function
    tracks_proj = damage_along_tracks_ratio(tracks_proj, func="RMSF", scaled=is_scaled)
    tracks_proj = tracks_proj.drop(columns=["v_half"]).rename(
        columns={"applied.ratio": "RMSF"}
    )

    tracks_proj = damage_along_tracks_ratio(
        tracks_proj, func="TDR1.5", scaled=is_scaled
    )
    tracks_proj = tracks_proj.drop(columns=["v_half"]).rename(
        columns={"applied.ratio": "TDR1.5"}
    )

    tracks_proj = damage_along_tracks_ratio(
        tracks_proj, func="TDR1.0", scaled=is_scaled
    )
    tracks_proj = tracks_proj.drop(columns=["v_half"]).rename(
        columns={"applied.ratio": "TDR1.0"}
    )

    return tracks_proj
