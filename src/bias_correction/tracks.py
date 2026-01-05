import tomllib
from joblib import Parallel, delayed
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import ecdf
from loguru import logger
from tabulate import tabulate
import pyarrow.dataset as pds
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import pyarrow as pa 
import pyarrow.parquet as pq
import sys


def correct_bias_with_era5_and_save(seeds: list, 
                                    tracks_with_env_ds: pds.Dataset, 
                                    clim_obs_histo: pd.DataFrame, 
                                    clim_sim_histo: pd.DataFrame,  
                                    model: str, 
                                    experiment: str, 
                                    save_dir: Path,) -> None:
    save_dir = Path(save_dir)
    # TODO: add consistency by rescaling when fetching from pangeo API
    clim_sim_histo = clim_sim_histo.assign(nshr=clim_sim_histo["nshr"] / 100.) 
    # TODO: refactor for managing all seeds
    tracks = (
        tracks_with_env_ds.to_table(filter=(pds.field("seed").isin(seeds)))
        .to_pandas()
        .rename(columns={"psl": "MSLP", "hur": "nshr", "tos": "SST", "ta": "T_strat"})
    )
    if not tracks.empty:
        # TODO: clean
        tracks = tracks.assign(
            MSLP=tracks["MSLP"] / 100.0,
            nshr=tracks["nshr"] / 100.0,
            thermo_eff=((tracks["SST"]+273.15) - tracks["T_strat"]) / (tracks["SST"]+273.15),
        )

        for clim_var in ["thermo_eff", "nshr", "MSLP"]: 
            logger.info(f"Correcting bias for climate variable {clim_var}")
            tracks = apply_bias_correction(
                benchmark=clim_obs_histo,
                df_gcm_hist=clim_sim_histo,
                tracks=tracks,
                variable=clim_var,
            )
        # Convert to Arrow Table
        table = pa.Table.from_pandas(tracks)

        #Write Hive-style partitioned Parquet
        logger.info(f"Writing dataset to {save_dir / 'corrected_tracks' / model / experiment}")
        pq.write_to_dataset(
            table=table,
            root_path=save_dir / "corrected_tracks" / model / experiment,  # output path
            partition_cols=["seed", "year", "month"],  # Hive-style columns
            existing_data_behavior="overwrite_or_ignore",  # optional: clean write
        )
    # else:
    #     logger.info(f"No data to correct for year:{year}, month:{month}")
    


def apply_bias_correction(benchmark, df_gcm_hist, tracks, variable="thermo_eff"):
    """
    Applies bias correction to the variable column of a future tracks DataFrame
    using the CDF transformation method based on observed and historical GCM data.

    This function corrects the variable values in the 'tracks' DataFrame using
    the observed 'benchmark' data as a reference for the historical period and
    the 'df_gcm_hist' as the historical GCM data. The correction is performed by
    applying a CDF transformation (CDFt) to align the future GCM data ('tracks')
    with the observed historical data ('benchmark').

    Parameters:
    -----------
    - benchmark (pd.DataFrame): DataFrame containing observed historical data.
    - df_gcm_hist (pd.DataFrame): DataFrame containing historical GCM data.
    - tracks (pd.DataFrame): DataFrame containing future period data to be corrected.
    - variable (str, optional): The column to be corrected. Defaults to 'thermo_eff'. This column should be present in all three DataFrames.

    Returns:
    ---------
    - pd.DataFrame: The 'tracks' DataFrame with the 'variable' column bias-corrected
      based on the observed historical and historical GCM data.

    Note:
    -------
    - NaN values are excluded from the correction process and their positions in 'tracks'
      are preserved.

    """

    def collect_statistics(data):
        """Helper function to collect statistics for given data."""
        return {
            "Metric": [
                "Length",
                "Mean",
                "Std Dev",
                "Skewness",
                "Min",
                "25th Quantile",
                "75th Quantile",
                "Max",
            ],
            "Value": [
                len(data),
                round(np.mean(data), 3),
                round(np.std(data), 3),
                round(pd.Series(data).skew(), 3),
                round(np.min(data), 3),
                round(np.quantile(data, 0.25), 3),
                round(np.quantile(data, 0.75), 3),
                round(np.max(data), 3),
            ],
        }

    # Step 1: Extract thermo_eff values from each DataFrame, excluding NaNs
    O = benchmark[variable].dropna().values
    Gp = df_gcm_hist[variable].dropna().values
    Gf = tracks[variable].dropna().values

    # Step 2: Apply the CDFt function for bias correction
    # Note: This assumes CDFt function can handle and return numpy arrays directly
    CT = CDFt(O, Gp, Gf)
    corrected_thermo_eff = CT["DS"]

    # Step 3: Collect statistics for all datasets
    stats_dict = {
        "O (Observed (ERA5))": collect_statistics(O),
        "Gp (GCM Historical)": collect_statistics(Gp),
        "Gf (Future)": collect_statistics(Gf),
        "Corrected Gf (Future)": collect_statistics(corrected_thermo_eff),
    }
    print_summary_statistics(stats_dict)

    # Step 4: Replace the corrected thermo_eff values back into the tracks DataFrame
    # First, create a boolean mask to identify non-NaN indices for replacement
    non_nan_indices = ~tracks[variable].isna()

    # Then, replace only the non-NaN values with the corrected values
    # This step assumes the length of corrected_thermo_eff matches the number of non-NaN entries in tracks['thermo_eff']
    tracks.loc[non_nan_indices, variable] = corrected_thermo_eff
    # Now, 'tracks' contains the corrected 'thermo_eff' values aligned with their original indices

    return tracks


def print_summary_statistics(stats_dict):
    """
    Prints summary statistics for multiple data arrays in a single table using tabulate for clear terminal output.

    Parameters:
    -----------
    - stats_dict (dict): A dictionary where keys are dataset labels and values are dictionaries of statistics.

    """
    # Transform stats_dict to a format suitable for tabulate
    table_data = []
    for label, stats in stats_dict.items():
        row = [label] + stats["Value"]
        table_data.append(row)

    headers = ["Dataset"] + stats_dict[next(iter(stats_dict))]["Metric"]
    logger.info(tabulate(table_data, headers=headers, tablefmt="pretty"))


def CDFt(ObsRp, DataGp, DataGf, npas=1000, dev=2):
    mO = np.nanmean(ObsRp)
    mGp = np.nanmean(DataGp)
    DataGp2 = DataGp + (mO - mGp)
    DataGf2 = DataGf + (mO - mGp)

    # Calculate empirical CDFs
    FRp = ecdf(ObsRp)
    FGp = ecdf(DataGp2)
    FGf = ecdf(DataGf2)

    a = np.abs(np.nanmean(DataGf) - mGp)
    combined_data = np.concatenate(
        (ObsRp[~np.isnan(ObsRp)], DataGp[~np.isnan(DataGp)], DataGf[~np.isnan(DataGf)])
    )

    m = np.min(combined_data) - dev * a
    M = np.max(combined_data) + dev * a

    x = np.linspace(m, M, npas)

    FRP = FRp.cdf.evaluate(x)
    FGP = FGp.cdf.evaluate(x)
    FGF = FGf.cdf.evaluate(x)

    FGPm1_FGF = np.quantile(DataGp2, FGF[~np.isnan(FGF)])

    FRF = FRp.cdf.evaluate(FGPm1_FGF)

    if np.nanmin(ObsRp) < np.nanmin(DataGf2):
        i = np.argmax(x >= np.quantile(ObsRp, FRF[0]))
        j = np.argmax(x >= np.nanmin(DataGf2))

        k = i
        while j > 0 and k > 0:
            FRF[j] = FRP[k]
            j -= 1
            k -= 1

        if j > 0:
            FRF[:j] = 0

    if FRF[-1] < 1:
        i = np.argmax(x >= np.quantile(ObsRp, FRF[-1]))
        j = np.where(FRF != FRF[-1])[0][-1]

        if j == 0:
            raise ValueError("In CDFt, dev must be higher")

        dif = min(len(x) - j, len(x) - i)
        FRF[j : j + dif] = FRP[i : i + dif]
        k = j + dif

        if k < len(x):
            FRF[k:] = 1

    NaNs_indices = np.isnan(DataGf2)

    qntl = np.full(DataGf2.shape, np.nan)
    qntl[~NaNs_indices] = FGf.cdf.evaluate(DataGf2[~NaNs_indices])

    xx = np.interp(qntl, FRF, x, left=x[0], right=x[-1])

    FGp = ecdf(DataGp)
    FGf = ecdf(DataGf)
    FGP = FGp.cdf.evaluate(x)
    FGF = FGf.cdf.evaluate(x)

    return {"x": x, "FRp": FRP, "FGp": FGP, "FGf": FGF, "FRf": FRF, "DS": xx}


def get_histo_sim_from_benchmark(benchmark, histo_clim_ds):
    # First step prepare df gcm hist
    df_gcm_hist = Parallel(n_jobs=2)(
        delayed(process_task)(gp, clim_ds=histo_clim_ds, benchmark=benchmark)
        for idx, gp in tqdm(benchmark.groupby(["year", "month"]))
    )

    df_gcm_hist = pd.concat(df_gcm_hist, ignore_index=True)
    return df_gcm_hist


def process_task(gp, clim_ds, benchmark):
    df_gcm_hist = benchmark[
        [
            "SID",
            "LAT",
            "LON",
            "ISO_TIME_POSIXct",
            "BASIN",
            "step",
            "seed",
            "start_year",
            "Nyears",
        ]
    ].copy()
    df_gcm_hist = gpd.GeoDataFrame(
        df_gcm_hist,
        geometry=gpd.points_from_xy(df_gcm_hist.LON, df_gcm_hist.LAT),
        crs="EPSG:3857",
    )
    year, month = gp.year.iloc[0], gp.month.iloc[0]
    clim_df = (
        clim_ds.sel(
            time=f"{year:04d}-{month:02d}",
            # lat=slice(min_lat, max_lat),
            # lon=slice(min_lon, max_lon),
        )[["hur", "psl", "ta", "tos"]]
        .to_dataframe()
        .rename(columns={"psl": "MSLP", "hur": "nshr", "tos": "SST", "ta": "T_strat"})
    )
    clim_df = gpd.GeoDataFrame(
        clim_df, geometry=gpd.points_from_xy(clim_df.lon, clim_df.lat), crs="EPSG:3857"
    )
    # Merge environmental conditions
    df_gcm_hist = gpd.sjoin_nearest(
        df_gcm_hist, clim_df, how="left", distance_col="distance_track_env"
    )
    # TODO: is this step in the right place?
    # df_gcm_hist = central_pressure_dynamics(df_gcm_hist)

    return df_gcm_hist


def get_era5_benchmark(benchmark_path: Path):
    """Observed climate data for historical period."""
    benchmark = pd.read_csv(benchmark_path, parse_dates=["ISO_TIME_POSIXct"])
    benchmark["year"] = benchmark["ISO_TIME_POSIXct"].dt.year
    benchmark["month"] = benchmark["ISO_TIME_POSIXct"].dt.month
    return benchmark


def correct_all_clim_bias(
    main_config: dict,
    output_dir: Path,
)-> None:

    data_dir = Path(main_config["input_data_dir"])

    # Number seeds + jobs + batch size + nb steps
    n_seeds = main_config["n_seeds"]

    # Model and experiment
    model = main_config["models"][0]
    experiment = main_config["experiments"][0]

    tracks_with_env_dir = output_dir / "track_with_env" / model / experiment
    if not tracks_with_env_dir.exists():
        raise ValueError(f"Provided tracks_with_env_dir path {tracks_with_env_dir} does not exist.")

    tracks_with_env_corr_dir = output_dir / "track_with_env_corr" / model / experiment

    # Define the path to the intensified tracks file
    intensified_tracks_path = data_dir / "intensified_tracks/ACCESS-CM2/ssp585/"

    benchmark_path = data_dir / "bias_correction" / "ERA5_benchmark_100tracks_bias_corrections.csv"
    cmip_dir = data_dir / Path(main_config["climate_data_dir"])

    # Historical data for debiasing
    clim_obs_histo = get_era5_benchmark(benchmark_path=benchmark_path)
    clim_sim_histo_ds = xr.open_zarr(cmip_dir / f"{model}/historical", chunks="auto")[
        ["hurs", "psl", "ta", "tos"]
    ].rename({"hurs": "hur"})  # TODO: move rename to cmip6_pangeo.py
    clim_sim_histo = get_histo_sim_from_benchmark(
        benchmark=clim_obs_histo, histo_clim_ds=clim_sim_histo_ds
    )
    clim_sim_histo = clim_sim_histo.assign(
        MSLP=clim_sim_histo["MSLP"] / 100.0,
        thermo_eff=((clim_sim_histo["SST"] + 273.15) - clim_sim_histo["T_strat"])
        / (clim_sim_histo["SST"] + 273.15),
    )

    # Create a pyarrow scanner for the parquet file
    tracks_with_env_ds = pds.dataset(
        tracks_with_env_dir, format="parquet", partitioning="hive"
    )

    # todo: batch seeds
    for seed in range(n_seeds):  # range(gen_config["n_seeds"]):
        correct_bias_with_era5_and_save(
            seeds=[seed],
            tracks_with_env_ds=tracks_with_env_ds,
            clim_obs_histo=clim_obs_histo,
            clim_sim_histo=clim_sim_histo,
            model=model,
            experiment=experiment,
            save_dir=tracks_with_env_corr_dir,
        )


if __name__ == "__main__":

    config_path: str = "./config.toml"
    with open(config_path, "rb") as f:  # Open the file in binary mode
        config_files = tomllib.load(f)

    output_dir = Path(sys.argv[1])

    correct_all_clim_bias(main_config=config_files["main_params"], output_dir=output_dir)