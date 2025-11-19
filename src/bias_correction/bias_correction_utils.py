from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd




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