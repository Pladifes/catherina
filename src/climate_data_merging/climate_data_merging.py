import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from pathlib import Path
import pyarrow.dataset as pds
import pyarrow.compute as pc
import xarray as xr



def process_month(
    year: int,
    month: int,
    clim_ds: xr.Dataset,
    model: str,
    experiment: str,
    tracks: pds.Dataset,
    save_dir: Path,
):
    """Merges synthetic tracks with climate data for a given (year, month)."""
    scanner = pds.Scanner.from_dataset(
        tracks,
        filter=(
            (pc.field("year") == year)
            & (pc.field("month") == month)
            & pc.is_valid(pc.field("lat"))
            & pc.is_valid(pc.field("lon"))
        ),
    )

    table = scanner.to_table()
    if table.num_rows == 0:
        return None
    else:
        df = table.to_pandas()
        df = df.reset_index(drop=True) #Reset index for unicity

        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
        ).to_crs("EPSG: 3857")

        # Climate data - pre-filter for the current batch's lat/lon range
        min_lat, max_lat = df["lat"].min(), df["lat"].max()
        min_lon, max_lon = df["lon"].min(), df["lon"].max()
        # FIXME: issue with selecting 2D lat and lon
        clim_df = (
            clim_ds.sel(
                time=f"{year:04d}-{month:02d}",
                # lat=slice(min_lat, max_lat), # FIXME: uncomment if slow
                # lon=slice(min_lon, max_lon),
            )[["hurs", "psl", "ta", "tos"]]
            .to_dataframe()
            .rename(columns={"hurs": "hur"})
        )  # TODO: change hur to hurs everywhere

        clim_df = gpd.GeoDataFrame(
            clim_df,
            geometry=gpd.points_from_xy(clim_df.lon, clim_df.lat),
            crs="EPSG:4326",
        ).to_crs("EPSG: 3857")

        # Merge environmental conditions
        merged = gpd.sjoin_nearest(
            df, clim_df, how="left", distance_col="distance_track_env"
        )
        # Some environmental points are equidistant from tracks
        # So multiple rows are sometimes created for the same track
        # We remove them
        merged = merged[~merged.index.duplicated(keep="first")].drop(
            columns=["geometry"]
        )

        # Data completion
        merged["hur"] = merged["hur"].clip(upper=100.)
        # Forward-fill for ocean variables (e.g. last value is used as model input over land)
        merged = merged.assign(
                                tos=merged["tos"].ffill(),
                                # hur=merged["hur"].ffill(),
                                # ta=merged["ta"].ffill(),
                                psl=merged["psl"].ffill(),
                            )
        # TOS:
        # - linear interpolation for nans between two valid values
        # - forward fill for nans at the end of the track
        # merged = merged.assign(
        #     tos=merged["tos"]
        #     .interpolate(method="linear", limit_direction="both", limit_area="inside")
        #     .ffill(),
        #     hur=merged["hur"]
        #     .interpolate(method="linear", limit_direction="both", limit_area="inside")
        #     .ffill(),
        #     ta=merged["ta"]
        #     .interpolate(method="linear", limit_direction="both", limit_area="inside")
        #     .ffill(),
        #     psl=merged["psl"]
        #     .interpolate(method="linear", limit_direction="both", limit_area="inside")
        #     .ffill(),
        # )
        # FIXME: temporary fix for tracks with any nan
        #merged = drop_tracks_with_any_nan(merged)
        
        # Convert to PyArrow Table and write to Parquet
        table = pa.Table.from_pandas(merged)

        pq.write_to_dataset(
            table,
            root_path=save_dir,
            partition_cols=["seed", "year", "month"],
            existing_data_behavior="overwrite_or_ignore",
        )
        return table