import shutil
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import xarray as xr
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

from src.genesis.genesis_utils import get_basins_poly

# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


def get_seeds_from_ds(dataset: "pds.Dataset") -> list:
    seeds_table = ac.Declaration.from_sequence(
        [
            ac.Declaration("scan", ac.ScanNodeOptions(dataset)),
            ac.Declaration("aggregate", ac.AggregateNodeOptions([], ["seed"])),
        ]
    ).to_table()
    return seeds_table.column(0).to_pylist()

def get_pdrop_month_coefs(fpath: Path) -> pd.DataFrame:
    id2basin = {0: "EP", 1: "NA", 2: "NI", 3: "SI", 4: "SP", 5: "WP"}
    arr = np.load(fpath, allow_pickle=True)
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(
        {(outer_key, inner_key): values for outer_key, inner_dict in arr.item().items() for inner_key, values in inner_dict.items()},
        orient="index",
        columns=["A", "B", "C"]
    ).reset_index()
    df[["basin", "month"]] = df["index"].tolist()
    df["basin"] = df["basin"].map(id2basin)
    return df.drop(columns="index")

def merge_basin_labels(
    input_file: Path,
    output_file: Path,
    batch_size: int = 1_000_000,
    proj_crs="EPSG: 3857",
):
    # Define output schema
    schema = pa.schema(
        [
            ("time", pa.timestamp(unit="ns")),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("value", pa.float32()),
            ("variable", pa.string()),
            ("basin", pa.string()),
        ]
    )
    # Open the input Parquet file for streaming
    parquet_file = pq.ParquetFile(input_file)
    # Get the number of rows
    num_rows = parquet_file.metadata.num_rows

    # Delete output file if exists
    if output_file.is_file():
        output_file.unlink()
        warnings.warn(f"Deleting {output_file}...")
    # Read the reference dataset (smaller, fits in memory)
    gdf_ref = get_basins_poly().rename(columns={"basins": "basin"}).to_crs(proj_crs)

    # Process the input file in batches and write with a context manager
    with pq.ParquetWriter(output_file, schema) as writer:
        with tqdm(
            parquet_file.iter_batches(batch_size=batch_size, columns=schema.names),
            total=num_rows,
        ) as pbar:
            for batch in pbar:
                # Convert batch to Pandas DataFrame
                df_batch = batch.to_pandas().reset_index()
                df_batch["lon"] = normalise_lon(df_batch["lon"])

                # Convert to GeoDataFrame
                gdf_batch = gpd.GeoDataFrame(
                    df_batch,
                    geometry=gpd.points_from_xy(df_batch["lon"], df_batch["lat"]),
                    crs=proj_crs,
                )

                # Perform nearest spatial join
                gdf_joined = (
                    gpd.sjoin(gdf_batch, gdf_ref, how="left", predicate="within")
                    .to_crs("WGS 84")
                    .drop(columns=["index_right", "geometry"])
                )

                # Convert back to PyArrow Table and write
                table = pa.Table.from_pandas(gdf_joined, preserve_index=False, schema=schema)
                writer.write_table(table)
                pbar.update(len(batch))

        print(f"✅ Processing complete! Results saved to {output_file}")


def normalise_lon(lon: pd.Series):
    return np.where(lon == 180, 180, (lon + 180.0) % 360 - 180)


def preprocess_climate_vars(climate_data_dir: Path) -> None:
    """Directory containing zip files for different model-experiment pairs.

    Args:
        climate_data_dir (Path): _description_
    """
    # TODO: how to delete all temp files at once?
    model_dirs = list(d for d in climate_data_dir.iterdir() if d.is_dir())
    models_pbar = tqdm(model_dirs, desc="Models")
    for model_dir in models_pbar:
        # Climate data is extracted and stored in a 'temp' folder
        # If the temp folder does not exist, proceed to
        # data extraction
        model = model_dir.name
        models_pbar.set_postfix(model=model)
        # Initialize an empty list to store individual datasets
        nc_file_paths = []

        # Skip temp dir
        experiment_dirs = list(
            d for d in model_dir.iterdir() if d.is_dir() and d.stem != "temp"
        )
        experiments_pbar = tqdm(
            experiment_dirs, desc="Extracting zip files for all available experiments"
        )
        temp_root_dir = Path(model_dir / "temp")

        # Extract nc files from zip
        for experiment_dir in experiments_pbar:
            experiment = experiment_dir.name
            temp_dir = temp_root_dir / experiment
            temp_dir.mkdir(parents=True, exist_ok=True)

            if not any(temp_dir.glob("*.nc")):
                vars_zips = list(experiment_dir.glob("*.zip"))
                if vars_zips:
                    for zip_file in tqdm(
                        vars_zips,
                        desc="Extracting zip files for all available variables",
                        leave=False,
                    ):
                        try:
                            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                                nc_files = [
                                    f for f in zip_ref.namelist() if f.endswith(".nc")
                                ]
                                # TODO: if multiple versions of a climate variables are present
                                # TODO: in a zip files, keep the last one.
                                nc_file = nc_files[-1]
                                zip_ref.extract(nc_file, path=temp_dir)
                        except zipfile.BadZipFile as e:
                            logger.error(f"Error: Bad zip file {zip_file}")
                            logger.error(e)
                else:
                    logger.error(f"{experiment} has no zip files")
            else:
                logger.info("NC files have already been extracted!")

        # Convert xarrays from extracted nc files to dataframes saved in parquet
        # Loop over each folder in temp_root_dir
        temp_experiment_dirs = list(d for d in temp_root_dir.iterdir() if d.is_dir())
        temp_experiments_pbar = tqdm(temp_experiment_dirs, desc="Temp experiments")

        for temp_experiment_dir in temp_experiments_pbar:
            experiment = temp_experiment_dir.name
            temp_experiments_pbar.set_postfix(experiment=experiment)
            nc_file_paths = list(temp_experiment_dir.glob("*.nc"))
            pq_file_paths = list(temp_experiment_dir.glob("*.parquet"))
            # Check whether all parquet files have been generated
            if (len(pq_file_paths) < len(nc_file_paths)) & (len(nc_file_paths) > 0):
                logger.info(f"Processing {experiment} (xarray.Dataset -> pd.DataFrame)")
                nc_files_pbar = tqdm(nc_file_paths, desc="NC files")
                for nc_file in nc_files_pbar:
                    var = nc_file.stem.split("_")[0]
                    ds = xr.open_dataset(nc_file)
                    if ds["time"].dtype == object:
                        ds["time"] = xr.CFTimeIndex(
                            ds["time"][:].values
                        ).to_datetimeindex()
                    if ("latitude" in ds.coords) & ("longitude" in ds.coords):
                        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
                    df = (
                        ds[[var]]
                        .to_dataframe()
                        .dropna(subset=var)
                        .rename(columns={var: "value"})
                    )
                    if ("lat" in df.columns) & ("lon" in df.columns):
                        df.set_index(["lat", "lon"], append=True, inplace=True)
                    df["variable"] = var
                    nc_files_pbar.set_postfix(
                        model=model, experiment=experiment, var=var
                    )
                    df.to_parquet(nc_file.parent / f"{var}.parquet")

            elif len(nc_file_paths) == 0:
                logger.error(f"Missing extracted .nc files for  {model_dir.stem}")
            else:
                logger.info(
                    f"Climate data has already been extracted for model-experiment {model_dir.stem, experiment}!"
                )
                logger.info(
                    f"Please delete {model_dir / 'temp'} in order to re-run extraction!"
                )  # TODO: clean


def concat_parquet_files(
    parquet_files_dir: Path, save_path: Path = None, remove_dir: bool = False
) -> None:
    """Concatenate parquet files from a given directory into a single parquet file.

    Args:
        parquet_files_dir (Path): _description_
        save_path (Path): _description_
    """
    ds = pds.dataset(parquet_files_dir, format="parquet")

    files = ds.files

    # Read the first Parquet file to obtain the schema
    first_table = pq.read_table(files[0])
    schema = first_table.schema  # Get the schema from the first file

    # If save_path is not provided, save the concatenated file in the parent directory
    if save_path is None:
        save_path = parquet_files_dir.parent / f"{parquet_files_dir.name}.parquet"

    # Open the output Parquet file for writing
    with pq.ParquetWriter(save_path, schema=schema) as writer:
        # Write the first table
        writer.write_table(first_table)

        remaining_files = files[1:]
        with progress_bar as pbar:
            for file in pbar.track(
                remaining_files,
                # total=len(remaining_files),
                description="Concatenating parquet files",
            ):
                # Read the current Parquet file in chunks
                table = pq.read_table(file)

                # Write the table to the output file
                writer.write_table(table)

    # Delete old directory with multiple parquet files
    if remove_dir:
        shutil.rmtree(parquet_files_dir)
        logger.info(f"{parquet_files_dir} was deleted!")


def delete_temp_folders(base_dir):
    """Delete all 'temp' folders in a given dir.
    Used to delete unzipped climate variables.

    Args:
        base_dir (_type_): _description_

    Raises:
        ValueError: _description_
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise ValueError(f"The specified path {base_dir} is not a directory.")

    for temp_folder in base_path.rglob("temp"):
        if temp_folder.is_dir():
            try:
                shutil.rmtree(temp_folder)  # Deletes folder and all its contents
                print(f"Deleted folder: {temp_folder}")
            except Exception as e:
                print(f"Failed to delete {temp_folder}: {e}")

def add_year_col(hive_dir: Path) -> None:
    output_path = "output_data_with_year/"  # New dataset

    # Load dataset with Hive-style partitions
    dataset = pds.dataset(hive_dir, format="parquet", partitioning="hive")

    # Iterate by partition group to avoid loading all at once
    for fragment in tqdm(dataset.get_fragments()):
        # Load one fragment (Parquet file)
        table = fragment.to_table()

        # Extract time column
        time_col = table["time"]
        if not pa.types.is_timestamp(time_col.type):
            time_col = pc.strptime(
                time_col, format="%Y-%m-%d", unit="s"
            )  # Adjust format

        # Add the new 'year' column
        year_col = pc.year(time_col)
        table = table.append_column("year", year_col)

        # Get relative partition path from fragment
        relative_path = Path(str(fragment.path)).relative_to(hive_dir)
        full_output_path = Path(output_path) / relative_path.parent
        full_output_path.mkdir(parents=True, exist_ok=True)

        # Write the new file (same filename)
        pq.write_table(table, full_output_path / Path(fragment.path).name)

    print("✅ Done. All partitions updated with 'year' column.")