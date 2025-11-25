import json
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import polars as pl
from tqdm import tqdm

POLARS_DATA_SCHEMA = {
    "source_id": pl.Utf8,
    "source_name": pl.Utf8,
    "source_type": pl.Utf8,
    "iso3_country": pl.Utf8,
    "sector": pl.Utf8,
    "subsector": pl.Utf8,
    "start_time": pl.Utf8,
    "end_time": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "geometry_ref": pl.Utf8,
    "gas": pl.Utf8,
    "emissions_quantity": pl.Float64,
    "temporal_granularity": pl.Utf8,
    "activity": pl.Float64,
    "activity_units": pl.Utf8,
    "emissions_factor": pl.Float64,
    "emissions_factor_units": pl.Utf8,
    "capacity": pl.Float64,
    "capacity_units": pl.Utf8,
    "capacity_factor": pl.Float64,
    "other1": pl.Utf8,
    "other1_def": pl.Utf8,
    "other2": pl.Utf8,
    "other2_def": pl.Utf8,
    "other3": pl.Utf8,
    "other3_def": pl.Utf8,
    "other4": pl.Utf8,
    "other4_def": pl.Utf8,
    "other5": pl.Utf8,
    "other5_def": pl.Utf8,
    "other6": pl.Utf8,
    "other6_def": pl.Utf8,
    "other7": pl.Utf8,
    "other7_def": pl.Utf8,
    "other8": pl.Utf8,
    "other8_def": pl.Utf8,
    "other9": pl.Utf8,
    "other9_def": pl.Utf8,
    "other10": pl.Utf8,
    "other10_def": pl.Utf8,
    "created_date": pl.Utf8,
    "modified_date": pl.Utf8,
}


@dataclass
class ClimateTrace:
    """
    A class to handle loading, processing, and merging climate trace data from various files, including emissions sources
    and ownership data.

    Attributes:
    data_dir (Path): Path to the directory containing the climate trace data files.
    usecols_path (Path): Path to the JSON file specifying which columns to load from the data files.
    data_schema_path (Path): Path to the data schema file for dask dataframes (optional).
    data (DataFrame): Loaded climate trace data in either Polars or Dask format.

    Methods:
    load_data(data_schema_path, frame_type): Loads and returns the climate trace data from CSV files.
    read_csv_from_zip(zip_file, pat): Reads CSV files from a zip archive that match the specified pattern.
    get_valid_subsectors(sector_zip_path): Extracts valid subsectors with both emissions and ownership data from a zip file.
    merge_all_subsectors(subsectors, sector_dir): Merges data for all subsectors, returning merged assets and ownership data.
    merge_plant_ownership(subsector, subsector_src_path, subsector_ownership_path): Merges emissions data with corporate ownership.
    """

    data_dir: Path

    def __init__(
        self,
        data_dir: Path,
        usecols_path: Path,
        data_schema_path: Path = None,
        frame_type: str = "polars",
    ):
        """
        Initializes the ClimateTrace object, loading data from the provided directory and columns from the usecols file.

        Args:
        data_dir (Path): The directory containing the climate trace data.
        usecols_path (Path): Path to the JSON file specifying which columns to read.
        data_schema_path (Path, optional): Path to the data schema file (required for Dask frames). Defaults to None.
        frame_type (str, optional): Specifies the type of data frame to use ('polars' or 'dask'). Defaults to 'polars'.
        """
        self.data_dir = data_dir
        self.usecols_path = usecols_path
        with open(usecols_path, "r") as f:
            self.usecols = json.load(f)
        self.data_schema_path: Path = data_schema_path
        self.data: "DataFrame" = self.load_data(
            data_schema_path=data_schema_path, frame_type=frame_type
        )

    def load_data(self, frame_type: str, data_schema_path: Path = None) -> "DataFrame":
        """
        Loads the climate trace data from the given directory, either in Polars or Dask format, depending on the
        frame_type parameter.

        Args:
        data_schema_path (Path): Path to the data schema file (required for Dask frames).
        frame_type (str): Specifies the type of data frame to use ('polars' or 'dask').

        Returns:
        DataFrame: Loaded climate trace data in the specified frame type.
        """
        if frame_type == "dask":
            if data_schema_path is not None:
                with open(data_schema_path, "r") as f:
                    DASK_DATA_SCHEMA = json.load(f)

                data = dd.read_csv(
                    self.valid_fpaths,
                    blocksize=None,
                    dtype=DASK_DATA_SCHEMA,
                    usecols=self.usecols,
                    include_path_column=True,
                )
            else:
                raise Exception
        elif frame_type == "polars":
            zip_files = list(self.data_dir.glob("*.zip"))
            data = [
                self.read_csv_from_zip(zip_file=zip_file)
                for zip_file in tqdm(
                    zip_files, desc="Read data from zip file for each sector"
                )
            ]
            data = pl.concat(data, how="diagonal")

        return data

    def read_csv_from_zip(self, zip_file, pat: str = "_emissions_sources.csv"):
        """
        Reads all CSV files from the specified zip archive that match the provided pattern (default '_emissions-sources.csv').

        Args:
        zip_file (Path): Path to the zip file containing the CSV data.
        pat (str, optional): File name pattern to match within the zip archive. Defaults to '_emissions_sources.csv'.

        Returns:
        pl.DataFrame: Dataframe containing the concatenated CSV data that matches the specified columns and pattern.
        """
        dfs = []
        with zipfile.ZipFile(zip_file) as z:
            data_files = [fname for fname in z.namelist() if fname.endswith(pat)]
            for data_file in data_files:
                with z.open(data_file) as file:
                    df = pd.read_csv(file, nrows=1)
                if set(self.usecols).issubset(set(df.columns)):
                    with z.open(data_file) as file:
                        df = pl.read_csv(
                            file,
                            columns=self.usecols,
                            dtypes=POLARS_DATA_SCHEMA,
                        )
                        dfs.append(df)
                else:
                    warnings.warn(
                        f"{file} was not read! It may not contain the desired columns specified in 'usecols' attribute."
                    )
        if dfs:
            dfs = pl.concat(dfs, how="diagonal")
            dfs.select(self.usecols)
        else:
            dfs = pl.DataFrame(
                {
                    name: pl.Series(name=name, dtype=POLARS_DATA_SCHEMA[name])
                    for name in self.usecols
                }
            )
        return dfs

    def get_valid_subsectors(self, sector_zip_path: Path):
        """
        Extracts valid subsectors for which both emissions sources and ownership files are available from a zip archive.

        Args:
        sector_zip_path (Path): Path to the zip file containing sector data.

        Returns:
        list: Sorted list of subsectors with ownership data.
        set: Set of all subsectors found in the sources files.
        """
        with zipfile.ZipFile(sector_zip_path) as z:
            sources_files = [
                fname
                for fname in z.namelist()
                if fname.endswith("emissions_sources.csv")
            ]
            ownership_files = [
                fname for fname in z.namelist() if fname.endswith("ownership.csv")
            ]
            sector_sources = set([f.split("_")[0] for f in sources_files])
            sector_ownership = set([f.split("_")[0] for f in ownership_files])

            subsectors = sector_sources.intersection(sector_ownership)

            return sorted(list(subsectors)), sector_sources

    def merge_all_subsectors(self, subsectors: list, sector_dir: Path):
        """
        Merges data for all subsectors within a sector, combining asset-level emissions and ownership data.

        Args:
        subsectors (list): List of valid subsectors for the sector.
        sector_dir (Path): Path to the directory containing sector data.

        Returns:
        tuple: Merged dataframes for assets with ownership, assets without ownership, and ownership data.
        """
        all_mergedf, all_src, all_ownership = [], [], []
        for subsector in subsectors:
            subsector_src_path = sector_dir / f"{subsector}_emissions_sources.csv"
            subsector_ownership_path = (
                sector_dir / f"{subsector}_emissions_sources_ownership.csv"
            )
            try:
                mergedf, src, ownership = self.merge_plant_ownership(
                    subsector=subsector,
                    subsector_src_path=subsector_src_path,
                    subsector_ownership_path=subsector_ownership_path,
                )
                all_mergedf.append(mergedf)
                all_src.append(src)
                all_ownership.append(ownership)
            except Exception as e:
                print(f"Error occurred while merging {subsector}: {str(e)}")
                continue

        try:
            all_mergedf = pd.concat(all_mergedf, axis=0, ignore_index=True)
            all_src = (pd.concat(all_src, axis=0, ignore_index=True),)
            all_ownership = pd.concat(all_ownership, axis=0, ignore_index=True)

        except Exception:
            print(
                "Emissions sources or ownership files might be missing from the zip folder"
            )
            all_mergedf, all_src, all_ownership = None, None, None

        if all_src is not None:
            return all_mergedf, all_src[0], all_ownership

        else:
            return all_mergedf, all_src, all_ownership

    @staticmethod
    def merge_plant_ownership(
        subsector: str, subsector_src_path: Path, subsector_ownership_path: Path
    ) -> pd.DataFrame:
        """
        Merge asset-level emissions with corporate ownership data for a given subsector.

        Args:
            - subsector (str): Name of the subsector.
            - subsector_src_path (Path): Path to the emissions sources file for the subsector.
            - subsector_ownership_path (Path): Path to the ownership data file for the subsector.

        Returns:
                    pd.DataFrame: Merged dataframe of asset-level emissions and ownership data.
        """

        zf = zipfile.ZipFile(subsector_src_path.parent.parent)
        src = pd.read_csv(zf.open(f"{subsector}_emissions_sources.csv"))
        ownership = pd.read_csv(zf.open(f"{subsector}_emissions_sources_ownership.csv"))

        src["start_year"] = pd.to_datetime(src["start_time"]).dt.year
        src["start_month"] = pd.to_datetime(src["start_time"]).dt.month
        src["subsector"] = subsector.split("/")[1]
        ownership["start_year"] = pd.to_datetime(ownership["start_date"]).dt.year
        ownership["start_month"] = pd.to_datetime(ownership["start_date"]).dt.month
        ownership["subsector"] = subsector.split("/")[1]

        m1 = src["gas"] == "co2e_100yr"
        m2 = src["start_year"] <= 2023

        ops_dict = {}
        tuples_list = [
            ("emissions_quantity", "sum"),
            ("activity", "sum"),
            ("capacity", "max"),
        ]
        for pair in tuples_list:
            if pair[0] in src.columns:
                ops_dict[pair[0]] = pair[1]
        if "geometry_ref" in src.columns:
            annual_src = (
                src.loc[m1 & m2]
                .groupby(
                    [
                        "source_id",
                        "source_name",
                        "subsector",
                        "start_year",
                        "geometry_ref",
                        "lat",
                        "lon",
                    ]
                )
                .agg(ops_dict)
                .reset_index()
            )
        else:
            annual_src = (
                src.loc[m1 & m2]
                .groupby(["source_id", "source_name", "subsector", "start_year"])
                .agg(ops_dict)
                .reset_index()
            )

        if ("activity" in annual_src.columns) & ("capacity" in annual_src.columns):
            annual_src["capacity_factor"] = (
                annual_src["activity"] / annual_src["capacity"]
            )

        if ("geometry_ref" in annual_src.columns) & (
            "geometry_ref" in ownership.columns
        ):
            mergedf = pd.merge(
                annual_src,
                ownership[
                    [
                        "source_id",
                        "source_name",
                        "ultimate_parent_name",
                        "ultimate_parent_id",
                        "percent_interest_parent",
                        "geometry_ref",
                        "iso3_country",
                        "lat",
                        "lon",
                    ]
                ],
                on=["source_id", "geometry_ref", "source_name", "lat", "lon"],
                how="left",
            )
        else:
            mergedf = pd.merge(
                annual_src,
                ownership[
                    [
                        "source_id",
                        "source_name",
                        "ultimate_parent_name",
                        "ultimate_parent_id",
                        "percent_interest_parent",
                        "iso3_country",
                    ]
                ],
                on=["source_id", "source_name"],
                how="left",
            )
        if "activity" in mergedf.columns:
            mergedf["emissions_factor"] = (
                mergedf["emissions_quantity"] / mergedf["activity"]
            )

        return mergedf, src, ownership


if __name__ == "__main__":
    root_dir = Path(
        "C:/Users/Jeremy.dreumont/Documents/Physical Risk/Data/climate_trace"
    )
    usecols_path = Path(
        "C:/Users/Jeremy.dreumont/Documents/natura/datasets/ctrace_usecols.json"
    )
    dask_schema_path = Path(
        "C:/Users/Jeremy.dreumont/Documents/natura/datasets/dask_data_schema.json"
    )

    ctrace = ClimateTrace(
        data_dir=root_dir,
        usecols_path=usecols_path,
        data_schema_path=dask_schema_path,
        frame_type="polars",
    )
