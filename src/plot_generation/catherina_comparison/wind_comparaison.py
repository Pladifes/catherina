######IBTRACS PROC:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Breadown to smaller functions
# Check NI
# Bins for Saffir-Simpson scale categories in m/s
SAFFIR_SIM_CAT_MS = [32.92, 42.7, 49.3, 58.13, 70.47, 999]
CATEGORY_LABELS = ["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"]

IBTRACS_AGENCY_1MIN_WIND_FACTOR = {
    "USA_WIND": [1.0, 0.0],
    "TOKYO_WIND": [0.6, 23.3],
    "NEWDELHI_WIND": [1.0, 0.0],
    "REUNION_WIND": [0.88, 0.0],
    "BOM_WIND": [0.88, 0.0],
    "NADI_WIND": [0.88, 0.0],
    "WELLINGTON_WIND": [0.88, 0.0],
    "CMA_WIND": [0.871, 0.0],
    "HKO_WIND": [0.9, 0.0],
    "DS824_WIND": [1.0, 0.0],
    "TD9636_WIND": [1.0, 0.0],
    "TD9635_WIND": [1.0, 0.0],
    "NEUMANN_WIND": [0.88, 0.0],
    "MLC_WIND": [1.0, 0.0],
}


def skip_row_func(row_number):
    if (
        row_number == 1
    ):  # skip second row since it contain units, to read the types properly
        return True
    return False


def process_ibtracs_data(stop_year=2010, min_wind_speed=32.92):
    """
    Reads the ibtracs dataset, filters it, and prepares the wind data.
    Returns the processed dataframes required for frequency computation.
    """

    ibtracs = pd.read_csv(
        "ibtracs/ibtracs.ALL.list.v04r00.csv",
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
        (ibtracs["year"] > 1979)
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


def compute_storm_frequency(final_storms_df):
    """
    Computes the storm frequency for each basin and category.
    """

    # Categorize storms based on wind speed
    final_storms_df["category"] = pd.cut(
        final_storms_df["wind_ms"],
        bins=SAFFIR_SIM_CAT_MS,
        labels=CATEGORY_LABELS,
        right=False,
        include_lowest=True,
    )

    # Calculate frequency for each basin and category
    all_basins, all_categories, all_frequencies = [], [], []
    for basin in final_storms_df["BASIN"].unique():
        basin_subset = final_storms_df[final_storms_df["BASIN"] == basin]
        freq, _ = np.histogram(
            basin_subset["wind_ms"], bins=SAFFIR_SIM_CAT_MS, density=True
        )
        normed_freq = freq / freq.sum()
        all_basins.extend([basin] * len(CATEGORY_LABELS))
        all_categories.extend(CATEGORY_LABELS)
        all_frequencies.extend(normed_freq)

    # Create the output dataframe
    result_df = pd.DataFrame(
        {"BASIN": all_basins, "category": all_categories, "freq": all_frequencies}
    )

    return result_df


#### Catherina proc
def compute_freq_cath(tracks):
    """
    Compute the frequency of storms in different basins based on wind speed.

    Parameters:
    - tracks: DataFrame containing storm data.
              Expected columns include 'ISO_TIME_POSIXct', 'BASIN', 'wind', 'SID', and 'seed'.

    Returns:
    - df_mean: DataFrame with the mean frequency for each basin.
    - df_std: DataFrame with the standard deviation of frequency for each basin.
    """

    # Extract year from 'ISO_TIME_POSIXct' and filter based on wind speed
    tracks["year"] = pd.to_datetime(tracks["ISO_TIME_POSIXct"]).dt.year
    tracks["BASIN"] = tracks["BASIN"].fillna("NA")
    strong_wind_tracks = tracks[tracks["wind"] > 32.92]

    # Group by SID, seed, and BASIN to get max wind for each group
    max_wind_df = (
        strong_wind_tracks.groupby(["SID", "seed", "BASIN"])["wind"].max().reset_index()
    )

    # Initialize stats list to store frequency data
    stats_list = []
    basins = max_wind_df["BASIN"].unique()
    seeds = max_wind_df["seed"].unique()

    # Compute the normalized frequency for each basin and seed combination
    for basin in basins:
        for seed in seeds:
            subset = max_wind_df[
                (max_wind_df["BASIN"] == basin) & (max_wind_df["seed"] == seed)
            ]
            freq, _ = np.histogram(subset["wind"], bins=SAFFIR_SIM_CAT_MS, density=True)
            freq_normed = freq / freq.sum()
            stats_list.append({"BASIN": basin, "seed": seed, "freq": freq_normed})

    # Convert stats list to DataFrame
    df_stats = pd.DataFrame(stats_list)

    # Compute mean and standard deviation for each basin
    df_mean = (
        df_stats.groupby("BASIN")
        .apply(lambda group: group["freq"].apply(pd.Series).mean())
        .reset_index()
    )
    df_std = (
        df_stats.groupby("BASIN")
        .apply(lambda group: group["freq"].apply(pd.Series).std())
        .reset_index()
    )

    return df_mean, df_std


def process_ibtracs_data_corrected(stop_year=2010, min_wind_speed=32.92):
    """
    Reads the ibtracs dataset, filters it, and prepares the wind data.
    Returns the processed dataframes required for frequency computation.
    """

    ibtracs = pd.read_csv(
        "ibtracs/ibtracs.ALL.list.v04r00.csv",
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
        (ibtracs["year"] > 1979)
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
    for column in list(IBTRACS_AGENCY_1MIN_WIND_FACTOR.keys()):
        ibtracs_winds[column] = (
            pd.to_numeric(ibtracs_winds[column], errors="coerce")
            - IBTRACS_AGENCY_1MIN_WIND_FACTOR[column][1]
            / IBTRACS_AGENCY_1MIN_WIND_FACTOR[column][0]
        )

    ibtracs_winds["WMO_WIND"] = pd.to_numeric(
        ibtracs_winds["WMO_WIND"], errors="coerce"
    )

    agencies_to_convert = ["reunion", "bom", "nadi", "wellington", "newdeli"]
    ibtracs_winds.loc[
        ibtracs_winds["WMO_AGENCY"].isin(agencies_to_convert), "WMO_WIND"
    ] /= 0.88

    ibtracs_winds.loc[
        ibtracs_winds["WMO_AGENCY"] == "tokyo", "WMO_WIND"
    ] -= IBTRACS_AGENCY_1MIN_WIND_FACTOR["TOKYO_WIND"][1]

    ibtracs_winds.loc[
        ibtracs_winds["WMO_AGENCY"] == "tokyo", "WMO_WIND"
    ] /= IBTRACS_AGENCY_1MIN_WIND_FACTOR["TOKYO_WIND"][0]

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
