import pandas as pd
import numpy as np
from pathlib import Path


def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None  # returns NA when the conversion fails

def input_processing(ibtracs, year_lim_low=1989,year_lim_up=2015):
    # Quick processing made in the paper
    print("initiating basic input processing...")
    ibtracs["ISO_TIME_POSIXct"] = pd.to_datetime(
        ibtracs["ISO_TIME"], format="%Y-%m-%d %H:%M:%S", utc=True
    )
    ibtracs["month"] = ibtracs["ISO_TIME_POSIXct"].dt.month
    ibtracs["day"] = ibtracs["ISO_TIME_POSIXct"].dt.day
    ibtracs["year"] = ibtracs["ISO_TIME_POSIXct"].dt.year
    ibtracs["DayMonth"] = ibtracs["ISO_TIME_POSIXct"].dt.strftime("%d-%m")
    ibtracs["BASIN"] = np.where(ibtracs["BASIN"].isna(), "NA", ibtracs["BASIN"])
    ibtracs_proc = ibtracs[
        (ibtracs["year"] > year_lim_low)&
        (ibtracs["year"] < year_lim_up)
        & (ibtracs["NATURE"] != "ET")
        & (ibtracs["NATURE"] != "DS")
        & (ibtracs["BASIN"] != "SA")
    ]
    return ibtracs_proc


def get_average_pres(ibtracs_proc):
    print("retrieving the pressure from IBTrACS...")
    # function to extract and average reported pressures
    ibtracs_pres = ibtracs_proc[
        ["SID", "ISO_TIME_POSIXct", "NAME", "LAT", "LON", "BASIN"]
        + [col for col in ibtracs_proc.columns if col.endswith("_PRES")]
    ].iloc[1:]
    num_cols = ibtracs_pres.columns[7:]

    ibtracs_pres[num_cols] = ibtracs_pres[num_cols].applymap(convert_to_float)
    ibtracs_pres["Mean"] = ibtracs_pres[num_cols].mean(axis=1, skipna=True)
    ibtracs_av_pres = ibtracs_pres[
        ["SID", "ISO_TIME_POSIXct", "NAME", "LAT", "LON", "BASIN", "Mean"]
    ]
    ibtracs_av_pres = ibtracs_av_pres.dropna()
    ibtracs_av_pres = ibtracs_av_pres.rename(columns={"Mean": "pres"})
    return ibtracs_av_pres


def get_average_wind(ibtracs_proc):
    print("retrieving and convert winds from IBTrACS...")
    # Function to extract and average winds
    ibtracs_winds = ibtracs_proc[
        [
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
        + [col for col in ibtracs_proc.columns if col.endswith("_WIND")]
    ].iloc[1:]
    num_cols = ibtracs_winds.columns[9:]
    # print(num_cols)
    ibtracs_winds[num_cols] = ibtracs_winds[num_cols].applymap(convert_to_float)
    ibtracs_winds["USA_WIND"] = pd.to_numeric(ibtracs_winds["USA_WIND"]) * 0.88
    ibtracs_winds["NEWDELHI_WIND"] = (
        pd.to_numeric(ibtracs_winds["NEWDELHI_WIND"]) * 0.88
    )
    ibtracs_winds["CMA_WIND"] = pd.to_numeric(ibtracs_winds["CMA_WIND"]) * 0.88
    ibtracs_winds["WMO_WIND"] = pd.to_numeric(ibtracs_winds["WMO_WIND"])
    ibtracs_winds.loc[ibtracs_winds["WMO_AGENCY"] == "hurdat_atl", "WMO_WIND"] *= 0.88
    ibtracs_winds.loc[ibtracs_winds["WMO_AGENCY"] == "hurdat_epa", "WMO_WIND"] *= 0.88
    ibtracs_winds.loc[ibtracs_winds["WMO_AGENCY"] == "newdeli", "WMO_WIND"] *= 0.88
    ibtracs_winds = ibtracs_winds.drop(["WMO_AGENCY"], axis=1).iloc[:-1]
    ibtracs_winds["Mean"] = ibtracs_winds[num_cols].mean(axis=1, skipna=True)
    ibtracs_av_wind = ibtracs_winds[
        [
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
    ]
    ibtracs_av_wind = ibtracs_av_wind.dropna()
    ibtracs_av_wind = ibtracs_av_wind.rename(columns={"Mean": "wind"})
    return ibtracs_av_wind


def filter_pres(ibtracs_av_pres, threshold=40):
    # filtering (out) storms with pressure inferior to threshold
    storm_filter_df = (
        ibtracs_av_pres.groupby("SID")
        .apply(lambda x: x.iloc[x["pres"].idxmin()])
        .assign(pres_drop=lambda x: 1015 - x["pres"])
        .query(f"pres_drop > {threshold}")
        .reset_index(drop=True)
        .loc[:, ["SID", "BASIN"]]
    )
    return storm_filter_df


def filter_wind(df, threshold=35, convert=True):
    print("filtering storm with wind above threshold...")
    if convert:
        df["wind"] = df["wind"] * 0.514444
    # filtering (out) storms with wind speed inferior to threshold
    # group by SID and get the row with the maximum wind value for each group
    max_wind_df = df.groupby("SID").apply(
        lambda x: x.iloc[0]["wind"]
        if len(x) == 1
        else x["wind"][x["wind"] >= 0].dropna().max()
        if len(x) > 0
        and not x["wind"].isnull().all()
        and not x["wind"][x["wind"] >= 0].dropna().empty
        else None
    )
    # filter SIDs with wind below the threshold value
    filtered_df = max_wind_df[max_wind_df >= threshold]
    # reset the index
    filtered_data = df[df["SID"].isin(filtered_df.index)]
    return filtered_data


def full_processing(ibtracs, threshold=35):
    ibtracs_proc = input_processing(ibtracs)
    ibtracs_av_pres = get_average_pres(ibtracs_proc)
    ibtracs_av_wind = get_average_wind(ibtracs_proc)
    ibtracs_av_wind_filtered = filter_wind(ibtracs_av_wind, threshold)
    ibtracs_processed = pd.merge(
        ibtracs_av_wind_filtered,
        ibtracs_av_pres,
        on=["SID", "NAME", "BASIN", "ISO_TIME_POSIXct", "LON", "LAT"],
    )
    return ibtracs_processed

def read_ibtracs(fpath: Path,
                 start_year:int = 1990, 
                 stop_year:int = 2015,
                 nature: str = "TS", 
                 signed_coords: bool = False, 
                 **kwargs):
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