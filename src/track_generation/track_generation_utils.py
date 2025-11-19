import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd


def preprocess_resid_params(fit_data: pd.DataFrame, coord: str) -> pd.DataFrame:
    resids = fit_data[f"Residual_{coord}"]
    add_split_group(ibtracs_reg=resids)
    # Shift longitude scale from [0°, 360°] to [-180°, 180°]
    resids.loc[resids["lon"] > 180, "lon"] -= 360.0
    resids = resids.rename(columns={"mean": f"{coord}_mean", "sd": f"{coord}_sd"})
    resids = gpd.GeoDataFrame(
        resids, geometry=gpd.points_from_xy(x=resids.lon, y=resids.lat, crs="WGS 84")
    )
    return resids


def add_split_group(ibtracs_reg: pd.DataFrame) -> None:
    """Regression coefficients for estimating cyclone tracks dynamics.
    Based on gridded lat/lon distribution from IbTracs.

    Sources:
    - James and Mason
    - Bloemendaal 2022

    Args:
        ibtracs_reg (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    ibtracs_reg[["lat", "lon", "month"]] = (
        ibtracs_reg["group"].str.split("_", expand=True).astype(float)
    )

def preprocess_delta_coef(coefs: pd.DataFrame) -> pd.DataFrame:
    # Coefs are stored in melted format (e.g. column-wise)
    # Pivot to have a single coef value (y) per group value (x)
    coefs = coefs.pivot(index="group", columns="term", values="estimate").reset_index()
    add_split_group(ibtracs_reg=coefs)
    diff_prev_col = coefs.columns[coefs.columns.str.contains("_diff_prev")][0]
    # LAT or LON
    latlon = diff_prev_col.split("_")[0]
    coefs = coefs.rename(columns={"(Intercept)": f"{latlon}_intercept"})
    coefs = gpd.GeoDataFrame(
        coefs, geometry=gpd.points_from_xy(x=coefs.lon, y=coefs.lat, crs="WGS 84")
    )
    return coefs


def get_fit_data(fit_dir: Path) -> dict:
    # general basin-wise fit
    CoefLat_g = pd.read_csv(fit_dir / "dflatCoef.csv", sep=";", decimal=",")
    CoefLon_g = pd.read_csv(fit_dir / "dflonCoef.csv", sep=";", decimal=",")
    CoefLat0 = pd.read_csv(fit_dir / "dflatnormCoeff.csv", sep=";")
    CoefLon0 = pd.read_csv(fit_dir / "dflonnormCoeff.csv", sep=";", decimal=",")
    Residual_lat_g = pd.read_csv(
        fit_dir / "dflatresnormCoeff.csv", sep=";", decimal=","
    )
    Residual_lon_g = pd.read_csv(
        fit_dir / "dflonresnormCoeff.csv", sep=";", decimal=","
    )
    # grouped value
    CoefLat = pd.read_csv(fit_dir / "dflatCoef_group.csv", sep=";", decimal=",")
    CoefLon = pd.read_csv(fit_dir / "dflonCoef_group.csv", sep=";", decimal=",")
    Residual_lat = pd.read_csv(
        fit_dir / "dflatresnormCoeff_group.csv", sep=";", decimal=","
    )
    Residual_lon = pd.read_csv(
        fit_dir / "dflonresnormCoeff_group.csv", sep=";", decimal=","
    )

    # make sure north atlantic basin is well understood
    CoefLat_g.BASIN = CoefLat_g.BASIN.fillna("NA")
    CoefLon_g.BASIN = CoefLon_g.BASIN.fillna("NA")
    CoefLat0.BASIN = CoefLat0.BASIN.fillna("NA")
    CoefLon0.BASIN = CoefLon0.BASIN.fillna("NA")
    Residual_lat_g.BASIN = Residual_lat_g.BASIN.fillna("NA")
    Residual_lon_g.BASIN = Residual_lon_g.BASIN.fillna("NA")

    # make sure required fields are in numeric type
    CoefLat0 = CoefLat0.replace(",", ".", regex=True)
    CoefLat0.mod_est_sd = pd.to_numeric(CoefLat0.mod_est_sd)
    CoefLon0 = CoefLon0.replace(",", ".", regex=True)
    CoefLon0.mod_est_sd = pd.to_numeric(CoefLon0.mod_est_sd)
    Residual_lat = Residual_lat.replace(",", ".", regex=True)
    Residual_lat.sd = pd.to_numeric(Residual_lat.sd)
    Residual_lon = Residual_lon.replace(",", ".", regex=True)
    Residual_lon.sd = pd.to_numeric(Residual_lon.sd)

    return {
        "CoefLon": CoefLon,
        "CoefLat": CoefLat,
        "CoefLon0": CoefLon0,
        "CoefLat0": CoefLat0,
        "CoefLat_g": CoefLat_g,
        "CoefLon_g": CoefLon_g,
        "Residual_lon": Residual_lon,
        "Residual_lat": Residual_lat,
        "Residual_lat_g": Residual_lat_g,
        "Residual_lon_g": Residual_lon_g,
    }
