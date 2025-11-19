import sqlite3
import warnings
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd


class DamageFunction:
    """Damage functions for tropical cyclones.
    Vulnerability curves are available for 9 different regions based on (Ebenrez et al., 2021).
    """

    def __init__(
        self, method: str, vuln_db_path: Path, vuln2assets_country_map: dict = None
    ):
        self.method = method
        if method not in (valid_methods := ["TDR1.0", "TDR1.5", "RMSF"]):
            raise ValueError(
                f"The requested method {method} is not supported. "
                f"Valid methods include: {valid_methods}"
            )
        self.vuln_db_path: Path

        self.vuln = (
            self.get_vulnerability_curves(vuln_db_path)
            .drop_duplicates()  
            .reset_index(drop=True)
        )
        self.vhalf = self.vuln.loc[self.vuln["ID"] == f"GLB{method}", "v_half"].iloc[0]
        self.vuln2assets_country_map = vuln2assets_country_map

    def compute_damage_fraction(
        self, assets: gpd.GeoDataFrame, hazards: gpd.GeoDataFrame
    ) -> np.ndarray:
        if isinstance(assets, pd.DataFrame):
            assets = gpd.GeoDataFrame(
                assets,
                geometry=gpd.points_from_xy(assets.lon, assets.lat),
                crs="EPSG:3857",
            )
        if isinstance(hazards, pd.DataFrame):
            hazards = gpd.GeoDataFrame(
                hazards,
                geometry=gpd.points_from_xy(hazards.LON, hazards.LAT),
                crs="EPSG:3857",
            )
        # Add country-specific 'v_half' values for each physical asset
        # That is, the vulneratbility of each asset to a
        vh_df = self.vuln.loc[self.vuln["Method"] == self.method, ["v_half", "country"]]
        assets = pd.merge(
            assets,
            vh_df,
            left_on="iso3_country",
            right_on="country",
            how="left",
        )
        # Assign default vulnerability value to assets in countries for which
        # v_half was not estimated
        assets = assets.fillna({"v_half": self.vhalf})

        hazards = hazards.rename(columns={"wind": "v"})

        assets = gpd.sjoin_nearest(
            assets,
            hazards[["geometry", "v"]],
            how="left",
            distance_col="dist_nearest",
        )

        assets["damage_fraction"] = emanuel_2011(v=assets["v"], v_half=assets["v_half"])
        return assets

    @staticmethod
    def get_vulnerability_curves(catherina_fit_path: Path) -> pd.DataFrame:
        with sqlite3.connect(catherina_fit_path) as conn:
            # Read the SQL query into a pandas DataFrame
            vuln = pd.read_sql_query("SELECT * FROM CLIMADA_damage_funcs", conn)
        return vuln


def emanuel_2011(
    v: Union[float, np.ndarray],
    v_half: Union[float, np.ndarray] = 74.7,
    v_thresh: float = 25.7,
) -> Union[float, np.ndarray]:
    """
    Calculate the fractional damage of a property based on wind speed using: Emanuel, K. A.: Global warming effects on US hurricane damage (2011) damage function.

    The function computes the fraction of damage as a cubic function of the normalized wind speed exceeding a threshold.
    This fraction represents the proportion of the property value lost due to wind damage.

    Parameters:
    -----------
    - v (float): The sustained wind speed (in m/s) for which to calculate the damage.
    - v_thresh (float, optional): The threshold wind speed (in m/s) below which there is no damage. This value is fixed for all regions.
    Defaults to 25.7 m/s (threshold for the USA (cf. Emanuel, 2011) which was empirically supported for China).
    - v_half (float, optional): The hypothetical wind speed (in m/s) at which the relative impact reaches 50% of the exposed asset value.
    This value can change by region based on calibration from (Ebenrez, 2021).
    Defaults to 74.7 m/s, which corresponds to the mean value for US hurricanes (cf. Sealy and Strobl (2017)).

    Quote from Ebenrez:
    'While v_half is fitted during the calibration process, the lower threshold v_thresh is kept constant throughout the study.
    This is based on the finding by Lüthi (2019) that the variation in more than one of the linearly dependent parameters most
    likely results in an overfitting during calibration with physically implausible values for v_thresh in some world regions.

    On the chosen 10 km x 10 km grid, single buildings are not resolved. Therefore, damage is aggregated over several buildings
    in a grid cell, and not all buildings are expected to be damaged to the same degree. However, the wind-speed-dependent
    impact function is also implicitly accounting for the damage caused by storm surges and torrential rain when calibrated
    against reported damage data. For these two reasons, we allow for values of Vhalf lower and larger than the literature
    range for pure wind-induced building damage in the calibration.'

    'In summary, the nine calibration regions are the Caribbean with Central America and Mexico (NA1), the USA and Canada (NA2),
    North Indian Ocean (NI), Oceania with Australia (OC), South Indian Ocean without Australia (SI), South East Asia (WP1),
    the Philippines (WP2), mainland China (WP3), and the north West Pacific (WP4).'
    See Table A1 to download the list of countries per calibration region.

    Bibliography
    [Emanuel, 2011 - Global Warming Effects on U.S. Hurricane Damage](https://texmex.mit.edu/pub/emanuel/PAPERS/wcas_2011.pdf)
    [Ebenrez et al., 2021 - Regional tropical cyclone impact functions fro globally consistent risk assessments](https://nhess.copernicus.org/articles/21/393/2021/)
    [Elliott et al., 2015 - The local impact of typhoons on economic activity in China: a view from outer space](https://www.sciencedirect.com/science/article/abs/pii/S0094119015000340)

    Returns:
    ----------
    - float: A value between 0 and 1 representing the fractional damage to the property.
    """
    if isinstance(v_half, float):
        if v_half == 74.7:
            warnings.warn(
                "The default value for 'v_half' has not been changed. "
                "The current value was calibrated on US tropical cyclones.",
                UserWarning,
            )

    v_n = np.maximum(v - v_thresh, 0) / (v_half - v_thresh)
    D = v_n**3 / (1 + v_n**3)
    return D


if __name__ == "__main__":
    catherina_fit_path = Path(
        "C:/Users/hamada.saleh/Documents/datasets/catherina_sectoral_dataset_refactor/input/Catherina_fit.db"
    )
    # hazard
    # tracks_dir = Path(
    #     "C:/Users/hamada.saleh/Documents/datasets/catherina_sectoral_dataset_refactor/outputs/intensified_tracks/access_cm2/ssp5_8_5"
    # ).glob("*.csv")
    # tracks_fpath = next(tracks_dir)
    # tracks = pd.read_csv(tracks_fpath)
    # assets
    # assets_path = Path("D:/datasets_physical_risk/steel_emissions_sources.csv")
    # assets = pd.read_csv(assets_path).drop_duplicates(subset="iso3_country")
    method = "RMSF"
    dfunc = DamageFunction(method=method, vuln_db_path=catherina_fit_path)
    # damages = dfunc.compute_damage_fraction(assets=assets, hazards=tracks)
