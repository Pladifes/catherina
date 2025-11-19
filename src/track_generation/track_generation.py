import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pyarrow.compute as pc

from pathlib import Path
import geopandas as gpd
from loguru import logger

from tqdm import tqdm

from src.track_generation.track_generation_utils import (get_fit_data,
                                                         preprocess_delta_coef,
                                                         preprocess_resid_params)

def simulate_tc_tracks(
    genesis_ds,
    catherina_fit_path,
    max_steps: int,
    seeds: "Iterable",
    logfile: str,
    save_dir: Path = None
) -> gpd.GeoDataFrame:
    # Logger
    logger.remove()
    # Add a file sink for debugging logs
    logger.add(
        logfile, rotation="10 MB", enqueue=True
    )  # Rotates file when it reaches 10MB
    logger.debug(seeds)

    fit_data = get_fit_data(fit_dir=catherina_fit_path)

    coeflat = preprocess_delta_coef(fit_data["CoefLat"])
    coeflon = preprocess_delta_coef(fit_data["CoefLon"])
    lat_resids_params = preprocess_resid_params(fit_data=fit_data, coord="lat")
    lon_resids_params = preprocess_resid_params(fit_data=fit_data, coord="lon")

    scanner = pds.Scanner.from_dataset(
    genesis_ds,
    filter=(
        (pc.field("seed").isin(seeds))),
    )
    genesis_ds_filter = scanner.to_table().to_pandas()

    logger.info("Simulating TC displacement...")
    tracks = simulate_tc_displacement(
        synth_genesis=genesis_ds_filter,
        lon_resids_params=lon_resids_params,
        lat_resids_params=lat_resids_params,
        coeflat=coeflat,
        coeflon=coeflon,
        max_steps=max_steps,
    )
    
    table = pa.Table.from_pandas(tracks.drop(columns="geometry"))
    pq.write_to_dataset(
        table=table,
        root_path=save_dir,  # output path
        partition_cols=["seed", "year", "month"],  # Hive-style columns
        existing_data_behavior="overwrite_or_ignore",  # optional: clean write
    )


def simulate_tc_displacement(
    synth_genesis, 
    lon_resids_params, 
    lat_resids_params, 
    coeflat, 
    coeflon, 
    max_steps
):
    synth_genesis["index"] = range(len(synth_genesis))

    synth_genesis = gpd.GeoDataFrame(
        synth_genesis,
        geometry=gpd.points_from_xy(
            synth_genesis.lon, synth_genesis.lat, crs="WGS 84"
        ),
    )

    lon_resids_params = gpd.GeoDataFrame(
        lon_resids_params,
        geometry=gpd.points_from_xy(
            lon_resids_params.lon, lon_resids_params.lat, crs="WGS 84"
        ),
    )
    lat_resids_params = gpd.GeoDataFrame(
        lat_resids_params,
        geometry=gpd.points_from_xy(
            lat_resids_params.lon, lat_resids_params.lat, crs="WGS 84"
        ),
    )
    _synth_genesis = synth_genesis.rename(
        columns={
            "lat": "lat_0",
            "lon": "lon_0",
        }
    )

    for step in tqdm(list(range(1, max_steps)), desc="Add displacement steps", ascii=' >=', leave=False, position=2):
        _synth_genesis = update_coords(
            step=step,
            synth_genesis=_synth_genesis,
            lat_resids_params=lat_resids_params,
            lon_resids_params=lon_resids_params,
            coeflat=coeflat,
            coeflon=coeflon,
        )
        # Set geometry column based on updated coords
        # so that next displacement is based on new location
        _synth_genesis = _synth_genesis.set_geometry(
            gpd.points_from_xy(
                x=_synth_genesis[f"lon_{step}"], y=_synth_genesis[f"lat_{step}"]
            )
        )
    _synth_genesis = _synth_genesis.drop(
        columns=[f"delta_lon_{step}", f"delta_lat_{step}"]
    )

    #Modify data structure
    # Melt cyclone tracks
    to_keep = ["seed", "SID", "year", "month"] + _synth_genesis.columns[
        _synth_genesis.columns.str.contains(r"^(?:lat_|lon_)")
    ].tolist()
    tracks = _synth_genesis.loc[:, to_keep]
    tracks_melt = tracks.melt(
        id_vars=["seed", "SID", "year", "month"],
        var_name="variable",
        value_name="value",
    )
    tracks_melt[["coord", "step"]] = tracks_melt["variable"].str.split("_", expand=True)
    tracks_pivot = tracks_melt.pivot(
        index=["seed", "SID", "year", "month", "step"], columns="coord", values="value"
    ).reset_index()
    tracks_pivot = gpd.GeoDataFrame(
        tracks_pivot,
        geometry=gpd.points_from_xy(
            x=tracks_pivot["lon"], y=tracks_pivot["lat"], crs="WGS 84"
        ),
    )
    tracks_pivot = tracks_pivot.astype({"step": int}).sort_values(
        ["seed", "SID", "year", "month", "step"], ascending=True
    )

    # Add datetime
    # Assume time delta between displacements to be 3 hours
    tracks_pivot["datetime"] = pd.to_datetime(
        tracks_pivot[["year", "month"]].assign(day=1)
    ) + pd.to_timedelta(tracks_pivot["step"] * 3, unit="h")

    #Here we delet the "bad" trakcs: above lat 60, below lat -60, corssing equator, stale
    tracks_pivot_corrected = correct_tracks(tracks_pivot)

    return tracks_pivot_corrected


def update_coords(
    step: int,
    synth_genesis: gpd.GeoDataFrame,
    lat_resids_params: gpd.GeoDataFrame,
    lon_resids_params: gpd.GeoDataFrame,
    coeflat: gpd.GeoDataFrame,
    coeflon: gpd.GeoDataFrame,
) -> pd.DataFrame:
    # Sjoin regression params
    synth_genesis = sjoin_reg_params(
        synth_genesis=synth_genesis,
        lat_resids_params=lat_resids_params,
        lon_resids_params=lon_resids_params,
        coeflat=coeflat,
        coeflon=coeflon,
    )
    # Simulate residuals
    synth_genesis = (
        synth_genesis.groupby("seed")
        .apply(lambda g: g.assign(lat_eps=sim_eps(g, "lat"), lon_eps=sim_eps(g, "lon")))
        .reset_index(drop=True)
    )
    # Compute cyclone movement one step forward
    # Each column represent a coefficient from the nonlinear regression
    # Update changes in coordinates
    if step == 1:
        synth_genesis["delta_lat_0"] = 0.0
        synth_genesis["delta_lon_0"] = 0.0

    # TODO: normalise letter case (lower)
    for l in ["LON", "LAT"]:
        synth_genesis[f"delta_{l.lower()}_{step}"] = (
            synth_genesis[f"{l}_intercept"]  # a_0 / b_0
            + synth_genesis[f"{l}_diff_prev"]
            * synth_genesis[
                f"delta_{l.lower()}_{step-1}"
            ]  # a_1 * \Delta \xi_{t-1} / b_1 * \Delta \phi_{t-1}
        )
        if l == "LAT":
            synth_genesis[f"delta_{l.lower()}_{step}"] += synth_genesis[f"{l}_inv"] * (
                1.0 / synth_genesis[f"{l.lower()}_{step-1}"]
            )  # b_2 / \phi_t

        # TODO: also fillna with group values for missing coefs
        # TODO: merge gaussian params to each location
        # TODO: fillna with group values

    # Update coordinates
    synth_genesis = synth_genesis.assign(
        **{
            f"lon_{step}": synth_genesis[f"lon_{step-1}"]
            + synth_genesis[f"delta_lon_{step}"]
            + synth_genesis["lon_eps"],
            f"lat_{step}": synth_genesis[f"lat_{step-1}"]
            + synth_genesis[f"delta_lat_{step}"]
            + synth_genesis["lat_eps"],
        }
    )

    # Update geometry based on new coordinates
    synth_genesis.set_geometry(
        gpd.points_from_xy(
            x=synth_genesis[f"lon_{step}"], y=synth_genesis[f"lat_{step}"], crs="WGS 84"
        ),
        inplace=True,
    )
    # Drop parameters from the previous location
    synth_genesis = synth_genesis.drop(
        columns=[
            "lat_mean",
            "lat_sd",
            "lon_mean",
            "lon_sd",
            "LAT_intercept",
            "LAT_diff_prev",
            "LAT_inv",
            "LON_intercept",
            "LON_diff_prev",
            f"delta_lat_{step-1}",
            f"delta_lon_{step-1}",
            "lat_eps",
            "lon_eps",
        ]
    )
    return synth_genesis


def correct_tracks(tracks : pd.DataFrame) -> pd.DataFrame:
    # --- params ---
    tol_lon = 0.1   # degrees: "almost equal" longitude tolerance
    tol_lat = 0.1   # degrees: "almost equal" latitude tolerance
    min_run = 11     # consecutive points (e.g., >10 means >=11)
    eq_band = 0.0    # set >0 if you want a deadband around the equator, e.g. 0.1°

    # Flag SIDs that ever go out of bounds
    out_of_bounds_sids = tracks.loc[(tracks["lat"] > 60) | (tracks["lat"] < -60), "SID"].unique()

    # Ensure ordering
    df = tracks.sort_values(['SID', 'step']).copy()

    # --- helper: consecutive-run lengths where |diff| <= tol ---
    def consecutive_runs_within_tol(s: pd.Series, tol: float) -> pd.Series:
        # whether current value is "equal" (within tol) to previous
        eq = (s - s.shift()).abs().le(tol)
        # start a new run when not equal (or at start)
        run_id = (eq.fillna(False) == False).cumsum()
        # length of each run
        run_len = s.groupby(run_id).transform('size')
        # only count runs where condition is True; else set to 0
        return np.where(eq.fillna(False), run_len, 0).max()

    # Run lengths for lon/lat with tolerance, per SID
    df['lon_run_len'] = df.groupby('SID')['lon'].apply(lambda s: consecutive_runs_within_tol(s, tol_lon)).reset_index(level=0, drop=True)
    df['lat_run_len'] = df.groupby('SID')['lat'].apply(lambda s: consecutive_runs_within_tol(s, tol_lat)).reset_index(level=0, drop=True)

    # SIDs with long "flat" runs in lon or lat
    bad_flat_lon = df.loc[df['lon_run_len'] >= min_run, 'SID'].unique()
    bad_flat_lat = df.loc[df['lat_run_len'] >= min_run, 'SID'].unique()

    # SIDs that cross the equator
    if eq_band > 0:
        # consider a band around zero as "equator"
        crosses_eq = (
            (df.groupby('SID')['lat'].min() < -eq_band) &
            (df.groupby('SID')['lat'].max() >  eq_band)
        )
    else:
        crosses_eq = (
            (df.groupby('SID')['lat'].min() < 0) &
            (df.groupby('SID')['lat'].max() > 0)
        )
    bad_cross_eq = crosses_eq.index[crosses_eq].to_numpy()

    # Union of all bad SIDs
    bad_sids = np.unique(np.concatenate([bad_flat_lon, bad_flat_lat, bad_cross_eq,out_of_bounds_sids]))

    # Filter out whole tracks
    corrected_tracks = df.loc[~df['SID'].isin(bad_sids)].drop(columns=['lon_run_len','lat_run_len'])
    return corrected_tracks



# TODO: generate a dict of rngs where each key is a seed int and value np.Generator
def sim_eps(gp: "Groupby", coord: str) -> list:  # TODO: add dict arg for all rngs
    """_summary_

    Args:
        gp (Groupby):
        coord (str): lat or lon

    Returns:
        _type_: _description_
    """
    seed = gp["seed"].iloc[0]
    rng = np.random.default_rng(seed)
    coord_params = gp[[f"{coord}_mean", f"{coord}_sd"]].itertuples(
        index=False, name=None
    )
    return [rng.normal(loc=loc, scale=scale) for loc, scale in coord_params]


def sjoin_reg_params(
    synth_genesis: gpd.GeoDataFrame,
    lat_resids_params: gpd.GeoDataFrame,
    lon_resids_params: gpd.GeoDataFrame,
    coeflat: gpd.GeoDataFrame,
    coeflon: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    synth_genesis = sjoin_coefs_resids(
        synth_genesis=synth_genesis, coefs_resids=lat_resids_params
    )
    synth_genesis = sjoin_coefs_resids(
        synth_genesis=synth_genesis, coefs_resids=lon_resids_params
    )
    synth_genesis = sjoin_coefs_resids(
        synth_genesis=synth_genesis, coefs_resids=coeflon
    )
    synth_genesis = sjoin_coefs_resids(
        synth_genesis=synth_genesis, coefs_resids=coeflat
    )
    return synth_genesis


def sjoin_coefs_resids(
    synth_genesis: gpd.GeoDataFrame, coefs_resids: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    cols = ["month", "geometry"] + coefs_resids.columns[
        coefs_resids.columns.str.contains("_intercept|_diff_prev|_inv|_mean|_sd")
    ].tolist()
    toto = synth_genesis.to_crs("EPSG: 3857").sjoin_nearest(
        coefs_resids.loc[:, cols].to_crs("EPSG: 3857"),
        how="left",
        distance_col="distance_meters",
    )
    toto["month_flag"] = toto["month_left"] == toto["month_right"]
    toto = (
        toto.sort_values(by=["index", "month_flag"], ascending=[True, False])
        .rename(columns={"month_left": "month"})
        .drop(columns=["month_right", "index_right"])
        .drop_duplicates(["index"], keep="first")
    )  # TODO: replace temp index

    logger.debug(
        f"Genesis points with mismatched months: {(1 - toto['month_flag']).sum()}"
    )
    return toto.reset_index(drop=True)