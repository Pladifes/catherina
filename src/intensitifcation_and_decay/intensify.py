import pandas as pd
import numpy as np
import geopandas as gpd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger
from shapely.prepared import prep
from shapely import STRtree
from pyproj import Transformer



from src.genesis.genesis_utils import get_basins_poly

from src.intensitifcation_and_decay.intensify_utils import (X,
                                                            delta_Sm,
                                                            delta_pc,
                                                            mpd,
                                                            mpi,
                                                            qc,
                                                            qenv,
                                                            wpr,
                                                            wpr_inverse_pc,
                                                            get_dist_to_coast,
                                                            get_is_on_land_fast)

from src.climate_data_merging.climate_data_merging_utils import get_catherina_coefs

_LAND_UNION = None
_COAST_GDF  = None
_BASINS_4326 = None
_COAST_TREE = None


def _ensure_geo(ne_10m_land_zip, ne_10m_coastline_zip):
    global _LAND_UNION, _COAST_GDF, _BASINS_4326, _COAST_TREE
    if _LAND_UNION is None:
        land_gdf = gpd.read_file(ne_10m_land_zip).to_crs(4326)
        _LAND_UNION = prep(land_gdf.union_all())
    if _COAST_GDF is None:
        _COAST_GDF = gpd.read_file(ne_10m_coastline_zip).to_crs(4326)
    if _COAST_TREE is None:
        _COAST_TREE = STRtree(_COAST_GDF.geometry.values)
    if _BASINS_4326 is None:
        _BASINS_4326 = get_basins_poly(crs="EPSG:4326") 

def intensify_and_save(
    batch_seeds:list,
    corrected_tracks_dir: str, 
    catherina_fit_path: str,
    ne_10m_coastline_zip :str,
    ne_10m_land_zip: str,
    save_dir:str
) -> None:

    #Load land and coast data:
    _ensure_geo(ne_10m_land_zip, ne_10m_coastline_zip)
    
    # Re-load the dataset in each process (necessary for multiprocessing)

    dataset = pds.dataset(corrected_tracks_dir, format="parquet", partitioning="hive")
    part_fields = {f.name: f.type for f in dataset.partitioning.schema}

    #year_f = pc.cast(pc.field("year"), pa.int32()) if part_fields.get("year") == pa.string() else pc.field("year")
    #month_f = pc.cast(pc.field("month"), pa.int32()) if part_fields.get("month") == pa.string() else pc.field("month")
    seed_f = pc.cast(pc.field("seed"), pa.int64())  if part_fields.get("seed") == pa.string() else pc.field("seed")

    filt = seed_f.isin(batch_seeds)

    #Avoid validity checks in the Scanner if columns may be absent; filter later
    scanner = pds.Scanner.from_dataset(dataset, filter=filt)

    df = scanner.to_table().to_pandas()

    #Add basin
    gdf = gpd.GeoDataFrame(df,
                           geometry=gpd.points_from_xy(df.lon_left, df.lat_left),
                           crs=4326)
    ############# ADDED #################
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    print("==============================")
    print(gdf.columns.to_list())
    print("==============================")
    print(_BASINS_4326.columns.to_list())
    print("==============================")
    ###################################################
    gdf = gdf.sjoin(_BASINS_4326, 
                    how="left", 
                    predicate="within").drop(columns="index_right").\
                        dropna(subset=["basin"])

    #Get land
    #land = gpd.read_file(ne_10m_land_zip).union_all()

    #RH: add basin
    # df = gpd.GeoDataFrame(
    # df, geometry=gpd.points_from_xy(df.lon_left, df.lat_left), crs="EPSG:3857"
    # )
    # basins_poly = get_basins_poly(crs="EPSG:3857")
    # df = df.sjoin(basins_poly, how="left", predicate="within").drop(
    #     columns="index_right"
    # )
    # df = df.dropna(subset="basin")

    

    # Web Mercator is fine for local distances if not near poles; for robustness use equal-area/UTM per lat band if needed.

    if not gdf.empty:
        # Add wind speed + features for decay when on land
        gdf = preprocessing_cyclone(track=gdf, catherina_fit_path=catherina_fit_path)

        #df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df["lon_left"], y=df["lat_left"]), crs="EPSG:4326")
        gdf["is_on_land"] = get_is_on_land_fast(gdf.geometry,
                                               _LAND_UNION)
        gdf = get_dist_to_coast(
                intersect_gdf=gdf, ne_10m_coastline_zip=ne_10m_coastline_zip
        )
        #back to dataframe
        df = pd.DataFrame(gdf)

        #Intensify and decay
        df = intensify_track_forward(df)

        #Save
        pq.write_to_dataset(
            table=pa.Table.from_pandas(df.drop(columns='geometry')),
            root_path=save_dir,
            partition_cols=["seed","year","month"],
            existing_data_behavior="overwrite_or_ignore",
        )
        del df
    else:
        logger.info(f"No tracks to intensify for seeds: {batch_seeds}")

def preprocessing_cyclone(track: pd.DataFrame, catherina_fit_path) -> pd.DataFrame:
    # Merge WPR coefficients
    cath_coefs = get_catherina_coefs(catherina_fit_path=catherina_fit_path)
    track = (
        pd.merge(track.reset_index(), cath_coefs, on=["basin"], how="left")
        .rename(columns={"MSLP": "mslp_hpa"})
        .set_index(["seed", "SID", "step"])
    )

    # compute specific humidity
    track["q_env"] = qenv(
        rh=track["nshr"],
        mslp_hpa=track["mslp_hpa"],
        sst_kelvin=track["SST"] + 273.15,
    )
    mask_step0 = track.index.get_level_values("step") == 0
    track.loc[mask_step0, "wind_speed"] = 20.0  # m.s^-1
    track.loc[mask_step0, "delta_pc_hpa"] = 0
    track.loc[mask_step0, "pc_hpa"] = wpr_inverse_pc(
        a=track.loc[mask_step0, "a"],
        mslp_hpa=track.loc[mask_step0, "mslp_hpa"],
        wind_speed=track.loc[mask_step0, "wind_speed"],
        b=track.loc[mask_step0, "b"],
    )

    # TODO: replace with random number generation
    # Generate random normal error column eps_pc
    track = track.assign(eps_pc=np.random.normal(loc=0, scale=track["sigma_pc"]))
    return track


def intensify_track_forward(
    tracks: pd.DataFrame,
    max_steps: int = 175,
    dt_hours: int = 3,
    # --- land-decay parameters (set to your calibrated values) ---
    V_b: float = 15,
    R: float = 0.79,
    alpha: float = 0.044,   # 1/hour
    D_0: float = 1,     # km
    c1_tilde: float =  3.35 * 1e-4,
    t_0_L: float = 172,   # hours
    d1: float = -0.00186,
    min_land_steps: int = 3,
) -> pd.DataFrame:
    """
    Forward integration with land-decay + continuity and pressure overrides.

    - While on land (and after >= min_land_steps), wind follows exponential decay.
    - On the first step back over water, wind = previous step's final wind (continuity).
    - While on land, CURRENT pressure pc_hpa is computed from wind via wpr_inverse
      (not from climate/recurrence).
    - On exit (land->water), previous-step effective pressure pc_hpa_1_eff is
      adjusted via wpr_inverse using previous final wind for dynamic consistency.
    """

    tracks = tracks[~tracks.index.duplicated(keep="first")].copy()

    def at_step(i):
        return lambda x: x.index.get_level_values("step") == i

    # --- helpers for land decay ---
    def _f1(tLh):
        return c1_tilde * tLh * (t_0_L - tLh)

    def _f2(tLh):
        return d1 * tLh * (t_0_L - tLh)

    def _land_decay(V0, tL_hours, dist2coast_m):
        # meters -> km for the log term, and guard against log(0)
        dist_km = pd.Series(dist2coast_m, index=V0.index).astype(float).clip(lower=1e-6) / 1000.0
        tLh = pd.Series(tL_hours, index=V0.index).astype(float)
        V0s = pd.Series(V0, index=V0.index).astype(float)
        return (
            V_b
            + (R * V0s - V_b) * np.exp(-alpha * tLh)
            - _f1(tLh) * np.log(dist_km / D_0)
            + _f2(tLh)
        )

    # State carried across steps
    tracks["V_0"] = np.nan              # landfall wind for current on-land episode
    tracks["t_L_steps"] = np.nan        # consecutive on-land step count
    tracks["final_wind_speed"] = np.nan # wind after overrides
    # We'll keep base wind optionally if you want to inspect it:
    # tracks["wind_speed_base"] = np.nan

    stop_step = {}  # dict of tuples to int

    global_max_step = int(tracks.index.get_level_values("step").max())

    for i in range(1, global_max_step+1):
        step_i_full = tracks.xs(i, level="step", drop_level=False)
        if not len(step_i_full):
            continue

        # keep only storms not yet stopped
        active_idx = []
        for idx in step_i_full.index:
            seed = idx[tracks.index.names.index("seed")] if "seed" in tracks.index.names else idx[0]
            SID  = idx[tracks.index.names.index("SID")]  if "SID"  in tracks.index.names else idx[1]
            if (seed, SID) not in stop_step:
                active_idx.append(idx)
        if not active_idx:
            continue
        step_i = step_i_full.loc[active_idx]
        idx_i  = step_i.index
        mask_i = tracks.index.isin(idx_i)

        # ---- shifts from previous step (add shift(2) for rolling-3) ----
        tracks["delta_pc_hpa_1"] = tracks.groupby(level=['seed','SID'])["delta_pc_hpa"].shift(1)
        tracks["pc_hpa_1"]       = tracks.groupby(level=['seed','SID'])["pc_hpa"].shift(1)
        tracks["final_wind_1"]   = tracks.groupby(level=['seed','SID'])["final_wind_speed"].shift(1)
        tracks["final_wind_2"]   = tracks.groupby(level=['seed','SID'])["final_wind_speed"].shift(2)  # NEW
        prev_is_land             = (tracks["is_on_land"].
                                    astype("boolean").
                                    groupby(level=['seed','SID']).
                                    shift(1).fillna(False))
        tracks["V_0_1"]          = tracks.groupby(level=['seed','SID'])["V_0"].shift(1)
        tracks["t_L_steps_1"]    = tracks.groupby(level=['seed','SID'])["t_L_steps"].shift(1)

        # Effective previous-step pressure (we may adjust this on exit)
        tracks["pc_hpa_1_eff"] = tracks["pc_hpa_1"]

        # Initial recurrence for current pressure (may be overridden on land)
        tracks.loc[mask_i, "pc_hpa"] = (
            tracks.loc[mask_i, "delta_pc_hpa_1"] +
            tracks.loc[mask_i, "pc_hpa_1_eff"]
        )

        # Base/unconstrained wind for current step
        base_wind = wpr(
            a=step_i["a"],
            mslp_hpa=step_i["mslp_hpa"],
            pc_hpa=tracks.loc[idx_i, "pc_hpa"],
            b=step_i["b"],
        )
        tracks.loc[idx_i, "final_wind_speed"] = base_wind

        # --- LAND DECAY STATE (V_0 and t_L_steps) ---
        is_land_now   = step_i["is_on_land"].astype(bool)
        was_land_prev = prev_is_land.loc[idx_i].astype(bool)

        # Landfall: False -> True  (set V_0 from base wind at landfall)
        landfall_mask = is_land_now & (~was_land_prev)
        if landfall_mask.any():
            tracks.loc[landfall_mask.index[landfall_mask], "V_0"] = base_wind.loc[landfall_mask]

        # Continuing on land: carry V_0 forward
        cont_land_mask = is_land_now & was_land_prev
        if cont_land_mask.any():
            tracks.loc[cont_land_mask.index[cont_land_mask], "V_0"] = tracks.loc[cont_land_mask.index[cont_land_mask], "V_0_1"]

        # Off land now: clear V_0
        off_land_mask = (~is_land_now)
        if off_land_mask.any():
            tracks.loc[off_land_mask.index[off_land_mask], "V_0"] = np.nan

        # Update consecutive on-land steps: landfall -> 0, continue -> +1, off-land -> NaN
        if landfall_mask.any():
            tracks.loc[landfall_mask.index[landfall_mask], "t_L_steps"] = 0
        if cont_land_mask.any():
            prev_counts = tracks.loc[cont_land_mask.index[cont_land_mask], "t_L_steps_1"].fillna(0).astype(float)
            tracks.loc[cont_land_mask.index[cont_land_mask], "t_L_steps"] = prev_counts + 1
        if off_land_mask.any():
            tracks.loc[off_land_mask.index[off_land_mask], "t_L_steps"] = np.nan

        # --- APPLY OVERRIDES: (optional cap) -> land decay -> exit continuity ---
        # 2) Land decay (after >= min_land_steps)
        tL_steps = tracks.loc[idx_i, "t_L_steps"]
        decay_mask = is_land_now & (tL_steps >= min_land_steps)
        if decay_mask.any():
            tL_hours = (tL_steps[decay_mask] * dt_hours).astype(float)
            V0_now   = tracks.loc[decay_mask.index[decay_mask], "V_0"].astype(float)
            d2c_m    = step_i.loc[decay_mask, "dist2coast_meters"].astype(float)
            decayed  = _land_decay(V0_now, tL_hours, d2c_m)
            tracks.loc[decay_mask.index[decay_mask], "final_wind_speed"] = decayed

        # 3) Continuity at exit (first off-land after on-land): wind = previous final wind
        exit_mask = (~is_land_now) & (was_land_prev)
        if exit_mask.any():
            tracks.loc[exit_mask.index[exit_mask], "final_wind_speed"] = tracks.loc[exit_mask.index[exit_mask], "final_wind_1"]
            # Adjust previous-step effective pressure to be consistent with carried wind
            pc_prev_adj = wpr_inverse_pc(
                a=step_i.loc[exit_mask, "a"],
                mslp_hpa=step_i.loc[exit_mask, "mslp_hpa"],
                wind_speed=tracks.loc[exit_mask.index[exit_mask], "final_wind_1"],
                b=step_i.loc[exit_mask, "b"]
            )
            tracks.loc[exit_mask.index[exit_mask], "pc_hpa_1_eff"] = pc_prev_adj
            # Update current pc_hpa to reflect new pc_hpa_1_eff + delta_pc_hpa_1
            tracks.loc[exit_mask.index[exit_mask], "pc_hpa"] = (
                tracks.loc[exit_mask.index[exit_mask], "delta_pc_hpa_1"] +
                tracks.loc[exit_mask.index[exit_mask], "pc_hpa_1_eff"]
            )

        # ========= ON-LAND PRESSURE OVERRIDE (current step) =========
        # While on land at step i, set CURRENT pressure from wind via wpr_inverse
        if is_land_now.any():
            tracks.loc[is_land_now.index[is_land_now], "pc_hpa"] = wpr_inverse_pc(
                a=step_i.loc[is_land_now, "a"],
                mslp_hpa=step_i.loc[is_land_now, "mslp_hpa"],
                wind_speed=tracks.loc[is_land_now.index[is_land_now], "final_wind_speed"],
                b=step_i.loc[is_land_now, "b"]
            )
        # ============================================================

        # (Optional) If you also want previous-step pressure consistent WHILE on land,
        # uncomment the following to use the previous final wind for pc_hpa_1_eff:
        # if is_land_now.any():
        #     tracks.loc[is_land_now.index[is_land_now], "pc_hpa_1_eff"] = wpr_inverse(
        #         wind_speed=tracks.loc[is_land_now.index[is_land_now], "final_wind_1"],
        #         a=step_i.loc[is_land_now, "a"],
        #         b=step_i.loc[is_land_now, "b"],
        #         mslp_hpa=step_i.loc[is_land_now, "mslp_hpa"],
        #     )

        # ---- downstream thermodynamics using *pc_hpa_1_eff* ----
        tracks.loc[idx_i, "qc"] = qc(
            pc_shifted_hpa=tracks.loc[idx_i, "pc_hpa_1_eff"],
            sst_kelvin=step_i["SST"] + 273.15,
        )

        tracks.loc[idx_i, "delta_Sm"] = delta_Sm(
            mslp_hpa=step_i["mslp_hpa"],
            pc_shifted_hpa=tracks.loc[idx_i, "pc_hpa_1_eff"],
            qc_star=tracks.loc[idx_i, "qc"],
            q_env=step_i["q_env"],
            sst_kelvin=step_i["SST"] + 273.15,
        )

        tracks.loc[idx_i, "X"] = X(
            sst_kelvin=step_i["SST"] + 273.15,
            T_tropo_kelvin=step_i["T_strat"],
            delta_Sm=tracks.loc[idx_i, "delta_Sm"] ,
            lat_deg=step_i["lat_left"],
        )

        tracks.loc[idx_i, "MPI_hpa"] = mpi(
            mslp_hpa=step_i["mslp_hpa"],
            X=tracks.loc[idx_i, "X"],
        )

        tracks.loc[idx_i, "delta_pc_hpa"] = np.clip(
            delta_pc(
                c1=step_i["c_1"].values,
                c0=step_i["c_0"].values,
                c2=step_i["c_2"].values,
                c3=step_i["c_3"].values,
                eps_P=step_i["eps_pc"].values,
                delta_pc_shifted=tracks.loc[idx_i, "delta_pc_hpa_1"].values,
                pc=tracks.loc[idx_i, "pc_hpa"].values,  # includes land override where applicable
                Y=tracks.loc[idx_i, "MPI_hpa"].values,
            ),
            None, 50.0
        )

        # Central pressure constraint
        MPD_cap_sup = 30
        tracks.loc[idx_i, "mpd_hpa"] = mpd(
            A=tracks.loc[idx_i, "A"],
            B=tracks.loc[idx_i, "B"],
            C=tracks.loc[idx_i, "C"],
            sst_celsius=tracks.loc[idx_i, "SST"],
        ) + MPD_cap_sup
        tracks.loc[idx_i, "mslp_mpd_diff_hpa"] = tracks.loc[idx_i, "mslp_hpa"] - tracks.loc[idx_i, "mpd_hpa"]
        tracks.loc[idx_i, "pc_hpa"] = tracks.loc[idx_i, ["pc_hpa", "mslp_mpd_diff_hpa"]].max(axis=1)

        # Expose reported wind (optional): overwrite or keep separate
        tracks.loc[idx_i, "wind_speed"] = tracks.loc[idx_i, "final_wind_speed"]


        sst_now = tracks.loc[idx_i, "SST"]
        fw_now = tracks.loc[idx_i, "final_wind_speed"]
        fw_1   = tracks.loc[idx_i, "final_wind_1"]
        fw_2   = tracks.loc[idx_i, "final_wind_2"]

        have3  = fw_now.notna() & fw_1.notna() & fw_2.notna()
        roll3  = (fw_now + fw_1 + fw_2) / 3.0

        to_stop_mask = (sst_now < 15) | (fw_now < 5) | (have3 & (roll3 < 10.0))
        if to_stop_mask.any():
                # record stop step i for those storms that haven't been recorded yet
                to_stop_idx = roll3.index[to_stop_mask]
                for idx in to_stop_idx:
                    seed = idx[tracks.index.names.index("seed")] if "seed" in tracks.index.names else idx[0]
                    SID  = idx[tracks.index.names.index("SID")]  if "SID"  in tracks.index.names else idx[1]
                    stop_step.setdefault((seed, SID), i)  # first time elradey excluded

    if stop_step:
        # Build a boolean mask: keep rows where step <= stop for that (seed,SID)
        def _keep_row(index_tuple):
            # index_tuple is (seed, SID, step) in that order per your MultiIndex
            seed = index_tuple[tracks.index.names.index("seed")] if "seed" in tracks.index.names else index_tuple[0]
            SID  = index_tuple[tracks.index.names.index("SID")]  if "SID"  in tracks.index.names else index_tuple[1]
            step = index_tuple[tracks.index.names.index("step")] if "step" in tracks.index.names else index_tuple[-1]

            stp  = stop_step.get((seed, SID))
            return (stp is None) or (step <= stp-1) #We delete the row matching the condition

        keep_mask = tracks.index.map(_keep_row)
        tracks = tracks.loc[keep_mask].copy()

    return tracks