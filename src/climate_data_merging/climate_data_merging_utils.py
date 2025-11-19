import pandas as pd
from pathlib import Path
import sqlite3


def get_catherina_coefs(catherina_fit_path: Path) -> pd.DataFrame:
    # Global coefficients for wpr relationship
    with sqlite3.connect(catherina_fit_path) as conn:
        wpr_global_coefs = pd.read_sql_query(
            "SELECT * FROM FIT_WPRCoef", conn
        )  # TODO: is this used?
        wpr_basin_coefs = pd.read_sql_query(
            "SELECT * FROM FIT_WPRCoef_basin", conn
        ).pivot(index="group", columns="term", values="estimate")
        sst_mpd_coefs = (
            pd.read_sql_query("SELECT * FROM FIT_SSTMPDCoef", conn)
            .replace({"term": {".lin1": "A", ".lin2": "B"}})
            .pivot(index="BASIN", columns="term", values="estimate")
            .rename_axis("basin", axis=0)
        )
        pres_dyn_coefs = pd.read_sql_query(
            "SELECT term, estimate FROM FIT_dfpresCoef_thermo", conn, index_col="term"
        ).T
        pres_dyn_resids_std = (
            pd.read_sql_query(
                "SELECT term, estimate FROM FIT_dfpres_res_distrib_thermo",
                conn,
                index_col="term",
            )
            .T["sd"]
            .iloc[0]
        )
        pres_dyn_coefs["sigma_pc"] = pres_dyn_resids_std
        # Merge all coefs
        cath_coefs = sst_mpd_coefs.merge(
            wpr_basin_coefs, left_index=True, right_index=True
        )
        cath_coefs = cath_coefs.assign(**pres_dyn_coefs.iloc[0].to_dict())
        return cath_coefs
