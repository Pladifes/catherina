"""Regrid CMIP6 climate data from the Pangeo API and store the result as Zarr files."""

import json
from collections import defaultdict
from pathlib import Path

import cf_xarray
import dask
import intake
import numpy as np
import xarray as xr
import xesmf as xe
from tqdm import tqdm
from xmip.preprocessing import combined_preprocessing


def clean_cmip_vars(ds_clim):
    """Clean CMIP6 variables and remove unused dimensions and variables."""
    dims_to_drop = {"bnds", "plev"}
    dims_to_drop = dims_to_drop & set(ds_clim.dims)
    if dims_to_drop:
        ds_clim = ds_clim.drop_dims(dims_to_drop)

    vars_to_drop = ["height", "area", "time_bounds", "member_id", "dcpp_init_year"]
    to_drop = set(vars_to_drop).intersection(ds_clim.variables)
    ds_in = ds_clim.drop_vars(to_drop)

    if np.issubdtype(ds_in["y"].dtype, np.floating):
        assert (ds_clim["y"].min() >= -90.0) & (ds_clim["y"].max() <= 90.0)
        ds_in = ds_in.where(ds_in.y < 90, drop=True)
        ds_in = ds_in.where(ds_in.y > -90, drop=True)
        ds_in = ds_in.where(ds_in.x < 360, drop=True)
    elif np.issubdtype(ds_in["y"].dtype, int):
        pass

    return ds_in


def prep_cmip_for_xesmf(ds_in):
    """Prepare a CMIP6 dataset for regridding with xesmf."""
    if np.issubdtype(ds_in["y"].dtype, np.floating):
        if ds_in.y[0] > ds_in.y[-1]:
            ds_in = ds_in.sortby("y")

        for var_name in ["lat_bounds", "lon_bounds", "lat_verticies", "lon_verticies"]:
            if var_name in ds_in:
                ds_in = ds_in.drop_vars(var_name)

        if "vertex" in ds_in.dims:
            ds_in = ds_in.drop_dims("vertex")

        ds_in = ds_in.drop_vars(["lat", "lon"])
        ds_in = ds_in.rename({"y": "lat", "x": "lon"})
        ds_in["lat"].attrs.pop("bounds", None)
        ds_in["lon"].attrs.pop("bounds", None)

    elif np.issubdtype(ds_in["y"].dtype, int):
        assert (ds_in["lat"].ndim == 2) & (ds_in["lon"].ndim == 2)
        if bool({"vertices_lat", "lat_verticies"} & set(ds_in.variables)):
            ds_in = ds_in.rename({"lat_verticies": "lat_b", "lon_verticies": "lon_b"})
        else:
            ds_in = ds_in.cf.add_bounds(["lat", "lon"])
            ds_in = ds_in.rename({"bounds": "vertex", "lat_bounds": "lat_b", "lon_bounds": "lon_b"})

        ds_in["lat_b"], ds_in["lon_b"] = get_xesmf_corners(ds_in["lat_b"], ds_in["lon_b"])
        ds_in["lat"].attrs.update({
            "standard_name": "latitude",
            "units": "degrees_north",
            "bounds": "lat_b",
        })
        ds_in["lon"].attrs.update({
            "standard_name": "longitude",
            "units": "degrees_east",
            "bounds": "lon_b",
        })

    return ds_in


def get_xesmf_corners(lat_b, lon_b):
    """Convert CF bounds to the corners used by xesmf."""
    lat_bounds, lon_bounds = lat_b.chunk({"vertex": -1}), lon_b.chunk({"vertex": -1})
    lat_corners = cf_xarray.bounds_to_vertices(lat_bounds, bounds_dim="vertex")
    lon_corners = cf_xarray.bounds_to_vertices(lon_bounds, bounds_dim="vertex")
    return lat_corners, lon_corners


def get_ds_out(ds_in, resolution: float = 0.25):
    """Create a standard global output grid."""
    return xe.util.grid_global(resolution, resolution, lon1=360)


def get_model_scenario_bounds(json_file):
    """Read a JSON file of lat/lon bounds and aggregate by model and experiment."""
    with open(json_file, "r") as f:
        records = json.load(f)

    bounds = defaultdict(lambda: defaultdict(lambda: {
        "lat_min": float("inf"),
        "lat_max": float("-inf"),
        "lon_min": float("inf"),
        "lon_max": float("-inf"),
    }))

    for rec in records:
        src = rec["source_id"]
        exp = rec["experiment_id"]
        b = bounds[src][exp]
        b["lat_min"] = min(b["lat_min"], rec["lat_min"])
        b["lat_max"] = max(b["lat_max"], rec["lat_max"])
        b["lon_min"] = min(b["lon_min"], rec["lon_min"])
        b["lon_max"] = max(b["lon_max"], rec["lon_max"])

    return {src: dict(exp_dict) for src, exp_dict in bounds.items()}


def load_scenario_queries(json_file):
    """Load scenario definitions from scenario_experiment_combination.json."""
    with open(json_file, "r") as f:
        records = json.load(f)

    queries = []
    for record in records:
        for experiment_id in record.get("experiments", []):
            queries.append({
                "source_id": record["source_id"],
                "experiment_id": experiment_id,
                "institution_id": record.get("institution_id"),
                "member_id": record.get("member_id"),
            })

    return queries


def write_to_zarr(ds, store_dir, source_id, experiment_id):
    """Write a dataset to Zarr, creating or appending to the target path."""
    zarr_path = Path(store_dir) / f"{source_id}/{experiment_id}"
    
    if zarr_path.exists() and any(zarr_path.iterdir()):
        raise FileExistsError(
            f"Zarr store already exists: {zarr_path}\n"
            "Delete it first if you want to regenerate the dataset."
        )

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(
        zarr_path,
        mode="w",
        zarr_format=2,
    )
    print(
        f"Created Zarr store for {source_id} {experiment_id} "
        f"({ds.sizes['time']} time steps)"
    )


def run_regridding_workflow(
    scenario_queries,
    store_dir="./data/input/cmip6_data",
    url="https://storage.googleapis.com/cmip6/pangeo-cmip6.json",
    vars_tables=None,
    z_kwargs=None,
):
    """Run the regridding workflow for the selected scenario queries."""
    if vars_tables is None:
        vars_tables = [("ta", "Amon"), ("tos", "Omon"), ("psl", "Amon"), ("hurs", "Amon")]
    if z_kwargs is None:
        z_kwargs = {"consolidated": True, "decode_times": True, "use_cftime": True}

    vars_tables_dicts = [{"variable_id": variable_id, "table_id": table_id} for variable_id, table_id in vars_tables]

    col = intake.open_esm_datastore(url)

    for query in tqdm(scenario_queries, desc="models"):

        all_ds = {}
        for var_table in tqdm(vars_tables_dicts, desc="var"):
            query_var = dict(query) | var_table
            variable_id = var_table["variable_id"]

            cat = col.search(**query_var)
            if cat.df.empty:
                query_var["grid_label"] = "gr" if query_var.get("grid_label", "gn") == "gn" else "gn"
                cat = col.search(**query_var)
                if cat.df.empty:
                    print(f"No results for {query_var}")
                    continue

            with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                clim_dict = cat.to_dataset_dict(zarr_kwargs=z_kwargs, preprocess=combined_preprocessing)
                for clim_key, ds_clim in clim_dict.items():
                    ds_clim = ds_clim.squeeze()
                    ds_clim = ds_clim.sel(time=slice("1950-01-01", "2100-12-31"))

                    if variable_id == "ta":
                        ds_clim["ta"] = ds_clim["ta"].sel(plev=5000.0, method="nearest")

                    ds_in = clean_cmip_vars(ds_clim)
                    ds_in = prep_cmip_for_xesmf(ds_in)
                    ds_in = ds_in.convert_calendar("noleap", align_on="year")

                    time_index = xr.cftime_range(
                        start="1950-01-01",
                        end="2100-12-01",
                        freq="MS",
                        calendar="noleap",
                    )

                    ds_in = ds_in.resample(time="MS").mean()
                    ds_in = ds_in.reindex(time=time_index, method="nearest")

                    ds_out = get_ds_out(ds_in)
                    regrid_method = "conservative_normed" if variable_id == "tos" else "conservative"

                    if bool({"x", "y"} & set(ds_in.dims)):
                        ds_in[variable_id] = ds_in[variable_id].chunk({"time": 10, "y": -1, "x": -1})
                    else:
                        ds_in[variable_id] = ds_in[variable_id].chunk({"time": 10, "lat": -1, "lon": -1})

                    regridder = xe.Regridder(
                        ds_in,
                        ds_out,
                        regrid_method,
                        periodic=True,
                        ignore_degenerate=True,
                    )
                    ds_out = regridder(ds_in, keep_attrs=True)

                    for drop_var in ["lat_b", "lon_b", "mask"]:
                        if drop_var in ds_out.variables:
                            ds_out = ds_out.drop_vars(drop_var)

                    all_ds[variable_id] = ds_out

        if all_ds:
            ds_to_store = xr.merge(list(all_ds.values()))
            write_to_zarr(ds_to_store, store_dir, query["source_id"], query["experiment_id"])


if __name__ == "__main__":
    scenario_queries = load_scenario_queries("./scenario_experiment_combination.json")
    run_regridding_workflow(scenario_queries, store_dir="./data/input/cmip6_data")
