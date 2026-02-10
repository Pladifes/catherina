"""Fetch CMIP6 climate data from pangeo API and store as zarr files."""

import json
import pprint
import webbrowser
import cf_xarray as cfxr
import dask
import intake
from loguru import logger
import xarray as xr
import xesmf as xe
from dask.distributed import Client
from tqdm.notebook import tqdm
from xmip.preprocessing import combined_preprocessing
from pathlib import Path


def get_cmip6_data_from_pangeo_api(
    source_id: str,
    experiment_id: str,
    store_dir: Path,
    institution_id: str,
    member_id: str,
) -> None:
    query_list = generate_cmip6_queries(
        source_id, experiment_id, tbl_var, institution_id, member_id
    )
    queries_pbar = tqdm(query_list, desc="Processing pangeo queries")
    for query in queries_pbar:
        queries_pbar.set_postfix(climate_var=query["variable_id"])
        pprint.pp(query)
        cat = col.search(**query)

        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            clim_dset_dict = cat.to_dataset_dict(
                zarr_kwargs=z_kwargs, preprocess=combined_preprocessing
            )
            clim_pbar = tqdm(
                clim_dset_dict.keys(), desc="Processing model-experiment-variable"
            )
            for clim_key in clim_pbar:
                clim_pbar.set_postfix({"cmip6_setting": clim_key})
                ds_clim = clim_dset_dict[clim_key].squeeze()
                # Fetch data up to year 2100 (some models go to year 2300)
                ds_clim = ds_clim.sel(
                    time=slice("1950-01-01T00:00:00", "2100-12-31T00:00:00")
                )
                if query["variable_id"] == "ta":
                    # Select tropopause air temperature and relative humidity for given pressure levels
                    # TODO: change plev to 5000.0
                    ds_clim["ta"] = ds_clim["ta"].sel(plev=50.0, method="nearest")
                elif query["variable_id"] == "hur":
                    ds_clim["hur"] = ds_clim["hur"].sel(plev=1000.0, method="nearest")
                elif query["variable_id"] == "hurs":
                    ds_clim["hurs"] = ds_clim["hurs"].clip(0.0, 100.0)

                # Drop unnecessary dims and variables after selection
                # Some datasets do not contain "lat_bounds", "lon_bounds"
                vars_to_drop = [
                    "member_id",
                    "dcpp_init_year",
                    "lon_bounds",
                    "lat_bounds",
                ]
                vars_to_drop = set(ds_clim.variables).intersection(vars_to_drop)
                coords_to_reset = ["time_bounds", "lat_verticies", "lon_verticies"]
                coords_to_reset = set(ds_clim.coords).intersection(coords_to_reset)
                ds_clim = ds_clim.drop_vars(vars_to_drop).reset_coords(coords_to_reset)
                if set(["lat_verticies", "lon_verticies"]).issubset(ds_clim.coords):
                    ds_clim = ds_clim.rename(
                        {
                            "lat_verticies": "lat_vertices",
                            "lon_verticies": "lon_vertices",
                        }
                    )
                if "plev" in ds_clim.dims:
                    ds_clim = ds_clim.drop_dims("plev")
                if "plev_bnds" in ds_clim.variables:
                    ds_clim = ds_clim.drop_vars("plev_bnds")

                has_vertices = "vertex" in ds_clim.dims
                method = "conservative"  # default

                if query["table_id"] == "Amon":
                    ds_clim["lon_vertices"] = ds_clim["lon_vertices"].chunk(
                        {"vertex": -1}
                    )
                    ds_clim["lat_vertices"] = ds_clim["lat_vertices"].chunk(
                        {"vertex": -1}
                    )
                    # Get the bounds variable and convert them to "vertices" format
                    # Order=none, means that we do not know if the bounds are listed clockwise or counterclockwise, so we ask cf_xarray to try both.
                    lat_corners = cfxr.bounds_to_vertices(
                        ds_clim.lat_vertices, "vertex", order=None
                    )
                    lon_corners = cfxr.bounds_to_vertices(
                        ds_clim.lon_vertices, "vertex", order=None
                    )
                    ds_clim = ds_clim.assign(lon_b=lon_corners, lat_b=lat_corners)
                    ds_clim = ds_clim.drop_vars(["lon_vertices", "lat_vertices"])
                    ds_clim = ds_clim.rename({"y_vertices": "y_b", "x_vertices": "x_b"})

                    # Assign cf-compliant bounds for regridding
                    ds_clim["lon"].attrs["bounds"] = "lon_b"
                    ds_clim["lat"].attrs["bounds"] = "lat_b"
                    print(ds_clim.cf)
                elif query["table_id"] == "Omon":
                    if has_vertices:
                        method = "conservative"
                        # FIXME: correct workflow for variables with vertex data
                        # Assign cf-compliant bounds for regridding
                        # For some reason, bounds attributes are dropped after regridding
                        # Reset them if regridding
                        ds_clim["lat"].attrs["bounds"] = "lat_vertices"
                        ds_clim["lon"].attrs["bounds"] = "lon_vertices"
                    # TODO: handle case with no vertices
                    else:
                        method = "bilinear"

                var = query["variable_id"]
                ds_out = regrid_to_global(ds_clim, var, method)

                # Rechunk on time dimension before storing to zarr
                ds_out = ds_out.chunk({"time": 1, "y": -1, "x": -1})
                source_id = clim_key.split(".")[2]
                experiment_id = clim_key.split(".")[3]
                ds_out.to_zarr(
                    Path(store_dir) / f"{source_id}/{experiment_id}",
                    mode="a",
                    zarr_format=2,
                )


def regrid_to_global(
    ds_clim: xr.Dataset, var: str, method, resolution=0.25
) -> xr.Dataset:
    """
    Regrid a climate dataset to a global lat/lon grid.

    Parameters
    ----------
    ds_clim : xarray.Dataset
        Input dataset to be regridded.
    query : dict
        Dictionary containing at least the key "variable_id".
    method : str
        Regridding method (e.g. "conservative", "bilinear").
    resolution : float, optional
        Grid resolution in degrees (default is 0.25).

    Returns
    -------
    ds_out : xarray.Dataset
        Regridded dataset.
    """

    # Define output grid (resolution° lat x resolution° lon)
    ds_out = xe.util.grid_global(resolution, resolution)

    # Rechunk before regridding to control memory usage
    ds_clim[var] = ds_clim[var].chunk({"time": 10, "y": -1, "x": -1})

    # Create regridder
    regridder = xe.Regridder(
        ds_clim,
        ds_out,
        method,
        ignore_degenerate=True,
    )

    # Apply regridding
    ds_out = regridder(ds_clim)

    return ds_out


def generate_cmip6_queries(
    source_ids: list[str],
    experiment_ids: list[str],
    table_variables: dict[str, list[str]],
    institution_id: str,
    member_id: str,
    grid_label: list[str] = ["gn"],
) -> list[dict]:
    """
    Generate a list of CMIP6 query dictionaries for combinations of tables and variables.

    Parameters
    ----------
    source_ids : list[str]
        List of CMIP6 model identifiers (source_id)
    experiment_ids : list[str]
        List of CMIP6 experiment identifiers (e.g., historical, ssp126)
    table_variables : dict[str, list[str]]
        Dictionary mapping table_id to list of variable_ids
    member_id : str, optional
        Member identifier, by default "r1i1p1f1"
    grid_label : list[str], optional
        Grid labels to query, by default ['gn'] (native grid)

    Returns
    -------
    list[dict]
        List of query dictionaries for all combinations
    """
    query_list = []

    # Create queries for each table and variable combination
    for table, variables in table_variables.items():
        for variable in variables:
            query = {
                "experiment_id": experiment_ids,
                "table_id": table,
                "source_id": source_ids,
                "variable_id": variable,
                "member_id": member_id,
                "grid_label": grid_label,
                "institution_id": institution_id,
            }
            query_list.append(query)

    # Display a sample query for verification
    if query_list:
        print("\nSample query:")
        pprint.pp(query_list[0])

    return query_list


if __name__ == "__main__":
    # Start Dask Client
    client = Client(n_workers=1, threads_per_worker=4, memory_limit="4GB")
    print(f"Dask dashboard: {client.dashboard_link}")
    # TODO: launch dashboard in browser
    # webbrowser.open(client.dashboard_link)

    logger.info(f"Fetching CMIP6 climate data from pangeo API")
    logger.info(
        f"Catalog available at this url: {'https://storage.googleapis.com/cmip6/pangeo-cmip6.csv'}"
    )
    url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(url)
    z_kwargs = {"consolidated": True, "decode_times": True, "use_cftime": True}
    with open("./parameters_for_climate_scenario.json", "r") as f:
        queries = json.load(f)
    query = queries[4]
    # source_id in emsl refers to model_id in catherina

    pprint.pprint(query)
    # "MPI-ESM1-2-LR", "ACCESS-CM2"

    ### TODO lire json et dire que on a remplacé ipsl_cm5a2_inca par ipsl-cm6a-lr

    ### CHANGER ICI POUR SCENARIO FUTUR OU HISTORIQUE ###
    # experiment_id = ["historical"] # pour récupérer les données simulées historiques pour le débiaisage

    tbl_var = {
        "Amon": [
            "hurs",
            "psl",
            "ta",
        ],
        "Omon": ["tos"],
    }
    store_dir = "./data/input/cmip6_data"
    print(f"Save dir: {store_dir}")
    get_cmip6_data_from_pangeo_api(
        source_id=query["source_id"],
        experiment_id=query["experiment_id"],
        store_dir=store_dir,
        institution_id=query["institution_id"],
        member_id=query["member_id"],
    )
    print("------------ pangeo done ------------")
