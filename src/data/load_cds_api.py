import os
from typing import Dict, List

import cdsapi
import pandas as pd
from loguru import logger


def get_copernicus_data(
    models: List[str],
    experiments: List[str],
    output_dir: str,
    start_year=2025,
    end_year=2101,
) -> pd.DataFrame:
    """
    Load climate data for specified models and experiments, retrieving various climate variables.

    Parameters:
        models (list): Climate models, e.g., ['ipsl_cm6a_lr', 'access_cm2', 'ipsl_cm5a2_inca'].
        experiments (list): Climate experiments, e.g., ['ssp5_8_5', 'historical'].
        output_dir (str): Root directory where retrieved data will be stored.
        start_year (int): Starting year for projections (default: 2025).
        end_year (int): Ending year for projections (default: 2101).

    Returns:
        pd.DataFrame: DataFrame containing the availability of each variable for each model-experiment combination.
    """
    availability_df = pd.DataFrame(
        columns=["model", "experiment", "SST", "MSLP", "RH", "T_tropo"]
    )
    client = cdsapi.Client()

    def _retrieve_variable(
        model: str, experiment: str, variable: str, output_path: str, params: Dict
    ) -> int:
        """Helper function to retrieve a specific climate variable and update availability."""
        try:
            client.retrieve("projections-cmip6", params, output_path)
            return 1  # Success
        except Exception as e:
            logger.info(
                f"Error retrieving {variable} for model {model} and experiment {experiment}: {e}"
            )
            return 0  # Failure

    for model in track(models, description="Models"):
        for experiment in track(experiments, description="Experiments"):
            # Set year range based on the experiment type
            period = (
                [str(year) for year in range(1950, 2015)]
                if experiment == "historical"
                else [str(year) for year in range(start_year, end_year + 1)]
            )
            experiment_dir = os.path.join(output_dir, model, experiment)
            os.makedirs(experiment_dir, exist_ok=True)

            # Initialize the availability record for this model and experiment
            availability = {
                "model": model,
                "experiment": experiment,
                "SST": 0,
                "MSLP": 0,
                "RH": 0,
                "T_tropo": 0,
            }

            # Retrieve each variable, setting parameters and file paths
            availability["RH"] = _retrieve_variable(
                model,
                experiment,
                "RH",
                f"{experiment_dir}/RH.zip",
                {
                    "format": "zip",
                    "experiment": experiment,
                    "variable": "near_surface_relative_humidity",
                    "model": model,
                    "month": [str(month) for month in range(1, 13)],
                    "year": period,
                    "temporal_resolution": "monthly",
                },
            ) or _retrieve_variable(
                model,
                experiment,
                "RH",
                f"{experiment_dir}/RH.zip",
                {
                    "format": "zip",
                    "experiment": experiment,
                    "variable": "relative_humidity",
                    "level": "1000",
                    "model": model,
                    "month": [str(month) for month in range(1, 13)],
                    "year": period,
                    "temporal_resolution": "monthly",
                },
            )

            availability["SST"] = _retrieve_variable(
                model,
                experiment,
                "SST",
                f"{experiment_dir}/SST.zip",
                {
                    "format": "zip",
                    "experiment": experiment,
                    "variable": "sea_surface_temperature",
                    "model": model,
                    "month": [str(month) for month in range(1, 13)],
                    "year": period,
                    "temporal_resolution": "monthly",
                },
            )

            availability["MSLP"] = _retrieve_variable(
                model,
                experiment,
                "MSLP",
                f"{experiment_dir}/MSLP.zip",
                {
                    "format": "zip",
                    "experiment": experiment,
                    "variable": "sea_level_pressure",
                    "model": model,
                    "month": [str(month) for month in range(1, 13)],
                    "year": period,
                    "temporal_resolution": "monthly",
                },
            )

            availability["T_tropo"] = _retrieve_variable(
                model,
                experiment,
                "T_tropo",
                f"{experiment_dir}/T_tropo.zip",
                {
                    "format": "zip",
                    "experiment": experiment,
                    "variable": "air_temperature",
                    "level": "50",
                    "model": model,
                    "month": [str(month) for month in range(1, 13)],
                    "year": period,
                    "temporal_resolution": "monthly",
                },
            )

            # Append availability record to DataFrame
            availability_df = availability_df._append(availability, ignore_index=True)

    return availability_df


if __name__ == "__main__":
    import tomllib

    with open(
        "C:/Users/hamada.saleh/Documents/Python Scripts/catherina_version_ilb/catherina_config.toml",
        "rb",
    ) as f:  # Open the file in binary mode
        config = tomllib.load(f)

    # Run the retrieve_data function with the specified arguments
    availability_df = get_copernicus_data(
        models=config["models"],
        experiments=config["experiments"],
        output_dir=config["climate_data_dir"],
        start_year=config["start_year"],
        end_year=config["end_year"],
    )

    # Print the resulting DataFrame if desired
    logger.info(availability_df)
