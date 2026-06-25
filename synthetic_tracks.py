from joblib import Parallel, delayed
import itertools
import pyarrow.dataset as pds
from tqdm import tqdm
import xarray as xr
import warnings
from pathlib import Path
from shapely.prepared import prep
import tomllib
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from tqdm_joblib import tqdm_joblib
import more_itertools
import os
import re

from src.bias_correction.tracks import (
    correct_bias_with_era5_and_save,
    get_era5_benchmark,
    get_histo_sim_from_benchmark,
)

from src.genesis.genesis import simulate_tc_genesis
from src.genesis.genesis_utils import read_ibtracs
from src.track_generation.track_generation import simulate_tc_tracks
from src.climate_data_merging.climate_data_merging import process_month
from src.bias_correction.bias_correction import correct_bias_with_era5_and_save
from src.intensitifcation_and_decay.intensify import intensify_and_save

warnings.filterwarnings("ignore")


def main(
    config_path: str = "./config.toml",
) -> None:
    # Instantiate logger
    console = set_console_logger()
    logger.info("Running Catherina module.")
    # Load toml config file
    with open(
        config_path,
        "rb",
    ) as f:  # Open the file in binary mode
        config_files = tomllib.load(f)

    # Get config
    main_config = config_files["main_params"]
    gen_config = config_files["generation"]
    intens_config = config_files["intensification"]

    # Number seeds + jobs + batch size + nb steps
    n_seeds = main_config["n_seeds"]
    n_job = main_config["n_jobs"]
    batch_seed_size = main_config["batch_seeds_size"]
    max_step = gen_config["max_steps"]

    # Start and end years
    start_year = main_config["start_year"]
    end_year = main_config["end_year"]

    # Model and experiment
    model = main_config["models"][0]
    experiment = main_config["experiments"][0]

    # Get folder names for project
    # Project
    data_dir = Path(main_config["input_data_dir"])
    output_dir = Path(main_config["output_data_dir"])
    cyclones_tracks_dir = output_dir / "genesis" / model / experiment
    tracks_dir = output_dir / "track" / model / experiment
    tracks_with_env_dir = output_dir / "track_with_env" / model / experiment
    tracks_with_env_corr_dir = output_dir / "track_with_env_corr" / model / experiment
    intens_dir = output_dir / "intensified_tracks" / model / experiment
    # Additionnal data
    cmip_dir = data_dir / Path(main_config["climate_data_dir"])
    fit_dir = data_dir / Path(gen_config["fit_dir"])
    catherina_fit_path = fit_dir / "Catherina_fit.db"
    land_zip_path = data_dir / Path(intens_config["ne_10m_land_zip_path"])
    ne_10m_coastline_zip_path = data_dir / Path(
        intens_config["ne_10m_coastline_zip_path"]
    )
    bias_correction_path = data_dir / Path(main_config["bcorr_path"])

    # Generate synthetic genesis points
    usecols = [
        "SID",
        "ISO_TIME",
        "BASIN",
        "LON",
        "LAT",
        "NATURE",
        "TRACK_TYPE",
        "WMO_WIND",
        "WMO_PRES",
    ]

    # Get additionnal data
    ibtracs = read_ibtracs(
        fpath=Path(data_dir / gen_config["ibtracs_path"]), usecols=usecols
    )

    # Historical data for debiasing
    clim_obs_histo = get_era5_benchmark(benchmark_path=bias_correction_path)
    clim_sim_histo_ds = xr.open_zarr(cmip_dir / f"{model}/historical", chunks="auto")[
        ["hurs", "psl", "ta", "tos"]
    ].rename({"hurs": "hur"})  # TODO: move rename to cmip6_pangeo.py
    clim_sim_histo = get_histo_sim_from_benchmark(
        benchmark=clim_obs_histo, histo_clim_ds=clim_sim_histo_ds
    )
    clim_sim_histo = clim_sim_histo.assign(
        MSLP=clim_sim_histo["MSLP"] / 100.0,
        thermo_eff=((clim_sim_histo["SST"] + 273.15) - clim_sim_histo["T_strat"])
        / (clim_sim_histo["SST"] + 273.15),
    )

    # Check if output data already exists and if yes the number of seeds already run
    max_seed_existing = 0

    try:
        pattern = re.compile(r"seed=(\d+)")

        for folder in os.listdir(intens_dir):
            match = pattern.match(folder)
            if match:
                value_seed = int(match.group(1))
                if value_seed > max_seed_existing:
                    max_seed_existing = value_seed 
        for folder in os.listdir(cyclones_tracks_dir):
            match = pattern.match(folder)
            if match:
                value_seed = int(match.group(1))
                if value_seed > max_seed_existing:
                    max_seed_existing = value_seed

        max_seed_existing += 1

    except:
        pass

    print("Starting at seed number: ", max_seed_existing)

    # Logger
    logger.remove()
    logfile = "debug.log"
    # Add a file sink for debugging logs
    logger.add(
        logfile, rotation="10 MB", enqueue=True
    )  # Rotates file when it reaches 10MB

    """
    Simulate starting points
    TODO: set displace=True
    """
    logger.info("Create starting point...")

    simulate_tc_genesis(
        ibtracs=ibtracs,
        land_zip_path=land_zip_path,
        resolution=gen_config["resolution"],
        n_seeds=n_seeds,
        start_year=start_year,
        end_year=end_year,
        max_seed_existing=max_seed_existing,
        save_dir=cyclones_tracks_dir,
        displace=False,
    )

    """
    Create tracks
    TODO: simulate_tc_tracks
    """
    logger.info("Create tracks...")
    genesis_ds = pds.dataset(
        cyclones_tracks_dir,
        format="parquet",
        partitioning="hive",
    )

    batch_seeds = list(more_itertools.chunked(list(range(max_seed_existing, max_seed_existing+n_seeds)), n=batch_seed_size))

    with tqdm_joblib(batch_seeds, desc="Processing seeds", position=0, total=len(batch_seeds), leave=True,
    ) as progress_bar:
        Parallel(n_jobs=n_job, backend="loky")(
            delayed(simulate_tc_tracks)(
                genesis_ds= genesis_ds,
                catherina_fit_path = fit_dir,
                max_steps=max_step,
                seeds=seeds,
                logfile=logfile,
                save_dir=tracks_dir,
            )
            for batch_id, seeds in enumerate(batch_seeds)
        )

    """
    Add climate variables
    """
    logger.info("Add climate variable...")
    clim_ds = xr.open_zarr(cmip_dir / model / experiment)
    tracks = pds.dataset(tracks_dir, format="parquet", partitioning="hive")
    tracks = tracks.filter(pds.field("seed").isin((range(max_seed_existing, max_seed_existing+n_seeds))))

    yearmonth_batches = list(itertools.product(range(start_year, end_year), range(1, 12+1)))
    tasks = []
    for yearmonth in yearmonth_batches:
        tasks.append(yearmonth)

    Parallel(n_jobs=n_job, prefer="threads")(
                delayed(process_month)(
                    year=year,
                    month=month,
                    clim_ds=clim_ds,
                    model=model,
                    experiment=experiment,
                    tracks=tracks,
                    save_dir=tracks_with_env_dir,
                )
                for year, month in tqdm(tasks, desc="year-month")
            )

    """
    Debias climate variables
    """
    logger.info(
        "Debiasing climate variables for future tracks based on observed (ERA5) and simulated (CMIP6) historical data"
    )

    # Create a pyarrow scanner for the parquet file
    tracks_with_env_ds = pds.dataset(
        tracks_with_env_dir, format="parquet", partitioning="hive"
    )
    tracks_with_env_ds = tracks_with_env_ds.filter(pds.field("seed").isin((range(max_seed_existing, max_seed_existing+n_seeds))))


    print("--- Correcting climate bias ---")

    # TODO : uncomment to run all bias correction at once
    # correct_all_clim_bias(main_config=config_files["main_params"], output_dir=output_dir)

    # todo: batch seeds
    for seed in range(max_seed_existing, max_seed_existing+n_seeds):  # range(gen_config["n_seeds"]):
        correct_bias_with_era5_and_save(
            seeds=[seed],
            tracks_with_env_ds=tracks_with_env_ds,
            clim_obs_histo=clim_obs_histo,
            clim_sim_histo=clim_sim_histo,
            model=model,
            experiment=experiment,
            save_dir=tracks_with_env_corr_dir,
        )

    """
    Intensify and decay
    """
    logger.info("Intensifying tracks...")
    print("intensifying tracks...")
    seed_batches = list(
        itertools.batched(range(max_seed_existing, max_seed_existing+n_seeds), n=batch_seed_size)
    )  # or 4/8 depending on RAM
    tasks = []
    for seeds in seed_batches:
        tasks.append(seeds)

    Parallel(n_jobs=1, prefer="processes")(  # n=4
        delayed(intensify_and_save)(
            batch_seeds=list(batch_seeds),
            corrected_tracks_dir=tracks_with_env_corr_dir,
            catherina_fit_path=catherina_fit_path,
            ne_10m_coastline_zip=ne_10m_coastline_zip_path,
            ne_10m_land_zip=land_zip_path,
            save_dir=intens_dir,
        )
        for batch_seeds in tqdm(tasks, desc="seed")
    )


def set_console_logger(
    rotation: str = "1 MB",
    retention: str = "10 days",
    format: str = "{message} | {file} | line {line}",
) -> Console:
    """
    Configures and returns a logger instance.

    Args:
    - log_name (str): The name of the log file. Defaults to "main.log".
    - log_level (str): The logging level. Defaults to "DEBUG".
    - rotation (str): The rotation size of the log file. Defaults to "1 MB".
    - retention (str): The retention period of the log file. Defaults to "10 days".

    Returns:
    - logger: A configured logger instance.
    """
    project_dir = Path().cwd()
    log_dir = project_dir / ".logs"
    log_dir.mkdir(exist_ok=True)

    logger.remove()  # Remove default logger

    # Step 1: Set up Loguru to log warnings to a separate file
    logger.add(
        log_dir / "warnings.log",
        level="WARNING",
        rotation=rotation,
        retention=retention,
        format="{time} - {level} - {message}",
    )

    # Step 2: Create a function to handle warnings and route them to Loguru
    def log_warning_to_file(message, category, filename, lineno, file=None, line=None):
        # Log the warning message with loguru's warning level
        logger.warning(f"{filename}:{lineno} - {category.__name__}: {message}")

    # Step 3: Redirect warnings to the custom function
    warnings.showwarning = log_warning_to_file

    # TODO: redirect pandas warnings to loguru
    # Set up file output handler (save logs to a file)
    logger.add(
        log_dir / "main.log",
        level="INFO",
        rotation=rotation,
        retention=retention,
        format=format,
    )

    # Add the rich handler for console output
    console = Console()

    # Create a console output handler (RichHandler)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    logger.add(
        sink=rich_handler,
        level="INFO",
        format=format,
        filter=lambda record: record["level"].name == "INFO",
    )

    return console


if __name__ == "__main__":
    main()
