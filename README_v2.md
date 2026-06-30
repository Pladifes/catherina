# Catherina v2 — README_v2

## Purpose
This repository implements a synthetic tropical cyclone track generation pipeline, starting from CMIP6 climate data and ending with intensity-adjusted tracks. The main code path is:

- `src/data/cmip6_pangeo.py`
- `synthetic_tracks.py`

The notebooks in `notebooks/` provide user-facing workflows to filter CMIP6 data, download climate input from Pangeo, and run multiple scenario experiments.

---
## Main Pipeline

### `src/data/cmip6_pangeo.py`
This module downloads CMIP6 climate data from the Pangeo catalog and stores it as Zarr.

Main features:
- Query CMIP6 metadata from `https://storage.googleapis.com/cmip6/pangeo-cmip6.json`
- Select variables needed by the Catherina model: `hurs`, `psl`, `ta`, `tos`
- Regrid selected datasets to a common global latitude/longitude grid
- Save results into `data/input/cmip6_data/<model>/<experiment>`

This code is used by the notebook `Prepare_cmip6_data.ipynb` to fetch and prepare climate data before running synthetic track simulation.

### `synthetic_tracks.py`
This script executes the core Catherina workflow.

Steps performed:
- Load configuration from `config.toml`
- Read IBTrACS historical cyclone data
- Simulate synthetic genesis points with `src.genesis.genesis.simulate_tc_genesis`
- Generate cyclone tracks using pre-fitted coefficients in `data/input/fit`
- Attach climate variables to track points using CMIP6 data
- Debias climate variables with ERA5 benchmark data using `src.bias_correction.tracks`
- Apply intensity and decay modeling with `src.intensitifcation_and_decay.intensify`

Key directories created under `data/output/catherina/`:
- `genesis/<model>/<experiment>`
- `track/<model>/<experiment>`
- `track_with_env/<model>/<experiment>`
- `track_with_env_corr/<model>/<experiment>`
- `intensified_tracks/<model>/<experiment>`

This code is used by the notebook `Run_multiple_scenarios.ipynb` to run synthetic track simulation.

---
## Notebooks in `notebooks/`

### `Filter_cmip6_model_scenario.ipynb`
This notebook helps find CMIP6 data sources that provide the required variables for the chosen scenario and experiment.

It:
- loads the Pangeo CMIP6 metadata CSV `data/pangeo-cmip6.csv`
- filters for some experiments (in the example, `historical`, `ssp370`, `ssp585`)
- selects tables `Amon` and `Omon`
- filters variables `hurs`, `psl`, `ta`, `tos`
- produces a filtered list of source/model and member combinations that have a complete set of variables

### `Prepare_cmip6_data_regrid.ipynb`
This notebook downloads CMIP6 input data from the Pangeo catalog using `src.data.cmip6_pangeo_regrid.get_cmip6_data_from_pangeo_api`.

It:
- starts a Dask client for parallel download
- loads the Pangeo catalog via `intake-esm`
- reads model/experiment combinations from `scenario_experiment_combination.json`
- fetches climate variables and saves them to `data/input/cmip6_data`

### `Run_multiple_scenarios.ipynb`
This notebook automates running the synthetic track pipeline for multiple models and experiments.

It:
- updates `config.toml` values dynamically for each model/experiment pair
- sets the simulation year window for historical vs future experiments
- runs the Pixi task `pixi run synth`
- captures runtime output and return codes

This notebook is intended to help batch-run multiple scenarios after the required CMIP6 and historical input files are available.

---
## Draft notebooks in `_notebooks_draft/`

These notebooks are not part of the main pipeline but document exploratory or support workflows.

### `Reproduce_fit_coefficients.ipynb`
Explains how to reproduce the track generation fit coefficients from IBTrACS data.

It:
- loads IBTrACS cyclone tracks
- computes track displacement statistics
- fits regression models for track dynamics
- evaluates predictions and residuals

This notebook is informative but not currently aligned with the exact coefficient files stored in `data/input/fit`.

### `add_country_to_tracks_optimized.ipynb`
Adds country metadata to cyclone track points.

It:
- loads cyclone track output
- builds a spatial index for country polygons
- assigns `country` and `country_iso2` only for points over land

### `compute_damages_by_countries_optimized.ipynb`
This notebook focus on joining cyclone tracks with national exposure/damage data from LitPop.


### `compute_damages_denisa_data.ipynb`
This notebook focus on joining cyclone tracks with national exposure/damage data originating from LitPop and postprocessed by Denisa. It only works for USA and China.


### `working_paper_plots.ipynb`
Contains plotting and visualization on simulated data.

---
## User guide

### System requirements
- Linux is required for the main pipeline.
- If running on Windows, use WSL2 with a Linux distribution.
- The repository, build tools, and data paths are designed for Linux-style file handling.

### Install dependencies with Pixi
The repository uses `pixi` to manage the Python environment.

1. Install Pixi on Linux or WSL2:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Open a new shell or source the Pixi path if needed.

3. Install dependencies:
```bash
pixi install
```

4. Confirm `pixi` tasks are available by inspecting `pixi.toml`.

### Run the Pangeo download task

```bash
pixi run pangeo
```

This executes `src/data/cmip6_pangeo.py` and downloads selected CMIP6 variables to `data/input/cmip6_data`. 
A more user-friendly approach is available in the notebook `Prepare_cmip6_data.ipynb`.

### Run the core pipeline

From the repository root:
```bash
pixi run synth
```

Or directly:
```bash
python synthetic_tracks.py
```

If you want to use a custom config file:
```bash
python synthetic_tracks.py --config_path path/to/config.toml
```
A more user-friendly approach is available in the notebook `Run_multiple_scenarios.ipynb`.


### Notes on required input data

The pipeline needs several input files in `data/input`:
- `ibtracs.since1980.list.v04r01.csv`
- `bias_correction/ERA5_benchmark_100tracks_bias_corrections.csv`
- `fit/Catherina_fit.db`
- Natural Earth land/coastline shapefiles: `ne_10m_land.zip`, `ne_10m_coastline.zip`

The pangeo task should be run before the core pipeline, to load CMIP6 data in `data/input/cmip6_data/<model>/<experiment>`. 


## Details on the steps in `src/`

### `genesis/`
Generates synthetic tropical cyclone starting points (genesis locations) from historical IBTrACS observations. Produces realistic genesis point distributions and temporal patterns.

### `track_generation/`
Simulates complete TC track trajectories using a stochastic markov-based model fit to historical data. Generates realistic spatial and temporal patterns of TC movement.

### `climate_data_merging/`
Merges CMIP6 climate variables (relative humidity, mean sea level pressure, air temperature, sea surface temperature) with each track point. This step enriches the synthetic tracks with environmental conditions necessary for intensity modeling.

### `bias_correction/`
Corrects systematic biases in the CMIP6 climate variables using ERA5 observations as a benchmark. Uses **CDF transformation (CDFt)** technique:

- **Step 1**: Extracts climate variable distributions from:
  - Observed ERA5 historical data (ground truth)
  - CMIP6 simulated historical data
  - CMIP6 future/projected data

- **Step 2**: Applies CDF transformation to align the future model predictions with observed statistical properties, ensuring realistic climate variable distributions in the simulated tracks

- **Step 3**: Corrects three key variables:
  - `MSLP` (Mean Sea Level Pressure)
  - `nshr` (Normalized Relative Humidity)
  - `thermo_eff` (Thermodynamic Efficiency: relationship between SST and upper atmosphere temperature)

- **Output**: Debiased tracks with corrected climate variables for more realistic physical constraints on TC intensity

### `intensitifcation_and_decay/`
Models TC intensity changes using physics-based equations and empirical relationships:

**Key Features:**
- **Wind Speed Calculation**: Uses wind-pressure relationship (WPR) equations fitted to observed data
- **Environmental Factors**: Incorporates:
  - Sea surface temperature (SST)
  - Atmospheric humidity
  - Sea level pressure
  - Latitude effects
  - Maximum Potential Intensity (MPI) constraints

- **Land Interactions**: 
  - Detects when tracks move over land using Natural Earth coastline data
  - Applies exponential decay model when on land based on:
    - Time spent on land
    - Distance to coast
    - Wind speed at landfall
  - Smoothly transitions wind back to ocean values when TC returns to water

- **Intensification Mechanics**:
  - Calculates environmental saturation deficit (delta S_m)
  - Computes MPI based on environmental thermodynamic conditions
  - Integrates pressure deficit evolution using stochastic model
  - Applies physical constraints (e.g., central pressure cannot be lower than MSLP)

- **Track Termination**: Stops simulating when:
  - Wind speed drops below 5 m/s
  - SST drops below 15°C
  - 3-hour rolling average wind falls below 10 m/s

**Output**: Complete TC intensity evolution with realistic intensification, maintenance, and decay phases

### `climate_data_merging/climate_data_merging.py`
Handles spatial-temporal alignment of CMIP6 climate data with track points using nearest-neighbor spatial joins on a month-by-month basis for efficiency.

---
## Docker support

A `Dockerfile` is included, but it may not work perfectly in all environments.

### Build the image
```bash
docker build -t catherina_v2 .
```

### Run the container
```bash
docker run -it --rm \
  -v "$PWD/data/input":/app/data/input \
  -v "$PWD/data/output":/app/data/output \
  catherina_v2 bash
```

### Inside the container
```bash
pixi run synth
```

### Docker Compose
If you want an interactive container, you can use `docker-compose.yml`:
```bash
docker compose up --build -d
```
Then enter the container:
```bash
docker compose exec <service_name> bash
```
```

> Warning: Docker support is provided for convenience, but the repository has not been fully validated in containerized mode.

---
## Key files
- `synthetic_tracks.py` — core generation pipeline
- `src/data/cmip6_pangeo.py` — CMIP6 download and regridding
- `config.toml` — default run configuration
- `pixi.toml` — dependency and task definitions
- `scenario_experiment_combination.json` — model/experiment selection for CMIP6 downloads
- `notebooks/Prepare_cmip6_data.ipynb` — prepare climate input data
- `notebooks/Run_multiple_scenarios.ipynb` — batch run synthetic tracks

---
## Known caveats
- `_notebooks_draft/` contains experimental and partially-working notebooks.
- The pipeline assumes CMIP6 and ERA5 data are available locally, and the historical experiment has been downloaded before running future projections.
- Docker mode is experimental and may require manual debugging.
