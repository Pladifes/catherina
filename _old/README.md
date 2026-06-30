# Catherina v2 - Synthetic Tropical Cyclone Track Generation

A Python pipeline for generating synthetic tropical cyclone (TC) tracks using climate model data with bias correction and intensification modeling.

## Overview

This project simulates realistic tropical cyclone tracks from climate data by combining genesis point simulation, track generation, climate variable merging, bias correction, and intensification modeling.

## Execution Pipeline

The main workflow (`synthetic_tracks.py`) executes the following steps:

1. **Genesis Simulation**: Generate synthetic TC starting points based on IBTrACS historical data
2. **Track Generation**: Simulate complete TC tracks using the Catherina model fits
3. **Climate Variable Merging**: Attach CMIP6 climate variables (humidity, pressure, temperature, SST) to each track point
4. **Bias Correction**: Debias climate variables using ERA5 observations as benchmark
5. **Intensification & Decay**: Model TC intensity changes and handle land interactions

## Running the Code

### Prerequisites

```bash
# Install dependencies using Pixi (or your environment manager)
pixi install
```

### Configuration

Edit `config.toml` to specify:
- Input/output directories
- Climate models and experiments
- Number of seeds (synthetic realizations)
- Year range for simulation
- Number of parallel jobs

### Execution

```bash
python synthetic_tracks.py
```

The script will read the `config.toml` file by default. To use a custom config:

```bash
python synthetic_tracks.py --config_path /path/to/custom_config.toml
```

## Inputs

- **IBTrACS Data**: Historical tropical cyclone observations
- **CMIP6 Climate Data**: Multi-year monthly climate variables (humidity, pressure, temperature, SST)
- **ERA5 Data**: Historical observations for bias correction
- **Land/Coastline Data**: Natural Earth datasets for land interactions
- **Catherina Fit Database**: Pre-fitted statistical model for track generation

## Outputs

Generated in `output_data_dir/`:
- `genesis/`: Synthetic TC starting points (parquet format)
- `track/`: Complete simulated TC tracks
- `track_with_env/`: Tracks with attached climate variables
- `track_with_env_corr/`: Debiased tracks
- `intensified_tracks/`: Final tracks with intensity modeling

## Logging

- **Console Output**: Real-time progress with Rich formatting
- **Log Files**: Stored in `.logs/` directory:
  - `main.log`: General information
  - `warnings.log`: Warning messages
  - `debug.log`: Detailed debug information

## Key Dependencies

- `joblib`, `tqdm`: Parallel processing and progress tracking
- `xarray`, `pyarrow`: Data handling and storage
- `shapely`: Geometric operations
- `loguru`, `rich`: Advanced logging and console output

## Main Modules in `src/`

### `genesis/`
Generates synthetic tropical cyclone starting points (genesis locations) from historical IBTrACS observations. Produces realistic genesis point distributions and temporal patterns.

### `track_generation/`
Simulates complete TC track trajectories using a stochastic markov-based model fit to historical data. Generates realistic spatial and temporal patterns of TC movement.

### `climate_data_merging/`
Merges CMIP6 climate variables (relative humidity, mean sea level pressure, air temperature, sea surface temperature) with each track point. This step enriches the synthetic tracks with environmental conditions necessary for intensity modeling.

### `bias_correction/` ⭐ (Emphasis)
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

### `intensitifcation_and_decay/` ⭐ (Emphasis)
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


# Pixi Docker Compose Workflow for Interactive Development

This guide shows how to build and run your Pixi project in a Docker container using **Docker Compose**, keeping the container alive for interactive use with bind-mounted input/output folders.

---

## 1. Create `docker-compose.yml`

Place this in your project root:

```yaml
version: "3.9"

services:
  app:
    build: .
    image: pixi-test:latest
    volumes:
      - ./data/input:/app/data/input:ro
      - ./data/output:/app/data/output
    command: tail -f /dev/null
```

## 2. Build and start the container
```bash
docker compose up --build -d
```

## 3. Find container id
```bash
docker ps
```

## 4. Open an interactive shell
```bash
docker compose exec container-id bash
```

## 5. Run a script
```bash
pixi run synth
```

## 6. Stop the container
```bash
docker compose down
```