from catherina_comparison.landfall_comparison import *

##TODO: This function performs calculations on the first seed (for historical).
## Can be generalized to averaging different seeds.

with open('VERSION', 'r') as version_file:
    version = version_file.read().strip()

USECOLS = [
    "SID",
    "ISO_TIME_POSIXct",
    "BASIN",
    "step",
    "seed",
    "distance_to_coast",
    "step_on_land",
    "wind",
]

MODE = "local"

# Models to compare Ibtracs (historical) with
MODELS = ["ukesm1_0_ll", "access_cm2", "mpi_esm1_2_lr", #"ERA5",
           "miroc_es2l", "fgoals_g3"]#, "ipsl_cm5a2_inca"]

EXPERIMENT = "historical"  # historical or future
KEEP_HIST = True  # True to include historical data with experiments
START_YEAR = 2050  # To target only the seeds of that start year
N_year = 25
ftd_year = 2023
threshold = 35

if __name__ == "__main__":
    df = pd.DataFrame()
    if EXPERIMENT == "historical":
        for model in MODELS:
            print(f"Processing {model}..")
            if model == "ERA5":
                df_mod = pd.read_csv("outputs/1500y_hist.csv")
                df_mod = df_mod[df_mod["seed"] == 1]
            else:
                df_mod = load_data(
                    f"outputs/{model}/{EXPERIMENT}/AR_tracks_proc_{N_year}y_s1980seed1v{version}mod{model}_{EXPERIMENT}_corrected.csv",
                    USECOLS,
                    mode=MODE,
                )
            if df_mod is None:
                print(f"No data available for {model}")
                continue
            df_mod_stats = compute_cylone_stats(df_mod,
                                                threshold=threshold)
            df_mod_stats["model"] = model
            df = pd.concat([df, df_mod_stats])
            print(f"{model} processed..")

        # Ibtracs
        print("Processing IBTrACs..")
        filepath_ibtracs = "ibtracs/ibtracs.ALL.list.v04r00.csv"
        ibtracs = load_ibtracs_data(filepath_ibtracs, skip_row_func, mode=MODE)

        df_stats_ibtracs = preprocess_ibtracs_data(ibtracs,
                                                   ftd_year=ftd_year,
                                                   threshold=threshold)
        df = pd.concat([df, df_stats_ibtracs])
        plot_landfall_comparison(df)

    else:
        for model in MODELS:
            print(f"Processing {model}..")
            df_model = read_experiments_compute_stats(
                model, start_year=START_YEAR, keep_hist=KEEP_HIST, mode=MODE
            )
            df = pd.concat([df, df_model])
        plot_landfall_comparison_proj(df)
