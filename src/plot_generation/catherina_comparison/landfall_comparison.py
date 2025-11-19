import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from catherina_thermo_generation.cyclone_generation import full_processing, filter_wind
import os

plt.rcParams.update(
    {
        "axes.labelsize": 12,  # Sets the font size of the x and y labels
        "axes.titlesize": 16,  # Sets the size of the title font
        "axes.grid": True,  # Enables grid lines
        "axes.prop_cycle": plt.cycler(
            "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        ),  # Color cycle for plot elements
        "legend.fontsize": "medium",  # Sets the font size of the legend
        "xtick.labelsize": 12,  # Sets the font size of the x-tick labels
        "ytick.labelsize": 12,  # Sets the font size of the y-tick labels
        "font.size": 10,  # The default font size
    }
)


# PROCESS ERA5
def load_data(filepath, usecols, mode="S3"):
    """Load the data from a CSV file."""
    if mode == "local":
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath, usecols=usecols)
        else:
            return pd.read_excel(filepath, usecols=usecols)

    else:
        data = fetch_from_s3(filepath)
        df = extract_csv(data)
        return df


def compute_cylone_stats(df, n_seed=None, threshold=35):
    """Process data on all seeds or only 1 seed"""
    df.BASIN = df.BASIN.fillna("NA")
    df_wind = filter_wind(df, threshold=threshold, convert=False)
    df_land = df_wind[~df_wind['step_on_land'].isna()]

    # Group by 'SID' and filter groups where the minimum 'step_on_land' is less than 3
    df_land = df_land.groupby('SID').filter(lambda x: x['step_on_land'].max() > 5).reset_index()

    if n_seed:
        # Processing for a specific seed
        df_land = df_land[df_land["seed"] == n_seed]
        df_land["ISO_TIME_POSIXct"] = pd.to_datetime(df_land["ISO_TIME_POSIXct"])
        df_land["Year"] = df_land["ISO_TIME_POSIXct"].dt.year
        df_unique = df_land.drop_duplicates(subset=["SID", "BASIN"])
        df_final = (
            df_unique.groupby(["BASIN", "Year"])["SID"]
            .nunique()
            .reset_index(name="count")
        )

        # Compute statistics for specific seed
        df_final = (
            df_final.groupby(["BASIN"])["count"].agg(["mean", "std"]).reset_index()
        )
        df_final["model"] = f"ERA5_seed{n_seed}"
    else:
        # Processing for all seeds
        df_land["ISO_TIME_POSIXct"] = pd.to_datetime(df_land["ISO_TIME_POSIXct"])
        df_land["Year"] = df_land["ISO_TIME_POSIXct"].dt.year
        df_unique = df_land.drop_duplicates(subset=["SID", "seed", "BASIN", "Year"])
        df_final = (
            df_unique.groupby(["BASIN", "Year", "seed"])["SID"]
            .nunique()
            .reset_index(name="count")
        )

        # Averaging over seeds and compute statistics for all seeds
        df_final = df_final.groupby(["BASIN", "Year"])["count"].mean().reset_index()
        df_final = df_final.groupby("BASIN")["count"].agg(["mean", "std"]).reset_index()
        df_final["model"] = "ERA5"

    return df_final


def preprocess_ibtracs_data(df, ftd_year=2010, threshold=35):
    """Preprocess the data using a full processing function and additional filters."""
    # Full processing (assumed to be a custom function from the 'ok' module)
    df_processed = full_processing(df, threshold=threshold)

    # Convert the column to datetime objects and extract the year
    df_processed["ISO_TIME_POSIXct"] = pd.to_datetime(df_processed["ISO_TIME_POSIXct"])
    df_processed["Year"] = df_processed["ISO_TIME_POSIXct"].dt.year

    # Filter the data
    df_filtered = df_processed[df_processed["Year"] < ftd_year]
    # df_wind = df_filtered[df_filtered['wind'] >= wind_tresh]
    df_land = df_filtered[df_filtered["DIST2LAND"] == 0]
    df_unique = df_land.drop_duplicates(subset=["SID", "BASIN"])
    df_final = (
        df_unique.groupby(["BASIN", "Year"])["SID"].nunique().reset_index(name="count")
    )

    # Compute statistics
    stats = df_final.groupby(["BASIN"])["count"].agg(["mean", "std"]).reset_index()
    stats["model"] = "IBTrACS (historical)"
    return stats


# Define the skip row function
def skip_row_func(row_number):
    return row_number == 1  # skip the second row


def load_ibtracs_data(filepath, skip_func, mode="S3"):
    """Load the data from a CSV file, skipping rows based on a given function."""
    if mode == "local":
        return pd.read_csv(filepath, skiprows=skip_func, low_memory=False)
    else:
        data = fetch_from_s3(filepath)
        csv_file = BytesIO(data)
        return pd.read_csv(csv_file, skiprows=skip_func, low_memory=False)


def read_experiments_compute_stats(
    model, keep_hist=False, start_year=2050, mode="local"
):
    """
        Reads experimental data for a given model from a local directory and computes
    the average and standard deviation of the maximum wind values across experiments
    and seeds.

    Parameters:
    - model (str): The name of the model to read experiments for.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the following columns:
        - 'model': Name of the model.
        - 'experiment': Name of each experiment.
        - 'avg_max_wind': Average of the maximum wind values across seeds for each experiment.
        - 'std_max_wind': Standard deviation of the maximum wind values across seeds for each experiment.
    """
    df = pd.DataFrame(columns=["model", "experiment", "avg_max_wind", "std_max_wind"])
    model_path = os.path.join("outputs", model)
    if mode == "local":
        if os.path.isdir(model_path):
            for experiment in os.listdir(model_path):
                print(f"..Processing {experiment}")
                if experiment == "historical" and not keep_hist:
                    continue
                experiment_path = os.path.join(model_path, experiment)
                if os.path.isdir(experiment_path):
                    df_seeds = pd.DataFrame()  # Regroup for multiple seeds in one df
                    for file in os.listdir(experiment_path):  # loop on seeds
                        if file.endswith(".csv") and (
                            (experiment == "historical")
                            or (experiment != "historical" and str(start_year) in file)
                        ):
                            csv_path = os.path.join(experiment_path, file)
                            df_seed = pd.read_csv(csv_path)
                            df_seeds = pd.concat([df_seeds, df_seed])

                    if len(df_seeds) > 0:
                        mean, std = compute_max_wind_exp(df_seeds)
                        df.loc[len(df)] = [model, experiment, mean, std]
        else:
            print(f"Model folder '{model}' not found.")

    elif mode == "S3":
        file_paths = list_files_in_s3_folder(f"outputs/{model}")

        # convert the paths to a dataframe for easier manipulations
        df_paths = convert_paths_to_df(file_paths)
        for experiment in df_paths["experiment"].unique():
            print(f"..Processing {experiment}")
            df_seeds = pd.DataFrame()

            if experiment == "historical":
                if not keep_hist:
                    continue
                else:
                    df_experiment = df_paths[
                        (df_paths["experiment"] == experiment)
                        & (df_paths["start_year"] == 1980)
                    ]
                    for file in df_experiment["file"].unique():
                        print(file)
                        df_seed = extract_csv(
                            fetch_from_s3(f"outputs/{model}/{experiment}/{file}")
                        )
                        df_seeds = pd.concat([df_seeds, df_seed])

                    mean, std = compute_max_wind_exp(df_seeds)
                    df.loc[len(df)] = [model, experiment, mean, std]

            else:
                df_experiment = df_paths[
                    (df_paths["experiment"] == experiment)
                    & (df_paths["start_year"] == start_year)
                ]
                for file in df_experiment["file"].unique():
                    print(file)
                    df_seed = extract_csv(
                        fetch_from_s3(f"outputs/{model}/{experiment}/{file}")
                    )
                    df_seeds = pd.concat([df_seeds, df_seed])

                if len(df_seeds) > 1:
                    mean, std = compute_max_wind_exp(df_seeds)
                    df.loc[len(df)] = [model, experiment, mean, std]

    return df


def compute_max_wind_exp(df_seeds):
    """Calculates the mean and avg of max wind in a given experiment and a given model
    df_seeds contain multiple seeds
    """
    # groupby seed, compute max wind, calculate mean and std
    df_seeds = filter_wind(df_seeds, threshold=35, convert=False)

    max_wind = df_seeds.groupby("seed")["wind"].max()
    return max_wind.mean(), max_wind.std()


def convert_paths_to_df(file_paths):
    """
    Convert a list of file paths to a pandas DataFrame.

    This function takes a list of file paths (specifically formatted as found in an S3 bucket)
    and converts them into a DataFrame with specific columns. Each column represents a different
    part of the file path, along with the extracted start year from the file name.

    Parameters:
    file_paths (list of str): A list of file paths to convert. Each path is expected to be in the format:
                              'outputs/model_name/experiment_name/file_name.csv'

    Returns:
    pandas.DataFrame: A DataFrame with the columns ['outputs', 'model', 'experiment', 'file', 'start_year'].
                      Each row represents a file, with 'start_year' extracted from the file name.
    """

    df_paths = pd.DataFrame(
        columns=["outputs", "model", "experiment", "file", "start_year"]
    )
    for file in file_paths:
        l = file.split("/")
        str_year = l[-1].split("y_s")[1][:4]
        l.append(int(str_year))
        df_paths.loc[len(df_paths)] = l
    return df_paths


# Plot functions
def plot_landfall_comparison(df):
    sns.set_style("white")
    sns.set_context("talk")
    plt.figure(figsize=(10, 6))

    basins = sorted(df["BASIN"].unique())
    models = sorted(df["model"].unique())

    bar_width = 0.8 / len(models)
    x = np.arange(len(basins))  # the label locations

    colors = plt.cm.get_cmap("Set2", len(models))

    for i, model in enumerate(models):
        model_means = np.zeros(len(basins))
        model_stds = np.zeros(len(basins))

        for j, basin in enumerate(basins):
            if ((df["BASIN"] == basin) & (df["model"] == model)).any():
                model_data = df[(df["BASIN"] == basin) & (df["model"] == model)]
                model_means[j] = model_data["mean"]
                model_stds[j] = model_data["std"]
            else:
                model_means[j] = np.nan
                model_stds[j] = np.nan

        plt.bar(
            x + i * bar_width,
            model_means,
            yerr=model_stds,
            width=bar_width,
            capsize=5,
            label=model,
            color=colors(i),
            error_kw={"elinewidth": 1, "ecolor": "black"},
        )

    plt.xlabel("Basin")
    plt.ylabel("Count")
    #plt.title("Landfall count cyclone over 35 m/s")
    plt.xticks(x + bar_width * len(models) / 2, basins)
    plt.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Remove the top and right spines
    sns.despine()

    plt.savefig("outputs/landfall_historical_count.png", dpi=200)
    plt.show()


def plot_landfall_comparison_proj(df):
    sns.set_style("white")
    sns.set_context("talk")
    plt.figure(figsize=(13, 6))

    models = sorted(df["model"].unique())
    experiments = sorted(df["experiment"].unique())

    bar_width = 0.6 / len(experiments)
    x = np.arange(len(models))  # the label locations

    colors = plt.cm.get_cmap("Set2", len(experiments))

    for i, experiment in enumerate(experiments):
        model_means = []
        model_stds = []
        model_positions = []

        for j, model in enumerate(models):
            model_data = df[(df["model"] == model) & (df["experiment"] == experiment)]
            if not model_data.empty:
                mean_val = model_data["avg_max_wind"].values[0]
                std_val = model_data["std_max_wind"].values[0]

                # Only add to the plot if the mean value is not NaN
                if not np.isnan(mean_val):
                    model_means.append(mean_val)
                    model_stds.append(std_val if not np.isnan(std_val) else 0)
                    model_positions.append(x[j] + i * bar_width)

        plt.bar(
            model_positions,
            model_means,
            yerr=model_stds,
            width=bar_width,
            capsize=5,
            label=experiment,
            color=colors(i),
            error_kw={"elinewidth": 1, "ecolor": "black"},
        )

    plt.xlabel("Model")
    plt.ylabel("Average Max Wind")
    plt.title("Average Max Wind by Model and Experiment")
    plt.xticks(x + bar_width * len(experiments) / 2, models)
    plt.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.ylim(55, 100)
    sns.despine()
    plt.savefig("outputs/avg_max_wind_projections.png", dpi=200)
    plt.show()
