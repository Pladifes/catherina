import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings

warnings.filterwarnings("ignore")


# TODO: to be improved
def get_rcp(path):
    # Replace backslashes with forward slashes 
    # in local path has backslash but in s3 path has slashes 
    path = path.replace("\\", "/")
    
    # Split the path into segments
    segments = path.split("/")
    
    # Find the index of the segment containing "damages"
    try:
        damages_index = segments.index("damages")
    except ValueError:
        print("No 'damages' segment found")
        return None

    # Extract and return the segment just before "damages"
    if damages_index > 0:
        segment = segments[damages_index - 1]
        if segment == "historical":
            return segment
        else:
            # Modify the segment if it matches the sspX_Y_Z pattern
            parts = segment.split("_")
            if len(parts) == 3 and parts[0].startswith("ssp"):
                return f"SSP{parts[0][3:]}-RCP{parts[1]}{parts[2]}"
            else:
                print('Segment should be in this patter: sspX_Y_Z.')
                print(f'Pattern given: {segment}')
                return None
    else:
        print("No segment found before 'damages'")
        return  None 


def read_damage_files(df, mode, aggregate=True):
    df_all = pd.DataFrame()
    for _,row in df.iterrows():
        rcp = get_rcp(row['path'])
        if mode == 'local':
            df_tmp = pd.read_csv(row['path'])
        else:
            df_tmp = fetch_from_s3(row['path'])
            df_tmp = extract_csv(df_tmp)
        df_tmp["SSP_RCP"] = rcp 
        
        expected_columns = ["SED", "SED_sqrt", "SED_third", "SED_pop", "SED_gdp", "SED_none"]
        for col in expected_columns:
            if col not in df_tmp.columns:
                df_tmp[col] = np.nan

        if aggregate:
        # Group by and aggregate
            df_tmp = (
                df_tmp.groupby(["SSP_RCP", "year", "seed"])
                .agg({"SED": "sum",
                  "SED_sqrt": "sum",
                  "SED_third": "sum",
                  "SED_pop": "sum",
                  "SED_gdp": "sum",
                  "SED_none": "sum"})
                .reset_index()
             )
        
        df_all = pd.concat([df_tmp, df_all], ignore_index=True)
    
    return df_all



def plot_damage_scenarios(df, scenario_column='SSP_RCP', model_column='model',
                           seed=None,
                           damage_columns=['SED', 'SED_sqrt', 'SED_third', 'SED_pop', 'SED_gdp', 'SED_none']):
    if seed:
        df = df.loc[(df['seed'] == seed) | (df[scenario_column] == 'historical')]

    # Explicitly specify the order of scenarios
    ordered_scenarios = ['historical', 'SSP2-RCP45', 'SSP4-RCP34', 'SSP3-RCP70', 'SSP5-RCP85']
    df = df[df[scenario_column].isin(ordered_scenarios)]  # Filter df to only include the specified scenarios
    scenarios = ordered_scenarios
    models = df[model_column].unique()
    bar_positions = range(len(scenarios))

    sns.set_style("white")
    sns.set_context("talk")

    # Descriptions for each damage column
    damage_descriptions = {
        'SED': '(1,1) Projection without adaptation',
        'SED_gdp': '(1,0) Change of GDP per Capita',
        'SED_sqrt': '(1/2,1) Projection with sqrt adaptation',
        'SED_third': '(1/3,1) Projection with third adaptation',
        'SED_pop': '(0,1) Projection with population growth',
        'SED_none': '(0,0) No exposure growth'
    }

    colors = plt.cm.get_cmap("Set2", len(damage_columns))

    # Create a FacetGrid for faceting per model
    g = sns.FacetGrid(df, col=model_column, col_wrap=2, height=6, aspect=1.5)

    # Initialize the legend handles and labels
    handles, labels = [], []

    def barplot(x, y, **kwargs):
        ax = plt.gca()
        model_data = kwargs.pop('data')
        for i, damage_column in enumerate(damage_columns):
            grouped = model_data.groupby([scenario_column])[damage_column]
            mean_damage_values = (grouped.mean().reindex(scenarios, fill_value=np.nan) / 1e9).tolist()
            std_damage_values = (grouped.std().reindex(scenarios, fill_value=np.nan) / np.sqrt(
                grouped.count().reindex(scenarios, fill_value=np.nan)) / 1e9).tolist()
            p90_damage_values = (grouped.quantile(0.90).reindex(scenarios, fill_value=np.nan) / 1e9).tolist()
            p95_damage_values = (grouped.quantile(0.95).reindex(scenarios, fill_value=np.nan) / 1e9).tolist()

            label = damage_descriptions.get(damage_column, damage_column)

            bars = ax.bar([pos + i * 0.1 for pos in bar_positions],
                          mean_damage_values, width=0.1, label=label,
                          yerr=std_damage_values, color=colors(i),
                          alpha=0.6, edgecolor='black', linewidth=1)
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1)
            ax.bar([pos + i * 0.1 for pos in bar_positions], p90_damage_values, width=0.1, color=colors(i), alpha=0.3, linewidth=1)
            ax.bar([pos + i * 0.1 for pos in bar_positions], p95_damage_values, width=0.1, color=colors(i), alpha=0.1)

        historical_value = df[df[scenario_column] == 'historical']['SED'].mean()
        ax.axhline(y=historical_value / 1e9, color='black', linestyle='--',
                   label='Historical Value')
        ax.set_xticks([pos + 0.1 * (len(damage_columns) - 1) / 2 for pos in bar_positions])
        ax.set_xticklabels(scenarios)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Capture legend handles and labels
        nonlocal handles, labels
        handles, labels = ax.get_legend_handles_labels()

    g.map_dataframe(barplot, scenario_column, damage_columns)

    # Remove x and y axis titles
    for ax in g.axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True)


    # Add a legend for the percentiles
    handles.append(plt.Line2D([0], [0], color='black',
                              alpha=0.8, lw=4, label='Mean Damage'))
    handles.append(plt.Line2D([0], [0], color='black',
                              alpha=0.6, lw=4, label='90th Percentile'))
    handles.append(plt.Line2D([0], [0], color='black',
                              alpha=0.4, lw=4, label='95th Percentile'))

    # Add the legend to the top of the 4th model chart
    ax = g.axes.flatten()[3]  # Get the 4th model chart
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, 0.9), ncol=2, fontsize=12)

    plt.savefig("outputs/annual_damages_facet.png", dpi=200)
    plt.show()

    # Save the statistics to an Excel file
    stats_list = []
    for model in models:
        for scenario in scenarios:
            if not df[(df[scenario_column] == scenario) & (df[model_column] == model)].empty:
                scenario_data = df[(df[scenario_column] == scenario) & (df[model_column] == model)]
                for damage_column in damage_columns:
                    mean_damage = scenario_data[damage_column].mean() / 1e9
                    std_damage = scenario_data[damage_column].std() / np.sqrt(
                        scenario_data[damage_column].count()) / 1e9
                    p75_damage = scenario_data[damage_column].quantile(0.75) / 1e9
                    p95_damage = scenario_data[damage_column].quantile(0.95) / 1e9
                    stats_list.append([model, scenario, damage_column, mean_damage, std_damage, p75_damage, p95_damage])

    stats_df = pd.DataFrame(stats_list, columns=['Model', 'Scenario', 'Damage Type', 'Mean Damage (Billion USD)',
                                                 'STD Damage (Billion USD)', '75th Percentile (Billion USD)',
                                                 '95th Percentile (Billion USD)'])
    stats_df.to_excel("outputs/annual_damages_statistics.xlsx", index=False)


def plot_damage_increase_loess(df, metric='95%', remove_outliers=False):
    # Define custom color palette
    custom_palette = {
        'SSP2-RCP45': 'orange',
        'SSP3-RCP70': 'purple',
        'SSP4-RCP34': 'darkgreen',
        'SSP5-RCP85': 'darkred',
        'historical': 'darkblue'
    }

    is_percentile = '%' in metric
    num_perc = round(float(metric.strip('%')), 0) if is_percentile else None

    # Function to remove outliers
    def remove_outliers_func(data):
        z_scores = zscore(data['SED'])
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3)
        return data[filtered_entries]

    # Remove outliers if requested
    if remove_outliers:
        df = remove_outliers_func(df)

    # Separate historical data
    df['SED'] = df['SED'] / 1e9
    historical_data = df[df['year'] < 2005]

    # Filter years between 2025 and 2099
    scenario_data = df[(df['year'] >= 2025) & (df['year'] <= 2099)]

    if is_percentile:
        # Group by Scenario and Year and compute percentiles for scenarios
        scenario_values = scenario_data.groupby(['SSP_RCP', 'year'])['SED'].quantile([num_perc / 100]).unstack(level=-1)
        scenario_values.columns = [metric]

        # Group by Year and compute percentiles for historical data
        historical_values = historical_data.groupby(['year'])['SED'].quantile([num_perc / 100]).unstack(level=-1)
        historical_values.columns = [metric]
    else:
        # Group by Scenario and Year and compute mean for scenarios
        scenario_values = scenario_data.groupby(['SSP_RCP', 'year'])['SED'].mean().reset_index()
        scenario_values.columns = ['SSP_RCP', 'year', metric]

        # Group by Year and compute mean for historical data
        historical_values = historical_data.groupby(['year'])['SED'].mean().reset_index()
        historical_values.columns = ['year', metric]

    historical_values['SSP_RCP'] = 'historical'

    # Reset index for plotting
    scenario_values = scenario_values.reset_index()
    historical_values = historical_values.reset_index()

    # Plotting the trajectory
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    for scenario in scenario_values['SSP_RCP'].unique():
        scenario_data = scenario_values[scenario_values['SSP_RCP'] == scenario]
        combined_data = pd.concat([historical_values, scenario_data], ignore_index=True)

        # Fit LOESS model to combined data
        x_data = combined_data['year'].values
        y_data = combined_data[metric].values

        # Perform LOESS smoothing
        loess_result = lowess(y_data, x_data, frac=0.6)

        # Get smoothed values
        x_fit = loess_result[:, 0]
        y_fit = loess_result[:, 1]

        # Calculate confidence intervals using an expanding window
        y_fit_series = pd.Series(y_fit, index=x_fit)
        ci_width = y_fit_series.expanding().apply(lambda s: 1.96 * s.std()).values

        # Plot data points
        sns.scatterplot(x='year', y=metric, data=combined_data, label=f'Scenario {scenario} - {metric}', s=50,
                        color=custom_palette[scenario])

        # Plot LOESS fit
        plt.plot(x_fit, y_fit, label=f'{scenario} smooth', color=custom_palette[scenario])

        # Plot confidence intervals
        plt.fill_between(x_fit, y_fit - ci_width, y_fit + ci_width, alpha=0.3, color=custom_palette[scenario])

    # Plot historical data
    sns.scatterplot(x='year', y=metric,
                    data=historical_values,
                    label=f'Historical - {metric}', s=50,
                    color=custom_palette['historical'])

    plt.xlabel('Year')
    plt.ylabel(f'{metric} Damage Value (USD billions)')
    plt.title(f'{metric} Trajectories by SSP-RCP CMIP6 Scenarios', fontsize=14)
    plt.legend()
    plt.savefig(f"outputs/annual_{metric.replace('%', 'th')}_increase.png", dpi=200)
    plt.show()


def plot_damage_increase_loess_grouped(df, metric='95%', remove_outliers=False):
    # Define custom color palette
    custom_palette = {
        'SSP2-RCP45': 'orange',
        'SSP3-RCP70': 'purple',
        'SSP4-RCP34': 'darkgreen',
        'SSP5-RCP85': 'darkred',
        'historical': 'darkblue'
    }

    is_percentile = '%' in metric
    num_perc = round(float(metric.strip('%')), 0) if is_percentile else None

    # Function to remove outliers
    def remove_outliers_func(data):
        z_scores = zscore(data['SED'])
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3)
        return data[filtered_entries]

    # Remove outliers if requested
    if remove_outliers:
        df = remove_outliers_func(df)

    # Separate historical data
    df['SED'] = df['SED'] / 1e9
    historical_data = df[df['year'] < 2005]

    # Filter years between 2025 and 2099
    scenario_data = df[(df['year'] >= 2025) & (df['year'] <= 2099)]

    if is_percentile:
        # Group by Scenario, Country, and Year and compute percentiles for scenarios
        scenario_values = scenario_data.groupby(['SSP_RCP', 'iso3', 'year'])['SED'].quantile(
            [num_perc / 100]).unstack(level=-1).reset_index()
        scenario_values.columns = ['SSP_RCP', 'iso3', 'year', metric]

        # Group by Country and Year and compute percentiles for historical data
        historical_values = historical_data.groupby(['iso3', 'year'])['SED'].quantile([num_perc / 100]).unstack(
            level=-1).reset_index()
        historical_values.columns = ['iso3', 'year', metric]
    else:
        # Group by Scenario, Country, and Year and compute mean for scenarios
        scenario_values = scenario_data.groupby(['SSP_RCP', 'iso3', 'year'])['SED'].mean().reset_index()
        scenario_values.columns = ['SSP_RCP', 'iso3', 'year', metric]

        # Group by Country and Year and compute mean for historical data
        historical_values = historical_data.groupby(['iso3', 'year'])['SED'].mean().reset_index()
        historical_values.columns = ['iso3', 'year', metric]

    historical_values['SSP_RCP'] = 'historical'

    # Combine scenario and historical values for plotting
    combined_values = pd.concat([scenario_values, historical_values], ignore_index=True)

    # Plotting the trajectory with facet wrapping by country
    g = sns.FacetGrid(combined_values, col="iso3", col_wrap=3, height=3, aspect=1.5, sharey=False)

    def plot_loess(data, **kwargs):
        country = data['iso3'].iloc[0]
        historical_country_data = historical_values[historical_values['iso3'] == country]

        for scenario in data['SSP_RCP'].unique():
            scenario_data = data[data['SSP_RCP'] == scenario]
            combined_data = pd.concat([historical_country_data, scenario_data], ignore_index=True)

            # Fit LOESS model to combined data
            x_data = combined_data['year'].values
            y_data = combined_data[metric].values

            # Perform LOESS smoothing
            loess_result = lowess(y_data, x_data, frac=0.6)

            # Get smoothed values
            x_fit = loess_result[:, 0]
            y_fit = loess_result[:, 1]

            # Calculate confidence intervals using an expanding window
            y_fit_series = pd.Series(y_fit, index=x_fit)
            ci_width = y_fit_series.expanding().apply(lambda s: 1.96 * s.std()).values

            # Plot LOESS fit
            plt.plot(x_fit, y_fit, label=f'{scenario}', color=custom_palette[scenario])

            # Plot confidence intervals
            plt.fill_between(x_fit, y_fit - ci_width, y_fit + ci_width, alpha=0.3, color=custom_palette[scenario])

    g.map_dataframe(plot_loess)

    # g.set_axis_labels('Year', f'{metric} Damage Value (USD billions)')
    g.add_legend()
    plt.subplots_adjust(top=0.9)
   # g.fig.suptitle(f'{metric} Trajectories by SSP-RCP CMIP6 Scenarios by Country', fontsize=16)
    plt.savefig(f"outputs/annual_{metric.replace('%', 'th')}_increase_by_country.png", dpi=200)
    plt.show()
