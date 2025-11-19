import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from catherina_thermo_generation.cyclone_generation import lon_to_360

# List of scenarios and models to iterate over
scenarios = ["ssp5_8_5", "historical"]
models = ["ukesm1_0_ll"]#, "miroc_es2l", "fgoals_g3"]
start_def = 2075  # Start year
percentile = 98

if __name__ == '__main__':

    def lon_to_360(df):
        df['LON'] = df['LON'] % 360
        return df

    for scenario in scenarios:
        if scenario == "historical":
            start = 1980
        else:
            start = start_def
        for model in models:
            tracks = pd.DataFrame(columns=["LAT", "LON", "wind"])
            columns_to_read = ["LAT", "LON", "wind"]

            for i in range(1, 60):
                filename = f"outputs/{model}/{scenario}/AR_tracks_proc_25y_s{start}seed{i}v1.2mod{model}_{scenario}_corrected.csv"
                track_int = pd.read_csv(filename, usecols=columns_to_read, low_memory=False)
                tracks = pd.concat([tracks, track_int])

            tracks = lon_to_360(tracks)
            colors = [(0.8, 0.8, 0.9), (0.5, 0.0, 0.0)]  # Light Blue to Red
            cm = LinearSegmentedColormap.from_list("BlueToRed", colors, N=256)

            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(10, 5))
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.LAND, edgecolor="black")

            hb = ax.hexbin(
                tracks["LON"],
                tracks["LAT"],
                C=tracks["wind"],
                gridsize=75,
                mincnt=10,
                cmap=cm,
                vmin=50,
                vmax=100,
                reduce_C_function=lambda x: np.percentile(x, percentile),
                transform=ccrs.PlateCarree(),
            )

            cbar = fig.colorbar(hb, ax=ax, orientation="horizontal", pad=0.02, extend="both")
            cbar.set_label("Wind Intensity (speed in m/s)")

            plt.title(f"Wind Intensity Density ({percentile}th percentile value) - {model} - {scenario}")
            fig.savefig(f"outputs/windmaps_{model}_{scenario}_{start}_{start + 25}_perc{percentile}.pdf", format="pdf",
                        bbox_inches="tight")
            plt.show()

