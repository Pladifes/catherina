from catherina_comparison.wind_comparaison import *

# TODO: fix legend, check ibtracs' WP and NI

scenario = "historical" # "ssp5_8_5"
model = "ukesm1_0_ll"#]#,"mpi_esm1_2_lr"  "miroc_es2l", "fgoals_g3"]
start = 1980

ibtracs_proc = process_ibtracs_data()
ibtracs = compute_storm_frequency(ibtracs_proc)
columns_to_read = [
        "SID",
        "ISO_TIME_POSIXct",
        "BASIN",
        "wind",
        "step_on_land",
        "distance_to_coast",
        "start_year",
        "seed",
    ]

tracks = pd.DataFrame(columns=[
        "SID",
        "ISO_TIME_POSIXct",
        "BASIN",
        "wind",
        "step_on_land",
        "distance_to_coast",
        "start_year",
        "seed",
    ])

for i in range(1, 40):
    filename = f"outputs/{model}/{scenario}/AR_tracks_proc_25y_s{start}seed{i}v1.2mod{model}_{scenario}_corrected.csv"
    track_int = pd.read_csv(filename, usecols=columns_to_read, low_memory=False)
    tracks = pd.concat([tracks, track_int])

stats, std_dev = compute_freq_cath(tracks)

#%%
std_dev.iloc[:, -1] = 0

#%%
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
axes = axes.ravel()

for i, basin in enumerate(stats["BASIN"]):
    freq = stats.iloc[i, 1:].values
    error = std_dev.iloc[i, 1:].values
    ibtracs_freq = ibtracs[ibtracs["BASIN"] == basin]["freq"].values

    # X coordinates for categories
    x_coords = np.arange(len(CATEGORY_LABELS))

    # Plot bars for all categories except Category 5 on the primary Y axis
    axes[i].bar(
        x_coords[:-1] - 0.15,
        freq[:-1],
        yerr=error[:-1],
        alpha=0.7,
        capsize=5,
        width=0.3,
        label=f"{basin} Catherina",
    )
    axes[i].bar(
        x_coords[:-1] + 0.15,
        ibtracs_freq[:-1],
        alpha=0.7,
        width=0.3,
        color="green",
        label=f"{basin} Ibtracs",
    )
    axes[i].set_ylim(bottom=0)

    # Create a secondary Y axis
    ax2 = axes[i].twinx()

    # Plot bars for Category 5 on the secondary Y axis without labels
    ax2.bar(
        x_coords[-1] - 0.15,
        freq[-1],
        yerr=error[-1],
        alpha=0.7,
        capsize=5,
        width=0.3,
        color="darkblue",
    )
    ax2.bar(
        x_coords[-1] + 0.15, ibtracs_freq[-1], alpha=0.7, width=0.3, color="darkgreen"
    )

    # Set title and X-axis label for the primary Y axis
    axes[i].set_title(f"Cyclone Frequencies in {basin}")
    axes[i].set_xlabel("Category")
    axes[i].set_ylabel("Frequency")

    # Set Y-axis label for the secondary Y axis
    ax2.set_ylabel("Frequency (Cat 5)", color="darkblue")

    # Changing the color of the Y-axis labels to match the bars' color
    ax2.tick_params(axis="y", colors="darkblue")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # Add a vertical line to separate Category 5 from the rest
    axes[i].axvline(x=x_coords[-1] - 0.5, color="gray", linestyle="--", lw=1)

    # Setting the legend (only considering the primary axis)
    lines, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(lines, labels, loc="upper left")

    axes[i].set_xticks(x_coords)
    axes[i].set_xticklabels(CATEGORY_LABELS)
    axes[i].tick_params(axis="x", rotation=45)

    #axes[-1].axis("off")  # remove the last plot

plt.tight_layout()
plt.savefig("outputs/wind_hist_comparison_ibtracs.png")
plt.show()
