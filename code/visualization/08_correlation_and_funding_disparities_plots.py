# Imports
import json
import os
from collections import Counter

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.spatial import KDTree, distance

"""
This script performs data analysis and visualization for aid funding data,
focusing on the relationship between funding and disease burden across different countries and income groups.

The main steps include:
1. Loading and preprocessing funding data
2. Adding country codes, population data, and health spending data
3. Merging with Global Burden of Disease (GBD) data
4. Calculating funding gaps and creating visualizations
5. Generating correlation heatmaps
"""

# Constants and Configuration
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "aid_funding_data"
)
BACKGROUND_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "background_data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed", "aid_funding_data")

categories = [
    "HIV/AIDS and sexually transmitted infections",
    "Respiratory infections and tuberculosis",
    "Enteric infections",
    "Neglected tropical diseases and malaria",
    "Maternal and neonatal disorders",
    "Nutritional deficiencies",
    "Neoplasms",
    "Cardiovascular diseases",
    "Chronic respiratory diseases",
    "Digestive diseases",
    "Neurological disorders",
    "Mental disorders",
    "Substance use disorders",
    "Diabetes and kidney diseases",
    "Skin and subcutaneous diseases",
    "Sense organ diseases",
    "Musculoskeletal disorders",
]

# Load SIDS list
with open(os.path.join(BACKGROUND_DIR, "sids_list.json")) as f:
    sids_list = json.load(f)

sids_codes = [item[1] for item in sids_list]

# Function Definitions


def load_and_preprocess_funding_data() -> pd.DataFrame:
    """
    Load and preprocess the funding data with the LLM and keyword classifications.

    Returns:
        pd.DataFrame: Preprocessed dataframe with funding data grouped by classification, year, recipient, and income group.
    """

    # Load the aid funding data
    dtypes = {"CrsID": pl.Utf8}
    df = pl.read_csv(os.path.join(ARTIFACT_DIR, "final_df.csv"), dtypes=dtypes)

    # Convert Year to integer
    df = df.with_columns(pl.col("Year").cast(pl.Float32).cast(pl.Int32))

    # Filter out rows where 'llm_classification' is "Other"
    df = df.filter(pl.col("llm_classification") != "Other")

    # Split the multi-label column into individual labels
    df = df.with_columns(
        pl.col("llm_classification")
        .map_elements(
            lambda x: [
                classification.strip()
                for classification in str(x).split(",")
                if classification.strip()
            ]
        )
        .alias("llm_classification")
    )

    # Explode the DataFrame to create a row for each label
    df = df.explode("llm_classification")

    # Group and aggregate the aid funding data
    funding_df = df.group_by(
        ["llm_classification", "Year", "RecipientCode", "IncomegroupName"]
    ).agg(pl.col("adjusted_funding_llm").sum().alias("USD_Disbursement"))

    funding_df.write_csv(os.path.join(ARTIFACT_DIR, "funding_df.csv"))

    return funding_df.to_pandas()


def add_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add country codes to the aid funding data.

    Args:
        df (pd.DataFrame): Aid funding dataframe grouped by classification, year, recipient, and income group.

    Returns:
        pd.DataFrame: Dataframe with added country codes
    """

    # Load country codes
    with open(os.path.join(BACKGROUND_DIR, "country_codes.json"), "r") as f:
        country_codes = json.load(f)

    # Remove values where the iso column is empty -> removes the non-country regions i.e. "central europe, regional"
    country_codes = {
        key: value
        for key, value in country_codes.items()
        if value["iso"] != ""
    }

    recipient_code_to_name = {
        int(key): value["name"] for key, value in country_codes.items()
    }
    recipient_code_to_iso = {
        int(key): value["iso"] for key, value in country_codes.items()
    }

    # Add country codes to the aid funding data
    df["country_code"] = df["RecipientCode"].map(recipient_code_to_iso)
    df["country_name"] = df["RecipientCode"].map(recipient_code_to_name)
    return df


def add_population_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add population data to the aid funding data.

    Args:
        df (pd.DataFrame): Aid funding dataframe grouped by classification, year, recipient, and income group.

    Returns:
        pd.DataFrame: Merged dataframe with population data
    """
    population = pd.read_csv(
        os.path.join(BACKGROUND_DIR, "populationdata_from_worldbank.csv")
    )
    population = population.rename(columns={"Country Code": "country_code"})
    population = population.drop(
        columns=["Country Name", "Indicator Name", "Indicator Code"],
        errors="ignore",
    )
    population = population.melt(
        id_vars=["country_code"], var_name="Year", value_name="population"
    )

    df["Year"] = df["Year"].astype(int)
    population["Year"] = population["Year"].astype(int)

    merged_df = df.merge(population, on=["country_code", "Year"], how="left")
    merged_df["mean_population"] = merged_df.groupby("country_code")[
        "population"
    ].transform("mean")

    # Manually adjust countries with missing population data
    population_adjustments = {
        "COK": 15200,
        "ERI": 3497117,
        "MSR": 4989,
        "NIU": 15200,
        "WLF": 11700,
        "TKL": 1914,
    }
    for country_code, population in population_adjustments.items():
        merged_df.loc[
            merged_df["country_code"] == country_code, "mean_population"
        ] = population

    merged_df["USD_Disbursement_per_capita"] = (
        merged_df["USD_Disbursement"] * 1000000
    ) / (merged_df["population"])

    return merged_df[merged_df["country_code"].notna()]


def add_health_spend_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add health spend data to the aid funding data.

    Args:
        df (pd.DataFrame): Aid funding dataframe grouped by classification, year, recipient, and income group.

    Returns:
        pd.DataFrame: Merged dataframe with health spend data
    """
    health_spend_df = pd.read_csv(
        os.path.join(
            BACKGROUND_DIR, "IHME_HEALTH_SPENDING_1995_2021_Y2024M07D23.CSV"
        )
    )
    health_spend_df["healthspend_per_capita"] = (
        health_spend_df["ghes_per_cap_mean"]
        + health_spend_df["ppp_per_cap_mean"]
        + health_spend_df["oop_per_cap_mean"]
    )

    return df.merge(
        health_spend_df[["iso3", "healthspend_per_capita", "year"]],
        left_on=["country_code", "Year"],
        right_on=["iso3", "year"],
        how="left",
    ).drop(columns=["iso3", "year"])


def add_gni_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add GNI (Gross National Income) per capita data to the aid funding data.

    Args:
        df (pd.DataFrame): Aid funding dataframe grouped by classification, year, recipient, and income group.

    Returns:
        pd.DataFrame: Merged dataframe with GNI per capita data
    """
    # Load GNI data, skipping the first 4 rows which contain metadata
    gni_per_capita_df = pd.read_csv(
        os.path.join(
            BACKGROUND_DIR, "API_NY.GNP.PCAP.CD_DS2_en_csv_v2_665.csv"
        ),
        skiprows=4,
    )

    # Rename the country code column and melt the year columns
    gni_per_capita_df = gni_per_capita_df.rename(
        columns={"Country Code": "iso3"}
    )
    gni_per_capita_df = gni_per_capita_df.melt(
        id_vars=["iso3"],
        value_vars=[
            str(year) for year in range(2000, 2023)
        ],  # Adjust year range as needed
        var_name="year",
        value_name="gni_per_capita",
    )

    # Convert year to integer
    gni_per_capita_df["year"] = gni_per_capita_df["year"].astype(int)

    # Merge with the main dataframe
    return df.merge(
        gni_per_capita_df[["iso3", "year", "gni_per_capita"]],
        left_on=["country_code", "Year"],
        right_on=["iso3", "year"],
        how="left",
    ).drop(columns=["iso3", "year"])


def merge_with_gbd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the disease outcome data from the Global Burden of Disease (GBD) with the aid funding data.

    Args:
        df (pd.DataFrame): Aid funding dataframe grouped by classification, year, recipient, and income group.

    Returns:
        pd.DataFrame: Merged dataframe with GBD data
    """
    gbd_df = pd.read_csv(
        os.path.join(BACKGROUND_DIR, "IHME-GBD_2021_DATA-96af6e94-1.csv")
    )
    df_gbd_with_iso3 = pd.read_csv(
        os.path.join(BACKGROUND_DIR, "df_gbd_with_iso3.csv")
    )
    df_gbd_with_iso3.loc[
        df_gbd_with_iso3["Location Name"] == "Türkiye", "Location Name"
    ] = "Turkey"

    gbd_df = gbd_df.merge(
        df_gbd_with_iso3, left_on="location_id", right_on="Location ID"
    )
    gbd_df = gbd_df.rename(
        columns={"ISO3 Code": "country_code", "location_name": "country"}
    )
    gbd_df = gbd_df.dropna(subset=["country_code"])

    gbd_dalys_df = gbd_df[
        gbd_df["measure_name"] == "DALYs (Disability-Adjusted Life Years)"
    ].reset_index(drop=True)
    gbd_incidence_df = gbd_df[
        gbd_df["measure_name"] == "Incidence"
    ].reset_index(drop=True)
    gbd_incidence_df = gbd_incidence_df.rename(
        columns={"val": "incidence_rate"}
    )
    gbd_dalys_df = gbd_dalys_df.rename(columns={"val": "dalys_rate"})

    gbd_dalys_per_cause = (
        gbd_dalys_df.groupby(["cause_name", "country"])
        .agg({"dalys_rate": "mean"})
        .reset_index()
    )

    # Calculate total DALYs per country by summing across all causes
    total_dalys_per_country = (
        gbd_dalys_per_cause.groupby("country")["dalys_rate"]
        .sum()
        .reset_index()
    )

    # Merge total DALYs back into gbd_dalys_per_cause
    gbd_dalys_per_cause = gbd_dalys_per_cause.merge(
        total_dalys_per_country,
        on="country",
        how="left",
        suffixes=("", "_total"),
    )

    # Calculate DALYs percentage per country for each cause
    gbd_dalys_per_cause["dalys_percentage"] = (
        gbd_dalys_per_cause["dalys_rate"]
        / gbd_dalys_per_cause["dalys_rate_total"]
    )

    # Merge gbd_dalys_per_cause with gbd_dalys_df to include dalys_percentage
    gbd_dalys_df = gbd_dalys_df.merge(
        gbd_dalys_per_cause[
            ["cause_name", "country", "dalys_rate_total", "dalys_percentage"]
        ],
        left_on=["cause_name", "country"],
        right_on=["cause_name", "country"],
        how="left",
    )

    merged_df = df.merge(
        gbd_incidence_df[
            ["country_code", "incidence_rate", "year", "cause_name"]
        ],
        left_on=["country_code", "Year", "llm_classification"],
        right_on=["country_code", "year", "cause_name"],
        how="left",
    ).drop(columns=["year", "cause_name"])

    merged_df = merged_df.merge(
        gbd_dalys_df[
            [
                "country_code",
                "dalys_rate",
                "year",
                "cause_name",
                "dalys_percentage",
                "dalys_rate_total",
            ]
        ],
        left_on=["country_code", "Year", "llm_classification"],
        right_on=["country_code", "year", "cause_name"],
        how="left",
    ).drop(columns=["year", "cause_name"])

    return merged_df


def plot_geoframe_subplots_rwb(
    df: pd.DataFrame,
    file_name: str,
    no_indicator: set,
    column: str,
    folder_path: str = "all",
) -> None:
    """
    Plot geographic subplots based on the provided dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        file_name (str): Name of the file to save the plot
        no_indicator (set): Set of countries with no indicator data
        column (str): Column to use for plotting
    """

    file_path = os.path.join(BACKGROUND_DIR, "ne_110m_admin_0_countries.shp")
    gdf = gpd.read_file(file_path)

    # Exclude Antarctica
    gdf = gdf[gdf.NAME != "Antarctica"]

    merged_df = gdf.merge(
        df, left_on="ADM0_A3", right_on="country_code", how="left"
    )
    fig, ax = plt.subplots(
        figsize=(12, 8), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Create a custom colormap
    cmap = plt.get_cmap("RdBu_r")
    norm = colors.Normalize(vmin=-100, vmax=100)

    # Plot the data
    merged_df.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.2,
        missing_kwds={"color": "lightgray", "label": "No reported aid"},
        transform=ccrs.PlateCarree(),
    )

    if len(no_indicator) > 0:
        if not merged_df[merged_df["country_code"].isin(no_indicator)].empty:
            merged_df[merged_df["country_code"].isin(no_indicator)].plot(
                ax=ax,
                edgecolor="indianred",
                hatch="////",
                facecolor="whitesmoke",
                linewidth=0.2,
                legend=False,
                transform=ccrs.PlateCarree(),
            )

    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar
    cbar = plt.colorbar(
        sm, ax=ax, orientation="vertical", pad=0.02, aspect=20, shrink=0.5
    )
    cbar.set_label(
        "Gap in aid and need percentiles",
        rotation=270,
        labelpad=15,
        fontsize=14,
    )

    # Get current tick locations and create new labels with %
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}%" for tick in ticks])

    # Create custom legend for 'No reported aid' and 'No indicator data'
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="No reported aid",
            markerfacecolor="lightgray",
            markeredgecolor="lightgray",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="No disease data",
            markerfacecolor="whitesmoke",
            markeredgecolor="indianred",
            markersize=10,
            linestyle="none",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    # Remove x and y axis from the map plot
    ax.set_axis_off()
    ax.set_title(f"{file_name}", fontsize=16)

    plt.savefig(
        os.path.join(
            PLOTS_DIR, "Map_plots", folder_path, f"{file_name}_rwb.pdf"
        ),
        bbox_inches="tight",
    )
    plt.close()


def create_split_correlation_plots(
    df: pd.DataFrame,
    disease_cause: str,
    disease_metric: str,
    aid_metric: str,
    folder_path: str = "all",
) -> None:
    """
    Create enhanced correlation plots for aid disbursement vs DALYs rate for different income groups,
    with points colored uniformly based on income group and Spearman rank correlation lines.
    """

    # Removing NaN and inf values
    df = df.dropna(subset=[aid_metric, disease_metric])
    df = df[df[disease_metric] != np.inf]

    # Create figure
    fig = plt.figure(figsize=(18, 4.7), dpi=300)

    # Create GridSpec - no need for extra space for colorbar
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    income_group_names = ["LDCs", "LMICs", "UMICs"]
    markers = {"LDCs": "o", "LMICs": "o", "UMICs": "o"}

    # Define a color for each income group
    colors = {
        "LDCs": "#1f77b4",  # Blue
        "LMICs": "#ff7f0e",  # Orange
        "UMICs": "#2ca02c",  # Green
    }

    for idx, income_group in enumerate(income_group_names):
        mask = df["IncomegroupName"] == income_group
        data = df[mask]

        # Create normalized points for annotation spacing - FIXED ORDER
        x_max = data[disease_metric].max()  # disease_metric is x-axis
        y_max = data[aid_metric].max()  # aid_metric is y-axis
        normalized_points = np.array(
            [
                (x / x_max, y / y_max)
                for x, y in zip(data[disease_metric], data[aid_metric])
            ]
        )  # FIXED ORDER

        # Create scatter plot with uniform color based on income group - FIXED ORDER
        scatter = axes[idx].scatter(
            data[disease_metric],  # x-axis: disease metric
            data[aid_metric],  # y-axis: aid metric
            c=colors[income_group],
            s=80,
            marker=markers[income_group],
            edgecolor="black",
            linewidth=1,
            alpha=0.9,
        )

        if len(data):
            # Calculate Spearman rank correlation for reporting - FIXED ORDER
            spearman_corr, p_value = scipy.stats.spearmanr(
                data[disease_metric], data[aid_metric]
            )

            # But fit a regular linear regression line for visualization - FIXED ORDER
            slope, intercept = np.polyfit(
                data[disease_metric], data[aid_metric], 1
            )

            # Create x and y values for the line - FIXED ORDER
            x_range = np.linspace(
                data[disease_metric].min(), data[disease_metric].max(), 100
            )
            y_line = slope * x_range + intercept

            # Plot the correlation line
            axes[idx].plot(
                x_range,
                y_line,
                color=colors[income_group],
                linestyle="--",
                linewidth=2,
            )

            # Add correlation text
            corr_str = f"r = {round(spearman_corr, 2)}"
            p_str = f"p = {round(p_value, 2)}"
            n_str = f"n = {len(data)}"

            # Position the text in the upper portion of the plot - FIXED ORDER
            text_x = data[disease_metric].min() + 0.03 * (
                data[disease_metric].max() - data[disease_metric].min()
            )
            text_y = data[aid_metric].max() - 0.15 * (
                data[aid_metric].max() - data[aid_metric].min()
            )

            axes[idx].text(
                text_x,
                text_y,
                f"{corr_str}\n{p_str}\n{n_str}",
                fontsize=14,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7),
            )

        # Add country name annotations
        if (
            len(normalized_points) > 1
        ):  # Only create KDTree if we have more than 1 point
            tree = KDTree(normalized_points)

            # First pass: identify pairs of points that are close to each other but far from others
            isolated_pairs = {}
            for i, row in data.iterrows():
                current_point = np.array(
                    [row[disease_metric] / x_max, row[aid_metric] / y_max]
                ).reshape(
                    1, -1
                )  # FIXED ORDER

                # Get the 3 nearest neighbors (including self)
                k = min(len(normalized_points), 3)
                distances, indices = tree.query(current_point, k=k)

                # Skip if we don't have enough points
                if indices[0][0] >= len(normalized_points):
                    continue

                # If we have exactly 2 close points and the third is far (or doesn't exist)
                if (
                    k > 2
                    and distances[0][1] < 0.15
                    and (k == 2 or distances[0][2] > 0.3)
                ):
                    point_key = tuple(
                        sorted([i, indices[0][1]])
                    )  # Create unique key for the pair
                    isolated_pairs[point_key] = True

            # Second pass: annotate points considering isolated pairs
            for i, row in data.iterrows():
                current_point = np.array(
                    [row[disease_metric] / x_max, row[aid_metric] / y_max]
                ).reshape(
                    1, -1
                )  # FIXED ORDER

                # Get nearest neighbor
                k = min(len(normalized_points), 2)
                distances, indices = tree.query(current_point, k=k)

                if indices[0][0] >= len(normalized_points):
                    continue

                nearest_idx = indices[0][1] if k > 1 else indices[0][0]
                nearest_neighbor = normalized_points[nearest_idx]

                # Compare with the flattened current point
                x_distance = abs(current_point[0][0] - nearest_neighbor[0])
                y_distance = abs(current_point[0][1] - nearest_neighbor[1])

                # Check if this point is part of an isolated pair
                pair_key = tuple(sorted([i, nearest_idx]))
                is_isolated_pair = pair_key in isolated_pairs

                # Annotation criteria
                not_in_upper_left_corner = not (
                    row[disease_metric] < data[disease_metric].max() * 0.5
                    and row[aid_metric] > data[aid_metric].max() * 0.5
                )

                should_annotate = (
                    not_in_upper_left_corner
                    and ((x_distance > 0.1 or y_distance > 0.1))
                    or not_in_upper_left_corner
                    and (is_isolated_pair and i == min(pair_key))
                )

                if should_annotate:
                    # Skip countries with long names
                    if len(row["country_name"]) > 16:
                        continue
                    elif row["country_name"] == "Dominican Republic":
                        continue
                    elif row["country_name"] == "Papua New Guinea":
                        continue

                    # Determine text alignment based on x position - FIXED ORDER
                    if row[disease_metric] > 0.9 * x_max:
                        ha = "right"
                    elif row[disease_metric] < 0.1 * x_max:
                        ha = "left"
                    else:
                        ha = "center"

                    axes[idx].annotate(
                        row["country_name"],
                        (row[disease_metric], row[aid_metric]),  # FIXED ORDER
                        xytext=(0, 9),
                        textcoords="offset points",
                        ha=ha,
                        fontsize=12,
                        alpha=1,
                    )

        elif (
            len(normalized_points) == 1
        ):  # If there only is one point, always annotate it
            row = data.iloc[0]
            axes[idx].annotate(
                row["country_name"],
                (row[disease_metric], row[aid_metric]),  # FIXED ORDER
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                alpha=1,
            )

        # Set axis limits and styling - FIXED ORDER
        y_min = 0
        y_max = data[aid_metric].max()  # aid_metric is y-axis
        y_padding = (y_max - y_min) * 0.05
        axes[idx].set_ylim(y_min - y_padding, y_max + 2 * y_padding)

        # Styling - axis labels are now correct
        axes[idx].set_xlabel("DALYs per 100,000 People", fontsize=18)
        if idx == 0:
            axes[idx].set_ylabel(
                "Aid per capita (USD)", fontsize=18, labelpad=15
            )
        axes[idx].set_title(
            f"{income_group}", fontsize=20, pad=15, weight="bold"
        )
        axes[idx].tick_params(axis="both", which="major", labelsize=16)
        # axes[idx].grid(True, alpha=0.2)

    plt.tight_layout()

    # Save as PDF
    filename = disease_cause.replace("/", "–")
    filename = filename + "_" + disease_metric + "_" + aid_metric
    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "Correlation",
            folder_path,
            f"split_{filename}_by_income.pdf",
        ),
        bbox_inches="tight",
    )
    plt.close()


def create_heatmap(
    income_groups_dict,
    country_counts_dict,
    include_madcts_other_lics=True,
    include_all=True,
    include_incidence_rate=False,
    no_covid=False,
    covid_years=False,
):
    data = {
        "Aid Target": [
            "HIV/AIDS & STIs",
            "Respiratory infections & TB",
            "Enteric infections",
            "NTDs & malaria",
            "Maternal & neonatal disorders",
            "Nutritional deficiencies",
            "Neoplasms",
            "Cardiovascular diseases",
            "Chronic respiratory diseases",
            "Digestive diseases",
            "Neurological disorders",
            "Mental disorders",
            "Substance use disorders",
            "Diabetes & kidney diseases",
            "Skin & subcutaneous diseases",
            "Sense organ diseases",
            "Musculoskeletal disorders",
        ],
        "GBD Cause Name": [
            "HIV/AIDS and STIs",
            "Resp. infections and TB",
            "Enteric infections",
            "NTDs and malaria",
            "Maternal & neonatal dis.",
            "Nutritional deficiencies",
            "Neoplasms",
            "Cardiovascular diseases",
            "Chronic resp. diseases",
            "Digestive diseases",
            "Neurological disorders",
            "Mental disorders",
            "Substance use disorders",
            "Diabetes & kidney diseases",
            "Skin and subcut. diseases",
            "Sense organ diseases",
            "Musculoskeletal disorders",
        ],
        "LDCs": [entry["correlation"] for entry in income_groups_dict["LDCs"]],
        "LMICs": [
            entry["correlation"] for entry in income_groups_dict["LMICs"]
        ],
        "UMICs": [
            entry["correlation"] for entry in income_groups_dict["UMICs"]
        ],
    }

    if include_madcts_other_lics:
        data["MADCTs"] = [
            entry["correlation"] for entry in income_groups_dict["MADCTs"]
        ]
        data["Other LICs"] = [
            entry["correlation"] for entry in income_groups_dict["Other LICs"]
        ]

    if include_all:
        data["All"] = [
            entry["correlation"] for entry in income_groups_dict["All"]
        ]

    p_value_df = {
        "LDCs": [entry["p_value"] for entry in income_groups_dict["LDCs"]],
        "LMICs": [entry["p_value"] for entry in income_groups_dict["LMICs"]],
        "UMICs": [entry["p_value"] for entry in income_groups_dict["UMICs"]],
    }

    if include_madcts_other_lics:
        p_value_df["MADCTs"] = [
            entry["p_value"] for entry in income_groups_dict["MADCTs"]
        ]
        p_value_df["Other LICs"] = [
            entry["p_value"] for entry in income_groups_dict["Other LICs"]
        ]

    if include_all:
        p_value_df["All"] = [
            entry["p_value"] for entry in income_groups_dict["All"]
        ]

    # Convert the data into a pandas DataFrame
    corr_df = pd.DataFrame(data)
    p_value_df = pd.DataFrame(p_value_df)

    # Round only numeric values, not other values
    numeric_columns = ["LDCs", "LMICs", "UMICs"]
    if include_madcts_other_lics:
        numeric_columns.extend(["MADCTs", "Other LICs"])

    if include_all:
        numeric_columns.append("All")

    for col in numeric_columns:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce").round(3)

    # Set the index to 'Aid Target' and drop 'GBD Cause Name' column for heatmap creation
    corr_df.set_index("Aid Target", inplace=True)
    gbd_causes = corr_df[
        "GBD Cause Name"
    ].values  # Save GBD Cause Name values before dropping the column
    corr_df.drop("GBD Cause Name", axis=1, inplace=True)

    # Create the heatmap without annotations
    if include_madcts_other_lics:
        plt.figure(figsize=(6, 10), dpi=300)
    elif include_all:
        plt.figure(figsize=(5, 10), dpi=300)
    else:
        plt.figure(figsize=(4, 10), dpi=300)

    # Fill NaN values with 0 for coloring purposes, but keep original data for annotations
    corr_df_filled = corr_df.fillna(0)

    # Define custom colormap
    custom_colors = [
        "#D46780FF",
        "#DF91A3FF",
        "#F0C6C3FF",
        "#FDFBE4FF",
        "#D0D3A2FF",
        "#A3AD62FF",
        "#798234FF",
    ]
    custom_cmap = colors.ListedColormap(custom_colors)

    # Define bins for discretization
    bins = [-1.0, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 1.0]
    labels = [
        "Strong misalignment",
        "Moderate misalignment",
        "Weak misalignment",
        "Random allocation",
        "Weak alignment",
        "Moderate alignment",
        "Strong alignment",
    ]

    # Create a new DataFrame with discretized values
    corr_df_discrete = corr_df_filled.copy()
    for col in numeric_columns:
        corr_df_discrete[col] = pd.cut(
            corr_df_filled[col], bins=bins, labels=labels, include_lowest=True
        )

    # Create the heatmap with discrete values
    heatmap = sns.heatmap(
        corr_df_filled,
        annot=False,
        cmap=custom_cmap,
        cbar=False,
        linewidths=0,
        linecolor="white",
        center=0,
        vmin=-1,
        vmax=1,
    )

    # Function to calculate luminance and determine text color
    def get_text_color(background_color):
        r, g, b, _ = background_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.6 else "black"

    # Add annotations with the desired font weight and adaptive text color
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            value = corr_df.iloc[i, j]
            p_value = p_value_df.iloc[i, j]
            fontweight = "normal"

            if p_value is not None and p_value < 0.001:
                asterix = "***"
            elif p_value is not None and p_value < 0.01:
                asterix = "**"
            elif p_value is not None and p_value < 0.05:
                asterix = "*"
            else:
                asterix = ""

            sign = "–" if value < 0 else ""
            if value < 0:
                value = value * (-1)
            background_color = heatmap.collections[0].get_facecolor()[
                i * corr_df.shape[1] + j
            ]
            text_color = get_text_color(background_color)
            if not pd.isna(value):
                heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight=fontweight,
                    fontsize=10,
                )
            heatmap.text(
                j + 0.665,
                i + 0.5,
                asterix,
                ha="left",
                va="center",
                color="black",
                fontsize=10,
            )
            heatmap.text(
                j + 0.29,
                i + 0.5,
                sign,
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

            # Add N/A (Not enough Data} labels to the places where it's NaN on the heatmap
            if pd.isna(corr_df.iloc[i, j]):
                text = "N/A"
                heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

    # Add labels and title
    plt.xlabel("")
    plt.ylabel("")

    # Add GBD Cause Name labels to the right of the heatmap
    for i, indicator in enumerate(gbd_causes):
        plt.text(
            heatmap.get_xlim()[1] + 0.1,
            i + 0.5,
            indicator,
            va="center",
            color="white",
        )

    # Rotate y-axis labels to be horizontal
    plt.yticks(rotation=0)

    # Remove x and y axis labels and ticks
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(), rotation=0, ha="center", fontsize=10
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)
    heatmap.set(xlabel=None, ylabel=None)

    # Add axis labels
    plt.xlabel("Income groups", fontsize=12, fontweight="bold", labelpad=20)

    # Add legend for correlation ranges
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
        for color in custom_colors[::-1]
    ]  # Reverse the colors order
    plt.legend(
        legend_elements,
        labels[::-1],  # Reverse the labels order
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.05,
    )

    # Adjust layout to ensure equal spacing
    plt.subplots_adjust(left=0.15, right=0.85)

    # Save to PDF
    if include_madcts_other_lics:
        filename = (
            "ranked_spearman_correlation_heatmap_with_madcts_other_lics.pdf"
        )
    elif include_incidence_rate:
        filename = "ranked_spearman_correlation_heatmap_incidence_rate.pdf"
    elif include_all:
        filename = "ranked_spearman_correlation_heatmap_with_all.pdf"
    else:
        filename = "ranked_spearman_correlation_heatmap.pdf"

    if no_covid:
        filename = "ranked_spearman_correlation_heatmap_with_all_no_covid.pdf"
    if covid_years:
        filename = (
            "ranked_spearman_correlation_heatmap_with_all_covid_years.pdf"
        )
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches="tight")

    # Save to SVG
    if include_madcts_other_lics:
        filename = (
            "ranked_spearman_correlation_heatmap_with_madcts_other_lics.svg"
        )
    elif include_incidence_rate:
        filename = "ranked_spearman_correlation_heatmap_incidence_rate.svg"
    elif include_all:
        filename = "ranked_spearman_correlation_heatmap_with_all.svg"
    else:
        filename = "ranked_spearman_correlation_heatmap.svg"

    if no_covid:
        filename = "ranked_spearman_correlation_heatmap_with_all_no_covid.svg"
    if covid_years:
        filename = (
            "ranked_spearman_correlation_heatmap_with_all_covid_years.svg"
        )
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches="tight")
    plt.close()


def prepare_data_for_heatmap(
    merged_df: pd.DataFrame,
    aid_metric: str,
    disease_metric: str,
    no_covid: bool = False,
) -> pd.DataFrame:
    # Analysis and visualization
    income_groups_dict = {
        group: [] for group in merged_df["IncomegroupName"].unique()
    }
    income_groups_dict["All"] = []  # Add 'All' key for aggregate correlations

    # Initialize dictionary to track country counts for each income group
    country_counts_dict = {
        group: [] for group in merged_df["IncomegroupName"].unique()
    }
    country_counts_dict["All"] = []

    # Remove rows with missing values for correlation calculation
    merged_df_clean = merged_df.dropna(subset=[aid_metric, disease_metric])

    # Initialize empty list to collect rows for spearman_df
    spearman_rows = []

    # Calculate Spearman rank correlation for each income group and category
    for category in categories:
        category_df = merged_df_clean[
            merged_df_clean["llm_classification"] == category
        ]

        if no_covid:
            # Replace values for Respiratory infections and tuberculosis after 2019 with 2019 values
            if category == "Respiratory infections and tuberculosis":
                # Get 2019 data for this category
                data_2019 = category_df[category_df["Year"] == 2019].copy()

                if not data_2019.empty:
                    # For each country that has 2019 data, replace post-2019 values with 2019 values
                    for country_code in data_2019["country_code"].unique():
                        country_2019_data = data_2019[
                            data_2019["country_code"] == country_code
                        ]

                        # Get the 2019 values for this country
                        if len(country_2019_data) > 0:
                            values_2019 = country_2019_data.iloc[0]

                            # Replace post-2019 values with 2019 values for this country
                            mask = (
                                category_df["country_code"] == country_code
                            ) & (category_df["Year"] > 2019)
                            for col in [
                                "USD_Disbursement_per_capita",
                                "dalys_rate",
                                "incidence_rate",
                            ]:
                                if (
                                    col in category_df.columns
                                    and col in values_2019
                                ):
                                    category_df.loc[mask, col] = values_2019[
                                        col
                                    ]

        # Comment if you want to include Small Island Developing States
        category_df = category_df[
            ~category_df["country_code"].isin(sids_codes)
        ]

        # Calculate averages per country first
        category_df = (
            category_df.groupby(
                ["country_code", "country_name", "IncomegroupName"]
            )[[aid_metric, disease_metric]]
            .mean()
            .reset_index()
        )

        for income_group in [
            group for group in income_groups_dict.keys() if group != "All"
        ]:
            income_group_df = category_df[
                category_df["IncomegroupName"] == income_group
            ]

            if len(income_group_df) > 10:
                correlation, p_value = stats.spearmanr(
                    income_group_df[aid_metric],
                    income_group_df[disease_metric],
                )
                income_groups_dict[income_group].append(
                    {
                        "correlation": round(correlation, 2),
                        "p_value": round(p_value, 2),
                    }
                )
                country_counts_dict[income_group].append(len(income_group_df))
                spearman_rows.append(
                    {
                        "category": category,
                        "income_group": income_group,
                        "correlation": round(correlation, 2),
                        "p_value": round(p_value, 2),
                    }
                )
            else:
                income_groups_dict[income_group].append(
                    {"correlation": None, "p_value": None}
                )
                country_counts_dict[income_group].append(len(income_group_df))
                spearman_rows.append(
                    {
                        "category": category,
                        "income_group": income_group,
                        "correlation": None,
                        "p_value": None,
                    }
                )

        correlation, p_value = stats.spearmanr(
            category_df[aid_metric], category_df[disease_metric]
        )
        income_groups_dict["All"].append(
            {
                "correlation": round(correlation, 2),
                "p_value": round(p_value, 2),
            }
        )
        country_counts_dict["All"].append(len(category_df))
        spearman_rows.append(
            {
                "category": category,
                "income_group": "ALL",
                "correlation": round(correlation, 2),
                "p_value": round(p_value, 2),
            }
        )

    return income_groups_dict, country_counts_dict


# Main Execution
def main():
    # Load and preprocess data
    funding_df_path = os.path.join(PROCESSED_DATA_DIR, "funding_df.csv")
    if os.path.exists(funding_df_path):
        funding_df = pd.read_csv(funding_df_path)
    else:
        funding_df = load_and_preprocess_funding_data()

    merged_df_path = os.path.join(PROCESSED_DATA_DIR, "merged_df.csv")
    if os.path.exists(merged_df_path):
        merged_df = pd.read_csv(merged_df_path)
    else:
        funding_df = add_country_codes(funding_df)
        funding_df = add_population_data(funding_df)
        funding_df = add_health_spend_data(funding_df)
        merged_df = merge_with_gbd_data(funding_df)
        merged_df.to_csv(merged_df_path, index=False)

    merged_df = add_gni_data(merged_df)

    # Analysis and visualization
    income_groups_dict = {
        group: [] for group in merged_df["IncomegroupName"].unique()
    }
    income_groups_dict["All"] = []  # Add 'All' key for aggregate correlations

    # Initialize dictionary to track country counts for each income group
    country_counts_dict = {
        group: [] for group in merged_df["IncomegroupName"].unique()
    }
    country_counts_dict["All"] = []

    funding_gap_data = {}
    merged_df = merged_df[merged_df["Year"] != 2022.0]
    merged_df_no_covid = merged_df[
        (merged_df["Year"] != 2020.0) & (merged_df["Year"] != 2021.0)
    ]
    merged_df_covid_years = merged_df[
        (merged_df["Year"] == 2020.0) | (merged_df["Year"] == 2021.0)
    ]

    # Create necessary directories
    if not os.path.exists(os.path.join(PLOTS_DIR, "Correlation", "all")):
        os.makedirs(os.path.join(PLOTS_DIR, "Correlation", "all"))
    if not os.path.exists(os.path.join(PLOTS_DIR, "Correlation", "no_covid")):
        os.makedirs(os.path.join(PLOTS_DIR, "Correlation", "no_covid"))
    if not os.path.exists(
        os.path.join(PLOTS_DIR, "Correlation", "covid_years")
    ):
        os.makedirs(os.path.join(PLOTS_DIR, "Correlation", "covid_years"))
    if not os.path.exists(os.path.join(PLOTS_DIR, "Map_plots", "all")):
        os.makedirs(os.path.join(PLOTS_DIR, "Map_plots", "all"))
    if not os.path.exists(os.path.join(PLOTS_DIR, "Map_plots", "no_covid")):
        os.makedirs(os.path.join(PLOTS_DIR, "Map_plots", "no_covid"))
    if not os.path.exists(os.path.join(PLOTS_DIR, "Map_plots", "covid_years")):
        os.makedirs(os.path.join(PLOTS_DIR, "Map_plots", "covid_years"))

    def create_funding_gap_plots(df: pd.DataFrame, folder_path: str = "all"):
        for category in categories:
            category_df = df[df["llm_classification"] == category]

            # Comment if you want to include Small Island Developing States
            category_df = category_df[
                ~category_df["country_code"].isin(sids_codes)
            ]

            category_df_grouped = (
                category_df.groupby(["country_code", "country_name"])
                .agg(
                    {
                        "IncomegroupName": "first",
                        "USD_Disbursement_per_capita": "mean",
                        "incidence_rate": "mean",
                        "dalys_rate": "mean",
                    }
                )
                .reset_index()
            )

            filename = category.replace("/", "–")

            # Calculate within-group funding gaps
            for income_group in ["LDCs", "LMICs", "UMICs"]:
                group_countries = category_df_grouped[
                    category_df_grouped["IncomegroupName"] == income_group
                ].copy()

                # Calculate rankings
                group_countries["Necessity_Rank_percentile"] = (
                    group_countries["dalys_rate"].rank(
                        pct=True, ascending=True
                    )
                    * 100
                )
                group_countries["Funding_Rank_percentile"] = (
                    group_countries["USD_Disbursement_per_capita"].rank(
                        pct=True, ascending=True
                    )
                    * 100
                )
                group_countries["Funding_Gap_percentile"] = round(
                    group_countries["Necessity_Rank_percentile"]
                    - group_countries["Funding_Rank_percentile"]
                )

                # Update the original DataFrame
                category_df_grouped.loc[
                    category_df_grouped["IncomegroupName"] == income_group,
                    "Funding_Gap_percentile",
                ] = group_countries["Funding_Gap_percentile"].values
                category_df_grouped.loc[
                    category_df_grouped["IncomegroupName"] == income_group,
                    "Funding_Rank_percentile",
                ] = group_countries["Funding_Rank_percentile"].values
                category_df_grouped.loc[
                    category_df_grouped["IncomegroupName"] == income_group,
                    "Necessity_Rank_percentile",
                ] = group_countries["Necessity_Rank_percentile"].values

            # Create visualizations
            no_indicator = category_df_grouped[
                category_df_grouped["dalys_rate"].isna()
            ]["country_code"].tolist()
            plot_geoframe_subplots_rwb(
                category_df_grouped,
                filename,
                set(no_indicator),
                "Funding_Gap_percentile",
                folder_path,
            )
            create_split_correlation_plots(
                category_df_grouped,
                filename,
                "dalys_rate",
                "USD_Disbursement_per_capita",
                folder_path,
            )

            funding_gap_data[category] = category_df_grouped[
                ["Necessity_Rank_percentile", "Funding_Gap_percentile"]
            ]

    create_funding_gap_plots(merged_df)
    create_funding_gap_plots(merged_df_no_covid, "no_covid")
    create_funding_gap_plots(merged_df_covid_years, "covid_years")

    # Remove rows with missing values for correlation calculation
    merged_df_clean = merged_df.dropna(
        subset=["USD_Disbursement_per_capita", "dalys_rate"]
    )

    # income_groups_dict, country_counts_dict = prepare_data_for_heatmap(merged_df_clean, 'USD_Disbursement_per_capita', 'incidence_rate', no_covid=False)
    income_groups_dict, country_counts_dict = prepare_data_for_heatmap(
        merged_df_clean,
        "USD_Disbursement_per_capita",
        "dalys_rate",
        no_covid=False,
    )
    income_groups_dict_no_covid, country_counts_dict_no_covid = (
        prepare_data_for_heatmap(
            merged_df_clean,
            "USD_Disbursement_per_capita",
            "dalys_rate",
            no_covid=True,
        )
    )
    income_groups_dict_incidence_rate, country_counts_dict_incidence_rate = (
        prepare_data_for_heatmap(
            merged_df_clean,
            "USD_Disbursement_per_capita",
            "incidence_rate",
            no_covid=False,
        )
    )

    merged_df_clean_covid_years = merged_df_clean[
        merged_df_clean["Year"] > 2019
    ]
    income_groups_dict_covid_years, country_counts_dict_covid_years = (
        prepare_data_for_heatmap(
            merged_df_clean_covid_years,
            "USD_Disbursement_per_capita",
            "dalys_rate",
            no_covid=False,
        )
    )

    # Create final heatmaps
    create_heatmap(
        income_groups_dict,
        country_counts_dict,
        include_madcts_other_lics=False,
        include_all=True,
    )
    create_heatmap(
        income_groups_dict_no_covid,
        country_counts_dict_no_covid,
        include_madcts_other_lics=False,
        include_all=True,
        no_covid=True,
    )
    create_heatmap(
        income_groups_dict_covid_years,
        country_counts_dict_covid_years,
        include_madcts_other_lics=False,
        include_all=True,
        no_covid=False,
        covid_years=True,
    )
    create_heatmap(
        income_groups_dict_incidence_rate,
        country_counts_dict_incidence_rate,
        include_madcts_other_lics=False,
        include_all=True,
        include_incidence_rate=True,
    )


if __name__ == "__main__":
    main()
