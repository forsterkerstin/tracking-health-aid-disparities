# This file generates various plots to visualize aid funding data across different disease categories and years.
# It uses data from a CSV file and creates several types of plots including heatmaps, stacked bar charts, line graphs, and bar charts.

import json
import os
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Imports
import polars as pl
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Add this at the beginning of the file
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Configuration
config = {
    "input_file": "data/processed/aid_funding_data/final_df.csv",
    # The split of the disease categories is based on the split used by the GBD study.
    "disease_categories": {
        "Communicable, maternal, neonatal, and nutritional diseases": [
            "HIV/AIDS and sexually transmitted infections",
            "Respiratory infections and tuberculosis",
            "Enteric infections",
            "Neglected tropical diseases and malaria",
            "Maternal and neonatal disorders",
            "Nutritional deficiencies",
        ],
        "Non-communicable diseases": [
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
        ],
    },
}

disease_abbreviations = {
    "HIV/AIDS and sexually transmitted infections": "HIV/AIDS & STIs",
    "Respiratory infections and tuberculosis": "Respiratory inf. & TB",
    "Nutritional deficiencies": "Nutritional deficiencies",
    "Maternal and neonatal disorders": "Maternal & neonatal dis.",
    "Neglected tropical diseases and malaria": "NTDs & malaria",
    "Enteric infections": "Enteric inf.",
    "Mental disorders": "Mental disorders",
    "Neoplasms": "Neoplasms",
    "Sense organ diseases": "Sense organ dis.",
    "Cardiovascular diseases": "Cardiovascular dis.",
    "Diabetes and kidney diseases": "Diabetes & kidney dis.",
    "Substance use disorders": "Substance use dis.",
    "Musculoskeletal disorders": "Musculoskeletal dis.",
    "Neurological disorders": "Neurological dis.",
    "Chronic respiratory diseases": "Chronic respiratory dis.",
    "Skin and subcutaneous diseases": "Skin & subcut. dis.",
    "Digestive diseases": "Digestive dis.",
}

# Functions


def load_data(file_path: str, dtypes: Dict[str, Any]) -> pl.DataFrame:
    """
    Load data from a CSV file into a Polars DataFrame and filter out 'Other' classifications.

    Args:
        file_path (str): Path to the CSV file.
        dtypes (Dict[str, Any]): Dictionary specifying column data types.

    Returns:
        pl.DataFrame: Loaded and filtered DataFrame.
    """
    # Convert relative path to absolute path
    absolute_file_path = os.path.join(PROJECT_ROOT, file_path)
    df = pl.read_csv(absolute_file_path, dtypes=dtypes)
    return df.filter(pl.col("llm_classification") != "Other")


def split_classifications(x: str) -> List[str]:
    """
    Split multi-label classifications into a list of individual classifications.

    Args:
        x (str): String containing comma-separated classifications.

    Returns:
        List[str]: List of individual classifications.
    """
    return [
        classification.strip()
        for classification in str(x).split(",")
        if classification.strip()
    ]


def calculate_disbursement_stats(
    df: pl.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate disbursement statistics for each classification.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing disbursement statistics.
    """
    stats = {"count": Counter(), "total": {}, "avg": {}}
    for row in df.iter_rows(named=True):
        for cls in split_classifications(row["llm_classification"]):
            stats["count"][cls] += 1
            stats["total"][cls] = (
                stats["total"].get(cls, 0) + row["adjusted_funding_llm"]
            )

    stats["avg"] = {
        cls: stats["total"][cls] / stats["count"][cls]
        for cls in stats["total"]
    }
    return stats


def get_top_recipients(df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Get the top recipients for each classification based on total funding.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing top recipients for each classification.
    """
    top_recipients = {}
    for row in df.iter_rows(named=True):
        country = row["RecipientName"]
        for cls in split_classifications(row["llm_classification"]):
            if cls not in top_recipients:
                top_recipients[cls] = Counter()
            top_recipients[cls][country] += row["adjusted_funding_llm"]

    return {
        cls: dict(recipients.most_common(3))
        for cls, recipients in top_recipients.items()
    }


def calculate_time_series(df: pl.DataFrame) -> Dict[str, Counter]:
    """
    Calculate time series data for disbursements, aggregating by year and classification.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        Dict[str, Counter]: Dictionary containing time series data for each classification.
    """
    time_series = {}
    for row in df.iter_rows(named=True):
        year = row["Year"]
        for cls in split_classifications(row["llm_classification"]):
            if cls not in time_series:
                time_series[cls] = Counter()
            time_series[cls][year] += row["adjusted_funding_llm"]
    return time_series


def plot_heatmap(time_series: Dict[str, Counter], output_path: str):
    """
    Plot and save a heatmap of disbursements over time.

    Args:
        time_series (Dict[str, Counter]): Time series data.
        output_path (str): Path to save the plot.
    """
    df = pd.DataFrame(time_series).fillna(0).sort_index()
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.T, cmap="YlOrRd")
    plt.title(
        "Total Disbursement per Year by Classification from 2000 to 2022"
    )
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


def plot_stacked_bar(time_series: Dict[str, Counter], output_path: str):
    """
    Plot and save a stacked bar chart of disbursements over time.

    Args:
        time_series (Dict[str, Counter]): Time series data.
        output_path (str): Path to save the plot.
    """
    df = pd.DataFrame(time_series).fillna(0).sort_index()
    df.index = df.index.astype(float).astype(
        int
    )  # Ensure the index is numeric
    df = df[df.sum().sort_values(ascending=False).index]

    # Ensure specific categories are on top in the desired order
    categories_to_top = [
        "Enteric infections",
        "Maternal and neonatal disorders",
        "Neglected tropical diseases and malaria",
        "Nutritional deficiencies",
        "HIV/AIDS and sexually transmitted infections",
        "Respiratory infections and tuberculosis",
    ]
    other_columns = [col for col in df.columns if col not in categories_to_top]
    if "Year" in other_columns:
        other_columns.remove("Year")
    df["Non-communicable diseases"] = df[other_columns].sum(axis=1)
    df = df[["Non-communicable diseases"] + categories_to_top]

    df_billions = df / 1000

    # Convert color codes to proper hex format
    colors_list = [
        "#e377c2",
        "#ff7f0e",
        "#9467bd",
        "#2ca02c",
        "#d62728",
        "#8c564b",
        "#1f77b4",
    ]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)

    bottom = np.zeros(len(df_billions))
    for idx, category in enumerate(df_billions.columns):
        color = (
            colors_list[idx] if idx < len(colors_list) else None
        )  # fallback if more categories
        ax.bar(
            df_billions.index,
            df_billions[category],
            bottom=bottom,
            label=category,
            alpha=1,
            zorder=3,
            color=color,
        )
        bottom += df_billions[category]

    # ax.set_title('Development Aid for Health by Disease, 2000–2022', fontsize=32, pad=20)
    # ax.set_xlabel('Year', fontsize=28)
    ax.set_ylabel("Aid in billions (USD)", fontsize=36)
    ax.tick_params(axis="both", which="major", labelsize=32)

    handles, labels = ax.get_legend_handles_labels()
    order = list(range(len(handles) - 1, -1, -1))
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
        fontsize=26,
    )

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


def plot_stacked_bar_non_communicable(
    time_series: Dict[str, Counter], output_path: str
):
    """
    Plot and save a stacked bar chart of disbursements over time.

    Args:
        time_series (Dict[str, Counter]): Time series data.
        output_path (str): Path to save the plot.
    """
    df = pd.DataFrame(time_series).fillna(0).sort_index()
    df.index = df.index.astype(float).astype(
        int
    )  # Ensure the index is numeric
    df = df[df.sum().sort_values(ascending=False).index]

    # Ensure specific categories are on top in the desired order
    categories_to_top = [
        "Enteric infections",
        "Maternal and neonatal disorders",
        "Neglected tropical diseases and malaria",
        "Nutritional deficiencies",
        "HIV/AIDS and sexually transmitted infections",
        "Respiratory infections and tuberculosis",
    ]
    other_columns = [col for col in df.columns if col not in categories_to_top]
    if "Year" in other_columns:
        other_columns.remove("Year")

    other_columns = [
        "Mental disorders",
        "Neoplasms",
        "Substance use disorders",
        "Sense organ diseases",
        "Chronic respiratory diseases",
        "Cardiovascular diseases",
        "Diabetes and kidney diseases",
        "Musculoskeletal disorders",
        "Neurological disorders",
        "Digestive diseases",
        "Skin and subcutaneous diseases",
    ]

    other_columns = other_columns[::-1]

    df = df[other_columns]
    df_billions = df / 1000

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)

    # Define a color palette
    color_list = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
    ]
    colors = {
        category: color_list[i % len(color_list)]
        for i, category in enumerate(df_billions.columns)
    }

    bottom = np.zeros(len(df_billions))
    for category in df_billions.columns:
        ax.bar(
            df_billions.index,
            df_billions[category],
            bottom=bottom,
            label=category,
            alpha=1,
            zorder=3,
            color=colors[category],
        )
        bottom += df_billions[category]

    # ax.set_title('Development Aid for Health by Disease, 2000–2022', fontsize=32, pad=20)
    ax.set_ylabel("Aid in billions (USD)", fontsize=36)
    ax.tick_params(axis="both", which="major", labelsize=32)

    handles, labels = ax.get_legend_handles_labels()
    order = list(range(len(handles) - 1, -1, -1))
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
        fontsize=26,
    )

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


def plot_disease_share_of_aid(
    time_series: Dict[str, Counter], output_path: str
):
    """
    Plot and save a stacked area chart of the share of total aid for each disease category.

    Args:
        time_series (Dict[str, Counter]): Time series data.
        output_path (str): Path to save the plot.
    """
    df = pd.DataFrame(time_series).fillna(0).sort_index()
    df.index = df.index.astype(float).astype(
        int
    )  # Ensure the index is numeric
    df = df[df.sum().sort_values(ascending=False).index]

    # Ensure specific categories are on top in the desired order
    categories_to_top = [
        "Enteric infections",
        "Maternal and neonatal disorders",
        "Neglected tropical diseases and malaria",
        "Nutritional deficiencies",
        "HIV/AIDS and sexually transmitted infections",
        "Respiratory infections and tuberculosis",
    ]

    other_columns = [col for col in df.columns if col not in categories_to_top]
    if "Year" in other_columns:
        other_columns.remove("Year")
    df["Non-communicable Diseases"] = df[other_columns].sum(axis=1)
    df = df[["Non-communicable Diseases"] + categories_to_top]

    # Calculate percentage share
    df_percentage = df.div(df.sum(axis=1), axis=0) * 100

    fig, ax2 = plt.subplots(figsize=(15, 10))

    # Line Plot
    ax2.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    for i, column in enumerate(df_percentage.columns):
        ax2.plot(
            df_percentage.index,
            df_percentage[column],
            label=column,
            linewidth=3,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
        )

    ax2.set_title(
        "Percentage Breakdown of Yearly Global Health Aid by Disease Area, 2000–2022",
        fontsize=20,
    )
    ax2.set_ylabel("Share of Aid (%)", fontsize=20)
    ax2.tick_params(axis="both", which="major", labelsize=18)

    handles, labels = ax2.get_legend_handles_labels()
    order = list(range(len(handles) - 1, -1, -1))  # Reverse order
    ax2.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
        fontsize=12,
    )

    ax2.set_ylim(
        0, df_percentage.max().max() * 1.1
    )  # Set y-axis limit to slightly above the maximum value

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_line_graph_separated(
    time_series: Dict[str, Counter], output_path: str
):
    """
    Plot and save a line graph of disbursements over time for each classification,
    with a broken axis to accommodate large values.

    Args:
        time_series (Dict[str, Counter]): Time series data.
        output_path (str): Path to save the plot.
    """

    # Ensure specific categories are on top in the desired order
    categories_to_top = [
        "Enteric infections",
        "Maternal and neonatal disorders",
        "Neglected tropical diseases and malaria",
        "Nutritional deficiencies",
        "HIV/AIDS and sexually transmitted infections",
        "Respiratory infections and tuberculosis",
    ]

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(time_series).fillna(0)
    df.index = df.index.astype(float).astype(
        int
    )  # Ensure the index is numeric
    df = df.sort_index()

    # Reorder columns
    other_columns = [col for col in df.columns if col not in categories_to_top]
    if "Year" in other_columns:
        other_columns.remove("Year")
    df = df[categories_to_top + other_columns]

    # Create 'Non-communicable Diseases' column
    df["Non-communicable Diseases"] = df[other_columns].sum(axis=1)
    df = df[["Non-communicable Diseases"] + categories_to_top]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [1, 3.4]},
        constrained_layout=True,
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    for i, cls in enumerate(df.columns):
        ax1.plot(
            df.index,
            df[cls],
            label=cls,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i],
            linewidth=2,
        )
        ax2.plot(
            df.index,
            df[cls],
            label=cls,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i],
            linewidth=2,
        )

        last_year = df.index[-1]
        last_value = df[cls].iloc[-1]
        ax2.annotate(
            f"{last_value:.0f}",
            (last_year, last_value),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=12,
            color=colors[i],
        )

    ax1.set_ylim(23500, 26000)
    ax2.set_ylim(0, 8500)

    ax1.yaxis.set_major_locator(MultipleLocator(1000))
    ax2.yaxis.set_major_locator(MultipleLocator(1000))

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    ax1.tick_params(bottom=False)
    ax2.tick_params(top=False)

    ax1.axhline(y=23500, color="r", linestyle="--")
    ax2.axhline(y=8500, color="r", linestyle="--")

    # ax1.set_title('Development Aid for Health per Year by Disease Focus Area, 2000–2022', fontsize=16)
    ax2.set_ylabel("Development Aid in millions (USD)", fontsize=14)

    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax2.tick_params(axis="both", which="major", labelsize=12)

    handles, labels = ax1.get_legend_handles_labels()
    order = [6, 5, 4, 3, 2, 1, 0]  # Adjust this order as needed
    ax1.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
        fontsize=14,
    )

    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_line_graph(
    time_series: Dict[str, Counter],
    output_path: str,
    include_respiratory: bool = True,
):
    """
    Plot and save a line graph of disbursements over time for each classification.

    Args:
        time_series (Dict[str, Counter]): Time series data.
        output_path (str): Path to save the plot.
        include_respiratory (bool): Whether to include respiratory infections and tuberculosis.
    """

    # Ensure specific categories are on top in the desired order
    categories_to_top = [
        "Enteric infections",
        "Maternal and neonatal disorders",
        "Neglected tropical diseases and malaria",
        "Nutritional deficiencies",
        "HIV/AIDS and sexually transmitted infections",
    ]

    if include_respiratory:
        categories_to_top.append("Respiratory infections and tuberculosis")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(time_series).fillna(0)
    df.index = df.index.astype(float).astype(
        int
    )  # Ensure the index is numeric
    df = df.sort_index()

    # Reorder columns
    other_columns = [col for col in df.columns if col not in categories_to_top]
    if "Year" in other_columns:
        other_columns.remove("Year")
    if not include_respiratory:
        other_columns = [
            col
            for col in other_columns
            if col != "Respiratory infections and tuberculosis"
        ]
    df = df[categories_to_top + other_columns]

    # Create 'Non-communicable Diseases' column
    df["Non-communicable Diseases"] = df[other_columns].sum(axis=1)
    df = df[["Non-communicable Diseases"] + categories_to_top]

    # Convert millions to billions
    df = df / 1000

    fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    for i, cls in enumerate(df.columns):
        ax.plot(
            df.index,
            df[cls],
            label=cls,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i],
            linewidth=3,
        )

        last_year = df.index[-1]
        last_value = df[cls].iloc[-1]

        ax.annotate(
            f"{last_value:.1f}",
            (last_year, last_value),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=18,
            color=colors[i],
        )

    if include_respiratory:
        ax.set_ylim(0, 25.5)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        title_suffix = " (including Covid-19)"
    else:
        ax.set_ylim(0, 8)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        title_suffix = " (excluding Covid-19)"

    # ax.set_title(f'Development Aid for Health per Year by Disease, 2000–2022', fontsize=32, pad=20)
    # ax.set_xlabel('Year', fontsize=28)
    ax.set_ylabel("Development Aid in billions (USD)", fontsize=28)

    ax.tick_params(axis="both", which="major", labelsize=24)

    handles, labels = ax.get_legend_handles_labels()
    order = list(range(len(df.columns) - 1, -1, -1))  # Reverse order
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
        fontsize=24,
    )

    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar_chart(
    disbursement_data: Dict[str, float],
    output_path: str,
    last_5_years: bool = False,
):
    """
    Plot and save a bar chart of total disbursements by disease focus area.

    Args:
        disbursement_data (Dict[str, float]): Disbursement data.
        output_path (str): Path to save the plot.
        last_5_years (bool): Whether to plot only the last 5 years.
    """
    sorted_disbursement = sorted(
        disbursement_data.items(), key=lambda item: item[1], reverse=True
    )
    categories, values = zip(*sorted_disbursement)

    values_billions = [v / 1000 for v in values]
    total_billions = sum(values_billions)
    print("Total aid in billions (USD):", total_billions)

    main_color = "#4e79a7"
    main_color = "#CCE5FF"
    highlight_color = "#f28e2b"
    highlight_color = "#FFB6C1"
    colors = [main_color] * 6 + [highlight_color] * (len(categories) - 6)

    fig, ax = plt.subplots(figsize=(16, 8))

    bars = ax.bar(
        range(len(categories)), values_billions, color=colors, width=0.7
    )

    ax.tick_params(axis="y", labelsize=16)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [disease_abbreviations[cat] for cat in categories],
        rotation=45,
        ha="right",
        fontsize=18,
    )
    ax.set_ylabel("Total Development Aid in billions (USD)", fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=16,
        )

    communicable_total = (
        sum(
            disbursement_data[cls]
            for cls in config["disease_categories"][
                "Communicable, maternal, neonatal, and nutritional diseases"
            ]
            if cls in disbursement_data
        )
        / 1000
    )
    noncommunicable_total = (
        sum(
            disbursement_data[cls]
            for cls in config["disease_categories"][
                "Non-communicable diseases"
            ]
            if cls in disbursement_data
        )
        / 1000
    )

    total_disease_disbursement = communicable_total + noncommunicable_total
    communicable_share = (
        communicable_total / total_disease_disbursement
    ) * 100
    noncommunicable_share = (
        noncommunicable_total / total_disease_disbursement
    ) * 100

    comm_patch = plt.Rectangle((0, 0), 1, 1, fc=main_color, ec="white")
    noncomm_patch = plt.Rectangle((0, 0), 1, 1, fc=highlight_color, ec="white")
    ax.legend(
        [comm_patch, noncomm_patch],
        [
            f"Communicable, maternal, neonatal, and nutritional diseases ({communicable_share:.1f}%)",
            f"Non-communicable diseases ({noncommunicable_share:.1f}%)",
        ],
        loc="upper right",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


def plot_bar_chart_only_communicable(
    disbursement_data: Dict[str, float],
    output_path: str,
    last_5_years: bool = False,
):
    """
    Plot and save a bar chart of total disbursements by disease focus area.

    Args:
        disbursement_data (Dict[str, float]): Disbursement data.
        output_path (str): Path to save the plot.
        last_5_years (bool): Whether to plot only the last 5 years.
    """
    sorted_disbursement = sorted(
        disbursement_data.items(), key=lambda item: item[1], reverse=True
    )
    categories, values = zip(*sorted_disbursement)

    communicable_categories = [
        "HIV/AIDS and sexually transmitted infections",
        "Respiratory infections and tuberculosis",
        "Enteric infections",
        "Neglected tropical diseases and malaria",
        "Maternal and neonatal disorders",
        "Nutritional deficiencies",
        "Mental disorders",
    ]

    # Filter both categories and values to only include communicable categories
    filtered_data = [
        (cat, val)
        for cat, val in zip(categories, values)
        if cat in communicable_categories
    ]
    categories, values = zip(*filtered_data) if filtered_data else ([], [])

    values_billions = [v / 1000 for v in values]

    main_color = "#CCE5FF"
    colors = [main_color] * (len(categories) - 1) + ["#FFB6C1"]

    fig, ax = plt.subplots(figsize=(9, 8))

    bars = ax.bar(
        range(len(categories)), values_billions, color=colors, width=0.7
    )

    ax.tick_params(axis="y", labelsize=20)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [disease_abbreviations[cat] for cat in categories],
        rotation=45,
        ha="right",
        fontsize=20,
    )
    ax.set_ylabel("Total aid in billions (USD)", fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=20,
        )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300)
    # plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()


def plot_bar_chart_only_non_communicable(
    disbursement_data: Dict[str, float],
    output_path: str,
    last_5_years: bool = False,
):
    """
    Plot and save a bar chart of total disbursements by disease focus area.

    Args:
        disbursement_data (Dict[str, float]): Disbursement data.
        output_path (str): Path to save the plot.
        last_5_years (bool): Whether to plot only the last 5 years.
    """
    sorted_disbursement = sorted(
        disbursement_data.items(), key=lambda item: item[1], reverse=True
    )
    categories, values = zip(*sorted_disbursement)

    non_communicable_categories = [
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

    # Filter both categories and values to only include communicable categories
    filtered_data = [
        (cat, val)
        for cat, val in zip(categories, values)
        if cat in non_communicable_categories
    ]
    categories, values = zip(*filtered_data) if filtered_data else ([], [])

    values_billions = [v / 1000 for v in values]

    main_color = "#FFB6C1"
    colors = [main_color] * len(categories)

    fig, ax = plt.subplots(figsize=(16, 8))

    bars = ax.bar(
        range(len(categories)), values_billions, color=colors, width=0.6
    )

    ax.tick_params(axis="y", labelsize=20)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [disease_abbreviations[cat] for cat in categories],
        rotation=45,
        ha="right",
        fontsize=20,
    )
    # ax.set_ylabel('Total aid in billions (USD)', fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=20,
        )

    plt.tight_layout()
    # plt.savefig(output_path, format='pdf', dpi=300)
    plt.savefig(output_path, format="svg", dpi=300, transparent=True)
    plt.close()


def plot_bar_with_total(
    disbursement_data: Dict[str, float],
    disease_categories: Dict[str, List[str]],
    output_path: str,
):
    """
    Plot and save a stacked bar chart of total disbursements by disease focus area,
    including a total bar and distinguishing between communicable and non-communicable diseases.

    Args:
        disbursement_data (Dict[str, float]): Disbursement data.
        disease_categories (Dict[str, List[str]]): Categories of diseases.
        output_path (str): Path to save the plot.
    """
    sorted_disbursement = sorted(
        disbursement_data.items(), key=lambda item: item[1], reverse=True
    )
    categories, values = zip(*sorted_disbursement)

    # Calculate the share of communicable vs noncommunicable diseases
    communicable_total = sum(
        disbursement_data[cls]
        for cls in disease_categories[
            "Communicable, maternal, neonatal, and nutritional diseases"
        ]
        if cls in disbursement_data
    )
    noncommunicable_total = sum(
        disbursement_data[cls]
        for cls in disease_categories["Non-communicable diseases"]
        if cls in disbursement_data
    )

    # Calculate shares
    total_disease_disbursement = communicable_total + noncommunicable_total
    communicable_share = (
        communicable_total / total_disease_disbursement
    ) * 100
    noncommunicable_share = (
        noncommunicable_total / total_disease_disbursement
    ) * 100

    # Calculate total
    total = sum(values)

    # Define custom colors for the first 6 categories
    custom_colors = ["#4e79a7"] * 6 + ["#f28e2b"] * (len(categories) - 6)
    colors = custom_colors

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 10))

    # Convert to billions
    communicable_total_billions = communicable_total / 1000
    noncommunicable_total_billions = noncommunicable_total / 1000
    total_billions = total / 1000

    # Plot the bottom half of the total bar with a distinct color
    ax.bar(
        0,
        communicable_total_billions,
        edgecolor="#4e79a7",
        facecolor="none",
        width=0.5,
        hatch="\\\\\\\\\\",
        bottom=noncommunicable_total_billions,
        label=f"Communicable, maternal, neonatal, and nutritional diseases ({communicable_share:.1f}% of total aid)",
    )
    ax.text(
        0,
        total_billions + (total_billions * 0.01),
        f"{total_billions:.1f}",
        ha="center",
        va="bottom",
        rotation=0,
        fontsize=10,
    )

    # Plot the top half of the total bar with a hatch pattern
    ax.bar(
        0,
        noncommunicable_total_billions,
        edgecolor="#f28e2b",
        facecolor="none",
        width=0.5,
        hatch="/////",
        label=f"Non-communicable Diseases ({noncommunicable_share:.1f}% of total aid)",
    )

    # Plot the stacked bars
    top = total_billions
    for i, (value, color) in enumerate(zip(values, colors), start=1):
        value_billions = value / 1000
        ax.bar(
            i,
            value_billions,
            color=color,
            width=0.5,
            bottom=top - value_billions,
        )
        ax.text(
            i,
            top + (total_billions * 0.01),
            f"{value_billions:.1f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=10,
        )
        top -= value_billions

    # Customize the chart
    ax.set_xticks(range(len(categories) + 1))
    ax.set_xticklabels(
        ["Total"] + list(categories), rotation=45, ha="right", fontsize=12
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Development Aid in billions (USD)", fontsize=14)
    # ax.set_title('Total Development Aid for Health by Disease Focus Area, 2000–2022', fontsize=16)

    # Add legend
    ax.legend(fontsize=12)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


def create_aid_vs_dalys_comparison(time_series_df, gbd_data_path, output_path):
    """
    Creates and saves a bar plot comparing aid allocation percentages with disease burden (DALYs) percentages
    across different disease categories.

    Args:
        time_series_df (pd.DataFrame): DataFrame containing aid time series data
        gbd_data_path (str): Path to the GBD data CSV file containing DALYs data
        output_path (str): Path to save the output plot
    """
    # Load GBD data
    gbd_dalys_df = pd.read_csv(gbd_data_path)

    # Filter for DALYs data
    gbd_dalys_df = gbd_dalys_df[
        gbd_dalys_df["measure_name"]
        == "DALYs (Disability-Adjusted Life Years)"
    ].reset_index(drop=True)

    # Aggregate DALYs by cause
    merge1 = (
        gbd_dalys_df[gbd_dalys_df["cause_name"] != "Other infectious diseases"]
        .groupby("cause_name")
        .agg({"val": "mean"})
        .reset_index()
        .rename(columns={"val": "dalys"})
    )

    # Prepare aid data
    merge2 = (
        pd.DataFrame(time_series_df.sum(axis=0))
        .reset_index()
        .rename(columns={"index": "cause_name", 0: "aid_total"})
    )

    # Merge datasets
    merged_df = merge1.merge(merge2, on="cause_name", how="left")

    # Calculate percentages
    total_aid = merged_df["aid_total"].sum()
    merged_df["aid_percentage"] = (merged_df["aid_total"] / total_aid) * 100
    merged_df["dalys_percentage"] = (
        merged_df["dalys"] / merged_df["dalys"].sum()
    ) * 100
    merged_df["difference"] = (
        merged_df["aid_percentage"] - merged_df["dalys_percentage"]
    )
    merged_df = merged_df.sort_values("difference", ascending=False)

    # Define disease name abbreviations
    abbreviations = {
        "HIV/AIDS and sexually transmitted infections": "HIV/AIDS & STI",
        "Respiratory infections and tuberculosis": "Respiratory Inf. & TB",
        "Enteric infections": "Enteric Infections",
        "Neglected tropical diseases and malaria": "NTDs & Malaria",
        "Maternal and neonatal disorders": "Maternal & Neonatal",
        "Nutritional deficiencies": "Nutritional Def.",
        "Diabetes and kidney diseases": "Diabetes & Kidney",
        "Digestive diseases": "Digestive Dis.",
        "Neurological disorders": "Neurological Dis.",
        "Mental disorders": "Mental Dis.",
        "Substance use disorders": "Substance Use",
        "Musculoskeletal disorders": "Musculoskeletal",
        "Neoplasms": "Neoplasms",
        "Cardiovascular diseases": "Cardiovascular",
        "Chronic respiratory diseases": "Chronic Respiratory",
        "Skin and subcutaneous diseases": "Skin & Subcutaneous",
        "Sense organ diseases": "Sense Organ Dis.",
        "Transport injuries": "Transport Inj.",
        "Unintentional injuries": "Unintentional Inj.",
        "Self-harm and interpersonal violence": "Self-harm & Violence",
    }

    abbreviations = {
        "HIV/AIDS and sexually transmitted infections": "HIV/AIDS & STIs",
        "Respiratory infections and tuberculosis": "Respiratory inf. & TB",
        "Nutritional deficiencies": "Nutritional deficiencies",
        "Maternal and neonatal disorders": "Maternal & neonatal dis.",
        "Neglected tropical diseases and malaria": "NTDs & malaria",
        "Enteric infections": "Enteric inf.",
        "Mental disorders": "Mental disorders",
        "Neoplasms": "Neoplasms",
        "Sense organ diseases": "Sense organ dis.",
        "Cardiovascular diseases": "Cardiovascular dis.",
        "Diabetes and kidney diseases": "Diabetes & kidney dis.",
        "Substance use disorders": "Substance use dis.",
        "Musculoskeletal disorders": "Musculoskeletal dis.",
        "Neurological disorders": "Neurological dis.",
        "Chronic respiratory diseases": "Chronic respiratory dis.",
        "Skin and subcutaneous diseases": "Skin & subcut. dis.",
        "Digestive diseases": "Digestive dis.",
    }

    # Create plot
    plt.figure(figsize=(14, 6), dpi=300)

    # Set bar positions
    width = 0.35
    x = np.arange(len(merged_df))

    # Create bars
    daly_bars = plt.bar(
        x + width / 2,
        merged_df["dalys_percentage"],
        width,
        label="Disease burden (DALYs, %)",
        color="#e74c3c",
        alpha=0.8,
    )
    aid_bars = plt.bar(
        x - width / 2,
        merged_df["aid_percentage"],
        width,
        label="Aid allocation (%)",
        color="#3498db",
        alpha=0.8,
    )

    # Customize plot
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)
    plt.ylabel("Share of aid and burden", fontsize=16)
    plt.legend(fontsize=16)

    # Set x-axis labels
    abbreviated_names = [
        abbreviations.get(name, name) for name in merged_df["cause_name"]
    ]
    plt.xticks(x, abbreviated_names, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=100, decimals=0)
    )

    # Adjust layout
    plt.tight_layout()

    # Save plot and close figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    Main function to run the analysis and generate all plots.
    """
    # Check if required output files already exist
    results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    disbursement_stats_file = os.path.join(
        results_dir, "disbursement_stats.csv"
    )
    disbursement_stats_5yr_file = os.path.join(
        results_dir, "disbursement_stats_last_5_years.csv"
    )
    time_series_file = os.path.join(results_dir, "time_series.csv")

    if (
        os.path.exists(disbursement_stats_file)
        and os.path.exists(disbursement_stats_5yr_file)
        and os.path.exists(time_series_file)
    ):
        print("Files already exist, skipping computation")
        disbursement_stats_df = pd.read_csv(disbursement_stats_file)
        disbursement_stats_last_5_years_df = pd.read_csv(
            disbursement_stats_5yr_file
        )
        time_series_df = pd.read_csv(time_series_file)
        time_series_df.rename(
            columns={time_series_df.columns[0]: "Year"}, inplace=True
        )

        # Convert DataFrames to the expected dictionary format
        time_series_df = time_series_df.sort_values(by="Year")
        time_series_df = time_series_df.reset_index(drop=True)
        time_series_df.index = time_series_df.index + 2000
        time_series_df = time_series_df.sort_index()
        time_series = time_series_df.to_dict()

        # Reconstruct disbursement_stats dictionary
        disbursement_stats = {
            "count": dict(
                zip(
                    disbursement_stats_df["category"],
                    disbursement_stats_df["count"],
                )
            ),
            "total": dict(
                zip(
                    disbursement_stats_df["category"],
                    disbursement_stats_df["total"],
                )
            ),
            "avg": dict(
                zip(
                    disbursement_stats_df["category"],
                    disbursement_stats_df["avg"],
                )
            ),
        }

        # Reconstruct disbursement_stats_last_5_years dictionary
        disbursement_stats_last_5_years = {
            "count": dict(
                zip(
                    disbursement_stats_last_5_years_df["category"],
                    disbursement_stats_last_5_years_df["count"],
                )
            ),
            "total": dict(
                zip(
                    disbursement_stats_last_5_years_df["category"],
                    disbursement_stats_last_5_years_df["total"],
                )
            ),
            "avg": dict(
                zip(
                    disbursement_stats_last_5_years_df["category"],
                    disbursement_stats_last_5_years_df["avg"],
                )
            ),
        }
    else:
        dtypes = {"CrsID": pl.Utf8}
        df = load_data(config["input_file"], dtypes)
        df_filtered = df.filter(
            pl.col("IncomegroupName") != "Part I unallocated by income"
        )
        df_filtered_last_5_years = df_filtered.filter(pl.col("Year") >= 2018)

        disbursement_stats = calculate_disbursement_stats(df)
        disbursement_stats_last_5_years = calculate_disbursement_stats(
            df_filtered_last_5_years
        )
        top_recipients = get_top_recipients(df_filtered)
        time_series = calculate_time_series(df)

        data_dir = os.path.join(PROJECT_ROOT, "data")

        disbursement_stats_df = pd.DataFrame(
            {
                "category": list(disbursement_stats["count"].keys()),
                "count": list(disbursement_stats["count"].values()),
                "total": [
                    disbursement_stats["total"][k]
                    for k in disbursement_stats["count"].keys()
                ],
                "avg": [
                    disbursement_stats["avg"][k]
                    for k in disbursement_stats["count"].keys()
                ],
            }
        )

        # For disbursement_stats_last_5_years
        disbursement_stats_last_5_years_df = pd.DataFrame(
            {
                "category": list(
                    disbursement_stats_last_5_years["count"].keys()
                ),
                "count": list(
                    disbursement_stats_last_5_years["count"].values()
                ),
                "total": [
                    disbursement_stats_last_5_years["total"][k]
                    for k in disbursement_stats_last_5_years["count"].keys()
                ],
                "avg": [
                    disbursement_stats_last_5_years["avg"][k]
                    for k in disbursement_stats_last_5_years["count"].keys()
                ],
            }
        )

        # For time_series
        time_series_df = pd.DataFrame(time_series)

        # For top_recipients
        top_recipients_df = pd.DataFrame(
            [
                {"category": cat, "country": country, "funding": amount}
                for cat, recipients in top_recipients.items()
                for country, amount in recipients.items()
            ]
        )

        # Save to CSV
        disbursement_stats_df.to_csv(
            os.path.join(data_dir, "results", "disbursement_stats.csv"),
            index=False,
        )
        disbursement_stats_last_5_years_df.to_csv(
            os.path.join(
                data_dir, "results", "disbursement_stats_last_5_years.csv"
            ),
            index=False,
        )
        time_series_df.to_csv(
            os.path.join(data_dir, "results", "time_series.csv"), index=True
        )
        top_recipients_df.to_csv(
            os.path.join(data_dir, "results", "top_recipients.csv"),
            index=False,
        )

    # Use the correct path for saving plots
    plots_dir = os.path.join(PROJECT_ROOT, "plots", "aid_plots")

    plot_stacked_bar(
        time_series,
        os.path.join(
            plots_dir,
            "stacked_bar_disbursement_per_year_by_llm_classification.pdf",
        ),
    )
    plot_stacked_bar_non_communicable(
        time_series,
        os.path.join(
            plots_dir,
            "stacked_bar_disbursement_per_year_by_llm_classification_non_communicable.pdf",
        ),
    )
    plot_line_graph_separated(
        time_series,
        os.path.join(
            plots_dir,
            "line_disbursement_per_year_by_llm_classification_broken_axis_red_separator.pdf",
        ),
    )
    plot_line_graph(
        time_series,
        os.path.join(
            plots_dir,
            "line_disbursement_per_year_by_llm_classification_with_covid.pdf",
        ),
        include_respiratory=True,
    )
    plot_line_graph(
        time_series,
        os.path.join(
            plots_dir,
            "line_disbursement_per_year_by_llm_classification_without_covid.pdf",
        ),
        include_respiratory=False,
    )
    plot_disease_share_of_aid(
        time_series,
        os.path.join(plots_dir, "share_of_aid_by_disease_area.pdf"),
    )
    plot_bar_chart(
        disbursement_stats["total"],
        os.path.join(
            plots_dir,
            "usd_disbursement_per_llm_classification_without_total.pdf",
        ),
    )
    plot_bar_chart_only_communicable(
        disbursement_stats["total"],
        os.path.join(
            plots_dir,
            "usd_disbursement_per_llm_classification_without_total_only_communicable.svg",
        ),
    )
    plot_bar_chart_only_non_communicable(
        disbursement_stats["total"],
        os.path.join(
            plots_dir,
            "usd_disbursement_per_llm_classification_without_total_only_non_communicable.svg",
        ),
    )
    plot_bar_chart(
        disbursement_stats_last_5_years["total"],
        os.path.join(
            plots_dir,
            "usd_disbursement_per_llm_classification_without_total_last_5_years.pdf",
        ),
        last_5_years=True,
    )
    plot_bar_with_total(
        disbursement_stats["total"],
        config["disease_categories"],
        os.path.join(
            plots_dir, "total_usd_disbursement_per_llm_classification.pdf"
        ),
    )

    time_series_df_without_2022 = time_series_df[
        time_series_df.index != 2022.0
    ]
    create_aid_vs_dalys_comparison(
        time_series_df=time_series_df_without_2022,
        gbd_data_path=os.path.join(
            PROJECT_ROOT,
            "data",
            "raw",
            "background_data",
            "IHME-GBD_2021_DATA-96af6e94-1.csv",
        ),
        output_path=os.path.join(
            PROJECT_ROOT,
            "plots",
            "aid_plots",
            "total_aid_vs_dalys_percentage_comparison.pdf",
        ),
    )


if __name__ == "__main__":
    main()
