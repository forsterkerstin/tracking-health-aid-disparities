# Imports
import os
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Update the DATA_DIR and PLOT_DIR paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "classification")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "classification")


# Functions
def split_labels(labels: str) -> Set[str]:
    """
    Split a string of labels into a set of individual labels.

    Args:
        labels (str): A string of labels separated by commas or semicolons.

    Returns:
        Set[str]: A set of individual labels.
    """
    return set(
        label.strip().lower()
        for label in labels.replace(";", ",").split(",")
        if label.strip()
    )


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate various classification metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        Dict[str, float]: A dictionary containing various classification metrics.
    """
    precision_micro = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    non_zero_precisions = [p for p in class_precision if p > 0]
    non_zero_recalls = [r for r in class_recall if r > 0]
    non_zero_f1s = [f for f in class_f1 if f > 0]

    precision_macro = (
        sum(non_zero_precisions) / len(non_zero_precisions)
        if non_zero_precisions
        else 0.0
    )
    recall_macro = (
        sum(non_zero_recalls) / len(non_zero_recalls)
        if non_zero_recalls
        else 0.0
    )
    f1_macro = sum(non_zero_f1s) / len(non_zero_f1s) if non_zero_f1s else 0.0

    return {
        "Micro-averaged Precision": precision_micro,
        "Micro-averaged Recall": recall_micro,
        "Micro-averaged F1 Score": f1_micro,
        "Accuracy": accuracy,
        "Macro-averaged Precision": precision_macro,
        "Macro-averaged Recall": recall_macro,
        "Macro-averaged F1 Score": f1_macro,
    }


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the manually labelled classification data.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    df["keyword_classification"] = df["keyword_classification"].fillna("Other")
    df = df.dropna(subset=["manual_label"])

    columns_to_process = [
        "manual_label",
        "llm_classification",
        "keyword_classification",
    ]
    for column in columns_to_process:
        df[column] = df[column].apply(split_labels)

    return df


def get_all_unique_labels(df: pd.DataFrame, columns: List[str]) -> Set[str]:
    """
    Get all unique labels from specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to process.

    Returns:
        Set[str]: Set of all unique labels.
    """
    all_labels = set()
    for column in columns:
        for labels in df[column]:
            all_labels.update(label for label in labels if label)
    return all_labels


def calculate_overall_metrics(
    y_true: np.ndarray,
    df: pd.DataFrame,
    columns: List[str],
    mlb: MultiLabelBinarizer,
) -> pd.DataFrame:
    """
    Calculate overall metrics for each classification column.

    Args:
        y_true (np.ndarray): True labels.
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to process.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer.

    Returns:
        pd.DataFrame: DataFrame containing overall metrics.
    """
    overall_metrics_df = pd.DataFrame()
    for column in columns:
        y_pred = mlb.transform(df[column])
        metrics = calculate_metrics(y_true, y_pred)
        metrics["Model"] = column
        overall_metrics_df = pd.concat(
            [overall_metrics_df, pd.Series(metrics, name=column).to_frame().T],
            ignore_index=True,
        )
    return overall_metrics_df


def calculate_per_class_metrics(
    y_true: np.ndarray,
    df: pd.DataFrame,
    columns: List[str],
    mlb: MultiLabelBinarizer,
) -> pd.DataFrame:
    """
    Calculate per-class metrics for each classification column.

    Args:
        y_true (np.ndarray): True labels.
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to process.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer.

    Returns:
        pd.DataFrame: DataFrame containing per-class metrics.
    """
    per_class_metrics_df = pd.DataFrame()
    for column in columns:
        y_pred = mlb.transform(df[column])
        class_precision = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        class_recall = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        true_positives = (y_true & y_pred).sum(axis=0)

        for label, prec, rec, tp in zip(
            mlb.classes_, class_precision, class_recall, true_positives
        ):
            if label:  # Skip empty string labels
                per_class_metrics_df = pd.concat(
                    [
                        per_class_metrics_df,
                        pd.DataFrame(
                            [
                                {
                                    "Model": column,
                                    "Class": label,
                                    "Precision": prec,
                                    "Recall": rec,
                                    "True Positives": tp,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    return per_class_metrics_df


def plot_heatmap(df: pd.DataFrame, x_label: str, output_file: str):
    """
    Plot a heatmap from the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title of the plot.
        output_file (str): Output file path for saving the plot.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap="Purples", vmin=0, vmax=1, square=True, annot=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel(x_label)
    plt.ylabel("Categories assigned by the LLM Classification")
    plt.text(
        df.shape[1] + 4,
        df.shape[0] / 2,
        "Share of activities from a row category attributed to a column category",
        rotation=90,
        verticalalignment="center",
    )
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def plot_precision_recall_heatmap(df: pd.DataFrame, output_file: str):
    """
    Plot a heatmap of precision and recall for each disease class.

    Args:
        df (pd.DataFrame): Input DataFrame containing precision and recall values.
        output_file (str): Output file path for saving the plot.
    """
    plt.figure(figsize=(14, 14), dpi=300)
    custom_cmap = sns.light_palette("blue", as_cmap=True, input="xkcd")
    custom_cmap.set_under(color="whitesmoke")
    ax = sns.heatmap(
        df,
        annot=True,
        cmap=custom_cmap,
        square=True,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    plt.title("Precision and Recall for Each Disease Class (LLM)", pad=20)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def main():
    # Load and preprocess data
    manual_df = load_and_preprocess_data(
        os.path.join(DATA_DIR, "manually_labelled_classification_data.csv")
    )

    # Get all unique labels
    columns_to_process = [
        "manual_label",
        "llm_classification",
        "keyword_classification",
    ]
    all_labels = get_all_unique_labels(manual_df, columns_to_process)

    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))

    # Transform manual labels to binary format
    y_true = mlb.fit_transform(manual_df["manual_label"])

    # Calculate overall metrics
    overall_metrics_df = calculate_overall_metrics(
        y_true, manual_df, columns_to_process[1:], mlb
    )

    # Calculate per-class metrics
    per_class_metrics_df = calculate_per_class_metrics(
        y_true, manual_df, columns_to_process[1:], mlb
    )

    # Save results
    overall_metrics_df.to_csv(
        os.path.join(DATA_DIR, "overall_metrics_keywords_vs_llm.csv"),
        index=False,
    )
    per_class_metrics_df.to_csv(
        os.path.join(DATA_DIR, "per_class_metrics_keywords_vs_llm.csv"),
        index=False,
    )

    # Create confusion matrix for LLM classification vs manual labels
    y_pred_llm = mlb.transform(manual_df["llm_classification"])
    confusion_matrix_llm_manual = np.dot(y_true.T, y_pred_llm)
    confusion_matrix_llm_manual_normalized = (
        confusion_matrix_llm_manual
        / confusion_matrix_llm_manual.sum(axis=1)[:, np.newaxis]
    )

    # Create confusion matrix for LLM classification vs keyword classification
    y_pred_keyword = mlb.transform(manual_df["keyword_classification"])
    confusion_matrix_llm_keyword = np.dot(y_pred_keyword.T, y_pred_llm)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    confusion_matrix_llm_keyword_normalized = confusion_matrix_llm_keyword / (
        confusion_matrix_llm_keyword.sum(axis=1)[:, np.newaxis] + epsilon
    )

    # Replace diagonal values with precision scores
    precision_scores = per_class_metrics_df[
        per_class_metrics_df["Model"] == "llm_classification"
    ].set_index("Class")["Precision"]
    np.fill_diagonal(confusion_matrix_llm_manual_normalized, precision_scores)
    np.fill_diagonal(confusion_matrix_llm_keyword_normalized, precision_scores)

    # Create DataFrames for the normalized confusion matrices
    mlb.classes_ = [cls.capitalize() for cls in mlb.classes_]
    df_llm_manual_normalized = pd.DataFrame(
        confusion_matrix_llm_manual_normalized,
        index=mlb.classes_,
        columns=mlb.classes_,
    )
    df_llm_keyword_normalized = pd.DataFrame(
        confusion_matrix_llm_keyword_normalized,
        index=mlb.classes_,
        columns=mlb.classes_,
    )

    # Plot heatmap LLM vs manual classification
    plot_heatmap(
        df_llm_manual_normalized,
        "Manually labelled categories",
        os.path.join(
            PLOT_DIR, "heatmap_comparing_share_of_llm_vs_manual_labels.pdf"
        ),
    )

    # Plot heatmap LLM vs Keyword classification
    plot_heatmap(
        df_llm_keyword_normalized,
        "Categories assigned by the Keyword classification",
        os.path.join(
            PLOT_DIR,
            "heatmap_comparing_share_of_llm_vs_keyword_classification.pdf",
        ),
    )

    # Plot precision and recall heatmap
    per_class_metrics_df = pd.read_csv(
        os.path.join(DATA_DIR, "per_class_metrics_keywords_vs_llm.csv")
    )
    llm_metrics = per_class_metrics_df[
        per_class_metrics_df["Model"] == "llm_classification"
    ].set_index("Class")[["Precision", "Recall"]]
    plot_precision_recall_heatmap(
        llm_metrics, os.path.join(PLOT_DIR, "precision_recall_llm_heatmap.pdf")
    )


if __name__ == "__main__":
    main()
