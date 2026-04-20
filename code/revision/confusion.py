"""Normalized multi-label confusion matrices (heatmap PDFs)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.preprocessing import MultiLabelBinarizer  # noqa: E402

from revision.labels import build_label_display_map, split_labels
from revision.paths import plots_dir


def get_all_unique_labels(df: pd.DataFrame, columns: List[str]) -> Set[str]:
    all_labels: Set[str] = set()
    for column in columns:
        for labels in df[column]:
            all_labels.update(label for label in labels if label)
    return all_labels


def plot_heatmap(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    output_file: str,
    text_body: str | None = None,
    color: str = "Purples",
) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap=color, vmin=0, vmax=1, square=True, annot=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if text_body is None:
        text_body = "Share of activities from a row category attributed to a column category"
    plt.text(
        df.shape[1] + 4,
        df.shape[0] / 2,
        text_body,
        rotation=90,
        verticalalignment="center",
    )
    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def create_confusion_matrix_between_columns(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    output_file: str,
    *,
    col1_desc: str | None = None,
    col2_desc: str | None = None,
    color: str = "Greens",
) -> pd.DataFrame:
    """Row-normalized confusion matrix: col1 (rows) vs col2 (columns)."""
    work = df[[col1, col2]].dropna(subset=[col1, col2]).copy()
    display_map = build_label_display_map([work[col1], work[col2]])
    work[col1] = work[col1].apply(split_labels)
    work[col2] = work[col2].apply(split_labels)

    all_labels = sorted(get_all_unique_labels(work, [col1, col2]))
    mlb = MultiLabelBinarizer(classes=all_labels)
    y_1 = mlb.fit_transform(work[col1])
    y_2 = mlb.transform(work[col2])

    confusion_matrix = np.dot(y_1.T, y_2)
    epsilon = 1e-10
    row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix_normalized = confusion_matrix / (row_sums + epsilon)

    classes_display = [display_map.get(c, c.title()) for c in mlb.classes_]
    cm_df = pd.DataFrame(confusion_matrix_normalized, index=classes_display, columns=classes_display)

    x_label = col2_desc if col2_desc is not None else f"Categories assigned by `{col2}`"
    y_label = col1_desc if col1_desc is not None else f"Categories assigned by `{col1}`"
    plot_heatmap(cm_df, x_label, y_label, output_file, color=color)
    return cm_df


def save_pairwise_confusion_matrices(
    df: pd.DataFrame,
    basename: str,
) -> list[str]:
    """Save pairwise confusion matrices for available LLM runs and annotators."""
    out: list[str] = []
    pdir = plots_dir()

    if {"llm_classification_1", "llm_classification_2"}.issubset(df.columns):
        path = os.path.join(pdir, f"{basename}_llm1_vs_llm2.pdf")
        create_confusion_matrix_between_columns(
            df,
            "llm_classification_1",
            "llm_classification_2",
            path,
            col1_desc="Categories assigned by LLM run 1",
            col2_desc="Categories assigned by LLM run 2 (e.g. different seed)",
        )
        out.append(path)

    if {"llm_classification_1", "llm_classification_3"}.issubset(df.columns):
        path = os.path.join(pdir, f"{basename}_llm1_vs_llm3.pdf")
        create_confusion_matrix_between_columns(
            df,
            "llm_classification_1",
            "llm_classification_3",
            path,
            col1_desc="Categories assigned by LLM run 1",
            col2_desc="Categories assigned by alternative LLM",
        )
        out.append(path)

    if {"llm_classification_1", "llm_classification_4"}.issubset(df.columns):
        path = os.path.join(pdir, f"{basename}_llm1_vs_llm4.pdf")
        create_confusion_matrix_between_columns(
            df,
            "llm_classification_1",
            "llm_classification_4",
            path,
            col1_desc="Categories assigned by LLM run 1",
            col2_desc="Categories assigned by alternative LLM (DeepSeek benchmark)",
        )
        out.append(path)

    if {"manual_label", "manual_label_2"}.issubset(df.columns):
        path = os.path.join(pdir, f"{basename}_annotator1_vs_annotator2.pdf")
        create_confusion_matrix_between_columns(
            df,
            "manual_label",
            "manual_label_2",
            path,
            col1_desc="Manual annotator 1",
            col2_desc="Manual annotator 2",
        )
        out.append(path)

    return out
