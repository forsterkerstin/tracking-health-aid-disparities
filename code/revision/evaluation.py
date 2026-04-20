"""Micro/macro multi-label metrics and optional confusion export."""

from __future__ import annotations

import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from revision.labels import split_labels
from revision.paths import results_dir


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    non_zero_precisions = [p for p in class_precision if p > 0]
    non_zero_recalls = [r for r in class_recall if r > 0]
    non_zero_f1s = [f for f in class_f1 if f > 0]

    precision_macro = (
        sum(non_zero_precisions) / len(non_zero_precisions) if non_zero_precisions else 0.0
    )
    recall_macro = sum(non_zero_recalls) / len(non_zero_recalls) if non_zero_recalls else 0.0
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


def get_all_unique_labels(df: pd.DataFrame, columns: List[str]) -> Set[str]:
    all_labels: Set[str] = set()
    for column in columns:
        for labels in df[column]:
            all_labels.update(label for label in labels if label)
    return all_labels


def evaluate_multilabel(
    df: pd.DataFrame,
    col_true: str,
    col_pred: str,
    *,
    model_name: str,
    csv_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute overall and per-class metrics; write CSVs under ``results/revision/``."""
    df_proc = df[[col_true, col_pred]].dropna(subset=[col_true, col_pred]).copy()
    df_proc = df_proc[
        df_proc[col_true].astype(str).str.strip().ne("")
        & df_proc[col_pred].astype(str).str.strip().ne("")
        & ~df_proc[col_pred].astype(str).str.startswith("ERROR:")
    ].copy()
    df_proc[col_true] = df_proc[col_true].apply(split_labels)
    df_proc[col_pred] = df_proc[col_pred].apply(split_labels)

    columns_to_process = [col_true, col_pred]
    all_labels = get_all_unique_labels(df_proc, columns_to_process)
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))

    y_true = mlb.fit_transform(df_proc[col_true])
    y_pred = mlb.transform(df_proc[col_pred])

    overall = calculate_metrics(y_true, y_pred)
    overall["Model"] = model_name
    overall_df = pd.DataFrame([overall])

    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    true_positives = (y_true & y_pred).sum(axis=0)

    per_rows = []
    for label, prec, rec, f1v, tp in zip(
        mlb.classes_, class_precision, class_recall, class_f1, true_positives
    ):
        if label:
            per_rows.append(
                {
                    "Model": model_name,
                    "Class": label,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1v,
                    "True Positives": int(tp),
                }
            )
    per_df = pd.DataFrame(per_rows)

    rdir = results_dir()
    overall_df.to_csv(os.path.join(rdir, f"overall_metrics_{csv_prefix}.csv"), index=False)
    per_df.to_csv(os.path.join(rdir, f"per_class_metrics_{csv_prefix}.csv"), index=False)

    return overall_df, per_df
