"""Distribution comparison for the 1% reproducibility subset versus the full corpus."""

from __future__ import annotations

import pandas as pd

from revision.labels import STANDARD_DISEASES, build_label_display_map, split_labels
from revision.paths import results_dir


def _explode_named_disease_labels(df: pd.DataFrame, column: str) -> pd.Series:
    allowed = {label.casefold() for label in STANDARD_DISEASES}
    labels = df[column].apply(split_labels).explode().dropna()
    labels = labels[labels.isin(allowed)]
    return labels


def compare_subset_distribution(
    subset_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    subset_col: str = "llm_classification_1",
    full_col: str = "llm_classification",
) -> pd.DataFrame:
    """Compare named-disease label frequencies in the reproducibility subset and full corpus."""
    subset_labels = _explode_named_disease_labels(subset_df, subset_col)
    full_labels = _explode_named_disease_labels(full_df, full_col)
    display_map = build_label_display_map([subset_df[subset_col], full_df[full_col]])

    subset_counts = subset_labels.value_counts()
    full_counts = full_labels.value_counts()
    ordered_labels = sorted(full_counts.index, key=lambda label: full_counts[label], reverse=True)

    rows = []
    total_subset = int(subset_counts.sum())
    total_full = int(full_counts.sum())
    for label in ordered_labels:
        rows.append(
            {
                "Disease": display_map.get(label, label.title()),
                "Subset count": int(subset_counts.get(label, 0)),
                "Subset share (%)": (
                    100.0 * float(subset_counts.get(label, 0)) / total_subset if total_subset else 0.0
                ),
                "Full dataset count": int(full_counts.get(label, 0)),
                "Full dataset share (%)": (
                    100.0 * float(full_counts.get(label, 0)) / total_full if total_full else 0.0
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = results_dir() / "subset_distribution_comparison_named_diseases.csv"
    out_df.to_csv(out_path, index=False)
    return out_df
