"""Helpers for resolved manual-validation labels used in Revision II analyses."""

from __future__ import annotations

import pandas as pd

from revision.paths import resolved_validation_csv


def load_resolved_validation_table() -> pd.DataFrame | None:
    """Load the canonical resolved manual-validation CSV when available."""
    path = resolved_validation_csv()
    if path is None:
        return None

    resolved = pd.read_csv(path, low_memory=False)
    required = {"raw_text", "final_label"}
    missing = sorted(required - set(resolved.columns))
    if missing:
        raise ValueError(f"Resolved validation file {path} is missing required columns: {missing}")

    resolved = resolved.drop_duplicates(subset=["raw_text"], keep="first").copy()
    return resolved


def merge_resolved_ground_truth(
    df: pd.DataFrame,
    *,
    raw_text_col: str = "raw_text",
    fallback_col: str = "manual_label",
    output_col: str = "ground_truth_label",
) -> pd.DataFrame:
    """Attach resolved validation labels to a dataframe, falling back to `fallback_col`."""
    merged = df.copy()
    if fallback_col not in merged.columns:
        raise ValueError(f"Missing fallback label column {fallback_col!r}.")

    merged[output_col] = merged[fallback_col]

    resolved = load_resolved_validation_table()
    if resolved is None or raw_text_col not in merged.columns:
        return merged

    merged = merged.merge(
        resolved[["raw_text", "final_label"]].rename(columns={"raw_text": raw_text_col}),
        on=raw_text_col,
        how="left",
    )
    merged[output_col] = merged["final_label"].fillna(merged[fallback_col])
    merged = merged.drop(columns=["final_label"])

    return merged
