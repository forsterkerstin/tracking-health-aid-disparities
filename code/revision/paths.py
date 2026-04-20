"""Repository-relative paths for revision analyses."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """HealthDevAidNeed repository root (parent of ``code/``)."""
    return Path(__file__).resolve().parent.parent.parent


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolved_validation_csv() -> Path | None:
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "revision" / "resolved_double_annotated_labels.csv",
            r / "revision" / "resolved_double_annotated_labels.csv",
            r / "revision_data" / "resolved_double_annotated_labels.csv",
        ]
    )


def final_aid_df_csv() -> Path | None:
    p = repo_root() / "data" / "processed" / "aid_funding_data" / "final_df.csv"
    return p if p.is_file() else None


def robustness_results_csv() -> Path | None:
    """Reproducibility subset with all robustness-model outputs, including DeepSeek."""
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "revision" / "robustness_results.csv",
            r / "revision" / "robustness_results.csv",
            r / "revision_data" / "robustness_results.csv",
        ]
    )


def robustness_short_non_en_csv() -> Path | None:
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "revision" / "combined_short_df.csv",
            r / "data" / "processed" / "classification" / "combined_short_df.csv",
            r / "revision_data" / "combined_short_df.csv",
        ]
    )


def negation_labeled_csv() -> Path | None:
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "revision" / "df_health_negation_classified_and_labeled.csv",
            r / "data" / "df_health_negation_classified_and_labeled.csv",
            r / "data" / "processed" / "classification" / "df_health_negation_classified_and_labeled.csv",
            r / "revision_data" / "df_health_negation_classified_and_labeled.csv",
        ]
    )


def chunk_combined_csv() -> Path | None:
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "revision" / "chunk_df_combined_classified_final.csv",
            r / "data" / "chunk_df_combined_classified_final.csv",
            r / "code" / "visualization" / "chunk_df_combined_classified_final.csv",
            r / "revision_data" / "chunk_df_combined_classified_final.csv",
        ]
    )


def unique_raw_text_counts_csv() -> Path | None:
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "processed" / "classification" / "df_unique_raw_texts.csv",
            r / "revision_data" / "df_unique_raw_texts.csv",
        ]
    )


def classification_keywords_csv() -> Path | None:
    r = repo_root()
    return _first_existing(
        [
            r / "data" / "processed" / "classification" / "keywords_used_in_classification.csv",
            r / "revision_data" / "keywords_used_in_classification.csv",
        ]
    )


def plots_dir() -> Path:
    d = repo_root() / "plots" / "revision"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = repo_root() / "results" / "revision"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_revision_output_layout() -> None:
    """Create ``plots/revision`` and ``results/revision`` (and parent ``plots/``, ``results/``) if missing."""
    plots_dir()
    results_dir()


def validation_data_dir() -> Path:
    """Repo-root ``revision/`` for validation CSVs; created if missing."""
    d = repo_root() / "revision"
    d.mkdir(parents=True, exist_ok=True)
    return d
