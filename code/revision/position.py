"""Disease-mention position robustness analyses for Revision II."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import MultiLabelBinarizer  # noqa: E402

from revision.evaluation import calculate_metrics, get_all_unique_labels
from revision.labels import split_labels
from revision.paths import classification_keywords_csv, plots_dir, results_dir, unique_raw_text_counts_csv
from revision.validation import merge_resolved_ground_truth


def _load_keyword_dict_exact() -> dict[str, list[str]]:
    path = classification_keywords_csv()
    if path is None:
        raise FileNotFoundError("keywords_used_in_classification.csv not found.")

    keywords_df = pd.read_csv(path)
    raw_dict = {
        row["Classification Category"]: row["Keywords"].split(", ")
        for _, row in keywords_df.iterrows()
    }

    processed: dict[str, list[str]] = {}
    for category, keywords in raw_dict.items():
        processed[category] = [
            keyword.lower()
            for keyword in keywords
            if not any(other in keyword.lower() for other in keywords if keyword != other)
        ]
    return processed


def _earliest_keyword_position(
    text: str,
    keywords: list[str],
) -> tuple[str | None, int | None, float | None]:
    text_lower = text.lower()
    best_keyword = None
    best_pos = None
    for keyword in keywords:
        pos = text_lower.find(keyword)
        if pos == -1:
            continue
        if best_pos is None or pos < best_pos or (pos == best_pos and len(keyword) > len(best_keyword or "")):
            best_keyword = keyword
            best_pos = pos
    if best_pos is None:
        return None, None, None
    return best_keyword, best_pos, best_pos / max(len(text_lower), 1)


def save_keyword_position_corpus_summary() -> tuple[str, str, str]:
    """Save corpus-wide keyword position summary CSVs and quartile plot."""
    path = unique_raw_text_counts_csv()
    if path is None:
        raise FileNotFoundError("df_unique_raw_texts.csv not found.")

    keyword_dict = _load_keyword_dict_exact()
    all_keywords = sorted({keyword for kws in keyword_dict.values() for keyword in kws})

    unique_texts_df = pd.read_csv(path, usecols=["raw_text", "count"], low_memory=False)
    unique_texts_df["raw_text"] = unique_texts_df["raw_text"].fillna("").astype(str)
    unique_texts_df = unique_texts_df[unique_texts_df["raw_text"].str.strip() != ""].copy()
    unique_texts_df["count"] = unique_texts_df["count"].fillna(1).astype(int)

    match_info = unique_texts_df["raw_text"].apply(
        lambda text: _earliest_keyword_position(text, all_keywords)
    )
    unique_texts_df[["first_keyword", "first_keyword_char_pos", "first_keyword_rel_pos"]] = pd.DataFrame(
        match_info.tolist(),
        index=unique_texts_df.index,
    )
    unique_texts_df["has_keyword_match"] = unique_texts_df["first_keyword_rel_pos"].notna()
    unique_texts_df["first_quartile_match"] = unique_texts_df["first_keyword_rel_pos"].le(0.25)

    matched_df = unique_texts_df[unique_texts_df["has_keyword_match"]].copy()
    weighted_total_projects = int(unique_texts_df["count"].sum())
    weighted_matched_projects = int(matched_df["count"].sum())
    weighted_first_quartile_projects = int(
        matched_df.loc[matched_df["first_quartile_match"], "count"].sum()
    )

    quartile_labels = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
    matched_df["position_quartile"] = pd.cut(
        matched_df["first_keyword_rel_pos"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=quartile_labels,
        include_lowest=True,
    )
    quartile_summary = (
        matched_df.groupby("position_quartile", observed=False)["count"]
        .sum()
        .reindex(quartile_labels, fill_value=0)
        .rename("projects")
        .reset_index()
    )
    quartile_summary["share_of_keyword_matched_projects"] = (
        quartile_summary["projects"] / weighted_matched_projects if weighted_matched_projects else 0.0
    )

    position_summary = pd.DataFrame(
        [
            {"metric": "Total projects represented", "value": weighted_total_projects},
            {
                "metric": "Projects with >=1 keyword-method disease mention",
                "value": weighted_matched_projects,
            },
            {
                "metric": "Projects where earliest keyword-method disease mention is in first quartile",
                "value": weighted_first_quartile_projects,
            },
            {
                "metric": "Share of keyword-matched projects with earliest mention in first quartile",
                "value": (
                    weighted_first_quartile_projects / weighted_matched_projects
                    if weighted_matched_projects
                    else 0.0
                ),
            },
        ]
    )

    summary_csv = results_dir() / "keyword_position_summary_exact_keyword_method.csv"
    quartile_csv = results_dir() / "keyword_position_quartiles_exact_keyword_method.csv"
    position_summary.to_csv(summary_csv, index=False)
    quartile_summary.to_csv(quartile_csv, index=False)

    plt.figure(figsize=(7, 4.8), dpi=300)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    plt.bar(
        quartile_summary["position_quartile"],
        quartile_summary["share_of_keyword_matched_projects"] * 100,
        color=[colors[0], colors[2], colors[4], colors[6]],
    )
    plt.ylabel("Share of keyword-matched projects (%)", fontsize=12)
    plt.xlabel("Relative position of earliest keyword-method disease mention", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    figure_pdf = plots_dir() / "keyword_position_quartiles_exact_keyword_method.pdf"
    plt.savefig(figure_pdf, bbox_inches="tight")
    plt.close()

    return str(summary_csv), str(quartile_csv), str(figure_pdf)


def _classify_keyword_exact(
    text: str,
    classifications: dict[str, list[str]],
) -> tuple[str, float | None]:
    text_lower = str(text).lower()
    labels = [
        category
        for category, keywords in classifications.items()
        if any(keyword in text_lower for keyword in keywords)
    ]
    matched_keywords = [
        keyword
        for keywords in classifications.values()
        for keyword in keywords
        if keyword in text_lower
    ]
    earliest_pos = None
    for keyword in matched_keywords:
        pos = text_lower.find(keyword)
        if pos != -1 and (earliest_pos is None or pos < earliest_pos):
            earliest_pos = pos
    rel_pos = earliest_pos / max(len(text_lower), 1) if earliest_pos is not None else None
    return (", ".join(labels) if labels else "Other", rel_pos)


def _evaluate_subset(df: pd.DataFrame, group_name: str) -> dict:
    metric_df = df[["ground_truth_label", "llm_classification"]].dropna().copy()
    metric_df = metric_df[
        metric_df["ground_truth_label"].astype(str).str.strip().ne("")
        & metric_df["llm_classification"].astype(str).str.strip().ne("")
    ].copy()
    metric_df["ground_truth_label"] = metric_df["ground_truth_label"].apply(split_labels)
    metric_df["llm_classification"] = metric_df["llm_classification"].apply(split_labels)

    all_labels = get_all_unique_labels(metric_df, ["ground_truth_label", "llm_classification"])
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    y_true = mlb.fit_transform(metric_df["ground_truth_label"])
    y_pred = mlb.transform(metric_df["llm_classification"])
    metrics = calculate_metrics(y_true, y_pred)
    metrics.update(
        {
            "Group": group_name,
            "n_texts": int(len(metric_df)),
            "n_unique_labels": int(len(all_labels)),
        }
    )
    return metrics


def save_validation_position_robustness(validation_df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Save validation performance by keyword-mention position strata."""
    keyword_dict = _load_keyword_dict_exact()
    work = merge_resolved_ground_truth(validation_df, output_col="ground_truth_label")

    classification_results = work["raw_text"].fillna("").astype(str).apply(
        lambda text: _classify_keyword_exact(text, keyword_dict)
    )
    work[["keyword_classification_exact", "earliest_keyword_rel_pos"]] = pd.DataFrame(
        classification_results.tolist(),
        index=work.index,
    )

    matched = work[work["earliest_keyword_rel_pos"].notna()].copy()
    matched["position_group_q1"] = np.where(
        matched["earliest_keyword_rel_pos"] <= 0.25,
        "Early mention (Q1)",
        "Later mention (Q2-Q4)",
    )
    matched["position_group_halves"] = np.where(
        matched["earliest_keyword_rel_pos"] <= 0.50,
        "Early mention (Q1-Q2)",
        "Late mention (Q3-Q4)",
    )

    perf_q1 = pd.DataFrame(
        [
            _evaluate_subset(matched, "All matched texts"),
            _evaluate_subset(matched[matched["position_group_q1"] == "Early mention (Q1)"], "Early mention (Q1)"),
            _evaluate_subset(
                matched[matched["position_group_q1"] == "Later mention (Q2-Q4)"],
                "Later mention (Q2-Q4)",
            ),
        ]
    )[
        [
            "Group",
            "n_texts",
            "Accuracy",
            "Micro-averaged Precision",
            "Micro-averaged Recall",
            "Micro-averaged F1 Score",
            "Macro-averaged Precision",
            "Macro-averaged Recall",
            "Macro-averaged F1 Score",
        ]
    ]
    counts_q1 = (
        matched["position_group_q1"]
        .value_counts()
        .rename_axis("Group")
        .reset_index(name="n_texts")
    )
    counts_q1["share_of_keyword_matched_validation_texts"] = counts_q1["n_texts"] / len(matched)

    perf_halves = pd.DataFrame(
        [
            _evaluate_subset(matched[matched["position_group_halves"] == "Early mention (Q1-Q2)"], "Early mention (Q1-Q2)"),
            _evaluate_subset(matched[matched["position_group_halves"] == "Late mention (Q3-Q4)"], "Late mention (Q3-Q4)"),
        ]
    )[
        [
            "Group",
            "n_texts",
            "Accuracy",
            "Micro-averaged Precision",
            "Micro-averaged Recall",
            "Micro-averaged F1 Score",
            "Macro-averaged Precision",
            "Macro-averaged Recall",
            "Macro-averaged F1 Score",
        ]
    ]
    counts_halves = (
        matched["position_group_halves"]
        .value_counts()
        .rename_axis("Group")
        .reset_index(name="n_texts")
    )
    counts_halves["share_of_keyword_matched_validation_texts"] = counts_halves["n_texts"] / len(matched)

    perf_q1_csv = results_dir() / "llm_validation_performance_q1_vs_q2_q4_by_keyword_position.csv"
    counts_q1_csv = results_dir() / "llm_validation_position_group_counts_q1_vs_q2_q4_by_keyword_position.csv"
    perf_halves_csv = results_dir() / "llm_validation_performance_q1_q2_vs_q3_q4_by_keyword_position.csv"
    counts_halves_csv = results_dir() / "llm_validation_position_group_counts_q1_q2_vs_q3_q4_by_keyword_position.csv"

    perf_q1.to_csv(perf_q1_csv, index=False)
    counts_q1.to_csv(counts_q1_csv, index=False)
    perf_halves.to_csv(perf_halves_csv, index=False)
    counts_halves.to_csv(counts_halves_csv, index=False)

    return (
        str(perf_q1_csv),
        str(counts_q1_csv),
        str(perf_halves_csv),
        str(counts_halves_csv),
    )
