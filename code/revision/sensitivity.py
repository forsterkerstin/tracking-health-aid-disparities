"""Validation sample size sensitivity (convergence) plots."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from revision.labels import build_label_display_map, parse_label_string
from revision.paths import plots_dir


def multilabel_jaccard_score(true_labels, pred_labels):
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    union = true_set | pred_set
    if not union:
        return 1.0
    return len(true_set & pred_set) / len(union)


def convergence_overall_accuracy(
    df: pd.DataFrame,
    col_pred: str = "llm_classification",
    col_true: str = "manual_label",
    n_repeats: int = 50,
    num_sizes: int = 10,
    out_pdf: str | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(42)
    sizes = np.linspace(100, len(df), num=max(2, num_sizes), dtype=int)
    results = []

    parsed_df = df.copy()
    parsed_df[col_true] = parsed_df[col_true].apply(parse_label_string)
    parsed_df[col_pred] = parsed_df[col_pred].apply(parse_label_string)

    for n in sizes:
        accs = []
        for _ in range(n_repeats):
            sample = parsed_df.sample(n, random_state=int(rng.integers(0, 10_000)))
            sample_scores = [
                multilabel_jaccard_score(t, p)
                for t, p in zip(sample[col_true], sample[col_pred])
            ]
            accs.append(float(np.mean(sample_scores)))
        results.append({"n": int(n), "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs))})

    res_df = pd.DataFrame(results)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(res_df["n"], res_df["mean_acc"], marker="o", label="Mean accuracy")
    plt.fill_between(
        res_df["n"],
        res_df["mean_acc"] - res_df["std_acc"],
        res_df["mean_acc"] + res_df["std_acc"],
        alpha=0.2,
    )
    plt.xlabel("Number of validation samples", fontsize=14, labelpad=10)
    plt.ylabel("Overall accuracy (mean Jaccard)", fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    out = out_pdf or os.path.join(plots_dir(), "overall_accuracy_convergence.pdf")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    return res_df


DISEASE_ABBREV = {
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


def convergence_category_variance(
    df: pd.DataFrame,
    col_pred: str = "llm_classification",
    col_true: str = "manual_label",
    n_repeats: int = 200,
    num_sizes: int = 10,
    out_pdf: str | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(43)
    parsed_df = df.copy()
    parsed_df[col_true] = parsed_df[col_true].apply(parse_label_string)
    parsed_df[col_pred] = parsed_df[col_pred].apply(parse_label_string)

    all_labels = sorted({lbl for labels in parsed_df[col_true] for lbl in labels})
    display_map = build_label_display_map([df[col_true], df[col_pred]])

    max_n = min(len(parsed_df), int(0.8 * len(parsed_df)))
    sizes = np.linspace(50, max_n, num=max(2, num_sizes), dtype=int)
    rows = []

    for n in sizes:
        per_label_accs = {label: [] for label in all_labels}
        per_label_counts = {label: [] for label in all_labels}

        for _ in range(n_repeats):
            sample = parsed_df.sample(n, random_state=int(rng.integers(0, 10_000)))
            exploded = sample.explode(col_true)
            exploded = exploded[exploded[col_true].notna() & (exploded[col_true] != "")]
            if exploded.empty:
                continue
            for label, group in exploded.groupby(col_true):
                contains_label = group[col_pred].apply(lambda preds, lbl=label: lbl in preds)
                per_label_accs[label].append(float(contains_label.mean()))
                per_label_counts[label].append(len(group))

        for label in all_labels:
            accs = per_label_accs[label]
            counts = per_label_counts[label]
            if not accs:
                continue
            rows.append(
                {
                    "label": label,
                    "label_display": DISEASE_ABBREV.get(
                        display_map.get(label, label.title()),
                        display_map.get(label, label.title()),
                    ),
                    "n_total": int(n),
                    "mean_samples_for_category": float(np.mean(counts)),
                    "var_accuracy_for_category": float(np.var(accs)),
                    "mean_accuracy_for_category": float(np.mean(accs)),
                }
            )

    res_df = pd.DataFrame(rows)
    labels_with_zero_var = []
    labels_with_nonzero_var = []
    for label in all_labels:
        subset = res_df[res_df["label"] == label]
        if subset.empty:
            continue
        if (subset["var_accuracy_for_category"] == 0).all():
            labels_with_zero_var.append(label)
        else:
            labels_with_nonzero_var.append(label)

    ordered = labels_with_nonzero_var + labels_with_zero_var
    normalized_data = []
    for label in ordered:
        subset = res_df[res_df["label"] == label].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("mean_samples_for_category")
        max_samples = subset["mean_samples_for_category"].max()
        min_samples = subset["mean_samples_for_category"].min()
        if max_samples > min_samples:
            subset = subset.assign(
                x_normalized=(
                    (subset["mean_samples_for_category"] - min_samples)
                    / (max_samples - min_samples)
                )
                * 100
            )
        else:
            subset = subset.assign(x_normalized=0)

        max_var = subset["var_accuracy_for_category"].max()
        min_var = subset["var_accuracy_for_category"].min()
        if max_var > min_var:
            subset = subset.assign(
                y_normalized=(
                    (subset["var_accuracy_for_category"] - min_var) / (max_var - min_var)
                )
                * 100
            )
        elif max_var == min_var and max_var > 0:
            subset = subset.assign(y_normalized=100)
        else:
            subset = subset.assign(y_normalized=0)

        normalized_data.append(subset)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(normalized_data), 1)))

    for idx, subset in enumerate(normalized_data):
        label = subset["label"].iloc[0]
        display_name = display_map.get(label, label.title())
        label_display = DISEASE_ABBREV.get(display_name, display_name)
        if (subset["var_accuracy_for_category"] == 0).all():
            linestyle, alpha = "--", 0.6
        else:
            linestyle, alpha = "-", 0.8
        ax.plot(
            subset["x_normalized"],
            subset["y_normalized"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[idx % len(colors)],
            label=label_display,
            linestyle=linestyle,
            alpha=alpha,
        )

    ax.set_xlabel("Percent of category samples (normalized)", fontsize=20)
    ax.set_ylabel("Percent of variance (normalized)", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    out = out_pdf or os.path.join(plots_dir(), "category_specific_variance_in_accuracy.pdf")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    return res_df
