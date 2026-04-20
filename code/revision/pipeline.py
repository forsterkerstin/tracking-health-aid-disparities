"""Single entry point: run all revision supplementary analyses that do not require API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from revision.agreement import compute_kappa_lcr
from revision.confusion import (
    create_confusion_matrix_between_columns,
    save_pairwise_confusion_matrices,
)
from revision.downstream import (
    save_full_dataset_downstream_uncertainty,
    save_sample_downstream_uncertainty,
)
from revision.donors import save_donor_table
from revision.evaluation import evaluate_multilabel
from revision.flow_chart import export_project_flow_text
from revision.input_length import save_input_length_analysis
from revision.labels import extract_model_labels
from revision.paths import (
    chunk_combined_csv,
    ensure_revision_output_layout,
    final_aid_df_csv,
    negation_labeled_csv,
    plots_dir,
    results_dir,
    resolved_validation_csv,
    robustness_results_csv,
    robustness_short_non_en_csv,
    validation_data_dir,
)
from revision.position import (
    save_keyword_position_corpus_summary,
    save_validation_position_robustness,
)
from revision.sensitivity import convergence_category_variance, convergence_overall_accuracy
from revision.subset_distribution import compare_subset_distribution
from revision.validation import merge_resolved_ground_truth


def _ensure_code_on_path() -> Path:
    """Allow `import revision` when running as script from repo root."""
    root = Path(__file__).resolve().parent.parent.parent
    code_dir = root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    return root


def _load_robustness_sample() -> pd.DataFrame | None:
    """Load the canonical robustness sample, normalizing the DeepSeek column when present."""
    path = robustness_results_csv()
    if path is None:
        return None

    df = pd.read_csv(path, low_memory=False)

    if "llm_classification_4" not in df.columns and "llm_classification_new" in df.columns:
        df["llm_classification_4"] = df["llm_classification_new"]

    if "llm_classification_4" in df.columns:
        df["llm_classification_4"] = df["llm_classification_4"].apply(extract_model_labels)

    return df


def run_revision_analysis(*, quick: bool = False, verbose: bool = True) -> dict:
    """
    Run all analyses that only need local CSVs under the repository.

    Parameters
    ----------
    quick
        If True, use fewer bootstrap/sensitivity repeats for faster smoke tests.

    Returns
    -------
    dict
        Short summary of outputs written (paths and row counts).
    """
    _ensure_code_on_path()
    ensure_revision_output_layout()
    validation_data_dir()
    summary: dict = {"plots": [], "results": [], "skipped": []}

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    n_rep_sens = 12 if quick else 50
    n_cat_sens = 10 if quick else 120
    num_sizes = 5 if quick else 10
    n_boot = 80 if quick else 1000
    n_boot_sample_downstream = 120 if quick else 1000
    n_boot_full_downstream = 12 if quick else 80

    # --- 1) Manual validation: sensitivity / convergence ---
    log("Step 1/8: validation sample sensitivity with resolved ground truth...")
    man_path = resolved_validation_csv()
    if man_path is None:
        summary["skipped"].append("resolved_validation_csv (not found)")
    else:
        manual_df = pd.read_csv(man_path)
        if "final_label" not in manual_df.columns:
            raise ValueError(f"Resolved validation file {man_path} must contain 'final_label'.")
        manual_df = manual_df.rename(columns={"final_label": "ground_truth_label"})
        manual_df = manual_df[
            manual_df["ground_truth_label"].notna()
            & (manual_df["ground_truth_label"].astype(str).str.strip() != "")
        ]
        rng = np.random.default_rng(42)
        overall_df = convergence_overall_accuracy(
            manual_df,
            col_pred="llm_classification",
            col_true="ground_truth_label",
            n_repeats=n_rep_sens,
            num_sizes=num_sizes,
            rng=rng,
        )
        cat_df = convergence_category_variance(
            manual_df,
            col_pred="llm_classification",
            col_true="ground_truth_label",
            n_repeats=n_cat_sens,
            num_sizes=num_sizes,
            rng=np.random.default_rng(43),
        )
        pdir = plots_dir()
        summary["plots"].extend(
            [
                str(pdir / "overall_accuracy_convergence.pdf"),
                str(pdir / "category_specific_variance_in_accuracy.pdf"),
            ]
        )
        overall_df.to_csv(results_dir() / "sensitivity_overall_accuracy_by_n.csv", index=False)
        cat_df.to_csv(results_dir() / "sensitivity_category_variance_long.csv", index=False)
        summary["results"].append(str(results_dir() / "sensitivity_overall_accuracy_by_n.csv"))

        # Category-level manual vs LLM (aligns with supplementary confusion-matrix for validation performance)
        cm_val = plots_dir() / "validation_manual_vs_llm.pdf"
        create_confusion_matrix_between_columns(
            manual_df,
            "ground_truth_label",
            "llm_classification",
            str(cm_val),
            col1_desc="Resolved validation categories",
            col2_desc="LLM classification",
        )
        summary["plots"].append(str(cm_val))

    # --- 2) Inter-annotator + LLM stability (1% / 2% sample) ---
    log("Step 2/8: Cohen's kappa / LCR (annotators + LLM runs)...")
    agreement_rows = []
    dbl = resolved_validation_csv()
    if dbl is not None:
        ddf = pd.read_csv(dbl, low_memory=False)
        if {"manual_label", "manual_label_2"}.issubset(ddf.columns):
            r = compute_kappa_lcr(
                ddf, "manual_label", "manual_label_2", n_bootstrap=n_boot, verbose=False
            )
            r["comparison"] = "manual_vs_manual"
            agreement_rows.append(r)
            cms_ann = save_pairwise_confusion_matrices(ddf, "double_annotation")
            summary["plots"].extend(cms_ann)

    sdf = _load_robustness_sample()
    if sdf is not None:
        if {"llm_classification_1", "llm_classification_2"}.issubset(sdf.columns):
            r = compute_kappa_lcr(
                sdf, "llm_classification_1", "llm_classification_2", n_bootstrap=n_boot, verbose=False
            )
            r["comparison"] = "llm1_vs_llm2_same_architecture"
            agreement_rows.append(r)
        if {"llm_classification_1", "llm_classification_3"}.issubset(sdf.columns):
            r = compute_kappa_lcr(
                sdf, "llm_classification_1", "llm_classification_3", n_bootstrap=n_boot, verbose=False
            )
            r["comparison"] = "llm1_vs_llm3_alternative_model"
            agreement_rows.append(r)
        if {"llm_classification_1", "llm_classification_4"}.issubset(sdf.columns):
            r = compute_kappa_lcr(
                sdf, "llm_classification_1", "llm_classification_4", n_bootstrap=n_boot, verbose=False
            )
            r["comparison"] = "llm1_vs_llm4_deepseek_benchmark"
            agreement_rows.append(r)

        cms = save_pairwise_confusion_matrices(sdf, "robustness_sample")
        summary["plots"].extend(cms)
    else:
        summary["skipped"].append("robustness_results_csv (not found)")

    if agreement_rows:
        agr_df = pd.DataFrame(agreement_rows)
        # Flatten tuples for clean CSV (avoid np.float64 / tuple repr in cells)
        agr_df["kappa_ci_lower"] = agr_df["kappa_ci"].apply(lambda t: t[0])
        agr_df["kappa_ci_upper"] = agr_df["kappa_ci"].apply(lambda t: t[1])
        agr_df["lcr_ci_lower"] = agr_df["lcr_ci"].apply(lambda t: t[0])
        agr_df["lcr_ci_upper"] = agr_df["lcr_ci"].apply(lambda t: t[1])
        agr_df = agr_df.drop(columns=["kappa_ci", "lcr_ci"])
        out_a = results_dir() / "agreement_kappa_lcr.csv"
        agr_df.to_csv(out_a, index=False)
        summary["results"].append(str(out_a))

    # --- 3) Reproducibility subset distribution + downstream uncertainty ---
    log("Step 3/8: 1% subset distribution and downstream ODA-share uncertainty...")
    fin = final_aid_df_csv()
    aid = None
    if fin is not None:
        aid = pd.read_csv(fin, low_memory=False)

    if sdf is not None and aid is not None:
        subset_dist = compare_subset_distribution(sdf, aid)
        summary["results"].append(
            str(results_dir() / "subset_distribution_comparison_named_diseases.csv")
        )
    elif sdf is None:
        summary["skipped"].append("subset distribution comparison (robustness sample missing)")
    else:
        summary["skipped"].append("subset distribution comparison (final_df.csv missing)")

    if sdf is not None and "USD_Disbursement" in sdf.columns:
        summary["results"].extend(
            save_sample_downstream_uncertainty(
                sdf,
                n_bootstrap=n_boot_sample_downstream,
                seed=42,
            )
        )
    else:
        summary["skipped"].append("sample downstream uncertainty (robustness sample missing funding)")

    if fin is not None:
        summary["results"].extend(
            save_full_dataset_downstream_uncertainty(
                fin,
                n_bootstrap=n_boot_full_downstream,
                seed=42,
            )
        )
    else:
        summary["skipped"].append("full downstream uncertainty (final_df.csv not found)")

    # --- 4) Short / non-English subset metrics ---
    log("Step 4/8: short / non-English subset metrics...")
    sh = robustness_short_non_en_csv()
    if sh is not None:
        shdf = pd.read_csv(sh, low_memory=False)
        if {"manual_label", "llm_classification_1"}.issubset(shdf.columns):
            evaluate_multilabel(
                shdf,
                "manual_label",
                "llm_classification_1",
                model_name="llm_classification_1",
                csv_prefix="short_non_en_subset",
            )
            cm_short = plots_dir() / "short_non_en_manual_vs_llm1.pdf"
            create_confusion_matrix_between_columns(
                shdf,
                "manual_label",
                "llm_classification_1",
                str(cm_short),
                col1_desc="Manually labelled categories",
                col2_desc="LLM classification (run 1)",
            )
            summary["plots"].append(str(cm_short))
            summary["results"].extend(
                [
                    str(results_dir() / "overall_metrics_short_non_en_subset.csv"),
                    str(results_dir() / "per_class_metrics_short_non_en_subset.csv"),
                ]
            )
    else:
        summary["skipped"].append("combined_short_df (not found)")

    # --- 5) Negation prompt subset ---
    log("Step 5/8: negation-aware prompt evaluation...")
    neg = negation_labeled_csv()
    if neg is not None:
        ndf = pd.read_csv(neg, low_memory=False)
        ndf = merge_resolved_ground_truth(ndf, output_col="ground_truth_label")
        if {"ground_truth_label", "llm_classification_negation"}.issubset(ndf.columns):
            evaluate_multilabel(
                ndf,
                "ground_truth_label",
                "llm_classification_negation",
                model_name="negation_prompt",
                csv_prefix="negation_prompt",
            )
            summary["results"].extend(
                [
                    str(results_dir() / "overall_metrics_negation_prompt.csv"),
                    str(results_dir() / "per_class_metrics_negation_prompt.csv"),
                ]
            )
        if {"ground_truth_label", "llm_classification"}.issubset(ndf.columns):
            evaluate_multilabel(
                ndf,
                "ground_truth_label",
                "llm_classification",
                model_name="baseline_prompt",
                csv_prefix="negation_subset_baseline",
            )
            summary["results"].extend(
                [
                    str(results_dir() / "overall_metrics_negation_subset_baseline.csv"),
                    str(results_dir() / "per_class_metrics_negation_subset_baseline.csv"),
                ]
            )
    else:
        summary["skipped"].append("negation_labeled_csv (not found)")

    # --- 6) Chunk-based aggregation vs manual labels ---
    log("Step 6/8: chunk-aggregated classification...")
    ch = chunk_combined_csv()
    if ch is not None:
        cdf = pd.read_csv(ch, low_memory=False)
        cdf = merge_resolved_ground_truth(cdf, output_col="ground_truth_label")
        if {"ground_truth_label", "combined_classification"}.issubset(cdf.columns):
            evaluate_multilabel(
                cdf,
                "ground_truth_label",
                "combined_classification",
                model_name="chunk_combined",
                csv_prefix="chunk_combined",
            )
            summary["results"].extend(
                [
                    str(results_dir() / "overall_metrics_chunk_combined.csv"),
                    str(results_dir() / "per_class_metrics_chunk_combined.csv"),
                ]
            )
    else:
        summary["skipped"].append("chunk_combined_csv (not found)")

    # --- 7) Input-length and mention-position robustness ---
    log("Step 7/8: input-length and mention-position robustness...")
    if fin is not None and dbl is not None:
        summary_csv, figure_pdf = save_input_length_analysis(fin, dbl)
        summary["results"].append(summary_csv)
        summary["plots"].append(figure_pdf)
    else:
        summary["skipped"].append("input-length analysis (final_df or double annotations missing)")

    if dbl is not None:
        validation_df = pd.read_csv(dbl, low_memory=False)
        try:
            pos_summary_csv, pos_quartile_csv, pos_plot_pdf = save_keyword_position_corpus_summary()
            summary["results"].extend([pos_summary_csv, pos_quartile_csv])
            summary["plots"].append(pos_plot_pdf)
        except FileNotFoundError:
            summary["skipped"].append("keyword-position corpus summary (keyword inputs missing)")

        try:
            summary["results"].extend(save_validation_position_robustness(validation_df))
        except FileNotFoundError:
            summary["skipped"].append("keyword-position validation robustness (keyword inputs missing)")
    else:
        summary["skipped"].append("keyword-position analyses (double annotations missing)")

    # --- 8) Donor concentration + flow text from final_df ---
    log("Step 8/8: donor table + project flow text...")
    if aid is not None:
        dpath = save_donor_table(aid, top_n=35)
        summary["results"].append(dpath)
        fpath = export_project_flow_text(aid)
        summary["results"].append(fpath)
    else:
        summary["skipped"].append("final_df.csv (not found)")

    meta = results_dir() / "run_summary.json"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["results"].append(str(meta))

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run revision supplementary analyses (local data only).")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster run with fewer bootstrap/sensitivity repeats (smoke test).",
    )
    args = parser.parse_args()
    s = run_revision_analysis(quick=args.quick)
    print("Revision analysis finished.")
    print("Plots:", len(s.get("plots", [])))
    print("Result files:", len(s.get("results", [])))
    if s.get("skipped"):
        print("Skipped (missing inputs):", "; ".join(s["skipped"]))
    print(f"Details: {results_dir() / 'run_summary.json'}")


if __name__ == "__main__":
    main()
