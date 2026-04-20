"""Bootstrap-style downstream ODA share uncertainty analyses for Revision II."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from revision.labels import STANDARD_DISEASES, canonicalize_label
from revision.paths import results_dir


DOWNSTREAM_CATEGORIES = list(STANDARD_DISEASES) + ["Other"]
_DOWNSTREAM_LOOKUP = {canonicalize_label(label): label for label in DOWNSTREAM_CATEGORIES}
_DOWNSTREAM_INDEX = {label: idx for idx, label in enumerate(DOWNSTREAM_CATEGORIES)}


def parse_downstream_labels(value) -> list[str]:
    """Parse labels and keep only disease categories used in the downstream tables."""
    if pd.isna(value):
        return []
    parts = re.split(r"\s*[;,|]\s*", str(value).strip())
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        label = _DOWNSTREAM_LOOKUP.get(canonicalize_label(part))
        if label and label not in seen:
            out.append(label)
            seen.add(label)
    return out


def _build_allocation_matrix(
    df: pd.DataFrame,
    label_col: str,
    *,
    funding_col: str = "USD_Disbursement",
) -> np.ndarray:
    n = len(df)
    k = len(DOWNSTREAM_CATEGORIES)
    alloc = np.zeros((n, k), dtype=np.float64)

    funding = df[funding_col].fillna(0).to_numpy(dtype=np.float64)
    parsed = df[label_col].apply(parse_downstream_labels).tolist()

    for i, (labels, amount) in enumerate(zip(parsed, funding)):
        if amount <= 0 or not labels:
            continue
        weight = amount / len(labels)
        for label in labels:
            alloc[i, _DOWNSTREAM_INDEX[label]] += weight
    return alloc


def _exact_bootstrap_shares(
    alloc: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    totals = alloc.sum(axis=0)
    shares = np.divide(totals, totals.sum(), out=np.zeros_like(totals), where=totals.sum() != 0)

    boot = np.empty((n_bootstrap, alloc.shape[1]), dtype=np.float64)
    n = alloc.shape[0]
    for idx in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        total_b = counts @ alloc
        boot[idx, :] = np.divide(
            total_b,
            total_b.sum(),
            out=np.zeros_like(total_b),
            where=total_b.sum() != 0,
        )
    return shares, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def save_sample_downstream_uncertainty(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[str]:
    """
    Save downstream ODA share uncertainty tables for available robustness runs.
    """
    run_columns = [
        ("Run 1", "llm_classification_1"),
        ("Run 2", "llm_classification_2"),
        ("Run 3", "llm_classification_3"),
        ("Run 4", "llm_classification_4"),
    ]
    available = [(name, col) for name, col in run_columns if col in df.columns]
    if not available:
        return []

    summary: dict[str, np.ndarray] = {"Disease": np.array(DOWNSTREAM_CATEGORIES, dtype=object)}
    run1_shares = None
    run1_boot = None

    for idx, (name, column) in enumerate(available):
        alloc = _build_allocation_matrix(df[[column, "USD_Disbursement"]].copy(), column)
        shares, low, high = _exact_bootstrap_shares(alloc, n_bootstrap=n_bootstrap, seed=seed + idx)
        summary[f"{name} (%)"] = 100 * shares
        summary[f"{name} UI low (%)"] = 100 * low
        summary[f"{name} UI high (%)"] = 100 * high
        if name == "Run 1":
            run1_shares = shares
            run1_boot = np.stack([low, high], axis=1)

    summary_df = pd.DataFrame(summary)
    if "Run 1 (%)" in summary_df.columns:
        summary_df = summary_df.sort_values("Run 1 (%)", ascending=False)
    other_row = summary_df[summary_df["Disease"] == "Other"]
    non_other_rows = summary_df[summary_df["Disease"] != "Other"]
    summary_df = pd.concat([non_other_rows, other_row], ignore_index=True)

    out_csv = results_dir() / "downstream_oda_shares_runs_1_to_4_with_uncertainty.csv"
    summary_df.to_csv(out_csv, index=False)

    keep_cols = ["Disease"]
    if "Run 1 (%)" in summary_df.columns:
        keep_cols.extend(["Run 1 (%)", "Run 1 UI low (%)", "Run 1 UI high (%)"])
    for name, _ in available[1:]:
        keep_cols.append(f"{name} (%)")

    table_df = summary_df[keep_cols].copy()
    if "Run 1 (%)" in table_df.columns:
        table_df["Run 1"] = table_df["Run 1 (%)"].map(lambda x: f"{x:.2f}")
        table_df["Run 1 95% UI"] = table_df.apply(
            lambda row: f"[{row['Run 1 UI low (%)']:.2f}, {row['Run 1 UI high (%)']:.2f}]",
            axis=1,
        )
        table_df = table_df.drop(columns=["Run 1 (%)", "Run 1 UI low (%)", "Run 1 UI high (%)"])
        ordered_cols = ["Disease", "Run 1", "Run 1 95% UI"]
    else:
        ordered_cols = ["Disease"]
    for name, _ in available[1:]:
        src = f"{name} (%)"
        dst = name
        table_df[dst] = table_df[src].map(lambda x: f"{x:.2f}")
        table_df = table_df.drop(columns=[src])
        ordered_cols.append(dst)
    table_df = table_df[ordered_cols]

    tex = table_df.to_latex(
        index=False,
        escape=True,
        caption=(
            "Disease-specific ODA shares (\\%) across the main specification and sensitivity analyses. "
            "Run 1 is the main specification and is reported with bootstrap-based 95\\% uncertainty "
            "intervals; Runs 2--4 are sensitivity analyses based on an alternative seed or alternative "
            "LLM classifications."
        ),
        label="tab:disease_share_uncertainty_runs_1_4",
    )
    out_tex = results_dir() / "table_disease_share_uncertainty_runs_1_4.tex"
    out_tex.write_text(tex, encoding="utf-8")

    return [str(out_csv), str(out_tex)]


def save_full_dataset_downstream_uncertainty(
    final_data_path: Path,
    *,
    label_col: str = "llm_classification",
    funding_col: str = "USD_Disbursement",
    n_bootstrap: int = 120,
    seed: int = 42,
    chunksize: int = 4000,
    bootstrap_batch_size: int = 20,
) -> list[str]:
    """
    Save the full-dataset downstream uncertainty table.

    Notes
    -----
    The notebook materialized a project-by-disease allocation matrix for the full corpus, which
    is not feasible for the public 3.7M-row release. Here we compute the same downstream
    contribution totals in a streaming pass and use chunked Poisson bootstrap weights to obtain
    bootstrap-style uncertainty intervals without loading the full matrix into memory.
    """
    total = np.zeros(len(DOWNSTREAM_CATEGORIES), dtype=np.float64)
    boot_totals = np.zeros((n_bootstrap, len(DOWNSTREAM_CATEGORIES)), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for chunk in pd.read_csv(
        final_data_path,
        usecols=[label_col, funding_col],
        chunksize=chunksize,
        low_memory=False,
    ):
        alloc = _build_allocation_matrix(chunk, label_col, funding_col=funding_col)
        total += alloc.sum(axis=0)

        for start in range(0, n_bootstrap, bootstrap_batch_size):
            end = min(start + bootstrap_batch_size, n_bootstrap)
            weights = rng.poisson(1.0, size=(end - start, alloc.shape[0])).astype(np.float32)
            boot_totals[start:end, :] += weights @ alloc

    other_idx = _DOWNSTREAM_INDEX["Other"]
    named_mask = np.arange(len(DOWNSTREAM_CATEGORIES)) != other_idx
    share = np.zeros_like(total)
    denom = total[named_mask].sum()
    share[named_mask] = np.divide(
        total[named_mask],
        denom,
        out=np.zeros(named_mask.sum(), dtype=np.float64),
        where=denom != 0,
    )

    boot_share = np.zeros_like(boot_totals)
    boot_denom = boot_totals[:, named_mask].sum(axis=1)
    boot_share[:, named_mask] = np.divide(
        boot_totals[:, named_mask],
        boot_denom[:, None],
        out=np.zeros((n_bootstrap, named_mask.sum()), dtype=np.float64),
        where=boot_denom[:, None] != 0,
    )

    out_df = pd.DataFrame(
        {
            "Disease": DOWNSTREAM_CATEGORIES,
            "Total ODA (USD bn)": total / 1000.0,
            "Total ODA UI low (USD bn)": np.percentile(boot_totals, 2.5, axis=0) / 1000.0,
            "Total ODA UI high (USD bn)": np.percentile(boot_totals, 97.5, axis=0) / 1000.0,
            "Share (%)": 100 * share,
            "Share UI low (%)": 100 * np.percentile(boot_share, 2.5, axis=0),
            "Share UI high (%)": 100 * np.percentile(boot_share, 97.5, axis=0),
        }
    ).sort_values("Share (%)", ascending=False)

    other_row = out_df[out_df["Disease"] == "Other"]
    non_other_rows = out_df[out_df["Disease"] != "Other"]
    out_df = pd.concat([non_other_rows, other_row], ignore_index=True)

    out_csv = results_dir() / "downstream_oda_shares_full_dataset_with_uncertainty.csv"
    out_df.to_csv(out_csv, index=False)

    table_df = out_df.copy()
    table_df["ODA share (%)"] = table_df["Share (%)"].map(lambda x: f"{x:.2f}")
    table_df["95% UI"] = table_df.apply(
        lambda row: f"[{row['Share UI low (%)']:.2f}, {row['Share UI high (%)']:.2f}]",
        axis=1,
    )
    table_df = table_df[["Disease", "ODA share (%)", "95% UI"]]

    tex = table_df.to_latex(
        index=False,
        escape=True,
        caption=(
            "Disease-specific downstream ODA share estimates from the full dataset main classification, "
            "reported with bootstrap-based 95\\% uncertainty intervals."
        ),
        label="tab:full_dataset_downstream_uncertainty",
    )
    out_tex = results_dir() / "table_full_dataset_downstream_uncertainty.tex"
    out_tex.write_text(tex, encoding="utf-8")

    return [str(out_csv), str(out_tex)]
