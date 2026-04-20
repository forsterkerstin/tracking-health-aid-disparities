"""Input-length representativeness analysis for the validation sample."""

from __future__ import annotations

from array import array
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.paths import plots_dir, results_dir


def _word_counts_from_series(series: pd.Series) -> np.ndarray:
    texts = series.fillna("").astype(str).str.strip()
    texts = texts[texts != ""]
    return texts.str.split().str.len().to_numpy(dtype=np.uint32)


def _word_counts_from_csv(
    path: Path,
    *,
    text_col: str = "raw_text",
    chunksize: int = 50_000,
) -> np.ndarray:
    counts = array("I")
    for chunk in pd.read_csv(path, usecols=[text_col], chunksize=chunksize, low_memory=False):
        counts.extend(_word_counts_from_series(chunk[text_col]).tolist())
    return np.array(counts, dtype=np.uint32)


def _summary_row(lengths: np.ndarray, dataset_name: str) -> dict:
    q1, median, q3, p95 = np.percentile(lengths, [25, 50, 75, 95])
    return {
        "Dataset": dataset_name,
        "n": int(lengths.size),
        "Mean": round(float(lengths.mean()), 1),
        "Median": int(round(median)),
        "IQR": f"{int(round(q1))}-{int(round(q3))}",
        "P95": int(round(p95)),
    }


def _ecdf(values: np.ndarray, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    n = values.size
    if n <= max_points:
        y = np.arange(1, n + 1) / n
        return values, y

    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return values[idx], (idx + 1) / n


def save_input_length_analysis(
    full_data_path: Path,
    validation_data_path: Path,
) -> tuple[str, str]:
    """Write the input-length summary CSV and ECDF plot PDF."""
    full_word_counts = _word_counts_from_csv(full_data_path)
    validation_df = pd.read_csv(validation_data_path, usecols=["raw_text"], low_memory=False)
    validation_word_counts = _word_counts_from_series(validation_df["raw_text"])

    summary_df = pd.DataFrame(
        [
            _summary_row(full_word_counts, "Full dataset"),
            _summary_row(validation_word_counts, "Validation set"),
        ]
    )
    summary_csv = results_dir() / "input_length_summary_full_vs_validation.csv"
    summary_df.to_csv(summary_csv, index=False)

    x_cap = int(
        np.ceil(
            max(
                np.percentile(full_word_counts, 99),
                np.percentile(validation_word_counts, 99),
            )
        )
    )
    full_x, full_y = _ecdf(np.clip(full_word_counts, None, x_cap))
    val_x, val_y = _ecdf(np.clip(validation_word_counts, None, x_cap))

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    ax.step(full_x, full_y, where="post", linewidth=1.7, label="Full dataset", color=colors[0])
    ax.step(val_x, val_y, where="post", linewidth=1.7, label="Validation set", color=colors[2])
    ax.set_xlabel("Project description length (word count)", fontsize=8, labelpad=6)
    ax.set_ylabel("Cumulative proportion of projects", fontsize=8, labelpad=6)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, x_cap)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    figure_pdf = plots_dir() / "input_length_full_vs_validation.pdf"
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    return str(summary_csv), str(figure_pdf)
