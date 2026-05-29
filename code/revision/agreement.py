"""Cohen's kappa and label consistency rate (LCR) for paired label columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def compute_kappa_lcr(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    n_bootstrap: int = 1000,
    verbose: bool = False,
) -> dict:
    """
    Cohen's kappa and LCR on rows where both columns are single non-empty labels
    (no comma-separated multi-labels), matching the manuscript supplementary logic.
    """
    labels_df = df[[col1, col2]].dropna(subset=[col1, col2])
    labels_df = labels_df[
        (labels_df[col1].astype(str).str.strip() != "")
        & (labels_df[col2].astype(str).str.strip() != "")
    ]
    single = labels_df[
        (~labels_df[col1].astype(str).str.contains(",", regex=False))
        & (~labels_df[col2].astype(str).str.contains(",", regex=False))
    ].copy()

    y_pred = single[col1].astype(str).str.strip().values
    y_true = single[col2].astype(str).str.strip().values

    if len(y_pred) == 0:
        return {
            "kappa": float("nan"),
            "kappa_ci": (float("nan"), float("nan")),
            "lcr": float("nan"),
            "lcr_ci": (float("nan"), float("nan")),
            "n": 0,
            "n_matches": 0,
        }

    kappa = float(cohen_kappa_score(y_pred, y_true))
    bootstrap_kappas = []
    n_items = len(y_pred)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(n_items, size=n_items, replace=True)
        bootstrap_kappas.append(
            float(cohen_kappa_score(y_pred[indices], y_true[indices]))
        )
    kappa_ci_lower = float(np.percentile(bootstrap_kappas, 2.5))
    kappa_ci_upper = float(np.percentile(bootstrap_kappas, 97.5))

    lcr = float((y_pred == y_true).mean())
    n_matches = int((y_pred == y_true).sum())
    n = n_items

    z = stats.norm.ppf(0.975)
    p = lcr
    denominator = 1 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    lcr_ci_lower = float(max(0.0, center - margin))
    lcr_ci_upper = float(min(1.0, center + margin))

    if verbose:
        print(
            f"Cohen's kappa: {kappa:.3f} (95% CI: [{kappa_ci_lower:.3f}, {kappa_ci_upper:.3f}])"
        )
        print(
            f"Label consistency rate (LCR): {lcr * 100:.2f}% "
            f"(95% CI: [{lcr_ci_lower * 100:.2f}%, {lcr_ci_upper * 100:.2f}%])  n={n}"
        )

    return {
        "kappa": kappa,
        "kappa_ci": (kappa_ci_lower, kappa_ci_upper),
        "lcr": lcr,
        "lcr_ci": (lcr_ci_lower, lcr_ci_upper),
        "n": n,
        "n_matches": n_matches,
    }
