"""Top donor concentration table (Supplementary-style)."""

from __future__ import annotations

import os

import pandas as pd

from revision.paths import results_dir


def donor_concentration_table(df: pd.DataFrame, top_n: int = 35) -> pd.DataFrame:
    donor_sums = df.groupby("DonorName", observed=True)["USD_Disbursement"].sum()
    donor_sums_sorted = donor_sums.sort_values(ascending=False)
    donor_percent = (100 * donor_sums_sorted / donor_sums_sorted.sum()).round(2)
    donor_cum = donor_percent.cumsum().round(2)
    out = pd.DataFrame(
        {
            "DonorName": donor_sums_sorted.index.astype(str),
            "Percent_of_Total": donor_percent.values,
            "Cumulative_Percent_of_Total": donor_cum.values,
        }
    )
    return out.head(top_n)


def save_donor_table(df: pd.DataFrame, top_n: int = 35) -> str:
    table = donor_concentration_table(df, top_n=top_n)
    path = os.path.join(results_dir(), "donor_concentration_top35.csv")
    table.to_csv(path, index=False)
    return path
