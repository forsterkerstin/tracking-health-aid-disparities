"""Project inclusion flow text export, no Plotly required."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from revision.paths import results_dir

DISEASE_CATEGORIES = {
    "CMNNDs": [
        "HIV/AIDS and sexually transmitted infections",
        "Respiratory infections and tuberculosis",
        "Enteric infections",
        "Neglected tropical diseases and malaria",
        "Maternal and neonatal disorders",
        "Nutritional deficiencies",
    ],
    "NCDs": [
        "Neoplasms",
        "Cardiovascular diseases",
        "Chronic respiratory diseases",
        "Digestive diseases",
        "Neurological disorders",
        "Mental disorders",
        "Substance use disorders",
        "Diabetes and kidney diseases",
        "Skin and subcutaneous diseases",
        "Sense organ diseases",
        "Musculoskeletal disorders",
    ],
}

DISEASE_COL = "llm_classification"
TOTAL_PROJECTS_RAW = 4_700_000


def export_project_flow_text(
    final_df: pd.DataFrame,
    *,
    total_projects: int = TOTAL_PROJECTS_RAW,
    output_path: str | Path | None = None,
) -> str:
    """
    Write a line-oriented flow description from ``final_df`` (same logic as exploratory
    Sankey scripts, without generating an interactive figure).
    """
    projects_in_final = len(final_df)
    projects_missing = total_projects - projects_in_final

    cmnnd_set = set(DISEASE_CATEGORIES["CMNNDs"])
    ncd_set = set(DISEASE_CATEGORIES["NCDs"])

    s = final_df[DISEASE_COL].astype(str)
    cmnnd_count = s.isin(cmnnd_set).sum()
    ncd_count = s.isin(ncd_set).sum()
    other_count = projects_in_final - (cmnnd_count + ncd_count)

    lines: list[str] = []
    lines.append("// Enter Flows between Nodes, like this:")
    lines.append("//         Source [AMOUNT] Target")
    lines.append("")
    lines.append(f"Total Projects [{projects_in_final}] In final_df")
    lines.append(f"Total Projects [{projects_missing}] Missing from final_df")
    lines.append("")
    lines.append(f"In final_df [{cmnnd_count}] CMNNDs")
    lines.append(f"In final_df [{ncd_count}] NCDs")
    if other_count > 0:
        lines.append(f"In final_df [{other_count}] Other")
    lines.append("")

    cmnnd_children = []
    for disease in DISEASE_CATEGORIES["CMNNDs"]:
        disease_count = (s == disease).sum()
        cmnnd_children.append((disease, int(disease_count)))
    cmnnd_children.sort(key=lambda x: x[1], reverse=True)

    ncd_children = []
    for disease in DISEASE_CATEGORIES["NCDs"]:
        disease_count = (s == disease).sum()
        ncd_children.append((disease, int(disease_count)))
    ncd_children.sort(key=lambda x: x[1], reverse=True)

    for disease, disease_count in cmnnd_children:
        if disease_count > 0:
            lines.append(f"CMNNDs [{disease_count}] {disease}")
    if cmnnd_children:
        lines.append("")

    for disease, disease_count in ncd_children:
        if disease_count > 0:
            lines.append(f"NCDs [{disease_count}] {disease}")

    lines.append("")
    lines.append(f"// Parameters: total_projects_in_CRS={total_projects}, rows_in_final_df={projects_in_final}")

    out = Path(output_path) if output_path else results_dir() / "project_flow_sankey.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
