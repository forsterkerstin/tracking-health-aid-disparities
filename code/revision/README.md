# Supplementary analyses (revision)

This directory contains code to reproduce the **robustness, validation, and transparency analyses** reported in the revised manuscript (Nature Communications): resolved-label validation, inter-annotator agreement, stability of LLM labels across runs and models, downstream bootstrap uncertainty tables, behaviour on short and non-English text, negation-aware prompting, chunk-level aggregation, validation sample-size sensitivity, input-length representativeness, disease-mention position robustness, 1% subset distribution comparisons, donor concentration, and text exports supporting flow / inclusion summaries.

The implementation is intended for **independent verification** alongside the paper: inputs are tabular exports from your classified aid dataset and manual validation files; outputs are figures under `plots/revision/` and metrics under `results/revision/`.

---

## What the pipeline computes

Running the main entrypoint (below) executes, **when the corresponding input files are present**:

| Theme | Outputs (examples) |
|--------|-------------------|
| Manual validation | Convergence of agreement with sample size; manual vs LLM confusion matrix (PDF) |
| Two human annotators | Cohen’s κ, label-change rate (LCR) with bootstrap CIs; annotator–annotator confusion matrix |
| Repeated / alternative LLM runs | Same agreement metrics and run-to-run confusion matrices on a labelled subsample |
| Downstream ODA shares | Bootstrap-based uncertainty tables for the reproducibility subset and full dataset |
| Short / non-English subset | Micro/macro multilabel metrics and confusion matrix vs manual labels |
| Negation-aware prompt (if column present) | Metrics CSVs comparing manual labels to negation-prompt predictions |
| Chunk-aggregated classification | Metrics vs manual labels when a combined chunk column exists |
| Input length / mention position | Full-vs-validation text-length summary + ECDF; keyword-position summaries and validation metrics |
| Full aid table | Top-donor concentration table; structured text for project-flow / inclusion summaries |

Steps whose inputs are missing are **skipped** and recorded in `results/revision/run_summary.json` under `skipped`, so partial reruns are explicit.

---

## Default workflow (local files only)

**By default**, `python code/run_revision_analysis.py` performs **deterministic analysis on CSVs already on disk**. It does not require network access, API keys, or a GPU. This matches the usual replication setting: classified outputs and validation exports are either produced elsewhere in the project workflow or supplied as part of a data release.

Optional scripts to **regenerate** some of those CSVs using LLM APIs or corpus-scale NLP are documented in **[`OPTIONAL_INPUT_REGENERATION.md`](OPTIONAL_INPUT_REGENERATION.md)**. Use them only when you explicitly enable them (environment variables and credentials as described there).

---

## Quick start

From the **repository root** (after `pip install -r requirements.txt`):

```bash
python code/run_revision_analysis.py
```

Faster smoke test (fewer bootstrap and sensitivity repeats):

```bash
python code/run_revision_analysis.py --quick
```

If your working directory is `code/`:

```bash
python run_revision_analysis.py
```

---

## Input files

Paths are resolved in `paths.py` (including fallbacks). Typical expectations:

| Input | Role |
|--------|------|
| `data/processed/aid_funding_data/final_df.csv` | Donor concentration; project-flow text export |
| `data/revision/resolved_double_annotated_labels.csv` | Canonical manual-validation table with `manual_label`, `manual_label_2`, and resolved `final_label`; used for validation sensitivity, confusion, and annotator agreement |
| `data/revision/robustness_results.csv` | Canonical 1% robustness sample with `llm_classification_1`-`llm_classification_4`; used for run-to-run agreement, confusion matrices, and downstream uncertainty |
| `data/processed/classification/combined_short_df.csv` | Short / non-English subset metrics and confusion |
| `data/df_health_negation_classified_and_labeled.csv` (or path in `paths.py`) | Negation-prompt evaluation CSVs |
| `data/chunk_df_combined_classified_final.csv` (or fallback under `code/visualization/`) | Chunk-aggregated vs manual metrics |
| `data/processed/classification/df_unique_raw_texts.csv` | Corpus-wide disease-mention position summary |
| `data/processed/classification/keywords_used_in_classification.csv` | Exact keyword dictionary for position robustness analyses |

The repository root `.gitignore` excludes **`data/`** and **`plots/`** (and `.DS_Store`). A **`revision/`** directory at the repo root is supported for validation CSVs so that **validation-only** replication is possible without the full processed aid database.

---

## Outputs

After a successful run:

- **`plots/revision/`** — PDF figures (convergence, confusion heatmaps, robustness comparisons, etc.).
- **`results/revision/`** — CSV metrics, `agreement_kappa_lcr.csv`, donor table, `project_flow_sankey.txt`, and **`run_summary.json`** (lists written paths and any skipped steps).

Output directories are created automatically if they do not exist.

---

## Optional: regenerating inputs (LLM and NLP)

To **call external LLM APIs** (e.g. Together) or to **run a corpus-wide NegEx-style scan with spaCy**, follow **[`OPTIONAL_INPUT_REGENERATION.md`](OPTIONAL_INPUT_REGENERATION.md)**. That document specifies:

- which environment variable must be set to opt in;
- where to place API keys (environment variables and `config.json`);
- extra packages for spaCy / NegEx;
- how these jobs relate to the main classification scripts in `code/classification/`.

Those scripts are **inactive unless explicitly enabled**, so default runs and automated checks do not consume API quota or long CPU time.

---

## Module layout

| File | Purpose |
|------|---------|
| `pipeline.py` | Orchestrates analysis steps; CLI via `run_revision_analysis.py` |
| `paths.py` | Repository-relative paths and file fallbacks |
| `agreement.py` | Cohen’s κ and LCR (single-label rows) with bootstrap CIs |
| `sensitivity.py` | Validation sample-size convergence figures |
| `validation.py` | Merge resolved `final_label` ground truth onto validation-style CSVs |
| `downstream.py` | Downstream ODA share uncertainty tables |
| `input_length.py` | Full-vs-validation text-length summary and ECDF |
| `position.py` | Disease-mention position summary and validation robustness tables |
| `subset_distribution.py` | 1% subset versus full-corpus category distribution comparison |
| `confusion.py` | Row-normalized multi-label confusion heatmaps (PDF) |
| `evaluation.py` | Micro/macro multilabel metrics → CSV |
| `donors.py` | Top-donor concentration table |
| `flow_chart.py` | Text export for flow / inclusion summaries (not final figure artwork) |
| `run_llm_regeneration.py` | Optional Together batch classification (see `OPTIONAL_INPUT_REGENERATION.md`) |
| `run_spacy_negation_scan.py` | Optional corpus NegEx scan with spaCy (see `OPTIONAL_INPUT_REGENERATION.md`) |

---

## Repo-root `revision/` folder

You may place `resolved_double_annotated_labels.csv` in **`revision/`** at the repository root (next to `code/`) so validation analyses align with paths in `paths.py` without copying into `data/revision/`. Adjust `paths.py` if you standardise all inputs under `data/` only.
