#!/usr/bin/env python3
"""
Run all revision supplementary analyses (local CSVs only, no API calls).

From the **repository root**::

    python code/run_revision_analysis.py

Quick smoke test::

    python code/run_revision_analysis.py --quick

Optional paid / heavy steps (LLM APIs, spaCy corpus scan) are **not** run here.
They live under ``code/revision/`` and are **off unless you set env vars** — see
``code/revision/OPTIONAL_INPUT_REGENERATION.md`` and ``run_llm_regeneration.py`` /
``run_spacy_negation_scan.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives in ``code/``; the ``revision`` package is ``code/revision/``.
_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))

from revision.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
