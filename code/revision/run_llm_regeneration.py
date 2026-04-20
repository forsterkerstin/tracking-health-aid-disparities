#!/usr/bin/env python3
"""
Optional Together.ai LLM classification for revision subsamples.

**Default behaviour:** does nothing useful — prints a skip message and exits 0.
No API calls unless you explicitly enable and provide a key.

Enable:
  export REVISION_ENABLE_LLM=1
  export TOGETHER_API_KEY='...'    # or set llm_api_key in repo config.json

Example (small test, 5 rows):
  REVISION_ENABLE_LLM=1 TOGETHER_API_KEY=... \\
    python code/revision/run_llm_regeneration.py classify-together \\
      --input data/combined_df_1_percent.csv \\
      --output data/out_classified_test.csv \\
      --text-col raw_text \\
      --result-col llm_classification_test \\
      --max-rows 5

See OPTIONAL_INPUT_REGENERATION.md in this directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- repo / import path -------------------------------------------------
_CODE = Path(__file__).resolve().parent.parent
_REPO = _CODE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))


def _enabled() -> bool:
    v = os.environ.get("REVISION_ENABLE_LLM", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _skip_message() -> str:
    return """\
[run_llm_regeneration] SKIPPED (no API calls).

The revision paper pipeline already assumes classified CSVs exist.
To re-run LLM classification (uses time and API quota):

  export REVISION_ENABLE_LLM=1
  export TOGETHER_API_KEY='your_key'   # or set "llm_api_key" in config.json

Then run e.g.:
  python code/revision/run_llm_regeneration.py classify-together --help

Docs: code/revision/OPTIONAL_INPUT_REGENERATION.md
"""


def _load_api_key() -> str | None:
    k = os.environ.get("TOGETHER_API_KEY", "").strip()
    if k:
        return k
    cfg = _REPO / "config.json"
    if cfg.is_file():
        try:
            with open(cfg, encoding="utf-8") as f:
                data = json.load(f)
            k = str(data.get("llm_api_key", "")).strip()
            return k or None
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _load_model_name() -> str:
    cfg = _REPO / "config.json"
    if cfg.is_file():
        try:
            with open(cfg, encoding="utf-8") as f:
                data = json.load(f)
            m = str(data.get("model", "")).strip()
            if m:
                return m
        except (json.JSONDecodeError, OSError):
            pass
    return "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


def classify_single_together(
    text: str,
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 40,
    temperature: float = 0.0,
    seed: int = 42,
) -> str:
    import requests

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "messages": [{"role": "user", "content": f"{prompt}\n\n{text}"}],
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "choices" in data and data["choices"]:
        return (data["choices"][0].get("message") or {}).get("content") or ""
    raise RuntimeError(f"Unexpected Together response: {data!r}")


DEFAULT_DISEASE_PROMPT = """Classify the given development aid project description by its primary objective:

1. If the focus is not on addressing diseases or health conditions, return 'Other' and stop.

2. If the project is health-related, but addresses broad health initiatives and doesn't fit the provided specific disease categories perfectly, return 'General Health' and stop.

3. If disease is the main focus, evaluate if it fits into any of these specific disease categories: [HIV/AIDS and sexually transmitted infections, Respiratory infections and tuberculosis, Enteric infections, Neglected tropical diseases and malaria, Maternal and neonatal disorders, Nutritional deficiencies, Neoplasms, Cardiovascular diseases, Chronic respiratory diseases, Digestive diseases, Neurological disorders, Mental disorders, Substance use disorders, Diabetes and kidney diseases, Skin and subcutaneous diseases, Sense organ diseases, Musculoskeletal disorders]

Don't make assumptions if the text is not clear.

4. If the project fits one or more specific categories, return those categories names separated by commas.

Rules:
- Do not combine 'Other' or 'General Health' with disease categories.
- Only use labels from the provided list, 'Other', or 'General Health'.
- Respond with the label(s) only, without additional commentary.

Project description:
"""


DEFAULT_NEGATION_AWARE_PROMPT = """You are an expert health policy assistant. Your task is to identify which disease categories are positively targeted by this
development aid project, based on its description.
Read the project description carefully. ONLY include disease categories that are explicitly targeted, supported, or addressed
in the project. DO NOT include diseases that are:
- Mentioned as excluded
- Mentioned as not funded
- Mentioned only in background context
- Described as 'not a focus' or 'not covered'
You may assign one or more of the following 17 disease categories:
[HIV/AIDS and sexually transmitted infections, Respiratory infections and tuberculosis, Enteric infections, Neglected tropical
diseases and malaria, Maternal and neonatal disorders, Nutritional deficiencies, Neoplasms, Cardiovascular diseases,
Chronic respiratory diseases, Digestive diseases, Neurological disorders, Mental disorders, Substance use disorders,
Diabetes and kidney diseases, Skin and subcutaneous diseases, Sense organ diseases, Musculoskeletal disorders]
Answer as a comma-separated list of disease categories. Use "Other" or "General Health" if no specific disease is
addressed. Respond with the label(s) only, without additional commentary.
Project Description:
"""


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if args.prompt_kind == "negation-aware":
        return DEFAULT_NEGATION_AWARE_PROMPT
    return DEFAULT_DISEASE_PROMPT


def cmd_classify_together(args: argparse.Namespace) -> int:
    import pandas as pd

    api_key = _load_api_key()
    if not api_key:
        print(
            "REVISION_ENABLE_LLM is set but no API key found. "
            "Set TOGETHER_API_KEY or config.json llm_api_key.",
            file=sys.stderr,
        )
        return 1

    inp = Path(args.input)
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    df = pd.read_csv(inp, low_memory=False)
    if args.text_col not in df.columns:
        print(f"Missing text column {args.text_col!r}", file=sys.stderr)
        return 1

    n = len(df)
    if args.max_rows is not None:
        df = df.iloc[: int(args.max_rows)].copy()
        print(f"Limiting to {len(df)} of {n} rows.")

    prompt = _resolve_prompt(args)

    model = args.model or _load_model_name()
    print(f"Model: {model}")
    print("This will issue one HTTP request per row (parallel). Press Ctrl+C to abort.")

    texts = df[args.text_col].fillna("").astype(str)
    results: dict[int, str] = {}
    lock = threading.Lock()

    def worker(idx: int, text: str) -> None:
        try:
            out = classify_single_together(
                text,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            with lock:
                results[idx] = out.strip()
        except Exception as e:
            with lock:
                results[idx] = f"ERROR: {e}"

    indices = texts.index.tolist()
    values = texts.tolist()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, i, t) for i, t in zip(indices, values)]
        for k, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if k % 50 == 0 or k == len(futs):
                print(f"  completed {k}/{len(futs)}", flush=True)

    df[args.result_col] = [results[i] for i in indices]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {out} ({len(df)} rows) in {elapsed:.1f}s")
    return 0


def main() -> int:
    wants_help = "-h" in sys.argv or "--help" in sys.argv
    if not _enabled() and not wants_help:
        print(_skip_message())
        return 0

    p = argparse.ArgumentParser(description="Optional Together LLM batch classification (revision).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify-together", help="Classify a CSV column via Together chat completions")
    c.add_argument("--input", required=True, help="Input CSV path")
    c.add_argument("--output", required=True, help="Output CSV path")
    c.add_argument("--text-col", default="raw_text")
    c.add_argument("--result-col", default="llm_classification_new", help="Column name for model output")
    c.add_argument("--max-rows", type=int, default=None, help="Cap rows (for testing)")
    c.add_argument("--workers", type=int, default=5, help="Parallel HTTP workers")
    c.add_argument("--model", default=None, help="Override model id (default: config.json or Llama 3.1 70B)")
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--max-tokens", type=int, default=40)
    c.add_argument(
        "--prompt-file",
        default=None,
        help="UTF-8 file with full prompt prefix (else built-in disease prompt)",
    )
    c.add_argument(
        "--prompt-kind",
        choices=["default", "negation-aware"],
        default="default",
        help="Use a built-in prompt template unless overridden by --prompt/--prompt-file",
    )
    c.add_argument(
        "--prompt",
        default=None,
        help="Inline prompt prefix if not using --prompt-file",
    )
    c.set_defaults(func=cmd_classify_together)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
