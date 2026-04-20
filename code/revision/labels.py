"""Shared disease labels and multi-label parsing helpers."""

from __future__ import annotations

import re
from typing import Iterable, Set

import pandas as pd


STANDARD_DISEASES = [
    "HIV/AIDS and sexually transmitted infections",
    "Respiratory infections and tuberculosis",
    "Enteric infections",
    "Neglected tropical diseases and malaria",
    "Maternal and neonatal disorders",
    "Nutritional deficiencies",
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
]

SPECIAL_CATEGORIES = ["General Health", "Other"]
ALL_CATEGORIES = STANDARD_DISEASES + SPECIAL_CATEGORIES

_CANONICAL_LABEL_LOOKUP = {" ".join(label.split()).casefold(): label for label in ALL_CATEGORIES}
_PROXY_PATTERNS = [
    ("General Health", r"sexual and reproductive health|reproductive health|family planning|maternal health|child health|adolescent health"),
    ("HIV/AIDS and sexually transmitted infections", r"hiv|aids|sexually transmitted infections|stis?\b"),
    ("Respiratory infections and tuberculosis", r"tuberculosis|\btb\b|measles|influenza|respiratory infection"),
    ("Enteric infections", r"enteric|cholera|diarrh"),
    ("Neglected tropical diseases and malaria", r"malaria|neglected tropical|vector-borne|dengue|leishmania"),
    ("Maternal and neonatal disorders", r"maternal and neonatal|maternal\b|neonatal|newborn|obstetric fistula|fistula"),
    ("Nutritional deficiencies", r"nutrition|nutritional|malnutrition|iodine deficiency|food security"),
    ("Neoplasms", r"neoplasm|cancer|oncolog|tumou?r|hpv"),
    ("Cardiovascular diseases", r"cardiovascular|heart disease|hypertension|stroke"),
    ("Chronic respiratory diseases", r"asthma|chronic respiratory|copd"),
    ("Digestive diseases", r"digestive|gastro"),
    ("Neurological disorders", r"neurolog|epilep|parkinson|alzheimer"),
    ("Mental disorders", r"mental health|psychosocial|depression|anxiety|psychiatr"),
    ("Substance use disorders", r"substance use|tobacco|alcohol|drug use"),
    ("Diabetes and kidney diseases", r"diabetes|kidney|renal"),
    ("Skin and subcutaneous diseases", r"skin disease|dermat|cutaneous"),
    ("Sense organ diseases", r"deaf|hearing|vision|blind|sense organ"),
    ("Musculoskeletal disorders", r"musculoskeletal|physiotherapy|orthop|spastic"),
]


def clean_label_text(label) -> str:
    """Collapse repeated whitespace and strip surrounding spaces."""
    if label is None or pd.isna(label):
        return ""
    return " ".join(str(label).split())


def canonicalize_label(label) -> str:
    """Canonicalize a single atomic label for stable comparisons."""
    cleaned = clean_label_text(label)
    if not cleaned:
        return ""
    return cleaned.casefold()


def display_label(label) -> str:
    """Map a raw or canonical label to a stable display form when known."""
    canon = canonicalize_label(label)
    if not canon:
        return ""
    return _CANONICAL_LABEL_LOOKUP.get(canon, clean_label_text(label))


def parse_label_string(labels) -> list[str]:
    """Parse a multi-label field into canonicalized atomic labels."""
    if isinstance(labels, (list, tuple, set)):
        parsed = [canonicalize_label(label) for label in labels]
        return [label for label in parsed if label]
    if pd.isna(labels):
        return []
    normalized = str(labels).replace(".", ",").replace(";", ",")
    parsed = [canonicalize_label(label) for label in normalized.split(",")]
    return [label for label in parsed if label]


def build_label_display_map(series_list: Iterable[pd.Series]) -> dict[str, str]:
    """Choose one stable display label per canonical class across one or more series."""
    display_map: dict[str, str] = {}
    for series in series_list:
        for raw in series.dropna():
            for canon in parse_label_string(raw):
                if canon and canon not in display_map:
                    display_map[canon] = display_label(canon)
    for label in ALL_CATEGORIES:
        canon = canonicalize_label(label)
        display_map.setdefault(canon, label)
    return display_map


def _find_label_mentions(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for label in sorted(ALL_CATEGORIES, key=len, reverse=True):
        if label == "Other":
            pattern = r"\bOther\b"
            flags = 0
        else:
            pattern = rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])"
            flags = re.IGNORECASE
        for match in re.finditer(pattern, text, flags=flags):
            matches.append((match.start(), label))
    matches.sort(key=lambda item: item[0])
    return matches


def extract_model_labels(text) -> str:
    """
    Recover category labels from model outputs.
    """
    cleaned = clean_label_text(text)
    if not cleaned:
        return ""

    segments = []
    raw_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if raw_lines:
        segments.append(raw_lines[-1])
    if len(cleaned) > 240:
        segments.append(cleaned[-240:])
    segments.append(cleaned)

    for segment in segments:
        mentions = _find_label_mentions(segment)
        if not mentions:
            segment_lower = segment.casefold()
            proxy_hits: list[tuple[int, str]] = []
            for label, pattern in _PROXY_PATTERNS:
                match = re.search(pattern, segment_lower, flags=re.IGNORECASE)
                if match is not None:
                    proxy_hits.append((match.start(), label))
            if proxy_hits:
                proxy_hits.sort(key=lambda item: item[0])
                latest_pos = max(pos for pos, _ in proxy_hits)
                latest = [label for pos, label in proxy_hits if pos >= latest_pos - 40]
                return latest[-1]
            continue

        latest_pos = max(pos for pos, _ in mentions)
        window_mentions = [(pos, label) for pos, label in mentions if pos >= latest_pos - 120]
        latest_special = [label for pos, label in window_mentions if label in SPECIAL_CATEGORIES]
        if latest_special:
            return latest_special[-1]

        ordered_labels: list[str] = []
        seen: set[str] = set()
        for _, label in window_mentions:
            if label not in seen:
                ordered_labels.append(label)
                seen.add(label)
        if ordered_labels:
            return ", ".join(ordered_labels)

    return "Other"


def split_labels(labels) -> Set[str]:
    """Split comma/semicolon/period-separated labels into a normalized set."""
    return set(parse_label_string(labels))
