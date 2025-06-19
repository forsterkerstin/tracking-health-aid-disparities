from typing import List, Optional

import langid
import polars as pl
from langdetect import LangDetectException, detect

"""
This script processes CRS (Creditor Reporting System) data from multiple years,
performs language detection on specific columns, and saves the results.

The main steps include:
1. Loading and combining CRS data from 2018 to 2022
2. Cleaning the data by filtering out invalid entries
3. Detecting languages for ProjectTitle, LongDescription, and ShortDescription columns
4. Saving the processed data and language detection results
"""


# Data Loading Functions
def load_crs_data(file_path: str) -> Optional[pl.DataFrame]:
    """
    Load CRS data from a file.

    Args:
        file_path (str): Path to the CRS data file.

    Returns:
        Optional[pl.DataFrame]: Loaded DataFrame or None if an error occurs.
    """
    try:
        return pl.read_csv(
            file_path,
            separator="|",
            encoding="utf-8",
            ignore_errors=True,
            infer_schema_length=0,
            dtypes={"USD_Disbursement": pl.Float64, "Year": pl.Int64},
        )
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Language Detection Functions
def detect_language_langid(text: str) -> Optional[str]:
    """
    Detect language using langid library.

    Args:
        text (str): Text to detect language for.

    Returns:
        Optional[str]: Detected language code or None if text is empty.
    """
    if not text.strip():
        return None
    lang, _ = langid.classify(text)
    return lang


def detect_language_langdetect(text: str) -> Optional[str]:
    """
    Detect language using langdetect library.

    Args:
        text (str): Text to detect language for.

    Returns:
        Optional[str]: Detected language code or None if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return None


def combine_languages(row: dict) -> str:
    """
    Combine language detection results from langdetect and langid.
    If both libraries agree and the detected language is not English,
    return that language. Otherwise, default to English.

    Args:
        row (dict): Dictionary containing langdetect and langid results.

    Returns:
        str: Combined language result.
    """
    langdetect = row["language_langdetect"]
    langid = row["language_langid"]
    return langdetect if langdetect == langid and langdetect != "en" else "en"


def run_language_detection(df: pl.DataFrame, column_name: str) -> None:
    """
    Run language detection on a specific column and save results.
    This function performs the following steps:
    1. Extract unique texts from the specified column
    2. Detect languages using both langdetect and langid
    3. Combine the results using the combine_languages function
    4. Save the results to a CSV file

    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the column to detect language for.
    """
    unique_texts = df[column_name].unique()
    unique_texts_df = pl.DataFrame({column_name: unique_texts})

    unique_texts_df = unique_texts_df.with_columns(
        pl.col(column_name)
        .map_elements(detect_language_langdetect)
        .alias("language_langdetect"),
        pl.col(column_name)
        .map_elements(detect_language_langid)
        .alias("language_langid"),
    )

    unique_texts_df = unique_texts_df.with_columns(
        pl.struct(["language_langdetect", "language_langid"])
        .map_elements(combine_languages)
        .alias("language_combined")
    )

    column_name = column_name.lower()
    unique_texts_df.write_csv(
        f"data/processed/detected_languages/unique_{column_name}s_with_detected_language.csv"
    )


# Main Execution
def main():
    # Load and combine data from 2018 to 2022
    file_names = [
        f"data/raw/unprocessed_aid_data/CRS/CRS {year} data.txt"
        for year in range(2018, 2023)
    ]

    # Load the first file
    df_2018_2022 = pl.read_csv(
        file_names[0],
        separator="|",
        encoding="utf-8",
        ignore_errors=True,
        infer_schema_length=0,
        dtypes={"USD_Disbursement": pl.Float64, "Year": pl.Int64},
    )

    # Load and concatenate the remaining files
    for file_name in file_names[1:]:
        crs_df = load_crs_data(file_name)
        if crs_df is not None:
            df_2018_2022 = pl.concat([df_2018_2022, crs_df], how="vertical")

    # Data cleaning: Remove invalid entries
    df_2018_2022 = df_2018_2022.filter(
        (pl.col("USD_Disbursement").is_not_null())
        & (pl.col("USD_Disbursement").cast(pl.Float64) >= 0)
        & (pl.col("LongDescription").is_not_null())
        & (pl.col("Year").is_not_null())
    )

    # Save the filtered aid data
    df_2018_2022.write_csv(
        "data/processed/aid_funding_data/crs_df_2018_to_2022_filtered.csv"
    )

    # Run language detection
    translation_columns = [
        "ProjectTitle",
        "LongDescription",
        "ShortDescription",
    ]
    for column in translation_columns:
        run_language_detection(df_2018_2022, column)


if __name__ == "__main__":
    main()
