import json
from typing import Dict, List

import pandas as pd
import polars as pl

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

DATA_DIR = config["data_dir"]
GBD_KEYWORDS_PATH = (
    f"{DATA_DIR}/processed/classification/keywords_used_in_classification.csv"
)
UNIQUE_TEXTS_PATH = (
    f"{DATA_DIR}/processed/classification/df_unique_raw_texts.csv"
)
OUTPUT_PATH = f"{DATA_DIR}/processed/classification/unique_texts_classified_with_keywords_method.csv"


def load_gbd_keywords() -> Dict[str, List[str]]:
    """
    Load GBD keywords from a CSV file and process them into a dictionary.

    Returns:
        Dict[str, List[str]]: A dictionary with classification categories as keys and lists of keywords as values.
    """
    gbd_keywords_df = pd.read_csv(GBD_KEYWORDS_PATH)
    gbd_keywords_dict = {
        row["Classification Category"]: row["Keywords"].split(", ")
        for _, row in gbd_keywords_df.iterrows()
    }

    processed_dict = {}
    keyword_count = 0
    for category, keywords in gbd_keywords_dict.items():
        unique_keywords = [
            keyword.lower()
            for keyword in keywords
            if not any(
                other in keyword.lower()
                for other in keywords
                if keyword != other
            )
        ]
        processed_dict[category] = unique_keywords
        keyword_count += len(unique_keywords)

    print(f"Total unique keywords: {keyword_count}")
    return processed_dict


def keyword_classification(
    df: pl.DataFrame,
    classifications: Dict[str, List[str]],
    column_name: str = "raw_text",
    classification_column: str = "keyword_classification",
) -> pl.DataFrame:
    """
    Classify text in a DataFrame based on keywords.

    This function takes a DataFrame and a dictionary of classifications with their associated keywords.
    It then classifies each row in the specified column based on the presence of keywords.

    Args:
        df (pl.DataFrame): Input DataFrame.
        classifications (Dict[str, List[str]]): Dictionary of classification categories and their keywords.
        column_name (str): Name of the column containing text to classify.
        classification_column (str): Name of the output classification column.

    Returns:
        pl.DataFrame: DataFrame with added classification and keyword columns.
    """

    def classify_row(text: str) -> str:
        labels = [
            category
            for category, keywords in classifications.items()
            if any(keyword in text.lower() for keyword in keywords)
        ]
        return ", ".join(labels) if labels else "Other"

    def add_keyword(text: str) -> str:
        labels = [
            keyword
            for keywords in classifications.values()
            for keyword in keywords
            if keyword in text.lower()
        ]
        return ", ".join(labels) if labels else ""

    return df.with_columns(
        [
            pl.col(column_name)
            .map_elements(classify_row)
            .alias(classification_column),
            pl.col(column_name)
            .map_elements(add_keyword)
            .alias("keyword_column"),
        ]
    )


def main():
    """
    Main function to orchestrate the keyword classification process.

    This function loads GBD keywords, reads the input CSV file containing unique texts,
    applies the keyword classification, and saves the results to a new CSV file.
    """
    # Load and process GBD keywords
    gbd_keywords_dict = load_gbd_keywords()

    # Load and classify unique texts
    df_unique = pl.read_csv(UNIQUE_TEXTS_PATH)
    df_classified = keyword_classification(df_unique, gbd_keywords_dict)

    # Save results
    df_classified.write_csv(OUTPUT_PATH)
    print(f"Classified data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
