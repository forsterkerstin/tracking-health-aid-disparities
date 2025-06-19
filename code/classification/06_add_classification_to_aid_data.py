from typing import Dict, List

import polars as pl


def normalize_text(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Normalize text in a specified column by stripping whitespace and converting to lowercase.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column (str): Name of the column to normalize.

    Returns:
        pl.DataFrame: DataFrame with normalized text column.
    """
    return df.with_columns(
        pl.col(column)
        .str.strip_chars()
        .str.to_lowercase()
        .alias("normalized_text")
    )


def create_classification_dict(
    df: pl.DataFrame, text_column: str, classification_column: str
) -> Dict[str, str]:
    """
    Create a dictionary mapping normalized text to its classification.

    Args:
        df (pl.DataFrame): Input DataFrame.
        text_column (str): Name of the text column.
        classification_column (str): Name of the classification column.

    Returns:
        Dict[str, str]: Dictionary mapping text to classification.
    """
    return dict(zip(df[text_column], df[classification_column]))


def add_classifications(
    df: pl.DataFrame, keyword_dict: Dict[str, str], llm_dict: Dict[str, str]
) -> pl.DataFrame:
    """
    Add keyword and LLM classifications to the DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame.
        keyword_dict (Dict[str, str]): Dictionary for keyword classifications.
        llm_dict (Dict[str, str]): Dictionary for LLM classifications.

    Returns:
        pl.DataFrame: DataFrame with added classification columns.
    """
    return df.with_columns(
        [
            pl.col("raw_text")
            .str.strip_chars()
            .str.to_lowercase()
            .map_elements(lambda x: keyword_dict.get(x, None))
            .alias("keyword_classification"),
            pl.col("raw_text")
            .str.strip_chars()
            .str.to_lowercase()
            .map_elements(lambda x: llm_dict.get(x, "Other"))
            .alias("llm_classification"),
        ]
    )


def process_multilabels(
    df: pl.DataFrame, column_name: str, method: str
) -> pl.DataFrame:
    """
    Process multi-label classifications and adjust funding.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the classification column.
        method (str): Classification method (e.g., 'keyword' or 'llm').

    Returns:
        pl.DataFrame: Processed DataFrame with adjusted funding.
    """

    def split_classifications(x: str) -> List[str]:
        return [
            classification.strip()
            for classification in str(x).split(",")
            if classification.strip()
        ]

    df = df.with_columns(
        [
            pl.col(column_name)
            .map_elements(split_classifications)
            .map_elements(len)
            .alias(f"number_of_{method}_categories"),
            (
                pl.col("USD_Disbursement")
                / pl.col(column_name)
                .map_elements(split_classifications)
                .map_elements(len)
            ).alias(f"adjusted_funding_{method}"),
        ]
    )

    return df


def main():
    # Load data
    dtypes = {"CRSid": pl.Utf8, "CrsID": pl.Utf8}
    df_keyword = pl.read_csv(
        "data/processed/classification/unique_texts_classified_with_keywords_method.csv"
    )
    df_llm = pl.read_csv(
        "data/processed/classification/unique_texts_classified_with_llm_method.csv"
    )
    df = pl.read_csv(
        "data/processed/aid_funding_data/df_translated_2000_to_2022.csv",
        dtypes=dtypes,
    )

    # Normalize text and create classification dictionaries
    df_keyword = normalize_text(df_keyword, "raw_text")
    df_llm = normalize_text(df_llm, "raw_text")
    keyword_dict = create_classification_dict(
        df_keyword, "normalized_text", "keyword_classification"
    )
    llm_dict = create_classification_dict(
        df_llm, "normalized_text", "llm_classification"
    )

    # Add classifications to main DataFrame
    df = add_classifications(df, keyword_dict, llm_dict)

    # Process multi-labels and adjust funding
    df = process_multilabels(df, "keyword_classification", "keyword")
    df = process_multilabels(df, "llm_classification", "llm")

    # Save the final DataFrame
    df.write_csv("data/processed/aid_funding_data/final_df.csv")


if __name__ == "__main__":
    main()
