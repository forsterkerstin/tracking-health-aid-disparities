from typing import Dict

import polars as pl


def load_csv(
    file_path: str, separator: str = ",", dtypes: Dict = None
) -> pl.DataFrame:
    """
    Load a CSV file into a Polars DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        separator (str): CSV separator character.
        dtypes (Dict): Dictionary of column data types.

    Returns:
        pl.DataFrame: Loaded DataFrame.
    """
    try:
        df = pl.read_csv(
            file_path, separator=separator, dtypes=dtypes, ignore_errors=True
        )
        print(f"Data loaded successfully from {file_path}")
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None


def combine_and_filter_dataframes(
    df1: pl.DataFrame, df2: pl.DataFrame
) -> pl.DataFrame:
    """
    Combine two DataFrames vertically after aligning their columns, and filter out invalid rows.

    Args:
        df1 (pl.DataFrame): First DataFrame.
        df2 (pl.DataFrame): Second DataFrame.

    Returns:
        pl.DataFrame: Combined and filtered DataFrame.
    """
    common_columns = list(set(df1.columns).intersection(set(df2.columns)))
    df2 = df2.with_columns(
        [pl.col(col).cast(df1[col].dtype) for col in common_columns]
    )
    df1 = df1.select(common_columns)
    df2 = df2.select(common_columns)
    combined_df = pl.concat([df1, df2], how="vertical")

    return combined_df.filter(
        (pl.col("USD_Disbursement").is_not_null())
        & (pl.col("USD_Disbursement") >= 0)
        & (pl.col("raw_text").is_not_null())
        & (pl.col("raw_text") != "")
    )


def main():
    # Load the first CSV file: the aid funding data from 2000 to 2017
    dtypes = {"CRSid": pl.Utf8}
    df1 = load_csv(
        "data/processed/aid_funding_data/translated_df.csv",
        separator="|",
        dtypes=dtypes,
    )
    if df1 is not None:
        df1 = df1.rename({"CRSid": "CrsID"})

    # Load the second CSV file: the aid funding data from 2018 to 2022
    df2 = load_csv(
        "data/processed/aid_funding_data/df_2018_2022_translation_added.csv"
    )
    if df2 is not None:
        df2 = df2.rename({"LongDescription_language": "language"})

    # Combine and filter DataFrames
    if df1 is not None and df2 is not None:
        result_df = combine_and_filter_dataframes(df1, df2)

        # Write the result to a CSV file
        output_file = (
            "data/processed/aid_funding_data/df_translated_2000_to_2022.csv"
        )
        result_df.write_csv(output_file)
        print(f"Processed data written to {output_file}")
    else:
        print("Error: Unable to process data due to loading errors.")


if __name__ == "__main__":
    main()
