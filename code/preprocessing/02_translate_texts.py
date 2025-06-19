import json
import random
import time
from typing import Dict, Optional

import polars as pl
import requests

"""
This script processes and translates text data for the aid funding dataset.
It performs the following main tasks:
1. Translates non-English texts using Google Translate API
2. Adds language detection and translation information to the main dataset
3. Creates a consolidated 'raw_text' column with all translated text
"""

# Configuration
CONFIG_PATH = "config.json"


def load_config(path: str) -> Dict:
    """Load configuration from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


config = load_config(CONFIG_PATH)


def translate_text(
    text: str,
    target_language: str = "en",
    max_retries: int = 5,
    base_delay: int = 1,
) -> Optional[str]:
    """
    Translate text using Google Translate API with retry mechanism.

    Args:
        text (str): The text to translate.
        target_language (str): The target language code (default: 'en' for English).
        max_retries (int): Maximum number of retry attempts (default: 5).
        base_delay (int): Base delay for exponential backoff (default: 1 second).

    Returns:
        Optional[str]: Translated text or None if translation fails after max retries.
    """
    api_key = config["google_translate_api_key"]
    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    data = {"q": text, "target": target_language, "format": "text"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                return json.loads(response.text)["data"]["translations"][0][
                    "translatedText"
                ]
            elif response.status_code == 429:  # Rate limit error
                raise requests.exceptions.RequestException(
                    "Rate limit reached"
                )
            else:
                raise requests.exceptions.RequestException(
                    f"Error: {response.status_code}, {response.text}"
                )
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Max retries reached. Error: {e}")
                return None
            delay = (2**attempt) * base_delay + (random.random() * base_delay)
            print(f"Request failed. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    return None


def process_translations(column_name: str) -> None:
    """
    Process translations for a specific column.

    Args:
        column_name (str): Name of the column to process (e.g., "ProjectTitle").

    This function:
    1. Reads unique texts for the specified column
    2. Filters out non-English texts
    3. Translates non-English texts
    4. Saves the translated texts to a CSV file
    """
    # Read unique texts for the column
    unique_texts = pl.read_csv(
        f"{config['data_dir']}/processed/detected_languages/unique_{column_name.lower()}s_with_detected_language.csv"
    )

    # Filter non-English texts and translate them
    non_english_texts = unique_texts.filter(
        pl.col("language_combined") != "en"
    )
    print(non_english_texts)
    translated_texts = non_english_texts.with_columns(
        pl.col(column_name)
        .map_elements(translate_text)
        .alias(f"{column_name}Translated")
    )

    # Save translated texts
    translated_texts.write_csv(
        f"{config['data_dir']}/processed/translated_texts/unique_{column_name.lower()}s_translated.csv"
    )


def create_language_and_translation_dicts() -> Dict[str, Dict[str, str]]:
    """
    Create dictionaries for quick lookup of text languages and translations.

    Returns a dictionary containing language detection and translation dictionaries for each column.
    """
    # Create language detection dictionaries
    lang_dict_long = dict(
        zip(
            unique_longdescriptions_lang_detected["LongDescription"],
            unique_longdescriptions_lang_detected["language"],
        )
    )
    lang_dict_short = dict(
        zip(
            unique_shortdescriptions_lang_detected["ShortDescription"],
            unique_shortdescriptions_lang_detected["language"],
        )
    )
    lang_dict_title = dict(
        zip(
            unique_projecttitles_lang_detected["ProjectTitle"],
            unique_projecttitles_lang_detected["language"],
        )
    )

    # Create translation dictionaries
    translation_dict_long = dict(
        zip(
            unique_longdescriptions_translated["LongDescription"],
            unique_longdescriptions_translated["LongDescriptionTranslated"],
        )
    )
    translation_dict_short = dict(
        zip(
            unique_shortdescriptions_translated["ShortDescription"],
            unique_shortdescriptions_translated["ShortDescriptionTranslated"],
        )
    )
    translation_dict_title = dict(
        zip(
            unique_projecttitles_translated["ProjectTitle"],
            unique_projecttitles_translated["ProjectTitleTranslated"],
        )
    )

    return {
        "lang_dict_long": lang_dict_long,
        "lang_dict_short": lang_dict_short,
        "lang_dict_title": lang_dict_title,
        "translation_dict_long": translation_dict_long,
        "translation_dict_short": translation_dict_short,
        "translation_dict_title": translation_dict_title,
    }


def main():
    """
    Orchestrates the translation and data processing workflow.

    This function:
    1. Processes translations for each specified column
    2. Loads the main dataset and language detection/translation data
    3. Creates language and translation dictionaries
    4. Adds language detection and translated columns to the main dataframe
    5. Creates a consolidated 'raw_text' column
    6. Saves the processed dataframe
    """
    # Process translations for each column
    columns_to_translate = [
        "ProjectTitle",
        "LongDescription",
        "ShortDescription",
    ]
    for column in columns_to_translate:
        process_translations(column)

    # Load main dataset
    main_df = pl.read_csv(
        f"{config['data_dir']}/processed/aid_funding_data/crs_df_2018_to_2022_filtered.csv"
    )

    # Load language detection and translation data
    global unique_longdescriptions_lang_detected, unique_shortdescriptions_lang_detected, unique_projecttitles_lang_detected
    global unique_longdescriptions_translated, unique_shortdescriptions_translated, unique_projecttitles_translated

    unique_longdescriptions_lang_detected = pl.read_csv(
        f"{config['data_dir']}/processed/detected_languages/unique_longdescriptions_with_detected_language.csv"
    )
    unique_shortdescriptions_lang_detected = pl.read_csv(
        f"{config['data_dir']}/processed/detected_languages/unique_shortdescriptions_with_detected_language.csv"
    )
    unique_projecttitles_lang_detected = pl.read_csv(
        f"{config['data_dir']}/processed/detected_languages/unique_projecttitles_with_detected_language.csv"
    )

    unique_longdescriptions_translated = pl.read_csv(
        f"{config['data_dir']}/processed/translated_texts/unique_longdescriptions_translated.csv"
    )
    unique_shortdescriptions_translated = pl.read_csv(
        f"{config['data_dir']}/processed/translated_texts/unique_shortdescriptions_translated.csv"
    )
    unique_projecttitles_translated = pl.read_csv(
        f"{config['data_dir']}/processed/translated_texts/unique_projecttitles_translated.csv"
    )

    # Create language and translation dictionaries
    dicts = create_language_and_translation_dicts()

    # Add language columns to main dataframe
    main_df = main_df.with_columns(
        pl.col("LongDescription")
        .map_elements(dicts["lang_dict_long"].get)
        .alias("LongDescription_language"),
        pl.col("ShortDescription")
        .map_elements(dicts["lang_dict_short"].get)
        .alias("ShortDescription_language"),
        pl.col("ProjectTitle")
        .map_elements(dicts["lang_dict_title"].get)
        .alias("ProjectTitle_language"),
    )

    # Add translated columns to main dataframe
    main_df = main_df.with_columns(
        pl.when(
            pl.col("LongDescription").is_not_null()
            & (pl.col("LongDescription_language") == "en")
        )
        .then(pl.col("LongDescription"))
        .otherwise(
            pl.col("LongDescription").map_elements(
                dicts["translation_dict_long"].get
            )
        )
        .alias("LongDescriptionTranslated"),
        pl.when(
            pl.col("ShortDescription").is_not_null()
            & (pl.col("ShortDescription_language") == "en")
        )
        .then(pl.col("ShortDescription"))
        .otherwise(
            pl.col("ShortDescription").map_elements(
                dicts["translation_dict_short"].get
            )
        )
        .alias("ShortDescriptionTranslated"),
        pl.when(
            pl.col("ProjectTitle").is_not_null()
            & (pl.col("ProjectTitle_language") == "en")
        )
        .then(pl.col("ProjectTitle"))
        .otherwise(
            pl.col("ProjectTitle").map_elements(
                dicts["translation_dict_title"].get
            )
        )
        .alias("ProjectTitleTranslated"),
    )

    # Add raw_text column (concatenation of translated columns)
    main_df = main_df.with_columns(
        (
            pl.col("ProjectTitleTranslated")
            + " "
            + pl.col("ShortDescriptionTranslated")
            + " "
            + pl.col("LongDescriptionTranslated")
        ).alias("raw_text")
    )

    # Save the processed dataframe
    main_df.write_csv(
        f"{config['data_dir']}/processed/aid_funding_data/df_2018_2022_translation_added.csv"
    )


if __name__ == "__main__":
    main()
