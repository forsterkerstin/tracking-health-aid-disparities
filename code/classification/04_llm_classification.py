# Imports
import asyncio

# Configuration
import json
import time
from typing import Any, Dict, List

import aiohttp
import nest_asyncio
import pandas as pd
import polars as pl

with open("config.json", "r") as f:
    config = json.load(f)

API_KEY = config["llm_api_key"]
DATA_DIR = config["data_dir"]
RATE_LIMIT = config["rate_limit"]
MODEL = config["model"]

# Function definitions


def load_data(file_path: str, dtypes: Dict[str, Any]) -> pl.DataFrame:
    """
    Load CSV data into a Polars DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        dtypes (Dict[str, Any]): Dictionary of column names and their data types.

    Returns:
        pl.DataFrame: Loaded DataFrame.
    """
    return pl.read_csv(file_path, dtypes=dtypes)


def get_unique_values(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Get unique values and their counts from a specific column.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column (str): Column name to get unique values from.

    Returns:
        pl.DataFrame: DataFrame with unique values and their counts.
    """
    return df[column].value_counts()


async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
) -> str:
    """
    Fetch data from an API endpoint.

    Args:
        session (aiohttp.ClientSession): Aiohttp client session.
        url (str): API endpoint URL.
        payload (Dict[str, Any]): Request payload.
        headers (Dict[str, str]): Request headers.

    Returns:
        str: API response content.
    """
    async with session.post(url, json=payload, headers=headers) as response:
        json_response = await response.json()
        if "choices" in json_response and len(json_response["choices"]) > 0:
            return json_response["choices"][0]["message"]["content"]
        else:
            print(f"Unexpected response structure: {json_response}")
            return None


async def rate_limited_fetch(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    max_retries: int = 5,
) -> str:
    """
    Perform a rate-limited API fetch with retries.

    Args:
        session (aiohttp.ClientSession): Aiohttp client session.
        url (str): API endpoint URL.
        payload (Dict[str, Any]): Request payload.
        headers (Dict[str, str]): Request headers.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

    Returns:
        str: API response content.

    Raises:
        Exception: If max retries are exceeded.
    """
    semaphore = asyncio.Semaphore(RATE_LIMIT)
    attempt = 0
    while attempt < max_retries:
        try:
            async with semaphore:
                response = await fetch(session, url, payload, headers)
                await asyncio.sleep(
                    1 / RATE_LIMIT
                )  # Wait to respect the rate limit
                return response
        except aiohttp.ClientError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            if attempt < max_retries:
                sleep_time = 2**attempt
                print(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
    raise Exception("Max retries exceeded")


async def classify_texts(
    df: pd.DataFrame,
    text_column: str,
    prompt: str,
    url: str,
    headers: Dict[str, str],
) -> List[str]:
    """
    Process a batch of texts through the API.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column name containing the text to process.
        prompt (str): Prompt to use for the API.
        url (str): API endpoint URL.
        headers (Dict[str, str]): Request headers.

    Returns:
        List[str]: List of API responses.
    """
    texts = df[text_column].to_list()
    texts = [element.lower() for element in texts]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            payload = {
                "messages": [{"role": "user", "content": f"{prompt} {text}"}],
                "model": MODEL,
                "max_tokens": 40,
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1,
                "stop": ["<|eot_id|>"],
            }
            tasks.append(rate_limited_fetch(session, url, payload, headers))

        responses = await asyncio.gather(*tasks)
        return responses


async def main(
    df: pd.DataFrame, text_column: str, prompt: str, classification_column: str
) -> pd.DataFrame:
    """
    Main function to process the DataFrame and classify texts.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column name containing the text to classify.
        prompt (str): Prompt to use for classification.
        classification_column (str): Column name to store the classification results.

    Returns:
        pd.DataFrame: DataFrame with added classification column.
    """
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    responses = await classify_texts(df, text_column, prompt, url, headers)
    df = df.copy()
    df.loc[:, classification_column] = responses
    df.to_csv(
        f"{DATA_DIR}/processed/classification/unique_texts_classified_with_llm_method.csv"
    )
    return df


# Main execution
if __name__ == "__main__":
    nest_asyncio.apply()  # Allow nested event loops

    # Load and process initial data
    dtypes = {"CrsID": pl.Utf8}
    df = load_data(
        f"{DATA_DIR}/processed/aid_funding_data/df_translated_2000_to_2022.csv",
        dtypes,
    )
    print(df.shape)

    df_unique = get_unique_values(df, "raw_text")
    df_unique.write_csv(
        f"{DATA_DIR}/processed/classification/df_unique_raw_texts.csv",
        include_header=True,
    )

    # Define classification parameters
    text_column = "raw_text"
    classification_column = "llm_classification"
    prompt = """
    Classify the given development aid project description by its primary objective:

    1. If the primary focus is not on addressing diseases or health conditions, return 'Other' and stop. If the description is very ambiguous, return 'Other' and stop.

    2. If the project is health-related, but addresses broad health initiatives or doesn't fit the provided specific disease categories perfectly, return 'General Health' and stop.

    3. If disease is the main focus, evaluate if it fits into any of these specific disease categories: [HIV/AIDS and sexually transmitted infections, Respiratory infections and tuberculosis, Enteric infections, Neglected tropical diseases and malaria, Maternal and neonatal disorders, Nutritional deficiencies, Neoplasms, Cardiovascular diseases, Chronic respiratory diseases, Digestive diseases, Neurological disorders, Mental disorders, Substance use disorders, Diabetes and kidney diseases, Skin and subcutaneous diseases, Sense organ diseases, Musculoskeletal disorders]

    Don't make assumptions if the text is not crystal clear. If you can't be sure, return 'General Health'.

    4. If the project fits one or more specific categories, return those category names separated by commas.

    Rules:
    - Do not combine 'Other' or 'General Health' with disease categories.
    - Only use labels from the provided list, 'Other', or 'General Health'.
    - Respond with the label(s) only, without additional commentary.

    Analyze this project description:"""

    # Run classification
    asyncio.run(main(df_unique, text_column, prompt, classification_column))
