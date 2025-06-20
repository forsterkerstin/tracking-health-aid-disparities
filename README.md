# Tracking funding disparities in global health aid with machine learning

### Abstract

Reducing the global burden of disease is crucial for improving health outcomes worldwide. However, funding gaps leave vulnerable populations without necessary support for major health challenges, particularly in the least developed countries. In this paper, we develop a machine learning framework using large language models to track flows in official development assistance (ODA) earmarked for health and identify funding gaps. Specifically, we classified 3.7 million development aid projects from 2000 to 2022 (USD 332 billion) into 17 major categories of communicable, maternal, neonatal, and nutritional diseases (CMNNDs) and non-communicable diseases (NCDs). We then compared the rank of per capita ODA disbursement against the rank of disease burden (i.e., disability-adjusted life years [DALYs]) to identify relative funding gaps at the country level. Even though funding and disease burden are significantly correlated for many diseases, there are notable disparities. For example, NCDs account for 59.5% of global DALYs but received only 2.5% of health-related ODA. This disparity is particularly concerning because, in low and middle-income countries, there is an increasing double burden: not only from the traditional burden of CMNNDs, but also from the rising burden of NCDs. Our results also show that several regions face severe funding gaps across multiple diseases including Central Africa and parts of South Asia and West Africa. Our results identify health disparities to inform public policy decisions in development aid assistance. Overall, our machine learning framework helps identify health disparities and thereby supports the targeted allocation of health aid where it is most needed, thereby reducing the global burden of diseases.

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/forsterkerstin/monitoring-public-health.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run scripts**:
    The `data` and `plots` folders will be automatically created at runtime as needed by the scripts.

    Run the scripts in the following order: `preprocessing` $\rightarrow$ `classification` $\rightarrow$ `visualization`. You can run `main.py` to execute the core analysis scripts without incurring API costs.

    * **Preprocessing**:
        1.  `01_detect_language.py`
        2.  `02_translate_texts.py`
        3.  `03_combine_aid_funding_datasets.py`
    * **Classification**:
        1.  `04_llm_classification.py`
        2.  `05_keyword_classification.py`
        3.  `06_add_classification_to_aid_data.py`
    * **Visualization**:
        1.  `07_aid_funding_plots.py`
        2.  `08_correlation_and_funding_disparities_plots.py`
        3.  `09_comparing_classification_methods.py`

    *Note: `main.py` is provided to run only the visualization and analysis scripts, as the translation and classification scripts require paid API keys.*

## Project Overview

This project consists of three main stages:

1.  **Preprocessing (2018-2022 Data Extension)**
    * Detect languages in new entries
    * Translate non-English text to English
    * Preprocess and merge with the original 2000-2017 dataset

2.  **Classification**
    * Categorize projects using two methods:
        1. Large Language Model (LLM) classification
        2. Keyword-based classification
    * Merge and process classification results

3.  **Visualization and Analysis**
    * Generate insights from classified data
    * Analyze funding trends and disparities
    * Compare classification methods

## Preprocessing

### 1. Detect language of project descriptions in the 2018-2022 dataset (`01_detect_language.py`)

* Aggregates 2018-2022 aid funding data
* Filters and cleans data
* Detects languages using `langid` and `langdetect`

### 2. Translate non-English texts using Google's Translation API (`02_translate_texts.py`)

* Translates non-English texts in the `ProjectTitle`, `LongDescription`, and `ShortDescription` columns.
* Creates a unified "raw_text" column as a project description column.

### 3. Merging the aid datasets (`03_combine_aid_funding_datasets.py`)

* Merges 2000-2017 data with 2018-2022 data
* Applies consistent filtering and cleaning

## Classification

### 1. LLM Classification (`04_llm_classification.py`)

* Uses an LLM API (Together.xyz) for project description classification
* Assigns one or multiple categories to each project description

### 2. Keyword Classification (`05_keyword_classification.py`)

* Rule-based classification using predefined keywords
* Processes texts and assigns categories based on keyword presence

### 3. Classification Merging (`06_add_classification_to_aid_data.py`)

* Combines LLM and keyword classification results
* Processes multi-label classifications
* Adjusts funding allocation for multi-category projects

## Visualization and Analysis

### 1. Aid Funding Analysis (`07_aid_funding_plots.py`)

* Visualizes aid distribution trends
* Analyzes disease-specific funding patterns

### 2. Correlation and Disparity Analysis (`08_correlation_and_funding_disparities_plots.py`)

* Examines correlations between aid funding and disease burden
* Visualizes funding disparities across regions and diseases

### 3. Classification Method Comparison (`09_comparing_classification_methods.py`)

* Evaluates LLM and keyword classification methods
* Calculates performance metrics (precision, recall, F1-score)
* Visualizes classification accuracy and discrepancies

## Folder Structure

```
tracking-health-aid-disparities/
├── README.md
├── config.json
├── requirements.txt
├── code/
│   ├── preprocessing/
│   │   ├── 01_detect_language.py
│   │   ├── 02_translate_texts.py
│   │   └── 03_combine_aid_funding_datasets.py
│   ├── classification/
│   │   ├── 04_llm_classification.py
│   │   ├── 05_keyword_classification.py
│   │   └── 06_add_classification_to_aid_data.py
│   ├── visualization/
│   │   ├── 07_aid_funding_plots.py
│   │   ├── 08_correlation_and_funding_disparities_plots.py
│   │   └── 09_comparing_classification_methods.py
│   └── main.py
├── data/
│   ├── processed/
│   │   ├── aid_funding_data/
│   │   ├── classification/
│   │   ├── detected_languages/
│   │   └── translated_texts/
│   ├── raw/
│   │   ├── background_data/
│   │   └── unprocessed_aid_data/
│   └── results/
└── plots/
    ├── aid_plots/
    ├── classification/
    ├── correlation/
    └── map_plots/
