```markdown
# Multilingual Dataset Translation for Article Level Dataset

## Description
This script leverages the Google Translate API to automatically translates the Article Dataset from English into multiple target languages. It is specifically designed to translate datasets containing tweets and articles, supporting a range of languages.

## Features
- Supports translation of text data to multiple languages including major ones used on Twitter.
- Processes large datasets by iterating through each record.
- Utilizes a progress bar to display real-time progress of the translation process.
- Saves the translated dataset into a CSV file.

## Requirements
To run this script, ensure you have the following installed:
- Python 3.x
- pandas
- googletrans
- tqdm

## Usage
1. **Prepare Your Dataset**: Ensure your dataset is in CSV format with columns labeled 'id_str', 'label', 'tweet', and 'article'.
2. **Configure Target Languages**: Modify the `languages` list in the script to include the languages you want to target. Default languages are the top 10 languages used on Twitter.
3. **Run the Script**: Execute the script using Python from your command line:
```bash
python translate_script.py
```
4. **Retrieve Output**: Find the translated dataset in the same directory, named `articles_translated.csv`.

## Script Details

### Main Functions
- `translate_dataset_to_languages(dataframe, languages)`: Translates text fields in the dataset to each of the specified languages and compiles the results into a new DataFrame.

### Output
- **CSV Output**: The script outputs a file named `articles_translated.csv`, containing the original data along with their translations.