```markdown
# Multilingual Dataset Translation for Paragraph Level Dataset

## Description
This Python script automates the translation of paragraphs dataset in our implementation into multiple languages using the Google Translate API. It is designed to handle large batches of data, translating text fields from English into specified target languages and saving the translated content in a new file.

We were to translate the paragraphs dataset into top 10 languages in Twitter that are English, Japanese, Spanish, Portuguese, Arabic, French, Indonesian, Russian, Turkish, Hindi.

We tried translating to all 10 languages but since the volume of paragraphs is much more than articles, it is taking more than 8-9 hours to just iterate over less than 30-40% of the data and then we are getting server connection error in ORC. Hence we decreased the translations to 2 languages other than english on paragraph level.

Hence we are translating the data to these 2 languages -  Japanese and Spanish.

## Features
- Translate text from English to multiple languages.
- Handle batch processing of CSV files containing text data.
- Save the translated data into new CSV files.
- Retry translation on failure to handle transient API errors.

## Requirements
To run this script, you need:
- Python 3.x
- pandas library
- googletrans library
- tqdm library

## Usage
1. **Prepare Data Files**: Place your CSV files in the `split_files` directory. Each file should contain the columns 'id_str', 'label', 'tweet', and 'paragraph'.
2. **Configure Languages**: Edit the `languages` list in the script to include the languages you want to translate to. The default languages are English, Japanese, and Spanish.
3. **Run the Script**: Execute the script from your command line:
```bash
python translate_script.py
```
4. **Check Output**: Translated files will be stored in the `translated_files` directory. Each file will be named using the original file name with an added `_translated.csv` suffix.

## Script Details

### Functions
- `translate_text(text, dest_lang)`: Takes a string and a destination language code (e.g., 'es' for Spanish), and returns the translated string.
- `translate_dataset_to_languages(dataframe, languages)`: Processes each row in the DataFrame, translates the specified fields into each target language, and compiles the results into a new DataFrame.

### Error Handling
The script includes robust error handling to manage common issues such as network errors or API limits:
- **Automatic Retries**: On encountering a translation error, the script pauses briefly and retries the translation request.
- **Rate Limiting**: Delays are inserted between translation requests to avoid exceeding the Google Translate API's rate limits.

### Output
- **CSV Files**: For each input file, a corresponding translated file is saved in the `translated_files` directory.
- **Log Messages**: Progress are printed to the console to track the translation process.