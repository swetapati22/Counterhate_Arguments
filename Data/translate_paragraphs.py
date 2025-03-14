import os
import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import time

# Initializing the Google Translate API translator
translator = Translator()

# Directory where the original batch files are stored
input_dir = 'split_files'

# Directory where the translated files will be saved
output_dir = 'translated_files'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to translate text from English to a target language
def translate_text(text, dest_lang):
    try:
        # Attempt to translate text
        return translator.translate(text, src='en', dest=dest_lang).text
    except Exception as e:
        # Print any error encountered and wait 1 second before retrying
        print(f"Error: {e}")
        time.sleep(1) 
        # Retry translation
        return translator.translate(text, src='en', dest=dest_lang).text

# Function to translate an entire dataset to multiple languages
def translate_dataset_to_languages(dataframe, languages):
    translated_data = []  # List to store translated data rows

    # Iterate over each row in the DataFrame with a progress bar
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Translating tweets and paragraphs"):
        id_str = row['id_str']
        label = row['label']
        tweet = row['tweet']
        paragraph = row['paragraph']
        
        # Translate both tweet and paragraph into each target language
        for lang in languages:
            if lang == 'en':  # No translation needed if the language is English
                translated_tweet = tweet
                translated_paragraph = paragraph
            else:  # Translate the text to the target language
                translated_tweet = translate_text(tweet, lang)
                translated_paragraph = translate_text(paragraph, lang)
            
            # Append the translated data to the list
            translated_data.append({
                'id_str': id_str,
                'tweet': translated_tweet,
                'paragraph': translated_paragraph,
                'label': label,
                'language': lang
            })
            # Delay to reduce the rate of API requests
            time.sleep(0.5)
    # Convert the list of translated data into a DataFrame
    return pd.DataFrame(translated_data)

#Specifying the target languages for translation, including English:
#Top 10 languages in Twitter: 
#English, Japanese, Spanish, Portuguese, Arabic, French, Indonesian, Russian, Turkish, Hindi
#languages = ['en', 'ja', 'es', 'pt', 'ar', 'fr', 'id', 'ru', 'tr', 'hi']

#We tried translating to all 10 languages but since the volume of paragraphs is much more than articles, it is taking more than 8-9 hours to just iterate over less than 30-40% of the data and then we are getting server connection error in ORC. Hence we decreased the translations to 2 languages other than english on paragraph level.

# List of languages to translate to [fixing it at 2 languages -  Japanese and Spanish.
languages = ['en', 'ja', 'es']

# Process each file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        print(f"Processing {file_name}...")  # Log the file being processed
        df = pd.read_csv(f'{input_dir}/{file_name}')  # Read the file
        translated_df = translate_dataset_to_languages(df, languages)  # Translate the dataset
        output_file_name = file_name.replace('.csv', '_translated.csv')  # Name of the translated file
        translated_df.to_csv(f'{output_dir}/{output_file_name}', index=False)  # Save the translated dataset
        print(f"Saved translated data to {output_file_name}")  # Log saving of the file

print("Translation of all batches completed.")  # Log completion of the translation process