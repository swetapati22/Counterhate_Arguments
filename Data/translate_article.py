import googletrans
import pandas as pd
from googletrans import Translator
from tqdm import tqdm

# Initializing
translator = Translator()

def translate_dataset_to_languages(dataframe, languages):
    translated_data = []

    # Iterating over each row in the DataFrame with a progress bar
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Translating tweets and articles"):
        id_str = row['id_str']
        label = row['label']
        tweet = row['tweet']
        article = row['article']

        # Translating both tweet and article to each target language
        for lang in languages:
            if lang == 'en':
                translated_tweet = tweet
                translated_article = article
            else:
                translated_tweet = translator.translate(tweet, src='en', dest=lang).text
                translated_article = translator.translate(article, src='en', dest=lang).text
            
            translated_data.append({
                'id_str': id_str,
                'tweet': translated_tweet,
                'article': translated_article,
                'label': label,
                'language': lang
            })

    # Converting the list of translated data into a DataFrame
    return pd.DataFrame(translated_data)

#Specifying the target languages for translation, including English:
#Top 10 languages in Twitter: 
#English, Japanese, Spanish, Portuguese, Arabic, French, Indonesian, Russian, Turkish, Hindi
languages = ['en', 'ja', 'es', 'pt', 'ar', 'fr', 'id', 'ru', 'tr', 'hi']

# Reading the Data to Translate
df = pd.read_csv('articles.csv')

# Creating the Translated dataframe
translated_df = translate_dataset_to_languages(df, languages)

# Printing the result
print(translated_df.head())

# Saving the augmented dataset with translations
translated_df.to_csv('articles_translated.csv', index=False)