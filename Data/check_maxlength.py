import pandas as pd

def compute_length_statistics(file_path, text_column):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop any rows where the text column is NaN to avoid errors in length calculation
    df = df.dropna(subset=[text_column])
    
    # Calculate the length of each article/paragraph
    df['length'] = df[text_column].apply(len)
    
    # Find the minimum and maximum lengths
    min_length = df['length'].min()
    max_length = df['length'].max()
    
    print(f"Minimum length of {text_column} in {file_path}: {min_length}")
    print(f"Maximum length of {text_column} in {file_path}: {max_length}")

# Example usage
compute_length_statistics('articles.csv', 'article')
compute_length_statistics('paragraphs.csv', 'paragraph')
