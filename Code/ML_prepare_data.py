# Importing necessary libraries
import argparse 
import pandas as pd  
from sklearn.model_selection import train_test_split  # Library for splitting data into train and test sets
from transformers import AutoTokenizer  # Importing tokenizer from Hugging Face Transformers library
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler  
import os  
import warnings  
import unicodedata
warnings.filterwarnings('ignore')  

# Constant seed for reproducibility.
SEED = 42

# Function to preprocess tweets in the dataframe
def preprocess_dataframe_tweets(df, col):
    # Ensure the column exists and drop rows with missing values in this column
    df.dropna(subset=[col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create a new column for processed text to preserve the original text
    df[col + '_proc'] = df[col].apply(lambda x: unicodedata.normalize('NFKC', x))

    # Remove URLs
    df[col + '_proc'] = df[col + '_proc'].str.replace(r'http\S+', '', regex=True)

    # Remove RT, mentions, and hashtags
    df[col + '_proc'] = df[col + '_proc'].str.replace(r'(RT|rt)[ ]@[ ]\S+', '', regex=True)
    df[col + '_proc'] = df[col + '_proc'].str.replace(r'@\S+', '', regex=True)
    df[col + '_proc'] = df[col + '_proc'].str.replace(r'#\S+', '', regex=True)  # Remove hashtags if not needed

    # Handle or remove special HTML entities and other non-essential punctuation
    df[col + '_proc'] = df[col + '_proc'].str.replace(r'&lt;|&gt;|&amp;', ' ', regex=True)

    # Normalize whitespace
    df[col + '_proc'] = df[col + '_proc'].str.replace(r'\s+', ' ', regex=True)

    # Strip white spaces at both ends
    df[col + '_proc'] = df[col + '_proc'].str.strip()

# Function to encode tweets using BERT tokenizer
def bert_encode(df, tokenizer, exp, max_seq_length=256):
    input_ids = []  # List to store input IDs
    attention_masks = []  # List to store attention masks
    separator = " </s> "  # Separator for joining tokens
    for s in df[exp].values:
        sent = separator.join(filter(None, s))
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode
            add_special_tokens=True,  # Add special tokens <s> and </s>
            padding='max_length',  # Pad to maximum sequence length
            truncation=True,  # Truncate sequences if exceeding max length
            max_length=max_seq_length,  # Maximum sequence length
            return_attention_mask=True,  # Return attention mask
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Add the encoded sentence to the list
        input_ids.append(encoded_dict["input_ids"])

        # Add its attention mask (differentiates padding from non-padding)
        attention_masks.append(encoded_dict["attention_mask"])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {"input_word_ids": input_ids, "input_mask": attention_masks}

    return inputs

# Main function
def main():
    parser = argparse.ArgumentParser(description=__doc__)  # Creating argument parser object
    parser.add_argument(
        "--csv-file", required=True, help="Location of the data file in a .csv format."
    )  # Argument for CSV file location
    parser.add_argument(
        "--level",
        required=True,
        help="The level to work with, either 'paragraph' or 'article'.",
    )  # Argument for data level
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./Dataloaders",
        help="A directory to save the output files in.",
    )  # Argument for output directory
    args = parser.parse_args()  # Parsing command line arguments
    csv_file = args.csv_file

    if args.level not in ["paragraph", "article"]:
        raise Exception("Level must equal either 'paragraph' or 'article'.")
    else:
        level = args.level
        EXP = ["tweet", level]

    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)  # Reading CSV file into DataFrame
    else:
        raise Exception(f'CSV file "{csv_file}" not found.')

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  # Creating output directory if not exists

    # Splitting data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[EXP], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
    )

    df_train = X_train.join(y_train)
    df_val = X_val.join(y_val)
    df_test = X_test.join(y_test)
    
    # Saving raw splits to CSV
    df_train.to_csv(os.path.join(output_dir, "train_raw.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "validation_raw.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test_raw.csv"), index=False)

    for exp in EXP:
        preprocess_dataframe_tweets(df_train, col=exp)
        preprocess_dataframe_tweets(df_val, col=exp)
        preprocess_dataframe_tweets(df_test, col=exp)

    if level == "paragraph":
        BATCH_SIZE = 16  # Batch size for paragraph level
        tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-base", use_fast=True, normalization=True
        )  # Loading xlm-roberta-base tokenizer
    else:
        BATCH_SIZE = 24  # Batch size for article level
        tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-base", use_fast=True, normalization=True
        )  # Loading xlm-roberta-base tokenizer

    # Encoding and tokenizing data for training, validation, and testing
    tweet_train = bert_encode(df_train, tokenizer, EXP)
    tweet_train_labels = df_train["label"].astype(int)
    tweet_valid = bert_encode(df_val, tokenizer, EXP)
    tweet_valid_labels = df_val["label"].astype(int)
    tweet_test = bert_encode(df_test, tokenizer, EXP)
    tweet_test_labels = df_test["label"].astype(int)

    input_ids, attention_masks = tweet_train.values()

    # Combine the training inputs into a TensorDataset
    input_ids, attention_masks = tweet_train.values()
    labels = torch.tensor(tweet_train_labels.values, dtype=torch.long)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split
    input_ids, attention_masks = tweet_valid.values()
    labels = torch.tensor(tweet_valid_labels.values, dtype=torch.long)
    val_dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split
    input_ids, attention_masks = tweet_test.values()
    labels = torch.tensor(tweet_test_labels.values, dtype=torch.long)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create DataLoaders for training, validation, and testing
    # RandomSampler for training data, SequentialSampler for validation and testing data
    train_dataloader = DataLoader(
        train_dataset,  # Training samples
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=BATCH_SIZE,  # Batch size for training
    )

    validation_dataloader = DataLoader(
        val_dataset,  # Validation samples
        sampler=SequentialSampler(val_dataset),  # Read batches sequentially
        batch_size=BATCH_SIZE,  # Batch size for validation
    )

    testing_dataloader = DataLoader(
        test_dataset,  # Testing samples
        sampler=SequentialSampler(test_dataset),  # Read batches sequentially
        batch_size=BATCH_SIZE,  # Batch size for testing
    )

    # Saving DataLoaders to files
    torch.save(train_dataloader, os.path.join(output_dir, "train.pth"))
    torch.save(validation_dataloader, os.path.join(output_dir, "valid.pth"))
    torch.save(testing_dataloader, os.path.join(output_dir, "test.pth"))

# Entry point of the script
if __name__ == "__main__":
    main()
