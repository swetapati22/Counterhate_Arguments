Read Me File for the `prepare_data.py` Script:

This script is responsible for preparing and pre-processing tweet and article data for machine learning tasks. It includes functions for cleaning the text data, tokenizing using a transformer model tokenizer (RoBERTa or LongFormer based on the specified level), and creating DataLoader objects for training, validation, and testing.

## Dataset:

The script is designed to work with CSV files that contain tweets and corresponding article texts. The CSV should have at least two columns: one with tweets and one with articles, each row representing a paired instance.

In the case of the `paragraph.csv` file, it is expected to have at least the following columns:
- A unique identifier for the tweet, often labeled as 'id_str' or similar.
- The 'tweet' column with the original tweet's text content.
- The 'paragraph' column with a paragraph from an article that relates to or counters the tweet.
- A 'label' column that indicates whether the paragraph was deemed an effective counter to the tweet (usually with binary labels such as 1 for effective and 0 for ineffective).

Similarly, the `articles.csv` file should include columns for the unique identifier, the 'tweet' column, and an 'article' column that contains the counterargument in the form of an article's text, along with the corresponding 'label' column.

## Preprocessing Steps:

- Dropping any rows with null values in the specified columns.
- Removing URLs, user mentions, and non-ASCII characters from the text.
- Converting text to lowercase and stripping whitespace.
- Splitting data into training, validation, and testing sets with stratification.

### Role of Tokenizers in `prepare_data.py`:

The `prepare_data.py` script uses the `AutoTokenizer` from the `transformers` library to process the text from tweets and articles or paragraphs. The code uses the fllowing two tokenizers:

1. **For Paragraph-Level Data (`--level paragraph`)**:
   - The script utilizes the `roberta-base` tokenizer, which is well-suited for shorter texts like individual paragraphs. RoBERTa is a robustly optimized BERT variant that has been pre-trained on a large corpus of text and is known for its effectiveness in various NLP tasks.

2. **For Article-Level Data (`--level article`)**:
   - The script selects the `allenai/longformer-base-4096` tokenizer, designed to handle longer text sequences such as full-length articles. The LongFormer is an extended version of the standard transformer models that can process much longer sequences, which is essential for article-level text classification.

### Tokenization Process:

During tokenization, the following steps are performed:

- **Segmentation**: Text is split into words, subwords, or symbols that are meaningful in the language model's training corpus. For instance, "don't" may be split into "do" and "n't".
  
- **Conversion to Input IDs**: Each token is converted into an integer ID that corresponds to its entry in the model's vocabulary. This step transforms the text into a numerical form that the model can work with.
  
- **Attention Mask Generation**: Since the model requires input sequences of the same length, shorter sequences are padded with zeros. An attention mask is created to let the model differentiate between the actual data and the padding.
  
- **Sequence Truncation/Padding**: The tokenizer ensures that all sequences are truncated or padded to a uniform length, as defined by the `max_seq_length` parameter.

These tokenization steps convert raw text data into a structured format that transformer-based models, such as RoBERTa or LongFormer, can effectively process. The tokenized data preserves the linguistic content and context required for the models to learn and make predictions accurately.

## Function to Encode Data:

The `bert_encode` function converts text data into a format that can be processed by transformer models, including:
- Tokenizing text and converting it to input IDs.
- Generating attention masks for the tokens.
- Truncating or padding sequences to a uniform length.

## DataLoaders Creation:

After pre-processing and encoding the data, the script creates DataLoaders for the training, validation, and testing sets that will be used to feed data into the machine learning model during training and evaluation.

## Execution:

Run the script in a Python environment with the following command structure:

python prepare_data.py --csv-file <path_to_csv_file> --level <level> --output-dir <output_directory>
    
## Output:

The script saves three files to the specified output directory:
- `train.pth`: DataLoader for the training set.
- `valid.pth`: DataLoader for the validation set.
- `test.pth`: DataLoader for the test set.

## Instructions:

- Ensure the CSV file is properly formatted and located at the specified path.
- The script can be adjusted for different batch sizes or sequence lengths as needed.
- Running this script is a prerequisite for training and evaluating the machine learning models.

## Requirements:

- Python 3.6 or above.
- PyTorch, Transformers, Pandas, and Sklearn libraries.
- Adequate disk space in the output directory to store DataLoader objects.
- Other requirements are mentioned in requirements.txt file

Kindly ensure that you have the necessary Python environment and dependencies installed before running the script.
