# Modifications to `prepare_data.py`

### Overview
This document outlines the modifications made from the original `prepare_data.py` to enhance text processing and accommodate multilingual data handling.

### Changes Implemented

1. **Normalization Enhancement**
- **Description:** Introduced Unicode normalization using `unicodedata.normalize('NFKC', x)`. This helps in representing characters with diacritical marks uniformly.
- **Impact:** Ensures consistency in processing text across various languages, aiding in the normalization of characters and diacritics.

2. **Updated Text Processing**
- **Description:** Enhanced URL removal and simplified the cleaning of mentions, retweets, and hashtags. Regex patterns were updated for robust handling of modern web text.
- **Impact:** Improves the reliability of text input into the model by removing extraneous web text elements efficiently.

3. **Handling Special Characters**
- **Description:** Expanded processing to handle HTML entities and special characters more effectively by replacing or removing them.
- **Impact:** Ensures the quality of text data remains high, crucial for accurate model input.

4. **Whitespace Normalization**
- **Description:** Normalized whitespace by condensing multiple spaces into a single space.
- **Impact:** Helps in standardizing text input for tokenization and maintaining correct token boundaries.

5. **Tokenizer Update**
- **Description:** Unified the tokenizer across different data levels to `xlm-roberta-base`, suitable for multilingual text.
- **Impact:** Simplifies the preprocessing pipeline and enhances support for multilingual data.

6. **Batch Size Adjustment**
- **Description:** Standardized batch sizes for all data levels when using the `xlm-roberta-base` tokenizer.
- **Impact:** Adjusts to the requirements of the new tokenizer, ensuring efficient data handling during training.

7. **Saving raw splits to CSV**
- **Description:** Saving raw splits of train, validation and test to CSV files.
- **Impact:** This helps in easier error analysis going forward.

### Connection to Next File:
- **Detail:** Outputs from this script (encoded data: input IDs and attention masks) are used to create TensorDataset objects. These datasets are subsequently loaded through DataLoaders in `train.py` for model training and validation.

### Expected Improvements:
These enhancements are designed to make the script more robust in handling diverse and noisy web text, particularly for multilingual contexts. The use of a powerful, unified tokenizer aligns with the goal of simplifying and standardizing the preprocessing steps. 



# Modifications in `train.py`

### General Overview:
The modified version of `train.py` incorporates changes to adapt the training script for a different pre-trained model and addresses enhanced performance settings. This adaptation is aimed at improving model performance and optimizing the training process.

### Changes Implemented

1. **Model Change**: Switched from using `roberta-base` and `allenai/longformer-base-4096` to `xlm-roberta-base` for both paragraph and article levels. This change introduces a model capable of handling multiple languages, enhancing the model's applicability in diverse linguistic contexts.
   
2. **Optimizer and Scheduler Configuration**: Maintained the use of `AdamW` optimizer but ensured it is optimized for `xlm-roberta-base`. The learning rate (`lr=1e-5`) and epsilon (`eps=1e-8`) remain optimized for stability in gradients during training.

3. **Hyperparameter Tuning**:
   - The epoch count (`EPOCHS`) is set to 6 to balance between adequate training time and model convergence.
   - A learning rate scheduler (`get_linear_schedule_with_warmup`) without warmup steps is used, ensuring the learning rate decreases linearly from the initial rate set by the optimizer.

### Connection to Next Steps:
The modifications prepare the model for improved performance in multilingual settings and ensure detailed monitoring of training progress, setting the stage for robust performance evaluation in subsequent testing phases.

### Expected Improvements:
These changes are expected to enhance model understanding of various languages and improve predictive accuracy and validation metrics across diverse datasets. The detailed logging and monitoring will aid in quicker identification and resolution of issues during training.




# Modifications to `test.py`

### Summary of Changes
The modified `test.py` script enhances evaluation functionality and improves data output handling. These changes facilitate a more detailed examination of the model's performance and provide additional utilities for integrating test results into further analysis processes.

### Changes Implemented

1. **Integration with Raw Data:**
   - **Loading Raw Data:** The script now loads a CSV file (`test_raw.csv`) that contains the raw test data. This is crucial for comparing the predicted results with the actual data.
   - **Appending Predictions:** After model evaluation, the predictions are appended to the loaded DataFrame. This allows for a direct comparison between predicted and actual labels within the same DataFrame.
   - **Saving Enhanced Data:** The DataFrame, now containing both original data and predictions, is saved back to a CSV file (`test_with_predictions.csv`). This file is useful for further analysis and review.

2. **Output Management:**
   - **.npy File Saving:** Predictions are also saved as a `.npy` file, offering compatibility with workflows that utilize NumPy arrays for further data manipulation and analysis.

3. **Enhanced Console Output:**
   - **Direct Scoring Outputs:** The script now prints detailed performance metrics, such as Precision, Recall, F1-Score for each class, and Weighted F1-Score, directly to the console. This allows users to quickly assess model performance without needing additional scripts.
   
### Connection to Next Steps:
These modifications are intended to streamline the testing phase of model development, making it easier to assess model effectiveness and prepare data for subsequent stages of analysis or presentation. The enhancements in output management particularly aid in maintaining a consistent and organized workflow during model evaluations.

### Expected Improvements:
By saving the predictions directly within the context of the original test data, anyone can proceed with statistical analysis or error analysis phases without needing additional data manipulation steps. 



# Modifications in `error_analysis.py`

### Changes Implemented

1. **Data Handling Enhancements:**
   - **Text Length Calculation:** Introduced explicit column operations to calculate the length of text for both tweets and articles directly within the DataFrame using Pandas' `.apply()` method. This change ensures the script avoids `SettingWithCopyWarning` and maintains good practices in data manipulation.
   - **Direct CSV Output:** Added functionality to save erroneous data points directly to a CSV file named `mispredicted_examples.csv`. This facilitates easy access and further analysis of misclassified cases without additional scripting.

2. **Visualization Improvements:**
   - **Histograms for Text Lengths:** Added histograms to visualize the distribution of tweet and article lengths where the model predictions were incorrect. This visual representation helps in understanding patterns in the data that may correlate with prediction errors.

3. **Error Reporting:**
   - **Enhanced Output Details:** Now outputs basic descriptive statistics for the lengths of tweets and articles in the error cases, providing immediate insights into potential issues like text length impacting model performance.

### Expected Improvements

1. **Enhanced Analytical Insights:**
   - By including direct calculations and visualizations of text lengths in errors, analysts can more readily identify if text length correlates with prediction errors, guiding potential model adjustments or preprocessing steps.

2. **Streamlined Error Analysis:**
   - Saving misclassified examples directly to a CSV file simplifies the workflow for reviewing and analyzing errors, making it easier to pinpoint specific cases for review or to use as examples in reports or further research.

3. **Immediate Access to Data Insights:**
   - Quick access to statistical descriptions and visualizations of error distributions enables faster hypothesis testing regarding error patterns, such as whether shorter or longer texts tend to be misclassified more often.



# New Files made for translations are as mentioned below:
- You will find the details on this files in their own README file in their respective folders.

## 2 main files: 
1.  **translate_article**
2.  **translate_paragraphs**

## Supporting files for translation:
1. **translate_paragraphs**
2. **split_paragraph_data_6000**
3. **merge_paragraphs**

