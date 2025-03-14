# Implementing Authentic Counterhate Arguments

This project replicates and extends the study "Finding Authentic Counterhate Arguments: A Case Study with Public Figures", focusing on the identification and validation of counterhate arguments against individual-targeted online hate speech.

Inorder to re-implement the same, do the following steps: 

1. Clone the repository:
   ```sh
   git clone https://github.com/swabhipapneja/Implementing_Counter-hate_Paragraph.git
   ```

2. Navigate to the project directory:
   ```sh
   cd Implementing_Counter-hate_Paragraph
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Running prepare_data.py script for loading data specific to model inputs:
   ```sh
   python prepare_data.py --csv-file <path_to_csv_file> --level --output-dir <output_directory>
   ```
    Where:
    file_path: path of the .csv data file.
    level: the level to work on, either paragraph or article levels.
    output_path: directory to save the processed data (default is ./Dataloaders/).

   This script is responsible for preparing and pre-processing tweet and article data. It includes functions for cleaning the text data, tokenizing using a transformer model tokenizer (RoBERTa or LongFormer based on the specified level), and creating DataLoader objects for training, validation, and testing.
   
   Example for Article:
   ```sh
   python prepare_data.py --csv-file ../Data/articles.csv --level article --output-dir ../Dataloaders/
   ```
   Example for Paragraph:
   ```sh
   python prepare_data.py --csv-file ../Data/paragraphs.csv --level paragraph --output-dir ../Dataloaders/
   ```
   After Running these scripts respective Dataloaders folders will be created having these 3 files:
      1. train.pth: DataLoader for the training set.
      2. valid.pth: DataLoader for the validation set.
      3. test.pth: DataLoader for the test set.

6.  Implement the training loop for a specified number of epochs, evaluate the model on a validation dataset, and records the training and validation statistics for analysis:
   ```sh
   python train.py --data-dir {processed_data_path} --level {level} --output-dir {output_path}
   ```
Where:
processed_data_path: directory of the processed data (default is ./Dataloaders/).
level: the level to work on, either paragraph or article levels.
output_path: directory to save the trained model (default is ./Output/).

   Example for Article:
   ```sh
   python train.py --data-dir ../Dataloaders/ --level article --output-dir ../Output/
   ```
   Example for Paragraph:
   ```sh
   python train.py --data-dir ../Dataloaders/ --level paragraph --output-dir ../Output/
   ```


6. Evaluating a pre-trained machine learning model on a test dataset and calculating various performance metrics such as accuracy, precision, recall, and F1 score:
   ```sh
   python test.py --data-dir ./Dataloaders --trained-model-dir ./Output --output-dir ./Output
   ```
Where:
processed_data_path: directory of the processed data (default is ./Dataloaders/).
trained_model_path: directory of the trained model (default ./Output/).
output_path: directory to save the predictions (default is ./Output/).

   Example for Article:
   ```sh
   python test.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
   ```
   Example for Paragraph:
   ```sh
   python test.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
   ```

7. Performing Error Analysis:
   ```sh
   python test.py --data-dir ./Dataloaders --trained-model-dir ./Output --output-dir ./Output
   ```

   Example for Article:
   ```sh
   python error_analysis.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
   ```
   Example for Paragraph:
   ```sh
   python error_analysis.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
   ```

We have also provided 2 files that will help you run your code and replicate the results just by running these files.:
article_script.sh and paragraph_script.sh

## Citation

```
Papneja, S., & Pati, S. (2024). Implementing Authentic Counterhate Arguments. GitHub Repository, https://github.com/swabhipapneja/Implementing_Counter-hate_Paragraph
```

The original research paper:
```
@inproceedings{albanyan-etal-2023-finding,
    title = "Finding Authentic Counterhate Arguments: A Case Study with Public Figures",
    author = "Albanyan, Abdullah  and Hassan, Ahmed  and Blanco, Eduardo",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.855",
    doi = "10.18653/v1/2023.emnlp-main.855",
    pages = "13862--13876",
}
```
