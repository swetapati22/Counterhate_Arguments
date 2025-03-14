Read Me File for the `train.py` Script:

This script is designed to train a machine learning model using the Transformer architecture for sequence classification tasks. It supports training for different levels of text granularity, namely paragraphs and articles, which are paired with tweets to identify potential counterhate speech.

## Training Process:

The script implements the training loop for a specified number of epochs, evaluates the model on a validation dataset, and records the training and validation statistics for analysis.

### Key Components:

- **Model Initialization**: Depending on the level of granularity (`paragraph` or `article`), the script initializes a RoBERTa or LongFormer model respectively.
- **Optimizer and Scheduler**: AdamW optimizer with a learning rate scheduler for effective learning rate adjustments during training.
- **Training Loop**: Runs through the dataset in batches, performing forward and backward passes, and updating model weights.
- **Evaluation**: Calculates precision, recall, and F1 scores to gauge model performance on the validation set.

## Metrics:

- Precision (P): Measures the accuracy of the positive predictions.
- Recall (R): Captures the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
- F1 Score (F): The harmonic mean of precision and recall, providing a balance between the two.

## Execution:

Run the script in a Python environment with the following command structure:

python train.py --data-dir ./Dataloaders --level article --output-dir ./Output


## Output:

- The trained model is saved to the specified output directory.
- Training and validation performance statistics are printed to the console for each epoch.

## Prerequisites:

Before running the `train.py` script, ensure that the `prepare_data.py` script has been executed to preprocess and tokenize the input data.

## Requirements:

- Python 3.6 or above.
- PyTorch, Transformers, and Scikit-learn libraries.
- CUDA-capable hardware if training on GPU for enhanced performance.

## Notes:

- It is recommended to run the training script on a machine with adequate computational resources to handle the intensive training process of Transformer models.
- Make sure that the dataloaders saved by the `prepare_data.py` script are accessible in the specified data directory.
- Monitor the training and validation statistics to adjust hyperparameters or to perform early stopping if necessary for optimal performance.
