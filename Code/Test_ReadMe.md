Read Me File for the `test.py` Script:

This script evaluates a pre-trained machine learning model on a test dataset and calculates various performance metrics such as accuracy, precision, recall, and F1 score. It is designed to be used after the model has been trained using the `train.py` script.

## Model Evaluation:

The `test.py` script loads the test dataset and the pre-trained model, then proceeds with the evaluation phase which includes:

- Predicting classes for the test data using the pre-trained model.
- Calculating performance metrics to assess the quality of the model.

### Performance Metrics:

- **Accuracy**: The proportion of correct predictions over the total number of cases evaluated.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives for each class.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives for each class.
- **F1 Score**: The weighted average of Precision and Recall, used as a combined metric for binary classification tasks.
- **Weighted F1-Score**: The F1 Score calculated for each label, and then their average is weighted by the number of true instances for each label.

## Execution:

To execute the `test.py` script, use the following command structure:

python test.py --data-dir ./Dataloaders --trained-model-dir ./Output --output-dir ./Output


## Output:

After running the script, you will receive:

- A printed summary of the performance metrics on the console.
- A `.npy` file saved to the output directory, containing the model's predictions on the test set.

## Prerequisites:

The model must be trained and its parameters saved using the `train.py` script prior to evaluation. Ensure that the test dataset has been properly prepared and tokenized by the `prepare_data.py` script.

## Requirements:

- Python 3.6 or above.
- PyTorch and Transformers libraries, as well as the Scikit-learn library for computing metrics.
- A compatible CUDA environment if running on GPU to accelerate the computation.

## Notes:

- Verify that the `trained_model` directory contains the correct model files before execution.
- Check that the correct paths are specified for the data and output directories.
- The script handles potential numerical stability issues, such as converting logits to predictions in a way that avoids the `AxisError`.

Ensure all dependencies and environment settings are correctly configured to successfully run the evaluation without any issues.


