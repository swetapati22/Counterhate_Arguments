# Importing necessary libraries
import argparse
import os
import torch
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from transformers import AutoModelForSequenceClassification

# Setting device for computation to use GPU if available, otherwise default to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to calculate flat accuracy from predictions and true labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to format elapsed time into a string (hours:minutes:seconds)
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate precision, recall, and F1-score for each class and the weighted F1-score across all classes
def calculate_scores(preds, labels):
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)
    f1 = f1_score(labels, preds, average=None)
    weighted_f1 = f1_score(labels, preds, average='weighted')
    
    results = {
        "Precision Class 0": round(precision[0], 2),
        "Recall Class 0": round(recall[0], 2),
        "F1-Score Class 0": round(f1[0], 2),
        "Precision Class 1": round(precision[1], 2),
        "Recall Class 1": round(recall[1], 2),
        "F1-Score Class 1": round(f1[1], 2),
        "Weighted F1-Score": round(weighted_f1, 2)
    }
    return results

# Function to evaluate the model performance on the test dataset and perform error analysis
def test(model, test_dataloader):
    model.eval()  # Set the model to evaluation mode
    model.to(DEVICE)  # Move the model to the appropriate device
    all_preds = []  # Store all predictions
    all_label_ids = []  # Store all actual labels
    total_eval_loss = 0  # Total loss for all batches
    t0 = time.time()  # Record the start time

    incorrect_samples = []  # List to store details of incorrect predictions

    for batch in test_dataloader:
        b_input_ids = batch[0].to(DEVICE)  # Batch input IDs
        b_input_mask = batch[1].to(DEVICE)  # Batch input masks
        b_labels = batch[2].to(DEVICE)  # Batch labels

        with torch.no_grad():  # No need to track gradients
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        all_preds.extend(logits)
        all_label_ids.extend(label_ids)

        preds = np.argmax(logits, axis=1)
        errors = np.where(preds != label_ids)[0]
        for i in errors:
            incorrect_samples.append((b_input_ids[i].to('cpu').numpy(), preds[i], label_ids[i], logits[i]))

    all_preds = np.concatenate([p[np.newaxis, :] for p in all_preds], axis=0)
    all_label_ids = np.array(all_label_ids)

    preds = np.argmax(all_preds, axis=1)

    scores = calculate_scores(preds, all_label_ids)

    avg_val_loss = total_eval_loss / len(test_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    
    return preds, scores, incorrect_samples

# Main function to setup arguments and run test
def main():
    # Using the module's docstring if available, or providing a default description
    description = __doc__ if __doc__ is not None else "Default description for the script."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", required=False, default="./Dataloaders", help="Location of data files.")
    parser.add_argument("--trained-model-dir", required=False, default="./Output", help="Location of the saved trained model.")
    parser.add_argument("--output-dir", required=False, default="./Output", help="Output directory to save the predictions.")
    args = parser.parse_args()

    if not os.path.isdir(args.trained_model_dir):
        raise Exception(f'Trained model directory "{args.trained_model_dir}" does not exist.')

    print(f"Loading data from {os.path.join(args.data_dir, 'test.pth')}")
    test_dataloader = torch.load(os.path.join(args.data_dir, 'test.pth'))
    print(f"Data loaded: {len(test_dataloader)} batches.")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.trained_model_dir, "trained_model"))
    print("Model loaded successfully.")

    preds, scores, incorrect_samples = test(model, test_dataloader)

    np.save(os.path.join(args.output_dir, 'predictions.npy'), preds)
    np.save(os.path.join(args.output_dir, 'incorrect_samples.npy'), np.array(incorrect_samples, dtype=object))
    
    # Save incorrect samples to CSV
    df = pd.DataFrame(incorrect_samples, columns=['Input IDs', 'Predicted Label', 'True Label', 'Logits'])
    df.to_csv(os.path.join(args.output_dir, 'incorrect_samples1.csv'), index=False)
    
    print(scores)
    print(f"Number of incorrect samples: {len(incorrect_samples)}")
    
    print("Incorrect Samples stored in incorrect_samples.csv");
    print(df.info())

if __name__ == '__main__':
    main()