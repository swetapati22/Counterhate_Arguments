# Importing necessary libraries
import argparse  
import os  
import torch  
import numpy as np  
import time  
import datetime 
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score  # Library for computing classification metrics
from transformers import AutoModelForSequenceClassification  # Importing class for sequence classification model from Hugging Face Transformers library

# Setting device for computation (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to calculate flat accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # Getting the predicted labels
    labels_flat = labels.flatten()  # Flattening the actual labels
    return np.sum(pred_flat == labels_flat) / len(labels_flat)  # Computing accuracy

# Function to format time
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))  # Rounding elapsed time to nearest integer
    return str(datetime.timedelta(seconds=elapsed_rounded))  # Formatting elapsed time

# Function to calculate precision, recall, and F1-score for each class and weighted F1-score
def calculate_scores(preds, labels):
    precision = precision_score(labels, preds, average=None)  # Calculating precision for each class
    recall = recall_score(labels, preds, average=None)  # Calculating recall for each class
    f1 = f1_score(labels, preds, average=None)  # Calculating F1-score for each class
    weighted_f1 = f1_score(labels, preds, average='weighted')  # Calculating weighted F1-score
    
    # Storing results in a dictionary
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

# Function to test the model and evaluate performance
def test(model, test_dataloader):
    model.eval()  # Setting model to evaluation mode
    model.to(DEVICE)  # Moving model to appropriate device
    all_preds = []  # List to store all predictions
    all_label_ids = []  # List to store all actual labels
    total_eval_loss = 0  # Initializing total evaluation loss
    t0 = time.time()  # Start time

    # Iterating over batches in the test dataloader
    for batch in test_dataloader:
        b_input_ids = batch[0].to(DEVICE)  # Input IDs of the batch
        b_input_mask = batch[1].to(DEVICE)  # Input mask of the batch
        b_labels = batch[2].to(DEVICE)  # Labels of the batch

        with torch.no_grad():  # Disabling gradient calculation
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  # Forward pass
            loss = outputs.loss  # Calculating loss
            logits = outputs.logits  # Predicted logits

        total_eval_loss += loss.item()  # Accumulating total evaluation loss
        logits = logits.detach().cpu().numpy()  # Detaching and moving logits to CPU
        label_ids = b_labels.to('cpu').numpy()  # Moving labels to CPU

        all_preds.extend(logits)  # Extending all_preds list with logits
        all_label_ids.extend(label_ids)  # Extending all_label_ids list with label IDs

    # Fix for AxisError: ensure all_preds is a 2D array before applying argmax
    all_preds = np.concatenate([p[np.newaxis, :] for p in all_preds], axis=0)
    all_label_ids = np.array(all_label_ids)

    # Use argmax to convert logits to predicted labels
    preds = np.argmax(all_preds, axis=1)

    # Calculate average scores using the calculate_scores function
    scores = calculate_scores(preds, all_label_ids)

    # Calculating average evaluation loss and validation time
    avg_val_loss = total_eval_loss / len(test_dataloader)
    validation_time = format_time(time.time() - t0)

    # Print evaluation results
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    
    return preds, scores  # Returning predictions and scores

# Main function
def main():
    # Creating argument parser object
    parser = argparse.ArgumentParser(description=__doc__)  
    # Adding argument for data directory
    parser.add_argument("--data-dir", required=False, default="./Dataloaders", help="Location of data files.")  
    # Adding argument for trained model directory
    parser.add_argument("--trained-model-dir", required=False, default="./Output", help="Location of the saved trained model.")  
    # Adding argument for output directory
    parser.add_argument("--output-dir", required=False, default="./Output", help="Output directory to save the predictions.")  
    # Parsing command line arguments
    args = parser.parse_args()  
    
    # Checking if trained model directory exists
    if not os.path.isdir(args.trained_model_dir):
        raise Exception(f'Trained model directory "{args.trained_model_dir}" does not exist.')

    # Loading test data
    test_dataloader = torch.load(os.path.join(args.data_dir, 'test.pth'))

    # Loading trained model
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.trained_model_dir, "trained_model"))

    # Testing the model and obtaining predictions and scores
    preds, scores = test(model, test_dataloader)
    
    # Saving predictions as .npy file
    np.save(os.path.join(args.output_dir, 'predictions.npy'), preds)
    
    # Printing scores
    print(scores)

# Entry point of the script
if __name__ == "__main__":
    main()
