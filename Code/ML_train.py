# Importing necessary libraries
import argparse 
import torch  
import os  
from transformers import (  # Importing functions and classes from the Hugging Face Transformers library
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import precision_recall_fscore_support  
import random 
import numpy as np  
import time  
import datetime  

# Setting constants
EPOCHS = 6  # epochs for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device for computation (GPU if available, otherwise CPU)
SEED = 42  # Random seed for reproducibility

# Function to calculate flat accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # Getting the predicted labels
    labels_flat = labels.flatten()  # Flattening the actual labels
    return np.sum(pred_flat == labels_flat) / len(labels_flat)  # Computing accuracy

# Function to format time
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))  # Rounding elapsed time to nearest integer
    return str(datetime.timedelta(seconds=elapsed_rounded))  # Formatting elapsed time

# Function to train the model
def train(train_dataloader, validation_dataloader, model, scheduler, optimizer):
    training_stats = []  # List to store training statistics
    total_t0 = time.time()  # Start time of training

    # Loop over each epoch
    for epoch_i in range(EPOCHS):
        print(f"\n======== Epoch {epoch_i + 1} / {EPOCHS} ========")  
        print("Training...") 

        t0 = time.time()  # Start time of current epoch
        total_train_loss = 0  # Initializing total training loss
        model.train()  # Setting model to training mode

        # Loop over each batch in the training set
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)  # Computing elapsed time for current batch
                print(f"  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.")  # Printing progress

            b_input_ids = batch[0].to(DEVICE)  # Input IDs of the batch
            b_input_mask = batch[1].to(DEVICE)  # Input mask of the batch
            b_labels = batch[2].to(DEVICE)  # Labels of the batch

            model.zero_grad()  # Resetting gradients
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  # Forward pass
            loss = outputs.loss  # Calculating loss
            logits = outputs.logits  # Predicted logits
            total_train_loss += loss.item()  # Accumulating total training loss
            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients to avoid exploding gradients
            optimizer.step()  # Optimizing parameters
            scheduler.step()  # Updating learning rate scheduler

        avg_train_loss = total_train_loss / len(train_dataloader)  # Calculating average training loss
        training_time = format_time(time.time() - t0)  # Calculating total training time

        print(f"\n  Average training loss: {avg_train_loss:.2f}")  
        print(f"  Training epoch took: {training_time}")  

        print("\nRunning Validation...")  
        t0 = time.time()  # Start time of validation

        model.eval()  # Setting model to evaluation mode
        total_eval_accuracy = 0  # variable for total validation accuracy
        total_eval_loss = 0  # variable for total validation loss
        total_eval_precision = 0  # variable for total precision
        total_eval_recall = 0  # variable for total recall
        total_eval_f1 = 0  # variable for total F1-score

        # Validation loop
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(DEVICE)  # Input IDs of the batch
            b_input_mask = batch[1].to(DEVICE)  # Input mask of the batch
            b_labels = batch[2].to(DEVICE)  # Labels of the batch

            with torch.no_grad():  # Disabling gradient calculation
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  # Forward pass
                loss = outputs.loss  # Calculating loss
                logits = outputs.logits  # Predicted logits

            total_eval_loss += loss.item()  # Accumulating total validation loss
            logits = logits.detach().cpu().numpy()  # Detaching and moving logits to CPU
            label_ids = b_labels.to("cpu").numpy()  # Moving labels to CPU

            total_eval_accuracy += flat_accuracy(logits, label_ids)  # Accumulating total validation accuracy

            # Calculate precision, recall, f1-score
            precision, recall, f1, _ = precision_recall_fscore_support(label_ids, np.argmax(logits, axis=1), average='binary')
            total_eval_precision += precision  # Accumulating total precision
            total_eval_recall += recall  # Accumulating total recall
            total_eval_f1 += f1  # Accumulating total F1-score

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)  # Calculating average validation accuracy
        avg_val_precision = total_eval_precision / len(validation_dataloader)  # Calculating average precision
        avg_val_recall = total_eval_recall / len(validation_dataloader)  # Calculating average recall
        avg_val_f1 = total_eval_f1 / len(validation_dataloader)  # Calculating average F1-score
        avg_val_loss = total_eval_loss / len(validation_dataloader)  # Calculating average validation loss
        validation_time = format_time(time.time() - t0)  # Calculating validation time

        # Print validation results
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Precision: {avg_val_precision:.2f}")
        print(f"  Validation Recall: {avg_val_recall:.2f}")
        print(f"  Validation F1-Score: {avg_val_f1:.2f}")
        print(f"  Validation took: {validation_time}")

        # Record training statistics
        training_stats.append({
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Valid. Precision": avg_val_precision,
            "Valid. Recall": avg_val_recall,
            "Valid. F1-Score": avg_val_f1,
            "Training Time": training_time,
            "Validation Time": validation_time,
        })

    # Print training completion
    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")

# Main function
def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser()  # Creating argument parser object
    # Adding argument for data directory
    parser.add_argument("--data-dir", required=False, default="./Dataloaders", help="Location of data files.") 
     # Adding argument for level of data
    parser.add_argument("--level", required=True, help="The level to work with, either 'paragraph' or 'article'.") 
    # Adding argument for output directory
    parser.add_argument("--output-dir", required=False, default="./Output", help="Output directory to save the trained model.") 
    # Parsing command line arguments
    args = parser.parse_args()  

    # Checking validity of level argument
    if args.level not in ["paragraph", "article"]:
        raise Exception("Level must equal either 'paragraph' or 'article'.")

    # Loading data
    train_dataloader = torch.load(os.path.join(args.data_dir, "train.pth"))  # Loading training data
    validation_dataloader = torch.load(os.path.join(args.data_dir, "valid.pth"))  # Loading validation data

    # Loading or initializing model based on level
    if args.level == "paragraph":
        # Loading pre-trained RoBERTa model for paragraphs
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)  
    else:
        # Loading pre-trained Longformer model for articles
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)  

    # Moving model to appropriate device
    model.to(DEVICE)

    # Initializing optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)  
     # Total number of training steps
    total_steps = len(train_dataloader) * EPOCHS 
    # Initializing linear scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  

    # Training the model
    train(train_dataloader, validation_dataloader, model, scheduler, optimizer)

    # Saving the trained model
    model.save_pretrained(os.path.join(args.output_dir, "trained_model"))  # Saving the trained model

# Entry point of the script
if __name__ == "__main__":
    main()
