#!/bin/bash

# Stop execution if any command fails
set -e

# Install required Python packages
# pip install -r requirements.txt

# Prepare the data
run prepare_data.py --csv-file ../Data/articles.csv --level article --output-dir ../Dataloaders/

# Train the model
run train.py --data-dir ../Dataloaders/ --level article --output-dir ../Output/

# Test the model
run test.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/

# Error Analysis
run error_analysis.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/

