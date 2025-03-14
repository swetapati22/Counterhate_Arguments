```markdown
# Translated Files Merger

## Description
This Python script consolidates multiple translated CSV files into a single CSV file. It is designed to merge files that have been previously translated and stored in a directory.

## Features
- Merges multiple CSV files into one.
- Saves the merged file in the same directory.

## Usage
1. **Prepare your files**: Place all translated CSV files within a single directory.
2. **Configure the script**:
   - `input_dir`: Directory containing the translated files. Each file should end with '_translated.csv'.
3. **Run the script**: Execute the script using Python:
```bash
python merge_script.py
```
4. **Check the output**: The merged file named `paragraphs_translated_2_Batches.csv` will be saved in the same directory.

## Output
- A single merged CSV file named `paragraphs_translated_2_Batches.csv` will be created in the `translated_files` directory.