```markdown
# Text Length Statistics Calculator

## Description
This Python script calculates the minimum and maximum lengths of text entries within a specified column of a CSV file. It is primarily used for analyzing the length of texts such as articles or paragraphs in datasets.

## Features
- Calculates and prints the minimum and maximum lengths of text in a specific column.
- Handles missing data by removing rows where the text data is NaN.

## Usage
1. **Prepare your dataset**: Ensure your data is in a CSV format with at least one text column.
2. **Modify the script**: Adjust the `file_path` and `text_column` parameters in the function call to match your dataset's specifics.
   - `file_path`: Path to the CSV file.
   - `text_column`: Column name containing the text to analyze.
3. **Run the script**: Execute the script using Python to see the results printed:
```bash
python compute_length_stats.py
```