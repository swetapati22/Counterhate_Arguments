```markdown
# Dataset Splitter

## Description
This Python script splits a large CSV file into multiple smaller CSV files based on a specified batch size.

## Features
- Splits a single dataset into manageable chunks.
- Saves each chunk into a separate CSV file in a specified directory.

## Usage
1. **Set up your dataset**: Place your large CSV file in the same directory as the script, or update the `input_file` path in the script.
2. **Configure the script**:
   - `input_file`: Path to the input CSV file.
   - `output_dir`: Directory where the split files will be saved.
   - `batch_size`: Number of rows per split file.
3. **Run the script**: Execute the script using Python:
```bash
python split_script.py
```
4. **Check output**: The split files will be saved in the specified output directory.

## Output
- The script will output multiple CSV files in the specified directory, named sequentially (e.g., `para1.csv`, `para2.csv`, etc.).