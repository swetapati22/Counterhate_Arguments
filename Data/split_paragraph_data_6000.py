import pandas as pd
import os

# Path to the input CSV file
input_file = 'paragraphs.csv'

# Directory where the split files will be saved
output_dir = 'split_files'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the entire dataset
df = pd.read_csv(input_file)

# Define the size of each batch
batch_size = 6000

# Calculate the number of batches needed
num_batches = (len(df) + batch_size - 1) // batch_size

# Loop through each batch and save it as a separate CSV file
for i in range(num_batches):
    start_index = i * batch_size
    end_index = start_index + batch_size
    df_batch = df.iloc[start_index:end_index]
    # Save each batch file within the new directory
    df_batch.to_csv(f'{output_dir}/para{i+1}.csv', index=False)

print(f'Successfully split the data into {num_batches} files in the directory {output_dir}.')
