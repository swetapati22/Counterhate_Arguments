import os
import pandas as pd

# Directory where the translated files are stored
input_dir = 'translated_files'

# Merging all translated files into one DataFrame
all_files = [f'{input_dir}/{file}' for file in sorted(os.listdir(input_dir)) if file.endswith('_translated.csv')]
merged_df = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)

# Save the merged DataFrame to a single file
merged_df.to_csv(f'{input_dir}/paragraphs_translated_2_Batches.csv', index=False)
print("All translated batches have been merged and saved.")