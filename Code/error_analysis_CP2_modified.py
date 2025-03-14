import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('test_with_predictions.csv')

# Create a new column to check if each prediction was correct
df['correct_prediction'] = df['predicted_label'] == df['label']

# Analyze only the errors
errors_df = df[df['correct_prediction'] == False].copy()

# Add columns for text length to avoid SettingWithCopyWarning
errors_df.loc[:, 'tweet_length'] = errors_df['tweet'].apply(len)
errors_df.loc[:, 'article_length'] = errors_df['article'].apply(len)

# Save the errors to a new CSV file
errors_df.to_csv('mispredicted_examples.csv', index=False)

# Display basic statistics about text lengths in errors
print("Error Tweet Length Statistics:")
print(errors_df['tweet_length'].describe())


print("Error Article Length Statistics:")
print(errors_df['article_length'].describe())

# Visualize the distributions of tweet and article lengths
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(errors_df['tweet_length'], bins=20, color='red', alpha=0.7)
plt.title('Distribution of Tweet Lengths in Errors')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(errors_df['article_length'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Article Lengths in Errors')
plt.xlabel('Article Length')
plt.ylabel('Frequency')
plt.show()

# Print some examples of errors
print("\nExamples of Misclassified Texts:")
print(errors_df[['tweet', 'article', 'label', 'predicted_label']].head())
