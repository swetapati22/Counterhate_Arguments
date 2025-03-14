#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# #### In this notebook we perform error analysis on multi-linguality dimention based on 4 Experiments that we conducted:
# - Article Level
#     - Only Tweets
#     - Tweets with Articles
# - Paragraph Level 
#     - Only Tweets
#     - Tweets with Paragraphs

# # 1. Paragraphs (Tweet) - Misprediction Analysis

# In[2]:


#Load the translated dataset:
data1 = pd.read_csv('para1_translated.csv')


# In[3]:


#Load the test with predictions dataset:
data2 = pd.read_csv('ParaLevel_TweetOnly_test_with_predictions.csv')


# In[4]:


#Columns to merge on:
col1= 'tweet'
col2= 'paragraph'


# In[5]:


#Merge the datasets based on these columns:
merged_data = pd.merge(data2, data1[[col1, col2, 'language']], on=[col1, col2,], how='left')


# In[6]:


merged_data.head(5)


# In[7]:


#Total Test Data Size:
merged_data.shape


# In[8]:


#Create a new column to check if each prediction was correct:
merged_data['correct_prediction'] = merged_data['predicted_label'] == merged_data['label']


# In[9]:


#Analyze only the errors:
errors_df = merged_data[merged_data['correct_prediction'] == False].copy()


# In[10]:


errors_df.head(5)


# In[11]:


#Total number of mis-predictions:
errors_df.shape


# In[12]:


#Count the number of records per language:
language_counts = errors_df['language'].value_counts()


# In[13]:


#Plotting a bar graph:
plt.figure(figsize=(5, 3))
language_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Mis-predicted Records per Language')
plt.xlabel('Language')
plt.ylabel('Number of Mis-predicted Records')
plt.xticks(rotation=0)  
#Rotate labels to make them readable
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[14]:


#Choosing shades of green:
colors = ['#74c476', '#31a354', '#006d2c']
#Plotting a pie chart:
plt.figure(figsize=(5, 3))
plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Mis-prediction Distribution by Language')
plt.axis('equal')  
#Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[15]:


#Correct way to add columns for text length to avoid SettingWithCopyWarning
errors_df.loc[:, 'tweet_length'] = errors_df['tweet'].apply(len)
errors_df.loc[:, 'paragraph_length'] = errors_df['paragraph'].apply(len)


# In[16]:


#Display basic statistics about text lengths in errors:
print("Error Tweet Length Statistics:")
print(errors_df['tweet_length'].describe())


# In[17]:


print("Error paragraph Length Statistics:")
print(errors_df['paragraph_length'].describe())


# In[18]:


# Visualize the distributions of tweet and article lengths
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(errors_df['tweet_length'], bins=20, color='red', alpha=0.7)
plt.title('Distribution of Tweet Lengths in Errors')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')


# In[19]:


plt.subplot(1, 2, 2)
plt.hist(errors_df['paragraph_length'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Paragraph Lengths in Errors')
plt.xlabel('Paragraph Length')
plt.ylabel('Frequency')
plt.show()


# In[20]:


#Print some examples of errors
print("\nExamples of Misclassified Texts:")
errors_df[['tweet', 'paragraph', 'label', 'predicted_label']].head()


# In[21]:


# Save the errors to a new CSV file
errors_df.to_csv('mispredicted_examples_para_tweet_only.csv', index=False)


# # 2. Paragraphs (Tweet + Paragraph) - Misprediction Analysis

# In[22]:


#Load the translated dataset:
data1 = pd.read_csv('para1_translated.csv')


# In[23]:


#Load the test with predictions dataset:
data2 = pd.read_csv('ParaLevel_tweet_paragraph_with_predictions.csv')


# In[24]:


#Columns to merge on:
col1= 'tweet'
col2= 'paragraph'


# In[25]:


#Merge the datasets based on these columns:
merged_data = pd.merge(data2, data1[[col1, col2, 'language']], on=[col1, col2,], how='left')


# In[26]:


merged_data.head(5)


# In[27]:


#Total Test Data Size:
merged_data.shape


# In[28]:


#Create a new column to check if each prediction was correct:
merged_data['correct_prediction'] = merged_data['predicted_label'] == merged_data['label']


# In[29]:


#Analyze only the errors:
errors_df = merged_data[merged_data['correct_prediction'] == False].copy()


# In[30]:


errors_df.head(5)


# In[31]:


#Total number of mis-predictions:
errors_df.shape


# In[32]:


#Count the number of records per language:
language_counts = errors_df['language'].value_counts()


# In[33]:


#Plotting a bar graph:
plt.figure(figsize=(5, 3))
language_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Mis-predicted Records per Language')
plt.xlabel('Language')
plt.ylabel('Number of Mis-predicted Records')
plt.xticks(rotation=0)  
#Rotate labels to make them readable
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[34]:


#Choosing shades of green:
colors = ['#74c476', '#31a354', '#006d2c']
#Plotting a pie chart:
plt.figure(figsize=(5, 3))
plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Mis-prediction Distribution by Language')
plt.axis('equal')  
#Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[35]:


#Correct way to add columns for text length to avoid SettingWithCopyWarning
errors_df.loc[:, 'tweet_length'] = errors_df['tweet'].apply(len)
errors_df.loc[:, 'paragraph_length'] = errors_df['paragraph'].apply(len)


# In[36]:


#Display basic statistics about text lengths in errors:
print("Error Tweet Length Statistics:")
print(errors_df['tweet_length'].describe())


# In[37]:


print("Error paragraph Length Statistics:")
print(errors_df['paragraph_length'].describe())


# In[38]:


# Visualize the distributions of tweet and article lengths
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(errors_df['tweet_length'], bins=20, color='red', alpha=0.7)
plt.title('Distribution of Tweet Lengths in Errors')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')


# In[39]:


plt.subplot(1, 2, 2)
plt.hist(errors_df['paragraph_length'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Paragraph Lengths (Tweet+Paragraph) in Errors')
plt.xlabel('Paragraph Length')
plt.ylabel('Frequency')
plt.show()


# In[40]:


#Print some examples of errors
print("\nExamples of Misclassified Texts:")
errors_df[['tweet', 'paragraph', 'label', 'predicted_label']].head()


# In[41]:


# Save the errors to a new CSV file
errors_df.to_csv('mispredicted_examples_para_Tweet_and_Paragraph.csv', index=False)


# # 3. Articles (Tweet) - Misprediction Analysis

# In[42]:


#Load the translated dataset:
data1 = pd.read_csv('articles_translated.csv')


# In[43]:


#Load the test with predictions dataset:
data2 = pd.read_csv('ArticleLevel_TweetOnly_test_with_predictions.csv')


# In[44]:


#Columns to merge on:
col1= 'tweet'
col2= 'article'


# In[45]:


#Merge the datasets based on these columns:
merged_data = pd.merge(data2, data1[[col1, col2, 'language']], on=[col1, col2,], how='left')


# In[46]:


merged_data.head(5)


# In[47]:


#Total Test Data Size:
merged_data.shape


# In[48]:


#Create a new column to check if each prediction was correct:
merged_data['correct_prediction'] = merged_data['predicted_label'] == merged_data['label']


# In[49]:


#Analyze only the errors:
errors_df = merged_data[merged_data['correct_prediction'] == False].copy()


# In[50]:


errors_df.head(5)


# In[51]:


#Total number of mis-predictions:
errors_df.shape


# In[52]:


#Count the number of records per language:
language_counts = errors_df['language'].value_counts()


# In[53]:


#Plotting a bar graph:
plt.figure(figsize=(5, 3))
language_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Mis-predicted Records per Language')
plt.xlabel('Language')
plt.ylabel('Number of Mis-predicted Records')
plt.xticks(rotation=0)  # Rotate labels to make them readable
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[54]:


#Choosing shades of green:
colors = ['#74c476', '#31a354', '#006d2c']
#Plotting a pie chart:
plt.figure(figsize=(5, 3))
plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Mis-prediction Distribution by Language')
plt.axis('equal')  
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[57]:


#Correct way to add columns for text length to avoid SettingWithCopyWarning
errors_df.loc[:, 'tweet_length'] = errors_df['tweet'].apply(len)
errors_df.loc[:, 'article_length'] = errors_df['article'].apply(len)


# In[56]:


# # Save the errors to a new CSV file
# errors_df.to_csv('mispredicted_examples.csv', index=False)


# In[58]:


#Display basic statistics about text lengths in errors:
print("Error Tweet Length Statistics:")
print(errors_df['tweet_length'].describe())


# In[59]:


print("Error article Length Statistics:")
print(errors_df['article_length'].describe())


# In[60]:


# Visualize the distributions of tweet and article lengths
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(errors_df['tweet_length'], bins=20, color='red', alpha=0.7)
plt.title('Distribution of Tweet Lengths in Errors')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')


# In[61]:


plt.subplot(1, 2, 2)
plt.hist(errors_df['article_length'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Article Lengths in Errors')
plt.xlabel('Article Length')
plt.ylabel('Frequency')
plt.show()


# In[62]:


#Print some examples of errors
print("\nExamples of Misclassified Texts:")
errors_df[['tweet', 'article', 'label', 'predicted_label']].head()


# In[64]:


# Save the errors to a new CSV file
errors_df.to_csv('mispredicted_examples_article_Tweet_only.csv', index=False)

