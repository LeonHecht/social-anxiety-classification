#!/usr/bin/env python
# coding: utf-8

# ## Print Start time

# In[1]:


import time
print("------------------------------------------------")
print("Start-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")


# In[2]:


import os
print("Does df-save-path exist:", os.path.exists('data/augmented_train_dfs'))


# ## Load df

# In[3]:


import pandas as pd

df = pd.read_csv('data/SMM4H_2024_Task3_Training_1800.csv', usecols=['keyword', 'text', 'label'])

print(df)


# ## Clean text

# In[4]:


def clean_text(text):
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)


# ## Add Keywords to text

# In[5]:


df['text'] = df['text'] + " Keywords: " + df['keyword']


# ## Split data

# In[6]:


from sklearn.model_selection import train_test_split

# First, split the data into a training set and a temporary set (which will be further split into validation and test sets)
train_texts, temp_texts, y_train, temp_labels = train_test_split(df['text'], df['label'], test_size=0.4, random_state=42)

# Next, split the temporary set into validation and test sets
val_texts, test_texts, y_val, y_test = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)


# ## Cut Classes to X texts

# In[7]:


# Create a DataFrame from the training texts and labels
train_df = pd.DataFrame({'text': train_texts, 'label': y_train})

# Sample 200 texts from each class (or as many as are available for classes with fewer than 200 examples)
sampled_dfs = []
for label in train_df['label'].unique():
    class_sample_size = min(len(train_df[train_df['label'] == label]), 200)
    sampled_dfs.append(train_df[train_df['label'] == label].sample(n=class_sample_size, random_state=42))

# Concatenate the samples to create a balanced training DataFrame
train_df = pd.concat(sampled_dfs, ignore_index=True)


# In[8]:


# import matplotlib.pyplot as plt

# df_plot = balanced_train_df.copy()

# label_mapping = {1: 'positive', 2: 'neutral', 3: 'negative', 0: 'unrelated'}
# df_plot['label'] = df_plot['label'].map(label_mapping)

# # Contar el número de publicaciones en cada categoría
# class_counts = df_plot['label'].value_counts()
# print(class_counts)

# # Crear un gráfico de barras
# plt.figure(figsize=(8, 6))
# class_counts.plot(kind='bar')
# plt.title('Distribución de clases')
# plt.xlabel('Clase')
# plt.ylabel('Número de publicaciones')
# plt.xticks(rotation=0)
# plt.show()


# ## Backtranslate

# In[9]:


backtranslate = False
# Save the augmented training dataframe to a CSV file
train_df_path = 'data/augmented_train_dfs/train_df_plus_backtranslated_class_1_3.csv'

if backtranslate:

    from utils import backtranslation

    for label in {1, 3}:
        print(f"Backtranslating class {label}...")
        # Backtranslate and augment the data for underrepresented classes
        selected_texts = train_df[train_df['label'] == label]['text']
        print(f"length texts of label {label}", len(selected_texts))
        augmented_texts = backtranslation.backtranslate(selected_texts.to_list())
        augmented_df = pd.DataFrame({'text': augmented_texts, 'label': [label] * len(augmented_texts)})
        augmented_df.to_csv(f'data/augmented_train_dfs/backtranslated_class_{label}.csv', index=False)
        train_df = pd.concat([train_df, augmented_df])

    # Check the new class distribution after backtranslation
    print("Class distribution after backtranslation:", train_df['label'].value_counts())

    train_df.to_csv(train_df_path, index=False)


# ## Load train_df from csv

# In[10]:


# # If we don't backtranslate, load the existing augmented training DataFrame
# if not backtranslate:
#     train_df = pd.read_csv(train_df_path)
#     print("Class distribution:", train_df['label'].value_counts())


# ## Split train_df into texts and labels

# In[11]:


# Now you can extract the texts and labels
train_texts = train_df['text']
print("Train texts balanced", train_texts)
y_train = train_df['label']
print("y_train balanced", y_train)


# ## Run Model

# In[17]:


# from models import tune_transformer

# # model = 'DistilBert'
# model = 'RoBERTa'

# print("------------------------------------")
# print("Model:", model)
# print("------------------------------------")

# if model == 'DistilBert':
#     tune_transformer.run('distilbert-base-uncased', train_texts, val_texts, test_texts, y_train, y_val, y_test)
# elif model == 'RoBERTa':
#     tune_transformer.run('roberta-base', train_texts, val_texts, test_texts, y_train, y_val, y_test)

from models import feed_forward
import pandas as pd

feed_forward.run(train_texts.append(val_texts, ignore_index=True), test_texts, y_train.append(y_val, ignore_index=True), y_test)


# ## Print End Time

# In[ ]:


import time
print("------------------------------------------------")
print("End-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")

