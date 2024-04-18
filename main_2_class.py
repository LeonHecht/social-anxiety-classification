#!/usr/bin/env python
# coding: utf-8

# # Performing 2 classifications in a row

# ## Print Start time

# In[8]:


import time
print("------------------------------------------------")
print("Start-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")


# ## Specify model

# In[9]:


# model = 'distilbert-base-uncased'
# model = 'roberta-base'
# model = 'bert-large-uncased'
# model = 'xlnet-large-cased'
model = 'xlm-roberta-large'
# model = 'microsoft/deberta-v2-xxlarge'


# ## Load df

# In[10]:


import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]

def filter_sentences(row):
    # Assuming keywords are separated by commas and possibly spaces
    keywords = [lemmatizer.lemmatize(word) for word in row['keyword'].replace(' ', '').split(',')]
    text = row['text']
    sentences = sent_tokenize(text)
    
    # Filter sentences that contain at least one lemmatized keyword
    filtered_sentences = set()  # Use a set to prevent duplicates
    for index, sentence in enumerate(sentences):
        words = lemmatize_words(word_tokenize(sentence))
        if any(keyword in words for keyword in keywords):
            # Add previous sentence if it exists
            if index > 0:
                filtered_sentences.add(sentences[index - 1])
            # Add current sentence
            filtered_sentences.add(sentence)
            # Add next sentence if it exists
            if index < len(sentences) - 1:
                filtered_sentences.add(sentences[index + 1])

    return ' '.join(sorted(filtered_sentences)) if filtered_sentences else text  # Return original text if no keywords found


print("Reading data...")
df = pd.read_csv('data/SMM4H_2024_Task3_Training_1800.csv', usecols=['id', 'keyword', 'text', 'label'])
df_val = pd.read_csv('data/SMM4H_2024_Task3_Validation_600.csv', usecols=['id', 'keyword', 'text', 'label'])
print("Data read...")

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply the function to filter sentences in the text
df['text'] = df.apply(filter_sentences, axis=1)
df_val['text'] = df_val.apply(filter_sentences, axis=1)

print(df)


# ## Add keywords

# In[11]:


def add_keywords(df_, model):
    if model == 'distilbert-base-uncased' or model == 'roberta-base' or model == 'bert-large-uncased' or model == 'microsoft/deberta-v2-xxlarge':
        sep_token = '[SEP]'
    elif model == 'xlnet-large-cased':
        sep_token = '<sep>'
    elif model == 'xlm-roberta-large':
        sep_token = '</s>'
    
    df_['text'] = df_['text'] + f" {sep_token} " + df_['keyword']
    df_.drop(columns=['keyword'], inplace=True)
    return df_

df = add_keywords(df, model)


# ## Clean text

# In[12]:


# import emoji library
import emoji


def clean_text(text):
    import re
    # Perform emoji to text conversion
    text = emoji.demojize(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters and numbers
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['text'] = df['text'].apply(clean_text)


# ## Get df for Classification 1: 0 vs rest

# In[13]:


# Map class 1, 2, 3 to 1
df['label_1'] = df['label'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
df_val['label_1'] = df_val['label'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# print class count
print(df['label_1'].value_counts())


# ## Split data for first classification

# In[14]:


from sklearn.model_selection import train_test_split


train_texts, val_texts, y_train, y_val = train_test_split(
    df['text'], df['label_1'],
    test_size=0.3, random_state=42
)

test_texts = df_val['text']
y_test = df_val['label_1']


# ## Run model

# In[15]:


from models import tune_transformer

print("------------------------------------")
print("Model:", model)
print("------------------------------------")

print("Converting train, val and test texts to csv...")
train_texts.to_csv('data/train_texts.csv', index=False, header=False)
val_texts.to_csv('data/val_texts.csv', index=False, header=False)
test_texts.to_csv('data/test_texts.csv', index=False, header=False)

test_pred_labels = tune_transformer.run(model, 2, train_texts, val_texts, test_texts, y_train, y_val, y_test)

# replace original test labels with predicted labels
df_val['label_pred'] = test_pred_labels

# # save the dataframe with predicted labels to a csv file
# print("Saving predictions to csv...")
# df_val.to_csv('data/prediction_task3.tsv', sep='\t', index=False)


# ## Print End Time

# In[ ]:


import time
print("------------------------------------------------")
print("End-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")


# # Run 2nd classification

# ## Model for 2nd Classification

# In[ ]:


# model = 'distilbert-base-uncased'
# model = 'roberta-base'
# model = 'bert-large-uncased'
model = 'xlnet-large-cased'
# model = 'xlm-roberta-large'
# model = 'microsoft/deberta-v2-xxlarge'


# In[ ]:


# create 2nd df dropping the class 0
df_2 = df[df['label_1'] == 1]

# create 2nd df_val dropping the class 0 concerning 'label_pred'
df_val_2 = df_val[df_val['label_pred'] == 1]

print(df_2['label'].value_counts())

remapped_labels = {1: 0, 2: 1, 3: 2}

df_2['label'] = df_2['label'].map(remapped_labels)
df_val_2['label'] = df_val_2['label'].map(remapped_labels)

print(df_2['label'].value_counts())


# ## Split data for 2nd classification

# In[ ]:


from sklearn.model_selection import train_test_split


train_texts, val_texts, y_train, y_val = train_test_split(
    df_2['text'], df_2['label'],
    test_size=0.3, random_state=42
)

test_texts = df_val_2['text']
y_test = df_val_2['label']


# ## Run Model

# In[ ]:


from models import tune_transformer

print("------------------------------------")
print("Model:", model)
print("------------------------------------")

print("Converting train, val and test texts to csv...")
train_texts.to_csv('data/train_texts.csv', index=False, header=False)
val_texts.to_csv('data/val_texts.csv', index=False, header=False)
test_texts.to_csv('data/test_texts.csv', index=False, header=False)

test_pred_labels = tune_transformer.run(model, 3, train_texts, val_texts, test_texts, y_train, y_val, y_test)

remapped_labels2 = {0: 1, 1: 2, 2: 3}
# replace original test labels with predicted labels
df_val_2['label_pred2'] = test_pred_labels
df_val_2['label_pred2'] = df_val_2['label_pred2'].map(remapped_labels2)

# # save the dataframe with predicted labels to a csv file
# print("Saving predictions to csv...")
# df_val.to_csv('data/prediction_task3.tsv', sep='\t', index=False)


# ## Combine predictions

# In[ ]:


# Step 1: Convert 'rest' predictions to None
combined_predictions = ['0' if prediction == 0 else None for prediction in df_val['label_pred']]

# Step 2: Map 'rest' indices to the second step predictions
rest_indices = [i for i, prediction in enumerate(combined_predictions) if prediction is None]
for i, prediction in zip(rest_indices, df_val_2['label_pred2']):
    combined_predictions[i] = prediction  # Map to the original labels for class 1, 2, and 3

# Step 3: Update the DataFrame
df['predictions'] = combined_predictions

# Print classficiation report based on df['predictions'] and df['label']
from sklearn.metrics import classification_report
print(classification_report(df_val['label'], df['predictions'], target_names=['0', '1', '2', '3']))

