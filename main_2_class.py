#!/usr/bin/env python
# coding: utf-8

# # Performing 2 classifications in a row

# ## Print Start time

# In[ ]:


import time
print("------------------------------------------------")
print("Start-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")


# ## Specify model

# In[ ]:


# model = 'distilbert-base-uncased'
# model = 'roberta-base'
# model = 'bert-large-uncased'
# model = 'xlnet-large-cased'
model = 'xlm-roberta-large'
# model = 'microsoft/deberta-v2-xxlarge'


# ## Load df

# In[ ]:


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


# ## Get separation token by model

# In[ ]:


def get_sep_token(model):
    if model == 'distilbert-base-uncased' or model == 'roberta-base' or model == 'bert-large-uncased' or model == 'microsoft/deberta-v2-xxlarge':
        sep_token = '[SEP]'
    elif model == 'xlnet-large-cased':
        sep_token = '<sep>'
    elif model == 'xlm-roberta-large':
        sep_token = '</s>'
    return sep_token


# ## Add keywords

# In[ ]:


def add_keywords(df_, model):
    sep_token = get_sep_token(model)
    
    df_['text'] = df_['text'] + f" {sep_token} Keyword: " + df_['keyword']
    df_.drop(columns=['keyword'], inplace=True)
    return df_

df = add_keywords(df, model)
df_val = add_keywords(df_val, model)


# ## Clean text

# In[ ]:


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
df_val['text'] = df_val['text'].apply(clean_text)


# ## Get df for Classification 1: 0 vs rest

# In[ ]:


# Map class 1, 2, 3 to 1
df['labels_mapped_0_1'] = df['label'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
df_val['labels_mapped_0_1'] = df_val['label'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# print class count
print(df['labels_mapped_0_1'].value_counts())
print(df_val['labels_mapped_0_1'].value_counts())


# ## Split data for first classification

# In[ ]:


from sklearn.model_selection import train_test_split


train_texts, val_texts, y_train, y_val = train_test_split(
    df['text'], df['labels_mapped_0_1'],
    test_size=0.3, random_state=42
)

test_texts = df_val['text']
y_test = df_val['labels_mapped_0_1']


# ## Get train_df

# In[ ]:


# Create a DataFrame from the training texts and labels
train_df = pd.DataFrame({'text': train_texts, 'label': y_train})

# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution before augmenting with paraphrased texts:\n", class_counts)


# ## Augment train_df by paraphased texts

# In[ ]:


paraphrased_1_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
paraphrased_1_df_2 = pd.read_csv('data/augmented_train_dfs/Paraphrase2/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])

paraphrased_2_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_2.csv', usecols=['text', 'label', 'keyword'])

paraphrased_3_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
paraphrased_3_df_2 = pd.read_csv('data/augmented_train_dfs/Paraphrase2/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
paraphrased_3_df_3 = pd.read_csv('data/augmented_train_dfs/Paraphrase3/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])

paraphrased_df = pd.concat([paraphrased_1_df_1, paraphrased_1_df_2, paraphrased_2_df_1, paraphrased_3_df_1, paraphrased_3_df_2, paraphrased_3_df_3], ignore_index=True)
paraphrased_df['label'] = 1

# Add keywords to paraphrased dfs
paraphrased_df = add_keywords(paraphrased_df, model)

# # Add sentiment to paraphrased dfs
# # Apply the function to get the compound sentiment score for each post
# paraphrased_df['vader_sentiment'] = paraphrased_df['text'].apply(get_vader_sentiment)
# paraphrased_df = add_vader_sentiment(paraphrased_df, model)

# merge df with paraphrased dfs
train_df = pd.concat([train_df, paraphrased_df], ignore_index=True)


# ## Extract texts and labels from train_df

# In[ ]:


shuffled_train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Now you can extract the texts and labels
train_texts = shuffled_train_df['text']
print("Train texts balanced", train_texts)
# print datatype of y train values
y_train = shuffled_train_df['label']
print("Datatype of y_train", type(y_train))
print("y_train balanced", y_train)


# ## Print train_df class distribution after cutting/before augmentation

# In[ ]:


# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution after cutting:\n", class_counts)


# ## Run model

# In[ ]:


from models import tune_transformer_GPU_0

print("------------------------------------")
print("Model:", model)
print("------------------------------------")

print("Converting train, val and test texts to csv...")
train_texts.to_csv('data/train_texts.csv', index=False, header=False)
val_texts.to_csv('data/val_texts.csv', index=False, header=False)
test_texts.to_csv('data/test_texts.csv', index=False, header=False)

test_pred_labels = tune_transformer_GPU_0.run_optimization(model, 2, train_texts, val_texts, test_texts, y_train, y_val, y_test)

# replace original test labels with predicted labels
df_val['label_pred_1'] = test_pred_labels

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


# # create 2nd df dropping the class 0
# df_2 = df[df['labels_mapped_0_1'] == 1].copy()
# # create 2nd df_val dropping the class 0 concerning 'label_pred_1' (the predictions already made)
# df_val_2 = df_val[df_val['labels_mapped_0_1'] == 1].copy()

# print("Before remapping labels:")
# print(df_2['label'].value_counts())
# print(df_val_2['label'].value_counts())

# remapped_labels = {1: 0, 2: 1, 3: 2}

# print("df_val['label']\n", df_val['label'])

# df_2['label'] = df_2['label'].map(remapped_labels).astype(int)
# df_val_2['label'] = df_val_2['label'].map(remapped_labels).astype(int)

# print("After remapping labels:")
# print(df_2['label'].value_counts())
# print(df_val_2['label'].value_counts())


# ## Split data for 2nd classification

# In[ ]:


from sklearn.model_selection import train_test_split


train_texts, val_texts, y_train, y_val = train_test_split(
    df.loc[df['labels_mapped_0_1'] == 1, 'text'], 
    df.loc[df['labels_mapped_0_1'] == 1, 'label'],
    test_size=0.3, random_state=42
)

test_texts = df_val.loc[df_val['label_pred_1'] == 1, 'text']


# ## Get train_df

# In[ ]:


# Create a DataFrame from the training texts and labels
train_df = pd.DataFrame({'text': train_texts, 'label': y_train})

# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution before augmenting with paraphrased texts:\n", class_counts)


# ## Augment train_df by paraphased texts

# In[ ]:


paraphrased_1_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
paraphrased_1_df_2 = pd.read_csv('data/augmented_train_dfs/Paraphrase2/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])

paraphrased_2_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_2.csv', usecols=['text', 'label', 'keyword'])

paraphrased_3_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
paraphrased_3_df_2 = pd.read_csv('data/augmented_train_dfs/Paraphrase2/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
paraphrased_3_df_3 = pd.read_csv('data/augmented_train_dfs/Paraphrase3/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])

paraphrased_df = pd.concat([paraphrased_1_df_1, paraphrased_1_df_2, paraphrased_2_df_1, paraphrased_3_df_1, paraphrased_3_df_2, paraphrased_3_df_3], ignore_index=True)

# Add keywords to paraphrased dfs
paraphrased_df = add_keywords(paraphrased_df, model)

# # Add sentiment to paraphrased dfs
# # Apply the function to get the compound sentiment score for each post
# paraphrased_df['vader_sentiment'] = paraphrased_df['text'].apply(get_vader_sentiment)
# paraphrased_df = add_vader_sentiment(paraphrased_df, model)

# merge df with paraphrased dfs
train_df = pd.concat([train_df, paraphrased_df], ignore_index=True)


# ## Extract texts and labels from train_df

# In[ ]:


shuffled_train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Now you can extract the texts and labels
train_texts = shuffled_train_df['text']
print("Train texts balanced", train_texts)
# print datatype of y train values
y_train = shuffled_train_df['label']
print("Datatype of y_train", type(y_train))
print("y_train balanced", y_train)


# ## Remap labels

# In[ ]:


remapped_labels = {1: 0, 2: 1, 3: 2}
y_train = y_train.map(remapped_labels).astype(int)
y_val = y_val.map(remapped_labels).astype(int)
# y_test = y_test.map(remapped_labels).astype(int)


# ## Run Model B

# In[ ]:


from models import tune_transformer_GPU_0

print("------------------------------------")
print("Model:", model)
print("------------------------------------")

print("Converting train, val and test texts to csv...")
train_texts.to_csv('data/train_texts.csv', index=False, header=False)
val_texts.to_csv('data/val_texts.csv', index=False, header=False)
test_texts.to_csv('data/test_texts.csv', index=False, header=False)

test_pred_labels = tune_transformer_GPU_0.run_optimization(model, 3, train_texts, val_texts, test_texts, y_train, y_val, None)

print("type test_pred_labels:", type(test_pred_labels))
remapped_labels2 = {0: 1, 1: 2, 2: 3}
test_pred_labels = pd.Series(test_pred_labels)
test_pred_labels = test_pred_labels.map(remapped_labels2).astype(int)

print("Len test_pred_labels:", len(test_pred_labels))
print("Len df_val['label']:", len(df_val['label']))
print("Len df_val['label_pred_1']:", len(df_val['label_pred_1']))

# replace original test labels with predicted labels
df_val.loc[df_val['label_pred_1'] == 1, 'label_pred_1'] = test_pred_labels.values

print("Info on df_val['label_pred_1']:\n", df_val['label_pred_1'].value_counts())

# print classification report
from sklearn.metrics import classification_report
print(classification_report(df_val['label'], df_val['label_pred_1']))

# # save the dataframe with predicted labels to a csv file
# print("Saving predictions to csv...")
# df_val.to_csv('data/prediction_task3.tsv', sep='\t', index=False)


# ## Combine predictions

# In[ ]:


# # Step 1: Convert 'rest' predictions to None
# combined_predictions = ['0' if prediction == 0 else None for prediction in df_val['label_pred_1']]

# # Step 2: Map 'rest' indices to the second step predictions
# rest_indices = [i for i, prediction in enumerate(combined_predictions) if prediction is None]
# for i, prediction in zip(rest_indices, df_val_2['label_pred_2']):
#     combined_predictions[i] = prediction  # Map to the original labels for class 1, 2, and 3

# # Step 3: Update the DataFrame
# df['final_predictions'] = combined_predictions

# # Print classficiation report based on df['final_predictions'] and df['label']
# from sklearn.metrics import classification_report
# print(classification_report(df_val['label'], df['final_predictions'], target_names=['0', '1', '2', '3']))

# Inicializar la columna final_prediction con las predicciones de la primera fase
# df_val['final_prediction'] = df_val['label_pred_1']

# print("Info on df_val['final_prediction']:\n", df_val['final_prediction'].value_counts())

# # Actualizar las predicciones de la segunda fase en la columna final_prediction
# df_val[df_val['labels_mapped_0_1'] == 1]['final_prediction'] = df_val[df_val['labels_mapped_0_1'] == 1]['label_pred_2']

# print("Info on df_val['final_prediction']:\n", df_val['final_prediction'].value_counts())

# # Print classficiation report based on df['final_predictions'] and df['label']
# from sklearn.metrics import classification_report
# print(classification_report(df_val['label'], df_val['final_prediction'], target_names=['0', '1', '2', '3']))

