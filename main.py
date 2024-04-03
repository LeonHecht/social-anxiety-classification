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


# ## Specify Model

# In[ ]:


# model = 'DistilBert'
# model = 'RoBERTa'
# model = 'Bert-Large'
model = 'XLNet-Large'
# model = 'XLM-Roberta-Large'


# ## Load df

# In[3]:


import pandas as pd

df = pd.read_csv('data/SMM4H_2024_Task3_Training_1800.csv', usecols=['id', 'keyword', 'text', 'label'])
df_val = pd.read_csv('data/SMM4H_2024_Task3_Validation_600.csv', usecols=['id', 'keyword', 'text', 'label'])

# Contar el número de publicaciones en cada categoría
class_counts = df_val['label'].value_counts()
print("Class distribution of val df:\n", class_counts)

df = pd.concat([df, df_val], ignore_index=True)
print(df)


# ## Specify augmentation mode

# In[4]:


augmenting = False


# ## Add Keywords

# In[5]:


if not augmenting:
    if model == 'DistilBert' or model == 'RoBERTa' or model == 'Bert-Large':
        sep_token = '[SEP]'
    elif model == 'XLNet-Large':
        sep_token = '<sep>'
    elif model == 'XLM-Roberta-Large':
        sep_token = '</s>'
    
    df['text'] = df['text'] + f" {sep_token} " + df['keyword']
    df.drop(columns=['keyword'], inplace=True)


# ## Clean text

# In[6]:


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

# df['text'] = df['text'].apply(clean_text)


# ## Split data

# In[ ]:


from sklearn.model_selection import train_test_split


if not augmenting:
    # ----- Without Keywords (for training) -----
    # First, split the data into a training set and a temporary set (which will be further split into validation and test sets)
    train_ids, temp_ids, train_texts, temp_texts, y_train, temp_labels = train_test_split(
        df['id'], df['text'], df['label'],
        test_size=0.3, random_state=42
    )

    # Next, split the temporary set into validation and test sets
    val_ids, test_ids, val_texts, test_texts, y_val, y_test = train_test_split(
        temp_ids, temp_texts, temp_labels,
        test_size=0.5, random_state=42
    )

else:
    # ----- With Keywords (for augmenting) -----
    # First, split the data into a training set and a temporary set (which will be further split into validation and test sets)
    train_texts, temp_texts, train_keywords, temp_keywords, y_train, temp_labels = train_test_split(
        df['text'], df['keyword'], df['label'], test_size=0.3, random_state=42
    )

    # Next, split the temporary set into validation and test sets
    val_texts, test_texts, val_keywords, test_keywords, y_val, y_test = train_test_split(
        temp_texts, temp_keywords, temp_labels, test_size=0.5, random_state=42
    )


# In[ ]:


# import matplotlib.pyplot as plt

# df_plot = df.copy()

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


# ## Get train_df

# In[ ]:


if not augmenting:
    # ----- Without Keywords (for training) -----
    # Create a DataFrame from the training texts and labels
    train_df = pd.DataFrame({'id': train_ids, 'text': train_texts, 'label': y_train})
    real_id = '3u2w5k'
    real_label = 0

    real_id2 = '8ugfyp'
    real_label2 = 3
else:  
    # ----- With Keywords (for augmenting) -----
    # Create a DataFrame from the training texts and labels
    train_df = pd.DataFrame({'text': train_texts, 'label': y_train, 'keyword': train_keywords})

# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution before cutting:\n", class_counts)


# ## Augment train_df by paraphased texts

# In[ ]:


if not augmenting:
    ## ----------- Comment code if augmenting data ------------

    paraphrased_1_df = pd.read_csv('data/augmented_train_dfs/paraphrased_class_1.csv', usecols=['text', 'label'])
    paraphrased_3_df = pd.read_csv('data/augmented_train_dfs/paraphrased_class_3.csv', usecols=['text', 'label'])

    # merge df with paraphrased dfs
    train_df = pd.concat([train_df, paraphrased_1_df, paraphrased_3_df], ignore_index=True)


# ## Augment train_df by backtranslated texts

# In[ ]:


# if not augmenting:

    ## ----------- Comment code if augmenting data ------------

    # back_translated_1_df = pd.read_csv('data/augmented_train_dfs/backtranslated_class_1.csv', usecols=['text', 'label'])
    # back_translated_3_df = pd.read_csv('data/augmented_train_dfs/backtranslated_class_3.csv', usecols=['text', 'label'])

    # # merge df with backtranslated dfs
    # train_df = pd.concat([train_df, back_translated_1_df, back_translated_3_df], ignore_index=True)


# ## Cut Classes to X texts

# In[12]:


# from sklearn.utils import shuffle


# if not augmenting:
#     ## ----------- Comment code if augmenting data ------------

#     # Sample 200 texts from each class (or as many as are available for classes with fewer than 200 examples)
#     sampled_dfs = []
#     for label in train_df['label'].unique():
#         class_sample_size = min(len(train_df[train_df['label'] == label]), 181)
#         sampled_dfs.append(train_df[train_df['label'] == label].sample(n=class_sample_size, random_state=42))

#     # Concatenate the samples to create a balanced training DataFrame
#     train_df = pd.concat(sampled_dfs, ignore_index=True)
    
#     # Shuffle the DataFrame
#     train_df = shuffle(train_df, random_state=42)
#     print("Train df:\n", train_df)

#     # Assuming 'df' is your DataFrame and 'text' is the column name
#     rows_with_phrase = train_df[train_df['text'].str.contains("I want to experience young love")]

#     # Get the IDs of those rows
#     # ids = rows_with_phrase['id'].tolist()
#     # print("Real ID:", real_id)
#     # print("Train_df ids", ids)

#     # labels = rows_with_phrase['label'].tolist()
#     # print("Real label:", real_label)
#     # print("Train df labels", labels)

#     rows_with_phrase2 = train_df[train_df['text'].str.contains("I wrote this as a bit of a vent for myself")]

#     # Get the IDs of those rows
#     ids = rows_with_phrase2['id'].tolist()
#     print("Real ID:", real_id2)
#     print("Train_df ids", ids)

#     labels = rows_with_phrase2['label'].tolist()
#     print("Real label:", real_label2)
#     print("Train df labels", labels)


# ## Extract texts and labels from train_df

# In[ ]:


# Now you can extract the texts and labels
train_texts = train_df['text']
print("Train texts balanced", train_texts)
# print datatype of y train values
y_train = train_df['label']
print("Datatype of y_train", type(y_train))
print("y_train balanced", y_train)


# ## Print train_df class distribution after cutting/before augmentation

# In[ ]:


# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution after cutting:\n", class_counts)


# In[ ]:


# import matplotlib.pyplot as plt

# df_plot = train_df.copy()

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

# In[ ]:


backtranslate = False
# Save the augmented training dataframe to a CSV file
train_df_path = 'data/augmented_train_dfs/train_df_plus_backtranslated_class_1_3.csv'

if backtranslate:

    from utils import backtranslation

    for label in {1, 3}:
        print(f"Backtranslating class {label}...")
        # Backtranslate and augment the data for underrepresented classes
        selected_texts = train_df[train_df['label'] == label]['text']
        selected_keywords = train_df[train_df['label'] == label]['keyword']
        print(f"length texts of label {label}", len(selected_texts))
        augmented_texts = backtranslation.backtranslate_t5(selected_texts.to_list(), selected_keywords.to_list())
        augmented_df = pd.DataFrame({'text': augmented_texts, 'label': [label] * len(augmented_texts)})
        augmented_df.to_csv(f'data/augmented_train_dfs/backtranslated_t5_class_{label}.csv', index=False)
        train_df = pd.concat([train_df, augmented_df])

    # Check the new class distribution after backtranslation
    print("Class distribution after backtranslation:", train_df['label'].value_counts())

    train_df.to_csv(train_df_path, index=False)


# ## Paraphrase

# In[ ]:


paraphrase = False
# Save the augmented training dataframe to a CSV file
# train_df_path = 'data/augmented_train_dfs/train_df_plus_paraphased_class_1_3.csv'

if paraphrase:

    from utils import paraphrase

    for label in {1, 3}:
        print(f"Paraphrasing class {label}...")
        # Backtranslate and augment the data for underrepresented classes
        selected_texts = train_df[train_df['label'] == label]['text']
        selected_keywords = train_df[train_df['label'] == label]['keyword']
        print(f"length texts of label {label}", len(selected_texts))
        augmented_texts = paraphrase.paraphrase(selected_texts.to_list(), selected_keywords.to_list())
        augmented_df = pd.DataFrame({'text': augmented_texts, 'label': [label] * len(augmented_texts)})
        augmented_df.to_csv(f'data/augmented_train_dfs/paraphrased_class_{label}.csv', index=False)
        train_df = pd.concat([train_df, augmented_df])

    # Check the new class distribution after paraphrasing
    print("Class distribution after paraphrasing:", train_df['label'].value_counts())

    # train_df.to_csv(train_df_path, index=False)


# ## Print train_df class distribution after augmentation

# In[ ]:


## ----------- Comment code if augmenting data ------------

# # Contar el número de publicaciones en cada categoría
# class_counts = train_df['label'].value_counts()
# print("Class distribution after augmentation:\n", class_counts)


# ## Run Model

# In[ ]:


if not augmenting:
    from models import tune_transformer

    print("------------------------------------")
    print("Model:", model)
    print("------------------------------------")

    print("Converting train, val and test texts to csv...")
    train_texts.to_csv('data/train_texts.csv', index=False, header=False)
    val_texts.to_csv('data/val_texts.csv', index=False, header=False)
    test_texts.to_csv('data/test_texts.csv', index=False, header=False)

    if model == 'DistilBert':
        tune_transformer.run('distilbert-base-uncased', train_texts, val_texts, test_texts, y_train, y_val, y_test)
    elif model == 'RoBERTa':
        tune_transformer.run_optimization('roberta-base', train_texts, val_texts, test_texts, y_train, y_val, y_test)
    elif model == 'Bert-Large':
        tune_transformer.run('bert-large-uncased', train_texts, val_texts, test_texts, y_train, y_val, y_test)
    elif model == 'XLNet-Large':
        tune_transformer.run('xlnet-large-cased', train_texts, val_texts, test_texts, y_train, y_val, y_test)
    elif model == 'XLM-Roberta-Large':
        tune_transformer.run('xlm-roberta-large', train_texts, val_texts, test_texts, y_train, y_val, y_test)


# ## Print End Time

# In[ ]:


import time
print("------------------------------------------------")
print("End-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")

