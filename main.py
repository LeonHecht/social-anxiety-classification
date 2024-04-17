#!/usr/bin/env python
# coding: utf-8

# ## Print Start time

# In[13]:


import time
print("------------------------------------------------")
print("Start-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")


# In[14]:


import os
print("Does df-save-path exist:", os.path.exists('data/augmented_train_dfs'))


# ## Specify deploying mode

# In[15]:


deploying = False


# ## Specify Model

# In[16]:


# model = 'distilbert-base-uncased'
# model = 'roberta-base'
# model = 'bert-large-uncased'
model = 'xlnet-large-cased'
# model = 'xlm-roberta-large'
# model = 'microsoft/deberta-v2-xxlarge'


# ## Load df

# In[18]:


import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

def filter_sentences(row):
    # Assuming keywords are separated by commas and possibly spaces
    keywords = [lemmatizer.lemmatize(word) for word in row['keyword'].replace(' ', '').split(',')]
    text = row['text']
    sentences = sent_tokenize(text)
    
    # Filter sentences that contain at least one lemmatized keyword
    filtered_sentences = []
    for index, sentence in enumerate(sentences):
        words = lemmatize_words(word_tokenize(sentence))
        if any(keyword in words for keyword in keywords):
            # Add previous sentence if it exists
            if index > 0:
                filtered_sentences.append(sentences[index - 1])
            # Add current sentence
            filtered_sentences.append(sentence)
            # Add next sentence if it exists
            if index < len(sentences) - 1:
                filtered_sentences.append(sentences[index + 1])

    return ' '.join(filtered_sentences) if filtered_sentences else text  # Return original text if no keywords found

print("Reading data...")
df = pd.read_csv('data/SMM4H_2024_Task3_Training_1800.csv', usecols=['id', 'keyword', 'text', 'label'])
df_val = pd.read_csv('data/SMM4H_2024_Task3_Validation_600.csv', usecols=['id', 'keyword', 'text', 'label'])
print("Data read...")
# Keep the validation data apart when deploying
if not deploying:
    df = pd.concat([df, df_val], ignore_index=True)

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply the function to filter sentences in the text
df['text'] = df.apply(filter_sentences, axis=1)
if deploying:
    df_val['text'] = df_val.apply(filter_sentences, axis=1)

print(df)


# ## Specify augmentation mode

# In[4]:


augmenting = True


# ## Add Keywords

# In[5]:


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

if not augmenting:
    df = add_keywords(df, model)


# ## Clean text

# In[7]:


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


# ## Split data

# In[8]:


from sklearn.model_selection import train_test_split


if not augmenting and not deploying:
    # ----- Without Keywords (for training) -----
    # First, split the data into a training set and a temporary set (which will be further split into validation and test sets)
    train_texts, temp_texts, y_train, temp_labels = train_test_split(
        df['text'], df['label'],
        test_size=0.3, random_state=42
    )

    # Next, split the temporary set into validation and test sets
    val_texts, test_texts, y_val, y_test = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5, random_state=42
    )

elif augmenting:
    # ----- With Keywords (for augmenting) -----
    # First, split the data into a training set and a temporary set (which will be further split into validation and test sets)
    train_texts, temp_texts, train_keywords, temp_keywords, y_train, temp_labels = train_test_split(
        df['text'], df['keyword'], df['label'], test_size=0.3, random_state=42
    )

    # Next, split the temporary set into validation and test sets
    val_texts, test_texts, val_keywords, test_keywords, y_val, y_test = train_test_split(
        temp_texts, temp_keywords, temp_labels, test_size=0.5, random_state=42
    )
    
elif deploying:
    train_texts, val_texts, y_train, y_val = train_test_split(
        df['text'], df['label'],
        test_size=0.3, random_state=42
    )

    test_texts = df_val['text']
    y_test = df_val['label']

    


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

# In[9]:


if not augmenting:
    # ----- Without Keywords (for training) -----
    # Create a DataFrame from the training texts and labels
    train_df = pd.DataFrame({'text': train_texts, 'label': y_train})
else:  
    # ----- With Keywords (for augmenting) -----
    # Create a DataFrame from the training texts and labels
    train_df = pd.DataFrame({'text': train_texts, 'label': y_train, 'keyword': train_keywords})

# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution before cutting:\n", class_counts)


# ## Augment train_df by paraphased texts

# In[10]:


if not augmenting:
    ## ----------- Comment code if augmenting data ------------

    paraphrased_1_df = pd.read_csv('data/augmented_train_dfs/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
    paraphrased_3_df = pd.read_csv('data/augmented_train_dfs/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
    
    # Add keywords to paraphrased dfs
    paraphrased_1_df = add_keywords(paraphrased_1_df, model)
    paraphrased_3_df = add_keywords(paraphrased_3_df, model)

    print("Paraphrased 1 df:\n", paraphrased_1_df)
    
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


# ## Augment data by ChatGPT 4 generated texts

# In[11]:


if not augmenting:
    ## ----------- Comment code if augmenting data ------------

    chatGPT4_1_df = pd.read_csv('data/augmented_train_dfs/ChatGPT4_texts_class_1.csv', usecols=['text', 'label', 'keyword'])
    chatGPT4_3_df = pd.read_csv('data/augmented_train_dfs/ChatGPT4_texts_class_3.csv', usecols=['text', 'label', 'keyword'])

    # Add keywords to ChatGPT4 dfs
    chatGPT4_1_df = add_keywords(chatGPT4_1_df, model)
    chatGPT4_3_df = add_keywords(chatGPT4_3_df, model)

    # Print chatGPT4 dfs
    print("chatGPT4_1_df:\n", chatGPT4_1_df)

    # merge df with paraphrased dfs
    train_df = pd.concat([train_df, paraphrased_1_df, paraphrased_3_df], ignore_index=True)


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

# In[12]:


if not augmenting:
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


paraphrase = True
# Save the augmented training dataframe to a CSV file
# train_df_path = 'data/augmented_train_dfs/train_df_plus_paraphased_class_1_3.csv'

if paraphrase:

    from utils import paraphrase

    for label in {1, 2, 3}:
        print(f"Paraphrasing class {label}...")
        # Paraphrase and augment the data for underrepresented classes
        selected_texts = train_df[train_df['label'] == label]['text']
        selected_keywords = train_df[train_df['label'] == label]['keyword']
        print(f"length texts of label {label}", len(selected_texts))
        augmented_texts = paraphrase.paraphrase(selected_texts.to_list(), selected_keywords.to_list())
        # augmented_texts = [["t1", "t2", "t3", "t4"], ["t12", "t22", "t32", "t42"]]
        for i in range(len(augmented_texts[0])):
            print("i", i)
            curr_texts = [augmented_texts[j][i] for j in range(len(augmented_texts))]
            print(curr_texts)
            augmented_df = pd.DataFrame({'text': curr_texts, 'label': [label] * len(curr_texts), 'keyword': selected_keywords.to_list()})
            # augmented_df = pd.DataFrame({'text': curr_texts, 'label': [label] * len(curr_texts)})
            augmented_df.to_csv(f'data/augmented_train_dfs/Paraphrase{i+1}/paraphrased_class_{label}.csv', index=False)
        # train_df = pd.concat([train_df, augmented_df])

    # Check the new class distribution after paraphrasing
    # print("Class distribution after paraphrasing:", train_df['label'].value_counts())

    # train_df.to_csv(train_df_path, index=False)


# ## Print train_df class distribution after augmentation

# In[ ]:


## ----------- Comment code if augmenting data ------------

# # Contar el número de publicaciones en cada categoría
# class_counts = train_df['label'].value_counts()
# print("Class distribution after augmentation:\n", class_counts)


# ## Run Model

# In[ ]:


from models import tune_transformer

print("------------------------------------")
print("Model:", model)
print("------------------------------------")

if not augmenting:

    print("Converting train, val and test texts to csv...")
    train_texts.to_csv('data/train_texts.csv', index=False, header=False)
    val_texts.to_csv('data/val_texts.csv', index=False, header=False)
    test_texts.to_csv('data/test_texts.csv', index=False, header=False)

    test_pred_labels = tune_transformer.run(model, train_texts, val_texts, test_texts, y_train, y_val, y_test)
    
    # replace original test labels with predicted labels
    df_val['label'] = test_pred_labels

    # save the dataframe with predicted labels to a csv file
    print("Saving predictions to csv...")
    df_val.to_csv('data/prediction_task3.tsv', sep='\t', index=False)


# ## Print End Time

# In[ ]:


import time
print("------------------------------------------------")
print("End-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")

