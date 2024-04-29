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


# ## Specify deploying mode

# In[3]:


deploying = False
final_deploy = False

if final_deploy:
    print("-----------------FINAL DEPLOYMENT-----------------")


# ## Specify Model

# In[4]:


# model = 'distilbert-base-uncased'
# model = 'roberta-base'
# model = 'bert-large-uncased'
# model = 'xlnet-base-cased'
model = 'xlnet-large-cased'
# model = 'xlm-roberta-large'
# model = 'microsoft/deberta-v2-xxlarge'


# ## Load df

# In[6]:


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
# Keep the validation data apart when deploying
if not deploying:
    df = pd.concat([df, df_val], ignore_index=True)
if final_deploy:
    df_test = pd.read_csv('data/SMM4H_Task3_testposts.csv', usecols=['id', 'keyword', 'text'])

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply the function to filter sentences in the text
df['text'] = df.apply(filter_sentences, axis=1)
if deploying:
    df_val['text'] = df_val.apply(filter_sentences, axis=1)
elif final_deploy:
    df_test['text'] = df_test.apply(filter_sentences, axis=1)

print(df)


# ## Specify augmentation mode

# In[4]:


augmenting = True


# ## Get separation token by model

# In[ ]:


def get_sep_token(model):
    if model == 'distilbert-base-uncased' or model == 'roberta-base' or model == 'bert-large-uncased' or model == 'microsoft/deberta-v2-xxlarge' or 'albert' in model:
        sep_token = '[SEP]'
    elif 'xlnet' in model:
        sep_token = '<sep>'
    elif model == 'xlm-roberta-large':
        sep_token = '</s>'
    return sep_token


# ## Add VADER Sentiment to df

# In[ ]:


# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')

# # Initialize the VADER sentiment intensity analyzer
# sid = SentimentIntensityAnalyzer()

# # Function to get the compound sentiment score
# def get_vader_sentiment(text):
#     return sid.polarity_scores(text)['compound']

# def add_vader_sentiment(df_, model):
#     sep_token = get_sep_token(model)
    
#     df_['text'] = df_['text'] + f" {sep_token} Sentiment score: " + df_['vader_sentiment'].astype(str)
#     df_.drop(columns=['vader_sentiment'], inplace=True)
#     return df_

# # Apply the function to get the compound sentiment score for each post
# df['vader_sentiment'] = df['text'].apply(get_vader_sentiment)
# df = add_vader_sentiment(df, model)
# if deploying:
#     df_val['vader_sentiment'] = df_val['text'].apply(get_vader_sentiment)
#     df_val = add_vader_sentiment(df_val, model)


# ## Add Keywords

# In[5]:


def add_keywords(df_, model):
    sep_token = get_sep_token(model)
    
    df_['text'] = df_['text'] + f" {sep_token} Keyword: " + df_['keyword']
    df_.drop(columns=['keyword'], inplace=True)
    return df_

if not augmenting:
    df = add_keywords(df, model)
    if deploying:
        df_val = add_keywords(df_val, model)
    elif final_deploy:
        df_test = add_keywords(df_test, model)


# ## Clean text

# In[7]:


# import emoji library
import emoji


def clean_text(text):
    import re
    # Perform emoji to text conversion
    text = emoji.demojize(text)
    # Convert to lowercase
    # text = text.lower()
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters and numbers
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['text'] = df['text'].apply(clean_text)
if deploying:
    df_val['text'] = df_val['text'].apply(clean_text)
elif final_deploy:
    df_test['text'] = df_test['text'].apply(clean_text)


# ## Split data

# In[8]:


from sklearn.model_selection import train_test_split


if not deploying and not augmenting and not final_deploy:
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
elif deploying:
    train_texts, val_texts, y_train, y_val = train_test_split(
        df['text'], df['label'],
        test_size=0.3, random_state=42
    )

    test_texts = df_val['text']
    y_test = df_val['label']
elif augmenting:
    train_texts, val_texts, train_keywords, val_keywords, y_train, y_val = train_test_split(
    df['text'], df['keyword'], df['label'],
    test_size=0.3, random_state=42
    )

    test_texts = df_val['text']
    y_test = df_val['label']
elif final_deploy:
    train_texts, val_texts, y_train, y_val = train_test_split(
        df['text'], df['label'],
        test_size=0.3, random_state=42
    )

    test_texts = df_test['text']


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
    train_df = pd.DataFrame({'text': train_texts, 'label': y_train, 'keyword': train_keywords})

# Contar el número de publicaciones en cada categoría
class_counts = train_df['label'].value_counts()
print("Class distribution before augmenting with paraphrased texts:\n", class_counts)


# ## Augment train_df by paraphased texts

# In[10]:


import os


# if not augmenting:

#     # Base directory where paraphrase folders are located
#     base_dir = 'data/augmented_train_dfs/'

#     # List to hold all the dataframes to concatenate
#     dfs_to_concat = [train_df]

#     # Loop through paraphrase folders and their files
#     for i in range(1, 11):  # Looping from Paraphrase1 to Paraphrase10
#         folder_path = os.path.join(base_dir, f'Paraphrase{i}')
#         for class_file in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, class_file)
#             # Load each class CSV file
#             class_df = pd.read_csv(file_path, usecols=['text', 'label', 'keyword'])
#             # Add keywords to paraphrased dfs
#             class_df = add_keywords(class_df, model)
#             # Add sentiment to paraphrased dfs
#             # # Apply the function to get the compound sentiment score for each post
#             # class_df['vader_sentiment'] = class_df['text'].apply(get_vader_sentiment)
#             # class_df = add_vader_sentiment(class_df, model)
#             # Append the loaded dataframe to the list
#             dfs_to_concat.append(class_df)

#     # Concatenate all dataframes
#     train_df = pd.concat(dfs_to_concat, ignore_index=True)

# # Contar el número de publicaciones en cada categoría
# class_counts = train_df['label'].value_counts()
# print("Class distribution after augmenting with paraphrased texts:\n", class_counts)

# -----------------------------------------

if not augmenting:
    
    paraphrased_1_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
    paraphrased_1_df_2 = pd.read_csv('data/augmented_train_dfs/Paraphrase2/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
    paraphrased_1_df_3 = pd.read_csv('data/augmented_train_dfs/Paraphrase3/paraphrased_class_1.csv', usecols=['text', 'label', 'keyword'])
    
    paraphrased_2_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_2.csv', usecols=['text', 'label', 'keyword'])

    paraphrased_3_df_1 = pd.read_csv('data/augmented_train_dfs/Paraphrase1/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
    paraphrased_3_df_2 = pd.read_csv('data/augmented_train_dfs/Paraphrase2/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
    paraphrased_3_df_3 = pd.read_csv('data/augmented_train_dfs/Paraphrase3/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])
    paraphrased_3_df_4 = pd.read_csv('data/augmented_train_dfs/Paraphrase4/paraphrased_class_3.csv', usecols=['text', 'label', 'keyword'])

    paraphrased_df = pd.concat([paraphrased_1_df_1, paraphrased_1_df_2, paraphrased_1_df_3, paraphrased_2_df_1, paraphrased_3_df_1, paraphrased_3_df_2, paraphrased_3_df_3, paraphrased_3_df_4], ignore_index=True)

    # Add keywords to paraphrased dfs
    paraphrased_df = add_keywords(paraphrased_df, model)

    # # Add sentiment to paraphrased dfs
    # # Apply the function to get the compound sentiment score for each post
    # paraphrased_df['vader_sentiment'] = paraphrased_df['text'].apply(get_vader_sentiment)
    # paraphrased_df = add_vader_sentiment(paraphrased_df, model)

    punct_df = pd.read_csv('data/LoadOnEpoch/PunctInsertion/df1.csv', usecols=['text_augmented', 'label', 'keyword'])
    # punct_df = punct_df.loc[punct_df['label'] != 0]

    rnd_del_df = pd.read_csv('data/LoadOnEpoch/RandomDeletion/df1.csv', usecols=['text_augmented', 'label', 'keyword'])
    # rnd_del_df = rnd_del_df.loc[rnd_del_df['label'] != 0]

    rnd_swap_df = pd.read_csv('data/LoadOnEpoch/RandomSwap/df1.csv', usecols=['text_augmented', 'label', 'keyword'])
    # rnd_swap_df = rnd_swap_df.loc[rnd_swap_df['label'] != 0]

    rnd_insert_df = pd.read_csv('data/LoadOnEpoch/RandomInsertion/df1.csv', usecols=['text_augmented', 'label', 'keyword'])
    # rnd_insert_df = rnd_insert_df.loc[rnd_insert_df['label'] != 0]

    aug_df = pd.concat([punct_df, rnd_del_df, rnd_swap_df, rnd_insert_df], ignore_index=True)

    # rename text_augmented to text
    aug_df.rename(columns={'text_augmented': 'text'}, inplace=True)

    # Add keywords to augmented dfs
    aug_df = add_keywords(aug_df, model)

    # merge df with paraphrased dfs
    train_df = pd.concat([train_df, paraphrased_df, aug_df], ignore_index=True)


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


# if not augmenting:
#     ## ----------- Comment code if augmenting data ------------

#     chatGPT4_1_df = pd.read_csv('data/augmented_train_dfs/ChatGPT4_texts_class_1.csv', usecols=['text', 'label', 'keyword'])
#     chatGPT4_3_df = pd.read_csv('data/augmented_train_dfs/ChatGPT4_texts_class_3.csv', usecols=['text', 'label', 'keyword'])

#     # Add keywords to ChatGPT4 dfs
#     chatGPT4_1_df = add_keywords(chatGPT4_1_df, model)
#     chatGPT4_3_df = add_keywords(chatGPT4_3_df, model)

#     # Print chatGPT4 dfs
#     print("chatGPT4_1_df:\n", chatGPT4_1_df)

#     # merge df with paraphrased dfs
#     train_df = pd.concat([train_df, chatGPT4_1_df, chatGPT4_3_df], ignore_index=True)


# ## Cut Classes to X texts

# In[12]:


# from sklearn.utils import shuffle


# if not augmenting:

#     # Size of each class after sampling (Hyperparameter)
#     # Class 0 has 796 samples and was not augmented
#     class_size = 1500

#     # Sample 200 texts from each class (or as many as are available for classes with fewer than 200 examples)
#     sampled_dfs = []
#     for label in train_df['label'].unique():
#         class_sample_size = min(len(train_df[train_df['label'] == label]), class_size)
#         sampled_dfs.append(train_df[train_df['label'] == label].sample(n=class_sample_size, random_state=42))

#     # Concatenate the samples to create a balanced training DataFrame
#     train_df = pd.concat(sampled_dfs, ignore_index=True)


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

    from utils import paraphrase_humarin

    for label in {1, 2, 3}:
        print(f"Paraphrasing class {label}...")
        # Paraphrase and augment the data for underrepresented classes
        selected_texts = train_df.loc[train_df['label'] == label, 'text']
        selected_keywords = train_df.loc[train_df['label'] == label, 'keyword']
        print(f"length texts of label {label}", len(selected_texts))
        augmented_texts = paraphrase_humarin.paraphrase(selected_texts.to_list())
        # augmented_texts = [["t1", "t2", "t3", "t4"], ["t12", "t22", "t32", "t42"]]
        for i in range(len(augmented_texts[0])):
            print("i", i)
            curr_texts = [augmented_texts[j][i] for j in range(len(augmented_texts))]
            print(curr_texts)
            augmented_df = pd.DataFrame({'text': curr_texts, 'label': [label] * len(curr_texts), 'keyword': selected_keywords.to_list()})
            # augmented_df = pd.DataFrame({'text': curr_texts, 'label': [label] * len(curr_texts)})
            augmented_df.to_csv(f'data/augmented_train_dfs/ParaphraseVal{i+1}/paraphrased_class_{label}.csv', index=False)
        # train_df = pd.concat([train_df, augmented_df])

    # Check the new class distribution after paraphrasing
    # print("Class distribution after paraphrasing:", train_df['label'].value_counts())

    # train_df.to_csv(train_df_path, index=False)


# ## Run Model

# In[ ]:


from models import tune_transformer
# from models import tune_transformer_copy_GPU0 as tune_transformer

print("------------------------------------")
print("Model:", model)
print("------------------------------------")

if not augmenting:

    print("Converting train, val and test texts to csv...")
    train_texts.to_csv('data/train_texts.csv', index=False, header=False)
    val_texts.to_csv('data/val_texts.csv', index=False, header=False)
    test_texts.to_csv('data/test_texts.csv', index=False, header=False)

    test_pred_labels = tune_transformer.run(model, 4, train_texts, val_texts, test_texts, y_train, y_val, y_test=None)
    
    if final_deploy:
        # replace original test labels with predicted labels
        df_test['label'] = test_pred_labels

        # save the dataframe with predicted labels to a csv file
        print("Saving predictions to csv...")
        df_test.to_csv('data/prediction_task3.tsv', sep='\t', index=False)


# ## Print End Time

# In[ ]:


import time
print("------------------------------------------------")
print("End-Time")
# print current time in format: 2019-10-03 13:10:00
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("------------------------------------------------")

