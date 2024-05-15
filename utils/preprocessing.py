import pandas as pd
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Lemmatizer
lemmatizer = WordNetLemmatizer()

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


def load_df(deploying: bool,
            train_path: str,
            val_path: str,
            test_path: str) -> pd.DataFrame:
    print("Reading data...")
    df = pd.read_csv(train_path, usecols=['id', 'keyword', 'text', 'label'])
    df_val = pd.read_csv(val_path, usecols=['id', 'keyword', 'text', 'label'])
    
    # Apply the function to filter sentences in the text
    df['text'] = df.apply(filter_sentences, axis=1)
    df_val['text'] = df_val.apply(filter_sentences, axis=1)
    
    if deploying:
        df = pd.concat([df, df_val], ignore_index=True)
        df_test = pd.read_csv(test_path, usecols=['id', 'keyword', 'text'])
        df_test['text'] = df_test.apply(filter_sentences, axis=1)
        print("Data read...")
        return df, df_test
    else:
        print("Data read...")
        return df, df_val
    

def get_sep_token(model_checkpoint: str) -> str:
    if model_checkpoint == 'distilbert-base-uncased' or model_checkpoint == 'roberta-base' or model_checkpoint == 'bert-large-uncased' or model_checkpoint == 'microsoft/deberta-v2-xxlarge' or 'albert' in model_checkpoint:
        sep_token = '[SEP]'
    elif 'xlnet' in model_checkpoint:
        sep_token = '<sep>'
    elif model_checkpoint == 'xlm-roberta-large':
        sep_token = '</s>'
    return sep_token


def add_keywords(df_, model):
    sep_token = get_sep_token(model)
    
    df_['text'] = df_['text'] + f" {sep_token} Keyword: " + df_['keyword']
    df_.drop(columns=['keyword'], inplace=True)
    return df_


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


def preprocess_data(deploying: bool,
                    train_path: str,
                    val_path: str,
                    test_path: str,
                    model_checkpoint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    df, df_test = load_df(deploying, train_path, val_path, test_path)
    df = add_keywords(df, model_checkpoint)
    df_test = add_keywords(df_test, model_checkpoint)

    # Clean text
    df['text'] = df['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)

    return df, df_test
