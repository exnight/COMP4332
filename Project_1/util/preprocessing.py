import pandas as pd
import spacy
import re
from string import punctuation
from nltk.corpus import stopwords

from keras.utils import to_categorical

nlp = spacy.load('en_core_web_lg')
stop_words = set(stopwords.words('english') + list(punctuation) + ['-PRON-'])


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I | re.A).lower().replace('\n', '').strip()
    text = re.sub(' +', ' ', text)
    text = nlp(text)
    lemmatized = list()
    for token in text:
        lemma = token.lemma_
        if lemma not in stop_words and not lemma.isnumeric():
            lemmatized.append(''.join(lemma.split()))

    return " ".join(lemmatized)


def read_data():
    print('Reading raw data')
    df_train = pd.read_csv('../data/train.csv')
    df_valid = pd.read_csv('../data/valid.csv')
    df_test = pd.read_csv('../data/test.csv')

    # Drop unused columns
    df_train.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)
    df_valid.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)
    df_test.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    return df_train, df_valid, df_test


def load_data():
    df_train, df_valid, df_test = read_data()

    K = df_train['stars'].nunique()
    df_train['stars'] = df_train['stars'] - 1
    df_valid['stars'] = df_valid['stars'] - 1

    train_label = pd.DataFrame(to_categorical(df_train['stars'], num_classes=K))
    valid_label = pd.DataFrame(to_categorical(df_valid['stars'], num_classes=K))

    print('Outputing labels')
    train_label.to_csv('../build/train_label.csv', index=False)
    valid_label.to_csv('../build/valid_label.csv', index=False)

    print('Processing training data...')
    df_train['text'] = df_train['text'].apply(clean_text)

    print('Processing validation data...')
    df_valid['text'] = df_valid['text'].apply(clean_text)

    print('Processing testing data...')
    df_test['text'] = df_test['text'].apply(clean_text)

    df_valid['text'] = df_valid['text'].astype('str')
    df_train['text'] = df_train['text'].astype('str')
    df_test['text'] = df_test['text'].astype('str')

    df_train.drop('stars', inplace=True)
    df_valid.drop('stars', inplace=True)

    print('Outputing data')
    df_train.to_csv('../build/train.csv', index=False)
    df_valid.to_csv('../build/valid.csv', index=False)
    df_test.to_csv('../build/test.csv', index=False)


load_data()
