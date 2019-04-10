import numpy as np
import pandas as pd
import spacy
from keras.preprocessing.sequence import pad_sequences
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2 as L2

nlp_vec = spacy.load('en_vectors_web_lg')


def get_word_vec(text):
    seq = np.array([nlp_vec.vocab.get_vector(word) for word in text.split() if nlp_vec.vocab.has_vector(word)])
    if seq.size > 0:
        seq = pad_seq(seq)
    else:
        seq = np.zeros((150, 300))

    return seq


def pad_seq(seq):
    return pad_sequences(seq.transpose(), dtype='float32', maxlen=150).transpose()


def read_data():
    df_train = pd.read_csv('../build/data/train.csv')
    df_valid = pd.read_csv('../build/data/valid.csv')
    df_test = pd.read_csv('../build/data/test.csv')

    return df_train, df_valid, df_test


def load_data():
    df_train, df_valid, df_test = read_data()

    df_valid['text'] = df_valid['text'].astype('str')
    df_train['text'] = df_train['text'].astype('str')
    df_test['text'] = df_test['text'].astype('str')

    train_matrix = df_train['text'].apply(get_word_vec).values
    valid_matrix = df_valid['text'].apply(get_word_vec).values
    test_matrix = df_test['text'].apply(get_word_vec).values

    train_matrix = np.vstack(train_matrix).reshape(train_matrix.shape[0], 150, 300)
    valid_matrix = np.vstack(valid_matrix).reshape(valid_matrix.shape[0], 150, 300)
    test_matrix = np.vstack(test_matrix).reshape(test_matrix.shape[0], 150, 300)

    train_label = pd.read_csv('../build/data/train_label.csv')
    valid_label = pd.read_csv('../build/data/valid_label.csv')
    K = len(train_label.columns)

    return train_matrix, train_label, valid_matrix, valid_label, test_matrix, K


train_matrix, train_label, valid_matrix, valid_label, test_matrix, output_size = load_data()

embedding_size = 300  # spaCy provides word vectors of size 1x300
time_steps = 150  # 150 is the 90% quantile of length of lemmas

filters = 60
kernel_size = 4
padding = 'valid'
strides = 1
pool_size = 2

dropout_rate = 0.5
learning_rate = 0.001
batch_size = 64
epochs = 10

activation_def = 'relu'
optimizer_def = Adam()
regularizer_def = L2(0.0001)

# Build the LSTM-CNN model
model = Sequential()

model.add(LSTM(embedding_size, input_shape=(batch_size, time_steps, embedding_size), return_sequences=True, kernel_regularizer=regularizer_def))
model.add(LSTM(embedding_size, input_shape=(batch_size, time_steps, embedding_size), return_sequences=True, kernel_regularizer=regularizer_def))

model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation_def))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation_def))
model.add(MaxPooling1D(pool_size=pool_size))

model.add(Flatten())
model.add(Dropout(rate=dropout_rate))
model.add(Dense(output_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
print(model.summary())

# Training model
train_history = model.fit(train_matrix, train_label, epochs=epochs, batch_size=batch_size, validation=(valid_matrix, valid_label))

model.save('../build/models/lstm_cnn'+str(epochs)+'epoch.h5')
with open('../build/log/lstm_cnn_history.json', 'w') as fp:
    json.dump(train_history.history, fp)

train_score = model.evaluate(train_matrix, train_label, batch_size=batch_size)
print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))

valid_score = model.evaluate(valid_matrix, valid_label, batch_size=batch_size)
print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))
