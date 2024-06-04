import os
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Global settings
def estimate_accuracy(X_test, y_test, y_pred):
    loss, accuracy = model.evaluate(X_test, y_test, verbose = 0)
    print(f"\nTest loss:          {loss}")
    print(f"Test accuracy:      {accuracy}\n")

    # Convert probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis = 1)
    print(classification_report(y_test, y_pred_labels))
    print(confusion_matrix(y_test, y_pred_labels))


random_state = 2291
pd.set_option("display.max_colwidth", 200)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable warnings
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)
#


# RECURRENT NEURAL NETWORK (RNN)
# Load dataset:     https://www.kaggle.com/datasets/lakshmi25npathi/images
file_dataset = "input/Youtube02-KatyPerry.csv"
data = pd.read_csv(file_dataset, encoding = "latin-1")
data = data.drop("COMMENT_ID", axis = 1)
data = data.drop("AUTHOR", axis = 1)
data = data.drop("DATE", axis = 1)
feature_names = {
    "CONTENT": "Content",
    "CLASS": "Category"
}
data = data.rename(columns = feature_names)
data.info()
display(data)
unique_vals = data["Category"].unique()
print(unique_vals)
#


# Preprocess texts
def clean_text(text):
    tokens = word_tokenize(text)  # Split text into tokens

    # Remove punctuation, convert to lower case, clean stop words
    stop_words = stopwords.words("english")
    words = []
    for token in tokens:
        if token.isalpha():
            word = token.lower()
            if word not in stop_words:
                words.append(word)

    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in words:
        stemmed_word = stemmer.stem(word)
        stemmed_words.append(stemmed_word)
    stemmed_text = " ".join(stemmed_words)

    return stemmed_text


data_before = data.copy()
data["Content"] = data["Content"].apply(clean_text)
max_length = data["Content"].apply(len).max()
display(data_before.head(6))
display(data.head(6))
#


# Split dataset into training set and test set
X = data["Content"]
y = data["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts = X_train)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size:    {vocab_size}")
print(f"Max text length:    {max_length}\n")
#


# Convert texts to sequences of indexes. Then add padding
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
preprocessed_text = tokenizer.sequences_to_texts([X_train[0]])[0]
print("Text to integer sequences:\n"
      f"  (preprocessed) -> {preprocessed_text}\n"
      f"  (tokenized)    -> {X_train[0]}")
X_train = pad_sequences(X_train, maxlen = max_length)
X_test = pad_sequences(X_test, maxlen = max_length)
#


# Create RNN model
model = Sequential(
    [
        Embedding(input_dim = vocab_size, input_length = max_length, output_dim = 32),
        LSTM(units = 64, return_sequences = True),
        Dropout(0.2),
        LSTM(units = 64),
        Dropout(0.3),
        Dense(units = len(unique_vals), activation = "softmax")
    ]
)
model.summary()
#


# Compile and fit RNN
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
model.fit(X_train, y_train, batch_size = 128, epochs = 35, validation_split = 0.1)
y_pred = model.predict(X_test)
estimate_accuracy(X_test, y_test, y_pred)
#
