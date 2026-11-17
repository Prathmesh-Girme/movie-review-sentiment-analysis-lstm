########################################################
# Helper functions for the LSTM sentiment project
#
# No TensorFlow code in this file on purpose, so these functions can be
# tested quickly and reused by both the training and the prediction script.
########################################################
import re

import numpy as np

########################################################
# Settings (shared by training and prediction, so they can never differ)
########################################################
VOCAB_SIZE = 10000          # Keep the 10,000 most frequent words
MAX_LENGTH = 200            # Use only the first 200 words of each review
EMBEDDING_DIM = 32          # Each word is represented by 32 numbers
LSTM_UNITS = 64             # Size of the LSTM hidden state
EPOCHS = 3
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# IMDB reserves indexes 0, 1, 2 for padding, start and unknown tokens.
# Real words therefore start from index 3.
PAD_INDEX = 0
START_INDEX = 1
UNKNOWN_INDEX = 2
INDEX_OFFSET = 3


########################################################
# Function name : build_reverse_word_index
# Description   : Create a dictionary that maps number -> word
# Input         : word -> number dictionary from imdb.get_word_index()
# Output        : number -> word dictionary
# Author        : Prathmesh Girme
########################################################
def build_reverse_word_index(word_index):
    return {index + INDEX_OFFSET: word for word, index in word_index.items()}


########################################################
# Function name : decode_review
# Description   : Convert a list of numbers back into readable text
# Input         : Encoded review, reverse word index
# Output        : Review text
# Author        : Prathmesh Girme
########################################################
def decode_review(encoded_review, reverse_word_index):
    words = [
        reverse_word_index.get(number, "?")
        for number in encoded_review
        if number >= INDEX_OFFSET
    ]
    return " ".join(words)


########################################################
# Function name : sentiment_name
# Description   : Convert 1/0 (or a probability) into a label
# Input         : Label or probability
# Output        : "POSITIVE" or "NEGATIVE"
# Author        : Prathmesh Girme
########################################################
def sentiment_name(value):
    return "POSITIVE" if value >= 0.5 else "NEGATIVE"


########################################################
# Function name : pad_review
# Description   : Make a list of numbers exactly MAX_LENGTH long
# Input         : List of numbers
# Output        : numpy array of shape (MAX_LENGTH,)
# Author        : Prathmesh Girme
########################################################
def pad_review(numbers, max_length=MAX_LENGTH):
    """Same rule as pad_sequences(padding="pre", truncating="post").

    Long reviews keep their first words. Short reviews get zeros at the
    start, so the LSTM finishes on real words instead of zeros.
    """
    numbers = list(numbers)[:max_length]
    return np.array([PAD_INDEX] * (max_length - len(numbers)) + numbers, dtype="int32")


########################################################
# Function name : encode_review
# Description   : Turn new review text into the numbers the model expects
# Input         : Text, word -> number dictionary from imdb.get_word_index()
# Output        : numpy array of shape (1, MAX_LENGTH)
# Author        : Prathmesh Girme
########################################################
def encode_review(text, word_index, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    """Follow the same rules as imdb.load_data: a start token, words shifted by 3,
    and unknown or rare words (outside the top vocab_size) mapped to the unknown token."""
    numbers = [START_INDEX]
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        number = word_index.get(word)
        number = UNKNOWN_INDEX if number is None else number + INDEX_OFFSET
        numbers.append(number if number < vocab_size else UNKNOWN_INDEX)
    return pad_review(numbers, max_length).reshape(1, -1)
