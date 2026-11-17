"""Tests for sentiment_utils.py - no TensorFlow or dataset download needed.
Run with:  python -m pytest -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sentiment_utils as su

# A tiny word index in the same format as imdb.get_word_index() (rank starts at 1)
WORD_INDEX = {"the": 1, "movie": 2, "was": 3, "great": 4, "terrible": 5, "rareword": 9999}


def test_short_review_gets_zeros_at_the_start():
    padded = su.pad_review([1, 5, 6], max_length=6)
    assert padded.tolist() == [0, 0, 0, 1, 5, 6]


def test_long_review_keeps_its_first_words():
    padded = su.pad_review(list(range(1, 11)), max_length=4)
    assert padded.tolist() == [1, 2, 3, 4]


def test_padding_matches_keras_rule_when_available():
    pad_sequences = pytest.importorskip("tensorflow.keras.preprocessing.sequence").pad_sequences
    review = [1, 14, 22, 16]
    expected = pad_sequences([review], maxlen=10, padding="pre", truncating="post")[0]
    assert su.pad_review(review, 10).tolist() == expected.tolist()


def test_encode_review_follows_imdb_numbering():
    encoded = su.encode_review("The movie was GREAT!", WORD_INDEX, max_length=8)
    assert encoded.shape == (1, 8)
    # zeros, start token (1), then each word index shifted by 3
    assert encoded[0].tolist() == [0, 0, 0, 1, 1 + 3, 2 + 3, 3 + 3, 4 + 3]


def test_unknown_and_rare_words_become_the_unknown_token():
    encoded = su.encode_review("great zzzz rareword", WORD_INDEX, vocab_size=100, max_length=6)
    # "zzzz" is not in the index, "rareword" (9999 + 3) is outside the vocabulary
    assert encoded[0].tolist() == [0, 0, 1, 4 + 3, 2, 2]


def test_empty_text_still_gives_a_valid_input():
    encoded = su.encode_review("", WORD_INDEX, max_length=5)
    assert encoded[0].tolist() == [0, 0, 0, 0, 1]


def test_decode_review_skips_special_tokens_and_marks_unknown_numbers():
    reverse = su.build_reverse_word_index(WORD_INDEX)
    # 0, 1, 2 are padding, start and unknown, so they are skipped;
    # 4 and 5 are "the" and "movie"; 999 is not in the dictionary, so it becomes "?"
    assert su.decode_review([0, 1, 2, 4, 5, 999], reverse) == "the movie ?"


def test_encode_then_decode_round_trip():
    reverse = su.build_reverse_word_index(WORD_INDEX)
    encoded = su.encode_review("the movie was great", WORD_INDEX, max_length=10)
    assert su.decode_review(encoded[0], reverse) == "the movie was great"


@pytest.mark.parametrize("value,label", [(0, "NEGATIVE"), (1, "POSITIVE"), (0.49, "NEGATIVE"), (0.5, "POSITIVE")])
def test_sentiment_name(value, label):
    assert su.sentiment_name(value) == label
