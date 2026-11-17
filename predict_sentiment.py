########################################################
# Predict the sentiment of your own review with the saved LSTM model
#
# Usage   : python predict_sentiment.py "This movie was fantastic"
#           python predict_sentiment.py          (asks you to type a review)
# Needs   : outputs/lstm_imdb_model.keras (run lstm_sentiment_analysis.py first)
# Author  : Prathmesh Girme
########################################################
import os
import sys

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

import sentiment_utils as su

MODEL_PATH = os.path.join("outputs", "lstm_imdb_model.keras")


########################################################
# Function name : main
# Description   : Load the model and print the predicted sentiment
# Input         : Review text from the command line or from the keyboard
# Output        : Probability and label
# Author        : Prathmesh Girme
########################################################
def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"{MODEL_PATH} not found. Run: python lstm_sentiment_analysis.py")

    text = " ".join(sys.argv[1:]).strip() or input("Type a movie review: ").strip()
    if not text:
        sys.exit("No review text given.")

    model = load_model(MODEL_PATH)
    encoded = su.encode_review(text, imdb.get_word_index())
    probability = float(model.predict(encoded, verbose=0)[0][0])

    print("Probability of positive :", round(probability, 4))
    print("Predicted sentiment     :", su.sentiment_name(probability))


if __name__ == "__main__":
    main()
