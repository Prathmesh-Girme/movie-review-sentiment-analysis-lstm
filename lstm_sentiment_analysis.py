########################################################
# Movie Review Sentiment Analysis using LSTM (IMDB dataset)
#
# Pipeline : load -> show samples -> pad -> Embedding -> LSTM -> Dense(sigmoid)
#            -> train -> evaluate -> save model, metrics and plot
# Author   : Prathmesh Girme
########################################################
import json
import os
import platform
from datetime import datetime

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sentiment_utils as su

matplotlib.use("Agg")                      # save plots to files, no window needed
import matplotlib.pyplot as plt            # noqa: E402

LINE = "_" * 50
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "lstm_imdb_model.keras")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
PLOT_PATH = os.path.join(OUTPUT_DIR, "training_curve.png")
TEST_REVIEW_NUMBER = 0                     # Which test review to predict


########################################################
# Function name : build_model
# Description   : Embedding -> LSTM -> Dense (sigmoid)
# Input         : Nothing
# Output        : Compiled Keras model
# Author        : Prathmesh Girme
########################################################
def build_model():
    model = Sequential([
        Embedding(input_dim=su.VOCAB_SIZE, output_dim=su.EMBEDDING_DIM),
        LSTM(units=su.LSTM_UNITS),
        Dense(units=1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


########################################################
# Function name : save_training_plot
# Description   : Plot training and validation accuracy and loss per epoch
# Input         : Keras History object
# Output        : Nothing (writes outputs/training_curve.png)
# Author        : Prathmesh Girme
########################################################
def save_training_plot(history):
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    left.plot(epochs, history.history["accuracy"], marker="o", label="Train")
    left.plot(epochs, history.history["val_accuracy"], marker="o", label="Validation")
    left.set_title("Accuracy")
    left.set_xlabel("Epoch")
    left.legend()

    right.plot(epochs, history.history["loss"], marker="o", label="Train")
    right.plot(epochs, history.history["val_loss"], marker="o", label="Validation")
    right.set_title("Loss")
    right.set_xlabel("Epoch")
    right.legend()

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)


########################################################
# Function name : main
# Description   : Run the whole workflow
# Input         : Nothing
# Output        : Prints results and writes files to outputs/
# Author        : Prathmesh Girme
########################################################
def main():
    tf.keras.utils.set_random_seed(su.RANDOM_SEED)     # same results on every run
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(LINE)
    print("Movie Review Sentiment Analysis using LSTM")
    print(LINE)

    # Step 1 : Load the dataset (0 = negative, 1 = positive)
    print("Loading the dataset...")
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=su.VOCAB_SIZE)

    print("Number of training reviews :", len(X_train))
    print("Number of testing reviews  :", len(X_test))
    print("Positive share in train    :", round(float(np.mean(Y_train)), 3))

    # Step 2 : Show sample reviews
    reverse_word_index = su.build_reverse_word_index(imdb.get_word_index())

    print(LINE)
    print("Sample reviews")
    print(LINE)
    for i in range(4):
        print("Review number :", i + 1)
        print(su.decode_review(X_train[i], reverse_word_index))
        print("Sentiment :", su.sentiment_name(Y_train[i]))
        print(LINE)

    # Step 3 : Padding
    # "pre" padding puts the zeros at the start, so the LSTM finishes on
    # real words instead of zeros. This works better than "post" padding.
    X_train_padded = pad_sequences(X_train, maxlen=su.MAX_LENGTH, padding="pre", truncating="post")
    X_test_padded = pad_sequences(X_test, maxlen=su.MAX_LENGTH, padding="pre", truncating="post")

    print("Training data shape :", X_train_padded.shape)
    print("Testing data shape  :", X_test_padded.shape)

    # Step 4 : Build the model
    model = build_model()
    model.build(input_shape=(None, su.MAX_LENGTH))
    model.summary()

    # Step 5 : Train the model
    print("Model training...")
    history = model.fit(
        X_train_padded,
        Y_train,
        epochs=su.EPOCHS,
        batch_size=su.BATCH_SIZE,
        validation_split=su.VALIDATION_SPLIT,
    )
    print("Model training completed")

    # Step 6 : Evaluate the model on the test set (used once, after training)
    test_loss, test_accuracy = model.evaluate(X_test_padded, Y_test, verbose=0)
    print("Testing loss     :", round(float(test_loss), 4))
    print("Testing accuracy :", round(float(test_accuracy), 4))

    # Step 7 : Save the model, the metrics and the training plot
    model.save(MODEL_PATH)
    save_training_plot(history)

    metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "validation_accuracy": float(history.history["val_accuracy"][-1]),
        "test_accuracy": float(test_accuracy),
        "train_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "test_loss": float(test_loss),
        "accuracy_per_epoch": [float(v) for v in history.history["accuracy"]],
        "validation_accuracy_per_epoch": [float(v) for v in history.history["val_accuracy"]],
        "settings": {
            "vocab_size": su.VOCAB_SIZE, "max_length": su.MAX_LENGTH,
            "embedding_dim": su.EMBEDDING_DIM, "lstm_units": su.LSTM_UNITS,
            "epochs": su.EPOCHS, "batch_size": su.BATCH_SIZE,
            "validation_split": su.VALIDATION_SPLIT, "random_seed": su.RANDOM_SEED,
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
    }
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=2)
    print("Saved :", MODEL_PATH, "|", METRICS_PATH, "|", PLOT_PATH)

    # Step 8 : Predict one test review
    print(LINE)
    print("Review given to the model")
    print(LINE)
    print(su.decode_review(X_test[TEST_REVIEW_NUMBER], reverse_word_index))

    sample = X_test_padded[TEST_REVIEW_NUMBER:TEST_REVIEW_NUMBER + 1]
    probability = float(model.predict(sample, verbose=0)[0][0])

    print(LINE)
    print("Final result")
    print(LINE)
    print("Prediction probability :", round(probability, 4))
    print("Actual sentiment       :", su.sentiment_name(Y_test[TEST_REVIEW_NUMBER]))
    print("Predicted sentiment    :", su.sentiment_name(probability))


if __name__ == "__main__":
    main()
