# Movie Review Sentiment Analysis using LSTM

This project demonstrates an **end-to-end deep learning workflow** for classifying **IMDB movie reviews as positive or negative**. It turns review text into numbers, trains an **LSTM network built with Keras**, evaluates it on reviews it has never seen, and lets you **predict the sentiment of your own review** with the saved model.

It follows **industrial best practices** by:

- Using the **official IMDB train/test split** (25,000 reviews each) and evaluating on the test set **only once, after training**
- Holding out **20% of the training data for validation**, so overfitting shows up during training
- Using **pre-padding**, so the LSTM finishes on real words instead of zeros
- Fixing the **random seed**, so a run can be repeated
- Keeping **shared settings and text helpers in one file** (`sentiment_utils.py`), so training and prediction can never disagree on how a review is encoded
- Saving the **model, the metrics and the training curves** to files for reproducibility
- Reusing the saved model from a **separate prediction script**, without retraining
- Covering the text helpers with **automated tests** that need no TensorFlow and no dataset download

---

## Dependencies

Requires **Python 3.9+**. Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install tensorflow numpy matplotlib
```

---

## Dataset Information

**Source:** the **IMDB movie review dataset** that ships with Keras (`tensorflow.keras.datasets.imdb`). It downloads automatically the first time you run the project.

| Item | Value |
|------|-------|
| Training reviews | 25,000 |
| Test reviews | 25,000 |
| Classes | 2 (0 = negative, 1 = positive) |
| Balance | equal numbers of positive and negative reviews |
| Language | English |

**Things to know about this data:**

- **Reviews arrive as numbers, not text.** Each word is replaced by its frequency rank, so the project keeps only the **10,000 most frequent words**. Rarer words become an "unknown" token.
- **Numbers 0, 1 and 2 are reserved** for padding, the start of a review and unknown words. Real words start from 3, so decoding a review back to text shifts every number by 3.
- **Reviews have different lengths**, so every review is cut or padded to **exactly 200 words**. Words after the 200th are ignored.
- **Reviews are strongly polarised.** Only clearly positive or clearly negative reviews are included, so a neutral or mixed review is outside what the model was trained on.

---

## Workflow

### Data Preparation

- Load the dataset with a 10,000-word vocabulary
- Decode and print sample reviews to check that the numbers map back to readable text
- **Pad** every review to 200 words (zeros at the start, long reviews cut at the end)
- The padded training data has shape **(25,000, 200)** and the test data **(25,000, 200)**

### Model Construction

```text
Review ──► Embedding(10000, 32) ──► LSTM(64) ──► Dense(1, sigmoid) ──► Positive / Negative
```

| Setting | Value |
|---------|-------|
| Vocabulary size | 10,000 words |
| Review length | 200 words |
| Embedding size | 32 numbers per word |
| LSTM units | 64 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Epochs / batch size | 3 / 64 |
| Validation split | 20% of the training data |
| Random seed | 42 |

**Model summary** (input is a batch of reviews, each 200 numbers long):

| Layer | Output shape | Parameters |
|-------|--------------|------------|
| Embedding | (None, 200, 32) | 320,000 |
| LSTM (64 units) | (None, 64) | 24,832 |
| Dense (1, sigmoid) | (None, 1) | 65 |
| **Total trainable parameters** | | **344,897** |

Most of the parameters (320,000 of 344,897) sit in the Embedding layer, which learns a 32-number meaning for each of the 10,000 words.

### Training and Evaluation

- Train on **20,000 reviews** and validate on **5,000** (the 20% split of the 25,000 training reviews) for 3 epochs, tracking training and validation accuracy and loss after every epoch
- Evaluate **once** on the 25,000 test reviews
- Predict the sentiment of one test review as a final check

### Why LSTM instead of a simple RNN

A simple RNN forgets early words in long sentences (the **vanishing gradient** problem). An LSTM keeps a separate **cell state** with **gates** that decide what to remember and what to forget, so it handles long reviews much better. A sentence such as *"The first half was slow, but the ending made it worth watching"* needs the model to remember the whole sentence, not only the last words.

### Model Saving & Loading

- Save the trained model with `model.save` to `outputs/lstm_imdb_model.keras`
- Load it in a separate script with `load_model` and predict without retraining

---

## Running the Project

```bash
python lstm_sentiment_analysis.py
```

The first run downloads the dataset. Training runs on a normal CPU, and a GPU makes it faster.

**Predict your own review (after training once):**

```bash
python predict_sentiment.py "The acting was brilliant and the story kept me hooked"
```

With no text, the script asks you to type a review. It prints the **probability of a positive review** and the label (probability of 0.5 or more is positive).

**Generated files** (in `outputs/`):

```text
outputs/
├── lstm_imdb_model.keras     trained model (ignored by Git)
├── metrics.json              train / validation / test accuracy and loss, settings, library versions
└── training_curve.png        accuracy and loss for each epoch
```

**Project files:**

```text
movie-review-sentiment-analysis-lstm/
├── lstm_sentiment_analysis.py    train, evaluate and save
├── predict_sentiment.py          predict a new review with the saved model
├── sentiment_utils.py            settings, padding, encoding and decoding helpers
├── requirements.txt
└── tests/
    └── test_utils.py             automated tests for the helpers
```

---

## Testing

```bash
pip install pytest
python -m pytest -v
```

The tests need **no TensorFlow and no dataset download**. They check that:

- short reviews get zeros at the start and long reviews keep their first words
- new text is encoded the same way as the IMDB dataset (start token, numbers shifted by 3, unknown and rare words mapped to the unknown token)
- encoding a sentence and decoding it again gives the same sentence
- probabilities are turned into the right label, with 0.5 counting as positive

> These tests check the **text handling**, not the quality of the trained model. Model accuracy comes from running the training script and is written to `outputs/metrics.json`.

---

## Limits

- **Only the first 200 words are used.** A review that states its verdict at the end, after more than 200 words, loses that part.
- **Only 10,000 words are known.** Rare names and unusual words are treated as unknown.
- **Only two classes.** There is no "neutral" answer, and the model gives a confident-looking probability even for text unlike a movie review.
- **Sarcasm and mixed reviews are hard.** The model reads word patterns, so *"Oh great, another two hours of my life gone"* can be misread.
- **Trained on movie reviews only.** It should not be used for product reviews, tweets or other kinds of text without retraining.
- **Short training (3 epochs).** The settings are kept small so the project runs on any computer. Check `training_curve.png` to see whether validation accuracy was still improving or had started to fall.

---

## Author

**Prathmesh Girme**  
GitHub: [@Prathmesh-Girme](https://github.com/Prathmesh-Girme)
