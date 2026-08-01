# Amazon Sentiment Analysis

A binary sentiment classifier for Amazon product reviews, built with an LSTM written from scratch in PyTorch — no pre-trained embeddings, no high-level text library. The point was to build the whole pipeline by hand: vocabulary, encoding, padding, batching, and the recurrent model itself.

## The problem

Amazon reviews come with 1–5 star ratings, but star ratings are noisy sentiment labels — a 3-star review can read either way. This project collapses the scale into a binary target (ratings 0–2 → negative, 3–5 → positive) and learns sentiment directly from review text.

## Pipeline

1. **Vocabulary construction** — tokenise the full corpus, count word frequency with `Counter`, and build a `vocab_to_int` mapping ordered by frequency (index 0 reserved for padding).
2. **Encoding** — convert each review to a sequence of integer token IDs.
3. **Label transformation** — map the 1–5 star scale onto a binary sentiment label.
4. **Sequence normalisation** — pad and truncate reviews to a fixed length so they can be batched.
5. **Model** — `SentimentLSTM`: an embedding layer, a multi-layer LSTM with dropout, and a fully-connected layer with sigmoid activation for binary output.
6. **Training** — `BCELoss` with the Adam optimiser, evaluated on a held-out test split.

## Model

```
Embedding(vocab_size, embedding_dim)
   ↓
LSTM(embedding_dim, hidden_dim, n_layers, dropout, batch_first=True)
   ↓
Dropout → Linear → Sigmoid
```

## Stack

PyTorch · NumPy · Pandas

## Contents

```
Amazon_Sentimental_Analysis(Binary).ipynb   # full pipeline: preprocessing → training → evaluation
review.csv                                  # review text and star ratings
```

## Running it

Open the notebook in Jupyter or Colab and run the cells in order. A GPU is recommended for training but not required.

```bash
pip install torch numpy pandas
jupyter notebook "Amazon_Sentimental_Analysis(Binary).ipynb"
```
