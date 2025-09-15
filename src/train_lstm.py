import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle

from preprocess import basic_clean, pad_sequences

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model(vocab_size, embedding_dim, lstm_units, max_len):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, input_length=max_len),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/sample_tweets.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns (label: 0/1).")

    texts = df["text"].astype(str).apply(basic_clean).tolist()
    labels = df["label"].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=cfg["val_split"], random_state=cfg["seed"], stratify=labels)

    tok = Tokenizer(num_words=cfg["max_vocab"], oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    seq_train = tok.texts_to_sequences(X_train)
    seq_val = tok.texts_to_sequences(X_val)

    x_train = pad_sequences(seq_train, cfg["max_len"])
    x_val = pad_sequences(seq_val, cfg["max_len"])

    model = build_model(vocab_size=cfg["max_vocab"], embedding_dim=cfg["embedding_dim"], lstm_units=cfg["lstm_units"], max_len=cfg["max_len"])
    es = EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        callbacks=[es],
        verbose=2
    )

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "lstm_sentiment.keras")
    tok_path = os.path.join(args.models_dir, "tokenizer.pkl")
    meta_path = os.path.join(args.models_dir, "meta.json")

    model.save(model_path)
    with open(tok_path, "wb") as f:
        pickle.dump(tok, f)

    meta = {
        "max_len": cfg["max_len"],
        "labels": {"negative": 0, "positive": 1},
        "config": cfg
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved tokenizer: {tok_path}")
    print(f"Saved meta: {meta_path}")

if __name__ == "__main__":
    main()
