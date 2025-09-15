import os
import json
import pickle
import numpy as np
from typing import List, Tuple

from preprocess import basic_clean, pad_sequences

# Optional TensorFlow import is inside a function to allow a "toy" fallback
def _try_load_tf_model(model_path: str):
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

def load_artifacts(models_dir="models"):
    model_path = os.path.join(models_dir, "lstm_sentiment.keras")
    tok_path = os.path.join(models_dir, "tokenizer.pkl")
    meta_path = os.path.join(models_dir, "meta.json")
    toy_path = os.path.join(models_dir, "toy_weights.json")

    model = None
    tok = None
    meta = {"max_len": 48}

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if os.path.exists(model_path) and os.path.exists(tok_path):
        # real model
        tok = pickle.load(open(tok_path, "rb"))
        model = _try_load_tf_model(model_path)
        if model is None:
            print("Warning: Could not load TensorFlow model. Will attempt toy model fallback.")
    else:
        print("Info: LSTM artifacts not found. Falling back to toy model.")

    toy = None
    if os.path.exists(toy_path):
        with open(toy_path, "r", encoding="utf-8") as f:
            toy = json.load(f)

    return model, tok, meta, toy

def predict(texts: List[str], models_dir="models") -> List[Tuple[float, int]]:
    """Return list of (prob_positive, label) tuples."""
    model, tok, meta, toy = load_artifacts(models_dir=models_dir)
    cleaned = [basic_clean(t) for t in texts]

    if model is not None and tok is not None:
        seqs = tok.texts_to_sequences(cleaned)
        x = pad_sequences(seqs, meta.get("max_len", 48))
        probs = model.predict(x, verbose=0).reshape(-1)
        labels = (probs >= 0.5).astype(int).tolist()
        return list(zip(probs.tolist(), labels))

    # Toy fallback: simple keyword heuristic using toy_weights.json
    if toy is None:
        # default tiny lexicon
        toy = {"pos": ["love","great","good","happy","wonderful","amazing","best","excited","spectacular","recommend"],
               "neg": ["hate","bad","sad","terrible","awful","worst","disappointed","furious","broken","bland"]}

    results = []
    for t in cleaned:
        score = 0
        for w in toy["pos"]:
            if f" {w} " in f" {t} ":
                score += 1
        for w in toy["neg"]:
            if f" {w} " in f" {t} ":
                score -= 1
        prob = 1/(1+np.exp(-score))
        label = 1 if prob >= 0.5 else 0
        results.append((float(prob), int(label)))
    return results
