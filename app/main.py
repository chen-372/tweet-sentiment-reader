import os, sys, json
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# make src importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from infer import predict

app = FastAPI(title="Tweet Sentiment Classifier (LSTM)")

class PredictRequest(BaseModel):
    text: str

class PredictBatchRequest(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {
        "message": "Tweet Sentiment Classifier is running.",
        "endpoints": ["/predict", "/predict_batch"]
    }

@app.post("/predict")
def predict_one(req: PredictRequest):
    (prob, label) = predict([req.text])[0]
    return {"text": req.text, "prob_positive": prob, "label": int(label)}

@app.post("/predict_batch")
def predict_many(req: PredictBatchRequest):
    results = predict(req.texts)
    out = []
    for txt, (prob, label) in zip(req.texts, results):
        out.append({"text": txt, "prob_positive": prob, "label": int(label)})
    return {"results": out}
