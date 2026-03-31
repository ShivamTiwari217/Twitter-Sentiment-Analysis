# Install: pip install fastapi uvicorn torch
# Run:     uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import torch, json, time
import torch.nn.functional as F

PAD_IDX, UNK_IDX = 0, 1
LABEL_MAP = {"Positive":0,"Negative":1,"Neutral":2,"Irrelevant":3}
ID2LABEL  = {v:k for k,v in LABEL_MAP.items()}
MAX_LEN   = 100

# Paste clean_tweet() and LSTMSentimentClassifier here or import from model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("outputs/vocab.json") as f: vocab_data = json.load(f)
word2idx = vocab_data["word2idx"]; vocab_size = vocab_data["vocab_size"]
from model import LSTMSentimentClassifier
ckpt  = torch.load("outputs/best_model.pt", map_location=device)
model = LSTMSentimentClassifier(vocab_size=vocab_size,
    embed_dim=ckpt.get("embed_dim",128), num_classes=4,
    dropout=0.0, pad_idx=PAD_IDX).to(device)
model.load_state_dict(ckpt["state_dict"]); model.eval()

app = FastAPI(title="Twitter Sentiment API", version="1.0.0")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    t0  = time.perf_counter()
    enc = [word2idx.get(w,UNK_IDX) for w in req.text.lower().split()][:MAX_LEN]
    enc += [PAD_IDX]*(MAX_LEN-len(enc))
    ids = torch.tensor([enc],dtype=torch.long,device=device)
    with torch.no_grad(): probs=F.softmax(model(ids),dim=-1)[0].cpu().tolist()
    idx = int(max(range(4),key=lambda i:probs[i]))
    return {"label":ID2LABEL[idx],"confidence":round(probs[idx],4),
            "probabilities":{ID2LABEL[i]:round(probs[i],4) for i in range(4)},
            "latency_ms":round((time.perf_counter()-t0)*1000,2)}
