import gradio as gr, torch, json, re
import torch.nn.functional as F

PAD_IDX, UNK_IDX = 0, 1
LABEL_MAP = {"Positive":0,"Negative":1,"Neutral":2,"Irrelevant":3}
ID2LABEL  = {v:k for k,v in LABEL_MAP.items()}
CLASS_NAMES = list(LABEL_MAP.keys())
MAX_LEN = 100

# Paste your clean_tweet() and LSTMSentimentClassifier here
# or import them from a separate model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("outputs/vocab.json", encoding="utf-8") as f: vocab_data = json.load(f)
word2idx = vocab_data["word2idx"]; vocab_size = vocab_data["vocab_size"]
from model import LSTMSentimentClassifier
ckpt  = torch.load("outputs/best_model.pt", map_location=device)
model = LSTMSentimentClassifier(vocab_size=vocab_size,
    embed_dim=ckpt.get("embed_dim",128), hidden_dim=ckpt.get("hidden_dim",256),
    num_layers=ckpt.get("num_layers",2), num_classes=4, dropout=0.0, pad_idx=PAD_IDX).to(device)
model.load_state_dict(ckpt["state_dict"]); model.eval()

def predict_sentiment(text):
    enc = [word2idx.get(w, UNK_IDX) for w in text.lower().split()][:MAX_LEN]
    enc += [PAD_IDX] * (MAX_LEN - len(enc))
    ids = torch.tensor([enc], dtype=torch.long, device=device)
    with torch.no_grad(): probs = F.softmax(model(ids),dim=-1)[0].cpu().tolist()
    return {CLASS_NAMES[i]: round(probs[i],4) for i in range(4)}

demo = gr.Interface(fn=predict_sentiment,
    inputs=gr.Textbox(label="Tweet", lines=3),
    outputs=gr.Label(num_top_classes=4),
    title="Twitter Sentiment Analyser",
    theme=gr.themes.Soft())

if __name__ == "__main__": demo.launch(server_name="0.0.0.0")
