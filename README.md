# Twitter Sentiment Analysis — BiLSTM + Attention

A 4-class tweet sentiment classifier built with PyTorch. Classifies tweets as **Positive**, **Negative**, **Neutral**, or **Irrelevant** using a Bidirectional LSTM with soft attention pooling and GloVe Twitter embeddings.

🚀 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Shivam217/twitter-sentiment-analysis)

---

## Model Architecture

```
Embedding (GloVe 100d, frozen 3 epochs → fine-tuned)
       ↓
BiLSTM (hidden=256, layers=2, bidirectional → 512d output)
       ↓
Soft Attention Pooling
       ↓
Classifier (512 → 256 → 4)
```

**5,060,357 trainable parameters** · embed_dim=128 · dropout=0.4 · AdamW + cosine LR schedule

---

## Results

### Test set (held-out, 7,451 tweets)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.8447 | 0.8292 | 0.8369 |
| Negative | 0.9065 | 0.8098 | 0.8554 |
| Neutral | 0.8902 | 0.7990 | 0.8421 |
| Irrelevant | 0.6951 | 0.9444 | 0.8008 |
| **Macro avg** | **0.8341** | **0.8456** | **0.8338** |

**Test accuracy: 83.60%** · **Test macro F1: 0.8338**

### vs Baseline (TF-IDF + Logistic Regression)

| Model | Val Accuracy | Val Macro F1 |
| TF-IDF + LogReg | **0.9550** | **0.9536** |
| **BiLSTM + Attention + GloVe** | **93.00%** | **0.9286** |


### Training curves

![Training curves](outputs/training_curves.png)
<img width="2384" height="593" alt="training_curves" src="https://github.com/user-attachments/assets/a0edc99a-335f-4537-86cb-183768f1165b" />


### Confusion matrix

![Confusion matrix](outputs/confusion_matrix.png)
<img width="2011" height="740" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f62e40ed-8991-4494-8dea-6b67aa2d7e03" />

---

## Project Structure

```
├── Twitter_Sentiment_Analysis_Complete.ipynb
├── model.py
├── app.py                  # Gradio demo
├── fastapi_app.py          # REST API
├── Dockerfile
├── requirements.txt
└── outputs/
    ├── best_model.pt
    ├── vocab.json
    └── config.json
```

---

## Deployment

### Local (FastAPI)

```bash
pip install fastapi uvicorn torch
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

Docs at `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## Future Improvements

- Fine-tune `bertweet-base` for comparison — expected improvement on the Neutral/Irrelevant boundary
- Back-translation augmentation on minority classes
- Visualise per-token attention weights for explainability
- Async FastAPI handlers for higher inference throughput

---

## Acknowledgements

- Dataset: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- Embeddings: [GloVe Twitter 27B](https://nlp.stanford.edu/projects/glove/) — Pennington et al., Stanford NLP
