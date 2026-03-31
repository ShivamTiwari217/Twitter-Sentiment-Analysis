# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, mask):
        scores  = self.attn(lstm_out).squeeze(-1)
        scores  = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * lstm_out).sum(dim=1)

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=4, dropout=0.4, pad_idx=0):
        super().__init__()
        self.pad_idx       = pad_idx
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        lstm_out_dim    = hidden_dim * 2
        self.attention  = AttentionPooling(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"      in name:
                p.data.fill_(0)
                n = p.size(0)
                p.data[n//4:n//2].fill_(1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, token_ids):
        mask     = (token_ids != self.pad_idx)
        embedded = self.embed_dropout(self.embedding(token_ids))
        lengths  = mask.sum(dim=1).cpu().clamp(min=1)
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=token_ids.size(1))
        return self.classifier(self.attention(lstm_out, mask))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)