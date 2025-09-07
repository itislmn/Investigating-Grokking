import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=113, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len=2)
        emb = self.embedding(x)                      # (B, 2, d_model)
        h = self.transformer(emb)                   # (B, 2, d_model)
        h = h.mean(dim=1)                           # Pool over sequence
        logits = self.fc(h)                         # (B, vocab_size)
        return F.log_softmax(logits, dim=-1)        # log-softmax for stability
