import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax
from stablemax import stablemax, log_stablemax

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

        self.log_softmax = nn.LogSoftmax(dim=-1)
        #self.sparsemax = Sparsemax(dim=-1)
        #self.stablemax = lambda x: stablemax(x, dim=-1)
        #self.log_stablemax = lambda x: log_stablemax(x, dim=-1)

    def forward(self, x):
        # x: (batch_size, seq_len=2)
        emb = self.embedding(x)
        h = self.transformer(emb)
        h = h.mean(dim=1)
        logits = self.fc(h)
        return self.log_softmax(logits)        # log-softmax
        #return self.sparsemax(logits)          # sparsemax
        #return self.stablemax(logits)          # stablemax
        #return self.log_softmax(logits)        # log_stablemax

