import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embedded_size, heads):
        super(SelfAttention, self).__init__()
        self.embedded_size = embedded_size
        self.heads = heads
        self.head_dim = embedded_size // heads

        assert (self.head_dim * heads == self.embedded_size), "embed_size must be divisible by heads"

        self.values = nn.Linear(embedded_size, embedded_size)
        self.keys = nn.Linear(embedded_size, embedded_size)
        self.queries = nn.Linear(embedded_size, embedded_size)
        self.fc_out = nn.Linear(embedded_size, embedded_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embedded_size ** 0.5), dim=3)
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embedded_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedded_size, heads)
        self.norm1 = nn.LayerNorm(embedded_size)
        self.norm2 = nn.LayerNorm(embedded_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedded_size, forward_expansion * embedded_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedded_size, embedded_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedded_size, num_layers,
                 heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embedded_size = embedded_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedded_size)
        self.position_embedding = nn.Embedding(max_length, embedded_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedded_size, heads, dropout = dropout, forward_expansion = forward_expansion)
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embedded_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedded_size, heads)
        self.norm = nn.LayerNorm(embedded_size)
        self.transformer_block = TransformerBlock(embedded_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedded_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embedded_size)
        self.position_embedding = nn.Embedding(max_length, embedded_size)

        self.layers = nn.ModuleList([DecoderBlock(embedded_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embedded_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embedded_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
        ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
        src_vocab_size,
        embedded_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        )

        self.decoder = Decoder(
        trg_vocab_size,
        embedded_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # For classification, we no longer need complex masking
        N, trg_len = trg.shape
        trg_mask = torch.ones(N, 1, 1, trg_len).to(self.device)  # Identity mask for classification
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)  # Pass the target tensor directly
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1,
                 max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        x = self.embedding(src)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Mean pooling over sequence
        return self.fc(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape)



