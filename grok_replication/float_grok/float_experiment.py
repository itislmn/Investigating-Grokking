# =========================================================
# GROKKING NUMERICAL PRECISION EXPERIMENT
# Compare float64 / float32 / mixed-fp16
# =========================================================
import torch, torch.nn as nn, torch.optim as optim
import math, random, numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Synthetic Dataset: modular addition ----------
def make_mod_addition_dataset(modulus=97, split=0.8):
    X, Y = [], []
    for a in range(modulus):
        for b in range(modulus):
            X.append([a, b])
            Y.append((a + b) % modulus)
    X, Y = np.array(X), np.array(Y)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    split_idx = int(len(X) * split)
    return (torch.tensor(X[:split_idx]), torch.tensor(Y[:split_idx]),
            torch.tensor(X[split_idx:]), torch.tensor(Y[split_idx:]))

train_X, train_Y, val_X, val_Y = make_mod_addition_dataset(97)
train_X, val_X = train_X.long(), val_X.long()
train_Y, val_Y = train_Y.long(), val_Y.long()

# ---------- Transformer Model ----------
class TinyTransformer(nn.Module):
    def __init__(self, vocab=97, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab)
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # average pooling
        return self.fc(x)

# ---------- Training function ----------
def train_grokking(dtype_mode="float32", epochs=2000):
    model = TinyTransformer().to(device)

    # set model dtype (not input indices!)
    if dtype_mode == "float64":
        torch.set_default_dtype(torch.float64)
        model = model.to(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_mode == "amp_fp16" and device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": []}

    def accuracy(model, X, Y, batch=256):
        with torch.no_grad():
            acc = 0
            for i in range(0, len(X), batch):
                xb, yb = X[i:i+batch].to(device), Y[i:i+batch].to(device)
                preds = model(xb).argmax(dim=-1)
                acc += (preds == yb).sum().item()
            return acc / len(X)

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        xb, yb = train_X.to(device), train_Y.to(device)

        if dtype_mode == "amp_fp16" and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(xb)
                loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

        if ep % 10 == 0:
            train_acc = accuracy(model, train_X, train_Y)
            val_acc = accuracy(model, val_X, val_Y)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            print(f"[{dtype_mode}] Epoch {ep:04d} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

        if ep % 100 == 0:
            optimizer.param_groups[0]['lr'] *= 0.98

    torch.set_default_dtype(torch.float32)
    return history


# ---------- Run Experiments ----------
hist64 = train_grokking("float64", epochs=1000)
hist32 = train_grokking("float32", epochs=1000)
hist16 = train_grokking("amp_fp16", epochs=1000)

# ---------- Plot ----------
epochs = np.arange(0, len(hist64["train_acc"]))*10
plt.figure(figsize=(10,6))
plt.plot(epochs, hist64["train_acc"], label="Train float64", c="blue", alpha=0.5, solid_capstyle="round")
plt.plot(epochs, hist64["val_acc"], label="Val float64", c="blue", solid_capstyle="round")
plt.plot(epochs, hist32["train_acc"], label="Train float32", c="green", alpha=0.5, solid_capstyle="round")
plt.plot(epochs, hist32["val_acc"], label="Val float32", c="green", solid_capstyle="round")
plt.plot(epochs, hist16["train_acc"], label="Train fp16 (mixed)", c="red", alpha=0.5, solid_capstyle="round")
plt.plot(epochs, hist16["val_acc"], label="Val fp16 (mixed)", c="red", solid_capstyle="round")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Numerical Precision vs Grokking")
plt.legend()
plt.savefig('Plots/numerical_precision_vs_grokking.png')
plt.show()
