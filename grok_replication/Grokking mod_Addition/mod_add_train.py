import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mod_add_data import ModularAdditionDataset
from small_transformer import SmallTransformer
from visualize_mod_add import plot_grokking

# --- Accuracy helper ---
def accuracy(logits, y):
    preds = torch.argmax(logits, dim=-1)
    return (preds == y).float().mean().item()

# --- Dataset ---
train_dataset = ModularAdditionDataset(p=113, split="train", train_frac=0.5)
val_dataset   = ModularAdditionDataset(p=113, split="val", train_frac=0.5)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)

# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallTransformer(vocab_size=113).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.NLLLoss()   # since model outputs log-softmax

# --- Training loop ---
train_acc_hist, val_acc_hist = [], []

max_steps = 100_000
log_interval = 100

for step in range(max_steps):
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if step % log_interval == 0:
        # Training accuracy
        train_acc = accuracy(out, y) * 100
        train_acc_hist.append(train_acc)

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            val_correct = 0
            total = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out_val = model(xb)
                preds = torch.argmax(out_val, dim=-1)
                val_correct += (preds == yb).sum().item()
                total += yb.size(0)
            val_acc = val_correct / total * 100
        val_acc_hist.append(val_acc)

        print(f"Step {step}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")

# --- Plot grokking ---
plot_grokking(train_acc_hist, val_acc_hist)
