import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mod_add_data import ModularAdditionDataset
from small_transformer import SmallTransformer
from visualize_mod_add import plot_grokking
from visualisation_params import plot_training_dynamics

# --- Accuracy helper ---
def accuracy(logits, y):
    preds = torch.argmax(logits, dim=-1)
    return (preds == y).float().mean().item()

# --- Dataset ---
train_dataset = ModularAdditionDataset(p=113, split="train", train_frac=0.5)
val_dataset   = ModularAdditionDataset(p=113, split="val", train_frac=0.5)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128)

# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallTransformer(vocab_size=113).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0)
criterion = nn.NLLLoss()   # if model outputs log-softmax, log-stablemax
#criterion = nn.MSELoss()  # if model outputs sparse-softmax
#criterion = nn.KLDivLoss(reduction='batchmean') # if model outputs stablemax

# --- Training loop ---
train_acc_hist, val_acc_hist = [], []
loss_hist, grad_hist, wd_hist = [], [], []

max_steps = 100000
log_interval = 10
switch_step = 24000
scheduler = None

for step in range(max_steps+1):
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    if step % log_interval == 0:

        #metrics
        loss_hist.append(loss.item())
        grad_hist.append(grad_norm)
        wd_hist.append(optimizer.param_groups[0]["weight_decay"])

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
    if step == switch_step:
        for g in optimizer.param_groups:
            g['weight_decay'] = 1e-4
        for l in optimizer.param_groups:
            l['lr'] = 2e-3
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_steps - switch_step)

# --- Plot grokking ---
plot_grokking(train_acc_hist, val_acc_hist, log_interval, max_steps)
plot_training_dynamics(loss_hist, grad_hist, wd_hist, log_interval, switch_step)
