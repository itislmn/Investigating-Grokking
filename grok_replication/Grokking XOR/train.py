import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from simple_model import *
from data import *
from visualize import plot_grokking  # Import the plot function

# Step 0: Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mod_xor', help='Which dataset to use (e.g. mod_xor, mixed, range_flip)')
parser.add_argument('--num_samples', type=int, default=1000, help='Number of total samples (train+val)')
args = parser.parse_args()

# Step 1: Create a custom dataset class
class XORDataset(Dataset):
    def __init__(self, name, num_samples=256):
        if name == "mixed":
            self.X, self.y = generate_mixed_dataset(num_samples)
        elif name in dataset_registry:
            op_fn = dataset_registry[name]
            self.X, self.y = generate_op_table(op_fn, num_samples)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 2: Set up dataset and dataloaders with 50/50 split
batch_size = 32
dataset = XORDataset(args.dataset, args.num_samples)

# Split 50/50 train/val
train_size = len(dataset) // 2
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 3: Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = N
num_classes = N
model = TransformerClassifier(
    vocab_size=vocab_size,
    num_classes=num_classes,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.2,
    max_len=2
).to(device)

# Step 4: Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.025)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 25 else 0.95)

# Step 5: Training loop
num_epochs = 10
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, dim=1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            _, predicted = torch.max(output, dim=1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
    val_acc = val_correct / val_total * 100
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")

# Step 6: Plot
plot_grokking(train_accuracies, val_accuracies)
