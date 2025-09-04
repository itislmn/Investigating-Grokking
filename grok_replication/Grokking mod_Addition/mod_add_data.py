import torch
from torch.utils.data import Dataset


class ModularAdditionDataset(Dataset):
    def __init__(self, p=113, split="train", train_frac=0.5, seed=42):
        torch.manual_seed(seed)
        self.p = p
        self.data = []
        for a in range(p):
            for b in range(p):
                x = torch.tensor([a, b], dtype=torch.long)
                y = (a + b) % p
                self.data.append((x, y))

        n = len(self.data)
        split_idx = int(train_frac * n)
        torch.random.manual_seed(seed)
        perm = torch.randperm(n)

        if split == "train":
            self.data = [self.data[i] for i in perm[:split_idx]]
        else:
            self.data = [self.data[i] for i in perm[split_idx:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
