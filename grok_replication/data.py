import random
import torch

N = 32  # Domain size

# Define operations
def op_mod_xor(a, b):
    return ((a ^ b) + (a % (b + 1))) % N

def op_rotated_xor(a, b):
    return (a ^ ((b << 1 | b >> 1) & 0x3F)) % N

def op_bitmask_blend(a, b):
    return ((a & 0xF) | ((b & 0xF) << 2)) % N

def op_index_shuffle(a, b):
    return ((a * 3 + b * 5) % (N - 1))

def op_reverse_hybrid(a, b):
    return a if b % 2 == 0 else b

def op_range_flip(a, b):
    return (a + b) % N if a < b else (a - b) % N

# Registry of all available operations
dataset_registry = {
    "mod_xor": op_mod_xor,
    "rotated_xor": op_rotated_xor,
    "bitmask_blend": op_bitmask_blend,
    "index_shuffle": op_index_shuffle,
    "reverse_hybrid": op_reverse_hybrid,
    "range_flip": op_range_flip,
}

# Generate dataset for a single op
def generate_op_table(op_fn, num_samples=1000):
    X, y = [], []
    for _ in range(num_samples):
        a = random.randint(0, N - 1)
        b = random.randint(0, N - 1)
        label = op_fn(a, b)
        X.append([a, b])
        y.append(label)

    print("First 5 samples of data:")
    for i in range(5):
        print(f"Input: {X[i]}, Label: {y[i]}")
    return torch.tensor(X), torch.tensor(y)

# Generate dataset from mixed operations
def generate_mixed_dataset(num_samples=1000):
    ops = list(dataset_registry.values())
    X, y = [], []
    for _ in range(num_samples):
        a = random.randint(0, N - 1)
        b = random.randint(0, N - 1)
        op_fn = random.choice(ops)  # Random op for each sample
        label = op_fn(a, b)
        X.append([a, b])
        y.append(label)

    print("First 5 samples of mixed-op data:")
    for i in range(5):
        print(f"Input: {X[i]}, Label: {y[i]}")
    return torch.tensor(X), torch.tensor(y)

# TESTING
if __name__ == "__main__":
    print("Single-op dataset (mod_xor):")
    X1, y1 = generate_op_table(op_mod_xor, 10)

    print("\nMixed-op dataset:")
    X2, y2 = generate_mixed_dataset(10)
