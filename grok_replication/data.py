# data_custom.py
import torch
import random

N = 32  # domain size

def generate_op_table(op_fn, num_samples=1000):
    X, y = [], []
    for _ in range(num_samples):
        a = random.randint(0, N - 1)
        b = random.randint(0, N - 1)
        label = op_fn(a, b)
        X.append([a, b])
        y.append(label)
    return torch.tensor(X), torch.tensor(y)

# Define some binary operations
def op_mod_xor(a, b): return ((a ^ b) + (a % (b + 1))) % N
def op_rotated_xor(a, b): return (a ^ ((b << 1 | b >> 1) & 0x1F)) % N
def op_bitmask_blend(a, b): return ((a & 0xF) | ((b & 0xF) << 2)) % N
def op_index_shuffle(a, b): return ((a * 3 + b * 5) % (N - 1))
def op_reverse_hybrid(a, b): return a if b % 2 == 0 else b
def op_range_flip(a, b): return (a + b) % N if a < b else (a - b) % N

dataset_registry = {
    "mod_xor": lambda: generate_op_table(op_mod_xor),
    "rotated_xor": lambda: generate_op_table(op_rotated_xor),
    "bitmask_blend": lambda: generate_op_table(op_bitmask_blend),
    "index_shuffle": lambda: generate_op_table(op_index_shuffle),
    "reverse_hybrid": lambda: generate_op_table(op_reverse_hybrid),
    "range_flip": lambda: generate_op_table(op_range_flip),
}
