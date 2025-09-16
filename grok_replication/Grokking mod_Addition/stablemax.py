import torch
def stable_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 1, 1 / (1 - x))

def stablemax(input: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    transformed = stable_transform(input)
    Z = transformed.sum(dim=dim, keepdim=True) + eps
    return transformed / Z

def log_stablemax(input: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    transformed = stable_transform(input)
    Z = transformed.sum(dim=dim, keepdim=True) + eps
    return torch.log(transformed + eps) - torch.log(Z)
