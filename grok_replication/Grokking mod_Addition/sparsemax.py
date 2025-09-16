import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Sparsemax Implementation
# -------------------------
class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - input.mean(dim=self.dim, keepdim=True)  # Centering

        # Sort input in descending order
        z_sorted, _ = torch.sort(input, dim=self.dim, descending=True)
        k = torch.arange(1, input.size(self.dim) + 1, device=input.device).view(
            *[1 if i != self.dim else -1 for i in range(input.dim())]
        )

        # Compute cumulative sum and sparsity mask
        z_cumsum = z_sorted.cumsum(dim=self.dim)
        k_thresh = 1 + k * z_sorted > z_cumsum
        k_max = k_thresh.sum(dim=self.dim, keepdim=True)

        # Compute threshold tau
        z_cumsum_k = z_cumsum.gather(self.dim, k_max - 1)
        tau = (z_cumsum_k - 1) / k_max

        # Compute sparsemax output
        output = torch.clamp(input - tau, min=0)
        return output