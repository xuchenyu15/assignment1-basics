import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum

class RMSNorm(nn.Module): 
    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.ones(
                (d_model),
                device = device,
                dtype = dtype
            )
        )
        return

    def set_weight(self, weights: Float[Tensor, " d_model"]):
        if not weights.shape == self.weight.shape:
            raise ValueError("Shape doesn't match")
        with torch.no_grad():
            self.weight.copy_(weights)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.square().mean(dim = -1, keepdim = True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_type)