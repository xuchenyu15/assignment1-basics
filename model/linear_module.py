import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum

class LinearModule(nn.Module): 
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device = device,
                dtype = dtype
            )
        )
        self.init_weight()
        return 

    def init_weight(self):
        sigma = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std = sigma,
            a = -3 * sigma,
            b = 3 * sigma,
        )

    def set_weight(self, weights: Float[Tensor, " d_out d_in"]): 
        if not weights.shape == self.weight.shape:
            raise ValueError("Shape doesn't match")
        with torch.no_grad():
            self.weight.copy_(weights)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")