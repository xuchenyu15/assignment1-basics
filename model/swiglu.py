import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum
from .linear_module import LinearModule as LM

class SwiGLU(nn.Module): 
    def __init__(
        self, 
        d_model: int,
        d_ff: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = LM(d_model, d_ff, device, dtype)
        self.w2 = LM(d_ff, d_model, device, dtype)
        self.w3 = LM(d_model, d_ff, device, dtype)
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # self.w1(x) is the standard way to invoke a PyTorch layer, which include forward and surrounding machinery.
        # w1x = self.w1.forward(x)
        # w3x = self.w3.forward(x)
        # swiglu = self.w2.forward(silu * w3x)
        w1x = self.w1(x)
        silu = w1x * torch.sigmoid(w1x)
        w3x = self.w3(x)
        swiglu = self.w2(silu * w3x)

        return swiglu