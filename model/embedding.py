import torch.nn as nn
import torch
import math
from jaxtyping import Float, Int
from torch import Tensor

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device = device,
                dtype = dtype
            )
        )
        self.init_weight()
        return 

    def init_weight(self):
        sigma = math.sqrt(2 / (self.num_embeddings + self.embedding_dim))
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = sigma, 
            a = -3 * sigma,
            b = 3 * sigma 
        )

    def set_weight(self, weights: Float[Tensor, " vocab_size d_model"]):
        if not weights.shape == self.weight.shape:
            raise ValueError("Shape doesn't match")
        with torch.no_grad():
            self.weight.copy_(weights)
        return

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]: 
        return self.weight[token_ids]
    
