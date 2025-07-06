import itertools
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock

from modeling.base_vit import OpenCLIPViT


class DistanceEmbedding(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 2):
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        
        dims: List[int] = [self.in_dim] + [1 << k for k in range(4, 0, -1)] + [self.out_dim]
        self.embedding: nn.Sequential = nn.Sequential(*[*itertools.chain(*(
            (nn.Linear(*d), nn.ReLU())
            for d in itertools.pairwise(dims)
        ))][:-1])
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(self.out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.squeeze(self.decoder(self.embedding(x)), dim=-1))




