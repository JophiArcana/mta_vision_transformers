import types
from typing import Callable, Dict, List, Literal, Optional, Tuple

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock

from infrastructure.settings import SEED, DEVICE



ModeOptions = Literal[
    "default",
    "concatenation",
    "mean_substitution",
    "permutation",
    "mean_concatenation",
    "permutation_concatenation",
]
class OpenCLIPViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: CLIP = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)[0]
        self.model.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
