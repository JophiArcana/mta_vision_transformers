import types
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP
from open_clip.tokenizer import HFTokenizer
from open_clip.transformer import ResidualAttentionBlock

from infrastructure.settings import SEED, DEVICE



class OpenCLIPViT(nn.Module):
    INITIALIZE_KWARGS: Dict[str, Any] = {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "force_quick_gelu": True,
    }
    # Image preprocess
    preprocess_func: Callable[[Any], torch.Tensor] = open_clip.create_model_and_transforms(**INITIALIZE_KWARGS)[2]
    # Text preprocess
    _tokenizer: HFTokenizer = open_clip.get_tokenizer(INITIALIZE_KWARGS["model_name"])
    tokenizer_func: Callable[[Any], torch.Tensor] = lambda t: OpenCLIPViT._tokenizer(t)[0]
    
    def __init__(self):
        super().__init__()
        self.model: CLIP = open_clip.create_model_and_transforms(**OpenCLIPViT.INITIALIZE_KWARGS)[0]
        self.model.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
