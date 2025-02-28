import types
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import open_clip
import torch
import torch.nn as nn
from open_clip.model import CLIP
from open_clip.tokenizer import HFTokenizer

from infrastructure.settings import DEVICE



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
    tokenizer_func: Callable[[Any], torch.Tensor] = lambda t: OpenCLIPViT._tokenizer(t)
    
    def __init__(self):
        super().__init__()
        self.model: CLIP = open_clip.create_model_and_transforms(**OpenCLIPViT.INITIALIZE_KWARGS)[0].to(DEVICE)
        self.model.eval()
        
        self._cache: Dict[str, torch.Tensor] = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def load_cache(self, d: Dict[str, torch.Tensor]) -> None:
        self._cache.clear()
        self._cache.update(d)
    
    def update_cache(self, d: Dict[str, torch.Tensor]) -> None:
        self._cache.update(d)
