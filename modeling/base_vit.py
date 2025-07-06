import types
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import open_clip
import torch
import torch.nn as nn
from open_clip.model import CLIP
from open_clip.tokenizer import HFTokenizer

from infrastructure.settings import DEVICE
from transformers import BitImageProcessor, Dinov2Model


class BaseViT(nn.Module):
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"
    
    def __init__(self, model: nn.Module):
        nn.Module.__init__(self)
        self.model = model.to(DEVICE).eval()
        self._cache: Dict[str, torch.Tensor] = {}
    
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(*args, **kwargs)
    
    def load_cache(self, d: Dict[str, torch.Tensor]) -> None:
        self._cache.clear()
        self._cache.update(d)
    
    def update_cache(self, d: Dict[str, torch.Tensor]) -> None:
        self._cache.update(d)
    
    def model_args(self) -> str:
        k, v = map(list, zip(*vars(self).items()))
        idx = k.index("_cache")
        return "_".join((f"{k[i]}:{v[i]}" for i in range(idx + 1, len(k))))


class OpenCLIPViT(BaseViT):    
    INITIALIZE_KWARGS: Dict[str, Any] = {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "force_quick_gelu": True,
        "force_image_size": 224,
    }
    # Image preprocess
    preprocess_func: Callable[[Any], torch.Tensor] = open_clip.create_model_and_transforms(**INITIALIZE_KWARGS)[2]
    # Text preprocess
    tokenizer_func: HFTokenizer = open_clip.get_tokenizer(INITIALIZE_KWARGS["model_name"])
    
    def __init__(self):
        BaseViT.__init__(self, open_clip.create_model_and_transforms(**OpenCLIPViT.INITIALIZE_KWARGS)[0])
        self.model: CLIP
        
        
class DINOv2ViT(BaseViT):
    BASE_MODEL_NAME = "facebook/dinov2-large"
    image_processor: BitImageProcessor = BitImageProcessor.from_pretrained(BASE_MODEL_NAME)
    
    def __init__(self):
        BaseViT.__init__(self, Dinov2Model.from_pretrained(DINOv2ViT.BASE_MODEL_NAME))
        self.model: Dinov2Model
