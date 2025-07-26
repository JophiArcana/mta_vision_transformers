from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock

from modeling.base_vit import OpenCLIPViT


class MAClassifier(nn.Module):
    def __init__(self, model: CLIP, start: int, end: int):
        super().__init__()
        if model is None:
            model = open_clip.create_model_and_transforms(**OpenCLIPViT.INITIALIZE_KWARGS)[0]
        
        self.biases: List[nn.Parameter] = []
        self.lns: List[nn.LayerNorm] = []
        self.mlps: List[nn.Sequential] = []
        for r in model.visual.transformer.resblocks[start:end]:
            r: ResidualAttentionBlock
            self.biases.append(nn.Parameter(r.attn.out_proj.bias))
            self.lns.append(r.ln_2)
            self.mlps.append(r.mlp)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        for bias, ln, mlp in zip(self.biases, self.lns, self.mlps):
            x = x + bias.to(device)
            x = x + mlp.to(device)(ln.to(device)(x))
        return x
        # return torch.norm(x, dim=-1)




