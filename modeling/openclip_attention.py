import types
from typing import Callable, Dict, List, Literal, Optional, Tuple

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock, Transformer

from infrastructure.settings import SEED, DEVICE



ModeOptions = Literal[
    "default",
    "sink",
    "mask",
]
class OpenCLIPViT(nn.Module):
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"
    
    def __init__(
        self,
        mask_layer: int,
        mask: torch.Tensor,
        mode: ModeOptions = "default",
        cache: List[torch.Tensor] = [],
        stop_layer: int = None
    ):
        super().__init__()
        self._model: CLIP = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)[0]
        
        self.mask_layer: int = mask_layer
        self.mask: torch.Tensor = mask
        self.mode: ModeOptions = mode
        self.cache: torch.Tensor = [] if cache is None else cache
        self.stop_layer: int = stop_layer
        
        self.attention_returns: List[str] = ["attn_matrix",]
        

        def get_attention_func_for_layer(idx: int):
            def attention(
                _self: ResidualAttentionBlock,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None, 
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None
            ):
                assert k_x is None and v_x is None, "Only implemented for k_x and v_x as None"
                mask_condition = idx >= self.mask_layer and self.mask is not None

                attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
                
                qkv = F.linear(q_x, _self.attn.in_proj_weight, _self.attn.in_proj_bias)
                q_x, k_x, v_x = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.attn.num_heads)
                
                attn_weights = torch.matmul(q_x, k_x.mT) / (q_x.shape[-1] ** 0.5)
                if attn_mask is not None:
                    attn_weights = attn_weights + attn_mask
                
                if self.mode == "mask" and mask_condition:
                    attn_weights[mask[:, None, None, :].expand_as(attn_weights)] = -torch.inf
                
                attn_matrix = F.softmax(attn_weights, dim=-1)
                
                if self.mode == "sink" and mask_condition:
                    attn_matrix[mask[:, None, None, :].expand_as(attn_matrix)] = 0.0
                
                x = einops.rearrange(torch.matmul(attn_matrix, v_x), "b h n d -> b n (h d)")
                x = F.linear(x, _self.attn.out_proj.weight, _self.attn.out_proj.bias)
                    
                for k in self.attention_returns:
                    _self.get_submodule(OpenCLIPViT.return_module_name(k))(locals()[k])
                return x
            return attention
        
        for idx, blk in enumerate(self._model.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            for handle in self.attention_returns:
                blk.register_module(OpenCLIPViT.return_module_name(handle), nn.Identity())
            blk.attention = types.MethodType(get_attention_func_for_layer(idx), blk)
        
        


        
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            if len(self.cache) > 0:
                x = self.cache[-1].to(DEVICE)
            for idx, r in enumerate(_self.resblocks[len(self.cache):]):
                if idx == stop_layer:
                    break
                x = r(x, attn_mask=attn_mask)  
            return x
        
        self._model.visual.transformer.forward = types.MethodType(new_transformer_forward, self._model.visual.transformer)






        self._model.eval()
        
    def forward(self, x):
        return self._model(x)
