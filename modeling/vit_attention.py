import types
from typing import Callable, Dict, List, Literal, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.transformer import ResidualAttentionBlock, Transformer

from infrastructure.settings import SEED, DEVICE
from modeling.openclip_vit import OpenCLIPViT



class OpenCLIPAttentionViT(OpenCLIPViT):
    ModeOptions = Literal[
        "default",
        "sink",
        "mask",
    ]
    
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"
    
    def __init__(
        self,
        mode: ModeOptions,
        mask_layer: int,
        mask: torch.Tensor,
        cache: List[torch.Tensor] = [],
        stop_layer: int = None
    ):
        OpenCLIPViT.__init__(self)
        self.mode: OpenCLIPAttentionViT.ModeOptions = mode
        self.mask_layer: int = mask_layer
        self.mask: torch.Tensor = mask
        self.cache: torch.Tensor = [] if cache is None else cache
        self.stop_layer: int = stop_layer
        
        self.attention_returns: List[str] = ["attn_matrix",]


        # SECTION: Replace resblock.attention
        def get_attention_func_for_layer(idx: int):
            def attention(
                _self: ResidualAttentionBlock,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None, 
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
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
                    attn_weights[self.mask[:, None, None, :].expand_as(attn_weights)] = -torch.inf
                
                attn_matrix = F.softmax(attn_weights, dim=-1)
                
                if self.mode == "sink" and mask_condition:
                    attn_matrix[self.mask[:, None, None, :].expand_as(attn_matrix)] = 0.0
                
                x = einops.rearrange(torch.matmul(attn_matrix, v_x), "b h n d -> b n (h d)")
                x = F.linear(x, _self.attn.out_proj.weight, _self.attn.out_proj.bias)
                    
                for k in self.attention_returns:
                    _self.get_submodule(OpenCLIPAttentionViT.return_module_name(k))(locals()[k])
                return x
            return attention
        
        for idx, blk in enumerate(self.model.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            for handle in self.attention_returns:
                blk.register_module(OpenCLIPAttentionViT.return_module_name(handle), nn.Identity())
            blk.attention = types.MethodType(get_attention_func_for_layer(idx), blk)
        
        
        # SECTION: Replace transformer.forward
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            if len(self.cache) > 0:
                x = self.cache[-1].to(DEVICE)
            for idx, r in enumerate(_self.resblocks[len(self.cache):]):
                if idx == self.stop_layer:
                    break
                x = r(x, attn_mask=attn_mask)  
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)
