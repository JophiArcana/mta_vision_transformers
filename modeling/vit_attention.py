import types
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.transformer import ResidualAttentionBlock, Transformer

from infrastructure.settings import SEED, DEVICE
from modeling.base_vit import BaseViT, OpenCLIPViT



class OpenCLIPAttentionViT(OpenCLIPViT):
    ModeOptions = Literal[
        "sink",
        "mask",
    ]
    MaskOptions = Literal[
        "T -> T",
        "X -> T",
        "~{T} -> T",
        "~T -> T",
        "T -> X",
        "T -> ~T",
        "X -> X",
        "X = X",
        "T = T",
    ]

    @classmethod
    def process_mask(cls, mask: torch.Tensor, mask_type: MaskOptions) -> torch.Tensor:
        match mask_type:
            case "X -> T":
                return mask[:, None, :]                             # [bsz x 1 x n]
            case "T -> X":
                return mask[:, :, None]                             # [bsz x n x 1]
            case "T = T":
                return torch.diag_embed(mask)                       # [bsz x n x n]
            case "T -> T":
                return mask[:, :, None] * mask[:, None, :]          # [bsz x n x n]
            case "~{T} -> T":
                return mask[:, None, :] * ~mask[:, :, None]         # [bsz x n x n]
            case "~T -> T":
                return mask[:, None, :] * ~torch.diag_embed(mask)   # [bsz x n x n]
            case "T -> ~T":
                return mask[:, :, None] * ~torch.diag_embed(mask)   # [bsz x n x n]
            case "X -> X":
                return torch.tensor(True)   # torch.any(mask)                              # []
            case "X = X":
                return torch.eye(mask.shape[1]).to(torch.bool)      # [n x n]
            case _:
                raise ValueError(mask_type)
    
    def __init__(
        self,
        mask_config: Dict[int, Tuple[ModeOptions, MaskOptions]],
        attn_out_proj_bias: bool = True,
        stop_layer: int = None,
    ):
        OpenCLIPViT.__init__(self)
        self.mask_config: Dict[int, Tuple[OpenCLIPAttentionViT.ModeOptions, OpenCLIPAttentionViT.MaskOptions]] = mask_config.copy()
        self.attn_out_proj_bias: bool = attn_out_proj_bias
        self.stop_layer: int = stop_layer
        
        self.attention_returns: List[str] = ["attn_matrix", "unmasked_attn_matrix"]


        # SECTION: Replace resblock.attn.forward
        def get_attention_forward_func_for_layer(idx: int):
            def forward(
                _self: nn.MultiheadAttention,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True,
                is_causal: bool = False,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                assert key is query and value is query, "Only implemented for k_x and v_x as None"
                mask: torch.Tensor = self._cache.get("mask", None)
                mode, mask_type = self.mask_config.get(idx, (None, None))
                mask_condition = mode is not None and mask_type is not None and mask is not None
                
                qkv = F.linear(query, _self.in_proj_weight, _self.in_proj_bias)
                query, key, value = einops.rearrange(qkv, "bsz n (qkv h d) -> qkv bsz h n d", qkv=3, h=_self.num_heads)
                attn_weights = torch.matmul(query, key.mT) / (query.shape[-1] ** 0.5)
                
                if mask_condition:
                    index = OpenCLIPAttentionViT.process_mask(mask, mask_type).expand_as(attn_weights[:, 0])
                    index = index[:, None].expand_as(attn_weights)
                else:
                    index = None
                    
                if mode == "mask" and mask_condition:
                    attn_weights[index] = -torch.inf
                
                attn_matrix = F.softmax(attn_weights, dim=-1)
                unmasked_attn_matrix = attn_matrix.clone()
                if mode == "sink" and mask_condition:
                    attn_matrix[index] = 0.0
                
                x = einops.rearrange(torch.matmul(attn_matrix, value), "b h n d -> b n (h d)")
                if mask_type == "X -> X" and not self.attn_out_proj_bias:
                    x = F.linear(x, _self.out_proj.weight, None)
                else:
                    x = F.linear(x, _self.out_proj.weight, _self.out_proj.bias)
                    
                for k in self.attention_returns:
                    _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
                return x,
            return forward
        
        for idx, blk in enumerate(self.model.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            for handle in self.attention_returns:
                blk.attn.register_module(BaseViT.return_module_name(handle), nn.Identity())
            blk.attn.forward = types.MethodType(get_attention_forward_func_for_layer(idx), blk.attn)
        
        
        # SECTION: Replace transformer.forward
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            cache = self._cache.get("layer_output", [])
            if len(cache) > 0:
                x = cache[-1].to(DEVICE)
            for idx, r in enumerate(_self.resblocks[len(cache):], start=len(cache)):
                if idx == self.stop_layer:
                    break
                x = r(x, attn_mask=attn_mask)  
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)
