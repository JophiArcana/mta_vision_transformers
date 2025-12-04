import types
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from open_clip.transformer import ResidualAttentionBlock, Transformer

from infrastructure.settings import SEED, DEVICE
from modeling.base_vit import BaseViT, OpenCLIPViT
from modeling.vit_attention import OpenCLIPAttentionViT



class OpenCLIPProjectionViT(OpenCLIPAttentionViT):
    ModeOptions = Literal[
        "ReLU -> sum",
        "sum -> ReLU",
    ]
    
    def __init__(
        self,
        mask_config: Dict[int, Tuple[ModeOptions, OpenCLIPAttentionViT.MaskOptions]],
        clamp: float = 0.0,
    ):
        OpenCLIPViT.__init__(self)
        self.mask_config: Dict[int, Tuple[OpenCLIPProjectionViT.ModeOptions, OpenCLIPProjectionViT.MaskOptions]] = mask_config.copy()
        self.clamp: float = clamp
        
        self.attention_returns: List[str] = ["attn_matrix", "value_subspace"]



        # SECTION: Replace resblock.forward
        def get_resblock_forward_func_for_layer(idx: int):
            def forward(
                _self: ResidualAttentionBlock,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
            ):
                if idx in self.mask_config:
                    self.update_cache({"layer_input": q_x})
                    
                k_x = _self.ln_1_kv(k_x) if hasattr(_self, "ln_1_kv") and k_x is not None else None
                v_x = _self.ln_1_kv(v_x) if hasattr(_self, "ln_1_kv") and v_x is not None else None
                x = q_x + _self.ls_1(_self.attention(q_x=_self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
                x = x + _self.ls_2(_self.mlp(_self.ln_2(x)))
                return x
            return forward



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
                bsz = query.shape[0]
                
                mask: torch.Tensor = self._cache.get("mask", None)
                mode, mask_type = self.mask_config.get(idx, (None, None))
                mask_condition = mode is not None and mask_type is not None and mask is not None
                
                qkv = Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias)
                query, key, value = einops.rearrange(qkv, "bsz n (qkv h hd) -> qkv bsz h n hd", qkv=3, h=_self.num_heads)
                
                # SUBSECTION: Compute attention matrix
                attn_weights = torch.matmul(query, key.mT) / (query.shape[-1] ** 0.5)
                attn_matrix = Fn.softmax(attn_weights, dim=-1)                                                      # [bsz x n_heads x n x n]

                # SUBSECTION: Compute value subspace
                value = einops.rearrange(value, "bsz h n hd -> h hd (bsz n)")                                       # [n_heads x head_dim x (bsz x n)]
                attn_out_proj = einops.rearrange(_self.out_proj.weight, "d (h hd) -> h d hd", h=_self.num_heads)    # [n_heads x embed_dim x head_dim]
                V = einops.rearrange(attn_out_proj @ value, "h d (bsz n) -> bsz (h n) d", bsz=bsz)                  # [bsz x (n_heads x n) x embed_dim]
                
                # SUBSECTION: Compute the projection modification
                attn_out = einops.rearrange(attn_matrix, "bsz h n1 n2 -> bsz n1 (h n2)") @ V                        # [bsz x n x embed_dim]
                if mask_condition:
                    x = self._cache["layer_input"]
                    x = Fn.normalize(x, p=2, dim=-1)                                                                # [bsz x n x embed_dim]
                    
                    mask = OpenCLIPProjectionViT.process_mask(mask, mask_type)                                      # [bsz x x n x n]                    
                    projection = einops.rearrange(x @ V.mT, "bsz n1 (h n2) -> bsz h n1 n2", h=_self.num_heads)      # [bsz x n_heads x n x n]
                    
                    if mode == "ReLU -> sum":
                        scale = torch.sum(attn_matrix * torch.relu(self.clamp - projection), dim=1)                 # [bsz x n x n]
                    elif mode == "sum -> ReLU":
                        scale = torch.relu(self.clamp - torch.sum(attn_matrix * projection, dim=1))                 # [bsz x n x n]
                
                    scale = torch.sum(mask * scale, dim=2, keepdim=True)                                            # [bsz x n x 1]
                    attn_out = attn_out + x * scale
                attn_out = attn_out + _self.out_proj.bias

                # SECTION: Post-process values for returns
                value_subspace = einops.rearrange(V, "bsz (h n) d -> bsz h n d", h=_self.num_heads)

                for k in self.attention_returns:
                    _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
                return attn_out,
            return forward
        



        for idx, blk in enumerate(self.model.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            for handle in self.attention_returns:
                blk.attn.register_module(BaseViT.return_module_name(handle), nn.Identity())
            blk.forward = types.MethodType(get_resblock_forward_func_for_layer(idx), blk)
            blk.attn.forward = types.MethodType(get_attention_forward_func_for_layer(idx), blk.attn)






