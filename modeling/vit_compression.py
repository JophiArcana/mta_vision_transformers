import itertools
import types
from typing import Callable, Dict, List, Literal, Optional, OrderedDict, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.transformer import ResidualAttentionBlock, Transformer

from infrastructure.settings import SEED, DEVICE
from modeling.image_features import ImageFeatures
from modeling.base_vit import BaseViT, OpenCLIPViT



class OpenCLIPCompressionViT(OpenCLIPViT):
    ModeOptions = Literal[
        "default",
        "compression",
    ]
    
    def __init__(
        self,
        mode: ModeOptions,
        rank: int,
        mask_layer: int,
    ):
        OpenCLIPViT.__init__(self)
        self.mode: OpenCLIPCompressionViT.ModeOptions = mode
        self.rank: int = rank
        self.mask_layer: int = mask_layer
        
        self.attention_returns: List[str] = ["attn_matrix",]
        nn.MultiheadAttention.forward()

        # SECTION: Replace resblock.attn.forward
        def get_attention_forward_func_for_layer(idx: int):
            def attention(
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
                mask_dict: Dict[str, torch.Tensor] = self._cache.get("mask_dict", {})                
                compression_condition = idx >= self.mask_layer
                
                qkv = F.linear(query, _self.in_proj_weight, _self.in_proj_bias)
                query, key, value = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads)
                
                attn_weights = torch.matmul(query, key.mT) / (query.shape[-1] ** 0.5)
                attn_matrix = F.softmax(attn_weights, dim=-1)
                
                
                
                if self.mode == "compression" and compression_condition:
                    bsz = query.shape[0]
                    
                    mask_list: List[torch.Tensor] = [(torch.arange(ImageFeatures.N + 1) == 0).expand((bsz, -1))]
                    normal_mask = (torch.arange(ImageFeatures.N + 1) > 0).expand((bsz, -1))
                    for mask in mask_dict.values():
                        mask_list.append(normal_mask * mask)
                        normal_mask = normal_mask * ~mask
                    mask_list.append(normal_mask)
                    
                    if len(mask_list) > 2:
                        for image_idx, m1, m2 in itertools.product(range(bsz), mask_list, mask_list):
                            matrix_idx = (image_idx, slice(None, None, None), torch.where(m1[image_idx])[0][:, None], torch.where(m2[image_idx])[0][None, :])
                            U, S, V = torch.svd_lowrank(attn_matrix[matrix_idx], q=self.rank)
                            attn_matrix[matrix_idx] = U @ torch.diag_embed(S, dim1=-2, dim2=-1) @ V.mT
                    else:
                        for m1, m2 in itertools.product(mask_list, mask_list):
                            matrix_idx = (slice(None, None, None), slice(None, None, None), torch.where(m1[0])[0][:, None], torch.where(m2[0])[0][None, :])
                            U, S, V = torch.svd_lowrank(attn_matrix[matrix_idx], q=self.rank)
                            attn_matrix[matrix_idx] = U @ torch.diag_embed(S, dim1=-2, dim2=-1) @ V.mT



                
                x = einops.rearrange(torch.matmul(attn_matrix, value), "b h n d -> b n (h d)")
                x = F.linear(x, _self.out_proj.weight, _self.out_proj.bias)
                    
                for k in self.attention_returns:
                    _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
                return x,
            return attention
        
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
            for idx, r in enumerate(_self.resblocks[len(cache):]):
                x = r(x, attn_mask=attn_mask)  
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)
