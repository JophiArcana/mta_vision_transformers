import gc
import types
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP
from open_clip.transformer import ResidualAttentionBlock, Transformer
from torch.utils._pytree import tree_flatten

from core.monitor import Monitor
from core.attention_sink import mask_attention_sink
from infrastructure import utils
from infrastructure.settings import DEVICE
from modeling.image_features import ImageFeatures
from modeling.openclip_vit import OpenCLIPViT



class OpenCLIPAdaptiveViT(OpenCLIPViT):
    ModeOptions = Literal["sink", "mask"]
    ExtractOptions = Literal["MA", "AS"]
    
    _cache: Dict[str, torch.Tensor] = {}
    
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"

    @classmethod
    def _attention_matrix_hook_fn(cls, model_: nn.Module, input_: Any, output_: Any) -> Any:
        return torch.mean(einops.rearrange(
            tree_flatten(output_)[0][0],
            "b h n1 n2 -> b n1 n2 h"
        ), dim=-1)
    
    def __init__(
        self,
        mode: ModeOptions,
        extract: ExtractOptions,
        mask_layer: int,
        reset_layer: int,
        detection_layer: int,
    ):
        OpenCLIPViT.__init__(self)
        self.mode: OpenCLIPAdaptiveViT.ModeOptions = mode
        self.extract: OpenCLIPAdaptiveViT.ExtractOptions = extract
        self.mask_layer: int = mask_layer
        self.reset_layer: int = reset_layer
        self.detection_layer: int = detection_layer
        
        self.attention_returns: Tuple[str, ...] = ("attn_matrix",)
        self.forward_returns: Tuple[str, ...] = ("mask",)
        

        # SECTION: Replace resblock.attention
        def new_resblock_attention(
            _self: ResidualAttentionBlock,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None, 
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
        ):
            assert k_x is None and v_x is None, "Only implemented for k_x and v_x as None"
            mask_condition = attn_mask is not None

            qkv = F.linear(q_x, _self.attn.in_proj_weight, _self.attn.in_proj_bias)
            q_x, k_x, v_x = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.attn.num_heads)
            
            attn_weights = torch.matmul(q_x, k_x.mT) / (q_x.shape[-1] ** 0.5)
            if self.mode == "mask" and mask_condition:
                attn_weights[attn_mask[:, None, None, :].expand_as(attn_weights)] = -torch.inf
            
            attn_matrix = F.softmax(attn_weights, dim=-1)
            if self.mode == "sink" and mask_condition:
                attn_matrix[attn_mask[:, None, None, :].expand_as(attn_matrix)] = 0.0
            
            x = einops.rearrange(torch.matmul(attn_matrix, v_x), "b h n d -> b n (h d)")
            x = F.linear(x, _self.attn.out_proj.weight, _self.attn.out_proj.bias)
                
            for k in self.attention_returns:
                _self.get_submodule(OpenCLIPAdaptiveViT.return_module_name(k))(locals()[k])
            return x
        
        for blk in self.model.visual.transformer.resblocks:
            blk: ResidualAttentionBlock
            for handle in self.attention_returns:
                blk.register_module(OpenCLIPAdaptiveViT.return_module_name(handle), nn.Identity())
            blk.attention = types.MethodType(new_resblock_attention, blk)


        # SECTION: Replace transformer.forward
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            bsz = x.shape[0]
            
            cache: List[torch.Tensor] = []
            for idx, r in enumerate(_self.resblocks[:max(self.mask_layer, self.reset_layer)]):
                x = (r if idx < self.mask_layer else r.forward)(x, attn_mask=None)
                cache.append(x)

            mask: torch.Tensor = OpenCLIPAdaptiveViT._cache.get("mask", None)
            if mask is not None:
                assert mask.shape[0] == bsz
            else:
                monitor = Monitor(_self.resblocks[self.detection_layer], {
                    "return_attn_matrix": [
                        ("attention_matrix", OpenCLIPAdaptiveViT._attention_matrix_hook_fn)
                    ]
                }, device=DEVICE)
                d: Dict[str, List[torch.Tensor]] = monitor.reset()
                
                max_it: float = float("inf") if self.extract == "AS" else 1
                max_num_tokens: int = 1 if self.extract == "AS" else None
                scale: float = 1.0 if self.extract == "AS" else 0.4
                    
                with torch.no_grad():
                    mask = torch.full((bsz, ImageFeatures.N + 1), False)
                    it = 1
                    convergence = torch.full((bsz,), False)
                
                    original_mode = self.mode
                    self.mode = "mask"
                    while not torch.all(convergence):
                        if it > max_it:
                            break
                        
                        updated_x = cache[self.reset_layer - 1]
                        for r in _self.resblocks[self.reset_layer:self.detection_layer + 1]:
                            updated_x = r.forward(updated_x, attn_mask=mask)
                        
                        new_mask = mask_attention_sink(d["attention_matrix"].pop(), masked_tokens=mask, max_num_tokens=max_num_tokens, scale=scale)
                        mask = mask + new_mask
                
                        # SECTION: Check convergence
                        convergence = torch.sum(new_mask, dim=1) == 0
                        
                        # SECTION: Cleanup
                        del updated_x, new_mask
                        utils.empty_cache()
                    self.mode = original_mode
                monitor.delete()

            x = cache[self.mask_layer - 1]
            for r in _self.resblocks[self.mask_layer:]:
                x = r(x, attn_mask=mask)
            
            for k in self.forward_returns:
                _self.get_submodule(OpenCLIPAdaptiveViT.return_module_name(k))(locals()[k])
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)
        for handle in self.forward_returns:
            self.model.visual.transformer.register_module(OpenCLIPAdaptiveViT.return_module_name(handle), nn.Identity())
    
