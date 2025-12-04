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
from modeling.base_vit import BaseViT, OpenCLIPViT
from modeling.image_features import ImageFeatures


class OpenCLIPAdaptiveViT(OpenCLIPViT):
    ModeOptions = Literal["sink", "mask"]
    ExtractOptions = Literal["MA", "AS"]

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
        self.forward_returns: Tuple[str, ...] = ("ranked_mask",)
        
        nn.MultiheadAttention.forward
        # SECTION: Replace resblock.attn.forward
        def new_attn_forward(
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
            mask_condition = mask is not None

            qkv = F.linear(query, _self.in_proj_weight, _self.in_proj_bias)
            query, key, value = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads)
            
            attn_weights = torch.matmul(query, key.mT) / (query.shape[-1] ** 0.5)
            if self.mode == "mask" and mask_condition:
                attn_weights[mask[:, None, None, :].expand_as(attn_weights)] = -torch.inf
            
            attn_matrix = F.softmax(attn_weights, dim=-1)
            if self.mode == "sink" and mask_condition:
                attn_matrix[mask[:, None, None, :].expand_as(attn_matrix)] = 0.0
            
            x = einops.rearrange(torch.matmul(attn_matrix, value), "b h n d -> b n (h d)")
            x = F.linear(x, _self.out_proj.weight, _self.out_proj.bias)
                
            for k in self.attention_returns:
                _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
            return x,
        
        for idx, blk in enumerate(self.model.visual.transformer.resblocks):
            blk: ResidualAttentionBlock
            for handle in self.attention_returns:
                blk.attn.register_module(BaseViT.return_module_name(handle), nn.Identity())
            blk.attn.forward = types.MethodType(new_attn_forward, blk.attn)


        # SECTION: Replace transformer.forward
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            bsz = x.shape[0]
            
            cache: List[torch.Tensor] = []
            for idx, r in enumerate(_self.resblocks[:max(self.mask_layer, self.reset_layer)]):
                x = (r if idx < self.mask_layer else r.forward)(x, attn_mask=None)
                cache.append(x)

            ranked_mask: torch.Tensor = self._cache.get("ranked_mask", None)
            if ranked_mask is not None:
                assert ranked_mask.shape[0] == bsz
            else:
                monitor = Monitor(_self.resblocks[self.detection_layer].get_submodule("attn"), {
                    "return_attn_matrix": [
                        ("attention_matrix", OpenCLIPAdaptiveViT._attention_matrix_hook_fn)
                    ]
                }, device=DEVICE)
                d: Dict[str, List[torch.Tensor]] = monitor.reset()
                
                max_it: float = float("inf") if self.extract == "AS" else 1
                max_num_tokens: int = 1 if self.extract == "AS" else None
                scale: float = 1.0 if self.extract == "AS" else 0.3

                with torch.no_grad():
                    ranked_mask = torch.full((bsz, ImageFeatures.N + 1), torch.inf)
                    it, convergence = 1, False
                
                    original_mode = self.mode
                    self.mode = "mask"
                    while not convergence and it <= max_it:
                        updated_x = cache[self.reset_layer - 1]
                        self.load_cache({"mask": torch.isfinite(ranked_mask)})
                        for r in _self.resblocks[self.reset_layer:self.detection_layer + 1]:
                            updated_x = r.forward(updated_x)

                        new_mask = mask_attention_sink(d["attention_matrix"].pop(), masked_tokens=torch.isfinite(ranked_mask), max_num_tokens=max_num_tokens, scale=scale)
                        ranked_mask[new_mask] = it
                
                        convergence = not torch.any(new_mask).item()
                        it += 1
                        
                    self.mode = original_mode
                monitor.delete()

            self.load_cache({"mask": torch.isfinite(ranked_mask)})
            x = cache[self.mask_layer - 1]
            for r in _self.resblocks[self.mask_layer:]:
                x = r(x)
            
            for k in self.forward_returns:
                _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)
        for handle in self.forward_returns:
            self.model.visual.transformer.register_module(BaseViT.return_module_name(handle), nn.Identity())
    
