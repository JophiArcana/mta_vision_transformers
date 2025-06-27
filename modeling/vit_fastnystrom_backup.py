import itertools
import types
from typing import Callable, Dict, Iterable, List, Literal, Optional, OrderedDict, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from open_clip.transformer import ResidualAttentionBlock, Transformer
from pytorch3d.ops import sample_farthest_points

from infrastructure import utils
from infrastructure.settings import SEED, DEVICE
from modeling.openclip_vit import OpenCLIPViT



class OpenCLIPFastNystromCompressionViT(OpenCLIPViT):
    ModeOptions = Literal[
        "fps",
        "uniform",
    ]
    
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"
    
    @classmethod
    def invert(cls, A: torch.Tensor) -> torch.Tensor:
        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
        I = torch.eye(A.shape[-1], device=A.device)
        Z = 1 / torch.max(torch.sum(A, dim=-2, keepdim=True), dim=-1, keepdim=True).values * A.mT
        for _ in range(6):
            AZ = A @ Z
            Z = 0.25 * Z @ (13 * I - AZ @ (15 * I - AZ @ (7 * I - AZ)))
        return Z
    
    def __init__(
        self,
        mode: ModeOptions,
        num_sample: int,
        resample_fps: bool,
        use_layer_input: bool,
        mask_layers: Iterable[int],
    ):
        OpenCLIPViT.__init__(self)
        self.mode: OpenCLIPFastNystromCompressionViT.ModeOptions = mode
        self.num_sample: int = num_sample
        self.resample_fps: bool = resample_fps
        self.use_layer_input: bool = use_layer_input
        self.mask_layers: List[int] = [*mask_layers]
        
        self.attention_returns: List[str] = ["sample_indices"]
        

        # SECTION: Replace resblock.attn.forward
        def new_attention_forward(
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
            
            fps_input = query
            qkv = Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias)
            query, key, value = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads)
            
            bsz = query.shape[0]
            bsz_index = torch.arange(bsz)[:, None]
            head_dim = query.shape[-1]
            invsqrt_d = head_dim ** -0.5
            
            def index(t: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
                return t[bsz_index, :, sample_indices, :].transpose(dim0=1, dim1=2)
                
            # SECTION: Construct the queries and keys used to extrapolate the attention matrix            
            sample_indices = self._cache.get("sample_indices", None)
            if self.resample_fps or sample_indices is None:
                guarantee_mask = mask_dict["guarantee"]                                 # bool: [bsz x N]
                exclude_mask = mask_dict["exclude"]                                     # bool: [bsz x N]
                
                restricted_samples = self.num_sample - torch.sum(guarantee_mask, dim=1) # int: [bsz]
                max_restricted_samples: int = torch.max(restricted_samples).item()      # int       
                restricted_mask = ~guarantee_mask * ~exclude_mask                       # bool: [bsz x N]
    
                if self.mode == "fps":
                    # if self.use_layer_input:
                    #     fps_input: torch.Tensor = self._cache["layer_input"]

                    counts = torch.sum(restricted_mask, dim=1)                              # [bsz]
                    max_count: int = torch.max(counts).item()                               # int

                    topk_indices = torch.topk(restricted_mask.to(torch.int), k=max_count, dim=1).indices# int: [bsz x max_count]
                    fps_indices = sample_farthest_points(
                        fps_input[bsz_index, topk_indices],
                        lengths=counts, K=restricted_samples,
                    )[1]                                                                                # int: [bsz x max_restricted_samples]
                    sample_indices = torch.cat((
                        topk_indices, torch.full((bsz, 1), -1)
                    ), dim=1)[bsz_index, fps_indices]                                                   # int: [bsz x max_restricted_samples]
                
                else:
                    sort_weights = torch.rand(restricted_mask.shape) + restricted_mask.to(torch.float)  # float: [bsz x N]
                    sample_indices = torch.topk(sort_weights, k=max_restricted_samples, dim=1).indices  # int: [bsz x max_restricted_samples]
                    sample_indices[torch.arange(max_restricted_samples) >= restricted_samples[:, None]] = -1

                if self.num_sample > max_restricted_samples:
                    sample_indices = torch.cat((
                        sample_indices, torch.full((bsz, self.num_sample - max_restricted_samples), -1)
                    ), dim=1)
                sample_indices[sample_indices == -1] = torch.where(guarantee_mask)[1]
                
                self.update_cache({"sample_indices": sample_indices})

            qp, kp = index(query, sample_indices), index(key, sample_indices)                       # float: [bsz x h x num_sample x d]   
            B = torch.softmax((invsqrt_d * qp) @ key.mT, dim=-1)                                    # float: [bsz x h x num_sample x N]
            BT = torch.softmax(query @ (invsqrt_d * kp.mT), dim=-1)                                 # float: [bsz x h x N x num_sample]
            A = BT[bsz_index, :, sample_indices, :].transpose(dim0=1, dim1=2)                       # float: [bsz x h x num_sample x num_sample]
            x = (BT @ OpenCLIPFastNystromCompressionViT.invert(A)) @ (B @ value)                    # float: [bsz x h x N x d]

            x = einops.rearrange(x, "b h n d -> b n (h d)")
            x = _self.out_proj(x)
            
            for k in self.attention_returns:
                _self.get_submodule(OpenCLIPFastNystromCompressionViT.return_module_name(k))(locals()[k])
            return x,
        
        for idx in self.mask_layers:
            blk: ResidualAttentionBlock = self.model.visual.transformer.resblocks[idx]
            for handle in self.attention_returns:
                blk.attn.register_module(OpenCLIPFastNystromCompressionViT.return_module_name(handle), nn.Identity())
            blk.attn.forward = types.MethodType(new_attention_forward, blk.attn)
    
    
        # # SECTION: Replace transformer.forward
        # def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        #     for r in _self.resblocks:
        #         if self.use_layer_input:
        #             self.update_cache({"layer_input": x})
        #             utils.empty_cache()
        #         x = r(x, attn_mask=attn_mask)  
        #     return x
        
        # self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)

