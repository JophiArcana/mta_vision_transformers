import itertools
import types
from typing import Callable, Dict, Iterable, List, Literal, Optional, OrderedDict, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from open_clip.transformer import ResidualAttentionBlock, Transformer
from pytorch3d.ops import sample_farthest_points
from torch_kmeans import KMeans

from infrastructure import utils
from infrastructure.settings import SEED, DEVICE
from modeling.image_features import ImageFeatures
from modeling.base_vit import BaseViT, OpenCLIPViT



class OpenCLIPNystromCompressionViT(OpenCLIPViT):
    ModeOptions = Literal[
        "fps",
        "uniform",
        "manual",
        "kmeans",
        "random_mean",
    ]

    def __init__(
        self,
        mode: ModeOptions,
        inverse_approximation: Union[Literal["iterative"], int],
        num_sample: int,
        resample: bool,
        use_layer_input: bool,
        include_explicit_in_total_count: bool,
        mask_layers: Iterable[int],
    ):
        OpenCLIPViT.__init__(self)
        self.mode: OpenCLIPNystromCompressionViT.ModeOptions = mode
        self.inverse_approximation: Union[Literal["iterative"], int] = inverse_approximation
        self.num_sample: int = num_sample
        self.resample: bool = resample
        self.use_layer_input: bool = use_layer_input
        self.include_explicit_in_total_count: bool = include_explicit_in_total_count
        self.mask_layers: List[int] = [*mask_layers]
        
        self.attention_returns: List[str] = ["attn_matrix", "subsample_matrix"]

        
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
                compression_condition = idx in self.mask_layers
                
                fps_input = query
                qkv = Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias)
                query, key, value = einops.rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads)
                
                bsz = query.shape[0]
                bsz_index = torch.arange(bsz)[:, None]
                head_dim = query.shape[-1]
                invsqrt_d = head_dim ** -0.5
                
                attn_weights = invsqrt_d * (query @ key.mT)
                attn_matrix = Fn.softmax(attn_weights, dim=-1)
                
                subsample_matrix: torch.Tensor = torch.full((bsz, self.num_sample), -1)
                if compression_condition:
                    def index(t: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
                        return einops.rearrange(torch.cat((
                            t, torch.full((bsz, _self.num_heads, 1, head_dim), torch.inf)
                        ), dim=2)[bsz_index, :, sample_indices, :], "bsz s h d -> bsz h s d")
                    
                    def mean(t: torch.Tensor, cluster_indices: torch.Tensor) -> torch.Tensor:
                        cluster_mask = (cluster_indices[..., None] == torch.arange(self.num_sample))    # bool: [bsz x n x num_sample]
                        cluster_sums = torch.sum(t[:, None, :, None, :] * cluster_mask, dim=2)          # float: [bsz x h x d x num_sample]
                        cluster_counts = torch.sum(cluster_mask, dim=1)[:, None, :, None]               # int: [bsz x 1 x num_sample x 1]
                        return einops.rearrange(cluster_sums, "bsz h d s -> bsz h s d") / cluster_counts
                    
                    def get_mask_indices(
                        implicit_mask: torch.Tensor,            # [bsz x N]
                    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                        counts = torch.sum(implicit_mask, dim=1)                                # int: [bsz]
                        max_count = torch.max(counts).item()                                    # int
                        
                        sample_indices = torch.topk(implicit_mask.to(torch.int), k=max_count, dim=1).indices    # int: [bsz x max_count]
                        sample_indices[torch.arange(max_count) >= counts[:, None]] = -1                        
                        return sample_indices                                                                   # int: [bsz x max_count]
                    
                    # SECTION: Construct explicit mask and downscaling for implicit matrix
                    explicit_mask = mask_dict["explicit"]                   # bool: [bsz x N]
                    implicit_mask = ~explicit_mask                          # bool: [bsz x N]
                    implicit_indices = get_mask_indices(implicit_mask)      # int: [bsz x num_implicit]
                
                    implicit_attn_weights: torch.Tensor = einops.rearrange(attn_weights[bsz_index, :, implicit_indices], "bsz i h n -> bsz h i n")  # float: [bsz x h x num_implicit x N]
                    implicit_masked_attn_weights = torch.where(explicit_mask[:, None, None, :], -torch.inf, implicit_attn_weights)                  # float: [bsz x h x num_implicit x N]
                    downscale = torch.exp(torch.logsumexp(implicit_masked_attn_weights, dim=-1) - torch.logsumexp(implicit_attn_weights, dim=-1))   # float: [bsz x h x num_implicit]

                    qi, ki = index(query, implicit_indices), index(key, implicit_indices)   # float: [bsz x h x num_implicit x d]
                    
                    # SECTION: Construct the queries and keys used to extrapolate the attention matrix
                    if self.mode in ["fps", "uniform", "manual"]:
                        if self.mode in ["fps", "uniform"]:
                            
                            sample_indices = self._cache.get("sample_indices", None)
                            if self.resample or sample_indices is None:
                            
                                guarantee_mask = mask_dict["guarantee"]                                     # bool: [bsz x N]
                                exclude_mask = mask_dict["exclude"]                                         # bool: [bsz x N]
                                assert torch.all(torch.sum(guarantee_mask, dim=1) <= self.num_sample)
                                
                                total_samples = self.num_sample - torch.sum(explicit_mask, dim=1)           # int: [bsz]
                                max_total_samples: int = torch.max(total_samples).item()                    # int
                                
                                if self.include_explicit_in_total_count:
                                    restricted_samples = total_samples - torch.sum(guarantee_mask, dim=1)   # int: [bsz]
                                else:
                                    restricted_samples = self.num_sample - torch.sum(guarantee_mask, dim=1) # int: [bsz]
                                max_restricted_samples: int = torch.max(restricted_samples).item()          # int       
                                
                                restricted_mask = implicit_mask * ~guarantee_mask * ~exclude_mask           # bool: [bsz x N]
                                counts = torch.sum(restricted_mask, dim=1)                                  # [bsz]
                                max_count: int = torch.max(counts).item()                                   # int
                                
                                if self.mode == "fps":
                                    topk_indices = torch.topk(restricted_mask.to(torch.int), k=max_count, dim=1).indices    # int: [bsz x max_count]
                                    
                                    if self.use_layer_input:
                                        fps_input: torch.Tensor = self._cache["layer_input"]
                                    
                                    fps_indices = sample_farthest_points(
                                        fps_input[bsz_index, topk_indices],
                                        lengths=counts, K=restricted_samples,
                                    )[1].to(DEVICE)                                                                         # int: [bsz x max_restricted_samples]
                                    sample_indices = torch.cat((
                                        topk_indices, torch.full((bsz, 1), -1)
                                    ), dim=1)[bsz_index, fps_indices]                                                       # int: [bsz x max_restricted_samples]
                                
                                else:
                                    sort_weights = torch.rand(restricted_mask.shape) + restricted_mask.to(torch.float)      # float: [bsz x N]
                                    sample_indices = torch.topk(sort_weights, k=max_restricted_samples, dim=1).indices      # int: [bsz x max_restricted_samples]
                                    sample_indices[torch.arange(max_restricted_samples) >= restricted_samples[:, None]] = -1
                                
                                if max_total_samples > max_restricted_samples:
                                    sample_indices = torch.cat((
                                        sample_indices, torch.full((bsz, max_total_samples - max_restricted_samples), -1)
                                    ), dim=1)
                                sample_indices[(sample_indices == -1) * (torch.arange(max_total_samples) < total_samples[:, None])] = torch.where(guarantee_mask)[1]

                                self.update_cache({"sample_indices": sample_indices})

                        elif self.mode == "manual":
                            manual_mask = mask_dict["manual"]                               # bool: [bsz x N]
                            sample_indices = get_mask_indices(manual_mask)
                        
                        subsample_matrix = torch.cat((
                            sample_indices, torch.full((bsz, self.num_sample - sample_indices.shape[1]), -1)
                        ), dim=1)                                                           # int: [bsz x num_sample]
                        qp, kp = index(query, sample_indices), index(key, sample_indices)   # float: [bsz x h x num_sample x d]   

                    elif self.mode in ["kmeans", "random_mean"]:
                        if self.mode == "kmeans":
                            counts = torch.sum(implicit_mask, dim=1)                        # [bsz]
                            max_count: int = torch.max(counts).item()                       # int
                                                            
                            topk_indices = torch.topk(implicit_mask.to(torch.int), k=max_count, dim=1).indices  # int: [bsz x max_count]
                            KM = KMeans(n_clusters=self.num_sample, verbose=False)
                            km_indices = KM.fit_predict(x[bsz_index, topk_indices], k=counts)                   # int: [bsz x max_count]
                            
                            cluster_indices = torch.full((bsz, ImageFeatures.N + 1), -1)                        # int: [bsz x N]
                            cluster_indices[bsz_index, topk_indices] = km_indices
                        else:
                            cluster_indices = torch.randint(0, self.num_sample, (bsz, ImageFeatures.N + 1))     # int: [bsz x N]
                        cluster_indices[~implicit_mask] = -1
                        
                        qp, kp = mean(query, cluster_indices), mean(key, cluster_indices)         # float: [bsz x h x num_sample x d]

                    else:
                        raise ValueError(self.mode)

                    def compute_padded_softmax(_q: torch.Tensor, _k: torch.Tensor) -> torch.Tensor:
                        _w = _q @ _k.mT                                     # float: [bsz x h x q x k]
                        _w.nan_to_num_(nan=-torch.inf, posinf=-torch.inf)
                        _w = torch.softmax(invsqrt_d * _w, dim=-1)
                        return _w.nan_to_num_(nan=0.0)

                    A = compute_padded_softmax(qp, kp)                      # float: [bsz x h x num_sample x num_sample]
                    B = compute_padded_softmax(qp, ki)                      # float: [bsz x h x num_sample x num_implicit]
                    BT = compute_padded_softmax(qi, kp)                     # float: [bsz x h x num_implicit x num_sample]
                    C = (BT @ self.invert(A) @ B) * downscale[..., None]    # float: [bsz x h x num_implicit x num_implicit]

                    buffered_attn_matrix = Fn.pad(attn_matrix, (0, 1, 0, 1), mode="constant", value=torch.nan)  # float: [bsz x h x (N + 1) x (N + 1)]
                    buffered_attn_matrix[torch.arange(bsz)[:, None, None], :, implicit_indices[:, :, None], implicit_indices[:, None, :]] = einops.rearrange(C, "bsz h ni1 ni2 -> bsz ni1 ni2 h")
                    attn_matrix = buffered_attn_matrix[:, :, :-1, :-1]
                    
                x = einops.rearrange(attn_matrix @ value, "b h n d -> b n (h d)")                
                x = Fn.linear(x, _self.out_proj.weight, _self.out_proj.bias)
                    
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
            for r in _self.resblocks:
                if self.use_layer_input:
                    self.update_cache({"layer_input": x})
                    utils.empty_cache()
                x = r(x, attn_mask=attn_mask)  
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)

        

    def invert(self, A: torch.Tensor) -> torch.Tensor:
        if self.inverse_approximation == "iterative":
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
            I = torch.eye(A.shape[-1], device=A.device)
            Z = 1 / torch.max(torch.sum(A, dim=-2, keepdim=True), dim=-1, keepdim=True).values * A.mT
            for _ in range(6):
                AZ = A @ Z
                Z = 0.25 * Z @ (13 * I - AZ @ (15 * I - AZ @ (7 * I - AZ)))
            return Z
            
        elif isinstance(self.inverse_approximation, int):
            U, S, V = torch.svd_lowrank(A, q=self.inverse_approximation)        # [h x Ns x k], [h x k], [h x Ns x k]
            return V @ torch.diag_embed(1 / S) @ U.mT                           # [h x Ns x Ns]
        else:
            return torch.inverse(A)


