import itertools
import types
from typing import Callable, Dict, Iterable, List, Literal, Optional, OrderedDict, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from nystrom_ncut import AxisAlign, NystromNCut, KernelNCut, SampleConfig
from open_clip.transformer import ResidualAttentionBlock, Transformer
from pytorch3d.ops import sample_farthest_points

from infrastructure import utils
from infrastructure.settings import SEED, DEVICE
from modeling.image_features import ImageFeatures
from modeling.openclip_vit import OpenCLIPViT


class OpenCLIPUltraFastNystromCompressionViT(OpenCLIPViT):
    ModeOptions = Literal[
        # Sampling methods
        "fps",
        "uniform",
        "multiclass_spectral_clustering",
        # Averaging methods
        "kmeans",
        "segment_means",
        "spectral_clustering",
    ]
    
    CompressionModeOptions = Literal[
        "nystrom",
        "linear",
    ]
    
    @classmethod
    def return_module_name(cls, handle: str) -> str:
        return f"return_{handle}"
    
    @classmethod
    def supply_ncut(cls, num_sample: int) -> NystromNCut:
        # return KernelNCut(
        #     n_components=num_sample,
        #     kernel_dim=4096,
        #     affinity_type="rbf",
        #     sample_config=SampleConfig(method="full"),
        # )
        return NystromNCut(
            n_components=num_sample,
            affinity_type="rbf",
            sample_config=SampleConfig(method="full"),
            eig_solver="svd_lowrank"
        )

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
        compression_mode: CompressionModeOptions,
        num_sample: int,
        resample: bool,
        use_layer_input: bool,
        mask_layers: Iterable[int],
    ):
        OpenCLIPViT.__init__(self)
        self.mode: OpenCLIPUltraFastNystromCompressionViT.ModeOptions = mode
        self.compression_mode: OpenCLIPUltraFastNystromCompressionViT.CompressionModeOptions = compression_mode
        self.num_sample: int = num_sample
        self.resample: bool = resample
        self.use_layer_input: bool = use_layer_input
        self.mask_layers: List[int] = [*mask_layers]
        
        self.attention_returns: List[str] = []  # ["sample_indices"]

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
            
            bsz = query.shape[0]
            
            # SECTION: Construct the queries and keys used to extrapolate the attention matrix
            sample_input = query           
            if self.use_layer_input:
                sample_input: torch.Tensor = self._cache.pop("layer_input")
            N = sample_input.shape[1] - 1

            # SECTION: Construct function that outputs landmark features
            restricted_samples: int = self.num_sample - 1
            if self.mode in ["fps", "uniform", "multiclass_spectral_clustering"]:
                
                sample_indices = self._cache.get("sample_indices", None)
                if self.resample or sample_indices is None:
                    if self.mode == "fps": 
                        sample_indices = sample_farthest_points(sample_input[:, 1:], K=restricted_samples)[1] + 1                   # int: [bsz x max_restricted_samples]
     
                    elif self.mode == "uniform":
                        sample_indices = torch.topk(torch.rand((bsz, sample_input.shape[1] - 1)), k=restricted_samples, dim=1).indices + 1  # int: [bsz x max_restricted_samples]

                    elif self.mode == "multiclass_spectral_clustering":
                        NC = OpenCLIPUltraFastNystromCompressionViT.supply_ncut(restricted_samples)
                        AA = AxisAlign(sort_method="marginal_norm")

                        ncut_features = NC.fit_transform(sample_input[:, 1:, :])                            # float: [bsz x N x num_sample]
                        axis_aligned_features = AA.fit_transform(ncut_features, normalize=True, hard=False) # float: [bsz x N x num_sample]
                        sample_indices = torch.argmax(axis_aligned_features, dim=1) + 1
            
                    else:
                        raise ValueError(self.mode)

                    sample_indices = torch.cat((torch.full((bsz, 1), 0), sample_indices), dim=1)                    
                    self.update_cache({"sample_indices": sample_indices})
                
                def reduction(t: torch.Tensor) -> torch.Tensor:
                    return t[torch.arange(bsz)[:, None], :, sample_indices, :].transpose(dim0=1, dim1=2)

            elif self.mode in ["kmeans", "segment_means", "spectral_clustering"]:
                
                if self.mode in ["kmeans", "spectral_clustering"]:
                    cluster_indices = torch.full((bsz, N + 1), 0)
                    if self.mode == "spectral_clustering":
                        NC = OpenCLIPUltraFastNystromCompressionViT.supply_ncut(self.num_sample)
                        restricted_sample_input = NC.fit_transform(sample_input[:, 1:, :])
                    else:
                        restricted_sample_input = sample_input[:, 1:, :]
                
                    # OPTION: Using cuml
                    from cuml import KMeans
                    KM = KMeans(n_clusters=restricted_samples)
                    for image_idx in range(bsz):
                        cluster_indices[image_idx, 1:] = torch.tensor(KM.fit_predict(restricted_sample_input[image_idx])) + 1

                elif self.mode == "segment_means":
                    cluster_indices = torch.full((bsz, N + 1), -1)
                    cluster_indices[:, 1:] = torch.arange(N) // (N // self.num_sample)

                else:
                    raise ValueError(self.mode)

                def reduction(t: torch.Tensor) -> torch.Tensor:
                    cluster_mask = (cluster_indices[..., None] == torch.arange(self.num_sample))    # bool: [bsz x n x num_centers]
                    cluster_sums = torch.sum(einops.rearrange(
                        t, "bsz ... n d -> ... bsz n 1 d"
                    ) * cluster_mask[..., None], dim=-3)                                            # float: [... x bsz x num_centers x d]
                    cluster_counts = torch.sum(cluster_mask, dim=1)                                 # int: [bsz x num_centers]
                    return einops.rearrange(cluster_sums / cluster_counts[..., None], "... bsz s d -> bsz ... s d")
                
            else:
                raise ValueError(self.mode)


            invsqrt_d = _self.head_dim ** -0.5
            query, key, value = einops.rearrange(
                Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias),
                "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads,
            )

            if self.compression_mode == "nystrom":
                qp, kp = reduction(query), reduction(key)
                A = torch.softmax(invsqrt_d * (qp @ kp.mT), dim=-1)                 # float: [bsz x h x num_sample x num_sample]
                BT = torch.softmax(query @ (invsqrt_d * kp.mT), dim=-1)             # float: [bsz x h x N x num_sample]
                BV = Fn.scaled_dot_product_attention(qp, key, value)                # float: [bsz x h x num_sample x d]
                x = BT @ (OpenCLIPUltraFastNystromCompressionViT.invert(A) @ BV)    # float: [bsz x h x N x d]
            
            elif self.compression_mode == "linear":
                kp, vp = reduction(key), reduction(value)
                x = Fn.scaled_dot_product_attention(query, kp, vp)                  # float: [bsz x h x N x d]

            else:
                raise ValueError(self.compression_mode)

            # AinvBV = OpenCLIPUltraFastNystromCompressionViT.invert(A) @ BV                          # float: [bsz x h x num_sample x d]
            # out_proj_weight = einops.rearrange(_self.out_proj.weight, "D (h d) -> h d D", h=_self.num_heads)    # float: [h x d x D]
            # AinvBVout = AinvBV @ out_proj_weight[None]                                              # float: [bsz x h x num_sample x D]
            # x = torch.sum(BT @ AinvBVout, dim=1) + _self.out_proj.bias                              # float: [bsz x N x D]

            x = einops.rearrange(x, "b h n d -> b n (h d)")
            x = _self.out_proj(x)
            
            for k in self.attention_returns:
                _self.get_submodule(OpenCLIPUltraFastNystromCompressionViT.return_module_name(k))(locals()[k])
            return x,
        
        for idx in self.mask_layers:
            blk: ResidualAttentionBlock = self.model.visual.transformer.resblocks[idx]
            for handle in self.attention_returns:
                blk.attn.register_module(OpenCLIPUltraFastNystromCompressionViT.return_module_name(handle), nn.Identity())
            blk.attn.forward = types.MethodType(new_attention_forward, blk.attn)
    
    
        # SECTION: Replace transformer.forward
        def new_transformer_forward(_self: Transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            for r in _self.resblocks:
                if self.use_layer_input:
                    self.update_cache({"layer_input": x})
                x = r(x, attn_mask=attn_mask)  
            return x
        
        self.model.visual.transformer.forward = types.MethodType(new_transformer_forward, self.model.visual.transformer)

