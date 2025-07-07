import types
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from nystrom_ncut import AxisAlign, NystromNCut, KernelNCut, SampleConfig
from open_clip.transformer import ResidualAttentionBlock, Transformer
from pytorch3d.ops import sample_farthest_points
from transformers.models.dinov2.modeling_dinov2 import (
    BaseModelOutput,
    Dinov2SelfAttention,
    Dinov2Layer,
    Dinov2Encoder,
)

from modeling.base_vit import BaseViT, OpenCLIPViT, DINOv2ViT


class NystromCompressionViT(BaseViT):
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
        
        target_layer_cls: type,
        target_attention_cls: type,
        condition: Callable[[str, nn.Module], bool],
        new_forward: Callable,
    ):
        self.mode: NystromCompressionViT.ModeOptions = mode
        self.compression_mode: NystromCompressionViT.CompressionModeOptions = compression_mode
        self.num_sample: int = num_sample
        self.resample: bool = resample
        self.use_layer_input: bool = use_layer_input

        found_modules = []
        for name, module in self.named_modules():
            if isinstance(module, target_layer_cls) and module not in found_modules and condition(name, module):
                attention_modules = [child for child in module.modules() if isinstance(child, target_attention_cls)]
                assert len(attention_modules) == 1, f"Expected each layer to have only one attention module but got {len(attention_modules)}."
                for attention_module in attention_modules:
                    attention_module.forward = types.MethodType(new_forward, attention_module)
                if self.use_layer_input:
                    module.register_forward_pre_hook(self.register_layer_input)
                found_modules.append(module)
        
    def register_layer_input(self, module: nn.Module, input: Any):
        self.update_cache({"layer_input": input[0]})
    
    def get_reduction_func(
        self,
        x: torch.Tensor,    # float: [B x N x D]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        
        # SECTION: Construct the queries and keys used to extrapolate the attention matrix
        if self.use_layer_input:
            x = self._cache.pop("layer_input")
        sample_indices = prev_sample_indices = None if self.resample else self._cache.get("sample_indices", None)
        
        # SECTION: Construct function that outputs landmark features
        bsz = x.shape[0]
        N = x.shape[1] - 1
        restricted_samples: int = self.num_sample - 1
        if self.mode in ["fps", "uniform", "multiclass_spectral_clustering"]:
            if sample_indices is None:
                if self.mode == "fps": 
                    sample_indices = sample_farthest_points(x[:, 1:], K=restricted_samples)[1] + 1                          # int: [bsz x max_restricted_samples]

                elif self.mode == "uniform":
                    sample_indices = torch.topk(torch.rand((bsz, x.shape[1] - 1)), k=restricted_samples, dim=1).indices + 1 # int: [bsz x max_restricted_samples]

                elif self.mode == "multiclass_spectral_clustering":
                    NC = OpenCLIPNystromCompressionViT.supply_ncut(restricted_samples)
                    AA = AxisAlign(sort_method="marginal_norm")

                    ncut_features = NC.fit_transform(x[:, 1:, :])                                       # float: [bsz x N x num_sample]
                    axis_aligned_features = AA.fit_transform(ncut_features, normalize=True, hard=False) # float: [bsz x N x num_sample]
                    sample_indices = torch.argmax(axis_aligned_features, dim=1) + 1
        
                else:
                    raise ValueError(self.mode)

                sample_indices = torch.cat((torch.full((bsz, 1), 0), sample_indices), dim=1)

            def reduction(t: torch.Tensor) -> torch.Tensor:
                return t[torch.arange(bsz)[:, None], :, sample_indices, :].transpose(dim0=1, dim1=2)

        elif self.mode in ["kmeans", "segment_means", "spectral_clustering"]:
            
            if self.mode in ["kmeans", "spectral_clustering"]:
                cluster_indices = torch.full((bsz, N + 1), 0)
                if self.mode == "spectral_clustering":
                    NC = OpenCLIPNystromCompressionViT.supply_ncut(self.num_sample)
                    restricted_sample_input = NC.fit_transform(x[:, 1:, :])
                else:
                    restricted_sample_input = x[:, 1:, :]
            
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

            sample_indices = None

        else:
            raise ValueError(self.mode)
        
        if prev_sample_indices is None:
            self.update_cache({"sample_indices": sample_indices})
        
        return reduction
    
    def compute_compression(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reduction: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        match self.compression_mode:
            case "nystrom":
                invsqrt_d = query.shape[-1] ** -0.5
                qp, kp = reduction(query), reduction(key)
                A = torch.softmax(invsqrt_d * (qp @ kp.mT), dim=-1)     # float: [bsz x h x num_sample x num_sample]
                BT = torch.softmax(query @ (invsqrt_d * kp.mT), dim=-1) # float: [bsz x h x N x num_sample]
                BV = Fn.scaled_dot_product_attention(qp, key, value)    # float: [bsz x h x num_sample x d]
                x = BT @ (NystromCompressionViT.invert(A) @ BV)     # float: [bsz x h x N x d]
                
            case "linear":
                kp, vp = reduction(key), reduction(value)
                x = Fn.scaled_dot_product_attention(query, kp, vp)      # float: [bsz x h x N x d]

            case _:
                raise ValueError(self.compression_mode)

        return einops.rearrange(x, "b h n d -> b n (h d)")


class OpenCLIPNystromCompressionViT(OpenCLIPViT, NystromCompressionViT):
    def __init__(
        self,
        mode: NystromCompressionViT.ModeOptions,
        compression_mode: NystromCompressionViT.CompressionModeOptions = "nystrom",
        num_sample: int = 32,
        resample: bool = False,
        use_layer_input: bool = True,
        mask_layers: Iterable[int] = range(13, 24),
    ):
        OpenCLIPViT.__init__(self)

        # SECTION: Replace resblock.attn.forward        
        def condition(name: str, module: nn.Module) -> bool:
            allowed_names = {
                f"model.visual.transformer.resblocks.{idx}"
                for idx in mask_layers
            }
            return name in allowed_names

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
            
            # SECTION: Construct the queries and keys used to extrapolate the attention matrix
            reduction = self.get_reduction_func(query)
            query, key, value = einops.rearrange(
                Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias),
                "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads,
            )
            x = self.compute_compression(query, key, value, reduction)
            x = _self.out_proj(x)
            
            for k in self.attention_returns:
                _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
            return x,

        NystromCompressionViT.__init__(
            self,
            mode=mode,
            compression_mode=compression_mode,
            num_sample=num_sample,
            resample=resample,
            use_layer_input=use_layer_input,
            target_layer_cls=ResidualAttentionBlock,
            target_attention_cls=nn.MultiheadAttention,
            condition=condition,
            new_forward=new_attention_forward,
        )
        self.attention_returns: List[str] = []  # ["sample_indices"]


class DINOv2NystromCompressionViT(DINOv2ViT, NystromCompressionViT):
    def __init__(
        self,
        mode: NystromCompressionViT.ModeOptions,
        compression_mode: NystromCompressionViT.CompressionModeOptions = "nystrom",
        num_sample: int = 32,
        resample: bool = False,
        use_layer_input: bool = True,
        mask_layers: Iterable[int] = range(13, 24),
    ):
        DINOv2ViT.__init__(self)
    
        # SECTION: Replace layer.attention.attention.forward
        def condition(name: str, module: nn.Module) -> bool:
            allowed_names = {
                f"model.encoder.layer.{idx}"
                for idx in mask_layers
            }
            return name in allowed_names
        
        def new_attention_forward(
            _self: Dinov2SelfAttention,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
            
            # SECTION: Construct the queries and keys used to extrapolate the attention matrix
            reduction = self.get_reduction_func(hidden_states)
            query = _self.transpose_for_scores(_self.query(hidden_states))
            key = _self.transpose_for_scores(_self.key(hidden_states))
            value = _self.transpose_for_scores(_self.value(hidden_states))

            context_layer = self.compute_compression(query, key, value, reduction)
            outputs = (context_layer, None) if output_attentions else (context_layer,)

            for k in self.attention_returns:
                _self.get_submodule(BaseViT.return_module_name(k))(locals()[k])
            return outputs
        
        NystromCompressionViT.__init__(
            self,
            mode=mode,
            compression_mode=compression_mode,
            num_sample=num_sample,
            resample=resample,
            use_layer_input=use_layer_input,
            target_layer_cls=Dinov2Layer,
            target_attention_cls=Dinov2SelfAttention,
            condition=condition,
            new_forward=new_attention_forward,
        )
        self.attention_returns: List[str] = []  # ["sample_indices"]
