import time
import types
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from infrastructure import utils
from modeling.base_vit import BaseViT


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
    
    LAYER_INPUT = "layer_input"
    SAMPLE_INDICES = "sample_indices"
    
    @classmethod
    def supply_ncut(cls, num_sample: int):
        from nystrom_ncut import AxisAlign, NystromNCut, KernelNCut, SampleConfig
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
        I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        Z = 1 / torch.max(torch.sum(A, dim=-2, keepdim=True), dim=-1, keepdim=True).values * A.mT
        for _ in range(6):
            AZ = A @ Z
            Z = 0.25 * Z @ (13 * I - AZ @ (15 * I - AZ @ (7 * I - AZ)))
        return Z
    
    def __init__(
        self,
        mode: ModeOptions,
        compression_mode: CompressionModeOptions,
        num_sample: int | Tuple[int, int],
        resample: bool,
        use_layer_input: bool,
        targets: List[Tuple[type, type, type, Callable]],
        condition: Callable[[nn.Module, str], bool],
        preserve_keys: List[str] = [],
    ):
        self.mode: NystromCompressionViT.ModeOptions = mode
        self.compression_mode: NystromCompressionViT.CompressionModeOptions = compression_mode
        self.num_sample: Union[int, Tuple[int, int]] = num_sample
        self.resample: bool = resample
        self.use_layer_input: bool = use_layer_input

        def find_targets(module: nn.Module, target_classes: tuple[type, ...]) -> list[tuple[str, nn.Module]]:
            target_cls = target_classes[0]
            targets = [(cname, cmodule,) for cname, cmodule in utils.named_modules(module) if isinstance(cmodule, target_cls)]
            if len(target_classes) == 1:
                return targets
            else:
                target_classes = target_classes[1:]
                result = []
                for cname, cmodule in targets:
                    result.extend([
                        (f"{cname}.{ccname}", ccmodule,)
                        for ccname, ccmodule in find_targets(cmodule, target_classes)
                    ])
                return result

        for target_classes, fn in targets:
            tname, transformer = find_targets(self, target_classes[:1])[0]
            targets = []
            if self.use_layer_input:
                for layer_name, layer_module in find_targets(transformer, target_classes[1:-1]):
                    for name, module in find_targets(layer_module, target_classes[-1:]):
                        if condition(module, f"{tname}.{layer_name}.{name}"):
                            # module.is_modified_forward = True
                            module.forward = types.MethodType(fn, module)
                            
                    layer_module.register_forward_pre_hook(self.register_layer_input, with_kwargs=True)
            else:
                targets.extend()
                for name, module in find_targets(transformer, target_classes[1:]):
                    if condition(module, f"{tname}.{name}"):
                        # module.is_modified_forward = True
                        module.forward = types.MethodType(fn, module)

            TIME_KEY = f"{tname}_time"
            def timer_pre_hook(*args: Any) -> None:
                self._cache[TIME_KEY] = (self._cache.get(TIME_KEY, 0.0), time.perf_counter(),)
            transformer.register_forward_pre_hook(timer_pre_hook)

            def timer_post_hook(*args: Any) -> None:
                t, start = self._cache[TIME_KEY]
                self._cache[TIME_KEY] = t + (time.perf_counter() - start)
            transformer.register_forward_hook(timer_post_hook)
            
            def reset_cache(*args: Any) -> None:
                to_delete = [k for k in self._cache.keys() if k not in ([TIME_KEY] + preserve_keys)]
                for k in to_delete:
                    del self._cache[k]
            transformer.reset_cache_handle = transformer.register_forward_hook(reset_cache)

    def register_layer_input(self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.update_cache({self.LAYER_INPUT: (args, kwargs)})
    
    def get_reduction_func(
        self,
        x: torch.Tensor,                # float: [B x N x D]
        num_sample: int,                # int: s
        sample_indices: torch.Tensor,   # int: [B x s]
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
        
        if num_sample == 0:
            def reduction(t: torch.Tensor) -> torch.Tensor:
                return torch.zeros(t.shape[:-2] + (0,) + t.shape[-1:])
        
        # SECTION: Construct the queries and keys used to extrapolate the attention matrix
        if self.resample:
            sample_indices = None
        
        # SECTION: Construct function that outputs landmark features
        bsz = x.shape[0]
        N = x.shape[1] - 1
        restricted_samples: int = num_sample - 1
        if self.mode in ["fps", "uniform", "multiclass_spectral_clustering"]:
            if sample_indices is None:
                if self.mode == "fps":
                    from pytorch3d.ops import sample_farthest_points
                    sample_indices = sample_farthest_points(x[:, 1:], K=restricted_samples)[1] + 1                          # int: [bsz x max_restricted_samples]

                    # import fpsample
                    # sample_indices = torch.empty((bsz, restricted_samples), dtype=torch.int)
                    # for i, _x in enumerate(torch.unbind(x[:, 1:], dim=0)):
                    #     sample_indices[i] = torch.tensor(fpsample.bucket_fps_kdline_sampling(
                    #         _x.numpy(force=True), restricted_samples, h=7,
                    #     ) + 1)
                    
                elif self.mode == "uniform":
                    sample_indices = torch.topk(torch.rand((bsz, x.shape[1] - 1)), k=restricted_samples, dim=1).indices + 1 # int: [bsz x max_restricted_samples]

                elif self.mode == "multiclass_spectral_clustering":
                    from nystrom_ncut import AxisAlign, NystromNCut, KernelNCut, SampleConfig
                    
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
                    NC = OpenCLIPNystromCompressionViT.supply_ncut(num_sample)
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
                cluster_indices[:, 1:] = torch.arange(N) // (N // num_sample)

            else:
                raise ValueError(self.mode)

            def reduction(t: torch.Tensor) -> torch.Tensor:
                cluster_mask = (cluster_indices[..., None] == torch.arange(num_sample)) # bool: [bsz x n x num_centers]
                cluster_sums = torch.sum(einops.rearrange(
                    t, "bsz ... n d -> ... bsz n 1 d"
                ) * cluster_mask[..., None], dim=-3)                                    # float: [... x bsz x num_centers x d]
                cluster_counts = torch.sum(cluster_mask, dim=1)                         # int: [bsz x num_centers]
                return einops.rearrange(cluster_sums / cluster_counts[..., None], "... bsz s d -> bsz ... s d")

            sample_indices = None

        else:
            raise ValueError(self.mode)
        
        return reduction, (None if self.resample else sample_indices)
    
    def compute_compression(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reduction: Callable[[torch.Tensor], torch.Tensor],
        return_effective_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        match self.compression_mode:
            case "nystrom":
                invsqrt_d = query.shape[-1] ** -0.5
                qp, kp = reduction(query), reduction(key)
                A = torch.softmax(invsqrt_d * (qp @ kp.mT), dim=-1)     # float: [bsz x h x num_sample x num_sample]
                BV = Fn.scaled_dot_product_attention(qp, key, value)    # float: [bsz x h x num_sample x d]
                vp = NystromCompressionViT.invert(A) @ BV               # float: [bsz x h x num_sample x d]
                x = Fn.scaled_dot_product_attention(query, kp, vp)      # float: [bsz x h x N x d]
                
            case "linear":
                kp, vp = reduction(key), reduction(value)
                x = Fn.scaled_dot_product_attention(query, kp, vp)      # float: [bsz x h x N x d]

            case _:
                raise ValueError(self.compression_mode)

        x = einops.rearrange(x, "b h n d -> b n (h d)")
        if return_effective_kv:
            return x, (kp, vp)
        else:
            return x


"""
CLIP
"""
from open_clip.transformer import Transformer, ResidualAttentionBlock
from modeling.base_vit import OpenCLIPViT


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
        def condition(module: nn.Module, name: str) -> bool:
            allowed_names = {
                f"model.visual.transformer.resblocks.{idx}.attn"
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
            reduction, sample_indices = self.get_reduction_func(
                self._cache[self.LAYER_INPUT][0][0] if self.use_layer_input else query,
                self.num_sample, self._cache.get(self.SAMPLE_INDICES),
            )
            self.update_cache({self.SAMPLE_INDICES: sample_indices})
            
            query, key, value = einops.rearrange(
                Fn.linear(query, _self.in_proj_weight, _self.in_proj_bias),
                "b n (qkv h d) -> qkv b h n d", qkv=3, h=_self.num_heads,
            )
            x = self.compute_compression(query, key, value, reduction)
            x = _self.out_proj(x)
            
            return x,

        NystromCompressionViT.__init__(
            self,
            mode=mode,
            compression_mode=compression_mode,
            num_sample=num_sample,
            resample=resample,
            use_layer_input=use_layer_input,
            targets=[
                ((Transformer, ResidualAttentionBlock, nn.MultiheadAttention,), new_attention_forward,),
            ],
            condition=condition,
        )


"""
Dinov2
"""
from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention, Dinov2Layer
from modeling.base_vit import DINOv2ViT


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
        def condition(module: nn.Module, name: str) -> bool:
            allowed_names = {
                f"model.encoder.layer.{idx}.attention"
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
            reduction, sample_indices = self.get_reduction_func(
                self._cache[self.LAYER_INPUT][0][0] if self.use_layer_input else hidden_states,
                self.num_sample, self._cache.get(self.SAMPLE_INDICES),
            )
            self.update_cache({self.SAMPLE_INDICES: sample_indices})
            
            query = _self.transpose_for_scores(_self.query(hidden_states))
            key = _self.transpose_for_scores(_self.key(hidden_states))
            value = _self.transpose_for_scores(_self.value(hidden_states))

            context_layer = self.compute_compression(query, key, value, reduction)
            outputs = (context_layer, None) if output_attentions else (context_layer,)

            return outputs
        
        NystromCompressionViT.__init__(
            self,
            mode=mode,
            compression_mode=compression_mode,
            num_sample=num_sample,
            resample=resample,
            use_layer_input=use_layer_input,
            targets=[
                ((Dinov2Layer, Dinov2SelfAttention,), new_attention_forward,),
            ],
            condition=condition,
        )


"""
Stable diffusion
"""
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention
from modeling.base_vit import StableDiffusion3ViT


class StableDiffusion3NystromCompressionViT(StableDiffusion3ViT, NystromCompressionViT):
    TIMESTEP_KEY = "counter"
    TEXT_SAMPLE_INDICES = "text_sample_indices"
    
    def __init__(
        self,
        mode: NystromCompressionViT.ModeOptions,
        compression_mode: NystromCompressionViT.CompressionModeOptions = "nystrom",
        num_sample: int | Tuple[int, int] = 32,
        resample: bool = False,
        use_layer_input: bool = True,
        layer_condition: Callable[[nn.Module, str], bool] = None,
        timestep_condition: Callable[[int], bool] = None,
    ):
        StableDiffusion3ViT.__init__(self)
        
        # SECTION: Replace layer.attention.attention.forward
        if layer_condition is None:
            def condition(module: nn.Module, name: str) -> bool:
                mask_layers = [*range(18, 24)]
                return name in [
                    f"model.transformer.transformer_blocks.{idx}.attn" for idx in mask_layers
                ] + [
                    f"model.transformer.transformer_blocks.{idx}.attn2" for idx in mask_layers
                ]
        else:
            condition = layer_condition
        
        if timestep_condition is None:
            def timestep_condition(t: int) -> bool:
                return t < 14
        self.timestep_condition = timestep_condition

        def new_attn_forward(
            _self: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty
            
            N = hidden_states.shape[-2]

            # `sample` projections.
            def transpose(t: torch.Tensor) -> torch.Tensor:
                return einops.rearrange(t, "... n (h d) -> ... h n d", h=_self.heads)

            query = transpose(_self.to_q(hidden_states))
            key = transpose(_self.to_k(hidden_states))
            value = transpose(_self.to_v(hidden_states))

            if _self.norm_q is not None:
                query = _self.norm_q(query)
            if _self.norm_k is not None:
                key = _self.norm_k(key)

            # `context` projections.
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = transpose(_self.add_q_proj(encoder_hidden_states))
                encoder_hidden_states_key_proj = transpose(_self.add_k_proj(encoder_hidden_states))
                encoder_hidden_states_value_proj = transpose(_self.add_v_proj(encoder_hidden_states))

                if _self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = _self.norm_added_q(encoder_hidden_states_query_proj)
                if _self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = _self.norm_added_k(encoder_hidden_states_key_proj)

                query = torch.cat((query, encoder_hidden_states_query_proj), dim=-2)
                key = torch.cat((key, encoder_hidden_states_key_proj), dim=-2)
                value = torch.cat((value, encoder_hidden_states_value_proj), dim=-2)

            if self.timestep_condition(self._cache[StableDiffusion3NystromCompressionViT.TIMESTEP_KEY]):
                if isinstance(self.num_sample, int):
                    concatenated_hidden_states = self._cache[self.LAYER_INPUT][1]["hidden_states"] if self.use_layer_input else hidden_states
                    if encoder_hidden_states is not None:
                        to_concatenate = self._cache[self.LAYER_INPUT][1]["encoder_hidden_states"] if self.use_layer_input else encoder_hidden_states
                        concatenated_hidden_states = torch.cat((concatenated_hidden_states, to_concatenate), dim=-2)
                    
                    reduction, sample_indices = self.get_reduction_func(
                        concatenated_hidden_states,
                        self.num_sample, self._cache.get(self.SAMPLE_INDICES),
                    )
                    self.update_cache({self.SAMPLE_INDICES: sample_indices})
                else:
                    image_input = self._cache[self.LAYER_INPUT][1]["hidden_states"] if self.use_layer_input else hidden_states
                    image_reduction, image_sample_indices = self.get_reduction_func(image_input, self.num_sample[0], self._cache.get(self.SAMPLE_INDICES))
                    self.update_cache({self.SAMPLE_INDICES: image_sample_indices})
                    
                    if encoder_hidden_states is not None:
                        text_input = self._cache[self.LAYER_INPUT][1]["encoder_hidden_states"] if self.use_layer_input else encoder_hidden_states
                        text_reduction, text_sample_indices = self.get_reduction_func(text_input, self.num_sample[1], self._cache.get(self.TEXT_SAMPLE_INDICES))
                        self.update_cache({self.TEXT_SAMPLE_INDICES: text_sample_indices})

                        def reduction(t: torch.Tensor) -> torch.Tensor:
                            return torch.cat((image_reduction(t[..., :N, :]), text_reduction(t[..., N:, :])), dim=-2)
                    else:
                        reduction = image_reduction
                
                hidden_states = self.compute_compression(query, key, value, reduction)
            else:
                hidden_states = Fn.scaled_dot_product_attention(query, key, value)
                hidden_states = einops.rearrange(hidden_states, "... h n d -> ... n (h d)")
            
            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, :N],
                    hidden_states[:, N:],
                )
                if not _self.context_pre_only:
                    encoder_hidden_states = _self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = _self.to_out[0](hidden_states)
            # dropout
            hidden_states = _self.to_out[1](hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
        
        def increment_timestep_counter(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            K = StableDiffusion3NystromCompressionViT.TIMESTEP_KEY
            self.update_cache({K: self._cache.get(K, -1) + 1})
        
        def terminate_timestep_counter(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Tuple[Any, ...]) -> None:
            K = StableDiffusion3NystromCompressionViT.TIMESTEP_KEY
            if self._cache[K] == 27:
                del self._cache[K]

        self.model.transformer.register_forward_pre_hook(increment_timestep_counter, with_kwargs=True)
        self.model.transformer.register_forward_hook(terminate_timestep_counter, with_kwargs=True)

        NystromCompressionViT.__init__(
            self,
            mode=mode,
            compression_mode=compression_mode,
            num_sample=num_sample,
            resample=resample,
            use_layer_input=use_layer_input,
            targets=[
                ((SD3Transformer2DModel, JointTransformerBlock, Attention,), new_attn_forward,),
            ],
            condition=condition,
            preserve_keys=[
                StableDiffusion3NystromCompressionViT.TIMESTEP_KEY,
                StableDiffusion3NystromCompressionViT.SAMPLE_INDICES,
                StableDiffusion3NystromCompressionViT.TEXT_SAMPLE_INDICES,
            ],
        )


"""
LlaVa
"""
from modeling.base_vit import LlavaNextViT
from transformers.cache_utils import DynamicCache
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextCausalLMOutputWithPast,
)
from transformers.models.clip.modeling_clip import (
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPAttention,
)
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


class LlavaNextNystromCompressionViT(LlavaNextViT, NystromCompressionViT):
    def __init__(
        self,
        mode: NystromCompressionViT.ModeOptions,
        compression_mode: NystromCompressionViT.CompressionModeOptions = "nystrom",
        num_sample: int | Tuple[int, int] = 32,
        resample: bool = False,
        use_online_fps: bool = False,
        stream_size: int = 32,
        use_layer_input: bool = True,
        mask_layers: Iterable[int] = range(16, 32),
    ):
        LlavaNextViT.__init__(self)
        self.mask_layers = [*mask_layers]

        # SECTION: Replace layers.self_attn.forward        
        def condition(module: nn.Module, name: str) -> bool:
            # allowed_names = [
            #     f"model.vision_tower.vision_model.encoder.layers.{idx}.self_attn"
            #     for idx in mask_layers
            # ] + [
            #     f"model.language_model.model.layers.{idx}.self_attn"
            #     for idx in range(16, 32)
            # ]
            allowed_names = [
                f"model.language_model.model.layers.{idx}.self_attn"
                for idx in self.mask_layers
            ]
            return name in allowed_names

        def new_clip_self_attn_forward(
            _self: CLIPAttention,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # SECTION: Construct the queries and keys used to extrapolate the attention matrix
            num_sample = self.num_sample if isinstance(self.num_sample, int) else self.num_sample[0]
            reduction, sample_indices = self.get_reduction_func(
                self._cache[self.LAYER_INPUT][0][0] if self.use_layer_input else hidden_states,
                num_sample, self._cache.get(self.SAMPLE_INDICES),
            )
            self.update_cache({self.SAMPLE_INDICES: sample_indices})

            def transpose(t: torch.Tensor) -> torch.Tensor:
                return einops.rearrange(t, "... n (h d) -> ... h n d", h=_self.num_heads)

            query_states = transpose(_self.q_proj(hidden_states))
            key_states = transpose(_self.k_proj(hidden_states))
            value_states = transpose(_self.v_proj(hidden_states))

            attn_output = self.compute_compression(query_states, key_states, value_states, reduction)
            attn_output = _self.out_proj(attn_output)

            return attn_output, None

        def new_llama_self_attn_forward(
            _self: LlamaAttention,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[DynamicCache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            def transpose(t: torch.Tensor) -> torch.Tensor:
                return einops.rearrange(t, "... n (h d) -> ... h n d", h=_self.num_heads)

            assert hidden_states.ndim == 3
            bsz = hidden_states.shape[0]
            query_states = transpose(_self.q_proj(hidden_states))
            key_states = transpose(_self.k_proj(hidden_states))
            value_states = transpose(_self.v_proj(hidden_states))
            
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            reduction_features = self._cache[self.LAYER_INPUT][0][0] if self.use_layer_input else hidden_states
            if hidden_states.shape[-2] > 1:
                # TODO: compute_compression in this case should be running is_causal = True which is incompatible with Nystrom, think about whether this matters
                LLAMA_SAMPLE_INDICES = "llama_sample_indices"
                num_sample = self.num_sample if isinstance(self.num_sample, int) else self.num_sample[1]
                reduction, sample_indices = self.get_reduction_func(reduction_features, num_sample, self._cache.get(LLAMA_SAMPLE_INDICES))
                self.update_cache({LLAMA_SAMPLE_INDICES: sample_indices})

                attn_output, (kp, vp) = self.compute_compression(query_states, key_states, value_states, reduction, return_effective_kv=True)
                past_key_value.update(kp, vp, _self.layer_idx)

            elif use_online_fps:
                if not hasattr(past_key_value, "stream_increment"):
                    past_key_value.stream_increment = -1
                    past_key_value.valid_indices = torch.full((bsz, stream_size + 1,), -1)
                    past_key_value.fps_features = torch.full((bsz, stream_size + 1, reduction_features.shape[-1],), torch.nan, dtype=reduction_features.dtype)

                if _self.layer_idx == min(mask_layers):
                    past_key_value.stream_increment += 1
                    
                if past_key_value.stream_increment < stream_size:
                    if _self.layer_idx == min(mask_layers):
                        past_key_value.valid_indices[:, past_key_value.stream_increment] = past_key_value.stream_increment
                        past_key_value.fps_features[:, past_key_value.stream_increment, :] = reduction_features[:, 0, :]

                    past_key_value.update(key_states, value_states, _self.layer_idx)
                else:
                    if _self.layer_idx == min(mask_layers):
                        scale = 5000
                        if past_key_value.stream_increment == stream_size:
                            open_index = torch.full((bsz,), -1)
                            past_key_value.fps_features[:, -1, :] = reduction_features[:, 0, :]
                            past_key_value.exp_dist = torch.exp((-0.5 / scale) * torch.cdist(past_key_value.fps_features, past_key_value.fps_features) ** 2)
                        else:
                            open_index = past_key_value.open_index
                            past_key_value.fps_features[range(bsz), open_index] = reduction_features[:, 0, :]
                            exp_dist = torch.exp((-0.5 / scale) * torch.cdist(reduction_features, past_key_value.fps_features) ** 2)[:, 0, :]
                            past_key_value.exp_dist[range(bsz), :, open_index] = exp_dist
                            past_key_value.exp_dist[range(bsz), open_index, :] = exp_dist

                        past_key_value.open_index = torch.argmax(torch.sum(past_key_value.exp_dist, dim=-1), dim=-1)
                        kv_index = past_key_value.valid_indices[range(bsz), past_key_value.open_index]
                        past_key_value.valid_indices[range(bsz), past_key_value.open_index] = -1
                        past_key_value.valid_indices[range(bsz), open_index] = kv_index

                        past_key_value.to_replace = torch.arange(bsz)[open_index != past_key_value.open_index]
                        past_key_value.kv_index = (kv_index - stream_size)[past_key_value.to_replace]

                    past_key_value.key_cache[_self.layer_idx][past_key_value.to_replace, :, past_key_value.kv_index, :] = key_states[past_key_value.to_replace, :, 0, :]
                    past_key_value.value_cache[_self.layer_idx][past_key_value.to_replace, :, past_key_value.kv_index, :] = value_states[past_key_value.to_replace, :, 0, :]

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                )                
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, _self.layer_idx)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                )

            attn_output = einops.rearrange(attn_output, "... h n d -> ... n (h d)")
            attn_output = _self.o_proj(attn_output)
            
            return attn_output, None, past_key_value

        NystromCompressionViT.__init__(
            self,
            mode=mode,
            compression_mode=compression_mode,
            num_sample=num_sample,
            resample=resample,
            use_layer_input=use_layer_input,
            targets=[
                ((CLIPEncoder, CLIPEncoderLayer, CLIPAttention,), new_clip_self_attn_forward,),
                ((LlamaModel, LlamaDecoderLayer, LlamaAttention,), new_llama_self_attn_forward,),
            ],
            condition=condition,
        )

        def new_forward(
            _self: LlavaNextForConditionalGeneration,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            image_sizes: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[DynamicCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
        ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
            vision_feature_layer = (
                vision_feature_layer if vision_feature_layer is not None else _self.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else _self.config.vision_feature_select_strategy
            )

            inputs_embeds = _self.get_input_embeddings()(input_ids)
            if pixel_values is None:
                # Get the target length
                position_ids = torch.sum(attention_mask, dim=1, keepdim=True) - 1

            # TODO: @raushan retain only the new behavior after v4.47
            else:
                image_features = _self.get_image_features(
                    pixel_values,
                    image_sizes,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                image_features = _self.pack_image_features(
                    image_features,
                    image_sizes,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_newline=_self.image_newline,
                )[0]
                
                special_image_mask = (
                    (input_ids == _self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            
            outputs = _self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=None,
                num_logits_to_keep=num_logits_to_keep,
            )

            logits = outputs[0]

            return LlavaNextCausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features if pixel_values is not None else None,
            )

        self.model.forward = types.MethodType(new_forward, self.model)
