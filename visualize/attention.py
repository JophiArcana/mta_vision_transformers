import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import matplotlib.colors
import torch
import torch.nn.functional as Fn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core.decomposition import DecompositionOptions, supply_decompositions
from infrastructure import utils
from infrastructure.settings import DEVICE, OUTPUT_DEVICE
from modeling.image_features import ImageFeatures
from visualize.base import (
    VISUALIZED_INDICES, NUM_VISUALIZED_IMAGES, PLOT_SCALE,
    _visualize_cmap_with_values,
)



ScaleOptions = Literal["linear", "log", "arcsinh"]
# def visualize_attention_matrix_per_image(
#     features: ImageFeatures,
#     layer_idx: int,
#     mta_aliases: Dict[int, str],
#     transform_func: Callable[[torch.Tensor], torch.Tensor] = None,
#     per_head: bool = False,
#     rescale_func: Callable[[torch.Tensor], torch.Tensor] = None,
#     global_cmap: bool = True,
#     cmap_scale: ScaleOptions = "linear",
#     subsample: float = 1.0,
#     spacing: float = 0.1,
#     **kwargs: Any,
# ) -> None:
#     # Construct the rearranged attention weights
#     attention_weights = features.get(layer_idx=layer_idx, key="attention_matrix", include=(ImageFeatures.CLS, ImageFeatures.IMAGE,), with_batch=True)
#     visualize_attention_matrix_per_image(
#         layer_idx=layer_idx,
#         attention_weights=attention_weights,
#         name_to_mask={
#             alias: features.masks[ImageFeatures.process_key(k)][layer_idx]
#             for k, alias in mta_aliases.items()
#         },
#         transform_func=transform_func,
#         per_head=per_head,
#         rescale_func=rescale_func,
#         global_cmap=global_cmap,
#         cmap_scale=cmap_scale,
#         subsample=subsample,
#         spacing=spacing,
#         **kwargs,
#     )
def visualize_attention_matrix_per_image(
    layer_idx: int,
    attention_weights: torch.Tensor,
    name_to_mask: Dict[str, torch.Tensor],
    transform_func: Callable[[torch.Tensor], torch.Tensor] = None,
    per_head: bool = False,
    rescale_func: Callable[[torch.Tensor], torch.Tensor] = None,
    global_cmap: bool = True,
    cmap_scale: ScaleOptions = "linear",
    subsample: float = 1.0,
    spacing=0.1,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    bsz, N = attention_weights.shape[:2]

    if attention_weights.ndim < 4:
        per_head = False
    if not per_head and attention_weights.ndim == 4:
        attention_weights = torch.mean(attention_weights, dim=-1)
    
    # Construct token masks for each cateogory
    flattened_mta_dict: Dict[str, torch.Tensor] = {
        "CLS": (torch.arange(N) == 0).expand((bsz, -1)),
    }
    normal_mask = torch.all(torch.isfinite(attention_weights), dim=-1).to(OUTPUT_DEVICE)
    for k, mask in name_to_mask.items():
        if k != "":
            flattened_mta_dict[k] = normal_mask * mask
        normal_mask = normal_mask * ~mask
    normal_mask = normal_mask * (torch.rand((bsz, N)) < subsample)
    flattened_mta_dict["Normal"] = normal_mask

    order_weights = torch.arange(N).repeat((bsz, 1))    
    for k, flattened_mta_mask in enumerate(flattened_mta_dict.values(), start=-len(flattened_mta_dict)):
        order_weights[flattened_mta_mask] += (2 * ImageFeatures.N) * k
    order = torch.argsort(order_weights, dim=1, stable=True)
    counts = torch.sum(order_weights < 0, dim=1)

    def get_attention_weights_for_image_idx(image_idx: int) -> torch.Tensor:
        _order = order[image_idx, :counts[image_idx]]
        return attention_weights[image_idx, _order[:, None], _order[None, :]]

    # Compute the widths of the rescaled attention image
    widths = torch.stack([
        torch.sum(flattened_mta_mask, dim=1)
        for flattened_mta_mask in flattened_mta_dict.values()
    ], dim=1)
    cumulative_widths = Fn.pad(torch.cumsum(widths, dim=1), (1, 0), mode="constant", value=0)
    
    if rescale_func is None:
        rescale_func = lambda t: t
    rescaled_widths = rescale_func(torch.mean(widths.to(torch.float32), dim=0).clamp_min_(1.0)) + 2 * spacing
    cumulative_rescaled_widths = (0.0, *torch.cumsum(rescaled_widths, dim=0).tolist())
    
    cutoff = cumulative_rescaled_widths[-1]
    aliases = (*flattened_mta_dict.keys(),)

    global_vmin = torch.min(attention_weights[VISUALIZED_INDICES]).item()
    global_vmax = torch.max(attention_weights[VISUALIZED_INDICES]).item()
    
    scale_dict: Dict[str, Callable[[float, float], str]] = {
        "linear": "Normalize",
        "log": "LogNorm",
        "arcsinh": "AsinhNorm",
    }
    norm = getattr(matplotlib.colors, scale_dict[cmap_scale])

    def plot_rescaled_attention(fig: Figure, ax: Axes, attention_weights: torch.Tensor, image_idx: int) -> None:
        if transform_func is not None:
            attention_weights = transform_func(attention_weights)
        
        vmin = global_vmin if global_cmap else torch.min(attention_weights).item()
        vmax = global_vmax if global_cmap else torch.max(attention_weights).item()

        for i, j in itertools.product(
            range(len(flattened_mta_dict)),
            range(len(flattened_mta_dict))
        ):
            h0, h1 = cumulative_rescaled_widths[i] + spacing, cumulative_rescaled_widths[i + 1] - spacing
            w0, w1 = cumulative_rescaled_widths[j] + spacing, cumulative_rescaled_widths[j + 1] - spacing
            im = ax.imshow(
                attention_weights[
                    cumulative_widths[image_idx, i]:cumulative_widths[image_idx, i + 1],
                    cumulative_widths[image_idx, j]:cumulative_widths[image_idx, j + 1],
                ].numpy(force=True),
                extent=(w0, w1, h1, h0),
                norm=norm(vmin=vmin, vmax=vmax),
                interpolation="none", **kwargs
            )
            
            if i == 0:
                ax.text(
                    (w0 + w1) / 2, -0.5, aliases[j],
                    horizontalalignment="center", verticalalignment="center", fontsize="x-small",
                )
        
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05), orientation="vertical")
        ax.set_xlim(left=0, right=cutoff)
        ax.set_ylim(top=0, bottom=cutoff)
        ax.axis("off")

    if transform_func is not None:
        suffix = f"_{transform_func.__name__}"
    else:
        suffix = ""
    
    if per_head:
        for image_idx in VISUALIZED_INDICES:
            image_attention_weights = get_attention_weights_for_image_idx(image_idx)
            
            nrows, ncols = 4, 4
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(PLOT_SCALE * ncols, PLOT_SCALE * nrows))
            for head_idx, weights in enumerate(torch.unbind(image_attention_weights, dim=-1)):
                ax: Axes = axs[head_idx // ncols, head_idx % ncols]
                ax.set_title(f"Head {head_idx}", pad=16.0)
                plot_rescaled_attention(fig, ax, weights, image_idx)
            fig.suptitle(f"Layer {layer_idx}, Image {image_idx}: attention_matrix{suffix}")
            plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE))
        for ax_idx, image_idx in enumerate(VISUALIZED_INDICES):
            ax: Axes = axs[ax_idx]
            ax.set_title(f"Image {image_idx}", pad=16.0)
            plot_rescaled_attention(fig, ax, get_attention_weights_for_image_idx(image_idx), image_idx,)
        fig.suptitle(f"Layer {layer_idx}: attention_matrix{suffix}")
        plt.show()
    plt.close()


def visualize_attention_weights_from_ma_per_image(
    features: ImageFeatures,
    layer_idx: int,
    mta_dict: Dict[str, torch.Tensor],
    mta_aliases: Dict[str, str],
    exclude_self: bool = True,
    invert: bool = False,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    bsz = features.shape[1]
    attention_weights = features.get(layer_idx=layer_idx, key="attention_matrix", include=(ImageFeatures.IMAGE,), with_batch=True)
    attention_weights = torch.mean(attention_weights[:, :, ImageFeatures.image_indices, :], dim=3)
    
    for k, mask in mta_dict.items():
        attention_weights_from_ma = torch.zeros((bsz, ImageFeatures.N), device=DEVICE)
        for image_idx in range(bsz):
            indices = torch.where(mask[image_idx].flatten().to(DEVICE))[0]
            attention_weights_from_ma[image_idx] = torch.sum(attention_weights[image_idx, :, indices], dim=1)

            if exclude_self:
                attention_weights_from_ma[image_idx, indices] -= attention_weights[image_idx, indices, indices]
            
        attention_weights_from_ma = einops.rearrange(
            attention_weights_from_ma,
            "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
        )
        
        if invert:
            attention_weights_from_ma = 1 - attention_weights_from_ma
    
        _visualize_cmap_with_values(attention_weights_from_ma, f"Layer {layer_idx}: attention_weight_on_{mta_aliases[k]}", **kwargs)
        plt.show()
        plt.close()


def visualize_incoming_attention_per_image(
    layer_idx: int,
    attention_weights: torch.Tensor,
    exclude_self: bool = True,
    invert: bool = False,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    if attention_weights.ndim == 4:
        attention_weights = torch.mean(attention_weights, dim=-1)
        
    if exclude_self:
        attention_weights[:, torch.arange(ImageFeatures.N + 1), torch.arange(ImageFeatures.N + 1)] = 0.0
    
    attention_weights_from_cls = einops.rearrange(
        torch.mean(attention_weights[:, :, ImageFeatures.image_indices], dim=1),
        # attention_weights[:, :, 0],
        "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )

    if invert:
        attention_weights_from_cls = 1 - attention_weights_from_cls

    _visualize_cmap_with_values(attention_weights_from_cls, f"Layer {layer_idx}: incoming_attention_weight", **kwargs)
    plt.show()
    plt.close()


def visualize_attention_weights_per_image(
    features: ImageFeatures,
    layer_idx: int,
    mta_dict: Dict[int, torch.Tensor],
    mta_aliases: Dict[int, str],
    mode: Tuple[DecompositionOptions, int],
    rgb_assignment: torch.Tensor,
    per_head: bool = False,
    **kwargs: Any,
) -> None:
    _, bsz, N = features.shape[:3]
    
    attention_weights = features.get(layer_idx, key="attention_matrix", with_batch=True)[:, :, ImageFeatures.image_indices] # [bsz x N? x N x h]
    if attention_weights.ndim < 4:
        per_head = False
    if not per_head and attention_weights.ndim == 4:
        attention_weights = torch.mean(attention_weights, dim=-1)

    mode, ndim = mode
    decomposition = supply_decompositions({mode})[mode]

    num_heads = attention_weights.shape[-1] if per_head else 1
    for k, mta_mask in mta_dict.items():
        mta_count = torch.sum(mta_mask, dim=(1, 2))
        
        mta_attention_weights = torch.zeros((bsz, N, torch.max(mta_count).item() * num_heads))
        for image_idx in range(bsz):
            mta_attention_weights[image_idx, :, :mta_count[image_idx] * num_heads] = einops.rearrange(
                attention_weights[image_idx, :, mta_mask[image_idx].flatten()],
                "n ... -> n (...)",
            )
        
        pseudo_features = ImageFeatures.from_tensor(mta_attention_weights, mta_dict, DEVICE)

        image_features = pseudo_features.get(layer_idx=0, key="", include=(ImageFeatures.IMAGE,), with_batch=True)
        cls_features = pseudo_features.get(layer_idx=0, key="", include=(ImageFeatures.CLS,), with_batch=True)
        fit_features = image_features
        
        ax_names = ("x", "y", "z")
        fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE * 0.75), subplot_kw={"projection": f"{ndim}d"} if ndim != 2 else None)
        for ax_idx, image_idx in enumerate(VISUALIZED_INDICES):       
            ax: Axes = axs[ax_idx]

            utils.reset_seed()
            decomposition.fit(fit_features[image_idx])
            def compress(_features: torch.Tensor) -> torch.Tensor:
                return decomposition.transform(_features)[..., :ndim]

            ax.scatter(
                *compress(image_features[image_idx]).mT.numpy(force=True),
                color=rgb_assignment[image_idx].flatten(0, -2).numpy(force=True), s=1, **kwargs
            )
            ax.scatter(
                *compress(cls_features[image_idx]).mT.numpy(force=True),
                color="black", label="cls_token",
            )
            mta_key = min((float("inf"), *filter(lambda l: l >= layer_idx, mta_dict.keys())))
            if mta_key != float("inf"):
                mask = mta_dict[mta_key][image_idx].flatten()
                ax.scatter(
                    *compress(image_features[image_idx, mask]).mT.numpy(force=True),
                    color=rgb_assignment[image_idx].flatten(0, -2)[mask].numpy(force=True), s=10, **kwargs
                )

            for j in range(ndim):
                getattr(ax, f"set_{ax_names[j]}label")(f"projection{j}")
                # getattr(ax, f"set_{ax_names[j]}lim")(-1.0, 1.0)
            ax.legend()

        fig.suptitle(f"Layer {layer_idx}: {mta_aliases[k]}_attention_weights_{mode}")
        plt.show()
        plt.close()


@torch.no_grad
def compute_attention_contribution(
    features: ImageFeatures,
    layer_idx: int,
    model_dict: List[Dict[str, torch.Tensor]],
    mta_dict: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    bsz = features.shape[1]
    attention_input = features.get(layer_idx=layer_idx, key="attention_input", include=(ImageFeatures.IMAGE,), with_batch=True)  # [bsz x N x d] 
    attention_weights = einops.rearrange(
        features.get(layer_idx=layer_idx, key="attention_matrix", with_batch=True), # [bsz x N? x N? x h]
        "bsz n1 n2 h -> bsz h n1 n2", bsz=bsz,
    )[..., ImageFeatures.image_indices]                                             # [bsz x h x N? x N]

    head_dim, D = 64, attention_input.shape[-1]
    out = utils.linear_from_wb(model_dict[layer_idx]["out_w"], None).to(DEVICE)
    V = utils.linear_from_wb(
        model_dict[layer_idx]["QKVw"][2 * D:],
        model_dict[layer_idx]["QKVb"][2 * D:],
    ).to(DEVICE)
    values = values = einops.rearrange(V.forward(attention_input), "bsz n (h k) -> bsz h n k", k=head_dim)  # [bsz x h x N x k]
    
    masked_attention_outputs = {}
    for k, mta_mask in mta_dict.items():
        masked_attention_weights = attention_weights * einops.rearrange(mta_mask.to(DEVICE), "bsz h w -> bsz 1 1 (h w)")
        
        masked_attention_outputs[k] = out.forward(einops.rearrange(
            masked_attention_weights @ values,                      # [bsz x h x N? x k]
            "bsz h n k -> bsz n (h k)",
        )).to(OUTPUT_DEVICE)                                        # [bsz x N? x d]    
    torch.cuda.empty_cache()
    return masked_attention_outputs
