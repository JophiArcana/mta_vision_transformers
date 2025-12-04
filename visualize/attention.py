import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import matplotlib.colors
import numpy as np
import torch
import torch.nn.functional as Fn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nystrom_ncut import AxisAlign
from sklearn.manifold import TSNE

from core.decomposition import DecompositionOptions, supply_decompositions
from infrastructure import utils
from infrastructure.settings import DEVICE, OUTPUT_DEVICE
from modeling.image_features import ImageFeatures
from visualize.base import (
    VISUALIZED_INDICES, NUM_VISUALIZED_IMAGES, PLOT_SCALE, CMAPScaleOptions,
    _visualize_cmap_with_values, symmetrize,
)



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
    order_by_tsne: bool = False,
    per_head: bool = False,
    rescale_func: Callable[[torch.Tensor], torch.Tensor] = None,
    symmetric_cmap: bool = False,
    global_cmap: bool = True,
    cmap_scale: CMAPScaleOptions = "linear",
    subsample: float = 1.0,
    spacing: float = 0.1,
    spacing_color: str = "white",
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    bsz, N = attention_weights.shape[:2]

    if attention_weights.ndim < 4:
        per_head = False
    if not per_head and attention_weights.ndim == 4:
        attention_weights = torch.mean(attention_weights, dim=-1)
    # compressed_attention_weights = torch.mean(attention_weights, dim=-1) if per_head else attention_weights
    
    # Construct token masks for each cateogory
    flattened_mta_dict: OrderedDict[str, torch.Tensor] = OrderedDict({
        "CLS": (torch.arange(N) == 0).expand((bsz, -1)),
    })
    normal_mask = (torch.arange(N) > 0) * (torch.all(torch.isfinite(attention_weights).flatten(2, -1), dim=-1))
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

    global_vmin, global_vmax = torch.min(attention_weights[VISUALIZED_INDICES]).item(), torch.max(attention_weights[VISUALIZED_INDICES]).item()
    
    scale_dict: Dict[str, Callable[[float, float], str]] = {
        "linear": "Normalize",
        "log": "LogNorm",
        "arcsinh": "AsinhNorm",
    }
    norm = getattr(matplotlib.colors, scale_dict[cmap_scale])

    def plot_rescaled_attention(fig: Figure, ax: Axes, attention_weights: torch.Tensor, image_idx: int) -> None:
        if transform_func is not None:
            attention_weights = transform_func(attention_weights)
        
        sub_orders: List[torch.Tensor] = []
        for i in range(len(flattened_mta_dict)):
            sub_orders.append(torch.arange(widths[image_idx, i], device=DEVICE))
        
        vmin, vmax = (global_vmin, global_vmax) if global_cmap else (
            torch.min(attention_weights).item(),
            torch.max(attention_weights).item(),
        )
        if symmetric_cmap:
            vmin, vmax = symmetrize(vmin, vmax)

        ax.add_patch(Rectangle((0, 0), cumulative_rescaled_widths[-1], cumulative_rescaled_widths[-1], facecolor=spacing_color, zorder=-100)) 
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
                    ha="center", va="center", fontsize="medium",
                )
            if j == 0:
                ax.text(
                    -0.5, (h0 + h1) / 2, aliases[i],
                    ha="center", va="center", fontsize="medium", rotation="vertical",
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
                ax.set_title(f"Head {head_idx}", pad=24.0, fontsize="xx-large")
                plot_rescaled_attention(fig, ax, weights, image_idx)
            fig.suptitle(f"Layer {layer_idx}, Image {image_idx}: attention_matrix{suffix}")
            plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE))
        for ax_idx, image_idx in enumerate(VISUALIZED_INDICES):
            ax: Axes = axs[ax_idx] if NUM_VISUALIZED_IMAGES > 1 else axs
            ax.set_title(f"Image {image_idx}", pad=24.0, fontsize="xx-large")
            plot_rescaled_attention(fig, ax, get_attention_weights_for_image_idx(image_idx), image_idx,)
        fig.suptitle(f"Layer {layer_idx}: attention_matrix{suffix}")
        plt.show()
    plt.close()









def visualize_attention_matrix_per_layer(
    layer_indices: List[int],
    attention_weights: torch.Tensor,
    name_to_mask: Dict[str, torch.Tensor],
    transform_func: Callable[[torch.Tensor], torch.Tensor] = None,
    rescale_func: Callable[[torch.Tensor], torch.Tensor] = None,
    symmetric_cmap: bool = False,
    global_cmap: bool = True,
    cmap_scale: CMAPScaleOptions = "linear",
    subsample: float = 1.0,
    spacing: float = 0.1,
    spacing_color: str = "white",
    save_fname: str = None,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    bsz, N = attention_weights.shape[1:3]

    if attention_weights.ndim == 5:
        attention_weights = torch.mean(attention_weights, dim=-1)
    
    # Construct token masks for each cateogory
    flattened_mta_dict: OrderedDict[str, torch.Tensor] = OrderedDict({
        "CLS": (torch.arange(N) == 0).expand((bsz, -1)),
    })
    normal_mask = (torch.arange(N) > 0) * (torch.all(torch.isfinite(attention_weights).flatten(3, -1), dim=[0, -1]))
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
        return attention_weights[:, image_idx, _order[:, None], _order[None, :]]

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

    scale_dict: Dict[str, Callable[[float, float], str]] = {
        "linear": "Normalize",
        "log": "LogNorm",
        "arcsinh": "AsinhNorm",
    }
    norm = getattr(matplotlib.colors, scale_dict[cmap_scale])

    def plot_rescaled_attention(fig: Figure, ax: Axes, attention_weights: torch.Tensor, image_idx: int, global_vmin: float, global_vmax: float) -> None:
        if transform_func is not None:
            attention_weights = transform_func(attention_weights)
        
        sub_orders: List[torch.Tensor] = []
        for i in range(len(flattened_mta_dict)):
            sub_orders.append(torch.arange(widths[image_idx, i], device=DEVICE))
        
        vmin, vmax = (global_vmin, global_vmax) if global_cmap else (
            torch.min(attention_weights).item(),
            torch.max(attention_weights).item(),
        )
        if symmetric_cmap:
            vmin, vmax = symmetrize(vmin, vmax)

        ax.add_patch(Rectangle((0, 0), cumulative_rescaled_widths[-1], cumulative_rescaled_widths[-1], facecolor=spacing_color, zorder=-100)) 
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
                    ha="center", va="center", fontsize="large",
                )
            if j == 0:
                ax.text(
                    -0.5, (h0 + h1) / 2, aliases[i],
                    ha="center", va="center", fontsize="large", rotation="vertical",
                )
        
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05), orientation="vertical")
        ax.set_xlim(left=0, right=cutoff)
        ax.set_ylim(top=0, bottom=cutoff)
        ax.axis("off")

    # if transform_func is not None:
    #     suffix = f"_{transform_func.__name__}"
    # else:
    #     suffix = ""
    
    for image_idx in VISUALIZED_INDICES:
        image_attention_weights = get_attention_weights_for_image_idx(image_idx)
        global_vmin, global_vmax = torch.min(image_attention_weights[layer_indices]).item(), torch.max(image_attention_weights[layer_indices]).item()
        
        fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(PLOT_SCALE * len(layer_indices), PLOT_SCALE))
        for ax_idx, layer_idx in enumerate(layer_indices):
            ax: Axes = axs[ax_idx]
            ax.set_title(f"Layer {layer_idx}", pad=24.0, fontsize="xx-large")
            plot_rescaled_attention(fig, ax, image_attention_weights[layer_idx], image_idx, global_vmin, global_vmax)
        fig.suptitle(
            f"Attention Matrices from Layers {min(layer_indices)} to {max(layer_indices)}",
            fontsize=22, y=1.02,
        )
        if save_fname is not None:
            plt.savefig(f"{save_fname}_image{image_idx}.pdf", bbox_inches="tight")
        plt.show()
        plt.close()











def visualize_attention_suppression_per_image(
    features: ImageFeatures,
    layer_idx: int,
    model_weights: List[Dict[str, torch.Tensor]],
    empirical: bool,
    normalize: bool,
    
    name_to_mask: Dict[str, torch.Tensor],
    pairwise: bool = True,
    transform_func: Callable[[torch.Tensor], torch.Tensor] = None,
    order_by_tsne: bool = False,
    per_head: bool = False,
    rescale_func: Callable[[torch.Tensor], torch.Tensor] = None,
    global_cmap: bool = True,
    cmap_scale: CMAPScaleOptions = "linear",
    subsample: float = 1.0,
    spacing: float = 0.1,
    **kwargs: Any,
) -> torch.Tensor:
    bsz = features.shape[1]
    x = features.get(layer_idx, "layer_input", include=(ImageFeatures.CLS, ImageFeatures.IMAGE,))           # [(bsz x n) x embed_dim]
    ln1_x = features.get(layer_idx, "attention_input", include=(ImageFeatures.CLS, ImageFeatures.IMAGE,))   # [(bsz x n) x embed_dim]
    
    hD, D = 64, x.shape[-1]
    QKVw, QKVb = model_weights[layer_idx]["QKVw"], model_weights[layer_idx]["QKVb"]
    
    V = einops.rearrange(Fn.linear(ln1_x, QKVw[2 * D:], QKVb[2 * D:]), "bszp (h hd) -> h hd bszp", hd=hD)   # [n_heads x head_dim x (bsz x n)]
    attn_out = einops.rearrange(model_weights[layer_idx]["out_w"], "d (h hd) -> h d hd", hd=hD, d=D)        # [n_heads x embed_dim x head_dim]
    subspace = attn_out @ V                                                                                 # [n_heads x embed_dim x (bsz x n)]
    
    x = Fn.normalize(x, p=2, dim=1)                                                                         # [(bsz x n) x embed_dim]
    if normalize:
        subspace = Fn.normalize(subspace, p=2, dim=1)
    x = einops.rearrange(x, "(bsz n) d -> bsz n d", bsz=bsz)                                                # [bsz x n x embed_dim]
    subspace = einops.rearrange(subspace, "h d (bsz n) -> bsz h n d", bsz=bsz)                              # [bsz x n_heads x n x embed_dim]


    pairwise_suppression_projection = x[:, None, :, :] @ subspace.mT                                # [bsz x n_heads x n x n]
    pairwise_suppression_projection = einops.rearrange(pairwise_suppression_projection, "bsz h n1 n2 -> bsz n1 n2 h")   # [bsz x n x n x n_heads]
    
    if empirical:
        attn_matrix = features.get(layer_idx, "unmasked_attention_matrix", include=(ImageFeatures.CLS, ImageFeatures.IMAGE,), with_batch=True)  # [bsz x n x n x n_heads]
        assert attn_matrix.ndim == 4
        attn_matrix = Fn.normalize(attn_matrix, p=1, dim=-1)
        
        pairwise_suppression_projection = pairwise_suppression_projection * attn_matrix

    kwargs["cmap"] = "bwr"
    if pairwise:
        visualize_attention_matrix_per_image(
            layer_idx=layer_idx,
            attention_weights=pairwise_suppression_projection,
            name_to_mask=name_to_mask,
            transform_func=transform_func,
            order_by_tsne=order_by_tsne,
            per_head=per_head,
            rescale_func=rescale_func,
            symmetric_cmap=True,
            global_cmap=global_cmap,
            cmap_scale=cmap_scale,
            subsample=subsample,
            spacing=spacing,
            spacing_color="black",
            **kwargs,
        )
    else:
        # suppression_projection = pairwise_suppression_projection[:, torch.arange(ImageFeatures.N + 1), torch.arange(ImageFeatures.N + 1), :]
        suppression_projection = torch.linalg.norm(pairwise_suppression_projection, dim=2) ** 2
        if per_head:
            for head_idx, projection in enumerate(torch.unbind(suppression_projection, dim=-1)):
                _visualize_cmap_with_values(einops.rearrange(
                    projection[:, ImageFeatures.image_indices],
                    "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
                ), f"Layer {layer_idx}, head {head_idx}: suppression_projection", **kwargs)
                plt.show()
                plt.close()
        else:
            _visualize_cmap_with_values(einops.rearrange(
                torch.mean(suppression_projection, dim=-1)[:, ImageFeatures.image_indices],
                "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
            ), f"Layer {layer_idx}: suppression_projection", **kwargs)
            plt.show()
            plt.close()
    
    return pairwise_suppression_projection









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

    _visualize_cmap_with_values(attention_weights_from_cls, f"Layer {layer_idx}: incoming_attention_weight", **kwargs)
    plt.show()
    plt.close()


def visualize_attention_to_MA_per_image(
    layer_idx: int,
    attention_weights: torch.Tensor,
    mta_mask: torch.Tensor,
    per_head: bool = False,
    exclude_self: bool = False,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    if attention_weights.ndim < 4:
        per_head = False

    def plot_attention_to_MA(attention_matrix: torch.Tensor, title: str) -> None:
        attention_matrix = attention_matrix.clone()
        if exclude_self:
            attention_matrix[:, torch.arange(ImageFeatures.N + 1), torch.arange(ImageFeatures.N + 1)] = 0.0
        
        attention_weights_to_MA = einops.rearrange(
            torch.sum(attention_matrix[:, ImageFeatures.image_indices, :] * mta_mask[:, None, :], dim=-1),
            "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
        )

        _visualize_cmap_with_values(attention_weights_to_MA, title, **kwargs)
        plt.show()
        plt.close()

    if per_head:
        for head_idx in range(attention_weights.shape[-1]):
            plot_attention_to_MA(attention_weights[..., head_idx], f"Layer {layer_idx}, head {head_idx}: attention_to_MA_weight", **kwargs)
    else:
        if attention_weights.ndim == 4:
            attention_weights = torch.mean(attention_weights, dim=-1)
        plot_attention_to_MA(attention_weights, f"Layer {layer_idx}: attention_to_MA_weight", **kwargs)


def visualize_attention_from_CLS_per_image(
    layer_idx: int,
    attention_weights: torch.Tensor,
    mta_mask: torch.Tensor,
    per_head: bool = False,
    exclude_MA: bool = True,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    if attention_weights.ndim < 4:
        per_head = False

    def plot_attention_from_CLS(attention_matrix: torch.Tensor, title: str) -> None:
        attention_matrix = attention_matrix.clone()
        if exclude_MA:
            attention_matrix = attention_matrix * ~mta_mask[:, None, :]
        
        attention_weights_from_CLS = einops.rearrange(
            attention_matrix[:, 0, ImageFeatures.image_indices],
            "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
        )

        _visualize_cmap_with_values(attention_weights_from_CLS, title, **kwargs)
        plt.show()
        plt.close()

    if per_head:
        for head_idx in range(attention_weights.shape[-1]):
            plot_attention_from_CLS(attention_weights[..., head_idx], f"Layer {layer_idx}, head {head_idx}: attention_from_CLS_weight", **kwargs)
    else:
        if attention_weights.ndim == 4:
            attention_weights = torch.mean(attention_weights, dim=-1)
        plot_attention_from_CLS(attention_weights, f"Layer {layer_idx}: attention_from_CLS_weight", **kwargs)


def visualize_attention_sink_decay(
    attention_weights: torch.Tensor,
    ranked_AS_mask: torch.Tensor,
    mode: str,
    title: str,
    k: int = 20,
    use_cls_proxy: bool = True,
    lock_tokens: bool = False,
    cmap: str = "magma",
    max_labels: int = 8,
    **kwargs: Any,
) -> None:
    bsz = attention_weights.shape[0]
    
    if attention_weights.ndim == 4:
        if use_cls_proxy:
            attention_weights = attention_weights[:, :, 0, :]
        else:
            attention_weights = torch.mean(attention_weights, dim=2)

    counts = torch.sum(ranked_AS_mask.isfinite(), dim=1)                                                    # [bsz]
    if lock_tokens:
        indices = torch.topk(ranked_AS_mask, k=k, dim=1, largest=False).indices
        topk_values = attention_weights[torch.arange(bsz)[:, None], :, indices].mT
    else:
        topk_values = torch.topk(attention_weights[:, :, ImageFeatures.image_indices], k=k, dim=2).values   # [bsz x num_iter x k]
 
    fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE * 0.75))
    for ax_idx, image_idx in enumerate(VISUALIZED_INDICES):
        ax: Axes = axs[ax_idx]
        colors: np.ndarray = matplotlib.colormaps.get_cmap(cmap)(torch.linspace(1, 0.3, n_it := (counts[image_idx].item() + 1)).numpy(force=True))
        for it in range(n_it):
            if it == 0:
                label: str = f"No {mode}"
            elif it < max_labels or it == n_it - 1:
                label = f"{mode.capitalize()} {it}"
            else:
                label = None
            
            ax.plot(
                torch.arange(1, k + 1).numpy(force=True), topk_values[image_idx, it].numpy(force=True),
                color=colors[it], zorder=it, marker=".", label=label, **kwargs,
            )
        ax.set_xlabel(f"Sorted attention sinks")
        # ax.set_yscale("log")
        ax.set_title(f"Image {image_idx}", fontsize="x-large")
        ax.legend(fontsize=8, loc="upper right")
        
    fig.suptitle(title, fontsize="xx-large", y=1.02)
    plt.show()
    plt.close()


def visualize_attention_sink_decay_by_type(
    attention_weights_dict: Dict[str, torch.Tensor],
    ranked_AS_mask: torch.Tensor,
    mode_dict: Dict[str, str],
    title: str,
    k: int = 20,
    use_cls_proxy: bool = True,
    cmap: str = "magma",
    max_labels: int = 6,
    save_dir: str = None,
    **kwargs: Any,
) -> None:
    bsz = ranked_AS_mask.shape[0]
    counts = torch.sum(ranked_AS_mask.isfinite(), dim=1)                                                    # [bsz]
    indices = torch.topk(ranked_AS_mask, k=k, dim=1, largest=False).indices

    attention_weights_dict = attention_weights_dict.copy()
    for mask_type, v in attention_weights_dict.items():
        if v.ndim == 4:
            if use_cls_proxy:
                attention_weights_dict[mask_type] = v[:, :, 0, :]
            else:
                attention_weights_dict[mask_type] = torch.mean(v, dim=2)

    for image_idx in VISUALIZED_INDICES:    
        fig, axs = plt.subplots(
            nrows=1, ncols=len(attention_weights_dict), figsize=(PLOT_SCALE * len(attention_weights_dict) * 1.5, PLOT_SCALE),
            sharey=True,
        )
        for ax_idx, (mask_type, attention_weights) in enumerate(attention_weights_dict.items()):
            ax: Axes = axs[ax_idx]
            colors: np.ndarray = matplotlib.colormaps.get_cmap(cmap)(torch.linspace(0.9, 0.0, n_it := (counts[image_idx].item() + 1)).numpy(force=True))
            
            mode = mode_dict[mask_type]
            topk_values = attention_weights[torch.arange(bsz)[:, None], :, indices].mT
            for it in range(n_it):
                if it == 0:
                    label: str = f"No {mode}"
                elif it < max_labels or it == n_it - 1:
                    label = f"{mode.capitalize()} {it}"
                else:
                    label = None
                
                if it in [0, n_it - 1]:
                    ax.plot(
                        torch.arange(1, k + 1).numpy(force=True), topk_values[image_idx, it].numpy(force=True),
                        color=colors[it], zorder=it, marker=".", markersize=16, linewidth=3, label=label, **kwargs,
                    )
                else:
                    ax.plot(
                        torch.arange(1, k + 1).numpy(force=True), topk_values[image_idx, it].numpy(force=True),
                        color=colors[it], zorder=it, marker=".", linewidth=1, label=label, **kwargs,
                    )
            ax.grid(axis="both", color="gray", linestyle="--", alpha=0.5)
                
            ax.set_xlabel(f"Ranked attention sinks", fontsize=18)
            ax.set_xticks(torch.arange(1, k + 1, 2).numpy(force=True))
            ax.set_ylabel("Attention to CLS", fontsize=18)
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.set_title(mask_type, fontsize=20)
            ax.legend(fontsize=16, loc="upper right")
            
        fig.suptitle(title, fontsize=24, y=1.02)
        if save_dir is not None:
            plt.savefig(f"{save_dir}/attention_sink_decay_image{image_idx}.pdf", bbox_inches="tight")
        plt.show()
        plt.close()




def visualize_layer_output_decay_by_type(
    layer_output_dict: Dict[str, torch.Tensor],
    ranked_AS_mask: torch.Tensor,
    mode_dict: Dict[str, str],
    title: str,
    k: int = 20,
    use_cls_proxy: bool = True,
    cmap: str = "magma",
    max_labels: int = 6,
    save_dir: str = None,
    **kwargs: Any,
) -> None:
    bsz = ranked_AS_mask.shape[0]
    counts = torch.sum(ranked_AS_mask.isfinite(), dim=1)                                                    # [bsz]
    indices = torch.topk(ranked_AS_mask, k=k, dim=1, largest=False).indices

    layer_output_dict = {k: torch.norm(v, dim=-1) for k, v in layer_output_dict.items()}



    for image_idx in VISUALIZED_INDICES:    
        fig, axs = plt.subplots(
            nrows=1, ncols=len(layer_output_dict), figsize=(PLOT_SCALE * len(layer_output_dict) * 1.5, PLOT_SCALE),
            sharey=True,
        )
        for ax_idx, (mask_type, layer_output) in enumerate(layer_output_dict.items()):
            ax: Axes = axs[ax_idx]
            colors: np.ndarray = matplotlib.colormaps.get_cmap(cmap)(torch.linspace(0.9, 0.0, n_it := (counts[image_idx].item() + 1)).numpy(force=True))
            
            mode = mode_dict[mask_type]
            topk_values = layer_output[torch.arange(bsz)[:, None], :, indices].mT
            
            for it in range(n_it):
                if it == 0:
                    label: str = f"No {mode}"
                elif it < max_labels or it == n_it - 1:
                    label = f"{mode.capitalize()} {it}"
                else:
                    label = None
                
                if it in [0, n_it - 1]:
                    ax.plot(
                        torch.arange(1, k + 1).numpy(force=True), topk_values[image_idx, it].numpy(force=True),
                        color=colors[it], zorder=it, marker=".", markersize=16, linewidth=3, label=label, **kwargs,
                    )
                else:
                    ax.plot(
                        torch.arange(1, k + 1).numpy(force=True), topk_values[image_idx, it].numpy(force=True),
                        color=colors[it], zorder=it, marker=".", linewidth=1, label=label, **kwargs,
                    )
            ax.grid(axis="both", color="gray", linestyle="--", alpha=0.5)
            
            
            ax.set_xlabel(f"Ranked attention sinks", fontsize=18)
            ax.set_xticks(torch.arange(1, k + 1, 2).numpy(force=True))
            ax.set_ylabel("Block Output Norm", fontsize=18)
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.set_title(mask_type, fontsize=20)
            ax.legend(fontsize=16, loc="upper right")
            
            # ax.set_xlabel(f"Unmasked Block Output Norm")
            # ax.set_xscale("functionlog", functions=[np.log, np.exp])
            # # ax.set_xticks(torch.arange(0, k + 1, 2).numpy(force=True))
            # ax.yaxis.set_tick_params(labelbottom=True)
            # # ax.set_yscale("log")
            # ax.set_yscale("functionlog", functions=[np.log, np.exp])
            # ax.set_title(mask_type, fontsize="x-large")
            # ax.legend(fontsize=8, loc="upper right")
            
        fig.suptitle(title, fontsize=24, y=1.02)
        if save_dir is not None:
            plt.savefig(f"{save_dir}/layer_output_decay_image{image_idx}.pdf", bbox_inches="tight")
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
