import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import matplotlib.colors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nystrom_ncut import NCut, SampleConfig, AxisAlign
from sklearn.base import TransformerMixin
from tensordict import TensorDict
from torch.utils._pytree import tree_flatten
from torch_pca import PCA

from core.qk import qk_intersection, qk_projection_variance
from infrastructure import utils
from infrastructure.settings import DEVICE, OUTPUT_DEVICE, SEED
from modeling.image_features import ImageFeatures

num_visualized_images = 4
s: float = 5.0
DEFAULT_LAYER_INDICES = [*range(9, 12)]

new = True
n_components = 100
num_sample = 20000


def reset_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def construct_per_layer_output_dict(_per_metric_output_dict: Dict[str, np.ndarray[torch.Tensor]]) -> List[TensorDict]:
    return [
        TensorDict(dict(zip(_per_metric_output_dict.keys(), next(zip(*v))))).auto_device_().auto_batch_size_()
        for v in zip(*_per_metric_output_dict.values())
    ]


class ComposeDecomposition(TransformerMixin):
    def __init__(self, decompositions: List[TransformerMixin]):
        self.decompositions = decompositions
        self.is_fitted: bool = False

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        for decomposition in self.decompositions:
            reset_seed()
            if getattr(decomposition, "is_fitted", False):
                X = decomposition.transform(X)
            else:
                X = decomposition.fit_transform(X)
        self.is_fitted = True
        return X

    def fit(self, X: torch.Tensor) -> Any:
        self.fit_transform(X)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        for decomposition in self.decompositions:
            X = decomposition.transform(X)
        return X

    
def generate_NCUT(distance: str = "rbf"):
    if new:
        return NCut(
            n_components=n_components,
            distance=distance,
            adaptive_scaling=True,
            sample_config=SampleConfig(
                method="fps",
                # method="fps_recursive",
                num_sample=num_sample,
                fps_dim=12,
                # n_iter=1,
            ),
            eig_solver="svd_lowrank"
        )
    else:
        from ncut_pytorch import NCUT
        return NCUT(num_eig=n_components, num_sample=num_sample, distance=distance)


DecompositionOptions = Literal["linear", "ncut", "recursive_ncut", "count", "norm", "marginal_norm"]
def supply_decompositions(modes: Set[DecompositionOptions]) -> Dict[str, TransformerMixin]:
    base_nc = generate_NCUT(distance="rbf")
    recursive_nc = generate_NCUT(distance="cosine")
    result = {
        "linear": PCA(n_components=n_components),
        "ncut": base_nc,
        "recursive_ncut": ComposeDecomposition([base_nc, recursive_nc]),
        **{
            k: ComposeDecomposition([
                base_nc, recursive_nc,
                AxisAlign(sort_method=k),
            ]) for k in ("count", "norm", "marginal_norm",)
        },
        "ncut_pca": ComposeDecomposition([
            generate_NCUT(),
            PCA(n_components=n_components),
        ]),
    }
    return {k: v for k, v in result.items() if k in modes}


def generate_rgb_from_tsne_3d(
    features: torch.Tensor,     # [N x D]
) -> torch.Tensor:              # [N x 3]
    if new:
        from nystrom_ncut import rgb_from_tsne_3d, rgb_from_euclidean_tsne_3d
        # return rgb_from_tsne_3d(features)
        return rgb_from_euclidean_tsne_3d(features, num_sample=1000)
    else:
        from ncut_pytorch import rgb_from_tsne_3d
        return rgb_from_tsne_3d(features)[1]


def demean(t: torch.Tensor) -> torch.Tensor:
    return t - torch.mean(t.flatten(0, -2), dim=0)


def svd(t: torch.Tensor, center: bool) -> torch.Tensor:
    mean = torch.mean(t.flatten(0, -2), dim=0) if center else 0.0
    return torch.linalg.svd(t - mean, full_matrices=False)

    
def visualize_images_with_mta(
    original_images: torch.Tensor,
    mta_mask: torch.Tensor = None,
) -> None:
    mta_mask = einops.rearrange(
        mta_mask[:, ImageFeatures.image_indices],
        "b (h w) -> b h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(2 * num_visualized_images, 2))
    for image_idx, original_image in enumerate(shift_channels(original_images[:num_visualized_images])):
        if mta_mask is not None:
            mask = transforms.Resize(original_image.shape[:2])(mta_mask[None, image_idx].to(dtype=torch.float))[0, ..., None]
            image = (1 - mask) * original_image + mask * torch.tensor((1.0, 0.0, 0.0))
        else:
            image = original_image
        
        axs[image_idx].imshow(image.numpy(force=True))
        axs[image_idx].axis("off")
    fig.suptitle("original_image")
    plt.show()


def shift_channels(images_: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(images_, "bsz c h w -> bsz h w c")


def get_rgb_colors(features: ImageFeatures, layer_idx: int, key: str, use_all: bool) -> torch.Tensor:
    reset_seed()

    ncut = generate_NCUT()
    image_features = features.get(layer_idx=layer_idx, key=key, include=(ImageFeatures.IMAGE,)) # [(bsz h w) x D]
    if use_all:
        fit_features = features.get(layer_idx=layer_idx, key=key)                               # [? x D]
        ncut_features = ncut.fit(fit_features).transform(image_features)                        # [(bsz h w) x d]
    else:
        ncut_features = ncut.fit_transform(image_features)                                      # [(bsz h w) x d]

    rgb_colors = einops.rearrange(
        generate_rgb_from_tsne_3d(ncut_features),
        "(bsz h w) c -> bsz h w c", h=ImageFeatures.H, w=ImageFeatures.W,
    ).to(OUTPUT_DEVICE)
    return rgb_colors


def visualize_features_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    mta_mask: torch.Tensor = None,
    use_all: bool = False,
    highlight: torch.Tensor = None,
) -> None:
    rgb_features = get_rgb_colors(features, layer_idx=layer_idx, key=metric_name, use_all=use_all)   # [B x H x W x 3]

    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(s * num_visualized_images, s))
    for image_idx, image_features in enumerate(rgb_features[:num_visualized_images]):
        axs[image_idx].imshow(image_features.numpy(force=True))
        axs[image_idx].axis("off")
    fig.suptitle(f"Layer {layer_idx}: {metric_name}")

    def draw_square(image_idx: int, h_idx: int, w_idx: int, color: str) -> None:
        axs[image_idx].plot(
            w_idx - 0.5 + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
            h_idx - 0.5 + torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
            color=color, linewidth=4.0,
        )

    if mta_mask is not None:
        for image_idx, h_idx, w_idx in torch.argwhere(einops.rearrange(
            mta_mask[:num_visualized_images, ImageFeatures.image_indices],
            "b (h w) -> b h w", h=ImageFeatures.H, w=ImageFeatures.W,
        )):
            draw_square(image_idx, h_idx, w_idx, "black")
    if highlight is not None:
        for image_idx, h_idx, w_idx in highlight:
            if image_idx < num_visualized_images:
                draw_square(image_idx, h_idx, w_idx, "white")
    plt.show()


def visualize_qk_projection_per_image(
    features: ImageFeatures,
    layer_idx: int,
    model_dict: List[Dict[str, torch.Tensor]],
    p: float = 2.0,
    aggregate_func: Callable[[torch.Tensor, int], torch.Tensor] = None,
) -> None:
    layer_output = features.get(layer_idx=layer_idx, key="attention_input", include=(ImageFeatures.IMAGE,)) # [(bsz h w) x d]

    QKVw = model_dict[layer_idx]["QKVw"].to(DEVICE)
    QKVb = model_dict[layer_idx]["QKVb"].to(DEVICE)
    
    head_dim = 64
    D = layer_output.shape[-1]
    Qw = QKVw[:D].reshape(-1, head_dim, D)
    Kw = QKVw[D:2 * D].reshape(-1, head_dim, D)

    Qb = QKVb[:D].reshape(-1, head_dim)
    Kb = QKVb[D:2 * D].reshape(-1, head_dim)
    
    qk = qk_intersection(Qw, Kw, Qb, Kb)
    if aggregate_func is not None:
        projection_variance = tree_flatten(aggregate_func(qk_projection_variance(layer_output, qk, p, joint=False), dim=-2))[0][0]
        aggregate_name = aggregate_func.__name__
    else:
        projection_variance = qk_projection_variance(layer_output, qk, p, joint=True)
        aggregate_name = "joint"

    projection_variance = einops.rearrange(
        projection_variance,
        "(bsz h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    _visualize_cmap_with_values(projection_variance, f"Layer {layer_idx}: qk_projection_{aggregate_name}_variance", cmap="gray")
    plt.show()


def visualize_qk_projection_per_image2(
    features: ImageFeatures,
    layer_idx: int,
    model_dict: List[Dict[str, torch.Tensor]],
    p: float = 2.0,
    aggregate_func: Callable[[torch.Tensor, int], torch.Tensor] = None,
) -> None:
    layer_output = features.get(layer_idx=layer_idx, key="attention_input", include=(ImageFeatures.IMAGE,)) # [(bsz h w) x d]

    QKVw = model_dict[layer_idx]["QKVw"].to(DEVICE)
    QKVb = model_dict[layer_idx]["QKVb"].to(DEVICE)
    
    head_dim = 64
    D = layer_output.shape[-1]
    Qw = QKVw[:D].reshape(-1, head_dim, D)
    Kw = QKVw[D:2 * D].reshape(-1, head_dim, D)

    Qb = QKVb[:D].reshape(-1, head_dim)
    Kb = QKVb[D:2 * D].reshape(-1, head_dim)
    
    print(Qw.shape, Qb.shape, Kw.shape, Kb.shape)
    raise Exception()
    
    
    qk = qk_intersection(Qw, Kw, Qb, Kb)
    if aggregate_func is not None:
        projection_variance = tree_flatten(aggregate_func(qk_projection_variance(layer_output, qk, p, joint=False), dim=-2))[0][0]
        aggregate_name = aggregate_func.__name__
    else:
        projection_variance = qk_projection_variance(layer_output, qk, p, joint=True)
        aggregate_name = "joint"





    projection_variance = einops.rearrange(
        projection_variance,
        "(bsz h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    _visualize_cmap_with_values(projection_variance, f"Layer {layer_idx}: qk_projection_{aggregate_name}_variance", cmap="gray")
    plt.show()
    

def visualize_feature_norms_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    p: float = 2.0,
    **kwargs: Any
) -> torch.Tensor:
    feature_norms = einops.rearrange(
        torch.norm(features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,)), p=p, dim=-1),
        "(bsz h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    _visualize_cmap_with_values(feature_norms, f"Layer {layer_idx}: {metric_name}_norm", **kwargs)
    plt.show()
    return feature_norms


ScaleOptions = Literal["linear", "log", "arcsinh"]
def visualize_attention_matrix_per_image(
    features: ImageFeatures,
    layer_idx: int,
    mta_aliases: Dict[int, str],
    transform_func: Callable[[torch.Tensor], torch.Tensor] = None,
    per_head: bool = False,
    rescale_func: Callable[[torch.Tensor], torch.Tensor] = None,
    global_cmap: bool = True,
    cmap_scale: ScaleOptions = "linear",
    subsample: float = 1.0,
    spacing: float = 0.1,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    attention_weights = features.get(layer_idx=layer_idx, key="attention_matrix", include=(ImageFeatures.CLS, ImageFeatures.IMAGE,), with_batch=True)
    if attention_weights.ndim < 4:
        per_head = False
    
    _visualize_attention_matrix_per_image(
        layer_idx=layer_idx,
        attention_weights=attention_weights,
        name_to_mask={
            alias: features.masks[ImageFeatures.process_key(k)][layer_idx]
            for k, alias in mta_aliases.items()
        },
        transform_func=transform_func,
        per_head=per_head,
        rescale_func=rescale_func,
        global_cmap=global_cmap,
        cmap_scale=cmap_scale,
        subsample=subsample,
        spacing=spacing,
        **kwargs,
    )


def visualize_attention_matrix_per_image_with_masks(
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
    _visualize_attention_matrix_per_image(
        layer_idx=layer_idx,
        attention_weights=attention_weights,
        name_to_mask=name_to_mask,
        transform_func=transform_func,
        per_head=per_head,
        rescale_func=rescale_func,
        global_cmap=global_cmap,
        cmap_scale=cmap_scale,
        subsample=subsample,
        spacing=spacing,
        **kwargs,
    )


def _visualize_attention_matrix_per_image(
    layer_idx: int,
    attention_weights: torch.Tensor,
    name_to_mask: Dict[str, torch.Tensor],
    transform_func: Callable[[torch.Tensor], torch.Tensor],
    per_head: bool,
    rescale_func: Callable[[torch.Tensor], torch.Tensor],
    global_cmap: bool,
    cmap_scale: ScaleOptions,
    subsample: float,
    spacing: float,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    bsz, N = attention_weights.shape[:2]

    if not per_head:
        attention_weights = torch.mean(attention_weights, dim=-1)
    
    # Construct token masks for each cateogory
    flattened_mta_dict: Dict[str, torch.Tensor] = {
        "CLS": (torch.arange(N) == 0).expand((bsz, -1)),
    }
    normal_mask = torch.all(torch.isfinite(attention_weights), dim=-1).to(OUTPUT_DEVICE)
    for k, mask in name_to_mask.items():
        if k != "exclude":
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
    rescaled_widths = rescale_func(torch.mean(widths.to(torch.float32), dim=0)) + 2 * spacing
    cumulative_rescaled_widths = (0.0, *torch.cumsum(rescaled_widths, dim=0).tolist())
    
    cutoff = cumulative_rescaled_widths[-1]
    aliases = (*flattened_mta_dict.keys(),)

    global_vmin = torch.min(attention_weights[:num_visualized_images]).item()
    global_vmax = torch.max(attention_weights[:num_visualized_images]).item()
    
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
        for image_idx in range(num_visualized_images):
            image_attention_weights = get_attention_weights_for_image_idx(image_idx)
            
            nrows, ncols = 4, 4
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(s * ncols, s * nrows))
            for head_idx, weights in enumerate(torch.unbind(image_attention_weights, dim=-1)):
                ax: Axes = axs[head_idx // ncols, head_idx % ncols]
                ax.set_title(f"Head {head_idx}", pad=16.0)
                plot_rescaled_attention(fig, ax, weights, image_idx)
            fig.suptitle(f"Layer {layer_idx}, Image {image_idx}: attention_matrix{suffix}")
            plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(s * num_visualized_images, s))
        for image_idx in range(num_visualized_images):
            ax: Axes = axs[image_idx]
            ax.set_title(f"Image {image_idx}", pad=16.0)
            plot_rescaled_attention(fig, axs[image_idx], get_attention_weights_for_image_idx(image_idx), image_idx,)
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


def visualize_incoming_attention_per_image(
    layer_idx: int,
    attention_weights: torch.Tensor,
    exclude_self: bool = True,
    invert: bool = False,
    **kwargs: Any,
) -> None:
    # Construct the rearranged attention weights
    attention_weights = torch.mean(attention_weights, dim=-1)
    if exclude_self:
        attention_weights[:, range(ImageFeatures.N + 1), range(ImageFeatures.N + 1)] = 0.0
    
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
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(s * num_visualized_images, s * 0.75), subplot_kw={"projection": f"{ndim}d"} if ndim != 2 else None)
        for image_idx in range(num_visualized_images):       
            ax: Axes = axs[image_idx]

            reset_seed()
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
    

def visualize_pc_projection_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    modes: List[Tuple[DecompositionOptions, int]] = [("linear", 0)],
    with_hist: bool = False,
    **kwargs: Any,
) -> None:
    image_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,))
    fit_features = image_features

    decompositions = supply_decompositions(set(next(zip(*modes))))
    projections = {}
    for k, decomposition in decompositions.items():
        reset_seed()
        if getattr(decomposition, "is_fitted", False):
            output_features = decomposition.transform(image_features)
        else:
            output_features = decomposition.fit(fit_features).transform(image_features)
        projections[k] = einops.rearrange(
            output_features,
            "(bsz h w) d -> bsz h w d", h=ImageFeatures.H, w=ImageFeatures.W,
        )

    for mode, eig_idx in modes:
        feature_projections = projections[mode][..., eig_idx]
        _visualize_cmap_with_values(feature_projections, f"Layer {layer_idx}: {metric_name}_{mode}{eig_idx}_projection", **kwargs)
        plt.show()
        
        if with_hist:
            plt.rcParams["figure.figsize"] = (3.0, 2.0)
            plt.hist(feature_projections.flatten(), bins=100)
            plt.show()


def compare_pc_projection_across_layers(
    features: ImageFeatures,
    layer_idx1: int,
    layer_idx2: int,
    metric_name: str,
    rgb_assignment: torch.Tensor,       # [bsz x H x W x 3]
    mode: Tuple[DecompositionOptions, int] = ("linear", 0),
    highlight: torch.Tensor = None,
    **kwargs: Any
) -> None:
    
    decomposition = supply_decompositions({mode[0]})
    def compute_output_features(layer_idx: int) -> torch.Tensor:
        image_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,))
        fit_features = image_features
        reset_seed()
        return decomposition.fit(fit_features).transform(image_features)[..., mode[1]]    # [(bsz h w)]
    
    x = compute_output_features(layer_idx1)
    y = compute_output_features(layer_idx2)
    
    ax = plt.gca()
    ax.scatter(
        x.numpy(force=True), y.numpy(force=True),
        color=rgb_assignment.flatten(0, -2).numpy(force=True), s=1, zorder=12, **kwargs,
    )
    
    if highlight is not None:
        image_idx, h_idx, w_idx = torch.unbind(highlight, dim=-1)
        highlight_idx = image_idx * ImageFeatures.N + h_idx * ImageFeatures.W + w_idx
        ax.scatter(
            x[highlight_idx].numpy(force=True), y[highlight_idx].numpy(force=True),
            color=(0.5 * rgb_assignment.flatten(0, -2)[highlight_idx]).numpy(force=True), s=20, zorder=12, marker="*"
        )
    
    ax.set_xlabel(f"layer{layer_idx1}")
    ax.set_ylabel(f"layer{layer_idx2}")
    ax.set_title(f"{metric_name}_{mode[0]}_pc{mode[1]}_projection")
    
    ax_hist = ax.twinx()
    ax_hist.hist(x.numpy(force=True), bins=50, density=True, alpha=0.2)
    ax_hist.set_ylabel("density")
    ax_hist.set_yscale("log")
    plt.show()

def _visualize_cmap_with_values(t: torch.Tensor, title: str, global_cmap: bool = True, **kwargs: Any) -> Tuple[Figure, Any]:
    global_vmin = torch.min(t[:num_visualized_images]).item()
    global_vmax = torch.max(t[:num_visualized_images]).item()
    
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(s * num_visualized_images, s))
    for image_idx, image_norms in enumerate(t[:num_visualized_images]):
        if global_cmap:
            vmin = global_vmin
            vmax = global_vmax
        else:
            vmin = torch.min(image_norms).item()
            vmax = torch.max(image_norms).item()
        
        ax: Axes = axs[image_idx]
        im = ax.imshow(image_norms.numpy(force=True), vmin=vmin, vmax=vmax, **kwargs)
        fig.colorbar(im, cax=make_axes_locatable(axs[image_idx]).append_axes("right", size="5%", pad=0.05), orientation="vertical")
        ax.axis("off")
        ax.set_title(f"Image {image_idx}", pad=4.0)
    
    fig.suptitle(title)
    return fig, axs


def visualize_feature_norms_per_layer(
    features: ImageFeatures,
    metric_name: str,
    mta_dict: Dict[int, torch.Tensor],  # [bsz x H x W]
    rgb_assignment: torch.Tensor,       # [bsz x H x W x 3]
    fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    if fns is None:
        fns = {"norm", lambda t: torch.norm(t, p=2, dim=-1)}

    cls_features = features.get(key=metric_name, include=(ImageFeatures.CLS,))      # [L x bsz x D]
    image_features = features.get(key=metric_name, include=(ImageFeatures.IMAGE,))  # [L x (bsz H W) x D]
    mta_features_dict = {
        k: features.get(key=metric_name, include=(k,))                              # [L x M? x D]
        for k in mta_dict.keys()
    }
    all_mta_mask = torch.any(torch.stack([*mta_dict.values()], dim=0), dim=0)       # [bsz x H x W]

    colors = [*matplotlib.colors.XKCD_COLORS.values()]
    fig, axs = plt.subplots(nrows=1, ncols=len(fns), figsize=(7 * len(fns), 5), sharex=True, sharey=True)
    for i, (fn_name, fn) in enumerate(fns.items()):
        mta_norms_dict = {k: fn(v) for k, v in mta_features_dict.items()}   # [L x M?]
        for k, mta_norms in mta_norms_dict.items():
            start_idx = k + 1
            for token_idx, mta_token in enumerate(torch.unbind(mta_norms, dim=1)):
                # nan_idx = torch.argmax(~torch.isnan(token_feature_norms).to(torch.int))
                # mta_idx = torch.argwhere(einops.rearrange(mta_mask, "b h w -> b (h w)"))[token_idx - min_token_idx - 1][-1]
                # axs[i].plot(
                #     [nan_idx - 1, nan_idx],
                #     [feature_norms[nan_idx - 1, image_idx, mta_idx - min_token_idx], token_feature_norms[nan_idx]],
                #     color="midnightblue", linewidth=0.5, linestyle="--", alpha=0.5,
                # )

                axs[i].plot(
                    torch.arange(len(mta_token))[start_idx:].numpy(force=True),
                    mta_token[start_idx:].numpy(force=True), marker=".",
                    color=colors[k], linewidth=0.5, label=f"{ImageFeatures.process_key(k)}_register_token" if token_idx == 0 else None
                )

        cls_norms = fn(cls_features)                                # [L x bsz]
        for image_idx, cls_token in enumerate(torch.unbind(cls_norms, dim=1)):
            axs[i].plot(
                cls_token.numpy(force=True), marker=".",
                color="black", linewidth=1, zorder=12, label="cls_token" if image_idx == 0 else None
            )

        image_norms = fn(image_features)                            # [L x (bsz H W)]
        for token_idx, (image_token, rgb) in enumerate(zip(
            torch.unbind(image_norms, dim=1),
            torch.unbind(rgb_assignment.flatten(0, -2), dim=0),
        )):
            image_idx = token_idx // ImageFeatures.N
            h_idx = (token_idx % ImageFeatures.N) // ImageFeatures.W
            w_idx = token_idx % ImageFeatures.W

            if all_mta_mask[image_idx, h_idx, w_idx]:
                mta_kwargs = {"linewidth": 1, "linestyle": "-",}
            else:
                mta_kwargs = {"linewidth": 0.5, "linestyle": "-.",}

            axs[i].plot(
                image_token.numpy(force=True), marker=".",
                color=rgb.numpy(force=True), **mta_kwargs,
            )

        axs[i].set_title(f"{metric_name}_{fn_name}")
        axs[i].set_xlabel("layer")
        axs[i].xaxis.grid(True)
        axs[i].set_ylabel(fn_name)
        axs[i].set_yscale("log")
        
        axs[i].legend()
    plt.show()


def visualize_feature_values_by_pca(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    modes: Set[DecompositionOptions],
    mta_dict: Dict[int, torch.Tensor],  # [bsz x H x W]
    rgb_assignment: torch.Tensor,       # [bsz x H x W x 3]
    ndim: int = 2,
    with_cls: bool = True,
    highlight: torch.Tensor = None,
    subsample: float = 1.0,
    **kwargs: Any,
) -> None:
    fit_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,))
    
    decompositions = supply_decompositions(modes)
    for decomposition in decompositions.values():
        if not getattr(decomposition, "is_fitted", False):
            reset_seed()
            decomposition.fit(fit_features)

    ax_names = ("x", "y", "z")
    fig, axs = plt.subplots(nrows=1, ncols=len(decompositions), figsize=(s * len(decompositions), s * 0.75), subplot_kw={"projection": f"{ndim}d"} if ndim != 2 else None)
    for i, (decomposition_name, decomposition) in enumerate(decompositions.items()):        
        ax: Axes = axs if len(decompositions) == 1 else axs[i]

        def compress(_features: torch.Tensor) -> torch.Tensor:
            return decomposition.transform(_features)[..., :ndim]

        image_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,)) # [(bsz H W) x D]
        subsample_mask = torch.rand(image_features.shape[:1]) < subsample
        ax.scatter(
            *compress(image_features[subsample_mask]).mT.numpy(force=True),
            color=rgb_assignment.flatten(0, -2)[subsample_mask].numpy(force=True), s=1, **kwargs
        )

        if with_cls:
            cls_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.CLS,)) # [bsz x D]
            ax.scatter(
                *compress(cls_features).mT.numpy(force=True),
                color="black", label="cls_token",
            )

        # mta_key = min((float("inf"), *filter(lambda l: l >= layer_idx, mta_dict.keys())))
        # if mta_key != float("inf"):
        #     mta_features = features.get(layer_idx=layer_idx, key=metric_name, include=(mta_key,))           # [M? x D]
        #     ax.scatter(
        #         *compress(mta_features).mT.numpy(force=True),
        #         color=rgb_assignment[mta_dict[mta_key]].numpy(force=True), s=10, **kwargs
        #     )

        if highlight is not None:
            image_idx, h_idx, w_idx = torch.unbind(highlight, dim=-1)
            highlight_idx = image_idx * ImageFeatures.N + h_idx * ImageFeatures.W + w_idx
            ax.scatter(
                *compress(image_features[highlight_idx]).mT.numpy(force=True),
                color=rgb_assignment.flatten(0, -2)[highlight_idx].numpy(force=True), s=20, marker="*"
            )

        for j in range(ndim):
            getattr(ax, f"set_{ax_names[j]}label")(f"projection{j}")
        ax.set_title(f"{metric_name}_{decomposition_name}")
        ax.legend()

    plt.show()




