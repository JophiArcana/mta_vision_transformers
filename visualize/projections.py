from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch.utils._pytree import tree_flatten

from core.decomposition import DecompositionOptions, supply_decompositions
from core.qk import qk_intersection, qk_projection_variance
from infrastructure import utils
from infrastructure.settings import DEVICE
from modeling.image_features import ImageFeatures
from visualize.base import (
    PLOT_SCALE,
    mask_to_highlight,
    _visualize_cmap_with_values,
)



def visualize_pc_projection_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    modes: List[Tuple[DecompositionOptions, int]] = [("linear", 0)],
    **kwargs: Any,
) -> None:
    image_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,))
    fit_features = image_features

    decompositions = supply_decompositions(set(next(zip(*modes))))
    projections = {}
    for k, decomposition in decompositions.items():
        utils.reset_seed()
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
        plt.close()


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
        utils.reset_seed()
        return decomposition.fit(fit_features).transform(image_features)[..., mode[1]]    # [(bsz h w)]
    
    x = compute_output_features(layer_idx1)
    y = compute_output_features(layer_idx2)
    
    ax = plt.gca()
    ax.scatter(
        x.numpy(force=True), y.numpy(force=True),
        color=rgb_assignment.flatten(0, -2).numpy(force=True), s=1, zorder=12, **kwargs,
    )
    
    if highlight is not None:
        if highlight.dtype == torch.bool:
            highlight = mask_to_highlight(highlight)
        
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
    plt.close()



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
    plt.close()


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


def visualize_feature_values_by_pca(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    modes: Set[DecompositionOptions],
    mta_mask: torch.Tensor,             # [bsz x (N + 1)]
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
            utils.reset_seed()
            decomposition.fit(fit_features)

    ax_names = ("x", "y", "z")
    fig, axs = plt.subplots(nrows=1, ncols=len(decompositions), figsize=(PLOT_SCALE * len(decompositions), PLOT_SCALE * 0.75), subplot_kw={"projection": f"{ndim}d"} if ndim != 2 else None)
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

        # SECTION: Plot CLS tokens
        if with_cls:
            cls_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.CLS,)) # [bsz x D]
            ax.scatter(
                *compress(cls_features).mT.numpy(force=True),
                color="black", label="cls_token",
            )

        # SECTION: Plot AS tokens
        if mta_mask is not None:
            flattened_mta_mask = mta_mask[:, ImageFeatures.image_indices].flatten()
            binary_mask = flattened_mta_mask > 0
            
            lo, hi, decay = 10, 1000, 0.7
            if mta_mask.dtype == torch.bool:
                size = lo
            elif mta_mask.dtype in [torch.int, torch.long]:
                rank = flattened_mta_mask[binary_mask]
                size = lo + (hi - lo) * (decay ** (rank - 1))
            else:
                raise ValueError(mta_mask.dtype)
        
            ax.scatter(
                *compress(image_features[binary_mask]).mT.numpy(force=True),
                color=rgb_assignment.flatten(0, -2)[binary_mask].numpy(force=True), edgecolors="black", s=size, **kwargs
            )
         
        # SECTION: Plot highlighted tokens
        if highlight is not None:
            if highlight.dtype == torch.bool:
                highlight = mask_to_highlight(highlight)
            
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
    plt.close()
