import itertools
from typing import Any, Callable, Dict, List, Literal, Set, Tuple, Union

import einops
import matplotlib.colors
import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensordict import TensorDict

from core.decomposition import generate_NCUT
from infrastructure import utils
from infrastructure.settings import DEVICE, OUTPUT_DEVICE, SEED
from modeling.image_features import ImageFeatures

VISUALIZED_INDICES = [0, 1, 2, 3, 4, 5]    # [45, 46, 47, 48, 49]
NUM_VISUALIZED_IMAGES = len(VISUALIZED_INDICES)
PLOT_SCALE: float = 5.0
CMAPScaleOptions = Literal["linear", "log", "arcsinh"]



def construct_per_layer_output_dict(_per_metric_output_dict: Dict[str, np.ndarray[torch.Tensor]]) -> List[TensorDict]:
    result: List[TensorDict] = [
        TensorDict(dict(zip(_per_metric_output_dict.keys(), (None if _v is None else _v[-1] for _v in v)))).auto_device_().auto_batch_size_(batch_dims=2)
        for v in zip(*_per_metric_output_dict.values())
    ]
    for idx, td in enumerate(result):
        if td._has_non_tensor:
            result[idx] = None
    return result + [None] * (ImageFeatures.NUM_LAYERS - len(result))


def mask_to_highlight(t: torch.Tensor) -> torch.Tensor:
    return torch.argwhere(einops.rearrange(
        t[..., ImageFeatures.image_indices],
        "... (h w) -> ... h w", h=ImageFeatures.H, w=ImageFeatures.W,
    ))


def generate_rgb_from_tsne_3d(
    features: torch.Tensor,     # [N x D]
) -> torch.Tensor:              # [N x 3]
    if True:
        from nystrom_ncut import rgb_from_tsne_3d, rgb_from_euclidean_tsne_3d
        # return rgb_from_tsne_3d(features)
        return rgb_from_euclidean_tsne_3d(features, num_sample=1000)
    else:
        from ncut_pytorch import rgb_from_tsne_3d
        return rgb_from_tsne_3d(features)[1]


def symmetrize(lo: float, hi: float) -> Tuple[float, float]:
    return -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))


def visualize_images_with_mta(
    original_images: torch.Tensor,
    mta_mask: torch.Tensor = None,
) -> None:
    original_images = shift_channels(original_images)
    if mta_mask is not None:
        mta_mask = einops.rearrange(
            mta_mask[:, ImageFeatures.image_indices],
            "b (h w) -> b h w", h=ImageFeatures.H, w=ImageFeatures.W,
        )
    
    fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE))
    for ax_idx, image_idx in enumerate(VISUALIZED_INDICES):
        original_image = original_images[image_idx]
        ax: Axes = axs[ax_idx]
        if mta_mask is not None:
            mask = transforms.Resize(original_image.shape[:2])(mta_mask[None, image_idx].to(dtype=torch.float))[0, ..., None]
            image = (1 - mask) * original_image + mask * torch.tensor((1.0, 0.0, 0.0))
        else:
            image = original_image
        
        ax.imshow(image.numpy(force=True))
        ax.axis("off")
    fig.suptitle("original_image")
    plt.show()
    plt.close()
    

def shift_channels(images_: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(images_, "bsz c h w -> bsz h w c")


def get_rgb_colors(features: ImageFeatures, layer_idx: int, key: str, use_all: bool) -> torch.Tensor:
    utils.reset_seed()

    ncut = generate_NCUT()
    image_features = features.get(layer_idx=layer_idx, key=key, include=(ImageFeatures.IMAGE,)) # [(bsz h w) x D]
    if use_all:
        fit_features = features.get(layer_idx=layer_idx, key=key)                               # [? x D]
        ncut_features = ncut.fit(fit_features).transform(image_features)
    else:
        # fit_features = features.get(layer_idx=layer_idx, key=key, include=(ImageFeatures.IMAGE,), exclude=("MA",))  # [? x D]
        ncut_features = ncut.fit_transform(image_features)
    # print(fit_features.shape, image_features.shape)
    # ncut_features = ncut.fit(fit_features).transform(image_features)                            # [(bsz h w) x d]

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

    fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE))
    for ax_idx, image_features in enumerate(rgb_features[VISUALIZED_INDICES]):
        ax: Axes = axs[ax_idx]
        ax.imshow(image_features.numpy(force=True))
        ax.axis("off")
    fig.suptitle(f"Layer {layer_idx}: {metric_name}")

    def draw_square(ax_idx: int, h_idx: int, w_idx: int, color: str) -> None:
        ax: Axes = axs[ax_idx]
        ax.plot(
            w_idx - 0.5 + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
            h_idx - 0.5 + torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
            color=color, linewidth=4.0,
        )

    if mta_mask is not None:
        for ax_idx, h_idx, w_idx in torch.argwhere(einops.rearrange(
            mta_mask[VISUALIZED_INDICES][:, ImageFeatures.image_indices],
            "b (h w) -> b h w", h=ImageFeatures.H, w=ImageFeatures.W,
        )):
            draw_square(ax_idx, h_idx, w_idx, "black")

    if highlight is not None:
        if highlight.dtype == torch.bool:
            highlight = mask_to_highlight(highlight)
        
        for image_idx, h_idx, w_idx in highlight:
            if image_idx in VISUALIZED_INDICES:
                draw_square(VISUALIZED_INDICES.index(image_idx), h_idx, w_idx, "white")
    plt.show()
    plt.close()

    
def visualize_feature_norms_per_image(
    features: Union[torch.Tensor, ImageFeatures],
    layer_idx: int = None,
    metric_name: str = None,
    title: str = None,
    p: float = 2.0,
    **kwargs: Any
) -> torch.Tensor:
    if not torch.is_tensor(features):
        features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.CLS, ImageFeatures.IMAGE,), require_valid=False, with_batch=True)
        title = f"Layer {layer_idx}: {metric_name}_norm"
        
    feature_norms = einops.rearrange(
        torch.norm(features[:, ImageFeatures.image_indices], p=p, dim=-1),
        "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    
    _visualize_cmap_with_values(feature_norms, title, **kwargs)
    plt.show()
    plt.close()
    return feature_norms


def visualize_feature_norms_per_image_with_tensor(
    features: torch.Tensor,
    layer_idx: int,
    metric_name: str,
    p: float = 2.0,
    **kwargs: Any
) -> torch.Tensor:
    feature_norms = einops.rearrange(
        torch.norm(features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,), require_valid=False), p=p, dim=-1),
        "(bsz h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    
    _visualize_cmap_with_values(feature_norms, f"Layer {layer_idx}: {metric_name}_norm", **kwargs)
    plt.show()
    plt.close()
    return feature_norms


def visualize_feature_dot_products_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name1: str,
    metric_name2: str,
    normalize: bool = True,
    **kwargs: Any
) -> torch.Tensor:
    feature1 = features.get(layer_idx=layer_idx, key=metric_name1, include=(ImageFeatures.IMAGE,), require_valid=False, with_batch=True)
    feature2 = features.get(layer_idx=layer_idx, key=metric_name2, include=(ImageFeatures.IMAGE,), require_valid=False, with_batch=True)
    if normalize:
        feature1 = Fn.normalize(feature1, p=2, dim=-1)
        feature2 = Fn.normalize(feature2, p=2, dim=-1)
    
    dot_product = einops.rearrange(
        torch.sum(feature1 * feature2, dim=-1),
        "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    
    _visualize_cmap_with_values(
        dot_product, f"Layer {layer_idx}: ({metric_name1}, {metric_name2})_dot_product",
        symmetric_cmap=True, cmap="bwr", **kwargs
    )
    plt.show()
    plt.close()
    return dot_product


def _visualize_cmap_with_values(
    t: torch.Tensor,
    title: str,
    symmetric_cmap: bool = False,
    global_cmap: bool = True,
    cmap_scale: CMAPScaleOptions = "linear",
    write_values: bool = False,
    **kwargs: Any
) -> Tuple[Figure, np.ndarray[Axes]]:
    scale_dict: Dict[str, Callable[[float, float], str]] = {
        "linear": "Normalize",
        "log": "LogNorm",
        "arcsinh": "AsinhNorm",
    }
    norm = getattr(matplotlib.colors, scale_dict[cmap_scale])
    
    global_vmin = torch.min(t[VISUALIZED_INDICES]).item()
    global_vmax = torch.max(t[VISUALIZED_INDICES]).item()
    
    fig, axs = plt.subplots(nrows=1, ncols=NUM_VISUALIZED_IMAGES, figsize=(PLOT_SCALE * NUM_VISUALIZED_IMAGES, PLOT_SCALE))
    for ax_idx, image_idx in enumerate(VISUALIZED_INDICES):
        image_features = t[image_idx]
        vmin, vmax = (global_vmin, global_vmax) if global_cmap else (
            torch.min(image_features).item(),
            torch.max(image_features).item(),
        )
        if symmetric_cmap:
            vmin, vmax = symmetrize(vmin, vmax)
        
        ax: Axes = axs[ax_idx]
        im = ax.imshow(image_features.numpy(force=True), norm=norm(vmin=vmin, vmax=vmax), **kwargs)
        if write_values:
            for i, j in itertools.product(range(ImageFeatures.H), range(ImageFeatures.W)):
                ax.text(x=j, y=i, s=f"{image_features[i, j]:.1f}", fontdict={"fontsize": 6, "ha": "center", "va": "center"})
        
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05), orientation="vertical")
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
    plt.close()






