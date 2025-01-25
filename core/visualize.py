from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import matplotlib.colors
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.base import BaseEstimator
from torch_pca import PCA

from core.qk import qk_intersection, qk_projection_variance
from infrastructure.settings import DEVICE, OUTPUT_DEVICE, SEED
from model.image_features import ImageFeatures

num_visualized_images = 6
DEFAULT_LAYER_INDICES = [*range(9, 12)]

new = True
n_components = 100


def reset_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


class ComposeDecomposition(BaseEstimator):
    def __init__(self, decompositions: List[BaseEstimator]):
        self.decompositions = decompositions

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        for decomposition in self.decompositions:
            X = decomposition.fit_transform(X)
        return X

    def fit(self, X: torch.Tensor) -> Any:
        self.fit_transform(X)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        for decomposition in self.decompositions:
            X = decomposition.transform(X)
        return X

    
def generate_NCUT():
    num_sample = 20000
    if new:
        from nystrom_ncut import NCut, SampleConfig
        return NCut(
            n_components=n_components,
            sample_config=SampleConfig(
                method="fps",
                # method="fps_recursive",
                num_sample=num_sample,
                fps_dim=12,
                # n_iter=1,
            ),
            distance="rbf",
            eig_solver="svd_lowrank"
        )
    else:
        from ncut_pytorch import NCUT
        return NCUT(num_eig=n_components, num_sample=num_sample, distance="rbf")
    
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


def massive_token_heuristic(layer_idx: int, per_metric_output_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    log_norms = einops.rearrange(
        torch.norm(per_metric_output_dict["layer_output"][15][0][:, ImageFeatures.image_indices, :], p=2, dim=-1).log(),
        "bsz (h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    flattened_norms = torch.sort(torch.flatten(log_norms), dim=0).values
    cutoff = torch.argmax(torch.diff(flattened_norms, dim=0), dim=0)
    mask = log_norms > flattened_norms[cutoff]
    return mask


def get_rgb_colors(features: ImageFeatures, layer_idx: int, key: str, use_all: bool) -> torch.Tensor:
    reset_seed()

    ncut = generate_NCUT()
    image_features = features.get(layer_idx=layer_idx, key=key, include=(ImageFeatures.IMAGE,)).to(DEVICE)  # [(bsz h w) x D]
    if use_all:
        fit_features = features.get(layer_idx=layer_idx, key=key).to(DEVICE)                                # [? x D]
        ncut_features = ncut.fit(fit_features).transform(image_features)                                    # [(bsz h w) x d]
    else:
        ncut_features = ncut.fit_transform(image_features)                                                  # [(bsz h w) x d]

    rgb_colors = einops.rearrange(
        generate_rgb_from_tsne_3d(ncut_features),
        "(bsz h w) c -> bsz h w c", h=ImageFeatures.H, w=ImageFeatures.W,
    ).to(OUTPUT_DEVICE)
    return rgb_colors


def visualize_features_per_image(
    features: ImageFeatures,
    layer_idx: int,
    mta_mask: torch.Tensor = None,
    use_all: bool = False,
    highlight: torch.Tensor = None,
    include: Set[str] = None,
) -> None:
    if include is None:
        include = features.features.keys(include_nested=True, leaves_only=True)

    for metric_name in include:
        rgb_features = get_rgb_colors(features, layer_idx=layer_idx, key=metric_name, use_all=use_all)   # [B x H x W x 3]

        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
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
            for image_idx, h_idx, w_idx in torch.argwhere(mta_mask[:num_visualized_images]):
                draw_square(image_idx, h_idx, w_idx, "black")
        if highlight is not None:
            for image_idx, h_idx, w_idx in highlight:
                draw_square(image_idx, h_idx, w_idx, "white")
        plt.show()


def visualize_qk_projection_per_image(
    features: ImageFeatures,
    model_dict: List[Dict[str, torch.Tensor]],
    layer_idx: int,
    bias: bool = False,
    p: float = 2.0,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = None,
    aggregate_name: str = "",
) -> None:
    if aggregate_func is None:
        aggregate_func = lambda t: torch.mean(t, dim=-2)

    layer_output = features.get(layer_idx=layer_idx, key="attention_input", include=(ImageFeatures.IMAGE,)).to(DEVICE)  # [(bsz h w) x d]

    QKVw = model_dict[layer_idx]["QKVw"].to(DEVICE)
    QKVb = model_dict[layer_idx]["QKVb"].to(DEVICE)
    
    head_dim = 64
    D = layer_output.shape[-1]
    Qw = QKVw[:D].reshape(-1, head_dim, D)
    Kw = QKVw[D:2 * D].reshape(-1, head_dim, D)
    
    Qb = QKVb[:D].reshape(-1, head_dim)
    Kb = QKVb[D:2 * D].reshape(-1, head_dim)
    
    if bias:
        qk = qk_intersection(Qw, Kw, Qb, Kb)
    else:
        qk = qk_intersection(Qw, Kw)
    
    projection_variance = einops.rearrange(
        aggregate_func(qk_projection_variance(layer_output, qk, p)),
        "(bsz h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    _visualize_cmap_with_values(projection_variance, f"qk_projection_{aggregate_name}variance", cmap="gray")
    

def visualize_feature_norms_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    **kwargs: Any
) -> torch.Tensor:
    feature_norms = einops.rearrange(
        torch.norm(features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,)).to(DEVICE), p=2, dim=-1),
        "(bsz h w) -> bsz h w", h=ImageFeatures.H, w=ImageFeatures.W,
    )
    _visualize_cmap_with_values(feature_norms, f"{metric_name}_norm", **kwargs)
    return feature_norms


def visualize_pc_projection_per_image(
    features: ImageFeatures,
    layer_idx: int,
    metric_name: str,
    modes: List[Tuple[Literal["linear", "ncut", "ncut_pca"], int]] = [("linear", 0)],
    with_hist: bool = False,
    **kwargs: Any
) -> None:
    image_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,)).to(DEVICE)
    fit_features = image_features

    decompositions = {
        "linear": PCA(n_components=n_components),
        "ncut": generate_NCUT(),
    }

    projections = {
        k: einops.rearrange(
            decomposition.fit(fit_features).transform(image_features),
            "(bsz h w) d -> bsz h w d", h=ImageFeatures.H, w=ImageFeatures.W,
        ) for k, decomposition in decompositions.items()
    }
    # ncut_V = torch.linalg.svd(ncut_feature_projections.flatten(0, -2), full_matrices=False)[-1].mT
    # ncut_pca_feature_projections = ncut_feature_projections @ ncut_V

    def plot_pc(mode: str, eig_idx: int) -> None:
        feature_projections = projections[mode][..., eig_idx]
        _visualize_cmap_with_values(feature_projections, f"{metric_name}_{mode}_pc{eig_idx}_projection", **kwargs)
        
        if with_hist:
            plt.rcParams["figure.figsize"] = (3.0, 2.0)
            plt.hist(feature_projections.flatten(), bins=100)
            plt.show()
    
    for mode, eig_idx in modes:
        plot_pc(mode, eig_idx)


def _visualize_cmap_with_values(t: torch.Tensor, title: str, **kwargs: Any) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
    
    vmin = torch.min(t[:num_visualized_images]).item()
    vmax = torch.max(t[:num_visualized_images]).item()
    for image_idx, image_norms in enumerate(t[:num_visualized_images]):
        im = axs[image_idx].imshow(image_norms.numpy(force=True), vmin=vmin, vmax=vmax, **kwargs)
        fig.colorbar(im, cax=make_axes_locatable(axs[image_idx]).append_axes("right", size="5%", pad=0.05), orientation="vertical")
        axs[image_idx].axis("off")
    
    fig.suptitle(title)
    plt.show()


def visualize_feature_norms_per_layer(
    features: ImageFeatures,
    metric_name: str,
    mta_dict: Dict[int, torch.Tensor],  # [bsz x H x W]
    rgb_assignment: torch.Tensor,       # [bsz x H x W x 3]
    fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    if fns is None:
        fns = {"norm", lambda t: torch.norm(t, p=2, dim=-1)}

    cls_features = features.get(key=metric_name, include=(ImageFeatures.CLS,)).to(DEVICE)       # [L x bsz x D]
    image_features = features.get(key=metric_name, include=(ImageFeatures.IMAGE,)).to(DEVICE)   # [L x (bsz H W) x D]
    mta_features_dict = {
        k: features.get(key=metric_name, include=(k,)).to(DEVICE)                               # [L x M? x D]
        for k in mta_dict.keys()
    }
    all_mta_mask = torch.any(torch.stack([*mta_dict.values()], dim=0), dim=0)                   # [bsz x H x W]

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


# TODO: asdf
def visualize_feature_values_by_pca(
    features: ImageFeatures,
    layer_idx: int,
    mta_dict: Dict[int, torch.Tensor],  # [bsz x H x W]
    rgb_assignment: torch.Tensor,       # [bsz x H x W x 3]
    include: Set[str] = None,
    ndim: int = 2,
    with_cls: bool = True,
    highlight: torch.Tensor = None,
    subsample: float = 0.5,
    **kwargs: Any,
) -> None:
    if include is None:
        include = features.features.keys(include_nested=True, leaves_only=True)

    decompositions = {
        "linear": PCA(n_components=n_components),
        "ncut": generate_NCUT(),
        "ncut_pca": ComposeDecomposition([
            generate_NCUT(),
            PCA(n_components=n_components),
        ])
    }

    fig, axs = plt.subplots(nrows=1, ncols=3 * len(include), figsize=(3 * 4 * len(include), 3), subplot_kw={"projection": f"{ndim}d"} if ndim != 2 else None)
    for i, metric_name in enumerate(include):

        fit_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.CLS, ImageFeatures.IMAGE)).to(DEVICE)
        for decomposition in decompositions.values():
            reset_seed()
            decomposition.fit(fit_features)

        ax_names = ("x", "y", "z")
        for j, (decomposition_name, decomposition) in enumerate(decompositions.items()):
            ax = axs[3 * i + j]

            def compress(_features: torch.Tensor) -> torch.Tensor:
                return decomposition.transform(_features)[..., :ndim]

            image_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.IMAGE,)).to(DEVICE)  # [(bsz H W) x D]
            subsample_mask = torch.rand(image_features.shape[:1]) < subsample
            ax.scatter(
                *compress(image_features[subsample_mask]).mT.numpy(force=True),
                color=rgb_assignment.flatten(0, -2)[subsample_mask].numpy(force=True), s=1, **kwargs
            )

            if with_cls:
                cls_features = features.get(layer_idx=layer_idx, key=metric_name, include=(ImageFeatures.CLS,)).to(DEVICE)  # [bsz x D]
                ax.scatter(
                    *compress(cls_features).mT.numpy(force=True),
                    color="black", label="cls_token",
                )

            mta_key = min((float("inf"), *filter(lambda l: l >= layer_idx, mta_dict.keys())))
            if mta_key != float("inf"):
                mta_features = features.get(layer_idx=layer_idx, key=metric_name, include=(mta_key,)).to(DEVICE)            # [M? x D]
                ax.scatter(
                    *compress(mta_features).mT.numpy(force=True),
                    color=rgb_assignment[mta_dict[mta_key]].numpy(force=True), s=10, **kwargs
                )

            if highlight is not None:
                image_idx, h_idx, w_idx = torch.unbind(highlight, dim=-1)
                highlight_idx = image_idx * ImageFeatures.N + h_idx * ImageFeatures.W + w_idx
                ax.scatter(
                    *compress(image_features[highlight_idx]).mT.numpy(force=True),
                    color=rgb_assignment.flatten(0, -2)[highlight_idx].numpy(force=True), s=400, marker="*"
                )

            for k in range(ndim):
                getattr(ax, f"set_{ax_names[k]}label")(f"projection_direction{k}")
            ax.set_title(f"{metric_name}_{decomposition_name}")
            ax.legend()

    plt.show()




