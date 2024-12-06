import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import matplotlib.colors
import numpy as np
import scipy.spatial
import torch
import torch.nn.functional as Fn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from sklearn.manifold import TSNE
from torch.utils._pytree import tree_flatten

from infrastructure.settings import DEVICE, OUTPUT_DEVICE, SEED
from infrastructure import utils
from core.monitor import Monitor
from core.qk import qk_intersection, qk_projection_variance


H = W = 16
N = H * W
num_visualized_images = 8
DEFAULT_LAYER_INDICES = [*range(9, 12)]


def get_image_tokens(x_: torch.Tensor) -> torch.Tensor:
    return x_[..., 1:N + 1, :]


def pad_tensor_outputs(ts: List[torch.Tensor], dim: int) -> List[torch.Tensor]:
    result, d = [], max(t.shape[dim] for t in ts)
    for t in ts:
        t = t.to()
        l = t.shape[dim]
        pad_length = d - l
        pad = torch.full((*t.shape[:dim], pad_length, *t.shape[dim + 1:]), torch.nan)
        result.append(torch.cat((t, pad), dim=dim))
    return result

    
def visualize_images_with_mta(original_images: torch.Tensor, mta_mask: torch.Tensor) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(2 * num_visualized_images, 2))
    for image_idx, original_image in enumerate(shift_channels(original_images[:num_visualized_images])):
        mask = transforms.Resize(original_image.shape[:2])(mta_mask[None, image_idx].to(dtype=torch.float))[0, ..., None]
        image = (1 - mask) * original_image + mask * torch.tensor((1.0, 0.0, 0.0))
        
        axs[image_idx].imshow(image.numpy(force=True))
        axs[image_idx].axis("off")
    fig.suptitle("original_image")
    plt.show()


def shift_channels(images_: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(images_, "bsz c h w -> bsz h w c")


def massive_token_heuristic(stacked_layer_output_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    log_norms = einops.rearrange(
        torch.norm(get_image_tokens(stacked_layer_output_dict["layer_output"][15]), p=2, dim=-1).log(),
        "bsz (h w) -> bsz h w", h=H, w=W
    )
    flattened_norms = torch.sort(torch.flatten(log_norms), dim=0).values
    cutoff = torch.argmax(torch.diff(flattened_norms, dim=0), dim=0)
    mask = log_norms > flattened_norms[cutoff]
    return mask


def get_rgb_colors(t: torch.Tensor, mta_mask: torch.Tensor, use_all: bool) -> torch.Tensor:
    if mta_mask is not None:
        r = torch.tensor(matplotlib.colors.to_rgb("red"))
        b = torch.tensor(matplotlib.colors.to_rgb("blue"))
        return torch.where(mta_mask[..., None], r, b)
    else:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        bsz, n, _ = t.shape
        ncut = NCUT(num_eig=100, distance="rbf", indirect_connection=False, device=DEVICE)
        if use_all:
            mask = ~torch.any(torch.isnan(t), dim=-1)
        else:
            mask = (torch.arange(bsz)[:, None], (1 <= torch.arange(n)) & (torch.arange(n) < N + 1))
        
        ncut_features, _ = ncut.fit_transform(t[mask])
        result = torch.full((bsz, n, 3), torch.nan)
        result[mask] = rgb_from_tsne_3d(ncut_features)[1]
            
        return einops.rearrange(
            get_image_tokens(result),
            "bsz (h w) c -> bsz h w c", h=H, w=W
        ).to(OUTPUT_DEVICE)


def visualize_features_per_image(
    output_dict: Dict[str, torch.Tensor],
    mta_mask: torch.Tensor = None,
    use_all: bool = False,
    highlight: torch.Tensor = None,
    include: Set[str] = None,
) -> None:
    for metric_name, t in output_dict.items():
        if include and metric_name not in include:
            continue

        rgb_features = get_rgb_colors(t, None, use_all)
        
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
        for image_idx, image_features in enumerate(rgb_features[:num_visualized_images]):
            axs[image_idx].imshow(image_features.numpy(force=True))
            axs[image_idx].axis("off")
        fig.suptitle(metric_name)
        
        if mta_mask is not None:
            for image_idx, h_idx, w_idx in torch.argwhere(mta_mask[:num_visualized_images]):
                axs[image_idx].plot(
                    w_idx - 0.5 + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
                    h_idx - 0.5 + torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
                    color="black", linewidth=4.0,
                )
                
        if highlight is not None:
            for image_idx, h_idx, w_idx in highlight:
                axs[image_idx].plot(
                    w_idx - 0.5 + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
                    h_idx - 0.5 + torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
                    color="white", linewidth=4.0,
                )
        
        plt.show()


def visualize_qk_projection_per_image(
    output_dict: Dict[str, torch.Tensor],
    bias: bool = False,
    p: float = 2.0,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = None,
    aggregate_name: str = "",
) -> None:
    if aggregate_func is None:
        aggregate_func = lambda t: torch.mean(t, dim=-2)
    
    layer_output = get_image_tokens(output_dict["attention_input"])
    QKVw = output_dict["QKVw"]
    QKVb = output_dict["QKVb"]
    
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
        "bsz (h w) -> bsz h w", h=H, w=W
    )
    _visualize_cmap_with_values(projection_variance, f"qk_projection_{aggregate_name}variance", cmap="gray")
    

def visualize_feature_norms_per_image(metric_name: str, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    feature_norms = einops.rearrange(
        torch.norm(get_image_tokens(t), p=2, dim=-1),
        "bsz (h w) -> bsz h w", h=H, w=W
    )
    _visualize_cmap_with_values(feature_norms, f"{metric_name}_norm", **kwargs)
    return feature_norms


def visualize_pc_projection_per_image(
    metric_name: str,
    t: torch.Tensor,
    modes: List[Tuple[Literal["linear", "ncut"], int]] = [("linear", 0)],
    with_hist: bool = False,
    **kwargs: Any
) -> None:
    original_features = get_image_tokens(t)
    
    num_eig = 100
    ncut = NCUT(num_eig=num_eig, distance="rbf", indirect_connection=False, device=OUTPUT_DEVICE)
    ncut_features = ncut.fit_transform(original_features.flatten(0, -2))[0].unflatten(0, original_features.shape[:-1])
    
    feature_dict = {
        "linear": original_features,
        "ncut": ncut_features,
    }
    
    def plot_pc(mode: str, eig_idx: int) -> None:
        features = feature_dict[mode]
        feature_mean = torch.mean(features.flatten(0, -2), dim=0)
        demeaned_features = features - feature_mean
        V = torch.linalg.svd(demeaned_features.flatten(0, -2), full_matrices=False)[-1].mT[:, eig_idx]
        feature_projections = einops.rearrange(
            demeaned_features @ V,
            "bsz (h w) -> bsz h w", h=H, w=W
        )
        _visualize_cmap_with_values(feature_projections, f"{metric_name}_{mode}_pc{eig_idx}_projection", **kwargs)
        
        # # cts = [1.6, 1.7, 1.8, 1.9, 2.0]
        # cts = [*-torch.arange(1.2, 2.0, 0.1)]
        # print([torch.sum(feature_projections < ct).item() for ct in cts])
        # for ct in cts:
        #     _visualize_cmap_with_values(feature_projections < ct, f"{metric_name}_pc_projection_binary", **kwargs)
        
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
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    rgb_assignment: torch.Tensor,
    fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    bsz = t.shape[0]
    if fns is None:
        fns = {"norm", lambda t: torch.norm(t, p=2, dim=-1)}
        
    fig, axs = plt.subplots(nrows=1, ncols=len(fns), figsize=(7 * len(fns), 5), sharex=True, sharey=True)
    for i, (fn_name, fn) in enumerate(fns.items()):
        feature_norms = fn(t)
        
        for image_idx, token_idx in itertools.product(range(bsz), range(feature_norms.shape[-1])):
            token_feature_norms = feature_norms[:, image_idx, token_idx]
            if token_idx == 0:
                axs[i].plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color="black", linewidth=1, zorder=12, label="cls_token" if image_idx == 0 else None
                )
            elif token_idx > N:
                # nan_idx = torch.argmax(~torch.isnan(token_feature_norms).to(torch.int))
                # mta_idx = torch.argwhere(einops.rearrange(mta_mask, "b h w -> b (h w)"))[token_idx - min_token_idx - 1][-1]
                # axs[i].plot(
                #     [nan_idx - 1, nan_idx],
                #     [feature_norms[nan_idx - 1, image_idx, mta_idx - min_token_idx], token_feature_norms[nan_idx]],
                #     color="midnightblue", linewidth=0.5, linestyle="--", alpha=0.5,
                # )
                
                axs[i].plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color="gold", linewidth=0.5, label="register_token" if token_idx == N + 1 and image_idx == 0 else None
                )
            else:
                patch_idx = token_idx - 1
                h_idx, w_idx = patch_idx // W, patch_idx % W
                
                if mta_mask[image_idx, h_idx, w_idx]:
                    mta_kwargs = {"linewidth": 1, "linestyle": "-",}
                else:
                    mta_kwargs = {"linewidth": 0.5, "linestyle": "-.",}
                    
                axs[i].plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color=rgb_assignment[image_idx, h_idx, w_idx].numpy(force=True), **mta_kwargs,
                )
    
        axs[i].set_title(f"{metric_name}_{fn_name}")
        axs[i].set_xlabel("layer")
        axs[i].xaxis.grid(True)
        axs[i].set_ylabel(fn_name)
        axs[i].set_yscale("log")
        
        axs[i].legend()
    plt.show()


def visualize_feature_values_by_pca(
    output_dict: Dict[str, torch.Tensor],
    mta_mask: torch.Tensor,
    rgb_assignment: torch.Tensor,
    mean: Literal[None, "local", "global"],
    ndim: int = 2,
    with_ma: bool = True,
    highlight: torch.Tensor = None,
    convex_hull: bool = True,
    subsample: float = 0.5,
    include: Set[str] = None,
    **kwargs: Any,
) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    if include is None:
        include = output_dict.keys()
        
    fig, axs = plt.subplots(nrows=1, ncols=2 * len(include), figsize=(4 * len(include), 2), subplot_kw={"projection": f"{ndim}d"} if ndim != 2 else None)
    for i, metric_name in enumerate(include):
        original_feature_values = output_dict[metric_name][:, :N + 1]
        
        # if feature_values.shape[-2] == N + 1 or torch.any(torch.isnan(feature_values)):
        #     mask = torch.full(feature_values.shape[:2], False)
        #     mask[:, 1:N + 1] = mta_mask.flatten(1, 2)
        # else:
        #     mask = torch.full(feature_values.shape[:-1], False)
        #     mask[:, N + 1:] = True
        # valid_mask = ~torch.any(torch.isnan(feature_values), dim=-1)
        # inverse_mask = valid_mask & ~mask
        
        mask = Fn.pad(mta_mask.flatten(1, 2), (1, 0), mode="constant", value=False)
        
        num_eig = 100
        ncut = NCUT(num_eig=num_eig, distance="rbf", indirect_connection=False, device=OUTPUT_DEVICE)
        ncut_feature_values = ncut.fit_transform(original_feature_values.flatten(0, -2))[0].unflatten(0, original_feature_values.shape[:-1])
        
        def plot_pca(ax, feature_values: torch.Tensor, title: str) -> None:
            bsz, _, D = feature_values.shape
            
            def svd(t: torch.Tensor, center: bool) -> torch.Tensor:
                mean = torch.mean(t.flatten(0, -2), dim=0) if center else 0.0
                return torch.linalg.svd(t - mean, full_matrices=False)
            
            # global_feature_mean = 0 if mean is None else torch.mean(feature_values[inverse_mask], dim=0)
            global_feature_mean = 0 if mean is None else torch.mean(feature_values.flatten(0, -2), dim=0)
            demeaned_feature_values = feature_values - global_feature_mean

            ax_names = ("x", "y", "z")
            # if with_ma:
            if False:
                V_mta = svd(demeaned_feature_values[mask], center=(mean == "local"))[-1].mT[:, :1]
                proj = torch.eye(D) - V_mta @ torch.linalg.pinv(V_mta)
                V_nonmta = svd(demeaned_feature_values[~mask] @ proj, center=(mean == "local"))[-1].mT[:, :ndim - 1]
                V = torch.cat((V_mta, V_nonmta), dim=-1)
                getattr(ax, f"set_{ax_names[0]}label")("ma_direction")
                for i in range(ndim - 1):
                    getattr(ax, f"set_{ax_names[i + 1]}label")(f"non_ma_direction{i}")
            else:
                V = svd(get_image_tokens(demeaned_feature_values).flatten(0, -2), center=(mean == "local"))[-1].mT[:, :ndim]
                for i in range(ndim):
                    getattr(ax, f"set_{ax_names[i]}label")(f"non_ma_direction{i}")
                    
            compressed_features = demeaned_feature_values @ V
            
            def to_rgb_mask(m: torch.Tensor) -> torch.Tensor:
                return m[:, 1:N + 1].view(bsz, H, W)
            
            subsample_mask = ~mask & (1 <= torch.arange(N + 1)) & (torch.rand(mask.shape) < subsample)
            if with_ma:
                ax.scatter(*compressed_features[mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(mask)].numpy(force=True), s=10, **kwargs)
            ax.scatter(*compressed_features[subsample_mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(subsample_mask)].numpy(force=True), s=1, **kwargs)
            # ax.scatter(*compressed_features[:, 0].mT.numpy(force=True), s=30, color="black", label="cls_token")
            
            if highlight is not None:
                image_idx, h_idx, w_idx = torch.unbind(highlight, dim=-1)
                highlight_mask = torch.full_like(mask, False)
                highlight_mask[image_idx, h_idx * W + w_idx + 1] = True

                ax.scatter(
                    *compressed_features[highlight_mask].mT.numpy(force=True),
                    color=rgb_assignment[to_rgb_mask(highlight_mask)].numpy(force=True),
                    s=400, marker="*"
                )
            
            if convex_hull:
                hull_kwargs = {"linewidth": 0.3, "edgecolor": "black", "alpha": 1.0}
                
                vertices = compressed_features[mask]
                hull = scipy.spatial.ConvexHull(vertices.numpy(force=True))
                ax.plot_trisurf(*vertices.mT.numpy(force=True), triangles=hull.simplices, cmap="pink", **hull_kwargs)
                
                cls_vertices = compressed_features[:, 0]
                cls_hull = scipy.spatial.ConvexHull(cls_vertices.numpy(force=True))
                ax.plot_trisurf(*cls_vertices.mT.numpy(force=True), triangles=cls_hull.simplices, cmap="bone", **hull_kwargs)
            
            ax.set_title(title)
            ax.legend()
        
        plot_pca(axs[2 * i], original_feature_values, f"{metric_name}_pca_values")
        plot_pca(axs[2 * i + 1], ncut_feature_values, f"{metric_name}_ncut_pca_values")

    plt.show()


def visualize_fc_weights(
    t: torch.Tensor,
    weights: torch.Tensor,
    mta_mask: torch.Tensor,
    mean: Literal[None, "local", "global"],
    layer_indices: List[int] = None,
) -> None:
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(7 * len(layer_indices), 5))
    for i, layer_idx in enumerate(layer_indices):
        feature_values = einops.rearrange(t[layer_idx, :, -H * W:], "bsz (h w) c -> bsz h w c", h=H, w=W)
        global_feature_mean = 0 if mean is None else torch.mean(feature_values.flatten(0, -2), dim=0)
        demeaned_feature_values = feature_values - global_feature_mean
        
        V_ma = torch.pca_lowrank(demeaned_feature_values[mta_mask], q=1, center=(mean == "local"))[-1][:, 0]
        V_nonma = torch.pca_lowrank(demeaned_feature_values[~mta_mask], q=1, center=(mean == "local"))[-1][:, 0]
        
        # V_ma = Fn.one_hot(torch.argmax(feature_values[mta_mask].norm(dim=0)), num_classes=feature_values.shape[-1]).to(torch.float)
        # V_nonma = Fn.one_hot(torch.argmax(feature_values[~mta_mask].norm(dim=0)), num_classes=feature_values.shape[-1]).to(torch.float)
        
        print(((demeaned_feature_values[mta_mask] @ V_ma) ** 2).mean(dim=0), ((demeaned_feature_values[~mta_mask] @ V_nonma) ** 2).mean(dim=0))
        
        layer_weights = weights[layer_idx]
        ma_projection, nonma_projection = V_ma @ layer_weights, V_nonma @ layer_weights
        
        k = 5
        print(f"Layer {layer_idx}:")
        print(f"\tMA direction projection:     {torch.topk(ma_projection, k=k).values.tolist()}")
        print(f"\tnon-MA direction projection: {torch.topk(nonma_projection, k=k).values.tolist()}")
        
        axs[i].hist(ma_projection, bins=100, label="mt_projection")
        axs[i].hist(nonma_projection, bins=100, label="non_mt_projection")
        
        axs[i].set_title(f"Layer {layer_idx}: mt_direction_projection")
        axs[i].set_xlabel("pca_projection")
        axs[i].set_ylabel("count")
        
        # mt_channel_norms = feature_values[mta_mask].mean(dim=0)
        # non_mt_channel_norms = feature_values[~mta_mask].mean(dim=0)
        # weight_channel_norms = (layer_weights ** 2).sum(dim=1) ** 0.5
    
        # m1, v1 = feature_values[mta_mask].mean(dim=0), feature_values[mta_mask].var(dim=0)
        # m2, v2 = feature_values[~mta_mask].mean(dim=0), feature_values[~mta_mask].var(dim=0)
        # sym_kldiv = -1 + (v1 ** 2 + v2 ** 2 + (v1 + v2) * (m1 - m2) ** 2) / (2 * v1 * v2)
        
        # # axs[0, i].scatter(weight_channel_norms.numpy(force=True), mt_channel_norms.numpy(force=True), label="mt")
        # # axs[0, i].scatter(weight_channel_norms.numpy(force=True), non_mt_channel_norms.numpy(force=True), label="non_mt")
        
        # # axs[0, i].set_title(f"Layer {layer_idx}: channel_norm_values")
        # # axs[0, i].set_xlabel("weight_channel_norm")
        # # axs[0, i].set_ylabel("token_channel_norm")
        # # axs[0, i].set_yscale("log")
        
        # # axs[0, i].legend()
        
        # # axs[i].scatter(weight_channel_norms.numpy(force=True), sym_kldiv.numpy(force=True), color="black", label="sym_kldiv")
        # mask = mt_channel_norms > non_mt_channel_norms
        # axs[i].scatter(weight_channel_norms[mask].numpy(force=True), (mt_channel_norms - non_mt_channel_norms)[mask].numpy(force=True), zorder=1, label="mt / non_mt")
        # axs[i].scatter(weight_channel_norms[~mask].numpy(force=True), (non_mt_channel_norms - mt_channel_norms)[~mask].numpy(force=True), zorder=0, label="non_mt / mt")
        
        # axs[i].set_title(f"Layer {layer_idx}: channel_norm_ratios")
        # axs[i].set_xlabel("weight_channel_norm")
        # axs[i].set_ylabel("token_channel_norm_ratio")
        # axs[i].set_yscale("log")
        
        axs[i].legend()

    plt.show()



def visualize_model_outputs(
    monitor: Monitor,
    original_images: torch.Tensor,
    input_images: torch.Tensor,
) -> None:
    model = monitor.model
    
    # SECTION: Visualize original images
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
    for image_idx, original_image in enumerate(shift_channels(original_images[:num_visualized_images])):
        axs[image_idx].imshow(original_image.numpy(force=True))
        axs[image_idx].axis("off")
    fig.suptitle("original_image")
    plt.show()
    
    output_dict = monitor.reset()
    with torch.no_grad():
        output = model.forward(input_images)
        
    stacked_layer_output_dict = OrderedDict([
        (metric, torch.stack([
            metric_output[0]
            for metric_output in metric_outputs
        ], dim=0))
        for metric, metric_outputs in output_dict.items()
    ])
    
    # SECTION: Massive token heuristic
    mta_mask = massive_token_heuristic(stacked_layer_output_dict)
    print(f"{mta_mask.sum().item()}/{mta_mask.numel()}")
    
    
    # SECTION: Per layer visualization code
    _, rgb_assignment = visualize_features_per_image(None, stacked_layer_output_dict["layer_output"][-1], plot=False)    
    # for metric, stacked_metric_output in stacked_layer_output_dict.items():
    #     visualize_feature_norms_per_layer(metric, stacked_metric_output, fn=lambda t: torch.norm(t, torch.inf, dim=-1))
    
        
    # SECTION: Histogram value visualization
    
    include = {
        "mlp_fc1_input": {"align_layers": True},
        "mlp_fc1_output": {"fn": lambda t: torch.mean(t, dim=0), "align_layers": False},
        # "mlp_fc1_output_no_bias": {"fn": lambda t: torch.mean(t, dim=0), "align_layers": False},
    }
    # include = {"layer_norm2_input", "layer_norm2_output"}
    # for metric, metric_kwargs in include.items():
    #     visualize_feature_values_by_channel(metric, stacked_layer_output_dict[metric], **metric_kwargs)
    
    visualize_feature_values_by_pca("mlp_fc1_input", stacked_layer_output_dict["mlp_fc1_input"])
    
    raise Exception()

    layer_output_dicts = [
        OrderedDict(zip(output_dict.keys(), layer_outputs))
        for layer_outputs in zip(*output_dict.values())
    ]
    
    include: Set[str] = {"layer_output"}
    rgb_assignment: torch.Tensor = None
    for layer_idx, layer_output_dict in enumerate(layer_output_dicts):
        print(f"Layer {layer_idx} {'=' * 120}")
        
        feature_norm_dict: Dict[str, torch.Tensor] = {}
        for metric, metric_output in layer_output_dict.items():
            if metric in include:
                visualize_features_per_image(metric, metric_output[0])
            feature_norm_dict[metric] = visualize_feature_norms_per_image(metric, metric_output[0])
        
        x_metric, y_metric = "layer_norm1", "layer_output"
        plt.scatter(
            feature_norm_dict[x_metric].numpy(force=True),
            feature_norm_dict[y_metric].numpy(force=True),
            color="black", s=2
        )
        plt.title(f"Layer {layer_idx}")
        plt.xlabel(f"{x_metric}_norm")
        plt.ylabel(f"{y_metric}_norm")
        plt.show()
        
        # if layer_idx == len(layer_output_dicts) - 1:
            
        


    raise Exception()
    



