import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import einops
import matplotlib.colors
import scipy.spatial
import torch
import torch.nn.functional as Fn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from sklearn.manifold import TSNE
from torch.utils._pytree import tree_flatten

from infrastructure.settings import DEVICE, OUTPUT_DEVICE
from infrastructure import utils
from core.monitor import Monitor


H = W = 16
num_visualized_images = 8
DEFAULT_LAYER_INDICES = [*range(9, 12)]


def pad_tensor_outputs(ts: List[torch.Tensor], dim: int) -> List[torch.Tensor]:
    result, d = [], max(t.shape[dim] for t in ts)
    for t in ts:
        t = t.to()
        l = t.shape[dim]
        pad_length = d - l
        pad = torch.full((*t.shape[:dim], pad_length, *t.shape[dim + 1:]), torch.nan)
        result.append(torch.cat((
            torch.index_select(t, dim, torch.arange(l - H * W)),
            pad, torch.index_select(t, dim, torch.arange(l - H * W, l)),
        ), dim=dim))
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


def get_mta_mask_from_indices(mta_indices: torch.Tensor):
    bsz = mta_indices.shape[0]
    mask = torch.full((bsz, H, W), False)
    mask[torch.arange(bsz)[:, None], mta_indices // W, mta_indices % W] = True
    return mask


def massive_token_heuristic(stacked_layer_output_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    log_norms = einops.rearrange(
        torch.norm(stacked_layer_output_dict["layer_output"][15, :, -H * W:], p=2, dim=-1).log(),
        "bsz (h w) -> bsz h w", h=H, w=W
    )
    flattened_norms = torch.sort(torch.flatten(log_norms), dim=0).values
    cutoff = torch.argmax(torch.diff(flattened_norms, dim=0), dim=0)
    mask = log_norms > flattened_norms[cutoff]
    return mask


ncut = NCUT(num_eig=100, distance="rbf", indirect_connection=False, device=DEVICE)
def visualize_features_per_image(metric_name: str, t: torch.Tensor, plot: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = t.shape[0]
    
    ncut_features, eigenvalues = ncut.fit_transform(t.flatten(0, 1))
    rgb_features = rgb_from_tsne_3d(ncut_features)[1]
    rgb_features = einops.rearrange(
        rgb_features.reshape(bsz, -1, 3)[:, -H * W:],
        "bsz (h w) c -> bsz h w c", h=H, w=W
    )
    
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
        for image_idx, image_features in enumerate(rgb_features[:num_visualized_images]):
            axs[image_idx].imshow(image_features.numpy(force=True))
            axs[image_idx].axis("off")
        fig.suptitle(metric_name)
        plt.show()

    return einops.rearrange(
        ncut_features.unflatten(0, (bsz, -1))[:, -H * W:],
        "bsz (h w) c -> bsz h w c", h=H, w=W
    ), rgb_features.to(OUTPUT_DEVICE)
    

def get_rgb_colors(t: torch.Tensor, mta_mask: torch.Tensor, binary: bool = False) -> torch.Tensor:
    if binary:
        r = torch.tensor(matplotlib.colors.to_rgb("red"))
        b = torch.tensor(matplotlib.colors.to_rgb("blue"))
        return torch.where(mta_mask[..., None], r, b)
    else:
        return visualize_features_per_image(None, t, plot=False)[1]
    

def visualize_feature_norms_per_image(metric_name: str, t: torch.Tensor) -> torch.Tensor:
    feature_norms = einops.rearrange(
        torch.norm(t, p=2, dim=-1)[:, -H * W:],
        "bsz (h w) -> bsz h w", h=H, w=W
    )
    assert torch.all(feature_norms >= 0), "Computed norms should be greater than 0."
    
    fig, axs = plt.subplots(nrows=1, ncols=num_visualized_images, figsize=(4 * num_visualized_images, 4))
    for image_idx, image_norms in enumerate(feature_norms[:num_visualized_images]):
        axs[image_idx].imshow(image_norms.numpy(force=True), cmap="gray")
        axs[image_idx].axis("off")
    fig.suptitle(f"{metric_name}_norm")
    plt.show()
    return feature_norms


def visualize_feature_norms_per_layer(
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    mta_indices: torch.Tensor,
    rgb_assignment: torch.Tensor,
    fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    bsz = t.shape[0]
    if fns is None:
        fns = {"norm", lambda t: torch.norm(t, p=2, dim=-1)}
        
    fig, axs = plt.subplots(nrows=1, ncols=len(fns), figsize=(7 * len(fns), 5), sharex=True, sharey=True)
    for i, (fn_name, fn) in enumerate(fns.items()):
        feature_norms = fn(t)
        
        min_token_idx = H * W - feature_norms.shape[-1]
        for image_idx, token_idx in itertools.product(range(bsz), range(min_token_idx, H * W)):
            h_idx, w_idx = token_idx // W, token_idx % W
            token_feature_norms = feature_norms[:, image_idx, token_idx - min_token_idx]
            
            if token_idx == min_token_idx:
                axs[i].plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color="black", linewidth=1, zorder=12, label="cls_token" if image_idx == 0 else None
                )
            elif token_idx < 0:
                if mta_indices is not None:
                    nan_idx = torch.argmax(~torch.isnan(token_feature_norms).to(torch.int))
                    mta_idx = mta_indices[image_idx, token_idx - min_token_idx - 1]
                    axs[i].plot(
                        [nan_idx - 1, nan_idx],
                        [feature_norms[nan_idx - 1, image_idx, mta_idx - min_token_idx], token_feature_norms[nan_idx]],
                        color="midnightblue", linewidth=0.5, linestyle="--",
                    )
                
                axs[i].plot(
                    token_feature_norms.numpy(force=True), marker=".",
                    color="gold", linewidth=0.5, label="register_token" if token_idx == min_token_idx + 1 and image_idx == 0 else None
                )
            else:
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


def visualize_feature_values_by_token(
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    rgb_assignment: torch.Tensor,
    layer_indices: List[int] = None,
) -> None:
    bsz = t.shape[1]
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(7 * len(layer_indices), 5), sharey=True)
    for i, layer_idx in enumerate(layer_indices):
        feature_values = t[layer_idx]
        
        m, s = feature_values.mean().item(), feature_values.std().item()
        left, right = m - 3 * s, m + 3 * s
        bins = torch.arange(left, right, s / 20).tolist()
        
        for image_idx, token_idx in itertools.product(range(bsz), range(-1, H * W)):
            h_idx, w_idx = token_idx // W, token_idx % W
            token_feature_values = feature_values[image_idx, token_idx + 1]
            
            p = torch.rand(()) < 0.3
            if token_idx != -1 and (mta_mask[image_idx, h_idx, w_idx] or p):
                axs[i].hist(
                    token_feature_values.numpy(force=True), bins=bins, histtype="step", density=True,
                    color=rgb_assignment[image_idx, h_idx, w_idx].numpy(force=True), zorder=20
                )
            elif token_idx == -1 and p:
                axs[i].hist(
                    token_feature_values.numpy(force=True), bins=bins, histtype="step", density=True,
                    color="black", zorder=12, label="cls_token" if image_idx == 0 else None
                )
        
        # axs[i].autoscale(False)
        # axs[i].plot([0, 0], [0, top := 1.5], color="black", linestyle="--")
        
        axs[i].set_title(f"Layer {layer_idx}: {metric_name}_values")
        axs[i].set_xlabel("value")
        axs[i].set_xlim(left=left, right=right)
        axs[i].set_ylabel("count")
        # axs[i].set_ylim(top=top)
    
    plt.show()


def visualize_feature_values_by_channel(
    metric_name: str,
    t: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor] = None,
    align_layers: bool = False,
    layer_indices: List[int] = None,
) -> None:
    if fn is None:
        fn = lambda t: torch.norm(t, dim=0)
    if align_layers:
        channel_feature_norms = fn(torch.flatten(t[layer_indices], 0, -2))
        channel_colors = plt.cm.get_cmap("plasma")((channel_feature_norms - channel_feature_norms.min()) / (channel_feature_norms.max() - channel_feature_norms.min()))
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(1 * len(layer_indices), 5 / 7), sharey=True)
    for i, layer_idx in enumerate(layer_indices):
        feature_values = t[layer_idx]
        
        if not align_layers:
            channel_feature_norms = fn(torch.flatten(feature_values, 0, -2))
            channel_colors = plt.cm.get_cmap("plasma")((channel_feature_norms - channel_feature_norms.min()) / (channel_feature_norms.max() - channel_feature_norms.min()))
        
        k = 300
        m, s = feature_values.mean().item(), feature_values.std().item()
        left, right = feature_values.flatten().topk(k=k, largest=False).values[-1], feature_values.flatten().topk(k=k, largest=True).values[-1]
        bins = torch.arange(left, right, s / 20).tolist()
        
        for channel_idx in range(feature_values.shape[2]):
            channel_feature_values = feature_values[:, :, channel_idx]
            
            axs[i].hist(
                channel_feature_values.flatten().numpy(force=True), bins=bins, histtype="step", density=True,
                color=channel_colors[channel_idx], zorder=channel_feature_norms[channel_idx]
            )
        
        axs[i].set_title(f"Layer {layer_idx}: {metric_name}_values")
        axs[i].set_xlabel("value")
        axs[i].set_xlim(left=left, right=right)
        axs[i].set_ylabel("count")
        # axs[i].set_ylim(top=top)
    
    plt.show()


def visualize_feature_norms_by_channel(
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    layer_indices: List[int] = None,
) -> None:
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=2, ncols=len(layer_indices), figsize=(2 * len(layer_indices), 10 / 7), sharey=True)
    for i, layer_idx in enumerate(layer_indices):
        feature_values = einops.rearrange(t[layer_idx, :, -H * W:], "bsz (h w) c -> bsz h w c", h=H, w=W)
        demeaned_feature_values = feature_values - torch.mean(feature_values.flatten(0, -2), dim=0)
        
        def plot_channel_norm(ax, values: torch.Tensor, prefix: str):
            ax.plot((values[mta_mask] ** 2).mean(dim=0) ** 0.5, zorder=1, label=f"{prefix}mt_channel_norm")
            ax.plot((values[~mta_mask] ** 2).mean(dim=0) ** 0.5, zorder=0, label=f"{prefix}non_mt_channel_norm")
            
            ax.set_title(f"Layer {layer_idx}: {metric_name}_{prefix}channel_norm")
            ax.legend()
            
        plot_channel_norm(axs[0, i], feature_values, "")
        plot_channel_norm(axs[1, i], demeaned_feature_values, "demeaned_")
    
    plt.show()


def visualize_feature_values_by_pca(
    metric_name: str,
    t: torch.Tensor,
    mta_mask: torch.Tensor,
    rgb_assignment: torch.Tensor,
    mean: Literal[None, "local", "global"],
    projection_mode: Literal["pca", "tsne"],
    weights: torch.Tensor = None,
    convex_hull: bool = True,
    layer_indices: List[int] = None,
) -> None:
    bsz = t.shape[1]
    if layer_indices is None:
        layer_indices = DEFAULT_LAYER_INDICES
    
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_indices), figsize=(2 * len(layer_indices), 2), subplot_kw={"projection": "3d"})
    for i, layer_idx in enumerate(layer_indices):
        feature_values = t[layer_idx]
        
        if feature_values.shape[-2] == H * W + 1 or torch.any(torch.isnan(feature_values)):
            mask = torch.cat((
                torch.full((bsz, feature_values.shape[-2] - H * W), False),
                torch.flatten(mta_mask, start_dim=1, end_dim=2),
            ), dim=-1)
        else:
            mask = torch.full(feature_values.shape[:-1], False)
            mask[:, 1:-H * W] = True
        inverse_mask = ~torch.any(torch.isnan(feature_values), dim=-1) & ~mask
                
        if projection_mode == "pca":    
            global_feature_mean = 0 if mean is None else torch.mean(feature_values[~torch.any(torch.isnan(feature_values), dim=-1)], dim=0)
            demeaned_feature_values = feature_values - global_feature_mean

            _, S_ma, V_ma = torch.pca_lowrank(demeaned_feature_values[mask], q=1, center=(mean == "local"))
            S_ma /= torch.sum(mask) ** 0.5
            _, S_nonma, V_nonma = torch.pca_lowrank(demeaned_feature_values[inverse_mask], q=2, center=(mean == "local"))
            S_nonma /= torch.sum(inverse_mask) ** 0.5
            
            V = torch.cat((V_ma, V_nonma), dim=-1)
            compressed_features = demeaned_feature_values @ V
            
        elif projection_mode == "tsne":
            compressed_features = torch.tensor(TSNE(n_components=3).fit_transform(feature_values.flatten(0, -2).numpy(force=True))).reshape(*mta_mask.shape, -1)   
        
        def to_rgb_mask(m: torch.Tensor) -> torch.Tensor:
            return m[:, -H * W:].view(bsz, H, W)
        
        subsample_mask = inverse_mask & (torch.arange(-feature_values.shape[-2], 0) >= -H * W) & (torch.rand(mask.shape) < 0.1)
        
        axs[i].scatter(*compressed_features[mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(mask)].numpy(force=True), s=10)
        axs[i].scatter(*compressed_features[subsample_mask].mT.numpy(force=True), color=rgb_assignment[to_rgb_mask(subsample_mask)].numpy(force=True), s=1, alpha=1.0)
        axs[i].scatter(*compressed_features[:, 0].mT.numpy(force=True), s=20, color="black", label="cls_token")
        
        if convex_hull:
            hull_kwargs = {"linewidth": 0.3, "edgecolor": "black", "alpha": 1.0}
            
            vertices = compressed_features[mask]
            hull = scipy.spatial.ConvexHull(vertices.numpy(force=True))
            axs[i].plot_trisurf(*vertices.mT.numpy(force=True), triangles=hull.simplices, cmap="pink", **hull_kwargs)
            
            cls_vertices = compressed_features[:, 0]
            cls_hull = scipy.spatial.ConvexHull(cls_vertices.numpy(force=True))
            axs[i].plot_trisurf(*cls_vertices.mT.numpy(force=True), triangles=cls_hull.simplices, cmap="bone", **hull_kwargs)
        
        axs[i].set_title(f"Layer {layer_idx}: {metric_name}_pca_values")
        axs[i].legend()

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
    



